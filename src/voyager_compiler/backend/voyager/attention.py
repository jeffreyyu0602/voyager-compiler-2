"""A flash attention on Voyager (``bufferize/attention_v3.py``):
``VoyagerAttention`` fits the FA3 kernel's buffers against the usable
scratchpad, and ``attention_tile_latency`` scores a tiling the way the
vector unit's tilings are scored (``vector.py``), against the FA3 schedule
instead.
"""

import math

import torch
from torch.fx import Node

from voyager_compiler.codegen.node_info import get_arg_value, tensor_alloc_bytes
from voyager_compiler.codegen.transform.tiling.lowering import (
    attention_operand_shapes,
    node_dtype_bits,
    operand_bytes,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    OpModel,
    Placement,
    pad_head,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    attention_kv_last,
    sweep_cycles,
)

from .vector import KERNEL_LAUNCH_OVERHEAD, tensor_bytes, transfer_cost


def attention_tile_latency(node, tiles, grid, config, matrix, bool_mask):
    """Latency and DRAM traffic of a flash-attention node under a tiling.

    Prices the FA3 schedule (``bufferize/attention_v3.py``) a step at a
    time.  The scores product runs first; the context product then runs on
    the matrix unit while the vector unit runs the softmax chain, which had
    to wait for the scores -- so a step costs the scores plus the busier of
    the two, then the rescale that folds the context in.  A one-row query
    tile makes the products matrix-vector, run on the vector unit like any
    (``gemv_compute_cycles``), so its step is the plain sum.  A query block's
    first step is a boundary: the vector unit resets its state, runs the
    softmax, then finalizes the previous block once that block's context has
    landed, so the finalize takes the rescale's place beside the context.
    The products are priced by their searches (``matrix``, at the head
    ``pad_head`` pads to); each vector pass at the bandwidth-bound rate
    ``vector_op_utilization`` charges, plus its launch.  Query and output
    move once per query block, while key, value, mask and every block scale
    reload on each step, and the DMAs overlap compute the way
    ``sweep_cycles`` prices a double-buffered sweep.
    Under ``is_causal`` only the live pairs are steps
    (``attention_kv_last``) and the mask tiles stream from the kernel's
    table, one per step, as int1 codes under ``bool_mask`` and else at the
    scores' width.  A split cache's residual
    (``bufferize/attention_v3.py``) runs serially at each boundary -- its
    two products around a softmax over its R positions, then the rescale
    of the output -- and its operands move once per boundary.

    Args:
        node: The attention node being tiled.
        tiles: Operand FX node -> its SRAM tile shape
            (``attention_operand_shapes``).
        grid: The FA3 loop grid, ``(*kv_batch, num_q_blocks, num_kv_blocks)``.
        config (AcceleratorConfig): The hardware description.
        matrix: Each product's cycles for one step, keyed as
            ``_attention_products`` names them: ``scores`` and ``context``
            for one step's two, plus ``residual_scores`` and
            ``residual_context`` under a residual.
        bool_mask: Whether the causal table's tiles are int1 codes.

    Returns:
        ``(cycles, DRAM bytes)``; the bytes break a latency tie.
    """
    query, key, value = node.args[0], node.args[1], node.args[2]
    causal = get_arg_value(node, 5, "is_causal", False)
    boundaries = math.prod(grid[:-1])  # one per query block
    tq, head_dim = tiles[query][-2], tiles[query][-1]
    tkv = tiles[key][-1]
    if causal:
        sq = query.value.shape[-2]
        steps = math.prod(grid[:-2]) * sum(
            attention_kv_last(q, tq, tkv, sq) + 1 for q in range(grid[-2])
        )
    else:
        steps = math.prod(grid)

    # The vector unit's passes, at the softmax's width (the output's): one
    # lane group per cycle, bound by the beats its bytes take.
    group_bytes = config.vector_lanes * node_dtype_bits(node) / 8
    util = min(1.0, 1.0 / math.ceil(group_bytes / config.bytes_per_cycle))

    def passes(count, elems):
        cycles = math.ceil(math.ceil(elems / config.vector_lanes) / util)
        return count * (cycles + KERNEL_LAUNCH_OVERHEAD)

    rows, tile, out = tq, tq * tkv, tq * head_dim
    # The softmax chain: rowmax, exponentials and rowsum over the score tile
    # (and P's quantize under MX); the running max, the rescale factor, its
    # copy and the running sum over a column.  Then the fused
    # rescale-accumulate over the output tile, or at a boundary the reset of
    # the two columns before the softmax and, after it, the accumulate, the
    # finalize and the zeroing of the output tile.
    softmax = passes(4 if node.kwargs.get("block_size") else 3, tile)
    softmax += passes(4, rows)
    rescale = passes(1, out)
    boundary = passes(3, out) + passes(2, rows)
    residual = get_arg_value(node, 7, "key_residual", None)
    if residual is not None:
        length = tiles[residual][-1]
        boundary += matrix["residual_scores"] + matrix["residual_context"]
        boundary += passes(3, tq * length) + passes(4, rows) + rescale
    scores, context = matrix["scores"], matrix["context"]
    if tq == 1:
        # One query row: the products are matrix-vector and run on the
        # vector unit beside the softmax, so nothing overlaps.
        interior = scores + context + softmax + rescale
        boundary_step = scores + context + softmax + boundary
    else:
        interior = scores + max(context, softmax) + rescale
        boundary_step = scores + max(context, softmax + boundary)

    lat = config.access_latency_cycles
    bpc = config.bytes_per_cycle
    # ``sweep_cycles`` reads the store off the front.
    out_bytes = operand_bytes(tiles[node], node)
    dmas = [(transfer_cost(tiles[node], out_bytes, lat, bpc), boundaries)]
    traffic = boundaries * out_bytes
    for operand, transfers in (
        (query, boundaries),
        (key, steps),
        (value, steps),
        (get_arg_value(node, 3, "attn_mask", None), steps),
        (node.kwargs.get("query_scale"), boundaries),
        (node.kwargs.get("key_scale"), steps),
        (node.kwargs.get("value_scale"), steps),
        (get_arg_value(node, 6, "query_residual", None), boundaries),
        (residual, boundaries),
        (get_arg_value(node, 8, "value_residual", None), boundaries),
        (get_arg_value(node, 9, "residual_mask", None), boundaries),
    ):
        if not isinstance(operand, Node):
            continue
        shape = tiles[operand]
        n_bytes = operand_bytes(shape, operand)
        dmas.append((transfer_cost(shape, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes
    if causal:
        mask_bits = 1 if bool_mask else node_dtype_bits(node)
        mask_bytes = math.ceil(tq * tkv * mask_bits / 8)
        dmas.append((transfer_cost((tq, tkv), mask_bytes, lat, bpc), steps))
        traffic += steps * mask_bytes

    if config.double_buffered_l2:
        latency = sweep_cycles(dmas, steps, interior)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += steps * interior
    latency += boundaries * (boundary_step - interior)
    return latency, traffic


class VoyagerAttention(OpModel):
    """A flash attention on Voyager (``bufferize/attention_v3.py``): the FA3
    kernel's buffers against the usable scratchpad, priced by
    ``attention_tile_latency`` over its products' compute.

    Attributes:
        node: The attention node.
        space: Its ``AttentionSpace``.
        key: ``None``: an attention search shares no result.
        config (AcceleratorConfig): The hardware description.
        acc_dtype: The dtype the kernel keeps its softmax state in.
        bool_mask: The causal table's tiles are int1 codes rather than the
            additive fill at ``acc_dtype``.
    """

    def __init__(self, node, space, config, acc_dtype, bool_mask):
        self.node = node
        self.space = space
        self.key = None
        self.config = config
        self.acc_dtype = acc_dtype
        self.bool_mask = bool_mask

    def fit(self, tiling):
        """``Placement()`` when the FA3 kernel's SRAM allocations under
        ``tiling`` fit the usable scratchpad, else ``None``.

        The allocations are ``_FA3Pipeline.forward``'s: two slots of every
        operand tile but the output's one, the running softmax state and the
        score tile at ``acc_dtype``, and the probabilities -- under MX their
        codes at the query's dtype plus their block scales, else a buffer at
        V's dtype when the softmax runs at a dtype other than the operands'
        -- each sized the way ``plan_memory`` will place it.  The operand
        tiles and the P@V product hold the head at ``pad_head``: the query's,
        value's and their scales' last dim, the transposed key's and its
        scales' first.  A split cache's residual adds its own score tile,
        product, max snapshot and, at a dtype the matrix unit cannot take,
        probabilities.
        """
        node, config, acc_dtype = self.node, self.config, self.acc_dtype
        tiles = attention_operand_shapes(node, tiling)
        query, key, value = node.args[0], node.args[1], node.args[2]
        tq, head_dim = tiles[query]
        tkv = tiles[value][0]
        head_pad = pad_head(head_dim, self.space)
        kw = node.kwargs
        block_size = kw.get("block_size")
        grown = {
            query: (-1, head_pad),
            key: (0, head_pad),
            value: (-1, head_pad),
        }
        if block_size is not None:
            head_blocks = math.ceil(head_pad / block_size)
            grown[kw["query_scale"]] = (-1, head_blocks)
            grown[kw["key_scale"]] = (0, head_blocks)
            grown[kw["value_scale"]] = (-1, head_pad)
        residual = get_arg_value(node, 7, "key_residual", None)
        if residual is not None:
            grown[get_arg_value(node, 6, "query_residual")] = (-1, head_pad)
            grown[residual] = (0, head_pad)
            grown[get_arg_value(node, 8, "value_residual")] = (-1, head_pad)
        padded = dict(tiles)
        for operand, (dim, size) in grown.items():
            shape = list(tiles[operand])
            shape[dim] = size
            padded[operand] = tuple(shape)

        total = sum(
            (1 if operand is node else 2) * tensor_bytes(operand, shape, config)
            for operand, shape in padded.items()
        )

        def acc_bytes(shape):
            return tensor_alloc_bytes(
                math.prod(shape),
                acc_dtype,
                config.bank_width,
                config.vector_lanes,
            )

        # m, row_tmp, alpha and the two slots of l; o, pv_buf; the
        # double-buffered s_buf.
        total += 5 * acc_bytes((tq, 1))
        total += acc_bytes((tq, head_dim)) + acc_bytes((tq, head_pad))
        total += 2 * acc_bytes((tq, tkv))
        if block_size is not None:
            total += 2 * tensor_bytes(query, (tq, tkv), config)
            total += 2 * tensor_bytes(
                node.kwargs["query_scale"], (tq, tkv // block_size), config
            )
        elif acc_dtype != value.value.dtype:
            total += 2 * tensor_bytes(value, (tq, tkv), config)
        if get_arg_value(node, 5, "is_causal", False):
            # The causal mask tiles stream through two slots like a mask's.
            total += 2 * tensor_alloc_bytes(
                tq * tkv,
                torch.bool if self.bool_mask else acc_dtype,
                config.bank_width,
                config.vector_lanes,
            )
        if residual is not None:
            length = tiles[residual][1]
            total += acc_bytes((tq, length))
            total += acc_bytes((tq, head_pad)) + acc_bytes((tq, 1))
            value_residual = get_arg_value(node, 8, "value_residual")
            if acc_dtype != value_residual.value.dtype:
                total += tensor_bytes(value_residual, (tq, length), config)
        if total > config.usable_scratchpad_size:
            return None
        return Placement()

    def cost(self, tiling, placement):
        """``attention_tile_latency`` of ``tiling``, each product charged its
        result's one-step ``compute`` (``tiling.products``); ``placement`` is
        the contract's."""
        matrix = {
            name: result.cost.compute
            for name, result in tiling.products.items()
        }
        latency, traffic = attention_tile_latency(
            self.node,
            attention_operand_shapes(self.node, tiling),
            tiling.grid,
            self.config,
            matrix,
            self.bool_mask,
        )
        return Cost(latency, traffic)
