"""The tile search of a flash attention.

``attention_op_tiling`` enumerates the query / key tile lengths, keeps those
the backend's model fits on chip, maps each one's products -- bare 2-D GEMMs
of the tile shapes -- through the GEMM search (``get_tiling``), prices every
candidate over its products' results and stamps the winner as
``meta["tile_result"]``.
"""

import logging
import math
from dataclasses import replace

import torch

import interstellar
from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.transform.tiling.matrix_search import (
    get_tiling,
    prefetch_tilings,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    AttentionTiling,
    TileResult,
    pad_head,
)
from voyager_compiler.codegen.transform.tiling.tiler import TileConstraint, pick
from voyager_compiler.shape_prop import set_node_value

logger = logging.getLogger(__name__)
le = interstellar.le


def _product_node(name, out_dtype, left, right, block_size, scales, codes):
    """A bare 2-D product ``left @ right`` as a node of its own graph, for
    the mapping search: placeholders carrying the operands' tile shapes and
    the logical dtypes of the nodes the tiles are cut from.

    Args:
        name: The node's name, for the search's logging.
        out_dtype: The product's torch dtype.
        left: ``(shape, source)`` -- the tile shape and the FX node it is
            cut from, whose dtype it takes.  Likewise ``right``.
        block_size: The MX block, or ``None`` for an unquantized product.
        scales: Under MX, the ``(shape, source)`` of each operand's block
            scales, left then right.
        codes: Under MX, the two codebook nodes, left then right, each
            ``None`` for a format without one (``fp4``).
    """
    graph = torch.fx.Graph()

    def placeholder(label, source, shape=None):
        ph = graph.placeholder(f"{name}_{label}")
        shape = tuple(source.value.shape) if shape is None else shape
        set_node_value(ph, torch.empty(shape, dtype=source.value.dtype))
        ph.meta["dtype"] = source.meta.get("dtype")
        return ph

    a = placeholder("a", left[1], left[0])
    b = placeholder("b", right[1], right[0])
    if block_size is None:
        node = graph.call_function(
            torch.ops.aten.matmul.default, (a, b), name=name
        )
    else:
        (a_scale, b_scale), (a_code, b_code) = scales, codes
        kwargs = {
            "input_scale": placeholder("a_scale", a_scale[1], a_scale[0]),
            "weight_scale": placeholder("b_scale", b_scale[1], b_scale[0]),
            "block_size": block_size,
        }
        for label, code in (("input_code", a_code), ("weight_code", b_code)):
            if code is not None:
                kwargs[label] = placeholder(label, code)
        node = graph.call_function(
            torch.ops.quantized_ops.matmul_mx.default, (a, b), kwargs, name=name
        )
    set_node_value(
        node, torch.empty((left[0][0], right[0][1]), dtype=out_dtype)
    )
    return node


def _attention_products(node, tq, tkv, head_pad):
    """The two products a ``(tq, tkv)`` attention tile runs, as bare 2-D
    GEMM nodes of the tile shapes (``_product_node``): ``"scores"``, the
    query tile times the transposed key tile, and ``"context"``, the
    probabilities times the value tile.  The probabilities take the query's
    dtype and scales, as the kernel quantizes them so.  The products hold
    the head at ``head_pad`` (``pad_head``), the kernel's tile width."""
    query, key, value = node.args[0], node.args[1], node.args[2]
    block_size = node.kwargs.get("block_size")
    out_dtype = node.value.dtype
    kw = node.kwargs
    if block_size is None:
        codes = scores_scales = context_scales = ()
    else:
        codes = (kw.get("input_code"), kw.get("weight_code"))
        # A block wider than the head is one truncated block.
        head_blocks = math.ceil(head_pad / block_size)
        scores_scales = (
            ((tq, head_blocks), kw["query_scale"]),
            ((head_blocks, tkv), kw["key_scale"]),
        )
        context_scales = (
            ((tq, tkv // block_size), kw["query_scale"]),
            ((tkv // block_size, head_pad), kw["value_scale"]),
        )
    scores = _product_node(
        f"attention_scores_{tq}x{tkv}",
        out_dtype,
        ((tq, head_pad), query),
        ((head_pad, tkv), key),
        block_size,
        scores_scales,
        codes,
    )
    context = _product_node(
        f"attention_context_{tq}x{tkv}",
        out_dtype,
        ((tq, tkv), query),
        ((tkv, head_pad), value),
        block_size,
        context_scales,
        codes,
    )
    products = {"scores": scores, "context": context}
    residual = get_arg_value(node, 7, "key_residual", None)
    if residual is not None:
        # The split cache's residual: plain products over its R positions,
        # the probabilities at the residual query's dtype.
        length = residual.value.shape[-2]
        query_residual = get_arg_value(node, 6, "query_residual")
        products["residual_scores"] = _product_node(
            f"attention_residual_scores_{tq}x{length}",
            out_dtype,
            ((tq, head_pad), query_residual),
            ((head_pad, length), residual),
            None,
            (),
            (),
        )
        products["residual_context"] = _product_node(
            f"attention_residual_context_{tq}x{length}",
            out_dtype,
            ((tq, length), query_residual),
            ((length, head_pad), get_arg_value(node, 8, "value_residual")),
            None,
            (),
            (),
        )
    return products


def _product_result(product, tiler):
    """The ``TileResult`` ``get_tiling`` stamps on ``product`` -- a mapping,
    or a grid tiling for a one-row product (``gemv_op_tiling``) -- or
    ``None`` unless it maps as the one on-chip tile ``attention_op_tiling``
    pins it to."""
    counts, _ = get_tiling(product, tiler)
    if counts is None or math.prod(counts) != 1:
        return None
    return product.meta["tile_result"]


def _divisors_descending(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def attention_op_tiling(node, op, tiler, *, sq_eff, kv_batch):
    """Block counts for a flash-attention node, ``(num_q_blocks,
    num_kv_blocks)``.

    Enumerates the query / key tile lengths that divide the (GQA-folded)
    query rows and the key rows in multiples of ``op.space.unit``, keeps
    those ``op`` fits on chip, maps each one's products
    (``_attention_products``, the head padded by ``pad_head``) through
    ``get_tiling`` -- so identical shapes share one search, run concurrently
    by ``prefetch_tilings`` -- prices each candidate with ``op.cost`` over
    its products' results, and keeps the ``pick`` under the tiler's
    ``runtime_tolerance``.  The winner is stamped as
    ``node.meta["tile_result"]``: its tiling's ``products`` hold each
    product's result, for the builder to copy onto the kernels that run
    them.

    Args:
        node: The attention node to tile.
        op: Its ``OpModel`` (``RuntimeModel.prepare_attention``).
        tiler: The ``Tiler``.
        sq_eff: Query rows after the GQA fold (``plan_gqa_fold``).
        kv_batch: The key / value batch dims, which lead the loop grid.

    Returns:
        ``(num_q_blocks, num_kv_blocks)``.

    Raises:
        RuntimeError: when no tiling of the operands fits the scratchpad, or
            no tiling's products map on chip.
    """
    query, key = node.args[0], node.args[1]
    skv = key.value.shape[-2]
    # Under MX the query and key block along head_dim, the value and the
    # probabilities along the keys, so a key tile holds whole blocks.
    block_size = node.kwargs.get("block_size")
    logger.info(f"Running L2 tiling for attention: {node}")

    unit = op.space.unit
    # A causal query tile must not straddle a head under the GQA fold
    # (``attention_kv_last``).
    causal = get_arg_value(node, 5, "is_causal", False)
    sq = query.value.shape[-2]
    candidates = []  # (tiling, placement)
    for tq in _divisors_descending(sq_eff):
        if tq % unit and tq != sq_eff:
            continue
        if causal and sq % tq:
            continue
        for tkv in _divisors_descending(skv):
            if tkv % unit and tkv != skv:
                continue
            if block_size is not None and tkv % block_size:
                continue
            tiling = AttentionTiling(
                tile_sizes=(tq, tkv),
                grid=tuple(kv_batch) + (sq_eff // tq, skv // tkv),
            )
            placement = op.fit(tiling)
            if placement is None:
                continue
            candidates.append((tiling, placement))
    if not candidates:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad"
        )

    head_pad = pad_head(query.value.shape[-1], op.space)
    products = {
        tiling.tile_sizes: _attention_products(
            node, *tiling.tile_sizes, head_pad
        )
        for tiling, _ in candidates
    }
    # The kernel holds each product's operands whole on chip, so its L2 tile
    # is pinned to the product and the search maps only the L1 / PE levels.
    for pair in products.values():
        for product in pair.values():
            m, k = product.args[0].value.shape
            n = product.args[1].value.shape[-1]
            tiler.constraints[product] = TileConstraint(
                exact=((le.IC, k), (le.OC, n), (le.OX, m))
            )
    prefetch_tilings(
        [p for pair in products.values() for p in pair.values()], tiler
    )

    scored = []
    for tiling, placement in candidates:
        try:
            results = {
                name: _product_result(product, tiler)
                for name, product in products[tiling.tile_sizes].items()
            }
        except RuntimeError:  # a product no mapping fits on chip
            continue
        if any(result is None for result in results.values()):
            continue
        tiling = replace(tiling, products=results)
        cost = op.cost(tiling, placement)
        scored.append(TileResult(tiling, placement, cost))
        logger.info(
            "[tiling] %s: %d x %d -> %.0f cycles, %.1f MB",
            node.name,
            *tiling.tile_sizes,
            cost.latency,
            cost.dram_bytes / 1e6,
        )
    if not scored:
        raise RuntimeError(f"{node}: no tiling of its products maps on chip")

    best = pick(scored, tiler.runtime_tolerance)
    node.meta["tile_result"] = best
    return best.tiling.grid[-2:]
