"""Facts about a node that hold on every backend: how the shared builders
lower it.

A tile search reads the graph for the operands a fused tail adds, whether
that tail takes a pass of its own, how a GEMM loops its batch, which tile of
each buffer a candidate tiling cuts, and which buffer a placement member
names.  The builders fix all of it, whatever hardware runs the kernel, so a
runtime model calls what it needs here in ``prepare`` and freezes the
answers into its ``OpModel``.
"""

import math
from typing import Optional

import torch
from torch.fx import Node

import interstellar
from voyager_compiler.codegen.node_info import (
    _pair,
    activation_transforms,
    bound_operands,
    compute_output_tiled_shapes,
    compute_tiled_shape,
    csr_quantize_node,
    dtype_byte_size,
    get_anchor_node,
    get_arg_value,
    is_bmm,
    is_conv2d,
    normalize_shape,
    quant_param_arg_nodes,
    reduced_input_tiling,
    require_allocation,
    stream_breaking_quantize,
    trailing_mha_perm,
    weight_is_ck,
    weight_transforms,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Fused,
    KernelInput,
    Role,
)
from voyager_compiler.ops.layout import (
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    OIHW_TO_HWIO,
    unproject,
)

le = interstellar.le


# --------------------------------------------------------------------------
# Element widths
# --------------------------------------------------------------------------


def get_dtype_width(dtype) -> int:
    """Element width in bits, derived from the canonical ``dtype_byte_size``
    so the dtype-name parsing lives in exactly one place."""
    return round(dtype_byte_size(dtype) * 8)


def node_dtype_bits(node, default: Optional[int] = None):
    """Element width in bits of ``node``'s tensor, read from the graph.

    Prefers the compiler's tracked storage dtype (``meta['dtype']``, e.g. an
    NF4 weight) over the traced tensor dtype.  A multi-output node --
    ``quantize_mx`` gives ``(scale, quantized)`` -- carries one tracked dtype
    per output and gives one width per output in that order; every other node
    gives a single width.

    Args:
        node: The FX node to size, or ``None`` for an absent operand.
        default: Width to return when there is no node to size; omit to raise.

    Returns:
        The element width in bits, or one per output for a multi-output node.

    Raises:
        ValueError: If ``node`` is not an FX node and ``default`` is ``None``.
    """
    if not isinstance(node, Node):
        if default is None:
            raise ValueError(f"node {node} has no dtype to size the operand")
        return default

    dtype = node.meta.get("dtype")
    value = getattr(node, "value", None)
    if value is None:
        value = node.meta.get("val")

    if isinstance(dtype, (list, tuple)):
        # One tracked dtype per output; a ``None`` entry is an output left at
        # the dtype it was traced with.
        return [
            get_dtype_width(d if d is not None else v.dtype)
            for d, v in zip(dtype, value)
        ]
    if dtype is not None:
        return get_dtype_width(dtype)
    if isinstance(value, (list, tuple)):
        return [get_dtype_width(v.dtype) for v in value]
    return get_dtype_width(value.dtype)


def operand_bytes(shape, node):
    """Physical bytes of a tiled operand.  ``shape`` is a single tile shape, or
    a sequence of per-output shapes for a multi-output node, which then pays for
    each output at its own width."""
    bits = node_dtype_bits(node)
    if isinstance(bits, int):
        return math.ceil(math.prod(shape) * bits / 8)
    return sum(math.ceil(math.prod(s) * w / 8) for s, w in zip(shape, bits))


# --------------------------------------------------------------------------
# A GEMM / conv and its fused tail
# --------------------------------------------------------------------------


def layer_key(anchor):
    """What a GEMM / conv ``anchor`` alone says about its mapping, hashable:
    its op, operand and output shapes, the conv stride / padding / dilation,
    the operand, scale and outlier-CSR element widths and the microscaling
    block.  A model's ``OpModel.key`` adds what the anchor does not say --
    the output dtype, the fused tail -- so identical layers share one search.
    The op is named by its string, so a key pickles."""
    val = anchor.value
    out_shape = tuple(val.shape) if isinstance(val, torch.Tensor) else None
    key = [
        str(anchor.target),
        tuple(anchor.args[0].shape),
        tuple(anchor.args[1].shape),
        out_shape,
        node_dtype_bits(anchor.args[0]),
        node_dtype_bits(anchor.args[1]),
        node_dtype_bits(anchor.kwargs.get("input_scale"), 0),
        node_dtype_bits(anchor.kwargs.get("weight_scale"), 0),
        node_dtype_bits(anchor.kwargs.get("A_indptr"), 0),
        node_dtype_bits(anchor.kwargs.get("A_data"), 0),
        node_dtype_bits(anchor.kwargs.get("A_indices"), 0),
        anchor.kwargs.get("block_size"),
    ]
    if is_conv2d(anchor):
        key += [
            _pair(get_arg_value(anchor, 3, "stride", 1)),
            _pair(get_arg_value(anchor, 4, "padding", 0)),
            _pair(get_arg_value(anchor, 5, "dilation", 1)),
        ]
    return tuple(key)


def matrix_layer(anchor):
    """The interstellar ``Layer`` of a GEMM / conv ``anchor``, from its
    current (pre-tiling) shapes.  A GEMM is a 1x1 conv: its rows ride the
    output width, the batch of a batched matmul left to the builder."""
    weight_shape = anchor.args[1].shape
    transposed = anchor.meta.get("transposed", False)

    if is_conv2d(anchor):
        w_dims = OIHW_TO_HWIO if transposed else None
        in_dims = NCHW_TO_NHWC if transposed else None
        out_channels, in_channels, kH, kW = unproject(weight_shape, w_dims)
        _, _, height, width = unproject(anchor.shape, in_dims)
        stride_h, stride_w = _pair(get_arg_value(anchor, 3, "stride", 1))
    else:
        width = (
            anchor.shape[-2] if is_bmm(anchor) else math.prod(anchor.shape[:-1])
        )

        if weight_is_ck(anchor):
            in_channels, out_channels = weight_shape[-2:]
        else:
            out_channels, in_channels = weight_shape[-2:]

        kH = kW = height = stride_h = stride_w = 1

    return interstellar.Layer(
        nifm=in_channels,
        nofm=out_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


def _output_pos_to_loop_dim(anchor):
    """Map each output-tensor dim position to its interstellar output loop dim
    (``ON`` batch / ``OC`` channels / ``OY`` height / ``OX`` width), in the
    anchor's physical layout — so a fused operand's shape can be broadcast onto
    the output tile.  conv: NCHW or NHWC (``meta['transposed']``); gemm:
    ``(batch.., M, N)`` with ``M -> OX``, ``N -> OC``, batch dims ``-> ON``.
    """
    if is_conv2d(anchor):
        if anchor.meta.get("transposed", False):
            return [le.ON, le.OY, le.OX, le.OC]  # NHWC
        return [le.ON, le.OC, le.OY, le.OX]  # NCHW
    ndim = anchor.value.ndim
    return [le.ON] * (ndim - 2) + [le.OX, le.OC]


def _operand_placeholders(root):
    """External placeholder operands feeding ``root``'s subtree (``root``
    inclusive), tracing through pre-processing ops (``dequantize`` / reshape)
    and skipping each op's quantization codebook / qmap args.
    """
    leaves, stack, visited = [], [root], set()
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n.op == "placeholder":
            leaves.append(n)
            continue
        codebooks = quant_param_arg_nodes(n)
        for inp in n.all_input_nodes:
            if inp not in codebooks:
                stack.append(inp)
    return leaves


def fused_operand_nodes(node, anchor):
    """The fused post-op operand placeholders of ``node``'s submodule, in the
    order ``fused_operand_specs`` sizes them — so a ``Fused(i)`` member of a
    placement names ``fused_operand_nodes(node, anchor)[i]``.

    A post-op operand is any submodule placeholder that is *not* one of the
    anchor's own operands (act / weight / scales, traced through any input
    dequantize / reshape — those are counted by the interstellar ``Layer``) and
    is not a codebook / qmap or scalar.  Defining it by exclusion catches an
    operand fed through a ``dequantize`` (e.g. the attention mask).  The
    submodule is already ShapeProp'd (placeholders carry ``.value``).  Empty
    for a bare node.
    """
    submod = node.meta.get("submodule")
    if submod is None:
        return []

    anchor_operands = set(_operand_placeholders(anchor))
    codebooks = set()
    for n in submod.graph.nodes:
        codebooks |= quant_param_arg_nodes(n)

    return [
        p
        for p in submod.graph.nodes
        if p.op == "placeholder"
        and p.value.numel() != 1
        and p not in anchor_operands
        and p not in codebooks
    ]


def fused_operand_specs(node, anchor):
    """Per fused post-op operand of a fused ``call_module`` ``node``: a
    ``(dims, dtype_bits)`` pair, where ``dims`` are the interstellar output loop
    dims (a subset of ``ON/OC/OY/OX``) the operand is *tiled* along — broadcast
    (size-1) dims dropped, so its tile size is ``prod(out_tile[d] for d in
    dims)``.  Empty for a bare node, or one whose fused ops add no tiled tensor
    operand (codebooks / scalars don't count).  The operands themselves come
    from ``fused_operand_nodes``, in the same order.
    """
    operands = fused_operand_nodes(node, anchor)
    if not operands:
        return []

    pos_to_loop_dim = _output_pos_to_loop_dim(anchor)
    out_ndim = anchor.value.ndim

    specs = []
    for p in operands:
        op_shape = tuple(p.shape)
        offset = out_ndim - len(op_shape)  # right-align (broadcast)
        dims = tuple(
            pos_to_loop_dim[offset + i]
            for i, sz in enumerate(op_shape)
            if sz > 1
        )
        specs.append((dims, node_dtype_bits(p)))
    return specs


def tail_extra_passes(node):
    """Whether the fused tail of a GEMM / conv ``node`` takes a pass of its
    own over the finished tile, ``(single_k, split_k)``: after a reduction
    that runs in one round, and after one split across rounds.

    It does when its quantize breaks the stream or the pipeline has no stage
    left for it (fusion's ``single_k_tail_fusible`` /
    ``split_k_tail_fusible``): after the drain on a single round, after the
    accumulate on a split one, which takes one more.  A CSR-producing
    epilogue (``_EpilogueTail``) re-dices its tile at the consumers' slice
    width: ops between the anchor and the ``quantize_mx_outlier`` run once on
    the whole tile, and their result is staged whenever a slice is finer
    than the column tile.  The slice count is a consumer property the search
    cannot see, so every prefixed producer tail takes the single round's
    pass.
    """
    anchor = get_anchor_node(node)
    sub_gm = node.meta.get("submodule")
    breaks_stream = stream_breaking_quantize(sub_gm) is not None
    single_k = breaks_stream or not node.meta.get("single_k_tail_fusible", True)
    out_quant = csr_quantize_node(node)
    if not single_k and out_quant is not None:
        single_k = (
            isinstance(out_quant.args[0], Node)
            and out_quant.args[0].target is not anchor.target
        )
    split_k_fusible = node.meta.get("split_k_tail_fusible", False)
    split_k = sub_gm is not None and (breaks_stream or not split_k_fusible)
    return single_k, split_k


def tail_keeps_shape(node):
    """Whether a GEMM / conv ``node``'s output has its anchor's shape, so a
    single round's extra pass runs in place on the output slot rather than
    through scratch (``_gemm_scratch_and_kernel``)."""
    if isinstance(node.value, (tuple, list)):
        return False
    return tuple(node.value.shape) == tuple(get_anchor_node(node).value.shape)


def output_dtype(node):
    """The dtype ``node``'s group stores its result in: ``meta["dtype"]``,
    or one per output for a multi-output tail (``quantize_mx`` gives
    ``(scale, values)``).  A CSR producer's stream is unquantized, so the
    meta leaves its entries empty; an output without one takes the dtype it
    was traced with (the rule ``node_dtype_bits`` documents)."""
    out_dtype = node.meta.get("dtype")
    vals = getattr(node, "value", None)
    if isinstance(vals, (list, tuple)):
        tracked = (
            out_dtype
            if isinstance(out_dtype, (list, tuple))
            else [None] * len(vals)
        )
        out_dtype = [
            d if d is not None else v.dtype for d, v in zip(tracked, vals)
        ]
    return out_dtype


def gemm_batch(anchor):
    """Batch elements the builder loops a GEMM ``anchor`` over, outside its
    mapping: a batched matmul's leading dims, else 1."""
    return math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1


def weight_batch(anchor):
    """Distinct weight tiles over the ``gemm_batch`` loop: a GQA repeat reads
    one tile for several batch elements."""
    weight = anchor.args[1]
    repeat = weight_transforms(weight)[2]
    weight_repeat = (
        math.prod(repeat[: max(0, len(weight.shape) - 2)]) if repeat else 1
    )
    return gemm_batch(anchor) // weight_repeat


def weight_k_major(anchor):
    """Whether a GEMM ``anchor``'s weight tile lands ``[contraction, out]``:
    its stored layout (``weight_is_ck``), flipped by a transpose fused onto
    it."""
    return weight_is_ck(anchor) != weight_transforms(anchor.args[1])[1]


def outlier_stats(node):
    """``(outlier_pct, out_outlier_pct, outlier_rate)`` of a GEMM / conv
    ``node``: the fraction of the activation an outlier CSR its anchor
    consumes holds, the fraction of the output the CSR its fused tail emits
    may hold (that quantize's ``max_pct``), and the consumed CSR's measured
    outlier rate.  Each is 0 without such a CSR."""
    anchor = get_anchor_node(node)
    outlier_pct = 0.0
    outlier_rate = 0.0
    a_data = anchor.kwargs.get("A_data")
    if a_data is not None and getattr(a_data, "value", None) is not None:
        act = anchor.args[0].value
        elements = act.shape[-2] * act.shape[-1]
        outlier_pct = a_data.value.shape[-1] / elements
        outlier_rate = anchor.meta["outlier_rate"]

    out_outlier_pct = 0.0
    out_quant = csr_quantize_node(node)
    if out_quant is not None:
        out_outlier_pct = get_arg_value(out_quant, 9, "max_pct", 0.01)
    return outlier_pct, out_outlier_pct, outlier_rate


def mha_head(node):
    """The head dim an MHA relayout ending ``node``'s fused tail cuts the
    GEMM's output into, or ``None`` without one.  The bufferizer folds the
    relayout into the store, which cannot write a tile that splits a head,
    so the output-feature tile has to hold whole heads.  Fusion has already
    run, so the permute is the tail of the submodule rather than a user."""
    anchor = get_anchor_node(node)
    submod = node.meta.get("submodule")
    if submod is None or is_conv2d(anchor):
        return None
    perm = trailing_mha_perm(
        [n for n in submod.graph.nodes if n.op == "call_function"]
    )
    if perm is None or perm.value.ndim <= anchor.value.ndim:
        return None
    return perm.value.shape[-1]


def resolve(member, node, anchor):
    """The input node a :class:`Placement` member names: the one ``node``'s
    kernel loads the operand from -- for a ``Role``, the anchor's operand
    traced beneath the transforms the builders fold into its fetch
    (``activation_transforms`` / ``weight_transforms``), bound to the node
    ``node`` passes in (``bound_operands``).

    Args:
        member: A ``KernelInput``, a ``Fused`` or an operand ``Role``: any
            but ``Role.OUTPUT`` and ``Role.SCRATCH``, which name no input.
        node: The node being built: a fused ``call_module``, or the op.
        anchor: Its anchor (``get_anchor_node``).

    Returns:
        The node, or ``None`` when the kernel holds no such buffer -- no
        bias, no scale, a fused operand the tail does not have.
    """
    if isinstance(member, KernelInput):
        return node.all_input_nodes[member.index]
    if isinstance(member, Fused):
        fused = fused_operand_nodes(node, anchor)
        source = fused[member.index] if member.index < len(fused) else None
    elif member is Role.INPUT:
        source = activation_transforms(anchor.args[0])[0]
    elif member is Role.WEIGHT:
        source = weight_transforms(anchor.args[1])[0]
    elif member is Role.BIAS:
        source = get_arg_value(anchor, 2, "bias")
    elif member is Role.INPUT_SCALE or member is Role.WEIGHT_SCALE:
        # A scale role is named as the anchor's keyword.
        scale = anchor.kwargs.get(member.value)
        source = (
            weight_transforms(scale)[0] if isinstance(scale, Node) else None
        )
    else:  # Role.CSR
        source = anchor.kwargs.get("A_indptr")
    if not isinstance(source, Node):
        return None
    bound = bound_operands(node, node.meta.get("submodule"))
    return bound.get(source, source)


# --------------------------------------------------------------------------
# The buffers a grid or attention tiling cuts
# --------------------------------------------------------------------------


def output_shape(node):
    """The shape of ``node``'s output; of its last output for a
    multi-output op, which names the iteration space."""
    return (
        node.value.shape
        if isinstance(node.value, torch.Tensor)
        else node.value[-1].shape
    )


def _gemv_role_tiles(anchor, c_tile, k_tile):
    """Each operand role of a GEMV ``anchor`` -> its tile under a ``(1,
    c_tile, k_tile)`` tile: the activation's leading dims at 1, and a batched
    weight's batch dims too.  A linear names its weight ``weight``, a matmul
    ``other``."""
    leading = (1,) * (len(anchor.args[0].shape) - 1)
    batch = leading[:-1]
    c_scaled = c_tile // anchor.kwargs.get("block_size", 1)

    if weight_is_ck(anchor):
        weight, weight_scale = (c_tile, k_tile), (c_scaled, k_tile)
    else:
        weight, weight_scale = (k_tile, c_tile), (k_tile, c_scaled)

    if is_bmm(anchor):
        weight, weight_scale = batch + weight, batch + weight_scale

    return {
        "input": leading + (c_tile,),
        "weight": weight,
        "other": weight,
        "bias": (k_tile,),
        "input_scale": leading + (c_scaled,),
        "weight_scale": weight_scale,
        "output": leading + (k_tile,),
    }


def _prologue_tile(tile, operand_shape, param_shape):
    """Tile of a parameter read beside an operand: the operand's tile, shrunk
    along every dim by the ratio of the two full shapes.  Ranks that do not
    line up fall back to the whole parameter."""
    if not len(tile) == len(operand_shape) == len(param_shape):
        return tuple(param_shape)
    return tuple(
        max(1, math.ceil(t * q / f))
        for t, f, q in zip(tile, operand_shape, param_shape)
    )


def gemv_operand_shapes(node, tiling):
    """Each buffer a matrix-vector GEMM's kernel holds -> its tile under
    ``tiling``, a ``GridTiling`` over ``(X, C, K)``: keyed by the FX node it
    belongs to, ``node`` itself for the output.

    The anchor's own operands are sized by the role they play there, resolved
    to the nodes the kernel really loads.  A fused kernel loads more than them:
    the tail streams a residual or a mask of its own, which has no role and is
    diced by the output block, the way the builder dices it.  The output is
    whatever fusion left of it -- a ``quantize_mx`` tail makes it a pair, an
    MHA relayout re-cuts ``N`` into heads -- diced by the same blocks.
    """
    anchor = get_anchor_node(node)
    _, c_tile, k_tile = tiling.tile_sizes
    tiles = _gemv_role_tiles(anchor, c_tile, k_tile)
    out_tile = tiles.pop("output")
    shapes = normalize_shape(
        anchor, tiles, bound_operands(node, node.meta.get("submodule"))
    )
    divisor = tuple(max(1, s // t) for s, t in zip(anchor.shape, out_tile))

    if node is not anchor:
        submod = node.meta["submodule"]
        bound = bound_operands(node, submod)
        # Key an anchor operand by the placeholder the kernel loads beneath
        # its expand / transpose / dequantize (what the planner allocates).
        # Whatever else that prologue reads (a dequantize's scale and zero
        # point) has no role; it takes the operand's tile scaled by shape.
        for n in list(shapes):
            if n.graph is not submod.graph:
                continue
            tile = shapes.pop(n)
            source = weight_transforms(n)[0]
            for p in _operand_placeholders(n):
                outer = bound.get(p, p)
                if outer in shapes or not require_allocation(outer):
                    continue
                shapes[outer] = (
                    tile
                    if p is source
                    else _prologue_tile(
                        tile, tuple(n.shape), tuple(outer.shape)
                    )
                )

        # What the tail brought of its own -- a residual, a mask -- has no role
        # and is diced by the output block, the way the builder dices it.
        for n in node.all_input_nodes:
            if n not in shapes and require_allocation(n):
                shapes[n] = compute_tiled_shape(tuple(n.shape), divisor)

    if tuple(output_shape(node)) == tuple(anchor.shape):
        shapes[node] = compute_output_tiled_shapes(node, divisor)
    else:
        # An MHA relayout re-cut ``N`` into ``(heads, head_dim)``, so
        # ``divisor`` -- indexed against the anchor's dims -- no longer lines
        # up.  The block count survives, and these shapes are only ever read
        # for their product.
        blocks = math.prod(divisor)
        values = node.value
        values = values if isinstance(values, (list, tuple)) else [values]
        out = [((math.prod(v.shape) + blocks - 1) // blocks,) for v in values]
        shapes[node] = out if len(out) > 1 else out[0]

    return shapes


def vector_operand_shapes(node, tiling):
    """Each buffer a vector op's kernel holds -> its tile under ``tiling``, a
    ``GridTiling`` over the op's output (its input for a reduction to one
    value): every allocated input, right-aligned onto the grid or read whole
    along the dims a reduction drops (``reduced_input_tiling``), and
    ``node`` itself for the output."""
    in_divisor = reduced_input_tiling(get_anchor_node(node), tiling.grid)
    shapes = {
        n: compute_tiled_shape(tuple(n.shape), in_divisor)
        for n in node.all_input_nodes
        if require_allocation(n)
    }
    shapes[node] = compute_output_tiled_shapes(node, tiling.grid)
    return shapes


def _pool_input_extent(tile, stride, dilation, kernel_size):
    """Input extent covered by ``tile`` consecutive pooling outputs."""
    return (tile - 1) * stride + dilation * (kernel_size - 1) + 1


def pool_operand_shapes(node, tiling):
    """Each buffer a pool's kernel holds -> its tile under ``tiling``, a
    ``GridTiling`` over ``(N, H, W, C)`` (the output's spatial dims) for a
    windowed pool and over ``(N, C)`` for an adaptive one, whose window spans
    its whole input.

    A windowed pool's input tile is the halo its outputs read, from the
    stride and dilation (padding does not change the footprint).  The tiles
    take the node's own layout, NHWC for the layout twins (``quantized_ops``)
    and NCHW for the aten ops, so the fit sizes what the planner allocates.
    The geometry comes off the anchor, which also resolves the halo to the
    node the kernel really loads, while whatever a fused tail loads of its
    own is diced by the output block, the way the builder dices it.
    """
    anchor = get_anchor_node(node)
    nhwc = anchor.target in NHWC_OP_VARIANTS.values()
    if "adaptive" in str(anchor.target):
        tile_N, tile_C = tiling.tile_sizes
        if nhwc:  # (N, H, W, C)
            H_in, W_in = anchor.args[0].shape[1], anchor.args[0].shape[2]
            H_out, W_out = anchor.shape[1], anchor.shape[2]
            in_tile = (tile_N, H_in, W_in, tile_C)
            out_tile = (tile_N, H_out, W_out, tile_C)
        else:  # (N, C, H, W)
            H_in, W_in = anchor.args[0].shape[2], anchor.args[0].shape[3]
            H_out, W_out = anchor.shape[2], anchor.shape[3]
            in_tile = (tile_N, tile_C, H_in, W_in)
            out_tile = (tile_N, tile_C, H_out, W_out)
    else:
        tile_N, tile_H, tile_W, tile_C = tiling.tile_sizes

        stride = _pair(get_arg_value(anchor, 2, "stride", 1))
        dilation = _pair(get_arg_value(anchor, 4, "dilation", 1))
        kernel_size = _pair(get_arg_value(anchor, 1, "kernel_size"))

        tile_H_in = _pool_input_extent(
            tile_H, stride[0], dilation[0], kernel_size[0]
        )
        tile_W_in = _pool_input_extent(
            tile_W, stride[1], dilation[1], kernel_size[1]
        )

        if nhwc:  # (N, H, W, C)
            in_tile = (tile_N, tile_H_in, tile_W_in, tile_C)
            out_tile = (tile_N, tile_H, tile_W, tile_C)
        else:  # (N, C, H, W)
            in_tile = (tile_N, tile_C, tile_H_in, tile_W_in)
            out_tile = (tile_N, tile_C, tile_H, tile_W)

    shapes = normalize_shape(
        anchor,
        {"input": in_tile},
        bound_operands(node, node.meta.get("submodule")),
    )
    divisor = tuple(max(1, s // t) for s, t in zip(anchor.shape, out_tile))
    if node is not anchor:
        for n in node.all_input_nodes:
            if n not in shapes and require_allocation(n):
                shapes[n] = compute_tiled_shape(tuple(n.shape), divisor)
    shapes[node] = compute_output_tiled_shapes(node, divisor)
    return shapes


def attention_operand_shapes(node, tiling):
    """Each operand of a flash-attention ``node`` -> its FA3 SRAM tile under
    ``tiling``, an ``AttentionTiling`` of query / key tile lengths ``(tq,
    tkv)``: the query, the (transposed) key, the value, the output
    (``node``), the mask when there is one, and under MX each operand's
    block scales."""
    tq, tkv = tiling.tile_sizes
    query, key, value = node.args[0], node.args[1], node.args[2]
    head_dim = query.value.shape[-1]
    tiles = {
        query: (tq, head_dim),
        key: (head_dim, tkv),
        value: (tkv, head_dim),
        node: (tq, head_dim),
    }
    mask = get_arg_value(node, 3, "attn_mask", None)
    if isinstance(mask, Node):
        tiles[mask] = (tq, tkv)
    block_size = node.kwargs.get("block_size")
    if block_size is not None:
        # A block wider than the head is one truncated block; along the
        # keys the tile is a whole number of blocks.
        head_blocks = math.ceil(head_dim / block_size)
        tiles[node.kwargs["query_scale"]] = (tq, head_blocks)
        tiles[node.kwargs["key_scale"]] = (head_blocks, tkv)
        tiles[node.kwargs["value_scale"]] = (tkv // block_size, head_dim)
    residual = get_arg_value(node, 7, "key_residual", None)
    if residual is not None:
        # The split cache's residual (``attention_v3``): its R
        # positions whole, K transposed like the main key's.
        length = residual.value.shape[-2]
        tiles[get_arg_value(node, 6, "query_residual")] = (tq, head_dim)
        tiles[residual] = (head_dim, length)
        tiles[get_arg_value(node, 8, "value_residual")] = (length, head_dim)
        tiles[get_arg_value(node, 9, "residual_mask")] = (tq, length)
    return tiles
