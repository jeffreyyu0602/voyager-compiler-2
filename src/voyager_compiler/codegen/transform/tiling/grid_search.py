"""The grid tile search: vector, GEMV and pool ops.

Each such op is cut into a grid of tiles, one dim at a time
(``get_valid_tiling``), under the builders' rules: dims kept whole, whole
quantization blocks, the MHA head, a constraint's pins.  The backend's
``OpModel`` judges each candidate (``RuntimeModel.prepare``): ``fit`` places
it on chip, at the least extra bank sharing it needs (``Placement.tier``),
and ``cost`` prices it.  The candidates at the lowest tier compete, and the
winner is stamped on the node as ``meta["tile_result"]``.
"""

import itertools
import logging
import math
from functools import partial
from typing import Generator, Optional, Tuple

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    DYNAMIC_QUANTIZE_OPS,
    NORM_OPS,
    SOFTMAX_OPS,
    get_anchor_node,
    get_arg_value,
    is_axis_reduction,
    is_bmm,
    is_elementwise_op,
    is_full_reduction,
    is_pooling,
    weight_is_ck,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    mha_head,
    output_shape,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    GridTiling,
    TileResult,
)
from voyager_compiler.codegen.transform.tiling.tiler import (
    DEFAULT_RUNTIME_TOLERANCE,
    pick,
)
from voyager_compiler.ops.layout import NHWC_OP_VARIANTS

logger = logging.getLogger(__name__)

__all__ = [
    "gemv_op_tiling",
    "pool_op_tiling",
    "vector_op_tiling",
]

# Which ``(X, C, K)`` tile a GEMV's search sizes each interstellar loop by:
# the rows, the input features (the reduction) and the output features.
_GEMV_LOOP_TILE = {
    interstellar.le.OX: 0,
    interstellar.le.IC: 1,
    interstellar.le.OC: 2,
}


def get_valid_tiling(
    input_shape: Tuple[int, ...],
    multiple_of: Optional[Tuple[int, ...]] = None,
    order: Optional[Tuple[int, ...]] = None,
    last_dim: Optional[int] = None,
    pinned: Optional[dict] = None,
    exhaustive: bool = False,
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    """Yield tile shapes from the full shape downwards.

    Reduces one dimension at a time to its next valid size, in ``order``,
    leaving the dimensions already reduced at their smallest; ``exhaustive``
    instead yields every combination of valid sizes.  A size is valid when it
    divides the dimension and is a multiple of ``multiple_of``.

    Args:
        input_shape: The shape being tiled.
        multiple_of: Required tile multiple per dim, right-aligned to the shape
            and padded with 1s (1 is free).
        order: Dim indices to reduce, first reduced first; default is left to
            right.
        last_dim: Dims from here onwards stay at full size.
        pinned: ``{dim: size}`` tiles held at one size while the other dims
            reduce; nothing is yielded when a pinned size is not valid.
        exhaustive: Yield the whole grid of valid sizes, ignoring ``order``.

    Yields:
        ``(tile_shape, tiling_factors)``, largest tile first.
    """
    ndim = len(input_shape)

    def resolve(i):
        return i + ndim if i < 0 else i

    fixed = (
        set(range(resolve(last_dim), ndim)) if last_dim is not None else set()
    )
    traversal = [resolve(i) for i in order] if order else list(range(ndim))

    multiples = list(multiple_of) if multiple_of else []
    multiples = [1] * (ndim - len(multiples)) + multiples

    # Valid tile sizes per dim, largest first.  A fixed dim has only one.
    sizes = {}
    for dim, extent in enumerate(input_shape):
        if dim in fixed:
            sizes[dim] = [extent]
            continue
        valid = [
            s
            for s in range(extent, 0, -1)
            if extent % s == 0 and s % multiples[dim] == 0
        ]
        if not valid:
            logger.warning(
                f"No valid tiling for dim {dim} (size={extent}, "
                f"multiple_of={multiples[dim]}); keeping full size."
            )
            valid = [extent]
        sizes[dim] = valid
    for dim, size in (pinned or {}).items():
        dim = resolve(dim)
        if input_shape[dim] % size or size % multiples[dim]:
            return
        sizes[dim] = [size]

    if exhaustive:
        for tile in itertools.product(*(sizes[dim] for dim in range(ndim))):
            yield tile, tuple(e // t for e, t in zip(input_shape, tile))
        return

    tile = [sizes[dim][0] for dim in range(ndim)]

    def current():
        return tuple(tile), tuple(e // t for e, t in zip(input_shape, tile))

    yield current()
    for dim in traversal:
        for size in sizes[dim][1:]:  # a fixed dim has nothing left to give
            tile[dim] = size
            yield current()


def _search_grid(op, full_shape, tolerance=0.0, **limits):
    """The winning tiling of ``full_shape`` as a ``TileResult``, or ``None``
    when no tiling fits on chip.

    Walks ``get_valid_tiling`` once, keeping the tilings ``op`` fits at the
    lowest tier seen: the least extra bank sharing any of them needs.  A
    tiling that fits at a tier fits at every higher one, so these are the
    tilings the least sharing admits.  An unranked space
    (``GridSpace.ranked``) takes the first of them along the walk -- the
    largest -- unpriced; a ranked one walks every combination of sizes,
    since a larger tile is not always a faster one, prices the kept tilings
    and takes the ``pick``.

    Args:
        op: The op's ``OpModel``; its ``space`` is a ``GridSpace``.
        full_shape: The shape being tiled.
        tolerance: How much slower than the fastest a tiling may be and
            still win on traffic (``pick``).
        **limits: ``get_valid_tiling``'s ``multiple_of``, ``order``,
            ``last_dim`` and ``pinned``: the builders' rules.
    """
    ranked = op.space.ranked
    tier, kept = math.inf, []  # the lowest tier seen, its fits in order
    for tile_sizes, grid in get_valid_tiling(
        full_shape, exhaustive=ranked, **limits
    ):
        tiling = GridTiling(tile_sizes, grid)
        placement = op.fit(tiling)
        if placement is None or placement.tier > tier:
            continue
        if placement.tier < tier:
            tier, kept = placement.tier, []
        kept.append((tiling, placement))
        if not ranked and tier == 0:
            break
    if not kept:
        return None
    if not ranked:
        tiling, placement = kept[0]
        return TileResult(tiling, placement, cost=None)
    return pick(
        [
            TileResult(tiling, placement, op.cost(tiling, placement))
            for tiling, placement in kept
        ],
        tolerance,
    )


def gemv_op_tiling(node, op, constraint):
    """Per-dim tile counts for a matrix-vector GEMM, bare or fused.

    Called from ``get_tiling`` during bufferization, so it sees the layout the
    transform settled on and the operands fusion left on the kernel.  The
    backend's model gave the GEMM a ``GridSpace`` (``RuntimeModel.prepare``),
    so this is the whole tile search for one.  The winner is stamped as
    ``node.meta["tile_result"]``.

    Args:
        node: The op to tile -- a fully-connected GEMM, or a fused
            ``call_module`` around one.
        op: Its ``OpModel``, which ``get_tiling`` prepared to route it here.
        constraint: A ``TileConstraint`` (``tiler.py``) on the rows
            (``le.OX``), input-feature (``le.IC``) or output-feature
            (``le.OC``) tile, or ``None``.

    Returns:
        ``(n_m, n_n, n_k)`` -- the M, output and reduction tile counts.

    Raises:
        RuntimeError: when no tiling of its operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    input_shape = anchor.args[0].shape
    X = input_shape[-2] if is_bmm(anchor) else math.prod(input_shape[:-1])
    C = input_shape[-1]
    weight_shape = anchor.args[1].shape
    K = weight_shape[-1] if weight_is_ck(anchor) else weight_shape[-2]

    # A C tile short of a whole microscaling block leaves the scale tile
    # with no elements to expand against the input, and an N tile that cuts
    # a head in half cannot be stored in an MHA relayout (``mha_head``).
    block_size = anchor.kwargs.get("block_size") or 1
    multiple = [1, block_size, math.lcm(op.space.lanes, mha_head(node) or 1)]
    pinned = {}
    if constraint is not None:
        for loop, value in constraint.exact:
            pinned[_GEMV_LOOP_TILE[loop]] = value
        for loop, value in constraint.multiple:
            dim = _GEMV_LOOP_TILE[loop]
            multiple[dim] = math.lcm(multiple[dim], value)

    logger.info(f"Running L2 tiling for GEMV: {node}")

    result = _search_grid(
        op,
        (X, C, K),
        DEFAULT_RUNTIME_TOLERANCE,
        multiple_of=tuple(multiple),
        pinned=pinned,
    )
    if result is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )
    node.meta["tile_result"] = result

    x_tiled, c_tiled, k_tiled = result.tiling.tile_sizes
    return X // x_tiled, K // k_tiled, C // c_tiled


def _vector_op_tiling_limits(node, vector_unit_width):
    """``(last_dim, multiple_of)`` for a vector op's tile search.

    Args:
        node: The vector op whose dimensions are being constrained.
        vector_unit_width: Lanes the last dim has to fill.
    """
    # A tile of the output reads each reduced dim whole; the kept dims tile
    # freely, the last in whole lane groups.
    if is_axis_reduction(node):
        return None, (vector_unit_width if node.shape[-1] > 1 else 1,)
    # A reduction to one value tiles its input the same way.
    if is_full_reduction(node):
        return None, (vector_unit_width if node.args[0].shape[-1] > 1 else 1,)

    # Certain dimensions cannot be tiled, e.g., transpose and reduction dims
    last_dim = -1
    multiple_of = (vector_unit_width,)
    dequantize = node.target == torch.ops.quantized_ops.dequantize.default
    if node.target in SOFTMAX_OPS:
        last_dim = get_arg_value(node, 1, "dim", -1)
    elif node.target in NORM_OPS:
        normalized_shape = get_arg_value(node, 1, "normalized_shape", None)
        last_dim = (
            -len(normalized_shape) if normalized_shape is not None else -1
        )
    elif node.target in DYNAMIC_QUANTIZE_OPS or (
        dequantize and get_arg_value(node, 4, "block_size", None)
    ):
        axes_arg = 3 if dequantize else 2
        axes = get_arg_value(node, axes_arg, "axes", None)
        block_size = get_arg_value(node, axes_arg + 1, "block_size", None)
        ndim = len(node.args[0].shape)

        # A quantization block must not straddle a tile boundary, so a tile on a
        # quantization axis holds a whole number of blocks; the last dim also
        # respects the hardware unroll.  A dequantize keeps its last dim whole,
        # as any elementwise op does.
        last_dim = -1 if dequantize else None
        axes = set(a % ndim for a in (axes or ()))
        multiple_of = tuple(
            (
                math.lcm(block_size if i in axes else 1, vector_unit_width)
                if i == ndim - 1
                else block_size if i in axes else 1
            )
            for i in range(ndim)
        )
    elif node.target == torch.ops.aten.transpose.int:
        last_dim = min(*node.args[1:])
    elif node.target == torch.ops.aten.permute.default:
        last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)

    return last_dim, multiple_of


def vector_op_tiling(node, tiler):
    """Per-dim tile counts for a vector op, bare or fused.

    Called from the builders during bufferization, so it sees whatever fusion
    left behind however it ran.  A fused submodule is searched as a whole: the
    backend's model fits and prices every operand the kernel loads, not just
    the anchor's.  The anchor only supplies the dim constraints.  The winner
    is stamped as ``node.meta["tile_result"]``.

    Args:
        node: The op to tile -- a vector op, or a fused ``call_module`` around
            one.  Its output shape drives the grid.
        tiler: The ``Tiler``; its model prepares the op
            (``RuntimeModel.prepare``).

    Returns:
        Tile counts over ``node``'s output -- over its input for a reduction
        to one value (``is_full_reduction``) -- or ``None`` when it has no
        anchor.

    Raises:
        RuntimeError: when no tiling of the op's operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    if anchor is None:
        return None
    op = tiler.model.prepare(node)
    last_dim, multiple_of = _vector_op_tiling_limits(anchor, op.space.lanes)

    logger.info(f"Running L2 tiling for vector op: {node}")

    # A reduction to one value has no output to tile: every tile of its input
    # folds into that one value.
    shape = (
        anchor.args[0].shape
        if is_full_reduction(anchor)
        else output_shape(node)
    )
    search = partial(_search_grid, op, shape, multiple_of=multiple_of)
    result = search(last_dim=last_dim)
    # An elementwise op whose whole rows do not fit splits its rows too, in
    # whole lane groups.
    if result is None and is_elementwise_op(anchor) and last_dim is not None:
        result = search(last_dim=None)
    if result is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )
    node.meta["tile_result"] = result
    return tuple(s // ts for s, ts in zip(shape, result.tiling.tile_sizes))


def pool_op_tiling(node, tiler):
    """Per-dim tile counts for a pooling op, bare or fused.

    Called from ``build_pool`` during bufferization, so it sees whatever fusion
    left behind.  A fused submodule is searched as a whole: the backend's model
    sizes the pool's halo and whatever the tail loads of its own
    (``pool_operand_shapes``).  The winner is stamped as
    ``node.meta["tile_result"]``.

    Args:
        node: The op to tile -- a pooling op, or a fused ``call_module`` around
            one.
        tiler: The ``Tiler``; its model prepares the op
            (``RuntimeModel.prepare``).

    Returns:
        Tile counts over ``(N, H_out, W_out, C)`` for a non-adaptive pool or
        ``(N, C)`` for an adaptive one, or ``None`` when the op is not a pool
        this search tiles.

    Raises:
        RuntimeError: when no tiling of the op's operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    if anchor is None or not is_pooling(anchor):
        return None

    # The NHWC twins keep the name of the aten op they replace (see layout_ops)
    # and differ only in namespace, so matching on the op name covers both.
    name = str(anchor.target)
    nhwc = anchor.target in NHWC_OP_VARIANTS.values()

    if name.endswith("max_pool2d.default"):
        if nhwc:
            N, H_out, W_out, C = anchor.shape
        else:
            N, C, H_out, W_out = anchor.shape
        logger.info(f"Running L2 tiling for non-adaptive pool op: {node}")
        full_shape = (N, H_out, W_out, C)
        order = (3, 0, 1, 2)
    elif "adaptive" in name:
        full_shape = (anchor.shape[0], anchor.shape[-1 if nhwc else 1])
        logger.info(f"Running L2 tiling for adaptive pool op: {node}")
        order = (1, 0)
    else:
        return None

    op = tiler.model.prepare(node)
    # The channels, the last dim, fill whole lane groups.
    multiple_of = (1,) * (len(full_shape) - 1) + (op.space.lanes,)
    result = _search_grid(op, full_shape, multiple_of=multiple_of, order=order)
    if result is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )
    node.meta["tile_result"] = result
    return tuple(s // ts for s, ts in zip(full_shape, result.tiling.tile_sizes))
