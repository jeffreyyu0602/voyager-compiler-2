"""Trainium's vector ops and pools: the fit and cost of a vector op
(``TrainiumVector``) and of an NCHW 2-D max pool (``TrainiumPool``), both
built by the shared builders.

- A vector op makes its passes over the tile 128 rows at a time, on ScalarE
  for a transcendental and on VectorE otherwise; a K-major RMS norm squares
  its blocks on ScalarE, sums them across the partitions on TensorE and
  scales them on VectorE, 512 columns at a time (``rms_norm_kmajor_busy``).
- A vector op lays its tile out as the translator does: a flat op -- an
  elementwise chain whose operands are each the result's size or one
  element, or an RMS norm alone on one row -- cuts the tile in row-major
  order across the partitions (``spread_partitions``) and runs on every
  one, a norm summing its partitions' squares by a ones-matmul
  (``rms_norm_spread_busy``); any other op puts its rows on the
  partitions, a one-row operand copied onto each partition the op's rows
  take.
- A 2-D max pool puts one output row on each partition: partition ``p``
  holds the ``kH`` input rows output row ``p`` reads side by side on its
  free axis, loaded as one strip, and VectorE takes ``kH - 1`` vertical
  then ``kW - 1`` horizontal maxes.
"""

import math
from collections import Counter

import torch

from voyager_compiler.codegen.node_info import (
    _pair,
    get_anchor_node,
    get_arg_value,
    require_allocation,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    pool_operand_shapes,
    vector_operand_shapes,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    OpModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import vector_transfers

from .utils import (
    ACCUMULATORS,
    F32,
    LOCAL,
    PARTITIONS,
    QUANTIZE_MX,
    RMS_NORM_OPS,
    ROW_OPS,
    ROW_PASS_BITS,
    SRAM,
    TrainiumGrid,
    TrainiumPlacement,
    arithmetic_ops,
    chain_ops,
    chain_tiles,
    extents,
    fits_psum,
    group_ops,
    group_tiles,
    moved_bytes,
    output_buffers,
    rows_cols,
    sbuf_bytes,
    sbuf_placed_bytes,
    scratch_regions,
    spread_partitions,
    target_engines,
    widest_bits,
)

RMS_NORM_KMAJOR = torch.ops.quantized_ops.rms_norm_kmajor.default
# The 128-row blocks of a K-major RMS norm's tile ScalarE squares in one
# instruction.
SQUARE_BLOCKS = 8


def rms_norm_kmajor_tiles(hidden, dtype, cols, free, post):
    """The tiles the translator computes a K-major RMS norm over ``hidden``
    rows of ``dtype`` in (``translate.Kernel._rms_norm_kmajor``), on a tile
    ``cols`` wide ``free`` columns at a time (a matmul's moving free size),
    its result read by the chain ``post`` (``chain_ops``) if any: a ones
    tile and the PSUM banks its sums land in, the weight's float32 columns,
    ``SQUARE_BLOCKS`` blocks' squares and the rstd, and the float32 result
    the chain after it reads, with that chain's."""
    blocks = math.ceil(hidden / PARTITIONS)
    tiles = {
        ("one", PARTITIONS, dtype, 1, SRAM),
        ("acc", free, F32, ACCUMULATORS, LOCAL),
        ("wc", blocks, F32, 1, SRAM),
        ("sq", SQUARE_BLOCKS * free, dtype, 2, SRAM),
        ("rq", free, F32, 2, SRAM),
    }
    if post:
        tiles.add(("rw", free, F32, 2, SRAM))
        for width in extents(cols, free):
            tiles |= chain_tiles(post, width)
    return tiles


def rms_norm_kmajor_busy(model, rows, cols, hidden):
    """Each engine's busy time on a K-major RMS norm of a ``rows x cols``
    tile, ``rows`` stacking blocks of ``hidden``, the normalized dim, on
    the partitions.  Per ``hidden`` rows and 512 columns: ScalarE squares
    ``SQUARE_BLOCKS`` 128-row blocks an instruction, TensorE sums them
    across the partitions by a ones-matmul per block accumulating into
    PSUM, VectorE takes the mean square plus ``eps`` and ScalarE its
    ``rsqrt``, then VectorE scales each block by it and the weight.
    VectorE first copies the weight to float32 columns, once."""
    config = model.config
    blocks = math.ceil(hidden / config.partitions)
    busy = Counter()
    for w in extents(cols, config.moving_free_max):
        for n in extents(blocks, SQUARE_BLOCKS):
            busy["scalar"] += model.engine_ns("scalar", n * w, 32)
        busy["tensor"] += (
            blocks * (w + config.tensor_overhead_cycles) / config.frequency
        )
        busy["vector"] += model.engine_ns("vector", w, 32)
        busy["scalar"] += model.engine_ns("scalar", w, 32)
        busy["vector"] += blocks * model.engine_ns("vector", w, ROW_PASS_BITS)
    lead = math.ceil(rows / hidden)
    busy = Counter({engine: lead * ns for engine, ns in busy.items()})
    busy["vector"] += model.engine_ns("vector", blocks, 32)
    return busy


# Ops the translator runs alone, not as an elementwise chain
# (``translate.Kernel._compute``).
NOT_FLAT = (
    torch.ops.aten.amax.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.eye.default,
    QUANTIZE_MX,
    RMS_NORM_KMAJOR,
)


def rms_norm_spread_tiles(free):
    """The tiles the translator computes an RMS norm of a row cut ``free``
    elements to a partition in (``translate.Kernel._rms_norm_spread``): a
    float32 ones tile and the PSUM column the partitions' total lands in,
    the squares, each partition's sum and the rstd."""
    return {
        ("one", PARTITIONS, F32, 1, SRAM),
        ("acc", 1, F32, 2, LOCAL),
        ("sq", free, F32, 2, SRAM),
        ("ss", 1, F32, 2, SRAM),
        ("rs", 1, F32, 2, SRAM),
    }


def rms_norm_spread_busy(model, free):
    """Each engine's busy time on an RMS norm of a row cut ``free`` elements
    to a partition: ScalarE squares each partition's elements, summing
    them on the way, TensorE totals the sums onto every partition by a
    ones-matmul, VectorE takes the mean square plus ``eps``, ScalarE its
    rsqrt, and VectorE scales the row by it and the weight.  Each waits on
    the one before."""
    config = model.config
    return Counter(
        {
            "scalar": model.engine_ns("scalar", free, ROW_PASS_BITS)
            + model.engine_ns("scalar", 1, ROW_PASS_BITS),
            "tensor": (1 + config.tensor_overhead_cycles) / config.frequency,
            "vector": model.engine_ns("vector", 1, ROW_PASS_BITS)
            + model.engine_ns("vector", free, ROW_PASS_BITS),
        }
    )


def _op_passes(node):
    """``(engine, share, bits)`` per pass a vector op's arithmetic makes over
    its tile: ``share`` is the fraction of the tile's columns it covers (an
    op on a reduced statistic covers one) and ``bits`` its widest tensor.  A
    ``quantize_mx`` is left to ``quantize_mx_busy`` and a K-major RMS norm
    to ``rms_norm_kmajor_busy``."""
    ops = [
        op
        for op in arithmetic_ops(node)
        if op.target not in (QUANTIZE_MX, RMS_NORM_KMAJOR)
    ]

    def tensors(op):
        return [
            n
            for n in (op, *op.all_input_nodes)
            if isinstance(getattr(n, "value", None), torch.Tensor)
        ]

    full = max((n.value.numel() for op in ops for n in tensors(op)), default=1)
    passes = []
    for op in ops:
        operands = tensors(op)
        if not operands:
            continue
        share = max(n.value.numel() for n in operands) / full
        bits = max(widest_bits(n) for n in operands)
        if op.target in ROW_OPS:
            bits = max(bits, ROW_PASS_BITS)
        passes += [(e, share, bits) for e in target_engines(op.target)]
    return passes


class TrainiumVector(TrainiumGrid):
    """A vector op: each step makes the group's passes over the tile's
    largest buffer.

    Attributes:
        passes: ``(engine, share, bits)`` per pass the group's arithmetic
            makes over a tile (``_op_passes``).
        kmajor_hidden: The normalized dim of the K-major RMS norm the group
            runs (``rms_norm_kmajor_busy``), 0 without one.
        kmajor_dtype: That RMS norm's element type.
        kmajor_post: The chain (``chain_ops``) after that RMS norm.
        group: The group of ops the translator runs (``group_ops``).
        flat: The translator may cut a tile of the group across the
            partitions (``row_parts``): it is an elementwise chain, or an
            RMS norm alone (on one row).
        inputs: The shapes of the group's stored inputs
            (``require_allocation``; a lookup table is not one), read only
            for a ``flat`` group.
    """

    def __init__(self, node, model):
        super().__init__(node, model)
        self.passes = _op_passes(node)
        norm = next(
            (op for op in arithmetic_ops(node) if op.target is RMS_NORM_KMAJOR),
            None,
        )
        self.kmajor_hidden = norm.args[0].value.shape[-2] if norm else 0
        self.kmajor_dtype = norm.args[0].value.dtype if norm else None
        submod = node.meta.get("submodule")
        ops = [node]
        if submod is not None:
            ops = [n for n in submod.graph.nodes if n.op == "call_function"]
        self.kmajor_post = chain_ops(ops[ops.index(norm) + 1 :]) if norm else ()
        self.group = group_ops(ops)
        row_op = self.group[1]
        self.flat = (
            not any(op.target in NOT_FLAT for op in ops)
            and (row_op is None or (row_op in RMS_NORM_OPS and len(ops) == 1))
        )
        self.inputs = [
            tuple(n.value.shape)
            for n in node.all_input_nodes
            if self.flat and require_allocation(n)
        ]

    def row_parts(self, out_tile):
        """``(parts, spread)`` for an output tile ``out_tile``: the
        partitions the translator lays the kernel's tiles on, and whether it
        cuts them across them.  A flat op -- an RMS norm only on one row --
        cuts its tiles evenly across the most partitions it can
        (``spread_partitions``), an operand of one element copied onto each;
        any other puts its rows on the partitions, a one-row operand copied
        onto each partition its rows take.  A flat op spreads only when each
        stored input's tile -- its shape cut to ``out_tile``, a broadcast
        dim staying 1 -- is the output tile's size or one element."""
        rows, cols = rows_cols(out_tile)
        size = rows * cols
        parts = spread_partitions(size)
        flat = self.flat and all(
            math.prod(min(d, t) for d, t in zip(s[::-1], out_tile[::-1]))
            in (1, size)
            for s in self.inputs
        )
        if flat and (rows == 1 or self.group[1] is None) and parts > 1:
            return parts, True
        return min(rows, PARTITIONS), False

    def placement(self, tiling, shapes, moves):
        """The kernel's ``TrainiumPlacement``, its tiles cut into ``shapes``
        under ``tiling``: no buffers -- the shared builder allocates none --
        and the regions the translator computes the op in (``scratch``),
        ``None`` when they do not fit PSUM.  ``moves`` is the contract's."""
        scratch = self.scratch(shapes[self.node])
        if not fits_psum(self.model.config, (), scratch):
            return None
        return TrainiumPlacement(scratch=scratch)

    def scratch(self, out_tile):
        """The regions the translator computes the op in, for an output tile
        ``out_tile`` (one per output for several), as the op group ``op``'s
        (``scratch_regions``): a K-major RMS norm's
        (``rms_norm_kmajor_tiles``), an RMS norm's of a row cut across the
        partitions (``rms_norm_spread_tiles``), or the group's over the
        tile's rows, a tile so cut ``row_parts`` elements to a partition
        (``group_tiles``)."""
        if self.kmajor_hidden:
            tiles = rms_norm_kmajor_tiles(
                self.kmajor_hidden,
                self.kmajor_dtype,
                out_tile[-1],
                self.model.config.moving_free_max,
                self.kmajor_post,
            )
        else:
            if out_tile and isinstance(out_tile[0], (tuple, list)):
                # A quantize_mx's output: (scales, codes).
                out_tile = out_tile[-1]
            parts, spread = self.row_parts(out_tile)
            cols = math.prod(out_tile) // parts if spread else out_tile[-1]
            if spread and self.group[1] is not None:
                tiles = rms_norm_spread_tiles(cols)
            else:
                tiles = group_tiles(self.group, cols)
        return scratch_regions("op", tiles)

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``vector_operand_shapes``)."""
        return vector_operand_shapes(self.node, tiling)

    def transfers(self, shapes, tiling):
        """Each buffer the sweep moves, with its tile in ``shapes`` and how
        often (``vector_transfers``)."""
        return vector_transfers(self.node, shapes, tiling.grid)

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes, the tile
        laid out as ``row_parts`` says: an RMS norm of a row cut across
        the partitions makes its instructions one after another
        (``rms_norm_spread_busy``).  ``placement`` is the contract's."""
        model = self.model
        shapes = self.operand_shapes(tiling)
        buffers = [b for n, s in shapes.items() for b in output_buffers(s, n)]
        rows, cols = max((rows_cols(s) for s, _ in buffers), key=math.prod)
        out_tile = output_buffers(shapes[self.node], self.node)[-1][0]
        parts, spread = self.row_parts(out_tile)
        if spread:
            rows, cols = parts, rows * cols // parts
        mx = self._mx_tiles(shapes)
        moves = self.transfers(shapes, tiling)
        steps = math.prod(tiling.grid)
        stores = sum(c for n, _, c in moves if n is self.node)
        if spread and self.group[1] is not None:
            busy = rms_norm_spread_busy(model, cols)
            busy["scalar"] += model.store_issue_ns(stores, steps)
            compute = sum(busy.values())
        else:
            busy = model.passes_busy(self.passes, rows, cols)
            if self.quantizes_mx:
                busy.update(model.quantize_mx_busy(rows, cols))
            if self.kmajor_hidden:
                busy.update(
                    rms_norm_kmajor_busy(model, rows, cols, self.kmajor_hidden)
                )
            busy["scalar"] += model.store_issue_ns(stores, steps)
            compute = max(busy.values())
        dmas = [
            (
                model.buffer_dma_ns(
                    n,
                    s,
                    mx,
                    n is not self.node,
                    parts,
                    spread and math.prod(s) > 1,
                ),
                c,
            )
            for n, s, c in moves
        ]
        latency = model.sweep_ns(dmas, steps, compute)
        return Cost(latency * model.config.frequency, moved_bytes(moves))


class TrainiumPool(OpModel):
    """An NCHW 2-D max pool, one output row per partition: partition ``p``
    holds output row ``p``'s ``kH`` input rows side by side, so a tile has at
    most a partition's worth of output rows.

    Attributes:
        node: The pool, a bare op: nothing fuses into one.
        model: The ``TrainiumModel``.
        space: ``TrainiumModel.grid_space``.
        key: ``None``: a grid search is not shared.
        input: The node the pool reads, whose tile is the halo.
        kernel: ``(kH, kW)``.
        bits: The element width.
    """

    key = None

    def __init__(self, node, model):
        self.node = node
        self.model = model
        self.space = model.grid_space
        anchor = get_anchor_node(node)
        self.input = anchor.args[0]
        self.kernel = _pair(get_arg_value(anchor, 1, "kernel_size"))
        self.bits = widest_bits(node)

    def _tiles(self, tiling):
        """``(rows, cols, in_cols)`` under ``tiling``: the output tile as
        rows and columns, and the halo's columns."""
        shapes = pool_operand_shapes(self.node, tiling)
        rows, cols = rows_cols(shapes[self.node])
        return rows, cols, shapes[self.input][-1]

    def fit(self, tiling):
        """The kernel's ``TrainiumPlacement`` -- the two tiles the
        translator takes each strip's vertical max in (``pool.pv``) -- when
        the output rows fit the partitions and, in their pipeline slots, the
        strip tile (``kH`` halo rows per output row) and the output tile fit
        SBUF beside them, else ``None``."""
        config = self.model.config
        p = config.partitions
        rows, cols, in_cols = self._tiles(tiling)
        if rows > p:
            return None
        strips = sbuf_bytes(rows, self.kernel[0] * in_cols, self.bits, p)
        output = sbuf_bytes(rows, cols, self.bits, p)
        vertical = {("pv", in_cols, self.input.value.dtype, 2, SRAM)}
        scratch = scratch_regions("pool", vertical)
        needed = config.num_slots * (strips + output)
        needed += sbuf_placed_bytes(config, (), scratch)
        if needed > config.usable_scratchpad_size:
            return None
        return TrainiumPlacement(scratch=scratch)

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes: each step
        loads its strips and stores its output tile.  ``placement`` is the
        contract's."""
        model = self.model
        rows, cols, in_cols = self._tiles(tiling)
        kh, kw = self.kernel
        steps = math.prod(tiling.grid)
        compute = (kh - 1) * model.engine_ns("vector", in_cols, self.bits)
        compute += (kw - 1) * model.engine_ns("vector", cols, self.bits)
        dmas = [
            (model.dma_ns(rows, cols, self.bits), steps),
            (model.dma_ns(rows * kh, in_cols, self.bits), steps),
        ]
        latency = model.sweep_ns(dmas, steps, compute)
        moved = steps * rows * (cols + kh * in_cols) * self.bits // 8
        return Cost(latency * model.config.frequency, moved)
