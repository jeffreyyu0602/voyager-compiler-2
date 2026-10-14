"""Trainium's runtime and fit model, ``TrainiumModel``.

What the shared tile search asks about one NeuronCore, answered from its
``TrainiumConfig``.  A tile lives in SBUF with its rows on the partitions and
its last dim free, and the translator runs each step of a sweep the way this
model prices it:

- a GEMM step ``[rows, depth] x [depth, cols]`` transposes its activation tile
  on TensorE one 128 x 128 block at a time and copies each block out of PSUM,
  runs one ``nc_matmul`` per row chunk, contraction chunk and moving-operand
  slice into PSUM, and evicts each finished slice to SBUF;
- a vector op makes its passes over the tile 128 rows at a time, on ScalarE
  for a transcendental and on VectorE otherwise;
- a DMA moves a tile in one ``dma_copy``: its issue, then its rows' bytes,
  packet overhead included, at the core's DMA bandwidth;
- a 2-D max pool puts one output row on each partition: partition ``p``
  holds the ``kH`` input rows output row ``p`` reads side by side on its free
  axis, loaded as one strip, and VectorE takes ``kH - 1`` vertical then
  ``kW - 1`` horizontal maxes.

An MX (microscaling) tile is stored contraction-major: 4 consecutive
contraction elements per 32-bit word, a 512-element chunk per 128
partitions, chunks side by side on the free dim, and one E8M0 scale byte per
32 elements in a tile of the same layout, filled only on each quadrant's
first 4 partitions.  An MX GEMM step runs one ``nc_matmul_mx`` per 512-K
chunk with no transposes, and an op ending in ``quantize_mx`` transposes its
result on TensorE 128 columns at a time before quantizing each chunk.

The engines run concurrently, so a step costs its busiest engine, and the
shared schedule overlaps a step's DMAs with the compute around it
(``sweep_cycles``).  Times are nanoseconds here and TensorE cycles to the
search.
"""

import math
from collections import Counter
from dataclasses import dataclass
from functools import partial

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    LAYER_NORM_OPS,
    NORM_OPS,
    RMS_NORM_OPS,
    SOFTMAX_OPS,
    _pair,
    activation_prologue,
    csr_quantize_node,
    get_anchor_node,
    get_arg_value,
    is_compute_op,
    is_conv2d,
    is_fully_connected,
    is_gemm_op,
    is_pooling,
    tail_targets,
)

# The bufferizer's own rules for which nodes become kernels and which reuse
# another's build.
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _bufferize_key,
    _is_compute,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    fused_operand_specs,
    gemm_batch,
    gemv_operand_shapes,
    get_dtype_width,
    layer_key,
    node_dtype_bits,
    operand_bytes,
    outlier_stats,
    output_dtype,
    pool_operand_shapes,
    tail_extra_passes,
    tail_keeps_shape,
    vector_operand_shapes,
    weight_batch,
    weight_k_major,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    GridSpace,
    MatrixSpace,
    OpModel,
    Placement,
    RuntimeModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    INPUT_DIMS,
    OUTPUT_DIMS,
    WEIGHT_DIMS,
    batch_loads,
    gemv_transfers,
    l3_steps,
    operand_roles,
    sweep_cycles,
    tile_extent,
    total_extent,
    vector_transfers,
)

le = interstellar.le

QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
MAX_POOL2D = torch.ops.aten.max_pool2d.default

# An MX word packs this many consecutive contraction elements
# (``float8_e4m3fn_x4``), so one ``nc_matmul_mx`` contracts the partitions
# times as many: 512 on 128 partitions.
MX_PACK = 4
# Trn3's MX block, 8 partitions of 4 elements.
MX_BLOCK = 32
# A quadrant of 32 partitions holds 4 blocks, whose scales ``nc_matmul_mx``
# reads off the quadrant's first 4 partitions: a scale tile moves in one DMA
# of 4 rows per quadrant.
QUADRANT = 32
QUADRANT_SCALES = QUADRANT * MX_PACK // MX_BLOCK
# The MX arrays of a GEMM's operands by role: the axis they contract along
# (the activation's last, the K-major weight's first) and whether they are
# scales.
MX_ROLES = {
    "input": (-1, False),
    "input_scale": (-1, True),
    "weight": (-2, False),
    "weight_scale": (-2, True),
}

# Ops ScalarE evaluates through its activation table; VectorE runs every
# other arithmetic op.
SCALAR_OPS = frozenset(
    (
        torch.ops.aten.exp.default,
        torch.ops.aten.log.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.rsqrt.default,
    )
)

# The engines of the passes a softmax makes over its rows: the row maximum
# (VectorE), the exponentials and their sum (ScalarE), the scaling by the
# sum's reciprocal (VectorE).  A layer norm makes three VectorE passes: its
# rows' moments, then ``(x - mean) * weight`` and ``* rstd + bias``, or with
# one affine operand the normalization and that operand.  A norm without an
# affine makes two, so it is priced one pass high.  An RMS norm makes two:
# its rows' moments, then ``x * rstd * weight``.
SOFTMAX_PASSES = ("vector", "scalar", "vector")
LAYER_NORM_PASSES = ("vector", "vector", "vector")
RMS_NORM_PASSES = ("vector", "vector")

# The ops that reduce each row of their input, so a tile holds whole rows.
ROW_OPS = NORM_OPS | SOFTMAX_OPS


def sbuf_bytes(rows, cols, bits, partitions):
    """SBUF a ``rows x cols`` tile of ``bits``-wide elements takes: its rows
    spread over the partitions, each partition holding a whole row block."""
    row_bytes = math.ceil(cols * bits / 8)
    return partitions * math.ceil(rows / partitions) * row_bytes


def target_engines(target):
    """The engines of the passes an op of ``target`` makes over its tile, in
    order."""
    if target in SOFTMAX_OPS:
        return SOFTMAX_PASSES
    if target in LAYER_NORM_OPS:
        return LAYER_NORM_PASSES
    if target in RMS_NORM_OPS:
        return RMS_NORM_PASSES
    return ("scalar" if target in SCALAR_OPS else "vector",)


def _tail_engines(targets):
    """The engine of each pass ops of ``targets`` make over a tile, in order;
    a ``quantize_mx`` is left to ``quantize_mx_busy``."""
    return tuple(
        engine
        for target in targets
        if target is not QUANTIZE_MX
        for engine in target_engines(target)
    )


def _chunks(n, size):
    """The extents of ``n`` cut into pieces of at most ``size``."""
    return [min(size, n - start) for start in range(0, n, size)]


def _rows_cols(shape):
    """A tile shape as ``(rows, cols)``: its last dim is free, and every
    leading dim stacks onto the partitions."""
    if not shape:
        return 1, 1
    return math.prod(shape[:-1]), shape[-1]


def _bits(node):
    """The widest element among ``node``'s outputs."""
    bits = node_dtype_bits(node)
    return max(bits) if isinstance(bits, list) else bits


def _buffers(shape, node):
    """``(tile shape, element bits)`` per output a tile of ``node`` holds:
    one, or one per output of a multi-output op."""
    if shape and isinstance(shape[0], (tuple, list)):
        return list(zip(shape, node_dtype_bits(node)))
    return [(shape, _bits(node))]


def _traffic(moves):
    """DRAM bytes ``moves`` -- ``(buffer, tile shape, transfers)`` -- carry."""
    return sum(count * operand_bytes(shape, n) for n, shape, count in moves)


def _group_ops(node):
    """The arithmetic ops ``node`` runs: its fused submodule's, or itself."""
    submod = node.meta.get("submodule")
    ops = submod.graph.nodes if submod is not None else [node]
    return [n for n in ops if n.op == "call_function" and is_compute_op(n)]


def _quantizes_mx(node):
    """Whether ``node``'s ops end in a ``quantize_mx``, so its output is the
    pair (scales, codes), quantized along its last dim."""
    ops = _group_ops(node)
    return bool(ops) and ops[-1].target is QUANTIZE_MX


def _multiplies_mx(node):
    """Whether ``node``'s GEMM takes MX operands."""
    anchor = get_anchor_node(node) or node
    return (anchor.kwargs.get("block_size") or 1) > 1


def _mx_extents(shape, axis, block):
    """``(depth, other)`` of an MX tile ``shape``: the contraction it spans
    along ``axis``, whose elements each stand for ``block`` (32 for a scale
    tile, 1 for codes), and its elements along every other dim."""
    return shape[axis] * block, math.prod(shape) // shape[axis]


def _whole_chunks(depth, partitions):
    """Whether an MX GEMM contracting ``depth`` runs whole 512-element
    chunks, one ``nc_matmul_mx`` each."""
    return depth > 0 and depth % (partitions * MX_PACK) == 0


def _quantizable(extent, partitions):
    """Whether ``quantize_mx`` writes ``extent`` contraction elements in
    place: whole chunks, or whole quadrants inside one chunk (an attention
    head's 128)."""
    chunk = partitions * MX_PACK
    return (
        extent > 0
        and extent % (QUADRANT * MX_PACK) == 0
        and (extent % chunk == 0 or chunk % extent == 0)
    )


def _op_passes(node):
    """``(engine, share, bits)`` per pass a vector op's arithmetic makes over
    its tile: ``share`` is the fraction of the tile's columns it covers (an
    op on a reduced statistic covers one) and ``bits`` its widest tensor.  A
    ``quantize_mx`` is left to ``quantize_mx_busy``."""
    ops = [op for op in _group_ops(node) if op.target is not QUANTIZE_MX]

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
        bits = max(_bits(n) for n in operands)
        passes += [(e, share, bits) for e in target_engines(op.target)]
    return passes


class TrainiumModel(RuntimeModel):
    """One NeuronCore: TensorE, VectorE and ScalarE beside SBUF and PSUM,
    and DMA to HBM.

    NKI allocates SBUF and PSUM itself, so a tiling that fits takes an empty
    ``Placement``.

    Attributes:
        matrix_space: How a GEMM is mapped: a 128 x 128 array, SBUF as one
            byte pool and HBM.  L1 holds no tile of its own -- the
            translator cuts each step into instructions itself -- so its
            blockings are pinned to 1 and the search maps the PE partition,
            the SBUF tile and the HBM grid.  The reduction stays innermost at
            L3, where the builders accumulate it.
        grid_space: How a vector op or a matrix-vector GEMM is cut: lanes of
            1, since a tile's rows go on the partitions and its last dim is
            free, and every tiling that fits is priced.
    """

    def __init__(self, config, accumulate_fp32):
        super().__init__(config, accumulate_fp32)
        p, q = config.pe_array_size
        architecture = interstellar.Resource(
            buf_capacity_list=[
                [1, 1, 1],
                # Unbounded: L1 only passes the PE tile through.
                [math.inf, math.inf, math.inf],
                [config.usable_scratchpad_size],
                [config.dram_size * 1024**3],  # GB -> bytes
            ],
            buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100], [1000]],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            para_count_list=[p * q, 1, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            mac_capacity=0,
            partition_mode=[0, 0, 0, 0],
            invalid_underutilized=False,
            bank_size_list=[None, None, None, None],
        )
        unit = {"blocking_size": 1, "partitioning_size": 1}
        hints = {
            loop: {"level1": dict(unit)} for loop in ("OX", "OY", "OC", "ON")
        }
        hints["IC"] = {
            "level0": {"order": 1, "partitioning_size": p},
            "level1": dict(unit),
            "level3": {"order": 0},
        }
        hints["OC"]["level0"] = {"order": 0, "partitioning_size": q}
        for loop in ("FX", "FY"):
            hints[loop] = {f"level{i}": dict(unit) for i in range(4)}
        schedule_data = interstellar.extract_input.extract_schedule_info(
            {"schedule_hint": hints}, 4
        )
        schedule = interstellar.Schedule(
            schedule_data["schedule_hint"], schedule_data["partition_loops"]
        )
        self.matrix_space = MatrixSpace(architecture, schedule)
        self.grid_space = GridSpace(lanes=1, ranked=True)

    def prepare(self, node):
        """The model of ``node``, bare or fused: a ``TrainiumGemm`` for a
        GEMM, a ``TrainiumGemv`` for a matrix-vector one, which the grid
        search tiles, a ``TrainiumPool`` for an NCHW 2-D max pool and a
        ``TrainiumVector`` for a vector op.

        Raises:
            NotImplementedError: ``node`` is a convolution or another pool,
                which do not lower to Trainium.
        """
        anchor = get_anchor_node(node)
        if anchor.target is MAX_POOL2D:
            return TrainiumPool(node, self)
        if is_conv2d(anchor) or is_pooling(anchor):
            raise NotImplementedError(f"{anchor}: not lowered on Trainium")
        if not is_gemm_op(anchor):
            return TrainiumVector(node, self)
        if is_fully_connected(anchor):
            return TrainiumGemv(node, self)
        return TrainiumGemm.from_node(node, self)

    def _engine_ns(self, engine, elements, bits):
        """One instruction on ``engine`` over ``elements`` per partition of
        ``bits``-wide values."""
        config = self.config
        if engine == "scalar":
            rate, ghz = config.scalar_elements, config.scalar_ghz
        elif bits <= 16:
            rate, ghz = config.vector_elements_16, config.vector_ghz
        else:
            rate, ghz = config.vector_elements_32, config.vector_ghz
        cycles = math.ceil(elements / rate) + config.instruction_overhead_cycles
        return cycles / ghz

    def _dma_ns(self, rows, cols, bits):
        """One ``dma_copy`` of a ``rows x cols`` tile of ``bits``-wide
        elements."""
        config = self.config
        row_bytes = math.ceil(cols * bits / 8)
        moved = rows * (row_bytes + config.dma_row_overhead_bytes)
        return config.dma_issue_ns + moved / config.dram_bandwidth

    def _mx_layout(self, depth, other):
        """``(partitions, words)`` of an MX tile contracting ``depth`` beside
        ``other`` elements: a word per 4 contraction elements, the first 512
        on the partitions and every further chunk of 512 beside them."""
        p = self.config.partitions
        chunks = math.ceil(depth / (p * MX_PACK))
        return min(p, math.ceil(depth / MX_PACK)), chunks * other

    def _mx_sbuf(self, depth, other, scales):
        """SBUF bytes of an MX codes tile, or of its ``scales``: one byte per
        word's place, though only each quadrant's first 4 partitions hold
        one."""
        partitions, words = self._mx_layout(depth, other)
        bits = 8 if scales else 32
        return sbuf_bytes(partitions, words, bits, self.config.partitions)

    def _mx_dma_ns(self, depth, other, scales):
        """One transfer of an MX codes tile in one ``dma_copy``, or of its
        ``scales`` in one per quadrant."""
        partitions, words = self._mx_layout(depth, other)
        if not scales:
            return self._dma_ns(partitions, words, 32)
        quadrants = math.ceil(partitions / QUADRANT)
        return quadrants * self._dma_ns(QUADRANT_SCALES, words, 8)

    def _buffer_sbuf(self, n, shape, mx):
        """SBUF bytes of buffer ``n``'s tile ``shape``: one array per output,
        rows on the partitions, or the MX layout ``mx``
        (``TrainiumGrid._mx_tiles``) gives it."""
        if n in mx:
            return sum(self._mx_sbuf(*array) for array in mx[n])
        p = self.config.partitions
        return sum(
            sbuf_bytes(*_rows_cols(s), bits, p)
            for s, bits in _buffers(shape, n)
        )

    def _buffer_dma_ns(self, n, shape, mx):
        """One transfer of buffer ``n``'s tile ``shape``: a ``dma_copy`` per
        output, or the MX layout's transfers."""
        if n in mx:
            return sum(self._mx_dma_ns(*array) for array in mx[n])
        return sum(
            self._dma_ns(*_rows_cols(s), bits) for s, bits in _buffers(shape, n)
        )

    def quantize_mx_busy(self, rows, cols):
        """Each engine's busy time quantizing a ``rows x cols`` tile, rows on
        the partitions, to MX along its columns: per 128 rows and 512
        columns, TensorE transposes every 4th column into PSUM 4 times,
        VectorE copies each out into a staging tile and runs one
        ``quantize_mx`` over it."""
        config = self.config
        overhead = config.instruction_overhead_cycles
        chunks = math.ceil(cols / (config.partitions * MX_PACK))
        busy = Counter()
        for r in _chunks(rows, config.partitions):
            transposes = MX_PACK * chunks
            busy["tensor"] += transposes * (r + overhead) / config.frequency
            busy["vector"] += transposes * self._engine_ns("vector", r, 16)
            busy["vector"] += chunks * self._engine_ns(
                "vector", MX_PACK * r, 16
            )
        return busy

    def passes_busy(self, passes, rows, cols):
        """Each engine's busy time making ``passes`` -- ``(engine, share,
        bits)`` -- over a ``rows x cols`` tile, 128 rows per instruction."""
        row_chunks = math.ceil(rows / self.config.partitions)
        busy = Counter()
        for engine, share, bits in passes:
            elements = math.ceil(cols * share)
            busy[engine] += row_chunks * self._engine_ns(engine, elements, bits)
        return busy

    def gemm_step_ns(
        self,
        rows,
        depth,
        cols,
        bits,
        weight_k_major,
        tail_engines=(),
        tail_quantizes_mx=False,
        num_k=1,
        mx=False,
        prologue_engines=(),
    ):
        """One GEMM step's compute, ``[rows, depth] x [depth, cols]``: the
        busiest engine.

        TensorE transposes each 128 x 128 block of the activation -- and of
        a weight stored ``[cols, depth]`` -- and multiplies the activation
        into PSUM one moving slice at a time; VectorE copies the transposed
        blocks out of PSUM and evicts the finished slices.  FP32 operands
        multiply at ``fp32_matmul_rate`` of the 16-bit rate and transpose at
        ``fp32_transpose_rate``.  MX operands are stored contraction-major,
        so they need no transposes, and each ``nc_matmul_mx`` contracts 512.
        The tail's passes run over an output tile once its reduction ends,
        so each of its ``num_k`` steps carries a share of them; the
        prologue's passes run over the activation tile on every step.

        Args:
            rows: Rows of the activation and output tiles.
            depth: The contraction.
            cols: Columns of the weight and output tiles.
            bits: ``(activation, weight, output)`` element widths.
            weight_k_major: The weight tile lands ``[depth, cols]``, the
                contraction on the partitions.
            tail_engines: The engine of each pass the tail's arithmetic
                makes over an output tile, in order (``_tail_engines``).
            tail_quantizes_mx: The tail runs a ``quantize_mx``.
            num_k: Steps of the reduction.
            mx: The operands are MX.
            prologue_engines: The engine of each pass the prologue's
                arithmetic makes over an activation tile, in order
                (``_tail_engines``).
        """
        config = self.config
        act_bits, weight_bits, out_bits = bits
        overhead = config.instruction_overhead_cycles
        widths = _chunks(cols, config.moving_free_max)
        tensor_cycles = 0.0
        busy = Counter()
        if mx:
            rate = 1.0
            depth_chunks = math.ceil(depth / (config.partitions * MX_PACK))
            transposes = []
        else:
            width = max(act_bits, weight_bits)
            if width <= 8:
                rate = config.fp8_matmul_speedup
            elif width <= 16:
                rate = 1.0
            else:
                rate = config.fp32_matmul_rate
            depth_chunks = math.ceil(depth / config.partitions)
            transposes = [
                (r, act_bits) for r in _chunks(rows, config.partitions)
            ]
            if not weight_k_major:
                transposes += [
                    (c, weight_bits) for c in _chunks(cols, config.partitions)
                ]
        for extent, element_bits in transposes:
            if element_bits > 16:
                extent_cycles = extent / config.fp32_transpose_rate
            else:
                extent_cycles = extent
            tensor_cycles += depth_chunks * (extent_cycles + overhead)
            busy["vector"] += depth_chunks * self._engine_ns(
                "vector", extent, element_bits
            )
        for _ in _chunks(rows, config.pe_array_size[1]):
            tensor_cycles += depth_chunks * sum(
                w / rate + overhead for w in widths
            )
            busy["vector"] += sum(
                self._engine_ns("vector", w, 32) for w in widths
            )
        busy["tensor"] = tensor_cycles / config.frequency
        tail = [(e, 1.0, out_bits) for e in tail_engines]
        tail_busy = self.passes_busy(tail, rows, cols)
        if tail_quantizes_mx:
            tail_busy.update(self.quantize_mx_busy(rows, cols))
        for engine, ns in tail_busy.items():
            busy[engine] += ns / num_k
        prologue = [(e, 1.0, act_bits) for e in prologue_engines]
        busy.update(self.passes_busy(prologue, rows, depth))
        return max(busy.values())

    def sweep_ns(self, dmas, steps, compute):
        """A sweep of ``steps`` grid steps of ``compute`` each, moving
        ``dmas`` -- ``(ns, transfers)`` per buffer, the output first --
        overlapped by the shared schedule when double-buffered; the first
        DMA's latency and the kernel's fixed cost come on top."""
        config = self.config
        if config.double_buffered_l2:
            sweep = sweep_cycles(dmas, steps, compute)
        else:
            sweep = sum(ns * count for ns, count in dmas) + steps * compute
        return sweep + config.dram_access_latency + config.kernel_overhead_ns


@dataclass(frozen=True)
class TrainiumGemm(OpModel):
    """A GEMM under ``TrainiumModel``: the SBUF fit of each tile interstellar
    tries and the runtime of each mapping it finds.

    Judged in a forked search worker, so it holds plain values only, read
    off the graph when it is prepared (``from_node``), and dispatches no
    torch op.

    Attributes:
        space: ``TrainiumModel.matrix_space``.
        key: What the search is cached under, the same facts on every
            backend: the layer (``layer_key``), its output dtype, its fused
            tail's operands and ops (by name), whether that tail takes passes
            of its own and the scratch regions they may take, its outlier
            CSRs, and its prologue's ops (by name) and operands.
        model: The ``TrainiumModel``, which prices each step.
        bits: ``(activation, weight, output)`` element widths.
        output_scale_bits: The output's block-scale width, 0 unless it is
            MX.
        bias_bits: The bias's element width, 0 without one.
        block_size: The operands' MX block, 1 unless they are MX.
        batch: Batch elements the builder loops outside the mapping.
        weight_batch: Distinct weight tiles over that loop (a GQA repeat
            reads one tile for several elements).
        weight_k_major: The weight tile lands ``[depth, cols]``, the
            contraction on the partitions.
        fused_specs: ``(dims, bits)`` per operand the fused tail adds
            (``fused_operand_specs``).
        has_tail: The GEMM fuses ops after it.
        tail_engines: The engine of each pass the tail's arithmetic makes
            over an output tile, in order (``_tail_engines``).
        tail_quantizes_mx: The tail runs a ``quantize_mx``.
        prologue_specs: ``(dims, bits)`` per operand the prologue computing
            the activation reads (``activation_prologue``), on the
            activation's loop dims; with any, they load in its place.
        prologue_engines: The engine of each pass the prologue's arithmetic
            makes over an activation tile, in order (``_tail_engines``).
        whole_depth: The prologue runs a row op, so the contraction is never
            split.
        row_width: The output's last dim when the tail runs a row op, which
            an output tile then spans whole; 0 otherwise.
    """

    space: MatrixSpace
    key: tuple
    model: TrainiumModel
    bits: tuple
    output_scale_bits: int
    bias_bits: int
    block_size: int
    batch: int
    weight_batch: int
    weight_k_major: bool
    fused_specs: tuple
    has_tail: bool
    tail_engines: tuple
    tail_quantizes_mx: bool
    prologue_specs: tuple
    prologue_engines: tuple
    whole_depth: bool
    row_width: int

    @classmethod
    def from_node(cls, node, model):
        """The ``TrainiumGemm`` of a GEMM ``node``, bare or fused, under
        ``model``."""
        anchor = get_anchor_node(node)
        out_dtype = output_dtype(node)
        mx_out = isinstance(out_dtype, (list, tuple))
        values_dtype = out_dtype[-1] if mx_out else out_dtype
        out_bits = (
            get_dtype_width(values_dtype)
            if values_dtype
            else node_dtype_bits(anchor)
        )
        fused_specs = tuple(fused_operand_specs(node, anchor))
        has_tail = node.meta.get("submodule") is not None
        tail_ops = tail_targets(node)
        single_k, split_k = tail_extra_passes(node)
        prologue = activation_prologue(node)
        prologue_ops = tuple(n.target for n in prologue)
        # A prologue operand tiles along the activation's loop dims it
        # spans, right-aligned on them, its broadcast dims dropped.
        act_dims = [le.ON] * (anchor.args[0].value.ndim - 2) + [le.OX, le.IC]
        operands = dict.fromkeys(
            p
            for op in prologue
            for p in op.all_input_nodes
            if p.op == "placeholder" and p.value.numel() != 1
        )
        prologue_specs = tuple(
            (
                tuple(
                    d
                    for d, size in zip(act_dims[-len(p.shape) :], p.shape)
                    if size > 1
                ),
                node_dtype_bits(p),
            )
            for p in operands
        )
        key = layer_key(anchor) + (
            tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
            fused_specs,
            has_tail,
            tuple(str(target) for target in tail_ops),
            single_k,
            split_k,
            tail_keeps_shape(node),
            # The scratch regions a split reduction's extra pass may take:
            # one in a CSR-producing nest, else two.
            2 if split_k and csr_quantize_node(node) is None else 1,
            *outlier_stats(node),
            tuple(str(target) for target in prologue_ops),
            prologue_specs,
        )
        return cls(
            space=model.matrix_space,
            key=key,
            model=model,
            bits=(
                node_dtype_bits(anchor.args[0]),
                node_dtype_bits(anchor.args[1]),
                out_bits,
            ),
            output_scale_bits=get_dtype_width(out_dtype[-2]) if mx_out else 0,
            bias_bits=node_dtype_bits(
                get_arg_value(anchor, 2, "bias", None), 0
            ),
            block_size=anchor.kwargs.get("block_size") or 1,
            batch=gemm_batch(anchor),
            weight_batch=weight_batch(anchor),
            weight_k_major=weight_k_major(anchor),
            fused_specs=fused_specs,
            has_tail=has_tail,
            tail_engines=_tail_engines(tail_ops),
            tail_quantizes_mx=QUANTIZE_MX in tail_ops,
            prologue_specs=prologue_specs,
            prologue_engines=_tail_engines(prologue_ops),
            whole_depth=any(t in ROW_OPS for t in prologue_ops),
            row_width=(
                anchor.value.shape[-1]
                if any(t in ROW_OPS for t in tail_ops)
                else 0
            ),
        )

    @staticmethod
    def _tile(extent):
        """``(rows, depth, cols)`` of the tile ``extent(loop)`` describes."""
        rows = extent(le.OX) * extent(le.OY) * extent(le.ON)
        return rows, extent(le.IC), extent(le.OC)

    @staticmethod
    def _prologue_tile(dims, rows, depth):
        """``(rows, cols)`` of a prologue operand's tile spanning ``dims``
        beside a ``rows x depth`` activation tile."""
        return (rows if le.OX in dims else 1, depth if le.IC in dims else 1)

    def fit(self, mapping):
        """``Placement()`` when the L2 tile fits SBUF, else ``None``.

        The tile's SBUF bytes are the activation's, the weight's with its
        bias and the output's with the fused tail's operands and a split
        reduction's float32 accumulator, each in its pipeline slots, and they
        must fit the usable scratchpad; a prologue's operands take the
        activation's place.  MX operands and an MX output take their codes
        and scales in the MX layout.  A tile the translator cannot lower
        fits at no size: one not in the MX layout's whole chunks
        (``_whole_chunks``, ``_quantizable``), one splitting the contraction
        under a prologue row op, or one splitting the output's rows under a
        tail row op.
        """
        model = self.model
        config = model.config
        p = config.partitions
        act_bits, weight_bits, out_bits = self.bits

        def slots(dims, distinct):
            tiles = distinct * math.prod(
                mapping.loop_blockings[d][3] for d in dims
            )
            return config.num_slots if tiles > 1 else 1

        rows, depth, cols = self._tile(partial(total_extent, mapping, level=2))
        mx_in = self.block_size > 1
        mx_out = self.output_scale_bits > 0
        split_k = mapping.loop_blockings[le.IC][3] > 1
        if (mx_in and not _whole_chunks(depth, p)) or (
            mx_out and not _quantizable(cols, p)
        ):
            return None
        if (self.whole_depth and split_k) or (
            self.row_width and cols != self.row_width
        ):
            return None
        if mx_in:
            act = model._mx_sbuf(depth, rows, False)
            act += model._mx_sbuf(depth, rows, True)
            weight = model._mx_sbuf(depth, cols, False)
            weight += model._mx_sbuf(depth, cols, True)
        else:
            weight_tile = (
                (depth, cols) if self.weight_k_major else (cols, depth)
            )
            act = sbuf_bytes(rows, depth, act_bits, p)
            weight = sbuf_bytes(*weight_tile, weight_bits, p)
        if mx_out:
            out = model._mx_sbuf(cols, rows, False)
            out += model._mx_sbuf(cols, rows, True)
        else:
            out = sbuf_bytes(rows, cols, out_bits, p)
        if self.prologue_specs:
            act = sum(
                slots(dims, self.batch)
                * sbuf_bytes(*self._prologue_tile(dims, rows, depth), bits, p)
                for dims, bits in self.prologue_specs
            )
        else:
            act *= slots(INPUT_DIMS, self.batch)
        weight += sbuf_bytes(1, cols, self.bias_bits, p)
        weight *= slots(WEIGHT_DIMS, self.weight_batch)
        out *= slots(OUTPUT_DIMS, self.batch)
        for dims, bits in self.fused_specs:
            tail_rows = rows if le.OX in dims else 1
            tail_cols = cols if le.OC in dims else 1
            out += slots(dims, self.batch) * sbuf_bytes(
                tail_rows, tail_cols, bits, p
            )
        if split_k and (self.has_tail or model.accumulate_fp32):
            out += sbuf_bytes(rows, cols, 32, p)
        if act + out + weight > config.usable_scratchpad_size:
            return None
        return Placement()

    def cost(self, mapping, placement):
        """The whole GEMM in TensorE cycles -- its L3 sweep, each step's
        compute overlapping the DMAs of the operands that change -- and, as
        ``compute``, one step of the GEMM alone, its prologue and tail left
        out.
        ``placement`` is the contract's."""
        model = self.model
        frequency = model.config.frequency
        act_bits, weight_bits, out_bits = self.bits
        rows, depth, cols = self._tile(partial(tile_extent, mapping, level=2))
        num_k = mapping.loop_blockings[le.IC][3]
        steps = l3_steps(mapping) * self.batch
        mx_in = self.block_size > 1
        compute = model.gemm_step_ns(
            rows,
            depth,
            cols,
            self.bits,
            self.weight_k_major,
            self.tail_engines,
            self.tail_quantizes_mx,
            num_k,
            mx_in,
            self.prologue_engines,
        )
        if mx_in:
            act_ns = model._mx_dma_ns(depth, rows, False)
            act_ns += model._mx_dma_ns(depth, rows, True)
            weight_ns = model._mx_dma_ns(depth, cols, False)
            weight_ns += model._mx_dma_ns(depth, cols, True)
        else:
            weight_tile = (
                (depth, cols) if self.weight_k_major else (cols, depth)
            )
            act_ns = model._dma_ns(rows, depth, act_bits)
            weight_ns = model._dma_ns(*weight_tile, weight_bits)
        if self.bias_bits:
            weight_ns += model._dma_ns(1, cols, self.bias_bits)
        if self.output_scale_bits:
            out_ns = model._mx_dma_ns(cols, rows, False)
            out_ns += model._mx_dma_ns(cols, rows, True)
        else:
            out_ns = model._dma_ns(rows, cols, out_bits)
        if self.prologue_specs:
            act_dmas = [
                (
                    model._dma_ns(
                        *self._prologue_tile(dims, rows, depth), bits
                    ),
                    batch_loads(
                        mapping,
                        dims,
                        self.batch,
                        self.batch if le.ON in dims else 1,
                    ),
                )
                for dims, bits in self.prologue_specs
            ]
        else:
            act_dmas = [
                (
                    act_ns,
                    batch_loads(mapping, INPUT_DIMS, self.batch, self.batch),
                )
            ]
        dmas = [
            (out_ns, steps // num_k),
            *act_dmas,
            (
                weight_ns,
                batch_loads(
                    mapping, WEIGHT_DIMS, self.batch, self.weight_batch
                ),
            ),
        ]
        for dims, bits in self.fused_specs:
            tail_rows = rows if le.OX in dims else 1
            tail_cols = cols if le.OC in dims else 1
            distinct = self.batch if le.ON in dims else 1
            dmas.append(
                (
                    model._dma_ns(tail_rows, tail_cols, bits),
                    batch_loads(mapping, dims, self.batch, distinct),
                )
            )
        latency = model.sweep_ns(dmas, steps, compute)
        gemm = model.gemm_step_ns(
            rows, depth, cols, self.bits, self.weight_k_major, mx=mx_in
        )
        return Cost(latency * frequency, compute=gemm * frequency)


class TrainiumGrid(OpModel):
    """A vector op or a matrix-vector GEMM under ``TrainiumModel``, cut into
    a grid of SBUF tiles by the shared grid search: the fit both share, of
    the tiles each kind of op cuts its buffers into (``operand_shapes``) and
    how often its sweep moves each (``transfers``).

    Judged in the parent, so it keeps its node, and reads what every
    candidate shares off the graph once, when it is prepared.

    Attributes:
        node: The op, bare or a fused ``call_module``.
        model: The ``TrainiumModel``.
        space: ``TrainiumModel.grid_space``.
        key: ``None``: a grid search is not shared.
        mx_operands: Each MX array a GEMM on MX operands loads -> ``(axis,
            scales)`` (``MX_ROLES``); empty for any other op.
        quantizes_mx: The op ends in a ``quantize_mx``, so its output is the
            pair (scales, codes).
    """

    key = None

    def __init__(self, node, model):
        self.node = node
        self.model = model
        self.space = model.grid_space
        roles = operand_roles(node) if _multiplies_mx(node) else {}
        self.mx_operands = {
            n: MX_ROLES[role]
            for n, (role, _) in roles.items()
            if role in MX_ROLES
        }
        self.quantizes_mx = _quantizes_mx(node)

    def _mx_tiles(self, shapes):
        """The MX tiles of the kernel, its buffers cut into ``shapes``:
        buffer -> ``(depth, other, scales)`` per array it holds
        (``_mx_extents``).  A GEMM on MX operands loads its activation and
        weight codes and their scales, contracting along the activation's
        last dim and the weight's first (K-major); an op ending in
        ``quantize_mx`` stores scales and codes, quantized along its last
        dim."""
        tiles = {}
        for n, (axis, scales) in self.mx_operands.items():
            shape = shapes.get(n)
            if shape is None:
                continue
            block = MX_BLOCK if scales else 1
            tiles[n] = [(*_mx_extents(shape, axis, block), scales)]
        if self.quantizes_mx:
            # Both take the codes' extents: a tile narrower than a block has
            # no scale column at all.
            depth, other = _mx_extents(shapes[self.node][1], -1, 1)
            tiles[self.node] = [(depth, other, True), (depth, other, False)]
        return tiles

    def _mx_lowerable(self, mx):
        """Whether the translator lowers the MX tiles ``mx``
        (``_mx_tiles``): the GEMM's operands contract whole chunks
        (``_whole_chunks``) and the op's own output is ``_quantizable``."""
        node = self.node
        p = self.model.config.partitions
        return all(
            (_quantizable(depth, p) if n is node else _whole_chunks(depth, p))
            for n, arrays in mx.items()
            for depth, _, _ in arrays
        )

    def fit(self, tiling):
        """``Placement()`` when the translator lowers the tile's MX arrays
        (``_mx_lowerable``) and every buffer fits SBUF, else ``None``.  A
        row op's statistics live in the translator's temporaries, inside the
        ``scratchpad_offset`` reserve.

        A buffer takes the slots the bufferizer gives it: two for one the
        sweep moves more than once, one for one it moves once and holds.
        """
        model = self.model
        config = model.config
        shapes = self.operand_shapes(tiling)
        mx = self._mx_tiles(shapes)
        if not self._mx_lowerable(mx):
            return None
        needed = 0
        for n, shape, count in self.transfers(shapes, tiling):
            slots = config.num_slots if count > 1 else 1
            needed += slots * model._buffer_sbuf(n, shape, mx)
        return Placement() if needed <= config.usable_scratchpad_size else None


class TrainiumVector(TrainiumGrid):
    """A vector op: each step makes the group's passes over the tile's
    largest buffer.

    Attributes:
        passes: ``(engine, share, bits)`` per pass the group's arithmetic
            makes over a tile (``_op_passes``).
    """

    def __init__(self, node, model):
        super().__init__(node, model)
        self.passes = _op_passes(node)

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``vector_operand_shapes``)."""
        return vector_operand_shapes(self.node, tiling)

    def transfers(self, shapes, tiling):
        """Each buffer the sweep moves, with its tile in ``shapes`` and how
        often (``vector_transfers``)."""
        return vector_transfers(self.node, shapes, tiling.grid)

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes.
        ``placement`` is the contract's."""
        model = self.model
        shapes = self.operand_shapes(tiling)
        buffers = [b for n, s in shapes.items() for b in _buffers(s, n)]
        rows, cols = max((_rows_cols(s) for s, _ in buffers), key=math.prod)
        busy = model.passes_busy(self.passes, rows, cols)
        if self.quantizes_mx:
            busy.update(model.quantize_mx_busy(rows, cols))
        mx = self._mx_tiles(shapes)
        moves = self.transfers(shapes, tiling)
        dmas = [(model._buffer_dma_ns(n, s, mx), c) for n, s, c in moves]
        compute = max(busy.values(), default=0.0)
        latency = model.sweep_ns(dmas, math.prod(tiling.grid), compute)
        return Cost(latency * model.config.frequency, _traffic(moves))


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
        self.bits = _bits(node)

    def _tiles(self, tiling):
        """``(rows, cols, in_cols)`` under ``tiling``: the output tile as
        rows and columns, and the halo's columns."""
        shapes = pool_operand_shapes(self.node, tiling)
        rows, cols = _rows_cols(shapes[self.node])
        return rows, cols, shapes[self.input][-1]

    def fit(self, tiling):
        """``Placement()`` when the output rows fit the partitions and, in
        their pipeline slots, the strip tile (``kH`` halo rows per output
        row), the output tile and the translator's vertical-max temporary
        fit SBUF, else ``None``."""
        config = self.model.config
        p = config.partitions
        rows, cols, in_cols = self._tiles(tiling)
        if rows > p:
            return None
        strips = sbuf_bytes(rows, self.kernel[0] * in_cols, self.bits, p)
        output = sbuf_bytes(rows, cols, self.bits, p)
        vertical = sbuf_bytes(rows, in_cols, self.bits, p)
        needed = config.num_slots * (strips + output + vertical)
        return Placement() if needed <= config.usable_scratchpad_size else None

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes: each step
        loads its strips and stores its output tile.  ``placement`` is the
        contract's."""
        model = self.model
        rows, cols, in_cols = self._tiles(tiling)
        kh, kw = self.kernel
        steps = math.prod(tiling.grid)
        compute = (kh - 1) * model._engine_ns("vector", in_cols, self.bits)
        compute += (kw - 1) * model._engine_ns("vector", cols, self.bits)
        dmas = [
            (model._dma_ns(rows, cols, self.bits), steps),
            (model._dma_ns(rows * kh, in_cols, self.bits), steps),
        ]
        latency = model.sweep_ns(dmas, steps, compute)
        moved = steps * rows * (cols + kh * in_cols) * self.bits // 8
        return Cost(latency * model.config.frequency, moved)


class TrainiumGemv(TrainiumGrid):
    """A matrix-vector GEMM: an ``(X, C, K)`` tile is a GEMM step of ``X``
    rows contracting ``C`` into ``K`` columns.

    Attributes:
        bits: ``(activation, weight, output)`` element widths.
        weight_k_major: The weight tile lands ``[depth, cols]``, the
            contraction on the partitions.
        tail_engines: The engine of each pass the tail's arithmetic makes
            over an output tile, in order (``_tail_engines``).
        tail_quantizes_mx: The tail runs a ``quantize_mx``.
        multiplies_mx: The GEMM takes MX operands.
    """

    def __init__(self, node, model):
        super().__init__(node, model)
        anchor = get_anchor_node(node)
        tail_ops = tail_targets(node)
        self.bits = (_bits(anchor.args[0]), _bits(anchor.args[1]), _bits(node))
        self.weight_k_major = weight_k_major(anchor)
        self.tail_engines = _tail_engines(tail_ops)
        self.tail_quantizes_mx = QUANTIZE_MX in tail_ops
        self.multiplies_mx = _multiplies_mx(node)

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``gemv_operand_shapes``)."""
        return gemv_operand_shapes(self.node, tiling)

    def transfers(self, shapes, tiling):
        """Each buffer the sweep moves, with its tile in ``shapes`` and how
        often (``gemv_transfers``)."""
        return gemv_transfers(self.node, shapes, tiling.grid)[1]

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes.
        ``placement`` is the contract's."""
        model = self.model
        rows, depth, cols = tiling.tile_sizes
        compute = model.gemm_step_ns(
            rows,
            depth,
            cols,
            self.bits,
            self.weight_k_major,
            self.tail_engines,
            self.tail_quantizes_mx,
            tiling.grid[1],
            self.multiplies_mx,
        )
        shapes = self.operand_shapes(tiling)
        mx = self._mx_tiles(shapes)
        grid, moves = gemv_transfers(self.node, shapes, tiling.grid)
        dmas = [(model._buffer_dma_ns(n, s, mx), c) for n, s, c in moves]
        latency = model.sweep_ns(dmas, math.prod(grid), compute)
        return Cost(latency * model.config.frequency, _traffic(moves))


def kernel_nodes(gm):
    """The nodes the bufferizer turns into kernels, one each and in order,
    with the key under which it reuses the build of an identical one.  Taken
    before ``compile``, which replaces them."""
    return [(n, _bufferize_key(n)) for n in gm.graph.nodes if _is_compute(n)]


def predicted_seconds(nodes, config):
    """The latency the tile search predicted for each kernel of ``nodes``
    (``kernel_nodes``): the cost of its winning tiling, stamped as
    ``meta["tile_result"]`` on a GEMM's anchor and on a GEMV or vector op
    itself.  A kernel whose build was reused takes the prediction of the one
    it reuses.

    Returns:
        Seconds per kernel, in order.
    """
    priced, predicted = {}, []
    for node, key in nodes:
        anchor = get_anchor_node(node) or node
        result = anchor.meta.get("tile_result") or node.meta.get("tile_result")
        if result is not None:
            seconds = result.cost.latency / (config.frequency * 1e9)
            priced.setdefault(key, seconds)
        predicted.append(priced[key])
    return predicted
