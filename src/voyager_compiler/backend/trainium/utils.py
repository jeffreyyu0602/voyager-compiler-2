"""What Trainium's op files share.

``TrainiumPlacement``, the SBUF and PSUM bytes of the buffers a kernel is
placed (``fits_psum``, ``sbuf_placed_bytes``), the scratch regions the
translator computes an op group in (``scratch_regions``), the engines of an
op's passes, the facts of the MX layout and ``TrainiumGrid``, the fit a
vector op and a matrix-vector GEMM share.
"""

import math
from dataclasses import dataclass

import torch

from voyager_compiler.codegen.node_info import (
    LAYER_NORM_OPS,
    NORM_OPS,
    RMS_NORM_OPS,
    SOFTMAX_OPS,
    get_anchor_node,
    is_compute_op,
    is_nop,
)
from voyager_compiler.codegen.transform.bufferize import MemoryLevel
from voyager_compiler.codegen.transform.tiling.lowering import (
    node_dtype_bits,
    operand_bytes,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    OpModel,
    Placement,
)
from voyager_compiler.codegen.transform.tiling.traversal import operand_roles

# SBUF partitions: TensorE's rows, the same on every core.
PARTITIONS = 128
# neuronx-cc takes a ``double_row`` matmul operand ``[128, 2, X]`` only with
# its two 128-row chunks a multiple of this many elements apart.
DOUBLE_ROW_STEP = 16
# The largest step of an access pattern's free dim: neuronx-cc holds it in an
# int16.
AP_STEP_MAX = 2**15 - 1
# Bytes of each SBUF partition a kernel leaves free for the loads of the
# kernels after it, which the step compile starts under it: without them
# the step spills tiles of both.
HOISTED_SBUF = 48 << 10
# The memory levels a kernel's buffers are placed in: SBUF and PSUM.
SRAM, LOCAL = int(MemoryLevel.SRAM), int(MemoryLevel.LOCAL)

QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
SDPA_KMAJOR = torch.ops.quantized_ops.sdpa_kmajor.default

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
# sum's reciprocal (VectorE).  A row op's passes move one element a cycle
# whatever its input's width: a reduction does, and its other passes hold
# float32 (``ROW_PASS_BITS``).  A layer norm makes three VectorE passes: its
# rows' moments, then ``(x - mean) * weight`` and ``* rstd + bias``, or with
# one affine operand the normalization and that operand.  A norm without an
# affine makes two, so it is priced one pass high.  An RMS norm makes two:
# its rows' moments, then ``x * rstd * weight``; a K-major one is left to
# ``rms_norm_kmajor_busy`` in ``vector.py``.
SOFTMAX_PASSES = ("vector", "scalar", "vector")
LAYER_NORM_PASSES = ("vector", "vector", "vector")
RMS_NORM_PASSES = ("vector", "vector")

# The ops that reduce each row of their input, so a tile holds whole rows.
ROW_OPS = NORM_OPS | SOFTMAX_OPS
ROW_PASS_BITS = 32


def sbuf_bytes(rows, cols, bits, partitions):
    """SBUF a ``rows x cols`` tile of ``bits``-wide elements takes: its rows
    spread over the partitions, each partition holding a whole row block."""
    row_bytes = math.ceil(cols * bits / 8)
    return partitions * math.ceil(rows / partitions) * row_bytes


# The PSUM tiles a GEMM's matmul results rotate through, and those a
# transposing load's blocks and the translator's single-block transposes
# rotate through.
ACCUMULATORS = 4
TRANSPOSE_TILES = 4


def psum_bytes(config, buffers):
    """PSUM bytes per partition ``buffers`` take, ``(shape, bits, slots)``
    each: rows on the partitions, an element of under 16 bits two bytes
    apart, as TensorE writes an 8-bit transpose."""
    total = 0
    for shape, bits, slots in buffers:
        rows, cols = rows_cols(shape)
        blocks = math.ceil(rows / config.partitions)
        total += slots * blocks * cols * max(bits, 16) // 8
    return total


def footprints(buffers, scratch):
    """``(shape, bits, slots, space)`` of each of ``buffers``
    (``TrainiumPlacement.buffers``) and ``scratch`` (``Placement.scratch``),
    a buffer's element width the stored one."""
    return [
        (shape, bits, slots, space)
        for _, shape, _, bits, slots, space in buffers
    ] + [
        (shape, 8 * dtype.itemsize, slots, space)
        for _, shape, dtype, slots, space in scratch
    ]


def fits_psum(config, buffers, scratch):
    """Whether a kernel's ``buffers`` and ``scratch`` in PSUM fit the core,
    each slot within a bank (``footprints``)."""
    psum = [
        (shape, bits, slots)
        for shape, bits, slots, space in footprints(buffers, scratch)
        if space == LOCAL
    ]
    if any(
        psum_bytes(config, [(shape, bits, 1)]) > config.psum_bank_bytes
        for shape, bits, _ in psum
    ):
        return False
    return psum_bytes(config, psum) <= config.psum_bytes


def sbuf_placed_bytes(config, buffers, scratch):
    """SBUF bytes a kernel's ``buffers`` and ``scratch`` in SBUF take
    (``footprints``)."""
    p = config.partitions
    return sum(
        slots * sbuf_bytes(*rows_cols(shape), bits, p)
        for shape, bits, slots, space in footprints(buffers, scratch)
        if space == SRAM
    )


def half_bank_blocks(config, bits):
    """The 128-row blocks of ``bits``-wide elements TensorE transposes side
    by side into a PSUM tile of half a bank, so two such tiles share a
    bank: one copy moves them all out of PSUM.  An 8-bit transpose writes
    every other element, so an element takes 16 bits at least."""
    block = config.partitions * max(bits, 16) // 8
    return max(1, config.psum_bank_bytes // 2 // block)


@dataclass(frozen=True)
class TrainiumPlacement(Placement):
    """A tiling's placement on a NeuronCore: beside what every backend
    places, the buffers its builder allocates.

    Attributes:
        buffers: The buffers the kernel's builder allocates as placed, at
            every level -- SBUF tiles beside the operands' and the PSUM
            boxes its matmuls and transposes write: ``(role, shape, dtype,
            bits, slots, space)`` each, ``dtype`` the one it is allocated
            at, ``bits`` the width it is stored at (8 for FP8 values the
            graph carries in bfloat16), ``space`` the ``MemoryLevel``, one
            slot allocated without a slot dim.
    """

    buffers: tuple = ()


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


def pass_engines(targets):
    """The engine of each pass ops of ``targets`` make over a tile, in order;
    a ``quantize_mx`` is left to ``quantize_mx_busy``."""
    return tuple(
        engine
        for target in targets
        if target is not QUANTIZE_MX
        for engine in target_engines(target)
    )


def extents(n, size):
    """The extents of ``n`` cut into pieces of at most ``size``."""
    return [min(size, n - start) for start in range(0, n, size)]


def pair_distance(steps):
    """Chunks apart the two 128-deep chunks of a ``double_row`` matmul are,
    its operand tiles holding consecutive chunks ``steps`` elements apart:
    the fewest that put each operand's pair a multiple of
    ``DOUBLE_ROW_STEP`` elements apart, ``None`` when a pair that far apart
    is more than ``AP_STEP_MAX`` elements apart."""
    distance = max(
        DOUBLE_ROW_STEP // math.gcd(s, DOUBLE_ROW_STEP) for s in steps
    )
    if max(abs(s) for s in steps) * distance > AP_STEP_MAX:
        return None
    return distance


def contraction_runs(depth, distance):
    """The 128-deep chunks, ``(start, stop)``, each ``nc_matmul`` contracts
    along a contraction of ``depth``: given a ``distance`` (FP8 operands,
    ``pair_distance``), one ``double_row`` matmul per chunk of the first
    half of each whole block of ``2 * distance`` chunks, with the chunk
    ``distance`` after it; every other chunk, and every chunk without a
    ``distance``, alone."""
    chunks = [
        (lo, min(depth, lo + PARTITIONS)) for lo in range(0, depth, PARTITIONS)
    ]
    if distance is None:
        return [[c] for c in chunks]
    block = 2 * distance
    paired = depth // (block * PARTITIONS) * block
    pairs = [
        [chunks[b + c], chunks[b + c + distance]]
        for b in range(0, paired, block)
        for c in range(distance)
    ]
    return pairs + [[c] for c in chunks[paired:]]


def rows_cols(shape):
    """A tile shape as ``(rows, cols)``: its last dim is free, and every
    leading dim stacks onto the partitions."""
    if not shape:
        return 1, 1
    return math.prod(shape[:-1]), shape[-1]


def widest_bits(node):
    """The widest element among ``node``'s outputs."""
    bits = node_dtype_bits(node)
    return max(bits) if isinstance(bits, list) else bits


def output_buffers(shape, node):
    """``(tile shape, element bits)`` per output a tile of ``node`` holds:
    one, or one per output of a multi-output op."""
    if shape and isinstance(shape[0], (tuple, list)):
        return list(zip(shape, node_dtype_bits(node)))
    return [(shape, widest_bits(node))]


def moved_bytes(moves):
    """DRAM bytes ``moves`` -- ``(buffer, tile shape, transfers)`` -- carry."""
    return sum(count * operand_bytes(shape, n) for n, shape, count in moves)


def arithmetic_ops(node):
    """The arithmetic ops ``node`` runs: its fused submodule's, or itself."""
    submod = node.meta.get("submodule")
    ops = submod.graph.nodes if submod is not None else [node]
    return [n for n in ops if n.op == "call_function" and is_compute_op(n)]


def multiplies_mx(node):
    """Whether ``node``'s GEMM takes MX operands."""
    anchor = get_anchor_node(node) or node
    return (anchor.kwargs.get("block_size") or 1) > 1


# The relayouts a chain of the translator's only re-maps elements through.
CHAIN_RELAYOUTS = (torch.ops.aten.transpose.int, torch.ops.aten.view.default)
# The ops the translator's chains run with tiles of their own
# (``translate.Kernel._apply``).
DIV = torch.ops.aten.div.Tensor
WHERE = (
    torch.ops.aten.where.self,
    torch.ops.aten.where.ScalarSelf,
    torch.ops.aten.where.ScalarOther,
    torch.ops.aten.where.Scalar,
)
QUANTIZE = torch.ops.quantized_ops.quantize.default
SINES = (torch.ops.aten.sin.default, torch.ops.aten.cos.default)
# Columns of one vector instruction's chunk and elements per partition one
# ``bn_stats`` takes, as the translator cuts its ops.
CHUNK_FREE = 2048
STATS_FREE = 512
F32, BF16 = torch.float32, torch.bfloat16
# A column of one value per row, ``[128, 1]`` float32: a scalar read from
# DRAM, a column read as another type, a reciprocal.
SCALAR_COLUMNS = ("sc", 1, F32, 2, SRAM)


def spread_partitions(cols):
    """The partitions a flat row of ``cols`` elements is cut across: the
    most, up to 128, that cut it evenly (``translate.Layout.spread``)."""
    return max(p for p in range(1, PARTITIONS + 1) if cols % p == 0)


def scratch_name(kind, cols):
    """The name of an op's scratch tile of ``kind``, ``[128, cols]``, after
    its op group's prefix (``scratch_regions``); the translator reads the
    tile by it."""
    return f"{kind}_{cols}"


def scratch_regions(group, tiles):
    """``tiles`` -- ``(kind, columns, dtype, slots, space)`` each, a tile
    ``[128, columns]`` -- as the scratch regions of an op group ``group``
    (``Placement.scratch``), each named ``"<group>.<kind>_<columns>"``."""
    return tuple(
        sorted(
            (
                f"{group}.{scratch_name(kind, cols)}",
                (PARTITIONS, cols),
                dtype,
                slots,
                space,
            )
            for kind, cols, dtype, slots, space in tiles
        )
    )


def group_regions(regions, group):
    """The entries of ``regions`` -- name -> region -- of the op group
    ``group``, by name (``scratch_regions``)."""
    return {
        name: region
        for name, region in regions.items()
        if name.startswith(f"{group}.")
    }


def chain_ops(ops):
    """``(target, operand kinds)`` of each op of ``ops`` -- the FX nodes of
    one chain the translator runs, in order -- that computes: an operand
    another op of ``ops`` computes is a ``"value"``, one of a single
    element a ``"scalar"`` the kernel reads from DRAM, one value per row of
    a wider result a ``"column"`` (``"column32"`` stored in float32), any
    other a ``"tile"``.  A view, a relayout or a cast computes nothing."""
    chain = []
    for op in ops:
        if is_nop(op) or op.target in CHAIN_RELAYOUTS or not is_compute_op(op):
            continue
        kinds = []
        for arg in op.args:
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg in ops:
                kinds.append("value")
            elif arg.value.numel() == 1:
                kinds.append("scalar")
            elif arg.shape[-1] == 1 and op.shape[-1] > 1:
                kinds.append("column32" if widest_bits(arg) == 32 else "column")
            else:
                kinds.append("tile")
        chain.append((op.target, tuple(kinds)))
    return tuple(chain)


def chain_tiles(chain, width):
    """The tiles the translator computes ``chain`` (``chain_ops``) in, on
    128-row chunks ``width`` wide (``translate.Kernel._chain``), as
    ``(kind, columns, dtype, slots, space)``: a float32 pair for each result
    before the chain's last, bfloat16 for the one a ``quantize_mx`` reads,
    which also stages its transposes in SBUF and PSUM; columns for a
    scalar, a column of another type or a reciprocal of one; a float32 pair
    for a reciprocal of a tile or a ``where``'s terms; a bfloat16 pair for
    a quantized quotient and an int32 pair for a sine's turns."""
    tiles = set()
    targets = [target for target, _ in chain]
    for i, (target, kinds) in enumerate(chain):
        if target is QUANTIZE_MX:
            tiles |= {
                ("q", width, BF16, 2, SRAM),
                ("qs", MX_PACK * PARTITIONS, BF16, 2, SRAM),
                ("tp", PARTITIONS, BF16, TRANSPOSE_TILES, LOCAL),
            }
            continue
        if i < len(chain) - 1:
            quantized = targets[i + 1] is QUANTIZE_MX
            kind, dtype = ("q", BF16) if quantized else ("t", F32)
            tiles.add((kind, width, dtype, 2, SRAM))
        if {"scalar", "column"} & set(kinds):
            tiles.add(SCALAR_COLUMNS)
        if target in (DIV, QUANTIZE):
            # The reciprocal of the divisor or scale.
            if kinds[1] in ("tile", "value"):
                tiles.add(("t", width, F32, 2, SRAM))
            else:
                tiles.add(SCALAR_COLUMNS)
        if target is QUANTIZE:
            tiles.add(("qt", width, BF16, 2, SRAM))
        if target in WHERE:
            tiles.add(("t", width, F32, 2, SRAM))
            if kinds[0] not in ("tile", "value"):
                tiles.add(SCALAR_COLUMNS)
        if target in SINES:
            tiles |= {
                ("t", width, F32, 2, SRAM),
                ("k", width, torch.int32, 2, SRAM),
            }
    return tiles


def row_op_tiles(target, cols, post):
    """The tiles the translator computes a row op of ``target`` in, over
    rows ``cols`` wide (``translate.Kernel._rowwise``), its result read by
    the chain ``post`` (``chain_ops``) if any: a norm's statistics or a
    softmax's columns and float32 exponentials, and the float32 result the
    chain after it reads, with that chain's."""
    if target in SOFTMAX_OPS:
        tiles = {
            ("mx", 1, F32, 2, SRAM),
            ("sm", 1, F32, 2, SRAM),
            ("t", cols, F32, 2, SRAM),
        }
    else:
        pieces = math.ceil(cols / STATS_FREE)
        tiles = {
            ("st", 6 * pieces, F32, 2, SRAM),
            ("mv", 2, F32, 2, SRAM),
            ("rs", 1, F32, 2, SRAM),
        }
    if post:
        tiles |= {("rw", cols, F32, 2, SRAM)} | chain_tiles(post, cols)
    return tiles


def group_tiles(group, cols):
    """The tiles the translator computes ``group`` (``group_ops``) in, over
    rows ``cols`` wide: its chain's before its row op, if any, run 2048
    columns at a time (``chain_tiles``), and its row op's
    (``row_op_tiles``)."""
    chain, row_op, post = group
    tiles = set()
    for width in extents(cols, CHUNK_FREE):
        tiles |= chain_tiles(chain, width)
    if row_op is not None:
        tiles |= row_op_tiles(row_op, cols, post)
    return tiles


def group_ops(ops):
    """``(chain, row op, post)`` of the FX nodes ``ops`` a group runs, in
    order: the chain (``chain_ops``) before its row op -- all of them
    without one -- the row op's target, or ``None``, and the chain after
    it."""
    row = next((i for i, op in enumerate(ops) if op.target in ROW_OPS), None)
    if row is None:
        return chain_ops(ops), None, ()
    return chain_ops(ops[:row]), ops[row].target, chain_ops(ops[row + 1 :])


def _mx_extents(shape, axis, block):
    """``(depth, other)`` of an MX tile ``shape``: the contraction it spans
    along ``axis``, whose elements each stand for ``block`` (32 for a scale
    tile, 1 for codes), and its elements along every other dim."""
    return shape[axis] * block, math.prod(shape) // shape[axis]


def whole_chunks(depth, partitions):
    """Whether an MX GEMM contracting ``depth`` runs whole 512-element
    chunks, one ``nc_matmul_mx`` each."""
    return depth > 0 and depth % (partitions * MX_PACK) == 0


def quantizable(extent, partitions):
    """Whether ``quantize_mx`` writes ``extent`` contraction elements in
    place: whole chunks, or whole quadrants inside one chunk (an attention
    head's 128)."""
    chunk = partitions * MX_PACK
    return (
        extent > 0
        and extent % (QUADRANT * MX_PACK) == 0
        and (extent % chunk == 0 or chunk % extent == 0)
    )


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
            pair (scales, codes), quantized along its last dim.
    """

    key = None

    def __init__(self, node, model):
        self.node = node
        self.model = model
        self.space = model.grid_space
        roles = operand_roles(node) if multiplies_mx(node) else {}
        self.mx_operands = {
            n: MX_ROLES[role]
            for n, (role, _) in roles.items()
            if role in MX_ROLES
        }
        ops = arithmetic_ops(node)
        self.quantizes_mx = bool(ops) and ops[-1].target is QUANTIZE_MX

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

    def row_parts(self, out_tile):
        """``(parts, spread)`` for an output tile ``out_tile``: its rows on
        the partitions, none cut across them; a flat vector op overrides
        it (``TrainiumVector.row_parts``)."""
        return min(rows_cols(out_tile)[0], PARTITIONS), False

    def _mx_lowerable(self, mx):
        """Whether the translator lowers the MX tiles ``mx``
        (``_mx_tiles``): the GEMM's operands contract whole chunks
        (``whole_chunks``) and the op's own output is ``quantizable``."""
        node = self.node
        p = self.model.config.partitions
        return all(
            (quantizable(depth, p) if n is node else whole_chunks(depth, p))
            for n, arrays in mx.items()
            for depth, _, _ in arrays
        )

    def fit(self, tiling):
        """The kernel's ``TrainiumPlacement`` under ``tiling``
        (``placement``) when the translator lowers the tile's MX arrays
        (``_mx_lowerable``) and its buffers fit SBUF, else ``None``.

        A buffer takes the slots the bufferizer gives it: two for one the
        sweep moves more than once, one for one it moves once and holds,
        laid out as ``row_parts`` says.  The placement's buffers and scratch
        count too.
        """
        model = self.model
        config = model.config
        shapes = self.operand_shapes(tiling)
        mx = self._mx_tiles(shapes)
        if not self._mx_lowerable(mx):
            return None
        moves = self.transfers(shapes, tiling)
        placement = self.placement(tiling, shapes, moves)
        if placement is None:
            return None
        needed = sbuf_placed_bytes(config, placement.buffers, placement.scratch)
        out_tile = output_buffers(shapes[self.node], self.node)[-1][0]
        parts, spread = self.row_parts(out_tile)
        for n, shape, count in moves:
            slots = config.num_slots if count > 1 else 1
            needed += slots * model.buffer_sbuf(
                n, shape, mx, parts, spread and math.prod(shape) > 1
            )
        if needed > config.usable_scratchpad_size:
            return None
        return placement
