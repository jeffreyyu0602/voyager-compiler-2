"""Carry out Voyager's ``model.txt`` for one Trainium NeuronCore in NKI.

Voyager compiled the model for the core (``TrainiumConfig``): it tiled it,
bufferized each op into a loop nest over pipelined SBUF slots and wrote the
program as ``voyager`` IR.  This module decides nothing.  It runs each
kernel's loops and conditions at translation time, so an NKI kernel is the
straight-line sequence of transfers and compute steps the schedule makes, on
the tiles it names.  NKI places the buffers and orders the instructions by
their dependences, so the semaphores and waits are dropped.

A tile in SBUF has its rows -- every dim but the last, flattened -- on the
partitions, 128 at a time, and its last dim free: a slot of ``B`` row blocks
of ``C`` columns is an NKI tensor ``[P, B * C]``.  A flat tile -- one every
op of the kernel touching it reads or writes whole, element for element, in
an elementwise chain (or, one row, an RMS norm alone), and TensorE never
reads -- is cut in row-major order across the partitions, each op on it
running there (``Layout.spread``): HBM still holds it as rows.  Any other
tile read only as an elementwise operand, of one row or of fewer rows than
an elementwise or row-wise op reading it, is copied onto each partition the
op takes when it is loaded, row ``p % rows`` onto partition ``p``
(``Layout.replicated``).
Every access is an ``.ap()`` access pattern, found by mapping each element's
logical coordinates to its offset and factoring the offsets into one nesting
per instruction (``fit``), so views and relayouts fused into a kernel need
no case of their own.

A dense GEMM and a flash attention reach the translator as their builders
(``gemm.py``, ``attention.py``) wrote them: transposes and matmuls into
PSUM boxes, the copies and the ops out of them, a split reduction's sum and
an attention's scores held in PSUM, each one IR op.  A vector op runs on
128-row chunks -- reading a PSUM tile of whole 128-row blocks, on all its
blocks at once.  The translator makes no buffer of its own: an op computes
its intermediates -- a chain's results before its last, a norm's
statistics, a ``quantize_mx``'s rearranged input and its transposes in
PSUM -- in the scratch regions the IR hands it (``with_scratch``), which
the tile search placed.  An operand the DMA would load transposed reaches it
as a load of its tile as stored and TensorE transposes; one it still loads
transposed (of several elements whose rows are not whole 128-row runs) the
DMA moves element by element.

A box the IR places in PSUM (``MEMORY_LEVEL_LOCAL_BUFFER``) is declared
there and laid out as an SBUF tile is, an 8-bit element two bytes apart.  A
transpose, a K-major GEMM or an MX one alone writing such a box is one
``nc_transpose``, ``nc_matmul`` or ``nc_matmul_mx``, and a fused group a
builder formed as a fused pattern (``patterns.py``) is the instruction the
pattern names (``PATTERN_HANDLERS``); an ``eye`` is a ``memset`` and an
``affine_select``.  Each is emitted per 128 rows -- a matmul also per 512
columns of its result and per 128-deep chunk of its contraction (two whole
ones in an FP8 ``double_row`` matmul, ``_nc_matmul``; 512-deep MX), a
transpose per 128 x 128 block.

A GEMM on a K-major activation (its ``layout`` ``"cx,ck->xk"``, ``[..., K, M]``)
reads it as stored, its contraction already on the partitions: the
activation is the stationary operand, or the moving one and the weight the
stationary when the GEMM writes its output transposed (``"cx,ck->kx"``,
``[..., N, M]``), and nothing is transposed.  Across the partitions of a
K-major tile, an RMS norm sums the squares by a ones-matmul on TensorE;
RoPE's rotate-half arrives as a K-major ``matmul`` by a signed permutation
constant, an ordinary GEMM.

An MX (microscaling) array -- the codes or the block scales a ``linear_mx``
multiplies -- is stored contraction-major, the way Trn3's ``nc_matmul_mx``
reads it (``Mx``), and a ``linear_mx`` reads part of such a tile.  An op
ending in ``quantize_mx`` transposes its result onto the partitions 4
columns apart and quantizes it there with ``quantize_mx``.

``translate`` writes ``kernels.py`` (one ``@nki.jit`` function per distinct
kernel), ``program.py`` (the host program: the DRAM-only ops as torch ops,
and a call per kernel) and ``manifest.json``.
"""

import hashlib
import json
import math
import operator
import os
import re
from dataclasses import dataclass, replace

import numpy as np
import torch
from google.protobuf import text_format

from voyager_compiler.codegen import voyager_ir_pb2 as ir
from voyager_compiler.ops.layout import MATMUL_LAYOUT, GemmLayout
from voyager_compiler.quantization.qspec import float_format
from voyager_compiler.shape_prop import fetch_attr

from .model import stream_blocks
from .patterns import (
    ACTIVATION_REDUCE,
    MATMUL_ACCUMULATE,
    SCALED_IDENTITY,
    TENSOR_SCALAR,
    TENSOR_SCALAR_REDUCE,
)
from .utils import (
    CHUNK_FREE,
    DOUBLE_ROW_STEP,
    PARTITIONS,
    STATS_FREE,
    contraction_runs,
    pair_distance,
    scratch_name,
    spread_partitions,
)
from .vector import SQUARE_BLOCKS

IMMEDIATE = ir.MEMORY_LEVEL_IMMEDIATE
SCRATCHPAD = ir.MEMORY_LEVEL_SCRATCHPAD
LOCAL = ir.MEMORY_LEVEL_LOCAL_BUFFER  # PSUM
DRAM = ir.MEMORY_LEVEL_DRAM
ON_CHIP = (SCRATCHPAD, LOCAL)

# A boolean is a byte, 0 or 1; an int64 (a position) is carried as int32,
# which is as wide as NKI computes.
NL_DTYPES = {
    "bfloat16": "nl.bfloat16",
    "float16": "nl.float16",
    "float32": "nl.float32",
    "int32": "nl.int32",
    "int64": "nl.int32",
    "bool": "nl.uint8",
}
# Bytes of each type's element on chip (an 8-bit float's is 1).
NL_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 4,
    "bool": 1,
}
# NKI's 8-bit floats, by exponent bits and ``FloatFormat.ieee``: Trn2
# computes in the IEEE formats, Trn3 also in OCP's E4M3FN.
NL_FP8 = {
    (4, True): "nl.float8_e4m3",
    (4, False): "nl.float8_e4m3fn",
    (5, True): "nl.float8_e5m2",
}

# Scalar ops the control runs, by the IR's target.
SCALAR_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "and_": lambda a, b: bool(a and b),
    "or_": lambda a, b: bool(a or b),
}
ALLOCATIONS = ("voyager::alloc", "voyager::zeros", "voyager::fill")

# Elementwise ops: target -> NKI ALU op, or activation.
BINARY_OPS = {
    "aten::add": "nl.add",
    "aten::sub": "nl.subtract",
    "aten::mul": "nl.multiply",
    "aten::le": "nl.less_equal",
    "aten::lt": "nl.less",
    "aten::ge": "nl.greater_equal",
    "aten::gt": "nl.greater",
    "aten::eq": "nl.equal",
    "aten::ne": "nl.not_equal",
    "aten::maximum": "nl.maximum",
    "aten::__and__": "nl.logical_and",
}
# Reductions over a tile's last dim: target -> NKI ALU op.
REDUCTIONS = {"aten::amax": "nl.maximum", "aten::sum": "nl.add"}
# Fills: target -> the kwarg giving the value, ``None`` for zero.
FILLS = {"aten::full_like": "fill_value", "aten::zeros_like": None}
ACTIVATIONS = {
    "aten::silu": "nl.silu",
    "aten::exp": "nl.exp",
    "aten::sigmoid": "nl.sigmoid",
    "aten::tanh": "nl.tanh",
    "aten::rsqrt": "nl.rsqrt",
    "aten::gelu": "nl.gelu",
}
# Per-tensor quantization: ``quantize`` divides by the scale and saturates
# at the format's largest value, writing the FP8 tile; ``dequantize``
# multiplies the scale back.
QUANTIZE = "quantized_ops::quantize"
DEQUANTIZE = "quantized_ops::dequantize"

# Sine and cosine as ScalarE's sine of an angle brought into [-pi, pi]:
# the shift taking each to a sine.
SINES = {"aten::sin": 0.0, "aten::cos": 0.5 * math.pi}
RELAYOUTS = ("aten::transpose", "aten::view", "aten::reshape")
# The GEMMs on a CK weight, each storing its operands as its ``layout``
# says (``_gemm_layout``).
GEMMS = ("quantized_ops::linear", "quantized_ops::matmul", "aten::matmul")
# An RMS norm of a K-major ``[..., H, T]`` tile over ``H``, across the
# partitions.
RMS_NORM_KMAJOR = "quantized_ops::rms_norm_kmajor"
MAX_POOL2D = "aten::max_pool2d"
LINEAR_MX = "quantized_ops::linear_mx"
QUANTIZE_MX = "quantized_ops::quantize_mx"
# The row-wise ops, each lowered by a ``Kernel`` method over whole rows.
ROWWISE = {
    "aten::layer_norm": "_layer_norm",
    "aten::rms_norm": "_rms_norm",
    "aten::softmax": "_softmax",
}
# The ``Kernel`` method emitting a fused group of each fused pattern
# (``patterns.py``), by the pattern's name.
PATTERN_HANDLERS = {
    TENSOR_SCALAR: "_tensor_scalar",
    TENSOR_SCALAR_REDUCE: "_tensor_scalar_reduce",
    ACTIVATION_REDUCE: "_activation_reduce",
    MATMUL_ACCUMULATE: "_matmul_accumulate",
    SCALED_IDENTITY: "_scaled_identity",
}

# MX arrays as ``nc_matmul_mx`` reads them: a 32-bit word packs 4
# consecutive contraction elements, so the partitions hold a chunk of 512,
# and the E8M0 scale of each block of 32 sits on the first 4 partitions of
# its quadrant of 32.
MX_PACK = 4
MX_BLOCK = 32
MX_CHUNK = PARTITIONS * MX_PACK
QUADRANT = 32
QUADRANT_SCALES = QUADRANT * MX_PACK // MX_BLOCK
CHUNK_SCALES = MX_CHUNK // MX_BLOCK
# NKI's packed MX types, by element exponent bits.
NL_MX = {4: "nl.float8_e4m3fn_x4", 5: "nl.float8_e5m2_x4"}
# The exponent ``quantize_mx`` leaves a block's largest element, by element
# exponent bits: Trn3's default, its scale 2**(floor(log2(amax)) - emax),
# one below OCP's.  Voyager's own is floor(log2(quant_max)).
MX_EMAX = {4: 7, 5: 14}
# The operands of a ``linear_mx``: whether each holds scales, and whether
# its contraction leads (a K-major weight).
MX_OPERANDS = {
    "input": (False, False),
    "input_scale": (True, False),
    "weight": (False, True),
    "weight_scale": (True, True),
}


class CrossesPartitions(Exception):
    """A result no instruction writes into ``box`` within the partitions."""

    def __init__(self, box):
        super().__init__(box)
        self.box = box


@dataclass
class Box:
    """A storage object: one slot's shape (a 0-d one as one element), its
    element type, memory level and slots; a ``pipelined`` box takes a slot
    index first in a reference."""

    name: str
    shape: tuple
    dtype: str
    level: int
    slots: int
    pipelined: bool

    @property
    def rows(self):
        return math.prod(self.shape[:-1])

    @property
    def cols(self):
        return self.shape[-1]


@dataclass
class Layout:
    """A box's slot on chip: ``parts`` partitions by ``blocks`` row blocks of
    room for ``cols`` columns (at least the box's), row ``r`` on partition
    ``r % span`` of block ``r // span``, its elements ``spacing`` apart; a
    ``replicated`` box's row ``p % rows`` on each partition ``p`` of its
    ``parts``.  A box with ``span`` 1 is all on one partition.  A ``spread``
    box is its elements cut in row-major order across the partitions,
    ``cols`` to each, element ``e`` on partition ``e // cols``."""

    parts: int
    blocks: int
    cols: int
    replicated: bool
    span: int = PARTITIONS
    spacing: int = 1
    spread: bool = False

    @property
    def free(self):
        return self.blocks * self.cols * self.spacing


@dataclass
class OpUse:
    """What a compute op does with the kernel's on-chip boxes, as the
    analysis walk records it (``Kernel._lay_out``).

    Attributes:
        flat: The op can run on its result cut across the partitions: an
            elementwise chain, or an RMS norm alone on one row, whose result
            and every operand are ``cols`` elements in SBUF or a single
            element.
        parts: The partitions the op's rows take otherwise.
        cols: The elements of its result.
        rows: The boxes it holds as rows of its result: its result and
            the operands of its result's size.
        touched: Every on-chip box it reads or writes.
        broadcast: The boxes an elementwise or row-wise op reads with fewer
            rows than its ``parts``, each copied onto its partitions.
    """

    flat: bool
    parts: int
    cols: int
    rows: frozenset
    touched: frozenset
    broadcast: frozenset


@dataclass
class View:
    """A reference resolved: the box and slot it reads, the window of the
    slot it covers (from ``start``, ``step`` apart) in the slot's dims
    ``dims`` -- the box's, or a reshape of them -- and the shape it is seen
    as."""

    box: Box
    slot: int
    start: tuple
    step: tuple
    window: tuple
    shape: tuple
    dims: tuple

    def covers_box(self):
        """Whether the view is its whole slot, in order."""
        return (
            self.window == self.box.shape
            and not any(self.start)
            and all(t == 1 for t in self.step)
        )

    def positions(self, linear):
        """Row-major positions in the slot of the elements at row-major
        positions ``linear`` of ``shape``."""
        window = np.unravel_index(linear, self.window)
        coords = [s + w * t for s, w, t in zip(self.start, window, self.step)]
        return np.ravel_multi_index(coords, self.dims)

    def key(self):
        """The view as a hashable tuple."""
        return (
            self.box.name,
            self.slot,
            self.start,
            self.step,
            self.window,
            self.shape,
        )


@dataclass
class Chunk:
    """A result on one ``[r, w]`` chunk: the tensor holding it and its
    elements per partition, its logical shape and each element's
    coordinates in it (``[ndim, r, w]``); ``flat`` gives the offsets and
    ``dtype`` the element type when it is written into a tile -- box
    ``box`` -- rather than a float32 scratch tile."""

    var: str
    free: int
    shape: tuple
    coords: np.ndarray
    flat: np.ndarray = None
    dtype: str = "float32"
    box: str = None

    def offsets(self):
        """The ``[r, w]`` flat offsets of the chunk's elements."""
        if self.flat is not None:
            return self.flat
        r, w = self.coords.shape[1:]
        return np.arange(r)[:, None] * self.free + np.arange(w)[None, :]

    def reshaped(self, shape):
        linear = np.ravel_multi_index(self.coords, self.shape)
        coords = np.stack(np.unravel_index(linear, shape))
        return replace(self, shape=tuple(shape), coords=coords)

    def transposed(self, d0, d1):
        order = list(range(len(self.shape)))
        d0, d1 = d0 % len(order), d1 % len(order)
        order[d0], order[d1] = order[d1], order[d0]
        shape = tuple(self.shape[d] for d in order)
        return replace(self, shape=shape, coords=self.coords[order])


@dataclass(frozen=True)
class Mx:
    """Where an MX array keeps its elements: ``depth`` contraction elements
    beside ``other`` (an activation's rows, a weight's columns), the
    contraction leading in a ``k_major`` array; a ``scales`` array holds one
    scale per block of 32 where a codes array holds the elements.

    In HBM codes are ``[128, chunks, other]`` words, a chunk of 512 elements
    4 to a word on 128 rows, and scales ``[16, chunks, other]`` bytes, a
    chunk's 16 blocks on 16 rows.  On chip both are ``[partitions, chunks *
    padded]``, each chunk's ``padded`` columns beside the one before, the
    scales on each quadrant's first 4 partitions."""

    scales: bool
    depth: int
    other: int
    k_major: bool

    def coords(self, linear):
        """``(other, k)`` of the elements at row-major positions ``linear``
        of the array's logical shape; ``k`` counts blocks in scales."""
        extent = self.depth // MX_BLOCK if self.scales else self.depth
        if self.k_major:
            k, x = np.divmod(linear, self.other)
        else:
            x, k = np.divmod(linear, extent)
        return x, k

    def _row_chunk(self, k):
        """The HBM row and the chunk of ``k`` (an element, or a block)."""
        if self.scales:
            return k % CHUNK_SCALES, k // CHUNK_SCALES
        return (k % MX_CHUNK) // MX_PACK, k // MX_CHUNK

    def hbm_shape(self):
        rows = CHUNK_SCALES if self.scales else PARTITIONS
        return (rows, self.depth // MX_CHUNK, self.other)

    def hbm_offsets(self, x, k):
        """Offsets in the HBM array, in words or bytes, of ``(x, k)``."""
        row, chunk = self._row_chunk(k)
        return (row * (self.depth // MX_CHUNK) + chunk) * self.other + x

    def tile(self, padded):
        """``(partitions, free)`` of an on-chip tile."""
        parts = min(PARTITIONS, self.depth // MX_PACK)
        return parts, math.ceil(self.depth / MX_CHUNK) * padded

    def tile_offsets(self, x, k, padded):
        """Offsets in an on-chip tile of ``(x, k)``."""
        row, chunk = self._row_chunk(k)
        if self.scales:
            quadrant, row = divmod(row, QUADRANT_SCALES)
            row += quadrant * QUADRANT
        return row * self.tile(padded)[1] + chunk * padded + x


def _mx_array(ref, scales, k_major):
    """``Mx`` of the HBM box a ``linear_mx`` operand is loaded from, through
    reference ``ref``: a weight's contraction leads, an activation's is its
    trailing elements, as many as the last dim of the view the load reads
    the whole box through."""
    shape = tuple(ref.box.shape)
    if k_major:
        depth, other = shape[0], math.prod(shape[1:])
    else:
        view = tuple(ref.output_shape) if not ref.offsets else shape
        depth = (view or shape)[-1]
        other = math.prod(shape) // depth
    if scales:
        depth *= MX_BLOCK
    if depth % MX_CHUNK:
        raise NotImplementedError(f"{ref.box.node}: MX depth {depth}")
    return Mx(scales, depth, other, k_major)


def mx_arrays(model):
    """The MX arrays among ``model``'s HBM boxes, by name -- what each
    ``linear_mx`` operand is loaded from -- and the scales of each codes
    array, by the codes' name."""
    loads, prims = {}, []

    def walk(ops):
        for op in ops:
            kind = op.WhichOneof("op_type")
            if kind == "loop":
                walk(op.loop.for_loop.body.ops)
            elif kind == "cond":
                walk(op.cond.true_region.ops)
                walk(op.cond.false_region.ops)
            elif kind == "async":
                walk(getattr(op, "async").body.ops)
            elif kind == "fused":
                prims.extend(op.fused.op_list)
            elif op.prim.target == "voyager::async_copy":
                src = op.prim.kwargs["src"].tensor_box
                dst = op.prim.kwargs["dst"].tensor_box
                if src.box.memory.level == DRAM:
                    loads.setdefault(dst.box.node, []).append(src)
            else:
                prims.append(op.prim)

    walk(model.ops)
    arrays, pairs = {}, {}
    for prim in prims:
        if prim.target != LINEAR_MX:
            continue
        found = {}
        for key, (scales, k_major) in MX_OPERANDS.items():
            tile = prim.kwargs[key].tensor_box.box.node
            for ref in loads.get(tile, ()):
                array = _mx_array(ref, scales, k_major)
                if arrays.setdefault(ref.box.node, array) != array:
                    raise NotImplementedError(f"{ref.box.node}: two MX views")
                found[key] = ref.box.node
        for codes, scales in (
            ("input", "input_scale"),
            ("weight", "weight_scale"),
        ):
            if codes in found and scales in found:
                pairs[found[codes]] = found[scales]
    return arrays, pairs


def box_of(message):
    """``Box`` of a ``TensorBox`` message."""
    return Box(
        name=message.node,
        shape=tuple(message.shape) or (1,),
        dtype=message.dtype,
        level=message.memory.level,
        slots=message.bank_count or 1,
        pipelined=message.HasField("bank_count"),
    )


def _fp8(dtype):
    """The format of IR ``dtype`` when it is an 8-bit float, else None."""
    fmt = float_format(dtype)
    return fmt if fmt is not None and fmt.bits == 8 else None


def _element_bytes(dtype):
    """Bytes of one element of IR ``dtype`` on chip."""
    return 1 if _fp8(dtype) else NL_BYTES[dtype]


def nl_dtype(dtype):
    """The NKI type of IR ``dtype``."""
    fmt = _fp8(dtype)
    if fmt is not None:
        return NL_FP8[fmt.ebits, fmt.ieee]
    return NL_DTYPES[dtype]


def fit(operands, rows):
    """Access patterns walking every operand of one instruction in step.

    Args:
        operands: ``(free, offsets)`` per operand: ``free`` is the elements
            per partition of an on-chip tensor (``None`` for HBM) and
            ``offsets`` the ``[rows, n]`` flat offsets of its elements, row
            by row.
        rows: Partitions the instruction spans.

    Returns:
        ``(pattern, offset)`` per operand, or ``None`` when an on-chip
        operand's rows are not one partition apart or a row leaves its
        partition, an HBM operand's rows are not evenly apart, or no nesting
        of at most three free dims walks every operand.
    """
    steps = []
    for free, offsets in operands:
        step = int(offsets[1, 0] - offsets[0, 0]) if rows > 1 else 0
        if rows > 1:
            expected = offsets[0] + step * np.arange(rows)[:, None]
            if not np.array_equal(offsets, expected):
                return None
            if free is not None and step != free:
                return None
        # An on-chip row stays on its partition.
        if free is not None:
            partitions = offsets // free
            if np.any(partitions != partitions[:, :1]):
                return None
        steps.append(free if free is not None else step)
    first = np.stack([offsets[0] for _, offsets in operands])
    dims = _nest(first)
    if dims is None or len(dims) > 3:
        return None
    return [
        ([[steps[k], rows]] + [[s[k], n] for n, s in dims], int(first[k, 0]))
        for k in range(len(operands))
    ]


def _nest(offsets):
    """Factor offsets ``[k, n]`` into one nesting all ``k`` operands share:
    ``[(count, strides)]``, outermost first, or ``None``."""
    k = offsets.shape[0]
    dims = []
    current = offsets
    while current.shape[1] > 1:
        m = current.shape[1]
        diffs = np.diff(current, axis=1)
        stride = diffs[:, 0]
        bad = np.any(diffs != stride[:, None], axis=0)
        run = int(np.argmax(bad)) + 1 if bad.any() else m
        if m % run:
            return None
        runs = current.reshape(k, m // run, run)
        if not np.all(np.diff(runs, axis=2) == stride[:, None, None]):
            return None
        dims.append((run, tuple(int(s) for s in stride)))
        current = runs[:, :, 0]
    return dims[::-1] or [(1, (0,) * k)]


def _ap(var, pattern, dtype=None):
    """Source of an access pattern on ``var``, seen as ``dtype`` if given."""
    steps, offset = pattern
    seen = f"dtype={dtype}, " if dtype is not None else ""
    return f"{var}.ap({seen}pattern={steps}, offset={offset})"


def _rows_cols(shape):
    return math.prod(shape[:-1]), shape[-1]


def _block_positions(offsets, sizes, shape):
    """Row-major positions in ``shape`` of the ``sizes`` block at
    ``offsets``."""
    block = np.indices(sizes).reshape(len(sizes), -1)
    block += np.array(offsets)[:, None]
    return np.ravel_multi_index(block, shape)


def _runs(n, size):
    """``(start, stop)`` of each run of at most ``size`` of ``n``."""
    return [(s, min(n, s + size)) for s in range(0, n, size)]


def _gemm_layout(gemm):
    """How GEMM prim ``gemm`` stores its operands and result: its ``layout``
    argument, or ``aten::matmul``'s, which takes none."""
    layout = gemm.kwargs.get("layout")
    return GemmLayout.parse(
        MATMUL_LAYOUT if layout is None else layout.str_value
    )


def _output_transposed(gemm):
    """Whether GEMM prim ``gemm`` writes its result ``[..., N, M]``."""
    return _gemm_layout(gemm).output == "kx"


def _delinearize(linear, basis):
    """``voyager::delinearize_index``: every coordinate wraps."""
    out = [0] * len(basis)
    for d in range(len(basis) - 1, -1, -1):
        out[d] = linear % basis[d]
        linear //= basis[d]
    return out


class Kernel:
    """One kernel's translation: two walks of its IR, the first recording
    what each box is used for and the second emitting NKI.

    It declares the boxes the IR names and makes none of its own: an op
    computes its intermediates in the scratch regions the IR hands it
    (``temp``).

    Attributes:
        immediates: The value of each immediate box, by name.
        mx: The ``Mx`` of each HBM box holding an MX array, by name.
        config: The ``TrainiumConfig`` it runs on.
        boxes: Every box the kernel touches, by name.
        reads: The HBM boxes it reads, its arguments, in order.
        writes: The HBM boxes it writes, its results, in order.
        stored: Each box of ``writes`` -> which of its elements, row-major,
            the kernel stores.
        windows: Each pool input box -> ``(kH, sH, dH, out_rows, out_h)``:
            its kernel height, stride and dilation along H, and the rows and
            height of the pool's output tile.
        lines: The body, statement by statement.
    """

    def __init__(self, ops, immediates, mx, config):
        self.immediates = immediates
        self.mx = mx
        self.config = config
        # Boxes held on one partition: a result laid out across partitions
        # that no instruction writes within them (a relayout of one row).
        self.one_partition = set()
        while True:
            try:
                self._translate(ops)
                break
            except CrossesPartitions as crossing:
                if crossing.box in self.one_partition:
                    raise NotImplementedError(f"{crossing.box}: relayout")
                self.one_partition.add(crossing.box)
        if set(self.reads) & set(self.writes):
            raise NotImplementedError("a kernel reads an HBM box it writes")

    def _translate(self, ops):
        """Walk ``ops`` twice: to learn each box's uses, then to emit."""
        self.boxes = {}
        self.layouts = {}
        # An SBUF tile holding an MX array: its ``Mx`` and its rows padded
        # to an even number (a stationary operand's width must be even).
        self.mx_tiles = {}
        self.reads, self.writes = [], []
        self.stored = {}
        self.windows = {}
        self.uses = {}
        self.op_uses = []
        self.lines = []
        # How many times each scratch region has been taken (``temp``).
        self.turns = {}
        # Emit-pass results by their inputs: transfers replayed shifted
        # (``_shifted_transfer``) and GEMM operand access patterns.
        self.memo = {}
        # The patterns ``_pattern_dma`` emits, while a transfer is recorded.
        self.dma_log = None
        # The kernel runs exponentials with their sums on ScalarE
        # (``ACTIVATION_REDUCE``), so its other ALU passes run on VectorE
        # and its fills on GpSimd.
        self.scalar_exps = False
        for analyze in (True, False):
            self.analyze = analyze
            self.env = {}
            self._walk(ops)
            if analyze:
                self._lay_out()

    # The IR walk, control specialized.

    def _scalar(self, value):
        kind = value.WhichOneof("value")
        if kind == "node":
            return self.env[value.node]
        return getattr(value, kind)

    def _arg(self, arg):
        kind = arg.WhichOneof("arg_type")
        if kind == "tensor_box":
            return arg.tensor_box
        if kind == "tensor_box_list":
            return list(arg.tensor_box_list.values)
        if kind == "scalar":
            return self._scalar(arg.scalar)
        if kind == "scalar_list":
            return [self._scalar(v) for v in arg.scalar_list.values]
        if kind == "str_value":
            return None if arg.str_value == "None" else arg.str_value
        return None

    def _kwargs(self, prim):
        return {k: self._arg(v) for k, v in prim.kwargs.items()}

    def _walk(self, ops):
        for op in ops:
            kind = op.WhichOneof("op_type")
            if kind == "loop":
                self._loop(op)
            elif kind == "cond":
                taken = self._scalar(op.cond.predicate)
                region = op.cond.true_region if taken else op.cond.false_region
                self._walk(region.ops)
                for out, value in zip(op.outputs, region.yields):
                    self.env[out.name] = self._scalar(value)
            elif kind == "async":
                self._walk(getattr(op, "async").body.ops)
            elif kind == "prim" and op.prim.op == "cpu":
                self._cpu(op)
            elif kind in ("prim", "fused"):
                self._compute(op)
            else:
                raise NotImplementedError(f"{op.name}: {kind}")

    def _loop(self, op):
        kind = op.loop.WhichOneof("loop_type")
        if kind != "for_loop":
            raise NotImplementedError(f"{op.name}: {kind}")
        loop = op.loop.for_loop
        state = [self._scalar(a.initial) for a in loop.iter_args]
        start, end = self._scalar(loop.start), self._scalar(loop.end)
        for i in range(start, end, self._scalar(loop.step)):
            self.env[loop.iv] = i
            for arg, value in zip(loop.iter_args, state):
                self.env[arg.name] = value
            self._walk(loop.body.ops)
            state = [self._scalar(v) for v in loop.body.yields]
        for out, value in zip(op.outputs, state):
            self.env[out.name] = value

    def _cpu(self, op):
        prim = op.prim
        kw = self._kwargs(prim)
        if prim.target in ALLOCATIONS:
            for out in op.outputs:
                if out.WhichOneof("result_type") == "tensor_box":
                    self._box(out.tensor_box)
        elif prim.target == "voyager::async_copy":
            self._copy(kw)
        elif prim.target == "voyager::delinearize_index":
            coords = _delinearize(kw["linear"], kw["basis"])
            for out, value in zip(op.outputs, coords):
                self.env[out.name] = value
        elif prim.target == "sym_ite":
            self.env[op.outputs[0].name] = kw["t"] if kw["b"] else kw["f"]
        elif prim.target in SCALAR_OPS:
            value = SCALAR_OPS[prim.target](kw["input"], kw["other"])
            self.env[op.outputs[0].name] = value
        elif prim.target != "voyager::async_wait":
            raise NotImplementedError(f"{op.name}: cpu op {prim.target}")

    # Boxes, layouts and references.

    def _box(self, message):
        box = self.boxes.get(message.node)
        if box is None:
            box = self.boxes[message.node] = box_of(message)
        return box

    def _use(self, box, role):
        if self.analyze:
            self.uses.setdefault(box.name, set()).add(role)

    def _lay_out(self):
        """Every on-chip box's layout: rows on the partitions, a flat box cut
        across them (``_spread_ops``), any other box only ever read as an
        elementwise operand, of one row or broadcast along an op's rows
        (``OpUse.broadcast``), copied onto each partition an op reading it
        takes (``replicated``), and a pool input as its windows
        (``windows``).
        PSUM holds an 8-bit element in two bytes, the way TensorE writes an
        FP8 transpose, and starts each row block on a 4-byte boundary, where
        an instruction's PSUM access must start."""
        spread_ops = self._spread_ops()
        spread = {name for use in spread_ops for name in use.rows}
        for box in self.boxes.values():
            if box.level not in ON_CHIP or box.name in self.mx_tiles:
                continue
            uses = self.uses.get(box.name, set())
            spacing = 2 if box.level == LOCAL and _fp8(box.dtype) else 1
            cols = box.cols
            if box.level == LOCAL:
                width = spacing * _element_bytes(box.dtype)
                cols = math.ceil(cols * width / 4) * 4 // width
            if box.name in self.one_partition:
                self.layouts[box.name] = Layout(
                    1, box.rows, cols, False, span=1, spacing=spacing
                )
                continue
            if box.name in spread:
                elements = math.prod(box.shape)
                parts = spread_partitions(elements)
                self.layouts[box.name] = Layout(
                    parts, 1, elements // parts, False, spread=True
                )
                continue
            replicated = (
                "operand" in uses
                and "dst" not in uses
                and (
                    box.rows == 1
                    or any(box.name in use.broadcast for use in self.op_uses)
                )
            )
            if replicated:
                parts = max(
                    (
                        (
                            spread_partitions(use.cols)
                            if use in spread_ops
                            else use.parts
                        )
                        for use in self.op_uses
                        if box.name in use.touched
                    ),
                    default=PARTITIONS,
                )
            else:
                parts = min(box.rows, PARTITIONS)
            blocks = math.ceil(box.rows / PARTITIONS)
            self.layouts[box.name] = Layout(
                parts, blocks, cols, replicated, spacing=spacing
            )
        # A pool input holds output row p's kH input rows on partition p.
        for name, (kh, _, _, out_rows, _) in self.windows.items():
            cols = kh * self.boxes[name].cols
            self.layouts[name] = Layout(out_rows, 1, cols, False)

    def _spread_ops(self):
        """The ops that run on rows cut across the partitions: the flat ones
        (``OpUse.flat``) whose rows no other op touches, TensorE never reads
        and none lies on one partition -- an op that cannot holds its rows
        against the flat ops sharing them, until none is left out."""
        spread = [use for use in self.op_uses if use.flat]
        while True:
            held = set(self.one_partition) | {
                name for name, uses in self.uses.items() if "gemm" in uses
            }
            for use in self.op_uses:
                if use not in spread:
                    held |= use.touched
            kept = [use for use in spread if not use.rows & held]
            if len(kept) == len(spread):
                return spread
            spread = kept

    def _view(self, ref):
        """``View`` of a ``TensorBoxRef``; a window of a pipelined box leads
        with its slot, unless the box has one slot."""
        box = self._box(ref.box)
        offsets = [self._scalar(o) for o in ref.offsets]
        sizes, steps = list(ref.sizes), list(ref.strides)
        slot = 0
        if box.pipelined and len(offsets) > len(ref.box.shape):
            slot = offsets.pop(0)
            sizes, steps = sizes[1:], steps[1:]
        dims = tuple(ref.box.shape)
        if not offsets:
            offsets, sizes = [0] * len(box.shape), list(box.shape)
            steps = [1] * len(box.shape)
            dims = box.shape
        shape = tuple(ref.output_shape) or tuple(sizes) or (1,)
        return View(
            box,
            slot,
            tuple(offsets),
            tuple(steps),
            tuple(sizes) or (1,),
            shape,
            dims,
        )

    def var(self, box, slot=0):
        """The kernel variable holding ``box`` (one slot of it on chip)."""
        if box.level == DRAM:
            if box.name in self.writes:
                return f"out{self.writes.index(box.name)}"
            return f"in{self.reads.index(box.name)}"
        return f"s{list(self.boxes).index(box.name)}_{slot}"

    def free(self, box):
        """Elements per partition of ``box``'s tensor; ``None`` in HBM."""
        if box.level == DRAM:
            return None
        if box.name in self.mx_tiles:
            mx, padded = self.mx_tiles[box.name]
            return mx.tile(padded)[1]
        return self.layouts[box.name].free

    def _mx_tile(self, box, scales, k_major):
        """Record on-chip ``box`` as an MX tile, contracting along its last
        dim, or its first if ``k_major``; an op may read or write part of
        it."""
        *lead, last = box.shape
        if k_major:
            depth, other = box.shape[0], math.prod(box.shape[1:])
        else:
            depth, other = last, math.prod(lead)
        mx = Mx(scales, depth * MX_BLOCK if scales else depth, other, k_major)
        padded = other if k_major else other + other % 2
        if self.mx_tiles.setdefault(box.name, (mx, padded)) != (mx, padded):
            raise NotImplementedError(f"{box.name}: two MX layouts")

    def flat(self, view, linear, partitions=None):
        """Flat offsets in its tensor of ``view``'s elements at positions
        ``linear``; a replicated box is read on ``partitions``, a spread
        one where it is cut (``Layout.spread``)."""
        box = view.box
        # The view is the box reshaped when it covers it: position is the
        # box's own.
        position = linear if view.covers_box() else view.positions(linear)
        if box.level == DRAM:
            return position
        layout = self.layouts[box.name]
        if layout.spread:
            part, col = np.divmod(position, layout.cols)
            return part * layout.free + col
        row, col = np.divmod(position, box.cols)
        if layout.replicated:
            # Partition ``p`` holds row ``p % rows`` only.
            if np.any(row != partitions % box.rows):
                raise CrossesPartitions(box.name)
            return partitions * layout.free + col
        part, block = row % layout.span, row // layout.span
        return part * layout.free + (block * layout.cols + col) * layout.spacing

    def emit(self, line):
        self.lines.append(line)

    def temp(self, kind, cols):
        """The next slot of the op's scratch tile of ``kind``, ``[128,
        cols]``: the region the IR hands the op under that name
        (``scratch_name``), its slots taken in turn across the kernel.

        Raises:
            NotImplementedError: The op has no such region.
        """
        suffix = "." + scratch_name(kind, cols)
        for prim in self.prims:
            for key, arg in prim.kwargs.items():
                if key.endswith(suffix):
                    box = self._view(arg.tensor_box).box
                    turn = self.turns.get(box.name, 0)
                    self.turns[box.name] = turn + 1
                    return self.var(box, turn % box.slots)
        raise NotImplementedError(
            f"{self.prims[0].name}: no {kind} scratch of {cols} columns"
        )

    # Transfers.

    def _copy(self, kw):
        """``voyager::async_copy``: the block of the HBM view at ``indices``
        to or from an SBUF slot."""
        src, dst = self._view(kw["src"]), self._view(kw["dst"])
        if any(kw.get("pad") or ()) or kw.get("count"):
            raise NotImplementedError("padded or partial transfer")
        load = src.box.level == DRAM
        hbm, sbuf = (src, dst) if load else (dst, src)
        if hbm.box.level != DRAM or sbuf.box.level != SCRATCHPAD:
            raise NotImplementedError("a transfer not between HBM and SBUF")
        names = self.reads if load else self.writes
        if hbm.box.name not in names:
            names.append(hbm.box.name)
        self._use(sbuf.box, "load" if load else "store")
        if self.analyze:
            return
        sizes = tuple(kw["sizes"]) or (1,)
        steps = kw.get("strides") or sizes
        dims = kw.get("dims")
        offsets = [0] * len(sizes)
        for i, index in enumerate(kw["indices"]):
            d = dims[i] if dims is not None else i
            offsets[d] = index * steps[d]
        hbm_linear = _block_positions(offsets, sizes, hbm.shape)
        if not load:
            stored = self.stored.setdefault(
                hbm.box.name, np.zeros(math.prod(hbm.box.shape), bool)
            )
            stored[self.flat(hbm, hbm_linear)] = True
        if (
            hbm.box.name not in self.mx
            and not kw.get("transposed")
            and sbuf.box.name not in self.windows
            and (hbm.covers_box() or hbm.shape == hbm.window)
        ):
            self._shifted_transfer(hbm, offsets, sizes, hbm_linear, sbuf, load)
            return
        if hbm.box.name in self.mx:
            if kw.get("transposed"):
                raise NotImplementedError("a transposing MX transfer")
            self._mx_transfer(hbm, hbm_linear, sbuf, load)
        elif kw.get("transposed"):
            # The DMA transposes element by element: a builder lands a tile
            # it can transpose on TensorE as stored (``_BufferedRef.landing``).
            a, b = sizes[-2:]
            linear = hbm_linear.reshape(-1, a, b).swapaxes(1, 2).reshape(-1)
            self._transfer(hbm, linear, sbuf, True)
        elif sbuf.box.name in self.windows:
            self._window_load(hbm, hbm_linear, sizes, sbuf)
        else:
            self._transfer(hbm, hbm_linear, sbuf, load)

    def _shifted_transfer(self, hbm, offsets, sizes, hbm_linear, sbuf, load):
        """``_transfer`` of the ``sizes`` block of ``hbm`` at ``offsets``
        (its elements ``hbm_linear``), replayed from an earlier copy of the
        same shape when there is one.

        ``hbm`` is its box reshaped or a window of it seen in the window's
        shape, so the HBM offsets of two such blocks differ by one constant,
        the difference of their first elements': a replay emits the recorded
        patterns with their HBM offsets shifted by it."""
        key = (
            "dma",
            hbm.box.name,
            hbm.step,
            hbm.window,
            hbm.shape,
            sbuf.key(),
            sizes,
            load,
        )
        # Raveling the block's last element checks it is in bounds.
        last = [o + s - 1 for o, s in zip(offsets, sizes)]
        corners = np.ravel_multi_index(np.array([offsets, last]).T, hbm.shape)
        base = int(self.flat(hbm, corners)[0])
        if key not in self.memo:
            self.dma_log = []
            self._transfer(hbm, hbm_linear, sbuf, load)
            self.memo[key] = [
                (on_chip, steps, offset - base)
                for on_chip, (steps, offset) in self.dma_log
            ]
            self.dma_log = None
            return
        hbm_var = self.var(hbm.box)
        for on_chip, steps, offset in self.memo[key]:
            self._emit_dma(on_chip, _ap(hbm_var, (steps, offset + base)), load)

    def _mx_transfer(self, hbm, hbm_linear, sbuf, load):
        """Copy an MX array between HBM and an SBUF tile, word by word for
        codes and byte by byte for scales, both laid out the ``Mx`` way:
        codes in one ``dma_copy`` when a pattern walks them, scales in one
        per quadrant (their partitions are not adjacent)."""
        array = self.mx[hbm.box.name]
        if sbuf.box.name not in self.mx_tiles or not sbuf.covers_box():
            raise NotImplementedError(f"{sbuf.box.name}: an MX transfer")
        tile, padded = self.mx_tiles[sbuf.box.name]
        hx, hk = array.coords(self.flat(hbm, hbm_linear))
        sx, sk = tile.coords(np.arange(hbm_linear.size))
        dtype = None
        if not array.scales:
            word = sk % MX_PACK == 0
            hx, hk, sx, sk = hx[word], hk[word], sx[word], sk[word]
            dtype = NL_MX[_fp8(hbm.box.dtype).ebits]
        s = tile.tile_offsets(sx, sk, padded)
        h = array.hbm_offsets(hx, hk)
        order = np.argsort(s, kind="stable")
        s, h = s[order], h[order]
        free = tile.tile(padded)[1]
        rows = len(np.unique(s // free))
        sbuf_var, hbm_var = self.var(sbuf.box, sbuf.slot), self.var(hbm.box)
        s, h = s.reshape(rows, -1), h.reshape(rows, -1)
        self._dma(sbuf_var, free, s, hbm_var, h, load, dtype)

    def _transfer(self, hbm, hbm_linear, sbuf, load):
        """Copy between the HBM elements ``hbm_linear`` of ``hbm`` and
        ``sbuf``'s, in ``sbuf``'s row-major order: the whole tile in one
        ``dma_copy`` when a pattern walks it -- a load in one per piece of
        its 128-row blocks (``stream_blocks``), a spread tile cut across the
        partitions as it lies -- else 128 rows at a time."""
        layout = self.layouts[sbuf.box.name]
        rows, cols = _rows_cols(sbuf.shape)
        sbuf_var, hbm_var = self.var(sbuf.box, sbuf.slot), self.var(hbm.box)

        def offsets(linear):
            r = linear.shape[0]
            s = self.flat(sbuf, linear, np.arange(r)[:, None])
            return s, self.flat(hbm, hbm_linear[linear])

        if layout.replicated:
            # A pattern repeats one row onto every partition, but not
            # several: a tile of several rows is loaded once per copy.
            parts = layout.parts if load else rows
            linear = np.arange(parts)[:, None] % rows * cols + np.arange(cols)
            s, h = offsets(linear)
            for a, b in _runs(parts, rows if rows > 1 else parts):
                self._dma(sbuf_var, layout.free, s[a:b], hbm_var, h[a:b], load)
            return
        if layout.spread:
            linear = np.arange(rows * cols).reshape(layout.parts, -1)
            s, h = offsets(linear)
            self._dma(sbuf_var, layout.free, s, hbm_var, h, load)
            return
        parts, blocks = layout.parts, layout.blocks
        if parts * blocks == rows:
            block = parts * cols * _element_bytes(sbuf.box.dtype)
            group = (
                stream_blocks(blocks, block) if load and parts > 1 else blocks
            )
            pieces = []
            for first in range(0, blocks, group):
                # Partition ``p`` holds row ``b * span + p`` of block ``b``.
                piece = np.arange(first, min(first + group, blocks))
                row = np.arange(parts)[:, None] + piece * layout.span
                linear = row[:, :, None] * cols + np.arange(cols)
                s, h = offsets(linear.reshape(parts, -1))
                pieces.append(fit([(layout.free, s), (None, h)], parts))
            if None not in pieces:
                for aps in pieces:
                    self._pattern_dma(sbuf_var, aps, hbm_var, load)
                return
        for a, b in _runs(rows, PARTITIONS):
            linear = np.arange(a * cols, b * cols).reshape(b - a, cols)
            s, h = offsets(linear)
            self._dma(sbuf_var, layout.free, s, hbm_var, h, load)

    def _window_load(self, hbm, hbm_linear, sizes, sbuf):
        """Load a pool's halo block (``sizes``, the HBM elements
        ``hbm_linear``) as its windows: partition ``r``, output row ``oh``
        of leading index ``lead``, holds halo rows ``lead * ih + oh * sH + k
        * dH`` for ``k < kH`` side by side, ``iw`` columns each."""
        kh, sh, dh, out_rows, out_h = self.windows[sbuf.box.name]
        ih, iw = sizes[-2], sizes[-1]
        lead, oh = np.divmod(np.arange(out_rows), out_h)
        row = lead[:, None] * ih + oh[:, None] * sh + np.arange(kh) * dh
        linear = (row[:, :, None] * iw + np.arange(iw)).reshape(out_rows, -1)
        free = kh * iw
        s = np.arange(out_rows)[:, None] * free + np.arange(free)
        h = self.flat(hbm, hbm_linear[linear])
        sbuf_var = self.var(sbuf.box, sbuf.slot)
        self._dma(sbuf_var, free, s, self.var(hbm.box), h, True)

    def _try_dma(self, sbuf_var, free, s, hbm_var, h, load, dtype=None):
        """One ``dma_copy`` of offsets ``s`` (SBUF) and ``h`` (HBM) if a
        pattern walks both, the HBM tensor seen as ``dtype`` when given;
        whether it did."""
        aps = fit([(free, s), (None, h)], s.shape[0])
        if aps is None:
            return False
        self._pattern_dma(sbuf_var, aps, hbm_var, load, dtype)
        return True

    def _pattern_dma(self, sbuf_var, aps, hbm_var, load, dtype=None):
        """``dma_copy`` along the SBUF and HBM patterns ``aps``, the HBM
        tensor seen as ``dtype`` when given; logged in ``dma_log`` while a
        transfer is recorded."""
        on_chip = _ap(sbuf_var, aps[0])
        if self.dma_log is not None:
            self.dma_log.append((on_chip, aps[1]))
        self._emit_dma(on_chip, _ap(hbm_var, aps[1], dtype), load)

    def _emit_dma(self, on_chip, hbm, load):
        """``dma_copy`` between access patterns ``on_chip`` and ``hbm``."""
        dst, src = (on_chip, hbm) if load else (hbm, on_chip)
        self.emit(f"nisa.dma_copy(dst={dst}, src={src})")

    def _dma(self, sbuf_var, free, s, hbm_var, h, load, dtype=None):
        """``dma_copy`` the rows of offsets ``s`` (SBUF) and ``h`` (HBM),
        halving them until a pattern walks each part."""
        if self._try_dma(sbuf_var, free, s, hbm_var, h, load, dtype):
            return
        rows = s.shape[0]
        if rows > 1:
            half = rows // 2
            for part in (slice(0, half), slice(half, rows)):
                self._dma(
                    sbuf_var, free, s[part], hbm_var, h[part], load, dtype
                )
            return
        for c in range(s.shape[1]):
            one = slice(c, c + 1)
            self._dma(
                sbuf_var, free, s[:, one], hbm_var, h[:, one], load, dtype
            )

    # Compute.

    def _compute(self, op):
        """A compute op: a prim, or a fused group, emitted as the instruction
        its fused pattern names (``PATTERN_HANDLERS``) when it has one.

        Raises:
            NotImplementedError: The group's pattern is not one of
                Trainium's, or a GEMM writes no PSUM box alone.
        """
        pattern = None
        if op.WhichOneof("op_type") == "prim":
            prims = [op.prim]
        else:
            prims = list(op.fused.op_list)
            if op.fused.HasField("pattern"):
                pattern = op.fused.pattern
                if pattern not in PATTERN_HANDLERS:
                    raise NotImplementedError(f"{op.name}: pattern {pattern}")
        # The op's prims, whose scratch regions ``temp`` reads.
        self.prims = prims
        anchor = prims[0]
        if prims[-1].target == QUANTIZE_MX:
            # Its results are the scales, then the codes.
            dst = tuple(self._view(o.destination) for o in op.outputs[:2])
            for view, scales in zip(dst, (True, False)):
                self._use(view.box, "dst")
                self._mx_tile(view.box, scales, False)
        elif pattern is not None:
            # A group's results, in the order it gives them.
            dst = tuple(self._view(o.destination) for o in op.outputs)
            for view in dst:
                self._use(view.box, "dst")
        else:
            dst = self._view(op.outputs[0].destination)
            self._use(dst.box, "dst")
        if anchor.target == MAX_POOL2D:
            self._max_pool(prims, dst)
            return
        # A lone op writing PSUM, which TensorE's transpose or matmul does.
        into_psum = (
            len(prims) == 1 and isinstance(dst, View) and dst.box.level == LOCAL
        )
        psum_transpose = into_psum and anchor.target == "aten::transpose"
        # A fused pattern's GEMM is the pattern's instruction, not a step.
        gemm = pattern is None and anchor.target in (*GEMMS, LINEAR_MX)
        kmajor_gemm = (
            anchor.target in GEMMS and _gemm_layout(anchor).activation == "cx"
        )
        # TensorE reads its operands as stored (``_lay_out``).
        on_tensor_engine = (
            gemm or pattern == MATMUL_ACCUMULATE or psum_transpose
        )
        gemm_operands = []
        if anchor.target == LINEAR_MX:
            for key, (scales, k_major) in MX_OPERANDS.items():
                view = self._view(anchor.kwargs[key].tensor_box)
                self._mx_tile(view.box, scales, k_major)
                gemm_operands.append(view)
        elif on_tensor_engine:
            gemm_operands = [
                self._view(anchor.kwargs[k].tensor_box)
                for k in ("input", "weight", "other", "bias")
                if k in anchor.kwargs
                and anchor.kwargs[k].WhichOneof("arg_type") == "tensor_box"
            ]
        for view in gemm_operands:
            self._use(view.box, "gemm")
        if self.analyze:
            self.scalar_exps |= pattern == ACTIVATION_REDUCE
            for prim in prims[1:] if on_tensor_engine else prims:
                for arg in prim.kwargs.values():
                    if arg.WhichOneof("arg_type") != "tensor_box":
                        continue
                    if not arg.tensor_box.box.HasField("memory"):
                        continue
                    box = self._view(arg.tensor_box).box
                    self._use(box, "operand")
                    if box.level == DRAM and box.name not in self.reads:
                        self.reads.append(box.name)
            flat = (
                isinstance(dst, View)
                and pattern is None
                and not on_tensor_engine
                and not into_psum
                and anchor.target
                not in (MAX_POOL2D, RMS_NORM_KMAJOR, "aten::eye", *REDUCTIONS)
                and prims[-1].target != QUANTIZE_MX
            )
            self.op_uses.append(self._op_use(prims, dst, flat))
            return
        if pattern is not None:
            getattr(self, PATTERN_HANDLERS[pattern])(prims, dst)
        elif gemm and into_psum and anchor.target == LINEAR_MX:
            self._nc_matmul_mx(anchor, dst, accumulate=False)
        elif gemm and into_psum and kmajor_gemm:
            self._nc_matmul(anchor, dst, accumulate=False)
        elif gemm:
            raise NotImplementedError(
                f"{anchor.name}: a GEMM other than a K-major or MX one alone "
                "into PSUM"
            )
        elif anchor.target == RMS_NORM_KMAJOR:
            self._rms_norm_kmajor(prims, dst)
        elif anchor.target in REDUCTIONS:
            self._reduce(prims, dst)
        elif psum_transpose:
            self._psum_transpose(anchor, dst)
        elif anchor.target == "aten::eye":
            self._eye(dst, 1.0)
        elif any(p.target in ROWWISE for p in prims):
            self._rowwise(prims, dst)
        else:
            self._elementwise(prims, dst)

    def _op_use(self, prims, dst, flat):
        """The ``OpUse`` of a compute op of ``prims`` writing ``dst``, which
        may run flat if ``flat`` and its shapes allow: an elementwise chain,
        or an RMS norm alone on one row, whose result is in SBUF and whose
        operands are each the result's size or one element.  An op's rows
        take ``min(rows, 128)`` partitions of its first result, a row op's
        of its input; another op's, every partition.  An operand of fewer
        rows than the partitions an elementwise or row op's rows take is
        broadcast along them (``OpUse.broadcast``)."""
        operands = [
            self._view(arg.tensor_box)
            for prim in prims
            for key, arg in prim.kwargs.items()
            if arg.WhichOneof("arg_type") == "tensor_box"
            and arg.tensor_box.box.HasField("memory")
            and "." not in key
        ]
        touched = {v.box.name for v in operands if v.box.level in ON_CHIP}
        views = [dst] if isinstance(dst, View) else list(dst)
        touched |= {v.box.name for v in views}
        rowwise = [p for p in prims if p.target in ROWWISE]
        if not flat:
            parts = PARTITIONS
        elif rowwise:
            shape = tuple(rowwise[0].kwargs["input"].tensor_box.output_shape)
            parts = min(_rows_cols(shape)[0], PARTITIONS)
        else:
            parts = min(
                _rows_cols(self._result_shape(prims[0], {}))[0], PARTITIONS
            )
        broadcast = frozenset(
            v.box.name for v in operands if flat and v.box.rows < parts
        )
        cols = math.prod(dst.box.shape) if isinstance(dst, View) else 0
        rows = {dst.box.name} if isinstance(dst, View) else set()
        flat = (
            flat
            and (
                not rowwise
                or [p.target for p in prims] == ["aten::rms_norm"]
                and dst.box.rows == 1
            )
            and dst.box.level == SCRATCHPAD
            and dst.covers_box()
            and spread_partitions(cols) > 1
        )
        for view in operands:
            if math.prod(view.shape) == 1 and view.box.rows == 1:
                continue
            if (
                view.box.level == SCRATCHPAD
                and math.prod(view.box.shape) == cols
                and view.covers_box()
            ):
                rows.add(view.box.name)
            else:
                flat = False
        return OpUse(
            flat, parts, cols, frozenset(rows), frozenset(touched), broadcast
        )

    def _gemm_extents(self, gemm, act, weight):
        """``(batch, batch_a, batch_w, m, k, n)`` of a GEMM step: its result's
        batch and its operands', the rows, the contraction and the columns;
        a K-major activation is ``[..., K, M]``."""
        if _gemm_layout(gemm).activation == "cx":
            *batch_a, k, m = act.shape
        else:
            *batch_a, m, k = act.shape
        *batch_w, k_w, n = weight.shape
        if k != k_w:
            raise NotImplementedError("a GEMM weight not [K, N]")
        batch = tuple(np.broadcast_shapes(tuple(batch_a), tuple(batch_w)))
        return batch, tuple(batch_a), tuple(batch_w), m, k, n

    def _block(self, view, index, chunks, x, width):
        """The access to a GEMM operand ``[..., K, X]`` held contraction on
        the partitions: the rows of the contraction in ``chunks``, ``(start,
        stop)`` each (``contraction_runs``), and columns ``x .. x + width``
        of its batch element ``index``.  A ``double_row`` matmul's two
        128-row chunks are read ``[128, 2, width]``, one step of the first
        free dim apart.

        Raises:
            NotImplementedError: No access pattern of at most three free
                dims reads the block, or one reads a ``double_row`` block
                with its chunks other than a multiple of
                ``DOUBLE_ROW_STEP`` apart.
        """
        key = ("block", view.key(), index, tuple(chunks), x, width)
        if key not in self.memo:
            k, cols = view.shape[-2:]
            rows = index * k + np.concatenate([np.arange(*c) for c in chunks])
            linear = rows[:, None] * cols + x + np.arange(width)[None, :]
            offsets = self.flat(view, linear)
            first, *second = offsets.reshape(len(chunks), -1, width)
            aps = fit([(self.free(view.box), first)], len(first))
            if aps is not None and second:
                (steps, offset), apart = aps[0], second[0][0, 0] - first[0, 0]
                paired = [steps[0], [int(apart), 2], *steps[1:]]
                aps = None
                # The partition dim and at most three free dims.
                if (
                    len(paired) <= 4
                    and apart % DOUBLE_ROW_STEP == 0
                    and np.array_equal(second[0], first + apart)
                ):
                    aps = [(paired, offset)]
            if aps is None:
                raise NotImplementedError(f"{view.box.name}: a GEMM block")
            self.memo[key] = _ap(self.var(view.box, view.slot), aps[0])
        return self.memo[key]

    def _quantize_mx(self, prim, source, dst):
        """``quantize_mx`` of the stored chunk ``source`` into ``dst``, its
        (scales, codes) tiles: the chunk in bfloat16, then per run of its
        columns within one 512-element chunk of the contraction, every 4th
        column transposed onto the partitions 4 times into a staging tile
        and ``quantize_mx`` over it -- 8 partitions of 4 columns are its block
        of 32 consecutive elements."""
        kw = self._kwargs(prim)
        scales_view, codes_view = dst
        fmt = _fp8(codes_view.box.dtype)
        emax = MX_EMAX.get(fmt.ebits) if fmt is not None else None
        if (
            emax is None
            or list(kw["axes"]) != [-1]
            or kw["block_size"] != MX_BLOCK
            or not kw["force_scale_power_of_two"]
            or math.floor(math.log2(kw["quant_max"])) != emax
        ):
            raise NotImplementedError(f"{prim.name}: not Trn3's MX rule")
        r, w = source.coords.shape[1:]
        if source.dtype != "bfloat16" or source.flat is not None:
            temp = self.temp("q", w)
            out = Chunk(temp, w, source.shape, source.coords, dtype="bfloat16")
            self._copy_chunk(out, source)
            source = out
        # Where the chunk's rows and columns land in the codes tile.
        if math.prod(codes_view.shape) != math.prod(source.shape):
            raise NotImplementedError(f"{prim.name}: into another shape")
        box = codes_view.box
        mx, padded = self.mx_tiles[box.name]
        linear = np.ravel_multi_index(source.coords, source.shape)
        x, k = mx.coords(codes_view.positions(linear))
        x0, k0 = int(x[0, 0]), int(k[0, 0])
        if not (
            np.all(x == x0 + np.arange(r)[:, None])
            and np.all(k == k0 + np.arange(w))
        ):
            raise NotImplementedError(f"{prim.name}: a scattered chunk")
        codes = self.var(codes_view.box, codes_view.slot)
        scales = self.var(scales_view.box, scales_view.slot)
        free = mx.tile(padded)[1]
        stage_free = MX_PACK * PARTITIONS
        # Runs of the columns within one chunk of the contraction.
        cuts = range(-k0 % MX_CHUNK, w, MX_CHUNK)
        bounds = [0] + [c for c in cuts if c > 0] + [w]
        for lo, hi in zip(bounds, bounds[1:]):
            kk = k0 + lo
            parts = (hi - lo) // MX_PACK
            if (hi - lo) % (QUADRANT * MX_PACK) or kk % (QUADRANT * MX_PACK):
                raise NotImplementedError(f"{prim.name}: columns {kk}+")
            staging = self.temp("qs", stage_free)
            for j in range(MX_PACK):
                data = (
                    f"{source.var}.ap(pattern=[[{source.free}, {r}], "
                    f"[{MX_PACK}, {parts}]], offset={lo + j})"
                )
                transposed = self._transpose(data, r, parts)
                self.emit(
                    f"nisa.tensor_copy(dst={staging}.ap(pattern="
                    f"[[{stage_free}, {parts}], [{MX_PACK}, {r}]], "
                    f"offset={j}), src={transposed})"
                )
            part = (kk % MX_CHUNK) // MX_PACK
            offset = part * free + (kk // MX_CHUNK) * padded + x0
            region = f"pattern=[[{free}, {parts}], [1, {r}]], offset={offset}"
            self.emit(
                f"nisa.quantize_mx(dst={codes}.ap({region}), "
                f"src={staging}[0:{parts}, 0:{MX_PACK * r}], "
                f"dst_scale={scales}.ap({region}))"
            )

    def _elementwise(self, prims, dst):
        """Elementwise ops, chunk by chunk of the first op's result: 128 rows
        and up to ``CHUNK_FREE`` columns at a time, a tile cut across the
        partitions (``Layout.spread``) on all of them, or, reading a tile
        from PSUM in whole 128-row blocks, every block at once -- PSUM holds
        a tile's blocks side by side within a bank."""
        shape = self._result_shape(prims[0], {})
        rows, cols = _rows_cols(shape)
        # A scratch region (``temp``), named ``<group>.<kind>_<cols>``, is
        # not an operand.
        reads_psum = any(
            arg.tensor_box.box.memory.level == LOCAL
            for prim in prims
            for key, arg in prim.kwargs.items()
            if arg.WhichOneof("arg_type") == "tensor_box" and "." not in key
        )
        layout = (
            self.layouts.get(dst.box.name) if isinstance(dst, View) else None
        )
        if layout is not None and layout.spread:
            # Partition ``p`` holds the tile's elements ``p * cols ..``.
            for lo, hi in _runs(layout.cols, CHUNK_FREE):
                linear = np.arange(layout.parts)[:, None] * layout.cols
                linear = linear + np.arange(lo, hi)[None, :]
                coords = np.stack(np.unravel_index(linear, shape))
                self._chain(prims, {}, dst, Chunk(None, None, shape, coords))
            return
        if reads_psum and rows > PARTITIONS and rows % PARTITIONS == 0:
            # Partition ``p`` holds row ``b * 128 + p`` of each block ``b``.
            row = np.arange(PARTITIONS)[:, None] + np.arange(
                0, rows, PARTITIONS
            )
            linear = (row[:, :, None] * cols + np.arange(cols)).reshape(
                PARTITIONS, -1
            )
            coords = np.stack(np.unravel_index(linear, shape))
            self._chain(prims, {}, dst, Chunk(None, None, shape, coords))
            return
        for a, b in _runs(rows, PARTITIONS):
            for lo, hi in _runs(cols, CHUNK_FREE):
                linear = np.arange(a, b)[:, None] * cols
                linear = linear + np.arange(lo, hi)[None, :]
                coords = np.stack(np.unravel_index(linear, shape))
                self._chain(prims, {}, dst, Chunk(None, None, shape, coords))

    def _result_shape(self, prim, values):
        """The shape of ``prim``'s result."""
        shapes = []
        for key in ("condition", "input", "other"):
            arg = prim.kwargs.get(key)
            if arg is None or arg.WhichOneof("arg_type") != "tensor_box":
                continue
            ref = arg.tensor_box
            if ref.box.HasField("memory"):
                shapes.append(self._view(ref).shape)
            else:
                shapes.append(
                    tuple(ref.output_shape) or values[ref.box.node].shape
                )
        return tuple(np.broadcast_shapes(*shapes))

    def _chain(self, prims, values, dst, grid=None):
        """Run ``prims`` on one chunk and write the last result to ``dst``.

        ``values`` holds the chunk's results so far, by name; ``grid`` is the
        chunk of the first op's result when no result exists yet.  A
        relayout only re-maps the coordinates of what it reads; the last
        computing op writes ``dst`` directly, through the relayouts after
        it, and with no computing op the last relayout's result is copied
        there.
        """
        computing = [
            i for i, p in enumerate(prims) if p.target not in RELAYOUTS
        ]
        last = computing[-1] if computing else -1
        result = next(iter(values.values()), grid)
        for index, prim in enumerate(prims):
            if prim.target in RELAYOUTS:
                values[prim.name] = result = self._relayout(prim, values)
                continue
            if prim.target == QUANTIZE_MX:
                if index != len(prims) - 1:
                    raise NotImplementedError(
                        f"{prim.name}: a quantize_mx before the last op"
                    )
                arg = prim.kwargs["input"].tensor_box
                if not arg.box.HasField("memory"):
                    source = self._read(arg, values)
                else:
                    # A stored tile, read where it lies.
                    kind, value = self._operand(
                        prim.kwargs["input"], values, grid
                    )
                    if kind != "tile":
                        raise NotImplementedError(f"{prim.name}: {kind}")
                    var, free, flat = value
                    dtype = self._view(arg).box.dtype
                    source = Chunk(
                        var, free, grid.shape, grid.coords, flat, dtype
                    )
                self._quantize_mx(prim, source, dst)
                return
            chunks = [
                self._read(a.tensor_box, values)
                for a in prim.kwargs.values()
                if a.WhichOneof("arg_type") == "tensor_box"
                and not a.tensor_box.box.HasField("memory")
            ]
            source = chunks[0] if chunks else grid
            shape = self._result_shape(prim, values)
            if shape != source.shape:
                raise NotImplementedError(f"{prim.name}: broadcasts a result")
            following = [p.target for p in prims[index + 1 :]]
            following = [t for t in following if t not in RELAYOUTS]
            if index == last:
                # Where each element lands in ``dst``, past the relayouts.
                landing = {
                    **values,
                    prim.name: Chunk(None, None, shape, source.coords),
                }
                moved = landing[prim.name]
                for after in prims[index + 1 :]:
                    moved = landing[after.name] = self._relayout(after, landing)
                out = replace(
                    self._destination(dst, moved),
                    shape=shape,
                    coords=source.coords,
                )
            elif following[:1] == [QUANTIZE_MX]:
                # What ``quantize_mx`` reads, in its own type.
                r, w = source.coords.shape[1:]
                temp = self.temp("q", w)
                out = Chunk(temp, w, shape, source.coords, dtype="bfloat16")
            else:
                r, w = source.coords.shape[1:]
                temp = self.temp("t", w)
                out = Chunk(temp, w, shape, source.coords)
            self._apply(prim, out, values, source)
            values[prim.name] = result = out
        if not computing:
            self._copy_chunk(self._destination(dst, result), result)

    def _relayout(self, prim, values):
        """The result relayout ``prim`` reads (``values``, by name), its
        coordinates re-mapped as ``prim`` moves its elements."""
        result = self._read(prim.kwargs["input"].tensor_box, values)
        if prim.target != "aten::transpose":
            return result
        return result.transposed(
            prim.kwargs["dim0"].scalar.int_value,
            prim.kwargs["dim1"].scalar.int_value,
        )

    def _read(self, ref, values):
        """The stored result a memoryless reference names, seen as its
        view."""
        chunk = values[ref.box.node]
        shape = tuple(ref.output_shape)
        return (
            chunk.reshaped(shape) if shape and shape != chunk.shape else chunk
        )

    def _destination(self, dst, result):
        """``dst`` as the chunk ``result`` is written to."""
        if math.prod(result.shape) != math.prod(dst.shape):
            raise NotImplementedError("a result and its destination differ")
        linear = np.ravel_multi_index(result.coords, result.shape)
        return Chunk(
            self.var(dst.box, dst.slot),
            self.free(dst.box),
            result.shape,
            result.coords,
            self.flat(dst, linear),
            dst.box.dtype,
            dst.box.name,
        )

    def _operand(self, arg, values, source):
        """An elementwise operand: ``("tile", (var, free, offsets))``,
        ``("column", ap)`` (one value per row) or ``("scalar", literal)``."""
        if arg.WhichOneof("arg_type") == "scalar":
            return "scalar", repr(float(self._scalar(arg.scalar)))
        ref = arg.tensor_box
        if not ref.box.HasField("memory"):
            chunk = self._read(ref, values)
            return "tile", (chunk.var, chunk.free, chunk.offsets())
        view = self._view(ref)
        coords = source.coords
        r = coords.shape[1]
        if view.box.level == IMMEDIATE:
            return "scalar", repr(self.immediates[view.box.name])
        if view.box.level == DRAM:
            # A scalar read where it lies, as Voyager reads it: loaded onto
            # every partition, one value per row.
            if math.prod(view.shape) != 1:
                raise NotImplementedError("an elementwise operand in HBM")
            column = self.temp("sc", 1)
            offset = int(self.flat(view, np.zeros(1, dtype=int))[0])
            self.emit(
                f"nisa.dma_copy(dst={column}, src={self.var(view.box)}.ap("
                f"pattern=[[0, {PARTITIONS}], [1, 1]], offset={offset}))"
            )
            return "column", f"{column}[0:{r}, 0:1]"
        # The operand's coordinates, broadcast from the result's.
        lead = len(source.shape) - len(view.shape)
        own = [
            coords[lead + d] if size > 1 else np.zeros_like(coords[0])
            for d, size in enumerate(view.shape)
        ]
        linear = np.ravel_multi_index(own, view.shape)
        # One value per row of a result wider than one column.
        per_row = view.shape[-1] == 1 and source.shape[-1] > 1
        flat = self.flat(view, linear, np.arange(r)[:, None])
        var, free = self.var(view.box, view.slot), self.free(view.box)
        if per_row:
            aps = fit([(free, flat[:, :1])], r)
            if aps is None:
                raise NotImplementedError("a column no pattern walks")
            column = _ap(var, aps[0])
            if view.box.dtype != "float32":
                # A per-row scalar operand is float32 to the engines.
                temp = self.temp("sc", 1)
                self.emit(
                    f"nisa.tensor_copy(dst={temp}[0:{r}, 0:1], src={column})"
                )
                column = f"{temp}[0:{r}, 0:1]"
            return "column", column
        return "tile", (var, free, flat)

    def _apply(self, prim, out, values, source):
        """Emit the instruction computing ``prim`` into ``out``."""
        target = prim.target
        if target in FILLS:
            aps = fit([(out.free, out.offsets())], source.coords.shape[1])
            if aps is None:
                raise CrossesPartitions(out.box or prim.name)
            key = FILLS[target]
            value = float(self._scalar(prim.kwargs[key].scalar)) if key else 0.0
            # Beside exponentials on ScalarE (``scalar_exps``), which keep
            # VectorE busy too, a fill runs on GpSimd.
            on = ", engine=nisa.gpsimd_engine" if self.scalar_exps else ""
            self.emit(
                f"nisa.memset(dst={_ap(out.var, aps[0])}, value={value!r}"
                f"{on})"
            )
            return
        keys = ("condition", "input", "other")
        if target in (QUANTIZE, DEQUANTIZE):
            keys = ("input", "scale")
            extra = [
                k
                for k in ("zero_point", "axes", "block_size")
                if self._kwargs(prim).get(k) is not None
            ]
            if extra:
                raise NotImplementedError(f"{prim.name}: {extra}")
        operands = [
            self._operand(prim.kwargs[key], values, source)
            for key in keys
            if key in prim.kwargs
        ]
        tiles = [value for kind, value in operands if kind == "tile"]
        # Beside exponentials on ScalarE (``scalar_exps``), a
        # ``tensor_scalar`` runs on VectorE.
        engine = ", engine=nisa.vector_engine" if self.scalar_exps else ""
        aps = fit(
            [(out.free, out.offsets())] + [(f, o) for _, f, o in tiles],
            source.coords.shape[1],
        )
        if aps is None:
            raise CrossesPartitions(out.box or prim.name)
        dst = _ap(out.var, aps[0])
        tile_aps = iter(_ap(v, ap) for (v, _, _), ap in zip(tiles, aps[1:]))
        args = [next(tile_aps) if k == "tile" else v for k, v in operands]
        kinds = [kind for kind, _ in operands]
        if target in BINARY_OPS:
            alpha = prim.kwargs.get("alpha")
            if alpha is not None and self._scalar(alpha.scalar) != 1:
                raise NotImplementedError(f"{prim.name}: alpha")
            op = BINARY_OPS[target]
            if target == "aten::__and__" and out.dtype != "bool":
                # ``logical_and`` is ``&`` only on 0/1 booleans.
                raise NotImplementedError(f"{prim.name}: a bitwise and")
            if kinds == ["tile", "tile"]:
                self.emit(
                    f"nisa.tensor_tensor(dst={dst}, data1={args[0]}, "
                    f"data2={args[1]}, op={op})"
                )
            elif kinds[0] == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={dst}, data={args[0]}, "
                    f"op0={op}, operand0={args[1]}{engine})"
                )
            elif kinds[1] == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={dst}, data={args[1]}, "
                    f"op0={op}, operand0={args[0]}, reverse0=True{engine})"
                )
            else:
                raise NotImplementedError(f"{prim.name}: no tile operand")
        elif target == "aten::div" and kinds[0] == "tile":
            inverse = self._reciprocal(args[1], kinds[1], source)
            if kinds[1] == "tile":
                self.emit(
                    f"nisa.tensor_tensor(dst={dst}, data1={args[0]}, "
                    f"data2={inverse}, op=nl.multiply)"
                )
            else:
                self.emit(
                    f"nisa.tensor_scalar(dst={dst}, data={args[0]}, "
                    f"op0=nl.multiply, operand0={inverse}{engine})"
                )
        elif target == "aten::where":
            self._where(dst, args, kinds, source.coords.shape[1:])
        elif target == QUANTIZE:
            fmt = _fp8(out.dtype)
            if fmt is None:
                raise NotImplementedError(f"{prim.name}: into {out.dtype}")
            # The quotient is rounded to bfloat16, as Voyager's quantize
            # rounds input / scale, then saturated and cast to FP8.
            inverse = self._reciprocal(args[1], kinds[1], source)
            r, w = source.coords.shape[1:]
            temp = self.temp("qt", w)
            quotient = f"{temp}[0:{r}, 0:{w}]"
            self.emit(
                f"nisa.tensor_scalar(dst={quotient}, data={args[0]}, "
                f"op0=nl.multiply, operand0={inverse})"
            )
            self.emit(
                f"nisa.tensor_scalar(dst={dst}, data={quotient}, "
                f"op0=nl.minimum, operand0={fmt.max_norm!r}, "
                f"op1=nl.maximum, operand1={-fmt.max_norm!r})"
            )
        elif target == DEQUANTIZE:
            self.emit(
                f"nisa.tensor_scalar(dst={dst}, data={args[0]}, "
                f"op0=nl.multiply, operand0={args[1]})"
            )
        elif target == "aten::clone":
            self.emit(f"nisa.tensor_copy(dst={dst}, src={args[0]})")
        elif target == "aten::neg":
            self.emit(
                f"nisa.tensor_scalar(dst={dst}, data={args[0]}, "
                f"op0=nl.multiply, operand0=-1.0)"
            )
        elif target in ACTIVATIONS:
            self.emit(
                f"nisa.activation(dst={dst}, op={ACTIVATIONS[target]}, "
                f"data={args[0]})"
            )
        elif target in SINES:
            self._sine(dst, args[0], SINES[target], source.coords.shape[1:])
        else:
            raise NotImplementedError(f"{prim.name}: {target}")

    def _transpose(self, data, r, c):
        """Transpose ``data``, an ``[r, c]`` bfloat16 access, on TensorE into
        the next of a ``quantize_mx``'s PSUM scratch tiles (``temp``); the
        access to the ``[c, r]`` result there."""
        psum = self.temp("tp", PARTITIONS)
        result = f"{psum}.ap(pattern=[[{PARTITIONS}, {c}], [1, {r}]], offset=0)"
        self.emit(f"nisa.nc_transpose(dst={result}, data={data})")
        return result

    def _reciprocal(self, value, kind, source):
        """``1 / value`` as an operand: a literal for a scalar, else computed
        into a scratch tile of the same kind (there is no ALU divide)."""
        if kind == "scalar":
            return repr(1.0 / float(value))
        r, w = source.coords.shape[1:]
        if kind == "column":
            temp = self.temp("sc", 1)
            inverse = f"{temp}[0:{r}, 0:1]"
        else:
            temp = self.temp("t", w)
            inverse = f"{temp}[0:{r}, 0:{w}]"
        self.emit(f"nisa.reciprocal(dst={inverse}, data={value})")
        return inverse

    def _sine(self, dst, angle, shift, chunk):
        """``sin(angle + shift)`` on ScalarE, the angle first brought into
        [-pi, pi]: ``k`` whole turns off by a cast to int32 (whichever way it
        rounds), and one more if that leaves it above pi."""
        r, w = chunk
        shifted, reduced = (
            self.temp("t", w) + f"[0:{r}, 0:{w}]" for _ in range(2)
        )
        turns = self.temp("k", w) + f"[0:{r}, 0:{w}]"
        turn = 2 * math.pi
        self.emit(
            f"nisa.tensor_scalar(dst={shifted}, data={angle}, op0=nl.add, "
            f"operand0={shift!r})"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={turns}, data={shifted}, "
            f"op0=nl.multiply, operand0={1 / turn!r})"
        )
        self.emit(
            f"nisa.scalar_tensor_tensor(dst={reduced}, data={turns}, "
            f"op0=nl.multiply, operand0={-turn!r}, op1=nl.add, "
            f"operand1={shifted})"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={shifted}, data={reduced}, "
            f"op0=nl.greater, operand0={math.pi!r})"
        )
        self.emit(
            f"nisa.scalar_tensor_tensor(dst={reduced}, data={shifted}, "
            f"op0=nl.multiply, operand0={-turn!r}, op1=nl.add, "
            f"operand1={reduced})"
        )
        self.emit(f"nisa.activation(dst={dst}, op=nl.sin, data={reduced})")

    def _where(self, dst, args, kinds, chunk):
        """``where(c, x, y)`` as ``c * x + (1 - c) * y``: exact for a
        condition of 0 or 1, whatever ``x`` and ``y`` are.  The condition is
        a tile or one value per row; ``x`` and ``y`` may be scalars."""
        condition, x, y = args
        r, w = chunk
        if kinds[0] == "tile":
            other = self.temp("t", w)
            inverse = f"{other}[0:{r}, 0:{w}]"
        elif kinds[0] == "column":
            other = self.temp("sc", 1)
            inverse = f"{other}[0:{r}, 0:1]"
        else:
            raise NotImplementedError("where on a scalar condition")
        self.emit(
            f"nisa.tensor_scalar(dst={inverse}, data={condition}, "
            f"op0=nl.subtract, operand0=1.0, reverse0=True)"
        )
        terms = []
        for gate, value, kind in (
            (condition, x, kinds[1]),
            (inverse, y, kinds[2]),
        ):
            temp = self.temp("t", w)
            term = f"{temp}[0:{r}, 0:{w}]"
            if kind == "tile" and kinds[0] == "tile":
                self.emit(
                    f"nisa.tensor_tensor(dst={term}, data1={gate}, "
                    f"data2={value}, op=nl.multiply)"
                )
            elif kind == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={term}, data={value}, "
                    f"op0=nl.multiply, operand0={gate})"
                )
            elif kinds[0] == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={term}, data={gate}, "
                    f"op0=nl.multiply, operand0={value})"
                )
            else:
                raise NotImplementedError("where with no tile operand")
            terms.append(term)
        self.emit(
            f"nisa.tensor_tensor(dst={dst}, data1={terms[0]}, "
            f"data2={terms[1]}, op=nl.add)"
        )

    def _copy_chunk(self, out, result):
        """Write the stored ``result`` to ``out``."""
        aps = fit(
            [(out.free, out.offsets()), (result.free, result.offsets())],
            result.coords.shape[1],
        )
        if aps is None:
            raise CrossesPartitions(out.box)
        self.emit(
            f"nisa.tensor_copy(dst={_ap(out.var, aps[0])}, "
            f"src={_ap(result.var, aps[1])})"
        )

    def _rows_ap(self, view, start, r, lo, hi):
        """``view``'s rows ``start .. start + r``, columns ``lo .. hi``; a
        replicated view on ``r`` partitions."""
        cols = view.shape[-1]
        if self.layouts[view.box.name].replicated:
            rows = np.zeros((r, 1), dtype=int)
        else:
            rows = (start + np.arange(r))[:, None]
        linear = rows * cols + np.arange(lo, hi)[None, :]
        flat = self.flat(view, linear, np.arange(r)[:, None])
        aps = fit([(self.free(view.box), flat)], r)
        if aps is None:
            raise NotImplementedError("rows no pattern walks")
        return _ap(self.var(view.box, view.slot), aps[0])

    def _rowwise(self, prims, dst):
        """A row-wise op (a layer norm, an RMS norm, a softmax) and the ops
        fused before or after it, 128 rows at a time; an RMS norm alone of a
        row cut across the partitions by ``_rms_norm_spread``.  The ops
        before it write each block's rows to ``dst``, where the op then runs
        in place; the op writes ``dst`` itself, or a scratch tile the ops
        after it read."""
        index = next(i for i, p in enumerate(prims) if p.target in ROWWISE)
        pre, prim, post = prims[:index], prims[index], prims[index + 1 :]
        if isinstance(dst, View) and self.layouts[dst.box.name].spread:
            self._rms_norm_spread(prim, dst)
            return
        if pre and post:
            raise NotImplementedError(f"{prim.name}: ops before and after")
        rows_op = getattr(self, ROWWISE[prim.target])
        x = prim.kwargs["input"].tensor_box
        shape = tuple(x.output_shape)
        source = dst if pre else self._view(x)
        rows, cols = _rows_cols(shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            if pre:
                for lo, hi in _runs(cols, CHUNK_FREE):
                    linear = np.arange(start, stop)[:, None] * cols
                    linear = linear + np.arange(lo, hi)[None, :]
                    coords = np.stack(np.unravel_index(linear, shape))
                    self._chain(pre, {}, dst, Chunk(None, None, shape, coords))
            data = lambda lo, hi: self._rows_ap(source, start, r, lo, hi)
            if not post:
                rows_op(prim, data, r, self._rows_ap(dst, start, r, 0, cols))
                continue
            temp = self.temp("rw", cols)
            rows_op(prim, data, r, f"{temp}[0:{r}, 0:{cols}]")
            linear = (start + np.arange(r))[:, None] * cols + np.arange(cols)
            coords = np.stack(np.unravel_index(linear, shape))
            result = Chunk(temp, cols, shape, coords)
            self._chain(post, {prim.name: result}, dst)

    def _layer_norm(self, prim, rows, r, out):
        """Layer norm over the last dim of the ``r`` rows ``rows(lo, hi)``
        reads (columns ``lo .. hi``): the moments (``bn_stats`` /
        ``bn_aggr``), then normalize, scale and shift in ``out``.  With both
        affine operands that is two ``scalar_tensor_tensor`` passes,
        ``(x - mean) * weight`` and then ``* rstd + bias``."""
        kw = self._kwargs(prim)
        cols = prim.kwargs["input"].tensor_box.output_shape[-1]
        if list(kw["normalized_shape"]) != [cols]:
            raise NotImplementedError("a layer norm over more than one dim")
        affine = [
            (self._view(prim.kwargs[key].tensor_box), op)
            for key, op in (("weight", "nl.multiply"), ("bias", "nl.add"))
            if kw.get(key) is not None
        ]
        pieces = _runs(cols, STATS_FREE)
        data = rows(0, cols)
        stats = self.temp("st", 6 * len(pieces))
        for q, (lo, hi) in enumerate(pieces):
            self.emit(
                f"nisa.bn_stats(dst={stats}[0:{r}, {6 * q}:{6 * q + 6}], "
                f"data={rows(lo, hi)})"
            )
        moments = self.temp("mv", 2)
        self.emit(
            f"nisa.bn_aggr(dst={moments}[0:{r}, 0:2], "
            f"data={stats}[0:{r}, 0:{6 * len(pieces)}])"
        )
        rstd = self.temp("rs", 1)
        self.emit(
            f"nisa.tensor_scalar(dst={rstd}[0:{r}, 0:1], "
            f"data={moments}[0:{r}, 1:2], op0=nl.add, "
            f"operand0={float(kw['eps'])!r})"
        )
        self.emit(
            f"nisa.activation(dst={rstd}[0:{r}, 0:1], op=nl.rsqrt, "
            f"data={rstd}[0:{r}, 0:1])"
        )
        mean = f"{moments}[0:{r}, 0:1]"
        inv = f"{rstd}[0:{r}, 0:1]"
        if len(affine) == 2:
            weight, bias = (
                self._rows_ap(view, 0, r, 0, cols) for view, _ in affine
            )
            self.emit(
                f"nisa.scalar_tensor_tensor(dst={out}, data={data}, "
                f"op0=nl.subtract, operand0={mean}, op1=nl.multiply, "
                f"operand1={weight})"
            )
            self.emit(
                f"nisa.scalar_tensor_tensor(dst={out}, data={out}, "
                f"op0=nl.multiply, operand0={inv}, op1=nl.add, "
                f"operand1={bias})"
            )
            return
        self.emit(
            f"nisa.tensor_scalar(dst={out}, data={data}, "
            f"op0=nl.subtract, operand0={mean}, "
            f"op1=nl.multiply, operand1={inv})"
        )
        for view, op in affine:
            self.emit(
                f"nisa.tensor_tensor(dst={out}, data1={out}, "
                f"data2={self._rows_ap(view, 0, r, 0, cols)}, op={op})"
            )

    def _rms_norm(self, prim, rows, r, out):
        """RMS norm over the last dim of the ``r`` rows ``rows(lo, hi)``
        reads (columns ``lo .. hi``): the moments (``bn_stats`` /
        ``bn_aggr``), the mean square as ``mean ** 2 + variance`` plus eps,
        its rsqrt, then one pass scaling by that and the weight into
        ``out``."""
        kw = self._kwargs(prim)
        cols = prim.kwargs["input"].tensor_box.output_shape[-1]
        if list(kw["normalized_shape"]) != [cols]:
            raise NotImplementedError("an RMS norm over more than one dim")
        weight = self._view(prim.kwargs["weight"].tensor_box)
        pieces = _runs(cols, STATS_FREE)
        stats = self.temp("st", 6 * len(pieces))
        for q, (lo, hi) in enumerate(pieces):
            self.emit(
                f"nisa.bn_stats(dst={stats}[0:{r}, {6 * q}:{6 * q + 6}], "
                f"data={rows(lo, hi)})"
            )
        moments = self.temp("mv", 2)
        self.emit(
            f"nisa.bn_aggr(dst={moments}[0:{r}, 0:2], "
            f"data={stats}[0:{r}, 0:{6 * len(pieces)}])"
        )
        mean = f"{moments}[0:{r}, 0:1]"
        rstd = self.temp("rs", 1) + f"[0:{r}, 0:1]"
        self.emit(
            f"nisa.scalar_tensor_tensor(dst={rstd}, data={mean}, "
            f"op0=nl.multiply, operand0={mean}, op1=nl.add, "
            f"operand1={moments}[0:{r}, 1:2])"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={rstd}, data={rstd}, op0=nl.add, "
            f"operand0={float(kw['eps'])!r})"
        )
        self.emit(f"nisa.activation(dst={rstd}, op=nl.rsqrt, data={rstd})")
        self.emit(
            f"nisa.scalar_tensor_tensor(dst={out}, "
            f"data={rows(0, cols)}, op0=nl.multiply, "
            f"operand0={rstd}, op1=nl.multiply, "
            f"operand1={self._rows_ap(weight, 0, r, 0, cols)})"
        )

    def _rms_norm_spread(self, prim, dst):
        """RMS norm of a row cut across the partitions (``Layout.spread``)
        into ``dst``, laid out the same: each partition's squares on
        ScalarE, summed on the way, the partitions' sums totalled onto every
        partition by a ones-matmul into PSUM, the mean square plus eps and
        its rsqrt, then one pass scaling by it and the weight."""
        kw = self._kwargs(prim)
        layout = self.layouts[dst.box.name]
        parts, free = layout.parts, layout.cols
        if list(kw["normalized_shape"]) != [parts * free]:
            raise NotImplementedError("an RMS norm over more than one dim")
        linear = np.arange(parts * free).reshape(parts, free)
        x, weight = (
            self._view(prim.kwargs[key].tensor_box)
            for key in ("input", "weight")
        )
        rows = []
        for view in (x, weight, dst):
            aps = fit([(self.free(view.box), self.flat(view, linear))], parts)
            if aps is None:
                raise NotImplementedError(f"{view.box.name}: a spread row")
            rows.append(_ap(self.var(view.box, view.slot), aps[0]))
        data, scale, out = rows
        ones = self.temp("one", PARTITIONS)
        if ("ones", ones) not in self.memo:
            self.memo["ones", ones] = True
            self.emit(f"nisa.memset(dst={ones}, value=1.0)")
        squares = self.temp("sq", free) + f"[0:{parts}, 0:{free}]"
        sums = self.temp("ss", 1) + f"[0:{parts}, 0:1]"
        total = self.temp("acc", 1) + f"[0:{parts}, 0:1]"
        rstd = self.temp("rs", 1) + f"[0:{parts}, 0:1]"
        self.emit(
            f"nisa.activation(dst={squares}, op=nl.square, data={data}, "
            f"reduce_op=nl.add, reduce_res={sums}, "
            "reduce_cmd=nisa.reduce_cmd.reset_reduce)"
        )
        self.emit(
            f"nisa.nc_matmul(dst={total}, stationary={ones}[0:{parts}, "
            f"0:{parts}], moving={sums}, accumulate=False)"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={rstd}, data={total}, op0=nl.multiply, "
            f"operand0={1 / (parts * free)!r}, op1=nl.add, "
            f"operand1={float(kw['eps'])!r})"
        )
        self.emit(f"nisa.activation(dst={rstd}, op=nl.rsqrt, data={rstd})")
        self.emit(
            f"nisa.scalar_tensor_tensor(dst={out}, data={data}, "
            f"op0=nl.multiply, operand0={rstd}, op1=nl.multiply, "
            f"operand1={scale})"
        )

    def _rms_norm_kmajor(self, prims, dst):
        """RMS norm of a K-major ``[..., H, T]`` tile over ``H``, which lies
        across the partitions, into ``dst``.  Per 512 columns: the 128-row
        blocks squared on ScalarE, ``SQUARE_BLOCKS`` an instruction, the
        squares summed over the blocks and their partitions by ones-matmuls
        accumulating in PSUM -- which puts the sum on every partition -- the
        mean square's rsqrt, then per block one pass scaling by it and by
        the ``[H, 1]`` weight, one value per row.  The pass writes ``dst``
        itself, or a scratch tile the ops fused after the norm read."""
        prim, post = prims[0], prims[1:]
        kw = self._kwargs(prim)
        x = self._view(prim.kwargs["input"].tensor_box)
        weight = self._view(prim.kwargs["weight"].tensor_box)
        *lead, hidden, tokens = x.shape
        if hidden > PARTITIONS and hidden % PARTITIONS:
            raise NotImplementedError(f"{prim.name}: hidden size {hidden}")
        parts = min(hidden, PARTITIONS)
        blocks = hidden // parts
        ones = self.temp("one", PARTITIONS)
        column = self.temp("wc", blocks)
        if ("ones", ones) not in self.memo:
            self.memo["ones", ones] = True
            self.emit(f"nisa.memset(dst={ones}, value=1.0)")
        # The weight as float32 columns, block ``b``'s in column ``b``.
        linear = np.arange(parts)[:, None] + np.arange(blocks) * parts
        aps = fit([(self.free(weight.box), self.flat(weight, linear))], parts)
        self.emit(
            f"nisa.tensor_copy(dst={column}[0:{parts}, 0:{blocks}], "
            f"src={_ap(self.var(weight.box, weight.slot), aps[0])})"
        )
        eps = float(kw["eps"])
        free = self.config.moving_free_max
        width = SQUARE_BLOCKS * free
        for e in range(math.prod(lead)):
            starts = [e * hidden + b * parts for b in range(blocks)]
            for lo, hi in _runs(tokens, free):
                w = hi - lo
                acc = self.temp("acc", free)
                for g in range(0, blocks, SQUARE_BLOCKS):
                    group = range(g, min(blocks, g + SQUARE_BLOCKS))
                    square = self.temp("sq", width)
                    # The group's blocks side by side, partition ``p``
                    # holding row ``starts[b] + p`` of each block ``b``.
                    rows = np.array([starts[b] for b in group])
                    rows = np.arange(parts)[:, None] + rows
                    linear = rows[:, :, None] * tokens + np.arange(lo, hi)
                    linear = linear.reshape(parts, -1)
                    on_x = self.flat(x, linear, np.arange(parts)[:, None])
                    on_square = np.arange(parts)[:, None] * width
                    on_square = on_square + np.arange(len(group) * w)
                    aps = fit(
                        [(self.free(x.box), on_x), (width, on_square)], parts
                    )
                    if aps is None:
                        raise NotImplementedError("blocks no pattern walks")
                    self.emit(
                        f"nisa.activation(dst={_ap(square, aps[1])}, "
                        f"op=nl.square, "
                        f"data={_ap(self.var(x.box, x.slot), aps[0])})"
                    )
                    for n, b in enumerate(group):
                        self.emit(
                            f"nisa.nc_matmul(dst={acc}[0:{PARTITIONS}, "
                            f"0:{w}], stationary={ones}[0:{parts}, "
                            f"0:{PARTITIONS}], moving={square}[0:{parts}, "
                            f"{n * w}:{(n + 1) * w}], accumulate={b > 0})"
                        )
                rstd = self.temp("rq", free)
                rstd = f"{rstd}[0:{parts}, 0:{w}]"
                self.emit(
                    f"nisa.tensor_scalar(dst={rstd}, data={acc}[0:{parts}, "
                    f"0:{w}], op0=nl.multiply, operand0={1 / hidden!r}, "
                    f"op1=nl.add, operand1={eps!r})"
                )
                self.emit(
                    f"nisa.activation(dst={rstd}, op=nl.rsqrt, data={rstd})"
                )
                for b, start in enumerate(starts):
                    out = self._rows_ap(dst, start, parts, lo, hi)
                    if post:
                        temp = self.temp("rw", free)
                        out = f"{temp}[0:{parts}, 0:{w}]"
                    self.emit(
                        f"nisa.scalar_tensor_tensor(dst={out}, "
                        f"data={self._rows_ap(x, start, parts, lo, hi)}, "
                        f"op0=nl.multiply, operand0={column}[0:{parts}, "
                        f"{b}:{b + 1}], op1=nl.multiply, operand1={rstd})"
                    )
                    if post:
                        rows = start + np.arange(parts)
                        linear = rows[:, None] * tokens + np.arange(lo, hi)
                        coords = np.stack(np.unravel_index(linear, x.shape))
                        result = Chunk(temp, free, x.shape, coords)
                        self._chain(post, {prim.name: result}, dst)

    def _softmax(self, prim, rows, r, out):
        """Softmax over the last dim of the ``r`` rows ``rows(lo, hi)`` reads
        (columns ``lo .. hi``) into ``out``: the row maximum, negated, the
        exponentials on ScalarE summed on the way, and the scaling by the
        sum's reciprocal on VectorE, so ScalarE runs only the
        exponentials."""
        kw = self._kwargs(prim)
        shape = prim.kwargs["input"].tensor_box.output_shape
        cols = shape[-1]
        if kw["dim"] % len(shape) != len(shape) - 1:
            raise NotImplementedError("a softmax over a dim but the last")
        data = rows(0, cols)
        top = self.temp("mx", 1)
        total = self.temp("sm", 1)
        exps = self.temp("t", cols)
        self.emit(
            f"nisa.tensor_reduce(dst={top}[0:{r}, 0:1], op=nl.maximum, "
            f"data={data}, axis=1, negate=True, keepdims=True)"
        )
        self.emit(
            f"nisa.activation(dst={exps}[0:{r}, 0:{cols}], op=nl.exp, "
            f"data={data}, bias={top}[0:{r}, 0:1], reduce_op=nl.add, "
            f"reduce_res={total}[0:{r}, 0:1], "
            f"reduce_cmd=nisa.reduce_cmd.reset_reduce)"
        )
        self.emit(
            f"nisa.reciprocal(dst={total}[0:{r}, 0:1], "
            f"data={total}[0:{r}, 0:1])"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={out}, data={exps}[0:{r}, 0:{cols}], "
            f"op0=nl.multiply, operand0={total}[0:{r}, 0:1], "
            f"engine=nisa.vector_engine)"
        )

    def _reduce(self, prims, dst):
        """A max or sum over a stored tile's last dim into ``dst``, kept as
        a dim of 1: one ``tensor_reduce`` per 128 rows."""
        if len(prims) > 1:
            raise NotImplementedError(f"{prims[1].name}: fused to a reduction")
        prim = prims[0]
        kw = self._kwargs(prim)
        x = self._view(prim.kwargs["input"].tensor_box)
        last = len(x.shape) - 1
        if [d % len(x.shape) for d in kw["dim"]] != [last] or not kw["keepdim"]:
            raise NotImplementedError(f"{prim.name}: not over the last dim")
        rows, cols = _rows_cols(x.shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            self.emit(
                f"nisa.tensor_reduce(dst={self._rows_ap(dst, start, r, 0, 1)}, "
                f"op={REDUCTIONS[prim.target]}, "
                f"data={self._rows_ap(x, start, r, 0, cols)}, axis=1, "
                "keepdims=True)"
            )

    def _transposed(self, prim):
        """The stored tile ``aten.transpose`` prim ``prim`` swaps the last two
        dims of.

        Raises:
            NotImplementedError: It swaps other dims.
        """
        x = self._view(prim.kwargs["input"].tensor_box)
        ndim = len(x.shape)
        dims = sorted(
            prim.kwargs[k].scalar.int_value % ndim for k in ("dim0", "dim1")
        )
        if dims != [ndim - 2, ndim - 1]:
            raise NotImplementedError(f"{prim.name}: not the last two dims")
        return x

    def _max_pool(self, prims, dst):
        """A 2-D max pool of its input's windows (``_window_load``) into
        ``dst``: the ``kH`` rows of each window maxed into a scratch tile, then
        its ``kW`` columns, ``sW`` apart, into ``dst``."""
        if len(prims) > 1:
            raise NotImplementedError("ops fused after a pool")
        prim = prims[0]
        kwargs = self._kwargs(prim)
        if any(kwargs["padding"]) or kwargs["ceil_mode"]:
            raise NotImplementedError(f"{prim.name}: a padded pool")
        x = self._view(prim.kwargs["input"].tensor_box)
        kh, kw = kwargs["kernel_size"]
        sh, sw = kwargs["stride"]
        dh, dw = kwargs["dilation"]
        out_rows, out_h = dst.box.rows, dst.box.shape[-2]
        if self.analyze:
            self.windows[x.box.name] = (kh, sh, dh, out_rows, out_h)
            return
        cols, iw = dst.box.cols, x.box.cols
        src = self.var(x.box, x.slot)
        temp = self.temp("pv", iw)
        vertical = f"{temp}[0:{out_rows}, 0:{iw}]"

        def strip(k):
            return f"{src}[0:{out_rows}, {k * iw}:{(k + 1) * iw}]"

        def tap(k):
            return (
                f"{temp}.ap(pattern=[[{iw}, {out_rows}], [{sw}, {cols}]], "
                f"offset={k * dw})"
            )

        self.emit(
            f"nisa.tensor_tensor(dst={vertical}, data1={strip(0)}, "
            f"data2={strip(1)}, op=nl.maximum)"
        )
        for k in range(2, kh):
            self.emit(
                f"nisa.tensor_tensor(dst={vertical}, data1={vertical}, "
                f"data2={strip(k)}, op=nl.maximum)"
            )
        out = self._rows_ap(dst, 0, out_rows, 0, cols)
        self.emit(
            f"nisa.tensor_tensor(dst={out}, data1={tap(0)}, "
            f"data2={tap(1)}, op=nl.maximum)"
        )
        for k in range(2, kw):
            self.emit(
                f"nisa.tensor_tensor(dst={out}, data1={out}, "
                f"data2={tap(k)}, op=nl.maximum)"
            )

    # TensorE writing PSUM.

    def _nc_matmul(self, gemm, dst, accumulate):
        """K-major GEMM prim ``gemm`` into PSUM view ``dst``: per batch
        element, 128 rows and 512 columns of the result, one ``nc_matmul``
        per 128-deep chunk of the contraction -- of two whole chunks, a
        ``double_row`` one, when both operands are FP8: chunks as many apart
        as put each operand's pair a multiple of ``DOUBLE_ROW_STEP``
        elements apart (``pair_distance``, ``contraction_runs``) -- each
        after the first adding to the ones before, and the first too given
        ``accumulate``.  The activation is the stationary operand, or the
        moving one when the output is transposed (``_gemm_kmajor``).  A
        bias, a row on one partition, is the block's first ``nc_matmul``:
        of the bias by the IR's ones row (``temp``), contracting that one
        row."""
        act, weight = (
            self._view(gemm.kwargs[k].tensor_box)
            for k in ("input", "weight", "other")
            if k in gemm.kwargs
        )
        bias = None
        if self._kwargs(gemm).get("bias") is not None:
            bias = self._view(gemm.kwargs["bias"].tensor_box)
            ones = self.temp("one", self.config.moving_free_max)
            if ("ones", ones) not in self.memo:
                self.memo["ones", ones] = True
                self.emit(f"nisa.memset(dst={ones}, value=1.0)")
        batch, batch_a, batch_w, m, k, n = self._gemm_extents(gemm, act, weight)
        operands = [(act, batch_a, m), (weight, batch_w, n)]
        if _output_transposed(gemm):
            operands.reverse()
        (stationary, batch_s, rows), (moving, batch_m, cols) = operands
        fp8 = all(_fp8(view.box.dtype) is not None for view in (act, weight))
        distance = None
        if k > PARTITIONS and fp8:
            # Each operand's step from one 128-row chunk to the next.
            starts = [
                self.flat(view, np.array([0, PARTITIONS]) * view.shape[-1])
                for view in (act, weight)
            ]
            distance = pair_distance([int(b - a) for a, b in starts])
        runs = contraction_runs(k, distance)
        for e in range(math.prod(batch)):
            e_coords = [int(c) for c in np.unravel_index(e, batch)]
            es = _batch_index(e_coords, batch_s)
            em = _batch_index(e_coords, batch_m)
            for i, i_end in _runs(rows, PARTITIONS):
                r = i_end - i
                for j, j_end in _runs(cols, self.config.moving_free_max):
                    out = self._rows_ap(dst, e * rows + i, r, j, j_end)
                    if bias is not None:
                        # The bias spans the result's columns, or its rows
                        # when it is transposed.
                        if _output_transposed(gemm):
                            pair = (
                                self._rows_ap(bias, 0, 1, i, i + r),
                                f"{ones}[0:1, 0:{j_end - j}]",
                            )
                        else:
                            pair = (
                                f"{ones}[0:1, 0:{r}]",
                                self._rows_ap(bias, 0, 1, j, j_end),
                            )
                        self.emit(
                            f"nisa.nc_matmul(dst={out}, stationary={pair[0]}"
                            f", moving={pair[1]}, accumulate={accumulate})"
                        )
                    for c, chunks in enumerate(runs):
                        added = c > 0 or accumulate or bias is not None
                        mode = (
                            ", perf_mode=nisa.matmul_perf_mode.double_row"
                            if len(chunks) == 2
                            else ""
                        )
                        self.emit(
                            f"nisa.nc_matmul(dst={out}, stationary="
                            f"{self._block(stationary, es, chunks, i, r)}, "
                            "moving="
                            f"{self._block(moving, em, chunks, j, j_end - j)}"
                            f", accumulate={added}{mode})"
                        )

    def _nc_matmul_mx(self, gemm, dst, accumulate):
        """MX GEMM prim ``gemm`` (``linear_mx``) into PSUM view ``dst``: per
        128 rows and 512 columns of the result, one ``nc_matmul_mx`` per
        512-element chunk of the contraction, each after the first adding to
        the ones before, and the first too given ``accumulate``.  The
        operands are parts of on-chip tiles in the ``Mx`` layout -- some of
        an activation's rows, some of a weight's columns -- so nothing is
        transposed; a stationary chunk of odd width takes the tile's zeroed
        pad column with it, and writes a row of PSUM more.

        Raises:
            NotImplementedError: The GEMM has a bias or codebooks, blocks
                other than 32, or a contraction not in whole chunks.
        """
        kw = self._kwargs(gemm)
        extra = [
            k
            for k in ("bias", "input_code", "weight_code")
            if kw.get(k) is not None
        ]
        if extra or kw.get("block_size") != MX_BLOCK:
            raise NotImplementedError(f"{gemm.name}: {extra or kw}")
        act, act_scale, weight, weight_scale = (
            self._view(gemm.kwargs[key].tensor_box) for key in MX_OPERANDS
        )
        *lead, k = act.shape
        k_w, n = weight.shape
        if k != k_w or k % MX_CHUNK:
            raise NotImplementedError(f"{gemm.name}: contraction {k}")
        # The activation's first row and the weight's first column in
        # their tiles: the rows and columns the ``Mx`` layout places.
        first = np.zeros(1, dtype=int)
        x0 = int(act.positions(first)[0]) // act.box.cols
        y0 = int(weight.positions(first)[0]) % weight.box.cols
        rows_padded = self.mx_tiles[act.box.name][1]
        cols_padded = self.mx_tiles[weight.box.name][1]
        names = [
            self.var(v.box, v.slot)
            for v in (act, act_scale, weight, weight_scale)
        ]
        psum = self.var(dst.box, dst.slot)
        row0, col0 = dst.start[-2:]
        for i, i_end in _runs(math.prod(lead), PARTITIONS):
            width = i_end - i + (i_end - i) % 2
            for j, j_end in _runs(n, self.config.moving_free_max):
                w = j_end - j
                out = (
                    f"{psum}[{row0 + i}:{row0 + i + width}, "
                    f"{col0 + j}:{col0 + j_end}]"
                )
                for c in range(k // MX_CHUNK):
                    a = c * rows_padded + x0 + i
                    b = c * cols_padded + y0 + j
                    stationary = f"[0:{PARTITIONS}, {a}:{a + width}]"
                    moving = f"[0:{PARTITIONS}, {b}:{b + w}]"
                    self.emit(
                        f"nisa.nc_matmul_mx(dst={out}, "
                        f"stationary={names[0]}{stationary}, "
                        f"moving={names[2]}{moving}, "
                        f"stationary_scale={names[1]}{stationary}, "
                        f"moving_scale={names[3]}{moving}, "
                        f"accumulate={c > 0 or accumulate})"
                    )

    def _psum_transpose(self, prim, dst):
        """A stored tile ``[..., a, b]`` transposed into PSUM view ``dst``,
        ``[..., b, a]``: one ``nc_transpose`` per 128 x 128 block, by block
        row of the result."""
        x = self._transposed(prim)
        *lead, a, b = x.shape
        for e in range(math.prod(lead)):
            for j, j_end in _runs(b, PARTITIONS):
                for i, i_end in _runs(a, PARTITIONS):
                    out = self._rows_ap(dst, e * b + j, j_end - j, i, i_end)
                    block = self._rows_ap(x, e * a + i, i_end - i, j, j_end)
                    self.emit(f"nisa.nc_transpose(dst={out}, data={block})")

    def _eye(self, dst, value):
        """An identity matrix times ``value`` into ``dst``: per 128 rows, a
        ``memset`` of ``value``, then GpSimd's ``affine_select`` keeping the
        elements whose column is their row -- the block's first row plus the
        partition."""
        rows, cols = dst.shape
        for start, stop in _runs(rows, PARTITIONS):
            out = self._rows_ap(dst, start, stop - start, 0, cols)
            self.emit(f"nisa.memset(dst={out}, value={value!r})")
            self.emit(
                f"nisa.affine_select(dst={out}, pattern=[[-1, {cols}]], "
                f"channel_multiplier=1, on_true_tile={out}, "
                f"on_false_value=0.0, cmp_op=nl.equal, offset={start})"
            )

    # The fused patterns (``PATTERN_HANDLERS``): each group one instruction
    # per 128 rows, ``dst`` its results' views in order.

    def _tensor_scalar(self, prims, dst):
        """``(data <op0> column) <op1> constant``: a ``tensor_scalar`` of two
        operators per 128 rows, on VectorE, as ScalarE is the
        exponentials'."""
        first, second = prims
        (out,) = dst
        data, column = (
            self._view(first.kwargs[k].tensor_box) for k in ("input", "other")
        )
        constant = float(self._kwargs(second)["other"])
        rows, cols = _rows_cols(data.shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            self.emit(
                f"nisa.tensor_scalar(dst="
                f"{self._rows_ap(out, start, r, 0, cols)}, "
                f"data={self._rows_ap(data, start, r, 0, cols)}, "
                f"op0={BINARY_OPS[first.target]}, "
                f"operand0={self._rows_ap(column, start, r, 0, 1)}, "
                f"op1={BINARY_OPS[second.target]}, operand1={constant!r}, "
                "engine=nisa.vector_engine)"
            )

    def _tensor_scalar_reduce(self, prims, dst):
        """``data <op> constant`` and its rows reduced: one
        ``tensor_scalar_reduce`` per 128 rows, the rows reduced as
        stored."""
        first, reduction = prims[0], prims[-1]
        out, reduced = dst
        data = self._view(first.kwargs["input"].tensor_box)
        constant = float(self._kwargs(first)["other"])
        rows, cols = _rows_cols(data.shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            self.emit(
                f"nisa.tensor_scalar_reduce(dst="
                f"{self._rows_ap(out, start, r, 0, cols)}, "
                f"data={self._rows_ap(data, start, r, 0, cols)}, "
                f"op0={BINARY_OPS[first.target]}, operand0={constant!r}, "
                f"reduce_op={REDUCTIONS[reduction.target]}, "
                f"reduce_res={self._rows_ap(reduced, start, r, 0, 1)})"
            )

    def _activation_reduce(self, prims, dst):
        """``exp(data + bias)`` and its rows' sums: one ScalarE
        ``activation`` per 128 rows, summing its rows from zero."""
        add, function, _ = prims
        out, sums = dst
        data, bias = (
            self._view(add.kwargs[k].tensor_box) for k in ("input", "other")
        )
        rows, cols = _rows_cols(data.shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            self.emit(
                f"nisa.activation(dst={self._rows_ap(out, start, r, 0, cols)}"
                f", op={ACTIVATIONS[function.target]}, "
                f"data={self._rows_ap(data, start, r, 0, cols)}, "
                f"bias={self._rows_ap(bias, start, r, 0, 1)}, "
                "reduce_op=nl.add, "
                f"reduce_res={self._rows_ap(sums, start, r, 0, 1)}, "
                "reduce_cmd=nisa.reduce_cmd.reset_reduce)"
            )

    def _scaled_identity(self, prims, dst):
        """An identity times the constant its ``mul`` takes (``_eye``)."""
        _, mul = prims
        (out,) = dst
        self._eye(out, float(self._kwargs(mul)["other"]))

    def _matmul_accumulate(self, prims, dst):
        """A K-major or MX GEMM's product added to the PSUM partial sum it
        writes back: its ``nc_matmul``s or ``nc_matmul_mx``s, each
        accumulating.

        Raises:
            NotImplementedError: The partial sum is not the PSUM view the
                group writes.
        """
        gemm, add = prims
        (out,) = dst
        partial = next(
            self._view(arg.tensor_box)
            for arg in add.kwargs.values()
            if arg.WhichOneof("arg_type") == "tensor_box"
            and arg.tensor_box.box.HasField("memory")
        )
        if out.box.level != LOCAL or partial.key() != out.key():
            raise NotImplementedError(
                f"{add.name}: a sum other than the PSUM view it writes"
            )
        if gemm.target == LINEAR_MX:
            self._nc_matmul_mx(gemm, out, accumulate=True)
        else:
            self._nc_matmul(gemm, out, accumulate=True)

    # Source.

    def source(self, name):
        """The kernel as an ``@nki.jit`` function ``name``."""
        args = ", ".join(f"in{i}" for i in range(len(self.reads)))
        lines = ["@nki.jit", f"def {name}({args}):"]
        for i, box_name in enumerate(self.writes):
            box = self.boxes[box_name]
            if box_name in self.mx:
                mx = self.mx[box_name]
                shape = mx.hbm_shape()
                dtype = "nl.uint8" if mx.scales else "nl.int32"
            else:
                shape, dtype = box.shape, nl_dtype(box.dtype)
            lines.append(
                f"    out{i} = nl.ndarray({shape!r}, "
                f"dtype={dtype}, buffer=nl.shared_hbm)"
            )
        for box in self.boxes.values():
            if box.level not in ON_CHIP:
                continue
            if box.name in self.mx_tiles:
                mx, padded = self.mx_tiles[box.name]
                shape = mx.tile(padded)
                dtype = (
                    "nl.uint8" if mx.scales else NL_MX[_fp8(box.dtype).ebits]
                )
            else:
                layout = self.layouts[box.name]
                shape, dtype = (layout.parts, layout.free), nl_dtype(box.dtype)
            buffer = "nl.psum" if box.level == LOCAL else "nl.sbuf"
            for slot in range(box.slots):
                var = self.var(box, slot)
                lines.append(
                    f"    {var} = nl.ndarray({shape!r}, dtype={dtype}, "
                    f"buffer={buffer})"
                )
                if box.name in self.mx_tiles and padded != mx.other:
                    # The pad column a stationary chunk of odd width takes.
                    lines.append(f"    nisa.memset(dst={var}, value=0)")
        lines += ["    " + line for line in self.lines]
        outs = "".join(f"out{i}, " for i in range(len(self.writes)))
        lines.append(f"    return ({outs.rstrip(' ')})")
        return "\n".join(lines) + "\n"


def _batch_index(coords, batch):
    """The flat index into ``batch`` of the broadcast result batch
    ``coords``."""
    if not batch:
        return 0
    lead = len(coords) - len(batch)
    own = [coords[lead + d] if size > 1 else 0 for d, size in enumerate(batch)]
    return int(np.ravel_multi_index(own, batch))


def _is_host(op, host_scalars):
    """Whether top-level ``op`` runs in the host program: an ATen op on HBM
    tensors, a compute op writing HBM, or scalar arithmetic on a value the
    host computed."""
    if op.WhichOneof("op_type") != "prim":
        return False
    prim = op.prim
    if prim.target.startswith("aten::"):
        out = op.outputs[0] if op.outputs else None
        writes_hbm = (
            out is not None
            and out.WhichOneof("result_type") == "destination"
            and out.destination.box.memory.level == DRAM
        )
        return prim.op == "cpu" or writes_hbm
    if prim.target in SCALAR_OPS or prim.target == "sym_ite":
        return any(
            a.WhichOneof("arg_type") == "scalar"
            and a.scalar.WhichOneof("value") == "node"
            and a.scalar.node in host_scalars
            for a in prim.kwargs.values()
        )
    return False


def _hbm_allocation(op):
    """Whether top-level ``op`` allocates an HBM box."""
    if op.WhichOneof("op_type") != "prim" or op.prim.target not in ALLOCATIONS:
        return False
    return any(
        o.WhichOneof("result_type") == "tensor_box"
        and o.tensor_box.memory.level == DRAM
        for o in op.outputs
    )


def segments(model):
    """The top-level ops as ``("kernel", ops)``, ``("host", op)`` and
    ``("alloc", op)`` (an HBM allocation), in order.  A kernel starts at an
    on-chip allocation made after the one before it has done work; a host
    op or an HBM allocation runs before a kernel that has not."""
    items, current, worked, host_scalars = [], [], False, set()
    for op in model.ops:
        kind = op.WhichOneof("op_type")
        target = op.prim.target if kind == "prim" else None
        host = _is_host(op, host_scalars)
        if host or _hbm_allocation(op):
            if worked:
                items.append(("kernel", current))
                current, worked = [], False
            items.append(("host" if host else "alloc", op))
            if host:
                host_scalars.update(
                    o.name
                    for o in op.outputs
                    if o.WhichOneof("result_type") == "scalar"
                )
            continue
        if target in ALLOCATIONS and worked:
            items.append(("kernel", current))
            current, worked = [], False
        current.append(op)
        if (
            kind != "prim"
            or op.prim.op != "cpu"
            or target == "voyager::async_copy"
        ):
            worked = True
    if current:
        items.append(("kernel", current))
    return items


# What the program does to a kernel argument whose type NKI carries as
# another (``NL_DTYPES``), and the torch type it passes an 8-bit float in,
# by exponent bits: torch has no IEEE E4M3, and its float8_e4m3fn has the
# same bits up to that format's 240.
KERNEL_CASTS = {
    "int64": ".to(torch.int32)",
    "bool": ".to(torch.uint8)",
}
TORCH_FP8 = {4: "torch.float8_e4m3fn", 5: "torch.float8_e5m2"}


def _kernel_cast(dtype):
    """What the program does to a kernel argument of IR ``dtype``."""
    fmt = _fp8(dtype)
    if fmt is not None:
        return f".to({TORCH_FP8[fmt.ebits]})"
    return KERNEL_CASTS.get(dtype, "")


# Kwargs the host program drops, or gives the device its inputs are on: it
# runs where its tensors live.
HOST_DROPPED = ("layout", "pin_memory")
TORCH_DTYPES = {"bool": "torch.bool", "int64": "torch.int64"}
# The torch enums the IR names by their bare names (a dtype, ``strided``).
TORCH_ENUMS = (torch.dtype, torch.layout, torch.memory_format)


def _torch_dtype(dtype):
    return TORCH_DTYPES.get(dtype, f"torch.{dtype}")


def _host_scalar(value):
    kind = value.WhichOneof("value")
    if kind == "node":
        return f"t[{value.node!r}]"
    return repr(getattr(value, kind))


def _host_tensor(ref):
    """A reference as program source: the box, its window, its view."""
    source = f"t[{ref.box.node!r}]"
    if ref.offsets:
        source += _host_window(ref)
    shape = tuple(ref.output_shape)
    if shape != tuple(ref.sizes or ref.box.shape):
        source += f".reshape({shape})"
    return source


def _host_window(ref):
    """``[...]`` selecting a reference's window of its box."""
    slices = []
    for offset, size, step in zip(ref.offsets, ref.sizes, ref.strides):
        start = _host_scalar(offset)
        slices.append(f"{start}:{start} + {size * step}:{step}")
    return "[" + ", ".join(slices) + "]"


def _host_value(arg):
    """An ATen op's argument as program source."""
    kind = arg.WhichOneof("arg_type")
    if kind == "tensor_box":
        return _host_tensor(arg.tensor_box)
    if kind == "tensor_box_list":
        refs = arg.tensor_box_list.values
        return "[" + ", ".join(_host_tensor(r) for r in refs) + "]"
    if kind == "scalar":
        return _host_scalar(arg.scalar)
    if kind == "scalar_list":
        values = arg.scalar_list.values
        return "[" + ", ".join(_host_scalar(v) for v in values) + "]"
    value = arg.str_value
    if value == "None":
        return "None"
    if isinstance(getattr(torch, value, None), TORCH_ENUMS):
        return (
            _torch_dtype(value) if value in TORCH_DTYPES else f"torch.{value}"
        )
    return repr(value)


def _aten_call(name, kwargs):
    """Source calling ``aten::name`` on IR ``kwargs``: the overload whose
    schema takes them, by keyword (the IR's ``input`` is ATen's
    ``self``)."""
    given = {
        ("self" if k == "input" else k): v
        for k, v in kwargs.items()
        if k not in HOST_DROPPED
    }
    device = given.pop("device", None)
    packet = getattr(torch.ops.aten, name)
    for overload in packet.overloads():
        arguments = getattr(packet, overload)._schema.arguments
        names = {a.name for a in arguments}
        required = {a.name for a in arguments if not a.has_default_value()}
        if set(given) <= names and required <= set(given):
            args = ", ".join(f"{k}={_host_value(v)}" for k, v in given.items())
            if device is not None and "device" in names:
                args += ", device=device"
            return f"torch.ops.aten.{name}.{overload}({args})"
    raise NotImplementedError(f"aten::{name} on {sorted(given)}")


def host_statement(op):
    """``(reads, writes, line)`` running top-level host ``op``; ``writes``
    names an HBM box an in-place op rebinds as well as the op's result."""
    prim = op.prim
    # By name, so the program reads the same from run to run.
    kwargs = dict(sorted(prim.kwargs.items()))
    out = op.outputs[0]
    reads = []
    for arg in kwargs.values():
        kind = arg.WhichOneof("arg_type")
        if kind == "tensor_box":
            reads.append(arg.tensor_box.box.node)
        elif kind == "tensor_box_list":
            reads += [ref.box.node for ref in arg.tensor_box_list.values]
    if prim.target in SCALAR_OPS or prim.target == "sym_ite":
        values = {k: _host_scalar(v.scalar) for k, v in kwargs.items()}
        if prim.target == "sym_ite":
            value = f"{values['t']} if {values['b']} else {values['f']}"
        else:
            value = (
                f"operator.{prim.target}({values['input']}, {values['other']})"
            )
        return reads, [out.name], f"t[{out.name!r}] = {value}"
    name = prim.target.removeprefix("aten::")
    if name == "_local_scalar_dense":
        tensor = _host_tensor(kwargs["input"].tensor_box)
        value = f"{tensor}.item() if scalars is None else scalars[{out.name!r}]"
        return reads, [out.name], f"t[{out.name!r}] = {value}"
    if name == "copy_":
        # A whole overwrite: the source in the destination's dtype and
        # shape, which the IR gives (``aten.copy`` would mark a traced
        # parameter written without reading it, and an immediate
        # destination is no tensor of the program's).
        box = kwargs["input"].tensor_box.box
        src = _host_tensor(kwargs["src"].tensor_box)
        value = (
            f"{src}.to({_torch_dtype(box.dtype)})"
            f".expand({tuple(box.shape)}).clone()"
        )
        line = f"t[{out.name!r}] = t[{box.node!r}] = {value}"
        return reads, [out.name, box.node], line
    if name.endswith("_"):
        # In place: its functional twin, rebinding what it mutates.
        dst = kwargs["input"].tensor_box.box.node
        value = _aten_call(name[:-1], kwargs)
        return reads, [out.name, dst], f"t[{out.name!r}] = t[{dst!r}] = {value}"
    value = _aten_call(name, kwargs)
    if out.WhichOneof("result_type") == "destination":
        refs = [o.destination for o in op.outputs]
        lines = [f"t[{op.name!r}] = {value}"] + [
            f"t[{ref.box.node!r}]{_host_window(ref)} = t[{op.name!r}]"
            for ref in refs
        ]
        return reads, [ref.box.node for ref in refs], "\n    ".join(lines)
    return reads, [out.name], f"t[{out.name!r}] = {value}"


def host_allocation(op):
    """``(reads, writes, line)`` allocating top-level HBM ``op``'s box."""
    box = next(
        o.tensor_box
        for o in op.outputs
        if o.WhichOneof("result_type") == "tensor_box"
    )
    line = (
        f"t[{box.node!r}] = torch.zeros({tuple(box.shape)}, "
        f"dtype={_torch_dtype(box.dtype)}, device=device)"
    )
    return [], [box.node], line


PROGRAM_HEADER = '''\
"""Generated by voyager_compiler.backend.trainium.translate: do not edit."""

import operator

import torch


def run(
    inputs,
    params,
    call_kernel,
    mark=lambda kind, *values: values,
    scalars=None,
):
    """Run the model.

    Args:
        inputs: The model's inputs, by name.
        params: Its parameters, by name.
        call_kernel: ``(name, *tensors) -> results`` of a kernel.
        mark: ``(kind, *tensors) -> tensors``, called between the program's
            segments -- the ops before the decoder layers, each layer's and
            those after: ``"end"`` with what a segment passes on, ``"start"``
            with what the next takes from it.  The identity unless a trace
            marks the layers for the compiler.
        scalars: The value of each tensor read as a host scalar, by name, to
            take instead of reading it (a trace cannot read one); ``None``
            reads them.

    Returns:
        The model's outputs; the parameters it writes (its state, a KV
        cache) are stored back into ``params``.
    """
    t = dict(inputs, **params)
    device = next(iter(params.values())).device
'''

# A decoder layer's index in the name of a box it writes.
DECODER_LAYER = re.compile(r"(?:^|_)layers_(\d+)_")

KERNELS_HEADER = '''\
"""Generated by voyager_compiler.backend.trainium.translate: do not edit."""

import nki
import nki.isa as nisa
import nki.language as nl
'''


# The names the IR gives each layer's boxes and ops, and the addresses its
# memory plan picks: two kernels differing only in these are one kernel.
_NAMES = re.compile(r'\b(node|name|iv): "([^"]*)"')
_ADDRESSES = re.compile(r"\b(address|bank_stride_bytes): \d+")


def _structure(ops, immediates, mx):
    """A key two kernels share when they differ only in names and
    addresses, and the kernel's names in order of first use."""
    names = {}

    def canonical(match):
        index = names.setdefault(match.group(2), len(names))
        return f'{match.group(1)}: "{index}"'

    text = "".join(text_format.MessageToString(op) for op in ops)
    text = _NAMES.sub(canonical, _ADDRESSES.sub("", text))
    values = [immediates[n] for n in names if n in immediates]
    values += [mx[n] for n in names if n in mx]
    key = hashlib.sha1((text + repr(values)).encode()).hexdigest()
    return key, list(names)


def translate(model, out_dir, config, immediates):
    """Translate ``model`` into ``out_dir``.

    Args:
        model: A ``voyager`` IR ``Model``, or the path of its ``model.txt``.
        out_dir: Where the translation is written.
        config: The ``TrainiumConfig`` Voyager compiled ``model`` for.
        immediates: The value of each immediate box (a constant the IR
            names but does not hold), by name.

    Returns:
        The manifest written to ``manifest.json``.
    """
    if isinstance(model, str):
        with open(model) as f:
            model = text_format.Parse(f.read(), ir.Model())
    items = segments(model)
    os.makedirs(out_dir, exist_ok=True)
    mx, mx_scales = mx_arrays(model)
    sources, steps, lines, translated = {}, [], [], {}
    allocations = []
    for kind, item in items:
        if kind == "alloc":
            allocations.append((len(steps), item))
            continue
        if kind == "host":
            reads, writes, line = host_statement(item)
            if set(reads + writes) & set(mx):
                raise NotImplementedError(f"{item.name}: an MX array on host")
            step = {"kernel": None, "inputs": reads, "outputs": writes}
            step["ops"] = [item.name, item.name]
        else:
            key, names = _structure(item, immediates, mx)
            if key not in translated:
                kernel = Kernel(item, immediates, mx, config)
                body = kernel.source("kernel")
                digest = hashlib.sha1(body.encode()).hexdigest()[:12]
                name = "kernel_" + digest
                sources.setdefault(
                    name, body.replace("def kernel(", f"def {name}(", 1)
                )
                # An MX array goes as stored, already packed.
                casts = [
                    "" if b in mx else _kernel_cast(kernel.boxes[b].dtype)
                    for b in kernel.reads
                ]
                # Each result the kernel writes only part of -> the runs
                # ``[start, stop)`` of row-major positions it writes.
                partial = {}
                for i, b in enumerate(kernel.writes):
                    stored = np.concatenate(([0], kernel.stored[b], [0]))
                    if not stored[1:-1].all():
                        edges = np.flatnonzero(np.diff(stored))
                        partial[i] = edges.reshape(-1, 2).tolist()
                translated[key] = (
                    name,
                    casts,
                    [
                        [names.index(b) for b in boxes]
                        for boxes in (kernel.reads, kernel.writes)
                    ],
                    partial,
                )
            name, casts, indices, partial = translated[key]
            reads, writes = ([names[i] for i in group] for group in indices)
            step = {"kernel": name, "inputs": reads, "outputs": writes}
            step["partial"] = {writes[i]: runs for i, runs in partial.items()}
            step["ops"] = [item[0].name, item[-1].name]
            args = ", ".join(f"t[{b!r}]{c}" for b, c in zip(reads, casts))
            outs = ", ".join(f"t[{b!r}]" for b in writes)
            line = f"{outs}, = call_kernel({name!r}, {args})"
        steps.append(step)
        lines.append(line)
    # An HBM box no kernel writes is allocated by the program, in place.
    by_kernel = {b for s in steps if s["kernel"] for b in s["outputs"]}
    for index, op in reversed(allocations):
        reads, writes, line = host_allocation(op)
        if writes[0] in by_kernel:
            continue
        step = {"kernel": None, "inputs": reads, "outputs": writes}
        step["ops"] = [op.name, op.name]
        steps.insert(index, step)
        lines.insert(index, line)
    outputs = [b.node for b in model.outputs]
    with open(os.path.join(out_dir, "kernels.py"), "w") as f:
        f.write(KERNELS_HEADER + "\n\n" + "\n\n".join(sources.values()))
    statements = [
        (step["inputs"], step["outputs"], line)
        for step, line in zip(steps, lines)
    ]
    params = [b.node for b in model.parameters]
    written = {b for s in steps for b in s["outputs"]}
    state = [b for b in params if b in written]
    dtypes = _hbm_dtypes(model)
    indices = {b for b, dtype in dtypes.items() if dtype == "int64"}
    with open(os.path.join(out_dir, "program.py"), "w") as f:
        f.write(program_source(statements, outputs, state, indices))
    packed = {
        codes: {
            "scales": scales,
            "format": dtypes[codes],
            "depth": mx[codes].depth,
            "other": mx[codes].other,
            "k_major": mx[codes].k_major,
        }
        for codes, scales in mx_scales.items()
    }
    for name, array in mx.items():
        dtypes[name] = "uint8" if array.scales else "int32"
    manifest = {
        "inputs": [b.node for b in model.inputs],
        "params": params,
        "state": state,
        "scalars": [
            op.name
            for kind, op in items
            if kind == "host" and op.prim.target == "aten::_local_scalar_dense"
        ],
        "outputs": outputs,
        "immediates": immediates,
        "dtypes": dtypes,
        # Each MX codes array's scales and layout (``Mx``); both are stored
        # packed, the codes as int32 and the scales as E8M0 bytes.
        "mx": packed,
        "steps": steps,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def program_source(statements, outputs, state, indices):
    """Source of ``program.py``: ``statements`` -- ``(reads, writes,
    line)`` each -- in order, ``mark`` calls between the segments a decoder
    layer's statements make, the ``state`` parameters it writes stored back
    into ``params``, and ``outputs`` returned.

    A statement belongs to the layer named in a box it writes; one that
    writes none joins the layer before it, or, after the last layer's
    statements, the segment after the layers.  A box of ``indices`` -- an
    int64 tensor, such as the position a KV cache write takes -- crosses a
    segment unmarked: a marker on one fails the trace.
    """
    layers = []
    for _, writes, _ in statements:
        found = [DECODER_LAYER.search(w) for w in writes]
        layers.append(next((int(m.group(1)) for m in found if m), None))
    last = max((i for i, x in enumerate(layers) if x is not None), default=-1)
    segment, current, layer = [], 0, None
    for i, x in enumerate(layers):
        if x is not None and x != layer:
            current, layer = current + 1, x
        elif x is None and layer is not None and i > last:
            current, layer = current + 1, None
        segment.append(current)
    read_in = {}
    for (reads, _, _), s in zip(statements, segment):
        for name in reads:
            read_in.setdefault(name, set()).add(s)
    for name in outputs:
        read_in.setdefault(name, set()).add(current + 1)

    def mark(kind, names):
        values = ", ".join(f"t[{n!r}]" for n in names)
        return f"    {values}, = mark({kind!r}, {values})"

    lines = [PROGRAM_HEADER.rstrip("\n")]
    for i, (_, _, line) in enumerate(statements):
        if i > 0 and segment[i] != segment[i - 1]:
            done = segment[i - 1]
            written = [
                w
                for (_, writes, _), s in zip(statements, segment)
                if s == done
                for w in writes
                if w not in indices
            ]
            passed = [w for w in written if max(read_in.get(w, {0})) > done]
            taken = [w for w in written if done + 1 in read_in.get(w, ())]
            if passed:
                lines.append(mark("end", passed))
            if taken:
                lines.append(mark("start", taken))
        lines.append("    " + line)
    lines += [f"    params[{n!r}] = t[{n!r}]" for n in state]
    lines.append(f"    return ({''.join(f't[{n!r}], ' for n in outputs)})")
    return "\n".join(lines) + "\n"


def _boxes(message, refs, found):
    """Collect the first ``TensorBox`` of each name under ``message``: the
    declarations (model inputs, parameters, allocations), and with ``refs``
    the boxes references name too -- an immediate's only mention.  A
    reference through a view gives the view's dtype, not the box's."""
    if isinstance(message, ir.TensorBoxRef) and not refs:
        return found
    if isinstance(message, ir.TensorBox) and message.HasField("memory"):
        found.setdefault(message.node, message)
    for descriptor, value in message.ListFields():
        if descriptor.type != descriptor.TYPE_MESSAGE:
            continue
        if descriptor.message_type.GetOptions().map_entry:
            value = value.values()
        elif not descriptor.is_repeated:
            value = [value]
        for item in value:
            _boxes(item, refs, found)
    return found


def _hbm_dtypes(model):
    """The element type each HBM box of ``model`` is declared with."""
    return {
        name: box.dtype
        for name, box in _boxes(model, False, {}).items()
        if box.memory.level == DRAM
    }


def immediates(gm, model):
    """The value of each immediate box of ``model`` (a constant the IR names
    but does not hold), read off ``gm``'s attributes."""
    boxes = _boxes(model, True, {})
    return {
        n.name: fetch_attr(gm, n.target).item()
        for n in gm.graph.nodes
        if n.op == "get_attr"
        and n.name in boxes
        and boxes[n.name].memory.level == IMMEDIATE
    }


# Torch's type of each MX element format: the bytes a codes array packs.
TORCH_MX = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


def _pack_mx(value, mx, element):
    """An MX array's value as ``Mx`` stores it in HBM: codes as int32 words
    of 4 consecutive contraction elements' ``element`` bytes, the first in
    the low byte; scales as E8M0 bytes, ``2 ** (byte - 127)``, a scale
    below the smallest (an all-zero block's) as byte 0."""
    rows = mx.depth // MX_BLOCK if mx.scales else mx.depth
    if mx.k_major:
        flat = value.reshape(rows, mx.other)
    else:
        flat = value.reshape(mx.other, rows).T
    chunks = mx.depth // MX_CHUNK
    if mx.scales:
        exponent = torch.log2(flat.float()).round()
        exponent = torch.where(exponent.isfinite(), exponent, -127.0)
        byte = (exponent + 127).clamp(0, 254).to(torch.uint8)
        byte = byte.reshape(chunks, CHUNK_SCALES, mx.other)
        return byte.permute(1, 0, 2).contiguous()
    byte = flat.to(element).view(torch.uint8)
    byte = byte.reshape(chunks, PARTITIONS, MX_PACK, mx.other)
    words = byte.permute(1, 0, 3, 2).contiguous().view(torch.int32)
    return words.reshape(PARTITIONS, chunks, mx.other)


def _save(values, manifest, path):
    """Save ``values`` to ``path``, each MX array the manifest names packed
    as its kernels take it."""
    for codes, entry in manifest["mx"].items():
        for name, scales in ((codes, False), (entry["scales"], True)):
            if name in values:
                mx = Mx(
                    scales, entry["depth"], entry["other"], entry["k_major"]
                )
                element = TORCH_MX[entry["format"]]
                values[name] = _pack_mx(values[name], mx, element)
    torch.save(values, path)


def record(gm, args, manifest, path, steps=True):
    """Save to ``path`` the value of every HBM box ``manifest`` names, as
    running the bufferized ``gm`` on the flat ``args`` computes it: the
    inputs and parameters as they come in, a box a step reads as the first
    such step starts -- finished, and not yet reused by the memory plan --
    and the rest as the run leaves them, an MX array packed as its kernels
    take it.  These are the references a translation is checked against;
    without ``steps`` only the inputs and parameters are kept, what a trace
    needs, and nothing is run."""
    given = set(manifest["inputs"]) | set(manifest["params"])
    if not steps:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        values = {n.name: a for n, a in zip(placeholders, args)}
        values.update(
            (n.name, fetch_attr(gm, n.target))
            for n in gm.graph.nodes
            if n.op == "get_attr"
        )
        _save({k: v for k, v in values.items() if k in given}, manifest, path)
        return
    scalars = set(manifest["scalars"])
    before = {}
    for step in manifest["steps"]:
        before.setdefault(step["ops"][0], []).extend(step["inputs"])
    nodes = {n.name: n for n in gm.graph.nodes}
    values = {}

    def keep(name, value):
        if name not in values and isinstance(value, torch.Tensor):
            values[name] = value.detach().clone()

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            for name in before.get(node.name, ()):
                if nodes.get(name) in self.env:
                    keep(name, self.env[nodes[name]])
            value = super().run_node(node)
            if node.name in given:
                keep(node.name, value)
            if node.name in scalars:
                keep(node.name, torch.tensor(value))
            return value

    recorder = Recorder(gm, garbage_collect_values=False)
    recorder.run(*args)
    # What no step reads -- the model's outputs -- as the run leaves it.
    for step in manifest["steps"]:
        for name in step["outputs"]:
            if nodes.get(name) in recorder.env:
                keep(name, recorder.env[nodes[name]])
    _save(values, manifest, path)
