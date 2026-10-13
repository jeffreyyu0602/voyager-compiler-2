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
of ``C`` columns is an NKI tensor ``[P, B * C]``.  A one-row tile read as an
elementwise operand is replicated across the partitions when it is loaded.
Every access is an ``.ap()`` access pattern, found by mapping each element's
logical coordinates to its offset and factoring the offsets into one nesting
per instruction (``fit``), so views and relayouts fused into a kernel need
no case of their own.

A GEMM step transposes its activation on TensorE one 128 x 128 block at a
time, multiplies it into PSUM one 512-column slice at a time and evicts each
slice through the ops fused after the GEMM; a vector op runs on 128-row
chunks; a transposing load lands on TensorE as well.  This is the step
``TrainiumModel`` prices.

An MX (microscaling) array -- the codes or the block scales a ``linear_mx``
multiplies -- is stored contraction-major, the way Trn3's ``nc_matmul_mx``
reads it (``Mx``).  A ``linear_mx`` step runs one ``nc_matmul_mx`` per
512-element chunk of its contraction, and an op ending in ``quantize_mx``
transposes its result onto the partitions 4 columns apart and quantizes it
there with ``quantize_mx``.

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
from voyager_compiler.quantization.qspec import float_format
from voyager_compiler.shape_prop import fetch_attr

PARTITIONS = 128
# Columns of one matmul's moving operand: a 2 KiB PSUM bank of float32.
MOVING_FREE = 512
# Elements per partition one ``bn_stats`` takes.
STATS_FREE = 512
# Columns of one vector instruction's chunk.
CHUNK_FREE = 2048
# Rotating PSUM accumulators and transposition tiles.
ACCUMULATORS = 4
TRANSPOSES = 4

IMMEDIATE = ir.MEMORY_LEVEL_IMMEDIATE
SCRATCHPAD = ir.MEMORY_LEVEL_SCRATCHPAD
DRAM = ir.MEMORY_LEVEL_DRAM

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
}
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
GEMMS = ("quantized_ops::linear", "aten::matmul")
LINEAR_MX = "quantized_ops::linear_mx"
QUANTIZE_MX = "quantized_ops::quantize_mx"

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
    """A box's SBUF slot: ``parts`` partitions by ``blocks`` row blocks of
    ``cols`` columns, row ``r`` on partition ``r % span`` of block ``r //
    span``; a ``replicated`` one-row box on every partition.  A box with
    ``span`` 1 is all on one partition."""

    parts: int
    blocks: int
    cols: int
    replicated: bool
    span: int = PARTITIONS

    @property
    def free(self):
        return self.blocks * self.cols


@dataclass
class View:
    """A reference resolved: the box and slot it reads, the window of the
    slot it covers (from ``start``, ``step`` apart) and the shape it is seen
    as."""

    box: Box
    slot: int
    start: tuple
    step: tuple
    window: tuple
    shape: tuple

    def covers_box(self):
        """Whether the view is its whole slot, in order."""
        return (
            self.window == self.box.shape
            and not any(self.start)
            and all(t == 1 for t in self.step)
        )

    def box_coords(self, linear):
        """Slot coordinates of the elements at row-major positions
        ``linear`` of ``shape``."""
        window = np.unravel_index(linear, self.window)
        return [s + w * t for s, w, t in zip(self.start, window, self.step)]


@dataclass
class Chunk:
    """A result on one ``[r, w]`` chunk: the tensor holding it and its
    elements per partition, its logical shape and each element's
    coordinates in it (``[ndim, r, w]``); ``flat`` gives the offsets and
    ``dtype`` the element type when it is written into a tile -- box
    ``box`` -- rather than a float32 temporary."""

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


def _runs(n, size):
    """``(start, stop)`` of each run of at most ``size`` of ``n``."""
    return [(s, min(n, s + size)) for s in range(0, n, size)]


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

    Attributes:
        immediates: The value of each immediate box, by name.
        mx: The ``Mx`` of each HBM box holding an MX array, by name.
        boxes: Every box the kernel touches, by name.
        reads: The HBM boxes it reads, its arguments, in order.
        writes: The HBM boxes it writes, its results, in order.
        lines: The body, statement by statement.
    """

    def __init__(self, ops, immediates, mx):
        self.immediates = immediates
        self.mx = mx
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
        self.uses = {}
        self.lines = []
        self.pools = {}
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
        """Every SBUF box's layout: rows on the partitions, a one-row box
        only ever read as an elementwise operand replicated."""
        for box in self.boxes.values():
            if box.level != SCRATCHPAD or box.name in self.mx_tiles:
                continue
            if box.name in self.one_partition:
                layout = Layout(1, box.rows, box.cols, False, span=1)
                self.layouts[box.name] = layout
                continue
            uses = self.uses.get(box.name, set())
            replicated = (
                box.rows == 1 and "operand" in uses and "dst" not in uses
            )
            parts = PARTITIONS if replicated else min(box.rows, PARTITIONS)
            blocks = math.ceil(box.rows / PARTITIONS)
            self.layouts[box.name] = Layout(parts, blocks, box.cols, replicated)

    def _view(self, ref):
        """``View`` of a ``TensorBoxRef``."""
        box = self._box(ref.box)
        offsets = [self._scalar(o) for o in ref.offsets]
        sizes, steps = list(ref.sizes), list(ref.strides)
        slot = 0
        if box.pipelined and offsets:
            slot = offsets.pop(0)
            sizes, steps = sizes[1:], steps[1:]
        if not offsets:
            offsets, sizes = [0] * len(box.shape), list(box.shape)
            steps = [1] * len(box.shape)
        shape = tuple(ref.output_shape) or tuple(sizes) or (1,)
        return View(
            box, slot, tuple(offsets), tuple(steps), tuple(sizes) or (1,), shape
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

    def _mx_tile(self, view, scales, k_major):
        """Record ``view``'s box as an on-chip MX tile, contracting along
        the view's last dim, or its first if ``k_major``."""
        *lead, last = view.shape
        if k_major:
            depth, other = view.shape[0], math.prod(view.shape[1:])
        else:
            depth, other = last, math.prod(lead)
        mx = Mx(scales, depth * MX_BLOCK if scales else depth, other, k_major)
        padded = other if k_major else other + other % 2
        if self.mx_tiles.setdefault(view.box.name, (mx, padded)) != (
            mx,
            padded,
        ):
            raise NotImplementedError(f"{view.box.name}: two MX views")

    def flat(self, view, linear, partitions=None):
        """Flat offsets in its tensor of ``view``'s elements at positions
        ``linear``; a replicated box is read on ``partitions``."""
        box = view.box
        if view.covers_box():
            # The view is the box reshaped: position is the box's own.
            if box.level == DRAM:
                return linear
            row, col = np.divmod(linear, box.cols)
        else:
            coords = view.box_coords(linear)
            if box.level == DRAM:
                return np.ravel_multi_index(coords, box.shape)
            col = coords[-1]
            row = np.ravel_multi_index(coords[:-1], box.shape[:-1])
        layout = self.layouts[box.name]
        if layout.replicated:
            return partitions * layout.free + col
        part, block = row % layout.span, row // layout.span
        return part * layout.free + block * layout.cols + col

    def emit(self, line):
        self.lines.append(line)

    def pool(self, kind, shape, dtype, count):
        """The next of ``count`` rotating temporaries of ``kind``."""
        key = (kind, shape, dtype)
        if key not in self.pools:
            names = [f"{kind}{len(self.pools)}_{i}" for i in range(count)]
            self.pools[key] = [names, 0]
        entry = self.pools[key]
        entry[1] += 1
        return entry[0][(entry[1] - 1) % count]

    # Transfers.

    def _copy(self, kw):
        """``voyager::async_copy``: the block of the HBM view at ``indices``
        to or from an SBUF slot."""
        src, dst = self._view(kw["src"]), self._view(kw["dst"])
        if kw.get("pad") or kw.get("count"):
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
        block = np.indices(sizes).reshape(len(sizes), -1)
        block += np.array(offsets)[:, None]
        hbm_linear = np.ravel_multi_index(block, hbm.shape)
        if hbm.box.name in self.mx:
            if kw.get("transposed"):
                raise NotImplementedError("a transposing MX transfer")
            self._mx_transfer(hbm, hbm_linear, sbuf, load)
        elif kw.get("transposed"):
            self._transposed_load(hbm, hbm_linear, sizes, sbuf)
        else:
            self._transfer(hbm, hbm_linear, sbuf, load)

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
        ``dma_copy`` when a pattern walks it, else 128 rows at a time."""
        layout = self.layouts[sbuf.box.name]
        rows, cols = _rows_cols(sbuf.shape)
        sbuf_var, hbm_var = self.var(sbuf.box, sbuf.slot), self.var(hbm.box)

        def offsets(linear):
            r = linear.shape[0]
            s = self.flat(sbuf, linear, np.arange(r)[:, None])
            return s, self.flat(hbm, hbm_linear[linear])

        if layout.replicated:
            parts = PARTITIONS if load else 1
            linear = np.broadcast_to(np.arange(cols), (parts, cols))
            s, h = offsets(linear)
            self._dma(sbuf_var, layout.free, s, hbm_var, h, load)
            return
        parts, blocks = layout.parts, layout.blocks
        if parts * blocks == rows:
            # Partition ``p`` holds row ``b * span + p`` of each block ``b``.
            row = np.arange(parts)[:, None] + np.arange(blocks) * layout.span
            linear = row[:, :, None] * cols + np.arange(cols)
            s, h = offsets(linear.reshape(parts, -1))
            if self._try_dma(sbuf_var, layout.free, s, hbm_var, h, load):
                return
        for a, b in _runs(rows, PARTITIONS):
            linear = np.arange(a * cols, b * cols).reshape(b - a, cols)
            s, h = offsets(linear)
            self._dma(sbuf_var, layout.free, s, hbm_var, h, load)

    def _try_dma(self, sbuf_var, free, s, hbm_var, h, load, dtype=None):
        """One ``dma_copy`` of offsets ``s`` (SBUF) and ``h`` (HBM) if a
        pattern walks both, the HBM tensor seen as ``dtype`` when given;
        whether it did."""
        aps = fit([(free, s), (None, h)], s.shape[0])
        if aps is None:
            return False
        on_chip, hbm = _ap(sbuf_var, aps[0]), _ap(hbm_var, aps[1], dtype)
        dst, src = (on_chip, hbm) if load else (hbm, on_chip)
        self.emit(f"nisa.dma_copy(dst={dst}, src={src})")
        return True

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

    def _transposed_load(self, hbm, hbm_linear, sizes, sbuf):
        """Load an HBM block transposed into ``sbuf``: as stored into a
        temporary, then each 128 x 128 block transposed on TensorE."""
        *lead, a, b = sizes
        dtype = sbuf.box.dtype
        rows = math.prod(lead) * a
        temp_free = math.ceil(rows / PARTITIONS) * b
        temp = self.pool("tl", (PARTITIONS, temp_free), dtype, 2)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            linear = np.arange(start * b, stop * b).reshape(r, b)
            t = np.arange(r)[:, None] * temp_free + np.arange(b)[None, :]
            t += (start // PARTITIONS) * b
            h = self.flat(hbm, hbm_linear[linear])
            self._dma(temp, temp_free, t, self.var(hbm.box), h, True)
        sbuf_var = self.var(sbuf.box, sbuf.slot)
        free = self.free(sbuf.box)
        for e in range(math.prod(lead)):
            for i, i_end in _runs(a, PARTITIONS):
                for j, j_end in _runs(b, PARTITIONS):
                    r, c = i_end - i, j_end - j
                    row = e * a + i
                    if row % PARTITIONS + r > PARTITIONS:
                        raise NotImplementedError("a block across partitions")
                    offset = (row % PARTITIONS) * temp_free
                    offset += (row // PARTITIONS) * b + j
                    block = (
                        f"{temp}.ap(pattern=[[{temp_free}, {r}], [1, {c}]], "
                        f"offset={offset})"
                    )
                    transposed = self._transpose(block, r, c, dtype)
                    # Transposed element (y, x) is block element (x, y).
                    y, x = np.arange(c)[:, None], np.arange(r)[None, :]
                    linear = (e * b + j + y) * a + i + x
                    aps = fit([(free, self.flat(sbuf, linear))], c)
                    if aps is None:
                        raise NotImplementedError("transposed tile layout")
                    self.emit(
                        f"nisa.tensor_copy(dst={_ap(sbuf_var, aps[0])}, "
                        f"src={transposed})"
                    )

    # Compute.

    def _compute(self, op):
        if op.WhichOneof("op_type") == "prim":
            prims = [op.prim]
        else:
            prims = list(op.fused.op_list)
        anchor = prims[0]
        if prims[-1].target == QUANTIZE_MX:
            # Its results are the scales, then the codes.
            dst = tuple(self._view(o.destination) for o in op.outputs[:2])
            for view, scales in zip(dst, (True, False)):
                self._use(view.box, "dst")
                self._mx_tile(view, scales, False)
        else:
            dst = self._view(op.outputs[0].destination)
            self._use(dst.box, "dst")
        gemm = anchor.target in GEMMS or anchor.target == LINEAR_MX
        gemm_operands = []
        if anchor.target == LINEAR_MX:
            for key, (scales, k_major) in MX_OPERANDS.items():
                view = self._view(anchor.kwargs[key].tensor_box)
                self._mx_tile(view, scales, k_major)
                gemm_operands.append(view)
        elif gemm:
            gemm_operands = [
                self._view(anchor.kwargs[k].tensor_box)
                for k in ("input", "weight", "other")
                if k in anchor.kwargs
            ]
        for view in gemm_operands:
            self._use(view.box, "gemm")
        if self.analyze:
            for prim in prims[1:] if gemm else prims:
                for arg in prim.kwargs.values():
                    if arg.WhichOneof("arg_type") != "tensor_box":
                        continue
                    if not arg.tensor_box.box.HasField("memory"):
                        continue
                    box = self._view(arg.tensor_box).box
                    self._use(box, "operand")
                    if box.level == DRAM and box.name not in self.reads:
                        self.reads.append(box.name)
            return
        if anchor.target == LINEAR_MX:
            self._gemm_mx(prims, *gemm_operands, dst)
        elif gemm:
            self._gemm(prims, *gemm_operands, dst)
        elif anchor.target == "quantized_ops::layer_norm":
            self._rowwise(prims, dst, self._layer_norm)
        elif anchor.target == "quantized_ops::softmax":
            self._rowwise(prims, dst, self._softmax)
        else:
            self._elementwise(prims, dst)

    def _gemm(self, prims, act, weight, dst):
        """A GEMM step and the ops fused after it: per batch element and
        128-row chunk, the activation transposed block by block, then per
        512-column slice its matmuls into PSUM and the eviction."""
        *batch_a, m, k = act.shape
        *batch_w, k_w, n = weight.shape
        if k != k_w:
            raise NotImplementedError("a GEMM weight not [K, N]")
        batch = tuple(np.broadcast_shapes(tuple(batch_a), tuple(batch_w)))
        out_shape = batch + (m, n)
        dtype = act.box.dtype
        depth = _runs(k, PARTITIONS)
        act_var = self.var(act.box, act.slot)
        weight_var = self.var(weight.box, weight.slot)
        for e in range(math.prod(batch)):
            e_coords = [int(c) for c in np.unravel_index(e, batch)]
            ea = _batch_index(e_coords, batch_a)
            ew = _batch_index(e_coords, batch_w)
            for i, i_end in _runs(m, PARTITIONS):
                r = i_end - i
                act_t = self.pool(
                    "at", (PARTITIONS, len(depth) * PARTITIONS), dtype, 2
                )
                for c, (lo, hi) in enumerate(depth):
                    rows = (ea * m + i + np.arange(r))[:, None]
                    linear = rows * k + lo + np.arange(hi - lo)[None, :]
                    aps = fit([(self.free(act.box), self.flat(act, linear))], r)
                    transposed = self._transpose(
                        _ap(act_var, aps[0]), r, hi - lo, dtype
                    )
                    at = c * PARTITIONS
                    self.emit(
                        f"nisa.tensor_copy(dst={act_t}[0:{hi - lo}, "
                        f"{at}:{at + r}], src={transposed})"
                    )
                for j, j_end in _runs(n, MOVING_FREE):
                    w = j_end - j
                    acc = self.pool(
                        "acc",
                        (PARTITIONS, MOVING_FREE),
                        "float32",
                        ACCUMULATORS,
                    )
                    for c, (lo, hi) in enumerate(depth):
                        rows = (ew * k + lo + np.arange(hi - lo))[:, None]
                        linear = rows * n + j + np.arange(w)[None, :]
                        b = self.flat(weight, linear)
                        aps = fit([(self.free(weight.box), b)], hi - lo)
                        at = c * PARTITIONS
                        self.emit(
                            f"nisa.nc_matmul(dst={acc}[0:{r}, 0:{w}], "
                            f"stationary={act_t}[0:{hi - lo}, {at}:{at + r}], "
                            f"moving={_ap(weight_var, aps[0])}, "
                            f"accumulate={c > 0})"
                        )
                    grid = np.stack(
                        np.broadcast_arrays(
                            *[np.full((r, w), x) for x in e_coords],
                            (i + np.arange(r))[:, None],
                            (j + np.arange(w))[None, :],
                        )
                    )
                    chunk = Chunk(acc, MOVING_FREE, out_shape, grid)
                    self._chain(prims[1:], {prims[0].name: chunk}, dst)

    def _gemm_mx(self, prims, act, act_scale, weight, weight_scale, dst):
        """An MX GEMM step and the ops fused after it: per 128-row chunk and
        512-column slice, one ``nc_matmul_mx`` per 512-element chunk of the
        contraction into PSUM, then the eviction.  The operands are on chip
        in the ``Mx`` layout, so nothing is transposed; a stationary chunk of
        odd width takes the tile's zeroed pad column with it."""
        anchor = prims[0]
        kw = self._kwargs(anchor)
        extra = [
            k
            for k in ("bias", "input_code", "weight_code")
            if kw.get(k) is not None
        ]
        if extra or kw.get("block_size") != MX_BLOCK:
            raise NotImplementedError(f"{anchor.name}: {extra or kw}")
        *lead, k = act.shape
        k_w, n = weight.shape
        if k != k_w or k % MX_CHUNK:
            raise NotImplementedError(f"{anchor.name}: contraction {k}")
        m = math.prod(lead)
        rows_padded = self.mx_tiles[act.box.name][1]
        cols_padded = self.mx_tiles[weight.box.name][1]
        names = [
            self.var(v.box, v.slot)
            for v in (act, act_scale, weight, weight_scale)
        ]
        out_shape = (*lead, n)
        for i, i_end in _runs(m, PARTITIONS):
            r = i_end - i
            width = r + r % 2
            lead_coords = np.unravel_index(i + np.arange(r), tuple(lead))
            for j, j_end in _runs(n, MOVING_FREE):
                w = j_end - j
                acc = self.pool(
                    "acc", (PARTITIONS, MOVING_FREE), "float32", ACCUMULATORS
                )
                for c in range(k // MX_CHUNK):
                    a, b = c * rows_padded + i, c * cols_padded + j
                    stationary = f"[0:{PARTITIONS}, {a}:{a + width}]"
                    moving = f"[0:{PARTITIONS}, {b}:{b + w}]"
                    self.emit(
                        f"nisa.nc_matmul_mx(dst={acc}[0:{width}, 0:{w}], "
                        f"stationary={names[0]}{stationary}, "
                        f"moving={names[2]}{moving}, "
                        f"stationary_scale={names[1]}{stationary}, "
                        f"moving_scale={names[3]}{moving}, "
                        f"accumulate={c > 0})"
                    )
                grid = np.stack(
                    np.broadcast_arrays(
                        *[c[:, None] for c in lead_coords],
                        (j + np.arange(w))[None, :],
                    )
                )
                chunk = Chunk(acc, MOVING_FREE, out_shape, grid)
                self._chain(prims[1:], {anchor.name: chunk}, dst)

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
            temp = self.pool("q", (PARTITIONS, w), "bfloat16", 2)
            out = Chunk(temp, w, source.shape, source.coords, dtype="bfloat16")
            self._copy_chunk(out, source)
            source = out
        # Where the chunk's rows and columns land in the codes tile.
        if not codes_view.covers_box() or codes_view.shape != source.shape:
            raise NotImplementedError(f"{prim.name}: into part of a tile")
        mx, padded = self.mx_tiles[codes_view.box.name]
        x, k = mx.coords(np.ravel_multi_index(source.coords, source.shape))
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
            staging = self.pool("qs", (PARTITIONS, stage_free), "bfloat16", 2)
            for j in range(MX_PACK):
                data = (
                    f"{source.var}.ap(pattern=[[{source.free}, {r}], "
                    f"[{MX_PACK}, {parts}]], offset={lo + j})"
                )
                transposed = self._transpose(data, r, parts, "bfloat16")
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
        """Elementwise ops, chunk by chunk of the first op's result."""
        shape = self._result_shape(prims[0], {})
        rows, cols = _rows_cols(shape)
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
        computing op writes ``dst`` directly unless a relayout follows it,
        and then the relayout's result is copied there.
        """
        computing = [
            i for i, p in enumerate(prims) if p.target not in RELAYOUTS
        ]
        last = computing[-1] if computing else -1
        result = next(iter(values.values()), grid)
        for index, prim in enumerate(prims):
            if prim.target in RELAYOUTS:
                arg = prim.kwargs["input"].tensor_box
                result = self._read(arg, values)
                if prim.target == "aten::transpose":
                    result = result.transposed(
                        prim.kwargs["dim0"].scalar.int_value,
                        prim.kwargs["dim1"].scalar.int_value,
                    )
                values[prim.name] = result
                continue
            if prim.target == QUANTIZE_MX:
                if index != len(prims) - 1:
                    raise NotImplementedError(f"{prim.name}: placement")
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
            if index == last and last == len(prims) - 1:
                out = self._destination(dst, source)
            elif following[:1] == [QUANTIZE_MX]:
                # What ``quantize_mx`` reads, in its own type.
                r, w = source.coords.shape[1:]
                temp = self.pool("q", (PARTITIONS, w), "bfloat16", 2)
                out = Chunk(temp, w, shape, source.coords, dtype="bfloat16")
            else:
                r, w = source.coords.shape[1:]
                temp = self.pool("t", (PARTITIONS, w), "float32", 2)
                out = Chunk(temp, w, shape, source.coords)
            self._apply(prim, out, values, source)
            values[prim.name] = result = out
        if not computing or last != len(prims) - 1:
            self._copy_chunk(self._destination(dst, result), result)

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
        r, w = coords.shape[1:]
        if view.box.level == IMMEDIATE:
            return "scalar", repr(self.immediates[view.box.name])
        if view.box.level == DRAM:
            # A scalar read where it lies, as Voyager reads it: loaded onto
            # every partition, one value per row.
            if math.prod(view.shape) != 1:
                raise NotImplementedError("an elementwise operand in HBM")
            column = self.pool("sc", (PARTITIONS, 1), "float32", 2)
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
        flat = self.flat(view, linear, np.arange(r)[:, None])
        var, free = self.var(view.box, view.slot), self.free(view.box)
        if view.shape[-1] == 1 and w > 1:
            aps = fit([(free, flat[:, :1])], r)
            if aps is None:
                raise NotImplementedError("a column no pattern walks")
            column = _ap(var, aps[0])
            if view.box.dtype != "float32":
                # A per-row scalar operand is float32 to the engines.
                temp = self.pool("sc", (PARTITIONS, 1), "float32", 2)
                self.emit(
                    f"nisa.tensor_copy(dst={temp}[0:{r}, 0:1], src={column})"
                )
                column = f"{temp}[0:{r}, 0:1]"
            return "column", column
        return "tile", (var, free, flat)

    def _apply(self, prim, out, values, source):
        """Emit the instruction computing ``prim`` into ``out``."""
        target = prim.target
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
            if kinds == ["tile", "tile"]:
                self.emit(
                    f"nisa.tensor_tensor(dst={dst}, data1={args[0]}, "
                    f"data2={args[1]}, op={op})"
                )
            elif kinds[0] == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={dst}, data={args[0]}, "
                    f"op0={op}, operand0={args[1]})"
                )
            elif kinds[1] == "tile":
                self.emit(
                    f"nisa.tensor_scalar(dst={dst}, data={args[1]}, "
                    f"op0={op}, operand0={args[0]}, reverse0=True)"
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
                    f"op0=nl.multiply, operand0={inverse})"
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
            temp = self.pool("t", (PARTITIONS, w), "bfloat16", 2)
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

    def _transpose(self, data, r, c, dtype):
        """Transpose ``data``, an ``[r, c]`` access, on TensorE into PSUM;
        the access to the ``[c, r]`` result there.  An FP8 transpose writes
        every other element."""
        step = 2 if _fp8(dtype) is not None else 1
        psum = self.pool(
            "tp", (PARTITIONS, step * PARTITIONS), dtype, TRANSPOSES
        )
        result = (
            f"{psum}.ap(pattern=[[{step * PARTITIONS}, {c}], [{step}, {r}]], "
            f"offset=0)"
        )
        self.emit(f"nisa.nc_transpose(dst={result}, data={data})")
        return result

    def _reciprocal(self, value, kind, source):
        """``1 / value`` as an operand: a literal for a scalar, else computed
        into a temporary of the same kind (there is no ALU divide)."""
        if kind == "scalar":
            return repr(1.0 / float(value))
        r, w = source.coords.shape[1:]
        if kind == "column":
            temp = self.pool("sc", (PARTITIONS, 1), "float32", 2)
            inverse = f"{temp}[0:{r}, 0:1]"
        else:
            temp = self.pool("t", (PARTITIONS, w), "float32", 2)
            inverse = f"{temp}[0:{r}, 0:{w}]"
        self.emit(f"nisa.reciprocal(dst={inverse}, data={value})")
        return inverse

    def _sine(self, dst, angle, shift, chunk):
        """``sin(angle + shift)`` on ScalarE, the angle first brought into
        [-pi, pi]: ``k`` whole turns off by a cast to int32 (whichever way it
        rounds), and one more if that leaves it above pi."""
        r, w = chunk
        shifted, reduced = (
            self.pool("t", (PARTITIONS, w), "float32", 2) + f"[0:{r}, 0:{w}]"
            for _ in range(2)
        )
        turns = self.pool("k", (PARTITIONS, w), "int32", 2) + f"[0:{r}, 0:{w}]"
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
            other = self.pool("t", (PARTITIONS, w), "float32", 2)
            inverse = f"{other}[0:{r}, 0:{w}]"
        elif kinds[0] == "column":
            other = self.pool("sc", (PARTITIONS, 1), "float32", 2)
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
            temp = self.pool("t", (PARTITIONS, w), "float32", 2)
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

    def _rowwise(self, prims, dst, rows_op):
        """A row-wise op (a layer norm, a softmax) and the ops fused after
        it, 128 rows at a time: ``rows_op(prim, x, start, r, out)`` writes
        the op's rows ``start .. start + r`` of input ``x`` to ``out`` --
        ``dst`` itself, or a temporary the ops after it then read."""
        anchor = prims[0]
        x = self._view(anchor.kwargs["input"].tensor_box)
        rows, cols = _rows_cols(x.shape)
        for start, stop in _runs(rows, PARTITIONS):
            r = stop - start
            if len(prims) == 1:
                rows_op(
                    anchor, x, start, r, self._rows_ap(dst, start, r, 0, cols)
                )
                continue
            temp = self.pool("rw", (PARTITIONS, cols), "float32", 2)
            rows_op(anchor, x, start, r, f"{temp}[0:{r}, 0:{cols}]")
            linear = (start + np.arange(r))[:, None] * cols + np.arange(cols)
            coords = np.stack(np.unravel_index(linear, x.shape))
            result = Chunk(temp, cols, x.shape, coords)
            self._chain(prims[1:], {anchor.name: result}, dst)

    def _layer_norm(self, prim, x, start, r, out):
        """Layer norm of rows ``start .. start + r`` over the last dim: the
        moments (``bn_stats`` / ``bn_aggr``), then normalize, scale and
        shift."""
        kw = self._kwargs(prim)
        cols = x.shape[-1]
        if list(kw["normalized_shape"]) != [cols]:
            raise NotImplementedError("a layer norm over more than one dim")
        affine = [
            (self._view(prim.kwargs[key].tensor_box), op)
            for key, op in (("weight", "nl.multiply"), ("bias", "nl.add"))
            if kw.get(key) is not None
        ]
        pieces = _runs(cols, STATS_FREE)
        data = self._rows_ap(x, start, r, 0, cols)
        stats = self.pool("st", (PARTITIONS, 6 * len(pieces)), "float32", 2)
        for q, (lo, hi) in enumerate(pieces):
            self.emit(
                f"nisa.bn_stats(dst={stats}[0:{r}, {6 * q}:{6 * q + 6}], "
                f"data={self._rows_ap(x, start, r, lo, hi)})"
            )
        moments = self.pool("mv", (PARTITIONS, 2), "float32", 2)
        self.emit(
            f"nisa.bn_aggr(dst={moments}[0:{r}, 0:2], "
            f"data={stats}[0:{r}, 0:{6 * len(pieces)}])"
        )
        rstd = self.pool("rs", (PARTITIONS, 1), "float32", 2)
        self.emit(
            f"nisa.tensor_scalar(dst={rstd}[0:{r}, 0:1], "
            f"data={moments}[0:{r}, 1:2], op0=nl.add, "
            f"operand0={float(kw['eps'])!r})"
        )
        self.emit(
            f"nisa.activation(dst={rstd}[0:{r}, 0:1], op=nl.rsqrt, "
            f"data={rstd}[0:{r}, 0:1])"
        )
        norm = out
        if affine:
            temp = self.pool("t", (PARTITIONS, cols), "float32", 2)
            norm = f"{temp}[0:{r}, 0:{cols}]"
        self.emit(
            f"nisa.tensor_scalar(dst={norm}, data={data}, "
            f"op0=nl.subtract, operand0={moments}[0:{r}, 0:1], "
            f"op1=nl.multiply, operand1={rstd}[0:{r}, 0:1])"
        )
        for index, (view, op) in enumerate(affine):
            into = out if index == len(affine) - 1 else norm
            self.emit(
                f"nisa.tensor_tensor(dst={into}, data1={norm}, "
                f"data2={self._rows_ap(view, 0, r, 0, cols)}, op={op})"
            )

    def _softmax(self, prim, x, start, r, out):
        """Softmax of rows ``start .. start + r`` over the last dim: the row
        maximum, the exponentials summed on the way, the scaling by the
        sum's reciprocal."""
        kw = self._kwargs(prim)
        cols = x.shape[-1]
        if kw["dim"] % len(x.shape) != len(x.shape) - 1:
            raise NotImplementedError("a softmax over a dim but the last")
        data = self._rows_ap(x, start, r, 0, cols)
        top = self.pool("mx", (PARTITIONS, 1), "float32", 2)
        total = self.pool("sm", (PARTITIONS, 1), "float32", 2)
        exps = self.pool("t", (PARTITIONS, cols), "float32", 2)
        self.emit(
            f"nisa.tensor_reduce(dst={top}[0:{r}, 0:1], op=nl.maximum, "
            f"data={data}, axis=1, keepdims=True)"
        )
        self.emit(
            f"nisa.tensor_scalar(dst={top}[0:{r}, 0:1], "
            f"data={top}[0:{r}, 0:1], op0=nl.multiply, operand0=-1.0)"
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
            f"op0=nl.multiply, operand0={total}[0:{r}, 0:1])"
        )

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
            if box.level != SCRATCHPAD:
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
            for slot in range(box.slots):
                var = self.var(box, slot)
                lines.append(
                    f"    {var} = nl.ndarray({shape!r}, dtype={dtype}, "
                    f"buffer=nl.sbuf)"
                )
                if box.name in self.mx_tiles and padded != mx.other:
                    # The pad column a stationary chunk of odd width takes.
                    lines.append(f"    nisa.memset(dst={var}, value=0)")
        for (kind, shape, dtype), (names, _) in self.pools.items():
            buffer = "nl.psum" if kind in ("tp", "acc") else "nl.sbuf"
            for var in names:
                lines.append(
                    f"    {var} = nl.ndarray({shape!r}, "
                    f"dtype={nl_dtype(dtype)}, buffer={buffer})"
                )
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


def translate(model, out_dir, immediates):
    """Translate ``model`` into ``out_dir``.

    Args:
        model: A ``voyager`` IR ``Model``, or the path of its ``model.txt``.
        out_dir: Where the translation is written.
        immediates: The value of each immediate box (a constant the IR
            names but does not hold), by name.

    Returns:
        The manifest written to ``manifest.json``.
    """
    if isinstance(model, str):
        with open(model) as f:
            model = text_format.Parse(f.read(), ir.Model())
    os.makedirs(out_dir, exist_ok=True)
    mx, mx_scales = mx_arrays(model)
    sources, steps, lines, translated = {}, [], [], {}
    allocations = []
    for kind, item in segments(model):
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
                kernel = Kernel(item, immediates, mx)
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
                translated[key] = (
                    name,
                    casts,
                    [
                        [names.index(b) for b in boxes]
                        for boxes in (kernel.reads, kernel.writes)
                    ],
                )
            name, casts, indices = translated[key]
            reads, writes = ([names[i] for i in group] for group in indices)
            step = {"kernel": name, "inputs": reads, "outputs": writes}
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
    with open(os.path.join(out_dir, "program.py"), "w") as f:
        f.write(program_source(statements, outputs, state))
    dtypes = _hbm_dtypes(model)
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
            for kind, op in segments(model)
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


def program_source(statements, outputs, state):
    """Source of ``program.py``: ``statements`` -- ``(reads, writes,
    line)`` each -- in order, ``mark`` calls between the segments a decoder
    layer's statements make, the ``state`` parameters it writes stored back
    into ``params``, and ``outputs`` returned.

    A statement belongs to the layer named in a box it writes; one that
    writes none joins the layer before it, or, after the last layer's
    statements, the segment after the layers.
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
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.message_type.GetOptions().map_entry:
            value = value.values()
        elif not field.is_repeated:
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
