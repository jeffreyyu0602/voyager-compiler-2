"""
``voyager`` torch.library namespace: explicit bufferization / memory primitives.

These are intentionally distinct from ``quantized_ops`` so that compiler memory
semantics (logical storage objects, DMA tile loads/stores, accumulator init,
counters) are not confused with model tensor semantics.  They are emitted by
the bufferization lowering pass and consumed by the loop-aware code generator.

Primitives
----------
``voyager.alloc(size, dtype)``       the one op that declares storage: a DRAM
                                     buffer, an on-chip tile, or (0-d int64
                                     at the register level) a counter.
``voyager.zeros(t)`` / ``ones(t)``   fill an existing tensor in place; on a
                                     counter, ``zeros`` is its reset.
``voyager.subview(src, o, s, st)``   strided window onto a buffer (a slot).
``voyager.insert(src, dst)``         destination-passing compute-result write.
``voyager.async_copy(..., c)``       async tile DMA; adds 1 to counter ``c``
                                     when the tile has landed.
``voyager.async_wait(c, thr)``       the CPU blocks until ``c >= thr``.
``voyager.drain()``                  the CPU blocks until every unit is idle.
``voyager.increment_indices(...)``   multi-dim tile-index counter (carry).
``voyager.delinearize_index(...)``   linear loop counter -> multi-dim index.
``commit(subgraph, operands, ...)``  one asynchronous dispatch (a higher-order
                                     op): each unit starts its pass once the
                                     waits listed for it hold, and the last
                                     pass adds 1 to ``post``.
``with_scratch(subgraph, ...)``      an op beside the on-chip regions its
                                     kernel keeps (a higher-order op).

Counters only count up.  A wait holds once the counter has reached its
threshold, so a wait is idempotent and never consumes anything.
"""

from contextlib import contextmanager
from enum import IntEnum
from typing import Optional, Tuple

import torch
from torch._higher_order_ops.base_hop import (
    BaseHOP,
    FunctionWithNoFreeVars,
)
from torch.library import Library, impl

# "DEF" creates/owns the namespace.  Keep a module-level handle alive.
voyager_lib = Library("voyager", "DEF")

# The units a dispatch's waits are grouped by: the matrix-class unit of the
# dispatch (systolic array or MVM) and the vector unit.
UNITS = ("matrix", "vector")


class MemoryLevel(IntEnum):
    """Memory-hierarchy levels, ordered by distance from the compute units
    (smaller = nearer/faster).  ``voyager.alloc(..., space=)`` records which
    level a bufferized storage object lives in (e.g. an ``SRAM``
    software-pipeline slot vs the ``DRAM`` output buffer); shared wherever a
    buffer's level is needed."""

    REGISTER = 0  # PE register file
    LOCAL = 1  # local buffers (input and weight buffers)
    SRAM = 2  # on-chip scratchpad / SRAM
    DRAM = 3  # main system memory (e.g., DDR)


# ---------------------------------------------------------------------------
# Software pipelining.
#
# ``alloc`` takes a ``num_slots`` count.  ``num_slots == 0`` is an
# unpipelined storage object: the tensor has exactly ``size``.  ``num_slots >=
# 1`` prepends a slot dimension (``[num_slots, *size]``), so ``size`` stays the
# payload of *one* slot and ``buf[slot]`` reads/writes a slot.  The distinction
# matters to the code generator: a slot dimension is not a tensor dimension —
# it is serialized as ``TensorBox.bank_count`` / ``bank_stride_bytes``, and the
# slot a step picks is the leading offset of the operand's window.  A
# single-slot buffer (``num_slots == 1``) still keeps its dimension, so
# ``buf[0]`` addresses it uniformly.
# ---------------------------------------------------------------------------
UNPIPELINED = 0


def _pipelined_size(size, num_slots: int):
    return list(size) if num_slots == UNPIPELINED else [num_slots] + list(size)


# ---------------------------------------------------------------------------
# voyager.alloc — logical storage object (output / staging buffer, on-chip
# tile, or counter).  ``space`` is a ``MemoryLevel`` (int): which level of
# the hierarchy it lives in — ``DRAM`` (default: output / staging), ``SRAM``
# (an on-chip scratchpad tile, e.g. a software-pipeline ping-pong slot the
# DMA fills) or ``REGISTER`` (a hardware counter: ``alloc([], torch.int64,
# MemoryLevel.REGISTER)``, reset by ``voyager.zeros``).  Eager allocation is
# space-agnostic (a plain tensor); the level is metadata the planner /
# codegen read.
# ---------------------------------------------------------------------------
# ``space`` default 3 == ``MemoryLevel.DRAM``.
voyager_lib.define(
    "alloc(SymInt[] size, ScalarType dtype, int space=3, int num_slots=0) "
    "-> Tensor"
)


@impl(voyager_lib, "alloc", "CompositeExplicitAutograd")
def alloc(
    size: Tuple[int, ...],
    dtype: torch.dtype,
    space: int = MemoryLevel.DRAM,
    num_slots: int = UNPIPELINED,
) -> torch.Tensor:
    size = _pipelined_size(size, num_slots)
    if dtype.is_floating_point:
        return torch.randn(size).to(dtype)
    return torch.zeros(size, dtype=dtype)


@torch.library.register_fake("voyager::alloc")
def _alloc_fake(
    size: Tuple[int, ...],
    dtype: torch.dtype,
    space: int = MemoryLevel.DRAM,
    num_slots: int = UNPIPELINED,
) -> torch.Tensor:
    return torch.empty(_pipelined_size(size, num_slots), dtype=dtype)


# ---------------------------------------------------------------------------
# The race check: a happens-before audit of one eager run, independent of the
# order it runs in.  Every dispatch and copy is a producer, named by the
# counter it adds to and the count it brings it to.  Per storage range the
# check keeps the last writer and the readers since; an access must be
# covered against each producer it depends on -- by a wait of the commit on
# that counter at or above the count, a CPU wait already made at or above
# it, or a drain since -- or the run raises.  A sequential run satisfies
# every wait however few there are; this checks that the waits are there.
# Only contiguous views are audited (every on-chip slot and tile): a strided
# DRAM tile's byte span runs across its neighbours.
# ---------------------------------------------------------------------------
class _RaceCheck:
    def __init__(self):
        self.cpu = {}  # counter key -> highest threshold the CPU waited
        self.ranges = {}  # storage -> {(start, end): [writer, readers]}
        self.seq = 0  # dispatches and copies issued so far
        self.kind = {}  # seq -> "a commit" / "an async_copy"
        # Per counter key: its resets so far, and the highest count posted
        # on it since the last.  A reset closes what posted before it.
        self.generation = {}
        self.posted = {}
        self.drained = 0  # every producer at or below this seq is done

    @staticmethod
    def key(counter):
        return counter.untyped_storage().data_ptr()

    @staticmethod
    def span(t):
        """``(storage, start, end)`` in bytes of the memory ``t`` views."""
        size = t.element_size()
        start = t.storage_offset() * size
        if t.numel() == 0:
            return t.untyped_storage().data_ptr(), start, start
        last = sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
        return t.untyped_storage().data_ptr(), start, start + (last + 1) * size

    def covered(self, producer, waits):
        seq, key, count, _, generation = producer
        if seq <= self.drained or self.cpu.get(key, 0) >= count:
            return True
        if generation < self.generation.get(key, 0):
            return True
        return any(k == key and thr >= count for k, thr in waits)

    def producer(self, counter, unit_key):
        """The producer that just brought ``counter`` to its count."""
        key, count = self.key(counter), int(counter.item())
        self.posted[key] = max(self.posted.get(key, 0), count)
        return (self.seq, key, count, unit_key, self.generation.get(key, 0))

    def _entries(self, t):
        storage, start, end = self.span(t)
        table = self.ranges.setdefault(storage, {})
        hits = [v for (a, b), v in table.items() if a < end and start < b]
        return table.setdefault((start, end), [None, []]), hits

    def access(self, what, t, write, waits, unit_key, stream=None):
        """Check one access against the producers it depends on: the last
        writer (RAW, WAW) and, for a write, the readers since on another
        unit (WAR; ``unit_key`` is the posting counter, ``None`` for a
        copy, which every unit's readers bind).  A copy on counter
        ``stream`` is ordered behind the earlier copies on it: copies on
        one counter land in issue order."""
        if not t.is_contiguous():
            return
        _, hits = self._entries(t)
        for writer, readers in hits:
            needed = [writer] if writer is not None else []
            if write:
                needed += [
                    r for r in readers if unit_key is None or r[3] != unit_key
                ]
            for producer in needed:
                same_stream = producer[3] is None and producer[1] == stream
                if (stream is not None and same_stream) or self.covered(
                    producer, waits
                ):
                    continue
                verb = "wrote" if producer is writer else "read"
                raise AssertionError(
                    f"race check: {what} (#{self.seq + 1}) accesses memory "
                    f"{self.kind[producer[0]]} #{producer[0]} {verb}, with "
                    f"no wait on its counter reaching {producer[2]}"
                )

    def record(self, t, write, producer):
        if not t.is_contiguous():
            return
        entry, hits = self._entries(t)
        if write:
            for hit in hits:
                hit[0], hit[1] = producer, []
        else:
            for hit in hits:
                hit[1].append(producer)

    def reset(self, counter):
        """A counter reset: whatever counted on it -- on this storage,
        which a later counter may reuse -- must have been waited at its
        final count (every kernel ends quiescent); it then closes them."""
        key = self.key(counter)
        final = max(int(counter.item()), self.posted.get(key, 0))
        if final and self.cpu.get(key, 0) < final:
            raise AssertionError(
                f"race check: a counter that reached {final} is reset with "
                f"the CPU having waited it only to {self.cpu.get(key, 0)}"
            )
        self.cpu.pop(key, None)
        self.posted.pop(key, None)
        self.generation[key] = self.generation.get(key, 0) + 1


_race_check: Optional[_RaceCheck] = None


@contextmanager
def race_check():
    """Audit every dispatch, copy and counter reset run in this scope
    (``_RaceCheck``); raise on the first access not covered by a wait."""
    global _race_check
    prev, _race_check = _race_check, _RaceCheck()
    try:
        yield
    finally:
        _race_check = prev


# ---------------------------------------------------------------------------
# voyager.zeros / voyager.ones — fill an existing tensor in place.  Neither
# allocates: the tensor is an ``alloc`` (or a view of one).  ``zeros`` on an
# on-chip tile is a zero fill (a reduction accumulator's init); on a counter
# (a 0-d int64 ``alloc`` at ``MemoryLevel.REGISTER``) it is the reset to 0.
# They are ``voyager`` ops (not ``aten.zero_`` / ``fill_``) so the bufferizer
# can tell a buffer fill from a genuine compute op.
# ---------------------------------------------------------------------------
voyager_lib.define("zeros(Tensor(a!) self) -> ()")


@impl(voyager_lib, "zeros", "CompositeExplicitAutograd")
def zeros(self: torch.Tensor) -> None:
    if _race_check is not None and self.dim() == 0:
        _race_check.reset(self)
    self.zero_()


@torch.library.register_fake("voyager::zeros")
def _zeros_fake(self: torch.Tensor) -> None:
    return None


voyager_lib.define("ones(Tensor(a!) self) -> ()")


@impl(voyager_lib, "ones", "CompositeExplicitAutograd")
def ones(self: torch.Tensor) -> None:
    self.fill_(1)


@torch.library.register_fake("voyager::ones")
def _ones_fake(self: torch.Tensor) -> None:
    return None


# ---------------------------------------------------------------------------
# voyager.subview — a strided window onto a buffer, after MLIR's memref.subview:
# ``offsets`` / ``sizes`` / ``strides`` all carry one entry per *source* dim.
# ``squeeze_dim`` names the windowed dims to drop from the result (each of size
# 1), which is how memref.subview reduces rank — a slot dim is not a tensor dim,
# so a slot pick drops it.  It names no storage of its own; the code generator
# folds it into the operand's ``TensorBoxRef``: the window that reference makes
# onto the buffer.
#
# It must return a genuine *view*: a slot is a write destination
# (``insert(src, slot)``), and the FX graph stays executable, so a copy
# would land the write in a throwaway tensor.  ``as_strided`` gives the view and
# keeps the result's shape static -- only its storage offset is dynamic, which
# is what lets the slot index (``step % num_slots``) stay a runtime value.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "subview(Tensor(a) source, SymInt[] offsets, SymInt[] sizes, "
    "SymInt[] strides, int[] squeeze_dim=[]) -> Tensor(a)"
)


def _subview(
    source: torch.Tensor,
    offsets: Tuple[int, ...],
    sizes: Tuple[int, ...],
    strides: Tuple[int, ...],
    squeeze_dim: Tuple[int, ...] = (),
) -> torch.Tensor:
    window = [source.stride(d) * s for d, s in enumerate(strides)]
    offset = source.storage_offset() + sum(
        o * source.stride(d) for d, o in enumerate(offsets)
    )
    kept = [d for d in range(len(sizes)) if d not in squeeze_dim]
    return torch.as_strided(
        source,
        [sizes[d] for d in kept],
        [window[d] for d in kept],
        offset,
    )


impl(voyager_lib, "subview", "CompositeExplicitAutograd")(_subview)
torch.library.register_fake("voyager::subview")(_subview)


voyager_lib.define("insert(Tensor src, Tensor(a!) dst) -> ()")


@impl(voyager_lib, "insert", "CompositeExplicitAutograd")
def insert(src: torch.Tensor, dst: torch.Tensor) -> None:
    dst.copy_(src)


@torch.library.register_fake("voyager::insert")
def _insert_fake(src: torch.Tensor, dst: torch.Tensor) -> None:
    return None


# ---------------------------------------------------------------------------
# voyager.async_copy / voyager.async_wait / voyager.drain — the asynchronous
# DMA and the CPU's waits.
#
# ``async_copy(..., counter)`` copies a tile and adds 1 to ``counter`` once
# the tile has landed; the copies on one counter land in issue order.
# ``async_wait(counter, threshold)`` blocks the CPU until ``counter >=
# threshold``.  Both take a 0-d int64 ``counter``, **return nothing**, and
# declare it **mutable** (``Tensor(a!)`` / ``Tensor(b!)``) so the graph keeps
# their order.  ``drain()`` blocks the CPU until no unit has a pass running
# or queued; it says nothing about copies, which counters cover.
#
# The eager impls are the counter oracle: ``async_copy`` adds its 1 at once
# and ``async_wait`` asserts the threshold is already met, so running the
# bufferized graph in program order proves every wait holds when it runs.
#
# ``count`` (default: the whole tile) is how much of the block actually moves,
# and may be a runtime scalar. It never moves the transfer: ``sizes`` still
# names the block and still discriminates load from store, so a variable-length
# payload keeps a fixed, statically-sized home. A shorter ``sizes`` would flip
# the direction test instead.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "async_copy(Tensor src, Tensor(a!) dst, SymInt[] indices, SymInt[] sizes, "
    "Tensor(b!) counter, SymInt[]? dims=None, SymInt[]? strides=None, "
    "bool transposed=False, SymInt[]? pad=None, float? pad_value=None, "
    "SymInt[]? count=None) -> ()"
)


@impl(voyager_lib, "async_copy", "CompositeExplicitAutograd")
def async_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    counter: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
    count: Optional[Tuple[int, ...]] = None,
) -> None:
    if strides is None:
        strides = sizes

    if dims is None:
        offsets = tuple(i * s for i, s in zip(indices, strides))
    else:
        offsets = [0] * len(sizes)
        for index, dim in zip(indices, dims):
            offsets[dim] = index * strides[dim]
        offsets = tuple(offsets)

    if pad is not None:
        dst.fill_(0 if pad_value is None else pad_value)
        src_slices, dst_slices = [], []
        for offset, size, padding, extent in zip(
            offsets, sizes, pad, src.shape
        ):
            start = offset - padding
            stop = start + size
            clipped_start = max(0, min(start, extent))
            clipped_stop = max(0, min(stop, extent))
            src_slices.append(slice(clipped_start, clipped_stop))
            dst_slices.append(
                slice(clipped_start - start, clipped_stop - start)
            )
        dst[tuple(dst_slices)].copy_(src[tuple(src_slices)])
    else:
        block = tuple(
            slice(offset, offset + size) for offset, size in zip(offsets, sizes)
        )
        payload = (
            tuple(slice(0, extent) for extent in count)
            if count is not None
            else ...
        )
        if transposed:
            dst.copy_(src[block].mT)
        elif tuple(dst.shape) == tuple(sizes):
            dst[payload].copy_(src[block][payload])
        else:
            dst[block][payload] = src[payload]

    if _race_check is not None:
        rc = _race_check
        key = rc.key(counter)
        rc.access("an async_copy's source", src, False, (), None, key)
        rc.access("an async_copy's destination", dst, True, (), None, key)
    counter.add_(1)
    if _race_check is not None:
        rc.seq += 1
        rc.kind[rc.seq] = "an async_copy"
        producer = rc.producer(counter, None)
        rc.record(src, False, producer)
        rc.record(dst, True, producer)


@torch.library.register_fake("voyager::async_copy")
def _async_copy_fake(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    counter: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
    count: Optional[Tuple[int, ...]] = None,
) -> None:
    return None


voyager_lib.define("async_wait(Tensor(a!) counter, SymInt threshold) -> ()")


# The counter check in ``async_wait`` and ``commit`` is a runtime *oracle*: it
# verifies that a real run of the pipeline meets every wait by the time the
# program reaches it.  It is only meaningful when the whole loop executes in
# sequence.  Codegen's ShapeProp re-runs a single loop-body iteration in
# isolation — where a wait legitimately precedes the copies and dispatches of
# earlier iterations — so it disables the oracle via ``oracle_disabled()``
# (it needs only shapes/values, not the counts).
_oracle_enabled = True


@contextmanager
def oracle_disabled():
    """Suspend the counter oracle of ``async_wait`` / ``commit`` in this
    scope."""
    global _oracle_enabled
    prev = _oracle_enabled
    _oracle_enabled = False
    try:
        yield
    finally:
        _oracle_enabled = prev


def _check_wait(counter: torch.Tensor, threshold, what: str) -> None:
    """The oracle of one wait: ``counter`` has reached ``threshold``."""
    if not _oracle_enabled:
        return
    reached = int(counter.item())
    if reached < int(threshold):
        raise AssertionError(
            f"{what} waits for a counter at {reached} to reach {threshold}"
        )


@impl(voyager_lib, "async_wait", "CompositeExplicitAutograd")
def async_wait(counter: torch.Tensor, threshold: int) -> None:
    _check_wait(counter, threshold, "async_wait")
    if _race_check is not None:
        key = _race_check.key(counter)
        cpu = _race_check.cpu
        cpu[key] = max(cpu.get(key, 0), int(threshold))
    return None


@torch.library.register_fake("voyager::async_wait")
def _async_wait_fake(counter: torch.Tensor, threshold: int) -> None:
    return None


voyager_lib.define("drain() -> ()")


@impl(voyager_lib, "drain", "CompositeExplicitAutograd")
def drain() -> None:
    # Dispatches run to completion in the eager model: nothing to wait for.
    if _race_check is not None:
        _race_check.drained = _race_check.seq
    return None


@torch.library.register_fake("voyager::drain")
def _drain_fake() -> None:
    return None


# A drain reads and writes nothing the graph names; keep it through dead
# code elimination.
torch.fx.node._side_effectful_functions.add(torch.ops.voyager.drain.default)


# ---------------------------------------------------------------------------
# voyager.increment_indices — advance a nested tile-index counter by one step.
#
# Plain integer counters with innermost-first carry: the last index is bumped,
# and each overflow (``index[d] >= ends[d]``) carries +1 into ``index[d-1]``
# and wraps ``index[d]``.  The outermost index (d=0) is never wrapped — the
# enclosing ``while_loop``'s bound on it terminates the loop.  This folds the
# per-dimension ``add / floordiv / mod`` carry arithmetic into a single node so
# the bufferized graph stays readable instead of sprouting many scalar ops per
# loop step.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "increment_indices(SymInt[] indices, SymInt[] ends) -> SymInt[]"
)


def _advance_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    out = list(indices)
    out[-1] = out[-1] + 1
    for d in range(len(out) - 1, 0, -1):
        out[d - 1] = out[d - 1] + out[d] // ends[d]
        out[d] = out[d] % ends[d]
    return tuple(out)


@impl(voyager_lib, "increment_indices", "CompositeExplicitAutograd")
def increment_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    return _advance_indices(indices, ends)


@torch.library.register_fake("voyager::increment_indices")
def _increment_indices_fake(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    return _advance_indices(indices, ends)


# ---------------------------------------------------------------------------
# voyager.delinearize_index — recover a multi-dim tile index from a single
# linear index and a per-dimension ``basis`` (the tile counts), mirroring
# MLIR's ``affine.delinearize_index``.  A flattened (software-pipelined) loop
# carries just a *single* counter — a real ``for`` induction variable — and
# delinearizes it each iteration for addressing, instead of carrying the whole
# multi-dim index as loop state.  Row-major: ``linear = d0*(b1*…*bn) + … + dn``
# with ``0 <= di < basis[i]``.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "delinearize_index(SymInt linear, SymInt[] basis) -> SymInt[]"
)


def _delinearize(linear: int, basis: Tuple[int, ...]) -> Tuple[int, ...]:
    out = [0] * len(basis)
    for d in range(len(basis) - 1, -1, -1):
        out[d] = linear % basis[d]
        linear = linear // basis[d]
    return tuple(out)


@impl(voyager_lib, "delinearize_index", "CompositeExplicitAutograd")
def delinearize_index(linear: int, basis: Tuple[int, ...]) -> Tuple[int, ...]:
    return _delinearize(linear, basis)


@torch.library.register_fake("voyager::delinearize_index")
def _delinearize_index_fake(
    linear: int, basis: Tuple[int, ...]
) -> Tuple[int, ...]:
    return _delinearize(linear, basis)


# ---------------------------------------------------------------------------
# voyager.commit — one asynchronous dispatch (a higher-order op).
#
# Unlike the primitives above, ``commit`` takes a *subgraph* (a traceable
# callable / ``GraphModule``), so it is a ``HigherOrderOperator`` (via
# ``BaseHOP``), not a schema-based ``torch.library`` op — exactly like
# ``torch.cond`` / ``torch.while_loop``.  It survives tracing as one
# ``call_function`` node whose first argument is the traced ``GraphModule`` it
# calls; it therefore lives in the ``higher_order`` namespace
# (``torch.ops.higher_order.commit``), not the ``voyager`` one.
#
# ``dependencies`` maps a unit (``UNITS``) to the ``(counter, threshold)``
# waits that unit's pass of the region checks before it starts; passes of one
# unit start in dispatch order, so a wait names only a cross-unit or copy
# dependency.  ``post`` is the counter the region's last pass adds 1 to.  The
# eager implementation runs the region at once: it asserts every wait holds
# (the ``async_wait`` oracle, which ShapeProp can disable), runs the
# subgraph, then adds the post.
#
# Public entry is the ``commit(subgraph, operands, dependencies, post)``
# wrapper below: ``operands`` is an explicit list (like ``torch.cond`` /
# ``while_loop``), so ``dependencies`` / ``post`` may be passed positionally.
# The HOP's own ``__call__`` must stay variadic — HOP dispatch re-enters it
# with the operands already flattened, so a list parameter there would break
# re-dispatch — hence the wrapper splats the list into it.
# ---------------------------------------------------------------------------
class AsyncOp(BaseHOP):
    def __init__(self) -> None:
        super().__init__("commit")

    def __call__(self, subgraph, *operands, dependencies=None, post=None):
        if not isinstance(
            subgraph, (torch.fx.GraphModule, FunctionWithNoFreeVars)
        ):
            subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph, *operands, dependencies=dependencies, post=post
        )

    def _call_CompositeExplicitAutograd(
        self, subgraph, *operands, dependencies=None, post=None
    ):
        # Eager / ShapeProp execution: check every wait, run the region,
        # then post.  Tracing (Fake / Proxy modes) runs the subgraph for
        # shapes only and leaves the counters untouched — BaseHOP's mode
        # impls ignore these kwargs.
        for unit, waits in (dependencies or {}).items():
            for counter, threshold in waits:
                _check_wait(counter, threshold, f"the {unit} pass")
        accesses = []
        if _race_check is not None and isinstance(
            subgraph, torch.fx.GraphModule
        ):
            accesses = _commit_accesses(subgraph, operands)
            waits = [
                (_race_check.key(c), int(t))
                for pairs in (dependencies or {}).values()
                for c, t in pairs
            ]
            unit = _race_check.key(post) if post is not None else None
            name = f"a commit ({type(subgraph).__name__})"
            for t, write in accesses:
                _race_check.access(name, t, write, waits, unit)
        out = subgraph(*operands)
        if post is not None:
            post.add_(1)
        if accesses:
            rc = _race_check
            rc.seq += 1
            rc.kind[rc.seq] = f"a commit ({type(subgraph).__name__})"
            if post is None:
                # Nothing can wait on a dispatch that posts nothing: what
                # depends on it has to drain.
                producer = (rc.seq, None, 1, None, 0)
            else:
                producer = rc.producer(post, rc.key(post))
            for t, write in accesses:
                rc.record(t, write, producer)
        return out


commit_hop = AsyncOp()


def _commit_accesses(subgraph, operands):
    """``[(tensor, writes)]`` for a commit's tensor operands: the ones its
    body stores into (an ``insert`` destination, a ``with_scratch``
    region) are written, the rest read."""
    phs = [n for n in subgraph.graph.nodes if n.op == "placeholder"]
    written = set()
    for n in subgraph.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target is torch.ops.voyager.insert.default:
            written.add(n.args[1])
        elif n.target is scratch_hop:
            written.update(n.kwargs["scratch"].values())
    return [
        (t, ph in written)
        for ph, t in zip(phs, operands)
        if isinstance(t, torch.Tensor) and t.dim() > 0
    ]


def commit(subgraph, operands, dependencies=None, post=None):
    """Dispatch ``subgraph(*operands)`` as one asynchronous region.

    Args:
        subgraph: The region's computation, a callable or ``GraphModule``.
        operands: Its tensor operands, an explicit list, splatted into the
            underlying ``commit_hop`` HOP (which survives tracing as a
            single ``call_function`` node over the traced subgraph).
        dependencies: ``{unit: [(counter, threshold), ...]}`` over
            ``UNITS``; a unit absent or empty waits nothing.
        post: The counter the region's last pass adds 1 to, or ``None``.

    Returns:
        ``subgraph(*operands)``.

    Raises:
        ValueError: ``dependencies`` names a unit outside ``UNITS``.
    """
    dependencies = {
        unit: list(waits) for unit, waits in (dependencies or {}).items()
    }
    unknown = set(dependencies) - set(UNITS)
    if unknown:
        raise ValueError(f"commit waits on unknown units {sorted(unknown)}")
    return commit_hop(subgraph, *operands, dependencies=dependencies, post=post)


# ---------------------------------------------------------------------------
# with_scratch — an op's computation beside the on-chip regions its backend
# kernel keeps that no FX node names (a higher-order op, like ``commit``).
#
# ``subgraph(*operands)`` is the op, bare or a fused group; ``scratch`` maps
# each region's name to its buffer.  Nothing reads or writes the regions in
# FX, so eager and ShapeProp run the subgraph alone; the node naming them is
# what tells memory planning to keep them live across the op and the emitter
# to hand them to it.
# ---------------------------------------------------------------------------
class ScratchOp(BaseHOP):
    def __init__(self) -> None:
        super().__init__("with_scratch")

    def __call__(self, subgraph, *operands, scratch):
        if not isinstance(
            subgraph, (torch.fx.GraphModule, FunctionWithNoFreeVars)
        ):
            subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(subgraph, *operands, scratch=scratch)


scratch_hop = ScratchOp()


def with_scratch(subgraph, operands, scratch):
    """Run ``subgraph(*operands)`` beside the regions ``scratch`` names.

    With no region it is the plain call, so an op whose backend keeps none
    traces exactly as it would unwrapped.  Otherwise it traces to one
    ``scratch_hop`` node over the traced subgraph; in a fused group the
    regions belong to its anchor.

    Args:
        subgraph: The op's computation, a callable or ``GraphModule``.
        operands: Its tensor operands, an explicit list.
        scratch: ``{name: buffer}``, one entry per region.

    Returns:
        ``subgraph(*operands)``.
    """
    if not scratch:
        return subgraph(*operands)
    return scratch_hop(subgraph, *operands, scratch=scratch)
