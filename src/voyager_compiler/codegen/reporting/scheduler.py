"""The resource/timing state machine.

A compute occupies the unit(s) named by its ``OpInfo.units`` — ``mma``
(systolic matrix array), ``vector`` (vector unit), or both (a fused GEMM /
conv, whose tail runs on the VU).  Whether it advances the **program clock**
depends on how it is dispatched:

* **synchronous**: advances the program clock, so it serializes and forms the
  clock the waits reconcile against.
* **asynchronous** (dispatched inside a ``voyager.commit``): occupies its
  unit(s) but does **not** advance the program clock, so it runs while the clock
  moves on other units; the commit adds its completion to its counter,
  exactly as ``async_copy`` does.  Two ops on the same unit still serialize
  (shared ``*_free`` counter); overlap appears only across different units.

``async_copy`` *is* asynchronous: it occupies DRAM for its transfer without
advancing the program clock, so a prefetch overlaps compute.  A counter is
the list of completion cycles of what has been added to it, in order; a wait
for ``counter >= threshold`` resolves at the ``threshold``-th of them,
whether that post came from a DMA or an async compute.  ``async_wait``
blocks the program clock until its wait resolves; a commit's passes wait
per unit before they start.  A pass frees its unit(s) at the end of its
charged cycles but is done ``OpInfo.completion_cycles`` later, and its post
carries the done.

Every timestamp the machine holds -- the clocks, the unit-free cycles, the
posted completions and the parked commits' floors -- is exposed by
``snapshot``, so the loop walker can compare the state after two iterations
(``snapshot_recipe``) and, when one is the other advanced by a period,
advance the whole state by any number of periods at once
(``advance_snapshot`` + ``load_snapshot``) without walking the iterations in
between.  While ``trace`` is a list, every event also appends a
time-invariant tuple to it (its times relative to ``trace_base``), which is
how the walker recognises a repeating iteration.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from voyager_compiler.codegen.reporting.cost import dram_cycles, op_info
from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    LoopStats,
    OpInfo,
    TimingRecord,
)
from voyager_compiler.hardware_config import AcceleratorConfig

# Trace entry tags.
T_COMPUTE = 1
T_DMA = 2
T_POST = 3
T_COMMIT = 5
T_WAIT = 6
T_LOOP = 7
T_SKIP = 8

# The unit an ``OpInfo`` names -> the unit its commit waits are listed under.
_WAIT_UNIT = {"mma": "matrix", "vector": "vector"}


@dataclass
class _PendingCommit:
    """A committed op parked until every one of its ``waits`` -- ``(unit,
    counter, threshold)`` -- holds.  ``run`` walks its body (and adds to its
    counter) once that happens; ``floors[unit]`` is the running max of the
    cycle the program issued it at and the completion cycles the unit's
    resolved waits named, which that unit's passes start no earlier than."""

    node: object
    waits: list
    floors: dict
    run: object


@dataclass
class Checkpoint:
    """A copy of everything ``ResourceState.restore`` puts back."""

    clocks: tuple
    counter_posts: dict
    posts: dict
    commits: list
    flags: tuple
    counts: tuple
    bytes: tuple
    loop_stats: dict


class ResourceState:
    def __init__(self, cost: AcceleratorConfig, calibration=None):
        self.cost = cost
        self.calibration = calibration
        self.now = 0
        self.mma_free = 0
        self.vector_free = 0
        self.dram_free = 0
        self.cur_loop = -1  # id() of the while_loop currently being walked
        self.cur_kernel = ""  # the bufferized nest being walked
        self.kernel_groups: Dict[str, str] = {}
        self.launched: set = set()  # kernels charged their launch overhead
        self.loop_stats: Dict[int, LoopStats] = {}  # id(loop) -> stats
        self.read_bytes = 0
        self.write_bytes = 0
        # DRAM traffic (read + write) bucketed by tensor role.
        self.cat_bytes = {"weight": 0, "activation": 0, "kv": 0}
        self.records: List[TimingRecord] = []
        self.skips: List[LoopSkip] = []
        self.ops: Dict[int, OpInfo] = {}  # id(node) -> OpInfo
        self._op_names: set = set()
        # kernel -> how many of its ops get_op has numbered (the next slot)
        self._kernel_slots: Dict[str, int] = {}
        # counter -> the completion cycles of its posts, in order
        self.counter_posts: Dict[object, List[int]] = {}
        # id(async compute node) -> its end, handed to the following insert
        # so it can post the counter
        self.pending_post: Dict[int, int] = {}
        # committed (async) ops parked until their waits hold; each runs
        # (walks its body) only then.  See register_commit.
        self.pending_commits: List[_PendingCommit] = []
        self._resolving = False
        # set while a committed body is being walked: its ops floor their
        # start on commit_now (seeded to the issue cycle, advanced by each
        # body op) and on commit_floors[unit] (the cycle the unit's waits
        # resolved at), and occupy their unit(s) without advancing the
        # program clock.
        self.in_commit = False
        self.commit_now = 0
        self.commit_floors: Dict[str, int] = {}
        self.last_commit_node = None
        self.trace: Optional[list] = None
        self.trace_base = 0

    def _eid(self) -> int:
        return len(self.records)

    def _unit_free(self, unit: str) -> int:
        return self.mma_free if unit == "mma" else self.vector_free

    def _occupy(self, unit: str, end: int) -> None:
        if unit == "mma":
            self.mma_free = end
        else:
            self.vector_free = end

    def clock(self) -> int:
        """The clock a loop iteration is measured against: the program clock,
        or the commit clock while a committed body is being walked."""
        return self.commit_now if self.in_commit else self.now

    def get_op(self, node) -> OpInfo:
        """Memoize one ``OpInfo`` per static compute node (keyed by identity, so
        same-named nodes in different loop bodies stay distinct), numbered by
        its ``slot`` within the kernel before calibration prices it."""
        key = id(node)
        op = self.ops.get(key)
        if op is None:
            op = op_info(node, self.cost)
            name = op.key
            i = 2
            while op.key in self._op_names:
                op.key = f"{name}#{i}"
                i += 1
            self._op_names.add(op.key)
            op.kernel = self.cur_kernel
            slot = self._kernel_slots.get(self.cur_kernel, 0)
            op.slot = slot
            self._kernel_slots[self.cur_kernel] = slot + 1
            if self.calibration is not None:
                self.calibration.apply(
                    op, self.kernel_groups.get(self.cur_kernel)
                )
            self.ops[key] = op
        return op

    # -- event kinds --------------------------------------------------------

    def compute(self, node, path) -> TimingRecord:
        op = self.get_op(node)
        units_free = [self._unit_free(u) for u in op.units]
        if self.in_commit:
            # A committed op runs in program order on the datapath but off
            # the program clock: its start floors on the issue cycle
            # (commit_now, advanced by each body op), on the cycle its
            # unit(s)' waits resolved at, and on its unit(s), never on
            # self.now.  A fused op's vector pass works on the matrix output,
            # so the vector waits floor the end of its cycles, not the start.
            floors = []
            for u in op.units:
                floor = self.commit_floors[_WAIT_UNIT[u]]
                if u == "vector" and "mma" in op.units:
                    floor -= op.effective_cycles
                floors.append(floor)
            start = max(self.commit_now, *units_free, *floors)
        else:
            start = max(self.now, *units_free)
        end = start + op.effective_cycles
        done = end + op.completion_cycles
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="compute",
            resource=op.units,
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
            op_key=op.key,
        )
        self.records.append(rec)
        for u in op.units:
            self._occupy(u, end)
        if self.in_commit:
            # A committed op occupies its unit(s) but leaves the program clock;
            # its post carries the done.  A sync one advances the clock to it.
            self.commit_now = end
            self.last_commit_node = node
            self.pending_post[id(node)] = done
        else:
            self.now = done
        if self.trace is not None:
            base = self.trace_base
            self.trace.append(
                (T_COMPUTE, id(node), self.in_commit, start - base, end - base)
            )
        return rec

    def post_counter(self, key, src_node) -> None:
        """Add the async compute ``src_node``'s completion to counter
        ``key`` -- the compute-side twin of ``async_copy``'s post."""
        end = self.pending_post.pop(id(src_node), self.now)
        self.counter_posts.setdefault(key, []).append(end)

    def _dram_event(
        self, node, n_bytes: int, is_read: bool, category, path, sync: bool
    ) -> TimingRecord:
        start = max(self.now, self.dram_free)
        end = start + dram_cycles(n_bytes, self.cost)
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="load" if is_read else "store",
            resource=("dram",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
            bytes=n_bytes,
            is_read=is_read,
            category=category,
            sync=sync,
        )
        self.records.append(rec)
        self.dram_free = end
        if sync:
            self.now = end
        if is_read:
            self.read_bytes += n_bytes
        else:
            self.write_bytes += n_bytes
        if category in self.cat_bytes:
            self.cat_bytes[category] += n_bytes
        if self.trace is not None:
            base = self.trace_base
            self.trace.append(
                (T_DMA, id(node), n_bytes, is_read, start - base, end - base)
            )
        return rec

    def async_copy(
        self, node, n_bytes: int, is_load: bool, key, path, category=""
    ) -> TimingRecord:
        # A DMA occupies DRAM but does NOT advance the program clock: the
        # waits on its counter are what reconcile it.
        rec = self._dram_event(
            node, n_bytes, is_load, category, path, sync=False
        )
        self.counter_posts.setdefault(key, []).append(rec.end)
        if self.trace is not None:
            self.trace.append((T_POST, key))
        self._resolve_commits()
        return rec

    def _resolved(self, key, threshold) -> Optional[int]:
        """The cycle counter ``key`` reaches ``threshold``, or ``None`` while
        it has not been posted that often yet."""
        posts = self.counter_posts.get(key, ())
        if threshold > len(posts):
            return None
        return posts[threshold - 1] if threshold >= 1 else 0

    def register_commit(self, node, waits, run) -> None:
        """Park a committed op: record its waits (``(unit, counter,
        threshold)`` each) and the ``run`` thunk that walks its body, then
        try to resolve it at once.  In the common case every wait already
        holds (the copy_in that feeds it ran just before), so it runs
        immediately; otherwise it waits for the posts.  Either way it starts
        no earlier than the clock it is issued at.
        """
        issued = self.clock()
        floors = {unit: issued for unit in _WAIT_UNIT.values()}
        self.pending_commits.append(
            _PendingCommit(node, list(waits), floors, run)
        )
        if self.trace is not None:
            self.trace.append(
                (T_COMMIT, id(node), tuple((u, k) for u, k, _ in waits))
            )
        self._resolve_commits()

    def _resolve_commits(self) -> None:
        """Run every committed op whose waits now all hold, in dispatch
        order: each resolved wait's completion floors the start of its
        unit's passes, and once a commit has no wait left, its body runs off
        the program clock.  A run adds to its counter, which may in turn
        resolve the commit after it, so the scan repeats until it makes no
        progress; a commit stays parked behind the ones dispatched before
        it, as the units' queues keep it."""
        if self._resolving:
            return
        self._resolving = True
        try:
            while self.pending_commits:
                pc = self.pending_commits[0]
                unresolved = []
                for unit, key, threshold in pc.waits:
                    ready = self._resolved(key, threshold)
                    if ready is None:
                        unresolved.append((unit, key, threshold))
                    else:
                        pc.floors[unit] = max(pc.floors[unit], ready)
                pc.waits = unresolved
                if unresolved:
                    break
                self.pending_commits.pop(0)
                self.in_commit = True
                self.commit_now = min(pc.floors.values())
                self.commit_floors = pc.floors
                self.last_commit_node = None
                pc.run()
                self.in_commit = False
        finally:
            self._resolving = False

    def assert_commits_drained(self, where: str) -> None:
        """At graph end every committed op must have resolved; one still parked
        means a counter it waits on never reaches its threshold -- a deadlock
        in the emitted schedule.  This is *not* checked at each async_wait: a
        committed tile legitimately stays in flight across the retire of
        another (an async_wait blocks on its own counter, not on every
        commit)."""
        self._resolve_commits()
        if self.pending_commits:
            raise ValueError(
                f"unresolved committed op at {where}: a counter it waits on "
                f"never reaches its threshold -- the emitted schedule would "
                f"deadlock"
            )

    def dram_copy(self, node, reads, write, path) -> TimingRecord:
        """A DRAM->DRAM materialization (``pad`` / ``expand`` / ``cat``): read
        every input, then write the output.  Each ``(bytes, category)`` side is
        sized from its own tensor, so a copy that reads a weight and writes an
        activation lands in both buckets.

        It carries no counter, so nothing can wait on it and it cannot be
        overlapped -- it is synchronous and stalls the program clock.
        """
        for n_bytes, category in reads:
            self._dram_event(node, n_bytes, True, category, path, sync=True)
        n_bytes, category = write
        return self._dram_event(node, n_bytes, False, category, path, sync=True)

    def async_wait(self, node, key, threshold, path) -> TimingRecord:
        """The program clock blocks until counter ``key`` reaches
        ``threshold``.  Every post that can still come is from a parked
        commit, so those are resolved first; a threshold still out of reach
        is a deadlock in the emitted schedule."""
        self._resolve_commits()
        completion = self._resolved(key, threshold)
        if completion is None:
            raise ValueError(
                f"async_wait {node.name} waits for a counter to reach "
                f"{threshold}, which nothing issued so far posts -- the "
                f"emitted schedule would deadlock"
            )
        start = self.now
        end = max(self.now, completion)
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="async_wait",
            resource=("control",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
        )
        self.records.append(rec)
        self.now = end
        if self.trace is not None:
            base = self.trace_base
            self.trace.append((T_WAIT, id(node), key, start - base, end - base))
        return rec

    def launch(self, kernel: str) -> None:
        """Charge a calibrated kernel's launch overhead once, as a synchronous
        control event at its entry."""
        if self.calibration is None or kernel in self.launched:
            return
        self.launched.add(kernel)
        cycles = self.calibration.launch_cycles(self.kernel_groups.get(kernel))
        if not cycles:
            return
        rec = TimingRecord(
            eid=self._eid(),
            node_name=kernel,
            kind="launch",
            resource=("control",),
            start=self.now,
            end=self.now + cycles,
            kernel=kernel,
        )
        self.records.append(rec)
        self.now = rec.end

    # -- loops ----------------------------------------------------------------

    def enter_loop(self, node, trip_count: int) -> LoopStats:
        stats = self.loop_stats.get(id(node))
        if stats is None:
            stats = LoopStats(
                loop_uid=id(node),
                name=node.name,
                kernel=self.cur_kernel,
                trip_count=trip_count,
                start=self.clock(),
            )
            self.loop_stats[id(node)] = stats
        stats.entries += 1
        return stats

    def exit_loop(self, stats: LoopStats) -> None:
        stats.end = self.clock()

    # -- steady-state folding -------------------------------------------------

    def counters(self) -> Tuple[int, ...]:
        """The additive totals a skipped period multiplies: DRAM bytes by
        direction and category."""
        return (
            self.read_bytes,
            self.write_bytes,
            self.cat_bytes["weight"],
            self.cat_bytes["activation"],
            self.cat_bytes["kv"],
        )

    def add_counters(self, delta: Tuple[int, ...], times: int) -> None:
        self.read_bytes += delta[0] * times
        self.write_bytes += delta[1] * times
        self.cat_bytes["weight"] += delta[2] * times
        self.cat_bytes["activation"] += delta[3] * times
        self.cat_bytes["kv"] += delta[4] * times

    def snapshot(self) -> tuple:
        """Every timestamp the state holds, as nested tuples that compare by
        value: ``(in_commit, clocks, counters, posts, commits)`` with the
        counters and posts sorted by key; a parked commit's floors are
        listed by unit."""
        return (
            self.in_commit,
            (
                self.now,
                self.commit_now,
                self.mma_free,
                self.vector_free,
                self.dram_free,
            ),
            tuple(
                (k, tuple(self.counter_posts[k]))
                for k in sorted(self.counter_posts)
            ),
            tuple((k, self.pending_post[k]) for k in sorted(self.pending_post)),
            tuple(
                (
                    id(pc.node),
                    tuple(pc.waits),
                    tuple(pc.floors[u] for u in _WAIT_UNIT.values()),
                )
                for pc in self.pending_commits
            ),
        )

    def load_snapshot(self, snap: tuple) -> None:
        """Write a snapshot of the same shape as the current state back."""
        in_commit, clocks, counters, posts, commits = snap
        self.in_commit = in_commit
        (
            self.now,
            self.commit_now,
            self.mma_free,
            self.vector_free,
            self.dram_free,
        ) = clocks
        for k, values in counters:
            self.counter_posts[k] = list(values)
        for k, value in posts:
            self.pending_post[k] = value
        for pc, (_, _, floors) in zip(self.pending_commits, commits):
            pc.floors = dict(zip(_WAIT_UNIT.values(), floors))

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            clocks=(
                self.now,
                self.commit_now,
                self.mma_free,
                self.vector_free,
                self.dram_free,
            ),
            counter_posts={k: list(v) for k, v in self.counter_posts.items()},
            posts=dict(self.pending_post),
            commits=[
                _PendingCommit(pc.node, list(pc.waits), dict(pc.floors), pc.run)
                for pc in self.pending_commits
            ],
            flags=(
                self.in_commit,
                self.last_commit_node,
                self.cur_loop,
                self.cur_kernel,
            ),
            counts=(len(self.records), len(self.skips)),
            bytes=(self.read_bytes, self.write_bytes, dict(self.cat_bytes)),
            loop_stats={k: copy.copy(v) for k, v in self.loop_stats.items()},
        )

    def restore(self, cp: Checkpoint) -> None:
        """Put a checkpoint back.  ``loop_stats`` entries are restored in
        place, so a walker holding one keeps a live reference."""
        (
            self.now,
            self.commit_now,
            self.mma_free,
            self.vector_free,
            self.dram_free,
        ) = cp.clocks
        self.counter_posts = {k: list(v) for k, v in cp.counter_posts.items()}
        self.pending_post = dict(cp.posts)
        self.pending_commits = [
            _PendingCommit(pc.node, list(pc.waits), dict(pc.floors), pc.run)
            for pc in cp.commits
        ]
        (
            self.in_commit,
            self.last_commit_node,
            self.cur_loop,
            self.cur_kernel,
        ) = cp.flags
        del self.records[cp.counts[0] :]
        del self.skips[cp.counts[1] :]
        self.read_bytes, self.write_bytes = cp.bytes[0], cp.bytes[1]
        self.cat_bytes = dict(cp.bytes[2])
        for k in list(self.loop_stats):
            if k not in cp.loop_stats:
                del self.loop_stats[k]
        for k, saved in cp.loop_stats.items():
            self.loop_stats[k].__dict__.update(saved.__dict__)


@dataclass(frozen=True)
class Recipe:
    """How one period advances the state: which timestamps move by
    ``shift`` (the rest stay: an idle unit), and how many posts each
    counter gains -- the period's posts are the previous period's, one
    ``shift`` later, so a counter extends by its last ``added`` entries
    shifted, period after period.
    """

    shift: int
    clocks: Tuple[bool, ...]
    counters: Tuple[int, ...]  # posts added per period
    posts: Tuple[bool, ...]
    commits: Tuple[Tuple[bool, ...], ...]  # per commit, its floors


def _moving(a: int, b: int, shift: int) -> Optional[bool]:
    if b - a == shift:
        return True
    if b == a:
        return False
    return None


def _moving_all(a, b, shift) -> Optional[Tuple[bool, ...]]:
    flags = []
    for va, vb in zip(a, b):
        m = _moving(va, vb, shift)
        if m is None:
            return None
        flags.append(m)
    return tuple(flags)


def snapshot_recipe(a: tuple, b: tuple, shift: int) -> Optional[Recipe]:
    """The recipe taking snapshot ``a`` to ``b`` one period of ``shift``
    cycles later, or ``None`` when ``b`` is not ``a`` advanced by a period.

    A counter only grows: ``b`` keeps every post of ``a`` and adds the ones
    the period made, each ``shift`` after the matching post of the period
    before.
    """
    if a[0] != b[0]:
        return None
    clocks = _moving_all(a[1], b[1], shift)
    if clocks is None:
        return None
    if len(a[2]) != len(b[2]):
        return None
    counters = []
    for (ka, va), (kb, vb) in zip(a[2], b[2]):
        added = len(vb) - len(va)
        if ka != kb or added < 0 or added > len(va) or vb[: len(va)] != va:
            return None
        previous = va[len(va) - added :] if added else ()
        if any(y - x != shift for x, y in zip(previous, vb[len(va) :])):
            return None
        counters.append(added)
    if [k for k, _ in a[3]] != [k for k, _ in b[3]]:
        return None
    posts = _moving_all([v for _, v in a[3]], [v for _, v in b[3]], shift)
    if posts is None:
        return None
    if [c[:2] for c in a[4]] != [c[:2] for c in b[4]]:
        return None
    commits = []
    for ca, cb in zip(a[4], b[4]):
        moving = _moving_all(ca[2], cb[2], shift)
        if moving is None:
            return None
        commits.append(moving)
    return Recipe(shift, clocks, tuple(counters), posts, tuple(commits))


def advance_snapshot(snap: tuple, recipe: Recipe, periods: int):
    """``snap`` after ``periods`` more periods under ``recipe``."""
    total = periods * recipe.shift

    def move(values, flags):
        return tuple(v + total if m else v for v, m in zip(values, flags))

    counters = []
    for (k, values), added in zip(snap[2], recipe.counters):
        last = values[len(values) - added :] if added else ()
        grown = list(values)
        for period in range(1, periods + 1):
            grown.extend(v + period * recipe.shift for v in last)
        counters.append((k, tuple(grown)))
    return (
        snap[0],
        move(snap[1], recipe.clocks),
        tuple(counters),
        tuple(
            (k, v + total if m else v)
            for (k, v), m in zip(snap[3], recipe.posts)
        ),
        tuple(
            (n, waits, move(floors, m))
            for (n, waits, floors), m in zip(snap[4], recipe.commits)
        ),
    )
