"""Data structures shared across the reporting stages.

The estimator walks a bufferized FX graph once (``interpret`` +
``scheduler``), folding each loop's steady state as it goes, and the
reporting stage (``excel`` / ``perfetto`` / ``calibration``) reads the
result.  These dataclasses are the contract between them.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from voyager_compiler.codegen.transform.tiling.runtime_model import (
    PassTiming,
)
from voyager_compiler.hardware_config import AcceleratorConfig

# The lanes a compute pass can occupy: the systolic array, the matrix-vector
# unit, the SpMM unit and the vector unit.
COMPUTE_UNITS = ("mma", "mvm", "spmm", "vector")


@dataclass
class TimingRecord:
    """One scheduled execution of one FX node (a loop body node runs many
    times, so one node yields many records, tagged by ``iteration_path``).

    ``resource`` is the lane the event occupies.  A compute pass holds one
    of ``COMPUTE_UNITS``: an op with a matrix-class pass and a vector pass
    (a fused GEMM / conv, whose tail runs on the VU) yields one record per
    pass, the vector one marked ``rides``; a DMA holds ``("dram",)``; an
    ``async_wait`` holds ``("control",)`` -- a synchronization that uses no
    bandwidth.  A compute pass is stamped on its unit's timeline: ``ready``
    when its sync waits hold, ``start`` and ``end`` its compute span, and
    ``done`` when it is finished (a pass posts its counter at its done);
    other events leave ``ready`` and ``done`` at ``start`` and ``end``.
    ``kernel`` names the bufferized nest the event belongs to, ``op_key``
    the ``OpInfo`` a compute event was priced by.  ``sync`` marks a DRAM
    event the program clock waits on -- the whole-tensor materialization of
    a ``pad`` / ``permute`` / ``cat`` / ``expand``, which cannot overlap --
    as against a tile ``async_copy``, which can.
    """

    eid: int
    node_name: str
    kind: str  # compute | load | store | async_wait | launch
    resource: Tuple[str, ...]  # one of COMPUTE_UNITS, dram or control
    start: int
    end: int
    iteration_path: Tuple[int, ...] = ()
    loop_uid: int = -1  # id() of the enclosing while_loop (-1 = top level)
    kernel: str = ""
    bytes: int = 0  # DRAM ops only
    is_read: bool = False  # DRAM loads (vs stores)
    category: str = ""
    op_key: str = ""
    sync: bool = False  # DRAM: a materialization that stalls the program clock
    ready: int = -1
    done: int = -1
    # A fused op's vector pass, derived from the matrix-class pass it rides.
    rides: bool = False

    def __post_init__(self):
        if self.ready < 0:
            self.ready = self.start
        if self.done < 0:
            self.done = self.end


@dataclass
class OpInfo:
    """One static compute node (a row of the Operations sheet).  Many
    ``TimingRecord``s with the same ``key`` reference one ``OpInfo``.

    ``utilization`` is the fraction of peak the op sustains, so it costs
    ``ceil(ideal_cycles / utilization)`` cycles.  It is compute-only (DRAM
    is modeled separately, as ``async_copy`` events) and pre-computed by
    ``cost.py``.  A calibrated op carries the RTL-derived ``measured_cycles``
    instead, with ``calibration`` set to ``per-op``: the op's own RTL
    runtime priced it.

    The op runs as the passes the sync pass gives it: a matrix-class pass on
    ``engine`` (``mma``, ``mvm`` or ``spmm``; blank for none) and a vector
    pass when ``vector_pass``; ``posts_on`` names the one that posts
    (``matrix`` or ``vector``), and ``spmm`` marks a GEMM whose matrix-class
    pass runs on the SpMM unit beside the array.  ``timing`` is the tiler's
    ``PassTiming`` of a tiled matrix op, and ``measured_terms`` the pass
    terms RTL stamps measured (``calibration`` ``per-pass``).

    ``slot`` is the op's 0-based ordinal within its kernel, in the order
    the scheduler first walks the kernel's ops.  Kernels built from one
    nest walk their ops in the same order, so ``(build_key, slot)`` names
    the same op in every one of them; that is what a per-op measurement is
    keyed by.
    """

    key: str
    op_type: str  # gemm | conv | vector
    ideal_cycles: int
    detail: dict = field(default_factory=dict)
    units: Tuple[str, ...] = ("vector",)
    utilization: float = 1.0
    kernel: str = ""
    slot: int = -1
    measured_cycles: Optional[int] = None
    calibration: str = ""
    engine: str = ""
    vector_pass: bool = True
    posts_on: str = "vector"
    spmm: bool = False
    timing: Optional[PassTiming] = None
    measured_terms: Optional[Tuple[int, int, int]] = None

    @property
    def pass_terms(self) -> Tuple[int, int, int]:
        """``(fill, span, tail)`` of the op's main pass (its matrix-class
        pass, else its vector pass): a cold pass computes ``fill`` after it
        is ready, computes for ``span`` and is done ``tail`` after that.
        The measured terms when calibrated per pass, else the tiler's
        ``PassTiming``; an op priced per op or without one computes for its
        ``effective_cycles``, with no fill and no tail."""
        if self.measured_terms is not None:
            return self.measured_terms
        if self.timing is None or self.measured_cycles is not None:
            return 0, self.effective_cycles, 0
        t = self.timing
        return (
            math.ceil(t.buffer_fill),
            math.ceil(t.steady),
            math.ceil(t.skew + t.drain),
        )

    @property
    def analytic_cycles(self) -> int:
        """The cost model's price: ideal cycles stretched by utilization."""
        return math.ceil(self.ideal_cycles / self.utilization)

    @property
    def effective_cycles(self) -> int:
        """The op's charged cost: the measurement when calibrated, else the
        analytic price."""
        if self.measured_cycles is not None:
            return self.measured_cycles
        return self.analytic_cycles


@dataclass
class LoopSkip:
    """A run of loop iterations the walk did not execute: ``repeats``
    periods of ``period`` iterations, each a time-shift by ``shift`` cycles
    of the ``template`` records (the last walked period), starting at
    iteration ``first_step`` and cycle ``start``.  ``bytes`` is one
    period's DRAM traffic by counter (``read`` / ``write`` / a category).
    """

    loop_uid: int
    kernel: str
    after_eid: int  # last walked record before the skip (-1 = none)
    first_step: int
    period: int
    repeats: int
    template: Tuple[int, ...]  # eids of the template period's records
    shift: int
    start: int
    bytes: Dict[str, int] = field(default_factory=dict)

    @property
    def iterations(self) -> int:
        return self.period * self.repeats

    @property
    def end(self) -> int:
        return self.start + self.repeats * self.shift


@dataclass
class LoopStats:
    """What the walk did with one ``while_loop``: how many iterations it
    walked and skipped, the period it found and how long a period takes.
    A nested loop is entered once per walked outer iteration; the counts
    accumulate over every entry."""

    loop_uid: int
    name: str
    kernel: str
    trip_count: int
    entries: int = 0
    walked: int = 0
    skipped: int = 0
    period: int = 0  # 0 = no period found
    shift: int = 0  # cycles per period
    start: int = 0  # clock at the first entry
    end: int = 0  # clock at the last exit


@dataclass
class ScheduleResult:
    """Everything the reporting stage needs.

    ``records`` holds the walked events; ``skips`` the folded steady-state
    runs between them, so the two together describe the whole schedule.
    ``busy_*`` are exact union lengths of the compute, DRAM and either
    lanes' busy intervals over the whole makespan.  ``kernel_groups``
    maps each kernel to its ``calibration.KernelSignature``.
    """

    records: List[TimingRecord]
    ops: List[OpInfo]
    total_latency: int
    dram_read_bytes: int
    dram_write_bytes: int
    cost: AcceleratorConfig
    dram_weight_bytes: int = 0
    dram_activation_bytes: int = 0
    dram_kv_bytes: int = 0
    skips: List[LoopSkip] = field(default_factory=list)
    loops: List[LoopStats] = field(default_factory=list)
    busy_compute: int = 0
    busy_dram: int = 0
    busy_any: int = 0
    kernel_groups: Dict[str, object] = field(default_factory=dict)
