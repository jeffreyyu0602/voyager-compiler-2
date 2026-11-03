"""Per-kernel view of a schedule.

A kernel is one bufferized nest (see ``calibration``); ``kernel_rows``
folds a ``ScheduleResult``'s walked records, skips and loop stats into one
row per kernel -- its span, DRAM traffic, how much of its main loop was
walked versus folded, what one steady-state period runs and costs, and
whether the kernel is bound by compute or by DMA.  Both the workbook's
Kernels sheet and the calibration form are written from these rows.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    OpInfo,
    ScheduleResult,
)


@dataclass
class KernelRow:
    kernel: str
    anchor: str = ""
    group: str = ""  # calibration identity: meta['build_key']
    group_id: int = -1  # meta['build_group'], readable, per-compile
    num_shared_kernels: int = 0  # kernels sharing this build key
    start: int = 0
    end: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    trip_count: int = 0
    walked: int = 0
    skipped: int = 0
    period: int = 0
    repeats: int = 0
    period_cycles: int = 0
    compute_per_period: int = 0
    dram_per_period: int = 0
    ops_per_period: Dict[str, int] = field(default_factory=dict)
    analytic_per_iteration: float = 0.0
    calibration: str = ""
    measured_per_iteration: Optional[float] = None

    @property
    def span(self) -> int:
        return self.end - self.start

    @property
    def bound(self) -> str:
        if not self.period_cycles:
            return ""
        if self.compute_per_period >= self.dram_per_period:
            return "compute"
        return "dma"

    @property
    def ops_text(self) -> str:
        return " ".join(
            f"{k}x{v}" if v != 1 else k
            for k, v in sorted(self.ops_per_period.items())
        )


def _busy_cycles(recs) -> int:
    """Cycles at least one of ``recs`` is running: the union of their
    intervals, so ops overlapping on different units count once."""
    busy, end = 0, None
    for r in sorted(recs, key=lambda r: r.start):
        if end is None or r.start >= end:
            busy += r.end - r.start
            end = r.end
        elif r.end > end:
            busy += r.end - end
            end = r.end
    return busy


# The order a kernel's mixed calibration labels are joined in, finest first.
_LABEL_ORDER = ("per-pass", "per-op", "analytic")


def _calibration_label(period_ops: List[OpInfo]) -> str:
    """How a kernel's period was priced: one label when its ops agree, else
    the labels joined (``per-op+analytic`` when some ops had their own
    measurement and the rest kept the analytic price)."""
    present = {op.calibration or "analytic" for op in period_ops}
    labels = [label for label in _LABEL_ORDER if label in present]
    return "+".join(labels)


def _main_skip(skips: List[LoopSkip]) -> Optional[LoopSkip]:
    """The fold covering the most iterations: the kernel's steady state."""
    if not skips:
        return None
    return max(skips, key=lambda s: s.iterations)


def kernel_rows(result: ScheduleResult) -> List[KernelRow]:
    rows: Dict[str, KernelRow] = {}
    ops: Dict[str, OpInfo] = {op.key: op for op in result.ops}
    by_kernel: Dict[str, list] = {}
    for rec in result.records:
        by_kernel.setdefault(rec.kernel, []).append(rec)
    skips_by_kernel: Dict[str, list] = {}
    for skip in result.skips:
        skips_by_kernel.setdefault(skip.kernel, []).append(skip)
    loops_by_kernel: Dict[str, list] = {}
    for stats in result.loops:
        loops_by_kernel.setdefault(stats.kernel, []).append(stats)

    for kernel, recs in by_kernel.items():
        grp = result.kernel_groups.get(kernel)
        row = KernelRow(
            kernel=kernel,
            anchor=grp.anchor if grp else "",
            group=grp.key if grp else "",
            group_id=grp.group if grp else -1,
            num_shared_kernels=grp.num_shared_kernels if grp else 0,
            start=min(r.start for r in recs),
            end=max(r.done for r in recs),
        )
        skips = skips_by_kernel.get(kernel, [])
        for skip in skips:
            row.end = max(row.end, skip.end)
            row.read_bytes += skip.bytes.get("read", 0) * skip.repeats
            row.write_bytes += skip.bytes.get("write", 0) * skip.repeats
        for r in recs:
            if r.is_read:
                row.read_bytes += r.bytes
            elif r.bytes:
                row.write_bytes += r.bytes
        loops = loops_by_kernel.get(kernel, [])
        if loops:
            main = max(loops, key=lambda s: s.trip_count)
            row.trip_count = main.trip_count
            row.walked = main.walked
            row.skipped = main.skipped
            row.period = main.period
            row.period_cycles = main.shift
        skip = _main_skip(skips)
        if skip is not None:
            template = [result.records[e] for e in skip.template]
            row.repeats = sum(s.repeats for s in skips)
            # A fused op's riding vector pass is part of its matrix-class
            # pass's invocation, not one of its own.
            compute = [
                r for r in template if r.kind == "compute" and not r.rides
            ]
            row.compute_per_period = sum(r.end - r.start for r in compute)
            row.dram_per_period = sum(
                r.end - r.start for r in template if r.resource == ("dram",)
            )
            iterations = skip.period
        else:
            loop_recs = [r for r in recs if r.loop_uid != -1]
            compute = [
                r for r in loop_recs if r.kind == "compute" and not r.rides
            ]
            iterations = max(1, row.walked)
        counts = Counter(r.op_key for r in compute)
        row.ops_per_period = {
            k: v / iterations if v % iterations else v // iterations
            for k, v in counts.items()
        }
        # Per iteration, the time the schedule spends computing, overlaps
        # counted once.  Analytic cyc/iter scales it back by analytic over
        # effective, an effective-weighted approximation under per-op
        # prices.
        busy = _busy_cycles(compute)
        analytic = sum(ops[k].analytic_cycles * v for k, v in counts.items())
        effective = sum(ops[k].effective_cycles * v for k, v in counts.items())
        row.analytic_per_iteration = (
            busy * analytic / effective / iterations if effective else 0.0
        )
        measured = [
            ops[k]
            for k in counts
            if ops[k].measured_cycles is not None
            or ops[k].measured_terms is not None
        ]
        if measured:
            # The busy time comes out of the schedule, so it is the measured
            # figure under per-op prices.
            period_ops = [ops[k] for k in counts]
            row.calibration = _calibration_label(period_ops)
            row.measured_per_iteration = busy / iterations
        rows[kernel] = row
    return sorted(rows.values(), key=lambda r: r.start)


def coverage(rows: List[KernelRow], total_latency: int) -> float:
    """The share of the makespan spent in calibrated kernels."""
    if not total_latency:
        return 0.0
    covered = sum(r.span for r in rows if r.calibration)
    return min(1.0, covered / total_latency)
