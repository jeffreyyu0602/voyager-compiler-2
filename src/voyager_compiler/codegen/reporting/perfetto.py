"""Chrome / Perfetto trace export (``chrome://tracing`` JSON).

Emits every walked event: each compute pass on its unit's track -- Matrix
(systolic array), MVM (matrix-vector unit), SpMM, Vector -- the DRAM
interface on another, control (waits) on another.  A compute pass's bar is
its compute span; its ``args`` carry its stamps (``ready``, ``start``,
``end``, ``done``).  A fused op (a GEMM + elementwise epilogue) draws a
matrix-class pass and the vector pass that rides it, which spans the
matrix pass's start to its done.  A folded steady state -- iterations the
walk skipped as repeats of the period before them -- appears as one bar on
the Control track spanning the skipped time, so the trace stays readable
at any trip count.  Time is in cycles (the trace ``ts``/``dur`` unit is
arbitrary).
"""

import json
from typing import Dict

from voyager_compiler.codegen.reporting.model import ScheduleResult

_TID = {"mma": 0, "vector": 1, "dram": 2, "control": 3, "mvm": 4, "spmm": 5}
_TRACK = {
    0: "Matrix",
    1: "Vector",
    2: "DRAM",
    3: "Control",
    4: "MVM",
    5: "SpMM",
}


def perfetto_dict(result: ScheduleResult) -> Dict:
    events = []
    for tid, name in _TRACK.items():
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {"name": name},
            }
        )
    for r in result.records:
        for lane in r.resource:
            events.append(
                {
                    "name": f"{r.node_name} {r.iteration_path}",
                    "cat": r.kind,
                    "ph": "X",
                    "ts": r.start,
                    "dur": max(0, r.end - r.start),
                    "pid": 0,
                    "tid": _TID.get(lane, _TID["control"]),
                    "args": {
                        "eid": r.eid,
                        "kind": r.kind,
                        "kernel": r.kernel,
                        "bytes": r.bytes,
                        "is_read": r.is_read,
                        "iteration": list(r.iteration_path),
                        "ready": r.ready,
                        "start": r.start,
                        "end": r.end,
                        "done": r.done,
                        "rides": r.rides,
                    },
                }
            )
    for s in result.skips:
        events.append(
            {
                "name": f"{s.kernel} x{s.repeats} periods folded",
                "cat": "fold",
                "ph": "X",
                "ts": s.start,
                "dur": s.end - s.start,
                "pid": 0,
                "tid": _TID["control"],
                "args": {
                    "kernel": s.kernel,
                    "first_step": s.first_step,
                    "iterations": s.iterations,
                    "period": s.period,
                    "repeats": s.repeats,
                    "cycles_per_period": s.shift,
                },
            }
        )
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def write_perfetto(result: ScheduleResult, path: str) -> str:
    """Write the trace JSON to ``path`` (open it in ``chrome://tracing`` or
    ``ui.perfetto.dev``) and return the path."""
    with open(path, "w") as f:
        json.dump(perfetto_dict(result), f)
    return path
