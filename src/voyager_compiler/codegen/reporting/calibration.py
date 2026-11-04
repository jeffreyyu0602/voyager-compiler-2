"""Kernel build groups and RTL calibration of the compute cost model.

A **kernel** is one bufferized nest: the top-level nodes sharing a
``meta['scope']``, with the loops and committed bodies under them.  Two
kernels are the *same operation* exactly when bufferization built them from
the same cached nest.  It records that twice:

* ``meta['build_key']`` -- a digest of the build cache's own key, which is
  the node's structural + shape/dtype signature.  Content addressed, so the
  same nest gets the same key in a two-layer compile and in the full model;
  this is what carries an RTL measurement between them.
* ``meta['build_group']`` -- a counter, readable and close to the ``group``
  column of ``layers.txt``, but meaningless outside its own compile.

The sheet is keyed by the first and shows the second.

Calibration replaces the *compute* term of the estimate and nothing else,
measured in RTL with no DRAM model: the analytic DMA cost and the
scheduler's overlap stay.  The ``Op calibration`` sheet gives one op's own
steady-state cycles per invocation, keyed by build key and the op's
``slot`` in its kernel, and ``Calibration.apply`` prices the op at exactly
that; the scheduler then works out how the kernel's ops overlap.  The build
key does not cover tiling, so a row also carries the shapes and the tiling
(scratch slots and nest) it was measured at; an op shaped or tiled
otherwise, or with no row, keeps its analytic price.  A kernel's launch
term, charged once per kernel, comes from its op rows when it has any (the
cold first invocation over a steady one), else from the ``Calibration``
sheet, whose per-group cycles per iteration are reported beside the
estimate.

The ``Pass calibration`` sheet gives one op's pass terms, measured from the
RTL's per-pass stamps of its unit (``measure_pass_terms``): its compute
span, its cold fill and its tail, keyed and guarded as an ``Op
calibration`` row.  The scheduler places the op's passes with them in
place of the tiler's ``steady``, ``buffer_fill`` and ``skew + drain``; a
pass row takes precedence over an op row for the same op.

The same sheet carries the group's measured **power**, which corrects
nothing: the compiler has no power model, and the number exists only so the
reporting workbook can turn a latency into an energy.
"""

import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from torch.fx import GraphModule

from voyager_compiler.codegen.reporting.model import OpInfo

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"layers_(\d+)_")


@dataclass
class KernelGroup:
    """One kernel's build group: ``key`` is what the calibration sheet is
    keyed by, ``members`` every kernel bufferization built the same way."""

    key: str  # the calibration identity: build_key, else "k:<kernel>"
    group: int  # build_group, readable but per-compile
    anchor: str = ""
    members: List[str] = field(default_factory=list)

    @property
    def num_shared_kernels(self) -> int:
        """Kernels built from this nest, counting repeats inside a layer as
        well as across layers -- q, k and v in every layer is 3 x n_layers."""
        return len(self.members)


def kernel_groups(model: GraphModule) -> Dict[str, KernelGroup]:
    """Every kernel of ``model``, keyed by kernel name, carrying the build
    group bufferization stamped on its nodes.

    A kernel whose nodes carry no ``build_group`` (bufferization never built
    it) is its own singleton group, keyed by name, so it still gets a row.
    """
    groups: Dict[str, KernelGroup] = {}
    out: Dict[str, KernelGroup] = {}
    for node in model.graph.nodes:
        scope = node.meta.get("scope")
        if scope is not None:
            kernel, anchor = scope[0], str(scope[1])
        elif node.op != "placeholder" and node.meta.get("build_key"):
            # Bufferization built no nest here, so there is no scope; the node
            # is its own kernel, named and anchored as the graph names it.
            kernel, anchor = node.name, str(node.target)
        else:
            continue
        if kernel in out:
            continue
        group = node.meta.get("build_group")
        # build_key is content addressed, so it identifies the same nest in
        # any compile; build_group is a per-run counter and only groups
        # within this one.
        key = node.meta.get("build_key") or (
            f"g{group}" if group is not None else f"k:{kernel}"
        )
        entry = groups.get(key)
        if entry is None:
            entry = groups[key] = KernelGroup(
                key=key,
                group=group if group is not None else -1,
                anchor=anchor,
            )
        entry.members.append(kernel)
        out[kernel] = entry
    return out


@dataclass
class Measurement:
    """One calibrated group: the cycles one steady-state iteration took in
    RTL against the analytic cycles the form reported for it, plus the
    silicon power the group draws (0 when unmeasured)."""

    measured_per_iteration: float
    analytic_per_iteration: float
    launch_cycles: int = 0
    power: float = 0.0  # W, measured on silicon; not a correction
    # The cells it was read from, so a calibrated compile can write the
    # measurement back onto its own sheet.
    n1: float = 0.0
    t1: float = 0.0
    n2: float = 0.0
    t2: float = 0.0


@dataclass
class OpMeasurement:
    """One op's own RTL runtime: ``cycles`` per invocation in steady state,
    for the op at ``slot`` of every kernel built with ``key``.  The other
    fields say where the number came from and are carried, not used."""

    key: str
    slot: int
    cycles: float
    kernel: str = ""
    op: str = ""
    unit: str = ""
    passes: int = 0
    samples: int = 0
    method: str = ""  # how fill-ops priced it, e.g. service-time
    source: str = ""  # the regression_results folder it was read from
    shapes: str = ""  # op_shapes of the op measured; blank matches any
    tiling: str = ""  # cost.tiling_text of the op measured
    # The kernel's cold first invocation over a steady one, on the row of
    # the op that invocation is; 0 on the others.
    launch_cycles: int = 0


@dataclass
class PassMeasurement:
    """One op's pass terms measured in RTL (``measure_pass_terms``), for the
    op at ``slot`` of every kernel built with ``key``, measured at
    ``shapes`` and ``tiling``: ``fill``, ``span`` and ``tail`` stand in for
    the tiler's ``buffer_fill``, ``steady`` and ``skew + drain``.  The other
    fields say where the terms came from and are carried, not used."""

    key: str
    slot: int
    fill: int
    span: int
    tail: int
    kernel: str = ""
    op: str = ""
    passes: int = 0
    source: str = ""
    shapes: str = ""
    tiling: str = ""


def measure_pass_terms(stamps) -> Dict[object, Tuple[int, int, int]]:
    """Each op's ``(fill, span, tail)`` from the RTL stamps of the passes
    of one unit.

    A pass is warm when it was ready before its predecessor's compute end,
    so it computes right after it; the others, the first included, are
    cold.  An op's span is the median of ``E(n) - E(n-1)`` over its warm
    passes (the handoff between passes included), its fill the median of
    ``E(n) - R(n) - span`` over its cold ones, and its tail the median of
    ``D(n) - E(n)`` over its passes whose done is exposed: the unit's last
    pass and every one a cold pass follows (the output latency after the
    skew and drain included).  An op with no warm or no cold pass takes the
    span or the fill of all the unit's passes, and one with no exposed pass
    the median tail of its own passes.

    Args:
        stamps: ``(op, R, E, D)`` per pass, in issue order on the unit:
            the op the pass runs (any hashable label, e.g. its slot), its
            start handshake, its last row into the array and its done.

    Returns:
        Each op's ``(fill, span, tail)``.

    Raises:
        ValueError: when no pass of the unit is warm, so no span is
            measurable.
    """
    stamps = list(stamps)
    warm, cold, exposed = {}, {}, {}
    for n, (op, r, e, d) in enumerate(stamps):
        prev_e = stamps[n - 1][2] if n else None
        if prev_e is not None and r < prev_e:
            warm.setdefault(op, []).append(e - prev_e)
        else:
            cold.setdefault(op, []).append((r, e))
        follower = stamps[n + 1] if n + 1 < len(stamps) else None
        if follower is None or follower[1] >= e:
            exposed.setdefault(op, []).append(d - e)
    if not warm:
        raise ValueError("no warm pass: the compute span is not measurable")
    unit_span = statistics.median_low([x for xs in warm.values() for x in xs])
    unit_fill = statistics.median_low(
        [e - r - unit_span for xs in cold.values() for r, e in xs]
    )
    terms = {}
    for op in dict.fromkeys(op for op, *_ in stamps):
        span = statistics.median_low(warm.get(op, [unit_span]))
        fills = [e - r - span for r, e in cold.get(op, [])]
        fill = statistics.median_low(fills) if fills else unit_fill
        tails = exposed.get(op) or [d - e for o, _, e, d in stamps if o == op]
        tail = statistics.median_low(tails)
        terms[op] = (max(0, fill), span, max(0, tail))
    return terms


def shape_text(dims) -> str:
    """One shape as the Operations sheet writes it: ``1x256x512``."""
    if dims is None:
        return ""
    return "x".join(str(int(d)) for d in dims)


def op_shapes(op: OpInfo) -> str:
    """An op's input, weight and output shapes as one ``Shapes`` cell of the
    Op calibration sheet: ``input | weight | output``."""
    cells = []
    for name in ("input", "weight", "output"):
        cells.append(shape_text(op.detail.get(name)))
    return " | ".join(cells)


class Calibration:
    """Build group -> measurement, applied to each ``OpInfo`` as it is
    priced.  ``power`` is carried through untouched for the reporting
    workbook; only the per-op cycles and ``launch_cycles`` change a
    schedule."""

    def __init__(
        self,
        measurements: Dict[str, Measurement],
        op_measurements: Optional[Dict[Tuple[str, int], OpMeasurement]] = None,
        pass_measurements: Optional[
            Dict[Tuple[str, int], PassMeasurement]
        ] = None,
    ):
        self.measurements = measurements
        self.op_measurements = op_measurements or {}
        self.pass_measurements = pass_measurements or {}
        self.applied: Dict[str, int] = {}
        # (key, slot) -> (the row's shapes and tiling, the op's): rows not
        # applied because this compile shapes or tiles the op otherwise.
        self.mismatches: Dict[Tuple[str, int], Tuple[str, str]] = {}

    def apply(self, op: OpInfo, group: Optional[str]) -> None:
        """Place ``op``'s passes by its measured pass terms when the ``Pass
        calibration`` sheet has a row for ``(group, op.slot)`` measured at
        the op's shapes and tiling, else price it by its own measurement
        when the ``Op calibration`` sheet has one; otherwise the analytic
        price stands."""
        if not group:
            return
        per_pass = self.pass_measurements.get((group, op.slot))
        if per_pass is not None and self._matches(per_pass, op):
            op.measured_terms = (per_pass.fill, per_pass.span, per_pass.tail)
            op.calibration = "per-pass"
            self.applied[group] = self.applied.get(group, 0) + 1
            return
        per_op = self.op_measurements.get((group, op.slot))
        if per_op is None or not self._matches(per_op, op):
            return
        op.measured_cycles = max(1, round(per_op.cycles))
        op.calibration = "per-op"
        self.applied[group] = self.applied.get(group, 0) + 1

    def _matches(self, row, op: OpInfo) -> bool:
        """Whether ``row`` was measured at ``op``'s shapes (a blank row
        shape matches any) and tiling; a mismatch is noted."""
        shapes = op_shapes(op)
        tiling = op.detail.get("tiling", "")
        shapes_differ = bool(row.shapes) and shapes != row.shapes
        if shapes_differ or tiling != row.tiling:
            self._mismatch(row, f"{shapes} [{tiling}]")
            return False
        return True

    def _mismatch(self, row, found: str) -> None:
        """Note, once per row, an op whose shapes or tiling differ from its
        row's."""
        where = (row.key, row.slot)
        if where in self.mismatches:
            return
        measured = f"{row.shapes} [{row.tiling}]"
        self.mismatches[where] = (measured, found)
        logger.warning(
            "Calibration row %s slot %d (%s) was measured at %s; this "
            "compile has %s, so the row does not price the op",
            row.key,
            row.slot,
            row.op,
            measured,
            found,
        )

    def launch_cycles(self, group: Optional[str]) -> int:
        """The measured fixed cost of entering a kernel (0 when unmeasured):
        the sum over the group's op rows when it has any, else the
        ``Calibration`` sheet's launch term."""
        if not group:
            return 0
        rows = [
            m for (key, _), m in self.op_measurements.items() if key == group
        ]
        if rows:
            return sum(m.launch_cycles for m in rows)
        m = self.measurements.get(group)
        return m.launch_cycles if m is not None else 0

    def power(self, group: Optional[str]) -> float:
        """The group's measured power in W (0 when unmeasured)."""
        m = self.measurements.get(group) if group else None
        return m.power if m is not None else 0.0


# --------------------------------------------------------------------------
# The calibration sheet: what to simulate, and reading the results back
# --------------------------------------------------------------------------

FORM_SHEET = "Calibration"
FORM_HEADERS = [
    "Key",
    "Group",
    "Kernel",
    "Anchor",
    "Instances",
    "Trip",
    "Period iters",
    "Ops per iteration",
    "Analytic cyc/iter",
    "N1",
    "T1",
    "N2",
    "T2",
    "Measured cyc/iter",
    "Launch cycles",
    "Power (W)",
]
# Suggested run lengths: N1 is past the first iteration's pipeline fill, and
# N2 - N1 is the fewest whole periods spanning at least _MIN_SPAN iterations.
_N1 = 1
_MIN_SPAN = 8


def suggested_runs(trip: int, period: int) -> Tuple[int, Optional[int]]:
    """The iteration counts to simulate a kernel at, ``(N1, N2)``.

    N2 stays below ``trip`` so the final drain is left out; a kernel with no
    room above N1 runs whole, ``(trip, None)``, and is priced by T1 / trip.
    """
    n2 = min(_N1 + period * math.ceil(_MIN_SPAN / period), trip - 1)
    return (trip, None) if n2 <= _N1 else (_N1, n2)


def sheet_rows(rows) -> List:
    """One row per distinct build key, in the order the kernels first run.

    A key is the unit of measurement -- every kernel sharing it runs the same
    nest -- so one row per key is exactly the list of things to simulate, and
    ``Instances`` says how many kernels each stands for.

    Every node carries a key now, DMA-only ones included, so computeless
    kernels are left off: there is nothing in them for the RTL to measure.
    """
    seen, out = set(), []
    for row in rows:
        if row.group and row.ops_per_period and row.group not in seen:
            seen.add(row.group)
            out.append(row)
    return out


def calibration_sheet(wb, rows, calibration=None, formats=None) -> int:
    """Write the calibration sheet into an open ``xlsxwriter`` workbook.

    One row per distinct build key, in first-run order (see ``sheet_rows``).

    ``calibration`` fills the measurement columns of the keys it matched, so
    a calibrated compile's workbook says what priced it -- and leaves the
    keys it did not match blank, which is where the coverage went.

    Returns the number of rows written.
    """
    ws = wb.add_worksheet(FORM_SHEET)
    fmt = formats or {}
    bold = fmt.get("bold") or wb.add_format({"bold": True})
    edit = fmt.get("edit") or wb.add_format(
        {"bg_color": "#FFF2CC", "border": 1}
    )
    for c, h in enumerate(FORM_HEADERS):
        ws.write(0, c, h, bold)
    ws.set_column(_col("Key"), _col("Key"), 14)
    ws.set_column(_col("Group"), _col("Group"), 8)
    ws.set_column(_col("Kernel"), _col("Kernel"), 46)
    ws.set_column(_col("Anchor"), _col("Anchor"), 30)
    ws.set_column(_col("Ops per iteration"), _col("Ops per iteration"), 40)
    ws.freeze_panes(1, 0)

    measured = calibration.measurements if calibration is not None else {}
    n1c, t1c = _letter("N1"), _letter("T1")
    n2c, t2c = _letter("N2"), _letter("T2")
    mc = _letter("Measured cyc/iter")

    written = 0
    for r, row in enumerate(sheet_rows(rows), start=1):
        written += 1
        ws.write(r, _col("Key"), row.group)
        ws.write(
            r,
            _col("Group"),
            f"g{row.group_id}" if row.group_id >= 0 else "",
        )
        ws.write(r, _col("Kernel"), row.kernel)
        ws.write(r, _col("Anchor"), row.anchor)
        ws.write_number(r, _col("Instances"), row.num_shared_kernels)
        ws.write_number(r, _col("Trip"), row.trip_count)
        period = max(1, row.period)
        ws.write_number(r, _col("Period iters"), period)
        ws.write(r, _col("Ops per iteration"), row.ops_text)
        ws.write_number(
            r, _col("Analytic cyc/iter"), row.analytic_per_iteration
        )
        m = measured.get(row.group)
        has_t1 = m is not None and bool(m.n1)
        has_t2 = has_t1 and bool(m.n2)
        n1, n2 = suggested_runs(row.trip_count, period)
        if has_t1:
            ws.write_number(r, _col("N1"), m.n1, edit)
            ws.write_number(r, _col("T1"), m.t1, edit)
        else:
            ws.write_number(r, _col("N1"), n1)
            ws.write_blank(r, _col("T1"), None, edit)
        if has_t2:
            ws.write_number(r, _col("N2"), m.n2, edit)
            ws.write_number(r, _col("T2"), m.t2, edit)
        elif n2 is not None:
            ws.write_number(r, _col("N2"), n2)
            ws.write_blank(r, _col("T2"), None, edit)
        else:
            ws.write_blank(r, _col("N2"), None, edit)
            ws.write_blank(r, _col("T2"), None, edit)
        if m is not None and m.power:
            ws.write_number(r, _col("Power (W)"), m.power, edit)
        else:
            ws.write_blank(r, _col("Power (W)"), None, edit)
        one = r + 1
        ws.write_formula(
            r,
            _col("Measured cyc/iter"),
            f'=IFERROR(IF(AND({t2c}{one}<>"",{n2c}{one}<>{n1c}{one}),'
            f"({t2c}{one}-{t1c}{one})/({n2c}{one}-{n1c}{one}),"
            f'{t1c}{one}/{n1c}{one}),"")',
        )
        ws.write_formula(
            r,
            _col("Launch cycles"),
            f'=IFERROR(MAX(0,{t1c}{one}-{n1c}{one}*{mc}{one}),"")',
        )
    return written


OP_SHEET = "Op calibration"
OP_HEADERS = [
    "Key",
    "Kernel",
    "Slot",
    "Op",
    "Unit",
    "Passes",
    "Measured cycles",
    "Samples",
    "Method",
    "Source",
    "Shapes",
    "Tiling",
    "Launch cycles",
]


def op_calibration_sheet(wb, calibration, formats=None) -> int:
    """Write the ``Op calibration`` rows ``calibration`` carries into an open
    ``xlsxwriter`` workbook, so a per-op calibrated compile's workbook says
    what priced it and loads back as the same calibration.

    Writes nothing (no sheet) when there are no per-op rows.  Returns the
    number of rows written.
    """
    if calibration is None or not calibration.op_measurements:
        return 0
    ws = wb.add_worksheet(OP_SHEET)
    fmt = formats or {}
    bold = fmt.get("bold") or wb.add_format({"bold": True})
    for c, h in enumerate(OP_HEADERS):
        ws.write(0, c, h, bold)
    widths = {
        "Key": 14,
        "Kernel": 46,
        "Op": 32,
        "Source": 40,
        "Shapes": 40,
        "Tiling": 40,
    }
    for name, width in widths.items():
        column = OP_HEADERS.index(name)
        ws.set_column(column, column, width)
    ws.freeze_panes(1, 0)

    for r, m in enumerate(calibration.op_measurements.values(), start=1):
        values = [
            m.key,
            m.kernel,
            m.slot,
            m.op,
            m.unit,
            m.passes,
            m.cycles,
            m.samples,
            m.method,
            m.source,
            m.shapes,
            m.tiling,
            m.launch_cycles,
        ]
        for c, v in enumerate(values):
            if isinstance(v, (int, float)):
                ws.write_number(r, c, v)
            else:
                ws.write(r, c, v)
    return len(calibration.op_measurements)


PASS_SHEET = "Pass calibration"
PASS_HEADERS = [
    "Key",
    "Kernel",
    "Slot",
    "Op",
    "Compute span",
    "Cold fill",
    "Tail",
    "Passes",
    "Source",
    "Shapes",
    "Tiling",
]


def pass_calibration_sheet(wb, calibration, formats=None) -> int:
    """Write the ``Pass calibration`` rows ``calibration`` carries into an
    open ``xlsxwriter`` workbook, so a per-pass calibrated compile's
    workbook loads back as the same calibration.

    Writes nothing (no sheet) when there are no pass rows.  Returns the
    number of rows written.
    """
    if calibration is None or not calibration.pass_measurements:
        return 0
    ws = wb.add_worksheet(PASS_SHEET)
    fmt = formats or {}
    bold = fmt.get("bold") or wb.add_format({"bold": True})
    for c, h in enumerate(PASS_HEADERS):
        ws.write(0, c, h, bold)
    widths = {"Kernel": 46, "Op": 32, "Source": 40, "Shapes": 40, "Tiling": 40}
    for name, width in widths.items():
        column = PASS_HEADERS.index(name)
        ws.set_column(column, column, width)
    ws.freeze_panes(1, 0)
    for r, m in enumerate(calibration.pass_measurements.values(), start=1):
        values = [
            m.key,
            m.kernel,
            m.slot,
            m.op,
            m.span,
            m.fill,
            m.tail,
            m.passes,
            m.source,
            m.shapes,
            m.tiling,
        ]
        for c, v in enumerate(values):
            if isinstance(v, (int, float)):
                ws.write_number(r, c, v)
            else:
                ws.write(r, c, v)
    return len(calibration.pass_measurements)


def power_column() -> str:
    """The Power column's A1 letter, for a formula on another sheet."""
    return _letter("Power (W)")


def _col(name: str) -> int:
    """A header's zero-based column index."""
    return FORM_HEADERS.index(name)


def _letter(name: str) -> str:
    """A header's A1 column letter, so the formulas follow the header order
    instead of hard-coding letters that shift when a column is added.  The
    sheet is under 26 columns wide, so one letter covers it."""
    return chr(ord("A") + _col(name))


def load_calibration(path: str) -> Calibration:
    """Read a filled-in sheet back, from a standalone form or from a
    reporting workbook (both name the sheet ``Calibration``).

    A row with ``T1`` and ``T2`` gives ``(T2 - T1) / (N2 - N1)`` cycles per
    iteration and ``T1 - N1 * that`` launch cycles; a row with only ``T1``
    gives ``T1 / N1`` and no launch term.  The arithmetic is redone here
    rather than read out of the ``Measured cyc/iter`` cell, because a
    workbook written but never opened in Excel has no cached formula
    results.  ``Power (W)`` is read straight through.  Rows with neither a
    measurement nor a power are ignored.  The sheet carries one row per key,
    so a repeated key means the file was edited by hand; the last row wins.

    An ``Op calibration`` and a ``Pass calibration`` sheet, when present,
    are read too (see ``load_op_measurements`` and
    ``load_pass_measurements``); any of the three may be absent, not all.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    op_measurements = load_op_measurements(wb)
    pass_measurements = load_pass_measurements(wb)
    if FORM_SHEET not in wb.sheetnames:
        if not {OP_SHEET, PASS_SHEET} & set(wb.sheetnames):
            raise ValueError(
                f"{path} has no {FORM_SHEET!r}, {OP_SHEET!r} or "
                f"{PASS_SHEET!r} sheet"
            )
        return Calibration({}, op_measurements, pass_measurements)
    ws = wb[FORM_SHEET]
    header = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(header) if name in FORM_HEADERS}
    measurements: Dict[str, Measurement] = {}

    def cell(cells, name):
        i = col.get(name)
        return cells[i] if i is not None and i < len(cells) else None

    for cells in ws.iter_rows(min_row=2, values_only=True):
        group = cell(cells, "Key")
        if not group:
            continue
        t1 = cell(cells, "T1")
        power = cell(cells, "Power (W)")
        if t1 is None and power is None:
            continue
        n1, n2, t2 = (cell(cells, k) for k in ("N1", "N2", "T2"))
        analytic = float(cell(cells, "Analytic cyc/iter") or 0) or 1.0
        per_iteration = launch = 0.0
        if t1 is not None and n1:
            if t2 is not None and n2 is not None and n2 != n1:
                per_iteration = (float(t2) - float(t1)) / (
                    float(n2) - float(n1)
                )
                launch = float(t1) - float(n1) * per_iteration
            else:
                per_iteration = float(t1) / float(n1)
        measurements[str(group)] = Measurement(
            measured_per_iteration=per_iteration,
            analytic_per_iteration=analytic,
            launch_cycles=max(0, round(launch)),
            power=float(power) if power is not None else 0.0,
            n1=float(n1 or 0),
            t1=float(t1 or 0),
            n2=float(n2 or 0),
            t2=float(t2 or 0),
        )
    return Calibration(measurements, op_measurements, pass_measurements)


def load_op_measurements(wb) -> Dict[Tuple[str, int], OpMeasurement]:
    """The ``Op calibration`` rows of an open ``openpyxl`` workbook, keyed by
    ``(Key, Slot)``; empty when the sheet is absent.

    A row needs a key, a slot and a positive ``Measured cycles`` (one steady
    sample suffices: the RTL repeats a tile to the cycle); one whose numbers
    do not read as numbers is dropped.  A row from before the ``Tiling``
    column loads with a blank tiling, so it prices only an op no search
    tiled; a tiled op keeps its analytic price.  A repeated ``(Key, Slot)``
    means a hand edit; the last row wins.
    """
    if OP_SHEET not in wb.sheetnames:
        return {}
    ws = wb[OP_SHEET]
    header = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(header) if name in OP_HEADERS}

    def cell(cells, name):
        i = col.get(name)
        return cells[i] if i is not None and i < len(cells) else None

    out: Dict[Tuple[str, int], OpMeasurement] = {}
    for cells in ws.iter_rows(min_row=2, values_only=True):
        key = cell(cells, "Key")
        if not key:
            continue
        try:
            slot = int(cell(cells, "Slot"))
            cycles = float(cell(cells, "Measured cycles"))
            samples = int(cell(cells, "Samples") or 0)
            passes = int(cell(cells, "Passes") or 0)
            launch = round(float(cell(cells, "Launch cycles") or 0))
        except (TypeError, ValueError):
            continue
        if cycles <= 0:
            continue
        m = OpMeasurement(
            key=str(key),
            slot=slot,
            cycles=cycles,
            kernel=str(cell(cells, "Kernel") or ""),
            op=str(cell(cells, "Op") or ""),
            unit=str(cell(cells, "Unit") or ""),
            passes=passes,
            samples=samples,
            method=str(cell(cells, "Method") or ""),
            source=str(cell(cells, "Source") or ""),
            shapes=str(cell(cells, "Shapes") or ""),
            tiling=str(cell(cells, "Tiling") or ""),
            launch_cycles=max(0, launch),
        )
        out[(m.key, m.slot)] = m
    return out


def load_pass_measurements(wb) -> Dict[Tuple[str, int], PassMeasurement]:
    """The ``Pass calibration`` rows of an open ``openpyxl`` workbook, keyed
    by ``(Key, Slot)``; empty when the sheet is absent.

    A row needs a key, a slot and a positive ``Compute span``; a blank fill
    or tail reads as 0, and a row whose numbers do not read as numbers is
    dropped.  A repeated ``(Key, Slot)`` means a hand edit; the last row
    wins.
    """
    if PASS_SHEET not in wb.sheetnames:
        return {}
    ws = wb[PASS_SHEET]
    header = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(header) if name in PASS_HEADERS}

    def cell(cells, name):
        i = col.get(name)
        return cells[i] if i is not None and i < len(cells) else None

    out: Dict[Tuple[str, int], PassMeasurement] = {}
    for cells in ws.iter_rows(min_row=2, values_only=True):
        key = cell(cells, "Key")
        if not key:
            continue
        try:
            slot = int(cell(cells, "Slot"))
            span = round(float(cell(cells, "Compute span")))
            fill = round(float(cell(cells, "Cold fill") or 0))
            tail = round(float(cell(cells, "Tail") or 0))
            passes = int(cell(cells, "Passes") or 0)
        except (TypeError, ValueError):
            continue
        if span <= 0:
            continue
        m = PassMeasurement(
            key=str(key),
            slot=slot,
            fill=max(0, fill),
            span=span,
            tail=max(0, tail),
            kernel=str(cell(cells, "Kernel") or ""),
            op=str(cell(cells, "Op") or ""),
            passes=passes,
            source=str(cell(cells, "Source") or ""),
            shapes=str(cell(cells, "Shapes") or ""),
            tiling=str(cell(cells, "Tiling") or ""),
        )
        out[(m.key, m.slot)] = m
    return out
