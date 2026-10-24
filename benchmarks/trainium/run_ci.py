#!/usr/bin/env python
"""Regression CI for the Trainium backend.

Compiles a fixed set of one-layer design points with ``compile_lm.py``,
checks each one's kernels in the NKI CPU simulator (``check.py
--distinct``) and drops every point's outputs into a date+time folder under
a user-supplied output location.  Each point's Trainium ``model.txt``,
``layers.txt`` and ``kernels.txt`` (its distinct kernels, named by a digest
of their NKI source, with their call counts) are compared with the previous
run's, the way ``test/run_ci.py`` compares the Voyager CI's.

Usage (from the repo root, in the Voyager env)::

    python benchmarks/trainium/run_ci.py <output_dir> [--jobs 3] \\
        [--cores 32-63] [--only SUBSTR] [--neff] [--trace] [--device]
    python benchmarks/trainium/run_ci.py --list

``--neff`` also compiles each distinct kernel to a NEFF for its target,
``--trace`` traces the whole step into one NEFF (``trace.py``) and reports
its SB spills, and ``--device`` runs the traced step on the Trn2 host
(``~/claude/trn/host/run_point.sh``), which must already be up: the CI never
starts it.  The device runs one point at a time and is a Trn2, so
``--device`` takes only ``trn2`` points.  The device's ``run.json`` is
copied into the point's folder and its latency reported.  Every step runs
under ``taskset`` on its job's share of ``--cores``, at ``nice`` 10, with no
GPU.  A point passes when every step does; its ``tensors.pt`` (2.5 GB for
Llama-3.1-8B) is then deleted, and kept when a step fails.  Exits non-zero
on any FAILED or MISMATCH.
"""

import argparse
import importlib.util
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The Voyager CI shares this script's file name, so it is loaded by path.
_SPEC = importlib.util.spec_from_file_location(
    "voyager_ci", REPO_ROOT / "test" / "run_ci.py"
)
voyager_ci = importlib.util.module_from_spec(_SPEC)
sys.modules["voyager_ci"] = voyager_ci
_SPEC.loader.exec_module(voyager_ci)

COMPILE_LM = REPO_ROOT / "benchmarks" / "trainium" / "compile_lm.py"
RUNTIME = REPO_ROOT / "src" / "voyager_compiler" / "backend" / "trainium"
CHECK = RUNTIME / "runtime" / "check.py"
TRACE = RUNTIME / "runtime" / "trace.py"
TRN_ENV = Path("~/claude/trn/env.sh").expanduser()
RUN_POINT = Path("~/claude/trn/host/run_point.sh").expanduser()
# Where ``run_point.sh`` leaves each run's ``run.json``.
HOST_RUNS = RUN_POINT.parent / "runs"
# The Trn2 host has one NeuronCore: one point runs on it at a time.
DEVICE = threading.Lock()

TITLE = "trainium CI"
# The files each point's outputs are compared by.
OUTPUTS = ("model.txt", "layers.txt", "kernels.txt")
# Every point compiles a 1024-token step.
CONTEXT = 1024

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen3-4B"
GEMMA = "google/gemma-3-1b-it"


@dataclass(frozen=True)
class Point:
    """One design point: a model's step compiled for one NeuronCore.

    Attributes:
        network: The model's short name, the label's first segment.
        model: Its Hugging Face checkpoint.
        dtype: The ``compile_lm.py`` ``--dtype``.
        phase: ``prefill`` or ``decode``.
        target: The core, ``trn2`` or ``trn3``.
        variant: What sets the point apart, appended to the phase in its
            label.
        extra: Any other ``compile_lm.py`` flags.
        layers: How many decoder layers it compiles.
    """

    network: str
    model: str
    dtype: str
    phase: str
    target: str = "trn2"
    variant: str = ""
    extra: str = ""
    layers: int = 1

    @property
    def label(self):
        """The point's label, ``<network>/<dtype>/<phase>[_<variant>]``."""
        step = f"{self.phase}_{self.variant}" if self.variant else self.phase
        return f"{self.network}/{self.dtype}/{step}"


_MERGED = "--merge-linears"
# MX attention (``sdpa_mx``) does not lower to Trainium: an MX point
# decomposes its attention, whose matmuls stay BF16.
_EAGER = "--attention eager"

POINTS = [
    Point("llama31", LLAMA, "bf16", "prefill"),
    Point("llama31", LLAMA, "bf16", "decode"),
    Point("llama31", LLAMA, "fp8", "prefill", variant="merged", extra=_MERGED),
    Point("llama31", LLAMA, "fp8", "decode", variant="merged", extra=_MERGED),
    Point("llama31", LLAMA, "fp8", "prefill"),
    Point("llama31", LLAMA, "mxfp8", "prefill", "trn3", "eager", _EAGER),
    Point("llama31", LLAMA, "mxfp8", "decode", "trn3", "eager", _EAGER),
    Point("llama31", LLAMA, "mxfp8_e5m2", "prefill", "trn3", "eager", _EAGER),
    Point("llama31", LLAMA, "bf16", "prefill", variant="eager", extra=_EAGER),
    Point("llama31", LLAMA, "bf16", "decode", variant="2layers", layers=2),
    Point("qwen3", QWEN, "bf16", "prefill"),
    Point("qwen3", QWEN, "fp8", "decode"),
    Point("gemma3", GEMMA, "bf16", "prefill"),
    Point("gemma3", GEMMA, "bf16", "decode"),
]
assert len({p.label for p in POINTS}) == len(POINTS), "duplicate labels"


def steps(point, dest, args, run_name):
    """Lists the steps that build and check ``point``, in order.

    The build always runs, then the simulator check; ``--neff``, ``--trace``
    and ``--device`` add their steps, and ``--device`` implies ``--trace``,
    since the device runs the traced step.  Every step but the build runs in
    the Neuron env.

    Args:
        point: The design point.
        dest: The folder its outputs go to.
        args: The parsed command line.
        run_name: The name of its run on the Trn2 host.

    Returns:
        Each step's ``(name, argv)``.
    """
    build = [
        sys.executable,
        str(COMPILE_LM),
        f"--model={point.model}",
        f"--phase={point.phase}",
        f"--context={CONTEXT}",
        f"--layers={point.layers}",
        f"--target={point.target}",
        f"--dtype={point.dtype}",
        *shlex.split(point.extra),
        f"--out={dest}",
    ]
    out = shlex.quote(str(dest))
    run_json = shlex.quote(str(HOST_RUNS / run_name / "run.json"))
    trn = [
        ("check", f"python {CHECK} {out} --distinct"),
        ("neff", f"python {CHECK} {out} --neff --target {point.target}"),
        (
            "trace",
            f"python {TRACE} {out} --phase {point.phase} "
            f"--target {point.target} --weights-from {point.model}",
        ),
        (
            "device",
            f"bash {RUN_POINT} {out} {run_name} && cp {run_json} {out}/",
        ),
    ]
    wanted = {"check"}
    wanted |= {n for n in ("neff", "trace", "device") if getattr(args, n)}
    if args.device:
        wanted.add("trace")
    return [("build", build)] + [
        (name, ["bash", "-c", f"source {TRN_ENV} >/dev/null && {command}"])
        for name, command in trn
        if name in wanted
    ]


def write_kernels(dest):
    """Writes ``dest/kernels.txt`` from the translation's ``manifest.json``.

    Each line names a distinct kernel, in the order it is first called, with
    its call count and its first op.

    Args:
        dest: The point's output folder.
    """
    with open(dest / "manifest.json") as f:
        calls = [s for s in json.load(f)["steps"] if s["kernel"] is not None]
    counts, first = {}, {}
    for call in calls:
        counts[call["kernel"]] = counts.get(call["kernel"], 0) + 1
        first.setdefault(call["kernel"], call["ops"][0])
    lines = [f"{k} x{n} {first[k]}" for k, n in counts.items()]
    (dest / "kernels.txt").write_text("\n".join(lines) + "\n")


def run_point(point, run_dir, prev_run, args, env, cores):
    """Runs each step of ``point`` until one fails and compares its outputs.

    The steps run on a share of the cores taken from ``cores`` and given
    back after; the device step waits for the device.  Each step's output
    goes to ``run.log`` and its wall time to ``steps.json``.  The status is
    ``ok``, ``<step>_failed``, ``no_output`` (a build that exits cleanly
    without a Trainium ``model.txt``) or ``crashed`` (an exception in the CI
    itself, whose traceback becomes the excerpt).  A point whose steps did
    not all pass is FAILED unless its outputs MISMATCH.

    Args:
        point: The design point.
        run_dir: This run's folder.
        prev_run: The previous run's folder, or ``None``.
        args: The parsed command line.
        env: The steps' environment.
        cores: A queue of ``taskset`` core ranges, one per job.

    Returns:
        ``(label, status, verdict, excerpt)``, as ``voyager_ci.run_all``
        takes it.
    """
    label = point.label
    dest = run_dir / label
    dest.mkdir(parents=True)
    run_name = f"ci_{run_dir.name}_{label.replace('/', '_')}"
    status, seconds = "ok", {}
    share = cores.get()
    try:
        with open(dest / "run.log", "w") as log:
            for name, argv in steps(point, dest, args, run_name):
                argv = ["taskset", "-c", share, "nice", "-n", "10", *argv]
                log.write("$ " + shlex.join(argv) + "\n\n")
                log.flush()
                with DEVICE if name == "device" else nullcontext():
                    start = time.monotonic()
                    proc = subprocess.run(
                        argv, cwd=dest, env=env, stdout=log, stderr=log
                    )
                    seconds[name] = round(time.monotonic() - start, 1)
                log.write(f"\n[{name}: exit {proc.returncode}]\n\n")
                log.flush()
                if proc.returncode != 0:
                    status = f"{name}_failed"
                    break
                if name == "build":
                    if not (dest / "model.txt").exists():
                        status = "no_output"
                        break
                    write_kernels(dest)
        built = status not in ("build_failed", "no_output")
        verdict, excerpt = voyager_ci.compare(
            label, not built, run_dir, prev_run, OUTPUTS
        )
    except Exception:
        status, verdict, excerpt = "crashed", "FAILED", traceback.format_exc()
        with open(dest / "run.log", "a") as log:
            log.write(excerpt)
    finally:
        cores.put(share)
    (dest / "steps.json").write_text(json.dumps(seconds) + "\n")
    if status == "ok":
        (dest / "tensors.pt").unlink(missing_ok=True)
    elif verdict in ("MATCH", "NEW"):
        verdict = "FAILED"
    return label, status, verdict, excerpt


def summary(results, run_dir, prev_run):
    """Tabulates each point's step times, simulator results and latencies.

    A row gives the point's wall time by step, how many of its simulated
    kernels passed, its predicted step time (the previous run's beside it)
    and, when traced, its SB spills and, when run on the device, its median
    device latency.

    Args:
        results: Each point's ``(label, status, verdict, excerpt)``.
        run_dir: This run's folder.
        prev_run: The previous run's folder, or ``None``.

    Returns:
        The table's text.
    """

    def predicted(run, label):
        """Reads a point's predicted step time from a run.

        Args:
            run: The run's folder, or ``None``.
            label: The point's label.

        Returns:
            The time in ms, or ``None`` when ``run`` is ``None`` or the
            point wrote no ``predicted.json``.
        """
        path = run / label / "predicted.json" if run else None
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())["total_s"] * 1e3

    lines = ["", "per point (wall s by step; predicted ms, previous run's)"]
    for label, _, verdict, _ in results:
        dest = run_dir / label
        seconds = json.loads((dest / "steps.json").read_text())
        timing = " ".join(f"{k} {v:.0f}" for k, v in seconds.items())
        log = (dest / "run.log").read_text()
        passed = len(re.findall(r"^PASS \[", log, re.M))
        failed = len(re.findall(r"^FAIL \[", log, re.M))
        row = f"  {label:<32} {verdict:<8} {sum(seconds.values()):>5.0f} s"
        row += f" ({timing})  sim {passed}/{passed + failed}"
        now, before = predicted(run_dir, label), predicted(prev_run, label)
        if now is not None:
            row += f"  {now:.3f} ms"
            row += "" if before is None else f" ({before:.3f})"
        spills = dest / "log-neuron-cc.txt"
        if spills.exists():
            found = re.findall(r"SB spills = (\d+)", spills.read_text())
            row += f"  SB spills {','.join(sorted(set(found), key=int))}"
        device = dest / "run.json"
        if device.exists():
            p50 = json.loads(device.read_text())["p50_s"] * 1e3
            row += f"  device p50 {p50:.3f} ms"
        lines.append(row)
    return "\n".join(lines)


def main():
    """Runs the selected points, or lists their steps under ``--list``.

    Returns:
        The exit status: 1 when a point is FAILED or MISMATCH, else 0.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument(
        "--only",
        action="append",
        metavar="SUBSTR",
        help="run only the points whose label it selects (repeatable; "
        "matched as test/run_ci.py matches)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--cores",
        default="32-63",
        help="the CPU range FIRST-LAST (or one CPU) the jobs share, split "
        "evenly between them",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="OMP / MKL threads per step"
    )
    parser.add_argument("--neff", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()

    points = [
        p
        for p in POINTS
        if not args.only
        or any(voyager_ci.matches(p.label, s) for s in args.only)
    ]
    if args.list:
        for point in points:
            print(point.label)
            dest = Path("<run_dir>") / point.label
            for name, argv in steps(point, dest, args, "<name>"):
                print(f"    {name}: {shlex.join(argv)}")
        return 0
    if not args.output_dir:
        parser.error("output_dir is required (unless --list)")
    if not points:
        parser.error(f"--only {args.only} matched no points")
    trn3 = [p.label for p in points if p.target != "trn2"]
    if args.device and trn3:
        parser.error(
            f"--device runs on a Trn2; these points target trn3: {trn3} "
            "(select the trn2 points with --only)"
        )

    cpus = re.fullmatch(r"(\d+)(?:-(\d+))?", args.cores)
    if cpus is None:
        parser.error("--cores must be FIRST-LAST or one CPU")
    first, last = int(cpus[1]), int(cpus[2] or cpus[1])
    if last < first:
        parser.error("--cores must be FIRST-LAST with FIRST <= LAST")
    count = last - first + 1
    if not 1 <= args.jobs <= count:
        parser.error(f"--jobs must be 1 to {count}, the cores in --cores")
    cores = queue.Queue()
    for j in range(args.jobs):
        lo = first + j * count // args.jobs
        hi = first + (j + 1) * count // args.jobs - 1
        cores.put(f"{lo}-{hi}")

    run_dir, prev_run = voyager_ci.open_run(args.output_dir, TITLE, OUTPUTS)
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(REPO_ROOT / "src"),
    )
    results = voyager_ci.run_all(
        points,
        lambda p: run_point(p, run_dir, prev_run, args, env, cores),
        args.jobs,
    )
    report = voyager_ci.build_report(results, run_dir, prev_run, TITLE)
    report += "\n" + summary(results, run_dir, prev_run)
    (run_dir / "report.txt").write_text(report + "\n")
    print("\n" + report)
    return 1 if any(r[2] in voyager_ci.GATED for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
