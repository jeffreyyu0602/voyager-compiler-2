#!/usr/bin/env python
"""Local pre-push CI for the voyager codegen commands.

Runs every actively-used ``test_codegen.py`` invocation (the ones reached
through the accelerator's ``codegen.mk``) without dumping tensors, and drops
each run's artifacts into a date+time folder under a user-supplied output
location.  If a prior run exists in that location, the freshly produced
``model.txt`` of each command is compared against the previous run's and any
mismatch (or compile failure) is reported.

Usage (from the repo root, with the conda env active)::

    python test/run_ci.py <output_dir> [--jobs N] [--threads T]

``--jobs`` runs that many commands at once; ``--threads`` caps each one's CPU
threads, so ``N * T`` bounds the cores the run takes.  Exits non-zero if any
command fails to compile or any ``model.txt`` differs from the previous run,
so it can gate a pre-push hook.
"""

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Repo root = parent of this file's directory (test/run_ci.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CODEGEN = REPO_ROOT / "test" / "test_codegen.py"

# Number of unified-diff lines shown in the report for a mismatch.
DIFF_EXCERPT_LINES = 60

# ---------------------------------------------------------------------------
# Command table.
#
# Each Command is expanded at runtime into a test_codegen.py argv:
#     <python> test_codegen.py <model> <SCHEME_ARGS[scheme]>
#         --pe_array_size <unrolling> <extra>
#         --model_output_dir <run_dir>/<label>
# The shared per-scheme quantization/compile flags live once in SCHEME_ARGS
# (no --dump_tensors). To add coverage, add a Command (and a new scheme to
# SCHEME_ARGS if needed).
# ---------------------------------------------------------------------------

# Shared quantization/compile flags per scheme.
SCHEME_ARGS = {
    "E4M3": (
        "--activation fp8_e4m3 --weight fp8_e4m3 --bf16 "
        "--tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
    "P8_1": (
        "--activation posit8_1 --weight posit8_1 --bf16 "
        "--tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
    "INT8": (
        "--activation int8,qs=per_tensor_symmetric "
        "--weight int8,qs=per_tensor_symmetric --bias int24 --bf16 "
        "--calibration_steps 3 --tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
    "MXINT8": (
        "--activation int8,qs=microscaling,bs=16,pot=1 "
        "--weight int8,qs=microscaling,bs=16,pot=1 "
        "--bf16 --tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
    "MXNF4": (
        "--activation lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 "
        "--weight lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 --bf16 "
        "--residual fp8_e4m3 --quantize_fc --tensor_layout nhwc "
        "--gemm_layout xc,ck->xk --scratchpad_size 2097152 --num_banks 16 "
        "--conv2d_im2col"
    ),
    "BF16": "--bf16 --tensor_layout nhwc --gemm_layout xc,ck->xk",
    # Training schemes: the gradients (--error) are quantized too.  FP8 is
    # e4m3 forward and e5m2 gradients with per-tensor delayed scaling.
    "FP8": (
        "--activation fp8_e4m3,qs=per_tensor_symmetric,ahl=16 "
        "--weight fp8_e4m3,qs=per_tensor_symmetric,ahl=16 "
        "--error fp8_e5m2,qs=per_tensor_symmetric,ahl=16 "
        "--bf16 --tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
    "MXFP4": (
        "--activation fp4_e2m1,qs=microscaling,bs=32,pot=1 "
        "--weight fp4_e2m1,qs=microscaling,bs=32,pot=1 "
        "--error fp4_e2m1,qs=microscaling,bs=32,pot=1 "
        "--bf16 --tensor_layout nhwc --gemm_layout xc,ck->xk"
    ),
}

# The matrix search sizes its blocking against a scratchpad and a bank
# count, and has no answer when they are unset.  Only MXNF4 names its own, so
# every other scheme is given these -- the same pair MXNF4 asks for.
DEFAULT_TILER_ARGS = {"--scratchpad_size": "2097152", "--num_banks": "16"}

# One activation element, bytes, per scheme.  Each command's --bank_width is
# one input beat -- the PE array's column count times this -- which is the
# word the store bus writes whole, so the memory planner both aligns to it
# and reserves the tail slack a sub-word store beat overshoots by.
SCHEME_INPUT_BYTES = {
    "E4M3": 1,
    "P8_1": 1,
    "INT8": 1,
    "MXINT8": 1,
    "MXNF4": 0.5,
    "BF16": 2,
    "FP8": 1,
    "MXFP4": 0.5,
}
assert SCHEME_INPUT_BYTES.keys() == SCHEME_ARGS.keys()

# Reused per-command extra-flag groups.
_SINGLE = "--num_hidden_layers 1"
_FP16 = "--remove_fp32_casts"
_LLM = (
    "--context_length 1024 --num_hidden_layers 1 --quantize_attention_mask "
    + _FP16
)
_LLM_MP = _LLM + " --qconfig mxlut4_int6_e5m3_attn_head_int6"
_LLM_SPMM = _LLM + " --qconfig mxlut4_int6_e5m3_outlier"
_DB = "--double_buffered_l2"
_LLM_DB = _LLM + " " + _DB
_LLM_SPMM_DB = _LLM_SPMM + " " + _DB
_LLM_TRAIN = (
    "--training --context_length 1024 --num_hidden_layers 1 "
    "--accumulate_fp32 " + _FP16
)


@dataclass(frozen=True)
class Command:
    """One codegen invocation, expanded to argv via SCHEME_ARGS."""

    model: str  # test_codegen.py positional argument
    scheme: str  # key into SCHEME_ARGS
    unrolling: str  # --pe_array_size value, e.g. "16,16"
    network: str = ""  # output/label name (defaults to model)
    extra: str = ""  # any per-command extra flags


COMMANDS = [
    # -- E4M3 --
    Command("resnet18", "E4M3", "16,16"),
    Command("resnet18", "E4M3", "32,64"),
    Command("resnet18", "E4M3", "4,8"),
    Command("resnet50", "E4M3", "32,64"),
    Command("mobilebert", "E4M3", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilebert", "E4M3", "4,8", "mobilebert_encoder", _SINGLE),
    # -- P8_1 --
    Command("resnet18", "P8_1", "16,16"),
    Command("mobilebert", "P8_1", "16,16", "mobilebert_encoder", _SINGLE),
    # -- INT8 --
    Command("resnet18", "INT8", "16,16"),
    Command("mobilebert", "INT8", "16,16", "mobilebert_encoder", _SINGLE),
    # -- MXINT8 --
    Command("resnet18", "MXINT8", "16,16"),
    Command("mobilebert", "MXINT8", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilenet_v2", "MXINT8", "16,16"),
    # -- MXNF4 (llama) --
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill", _LLM),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_mp", _LLM_MP),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_spmm", _LLM_SPMM),
    Command("llama_decode", "MXNF4", "64,64", "llama_decode", _LLM),
    Command("llama_decode_kivi", "MXNF4", "64,64", "llama_decode_kivi", _LLM),
    # -- BF16 (an unquantized KV cache stays whole) --
    Command("llama_decode", "BF16", "64,64", "llama_decode", _LLM),
    # -- MXNF4 (vision / bert) --
    Command("resnet18", "MXNF4", "64,64"),
    Command("resnet50", "MXNF4", "64,64"),
    Command("vit", "MXNF4", "64,64", extra=_FP16),
    Command("bert", "MXNF4", "64,64"),
    # -- double-buffered L2 (conv / prefill / decode / sparse prefill each
    # pipeline apart) --
    Command("resnet18", "MXNF4", "64,64", "resnet18_db", _DB),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_db", _LLM_DB),
    Command("llama_decode", "MXNF4", "64,64", "llama_decode_db", _LLM_DB),
    Command(
        "llama_prefill", "MXNF4", "64,64", "llama_prefill_spmm_db", _LLM_SPMM_DB
    ),
    # -- training: one AdamW step, compiled into gradient/ and update/ --
    Command("llama_prefill", "BF16", "64,64", "llama_train", _LLM_TRAIN),
    Command("llama_prefill", "FP8", "64,64", "llama_train", _LLM_TRAIN),
    Command("llama_prefill", "MXFP4", "64,64", "llama_train", _LLM_TRAIN),
]

# Timestamp folder format; lexicographic sort == chronological sort.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

# The CI's name, heading its output and report.
TITLE = "voyager codegen CI"
# The files each command's outputs are compared by.
OUTPUTS = ("model.txt",)
# The verdicts that fail a run.
GATED = ("FAILED", "MISMATCH")


def _label(command):
    """``<network>/<scheme>/<unrolling>`` — unique per command (unrolling
    disambiguates commands sharing a network/scheme, e.g. resnet18 E4M3)."""
    network = command.network or command.model
    unroll = command.unrolling.replace(",", "x")
    return f"{network}/{command.scheme}/{unroll}"


assert len({_label(c) for c in COMMANDS}) == len(COMMANDS), (
    "COMMANDS have duplicate <network>/<scheme>/<unrolling> labels; give "
    "colliding commands distinct 'network' names"
)


def matches(label, pattern):
    """Does ``pattern`` select ``label`` (``network/scheme/unrolling``)?

    Matching is anchored at ``/`` segment boundaries, not raw substring: a
    single token matches a whole segment by prefix (so ``bert`` selects
    ``bert`` but not ``mobilebert_encoder``, while ``mobilebert`` still
    selects ``mobilebert_encoder`` and ``resnet`` selects both resnets); a
    ``a/b`` token matches as an anchored path prefix.
    """
    segs = label.lower().split("/")
    parts = pattern.lower().strip("/").split("/")
    if len(parts) == 1:
        return any(s.startswith(parts[0]) for s in segs)
    return len(parts) <= len(segs) and all(
        segs[i].startswith(parts[i]) for i in range(len(parts))
    )


def _build(command, run_dir):
    """Expand a Command into ``(label, dest, argv)`` for this run.

    The argv runs this repo's test_codegen.py with --model_output_dir pointed
    into the timestamped run folder; --dump_tensors is never added.
    ``--debug`` is always added: without it test_codegen never evaluates the
    lowered graph, so the run checks codegen only and no numeric comparison
    happens at all.
    """
    label = _label(command)
    dest = run_dir / label

    argv = [sys.executable, str(TEST_CODEGEN), command.model, "--debug"]
    argv += shlex.split(SCHEME_ARGS[command.scheme])
    argv += ["--pe_array_size", command.unrolling]
    argv += shlex.split(command.extra)
    for flag, value in DEFAULT_TILER_ARGS.items():
        if flag not in argv:
            argv += [flag, value]
    if "--bank_width" not in argv:
        cols = int(command.unrolling.split(",")[1])
        width = int(cols * SCHEME_INPUT_BYTES[command.scheme])
        argv += ["--bank_width", str(width)]
    argv += ["--model_output_dir", str(dest)]
    return label, dest, argv


def outputs(dest, names):
    """Finds the files named one of ``names`` that a command wrote.

    Args:
        dest: The command's output folder.
        names: The file names sought, e.g. ``model.txt``: one for inference,
            one per program (``gradient/``, ``update/``) for training.

    Returns:
        Their paths relative to ``dest``, sorted.
    """
    return sorted(
        p.relative_to(dest) for name in names for p in dest.rglob(name)
    )


def _run_one(label, dest, argv, env):
    """Run one command in ``env``; capture combined output to
    ``dest/run.log``.

    Returns a status string: ``ok`` / ``error`` (nonzero exit) /
    ``no_output`` (exited 0 but no model.txt) / ``numeric_drift`` (the
    lowered graph's output left test_codegen's tolerance) / ``unverified``
    (the model ran no comparison despite --debug).  test_codegen only warns
    about the last two, so they have to be read back out of the log; neither
    fails the run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "run.log"
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n\n")
        log.flush()
        proc = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        return "error"
    if not outputs(dest, OUTPUTS):
        return "no_output"
    log = log_path.read_text()
    if "Skipping output verification" in log:
        return "unverified"
    if "Results match" not in log:
        return "numeric_drift"
    return "ok"


def _find_previous_run(out_dir, current):
    """Newest timestamp dir under ``out_dir`` that isn't ``current``."""
    runs = sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_dir() and TS_RE.match(p.name) and p.name != current
    )
    return out_dir / runs[-1] if runs else None


def compare(label, failed, run_dir, prev_run, names):
    """Classifies one label's output files against the previous run's.

    Args:
        label: The command's label, its folder under each run.
        failed: The command wrote no outputs to compare.
        run_dir: This run's folder.
        prev_run: The previous run's folder, or ``None``.
        names: The names of the files compared (``outputs``).

    Returns:
        ``(verdict, diff_excerpt)``.  ``diff_excerpt`` is non-empty only for
        a MISMATCH.
    """
    if failed:
        return "FAILED", ""
    previous = (
        set() if prev_run is None else set(outputs(prev_run / label, names))
    )
    if not previous:
        return "NEW", ""

    programs = set(outputs(run_dir / label, names))
    diff = [
        f"only in {run.name}/{label}: {program}"
        for run, only in (
            (prev_run, previous - programs),
            (run_dir, programs - previous),
        )
        for program in sorted(only)
    ]
    for program in sorted(programs & previous):
        diff += difflib.unified_diff(
            (prev_run / label / program).read_text().splitlines(),
            (run_dir / label / program).read_text().splitlines(),
            fromfile=f"{prev_run.name}/{label}/{program}",
            tofile=f"{run_dir.name}/{label}/{program}",
            lineterm="",
        )
    if not diff:
        return "MATCH", ""
    return "MISMATCH", "\n".join(diff[:DIFF_EXCERPT_LINES])


def open_run(output_dir, title, names):
    """Makes this run's timestamped folder and prints what it compares.

    Args:
        output_dir: The folder holding every run (created if missing).
        title: The CI's name, printed first.
        names: The names of the files compared (``outputs``).

    Returns:
        ``(run_dir, prev_run)``: this run's folder and the newest earlier
        run's, or ``None`` when there is none.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(TS_FORMAT)
    run_dir = out_dir / timestamp
    run_dir.mkdir()

    prev_run = _find_previous_run(out_dir, timestamp)

    print(f"{title} -> {run_dir}")
    if prev_run is not None:
        compared = ", ".join(names)
        print(f"comparing {compared} against previous run: {prev_run.name}")
    else:
        print("no previous run found; this run is the baseline")
    return run_dir, prev_run


def run_all(items, run, jobs):
    """Runs ``run`` on each of ``items``, ``jobs`` at a time.

    Prints each item's verdict as it lands.

    Args:
        items: What is run, e.g. the commands.
        run: Runs one item and returns its ``(label, status, verdict,
            excerpt)``.
        jobs: How many run at once.

    Returns:
        Each item's ``run`` result, in the order of ``items``.
    """
    print(f"running {len(items)} commands, {jobs} at a time\n")
    results = [None] * len(items)
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(run, c): i for i, c in enumerate(items)}
        for done, future in enumerate(as_completed(futures), 1):
            label, status, verdict, excerpt = future.result()
            results[futures[future]] = (label, status, verdict, excerpt)
            suffix = "" if status == "ok" else f" ({status})"
            print(f"[{done}/{len(items)}] {label} ... {verdict}{suffix}")
            sys.stdout.flush()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Location for dated CI artifacts (created if missing).",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SUBSTR",
        help="Run only commands whose <network>/<scheme>/<unrolling> label "
        "contains SUBSTR (case-insensitive; repeatable).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List command labels and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Commands to run at once.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU threads each command may use (OMP / MKL / OpenBLAS); "
        "unset leaves each command its default.",
    )
    args = parser.parse_args()

    commands = COMMANDS
    if args.only:
        commands = [
            c for c in COMMANDS if any(matches(_label(c), p) for p in args.only)
        ]

    if args.list:
        for command in commands:
            _, _, argv = _build(command, Path("<run_dir>"))
            print(" ".join(shlex.quote(a) for a in argv))
        return 0

    if not args.output_dir:
        parser.error("output_dir is required (unless --list)")
    if not commands:
        parser.error(f"--only {args.only} matched no commands")

    run_dir, prev_run = open_run(args.output_dir, TITLE, OUTPUTS)

    env = dict(os.environ)
    if args.threads is not None:
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            env[var] = str(args.threads)

    def run(command):
        label, dest, argv = _build(command, run_dir)
        status = _run_one(label, dest, argv, env)
        failed = status in ("error", "no_output")
        verdict, excerpt = compare(label, failed, run_dir, prev_run, OUTPUTS)
        return label, status, verdict, excerpt

    # (label, status, verdict, diff_excerpt), in command order.
    results = run_all(commands, run, args.jobs)

    report = build_report(results, run_dir, prev_run, TITLE)
    (run_dir / "report.txt").write_text(report)
    print("\n" + report)

    # Gate: fail on any compile failure or model.txt mismatch.
    return 1 if any(r[2] in GATED for r in results) else 0


def build_report(results, run_dir, prev_run, title):
    """Renders the summary written to ``report.txt`` and stdout.

    A label in the previous run that is absent from ``results`` is reported
    MISSING; it is found as the first three path segments of a ``model.txt``
    there, so every label must be three segments and write one.

    Args:
        results: Each command's ``(label, status, verdict, excerpt)``.
        run_dir: This run's folder.
        prev_run: The previous run's folder, or ``None``.
        title: The CI's name, heading the report.

    Returns:
        The report's text, ending in ``RESULT: PASS`` or ``RESULT: FAIL``.
    """
    counts = {}
    for _, _, verdict, _ in results:
        counts[verdict] = counts.get(verdict, 0) + 1

    lines = []
    lines.append("=" * 70)
    lines.append(f"{title} report  ({run_dir.name})")
    if prev_run is not None:
        lines.append(f"compared against: {prev_run.name}")
    else:
        lines.append("compared against: (none - baseline run)")
    lines.append("=" * 70)

    order = ["MATCH", "NEW", "MISMATCH", "FAILED", "MISSING"]
    summary = "  ".join(f"{v}={counts[v]}" for v in order if v in counts)
    lines.append(f"totals: {summary}")
    lines.append("")

    # Detail any non-clean verdicts.
    for label, status, verdict, excerpt in results:
        if verdict in ("MATCH", "NEW"):
            continue
        detail = f"  - {verdict}: {label}"
        if status != "ok":
            detail += f" [{status}; see {label}/run.log]"
        lines.append(detail)
        if excerpt:
            for dl in excerpt.splitlines():
                lines.append(f"      {dl}")

    # Numeric warnings: reported, never gated.
    warned = [r for r in results if r[1] in ("numeric_drift", "unverified")]
    if warned:
        lines.append("")
        lines.append("numeric warnings (not gated):")
        for label, status, _, _ in warned:
            lines.append(f"  - {status}: {label} [see {label}/run.log]")

    # Labels present in the previous run but absent now.
    if prev_run is not None:
        current_labels = {r[0] for r in results}
        # A label is the first three path segments; training nests its
        # programs' model.txt one level further down.
        previous = {
            "/".join(p.relative_to(prev_run).parts[:3])
            for p in prev_run.rglob("model.txt")
        }
        for label in sorted(previous - current_labels):
            lines.append(f"  - MISSING: {label} (in {prev_run.name})")

    bad = any(r[2] in GATED for r in results)
    lines.append("")
    lines.append("RESULT: FAIL" if bad else "RESULT: PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
