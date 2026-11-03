#!/usr/bin/env python
"""Pre-push CI: what your code changes in every compiled CI config.

Compiles every CI command (a ``compile_model.py`` invocation; ``--list``
prints them) with the baseline and with your code, and compares each
Voyager ``model.txt`` after ``tools/canon_model.py`` renames its values,
together with the predicted latency and DRAM bytes that
``compile_model.py --report`` prints (training commands take no
``--report``). Each config that differs is then compiled at every local
commit, to name the commit that changed it.

- The baseline is ``git merge-base HEAD origin/main``, or ``--base``. It is
  built in a git worktree under ``<output_dir>/worktrees``, with the
  commands its own ``run_ci.py --list`` prints; every command runs with
  ``PYTHONPATH`` at its own tree's ``src/``. When that listing fails, every
  config is NEW. A run removes the worktrees an earlier one left.
- Your code is the checkout as it is, uncommitted and new files included.
- Every build, of any version, runs its command exactly as listed, so with
  the ``--debug`` that checks the lowered graph's output against the
  model's. The report gives each build's numeric verdict; a verdict that
  differs between two versions is a change, like one in ``model.txt``.
  When both builds fail the check, the outputs each one prints are
  compared, and printed outputs that differ are a change too.
- Every build is cached under ``<output_dir>/cache/<key>/<label>``. The key
  is git's ids of the sources the commands read (``SOURCES``), uncommitted
  and new files counted as ``git add -A`` would stage them, plus the Python,
  torch and transformers versions. A failed build is not cached, nor is your
  code's when its sources change during the run. A cached build that left
  out its listed ``--debug`` is rebuilt. ``--fresh`` rebuilds your code
  without the cache.
- Attribution walks ``git rev-list --reverse --first-parent base..HEAD``,
  then your uncommitted edits, comparing each with the one before. A commit
  whose sources match its parent's shares its parent's build. It is skipped,
  with a report line, when the baseline is not the first commit's parent.

A run writes ``<output_dir>/<time>/``: ``report.txt``, the uncut cleaned
diff of each config that differs (``<label>/model.diff``) and of each
commit that changed it (``commits/<hash>/<label>.diff``, with
``commits/your_code/`` for the uncommitted edits), and
``changelog_draft.md``. Verdicts: MATCH, MISMATCH, NEW (only in your code),
MISSING (only in the baseline) and FAILED (your code fails to compile). The
run exits non-zero when a command FAILED, or at once when another run holds
``<output_dir>``. A run stopped by SIGTERM, SIGHUP or Ctrl-C ends its
compiles and waits for them before it removes its worktrees.

Usage (from the repo root, with the conda env active)::

    python test/run_ci.py <output_dir> [--jobs N] [--threads T]
    python test/run_ci.py <output_dir> --base <commit> [--fresh]
    python test/run_ci.py --list
"""

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path

# Repo root = parent of this file's directory (test/run_ci.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
COMPILE_MODEL = REPO_ROOT / "scripts" / "compile_model.py"

# The name cleaner is a tool script, not a package.
sys.path.insert(0, str(REPO_ROOT / "tools"))
import canon_model  # noqa: E402

# ---------------------------------------------------------------------------
# Command table.
#
# Each Command is expanded at runtime into a compile_model.py argv:
#     <python> compile_model.py <model> <SCHEME_ARGS[scheme]>
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
        "--gemm_layout xc,ck->xk --scratchpad_size 2097152 "
        "--scratchpad_banks 16 --conv2d_im2col"
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
DEFAULT_TILER_ARGS = {
    "--scratchpad_size": "2097152",
    "--scratchpad_banks": "16",
}

# One activation element, bytes, per scheme.  Each command's
# --scratchpad_word_bytes is one input beat -- the PE array's column count
# times this -- which is the word the store bus writes whole, so the memory
# planner both aligns to it and reserves the tail slack a sub-word store beat
# overshoots by.
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
# MXNF4's vector unit reads a causal flash attention's mask table as int1.
_BOOL_MASK = " --bool_mask"
_LLM_MP = _LLM + " --qconfig mxlut4_int6_e5m3_attn_head_int6"
# A prompt of 960 tokens leaves a 64-token tail past its 128-position chunks.
_LLM_TAIL = _LLM.replace("--context_length 1024", "--context_length 960")
_RESIDUAL = " --residual_length 128"
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

    model: str  # compile_model.py positional argument
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
    Command(
        "llama_prefill", "MXNF4", "64,64", "llama_prefill", _LLM + _BOOL_MASK
    ),
    Command(
        "llama_prefill",
        "MXNF4",
        "64,64",
        "llama_prefill_mp",
        _LLM_MP + _BOOL_MASK,
    ),
    Command(
        "llama_prefill",
        "MXNF4",
        "64,64",
        "llama_prefill_spmm",
        _LLM_SPMM + _BOOL_MASK,
    ),
    Command(
        "llama_decode", "MXNF4", "64,64", "llama_decode", _LLM + _BOOL_MASK
    ),
    Command(
        "llama_decode_kivi",
        "MXNF4",
        "64,64",
        "llama_decode_kivi",
        _LLM + _RESIDUAL + _BOOL_MASK,
    ),
    # -- the prefill that fills the decode's KV cache, compiled with it: a
    # 128-position residual with a tail of 0 and of 64 tokens, the latter
    # under flash attention too, and the split each cache's blocks ask for --
    Command(
        "llama_cache",
        "MXNF4",
        "64,64",
        "llama_cache",
        _LLM + _RESIDUAL + _BOOL_MASK,
    ),
    Command(
        "llama_cache",
        "MXNF4",
        "64,64",
        "llama_cache_tail",
        _LLM_TAIL + _RESIDUAL + _BOOL_MASK,
    ),
    Command(
        "llama_cache",
        "MXNF4",
        "64,64",
        "llama_cache_sdpa",
        _LLM_TAIL + _RESIDUAL + " --attn_implementation sdpa" + _BOOL_MASK,
    ),
    # The Voyager test set's llama_prefill + llama_decode pair.
    Command(
        "llama_cache",
        "MXNF4",
        "64,64",
        "llama_cache_blocks",
        _LLM + _BOOL_MASK,
    ),
    # The Voyager test set's llama_prefill_mp + llama_decode_kivi pair: one
    # qconfig, the decode's cache KIVI.
    Command(
        "llama_cache_kivi",
        "MXNF4",
        "64,64",
        "llama_pair_mp",
        _LLM_MP + _RESIDUAL + _BOOL_MASK,
    ),
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
    Command(
        "llama_prefill",
        "MXNF4",
        "64,64",
        "llama_prefill_db",
        _LLM_DB + _BOOL_MASK,
    ),
    Command(
        "llama_decode",
        "MXNF4",
        "64,64",
        "llama_decode_db",
        _LLM_DB + _BOOL_MASK,
    ),
    Command(
        "llama_prefill",
        "MXNF4",
        "64,64",
        "llama_prefill_spmm_db",
        _LLM_SPMM_DB + _BOOL_MASK,
    ),
    # -- training: one AdamW step, compiled into gradient/ and update/ --
    Command("llama_prefill", "BF16", "64,64", "llama_train", _LLM_TRAIN),
    Command("llama_prefill", "FP8", "64,64", "llama_train", _LLM_TRAIN),
    Command("llama_prefill", "MXFP4", "64,64", "llama_train", _LLM_TRAIN),
]

# Timestamp folder format; lexicographic sort == chronological sort.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"
# The output folder ``--list`` names: ``<RUN_DIR>/<label>``.
RUN_DIR = "<run_dir>"
# What the commands read, relative to the repo root: the package, the
# driver and its loaders (under test/ at older commits), and the qconfig
# table scripts/models/llama.py imports from the
# language-modeling example.
SOURCES = (
    "src",
    "scripts",
    "test/compile_model.py",
    "test/utils",
    "examples/language_modeling/quantization_configs.py",
)
# What compile_model.py prints before the outputs of a failed numeric check.
FAILED_CHECK = "WARNING: output verification failed"
# A number in that printout; comparing only the numbers keeps a change of
# output structure (a bare tensor for a one-element tuple) from counting.
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
# The predicted cost ``compile_model.py --report`` prints.
REPORT_LINE = re.compile(
    r"\[report\] total_latency=(?P<total_latency>\S+) "
    r"dram_read=(?P<dram_read>\d+) dram_write=(?P<dram_write>\d+)"
)
VERDICTS = ("MATCH", "MISMATCH", "NEW", "MISSING", "FAILED")
# The signals that stop a run.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# Build statuses that leave nothing to compare.
BROKEN = ("error", "no_output")


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

    The argv runs this repo's compile_model.py with --model_output_dir pointed
    into the run folder; --dump_tensors is never added. ``--debug`` is always
    added: without it compile_model never evaluates the lowered graph, so the
    run checks codegen only and no numeric comparison happens at all.
    """
    label = _label(command)
    dest = run_dir / label

    argv = [sys.executable, str(COMPILE_MODEL), command.model, "--debug"]
    argv += shlex.split(SCHEME_ARGS[command.scheme])
    argv += ["--pe_array_size", command.unrolling]
    argv += shlex.split(command.extra)
    for flag, value in DEFAULT_TILER_ARGS.items():
        if flag not in argv:
            argv += [flag, value]
    if "--scratchpad_word_bytes" not in argv:
        cols = int(command.unrolling.split(",")[1])
        width = int(cols * SCHEME_INPUT_BYTES[command.scheme])
        argv += ["--scratchpad_word_bytes", str(width)]
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


def parse_listing(text):
    """Reads ``run_ci.py --list`` output: ``{label: argv}``."""
    commands = {}
    for line in text.splitlines():
        argv = shlex.split(line)
        dest = argv[argv.index("--model_output_dir") + 1]
        commands[dest.removeprefix(RUN_DIR + "/")] = argv
    return commands


def git(*args, env=None, check=True):
    """Runs git in the repo and returns its stripped stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def describe(commit):
    """``<short hash> <subject>`` of a commit."""
    return git("log", "-1", "--format=%h %s", commit)


def source_key(trees):
    """The cache key of a source tree.

    Args:
        trees: ``git ls-tree`` lines naming git's id of each of ``SOURCES``
            in the tree.

    Returns:
        A hex digest of them and of the Python, torch and transformers
        versions the commands run with.
    """
    versions = [
        platform.python_version(),
        metadata.version("torch"),
        metadata.version("transformers"),
    ]
    text = "\n".join([trees, *versions])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def commit_key(commit):
    """The cache key of a commit's sources; a path it lacks is skipped."""
    return source_key(git("ls-tree", commit, "--", *SOURCES))


def checkout_key(scratch):
    """The cache key of the checkout's sources as ``git add -A`` stages them.

    The files are staged into a copy of the index whose new objects go to a
    scratch object folder, so neither the real index nor the repository's
    objects change.

    Args:
        scratch: A folder for the index copy and objects, emptied after.
    """
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        objects = Path(tmp, "objects")
        objects.mkdir()
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(Path(tmp, "index"))
        env["GIT_OBJECT_DIRECTORY"] = str(objects)
        env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(
            REPO_ROOT / git("rev-parse", "--git-path", "objects")
        )
        shutil.copy(REPO_ROOT / git("rev-parse", "--git-path", "index"), tmp)
        present = [
            p
            for p in SOURCES
            if (REPO_ROOT / p).exists() or git("ls-files", "--", p)
        ]
        git("add", "-A", "--", *present, env=env)
        tree = git("write-tree", env=env)
        return source_key(git("ls-tree", tree, "--", *SOURCES, env=env))


@dataclass
class Version:
    """A source tree whose commands are compiled.

    Attributes:
        title: How the report names it: a short hash, or ``your code``.
        key: Its sources' cache key.
        commit: The commit it is, or ``None`` for your checkout.
        root: Where its files are; set when one of its commands compiles.
        commands: ``{label: argv}``, as its ``run_ci.py --list`` prints.
        error: Why its commands could not be listed, if they could not.
    """

    title: str
    key: str
    commit: str | None
    root: Path | None = None
    commands: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class Build:
    """One command compiled from one version.

    Attributes:
        status: ``ok``; ``error`` (it exited non-zero, or its version could
            not list its commands); ``no_output`` (it wrote no
            ``model.txt``); ``numeric_drift`` or ``unverified`` (its lowered
            graph left compile_model's tolerance, or was not checked, as in a
            commit whose commands list no ``--debug``); or ``absent`` (its
            version has no such command).
        dest: Its output folder, or ``None`` when it never ran.
        report: The predicted ``total_latency``, ``dram_read`` and
            ``dram_write``, or ``None`` when it printed none.
    """

    status: str
    dest: Path | None = None
    report: dict | None = None

    @property
    def broken(self):
        """Whether it left nothing to compare."""
        return self.status in BROKEN


class Cache:
    """Builds and command listings saved by source key."""

    def __init__(self, folder):
        self.folder = folder

    def load(self, key, label, argv):
        """The cached build of ``label`` under ``key``, or ``None``.

        Args:
            key: The version's source key.
            label: The command's label.
            argv: The command as the version lists it. A build whose
                ``run.log`` shows it ran without the listed ``--debug`` is
                not served.
        """
        dest = self.folder / key / label
        if not (dest / "result.json").exists():
            return None
        with open(dest / "run.log") as log:
            ran = shlex.split(log.readline().removeprefix("$ "))
        if "--debug" in argv and "--debug" not in ran:
            return None
        result = json.loads((dest / "result.json").read_text())
        return Build(result["status"], dest, result["report"])

    def store(self, key, label, build):
        """Moves a finished build into the cache and returns it there."""
        dest = self.folder / key / label
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(build.dest, dest)
        result = {"status": build.status, "report": build.report}
        (dest / "result.json").write_text(json.dumps(result))
        return Build(build.status, dest, build.report)

    def listing(self, key, text=None):
        """Reads, or with ``text`` saves, the command listing of ``key``."""
        path = self.folder / key / "commands.txt"
        if text is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return path.read_text() if path.exists() else None


class Worktrees:
    """Git worktrees of commits under one folder.

    Entering removes any worktree an earlier run left in the folder; leaving
    removes this run's.
    """

    def __init__(self, folder):
        self.folder = folder
        self.paths = {}

    def __enter__(self):
        if self.folder.exists():
            for path in self.folder.iterdir():
                git("worktree", "remove", "--force", str(path), check=False)
                shutil.rmtree(path, ignore_errors=True)
        git("worktree", "prune")
        return self

    def __exit__(self, *exc):
        for path in self.paths.values():
            git("worktree", "remove", "--force", str(path), check=False)
        git("worktree", "prune", check=False)

    def add(self, commit):
        """Checks out ``commit`` (once) and returns its folder."""
        if commit not in self.paths:
            path = self.folder / commit[:12]
            git("worktree", "add", "--detach", "--quiet", str(path), commit)
            self.paths[commit] = path
        return self.paths[commit]


@contextmanager
def locked(folder):
    """Holds ``folder``'s lock, so no two runs share it."""
    handle = open(folder / ".lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"another run_ci.py is using {folder}")
    try:
        yield
    finally:
        handle.close()


class Compiles:
    """The compiles a run starts, so that a stopped run can end them.

    Each compile runs in a session of its own, so its process group holds it
    and whatever it starts. A group is signalled only while its leader is
    unreaped, so its id cannot have passed to another process.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.running = set()
        self.stopping = False

    def run(self, argv, **kwargs):
        """Runs a compile to its end.

        Args:
            argv: The command.
            **kwargs: Passed to ``subprocess.Popen``.

        Returns:
            Its exit code, or ``-SIGTERM`` when the run is stopping, in which
            case it is not started.
        """
        with self.lock:
            if self.stopping:
                return -signal.SIGTERM
            proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
            self.running.add(proc.pid)
        # Wait without reaping: the group id stays this run's until it
        # leaves ``running``.
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        with self.lock:
            if self.stopping:
                # End anything it started and left running.
                kill_group(proc.pid, signal.SIGKILL)
            self.running.discard(proc.pid)
        return proc.wait()

    def stop(self, grace=10):
        """Ends the running compiles, waits for them, and starts no more.

        Args:
            grace: Seconds a compile has to exit on SIGTERM before SIGKILL.
        """
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
            with self.lock:
                self.stopping = True
                for pid in self.running:
                    kill_group(pid, sig)
            deadline = time.time() + wait if wait else float("inf")
            while self.running and time.time() < deadline:
                time.sleep(0.1)


def kill_group(pid, sig):
    """Sends ``sig`` to the process group led by ``pid``, if it remains."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


COMPILES = Compiles()


def stop_on_signal(signum, frame):
    """Stops the run, leaving later stop signals no effect.

    Later signals reach a handler that does nothing, so the cleanup, which
    ends the compiles before the worktrees go, runs whole. A handler, unlike
    ``SIG_IGN``, is not inherited by a compile started meanwhile, so that
    compile still ends on SIGTERM.

    Args:
        signum: The signal received.
        frame: Unused; the signal handler's signature requires it.
    """
    for sig in STOP_SIGNALS:
        if signal.getsignal(sig) is stop_on_signal:
            signal.signal(sig, lambda *_: None)
    # Not print(): the handler may interrupt a print to the same stream.
    name = signal.Signals(signum).name
    os.write(sys.stderr.fileno(), f"\nstopped by {name}\n".encode())
    raise SystemExit(128 + signum)


def tree_env(root, threads):
    """The environment a version's commands run in."""
    env = dict(os.environ)
    paths = [str(root / "src"), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    if threads is not None:
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            env[var] = str(threads)
    return env


def list_commands(version, cache, trees):
    """Fills in a commit's commands, from the cache or its own ``--list``.

    Args:
        version: The commit's ``Version``; its worktree is made if needed.
        cache: The ``Cache``.
        trees: The ``Worktrees``.
    """
    text = cache.listing(version.key)
    if text is None:
        version.root = trees.add(version.commit)
        proc = subprocess.run(
            [sys.executable, "test/run_ci.py", "--list"],
            cwd=version.root,
            env=tree_env(version.root, None),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            version.error = f"its run_ci.py --list failed:\n{proc.stderr}"
            return
        text = cache.listing(version.key, proc.stdout)
    version.commands = parse_listing(text)


def compile_one(version, label, dest, threads):
    """Runs a version's command into ``dest``.

    Args:
        version: The ``Version``; its ``root`` is set.
        label: The command's label.
        dest: The output folder.
        threads: CPU threads the command may use, or ``None``.

    Returns:
        The ``Build``.
    """
    root = version.root
    codegen = root / "scripts" / "compile_model.py"
    if not codegen.exists():
        codegen = root / "test" / "test_codegen.py"
    argv = list(version.commands[label])
    argv[:2] = [sys.executable, str(codegen)]
    argv[argv.index("--model_output_dir") + 1] = str(dest)
    prints_report = "total_latency=" in codegen.read_text()
    if prints_report and "--training" not in argv:
        argv.append("--report")

    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "run.log"
    with open(log_path, "w") as log:
        log.write(f"$ {shlex.join(argv)}\n\n")
        log.flush()
        code = COMPILES.run(
            argv,
            cwd=root,
            env=tree_env(root, threads),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log = log_path.read_text()
    found = REPORT_LINE.search(log)
    report = found.groupdict() if found else None
    if code != 0:
        status = "error"
    elif not outputs(dest, ("model.txt",)):
        status = "no_output"
    elif "Skipping output verification" in log:
        status = "unverified"
    elif "Results match" not in log:
        status = "numeric_drift"
    else:
        status = "ok"
    return Build(status, dest, report)


def build_all(requests, cache, trees, staging, args):
    """Compiles the requested builds that are not cached.

    A key and label requested by several versions compile once, by a commit
    when one asks: a commit's builds are cached as they finish, while your
    code's stay in ``staging`` for the caller to cache.

    Args:
        requests: ``(version, label, fresh)`` triples; ``fresh`` skips the
            cache.
        cache: The ``Cache``.
        trees: The ``Worktrees``.
        staging: The folder builds are made in.
        args: The parsed command line (``jobs``, ``threads``).

    Returns:
        ``{(version title, label): Build}`` and the number compiled.
    """
    builds = {}
    todo = {}
    for version, label, fresh in requests:
        if version.error:
            builds[version.title, label] = Build("error")
            continue
        if label not in version.commands:
            builds[version.title, label] = Build("absent")
            continue
        argv = version.commands[label]
        cached = None if fresh else cache.load(version.key, label, argv)
        if cached:
            builds[version.title, label] = cached
        else:
            todo.setdefault((version.key, label, fresh), []).append(version)
    for versions in todo.values():
        versions.sort(key=lambda v: v.commit is None)
        if versions[0].root is None:
            versions[0].root = trees.add(versions[0].commit)

    def run(item):
        (key, label, fresh), versions = item
        maker = versions[0]
        dest = staging / maker.title.replace(" ", "_") / label
        build = compile_one(maker, label, dest, args.threads)
        if maker.commit is not None and not fresh and not build.broken:
            build = cache.store(key, label, build)
        return build

    started = time.time()
    pool = ThreadPoolExecutor(args.jobs)
    try:
        futures = {pool.submit(run, item): item for item in todo.items()}
        for done, future in enumerate(as_completed(futures), 1):
            (_, label, _), versions = futures[future]
            build = future.result()
            for version in versions:
                builds[version.title, label] = build
            names = ", ".join(v.title for v in versions)
            print(
                f"[{done}/{len(todo)}] {names}: {label} ... {build.status} "
                f"({time.time() - started:.0f} s)",
                flush=True,
            )
    except BaseException:
        # Stopped or crashed: end the running compiles, which the pool
        # would otherwise wait out.
        COMPILES.stop()
        raise
    finally:
        pool.shutdown(cancel_futures=True)
    return builds, len(todo)


@dataclass
class Delta:
    """How one build of a config differs from another.

    Attributes:
        old: The earlier build.
        new: The later build.
        equal: Whether they compare equal: the same numeric verdict, the
            same printed outputs when both failed the check, the same
            programs, each with equal cleaned text and addresses, and the
            same predicted cost when the old build printed one (the new one
            must too).
        note: What changed when a side has nothing to compare.
        numerics: ``<old status> -> <new status>`` when the numeric verdict
            changed, else ``""``.
        outputs_changed: Whether both failed the numeric check and their
            logs print different numbers for the outputs.
        programs: ``{program: canon_model.Comparison}`` for each program
            both wrote.
        only: The programs one build wrote and the other did not.
        diff: The unified diff of the cleaned texts, uncut.
    """

    old: Build
    new: Build
    equal: bool = True
    note: str = ""
    numerics: str = ""
    outputs_changed: bool = False
    programs: dict = field(default_factory=dict)
    only: list = field(default_factory=list)
    diff: list = field(default_factory=list)


def delta(old, new):
    """Compares two builds of one config (see ``Delta``)."""
    if "absent" in (old.status, new.status):
        if old.status == new.status:
            return Delta(old, new)
        note = "added" if old.status == "absent" else "removed"
        return Delta(old, new, False, note)
    if old.broken or new.broken:
        if old.broken and new.broken:
            return Delta(old, new, True, "still fails to compile")
        note = "compiles again" if old.broken else "fails to compile"
        return Delta(old, new, False, note)

    result = Delta(old, new)
    if old.status != new.status:
        result.numerics = f"{old.status} -> {new.status}"
    elif old.status == "numeric_drift":
        old_out, new_out = (
            NUMBER.findall(
                (b.dest / "run.log").read_text().partition(FAILED_CHECK)[2]
            )
            for b in (old, new)
        )
        result.outputs_changed = old_out != new_out
    olds = outputs(old.dest, ("model.txt",))
    news = outputs(new.dest, ("model.txt",))
    result.only = sorted(map(str, set(olds) ^ set(news)))
    for program in sorted(set(olds) & set(news)):
        before = canon_model.canonicalize((old.dest / program).read_text())
        after = canon_model.canonicalize(
            (new.dest / program).read_text(), before
        )
        result.programs[str(program)] = canon_model.compare(before, after)
        result.diff += difflib.unified_diff(
            before.text.splitlines(),
            after.text.splitlines(),
            f"old/{program}",
            f"new/{program}",
            lineterm="",
        )
    result.equal = (
        not result.numerics
        and not result.outputs_changed
        and (old.report is None or old.report == new.report)
        and not result.only
        and all(c.equal for c in result.programs.values())
    )
    return result


LOST_COST = "no predicted cost, though the old build printed one"


def lost_cost(change):
    """Whether a ``Delta``'s new build compiled but lost the old one's cost."""
    return (
        bool(change.old.report)
        and not change.new.report
        and not change.new.broken
    )


def verdict(change):
    """The verdict on a config's ``Delta`` from the baseline to your code."""
    if change.new.status == "absent":
        return "MISSING"
    if change.new.broken:
        return "FAILED"
    if change.old.status == "absent":
        return "NEW"
    return "MATCH" if change.equal else "MISMATCH"


def cost(old, new):
    """The predicted cost of two builds, old -> new, or ``""``."""
    if not (old.report and new.report):
        return ""
    parts = []
    for key, unit in (
        ("total_latency", "cycles"),
        ("dram_read", "B"),
        ("dram_write", "B"),
    ):
        a, b = old.report[key], new.report[key]
        x, y = float(a), float(b)
        change = f" ({(y - x) / x:+.2%})" if x and x != y else ""
        parts.append(f"{key} {a} -> {b} {unit}{change}")
    return "; ".join(parts)


def explain(change):
    """Report lines for a ``Delta`` that is not equal."""
    lines = [change.note] if change.note else []
    if change.numerics:
        lines.append(f"numeric verdict: {change.numerics}")
    if change.outputs_changed:
        lines.append("printed outputs differ (both failed the numeric check)")
    if lost_cost(change):
        lines.append(LOST_COST)
    if change.only:
        lines.append(f"programs in one build only: {', '.join(change.only)}")
    if change.programs:
        edits = sum(
            1
            for line in change.diff
            if line[:1] in "+-" and line[:3] not in ("+++", "---")
        )
        lines.append(f"model.txt: {edits} cleaned lines changed")
    for program, result in change.programs.items():
        prefix = "" if len(change.programs) == 1 else f"{program}: "
        lines += [prefix + line for line in canon_model.summary(result)]
    if cost(change.old, change.new):
        lines.append(f"predicted: {cost(change.old, change.new)}")
    return lines


@dataclass
class Step:
    """One version of the attribution walk, compared with the one before.

    Attributes:
        version: A local commit, or your code for its uncommitted edits.
        title: How the change log names it.
        against: The title of the version it is compared with.
        shared: Its sources match that version's, so it shares its builds.
        changes: ``{label: Delta}`` for each traced config it changed.
        failed: The traced configs it failed to build.
        builds: ``{label: Build}`` for each traced config, unless shared.
    """

    version: Version
    title: str
    against: str = ""
    shared: bool = False
    changes: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)
    builds: dict = field(default_factory=dict)


def attribute(base, yours, commits, traced, cache, trees, run_dir, args):
    """Finds the commit, or the uncommitted edits, that changed each config.

    Args:
        base: The baseline's ``Version``.
        yours: Your code's ``Version``.
        commits: The local commits, oldest first; the first one's parent is
            the baseline and each other's the one before it.
        traced: ``{label: (baseline build, your build)}`` for each config
            that differs end to end.
        cache: The ``Cache``.
        trees: The ``Worktrees``.
        run_dir: This run's folder.
        args: The parsed command line.

    Returns:
        The ``Step`` list, oldest first, and the number of builds compiled.
    """
    steps = []
    for commit in commits:
        version = Version(git("rev-parse", "--short", commit), "", commit)
        version.key = commit_key(commit)
        steps.append(Step(version, describe(commit)))
    if yours.key != (steps[-1].version if steps else base).key:
        steps.append(Step(yours, "your uncommitted edits"))

    requests = []
    previous = base
    for step in steps:
        step.against = previous.title
        step.shared = step.version.key == previous.key
        if not step.shared and step.version is not yours:
            list_commands(step.version, cache, trees)
            requests += [(step.version, label, False) for label in traced]
        previous = step.version
    builds, compiled = build_all(
        requests, cache, trees, run_dir / "build", args
    )

    for label, (reference, end) in traced.items():
        good = None if reference.broken else reference
        good_title = base.title
        for step in (s for s in steps if not s.shared):
            title = step.version.title
            build = end if step.version is yours else builds[title, label]
            step.builds[label] = build
            change = delta(reference, build)
            if build.broken:
                step.failed.append(label)
            if reference.broken and not build.broken and good is not None:
                # Compiles again: say whether it matches the last good build.
                change = delta(good, build)
                change.note = (
                    "compiles again, "
                    + ("same as" if change.equal else "differs from")
                    + f" the last good build ({good_title})"
                )
                change.equal = False
            if not change.equal:
                step.changes[label] = change
                folder = run_dir / "commits" / title.replace(" ", "_")
                path = folder / f"{label}.diff"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(change.diff) + "\n")
            reference = build
            if not build.broken:
                good, good_title = build, title
    return steps, compiled


def changelog(steps, base, traced, skipped):
    """Drafts the change-log entries of the steps, newest first."""
    lines = [
        f"<!-- Drafted by test/run_ci.py against {base.title}. Only the "
        "configs that differ from the baseline to your code are traced per "
        "commit. Write each why by hand. -->",
        "",
    ]
    if skipped:
        lines += [f"Attribution skipped: {skipped}", ""]
    elif not traced:
        lines += ["No config differs from the baseline.", ""]
    for step in reversed(steps):
        commit = step.version.commit
        if commit:
            date = git("log", "-1", "--format=%ad", "--date=short", commit)
            lines += [f"## {step.title} ({date})", ""]
        else:
            lines += [f"## {step.title} (on {step.against})", ""]
        changed = [
            f"{label} ({c.note})" if c.note else label
            for label, c in sorted(step.changes.items())
            if (
                c.note
                or c.only
                or not all(p.equal for p in c.programs.values())
            )
        ]
        if step.shared:
            lines.append("- model.txt: no compiled source changed.")
        elif changed:
            lines.append(
                f"- model.txt: changed in {', '.join(changed)} "
                f"(vs {step.against})."
            )
            lines.append("  Why: (write by hand)")
        else:
            lines.append("- model.txt: unchanged in the traced configs.")
        costs = [
            f"{label} {cost(c.old, c.new)}"
            for label, c in step.changes.items()
            if cost(c.old, c.new) and c.old.report != c.new.report
        ]
        costs += [
            f"{label} {LOST_COST}"
            for label, c in sorted(step.changes.items())
            if lost_cost(c)
        ]
        if costs:
            others = "; others unchanged" if len(costs) < len(traced) else ""
            lines.append(f"- Predicted latency: {'; '.join(costs)}{others}.")
        else:
            lines.append("- Predicted latency: unchanged.")
        numerics = [
            f"{label} {c.numerics}"
            for label, c in sorted(step.changes.items())
            if c.numerics
        ]
        if numerics:
            lines.append(f"- Numeric verdict: {'; '.join(numerics)}.")
        if step.failed:
            lines.append(f"- Failed to build: {', '.join(step.failed)}.")
        lines += ["- Measured: none.", ""]
    return "\n".join(lines)


def config_lines(changes, verdicts, base, steps):
    """The report's lines on each config and on the attribution builds.

    Args:
        changes: ``{label: Delta}``, baseline to your code.
        verdicts: ``{label: verdict}``.
        base: The baseline's ``Version``.
        steps: The attribution ``Step`` list.

    Returns:
        The lines, one config after another.
    """
    lines = []
    for label in changes:
        change = changes[label]
        lines.append(f"{verdicts[label]:<9} {label}")
        # Each build of it, as (verdict name, log name, build): the
        # baseline's, each attribution build's, and your code's.
        statuses = [(base.title, "baseline", change.old)]
        statuses += [
            (s.version.title, s.version.title, s.builds[label])
            for s in steps
            if label in s.builds and s.version.commit is not None
        ]
        statuses.append(("your code", "your", change.new))
        lines.append(
            "    numerics: "
            + "; ".join(f"{title} {b.status}" for title, _, b in statuses)
        )
        if not change.equal:
            lines += [f"    {line}" for line in explain(change)]
            causes = [s.title for s in steps if label in s.changes]
            if causes:
                lines.append(f"    caused by: {'; '.join(causes)}")
            if change.diff:
                lines.append(f"    diff: {label}/model.diff")
        elif cost(change.old, change.new):
            lines.append(f"    predicted: {cost(change.old, change.new)}")
        # The log of each build whose check did not pass.
        for _, name, b in statuses:
            if b.status in ("ok", "absent"):
                continue
            log = f"; see {b.dest / 'run.log'}" if b.dest else ""
            lines.append(f"    {name} build: {b.status}{log}")
    for step in steps:
        for label in step.failed:
            lines.append(
                f"attribution: {step.version.title} failed to build {label}"
            )
        if step.version.error:
            lines.append(
                f"attribution: {step.version.title}: {step.version.error}"
            )
    return lines


def run(args, out, commands, trees):
    """Compiles, compares and reports; returns the exit code."""
    run_dir = out / datetime.now().strftime(TS_FORMAT)
    run_dir.mkdir()
    cache = Cache(out / "cache")
    head = git("rev-parse", "HEAD")
    if args.base:
        base_commit = git("rev-parse", "--verify", f"{args.base}^{{commit}}")
        origin = f"--base {args.base}"
    else:
        base_commit = git("merge-base", "HEAD", "origin/main")
        origin = "merge-base of HEAD and origin/main"
    base = Version(
        git("rev-parse", "--short", base_commit),
        commit_key(base_commit),
        base_commit,
    )
    yours = Version("your code", checkout_key(out), None, REPO_ROOT)
    for command in commands:
        label, _, argv = _build(command, Path(RUN_DIR))
        yours.commands[label] = argv
    dirty = yours.key != commit_key(head)

    lines = [
        f"voyager codegen CI  ({run_dir.name})",
        f"baseline:  {describe(base_commit)}  ({origin})",
        f"your code: {describe(head)}"
        + ("  + uncommitted edits to the sources" if dirty else ""),
    ]
    print("\n".join(lines) + f"\n-> {run_dir}\n", flush=True)

    list_commands(base, cache, trees)
    unlisted = base.error
    if unlisted:
        lines.append(f"baseline {base.title}: {unlisted.rstrip()}")
        lines.append("every config is compared as NEW")
        # Cleared so that its builds count as absent (NEW), not failed.
        base.error = ""
    labels = list(yours.commands) + [
        label
        for label in base.commands
        if label not in yours.commands
        and (not args.only or any(matches(label, p) for p in args.only))
    ]
    requests = [(base, label, False) for label in labels]
    requests += [(yours, label, args.fresh) for label in labels]
    builds, compiled = build_all(
        requests, cache, trees, run_dir / "build", args
    )

    steady = checkout_key(out) == yours.key
    for label in labels:
        build = builds[yours.title, label]
        staged = build.dest is not None and run_dir in build.dest.parents
        if steady and not args.fresh and staged and not build.broken:
            builds[yours.title, label] = cache.store(yours.key, label, build)

    changes = {}
    verdicts = {}
    for label in labels:
        change = delta(builds[base.title, label], builds[yours.title, label])
        changes[label] = change
        verdicts[label] = verdict(change)
        if change.diff:
            (run_dir / label).mkdir(parents=True, exist_ok=True)
            diff = "\n".join(change.diff) + "\n"
            (run_dir / label / "model.diff").write_text(diff)

    traced = {
        label: (change.old, change.new)
        for label, change in changes.items()
        if not change.equal
    }
    steps = []
    skipped = ""
    if traced:
        commits = git(
            "rev-list", "--reverse", "--first-parent", f"{base_commit}..{head}"
        ).split()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_commit, head],
            cwd=REPO_ROOT,
        )
        if unlisted:
            skipped = f"the baseline {base.title} could not list its commands"
        elif ancestor.returncode:
            skipped = f"the baseline {base.title} is not an ancestor of HEAD"
        elif commits and git("rev-parse", f"{commits[0]}^") != base_commit:
            skipped = (
                f"the first local commit {commits[0][:7]} does not have the "
                f"baseline {base.title} as its parent (a merge brought the "
                "baseline in)"
            )
        else:
            steps, more = attribute(
                base, yours, commits, traced, cache, trees, run_dir, args
            )
            compiled += more
    (run_dir / "changelog_draft.md").write_text(
        changelog(steps, base, traced, skipped)
    )

    counts = {v: list(verdicts.values()).count(v) for v in VERDICTS}
    lines += [
        f"builds:    {compiled} compiled; the rest came from the cache"
        + ("; your code was built fresh" if args.fresh else ""),
        "totals:    " + "  ".join(f"{v}={n}" for v, n in counts.items() if n),
    ]
    tallies = []
    for title, statuses in (
        (base.title, [c.old.status for c in changes.values()]),
        ("your code", [c.new.status for c in changes.values()]),
    ):
        tally = " ".join(
            f"{s}={statuses.count(s)}" for s in sorted(set(statuses))
        )
        tallies.append(f"{title} {tally}")
    lines.append("numerics:  " + "; ".join(tallies))
    if not steady:
        lines.append(
            "your sources changed during the run: their builds were not "
            "cached"
        )
    if skipped:
        lines.append(f"attribution skipped: {skipped}")
    lines.append("")
    lines += config_lines(changes, verdicts, base, steps)
    failed = counts["FAILED"]
    lines += [
        "",
        f"RESULT: FAIL ({failed} FAILED)" if failed else "RESULT: PASS",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "report.txt").write_text(report)
    print("\n" + report)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Folder for runs, the build cache and worktrees (created if "
        "missing); one run uses it at a time.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SUBSTR",
        help="Run only commands whose <network>/<scheme>/<unrolling> label "
        "SUBSTR selects (see matches(); repeatable).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print each command's argv, one per line, and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Commands to compile at once.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU threads each command may use (OMP / MKL / OpenBLAS); "
        "unset leaves each command its default.",
    )
    parser.add_argument(
        "--base",
        help="The commit to compare with (default: git merge-base HEAD "
        "origin/main).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rebuild your code without the cache; with --base HEAD on a "
        "clean tree, checks that a compile repeats.",
    )
    args = parser.parse_args()

    commands = COMMANDS
    if args.only:
        commands = [
            c for c in COMMANDS if any(matches(_label(c), p) for p in args.only)
        ]

    if args.list:
        for command in commands:
            _, _, argv = _build(command, Path(RUN_DIR))
            print(shlex.join(argv))
        return 0

    if not args.output_dir:
        parser.error("output_dir is required (unless --list)")
    if not commands:
        parser.error(f"--only {args.only} matched no commands")

    # Exit through the cleanup below when the run is stopped. A signal
    # already ignored, as nohup ignores SIGHUP, stays ignored.
    for sig in STOP_SIGNALS:
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, stop_on_signal)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with locked(out), Worktrees(out / "worktrees") as trees:
        return run(args, out, commands, trees)


if __name__ == "__main__":
    sys.exit(main())
