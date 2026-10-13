"""Check a translation's NKI kernels in the NKI CPU simulator.

Runs in the ``trn`` env (Neuron SDK, torch 2.9), which cannot import
``voyager_compiler``; run it by path, with ``~/claude/trn/env.sh`` sourced:

    python check.py <dir>              # simulate every kernel call
    python check.py <dir> --distinct   # one call of each distinct kernel
    python check.py <dir> --program    # run program.py end to end
    python check.py <dir> --neff --target trn2   # compile each distinct one

``<dir>`` holds ``translate``'s output beside ``tensors.pt``, the value of
every HBM box -- the inputs, the parameters and each kernel's results, as
Voyager's own bufferized graph computed them.  A kernel call runs on the
values of its inputs and is compared with the values of its outputs; the
program runs on the inputs and parameters alone.  The error is the largest
absolute difference over the largest reference magnitude; an MX output's
is taken block by block (``mx_error``).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import ml_dtypes
import nki
import numpy as np
import torch

# A bfloat16 result rounded once more is off by about 2**-8 of its largest
# magnitude.  A GEMM whose reduction spans several steps sums its partial
# results in bfloat16, as Voyager's schedule does, and two such sums rounded
# in another order differ by up to about 1.5%; more is a lowering error.
MAX_ERROR = 2e-2
# An FP8 result may round one step the other way, and a step is at most
# 2**-m of the largest magnitude, m the format's mantissa bits.  An MX output
# -- codes and their block scales -- is held to the same, as values, block by
# block against the block's largest reference magnitude: where Voyager's
# bfloat16 log2 rounds a block's maximum up, its scale is twice the device's
# and its codes half, the same values.
MAX_FP8_ERROR = {"fp8_e4m3": 2**-3, "fp8_e4m3_ieee": 2**-3, "fp8_e5m2": 2**-2}
# The cosine similarity a whole program's output keeps with the reference;
# lower for a program with MX arrays, by element format, whose elements
# round one step the other way wherever their BF16 input differs by a
# rounding, flips that add up over its quantizations.  The shortfall from 1
# grows with the square of the format's step: four times for E5M2's.
MIN_COSINE = 0.999
MIN_MX_COSINE = {"fp8_e4m3": 0.998, "fp8_e5m2": 0.992}
# The ``translate`` layout of an MX array (``Mx``): a codes word packs 4
# consecutive contraction elements, 128 of them hold a chunk of 512, and a
# chunk's 16 block scales are 16 rows of bytes.
MX_PACK = 4
MX_BLOCK = 32
MX_ELEMENTS = {
    "fp8_e4m3": ml_dtypes.float8_e4m3fn,
    "fp8_e5m2": ml_dtypes.float8_e5m2,
}


# The torch type a kernel takes a tensor of each IR type as, as
# ``program.py`` passes it: torch has no IEEE E4M3 (``fp8_e4m3_ieee``,
# Trn2's), and its e4m3fn has the same bits up to that format's 240.
KERNEL_TYPES = {
    "int64": torch.int32,
    "bool": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e4m3_ieee": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
# Torch types numpy holds through ``ml_dtypes``, bit for bit: the integer
# type of their bits and the ``ml_dtypes`` type.
NUMPY_TYPES = {
    torch.bfloat16: (torch.int16, ml_dtypes.bfloat16),
    torch.float8_e4m3fn: (torch.uint8, ml_dtypes.float8_e4m3),
    torch.float8_e5m2: (torch.uint8, ml_dtypes.float8_e5m2),
}


def to_kernel(tensor, dtype):
    """``tensor``, of IR type ``dtype``, as a kernel takes it."""
    return tensor.to(KERNEL_TYPES.get(dtype, tensor.dtype))


def to_numpy(tensor):
    """``tensor`` as a numpy array, a type numpy lacks kept bit for bit."""
    tensor = tensor.contiguous()
    if tensor.dtype in NUMPY_TYPES:
        bits, numpy_type = NUMPY_TYPES[tensor.dtype]
        return tensor.view(bits).numpy().view(numpy_type)
    return tensor.numpy()


# The ``ml_dtypes`` types the simulator returns, with the torch type of the
# same bits (E4M3 either way below 240).
TORCH_TYPES = {
    ml_dtypes.bfloat16: (np.int16, torch.bfloat16),
    ml_dtypes.float8_e4m3: (np.uint8, torch.float8_e4m3fn),
    ml_dtypes.float8_e4m3fn: (np.uint8, torch.float8_e4m3fn),
    ml_dtypes.float8_e5m2: (np.uint8, torch.float8_e5m2),
}


def to_torch(array):
    """A kernel's numpy result as a torch tensor, bit for bit."""
    array = np.asarray(array)
    if array.dtype.type in TORCH_TYPES:
        bits, torch_type = TORCH_TYPES[array.dtype.type]
        return torch.from_numpy(array.view(bits)).view(torch_type)
    return torch.from_numpy(array)


def load_module(directory, name):
    """``<directory>/<name>.py`` as a module."""
    path = os.path.join(directory, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(got, want):
    """Largest absolute difference over the largest finite reference
    magnitude; equal values, infinities included, differ by nothing."""
    got = to_torch(got).float() if not isinstance(got, torch.Tensor) else got
    got, want = got.float().reshape(want.shape), want.float()
    diff = (got - want).abs()
    diff[got == want] = 0
    finite = want[want.isfinite()].abs()
    scale = max(finite.max().item() if finite.numel() else 0.0, 1e-30)
    return diff.max().item() / scale


def mx_values(codes, scales, entry):
    """The values a packed MX array holds, ``[blocks, 32, other]``: int32
    ``codes`` ``[128, chunks, other]`` times E8M0 ``scales`` ``[16, chunks,
    other]``, laid out as manifest ``entry`` gives."""
    codes, scales = to_numpy(codes), to_numpy(scales)
    rows, chunks, other = codes.shape
    element = MX_ELEMENTS[entry["format"]]
    words = np.ascontiguousarray(codes).view(np.uint8).view(element)
    # [row, chunk, other, byte] -> contraction (chunk, row, byte) by other.
    elements = words.reshape(rows, chunks, other, MX_PACK).astype(np.float32)
    elements = elements.transpose(1, 0, 3, 2).reshape(-1, MX_BLOCK, other)
    exponent = scales.transpose(1, 0, 2).reshape(-1, other).astype(np.int32)
    return elements * np.exp2(exponent - 127.0)[:, None, :]


def mx_error(got_codes, got_scales, want_codes, want_scales, entry):
    """The largest error of an MX output's values over its block's largest
    reference magnitude, taken block by block (an all-zero block counts
    any nonzero value against the smallest normal float)."""
    got = mx_values(got_codes, got_scales, entry)
    want = mx_values(want_codes, want_scales, entry)
    diff = np.abs(got - want).max(axis=1)
    top = np.maximum(np.abs(want).max(axis=1), np.finfo(np.float32).tiny)
    return float((diff / top).max())


def simulate(kernels, name, args):
    """Run kernel ``name`` on torch tensors ``args`` in the simulator."""
    results = nki.simulate(getattr(kernels, name))(*map(to_numpy, args))
    if not isinstance(results, (list, tuple)):
        results = [results]
    return [to_torch(r) for r in results]


def check_calls(directory, kernels, manifest, tensors, only, distinct):
    """Simulate every kernel call -- the first of each distinct kernel when
    ``distinct`` -- or those numbered ``only``; returns how many failed."""
    failed, seen = 0, set()
    calls = [s for s in manifest["steps"] if s["kernel"] is not None]
    for i, call in enumerate(calls):
        if only is not None and i not in only:
            continue
        if distinct and call["kernel"] in seen:
            continue
        seen.add(call["kernel"])
        dtypes = manifest["dtypes"]
        args = [to_kernel(tensors[b], dtypes.get(b)) for b in call["inputs"]]
        got = dict(
            zip(call["outputs"], simulate(kernels, call["kernel"], args))
        )
        pairs = manifest.get("mx", {})
        codes_of = {e["scales"]: codes for codes, e in pairs.items()}
        errors, bars = [], []
        for b in call["outputs"]:
            # An MX array is checked with the other half of its pair.
            codes = codes_of.get(b, b)
            if codes not in pairs:
                errors.append(error(got[b], tensors[b]))
                bars.append(MAX_FP8_ERROR.get(dtypes.get(b), MAX_ERROR))
                continue
            scales = pairs[codes]["scales"]
            if codes not in got or scales not in got:
                raise NotImplementedError(f"{b}: half an MX pair written")
            errors.append(
                mx_error(
                    got[codes],
                    got[scales],
                    tensors[codes],
                    tensors[scales],
                    pairs[codes],
                )
            )
            bars.append(MAX_FP8_ERROR[pairs[codes]["format"]])
        ok = all(e <= bar for e, bar in zip(errors, bars))
        failed += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} [{i}] {call['ops'][0]} "
            f"({call['kernel']}): "
            + ", ".join(
                f"{b} {e:.2e}" for b, e in zip(call["outputs"], errors)
            ),
            flush=True,
        )
    return failed


def check_program(directory, kernels, manifest, tensors):
    """Run the program on its inputs and parameters, every kernel
    simulated; returns how many outputs are off.

    Rounding differences add up across a model's kernels, so an output
    passes on its direction -- a cosine similarity of at least
    ``MIN_COSINE`` with the reference, the lowest ``MIN_MX_COSINE`` of its
    MX formats in a program with MX arrays -- and its largest error is
    reported beside it.
    """
    program = load_module(directory, "program")
    inputs = {b: tensors[b] for b in manifest["inputs"]}
    params = {b: tensors[b] for b in manifest["params"]}
    got = program.run(
        inputs, params, lambda name, *args: simulate(kernels, name, args)
    )
    bar = min(
        [MIN_MX_COSINE[e["format"]] for e in manifest.get("mx", {}).values()],
        default=MIN_COSINE,
    )
    failed = 0
    for value, name in zip(got, manifest["outputs"]):
        want = tensors[name].float().flatten()
        value = value.float().flatten()
        cosine = torch.nn.functional.cosine_similarity(value, want, dim=0)
        ok = cosine.item() >= bar
        failed += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} output {name}: cosine "
            f"{cosine.item():.6f}, error {error(value, want):.2e}"
        )
    return failed


def build_neffs(directory, manifest, target):
    """Compile each distinct kernel for ``target`` to a NEFF under
    ``<dir>/neff/<target>/<kernel>``, each in its own process, unless one is
    there; returns how many failed."""
    failed = 0
    calls = [s for s in manifest["steps"] if s["kernel"] is not None]
    for call in {c["kernel"]: c for c in calls}.values():
        out = os.path.join(directory, "neff", target, call["kernel"])
        os.makedirs(out, exist_ok=True)
        # A kernel's name is a digest of its source: a NEFF there is its.
        if any(f.endswith(".neff") for f in os.listdir(out)):
            print(f"ok {call['kernel']}", flush=True)
            continue
        env = dict(
            os.environ,
            NKI_ARTIFACTS_DIR=out,
            NEURON_PLATFORM_TARGET_OVERRIDE=target,
        )
        command = [
            sys.executable,
            __file__,
            directory,
            "--compile",
            call["kernel"],
        ]
        with open(os.path.join(out, "compile.log"), "w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=log)
        neffs = [f for f in os.listdir(out) if f.endswith(".neff")]
        failed += not neffs
        print(f"{'ok' if neffs else 'MISSING'} {call['kernel']}", flush=True)
    return failed


def compile_one(directory, manifest, name):
    """Compile kernel ``name`` on zero arguments of its inputs' shapes."""
    kernels = load_module(directory, "kernels")
    tensors = torch.load(os.path.join(directory, "tensors.pt"))
    call = next(s for s in manifest["steps"] if s["kernel"] == name)
    dtypes = manifest["dtypes"]
    args = [
        to_numpy(to_kernel(torch.zeros_like(tensors[b]), dtypes.get(b)))
        for b in call["inputs"]
    ]
    getattr(kernels, name)(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--program", action="store_true")
    parser.add_argument("--neff", action="store_true")
    parser.add_argument("--target", choices=("trn2", "trn3"), default=None)
    parser.add_argument("--compile", default=None)
    parser.add_argument("--only", type=int, action="append", default=None)
    parser.add_argument("--distinct", action="store_true")
    args = parser.parse_args()
    if args.neff and args.target is None:
        parser.error("--neff needs --target")

    with open(os.path.join(args.directory, "manifest.json")) as f:
        manifest = json.load(f)
    if args.compile is not None:
        compile_one(args.directory, manifest, args.compile)
        return
    if args.neff:
        failed = build_neffs(args.directory, manifest, args.target)
        sys.exit(1 if failed else 0)
    kernels = load_module(args.directory, "kernels")
    tensors = torch.load(os.path.join(args.directory, "tensors.pt"))
    if args.program:
        failed = check_program(args.directory, kernels, manifest, tensors)
    else:
        failed = check_calls(
            args.directory,
            kernels,
            manifest,
            tensors,
            args.only,
            args.distinct,
        )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
