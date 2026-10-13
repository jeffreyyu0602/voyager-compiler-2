"""Run a traced step (``trace.py``) on a NeuronCore and time it.

Runs on a Trainium host in the ``trn`` env, by path:

    python run.py <dir> [--iterations N] [--warmup N]

Moves the step's parameters to NeuronCore 0 once, runs the warm-up, then
times each run with the host clock until the outputs are back.  Writes
``<dir>/step/run.json``: every run's latency, their percentiles, and how
each output ``tensors.pt`` holds a reference for compares with it.
Profiling wraps this script: set ``NEURON_RT_INSPECT_ENABLE=1`` and
``NEURON_RT_INSPECT_OUTPUT_DIR`` to capture a device profile.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch_neuronx


def compare(got, want):
    """``got`` against reference ``want``: the largest absolute difference
    over the largest reference magnitude, the cosine similarity, and the
    fraction of rows whose largest entry along the last dim agrees (for
    logits, the predicted token)."""
    got, want = got.float().reshape(want.shape), want.float()
    scale = max(want.abs().max().item(), 1e-30)
    cosine = torch.nn.functional.cosine_similarity(
        got.flatten(), want.flatten(), dim=0
    )
    agree = got.argmax(-1) == want.argmax(-1)
    return {
        "error": (got - want).abs().max().item() / scale,
        "cosine": cosine.item(),
        "argmax": agree.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    with open(os.path.join(args.directory, "manifest.json")) as f:
        manifest = json.load(f)
    tensors = torch.load(os.path.join(args.directory, "tensors.pt"))
    inputs = tuple(tensors[name] for name in manifest["inputs"])
    step_dir = os.path.join(args.directory, "step")
    step = torch.jit.load(os.path.join(step_dir, "step.pt"))
    torch_neuronx.move_trace_to_device(step, 0)

    for _ in range(args.warmup):
        outputs = step(*inputs)
    latencies = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        outputs = step(*inputs)
        latencies.append(time.perf_counter() - start)

    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    result = {
        "latencies_s": latencies,
        "p50_s": statistics.median(latencies),
        "p90_s": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
        "mean_s": statistics.mean(latencies),
        "outputs": {
            name: compare(got.cpu(), tensors[name])
            for got, name in zip(outputs, manifest["outputs"])
            if name in tensors
        },
    }
    with open(os.path.join(step_dir, "run.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "latencies_s"}))


if __name__ == "__main__":
    main()
