"""Compile one of the prior-work benchmark kernels codex measured on Trn2.

Voyager compiles the kernel as a model of its own -- exported, transformed,
tiled under ``TrainiumModel``, bufferized and planned -- writes ``model.txt``
to ``--out``, and ``translate`` turns it into NKI there, beside
``tensors.pt`` (the references), as ``compile_lm.py`` does for a causal LM:

    python benchmarks/trainium/compile_kernel.py --kernel layernorm \\
        --out <dir> [--target trn2] [--dtype fp32]

The kernels and shapes are codex's (``voyager-compiler-trainium-isa-10-09``
docs), all FP32 there:

- ``layernorm``: a LayerNorm over ``[4096, 8192]``.
- ``swiglu``: ``down(silu(gate(x)) * up(x))``, ``x`` ``[4096, 1024]``, 3072
  hidden.
- ``bmm_softmax``: ``softmax(q @ k)`` over the last dim, ``q`` ``[16, 4096,
  64]``, ``k`` ``[16, 64, 4096]``.
- ``gemm128``, ``gemm256``, ``gemm512``: ``a @ b`` of two inputs, ``M x N x
  K`` 128 x 128 x 128, 256 x 256 x 256 and 512 x 256 x 512.
- ``maxpool``: a 3 x 3 max pool, stride 1 and no padding, over ``[1, 1,
  4096, 4096]``.
- ``add_rmsnorm_matmul``: ``rms_norm(x + res) @ W``, ``x`` and ``res``
  ``[4096, 1024]``, ``W`` ``[1024, 2048]``.
- ``matmul_add_rmsnorm``: ``rms_norm(x @ W + res)``, ``x`` and ``res``
  ``[4096, 2048]``, ``W`` ``[2048, 2048]``.

In the two RMS norm chains the norm's eps is 1e-5, ``x`` and ``res`` are
inputs, and the norm's weight and ``W`` are parameters.  Each compiles to
one kernel: the add and the norm fuse into the GEMM as its prologue, or
after it as its tail.

``rmsnorm``, an RMSNorm (eps 1e-5) over ``[4096, 8192]``, is not codex's:
it is the ``layernorm`` kernel's shape for the norm Llama uses.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
)
from test_codegen import MXU_OPS, VECTOR_PIPELINE, _can_fuse  # noqa: E402

from voyager_compiler import (  # noqa: E402
    OpMatcher,
    compile,
    export_model,
    transform,
)
from voyager_compiler.backend.trainium import (  # noqa: E402
    LAYOUT,
    TRN2,
    TRN3,
    immediates,
    kernel_nodes,
    predicted_seconds,
    record,
    translate,
)

TARGETS = {"trn2": TRN2, "trn3": TRN3}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# Voyager's fusion patterns, plus an add then an RMS norm: alone, ahead of a
# GEMM as its prologue, or after one as its tail.
PATTERNS = VECTOR_PIPELINE + [
    [OpMatcher("add"), OpMatcher("rms_norm")],
    [
        OpMatcher("add"),
        OpMatcher("rms_norm"),
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("add"),
        OpMatcher("rms_norm"),
    ],
]


class LayerNorm(torch.nn.Module):
    def __init__(self, features=8192):
        super().__init__()
        self.norm = torch.nn.LayerNorm(features)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)
            self.norm.bias.normal_(0.0, 0.1)

    def forward(self, x):
        return self.norm(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, features=8192):
        super().__init__()
        self.norm = torch.nn.RMSNorm(features, eps=1e-5)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)

    def forward(self, x):
        return self.norm(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, hidden=1024, inner=3072):
        super().__init__()
        self.gate = torch.nn.Linear(hidden, inner, bias=False)
        self.up = torch.nn.Linear(hidden, inner, bias=False)
        self.down = torch.nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class BmmSoftmax(torch.nn.Module):
    def forward(self, q, k):
        return torch.softmax(q @ k, dim=-1)


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class MaxPool(torch.nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, 3, 1)


class AddRmsNormMatmul(torch.nn.Module):
    def __init__(self, features=1024, out_features=2048):
        super().__init__()
        self.norm = torch.nn.RMSNorm(features, eps=1e-5)
        self.weight = torch.nn.Parameter(
            torch.randn(features, out_features) / features**0.5
        )
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)

    def forward(self, x, res):
        return self.norm(x + res) @ self.weight


class MatmulAddRmsNorm(torch.nn.Module):
    def __init__(self, in_features=2048, features=2048):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, features) / in_features**0.5
        )
        self.norm = torch.nn.RMSNorm(features, eps=1e-5)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)

    def forward(self, x, res):
        return self.norm(x @ self.weight + res)


# Each kernel and its inputs' shapes.
KERNELS = {
    "layernorm": (LayerNorm, [(4096, 8192)]),
    "rmsnorm": (RMSNorm, [(4096, 8192)]),
    "swiglu": (SwiGLU, [(4096, 1024)]),
    "bmm_softmax": (BmmSoftmax, [(16, 4096, 64), (16, 64, 4096)]),
    "gemm128": (Matmul, [(128, 128), (128, 128)]),
    "gemm256": (Matmul, [(256, 256), (256, 256)]),
    "gemm512": (Matmul, [(512, 512), (512, 256)]),
    "maxpool": (MaxPool, [(1, 1, 4096, 4096)]),
    "add_rmsnorm_matmul": (AddRmsNormMatmul, [(4096, 1024), (4096, 1024)]),
    "matmul_add_rmsnorm": (MatmulAddRmsNorm, [(4096, 2048), (4096, 2048)]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    parser.add_argument("--target", choices=TARGETS, default="trn2")
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    config = TARGETS[args.target]
    dtype = DTYPES[args.dtype]
    module, shapes = KERNELS[args.kernel]
    model = module().to(dtype).eval()
    example_args = tuple(torch.randn(s, dtype=dtype) for s in shapes)

    gm = export_model(model, example_args)
    transform(gm, example_args, patterns=PATTERNS, config=config, **LAYOUT)
    nodes = kernel_nodes(gm)
    ir = compile(
        gm,
        example_args,
        config=config,
        output_dir=args.out,
        output_file=args.kernel,
        dump_tensors=False,
    )
    values = immediates(gm, ir)
    with open(os.path.join(args.out, "immediates.json"), "w") as f:
        json.dump(values, f)
    predicted = predicted_seconds(nodes, config)
    total = sum(predicted)
    with open(os.path.join(args.out, "predicted.json"), "w") as f:
        json.dump({"total_s": total, "kernels_s": predicted}, f, indent=1)
    manifest = translate(ir, args.out, values)
    kernels = [s for s in manifest["steps"] if s["kernel"] is not None]
    record(gm, list(example_args), manifest, f"{args.out}/tensors.pt")
    print(
        f"{len(kernels)} kernel calls, predicted "
        f"{total * 1e3:.3f} ms -> {args.out}"
    )


if __name__ == "__main__":
    main()
