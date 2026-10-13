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
from test_codegen import VECTOR_PIPELINE  # noqa: E402

from voyager_compiler import compile, export_model, transform  # noqa: E402
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


class LayerNorm(torch.nn.Module):
    def __init__(self, features=8192):
        super().__init__()
        self.norm = torch.nn.LayerNorm(features)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)
            self.norm.bias.normal_(0.0, 0.1)

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


# Each kernel and its inputs' shapes.
KERNELS = {
    "layernorm": (LayerNorm, [(4096, 8192)]),
    "swiglu": (SwiGLU, [(4096, 1024)]),
    "bmm_softmax": (BmmSoftmax, [(16, 4096, 64), (16, 64, 4096)]),
    "gemm128": (Matmul, [(128, 128), (128, 128)]),
    "gemm256": (Matmul, [(256, 256), (256, 256)]),
    "gemm512": (Matmul, [(512, 512), (512, 256)]),
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
    transform(
        gm, example_args, patterns=VECTOR_PIPELINE, config=config, **LAYOUT
    )
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
