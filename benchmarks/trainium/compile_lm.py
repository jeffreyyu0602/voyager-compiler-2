"""Compile a causal LM's prefill or decode step for one Trainium NeuronCore.

Voyager compiles the step as it would for its own accelerator -- exported,
quantized and transformed the way ``test/test_codegen.py`` does, tiled under
``TrainiumModel``, bufferized and planned -- and writes ``model.txt`` to
``--out``; ``translate`` turns it into NKI there, beside ``tensors.pt``, the
value of every HBM box as the bufferized graph computes it (check the
translation with ``backend/trainium/runtime/check.py``).

    python benchmarks/trainium/compile_lm.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --phase prefill \\
        --context 1024 --layers 1 --out <dir> [--target trn2] [--dtype bf16]

The attention is one flash-attention kernel (``--attention sdpa``, the
graph's ``scaled_dot_product_attention``) unless ``--attention eager``
decomposes it into matmul / softmax / matmul.

The layouts are Trainium's ``LAYOUT`` unless ``--tensor_layout``,
``--gemm_layout`` or ``--gemv_layout`` says otherwise: ``--gemm_layout
'xc,ck->xk'`` keeps the activations between GEMMs row-major (quote the
value: ``>`` is a shell redirect).

``--dtype fp8`` quantizes the linear layers' activations and weights, per
tensor, to the E4M3 Trn2 multiplies natively: the IEEE format
(``fp8_e4m3_ieee``), whose largest value is 240.  ``--dtype mxfp8`` (Trn3)
quantizes them to MXFP8 E4M3, blocks of 32 under a power-of-two scale set
the way Trn3's ``quantize_mx`` sets it, 2**(floor(log2(amax)) - 7): that is
``qmax=240``, since the offset is floor(log2(qmax)).  ``--dtype mxfp8_e5m2``
is the same in E5M2, 2**(floor(log2(amax)) - 14), ``qmax=16384``.
Attention and the rest stay BF16.
"""

import argparse
import json
import os
import sys

import torch
from torch.utils._pytree import tree_flatten

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
)
from test_codegen import FUSION_PATTERNS  # noqa: E402
from utils.models import llama  # noqa: E402

from voyager_compiler import (  # noqa: E402
    compile,
    get_default_quantizer,
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
from voyager_compiler.cli_args import (  # noqa: E402
    add_compile_args,
    add_layout_args,
    add_quantization_args,
)

TARGETS = {"trn2": TRN2, "trn3": TRN3}
DTYPES = {
    "bf16": None,
    "fp8": "fp8_e4m3_ieee,qs=per_tensor_symmetric",
    "mxfp8": "fp8_e4m3,qs=microscaling,bs=32,pot=1,qmax=240",
    "mxfp8_e5m2": "fp8_e5m2,qs=microscaling,bs=32,pot=1,qmax=16384",
}
# The accumulation of a scaled GEMM, which its bias spec names: TensorE sums
# into float32 PSUM.
ACCUMULATION = "fp32"


def voyager_args(args, config):
    """The ``test_codegen.py`` arguments the export helpers read: its
    defaults, the BF16 LLM flags ``run_ci.py`` compiles with, the layouts
    and the design point."""
    parser = argparse.ArgumentParser()
    add_quantization_args(parser)
    add_compile_args(parser)
    voyager = parser.parse_args([])
    voyager.model = f"llama_{args.phase}"
    voyager.model_name_or_path = args.model
    voyager.context_length = args.context
    voyager.num_hidden_layers = args.layers
    voyager.attn_implementation = args.attention
    voyager.bf16 = True
    voyager.remove_fp32_casts = True
    voyager.qconfig = None
    voyager.quantize_attention_mask = False
    voyager.residual_length = None
    voyager.spec_length = 1
    voyager.debug = False
    voyager.model_output_dir = args.out
    voyager.dump_tensors = False
    voyager.tensor_layout = args.tensor_layout
    voyager.gemm_layout = args.gemm_layout
    voyager.gemv_layout = args.gemv_layout
    voyager.merge_linears = args.merge_linears
    # The decode cache is sized in whole vector-lane groups of this core.
    voyager.pe_array_size = config.pe_array_size
    voyager.vector_unit_width = config.vector_unit_width
    return voyager


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--target", choices=TARGETS, default="trn2")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument(
        "--attention", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--merge-linears",
        action="store_true",
        help="run the linears reading one input as one GEMM "
        "(merge_sibling_linears)",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="keep only the inputs and parameters, not every step's result",
    )
    add_layout_args(parser)
    parser.set_defaults(**LAYOUT)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    config = TARGETS[args.target]
    voyager = voyager_args(args, config)

    model, tokenizer = llama.load_model(voyager)
    spec = DTYPES[args.dtype]
    quantizer = get_default_quantizer(
        input_activation=spec,
        weight=spec,
        bias=ACCUMULATION if spec is not None else None,
    )
    # Attention's matmuls stay BF16.
    quantizer.set_object_type(torch.ops.aten.matmul.default, None)
    gm, example_args, example_kwargs, _, transform_args, compile_args, _ = (
        llama.quantize_model(
            model,
            tokenizer,
            quantizer,
            FUSION_PATTERNS,
            voyager,
            reference=False,
        )
    )
    transform_args["config"] = config
    compile_args["config"] = config
    # The flash attention's causal table is the additive fill.
    compile_args["bool_mask"] = False
    transform(gm, example_args, example_kwargs, **transform_args)
    nodes = kernel_nodes(gm)
    model = compile(gm, example_args, example_kwargs, **compile_args)
    # Saved first, so the translation can be redone without compiling.
    values = immediates(gm, model)
    with open(os.path.join(args.out, "immediates.json"), "w") as f:
        json.dump(values, f)
    # Saved before translating, so a step the translator rejects still has
    # its prediction.
    predicted = predicted_seconds(nodes, config)
    total = sum(predicted)
    with open(os.path.join(args.out, "predicted.json"), "w") as f:
        json.dump({"total_s": total, "kernels_s": predicted}, f, indent=1)
    manifest = translate(model, args.out, config, values)
    kernels = [s for s in manifest["steps"] if s["kernel"] is not None]
    flat_args, _ = tree_flatten((example_args, example_kwargs))
    path = os.path.join(args.out, "tensors.pt")
    record(gm, flat_args, manifest, path, steps=not args.no_reference)
    print(
        f"{len(kernels)} kernel calls, predicted "
        f"{total * 1e3:.3f} ms -> {args.out}"
    )


if __name__ == "__main__":
    main()
