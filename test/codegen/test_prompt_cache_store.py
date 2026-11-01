"""A prefill fills a split decode cache as the decode's own fold would.

A tiny random Llama (1 layer, 2 heads, 1 KV head, head_dim 64) is
quantized as a ``llama_cache`` pair.  After both graphs are transformed,
the prefill's stores into the decode graph's cache buffers are compared,
bit for bit, with the bake the decode's transform made of the same prompt,
and the decode step run after the prefill is compared with its reference.
"""

import os
import shlex
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.fx import GraphModule
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from test_codegen import FUSION_PATTERNS  # noqa: E402
from utils.models import llama  # noqa: E402
from utils.models.utils import get_transform_args  # noqa: E402

from voyager_compiler import (  # noqa: E402
    add_compile_args,
    add_quantization_args,
    get_default_quantizer,
    transform,
)
from voyager_compiler.shape_prop import (  # noqa: E402
    constant_value,
    fetch_attr,
)

GRAD_ENABLED = False

# The tokenizer the random model is saved with: a public Llama one.
TOKENIZER = "amd/AMD-Llama-135m"
FLAGS = (
    "--activation lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 "
    "--weight lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 --bf16 "
    "--residual fp8_e4m3 --tensor_layout nhwc --gemm_layout xc,ck->xk "
    "--scratchpad_size 2097152 --num_banks 16 --pe_array_size 64,64 "
    "--remove_fp32_casts --bank_width 32"
)
RESIDUAL = 128
# (stage, prompt length, --residual_length): a KIVI cache with a 64-token
# tail and with none, an MX cache split by the flag, and the MX caches split
# as their blocks ask: the value cache alone, by 64.
CASES = {
    "kivi": ("llama_cache_kivi", 320, RESIDUAL),
    "kivi_whole": ("llama_cache_kivi", 384, RESIDUAL),
    "mx": ("llama_cache", 320, RESIDUAL),
    "mx_blocks": ("llama_cache", 320, None),
}


def stored_buffers(formats):
    """The buffers each cache format stores, by name."""
    names = {}
    for fmt in formats.values():
        buffers = list(fmt.buffers.values())
        if fmt.reencode is not None:
            buffers = [fmt.buffers[2], *fmt.reencode.buffers]
        names[fmt.cache] = [b.target for b in buffers]
    return names


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A random tiny Llama and its tokenizer, saved as a checkpoint for
    ``llama.load_model``."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=len(tokenizer),
        max_position_embeddings=1024,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_llama")
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module", params=sorted(CASES))
def pair(request, model_dir):
    stage, length, residual = CASES[request.param]
    args = add_compile_args(add_quantization_args()).parse_args(
        shlex.split(FLAGS)
    )
    vars(args).update(
        model=stage,
        model_name_or_path=model_dir,
        num_hidden_layers=1,
        context_length=length,
        residual_length=residual,
        spec_length=1,
        qconfig=None,
        quantize_attention_mask=True,
        attn_implementation="eager",
        model_output_dir=None,
        dump_tensors=False,
        debug=False,
    )
    model, tokenizer = llama.load_model(args)
    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
        error=args.error,
    )
    lm, kwargs = llama.quantize_graphs(model, tokenizer, quantizer, args)
    decode, prefill = lm.graphs["decode"], lm.graphs["prefill"]
    transform_args = get_transform_args(args, FUSION_PATTERNS)
    transform(decode, (), kwargs["decode"], **transform_args)
    formats = decode.meta["cache_formats"]
    stored = stored_buffers(formats)
    buffers = {
        name: fetch_attr(decode, name)
        for names in stored.values()
        for name in names
    }
    residuals = {
        name: buffer
        for name, buffer in decode.named_buffers()
        if name.endswith("_residual")
    }
    saved = {name: t.clone() for name, t in {**buffers, **residuals}.items()}
    transform(
        prefill, (), kwargs["prefill"], cache_formats=formats, **transform_args
    )
    # The constants the folder deferred are real only once compile installs
    # them; a run of the transformed graphs needs them installed too.
    for graph in (prefill, decode):
        for sub in graph.modules():
            if isinstance(sub, GraphModule):
                for node in sub.graph.find_nodes(op="get_attr"):
                    constant_value(sub, node)
    # The decode step on the bake the decode's own transform made.
    with torch.no_grad():
        reference = decode(**kwargs["decode"])
    return SimpleNamespace(
        lm=lm,
        kwargs=kwargs,
        length=length,
        formats=formats,
        stored=stored,
        buffers=buffers,
        residuals=residuals,
        saved=saved,
        reference=reference,
    )


def test_prefill_stores_the_bake(pair):
    """Run on emptied buffers, the transformed prefill writes each stored
    buffer's entries for the prompt's whole chunks as the decode's bake
    holds them, the residual's tail as the prepared prefill wrote it, and
    nothing above them."""
    for tensor in {**pair.buffers, **pair.residuals}.values():
        tensor.zero_()
    with torch.no_grad():
        pair.lm.graphs["prefill"](**pair.kwargs["prefill"])
    for cache, fmt in pair.formats.items():
        cache_len = fetch_attr(pair.lm.graphs["decode"], cache).shape[fmt.dim]
        chunk = (
            fmt.residual.value.shape[fmt.dim] if fmt.residual is not None else 1
        )
        done = pair.length // chunk * chunk
        for name in pair.stored[cache]:
            held, saved = pair.buffers[name], pair.saved[name]
            n = held.shape[fmt.dim] * done // cache_len
            assert torch.equal(
                held.narrow(fmt.dim, 0, n), saved.narrow(fmt.dim, 0, n)
            ), name
            assert not held.narrow(fmt.dim, n, held.shape[fmt.dim] - n).any()
        if fmt.residual is not None:
            name = f"{cache}_residual"
            held, saved = pair.residuals[name], pair.saved[name]
            tail = pair.length - done
            assert torch.equal(held[:, :, :tail], saved[:, :, :tail]), name
            assert not held[:, :, tail:].any()


def test_decode_after_prefill_matches_reference(pair):
    """The transformed decode step, reading what the prefill stored, gives
    the logits it gave on the decode's own bake of the prompt."""
    with torch.no_grad():
        logits = pair.lm.graphs["decode"](**pair.kwargs["decode"])
    assert torch.equal(logits, pair.reference)
