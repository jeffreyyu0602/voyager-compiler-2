"""A prefill fills a split decode cache as the decode's own fold would.

A tiny random Llama (1 layer, 2 heads, 1 KV head, head_dim 64) is quantized
as one ``QuantizedLM`` holding a prefill and the decode whose cache it fills,
as Voyager's causal-LM stages quantize them.  After both graphs are
transformed, the prefill's stores into the decode graph's cache buffers are
compared, bit for bit, with the bake the decode's transform made of the same
prompt, and the decode step run after the prefill is compared with its
reference.
"""

import copy
import math
import os
import shlex
import sys
from types import SimpleNamespace

import pytest
import torch
from datasets import load_dataset
from torch.fx import GraphModule
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.append(os.path.join(_REPO, "examples", "language_modeling"))
from compile_model import FUSION_PATTERNS  # noqa: E402
from models.compile_steps import get_transform_args  # noqa: E402
from quantization_configs import (  # noqa: E402
    annotate_attention_mask,
    annotate_kivi_cache,
    set_kivi_attention_qconfig,
    set_residual_attention_qconfig,
)

from voyager_compiler import (  # noqa: E402
    QuantizedLM,
    add_compile_args,
    add_quantization_args,
    get_default_quantizer,
    quantizes_kv_cache,
    split_kv_cache,
    transform,
)
from voyager_compiler.codegen import match_and_fuse_rms_norm  # noqa: E402
from voyager_compiler.shape_prop import (  # noqa: E402
    constant_value,
    fetch_attr,
)

GRAD_ENABLED = False

# The tokenizer the random model is saved with: a public Llama one.
TOKENIZER = "amd/AMD-Llama-135m"
# Generation slots the KV cache holds beyond the prompt.
DECODE_MAX_GEN = 128
# The vector lanes, which the cache length is a multiple of.
LANES = 64
FLAGS = (
    "--activation lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 "
    "--weight lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3 --bf16 "
    "--residual fp8_e4m3 --tensor_layout nhwc --gemm_layout xc,ck->xk "
    "--scratchpad_size 2097152 --scratchpad_banks 16 --pe_array_size 64,64 "
    "--remove_fp32_casts --scratchpad_word_bytes 32"
)
RESIDUAL = 128
# The KIVI cache's group scale format, the one the compiled stages store.
KIVI_SCALE = "fp8_e5m3"
# (KIVI cache, prompt length, residual length): a KIVI cache with a 64-token
# tail and with none, an MX cache split by the residual, and the MX caches
# split as their blocks ask: the value cache alone, by 64.
CASES = {
    "kivi": (True, 320, RESIDUAL),
    "kivi_whole": (True, 384, RESIDUAL),
    "mx": (False, 320, RESIDUAL),
    "mx_blocks": (False, 320, None),
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
    """A random tiny Llama and its tokenizer, saved as a checkpoint."""
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


def build(model, kivi, length, residual, args):
    """A prefill and the decode whose cache it fills, as one prepared
    ``QuantizedLM``: RMSNorms fused, the prefill's causal mask int1, the
    rotary matmul unquantized.  Under ``residual`` a decode cache the
    attention reads quantized is split into chunks and a full-precision
    residual, the prefill's writes alike; a KIVI cache's main halves are
    2-bit."""
    model.config.use_cache = False
    quantizer = get_default_quantizer(
        input_activation=args.activation, weight=args.weight, bias=args.bias
    )
    quantizer.set_module_name_object_type_order(
        r"model\.rotary_emb", torch.ops.aten.matmul.default, 0, None
    )
    quantizers = {p: copy.deepcopy(quantizer) for p in ("prefill", "decode")}
    layernorm = model.model.layers[0].input_layernorm
    split = {"done": False}

    def rewrite(gm, phase):
        seq = 1 if phase == "decode" else 128
        example = torch.randn(
            1, seq, layernorm.weight.shape[-1], dtype=model.dtype
        )
        match_and_fuse_rms_norm(gm, layernorm, (example,))
        if phase == "prefill":
            annotate_attention_mask(gm)
            if split["done"]:
                split_kv_cache(gm, residual, 0)
            return
        if residual is not None and (
            kivi or quantizes_kv_cache(gm, quantizers["decode"])
        ):
            split_kv_cache(gm, residual, 0)
            split["done"] = True
        if kivi:
            set_kivi_attention_qconfig(quantizers["decode"])
            annotate_kivi_cache(gm, scale=KIVI_SCALE)
        elif split["done"]:
            set_residual_attention_qconfig(quantizers["decode"])

    block = math.lcm(LANES, residual or 1)
    return QuantizedLM(
        model,
        quantizers,
        None,
        context_length=length,
        cache_len=-(-(length + DECODE_MAX_GEN) // block) * block,
        phases=("prefill", "decode"),
        rewrite=rewrite,
    )


@pytest.fixture(scope="module", params=sorted(CASES))
def pair(request, model_dir):
    kivi, length, residual = CASES[request.param]
    args = add_compile_args(add_quantization_args()).parse_args(
        shlex.split(FLAGS)
    )
    model = LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    lm = build(model, kivi, length, residual, args)
    # Calibrate on a wikitext prompt and the token after it, twice (delayed
    # scaling), then convert with the prompt in the cache.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    ids = tokenizer("\n\n".join(text), return_tensors="pt").input_ids
    prompt = ids[:, : length + 1]
    kwargs = {
        "prefill": {
            "input_ids": prompt[:, :length],
            "cache_position": torch.arange(length),
        },
        "decode": {
            "input_ids": prompt[:, length:],
            "cache_position": torch.arange(length, length + 1),
        },
    }
    for _ in range(2):
        lm.prefill(prompt[:, :length])
        lm.graphs["decode"](**kwargs["decode"])
    lm.eval()
    lm.prefill(prompt[:, :length])
    lm.convert(args.bias)
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
