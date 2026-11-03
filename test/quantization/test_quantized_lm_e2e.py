"""Training to inference, end to end, on one ``QuantizedLM``.

Each case builds a tiny GQA Llama's training, prefill and decode graphs as
one ``QuantizedLM``, trains 3 steps, and runs a fixed list of prompts --
the prefill, then decode steps -- three ways, back to back on one KV cache:

- **U**, the prepared (fake-quant) graphs;
- **V**, the converted graphs (``convert``), or the frozen ones;
- **W**, the compiled programs, built as the ``llama_cache`` stage builds
  them: the decode once, then the prefill at each prompt length, storing
  the prompt in the decode's cache format at the decode's DRAM addresses.
  Each program's Voyager ``model.txt`` is written under ``tmp_path``
  (``pytest --basetemp`` keeps them).

The cache is not emptied between W's prompts, so a shorter prompt after
longer ones must give what it gives on an emptied cache.  Every test here
is slow: a case runs for minutes on a CPU.  Run them with ``--run-slow``,
one with ``-k``.
"""

import copy
import os
import shlex
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch.fx import GraphModule
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"
    ),
)
from compile_model import FUSION_PATTERNS  # noqa: E402
from models import llama  # noqa: E402
from models.compile_steps import (  # noqa: E402
    get_compile_args,
    get_transform_args,
    single_cpu_thread,
)

# ``models.llama`` puts examples/language_modeling on the path.
from common import quantize_generator  # noqa: E402
from quantization_configs import (  # noqa: E402
    annotate_kivi_cache,
    set_kivi_attention_qconfig,
    set_residual_attention_qconfig,
)
from voyager_compiler import (  # noqa: E402
    Example,
    QuantizedLM,
    add_compile_args,
    add_quantization_args,
    compile,
    get_default_quantizer,
    lower,
    merge_sibling_linears,
    planned_buffer_layout,
    quantizes_kv_cache,
    split_kv_cache,
    transform,
)
from voyager_compiler.codegen import match_and_fuse_rms_norm  # noqa: E402
from voyager_compiler.quantization.model import _CACHE_BUFFER  # noqa: E402
from voyager_compiler.shape_prop import constant_value  # noqa: E402

GRAD_ENABLED = False
pytestmark = pytest.mark.slow

VOCAB = 256
HIDDEN = 256
CONTEXT = 1024
# The vector lanes, which a residual is a multiple of.
LANES = 64
COMPILE_FLAGS = (
    "--bf16 --tensor_layout nhwc --gemm_layout xc,ck->xk --scratchpad_size "
    "2097152 --scratchpad_banks 16 --pe_array_size 64,64 --remove_fp32_casts "
    "--scratchpad_word_bytes 32"
)
# Prompts, in order, as (length, decode steps after it; 0 for up to the
# cache's last slot): the usual 1024; a shorter prompt after it, then 192
# after both (stale cache); an 8-token decode tail; past the context; no
# whole chunk; to the cache's last slot.
PROMPTS = [
    (1024, 64),
    (320, 128),
    (192, 128),
    (200, 8),
    (1088, 8),
    (64, 8),
    (1024, 0),
]
# The prompt run again on an emptied cache after all of them.
STALE = 192

FP8 = "fp8_e4m3,qs=per_tensor_symmetric,ahl=16"
FP8_ERROR = "fp8_e5m2,qs=per_tensor_symmetric,ahl=16"
MXINT8 = "int8,qs=microscaling,bs=32,pot=1"
NVFP4 = "fp4_e2m1,qs=microscaling,bs=16,scale=fp8_e4m3,gs=1"
MXNF4 = "lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3"
MX_ERROR = "fp8_e5m2,qs=microscaling,bs=64"
# ``get_default_quantizer`` arguments per scheme for the inference graphs.
INFERENCE = {
    "bf16": {},
    "fp8": {"input_activation": FP8, "weight": FP8},
    "mxint8": {"input_activation": MXINT8, "weight": MXINT8},
    "nvfp4": {"input_activation": NVFP4, "weight": f"{NVFP4},ax=(-2,-1)"},
    "mxnf4": {"input_activation": MXNF4, "weight": MXNF4},
    "kivi": {"input_activation": MXNF4, "weight": MXNF4},
    "mixed": {"input_activation": MXNF4, "weight": MXNF4},
}
# What the training graphs add or change: the gradients' spec, NVFP4's full
# recipe, and the mixed scheme's FP8 training.
TRAINING = {
    "fp8": {"error": FP8_ERROR},
    "mxint8": {"error": MXINT8},
    "nvfp4": {
        "error": f"{NVFP4},sr=1",
        "random_hadamard_transform": ("wgrad",),
    },
    "mxnf4": {"error": MX_ERROR},
    "kivi": {"error": MX_ERROR},
    "mixed": {"input_activation": FP8, "weight": FP8, "error": FP8_ERROR},
}
# Schemes whose decode cache is KIVI's 2-bit one.
KIVI = ("kivi", "mixed")
# Bound on a logit's distance from the converted graphs' where the two are
# not bit-exact: the compiled programs, and a prepared graph that rounds
# differently (FP8's per-GEMM rounding, the converted ``sdpa_mx``).  The
# random model's logits are of scale 1.
TOLERANCE = {
    "bf16": 0.05,
    "fp8": 0.15,
    "mxint8": 0.08,
    "nvfp4": 0.5,
    "mxnf4": 0.5,
    "kivi": 0.5,
    "mixed": 0.5,
}
ALL_PHASES = ("prefill", "decode", "training")


@dataclass(frozen=True)
class Case:
    """One build: a scheme with ``--residual_length`` (None for the split
    rule's own choice) and an attention, plus the one-at-a-time options.
    None for ``mismatched_weights`` and ``backward_override`` takes the
    library's default."""

    scheme: str
    residual: Optional[int] = None
    attn: str = "eager"
    mha: bool = False
    merge_linears: bool = False
    double_buffered: bool = False
    quantize_mask: bool = False
    bool_mask: bool = True
    logits_to_keep: int = 1
    decode_tokens: int = 1
    freeze: bool = False
    mismatched_weights: Optional[str] = None
    backward_override: Optional[str] = None


def grid():
    """The approved grid: every scheme with and without the residual, under
    eager and flash attention; KIVI only with it.  NVFP4's compiled
    programs fail on its global scale, a lowering not written yet."""
    cases = []
    for scheme in INFERENCE:
        for residual in (128, None):
            if residual is None and scheme in KIVI:
                continue
            for attn in ("eager", "sdpa"):
                marks = ()
                if scheme == "nvfp4":
                    marks = pytest.mark.xfail(
                        raises=RuntimeError,
                        strict=True,
                        reason="NVFP4's global scale is not lowered",
                    )
                cases.append(
                    pytest.param(
                        Case(scheme, residual, attn),
                        id=f"{scheme}-r{residual}-{attn}",
                        marks=marks,
                    )
                )
    return cases


# The one-at-a-time options, each on the KIVI build unless it needs another.
VARIANTS = {
    "mha": Case("kivi", 128, mha=True),
    "merge_linears": Case("kivi", 128, merge_linears=True),
    "double_buffered": Case("kivi", 128, double_buffered=True),
    "quantize_mask": Case("kivi", 128, quantize_mask=True),
    "bool_mask_off": Case("mxnf4", attn="sdpa", bool_mask=False),
    "logits_all": Case("kivi", 128, logits_to_keep=0),
    "freeze": Case("kivi", 128, freeze=True),
    "dynamic_weights": Case("kivi", 128, mismatched_weights="dynamic"),
    "high_precision_backward": Case(
        "kivi", 128, backward_override="high_precision"
    ),
}


def tiny_llama(mha: bool, attn: str) -> LlamaForCausalLM:
    """A random 2-layer Llama of head_dim 64 in bf16: 4 heads over 2 KV
    heads, or over 4 under ``mha``."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4 if mha else 2,
        max_position_embeddings=2048,
        attn_implementation=attn,
    )
    return LlamaForCausalLM(config).to(torch.bfloat16)


def quantizer(scheme: str, phase: str):
    """``scheme``'s quantizer for ``phase``.  The rotary embedding's matmul
    stays unquantized; so does NVFP4's attention, as an NVFP4 cache is not
    lowered."""
    settings = dict(INFERENCE[scheme])
    if phase == "training":
        settings.update(TRAINING.get(scheme, {}))
    q = get_default_quantizer(**settings)
    q.set_module_name_object_type_order(
        r"model\.rotary_emb", torch.ops.aten.matmul.default, 0, None
    )
    if scheme == "nvfp4":
        for order in range(4):
            q.set_module_name_object_type_order(
                "self_attn", torch.ops.aten.matmul.default, order, None
            )
        q.set_object_type(
            torch.ops.aten.scaled_dot_product_attention.default, None
        )
    return q


def build(case: Case, phases=ALL_PHASES, model=None):
    """Build ``case`` as ``llama.quantize_model`` does.

    Under a residual length, a decode cache the attention reads quantized
    is split before quantization (``split_kv_cache``) and the prefill's
    cache writes alike; without one the caches stay whole, for
    ``transform`` to split a cache whose blocks run along the positions.
    A KIVI decode's main caches are quantized to 2 bits.

    Args:
        case: The build.
        phases: The phases to build.
        model: The causal LM; ``tiny_llama`` when None.

    Returns:
        ``(lm, split, ids)``: the ``QuantizedLM``, whether its decode
        caches were split before quantization, and the training batch.

    Raises:
        ValueError: A KIVI case has no residual length.
    """
    if model is None:
        model = tiny_llama(case.mha, case.attn)
    model.config.use_cache = False
    quantizers = {p: quantizer(case.scheme, p) for p in phases}
    layernorm = model.model.layers[0].input_layernorm
    hidden = layernorm.weight.shape[-1]
    split = {"done": False}

    def rewrite(gm, phase):
        if isinstance(model, LlamaForCausalLM):
            example = torch.randn(
                1, 8, hidden, dtype=model.dtype, device=model.device
            )
            match_and_fuse_rms_norm(gm, layernorm, (example,))
        if case.merge_linears and phase != "training":
            merge_sibling_linears(gm)
        if case.quantize_mask and phase == "prefill":
            llama._annotate_attention_mask(gm)
        if phase == "prefill" and split["done"]:
            split_kv_cache(gm, case.residual, 0)
        if phase != "decode":
            return
        if case.scheme in KIVI and case.residual is None:
            raise ValueError("a KIVI cache takes a residual length")
        if case.residual is not None and (
            case.scheme in KIVI or quantizes_kv_cache(gm, quantizers["decode"])
        ):
            split_kv_cache(gm, case.residual, 0)
            split["done"] = True
        if case.scheme in KIVI:
            set_kivi_attention_qconfig(quantizers["decode"])
            annotate_kivi_cache(gm)
        elif split["done"]:
            set_residual_attention_qconfig(quantizers["decode"])

    ids = torch.randint(0, VOCAB, (1, 64))
    extra = {}
    if case.backward_override is not None:
        extra["backward_override"] = case.backward_override
    cache_args = SimpleNamespace(
        residual_length=case.residual, context_length=CONTEXT
    )
    lm = QuantizedLM(
        model,
        quantizers,
        Example((), {"input_ids": ids, "labels": ids}),
        context_length=CONTEXT,
        cache_len=llama.max_cache_len(
            cache_args, SimpleNamespace(vector_lanes=LANES)
        ),
        phases=phases,
        logits_to_keep=case.logits_to_keep,
        rewrite=rewrite,
        decode_tokens=case.decode_tokens,
        **extra,
    )
    return lm, split["done"], ids


def expected_residuals(case: Case, num_layers: int = 2):
    """The residual each cache of ``case``'s compiled decode holds, as
    ``{cache buffer: positions}``: every quantized cache's by the residual
    length; without one, a cache whose blocks run along tokens -- MX V, and
    K beside it under flash attention -- by a lane block.  BF16 and NVFP4
    keep a bf16 cache, as does FP8 under flash attention, which its
    per-tensor quantizer leaves unquantized."""
    if case.scheme in ("bf16", "nvfp4") or (
        case.scheme == "fp8" and case.attn == "sdpa"
    ):
        return {}
    if case.residual is not None:
        kinds, length = ("key", "value"), case.residual
    elif case.scheme in ("mxint8", "mxnf4"):
        kinds = ("key", "value") if case.attn == "sdpa" else ("value",)
        length = LANES
    else:
        return {}
    return {
        f"{kind}_cache_{i}": length for i in range(num_layers) for kind in kinds
    }


def train(lm, ids) -> list:
    """Three AdamW steps through the training graphs; the losses."""
    lm.train()
    optimizer = torch.optim.AdamW(lm.model.parameters(), lr=1e-4)
    losses = []
    with torch.enable_grad():
        for _ in range(3):
            loss = lm(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss))
    lm.eval()
    return losses


def check_training_carries_over(lm, case: Case, ids) -> None:
    """Train, and check what the inference graphs then run on: the weights
    training updated, read in place, and under FP8's delayed scaling the
    scales training observed."""
    if not case.merge_linears:
        params = {p.data_ptr() for p in lm.model.parameters()}
        for name in lm.inference_graphs():
            graph = lm.graphs[name]
            assert {p.data_ptr() for p in graph.parameters()} <= params
    before = [p.detach().clone() for p in lm.model.parameters()]
    scales = {
        id(m): m.scale.clone()
        for m in lm.fake_quants()
        if isinstance(getattr(m, "scale", None), torch.Tensor)
    }
    losses = train(lm, ids)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert any(
        not torch.equal(a, p) for a, p in zip(before, lm.model.parameters())
    )
    if case.scheme == "fp8":
        assert any(
            not torch.equal(scales[id(m)], m.scale)
            for m in lm.fake_quants()
            if id(m) in scales
        )


def prompts(cache_len: int):
    """``{length: tokens}`` of every prompt and the decode steps after it."""
    torch.manual_seed(1)
    out = {}
    for length, steps in PROMPTS:
        steps = steps or cache_len - length
        out[length + steps] = (
            length,
            torch.randint(0, VOCAB, (1, length + steps)),
        )
    return list(out.values())


def generate(lm, tokens, length):
    """The logits of a prompt's last token and of each decode step after
    it, through ``lm.prefill`` and ``lm.step``."""
    logits = [lm.prefill(tokens[:, :length]).float().clone()]
    for position in range(length, tokens.shape[1]):
        token = tokens[0, position].item()
        logits.append(lm.step(token, position).float().clone())
    return torch.stack(logits)


def install_constants(graph: GraphModule) -> None:
    """Compute the constants the folder deferred, as ``compile`` does
    before it emits, so the lowered graph runs."""
    for sub in graph.modules():
        if isinstance(sub, GraphModule):
            for node in sub.graph.find_nodes(op="get_attr"):
                constant_value(sub, node)


def compile_settings(case: Case, out_dir):
    """``(transform kwargs, compile kwargs)`` for ``case``, outputs under
    ``out_dir``."""
    flags = COMPILE_FLAGS
    if case.double_buffered:
        flags += " --double_buffered_l2"
    args = add_compile_args(add_quantization_args()).parse_args(
        shlex.split(flags)
    )
    vars(args).update(
        model="llama_cache", model_output_dir=str(out_dir), dump_tensors=False
    )
    compile_args = get_compile_args(args)
    compile_args.pop("output_dir")
    compile_args["bool_mask"] = case.bool_mask
    return get_transform_args(args, FUSION_PATTERNS), compile_args


def compile_programs(lm, case: Case, out_dir, lengths, decode_tokens=1):
    """Compile the decode, then a prefill per prompt length, as the
    ``llama_cache`` stage does: each prefill holds the decode's cache
    buffers, stores the prompt in their format and places them at the
    decode's DRAM addresses.

    Returns:
        ``(decode, {length: prefill}, cache buffers)``, compiled and
        runnable, the buffers by name.
    """
    transform_args, compile_args = compile_settings(case, out_dir)
    decode = lm.graphs["decode"]
    shared = {
        name: buffer
        for name, buffer in decode.named_buffers()
        if _CACHE_BUFFER.fullmatch(name)
    }
    step = {
        "input_ids": torch.zeros(1, decode_tokens, dtype=torch.long),
        "cache_position": torch.arange(CONTEXT, CONTEXT + decode_tokens),
    }
    transform(decode, (), step, **transform_args)
    formats = decode.meta.get("cache_formats") or None
    compile(
        decode,
        (),
        step,
        output_dir=os.path.join(out_dir, "decode"),
        **compile_args,
    )
    prefills = {}
    for length in lengths:
        prefill = copy.deepcopy(lm.graphs["prefill"])
        # A deep copy holds copies of the cache; the program writes the
        # decode's own.
        for name, buffer in shared.items():
            if hasattr(prefill, name):
                delattr(prefill, name)
                prefill.register_buffer(name, buffer)
        prompt = {
            "input_ids": torch.zeros(1, length, dtype=torch.long),
            "cache_position": torch.arange(length),
        }
        transform(prefill, (), prompt, cache_formats=formats, **transform_args)
        compile(
            prefill,
            (),
            prompt,
            output_dir=os.path.join(out_dir, f"prefill_{length}"),
            dram_layout=planned_buffer_layout(
                decode, prefill, transform_args["config"]
            ),
            **compile_args,
        )
        prefills[length] = prefill
    for graph in (decode, *prefills.values()):
        install_constants(graph)
    return decode, prefills, shared


def generate_compiled(decode, prefills, tokens, length):
    """``generate`` on the compiled programs: the prefill takes the longest
    multiple of 64 it accepts, the decode the rest token by token."""
    done = min(length, CONTEXT) // 64 * 64
    logits = []
    with single_cpu_thread():
        if done:
            out = prefills[done](
                input_ids=tokens[:, :done], cache_position=torch.arange(done)
            )[0, -1]
            if done == length:
                logits.append(out.float().clone())
        for position in range(done, tokens.shape[1]):
            out = decode(
                input_ids=tokens[:, position : position + 1],
                cache_position=torch.tensor([position]),
            )[0, -1]
            if position >= length - 1:
                logits.append(out.float().clone())
    return torch.stack(logits)


def exact(case: Case) -> bool:
    """Whether ``case``'s prepared graphs give the converted graphs'
    logits bit for bit."""
    return case.scheme == "bf16" or (
        case.attn == "eager" and case.scheme in ("mxint8", "mxnf4") + KIVI
    )


def check_close(got, expected, case: Case, bit_exact: bool) -> None:
    if bit_exact:
        assert torch.equal(got, expected)
    else:
        deviation = float((got - expected).abs().max())
        assert deviation <= TOLERANCE[case.scheme], deviation


def convert(lm, case: Case) -> None:
    """Freeze or convert the inference graphs, as ``case`` asks."""
    if case.freeze:
        lm.freeze()
        return
    kwargs = {}
    if case.mismatched_weights is not None:
        kwargs["mismatched_weights"] = case.mismatched_weights
    lm.convert(None, **kwargs)


def run_case(case: Case, out_dir) -> None:
    """Build, train, then U, V and W (see the module's docstring), checking
    each against the last."""
    lm, split, ids = build(case)
    residuals = expected_residuals(case)
    assert split == (case.residual is not None and bool(residuals))
    check_training_carries_over(lm, case, ids)

    runs = prompts(lm.cache_len)
    u = [generate(lm, tokens, length) for length, tokens in runs]
    convert(lm, case)
    v = [generate(lm, tokens, length) for length, tokens in runs]
    for got, expected in zip(u, v):
        check_close(got, expected, case, exact(case) or case.freeze)
    if case.freeze:
        # A frozen graph is for evaluation; it is not compiled.
        return

    lengths = sorted({min(n, CONTEXT) // 64 * 64 for n, _ in runs} - {0})
    decode, prefills, shared = compile_programs(lm, case, out_dir, lengths)
    assert {
        name.removesuffix("_residual"): buffer.shape[2]
        for name, buffer in decode.named_buffers()
        if name.endswith("_residual")
    } == residuals
    for name in ["decode", *(f"prefill_{n}" for n in lengths)]:
        assert os.path.exists(os.path.join(out_dir, name, "model.txt"))
    for (length, tokens), expected in zip(runs, v):
        w = generate_compiled(decode, prefills, tokens, length)
        check_close(w, expected[-w.shape[0] :], case, bit_exact=False)
        assert all(torch.isfinite(b).all() for b in shared.values())
    length, tokens = next(r for r in runs if r[0] == STALE)
    stale = generate_compiled(decode, prefills, tokens, length)
    for buffer in shared.values():
        buffer.zero_()
    fresh = generate_compiled(decode, prefills, tokens, length)
    assert torch.equal(stale, fresh)


@pytest.mark.parametrize("case", grid())
def test_training_to_inference(case, tmp_path):
    run_case(case, tmp_path)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant(name, tmp_path):
    run_case(VARIANTS[name], tmp_path)


def test_decode_of_several_tokens(tmp_path):
    """A decode step writing 4 tokens (speculative decoding's verification
    step), which ``generate`` does not drive: one 1024-token prefill, then
    one 4-token step, prepared, converted and compiled."""
    case = Case("kivi", 128, decode_tokens=4)
    lm, _, ids = build(case)
    check_training_carries_over(lm, case, ids)
    tokens = torch.randint(0, VOCAB, (1, CONTEXT + 4))
    step = {
        "input_ids": tokens[:, CONTEXT:],
        "cache_position": torch.arange(CONTEXT, CONTEXT + 4),
    }

    def run(prefill, decode):
        with torch.no_grad():
            prefill(
                input_ids=tokens[:, :CONTEXT],
                cache_position=torch.arange(CONTEXT),
            )
            return decode(**step).float().clone()

    u = run(lm.graphs["prefill"], lm.graphs["decode"])
    lm.convert(None)
    v = run(lm.graphs["prefill"], lm.graphs["decode"])
    check_close(u, v, case, bit_exact=True)
    decode, prefills, _ = compile_programs(
        lm, case, tmp_path, [CONTEXT], decode_tokens=4
    )
    with single_cpu_thread():
        w = run(prefills[CONTEXT], decode)
    check_close(w, v, case, bit_exact=False)


def test_prefill_alone(tmp_path):
    """A prefill-only build keeps today's path: the attention reads its
    cache, which no decode graph holds.  Prepared, converted and compiled
    on one 1024-token prompt."""
    case = Case("mxnf4")
    lm, split, _ = build(case, phases=("prefill",))
    assert not split
    lm.prefill(torch.randint(0, VOCAB, (1, 320)))  # calibration
    lm.eval()
    prompt = {
        "input_ids": torch.randint(0, VOCAB, (1, CONTEXT)),
        "cache_position": torch.arange(CONTEXT),
    }
    with torch.no_grad():
        u = lm.graphs["prefill"](**prompt).float().clone()
    lm.convert(None)
    prefill = lm.graphs["prefill"]
    with torch.no_grad():
        v = prefill(**prompt).float().clone()
    check_close(u, v, case, bit_exact=True)
    transform_args, compile_args = compile_settings(case, tmp_path)
    transform(prefill, (), prompt, **transform_args)
    compile(
        prefill,
        (),
        prompt,
        output_dir=str(tmp_path / "prefill"),
        **compile_args,
    )
    install_constants(prefill)
    with torch.no_grad(), single_cpu_thread():
        w = prefill(**prompt).float()
    check_close(w, v, case, bit_exact=False)


def test_training_alone():
    """A training-only build trains, and its weights move."""
    case = Case("mxnf4")
    lm, _, ids = build(case, phases=("training",))
    check_training_carries_over(lm, case, ids)


def test_kivi_needs_a_residual_length():
    with pytest.raises(ValueError, match="residual length"):
        build(Case("kivi"))


def tiny_gemma() -> Gemma3ForCausalLM:
    """A random 2-layer Gemma 3 whose layers attend through a 128-token
    sliding window.  Its RMSNorm is left as Gemma writes it."""
    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        sliding_window=128,
        max_position_embeddings=2048,
        attn_implementation="eager",
    )
    return Gemma3ForCausalLM(config).to(torch.bfloat16)


def test_sliding_window():
    """A sliding-window layer beside a split decode cache is not lowered
    and raises; beside a whole bf16 cache it builds and generates."""
    with pytest.raises(NotImplementedError, match="sliding-window"):
        build(Case("mxnf4", 128), ("prefill", "decode"), model=tiny_gemma())
    lm, split, _ = build(
        Case("bf16"), ("prefill", "decode"), model=tiny_gemma()
    )
    assert not split
    prompt = torch.randint(0, VOCAB, (1, 70))
    assert lm.generate(prompt, 4).shape == (1, 74)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_replay_generates_the_cpu_tokens():
    """On a GPU, FP8's fake-quants run their Triton kernels and the decode
    step replays as a CUDA graph; frozen, the model generates the CPU's
    tokens."""
    case = Case("fp8", 128)
    tokens = {}
    for device in ("cpu", "cuda"):
        model = tiny_llama(False, "eager").to(device)
        lm, _, _ = build(case, ("prefill", "decode"), model=model)
        lm.eval()
        lm.prefill(torch.randint(0, VOCAB, (1, 320), device=device))
        lm.freeze()
        torch.manual_seed(2)
        tokens[device] = [
            lm.generate(torch.randint(0, VOCAB, (1, n)).to(device), 16).cpu()
            for n in (320, 100, CONTEXT)
        ]
    for cpu, gpu in zip(tokens["cpu"], tokens["cuda"]):
        assert torch.equal(cpu, gpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_lowered_kivi_cache_runs_on_gpu():
    """LongBench's fully quantized KIVI arm (``longbench.py --lower``),
    converted and lowered on a GPU: the prefill stores the prompt in the
    decode's 2-bit cache, and each decode step reads it through the
    dequantize fused with the attention's int6 quantize.  The lowered step,
    its folds run by the caller, replays as a CUDA graph.  Fed the same
    tokens across two residual folds, it stays within the KIVI tolerance of
    the converted decode, whose cache is not stored quantized; ``generate``
    lowers a prefill for each new prompt length."""
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (1, 600), device="cuda")
    prompt = 256
    lms = []
    for lowered in (False, True):
        lm = quantize_generator(
            tiny_llama(False, "eager").to("cuda"),
            "mxlut4_int6_e5m3_attn_head_int6",
            "mxlut4_int6_e5m3_a_int6",
            True,
            128,
            None,
            512,
            1024,
        )
        if lowered:
            lower(lm)
        else:
            lm.convert()
        lms.append(lm)
    converted, lowered = lms
    assert lowered.replay
    # Every layer's key and value cache is stored quantized.
    assert len(lowered.graphs["decode"].meta["cache_formats"]) == 4
    assert torch.equal(
        lowered.prefill(tokens[:, :prompt]),
        converted.prefill(tokens[:, :prompt]),
    )
    for position in range(prompt, tokens.shape[1]):
        token = tokens[0, position].item()
        got = lowered.step(token, position).float()
        expected = converted.step(token, position).float()
        assert (got - expected).abs().max() <= TOLERANCE["kivi"], position
    generated = lowered.generate(tokens[:, :320], 8)
    assert torch.equal(generated[:, :320], tokens[:, :320])
