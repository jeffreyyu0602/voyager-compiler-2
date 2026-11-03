"""A tiny Llama and a tiny regression module quantized for inference and
training at once: generation matches HF greedy decoding, the fake-quants
are shared, the modes switch the observers, and convert bakes the graphs."""

import copy
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torchao.quantization.pt2e import FakeQuantizeBase
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from voyager_compiler import gradient_program, split_kv_cache
from voyager_compiler.ops.layout import DEFAULT_WEIGHT_LAYOUT
from voyager_compiler.quantization.model import SEQ_BLOCK, _CACHE_BUFFER
from voyager_compiler.quantization.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.quantization import (
    Example,
    QuantizedLM,
    QuantizedModel,
    get_default_quantizer,
)

ACT = "fp8_e4m3,qs=per_tensor_symmetric"
INT8 = "int8,qs=per_tensor_symmetric"
MXINT8 = "int8,qs=microscaling,bs=32,pot=1,ax=-1"
ERROR = "fp8_e5m2,qs=per_tensor_symmetric"
VOCAB = 256
CONTEXT_LENGTH = 128
CACHE_LEN = 192
SPLIT = 64
NEW_TOKENS = 6


def quantizer(activation=ACT, weight=ACT, residual_attention=False):
    """The tiny Llama's quantizer; a decode graph over a split cache leaves
    the attention's residual halves, the second and fourth matmul of each
    layer, unquantized."""
    q = get_default_quantizer(
        input_activation=activation, weight=weight, error=ERROR
    )
    q.set_module_name("rotary_emb", None)
    if residual_attention:
        for order in (1, 3):
            q.set_module_name_object_type_order(
                "self_attn", torch.ops.aten.matmul.default, order, None
            )
    return q


def tiny_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(config)


def training_example():
    ids = torch.randint(0, VOCAB, (1, 64))
    return Example((), {"input_ids": ids, "labels": ids, "use_cache": False})


def split_caches(graph, phase):
    """Split the decode cache and, as the decode graph is rewritten first,
    the prefill's write of it."""
    if phase in ("prefill", "decode"):
        split_kv_cache(graph, SPLIT, 0)


def fake_quants(graph):
    return {
        name: child
        for name, child in graph.named_children()
        if isinstance(child, FakeQuantizeBase)
    }


@pytest.fixture(scope="module", params=["whole", "split"])
def lm(request):
    """The tiny Llama quantized over a whole or a split decode cache, with a
    copy of the model as it was before, for HF's own generation."""
    model = tiny_llama()
    reference = copy.deepcopy(model)
    split = request.param == "split"
    lm = QuantizedLM(
        model,
        {
            "prefill": quantizer(),
            "decode": quantizer(residual_attention=split),
            "training": quantizer(),
        },
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
        rewrite=split_caches if split else None,
    )
    lm.reference = reference
    return lm


@pytest.mark.parametrize("length", [8, 64, 70])
def test_generate_matches_hf_greedy(lm, length):
    for fake_quant in lm.fake_quants():
        fake_quant.disable_fake_quant()
    prompt = torch.randint(0, VOCAB, (1, length))
    with torch.no_grad():
        expected = lm.reference.generate(
            prompt, max_new_tokens=NEW_TOKENS, do_sample=False
        )
    assert torch.equal(lm.generate(prompt, NEW_TOKENS), expected)
    for fake_quant in lm.fake_quants():
        fake_quant.enable_fake_quant()


def test_fake_quants_are_shared(lm):
    prefill = fake_quants(lm.graphs["prefill"])
    decode = fake_quants(lm.graphs["decode"])
    kept = {id(m) for m in prefill.values()}
    decode_report, training_report = lm.reports
    shared = [n for n, r in decode_report.items() if r == "shared"]
    assert all(id(decode[n]) in kept for n in shared)
    # A split attention's main half shares the prefill's fake-quants too:
    # its key and value read the cache buffers, which the decode writes,
    # and its residual half is not counted.
    assert len(shared) == len(decode)
    gradients = lm.training_quantizers.gradients
    assert {n for n, r in training_report.items() if r == "shared"} == (
        set(training_report) - gradients
    )
    for name in set(training_report) - gradients:
        assert id(getattr(lm.training_quantizers, name)) in kept
    # Delayed scaling: a pass quantizes with the scale the passes before
    # observed, so the second pass is the first to use a calibrated scale.
    lm.enable_observers()
    ids = torch.randint(0, VOCAB, (1, 64))
    with torch.no_grad():
        for _ in range(2):
            lm.graphs["prefill"](input_ids=ids, cache_position=torch.arange(64))
    for name in shared:
        assert decode[name].scale.item() != 1.0


def observing(model):
    return {m for m in model.fake_quants() if m.observer_enabled}


def test_modes_switch_the_observers(lm):
    every = set(lm.fake_quants())
    gradients = {
        getattr(lm.training_quantizers, n)
        for n in lm.training_quantizers.gradients
    }
    # Eval fixes every scale but the gradients', which keep moving.
    lm.eval()
    assert observing(lm) == gradients
    assert not lm.training and not lm.model.training
    lm.train()
    assert observing(lm) == every
    assert lm.training and lm.model.training
    lm.eval()
    lm.enable_observers()
    assert observing(lm) == every
    lm.disable_observers()
    assert observing(lm) == gradients


def test_training_step_and_eval_loss(lm):
    ids = torch.randint(0, VOCAB, (1, 64))
    lm.train()
    lm.model.zero_grad()
    lm(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    assert all(p.grad is not None for p in lm.model.parameters())
    lm.eval()
    scales = [m.scale.clone() for m in lm.fake_quants()]
    with torch.no_grad():
        losses = [
            lm(input_ids=ids, labels=ids, use_cache=False).loss
            for _ in range(2)
        ]
    assert torch.isfinite(losses[0]) and torch.equal(*losses)
    for module, scale in zip(lm.fake_quants(), scales):
        assert torch.equal(module.scale, scale)


def test_generate_takes_stop_token_tuples(lm):
    prompt = torch.randint(0, VOCAB, (1, 8))
    stops = tuple(range(VOCAB))
    assert lm.generate(prompt, NEW_TOKENS, eos_token_id=stops).shape[1] == 9


def test_short_prompt_refreshes_frozen_reads(lm):
    """A prompt under one prefill block runs no prefill, but the main caches
    were emptied, so frozen reads of them are recomputed all the same."""
    if lm.split is None:
        pytest.skip("a whole cache has no frozen reads")
    refresh = lm._refresh = Mock()
    try:
        lm.generate(torch.randint(0, VOCAB, (1, 8)), NEW_TOKENS)
    finally:
        lm._refresh = None
    refresh.assert_called_once()


def test_cache_buffers_are_shared(lm):
    """Both inference graphs hold each cache buffer, the residuals of a
    split cache included, as one tensor, and the prefill graph itself
    writes the residuals."""
    prefill, decode = lm.graphs["prefill"], lm.graphs["decode"]
    names = [n for n, _ in decode.named_buffers() if _CACHE_BUFFER.match(n)]
    assert any(n.endswith("_residual") for n in names) == (lm.split is not None)
    for name in names:
        assert prefill.get_buffer(name) is decode.get_buffer(name)
    written = {
        n.args[0].target
        for n in prefill.graph.nodes
        if n.target is torch.ops.aten.index_copy_.default
    }
    assert {n for n in names if n.endswith("_residual")} <= written


@pytest.fixture(scope="module")
def tail_lm():
    """The tiny Llama over a decode cache split at two prefill blocks, so a
    prompt of an odd number of blocks leaves a tail in the residual."""
    return QuantizedLM(
        tiny_llama(),
        {"prefill": quantizer(), "decode": quantizer(residual_attention=True)},
        context_length=3 * SPLIT,
        cache_len=4 * SPLIT,
        phases=("prefill", "decode"),
        rewrite=lambda graph, phase: split_kv_cache(graph, 2 * SPLIT, 0),
    )


@pytest.mark.parametrize("length", [64, 128, 192, 200])
def test_prefill_writes_the_split_cache(tail_lm, length):
    """After a prompt, each main cache holds its whole chunks and nothing
    above them, and each residual the tail and nothing above it: the
    positions the prefill graph wrote equal the HF model's own cache bit
    for bit, as both attend over the prompt's keys alone; the decode
    steps past the prefill's blocks sum their attention in another order.
    """
    lm = tail_lm
    for fake_quant in lm.fake_quants():
        fake_quant.disable_fake_quant()
    prompt = torch.randint(0, VOCAB, (1, length))
    cache = DynamicCache()
    with torch.no_grad():
        lm.model(prompt, past_key_values=cache, use_cache=True)
        lm.prefill(prompt)
    for fake_quant in lm.fake_quants():
        fake_quant.enable_fake_quant()
    prefilled = min(length, lm.context_length) // SEQ_BLOCK * SEQ_BLOCK
    done = length // lm.split * lm.split
    tail = length - done

    def check(held, expected, start):
        """``held`` holds ``expected`` from position ``start`` and nothing
        above it."""
        # Past the prefill, HF's one forward and the decode steps sum in
        # another order.
        exact = prefilled - start if length == prefilled else 0
        exact = min(max(exact, 0), expected.shape[2])
        assert torch.equal(held[:, :, :exact], expected[:, :, :exact])
        assert torch.allclose(
            held[:, :, : expected.shape[2]], expected, atol=1e-6
        )
        assert not held[:, :, expected.shape[2] :].any()

    for i, layer in enumerate(cache.layers):
        for kind, expected in (("key", layer.keys), ("value", layer.values)):
            decode = lm.graphs["decode"]
            check(
                decode.get_buffer(f"{kind}_cache_{i}"), expected[:, :, :done], 0
            )
            check(
                decode.get_buffer(f"{kind}_cache_{i}_residual"),
                expected[:, :, done:],
                done,
            )


def test_freeze_bakes_the_weights_once():
    """Frozen, the training forward reads the baked weights raw: its loss
    is the prepared forward's, and the baked fake-quants are off."""
    lm = QuantizedLM(
        tiny_llama(),
        quantizer(),
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
    )
    inputs = {"input_ids": torch.randint(0, VOCAB, (1, 64)), "use_cache": False}
    inputs["labels"] = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(2):
            lm(**inputs)
        lm.eval()
        before = lm(**inputs).loss
        lm.freeze()
        after = lm(**inputs).loss
    assert torch.equal(before, after)
    weights = [m for m in lm.fake_quants() if not m.fake_quant_enabled]
    assert len(weights) == 7 * lm.config.num_hidden_layers + 1
    remaining = {id(m) for m in lm.graphs["prefill"].modules()}
    assert all(id(m) not in remaining for m in weights)


def test_frozen_split_cache_generates_alike():
    """Frozen over a split cache, generation past a fold gives the tokens
    the prepared graphs give, and a second freeze raises."""
    lm = QuantizedLM(
        tiny_llama(),
        {
            "prefill": quantizer(),
            "decode": quantizer(residual_attention=True),
            "training": quantizer(),
        },
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
        rewrite=split_caches,
    )
    prompt = torch.randint(0, VOCAB, (1, 120))
    new_tokens = 2 * SPLIT - prompt.shape[1] + NEW_TOKENS
    lm.generate(prompt, new_tokens)
    lm.eval()
    expected = lm.generate(prompt, new_tokens)
    lm.freeze()
    assert torch.equal(lm.generate(prompt, new_tokens), expected)
    with pytest.raises(ValueError):
        lm.freeze()
    # The fake-quants the frozen cache reads run outside the decode graph
    # still follow the mode.
    lm.train()
    assert all(bool(m.observer_enabled) for m in lm.fake_quants())
    lm.eval()
    inference = [
        m
        for m in lm.inference_quantizers.modules()
        if isinstance(m, FakeQuantizeBase)
    ]
    assert not any(bool(m.observer_enabled) for m in inference)


def test_prefill_alone_returns_every_logit():
    """Built with only a prefill graph keeping every position, the model
    gives the HF model's logits with its fake-quants off, and raises where
    it would need the graphs it was built without."""
    model = tiny_llama()
    reference = copy.deepcopy(model)
    lm = QuantizedLM(
        model,
        quantizer(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
        phases=("prefill",),
        logits_to_keep=0,
    )
    assert set(lm.graphs) == {"prefill"}
    for fake_quant in lm.fake_quants():
        fake_quant.disable_fake_quant()
    ids = torch.randint(0, VOCAB, (1, 64))
    with torch.no_grad():
        logits = lm.graphs["prefill"](
            input_ids=ids, cache_position=torch.arange(64)
        )
        expected = reference(ids).logits
    torch.testing.assert_close(logits, expected)
    with pytest.raises(RuntimeError):
        lm.prefill(torch.randint(0, VOCAB, (1, 70)))
    with pytest.raises(RuntimeError):
        lm(input_ids=ids, labels=ids)


def test_training_alone_trains():
    lm = QuantizedLM(
        tiny_llama(),
        quantizer(),
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
        phases=("training",),
    )
    assert "prefill" not in lm.graphs and "decode" not in lm.graphs
    inputs = {"input_ids": torch.randint(0, VOCAB, (1, 64)), "use_cache": False}
    inputs["labels"] = inputs["input_ids"]
    lm.train()
    lm(**inputs).loss.backward()
    assert lm.model.model.layers[0].self_attn.q_proj.weight.grad is not None
    with pytest.raises(RuntimeError):
        lm.prefill(torch.randint(0, VOCAB, (1, 64)))
    with pytest.raises(RuntimeError):
        lm.freeze()
    with pytest.raises(RuntimeError):
        lm.convert()


@pytest.mark.parametrize("phases", [(), ("prefill", "verify"), ("training",)])
def test_bad_phases_raise(phases):
    """No phase, an unknown one, or training without its example."""
    with pytest.raises(ValueError):
        QuantizedLM(
            tiny_llama(),
            quantizer(),
            context_length=CONTEXT_LENGTH,
            cache_len=CACHE_LEN,
            phases=phases,
        )


def test_state_dict_keeps_the_model_names(lm):
    keys = lm.state_dict()
    assert "model.model.layers.0.self_attn.q_proj.weight" in keys
    assert any(k.startswith("model.training_quantizers.") for k in keys)
    # Every fake-quant of the inference graphs is saved, shared or not.
    decode_report, _ = lm.reports
    for name in decode_report:
        assert f"inference_quantizers.decode.{name}.scale" in keys
    assert not lm.load_state_dict(keys).missing_keys


def test_phases_may_quantize_differently():
    """FP8 training, int8-activation prefill and FP8 decode: only the decode
    and training fake-quants of one operand are shared, and conversion
    gives the prefill its own copies of the weights."""
    lm = QuantizedLM(
        tiny_llama(),
        {
            "prefill": quantizer(activation=INT8, weight=INT8),
            "decode": quantizer(),
            "training": quantizer(),
        },
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
    )
    decode_report, training_report = lm.reports
    assert set(decode_report.values()) == {"mismatched"}
    gradients = lm.training_quantizers.gradients
    assert {n for n, r in training_report.items() if r == "mismatched"} == (
        set(training_report) - gradients
    )
    prompt = torch.randint(0, VOCAB, (1, 70))
    lm.generate(prompt, NEW_TOKENS)
    weight = lm.model.model.layers[0].self_attn.q_proj.weight
    raw = weight.detach().clone()
    lm.convert()
    assert torch.equal(weight, raw)
    assert lm.generate(prompt, NEW_TOKENS).shape == (1, 70 + NEW_TOKENS)


class Regression(nn.Module):
    """Two linears whose forward returns the MSE loss when given labels."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 32)
        self.b = nn.Linear(32, 8)

    def forward(self, x, labels=None):
        y = self.b(torch.relu(self.a(x)))
        return y if labels is None else nn.functional.mse_loss(y, labels)


def quantized_regression():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    quantized = QuantizedModel(
        Regression(),
        get_default_quantizer(input_activation=ACT, weight=ACT, error=ERROR),
        Example((x,)),
        Example((x,), {"labels": torch.randn(4, 8)}),
    )
    return quantized, x


def test_quantized_module_shares_weights_and_infers():
    quantized, x = quantized_regression()
    (report,) = quantized.reports
    kept = {id(m) for m in fake_quants(quantized.graphs["inference"]).values()}
    # Both linears' weights and inputs are shared; the gradients are not.
    shared = [n for n, r in report.items() if r == "shared"]
    assert len(shared) == 4
    assert all(
        id(getattr(quantized.training_quantizers, n)) in kept for n in shared
    )
    assert quantized.infer(x).shape == (4, 8)
    loss = quantized(x, labels=torch.randn(4, 8))
    assert loss.ndim == 0


def test_derived_fake_quants_follow_the_shared_ones():
    """The biases' fake-quants derive their scale from the input's and the
    weight's: after sharing, the training graphs' derive from the shared
    objects, which the gradient program's lowering looks up by name."""
    quantized, x = quantized_regression()
    inference = {id(m) for m in quantized.graphs["inference"].modules()}
    derived = [
        m
        for m in quantized.training_quantizers.modules()
        if isinstance(m, _DerivedObserverOrFakeQuantize)
    ]
    assert len(derived) == 2
    assert all(id(s) in inference for m in derived for s in m.obs_or_fqs)
    quantized.train()
    quantized(x, labels=torch.randn(4, 8)).backward()
    quantized.model.zero_grad(set_to_none=True)
    assert gradient_program(quantized.model) is not None


def test_convert_bakes_the_inference_graphs():
    lm = QuantizedLM(
        tiny_llama(),
        quantizer(),
        training_example(),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
    )
    prompt = torch.randint(0, VOCAB, (1, 70))
    for _ in range(2):
        lm.generate(prompt, NEW_TOKENS)
    lm.convert()
    assert lm.converted
    assert not fake_quants(lm.graphs["prefill"])
    assert not lm.inference_quantizers
    lm.eval()
    assert lm.generate(prompt, NEW_TOKENS).shape == (1, 70 + NEW_TOKENS)
    with pytest.raises(RuntimeError):
        lm(input_ids=prompt, labels=prompt, use_cache=False)
    with pytest.raises(RuntimeError):
        lm.train()
    quantized, x = quantized_regression()
    quantized.convert()
    assert quantized.infer(x).shape == (4, 8)
    with pytest.raises(RuntimeError):
        quantized(x)


def test_convert_shares_block_quantized_weights():
    """Prefill and decode read one set of block-quantized weight buffers,
    the very tensors, so a paired compile places each weight once."""
    lm = QuantizedLM(
        tiny_llama(),
        quantizer(weight=MXINT8),
        context_length=CONTEXT_LENGTH,
        cache_len=CACHE_LEN,
        phases=("prefill", "decode"),
    )
    lm.generate(torch.randint(0, VOCAB, (1, 70)), NEW_TOKENS)
    lm.convert()
    weights = {
        phase: {
            name: id(buffer)
            for name, buffer in lm.graphs[phase].named_buffers()
            if "_weight_" in name
        }
        for phase in ("prefill", "decode")
    }
    assert weights["prefill"]
    assert set(weights["prefill"].values()) == set(weights["decode"].values())
    for phase in ("prefill", "decode"):
        assert lm.graphs[phase].meta["weight_layout"] == DEFAULT_WEIGHT_LAYOUT
