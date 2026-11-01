"""Llama prefill / decode / training quantization and lowering for
``compile_model.py``.

One ``QuantizedLM`` holds every graph: the prefill and decode graphs over a
static KV cache, both keeping only the last position's logits
(``logits_to_keep=1``) so the vocabulary projection lowers as a
matrix-vector product, and the training graphs.  ``llama_prefill`` /
``llama_decode`` / ``llama_decode_kivi`` compile one of the inference graphs
and ``--training`` the training step; a speculative-decoding verification
step (``llama_verify``) is a decode graph of ``--spec_length`` tokens, and
``llama_cache`` / ``llama_cache_kivi`` compile the decode graph with the
prefill that fills its KV cache.
"""

import copy
import logging
import math
import os
import re
import sys
from contextlib import contextmanager
from unittest.mock import patch

import torch
from datasets import load_dataset
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    masking_utils,
)

from voyager_compiler import (
    Example,
    QuantizationSpec,
    QuantizedLM,
    ShapeProp,
    compile,
    merge_sibling_linears,
    planned_buffer_layout,
    quantizes_kv_cache,
    split_kv_cache,
    transform,
)
from voyager_compiler.codegen import match_and_fuse_rms_norm
from voyager_compiler.hardware_config import AcceleratorConfig

from .compile_steps import (
    get_compile_args,
    get_transform_args,
    single_cpu_thread,
)

# ``set_qconfig`` lives in the language-modeling example, not the package.
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../examples/language_modeling"
        )
    )
)
from quantization_configs import (  # noqa: E402
    KIVI_BLOCK_SIZE,
    KIVI_RESIDUAL_LENGTH,
    QUANTIZATION_CONFIGS,
    set_kivi_attention_qconfig,
    set_qconfig,
    set_residual_attention_qconfig,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

# Generation slots the decode KV cache holds beyond the prefilled context.
DECODE_MAX_GEN = 128

# The rotary embedding's ``inv_freq @ position`` matmul is not an MXU op and
# stays unquantized.  A regex, so it matches both the training scope
# (``model.rotary_emb``) and the static-cache wrapper's
# (``model.model.rotary_emb``).
_ROTARY_SCOPE = r"model\.rotary_emb"

# The 2-bit KV cache the compiler lowers: KIVI's groups (keys along the
# sequence per channel, values along the head dim per token), with the
# group scale and zero point in fp8_e5m3, the format of the block scale
# the fused dequantize reads them beside.
_KIVI_KEY_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-2,scale=fp8_e5m3"
)
_KIVI_VALUE_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-1,scale=fp8_e5m3"
)
_KV_CACHE = re.compile(r"^(key|value)_cache_\d+$")
_KV_BUFFER = re.compile(r"^(key|value)_cache_\d+(_residual)?$")

# The stages compiled from the decode graph, and those compiling the decode
# graph together with the prefill that fills its KV cache.
_DECODE_STAGES = ("llama_decode", "llama_decode_kivi", "llama_verify")
CACHE_STAGES = ("llama_cache", "llama_cache_kivi")


def load_model(args):
    """Load the language model (its first ``--num_hidden_layers`` decoder
    layers, when given) and its tokenizer.

    A vision-language checkpoint registers only under the
    image-text-to-text auto class, so it is loaded through that one and
    driven text-only; everything else goes through the causal-LM class.
    """
    if args.model_name_or_path is None:
        args.model_name_or_path = DEFAULT_MODEL

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    text_config = config.get_text_config()
    n = getattr(args, "num_hidden_layers", None)
    if n:
        text_config.num_hidden_layers = n
        # StaticCache sizes itself from layer_types, which num_hidden_layers
        # does not truncate (Qwama-0.5B: 2 layers, 24 cache entries).
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:n]
    auto_class = (
        AutoModelForCausalLM
        if type(config) in AutoModelForCausalLM._model_mapping
        else AutoModelForImageTextToText
    )
    model = auto_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer


def _prompt_ids(tokenizer, length):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    return encodings.input_ids[:, :length]


def training_inputs(tokenizer, args):
    """The training graphs' batch: the ``--context_length`` prompt as both
    input and labels."""
    input_ids = _prompt_ids(tokenizer, args.context_length)
    return {"input_ids": input_ids, "labels": input_ids}


def max_cache_len(args, config):
    """Context plus generation budget, rounded up to a vector-lane multiple
    so the KV tensors stay block-aligned, and to a whole number of
    ``--residual_length`` chunks when the flag is given."""
    block = math.lcm(config.vector_lanes, args.residual_length or 1)
    raw = args.context_length + DECODE_MAX_GEN
    return -(-raw // block) * block


@contextmanager
def _sdpa_is_causal():
    """Export sdpa attention with ``is_causal`` instead of a materialized
    causal mask.  transformers builds the mask whenever it is tracing: it
    refuses the ``is_causal`` skip, since a baked-in flag would be wrong for
    a variable-length graph, and it cannot run the check that the batch
    holds no packed sequences.  A fixed-length, single-sequence prefill
    export wants the flag and has no packing.  A sliding-window layer whose
    window is shorter than the sequence keeps its mask."""

    def skip(padding_mask, query_length, kv_length, kv_offset, window=None):
        return (
            query_length == kv_length
            and (window is None or kv_length < window)
            and (padding_mask is None or bool(padding_mask.all()))
        )

    with (
        patch.object(masking_utils, "_ignore_causal_mask_sdpa", skip),
        patch.object(
            masking_utils, "find_packed_sequence_indices", lambda ids: None
        ),
    ):
        yield


def annotate_kivi_cache(gm):
    """Quantize the main KV caches of a split decode graph to the 2-bit
    layout the compiler lowers (``_KIVI_KEY_SPEC`` / ``_KIVI_VALUE_SPEC``).

    The observer sits on the main cache's read into the attention, so the
    whole buffer -- completed chunks and the zeros above them -- is
    quantized on the way in, and ``quant_folding`` bakes it and moves the
    quantize onto each chunk's fold.  The fold's own handle on the buffer,
    whose ``index_copy_`` writes a ``where``, is not a read; the residual
    buffers are not annotated.

    Args:
        gm: Decode graph after ``split_kv_cache``; annotated in place.

    Raises:
        RuntimeError: A cache is written directly, i.e. the graph was not
            split.
    """
    key_qspec = QuantizationSpec.from_str(_KIVI_KEY_SPEC)
    value_qspec = QuantizationSpec.from_str(_KIVI_VALUE_SPEC)
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        match = _KV_CACHE.match(str(node.target))
        if match is None:
            continue
        writes = [
            u
            for u in node.users
            if u.target is torch.ops.aten.index_copy_.default
        ]
        if writes:
            if writes[0].args[3].target is not torch.ops.aten.where.self:
                raise RuntimeError(
                    f"{node.target} is written directly: split the graph "
                    "with split_kv_cache before annotating the caches"
                )
            continue
        annotate_output_qspec(
            node, key_qspec if match.group(1) == "key" else value_qspec
        )


def _annotate_attention_mask(gm):
    """Quantize the prefill's causal mask to int1: the ``where`` over the
    boolean mask that the attention scores' ``add`` reads.  The static-cache
    export builds the mask from ``cache_position`` at run time, so the
    quantize runs with the graph rather than folding into a constant.

    Raises:
        RuntimeError: No causal-mask ``where`` feeds an ``add`` and the
            attention is not sdpa (which runs with no mask tensor to
            quantize).
    """
    qspec = QuantizationSpec.from_str("int1,qs=per_tensor_symmetric,qmax=1")
    masks = [
        n
        for n in gm.graph.nodes
        if n.target is torch.ops.aten.where.ScalarOther
        and any(u.target is torch.ops.aten.add.Tensor for u in n.users)
    ]
    if not masks:
        sdpa = any(
            n.target is torch.ops.aten.scaled_dot_product_attention.default
            for n in gm.graph.nodes
        )
        if not sdpa:
            raise RuntimeError("no causal-mask where node feeds an add")
        logger.info("no attention mask under sdpa")
    for mask in masks:
        annotate_output_qspec(mask, qspec)


def quantize_model(model, tokenizer, quantizer, args):
    """Quantize ``model`` as one ``QuantizedLM``, prepared with its
    observers on, building the phases the run needs: the training graphs
    under ``--training``, the prefill for ``llama_prefill``, and both
    inference graphs for a decode stage, whose cache the prefill fills.

    Every phase quantizes as ``quantizer`` says, the rotary embedding's
    matmul excepted.  The prefill and decode quantizers add ``--qconfig``'s
    table.  Under ``--residual_length``, a decode cache the attention reads
    quantized is split into completed chunks and a full-precision residual
    (``split_kv_cache``), its attention's residual halves unquantized, and
    the prefill splits its writes alike; without the flag the caches stay
    whole, and ``transform`` splits a cache whose blocks run along the
    positions.  Under a KIVI stage the main caches are quantized to 2 bits,
    which needs ``--residual_length``.
    Each graph's RMSNorms are fused before quantization; under
    ``--quantize_attention_mask`` the prefill's causal mask is int1; under
    ``--merge_linears`` each inference graph runs the linears reading one
    input as one GEMM over its own stacked copy of their weights, which no
    other graph shares.

    Args:
        model: The Hugging Face model, in eval mode.
        tokenizer: The model's tokenizer, which builds the example prompt.
        quantizer: The PT2E quantizer every phase derives its own from.
        args: The parsed command-line arguments.

    Returns:
        The ``QuantizedLM``.

    Raises:
        ValueError: A verification step would cross a residual chunk
            boundary, or a KIVI stage has no ``--residual_length`` or one
            that is not a multiple of the 2-bit group.
    """
    quantizer.set_module_name_object_type_order(
        _ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )
    quantizers = {
        phase: copy.deepcopy(quantizer)
        for phase in ("prefill", "decode", "training")
    }
    kivi = args.model in ("llama_decode_kivi", "llama_cache_kivi")
    tokens = args.spec_length if args.model == "llama_verify" else 1
    if args.qconfig is not None:
        for phase in ("prefill", "decode"):
            set_qconfig(quantizers[phase], QUANTIZATION_CONFIGS[args.qconfig])

    # The training graphs take no KV cache: a ``use_cache`` flag among the
    # example inputs would become a non-tensor input of the gradient
    # program.  Inference passes its own ``use_cache``.
    model.config.use_cache = False

    # get_decoder() is the text decoder, wherever the model nests it.
    layernorm = model.get_decoder().layers[0].input_layernorm

    config = AcceleratorConfig.from_args(args)
    # Whether the decode graph's caches were split.
    split = {"done": False}

    def rewrite(gm, phase):
        seq = tokens if phase == "decode" else 128
        example_input = torch.randn(
            1, seq, layernorm.weight.shape[-1], dtype=model.dtype
        )
        match_and_fuse_rms_norm(gm, layernorm, (example_input,))
        if args.merge_linears and phase != "training":
            merge_sibling_linears(gm)
        if phase == "prefill" and args.quantize_attention_mask:
            _annotate_attention_mask(gm)
        if phase == "prefill":
            # The decode graph was rewritten first: the prefill writes a
            # cache split there as the decode does.
            if split["done"]:
                split_kv_cache(gm, args.residual_length, 0)
            return
        if phase != "decode":
            return
        if kivi:
            if args.residual_length is None:
                raise ValueError(
                    "a KIVI cache takes --residual_length: its groups run "
                    "along the positions"
                )
            if args.residual_length % KIVI_BLOCK_SIZE:
                raise ValueError(
                    f"--residual_length {args.residual_length} is not a "
                    f"multiple of the {KIVI_BLOCK_SIZE}-token KIVI group"
                )
        length = args.residual_length
        if length is not None and (
            kivi or quantizes_kv_cache(gm, quantizers["decode"])
        ):
            if args.context_length % length + tokens > length:
                raise ValueError(
                    f"a {tokens}-token step from position "
                    f"{args.context_length} crosses a {length}-position "
                    "residual chunk boundary"
                )
            split_kv_cache(gm, length, 0)
            split["done"] = True
        if kivi:
            set_kivi_attention_qconfig(quantizers["decode"])
            annotate_kivi_cache(gm)
        elif split["done"]:
            set_residual_attention_qconfig(quantizers["decode"])

    if args.training:
        phases = ("training",)
    elif args.model == "llama_prefill":
        phases = ("prefill",)
    else:
        phases = ("prefill", "decode")
    return QuantizedLM(
        model,
        quantizers,
        Example((), training_inputs(tokenizer, args)),
        context_length=args.context_length,
        cache_len=max_cache_len(args, config),
        phases=phases,
        rewrite=rewrite,
        decode_tokens=tokens,
        backward_override=args.backward_override,
    )


def quantize_graphs(model, tokenizer, quantizer, args):
    """Quantize, calibrate and convert the inference graphs the stage
    ``args.model`` names.

    Calibration prefills the ``--context_length`` prompt and, with a decode
    graph, decodes the tokens after it, twice (delayed scaling), so the
    fake-quants see real cache contents; the graphs are then converted
    with the prompt in their cache.

    Returns:
        ``(lm, kwargs)``: the ``QuantizedLM`` and, by graph name, the inputs
        each graph is compiled for: the prompt, or the tokens after it.
    """
    lm = quantize_model(model, tokenizer, quantizer, args)
    length = args.context_length
    tokens = args.spec_length if args.model == "llama_verify" else 1
    prompt = _prompt_ids(tokenizer, length + tokens)
    kwargs = {
        "prefill": {
            "input_ids": prompt[:, :length],
            "cache_position": torch.arange(length),
        },
        "decode": {
            "input_ids": prompt[:, length:],
            "cache_position": torch.arange(length, length + tokens),
        },
    }
    kwargs = {name: kwargs[name] for name in lm.graphs if name in kwargs}
    if next(lm.fake_quants(), None) is not None:
        for _ in range(2):
            lm.prefill(prompt[:, :length])
            if "decode" in lm.graphs:
                lm.graphs["decode"](**kwargs["decode"])
    lm.eval()
    lm.prefill(prompt[:, :length])
    lm.convert(args.bias)
    return lm, kwargs


def quantize_stage(model, tokenizer, quantizer, args):
    """The converted graph of the one-graph inference stage ``args.model``
    names (``llama_prefill`` / ``llama_decode`` / ``llama_decode_kivi`` /
    ``llama_verify``) and the inputs it is compiled for
    (``quantize_graphs``)."""
    lm, kwargs = quantize_graphs(model, tokenizer, quantizer, args)
    name = "decode" if args.model in _DECODE_STAGES else "prefill"
    return lm.graphs[name], kwargs[name]


def compile_cache_programs(model, tokenizer, quantizer, fusion_patterns, args):
    """Compile the decode graph and the prefill that fills its KV cache as
    one pair (``llama_cache`` / ``llama_cache_kivi``).

    The decode graph is transformed first; the prefill then stores the
    prompt's whole chunks in the decode graph's cache format
    (``transform``'s ``cache_formats``) and, compiled after it, places every
    buffer the two share at the decode graph's address
    (``planned_buffer_layout``).  Each program goes into its own directory
    under ``--model_output_dir``.

    Returns:
        ``(programs, reference, lowered)``, each keyed ``"prefill"`` and
        ``"decode"``: the compiled programs, the logits the converted graphs
        give run prefill then decode on the shared cache, and the same from
        the compiled graphs under ``--debug``, else ``None``.
    """
    lm, kwargs = quantize_graphs(model, tokenizer, quantizer, args)
    programs = {name: lm.graphs[name] for name in ("prefill", "decode")}

    def run():
        return {
            name: ShapeProp(program).propagate(
                *tree_flatten(((), kwargs[name]))[0]
            )
            for name, program in programs.items()
        }

    reference = run()
    transform_args = get_transform_args(args, fusion_patterns)
    transform(programs["decode"], (), kwargs["decode"], **transform_args)
    transform(
        programs["prefill"],
        (),
        kwargs["prefill"],
        cache_formats=programs["decode"].meta["cache_formats"],
        **transform_args,
    )
    compile_args = get_compile_args(args)
    out_dir = compile_args.pop("output_dir")
    layout = None
    for name in ("decode", "prefill"):
        compile(
            programs[name],
            (),
            kwargs[name],
            output_dir=os.path.join(out_dir, name),
            dram_layout=layout,
            **compile_args,
        )
        layout = planned_buffer_layout(
            programs[name], programs["prefill"], transform_args["config"]
        )
    if not args.debug:
        return programs, reference, None
    with single_cpu_thread():
        return programs, reference, run()


def kv_cache_state(gm):
    """A copy of every KV-cache buffer, to put back with
    ``restore_kv_cache`` before the graph is run again.

    The decode step is not idempotent: the step that completes a chunk folds
    the residual into the main cache, and a second run of it would then read
    the chunk from both.
    """
    return {
        name: buffer.clone()
        for name, buffer in gm.named_buffers()
        if _KV_BUFFER.match(name)
    }


def restore_kv_cache(gm, state):
    for name, saved in state.items():
        getattr(gm, name).copy_(saved)


def quantize_and_dump_model(model, tokenizer, quantizer, fusion_patterns, args):
    """Quantize the inference stage ``args.model`` names (``quantize_stage``),
    then transform and compile it.  Returns ``(gm, old_output,
    new_output)``; ``new_output`` is ``None`` unless ``--debug`` re-runs the
    lowered graph."""
    gm, example_kwargs = quantize_stage(model, tokenizer, quantizer, args)
    flatten_args, _ = tree_flatten(((), example_kwargs))
    old_output = ShapeProp(gm).propagate(*flatten_args)

    transform(
        gm, (), example_kwargs, **get_transform_args(args, fusion_patterns)
    )
    compile(gm, (), example_kwargs, **get_compile_args(args))
    gm.graph.print_tabular()

    new_output = None
    if args.debug:
        with single_cpu_thread():
            new_output = gm(**example_kwargs)
    return gm, old_output, new_output
