"""Pieces shared by the language-modeling evaluation scripts.

Model loading, the CLI options every script takes, sliding-window
perplexity, greedy prompt completion for the LongBench scripts, and the
quantized generator: a ``QuantizedLM`` over the compiler's prefill and
decode graphs, with their weights and KV-cache reads frozen.
"""

import json
import logging
import time

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from quantization_configs import (
    KIVI_RESIDUAL_LENGTH,
    QUANTIZATION_CONFIGS,
    annotate_kivi_cache,
    set_kivi_attention_qconfig,
    set_qconfig,
    set_residual_attention_qconfig,
)
from voyager_compiler import (
    QuantizedLM,
    get_default_quantizer,
    split_kv_cache,
)
from voyager_compiler.codegen import (
    match_and_fuse_rms_norm,
    remove_fp32_casts,
)
from voyager_compiler.quantization import load_codebooks
from voyager_compiler.quantization.codebook_optimizer import _quantized_operands

logger = logging.getLogger(__name__)


def add_model_args(parser):
    """Add the ``--model_id`` and ``--torch_dtype`` options to ``parser``."""
    parser.add_argument(
        "--model_id", required=True, help="Pretrained model identifier"
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this "
            "dtype. If `auto` is passed, the dtype will be automatically "
            "derived from the model's weights."
        ),
    )


def add_inference_args(parser):
    """Add the generation options shared by the LongBench scripts."""
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single GPU to run on; omit to shard across all visible GPUs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Prompt token budget; longer prompts are truncated in the "
            "middle. Defaults to the model's context length minus the "
            "generation length."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where predictions and scores go (default: pred*/<model>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate predictions that already exist in output_dir",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Do not wrap prompts in the tokenizer's chat template",
    )


def load_model_and_tokenizer(model_id, torch_dtype, **from_pretrained_kwargs):
    """Load a text-generation model and its tokenizer.

    A vision-language checkpoint registers only under the
    image-text-to-text auto class, so it is loaded through that one and
    driven text-only; everything else goes through the causal-LM class.

    Args:
        model_id: Hub id or local path of the checkpoint.
        torch_dtype: ``"auto"`` or the name of a torch dtype, as given on
            the command line.
        **from_pretrained_kwargs: Forwarded to ``from_pretrained``
            (``device_map``, ``attn_implementation``, ...).

    Returns:
        ``(model, tokenizer)``.
    """
    dtype = (
        torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)
    )
    config = AutoConfig.from_pretrained(model_id)
    auto_class = (
        AutoModelForCausalLM
        if type(config) in AutoModelForCausalLM._model_mapping
        else AutoModelForImageTextToText
    )
    model = auto_class.from_pretrained(
        model_id, dtype=dtype, **from_pretrained_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def evaluate_perplexity(
    model, encodings, max_length, stride, device, num_steps=None
):
    """Sliding-window perplexity evaluation. Returns a scalar tensor.

    If num_steps is set, exits early after that many windows — useful for
    activation observer calibration (caller can ignore the returned value).
    """
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    # Subtract max_length from seq_len so the last window is max_length long
    for i, begin_loc in enumerate(tqdm(range(0, seq_len - max_length, stride))):
        end_loc = min(begin_loc + max_length, seq_len)
        # May differ from stride on the last window.
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=False)

            # The loss is a CrossEntropyLoss mean over the valid labels. The
            # model scores only trg_len - 1 of them, because it shifts the
            # labels left by one internally.
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len or (num_steps is not None and i == num_steps - 1):
            break

    return torch.exp(torch.stack(nlls).mean())


def generate_answer(
    model, tokenizer, prompt, max_length, chat_template, **generate_kwargs
):
    """Greedily complete ``prompt`` and return the decoded completion.

    A prompt longer than ``max_length`` tokens is truncated in the middle so
    the instructions at both ends survive. With ``chat_template`` set, the
    prompt is sent as a single user turn through the tokenizer's chat
    template, which then supplies the special tokens.

    Args:
        model: Causal LM whose ``generate`` produces the completion.
        tokenizer: Its tokenizer.
        prompt: The prompt text.
        max_length: Prompt token budget.
        chat_template: Wrap the prompt in the chat template.
        **generate_kwargs: Forwarded to ``model.generate``; at least
            ``max_new_tokens``.
    """
    prompt_ids = tokenizer(prompt, truncation=False).input_ids
    if len(prompt_ids) > max_length:
        half = max_length // 2
        prompt = tokenizer.decode(
            prompt_ids[:half], skip_special_tokens=True
        ) + tokenizer.decode(prompt_ids[-half:], skip_special_tokens=True)
    if chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=not chat_template,
    ).to(model.device)
    context_length = inputs.input_ids.shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )[0]
    return tokenizer.decode(output[context_length:], skip_special_tokens=True)


# The scheme that quantizes nothing; the other names in QUANTIZATION_CONFIGS
# turn on the compiler's graph rewrites.  The softmax and RMS norm stay
# unquantized: the fp8 spec the compiler path puts on the softmax input sees
# the causal mask's ``finfo.min`` and scales every real score to zero.
BF16_CONFIG = "bf16"
# The rotary embedding's ``inv_freq @ position`` matmul stays unquantized; a
# regex, so it matches the model's own scope and the cache wrapper's.
ROTARY_SCOPE = r"model\.rotary_emb"


def add_quantization_args(parser):
    """Add the options that pick the quantized generator over the HF model:
    any of them set means the compiler's graphs run instead."""
    parser.add_argument(
        "--qconfig",
        default=None,
        choices=sorted(QUANTIZATION_CONFIGS),
        help=(
            "Scheme for the linears, attention and lm_head; with --kivi "
            "alone everything but the KV cache stays bf16"
        ),
    )
    parser.add_argument(
        "--decode_qconfig",
        default=None,
        choices=sorted(QUANTIZATION_CONFIGS),
        help=(
            "Scheme for the decode graph when it differs from --qconfig, "
            "e.g. mxlut4_int6_e5m3_a_int6 to decode with int6 activations"
        ),
    )
    parser.add_argument(
        "--kivi",
        action="store_true",
        help="Quantize the decode KV cache to KIVI's 2-bit layout",
    )
    parser.add_argument(
        "--residual_length",
        type=int,
        default=KIVI_RESIDUAL_LENGTH,
        help=(
            "Positions of the KIVI cache kept in full precision while their "
            "chunk fills; a multiple of the 64-token group"
        ),
    )
    parser.add_argument(
        "--codebooks",
        default=None,
        help="JSON of fitted tables to install, as codebook_eval.py dumps",
    )


def build_quantizer(qconfig, decode, kivi):
    """Return the quantizer for one graph.

    Args:
        qconfig: Name in ``QUANTIZATION_CONFIGS``.
        decode: The decode graph's quantizer: its split caches' residual
            halves of the attention stay unquantized.
        kivi: Point the attention matmuls at the int6 re-encode of a KIVI
            cache (decode only).
    """
    quantizer = get_default_quantizer()
    quantizer.set_module_name_object_type_order(
        ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )
    set_qconfig(quantizer, QUANTIZATION_CONFIGS[qconfig])
    if kivi:
        set_kivi_attention_qconfig(quantizer)
    elif decode:
        set_residual_attention_qconfig(quantizer)
    return quantizer


def rewrite_for_compiler(gm, model):
    """Apply the graph rewrites the compiler's Llama path makes before
    quantizing: RMSNorm fused into one rms_norm op, the softmax in bf16."""
    # get_decoder() is the text decoder, wherever the model nests it.
    layernorm = model.get_decoder().layers[0].input_layernorm
    hidden = layernorm.weight.shape[-1]
    example = torch.randn(1, 1, hidden, dtype=model.dtype, device=model.device)
    match_and_fuse_rms_norm(gm, layernorm, (example,))
    remove_fp32_casts(gm)


def install_codebooks(prefill, decode, path):
    """Install fitted tables into both prepared graphs.

    Both graphs name every operand under the cache wrapper's extra
    ``model.`` scope, so the tables are re-keyed; the decode graph quantizes
    its attention operands its own way, so only the tables it has an
    operand for go in.

    Args:
        prefill: Prepared prefill graph.
        decode: Prepared decode graph.
        path: JSON of tables, keyed by the model's own operand names, as
            ``codebook_eval.py`` dumps them.
    """
    with open(path) as handle:
        tables = json.load(handle)
    tables = {f"model.{name}": entry for name, entry in tables.items()}
    installed = load_codebooks(prefill, tables)
    logger.info("prefill: installed %d codebooks", len(installed))
    operands = _quantized_operands(decode, ())
    installed = load_codebooks(
        decode, {name: e for name, e in tables.items() if name in operands}
    )
    logger.info("decode: installed %d codebooks", len(installed))


def build_generator(
    model_id,
    torch_dtype,
    qconfig,
    decode_qconfig,
    kivi,
    residual_length,
    codebooks,
    context_length,
    cache_len,
    device,
):
    """Load the model and quantize it into a ``QuantizedLM`` with its
    prefill and decode graphs frozen.

    The graphs are quantized the way ``test_codegen.py``'s ``llama_prefill``
    / ``llama_decode_kivi`` paths quantize them, except that the softmax and
    RMS norm stay unquantized.  Both stay prepared, their observers on, and
    frozen (``QuantizedLM.freeze``): every weight is fake-quantized once,
    and what the decode graph reads from a split cache's main half once per
    chunk.  The training graphs the class captures take the prefill's
    quantizer.

    Args:
        model_id: Hub id or local path of the checkpoint.
        torch_dtype: ``"auto"`` or a torch dtype name, as on the CLI.
        qconfig: Name in ``QUANTIZATION_CONFIGS``; ``BF16_CONFIG`` leaves
            everything but the cache alone.
        decode_qconfig: The decode graph's scheme, or None for ``qconfig``.
        kivi: Quantize the decode KV cache to KIVI's 2-bit layout.
        residual_length: Positions of that cache kept in full precision
            while their chunk fills.
        codebooks: JSON of fitted tables to install, or None.
        context_length: Longest prompt the prefill graph takes, a multiple
            of ``SEQ_BLOCK``.
        cache_len: Slots in the KV cache, covering the longest prompt plus
            generation.
        device: Where everything runs.

    Returns:
        ``(lm, tokenizer)``.
    """
    started = time.monotonic()

    def mark(what):
        logger.info(
            "### %s (%.0f min)", what, (time.monotonic() - started) / 60
        )

    quantized = qconfig != BF16_CONFIG
    decode_qconfig = decode_qconfig or qconfig
    decode_quantized = decode_qconfig != BF16_CONFIG
    mark("loading the checkpoint")
    model, tokenizer = load_model_and_tokenizer(
        model_id, torch_dtype, attn_implementation="eager"
    )
    model.to(device)
    text_config = model.config.get_text_config()

    # A quantized cache -- KIVI's 2-bit one, or the int6 re-encode a
    # quantized qconfig's attention matmuls read -- is split; bf16 is not.
    # The decode graph is rewritten first; the prefill splits its writes
    # alike.
    split = residual_length if (kivi or decode_quantized) else None

    def rewrite(gm, phase):
        if (phase == "prefill" and quantized) or (
            phase == "decode" and decode_quantized
        ):
            rewrite_for_compiler(gm, model)
        if phase in ("prefill", "decode") and split is not None:
            split_kv_cache(gm, split, 0)
        if phase == "decode" and kivi:
            caches = annotate_kivi_cache(gm)
            expected = 2 * text_config.num_hidden_layers
            if caches != expected:
                raise RuntimeError(
                    f"annotated {caches} KV caches, expected {expected}"
                )

    mark(f"exporting the graphs over {cache_len} cache slots")
    lm = QuantizedLM(
        model,
        {
            "prefill": build_quantizer(qconfig, False, False),
            "decode": build_quantizer(
                decode_qconfig, True, kivi and decode_quantized
            ),
        },
        context_length=context_length,
        cache_len=cache_len,
        phases=("prefill", "decode"),
        rewrite=rewrite,
    )
    prefill, decode = lm.graphs["prefill"], lm.graphs["decode"]
    if codebooks:
        install_codebooks(prefill, decode, codebooks)
    mark("freezing")
    lm.freeze()
    lm.replay = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        mark(f"{torch.cuda.memory_allocated(device) / 2**30:.1f} GiB resident")
    return lm, tokenizer
