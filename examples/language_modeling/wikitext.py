import argparse
import logging

import torch
from datasets import load_dataset
from transformers.modeling_outputs import CausalLMOutput

from common import (
    add_model_args,
    evaluate_perplexity,
    load_model_and_tokenizer,
)
from quantization_configs import QUANTIZATION_CONFIGS, set_qconfig
from voyager_compiler import (
    QuantizedLM,
    add_experiment_args,
    get_default_quantizer,
    print_node_scope_tabular,
    with_execution_context,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    add_model_args(parser)
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride for processing the data",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Output directory for histograms"
    )
    parser.add_argument(
        "--qconfig", default=None, help="Quantization scheme for the model"
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="Print node scope information",
    )
    add_experiment_args(parser)
    return parser.parse_args()


def setup_quantized_model(
    model_id, quantizer, max_length, torch_dtype, device, print_model=False
):
    """Load the model and quantize it into a ``QuantizedLM`` with only a
    prefill graph, returning every position's logits, for
    ``evaluate_perplexity``'s windows: ``max_length`` tokens, or any shorter
    multiple of ``SEQ_BLOCK``.

    Args:
        model_id: Hub id or local path of the checkpoint.
        quantizer: Annotates every graph.
        max_length: Window length, a multiple of ``SEQ_BLOCK``.
        torch_dtype: ``"auto"`` or a torch dtype name, as on the CLI.
        device: The one device everything runs on.
        print_model: Print the forward graph and its node scopes.

    Returns:
        ``(lm, tokenizer)``.

    Raises:
        NotImplementedError: ``device`` is None, which asked for the graph
            sharded across every GPU.
    """
    if device is None:
        raise NotImplementedError(
            "A QuantizedLM runs on one device; sharding its graphs across "
            "GPUs is not supported.  Pass --gpu."
        )
    model, tokenizer = load_model_and_tokenizer(
        model_id, torch_dtype, attn_implementation="eager"
    )
    model.to(device)
    lm = QuantizedLM(
        model,
        quantizer,
        context_length=max_length,
        cache_len=max_length,
        phases=("prefill",),
        logits_to_keep=0,
    )
    if print_model:
        lm.graphs["prefill"].graph.print_tabular()
        print_node_scope_tabular(lm.graphs["prefill"])
    return lm, tokenizer


def prefill_loss(lm):
    """``evaluate_perplexity``'s model on ``lm``: a function returning the
    causal-LM loss of a window, computed from the prefill graph's logits
    with the model's own loss function.

    Args:
        lm: A ``QuantizedLM`` whose prefill graph returns every position's
            logits.

    Returns:
        ``loss(input_ids, labels, use_cache)``, whose ``use_cache`` only
        satisfies ``evaluate_perplexity``'s call.
    """
    prefill = lm.graphs["prefill"]
    vocab_size = lm.config.get_text_config().vocab_size

    def loss(input_ids, labels, use_cache):
        logits = prefill(
            input_ids=input_ids,
            cache_position=torch.arange(
                input_ids.shape[1], device=input_ids.device
            ),
        )
        return CausalLMOutput(
            loss=lm.model.loss_function(logits, labels, vocab_size)
        )

    return loss


@with_execution_context
def main(args):
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else None

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
    )
    quantizer.set_module_name("model.rotary_emb", None)

    if (qconfig := QUANTIZATION_CONFIGS.get(args.qconfig)) is not None:
        set_qconfig(quantizer, qconfig)

    model, tokenizer = setup_quantized_model(
        args.model_id,
        quantizer,
        args.max_length,
        args.torch_dtype,
        device,
        print_model=args.print_model,
    )

    if args.calibration_steps > 0:
        validation = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="validation"
        )
        calib_encodings = tokenizer(
            "\n\n".join(validation["text"]), return_tensors="pt"
        )

        evaluate_perplexity(
            prefill_loss(model),
            calib_encodings,
            args.max_length,
            args.stride,
            device,
            num_steps=args.calibration_steps,
        )

        model.disable_observers()

    model.freeze()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    ppl = evaluate_perplexity(
        prefill_loss(model),
        test_encodings,
        args.max_length,
        args.stride,
        device,
    )

    print(f"model:      {args.model_id}")
    print(f"max length: {args.max_length}")
    print(f"stride:     {args.stride}")
    print(f"perplexity: {ppl.item()}")


if __name__ == "__main__":
    main(parse_args())
