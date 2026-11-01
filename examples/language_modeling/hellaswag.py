"""Zero-shot HellaSwag accuracy of a causal language model.

Each example's four endings are scored by the log-likelihood the model gives
them after the context; ``acc`` picks the most likely ending and
``acc_norm`` the most likely per character, the normalization lm-eval and
lighteval report.  The context and endings are cleaned as lm-eval cleans
them.  The model is a Hugging Face model id (at ``--revision``) or a
checkpoint ``run_clm_no_trainer.py`` saved, rebuilt from ``--config_name``.

Example::

    python examples/language_modeling/hellaswag.py \\
        --checkpoint ~/claude/pretrain/bf16_2b/step_1000.pt
    python examples/language_modeling/hellaswag.py \\
        --model allenai/DataDecide-fineweb-edu-150M \\
        --revision step5000-seed-default
"""

import argparse
import json
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model id.")
    source.add_argument(
        "--checkpoint", help="Checkpoint run_clm_no_trainer.py saved."
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--config_name",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Config and tokenizer a checkpoint was trained with.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="RoPE base frequency a checkpoint was trained with.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score the first N only."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", default=None, help="JSON to write.")
    return parser.parse_args()


def clean(text):
    """lm-eval's HellaSwag cleanup of WikiHow markup."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_model(args, device):
    """The model and tokenizer to score, in bf16 on ``device``."""
    if args.model is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model.to(device).eval(), tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    config = AutoConfig.from_pretrained(args.config_name)
    config.rope_parameters["rope_theta"] = args.rope_theta
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Tied embeddings leave lm_head out of the trained parameters.
    loaded = model.load_state_dict(state["masters"], strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) - {"lm_head.weight"}:
        raise ValueError(f"{args.checkpoint} does not fit the config: {loaded}")
    return model.to(device, torch.bfloat16).eval(), tokenizer


@torch.no_grad()
def score(model, rows, device):
    """Log-likelihood of each row's ending tokens after its context tokens.

    Args:
        model: Causal language model.
        rows: ``(context ids, ending ids)`` pairs.
        device: Device the model is on.

    Returns:
        One summed log-likelihood per row.
    """
    width = max(len(c) + len(e) for c, e in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, (context, ending) in enumerate(rows):
        tokens = context + ending
        ids[i, : len(tokens)] = torch.tensor(tokens)
        mask[i, len(context) : len(tokens)] = True
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids).logits.float()
    logprobs = F.log_softmax(logits[:, :-1], dim=-1)
    taken = logprobs.gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (taken * mask[:, 1:]).sum(-1).tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args, device)
    data = load_dataset("Rowan/hellaswag", split="validation")
    if args.limit is not None:
        data = data.select(range(args.limit))

    rows, examples = [], []
    for example in data:
        context = clean(
            f"{example['activity_label']}: {example['ctx_a']} "
            f"{example['ctx_b'].capitalize()}"
        )
        endings = [clean(e) for e in example["endings"]]
        context_ids = tokenizer(context, add_special_tokens=False).input_ids
        for ending in endings:
            ending_ids = tokenizer(
                " " + ending, add_special_tokens=False
            ).input_ids
            rows.append((context_ids, ending_ids))
        examples.append((endings, int(example["label"])))

    loglikes = []
    for start in range(0, len(rows), args.batch_size):
        loglikes += score(model, rows[start : start + args.batch_size], device)

    correct = correct_norm = 0
    for i, (endings, label) in enumerate(examples):
        scores = loglikes[4 * i : 4 * i + 4]
        normed = [s / len(e) for s, e in zip(scores, endings)]
        correct += max(range(4), key=scores.__getitem__) == label
        correct_norm += max(range(4), key=normed.__getitem__) == label
    result = {
        "model": args.model or args.checkpoint,
        "revision": args.revision,
        "examples": len(examples),
        "acc": correct / len(examples),
        "acc_norm": correct_norm / len(examples),
    }
    print(json.dumps(result), flush=True)
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
