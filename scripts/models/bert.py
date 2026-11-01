import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

from voyager_compiler import (
    Example,
    QuantizedModel,
    compile,
    transform,
)

from .compile_steps import (
    get_transform_args,
    get_compile_args,
    single_cpu_thread,
)


def load_model(args):
    if args.model_name_or_path is None:
        args.model_name_or_path = "bert-base-uncased"

    layers = getattr(args, "num_hidden_layers", None)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager",
        **({"num_hidden_layers": layers} if layers else {}),
    )

    model.eval()
    model.to(torch.bfloat16 if args.bf16 else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer


def _inputs(batch):
    """The model's inputs in ``batch``, labels left out."""
    return {k: v for k, v in batch.items() if k != "labels"}


def quantize_model(model, quantizer, calibration_data, args):
    """Quantize the whole ``model``, embeddings to classifier, as one
    ``QuantizedModel`` prepared with its observers on, for training under
    ``--training`` and for inference otherwise; the pooler and classifier
    stay unquantized.  The graphs take a GLUE batch of ``calibration_data``
    as their example, the training ones with its labels."""
    batch = next(
        iter(
            DataLoader(
                calibration_data,
                collate_fn=default_data_collator,
                batch_size=1,
            )
        )
    )
    quantizer.set_module_name("pooler", None)
    quantizer.set_module_name("classifier", None)
    return QuantizedModel(
        model,
        quantizer,
        Example((), _inputs(batch)),
        Example((), batch),
        phases=("training",) if args.training else ("inference",),
        backward_override=args.backward_override,
    )


def quantize_and_dump_model(
    model, quantizer, calibration_data, fusion_patterns, args
):
    """Quantize ``model`` (``quantize_model``), calibrate its inference graph
    on ``--calibration_steps`` batches, convert it, then transform and
    compile it.  Returns ``(gm, old_output, new_output)``; ``new_output`` is
    ``None`` unless ``--debug`` re-runs the lowered graph."""
    quantized = quantize_model(model, quantizer, calibration_data, args)
    calibration_dataloader = DataLoader(
        calibration_data, collate_fn=default_data_collator, batch_size=1
    )
    example_kwargs = _inputs(next(iter(calibration_dataloader)))

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(calibration_dataloader, desc="Calibrating BERT")
        ):
            if i == args.calibration_steps:
                break
            quantized.infer(**_inputs(batch))
    quantized.eval()
    quantized.convert(args.bias)
    gm = quantized.graphs["inference"]

    old_output = gm(**example_kwargs)

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


def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(
        dataset, collate_fn=default_data_collator, batch_size=1
    )

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating BERT"):
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            correct_predictions += (prediction == inputs["labels"]).sum().item()
            total_samples += inputs["labels"].size(0)

    print(f"BERT Accuracy: {correct_predictions / total_samples:.4f}")


def evaluate_gm(gm, dataset):
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataset, desc="Evaluating Quantized BERT"):
            label = batch["labels"]
            outputs = gm(**_inputs(batch))
            logits = (
                outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            )
            prediction = torch.argmax(logits, dim=-1)
            if prediction == label:
                correct_predictions += 1
            total_samples += 1

    print(f"Quantized BERT Accuracy: {correct_predictions / total_samples:.4f}")
