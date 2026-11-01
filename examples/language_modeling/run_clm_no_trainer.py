#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain a causal language model, optionally quantized.

Adapted from Hugging Face's ``run_clm_no_trainer.py``, with the data loading
and optimizer setup of Karpathy's build-nanogpt.  The model is built from
``--config_name`` with random weights, or loaded from the pretrained
``--model_name_or_path`` to continue its training with a fresh optimizer,
and trains on the shards ``pretokenize.py`` writes, read in order in
fixed-size batches, so every run sees the same tokens and a resumed run
continues exactly.

The defaults follow SmolLM2-135M's pretraining recipe
(``huggingface/smollm``, ``text/pretraining/smollm2/config_smollm2_135M.yaml``):
2048-token sequences, 1M tokens per step (8 x 64 accumulation x 2048),
AdamW at 3e-3 with betas (0.9, 0.95) and weight decay 0.01, gradients
clipped to 1.0, and a warmup-stable-decay schedule that warms up linearly
over 2000 steps and decays linearly to zero over the last 20%.  The recipe
trains with a RoPE theta of 10000 (``--rope_theta 10000``), not the 100000
of the released config, which was raised for long-context training
afterwards.

The quantization flags pick the mode::

    flags                          forward GEMMs  backward GEMMs
    -----------------------------  -------------  --------------
    none                           unquantized    unquantized
    --weight --activation          quantized      unquantized (1)
    --weight --activation --error  quantized      quantized

(1) Reading the forward's original operands or their quantized values, as
``--backward_override`` says.

Quantized GEMMs are fake-quant, run eagerly, unless ``--compile_step``
compiles the training step -- forward, loss and backward -- as one program
(``TrainingStep``) that calls the same fake-quants, or ``--gpu_kernels``
runs them as real FP8 / NVFP4 kernels in that program, lowered by
``backend/gpu``'s ``lower_training``.  Under ``--compile_step`` each weight
is quantized once per optimizer step, not once per micro-batch, unless it
rounds stochastically, and the steps inside a codebook refit window still
run eagerly.
``lm_head``, the rotary embedding and attention's QK^T and PV stay
unquantized unless
``--quantize_lm_head`` / ``--quantize_attention``, as do the decoder layers
``--bf16_layers_at_start`` / ``--bf16_layers_at_end`` name.  The optimizer
updates FP32 master weights, from gradients accumulated in FP32 and clipped
to ``--max_grad_norm``; the model computes in bf16 with ``--bf16``.
Lookup-table formats (``lut<I>_to_<E>``) are refitted to the training
tensors every ``--codebook_refit_steps`` steps by ``CodebookRefitter``.

Example::

    python examples/language_modeling/run_clm_no_trainer.py \\
        --data_dir ~/claude/datasets/fineweb-edu-10BT-smollm2 \\
        --output_dir ~/claude/pretrain/mxint8 --max_train_steps 5000 --bf16 \\
        --activation int8,qs=microscaling,bs=32 \\
        --weight int8,qs=microscaling,bs=32 \\
        --error int8,qs=microscaling,bs=32
"""

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import time
from unittest.mock import patch

import numpy as np
import torch
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
    masking_utils,
    set_seed,
)

from voyager_compiler import (
    TrainingStep,
    add_experiment_args,
    get_default_quantizer,
    prepare_training,
    with_execution_context,
)
from voyager_compiler.backend.gpu.training import lower_training
from voyager_compiler.quantization import (
    CodebookRefitter,
    Weighting,
    in_refit_window,
    refit_window,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a causal language model."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory of token shards written by pretokenize.py.",
    )
    parser.add_argument(
        "--config_name",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model config to build the model from, with random weights.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help=(
            "Pretrained model to continue training, with its config in "
            "place of --config_name."
        ),
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=None,
        help="RoPE base frequency, in place of the config's.",
    )
    parser.add_argument(
        "--config_overrides",
        default=None,
        help=(
            "Config fields to override, e.g. "
            '"num_hidden_layers=6,hidden_size=512".'
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Tokens per sequence, and the model's context length.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        default="warmup_stable_decay",
        choices=["cosine_with_min_lr", "warmup_stable_decay"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Final learning rate as a fraction of --learning_rate.",
    )
    parser.add_argument(
        "--decay_ratio",
        type=float,
        default=0.2,
        help="Fraction of the steps warmup_stable_decay decays over.",
    )
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=20,
        help="Validation batches each evaluation averages the loss over.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Save a checkpoint every this many steps; always at the end.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Checkpoint to continue from.  Under the quantization settings "
            "it was saved with, the run continues exactly; under others it "
            "branches: weights, optimizer, schedule, data position and RNG "
            "carry over, and the quantizers start fresh."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="Quantize lm_head like the other linears.",
    )
    parser.add_argument(
        "--quantize_attention",
        action="store_true",
        help="Quantize attention's QK^T and PV matmuls.",
    )
    parser.add_argument(
        "--bf16_layers_at_start",
        type=int,
        default=0,
        help="Number of first decoder layers left unquantized.",
    )
    parser.add_argument(
        "--bf16_layers_at_end",
        type=int,
        default=0,
        help="Number of last decoder layers left unquantized.",
    )
    parser.add_argument(
        "--gpu_kernels",
        action="store_true",
        help=(
            "Run the quantized GEMMs as real FP8 / NVFP4 kernels "
            "(backend/gpu's lower_training) instead of fake-quant; needs "
            "--error."
        ),
    )
    parser.add_argument(
        "--compile_step",
        action="store_true",
        help=(
            "Compile the fake-quant training step -- forward, loss and "
            "backward -- as one program.  Steps inside a codebook refit "
            "window run eagerly."
        ),
    )
    parser.add_argument(
        "--codebook_refit_steps",
        type=int,
        default=0,
        help="Refit the forward lookup tables every this many steps; 0 never.",
    )
    parser.add_argument(
        "--codebook_collect_steps",
        type=int,
        default=4,
        help="Steps before each refit its statistics are collected over.",
    )
    parser.add_argument(
        "--codebook_weighting",
        choices=[Weighting.PARTNER.value, Weighting.FISHER_ACTIVATIONS.value],
        default=Weighting.PARTNER.value,
        help=(
            "What weights each sample of a refit: the partner's channel "
            "energy, or the activations' own loss-gradient Fisher diagonal."
        ),
    )
    parser.add_argument(
        "--codebook_refit_gradients",
        action="store_true",
        help="Refit the gradients' lookup tables too.",
    )
    add_experiment_args(parser)
    parser.set_defaults(log_level="INFO")
    args = parser.parse_args()
    if args.gpu_kernels and args.error is None:
        parser.error("--gpu_kernels lowers a quantized training step: --error")
    if args.gpu_kernels and args.codebook_refit_steps:
        parser.error("a codebook refit hooks fake-quants: no --gpu_kernels")
    if args.compile_step and args.gpu_kernels:
        parser.error("--gpu_kernels compiles the step too: no --compile_step")
    return args


@contextlib.contextmanager
def causal_sdpa():
    """Capture SDPA attention as ``is_causal`` instead of a causal mask.

    transformers builds the mask whenever it is tracing, since a baked-in
    flag would be wrong for a graph of variable length or with packed
    sequences.  Training batches have one fixed length and no padding.
    """

    def skip(padding_mask, query_length, kv_length, *args, **kwargs):
        return query_length == kv_length and (
            padding_mask is None or bool(padding_mask.all())
        )

    with (
        patch.object(masking_utils, "_ignore_causal_mask_sdpa", skip),
        patch.object(
            masking_utils, "find_packed_sequence_indices", lambda ids: None
        ),
    ):
        yield


class TokenStream:
    """Batches of token rows read in order from token shards.

    The shards are read one after another, and from the first again after
    the last; the tail of a shard too short for a whole batch is skipped.
    ``shard`` and ``offset`` are the read position.

    Args:
        files: Shard paths.
        dtype: Dtype the shards store token ids in.
        batch_size: Rows per batch.
        block_size: Tokens per row.
    """

    def __init__(self, files, dtype, batch_size, block_size):
        self.shards = [np.memmap(f, dtype=dtype, mode="r") for f in files]
        self.shape = (batch_size, block_size)
        self.shard = 0
        self.offset = 0

    def next_batch(self) -> torch.Tensor:
        size = self.shape[0] * self.shape[1]
        if self.offset + size > len(self.shards[self.shard]):
            self.shard = (self.shard + 1) % len(self.shards)
            self.offset = 0
        tokens = self.shards[self.shard][self.offset : self.offset + size]
        self.offset += size
        return torch.from_numpy(tokens.astype(np.int64)).view(self.shape)


def build_quantizer(args, num_layers):
    """The quantizer the flags build, with the unquantized GEMMs excluded.

    Args:
        args: Parsed command-line arguments.
        num_layers: Number of decoder layers in the model.

    Returns:
        The configured quantizer.
    """
    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
        error=args.error,
        random_hadamard_transform=args.random_hadamard_transform,
    )
    quantizer.set_module_name("rotary_emb", None)
    if not args.quantize_lm_head:
        quantizer.set_module_name("lm_head", None)
    if not args.quantize_attention:
        for op in (
            torch.ops.aten.matmul.default,
            torch.ops.aten.scaled_dot_product_attention.default,
        ):
            quantizer.set_module_name_object_type(r"self_attn$", op, None)
    end = num_layers - args.bf16_layers_at_end
    for layer in [*range(args.bf16_layers_at_start), *range(end, num_layers)]:
        quantizer.set_module_name(rf"layers\.{layer}$", None)
    return quantizer


def evaluate(model, loss_of, stream, num_batches, device):
    """Mean loss over ``stream``'s next ``num_batches``.

    The quantizers' state, such as a delayed scaling's amax history, is put
    back afterwards, so evaluating leaves training as it was.

    Args:
        model: The module training runs, switched to eval mode meanwhile.
        loss_of: Called with a batch of token ids; returns its loss.
        stream: Token stream to read the batches from.
        num_batches: Batches to average over.
        device: Device the batches go to.
    """
    quantizers = getattr(model, "training_quantizers", None)
    if quantizers is not None:
        saved = copy.deepcopy(quantizers.state_dict())
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            total += loss_of(stream.next_batch().to(device)).item()
    model.train()
    if quantizers is not None:
        quantizers.load_state_dict(saved)
    return total / num_batches


@with_execution_context
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    files = [os.path.join(args.data_dir, s["file"]) for s in meta["shards"]]
    shape = (args.per_device_train_batch_size, args.block_size)
    # The first shard is the validation split.
    train_stream = TokenStream(files[1:], meta["dtype"], *shape)

    if torch.cuda.is_available():
        device = torch.device(
            "cuda" if args.gpu is None else f"cuda:{args.gpu}"
        )
    else:
        logger.warning("CUDA not available; training on the CPU.")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path or args.config_name
    )
    if args.config_overrides is not None:
        config.update_from_string(args.config_overrides)
    if args.rope_theta is not None:
        config.rope_parameters["rope_theta"] = args.rope_theta
    config.max_position_embeddings = args.block_size
    config.use_cache = False
    # Quantizing QK^T and PV needs them as matmuls; otherwise attention runs
    # fused, which never materializes the score matrix.
    attn_implementation = "eager" if args.quantize_attention else "sdpa"
    if args.model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation=attn_implementation
        )
    if args.bf16:
        model.bfloat16()
    model.to(device)
    logger.info(f"#Params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Tokens: {meta['tokenizer']} ids from {args.data_dir}")

    example = torch.zeros(shape, dtype=torch.long, device=device)
    example = {"input_ids": example, "labels": example}
    specs = (args.activation, args.weight, args.error)
    if any(spec is not None for spec in specs):
        with causal_sdpa():
            model = prepare_training(
                model.train(),
                build_quantizer(args, config.num_hidden_layers),
                (),
                example,
                None,
                args.merge_linears,
                args.backward_override,
            )
    step = TrainingStep(
        model,
        compile=args.compile_step or args.gpu_kernels,
        lower=lower_training if args.gpu_kernels else None,
    )

    def loss_of(batch):
        return model(input_ids=batch, labels=batch).loss

    refitter = None
    if args.codebook_refit_steps:
        refitter = CodebookRefitter(
            model,
            weighting=Weighting(args.codebook_weighting),
            refit_gradients=args.codebook_refit_gradients,
        )

    names, params = zip(
        *((n, p) for n, p in model.named_parameters() if p.requires_grad)
    )
    masters = [p.detach().to(torch.float32, copy=True) for p in params]
    for master in masters:
        master.grad = torch.zeros_like(master)
    # Weight decay applies to matrices, not to norm gains.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [m for m in masters if m.dim() >= 2],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [m for m in masters if m.dim() < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    if args.lr_scheduler_type == "warmup_stable_decay":
        schedule_kwargs = {
            "num_decay_steps": round(args.decay_ratio * args.max_train_steps),
            "decay_type": "linear",
            "min_lr_ratio": args.min_lr_ratio,
        }
    else:
        schedule_kwargs = {"min_lr_rate": args.min_lr_ratio}
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        scheduler_specific_kwargs=schedule_kwargs,
    )

    # What the GEMMs are quantized to: a checkpoint saved under the same
    # settings resumes exactly, under others it starts a branch.
    quantization = {
        key: getattr(args, key)
        for key in (
            "activation",
            "weight",
            "error",
            "bias",
            "random_hadamard_transform",
            "quantize_lm_head",
            "quantize_attention",
            "bf16_layers_at_start",
            "bf16_layers_at_end",
        )
    }
    completed_steps = 0
    if args.resume_from_checkpoint is not None:
        state = torch.load(
            args.resume_from_checkpoint, map_location=device, weights_only=False
        )
        if set(state["masters"]) != set(names):
            raise ValueError(
                f"{args.resume_from_checkpoint} holds other parameters"
            )
        for name, master in zip(names, masters):
            master.copy_(state["masters"][name])
        if state["quantization"] == quantization:
            model.load_state_dict(state["model"])
            # The tables are not in the state dict.
            if refitter is not None:
                for name, table in state["codebooks"].items():
                    refitter.modules[name].install_qmap(table)
        else:
            with torch.no_grad():
                for param, master in zip(params, masters):
                    param.copy_(master)
            logger.info(
                f"Branching from a checkpoint quantized as "
                f"{state['quantization']}"
            )
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        train_stream.shard, train_stream.offset = state["stream"]
        completed_steps = state["step"]
        # Stochastic rounding draws from the generators, so a resumed run
        # carries on with the same draws.
        torch.set_rng_state(state["rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        logger.info(f"Resumed from {args.resume_from_checkpoint}")

    tokens_per_step = math.prod(shape) * args.gradient_accumulation_steps
    logger.info(f"Tokens per step: {tokens_per_step}")
    model.train()
    total_loss = 0.0
    start = time.time()
    while completed_steps < args.max_train_steps:
        # A refit window hooks the fake-quants, which only an eager step runs.
        if refitter is not None:
            every = args.codebook_refit_steps
            collect = args.codebook_collect_steps
            window = refit_window(refitter, completed_steps, every, collect)
            eager = in_refit_window(completed_steps, every, collect)
        else:
            window, eager = contextlib.nullcontext(), False
        # The weights change only between optimizer steps.
        with window, step.fixed_weights():
            for _ in range(args.gradient_accumulation_steps):
                batch = train_stream.next_batch().to(device)
                loss, grads = step(input_ids=batch, labels=batch, eager=eager)
                total_loss += loss.float()
                for master, grad in zip(masters, grads):
                    master.grad += grad / args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(masters, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=False)
        with torch.no_grad():
            for param, master in zip(params, masters):
                param.copy_(master)
        completed_steps += 1
        if (
            refitter is not None
            and completed_steps % args.codebook_refit_steps == 0
        ):
            fitted = refitter.refit()
            logger.info(
                f"refit {len(fitted)} codebooks at step {completed_steps}"
            )

        metrics = {}
        if completed_steps % args.logging_steps == 0:
            batches = args.logging_steps * args.gradient_accumulation_steps
            metrics["train_loss"] = total_loss.item() / batches
            metrics["lr"] = lr_scheduler.get_last_lr()[0]
            metrics["grad_norm"] = grad_norm.item()
            metrics["tokens_per_second"] = (
                args.logging_steps * tokens_per_step / (time.time() - start)
            )
            total_loss = 0.0
            start = time.time()
        last = completed_steps == args.max_train_steps
        if completed_steps % args.eval_steps == 0 or last:
            val_stream = TokenStream(files[:1], meta["dtype"], *shape)
            val_loss = evaluate(
                model, loss_of, val_stream, args.eval_batches, device
            )
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = math.exp(val_loss)
        if metrics:
            logger.info(f"step {completed_steps}: {metrics}")
            if wandb.run is not None:
                wandb.log(metrics, step=completed_steps)

        checkpoint = (
            args.checkpointing_steps is not None
            and completed_steps % args.checkpointing_steps == 0
        )
        if checkpoint or last:
            path = os.path.join(args.output_dir, f"step_{completed_steps}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "masters": dict(zip(names, masters)),
                    "quantization": quantization,
                    "codebooks": (
                        {
                            name: refitter.modules[name].qmap
                            for name in refitter.tables
                        }
                        if refitter is not None
                        else None
                    ),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "stream": (train_stream.shard, train_stream.offset),
                    "step": completed_steps,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None
                    ),
                },
                path,
            )
            logger.info(f"Saved {path}")


if __name__ == "__main__":
    main(parse_args())
