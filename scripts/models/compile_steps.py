import os
from contextlib import contextmanager

import torch
from voyager_compiler import (
    QuantizedModel,
    capture_training,
    compile,
    gradient_program,
    prepare_training,
    shared_dram_layout,
    transform,
    update_program,
)
from voyager_compiler.hardware_config import AcceleratorConfig
from voyager_compiler.shape_prop import ShapeProp


def get_transform_args(args, fusion_patterns):
    return {
        "patterns": fusion_patterns,
        "config": AcceleratorConfig.from_args(args),
        "tensor_layout": args.tensor_layout,
        "gemm_layout": args.gemm_layout,
        "gemv_layout": args.gemv_layout,
        "weight_layout": args.weight_layout,
        "fuse_reshape": not args.disable_reshape_fusion,
        "keep_fp32": not args.remove_fp32_casts,
    }


def get_compile_args(args):
    return {
        "config": AcceleratorConfig.from_args(args),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "runtime_tolerance": args.runtime_tolerance,
        "accumulate_fp32": args.accumulate_fp32,
        "bool_mask": args.bool_mask,
    }


@contextmanager
def single_cpu_thread():
    """Run the enclosed code on one CPU thread, then restore the old count."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


class Loss(torch.nn.Module):
    """A Hugging Face model whose forward returns its loss alone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs).loss


def compile_training_step(model, inputs, fusion_patterns, args, quantizer=None):
    """Compile one AdamW step of training ``model`` on the batch ``inputs``.

    The step is two programs sharing one DRAM layout: the gradient program
    -- forward, loss and backward -- and the update program, each compiled
    into its own directory under ``args.model_output_dir``.  They run on
    ``model``'s own tensors and ``inputs``, zeroed gradient buffers or, for
    the update, random gradients, and the optimizer's state as before a
    first step.  With quantization specs, the step trains through the
    fake-quants -- the backward too with ``--error``, else as
    ``--backward_override`` says -- after ``--calibration_steps`` eager
    steps have started their delayed scaling.  A dropout draws its mask per
    tile once compiled, so with dropout the compiled program's results
    differ from the reference's.

    Args:
        model: A ``QuantizedModel`` holding the training graphs, or a
            Hugging Face model that returns its loss given labels, captured
            here.
        inputs: One batch, labels included.
        fusion_patterns: The fusion patterns ``transform`` applies.
        args: The command line.
        quantizer: Picks the GEMM operands to quantize when ``model`` is
            captured here; a ``QuantizedModel`` brought its own.

    Returns:
        ``(programs, reference, lowered)``, each keyed ``"gradient"`` and
        ``"update"``: the compiled programs, the tensors each writes and
        returns when its transformed graph runs -- the reference, since
        ``--remove_fp32_casts`` changes the numbers on purpose -- and the
        same from the compiled graphs under ``--debug``, else ``None``.
    """
    specs = (args.activation, args.weight, args.error)
    quantized = any(spec is not None for spec in specs)
    if isinstance(model, QuantizedModel):
        # Its forward returns the Hugging Face output; Loss takes the loss.
        loss = Loss(model.train())
        model = model.model
    else:
        model = loss = Loss(model).train()
        if quantized:
            prepare_training(
                model,
                quantizer,
                (),
                inputs,
                None,
                backward_override=args.backward_override,
            )
        else:
            with torch.enable_grad():
                capture_training(model, (), inputs, None)
    if quantized:
        for _ in range(args.calibration_steps):
            with torch.enable_grad():
                loss(**inputs).backward()
        model.zero_grad(set_to_none=True)
    gradient = gradient_program(model)
    # A torch optimizer skips a parameter the loss never reaches, such as a
    # vision tower under a text-only batch: it steps the ones with gradients.
    grads = {n.name for n in gradient.graph.find_nodes(op="placeholder")}
    optimizer = torch.optim.AdamW(
        p
        for name, p in model.named_parameters()
        if name.replace(".", "_") + "_grad" in grads
    )
    programs = {
        "gradient": gradient,
        "update": update_program(model, optimizer),
    }

    tensors = {
        name.replace(".", "_"): tensor
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
    }
    example = {}
    for kind, program in programs.items():
        batch = iter(inputs.values())
        values = []
        for node in program.graph.find_nodes(op="placeholder"):
            val = node.meta["val"]
            if node.name in tensors:
                values.append(tensors[node.name])
            elif kind == "gradient" and not node.name.endswith("_grad"):
                values.append(next(batch))
            elif kind == "update" and node.name.endswith("_grad"):
                values.append(torch.randn(val.shape).to(val.dtype) * 1e-2)
            elif node.name.startswith("lr_"):
                values.append(torch.tensor(1e-3))
            else:
                values.append(torch.zeros(val.shape, dtype=val.dtype))
        example[kind] = values

    transform_args = get_transform_args(args, fusion_patterns)
    for kind, program in programs.items():
        transform(program, example[kind], **transform_args)
    layout = shared_dram_layout(
        list(programs.values()), transform_args["config"]
    )

    def run():
        outputs = {}
        for kind, program in programs.items():
            written = [v.clone() for v in example[kind]]
            result = ShapeProp(program).propagate(*written)
            # The masks and token ids are only read.
            outputs[kind] = [
                t for t in [*written, *(result or ())] if t.is_floating_point()
            ]
        return outputs

    reference = run()
    compile_args = get_compile_args(args)
    out_dir = compile_args.pop("output_dir")
    for kind, program in programs.items():
        compile(
            program,
            [v.clone() for v in example[kind]],
            output_dir=os.path.join(out_dir, kind),
            dram_layout=layout,
            **compile_args,
        )
    if not args.debug:
        return programs, reference, None
    with single_cpu_thread():
        return programs, reference, run()
