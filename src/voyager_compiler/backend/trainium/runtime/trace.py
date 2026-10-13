"""Trace a translation's program, kernels and host ops, into one NEFF.

Runs in the ``trn`` env, by path, on a CPU-only machine:

    python trace.py <dir> --phase {prefill,decode} [--target trn2]

Writes ``<dir>/step/``: ``step.pt``, the traced module a NeuronCore runs
(``run.py``), and its NEFF.  The kernels become NKI custom calls and the
host ops XLA ops of the same graph, so one launch runs the whole step, as
NxD Inference runs a prefill or a decode step, with its compiler flags and
decoder-layer markers.  Weights are the module's parameters, kept out of the
NEFF so they load once; the state the step writes stays on the device.
"""

import argparse
import importlib.util
import json
import os

import torch
import torch_neuronx
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

MARKERS = {"start": ModuleMarkerStartWrapper(), "end": ModuleMarkerEndWrapper()}

# The torch type a kernel takes a tensor of each IR type as (``check.py``).
KERNEL_TYPES = {
    "int64": torch.int32,
    "bool": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e4m3_ieee": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

# NxD Inference's compiler flags (models/model_wrapper.py, with its layer
# boundary markers).  A prefill step compiles in the modular flow, which
# splits the graph at the markers and compiles one decoder layer for all of
# them; a decode step compiles at -O2.  A step computing in Trn2's IEEE E4M3
# has its HLO's float8_e4m3fn compiled as that format.
COMPILER_FLAGS = (
    "--auto-cast=none --model-type=transformer --tensorizer-options="
    "'--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 "
    "--vectorize-strided-dma' --lnc=1 "
)
PHASE_FLAGS = {"prefill": "-O1", "decode": "-O2"}
HLO_FLAGS = {"prefill": "--modular-flow-mac-threshold=10 ", "decode": ""}
FP8_HLO_FLAG = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 "


def load_module(directory, name):
    """``<directory>/<name>.py`` as a module."""
    path = os.path.join(directory, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step(torch.nn.Module):
    """One run of the program: its inputs in, its outputs out.

    Weights are parameters, which the trace keeps out of the graph; the
    state the step writes (a KV cache) is buffers, which aliasing keeps on
    the device and updates in place.
    """

    def __init__(self, program, kernels, params, state, names, scalars):
        super().__init__()
        self.program = program
        self.kernels = kernels
        self.param_names = list(params)
        for i, (name, tensor) in enumerate(params.items()):
            if name in state:
                self.register_buffer(f"param{i}", tensor)
            else:
                parameter = torch.nn.Parameter(tensor, requires_grad=False)
                self.register_parameter(f"param{i}", parameter)
        self.state = state
        self.input_names = names
        self.scalars = scalars

    def forward(self, *tensors):
        params = {
            name: getattr(self, f"param{i}")
            for i, name in enumerate(self.param_names)
        }
        inputs = dict(zip(self.input_names, tensors))
        outputs = self.program.run(
            inputs, params, self.call_kernel, self.mark, self.scalars
        )
        # Reassigning the buffers is what aliasing tracks.
        for i, name in enumerate(self.param_names):
            if name in self.state:
                setattr(self, f"param{i}", params[name])
        return outputs

    def call_kernel(self, name, *args):
        """Kernel ``name``'s results, as a tuple even when there is one."""
        results = getattr(self.kernels, name)(*args)
        if isinstance(results, (list, tuple)):
            return tuple(results)
        return (results,)

    @staticmethod
    def mark(kind, *values):
        marked = MARKERS[kind](*values)
        return (marked,) if isinstance(marked, torch.Tensor) else tuple(marked)


def main(directory, phase, target):
    with open(os.path.join(directory, "manifest.json")) as f:
        manifest = json.load(f)
    tensors = torch.load(os.path.join(directory, "tensors.pt"))
    # Weights in the type the kernels take them, once, not every step.
    dtypes = manifest["dtypes"]
    params = {
        name: tensors[name].to(
            KERNEL_TYPES.get(dtypes[name], tensors[name].dtype)
        )
        for name in manifest["params"]
    }
    inputs = [tensors[name] for name in manifest["inputs"]]
    step = Step(
        load_module(directory, "program"),
        load_module(directory, "kernels"),
        params,
        set(manifest["state"]),
        manifest["inputs"],
        # The host scalars of the example run: the trace is specialized to
        # them (a decode step to its position).
        {name: tensors[name].item() for name in manifest["scalars"]},
    )
    step_dir = os.path.join(directory, "step")
    os.makedirs(step_dir, exist_ok=True)
    hlo_flags = HLO_FLAGS[phase]
    if "fp8_e4m3_ieee" in dtypes.values():
        # torch carries the IEEE E4M3 as float8_e4m3fn, the same bits up to
        # its 240: NKI's non-OCP mode, and NxD Inference's flag.
        os.environ["NKI_FP8_E4M3_MODE"] = "non_ocp"
        hlo_flags += FP8_HLO_FLAG
    hlo_flags += "--verify-hlo=true --recursive-layer-det=false"
    compiler_args = (
        f"--target={target} {COMPILER_FLAGS}{PHASE_FLAGS[phase]} "
        f"--internal-hlo2tensorizer-options='{hlo_flags}'"
    )
    traced = torch_neuronx.trace(
        step,
        tuple(inputs),
        compiler_workdir=step_dir,
        inline_weights_to_neff=False,
        enable_aliasing=True,
        compiler_args=compiler_args,
    )
    torch.jit.save(traced, os.path.join(step_dir, "step.pt"))
    print(f"traced -> {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--phase", choices=tuple(PHASE_FLAGS), required=True)
    parser.add_argument("--target", default="trn2")
    args = parser.parse_args()
    main(args.directory, args.phase, args.target)
