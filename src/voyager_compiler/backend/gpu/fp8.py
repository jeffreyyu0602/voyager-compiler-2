"""Real FP8 GEMMs for the linears and matmuls of a converted graph.

``convert_pt2e`` leaves an FP8 GEMM as ``quantize`` on each activation
operand, then the ``linear`` or ``matmul``, then ``dequantize(., s)`` when
the scheme has scales.  The codes are values on the FP8 grid held in
bfloat16, a linear's weight codes are a constant, and ``s`` is the product
of the operand scales, one per tensor or one per output channel; a direct
cast (``fp8_e4m3`` with no scheme) has unit scales and no dequantize.
Voyager's ``fp8_e4m3`` and ``fp8_e5m2`` grids are torch's ``float8_e4m3fn``
and ``float8_e5m2``, so the codes cast to real FP8 exactly.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.shape_prop import fetch_attr

QUANTIZE = torch.ops.quantized_ops.quantize.default
DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
LINEAR = torch.ops.aten.linear.default
MATMUL = torch.ops.aten.matmul.default

# Voyager's FP8 element formats and the torch dtypes on the same grid.
FP8_DTYPES = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

# torch._scaled_mm and torch._scaled_grouped_mm take K and N in multiples
# of 16.
GEMM_ALIGNMENT = 16
# The compute capability majors torch runs torch._scaled_grouped_mm on.
BATCHED_GEMM_MAJORS = (9, 10)


@dataclass
class Site:
    """One FP8 GEMM of a converted graph: a linear, or a matmul of two
    activations.

    Attributes:
        gemm: The ``aten.linear`` or ``aten.matmul``.
        output: The ``dequantize`` after ``gemm``, or ``gemm`` itself when
            the scheme has no scales.
        quantizes: The ``quantize`` of each activation operand, in operand
            order: one for a linear, two for a matmul.  Other sites may
            share them.
        dtypes: The torch FP8 dtype of each of ``quantizes``.
        kernel: The real GEMM, ``fp8_linear`` or ``fp8_bmm``, called on the
            FP8 activations and then the values of ``site_operands``.
        weight: A linear's weight codes, ``[N, K]`` (``get_attr``); None for
            a matmul.
        weight_dtype: Their torch FP8 dtype; None for a matmul.
        bias: A linear's bias (``get_attr``), or None.
    """

    gemm: Node
    output: Node
    quantizes: Tuple[Node, ...]
    dtypes: Tuple[torch.dtype, ...]
    kernel: Callable
    weight: Optional[Node] = None
    weight_dtype: Optional[torch.dtype] = None
    bias: Optional[Node] = None


def _set_args(node: Node, *names: Tuple[int, str]) -> bool:
    """Whether any of ``node``'s arguments at ``(index, name)`` is set."""
    return any(get_arg_value(node, i, name) is not None for i, name in names)


def _fp8_input(model: GraphModule, node: Node) -> Union[torch.dtype, str]:
    """The FP8 dtype ``node`` quantizes to, or why it is not a bfloat16
    activation quantized to FP8 against one scale."""
    if node.target is not QUANTIZE:
        return "input is not quantized"
    dtype = FP8_DTYPES.get(node.meta.get("dtype"))
    if dtype is None:
        return f"input is {node.meta.get('dtype')}, not FP8"
    if _set_args(
        node,
        (2, "zero_point"),
        (4, "block_size"),
        (6, "output_code"),
        (7, "seed"),
    ):
        return "input quantize is not a plain scale"
    scale = node.args[1]
    if scale.op != "get_attr":
        return "input scale is not a constant"
    scale = fetch_attr(model, scale.target)
    if scale.numel() != 1:
        return "input scale is not per tensor"
    if scale.dtype is not torch.bfloat16:
        return f"input is {scale.dtype}, not bfloat16"
    return dtype


def _scaled_output(
    model: GraphModule, gemm: Node, channels: int
) -> Union[Node, str]:
    """The node holding ``gemm``'s scaled result -- its dequantize, or
    ``gemm`` itself when it has none -- or why it is neither.

    Args:
        model: The converted graph.
        gemm: The linear or matmul.
        channels: How many scales a per-output-channel dequantize holds; 1
            where only one per tensor can be.
    """
    users = list(gemm.users)
    if not any(u.target is DEQUANTIZE for u in users):
        return gemm
    if len(users) != 1:
        return "output dequantize is not its only user"
    output = users[0]
    if _set_args(
        output, (2, "zero_point"), (4, "block_size"), (6, "output_qmap")
    ):
        return "output dequantize is not a plain scale"
    scale = output.args[1]
    if scale.op != "get_attr":
        return "output scale is not a constant"
    if fetch_attr(model, scale.target).numel() not in (1, channels):
        return "output scale is not per tensor or per output channel"
    return output


def _match_linear(model: GraphModule, linear: Node) -> Union[Site, str]:
    """``match_site`` for an ``aten.linear``."""
    x, weight = linear.args[0], linear.args[1]
    bias = get_arg_value(linear, 2, "bias")
    activation_dtype = _fp8_input(model, x)
    if isinstance(activation_dtype, str):
        return activation_dtype
    if weight.op != "get_attr":
        return "weight is not a constant"
    weight_dtype = FP8_DTYPES.get(weight.meta.get("dtype"))
    if weight_dtype is None:
        return f"weight is {weight.meta.get('dtype')}, not FP8"
    if activation_dtype is weight_dtype is torch.float8_e5m2:
        return "E5M2 x E5M2 has no FP8 GEMM"
    codes = fetch_attr(model, weight.target)
    if codes.dim() != 2 or any(d % GEMM_ALIGNMENT for d in codes.shape):
        return f"weight {tuple(codes.shape)} is not [N, K] in multiples of 16"
    if bias is not None and bias.op != "get_attr":
        return "bias is not a constant"
    output = _scaled_output(model, linear, codes.shape[0])
    if isinstance(output, str):
        return output
    return Site(
        linear,
        output,
        (x,),
        (activation_dtype,),
        fp8_linear,
        weight,
        weight_dtype,
        bias,
    )


def _match_matmul(model: GraphModule, matmul: Node) -> Union[Site, str]:
    """``match_site`` for an ``aten.matmul``."""
    quantizes = tuple(matmul.args[:2])
    dtypes = []
    for quantize in quantizes:
        dtype = _fp8_input(model, quantize)
        if isinstance(dtype, str):
            return dtype
        dtypes.append(dtype)
    if any(dtype is not torch.float8_e4m3fn for dtype in dtypes):
        return "the batched FP8 GEMM takes E4M3 only"
    values = [quantize.args[0].meta.get("val") for quantize in quantizes]
    if any(value is None for value in values):
        return "operand shapes are unknown"
    a, b = (value.shape for value in values)
    if len(a) < 2 or len(b) < 2 or a[:-2] != b[:-2]:
        return "operands do not share their batch dims"
    if a[-1] % GEMM_ALIGNMENT or b[-1] % GEMM_ALIGNMENT:
        return f"{tuple(a)} @ {tuple(b)}: K or N is not a multiple of 16"
    output = _scaled_output(model, matmul, 1)
    if isinstance(output, str):
        return output
    return Site(matmul, output, quantizes, tuple(dtypes), fp8_bmm)


def match_site(model: GraphModule, gemm: Node) -> Union[Site, str]:
    """The FP8 site ``gemm`` heads, or why it is not one.

    Args:
        model: The converted graph.
        gemm: A GEMM node of it.

    Returns:
        The ``Site``, or a short reason it cannot run as a real FP8 GEMM.
    """
    if gemm.target is LINEAR:
        return _match_linear(model, gemm)
    if gemm.target is MATMUL:
        return _match_matmul(model, gemm)
    return f"{gemm.target} is not lowered"


def site_operands(
    model: GraphModule, site: Site
) -> Dict[str, Optional[torch.Tensor]]:
    """The constants ``site.kernel`` takes after the FP8 activations.

    Args:
        model: The converted graph.
        site: One of its FP8 sites.

    Returns:
        By name, in argument order: a linear's ``weight`` (its codes as FP8,
        ``[N, K]``), ``scale`` and ``bias`` (in the output's units, or
        None); a matmul's ``scale``.  The scale is the one the GEMM applies,
        a float32 scalar or ``[1, N]`` per output channel.

    Raises:
        ValueError: A linear's weight holds a value off its FP8 grid.
    """
    input_scale = fetch_attr(model, site.quantizes[0].args[1].target)
    scale = torch.ones((), device=input_scale.device)
    if site.output is not site.gemm:
        scale = fetch_attr(model, site.output.args[1].target).float()
        scale = scale.reshape(()) if scale.numel() == 1 else scale.view(1, -1)
    if site.weight is None:
        return {"scale": scale}

    codes = fetch_attr(model, site.weight.target)
    weight = codes.to(site.weight_dtype)
    if not torch.equal(weight.to(codes.dtype), codes):
        raise ValueError(f"{site.weight.name}: values off the FP8 grid")
    bias = None
    if site.bias is not None:
        # The graph holds the bias quantized against the output's scale.
        bias = fetch_attr(model, site.bias.target).float() * scale.view(-1)
        bias = bias.to(codes.dtype)
    return {"weight": weight, "scale": scale, "bias": bias}


def quantize_fp8(
    input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """``input / scale`` rounded to the nearest FP8 value of ``dtype``,
    saturating at its largest, as Voyager's ``quantize`` rounds it."""
    limit = torch.finfo(dtype).max
    return (input / scale).clamp(-limit, limit).to(dtype)


def fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``input @ weight.T * scale + bias`` in bfloat16, as one
    ``torch._scaled_mm``.

    Args:
        input: FP8 activation, ``[..., K]``.
        weight: FP8 weight, ``[N, K]``.
        scale: float32 scalar, or ``[1, N]`` per output channel.
        bias: ``[N]`` in bfloat16, or None.

    Returns:
        ``[..., N]``.
    """
    a = input.reshape(-1, input.shape[-1]).contiguous()
    if scale.dim() == 0:
        scale_a, scale_b = scale, scale.new_ones(())
    else:
        # Per output channel: the row-wise mode wants a scale per activation
        # row as well.
        scale_a, scale_b = scale.new_ones(a.shape[0], 1), scale
    out = torch._scaled_mm(
        a, weight.t(), scale_a, scale_b, bias=bias, out_dtype=torch.bfloat16
    )
    return out.reshape(*input.shape[:-1], out.shape[-1])


def fp8_bmm(
    input: torch.Tensor, other: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``input @ other * scale`` in bfloat16 over shared batch dims, as one
    ``torch._scaled_grouped_mm``.

    Args:
        input: FP8 ``[..., M, K]``.
        other: FP8 ``[..., K, N]``, with ``input``'s batch dims.
        scale: float32 scalar.

    Returns:
        ``[..., M, N]``.
    """
    *batch, m, k = input.shape
    n = other.shape[-1]
    a = input.reshape(-1, m, k).contiguous()
    # The GEMM reads its second operand column-major: [G, N, K] in memory.
    b = other.reshape(-1, k, n).transpose(-2, -1).contiguous()
    groups = a.shape[0]
    # The grouped GEMM takes one scale per row of each operand.
    scale_a = scale.expand(groups, m).contiguous()
    scale_b = scale.new_ones(groups, n)
    out = torch._scaled_grouped_mm(
        a, b.transpose(-2, -1), scale_a, scale_b, out_dtype=torch.bfloat16
    )
    return out.reshape(*batch, m, n)
