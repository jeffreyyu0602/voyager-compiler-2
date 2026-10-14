"""Real NVFP4 GEMMs for the linears of a converted graph.

``convert_pt2e`` leaves an NVFP4 linear -- ``fp4_e2m1`` elements in blocks of
16 along K, each block with an ``fp8_e4m3`` scale, under one global scale per
tensor -- as::

    t = clamp_min(amax(abs(x)) / (6 * 448), tiny)
    s, q = quantize_mx(x, ..., global_scale=t)
    linear_mx(q, W_q, b, input_scale=s, weight_scale=S_w, alpha=t_w * t)

with the codes and block scales held as bfloat16 values on their grids, and
``W_q``, ``S_w`` and ``t_w`` constants.  ``quantize_nvfp4`` repeats the
fake-quant's bfloat16 arithmetic, so it picks the same codes, and packs them
as torch's block-scaled GEMM reads them; ``nvfp4_linear`` is that GEMM.

The GEMM reads one scale per 16 elements of a row.  A weight quantized in
16x16 blocks keeps its codes; ``pack_nvfp4`` gives each of a block's 16 rows
the block's scale.  ``quantize_nvfp4_square`` casts a weight in 16x16 blocks
itself, packed for its GEMM and, transposed, for its input gradient's.
"""

import operator
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.library import Library, impl
from torch.nn.functional import ScalingType, SwizzleType
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    pack_uint4,
    unpack_uint4,
)
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.ops import expand
from voyager_compiler.ops.quantized import _random_bytes
from voyager_compiler.quantization.fake_quantize import (
    _rotate,
    calculate_global_scale,
    hadamard_matrix,
)
from voyager_compiler.shape_prop import fetch_attr

from .fp8 import GEMM_ALIGNMENT

QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
LINEAR_MX = torch.ops.quantized_ops.linear_mx.default

# NVFP4's block length and the largest fp4_e2m1 and fp8_e4m3 values.
BLOCK_SIZE = 16
FP4_MAX = 6.0
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# The ``meta["dtype"]`` of an NVFP4 ``quantize_mx``: its scales', its codes'.
NVFP4_DTYPES = ("fp8_e4m3", "fp4_e2m1")

# The compute capability majors torch runs block-scaled GEMMs on, and the
# CUDA version they need.
BLOCK_SCALED_MAJORS = (10, 12)
BLOCK_SCALED_CUDA = (12, 8)

# The buffer of a lowered graph holding the 1 its ``nvfp4_linear`` calls
# take as ``one``.
ONE = "nvfp4_one"


@dataclass
class Site:
    """One NVFP4 linear of a converted graph, in ``fp8.Site``'s terms.

    Attributes:
        gemm: The ``linear_mx``.
        output: ``gemm`` as well: it applies every scale itself.
        quantizes: The ``quantize_mx`` of its activation, alone.  Other
            sites may share it.
        kernel: The real GEMM, ``nvfp4_linear``.
        weight: The weight's codes, ``[N, K]`` (``get_attr``).
        weight_scale: Their block scales (``get_attr``): ``[N, K / 16]``, or
            ``[N / 16, K / 16]`` for 16x16 blocks.
        alpha: The product of the weight's and the activation's global
            scales.
        bias: The bias (``get_attr``), or None.
    """

    gemm: Node
    output: Node
    quantizes: Tuple[Node]
    kernel: Callable
    weight: Node
    weight_scale: Node
    alpha: Node
    bias: Optional[Node]


def match_site(model: GraphModule, linear: Node) -> Union[Site, str]:
    """The NVFP4 site ``linear`` heads, or why it is not one.

    Args:
        model: The converted graph.
        linear: A ``linear_mx`` of it.

    Returns:
        The ``Site``, or a short reason it cannot run as a real NVFP4 GEMM.
    """
    codes, weight = linear.args[:2]
    quantize = codes.args[0] if codes.target is operator.getitem else codes
    if quantize.target is not QUANTIZE_MX:
        return "input is not quantized"
    if quantize.meta.get("dtype") != NVFP4_DTYPES:
        return f"input is {quantize.meta.get('dtype')}, not NVFP4"
    if quantize.args[2:5] != ((-1,), BLOCK_SIZE, FP4_MAX):
        return "input is not in NVFP4's blocks of 16 along K"
    if get_arg_value(quantize, 9, "global_scale") is None:
        return "input has no global scale"
    value = quantize.args[0].meta.get("val")
    if value is None or value.dtype is not torch.bfloat16:
        return "input is not bfloat16"
    weight_scale = linear.kwargs["weight_scale"]
    if (
        weight.op != "get_attr"
        or weight.meta.get("dtype") != NVFP4_DTYPES[1]
        or weight_scale.meta.get("dtype") != NVFP4_DTYPES[0]
    ):
        return "weight is not a constant in NVFP4"
    n, k = fetch_attr(model, weight.target).shape
    # The GEMM counts K in bytes of two codes.
    if k // 2 % GEMM_ALIGNMENT or n % GEMM_ALIGNMENT:
        return f"weight ({n}, {k}): N is not a multiple of 16 or K of 32"
    bias = get_arg_value(linear, 2, "bias")
    if bias is not None and bias.op != "get_attr":
        return "bias is not a constant"
    return Site(
        linear,
        linear,
        (quantize,),
        NVFP4_LINEAR,
        weight,
        weight_scale,
        linear.kwargs["alpha"],
        bias,
    )


def _block_scales(
    amax: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Each block's scale from its amax, as ``quantize_mx`` computes it: the
    amax over 6 and over the global scale, rounded to ``fp8_e4m3`` in the
    amax's dtype, and 1 where that is 0."""
    scale = amax / FP4_MAX / global_scale
    scale = scale.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).to(amax.dtype)
    return torch.where(scale > 0, scale, 1.0)


def quantize_nvfp4(
    input: torch.Tensor,
    axis: int = -1,
    seed: Optional[torch.Tensor] = None,
    rotate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``input`` to NVFP4 as Voyager's fake-quant does, packed for
    ``nvfp4_linear``.

    Repeats ``MXFakeQuantize`` and ``quantize_mx`` op for op in bfloat16: the
    random Hadamard transform, the global scale ``t`` from the tensor's amax,
    each block's scale -- its amax over 6 and over ``t``, rounded to
    ``fp8_e4m3``, and 1 where that is 0 -- and each element divided by its
    block scale times ``t`` and rounded to ``fp4_e2m1``.

    Args:
        input: bfloat16 tensor, a multiple of 16 long along ``axis``.
        axis: Axis the blocks run along, the GEMM's K.
        seed: int64 scalar keying stochastic rounding, by the random bits
            ``quantize`` draws for a tensor of ``input``'s shape; None
            rounds to nearest.
        rotate: Apply the random Hadamard transform along ``axis`` first.

    Returns:
        ``(codes, scales, t)``: the codes with ``axis`` moved last,
        ``[..., K / 2]``, and their block scales, as ``pack_nvfp4`` packs
        them over the codes' rows; and ``t``, a 0-d tensor.
    """
    if rotate:
        input = _rotate(input, axis, hadamard_matrix(input.device))
    global_scale = calculate_global_scale(input, FP4_MAX, E4M3_MAX)
    blocks = input.movedim(axis, -1).unflatten(-1, (-1, BLOCK_SIZE))
    scale = _block_scales(blocks.abs().amax(-1), global_scale)
    elements = blocks / (scale * global_scale).unsqueeze(-1)
    if seed is not None:
        random_bytes = _random_bytes(seed, input.shape, input.device)
        random_bytes = random_bytes.movedim(axis, -1).unflatten(
            -1, (-1, BLOCK_SIZE)
        )
        elements = _round_stochastically(elements, random_bytes)
    values = elements.flatten(-2)
    codes, scales = pack_nvfp4(values.flatten(0, -2), scale.flatten(0, -2))
    return codes.view(*values.shape[:-1], -1), scales, global_scale


def quantize_nvfp4_square(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Quantize a weight to NVFP4 in 16x16 blocks as Voyager's fake-quant
    does, packed for its forward GEMM and, transposed, for its dgrad.

    ``quantize_nvfp4``'s arithmetic over 16x16 blocks: the global scale ``t``
    from the weight's amax, each block's scale from its amax, and each
    element divided by its block scale times ``t``.

    Args:
        weight: bfloat16 ``[N, K]``, both multiples of 16.

    Returns:
        ``(codes, scales, transposed_codes, transposed_scales, t)``: the
        codes packed along K, ``[N, K / 2]``, and along N, ``[K, N / 2]``,
        each beside its block scales as ``pack_nvfp4`` packs them; and ``t``,
        a 0-d tensor.
    """
    global_scale = calculate_global_scale(weight, FP4_MAX, E4M3_MAX)
    blocks = weight.unflatten(-1, (-1, BLOCK_SIZE)).unflatten(
        -3, (-1, BLOCK_SIZE)
    )
    scale = _block_scales(blocks.abs().amax((-3, -1)), global_scale)
    element_scale = (scale * global_scale)[:, None, :, None]
    values = (blocks / element_scale).reshape(weight.shape)
    return (
        *pack_nvfp4(values, scale),
        *pack_nvfp4(values.mT, scale.mT),
        global_scale,
    )


def _round_stochastically(
    elements: torch.Tensor, random_bytes: torch.Tensor
) -> torch.Tensor:
    """Round each element to one of the two ``fp4_e2m1`` values around it by
    the rule and the float32 arithmetic of ``quantize``'s stochastic rounding.

    The two values come from the grid's spacing -- 0.5 below 2, 1 below 4 and
    2 up to 6 -- rather than from a search of its lookup table.

    Args:
        elements: Values already divided by their scales.
        random_bytes: Eight random bits per element.

    Returns:
        float32 values on the ``fp4_e2m1`` grid.
    """
    value = elements.float().clamp(-FP4_MAX, FP4_MAX)
    magnitude = value.abs()
    gap = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    near = torch.floor(magnitude / gap) * gap
    far = torch.where(near == magnitude, near, near + gap)
    away = magnitude + random_bytes / 256 * (far - near) >= far
    return torch.copysign(torch.where(away, far, near), value)


def pack_nvfp4(
    values: torch.Tensor, scales: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 elements and their block scales in the layout ``nvfp4_linear``
    reads.

    Args:
        values: ``[R, K]`` elements, each rounded to the nearest ``fp4_e2m1``
            value (Voyager's codes are on that grid already).
        scales: Their block scales, values on the ``fp8_e4m3`` grid:
            ``[R, K / 16]``, one per 16 elements of a row, or
            ``[R / 16, K / 16]`` for 16x16 blocks, whose rows each read
            their block's.

    Returns:
        ``(codes, scales)``: ``[R, K / 2]`` uint8, two codes per byte along
        K, the even element's in the low nibble; and the scales' fp8_e4m3
        bytes as cuBLAS reads block scales, in 128x4 tiles over ``R`` rounded
        up to 128 rows and ``K / 16`` up to 4 columns, zero-padded.
    """
    rows, length = values.shape
    scales = expand(scales, (rows, length // BLOCK_SIZE), BLOCK_SIZE)
    codes = pack_uint4(f32_to_f4_unpacked(values.float()))
    scales = to_blocked(scales.to(torch.float8_e4m3fn).view(torch.uint8))
    return codes, scales


def unpack_nvfp4(
    codes: torch.Tensor, scales: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The elements and block scales ``pack_nvfp4`` packed, in bfloat16.

    Args:
        codes: ``[..., K / 2]`` packed codes.
        scales: Their packed block scales.

    Returns:
        ``(values, block_scales)``: ``[..., K]`` and ``[..., K / 16]``.
    """
    values = f4_unpacked_to_f32(unpack_uint4(codes)).to(torch.bfloat16)
    *rows, length = values.shape
    blocks = from_blocked(
        scales, values.numel() // length, length // BLOCK_SIZE
    )
    block_scales = blocks.view(torch.float8_e4m3fn).reshape(*rows, -1)
    return values, block_scales.to(torch.bfloat16)


_LIBRARY = Library("voyager_gpu", "DEF")
_LIBRARY.define(
    "nvfp4_linear(Tensor input, Tensor input_scale, Tensor weight, "
    "Tensor weight_scale, Tensor alpha, Tensor one, Tensor? bias) -> Tensor"
)


@impl(_LIBRARY, "nvfp4_linear", "CUDA")
def nvfp4_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    one: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``input @ weight.T * alpha + bias`` in bfloat16, on packed NVFP4
    operands, as one ``F.scaled_mm`` with block scales per 16 elements and
    ``alpha`` as the second-level (``TensorWise``) scale, applied to the
    float32 product inside the GEMM.

    The scalars are tensors the caller holds, so a call creates none.

    Args:
        input: ``[..., K / 2]`` codes, as ``quantize_nvfp4`` packs them.
        input_scale: Their packed block scales.
        weight: ``[N, K / 2]`` codes, as ``pack_nvfp4`` packs them.
        weight_scale: Their packed block scales.
        alpha: float32 scalar holding the bfloat16 product of the two global
            scales.
        one: float32 scalar 1, the weight's second-level scale, since
            ``alpha`` holds both global scales.
        bias: ``[N]`` in bfloat16, or None.

    Returns:
        ``[..., N]`` in bfloat16.
    """
    a = input.reshape(-1, input.shape[-1]).view(torch.float4_e2m1fn_x2)
    # The GEMM reads its second operand column-major: [K / 2, N] over
    # [N, K / 2] in memory.
    b = weight.view(torch.float4_e2m1fn_x2).t()
    scale_a = input_scale.view(torch.float8_e4m3fn)
    scale_b = weight_scale.view(torch.float8_e4m3fn)
    blockwise, swizzled = ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4
    tensorwise, unswizzled = ScalingType.TensorWise, SwizzleType.NO_SWIZZLE
    out = F.scaled_mm(
        a,
        b,
        [scale_a, alpha],
        [blockwise, tensorwise],
        [scale_b, one],
        [blockwise, tensorwise],
        swizzle_a=[swizzled, unswizzled],
        swizzle_b=[swizzled, unswizzled],
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    return out.reshape(*input.shape[:-1], -1)


@torch.library.register_fake("voyager_gpu::nvfp4_linear")
def _(input, input_scale, weight, weight_scale, alpha, one, bias):
    shape = (*input.shape[:-1], weight.shape[0])
    return input.new_empty(shape, dtype=torch.bfloat16)


NVFP4_LINEAR = torch.ops.voyager_gpu.nvfp4_linear.default


def insert_gemm_scalars(
    model: GraphModule, alpha: Node, device: torch.device
) -> Tuple[Node, Node]:
    """Insert, at the graph's insertion point, the scalars an
    ``nvfp4_linear`` of ``model`` takes: ``alpha`` copied to float32 in the
    graph, where Inductor can fuse the copy into the op computing it, and a
    ``get_attr`` of the 1 all its GEMMs share, registered on first use.

    Args:
        model: The graph being lowered.
        alpha: The GEMM's bfloat16 product of global scales.
        device: The device ``model`` runs on.

    Returns:
        ``(alpha, one)``, as ``nvfp4_linear`` takes them.
    """
    if not hasattr(model, ONE):
        model.register_buffer(ONE, torch.ones((), device=device))
    graph = model.graph
    return (
        graph.call_function(
            torch.ops.aten._to_copy.default, (alpha,), {"dtype": torch.float32}
        ),
        graph.get_attr(ONE),
    )


def emulate_nvfp4_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    one: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``nvfp4_linear`` on any device, as the reference: the operands
    unpacked to bfloat16 and multiplied by ``linear_mx``, as the converted
    graph multiplies them.

    Args:
        input: As ``nvfp4_linear`` takes it.
        input_scale: As ``nvfp4_linear`` takes it.
        weight: As ``nvfp4_linear`` takes it.
        weight_scale: As ``nvfp4_linear`` takes it.
        alpha: As ``nvfp4_linear`` takes it.
        one: Taken for ``nvfp4_linear``'s signature only.
        bias: As ``nvfp4_linear`` takes it.

    Returns:
        ``[..., N]`` in bfloat16.
    """
    input, input_scale = unpack_nvfp4(input, input_scale)
    weight, weight_scale = unpack_nvfp4(weight, weight_scale)
    return torch.ops.quantized_ops.linear_mx(
        input,
        weight,
        bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        block_size=BLOCK_SIZE,
        alpha=alpha,
    )
