"""Hardware-layout forms of aten operators.

The data-layout passes re-store certain operands in hardware-friendly
layouts: NHWC activations + HWIO conv weights, and GEMM operands
contraction-major.  Each affected pool operator gets a ``quantized_ops``
twin with the SAME name and the SAME schema (extracted verbatim from the
aten ``OpOverload``, so the backend keeps seeing the familiar targets);
the generated eager impl permutes the layout-transformed inputs back to
aten's native layout, calls the original operator, and permutes the
result into the hardware layout.  ``quantized_ops.conv2d`` takes aten's
operands plus a ``layout`` argument, ``"nchw"`` or ``"nhwc"``, and the
GEMMs, ``quantized_ops.linear`` and ``quantized_ops.matmul``, one naming
how each operand is stored (``GemmLayout``).  The op and its arguments
therefore carry the layout contract: eager execution of a transformed
graph is correct with no process-global monkeypatching.

This module is a leaf (it imports only torch): the ops are registered
the moment ``quantized`` imports it, before any module that references
``torch.ops.quantized_ops.conv2d`` etc. at import time can load.
"""

from typing import List, NamedTuple, Optional, Tuple

import torch

NCHW_TO_NHWC = (0, 2, 3, 1)
NHWC_TO_NCHW = (0, 3, 1, 2)
OIHW_TO_HWIO = (2, 3, 1, 0)
HWIO_TO_OIHW = (3, 2, 0, 1)

# The two memory orders of the 4-D feature maps that conv2d, the 2-D pools
# and the ops connected to them work in: ``"nchw"`` (channels first,
# PyTorch's order) and ``"nhwc"`` (channels last).
TENSOR_LAYOUTS = ("nchw", "nhwc")
DEFAULT_TENSOR_LAYOUT = "nchw"

# A GEMM op's ``layout`` argument names how its operands and output are
# stored, in einsum form over their last two dims (batch dims left out),
# ``"<activation>,<weight>-><output>"``, in C (contraction), K (output
# features) and X (tokens, the activation's rows): the activation ``"xc"``
# or ``"cx"``, the weight ``"kc"`` or ``"ck"``, and the output ``"xk"``
# [tokens, out], as aten computes it, or ``"kx"``.  ``"xc,kc->xk"`` is
# PyTorch's, ``"xc,ck->xk"`` stores the weight [contraction, out], and
# ``"cx,ck->xk"`` also stores the activations between GEMMs
# contraction-major, [features, tokens], the order a Trainium TensorE
# contracts over its partitions.  A contraction-major activation needs the
# CK weight.  A matrix-vector GEMM's activation is a single row, the same
# bytes in either order, so it is always ``"xc"``.  The default serves both
# the matrix-matrix and the matrix-vector layout.
GEMM_LAYOUTS = ("xc,kc->xk", "xc,ck->xk", "cx,ck->xk")
GEMV_LAYOUTS = ("xc,kc->xk", "xc,ck->xk")
DEFAULT_GEMM_LAYOUT = "xc,kc->xk"
GEMM_ACTIVATION_LAYOUTS = ("xc", "cx")
GEMM_WEIGHT_LAYOUTS = ("kc", "ck")
DEFAULT_GEMM_OUTPUT = "xk"
GEMM_OUTPUT_LAYOUTS = (DEFAULT_GEMM_OUTPUT, "kx")
# The layouts ``aten.linear`` and ``aten.matmul`` compute in: the default of
# each GEMM op.
LINEAR_LAYOUT = "xc,kc->xk"
MATMUL_LAYOUT = "xc,ck->xk"


class GemmLayout(NamedTuple):
    """The parts of a GEMM op's ``layout`` argument: how its activation,
    weight and output are stored."""

    activation: str
    weight: str
    output: str

    @classmethod
    def parse(cls, layout: str) -> "GemmLayout":
        """The parts of ``layout``, a GEMM op's ``layout`` argument."""
        operands, output = layout.split("->")
        parts = cls(*operands.split(","), output)
        assert parts.activation in GEMM_ACTIVATION_LAYOUTS, layout
        assert parts.weight in GEMM_WEIGHT_LAYOUTS, layout
        assert parts.output in GEMM_OUTPUT_LAYOUTS, layout
        return parts

    def __str__(self) -> str:
        """The ``layout`` argument."""
        return f"{self.activation},{self.weight}->{self.output}"


def project(per_axis: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Reorder a per-logical-axis tuple into a tensor's physical order.

    Tile specs are written in logical axis order — feature maps NCHW, weights
    OIHW — and projected onto an operand's physical order only at the
    load/store boundary.

    Args:
        per_axis: Value per logical axis; ``per_axis[a]`` belongs to axis ``a``.
        dims: NCHW->physical permutation (physical position ``i`` holds logical
            axis ``dims[i]``), or ``None`` for the logical layout itself.

    Returns:
        The values in physical order.
    """
    if dims is None:
        return tuple(per_axis)
    return tuple(per_axis[a] for a in dims)


def unproject(physical: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Read a physical-order sequence back into logical NCHW / OIHW order.

    Inverse of :func:`project`.

    Args:
        physical: Values in the tensor's physical order (e.g. a ``shape``).
        dims: The same permutation :func:`project` takes, or ``None``.

    Returns:
        The values in logical axis order.
    """
    if dims is None:
        return tuple(physical)
    return tuple(physical[dims.index(a)] for a in range(len(physical)))


# The pools consuming NHWC activations (their twin takes and returns
# hardware-layout tensors).
NHWC_LAYOUT_OPS: List[torch._ops.OpOverload] = [
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
]

# The layout ops join the ``quantized_ops`` namespace ``quantized.py`` owns
# (a FRAGMENT extends it; the handle must stay alive for the
# registrations to persist).
_QUANTIZED_OPS_FRAGMENT = torch.library.Library("quantized_ops", "FRAGMENT")


def _define_nhwc_variant(aten_op):
    """Register the NHWC twin of the pool ``aten_op`` and return its
    ``OpOverload``.

    The twin has aten's name and schema; it takes its first (activation)
    argument NHWC and returns NHWC.  The impl is
    ``CompositeExplicitAutograd``, so fake tensors and export trace through
    it with no fake impl.
    """
    schema = aten_op._schema
    name = schema.name.split("::")[1]
    _QUANTIZED_OPS_FRAGMENT.define(str(schema).split("::", 1)[1])

    def nhwc_impl(input, *args):
        out = aten_op(input.permute(NHWC_TO_NCHW), *args)
        return out.permute(NCHW_TO_NHWC)

    torch.library.impl(
        _QUANTIZED_OPS_FRAGMENT, name, "CompositeExplicitAutograd"
    )(nhwc_impl)
    return getattr(torch.ops.quantized_ops, name).default


# ``{aten pool: its NHWC twin}``.
NHWC_OP_VARIANTS = {op: _define_nhwc_variant(op) for op in NHWC_LAYOUT_OPS}


# aten's ``conv2d`` on feature maps stored as ``layout`` says, a
# ``TENSOR_LAYOUTS`` entry: under ``"nhwc"`` the input and result are NHWC
# and a dense conv's weight is HWIO (a grouped conv's stays OIHW).
# ``out_dtype``, when given, is the dtype the convolution is summed in
# float32 and returned in, as for ``linear`` below.
_QUANTIZED_OPS_FRAGMENT.define(
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, "
    "SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], "
    "SymInt[2] dilation=[1, 1], SymInt groups=1, *, "
    f'str layout="{DEFAULT_TENSOR_LAYOUT}", ScalarType? out_dtype=None) '
    "-> Tensor"
)


@torch.library.impl(
    _QUANTIZED_OPS_FRAGMENT, "conv2d", "CompositeExplicitAutograd"
)
def _conv2d(
    input,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    *,
    layout=DEFAULT_TENSOR_LAYOUT,
    out_dtype=None,
):
    assert layout in TENSOR_LAYOUTS, layout
    nhwc = layout == "nhwc"
    if nhwc:
        input = input.permute(NHWC_TO_NCHW)
        if groups == 1:
            weight = weight.permute(HWIO_TO_OIHW)
    if out_dtype is not None:
        input, weight = input.float(), weight.float()
        bias = None if bias is None else bias.float()
    out = torch.ops.aten.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out.permute(NCHW_TO_NHWC) if nhwc else out


# The GEMMs in any layout: aten's ``linear`` and ``matmul`` on operands
# stored as ``layout`` says, their result stored as it says too.
# ``out_dtype``, when given, is the dtype the product is summed in float32
# and returned in, as a GEMM writing an accumulator does; unset, the product
# is the inputs' dtype.
_QUANTIZED_OPS_FRAGMENT.define(
    "linear(Tensor input, Tensor weight, Tensor? bias=None, *, "
    f'str layout="{LINEAR_LAYOUT}", ScalarType? out_dtype=None) -> Tensor'
)
_QUANTIZED_OPS_FRAGMENT.define(
    f'matmul(Tensor self, Tensor other, *, str layout="{MATMUL_LAYOUT}", '
    "ScalarType? out_dtype=None) -> Tensor"
)


def _gemm_in_layout(
    aten_op, native, activation, weight, layout, out_dtype, *rest
):
    """``aten_op`` on operands stored as ``layout`` says.

    Args:
        aten_op: ``aten.linear`` or ``aten.matmul``.
        native: The layout ``aten_op`` computes in.
        activation: The activation, stored as ``layout`` says.
        weight: The weight, stored as ``layout`` says.
        layout: The GEMM's ``layout`` argument.
        out_dtype: The dtype the product is summed in float32 and returned
            in, or ``None`` for the inputs' dtype.
        *rest: ``aten_op``'s other arguments: ``linear``'s bias.

    Returns:
        The product, stored as ``layout`` says.
    """
    stored, native = GemmLayout.parse(layout), GemmLayout.parse(native)
    if stored.activation != native.activation:
        activation = activation.mT
    if stored.weight != native.weight:
        weight = weight.mT
    if out_dtype is not None:
        activation, weight = activation.float(), weight.float()
        rest = [None if r is None else r.float() for r in rest]
    out = aten_op(activation, weight, *rest)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out.mT.contiguous() if stored.output != native.output else out


@torch.library.impl(
    _QUANTIZED_OPS_FRAGMENT, "linear", "CompositeExplicitAutograd"
)
def _linear(input, weight, bias=None, *, layout=LINEAR_LAYOUT, out_dtype=None):
    return _gemm_in_layout(
        torch.ops.aten.linear,
        LINEAR_LAYOUT,
        input,
        weight,
        layout,
        out_dtype,
        bias,
    )


@torch.library.impl(
    _QUANTIZED_OPS_FRAGMENT, "matmul", "CompositeExplicitAutograd"
)
def _matmul(input, other, *, layout=MATMUL_LAYOUT, out_dtype=None):
    return _gemm_in_layout(
        torch.ops.aten.matmul, MATMUL_LAYOUT, input, other, layout, out_dtype
    )


# The K-major twins ``normalize_activation_layout`` retargets to.  The
# attention twin reads its query and key ``[..., head_dim, tokens]``, its
# value and mask as aten's, and writes ``[..., tokens, head_dim]``, or
# ``[..., head_dim, tokens]`` given ``transpose_output``.  The RMS norm twin
# normalizes ``[..., hidden, tokens]`` over its hidden dim, the second-last,
# by a ``[hidden, 1]`` weight.
_QUANTIZED_OPS_FRAGMENT.define(
    "sdpa_kmajor(Tensor query, Tensor key, Tensor value, "
    "Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, "
    "float? scale=None, bool enable_gqa=False, bool transpose_output=False) "
    "-> Tensor"
)
_QUANTIZED_OPS_FRAGMENT.define(
    "rms_norm_kmajor(Tensor input, Tensor? weight=None, float? eps=None) "
    "-> Tensor"
)


@torch.library.impl(
    _QUANTIZED_OPS_FRAGMENT, "sdpa_kmajor", "CompositeExplicitAutograd"
)
def _sdpa_kmajor(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
    transpose_output=False,
):
    out = torch.ops.aten.scaled_dot_product_attention(
        query.mT.contiguous(),
        key.mT.contiguous(),
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.mT.contiguous() if transpose_output else out


@torch.library.impl(
    _QUANTIZED_OPS_FRAGMENT, "rms_norm_kmajor", "CompositeExplicitAutograd"
)
def _rms_norm_kmajor(input, weight=None, eps=None):
    hidden = [input.shape[-2]]
    if weight is not None:
        weight = weight.reshape(hidden)
    out = torch.ops.aten.rms_norm(input.mT, hidden, weight, eps)
    return out.mT.contiguous()
