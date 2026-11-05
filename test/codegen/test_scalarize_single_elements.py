"""One-element constants of elementwise ops become scalar operands.

``scalarize_single_elements`` passes a single-element constant that an
elementwise op combines with full tensors as an immediate of the op's scalar
overload, leaves the operands a quantized op takes whole as tensors, and keeps
every output bit-identical.
"""

import copy

import torch
from torch import nn

import voyager_compiler  # noqa: F401 (registers quantized_ops)
from voyager_compiler import ShapeProp, export_model
from voyager_compiler.codegen import (
    fold_constant_generators,
    scalarize_single_elements,
)

aten = torch.ops.aten

MIN = torch.finfo(torch.bfloat16).min


class Mask(nn.Module):
    """An attention mask's additive form, as transformers builds it."""

    def forward(self, mask):
        return torch.where(mask, torch.tensor(0.0), MIN)


class Bf16Mask(nn.Module):
    """The additive mask with a bfloat16 zero."""

    def forward(self, mask):
        return torch.where(mask, torch.tensor(0.0, dtype=torch.bfloat16), MIN)


class Arithmetic(nn.Module):
    """A sum, a product and a comparison with one-element constants, and a
    product with a computed one-element tensor."""

    def forward(self, x):
        total = x + torch.tensor(2.0)
        scaled = total * torch.tensor(0.1)
        below = x < torch.tensor(3)
        normed = scaled * x.sum()
        return normed, below


class Quantized(nn.Module):
    """A quantize whose scalar scale also scales the input by a product."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.5]).bfloat16())

    def forward(self, x):
        quantized = torch.ops.quantized_ops.quantize(x, self.scale)
        return quantized + x * self.scale


def scalarized(model, args):
    """``model`` exported on ``args`` with its constants folded, as
    ``transform`` runs it up to the pass, before and after the pass."""
    graph = export_model(model, args)
    ShapeProp(graph).propagate(*args)
    fold_constant_generators(graph)
    original = copy.deepcopy(graph)
    scalarize_single_elements(graph)
    return original, graph


def calls(graph):
    """The ``call_function`` nodes of ``graph``."""
    return [n for n in graph.graph.nodes if n.op == "call_function"]


def one_element_constants(node):
    """The one-element ``get_attr`` operands of ``node``."""
    return [
        a
        for a in node.all_input_nodes
        if a.op == "get_attr" and a.value.numel() == 1
    ]


def assert_bit_identical(original, graph, args):
    """Both graphs give the same outputs on ``args``, bit for bit."""
    expected = original(*args)
    actual = graph(*args)
    if isinstance(expected, torch.Tensor):
        expected, actual = (expected,), (actual,)
    assert len(expected) == len(actual)
    for want, got in zip(expected, actual):
        assert want.dtype == got.dtype
        assert torch.equal(want.view(torch.uint8), got.view(torch.uint8))


def test_mask_zero_becomes_scalar():
    args = (torch.rand(1, 1, 8, 8) > 0.5,)
    original, graph = scalarized(Mask(), args)
    (where,) = [n for n in calls(graph) if "where" in str(n.target)]
    assert where.target is aten.where.Scalar
    assert where.args[1:] == (0.0, MIN)
    assert not one_element_constants(where)
    assert_bit_identical(original, graph, args)


def test_bf16_mask_keeps_its_dtype():
    # where.Scalar is float32: the bfloat16 zero stays a tensor operand.
    args = (torch.rand(1, 1, 8, 8) > 0.5,)
    original, graph = scalarized(Bf16Mask(), args)
    (where,) = [n for n in calls(graph) if "where" in str(n.target)]
    assert where.target is aten.where.ScalarOther
    assert where.value.dtype == torch.bfloat16
    assert_bit_identical(original, graph, args)


def test_arithmetic_takes_scalar_overloads():
    args = (torch.randn(4, 64).bfloat16() * 4,)
    original, graph = scalarized(Arithmetic(), args)
    targets = [n.target for n in calls(graph)]
    assert aten.add.Scalar in targets
    assert aten.lt.Scalar in targets
    for node in calls(graph):
        if node.target in (aten.add.Scalar, aten.lt.Scalar):
            assert not one_element_constants(node)
    products = [n for n in calls(graph) if n.target is aten.mul.Scalar]
    assert [n.args[1] for n in products] == [torch.tensor(0.1).item()]
    # The computed sum stays a tensor operand.
    (computed,) = [n for n in calls(graph) if n.target is aten.mul.Tensor]
    assert computed.args[1].target is aten.sum.default
    assert_bit_identical(original, graph, args)


def test_quantize_scale_stays_a_tensor():
    args = (torch.randn(4, 64).bfloat16(),)
    original, graph = scalarized(Quantized(), args)
    (quantize,) = [
        n
        for n in calls(graph)
        if n.target is torch.ops.quantized_ops.quantize.default
    ]
    scale = quantize.args[1]
    assert scale.op == "get_attr"
    (product,) = [n for n in calls(graph) if "mul" in str(n.target)]
    assert product.target is aten.mul.Tensor
    assert product.args[1] is scale
    assert_bit_identical(original, graph, args)
