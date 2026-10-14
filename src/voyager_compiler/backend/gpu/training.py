"""Run a training step on a GPU with real FP8 and NVFP4 GEMMs.

``gradient_program`` writes a training step -- the forward, the loss and the
backward -- as one graph whose quantized GEMMs, forward (fprop), input
gradient (dgrad) and weight gradient (wgrad), read their operands' codes.  An
FP8 operand is a ``quantize`` against its delayed-scaling scale, which the
graph updates in place, read through views; its GEMM is scaled back by a
``dequantize``.  An NVFP4 operand is a ``quantize_mx`` blocked along its
GEMM's contraction axis, or in 16x16 blocks for a weight, after its random
Hadamard transform where the recipe has one; its GEMM is a ``linear_mx`` or
``matmul_mx`` scaled by ``alpha``.

``lower_training`` swaps each such GEMM for a real one, in place, as
``lower`` does for inference:

- each FP8 ``quantize`` becomes one cast to FP8 (``quantize_fp8``), and each
  GEMM with its dequantize one ``fp8_linear``.  That GEMM takes both operands
  K-major (TN), the only layout torch's FP8 GEMM takes, so an operand a dgrad
  or wgrad contracts along its rows is a transposed copy of its FP8 codes;
- each NVFP4 ``quantize_mx`` becomes one cast to packed codes
  (``quantize_nvfp4``, with its stochastic rounding's seed), and each GEMM one
  ``nvfp4_linear``.  A weight in 16x16 blocks is cast once
  (``quantize_nvfp4_square``) and packed both ways, so the dgrad reads the
  forward's codes transposed.

The delayed-scaling updates, the rotations and every other op run as written.
"""

import operator
from typing import Dict, List, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.quantization.training import _VIEWS

from . import nvfp4
from .fp8 import (
    DEQUANTIZE,
    FP8_DTYPES,
    LINEAR,
    MATMUL,
    QUANTIZE,
    fp8_linear,
    quantize_fp8,
)
from .nvfp4 import (
    BLOCK_SIZE,
    FP4_MAX,
    LINEAR_MX,
    NVFP4_DTYPES,
    NVFP4_LINEAR,
    QUANTIZE_MX,
    quantize_nvfp4,
    quantize_nvfp4_square,
)

__all__ = ["lower_training"]

aten = torch.ops.aten

MATMUL_MX = torch.ops.quantized_ops.matmul_mx.default

# The axes of a ``quantize_mx`` in 16x16 blocks.
SQUARE_BLOCKS = (-2, -1)


def _fp8_quantize(operand: Node) -> Node:
    """The FP8 ``quantize`` whose codes a GEMM reads, as they are or through
    views, as ``operand``.

    Raises:
        NotImplementedError: ``operand`` is not cast to FP8.
    """
    quantize = operand
    while quantize.target in _VIEWS:
        quantize = quantize.args[0]
    if (
        quantize.target is not QUANTIZE
        or quantize.meta.get("dtype") not in FP8_DTYPES
    ):
        raise NotImplementedError(f"{operand.name} is not cast to FP8")
    return quantize


def _insert_fp8_gemm(graph: Graph, gemm: Node) -> Node:
    """Insert the real GEMM computing an FP8 ``gemm`` and the dequantize after
    it, before that dequantize.

    A linear reads its ``[N, K]`` weight as it is; a matmul's ``[K, N]``
    second operand is copied transposed, so the GEMM reads it K-major.

    Args:
        graph: The program's graph, its FP8 casts in place.
        gemm: A linear, or a matmul whose second operand is 2-d, of two FP8
            operands; its only user scales it back.

    Returns:
        The GEMM.
    """
    (output,) = gemm.users
    input, other = gemm.args
    with graph.inserting_before(output):
        scale = graph.call_function(
            aten._to_copy.default, (output.args[1],), {"dtype": torch.float32}
        )
        if gemm.target is MATMUL:
            other = graph.call_function(
                aten.clone.default,
                (graph.call_function(aten.t.default, (other,)),),
                {"memory_format": torch.contiguous_format},
            )
        return graph.call_function(fp8_linear, (input, other, scale))


def _nvfp4_quantize(codes: Node) -> Node:
    """The NVFP4 ``quantize_mx`` whose codes a GEMM reads as ``codes``.

    Raises:
        NotImplementedError: ``codes`` are not NVFP4 codes.
    """
    quantize = codes.args[0] if codes.target is operator.getitem else codes
    if (
        quantize.target is not QUANTIZE_MX
        or quantize.meta.get("dtype") != NVFP4_DTYPES
        or quantize.args[3:5] != (BLOCK_SIZE, FP4_MAX)
        or get_arg_value(quantize, 9, "global_scale") is None
    ):
        raise NotImplementedError(f"{codes.name} is not cast to NVFP4")
    return quantize


def _insert_nvfp4_cast(
    graph: Graph, quantize: Node
) -> Dict[int, Tuple[Node, Node]]:
    """Insert the cast replacing an NVFP4 ``quantize``, before it, and make
    each ``alpha`` read the cast's global scale.

    Args:
        graph: The program's graph.
        quantize: An NVFP4 ``quantize_mx``.

    Returns:
        The cast's packed codes and block scales, by the axis of the tensor
        they are packed along: its blocks' axis, or both axes for 16x16
        blocks.

    Raises:
        NotImplementedError: ``quantize`` rounds 16x16 blocks stochastically,
            or blocks along two axes that are not the last two.
    """
    input, axes = quantize.args[0], tuple(quantize.args[2])
    seed = get_arg_value(quantize, 8, "seed")
    with graph.inserting_before(quantize):
        if len(axes) == 1:
            cast = graph.call_function(quantize_nvfp4, (input, axes[0], seed))
            packed_along = axes
        elif axes == SQUARE_BLOCKS and seed is None:
            cast = graph.call_function(quantize_nvfp4_square, (input,))
            packed_along = (-1, -2)
        else:
            raise NotImplementedError(
                f"{quantize.name}: blocks along {axes} have no NVFP4 cast"
            )
        outputs = [
            graph.call_function(operator.getitem, (cast, i))
            for i in range(2 * len(packed_along) + 1)
        ]
    get_arg_value(quantize, 9, "global_scale").replace_all_uses_with(
        outputs[-1]
    )
    return {
        axis: tuple(outputs[2 * i : 2 * i + 2])
        for i, axis in enumerate(packed_along)
    }


def _insert_nvfp4_gemm(
    program: GraphModule,
    gemm: Node,
    casts: Dict[Node, Dict[int, Tuple[Node, Node]]],
) -> Node:
    """Insert the casts of an NVFP4 ``gemm``'s operands no earlier GEMM
    inserted, and its real GEMM before it.

    Args:
        program: The gradient program.
        gemm: A ``linear_mx``, or a ``matmul_mx`` whose second operand is
            2-d.
        casts: Each NVFP4 ``quantize_mx`` cast so far -> its cast's packed
            operands, as ``_insert_nvfp4_cast`` returns them; updated.

    Returns:
        The GEMM.

    Raises:
        NotImplementedError: An operand is not NVFP4 codes blocked along
            the GEMM's contraction axis.
    """
    graph = program.graph
    # The axis each operand is contracted along: a linear's weight is
    # [N, K], a matmul's second operand [K, N].
    contraction = (-1, -1 if gemm.target is LINEAR_MX else -2)
    operands = []
    for codes, axis in zip(gemm.args[:2], contraction):
        quantize = _nvfp4_quantize(codes)
        if quantize not in casts:
            casts[quantize] = _insert_nvfp4_cast(graph, quantize)
        if axis not in casts[quantize]:
            raise NotImplementedError(
                f"{codes.name} is not blocked along {gemm.name}'s "
                "contraction axis"
            )
        operands.extend(casts[quantize][axis])
    bias = get_arg_value(gemm, 2, "bias")
    with graph.inserting_before(gemm):
        alpha, one = nvfp4.insert_gemm_scalars(
            program, gemm.kwargs["alpha"], gemm.meta["val"].device
        )
        return graph.call_function(NVFP4_LINEAR, (*operands, alpha, one, bias))


def _lower_fp8_gemms(graph: Graph) -> List[str]:
    """Replace each FP8 ``quantize`` with one cast to FP8 that every GEMM it
    fed reads, and each FP8 GEMM with its dequantize by one ``fp8_linear``.

    Args:
        graph: The program's graph, rewritten in place.

    Returns:
        The names of the GEMMs replaced.

    Raises:
        NotImplementedError: A GEMM scaled back by a dequantize reads an
            operand that is not cast to FP8.
    """
    gemms = [
        gemm
        for gemm in graph.nodes
        if gemm.target in (LINEAR, MATMUL)
        and any(user.target is DEQUANTIZE for user in gemm.users)
    ]
    quantizes = dict.fromkeys(
        _fp8_quantize(operand) for gemm in gemms for operand in gemm.args
    )
    for quantize in quantizes:
        dtype = FP8_DTYPES[quantize.meta["dtype"]]
        with graph.inserting_before(quantize):
            cast = graph.call_function(
                quantize_fp8, (*quantize.args[:2], dtype)
            )
        quantize.replace_all_uses_with(cast)
        graph.erase_node(quantize)
    for gemm in gemms:
        (output,) = gemm.users
        output.replace_all_uses_with(_insert_fp8_gemm(graph, gemm))
        graph.erase_node(output)
        graph.erase_node(gemm)
    return [gemm.name for gemm in gemms]


def _lower_nvfp4_gemms(program: GraphModule) -> List[str]:
    """Replace each NVFP4 ``quantize_mx`` with one cast to packed codes, and
    each ``linear_mx`` and ``matmul_mx`` by one ``nvfp4_linear``.

    Args:
        program: The gradient program, rewritten in place.

    Returns:
        The names of the GEMMs replaced.

    Raises:
        NotImplementedError: A GEMM reads an operand that is not NVFP4 codes
            blocked along its contraction axis.
    """
    graph = program.graph
    gemms = [
        gemm for gemm in graph.nodes if gemm.target in (LINEAR_MX, MATMUL_MX)
    ]
    casts = {}
    for gemm in gemms:
        gemm.replace_all_uses_with(_insert_nvfp4_gemm(program, gemm, casts))
        graph.erase_node(gemm)
    return [gemm.name for gemm in gemms]


def lower_training(program: GraphModule) -> Dict[str, str]:
    """Swap each quantized GEMM of a gradient program for a real one, in
    place.

    Every quantized GEMM is a linear's: a forward linear, or a backward
    matmul whose second operand is 2-d.  The ops the old casts leave unused,
    such as the global scales' amaxes, are dropped.

    Args:
        program: A program ``gradient_program`` built from a bfloat16 model.

    Returns:
        Per GEMM lowered, by name: ``"fp8"`` or ``"nvfp4"``.

    Raises:
        NotImplementedError: A quantized GEMM reads an operand that is not
            cast to FP8 or NVFP4, or NVFP4 codes it cannot read; ``program``
            is then left partly rewritten.
    """
    graph = program.graph
    lowered = dict.fromkeys(_lower_fp8_gemms(graph), "fp8")
    nvfp4_gemms = _lower_nvfp4_gemms(program)
    lowered.update(dict.fromkeys(nvfp4_gemms, "nvfp4"))
    graph.eliminate_dead_code()
    graph.lint()
    program.recompile()
    return lowered
