"""The builder registry: which builder lowers each compute node.

``bufferize_graph`` walks a table of ``BuilderRule``s and hands a node to the
first rule that matches it; a backend lowers a node its own way by putting its
rule ahead of ``SHARED_BUILDERS``.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import (
    DYNAMIC_QUANTIZE_OPS,
    NORM_OPS,
    SOFTMAX_OPS,
    csr_quantize_node,
    is_axis_reduction,
    is_conv2d,
    is_elementwise_op,
    is_full_reduction,
    is_gemm_op,
    is_pooling,
)
from voyager_compiler.codegen.transform.bufferize.attention import (
    build_attention,
)
from voyager_compiler.codegen.transform.bufferize.attention_v3 import (
    build_attention_fa3,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    build_conv2d,
    build_gemm,
    build_pointwise,
    build_pool,
)
from voyager_compiler.codegen.transform.bufferize.quantize_mx_outlier import (
    build_quantize_mx_outlier,
)
from voyager_compiler.codegen.transform.bufferize.sparse_gemm import (
    build_sparse_gemm,
)
from voyager_compiler.codegen.transform.bufferize.utils import BuildContext

Builder = Callable[[Node, BuildContext], Optional[GraphModule]]


@dataclass(frozen=True)
class BuilderRule:
    """One arm of the dispatch: which nodes it takes and how it lowers them.

    Attributes:
        name: The rule's name, for logs and for finding it in a table.
        matches: ``(node, anchor) -> bool``; ``anchor`` is the node's anchor
            op (the node itself unless it is a fused ``call_module``).
        build: Lowers the node to a bufferized ``GraphModule``, or returns
            ``None`` to leave it to the untiled path.
    """

    name: str
    matches: Callable[[Node, Node], bool]
    build: Builder


_ATTENTION_OPS = (
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.quantized_ops.sdpa_mx.default,
)
_REDUCTION_POINTWISE_OPS = (
    DYNAMIC_QUANTIZE_OPS
    | NORM_OPS
    | SOFTMAX_OPS
    | {
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.rms_norm_kmajor.default,
    }
)
# A standalone transpose / permute relayout — build_pointwise stores each tile
# identity and loads the input from the transposed source.  (Untiled ones fall
# to ``_build_for_untiled``; build_pointwise returns ``None`` when untiled.)
_RELAYOUT_POINTWISE_OPS = {
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
}


def _build_attention(node: Node, ctx: BuildContext) -> Optional[GraphModule]:
    """Lower an attention node with the builder ``ctx`` selects."""
    if ctx.flash_attention_v3:
        return build_attention_fa3(node, ctx)
    return build_attention(node, ctx)


def _build_quantize_mx_outlier(node: Node, ctx: BuildContext) -> GraphModule:
    """Lower a row-swept outlier producer, which has no fallback.

    ``build_pointwise`` would hand the quantize a full-K tile, whose column
    indices span all of K and which the SpMM engine cannot consume.

    Raises:
        ValueError: No row-swept nest could be built for the node.
    """
    sub_gm = build_quantize_mx_outlier(node, ctx)
    if sub_gm is None:
        raise ValueError(
            f"{node.name}: produces an outlier CSR but no row-swept nest "
            f"could be built for it"
        )
    return sub_gm


def _is_pointwise(node: Node, anchor: Node) -> bool:
    """Whether ``build_pointwise`` covers ``anchor``'s op."""
    return (
        is_elementwise_op(anchor)
        or is_axis_reduction(anchor)
        or is_full_reduction(anchor)
        or anchor.target in _REDUCTION_POINTWISE_OPS
        or anchor.target in _RELAYOUT_POINTWISE_OPS
    )


SHARED_BUILDERS: tuple[BuilderRule, ...] = (
    BuilderRule("conv2d", lambda node, anchor: is_conv2d(anchor), build_conv2d),
    # A GEMM whose activation carries an outlier CSR: the dense nest, plus
    # the per-step gather of the row tile's blocks.
    BuilderRule(
        "sparse_gemm",
        lambda node, anchor: anchor.kwargs.get("A_indptr") is not None,
        build_sparse_gemm,
    ),
    BuilderRule("gemm", lambda node, anchor: is_gemm_op(anchor), build_gemm),
    BuilderRule("pool", lambda node, anchor: is_pooling(anchor), build_pool),
    BuilderRule(
        "attention",
        lambda node, anchor: anchor.target in _ATTENTION_OPS,
        _build_attention,
    ),
    # A row-swept producer: the quantize needs a per-K-slice tiling its
    # anchor cannot give it, so it gets its own two-loop nest rather than
    # build_pointwise's single grid.
    BuilderRule(
        "quantize_mx_outlier",
        lambda node, anchor: csr_quantize_node(node) is not None,
        _build_quantize_mx_outlier,
    ),
    BuilderRule("pointwise", _is_pointwise, build_pointwise),
)
