import itertools
import logging
import math
import operator
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.aten_classifier import is_elementwise_op
from voyager_compiler.codegen.node_info import (
    AXES_ARG_INDEX_MAP,
    gemm_layout,
    get_arg_value,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_linear,
    is_matmul,
    is_nop,
    is_reshape_op,
    quant_param_arg_nodes,
    swaps_last_two_dims,
)
from voyager_compiler.codegen.transform.operator_fusion import (
    duplicate_shared_nodes,
)
from voyager_compiler.codegen.transform.rewrites import deduplicate_nodes
from voyager_compiler.export_utils import (
    create_getattr_from_value,
    derived_producer,
)
from voyager_compiler.ops.layout import (
    GEMM_WEIGHT_LAYOUTS,
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    NHWC_TO_NCHW,
    OIHW_TO_HWIO,
)
from voyager_compiler.shape_prop import (
    fetch_attr,
    propagate_shape,
    written_buffers,
)

logger = logging.getLogger(__name__)

__all__ = [
    "eliminate_reshape_with_no_effect",
    "normalize_activation_layout",
    "normalize_conv2d_layout",
    "normalize_gemm_weight_layout",
]


def extract_conv2d_graph(
    model: GraphModule, start: Node, visited: Set[Node]
) -> List[Node]:
    """
    Depth-first worklist traversal over both consumers (users) and
    producers (input nodes), restricted to reshape/elementwise/transpose
    /indexing/quantization ops that are considered fusable.
    """

    # ``slice`` / ``cat`` are what L2 tiling emits when it splits a conv along
    # the reduction dim and concatenates the partial outputs.  Both preserve
    # rank and carry an int ``dim`` at args[1], so their layout can be remapped
    # (see ``_rewrite_node_args_for_layout``).  ``stack`` (adds a dim),
    # ``select`` (drops a dim) and ``index`` (args[1] is not a dim) must stay
    # out of the island.
    ALLOW_LIST_OPS = {
        torch.ops.aten.pad.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.cat.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default,
    }

    def should_traverse(node: Node) -> bool:
        # Only include reshape if the input is a 4D tensor
        if is_reshape_op(node):
            return len(node.shape) == 4

        if node.target == operator.getitem:
            src = node.args[0]
            return src.target == torch.ops.quantized_ops.quantize_mx.default

        return (
            node.target in NHWC_OP_VARIANTS
            or node.target in ALLOW_LIST_OPS
            or is_elementwise_op(node)
        )

    stack = [start]
    nodes_in_graph = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes_in_graph.add(node)

        adjacent_nodes = list(node.users) + list(node.all_input_nodes)

        for n in adjacent_nodes:
            if n not in visited and should_traverse(n):
                stack.append(n)

    node_to_idx = {n: i for i, n in enumerate(model.graph.nodes)}
    return sorted(nodes_in_graph, key=lambda n: node_to_idx[n])


def remap_pad_after_permute(
    pad: Tuple[int, ...], dims: Tuple[int, ...], ndim: int
) -> Tuple[int, ...]:
    """
    Remap padding after permuting a tensor.

    Args:
        pad: Original pad tuple as in F.pad (starts from last dim).
        dims: Permutation dimensions.
        ndim: Number of dimensions in the original tensor.

    Returns:
        Tuple[int, ...]: New pad tuple corresponding to permuted tensor.
    """
    # number of padded dimensions
    k = len(pad) // 2
    assert k <= ndim, "Pad dimensions exceed tensor dimensions"

    # original padded dims (from last to first)
    original_padded_dims = list(range(ndim - k, ndim))

    dim_to_new_index = {d: dims.index(d) for d in range(ndim)}

    new_pad_pairs = {i: (0, 0) for i in range(ndim)}

    # Assign padding for dimensions that were originally padded
    for i, orig_dim in enumerate(reversed(original_padded_dims)):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad_pairs[dim_to_new_index[orig_dim]] = (left, right)

    # Collect pads in reverse order (last-first)
    new_pad = []
    for i in sorted(new_pad_pairs.keys(), reverse=True):
        new_pad.extend(new_pad_pairs[i])

    return tuple(new_pad)


def _permute_inputs(
    node: Node, model: GraphModule, nhwc_nodes: List[Node]
) -> Optional[Node]:
    """Bring one input of the NHWC group into that layout, and return the node
    whose value it left stale for the caller to restamp.

    A conv's weight (or weight scale) is permuted in place, and a 4-D
    activation goes through a permute.  An operand the group's elementwise
    ops broadcast onto its activations (a per-channel scale ``[C, 1, 1]``,
    a spatial mask ``[1, 1, H, W]``) is ranked up to 4-D first, and a
    constant one is relaid once, into a buffer of its own."""
    graph = model.graph

    # Case A: Input is a weight (Parameter) or weight scale.
    conv = next((n for n in node.users if is_conv2d(n)), None)
    if node.op == "get_attr" and conv is not None:
        if is_depthwise_conv(conv):
            return None

        w = get_arg_value(conv, 1, "weight")
        ws = conv.kwargs.get("weight_scale")
        if node not in [w, ws]:
            return None

        logger.debug(f"Permuting {conv} parameter: {node}")
        param = fetch_attr(model, node.target)
        param.data = param.data.permute(2, 3, 1, 0)

        node.meta["dims"] = OIHW_TO_HWIO
        return node

    # Case B: an activation flowing into the group from outside, or a
    # broadcast operand of its aten elementwise ops (a quantize's scales
    # follow its axis argument instead)
    readers = [u for u in node.users if u in nhwc_nodes]
    rank = len(node.shape)
    broadcast = (
        0 < rank <= 4
        and node.value.numel() > 1
        and all(
            is_elementwise_op(u) and u.target.namespace == "aten"
            for u in readers
        )
    )
    constant = node.op == "get_attr" and node.target not in written_buffers(
        graph
    )
    if not broadcast and (rank != 4 or constant):
        return None

    full = (1,) * (4 - rank) + tuple(node.shape)
    logger.debug(f"Relaying {node} {full} with dims {NCHW_TO_NHWC}")
    with graph.inserting_after(node):
        if constant:
            relaid = create_getattr_from_value(
                model,
                graph,
                node.name + "_nhwc",
                _relaid(node.value, full, NCHW_TO_NHWC),
                derived_producer(node, _relaid, full, NCHW_TO_NHWC),
            )
        else:
            view = node
            if rank < 4:
                view = graph.call_function(
                    torch.ops.aten.reshape.default, (node, full)
                )
                view.meta["dtype"] = node.meta.get("dtype")
                propagate_shape(view, model)
            relaid = graph.call_function(
                torch.ops.aten.permute.default, (view, NCHW_TO_NHWC)
            )
            relaid.meta["dtype"] = node.meta.get("dtype")

    relaid.meta["dims"] = NCHW_TO_NHWC
    for user in readers:
        user.replace_input_with(node, relaid)
    return relaid


def _rewrite_node_args_for_layout(node: Node) -> None:
    input_dims = node.all_input_nodes[0].meta.get("dims")
    node.meta["dims"] = input_dims

    args = tuple(node.args)

    if node.target == torch.ops.aten.pad.default:
        pad = remap_pad_after_permute(args[1], input_dims, node.value.ndim)
        node.update_arg(1, pad)

    if node.target in (torch.ops.aten.slice.Tensor, torch.ops.aten.cat.default):
        dim = get_arg_value(node, 1, "dim", 0)
        if dim < 0:
            dim = dim + len(input_dims)
        node.update_arg(1, input_dims.index(dim))

    if is_reshape_op(node):
        if node.target == torch.ops.aten.transpose.int:
            dims = (args[1], args[2])
        else:
            dims = args[1]
        dims = [d + max(input_dims) + 1 if d < 0 else d for d in dims]
        dims = tuple(input_dims.index(d) for d in dims)
        node.update_arg(1, dims)

    idx = AXES_ARG_INDEX_MAP.get(node.target)
    if idx is not None and idx < len(args) and args[idx] is not None:
        axes = [a + len(input_dims) if a < 0 else a for a in args[idx]]
        axes = tuple(input_dims.index(a) for a in axes)
        node.update_arg(idx, axes)

    if node.target in NHWC_OP_VARIANTS:
        node.target = NHWC_OP_VARIANTS[node.target]
        node.meta["transposed"] = True
    elif is_conv2d(node):
        # aten.conv2d takes no layout, so it runs as its quantized_ops twin.
        if node.target is torch.ops.aten.conv2d.default:
            node.target = torch.ops.quantized_ops.conv2d.default
        node.kwargs = {**node.kwargs, "layout": "nhwc"}
        node.meta["transposed"] = True


def normalize_conv2d_layout(model: GraphModule):
    graph = model.graph
    visited: Set[Node] = set()
    touched: Set[Node] = set()

    for node in list(graph.nodes):
        if node in visited or (
            node.target not in NHWC_OP_VARIANTS and not is_conv2d(node)
        ):
            continue

        # The connected group of nodes that can share the NHWC layout
        nhwc_nodes = extract_conv2d_graph(model, node, visited)
        touched.update(nhwc_nodes)

        for n in nhwc_nodes:
            for input_node in list(n.all_input_nodes):
                if input_node in nhwc_nodes or "dims" in input_node.meta:
                    continue

                stale = _permute_inputs(input_node, model, nhwc_nodes)
                if stale is not None:
                    touched.add(stale)

            for user in list(n.users.keys()):
                if user in nhwc_nodes or "dims" in user.meta:
                    continue

                logger.debug(
                    f"Insert permute before {user} with dims (0, 3, 1, 2)"
                )
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default, (n, NHWC_TO_NCHW)
                    )
                permute_node.meta["dtype"] = n.meta.get("dtype")
                user.replace_input_with(n, permute_node)
                touched.add(permute_node)

            _rewrite_node_args_for_layout(n)

    # Restamp what moved, in graph (topological) order
    for node in graph.nodes:
        if node in touched:
            propagate_shape(node, model)

    graph.lint()
    model.recompile()
    eliminate_reshape_with_no_effect(model)
    return model


def eliminate_reshape_with_no_effect(model: GraphModule):
    deleted_nodes = set()
    for node in list(model.graph.nodes):
        if not is_reshape_op(node) or node in deleted_nodes:
            continue

        curr_node = node
        input_node = node.all_input_nodes[0]

        group = []
        while len(curr_node.users) == 1 and (
            is_reshape_op(curr_node) or is_nop(curr_node)
        ):
            group.append(curr_node)
            curr_node = next(iter(curr_node.users))

        val = input_node.value
        orig_x = torch.arange(val.numel(), dtype=torch.int32)

        x = orig_x.reshape(val.shape)

        last_valid_idx = -1

        for i, gn in enumerate(group):
            args = torch.fx.graph.map_arg(gn.args, lambda n: x)
            x = gn.target(*args)

            if torch.equal(x.reshape(-1), orig_x):
                last_valid_idx = i

        del group[last_valid_idx + 1 :]

        if len(group) <= 1:
            continue

        logger.debug(f"Eliminating reshape group: {[n.name for n in group]}")

        output_shape = group[-1].value.shape

        with model.graph.inserting_before(node):
            reshape_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, output_shape),
            )

        propagate_shape(reshape_node)
        reshape_node.meta["dtype"] = group[-1].meta.get("dtype")

        group[-1].replace_all_uses_with(reshape_node)

        for n in reversed(group):
            model.graph.erase_node(n)
            deleted_nodes.add(n)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


ALLOWED_UPSTREAM_OPS: Set[any] = {
    torch.ops.aten.select.int,
    torch.ops.aten.pad.default,
    torch.ops.quantized_ops.dequantize.default,
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.quantize_mx.default,
}


def find_upstream_transpose_or_param(
    node: torch.fx.Node,
) -> Optional[List[torch.fx.Node]]:
    """Walk upstream from a transpose, along its single data operand, to
    another transpose or a constant param.  Returns the linear chain
    ``[node, ..., anchor]`` (nearest ``node`` first), or ``None``.

    A relayout op's non-data inputs are quantization tables
    (``quant_param_arg_nodes``) and are excluded; a node left with more than
    one input is a merge, not a relayout, so the walk stops -- keeping the
    result the single chain the eliminators consume."""
    if not swaps_last_two_dims(node):
        return None

    chain = [node]
    curr = node.args[0]
    while isinstance(curr, Node):
        if swaps_last_two_dims(curr) or curr.op == "get_attr":
            return chain + [curr]

        if not (curr.target in ALLOWED_UPSTREAM_OPS or is_nop(curr)):
            return None

        data_inputs = [
            i
            for i in curr.all_input_nodes
            if i not in quant_param_arg_nodes(curr)
        ]
        if len(data_inputs) != 1:
            return None

        chain.append(curr)
        curr = data_inputs[0]

    return None


def _insert_transposed_input(arg: Node, model: GraphModule):
    """``arg`` transposed: a transpose of the last two dims is undone, a
    constant is transposed once, into a buffer of its own, and a computed
    tensor, or a buffer the graph writes (a KV cache, whose transposed copy
    would go stale), goes through a transpose op."""
    if swaps_last_two_dims(arg):
        return arg.args[0]
    with model.graph.inserting_after(arg):
        if arg.op == "get_attr" and arg.target not in written_buffers(
            model.graph
        ):
            value = fetch_attr(model, arg.target)
            # A view of the weight, unless it shares a larger storage, which
            # saving the view would write out whole.
            transposed = create_getattr_from_value(
                model,
                model.graph,
                arg.name + "_T",
                value.mT,
                derived_producer(arg, torch.ops.aten.transpose.int, -2, -1),
                copy=value.untyped_storage().nbytes() != value.nbytes,
            )
        else:
            transposed = model.graph.call_function(
                torch.ops.aten.transpose.int, (arg, -2, -1)
            )
    transposed.meta["dtype"] = arg.meta.get("dtype")
    propagate_shape(transposed, model)
    return transposed


def _remap_dim_args_after_transpose(node: Node) -> None:
    """Restate a node's dim-referencing arg after a cancelling
    ``transpose(-2, -1)`` around it is removed: a quantize's blocked ``axes``
    through the inverse permutation, or a ``pad``'s pairs by swapping the last
    two dims (pad is last-dim-first, so a last-dim-only pad grows a zero pair
    for the second-last first)."""
    if (index := AXES_ARG_INDEX_MAP.get(node.target)) is not None:
        rank = len(node.shape)
        swap = {rank - 2: rank - 1, rank - 1: rank - 2}
        axes = (a % rank for a in node.args[index])
        node.update_arg(index, tuple(swap.get(a, a) for a in axes))
    elif node.target is torch.ops.aten.pad.default:
        pad = list(get_arg_value(node, 1, "pad"))
        while len(pad) < 4:
            pad.append(0)
        pad[0:2], pad[2:4] = pad[2:4], pad[0:2]
        node.update_arg(1, tuple(pad))


def eliminate_canceling_transposes(
    model: GraphModule,
    chain: List[Node],
    transposed_nodes: Dict[Node, Node],
) -> bool:
    """
    Optimizes a chain like [select_3, select_2, quantize_default_1, transpose_3]
    when there's a matching matmul-side transpose (user of chain[0]).

    Steps:
      1. Check if the two transposes cancel (considering selects).
      2. If yes, detach intermediate nodes and remove the redundant transpose.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    up_t = chain[0]
    down_t = chain[-1]
    if up_t.target != torch.ops.aten.transpose.int:
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    rank = len(up_t.shape)
    if rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the upstream transpose node
    chain = duplicate_shared_nodes(graph, chain[1:])

    # Rewrite graph to remove cancelling transposes
    for n in chain:
        for arg in n.all_input_nodes:
            if arg == up_t:
                n.replace_input_with(up_t, up_t.args[0])
                continue
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _remap_dim_args_after_transpose(n)
        propagate_shape(n, model)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not up_t.users:
        graph.erase_node(up_t)

    logger.info(f"Eliminated redundant transposes: {up_t} and {down_t}")

    return True


def fold_transpose_into_constant(
    model: GraphModule,
    chain: List[Node],
    transposed_nodes: Dict[Node, Node],
) -> bool:
    graph = model.graph

    attr_node = chain[0]
    down_t = chain[-1]
    if attr_node.op != "get_attr" or attr_node.target in written_buffers(graph):
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    rank = len(attr_node.shape)
    if rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the transpose node
    chain = duplicate_shared_nodes(model.graph, chain[1:])

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _remap_dim_args_after_transpose(n)
        propagate_shape(n, model)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not attr_node.users:
        graph.erase_node(attr_node)

    logger.info(f"Folded {down_t} into constant: {attr_node}")

    return True


def _insert_transpose_op(
    model: GraphModule, node: Node, user: Node, transposed_nodes: dict
) -> Optional[List[Node]]:
    """Inserts a transpose operation before the user node.

    A ``dequantize`` on the way in decodes element by element, so it is the same
    tensor either side of it -- but the transpose goes *above* it, because a
    decode the GEMM reads through is one it folds into its own fetch, where one
    it reads from is an op with a buffer of its own.  Its scale and zero point
    turn with it, and the axes it blocks along are restated.
    """
    # The legacy path splits the heads before the decode, so the GEMM's operand
    # arrives through a chain of ``select``s on the leading dims.
    chain = [node]
    while chain[-1].target == torch.ops.aten.select.int:
        chain.append(chain[-1].args[0])

    dq = chain[-1]
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if dq.target != torch.ops.quantized_ops.dequantize.default or any(
        n.args[1] != 0 for n in selects
    ):
        dq = None
    else:
        # Every head shares the decode, so give this GEMM one of its own to
        # turn; once all of them are turned the copies are identical again and
        # ``deduplicate_nodes`` folds them back.
        chain = duplicate_shared_nodes(model.graph, chain)
        dq = chain[0]

    # Transpose what the decode reads, or what the GEMM reads if there is none.
    src, consumer = (dq.args[0], dq) if dq is not None else (node, user)

    with model.graph.inserting_before(consumer):
        new_node = model.graph.call_function(
            torch.ops.aten.transpose.int, (src, -2, -1)
        )

    new_node.meta["dtype"] = src.meta.get("dtype")
    propagate_shape(new_node, model)
    consumer.replace_input_with(src, new_node)

    if dq is not None:
        for arg in list(dq.all_input_nodes):
            if arg is new_node or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            dq.replace_input_with(arg, transposed_nodes[arg])
        _remap_dim_args_after_transpose(dq)
        # The decode -- and every select over it -- now hands on a transposed
        # tensor, which is the one the GEMM wanted.
        for n in chain:
            propagate_shape(n, model)

    path = find_upstream_transpose_or_param(new_node)
    if path is None or len(path) < 2:
        return

    node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    path = sorted(path, key=lambda n: node_order[n])

    eliminate_canceling_transposes(model, path, transposed_nodes)
    fold_transpose_into_constant(model, path, transposed_nodes)


def _node_layout(node: Node, gemm_weight: str, gemv_weight: str) -> str:
    """The target weight layout of ``node``'s class: matrix-vector
    (fully-connected / batch-1) nodes follow ``gemv_weight``, matrix-matrix
    nodes ``gemm_weight``."""
    return gemv_weight if is_fully_connected(node) else gemm_weight


def _set_gemm_layout(node: Node, **parts: str) -> None:
    """Restate GEMM ``node``'s ``layout`` argument with ``parts`` -- its
    ``activation``, ``weight`` or ``output`` layout -- replaced."""
    layout = gemm_layout(node)._replace(**parts)
    node.kwargs = {**node.kwargs, "layout": str(layout)}


def _process_linear_node(model: GraphModule, node: Node) -> None:
    """Point a linear at a CK-stored copy of its KC-native weight and state
    that in its ``layout``.  The caller decides which nodes to flip."""
    logger.info(f"Transposing weight for linear node {node.name}")

    weight_node = node.args[1]
    transposed = _insert_transposed_input(weight_node, model)
    node.replace_input_with(weight_node, transposed)

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        node.replace_input_with(
            scale_node, _insert_transposed_input(scale_node, model)
        )

    # Mark mx / spmm_csr users as consuming a CK-stored weight.
    for user in list(transposed.users):
        if user.target == torch.ops.quantized_ops.linear_mx.default:
            _set_gemm_layout(user, weight="ck")
        elif user.target == torch.ops.quantized_ops.spmm_csr.default:
            user.kwargs = {**user.kwargs, "weight_layout": "ck"}

    if node.target == torch.ops.aten.linear.default:
        node.target = torch.ops.quantized_ops.linear.default
        _set_gemm_layout(node, weight="ck")


def _process_matmul_node(
    model: GraphModule, node: Node, transposed_nodes: dict
) -> None:
    """Flip a matmul's CK-native right operand storage to KC and state that
    in its ``layout``.  The caller decides which nodes to flip."""
    logger.info(f"Transposing weight for matmul node {node.name}")

    weight_node = node.args[1]
    _insert_transpose_op(model, weight_node, node, transposed_nodes)

    if (scale_node := node.kwargs.get("weight_scale")) is not None:
        _insert_transpose_op(model, scale_node, node, transposed_nodes)

    if node.target == torch.ops.aten.matmul.default:
        node.target = torch.ops.quantized_ops.matmul.default
    _set_gemm_layout(node, weight="kc")


def normalize_gemm_weight_layout(
    model: GraphModule,
    gemm_weight: str,
    gemv_weight: str,
) -> GraphModule:
    """Normalize every GEMM right-operand (weight) to one layout per
    operation class: matrix-matrix nodes to ``gemm_weight``, matrix-vector
    (fully-connected / batch-1) nodes to ``gemv_weight``; each is ``"kc"``
    ([out, contraction]) or ``"ck"`` ([contraction, out]).

    ``aten.linear`` is KC-native and ``aten.matmul`` CK-native, so
    exactly the nodes whose class targets the other layout have their
    weight storage flipped, and say so in their ``layout`` argument: an
    aten op is retargeted to the ``quantized_ops`` GEMM of its name, which
    takes one.  Eager execution stays correct in every configuration with
    no global patching.
    """
    assert gemm_weight in GEMM_WEIGHT_LAYOUTS, gemm_weight
    assert gemv_weight in GEMM_WEIGHT_LAYOUTS, gemv_weight

    transposed_nodes = {}

    for node in list(model.graph.nodes):
        # A node is flipped exactly when its class's target layout differs
        # from the op's native one (linear: KC, matmul: CK).
        if is_linear(node):
            if _node_layout(node, gemm_weight, gemv_weight) == "ck":
                _process_linear_node(model, node)
        elif is_matmul(node):
            if _node_layout(node, gemm_weight, gemv_weight) == "kc":
                _process_matmul_node(model, node, transposed_nodes)

    deduplicate_nodes(model)

    model.graph.lint()
    model.recompile()
    return model


# -------------------------------------------------------------------------
# K-major activations
# -------------------------------------------------------------------------
aten = torch.ops.aten
quantized_ops = torch.ops.quantized_ops

SDPA = aten.scaled_dot_product_attention.default
# The ops that read their activation K-major: the GEMMs taking a CK weight,
# each restated as the GEMM taking a ``layout`` argument, and attention, as
# its K-major twin.
TWINS = {
    quantized_ops.linear.default: quantized_ops.linear.default,
    aten.matmul.default: quantized_ops.matmul.default,
    SDPA: quantized_ops.sdpa_kmajor.default,
}
VIEW_OPS = {
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
    aten.unsqueeze.default,
}
MOVE_OPS = {aten.transpose.int, aten.permute.default}
SAME_LAYOUT_OPS = {aten.contiguous.default, aten.clone.default}
SHAPE_OPS = VIEW_OPS | MOVE_OPS | SAME_LAYOUT_OPS | {aten.expand.default}
DIM_OPS = {aten.slice.Tensor, aten.cat.default}
# A write of ``source`` into a buffer along a dim, in place: a KV cache's.
CACHE_WRITES = {aten.index_copy_.default, aten.index_copy.default}
RMS_NORM = aten.rms_norm.default
AGNOSTIC_OPS = SHAPE_OPS | DIM_OPS | {RMS_NORM}
# The ops whose output's layout follows their first input's.
SINGLE_INPUT_OPS = SHAPE_OPS | CACHE_WRITES | {aten.slice.Tensor, RMS_NORM}

# The side of the square blocks a relayout transposes.
BLOCK = 128

Perm = Tuple[int, ...]
# One layout change: (consumer, operand, operand's perm, perm wanted).
Relayout = Tuple[Node, Node, Perm, Perm]


def _identity(rank: int) -> Perm:
    return tuple(range(rank))


def _swapped(rank: int) -> Perm:
    """The last two dims swapped; a tensor of rank below 2 has one order."""
    perm = list(range(rank))
    if rank >= 2:
        perm[-2], perm[-1] = perm[-1], perm[-2]
    return tuple(perm)


def _norm_layout(norm: Node, perm: Perm) -> Optional[str]:
    """How an RMS norm runs on its input stored in ``perm``: ``"row"`` when
    the normalized dims stay innermost in order, ``"kmajor"`` when the one
    normalized dim lies second-last, on the partitions
    (``rms_norm_kmajor``), and ``None`` when neither holds.  Leading dims
    may lie in any order: they are batch."""
    rank = len(perm)
    count = len(norm.args[1])
    if perm[rank - count :] == tuple(range(rank - count, rank)):
        return "row"
    if count == 1 and perm[-2] == rank - 1:
        return "kmajor"
    return None


def _is_tensor(node) -> bool:
    return isinstance(node, Node) and isinstance(
        getattr(node, "value", None), torch.Tensor
    )


def _innermost(shape, perm: Perm) -> Optional[int]:
    """The logical axis stored innermost under ``perm``, unit axes skipped."""
    return next((a for a in reversed(perm) if shape[a] > 1), None)


def transposed_blocks(shape, src: Perm, dst: Perm) -> int:
    """The 128 x 128 blocks a relayout of a tensor of logical ``shape`` from
    physical perm ``src`` to ``dst`` transposes: none when the innermost
    non-unit axis stays innermost (a row reorder), else the blocks of the
    two axes that trade the free dim."""
    old, new = _innermost(shape, src), _innermost(shape, dst)
    if old == new:
        return 0
    cols, rows = shape[old], shape[new]
    batch = math.prod(shape) // (rows * cols)
    return batch * math.ceil(rows / BLOCK) * math.ceil(cols / BLOCK)


def _data_fraction(node: Node) -> float:
    """The share of ``node``'s elements that are distinct data: below 1 for
    a view of an ``expand`` (GQA's repeated K and V heads)."""
    fraction = 1.0
    while node.op == "call_function" and node.target in SHAPE_OPS:
        src = node.args[0]
        if node.target is aten.expand.default:
            fraction *= src.value.numel() / node.value.numel()
        node = src
    return fraction


def _view_perm(
    node: Node, perm: Perm, shapes: Dict[Node, tuple]
) -> Optional[Perm]:
    """The perm of a view's output when its input is stored in ``perm`` and
    the view is applied to that physical tensor, or ``None`` when no
    reshape of it holds the view's elements.  Of the perms that do, the one
    displacing the fewest unit axes, then inverting the fewest pairs."""
    src_shape, out_shape = shapes[node.args[0]], shapes[node]
    x = torch.arange(math.prod(src_shape), dtype=torch.int32)
    x = x.reshape(src_shape)
    physical = x.permute(perm)
    logical = x.reshape(out_shape)

    def displaced(p):
        pos = {a: i for i, a in enumerate(p)}
        units = sum(
            pos[a] != (pos[a - 1] + 1 if a else 0)
            for a in p
            if out_shape[a] == 1
        )
        return units, sum(a > b for a, b in itertools.combinations(p, 2))

    candidates = sorted(
        itertools.permutations(range(len(out_shape))), key=displaced
    )
    for out_perm in candidates:
        phys_shape = [out_shape[a] for a in out_perm]
        if torch.equal(logical.permute(out_perm), physical.reshape(phys_shape)):
            return out_perm
    return None


def _on_partitions(dim: int, perm: Perm) -> bool:
    """Whether logical ``dim`` lands on the partition axis, the physical
    second-last one."""
    return perm.index(dim % len(perm)) == len(perm) - 2


def _move_dims(node: Node, rank: int) -> Perm:
    """The input dim each output dim of a ``transpose`` / ``permute`` takes."""
    if node.target is aten.permute.default:
        return tuple(d % rank for d in node.args[1])
    a, b = (d % rank for d in node.args[1:3])
    dims = list(range(rank))
    dims[a], dims[b] = dims[b], dims[a]
    return tuple(dims)


def _cat_dim(cat: Node) -> int:
    return cat.args[1] if len(cat.args) > 1 else 0


def _slice_bounds(node: Node) -> Optional[Tuple[int, int, int]]:
    """``(dim, start, end)`` of a unit-step ``slice``, ``dim`` and ``end``
    resolved against its input, or ``None`` for another op."""
    if node.op != "call_function" or node.target is not aten.slice.Tensor:
        return None
    shape = node.args[0].value.shape
    defaults = [0, None, None, 1]
    dim, start, end, step = list(node.args[1:]) + defaults[len(node.args) - 1 :]
    if step != 1:
        return None
    dim %= len(shape)
    end = shape[dim] if end is None else min(end, shape[dim])
    return dim, start or 0, end


def _rotated_half(cat: Node) -> Optional[Node]:
    """The tensor ``x`` a ``cat`` rotates by half along its dim -- RoPE's
    ``cat([-x[h:], x[:h]])`` of a dim of ``2h`` -- or ``None``."""
    pieces = cat.args[0]
    if len(pieces) != 2 or pieces[0].target is not aten.neg.default:
        return None
    upper, lower = pieces[0].args[0], pieces[1]
    upper_bounds, lower_bounds = _slice_bounds(upper), _slice_bounds(lower)
    if upper_bounds is None or lower_bounds is None:
        return None
    x = lower.args[0]
    if upper.args[0] is not x:
        return None
    dim = _cat_dim(cat) % x.value.ndim
    size = x.value.shape[dim]
    half = size // 2
    if size % 2 or upper_bounds != (dim, half, size):
        return None
    return x if lower_bounds == (dim, 0, half) else None


def _cache_writes(graph) -> Set[Node]:
    """The cache writes (``CACHE_WRITES``) into a ``get_attr`` buffer that
    only cache writes read, which can store it in either layout."""
    readers: Dict[str, List[Node]] = {}
    for node in graph.nodes:
        if node.op == "get_attr":
            readers.setdefault(node.target, []).extend(node.users)
    writes = set()
    for target, users in readers.items():
        if users and all(
            u.target in CACHE_WRITES
            and u.args[0].op == "get_attr"
            and u.args[0].target == target
            for u in users
        ):
            writes.update(users)
    return writes


class _Island:
    """A set of nodes stored in one orientation, planned before rewriting.

    Attributes:
        nodes: The members, in graph order.
        kmajor: Whether the plan stores the island K-major.
        perms: Each member's physical perm under the plan.
        relayouts: The layout changes the plan needs.
    """

    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.kmajor = False
        self.perms: Dict[Node, Perm] = {}
        self.relayouts: List[Relayout] = []


def _extend(perm: Perm, rank: int) -> Perm:
    """``perm`` of a broadcast operand, ranked up to ``rank`` with leading
    size-1 axes."""
    lead = rank - len(perm)
    return _identity(lead) + tuple(a + lead for a in perm)


def _kmajor_operands(twin: Node) -> tuple:
    """The operands a twin reads K-major: a GEMM's activation, an
    attention's query and key."""
    return twin.args[:2] if twin.target is SDPA else twin.args[:1]


def _demands(node: Node, perm: Perm, twins: Set[Node]) -> List[Relayout]:
    """The relayouts ``node``'s readers need of it, stored in ``perm``, when
    each reads it in its own layout: an op getting a K-major twin (one of
    ``twins``) its ``_kmajor_operands`` K-major, anything else row-major."""
    rank = len(perm)
    relayouts = []
    for user in node.users:
        if user in twins and any(a is node for a in _kmajor_operands(user)):
            dst = _swapped(rank)
        else:
            dst = _identity(rank)
        if dst != perm:
            relayouts.append((user, node, perm, dst))
    return relayouts


def _plan_kmajor(
    island: _Island,
    members: Dict[Node, _Island],
    shapes: Dict[Node, tuple],
    twins: Set[Node],
) -> Tuple[Dict[Node, Perm], List[Relayout]]:
    """The perms and relayouts of storing ``island`` K-major: its GEMMs
    write swapped, its views and moves carry their input's storage, and an
    input from outside enters swapped.  A ``cat`` across the partitions that
    is no rotate-half, or an op the storage cannot express, reads its input
    row-major."""
    perms: Dict[Node, Perm] = {}
    relayouts: List[Relayout] = []

    def perm_of(arg: Node) -> Perm:
        if members.get(arg) is island:
            return perms[arg]
        return _identity(arg.value.ndim)

    def need(node, arg, src, dst):
        if src != dst:
            relayouts.append((node, arg, src, dst))

    for node in island.nodes:
        rank = len(shapes[node])
        if node in twins:
            perms[node] = _swapped(rank)
            continue

        data = node.args[0]
        if node.target is aten.cat.default:
            data = node.args[0][0]
        src = perm_of(data)
        single = node.target in SINGLE_INPUT_OPS
        if single and members.get(data) is not island:
            # An input from outside enters in the island's orientation.
            need(node, data, src, _swapped(len(src)))
            src = _swapped(len(src))

        out = None
        if node.target in SAME_LAYOUT_OPS | {aten.expand.default}:
            out = src
        elif node.target in VIEW_OPS:
            out = _view_perm(node, src, shapes)
        elif node.target in MOVE_OPS:
            moved = _move_dims(node, rank)
            out = tuple(moved.index(a) for a in src)
        elif node.target is aten.slice.Tensor:
            out = src
        elif node.target in CACHE_WRITES:
            # The buffer is stored in ``out`` once, so only what the write
            # stores into it is relaid every step.
            out = src
            source = node.args[3]
            need(node, source, perm_of(source), out)
        elif node.target is aten.cat.default:
            out = src
            if _on_partitions(_cat_dim(node), src) and not _rotated_half(node):
                out = _identity(rank)
            for arg in node.args[0]:
                need(node, arg, perm_of(arg), out)
        elif node.target is RMS_NORM:
            out = src if _norm_layout(node, src) else None
        else:  # elementwise
            # A quantization table is indexed by value, not position, so it
            # is read as it is.
            tables = quant_param_arg_nodes(node)
            tensors = [
                a for a in node.args if _is_tensor(a) and a not in tables
            ]
            full = [
                a
                for a in tensors
                if members.get(a) is island and shapes[a] == shapes[node]
            ]
            out = perms[full[0]] if full else _swapped(rank)
            for arg in tensors:
                if arg.value.numel() > 1:
                    need(node, arg, _extend(perm_of(arg), rank), out)
            perms[node] = out
            continue

        if out is None:
            need(node, data, src, _identity(len(src)))
            out = _identity(rank)
        perms[node] = out

    for node in island.nodes:
        relayouts += [
            r
            for r in _demands(node, perms[node], twins)
            if members.get(r[0]) is not island or r[0] in twins
        ]
    return perms, relayouts


def _plan_row(
    island: _Island, twins: Set[Node]
) -> Tuple[Dict[Node, Perm], List[Relayout]]:
    """The perms and relayouts of leaving ``island`` as it is: each GEMM
    reading a member as its activation takes it transposed."""
    perms = {n: _identity(n.value.ndim) for n in island.nodes}
    relayouts = [r for n in island.nodes for r in _demands(n, perms[n], twins)]
    return perms, relayouts


def _moved_blocks(island: _Island, shapes: Dict[Node, tuple]) -> float:
    """The blocks the moves of ``island`` transpose when it is left as it
    is: each stores its input in a new order, standalone or fused into the
    GEMM reading it (a ``Kᵀ``)."""
    blocks = 0.0
    for node in island.nodes:
        if node.target not in MOVE_OPS:
            continue
        data = node.args[0]
        rank = len(shapes[data])
        dims = _move_dims(node, rank)
        moved = transposed_blocks(shapes[data], _identity(rank), dims)
        blocks += moved * _data_fraction(data)
    return blocks


def _cost(relayouts: List[Relayout], shapes: Dict[Node, tuple]) -> float:
    """The transposed blocks of ``relayouts``, each operand and target perm
    once, constants free."""
    seen = {}
    for _, arg, src, dst in relayouts:
        if arg.op == "get_attr":
            continue
        shape = (1,) * (len(dst) - len(shapes[arg])) + shapes[arg]
        blocks = transposed_blocks(shape, src, dst) * _data_fraction(arg)
        seen[(arg, dst)] = blocks
    return sum(seen.values())


def _resolve_writes(
    relayouts: List[Relayout],
    writes: Set[Node],
    perms: Dict[Node, Perm],
    shapes: Dict[Node, tuple],
) -> List[Relayout]:
    """``relayouts`` with each cache write's relayout restated by ``perms``,
    the decided layouts (row-major where absent): the write stores its source
    in the buffer's layout, and drops out where the two agree."""
    resolved = []
    for consumer, arg, src, dst in relayouts:
        if consumer in writes and arg is consumer.args[3]:
            rank = len(shapes[arg])
            src = perms.get(arg, _identity(rank))
            dst = perms.get(consumer, _identity(rank))
            if src == dst:
                continue
        resolved.append((consumer, arg, src, dst))
    return resolved


def _islands(
    model: GraphModule, twins: Set[Node], writes: Set[Node]
) -> List[_Island]:
    """The graph's islands: the outputs of ``twins``, of the cache
    ``writes`` and of layout-agnostic ops over activations of rank 2 or
    more, joined along agnostic consumers.  A cache write joins no island
    through what it stores, which is relaid on its way in, and the writes
    into one buffer share an island."""
    parent: Dict[Node, Node] = {}

    def find(n):
        while parent[n] is not n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for node in model.graph.nodes:
        if node.op != "call_function" or not _is_tensor(node):
            continue
        if node.value.ndim < 2:
            continue
        activations = [a for a in node.all_input_nodes if a.op != "get_attr"]
        if node in twins or node in writes:
            parent[node] = node
        elif node.target in AGNOSTIC_OPS or is_elementwise_op(node):
            if not activations:
                continue
            parent[node] = node
            for arg in activations:
                if arg in parent:
                    parent[find(arg)] = find(node)

    buffers: Dict[str, Node] = {}
    for node in [n for n in writes if n in parent]:
        first = buffers.setdefault(node.args[0].target, node)
        parent[find(node)] = find(first)

    # The members reading one outside tensor share its relayout, so they
    # share an orientation; a twin or a cache write relays it on its own.
    for node in model.graph.nodes:
        if node in parent or node.op == "get_attr":
            continue
        readers = [
            u
            for u in node.users
            if u in parent and u not in twins and u not in writes
        ]
        for user in readers[1:]:
            parent[find(user)] = find(readers[0])

    groups: Dict[Node, List[Node]] = {}
    for node in model.graph.nodes:
        if node in parent:
            groups.setdefault(find(node), []).append(node)
    return [_Island(nodes) for nodes in groups.values()]


def _relaid(t: torch.Tensor, shape, dims) -> torch.Tensor:
    """``t`` viewed at ``shape``, then permuted by ``dims``."""
    return t.reshape(shape).permute(dims)


def _insert_relayout(
    model: GraphModule,
    arg: Node,
    src: Perm,
    dst: Perm,
    shape: tuple,
) -> Node:
    """``arg``, stored in ``src``, stored in ``dst`` instead: ranked up
    first if it broadcasts, a reshape when only unit axes move, else a
    permute.  A constant is relaid into a buffer of its own."""
    graph = model.graph
    lead = len(dst) - len(shape)
    full = (1,) * lead + shape
    phys = tuple(full[a] for a in src)
    dims = tuple(src.index(a) for a in dst)
    with graph.inserting_after(arg):
        if arg.op == "get_attr":
            return create_getattr_from_value(
                model,
                graph,
                arg.name + "_kmajor",
                _relaid(arg.value, phys, dims),
                derived_producer(arg, _relaid, phys, dims),
            )
        if [a for a in src if full[a] > 1] == [a for a in dst if full[a] > 1]:
            out = tuple(full[a] for a in dst)
            return graph.call_function(aten.reshape.default, (arg, out))
        view = arg
        if lead:
            view = graph.call_function(aten.reshape.default, (arg, phys))
        return graph.call_function(aten.permute.default, (view, dims))


def _relay_buffer(model: GraphModule, target: str, src: Perm, dst: Perm):
    """Store buffer ``target``, stored in ``src``, in ``dst`` instead."""
    value = fetch_attr(model, target)
    owner, _, name = target.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    dims = tuple(src.index(a) for a in dst)
    setattr(module, name, value.permute(dims).contiguous())


def _rewrite(
    model: GraphModule, node: Node, perm: Perm, shapes: Dict[Node, tuple]
) -> None:
    """Restate a K-major island member's args for its output stored in
    ``perm``."""
    rank = len(perm)
    out_shape = shapes[node]
    if node.target in VIEW_OPS:
        node.target = aten.reshape.default
        node.args = (node.args[0], [out_shape[a] for a in perm])
    elif node.target is aten.expand.default:
        node.args = (node.args[0], [out_shape[a] for a in perm])
    elif node.target in DIM_OPS | CACHE_WRITES:
        node.update_arg(1, perm.index(_cat_dim(node) % rank))
    elif node.target is RMS_NORM and _norm_layout(node, perm) == "kmajor":
        _, _, weight, eps = (list(node.args) + [None, None])[:4]
        if weight is not None:
            # The weight scales the normalized dim, now the second-last.
            with model.graph.inserting_after(weight):
                weight = create_getattr_from_value(
                    model,
                    model.graph,
                    weight.name + "_kmajor",
                    weight.value.reshape(-1, 1),
                    derived_producer(weight, torch.reshape, (-1, 1)),
                )
        node.target = quantized_ops.rms_norm_kmajor.default
        node.args = (node.args[0], weight, eps)


def _rotation(size: int, dtype: torch.dtype) -> torch.Tensor:
    """The signed permutation ``R`` whose ``Rᵀ x`` rotates the ``size``
    rows of ``x`` by half: ``cat([-x[h:], x[:h]])``."""
    half = size // 2
    rotation = torch.zeros(size, size, dtype=dtype)
    rows = torch.arange(half)
    rotation[rows + half, rows] = -1
    rotation[rows, rows + half] = 1
    return rotation


def _rotate_by_matmul(graph, cat: Node, x: Node, rotation: Node) -> None:
    """Replace ``cat``, the rotate-half of K-major ``x`` across its
    partitions, by ``matmul(x, rotation, layout="cx,ck->kx")``, ``rotationᵀ
    x`` (``_rotation``)."""
    with graph.inserting_before(cat):
        matmul = graph.call_function(
            quantized_ops.matmul.default,
            (x, rotation),
            {"layout": "cx,ck->kx"},
        )
    matmul.meta["dtype"] = cat.meta.get("dtype")
    cat.replace_all_uses_with(matmul)
    neg, lower = cat.args[0]
    upper = neg.args[0]
    for node in (cat, neg, upper, lower):
        if not node.users:
            graph.erase_node(node)


def normalize_activation_layout(model: GraphModule) -> GraphModule:
    """Store the activations K-major wherever that transposes fewer blocks.

    A Trainium TensorE contracts over the SBUF partitions, so a GEMM wants its
    activation ``[M, K]`` stored ``[K, M]``.  ``normalize_gemm_weight_layout``
    stores the weights that way; this pass does it for the activations.  It
    splits the graph into islands -- the GEMM outputs and the layout-agnostic
    ops joined to them (elementwise ops, views, moves, ``expand``, ``slice``,
    ``cat``, RMS norms) -- and stores each island either as it is or K-major:
    its GEMM outputs' last two dims swapped and the swap carried through its
    views and moves, whichever transposes fewer 128 x 128 blocks where the
    island meets a reader that wants the other layout.  Every matrix-matrix
    GEMM reads its activation K-major and writes its output in its island's
    orientation, as its ``layout`` argument says (``"cx,ck->xk"``, or
    ``"cx,ck->kx"`` for an output ``[N, M]``).  So does every attention,
    ``sdpa_kmajor`` (its output by ``transpose_output``):
    TensorE contracts the head dim of ``Q Kᵀ`` over the partitions, so it
    reads its query and key K-major, ``[..., head_dim, tokens]``, and its
    value and mask as they are.  It writes a transposed output by transposing
    each output tile once, as many blocks as the relayout that output saves,
    so the plan counts neither.  A KV cache write (``CACHE_WRITES``) into a
    buffer only cache writes read keeps the buffer in its island's
    orientation: the buffer is relaid once, in place under its own name, and
    each step relays only what it writes, so a decode step's key cache is
    stored ``[..., head_dim, positions]`` for ``Q Kᵀ`` to read as it is.
    Where a layout still changes, an ``aten.permute`` does it.  The graph
    computes the physical tensors, so it stays executable.

    Two ops reduce or move data across the partitions of a K-major tensor, and
    both run on TensorE: an RMS norm whose normalized dim lies on the
    partitions -- the hidden dim, or a head's -- becomes ``rms_norm_kmajor``
    (a ones-matmul sums the squares), and RoPE's rotate-half
    -- ``cat([-x2, x1])`` of the two halves of the head dim -- becomes a
    K-major ``matmul`` by a signed permutation.  A GEMM whose activation is a
    single row or column -- a matrix-vector product, an outer product -- keeps
    its op: its activation has the same bytes either way.

    Args:
        model: The graph, after ``normalize_gemm_weight_layout`` has stored
            every GEMM weight CK; rewritten in place.

    Returns:
        ``model``.
    """
    graph = model.graph
    shapes = {n: tuple(n.value.shape) for n in graph.nodes if _is_tensor(n)}
    twins = {
        n
        for n in graph.nodes
        if n.target is SDPA
        or (n.target in TWINS and min(shapes[n.args[0]][-2:]) > 1)
    }
    writes = _cache_writes(graph)
    islands = _islands(model, twins, writes)
    members = {n: island for island in islands for n in island.nodes}

    # A cache write joins its buffer's island but not its source's, so the
    # buffers' islands are decided first, and every island prices each write
    # it takes part in by the layouts already decided, whether or not its
    # plan relays the write's source.
    decided: Dict[Node, Perm] = {}
    for island in sorted(islands, key=lambda i: writes.isdisjoint(i.nodes)):
        row = _plan_row(island, twins)
        kmajor = _plan_kmajor(island, members, shapes, twins)
        crossings = [
            (write, write.args[3], None, None)
            for write in writes
            if island in (members.get(write), members.get(write.args[3]))
        ]
        costs = [
            _cost(
                _resolve_writes(
                    relayouts + crossings, writes, decided | perms, shapes
                ),
                shapes,
            )
            for perms, relayouts in (row, kmajor)
        ]
        costs[0] += _moved_blocks(island, shapes)
        island.kmajor = costs[1] < costs[0]
        island.perms, island.relayouts = kmajor if island.kmajor else row
        if island.kmajor:
            decided |= island.perms
        logger.debug(
            f"island {island.nodes[0].name} ({len(island.nodes)} nodes): "
            f"row {costs[0]:.0f}, kmajor {costs[1]:.0f} blocks"
        )

    perms = {
        n: p
        for island in islands
        if island.kmajor
        for n, p in island.perms.items()
    }
    # A rotate-half across the partitions of a K-major island: the cat,
    # the tensor it rotates and the partitions' extent, read before the
    # rewrite restates them.
    rotations = [
        (n, x, shapes[x][_cat_dim(n) % len(shapes[x])])
        for n in perms
        if n.target is aten.cat.default
        and _on_partitions(_cat_dim(n), perms[n])
        and (x := _rotated_half(n)) is not None
    ]

    relayouts = [
        relayout
        for island in islands
        for relayout in _resolve_writes(island.relayouts, writes, perms, shapes)
    ]

    # A twin fed by a fixed op (softmax, the embedding) reads it transposed.
    for node in [n for n in graph.nodes if n in twins]:
        for arg in _kmajor_operands(node):
            if arg not in members:
                rank = len(shapes[arg])
                relayouts.append((node, arg, _identity(rank), _swapped(rank)))

    made = {}
    relaid_buffers = set()
    for consumer, arg, src, dst in relayouts:
        if consumer in writes and arg is consumer.args[0]:
            # A buffer is relaid in place, under its own name, once.
            if arg.target not in relaid_buffers:
                _relay_buffer(model, arg.target, src, dst)
                relaid_buffers.add(arg.target)
            continue
        if (arg, dst) not in made:
            made[(arg, dst)] = _insert_relayout(
                model, arg, src, dst, shapes[arg]
            )
        consumer.replace_input_with(arg, made[(arg, dst)])

    # A move in a K-major island stores nothing new: its readers read the
    # physical tensor it moves.
    moved = {}
    for node in list(perms):
        if node.target in MOVE_OPS:
            moved[node] = node.args[0]
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        elif node not in twins:
            _rewrite(model, node, perms[node], shapes)

    for node in twins:
        transposed = perms.get(node) == _swapped(len(shapes[node]))
        node.target = TWINS[node.target]
        if node.target is quantized_ops.sdpa_kmajor.default:
            node.kwargs = {**node.kwargs, "transpose_output": transposed}
        else:
            _set_gemm_layout(
                node, activation="cx", output="kx" if transposed else "xk"
            )

    tables = {}
    for cat, x, size in rotations:
        while x in moved:
            x = moved[x]
        key = (size, cat.value.dtype)
        if key not in tables:
            with graph.inserting_before(cat):
                tables[key] = create_getattr_from_value(
                    model, graph, "rotate_half", _rotation(*key)
                )
        _rotate_by_matmul(graph, cat, x, tables[key])

    for node in graph.nodes:
        if node.op in ("call_function", "get_attr"):
            propagate_shape(node, model)

    graph.lint()
    model.recompile()
    return model
