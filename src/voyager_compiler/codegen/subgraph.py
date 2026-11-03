"""Moving code between FX graphs.

Outlining -- lift a run of nodes into a submodule and call it -- and its
inverse, splicing an exported ``GraphModule`` back into a parent graph in
place of a node.  Plus the naming rules both directions use.
"""

import logging
import operator
from typing import Dict, List, Optional

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg
from transformers.utils.import_utils import is_torch_greater_or_equal

from voyager_compiler.codegen.iteration_space import (
    NormalizationError,
    normalize_group,
)
from voyager_compiler.codegen.node_info import (
    NORM_OPS,
    get_arg_value,
    is_gemm_op,
    is_nop,
    is_reshape_op,
    swaps_last_two_dims,
)
from voyager_compiler.export_utils import create_getattr_from_value
from voyager_compiler.shape_prop import (
    ShapeProp,
    fetch_attr,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)


def copy_graph_module(
    gm: GraphModule, remap: Optional[Dict[Node, Node]] = None
) -> GraphModule:
    """An independent-graph copy of ``gm`` (recursively for child
    GraphModules) that preserves each node's ``meta`` and ``.value`` and shares
    the constant params / buffers — WITHOUT recompiling ``forward``.

    The body stays runnable: ``node_copy`` preserves node names, so the shared
    (already-compiled) ``forward`` works on the copied graph — we skip only the
    *recompile*, which ``deepcopy`` and the ``.graph`` setter both force.  But a
    shared body *object* breaks per-site meta (e.g. scratchpad Segments stamped
    on its nodes), so each splice needs its own node objects; this gives them
    cheaply.

    ``object.__new__`` (not ``gm.__class__.__new__``) is required: fx's
    ``GraphModule.__new__`` mints a fresh ``forward``-less subclass each call,
    whereas we want to keep ``gm``'s class (which carries the compiled
    ``forward``).
    """
    root = remap is None
    if remap is None:
        remap = {}

    new = object.__new__(gm.__class__)
    new.__dict__ = dict(gm.__dict__)  # share params / buffers / forward code
    new._modules = {
        k: copy_graph_module(v, remap) if isinstance(v, GraphModule) else v
        for k, v in gm._modules.items()
    }
    new_graph = Graph()
    for n in gm.graph.nodes:
        c = new_graph.node_copy(n, lambda x: remap[x])
        remap[n] = c
        if (val := getattr(n, "value", None)) is not None:
            c.value, c.shape = val, getattr(n, "shape", None)
    new._graph = new_graph

    return new


def replace_node_with_graph_module(
    model: GraphModule,
    source: Node,
    replacement: GraphModule,
    value_remap=None,
    propagate: bool = True,
) -> List[Node]:
    """Copy ``replacement`` (an exported subgraph) into ``model`` just before
    ``source``, mapping its placeholders to ``source.all_input_nodes``; return
    the output value node(s).

    ``propagate`` re-runs ``propagate_shape`` on each copied node to recover its
    shape / value.  Pass ``False`` when ``replacement`` is already
    shape-propagated and its placeholder shapes match this site (e.g. a cached
    bufferized nest): the precomputed value is carried over instead, which
    avoids re-executing the whole tiled ``while_loop`` per splice and the
    O(N^2) ``named_modules`` rescans ``propagate_shape`` does each call.
    """
    graph = model.graph
    taken = _taken_names(model)
    if value_remap is None:
        value_remap = {}

    arg_nodes = iter(source.all_input_nodes)
    output = None

    for n in list(replacement.graph.nodes):
        if n.op == "placeholder":
            value_remap[n] = next(arg_nodes, None)
            continue
        if n.op == "output":
            output = n.args[0]
            if len(output) == 1:
                source.replace_all_uses_with(value_remap[output[0]])
            else:
                for user in list(source.users):
                    assert user.target == operator.getitem
                    idx = user.args[1]
                    user.replace_all_uses_with(value_remap[output[idx]])
            continue
        with graph.inserting_before(source):
            if n.op == "get_attr":
                attr = fetch_attr(replacement, n.target)
                if isinstance(attr, GraphModule):
                    # cond / body subgraph of a while_loop: register an
                    # independent copy (shared body objects would collide when
                    # later passes stamp per-site meta on their nodes).
                    name = _fresh_name(n.target, taken)
                    setattr(model, name, copy_graph_module(attr))
                    value_remap[n] = graph.create_node("get_attr", name)
                else:
                    value_remap[n] = create_getattr_from_value(
                        model, graph, n.target, attr, n.meta.get("producer")
                    )
                    if "dtype" in n.meta:
                        # The same tensor keeps its logical dtype (a
                        # kernel's int1 mask table).
                        value_remap[n].meta["dtype"] = n.meta["dtype"]
            elif n.op == "call_module":
                # A fused compute submodule (the attention builders run
                # _fuse_passes, leaving top-level fused call_modules): register
                # an independent copy under a fresh name, like a body subgraph,
                # and retarget the copied call at it.
                sub = fetch_attr(replacement, n.target)
                name = _fresh_name(n.target, taken)
                setattr(model, name, copy_graph_module(sub))
                new = graph.node_copy(n, lambda x: value_remap[x])
                new.target = name
                value_remap[n] = new
            else:
                value_remap[n] = graph.node_copy(n, lambda n: value_remap[n])
            taken.add(value_remap[n].name)
            if propagate:
                propagate_shape(value_remap[n], model)

    return [value_remap[n] for n in output]


def create_subgraph(nodes: List[Node], results: List[Node]):
    """``nodes`` as a ``GraphModule`` returning ``results``' values, a tuple
    of them when there are several, and the values it takes as operands."""
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    for node in nodes:
        for n in node.all_input_nodes:
            if n not in value_remap:
                value_remap[n] = new_graph.placeholder(n.name)
                new_args.append(n)
        value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

    if len(results) == 1:
        new_graph.output(value_remap[results[0]])
    else:
        new_graph.output(tuple(value_remap[n] for n in results))
    new_graph.lint()
    gm = torch.fx.GraphModule(torch.nn.Module(), new_graph)
    return gm, tuple(new_args)


OP_PARAM_ARG_INDEX = {
    torch.ops.aten.conv2d.default: 1,
    torch.ops.aten.linear.default: 1,
    torch.ops.quantized_ops.conv2d.default: 1,
    torch.ops.quantized_ops.conv2d_mx.default: 1,
    torch.ops.quantized_ops.linear.default: 1,
    torch.ops.quantized_ops.linear_mx.default: 1,
    **dict.fromkeys(NORM_OPS, 2),
}


def get_unique_node_name(node: Node):
    """
    Generate a unique and meaningful name for the node based on its parameter.
    """
    if (pos := OP_PARAM_ARG_INDEX.get(node.target)) is not None:
        # A norm without an affine has no weight to name it by.
        weight_node = get_arg_value(node, pos, "weight")
        # There are cases where weights are sliced, or read through the
        # transpose a weight's load fuses. Trace up to find the get_attr node
        # and use the parameter name
        while isinstance(weight_node, Node):
            if not (
                weight_node.target == torch.ops.aten.slice.Tensor
                or swaps_last_two_dims(weight_node)
            ):
                break
            weight_node = weight_node.args[0]

        if isinstance(weight_node, Node) and weight_node.op == "get_attr":
            return weight_node.name.split("_weight")[0]

    return node.name


def _taken_names(module: GraphModule) -> set:
    """Names a fresh node or submodule of ``module`` must avoid.

    These are its graph's node names plus its direct child module names. A
    fresh name never contains a dot, so deeper submodules cannot clash.
    """
    taken = {n.name for n in module.graph.nodes}
    taken.update(k for k, m in module._modules.items() if m is not None)
    return taken


def _fresh_name(prefix: str, taken: set) -> str:
    """Picks the first of ``prefix``, ``prefix_1``, ``prefix_2``, ... not taken.

    Dots in ``prefix`` become underscores first.

    Args:
        prefix: Base name, e.g. a node target.
        taken: Names already in use; updated with the result.

    Returns:
        The picked name.
    """
    prefix = prefix.replace(".", "_")
    name, i = prefix, 0
    while name in taken:
        i += 1
        name = f"{prefix}_{i}"
    taken.add(name)
    return name


def get_new_node_name_with_prefix(prefix: str):
    """
    Generate a new attribute name with a given prefix that is not already used
    in the module's graph.
    """

    def get_new_node_name(module: torch.nn.Module):
        return _fresh_name(prefix, _taken_names(module))

    return get_new_node_name


def get_submodule_name(module, nodes: List[Node]):
    prefix = "submodule"
    if is_torch_greater_or_equal("2.5"):
        anchor_node = None
        for n in nodes:
            if n.target in OP_PARAM_ARG_INDEX or is_gemm_op(n):
                anchor_node = n
                break
            if (
                n.op == "call_function"
                and not is_nop(n)
                and not is_reshape_op(n)
                and (
                    anchor_node is None
                    or anchor_node.target
                    == torch.ops.quantized_ops.dequantize.default
                )
            ):
                anchor_node = n
        if anchor_node is not None:
            prefix = get_unique_node_name(anchor_node)
            if len(nodes) > 1:
                prefix += "_fused"

    get_new_node_name = get_new_node_name_with_prefix(prefix)
    return get_new_node_name(module)


def rename_nodes_with_param_names(model: GraphModule):
    if not is_torch_greater_or_equal("2.5"):
        return
    graph = model.graph
    for node in list(graph.nodes):
        if node.target in OP_PARAM_ARG_INDEX:
            node.name = get_submodule_name(model, [node])
    graph.lint()
    model.recompile()


def create_and_insert_subgraph(
    nodes: List[Node],
    model: torch.nn.Module,
    node_order: Dict[Node, int] = None,
    normalize_iteration_space: bool = False,
    results: Optional[List[Node]] = None,
) -> Optional[Node]:
    """Outline ``nodes`` into a submodule of ``model`` called where the last
    of them stands.

    Args:
        nodes: The nodes to outline; sorted in place into program order.
        model: The module whose graph holds them.
        node_order: Each node's position in the graph; computed if ``None``.
        normalize_iteration_space: Whether to normalize the group to its
            anchor's iteration space, leaving it unfused if it cannot.
        results: The nodes whose values the call returns, each read through
            a ``getitem`` when there are several; ``None`` for the last
            node's alone.

    Returns:
        The ``call_module`` node, or ``None`` if normalization failed.
    """
    if node_order is None:
        node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    nodes.sort(key=lambda n: node_order[n])
    if results is None:
        results = [nodes[-1]]
    submodule, new_args = create_subgraph(nodes, results)
    node_name = get_submodule_name(model, nodes)
    setattr(model, node_name, submodule)
    with model.graph.inserting_after(nodes[-1]):
        new_node = model.graph.create_node(
            "call_module", node_name, new_args, {}
        )
    new_node.meta["submodule"] = submodule
    items = []
    if len(results) == 1:
        if (dtype := results[0].meta.get("dtype", None)) is not None:
            new_node.meta["dtype"] = dtype
        results[0].replace_all_uses_with(new_node, lambda u: u not in nodes)
    else:
        for index, node in enumerate(results):
            with model.graph.inserting_after(items[-1] if items else new_node):
                items.append(
                    model.graph.call_function(
                        operator.getitem, (new_node, index)
                    )
                )
            node.replace_all_uses_with(items[-1], lambda u: u not in nodes)

    # An operand is cloned so propagation cannot write through it, but a group
    # may also take a scalar -- an index or a running offset read out of a
    # buffer, which the datapath walk stops at -- and those are passed as is.
    args = map_arg(
        new_node.args,
        lambda n: (
            n.value.clone() if isinstance(n.value, torch.Tensor) else n.value
        ),
    )
    result = ShapeProp(submodule).propagate(*args)
    set_node_value(new_node, result)
    if items:
        for item, value in zip(items, result):
            set_node_value(item, value)

    # Normalize the fused submodule to the anchor's iteration space; leave
    # the group unfused if it cannot share one iteration space.
    if normalize_iteration_space:
        try:
            normalize_group(model, new_node)
        except NormalizationError as exc:
            logger.warning("normalization failed: %s: %s", new_node, exc)
            new_node.replace_all_uses_with(nodes[-1])
            delattr(model, node_name)
            model.graph.erase_node(new_node)
            return None

    for node in reversed(nodes):
        if not node.users:
            model.graph.erase_node(node)
    return new_node
