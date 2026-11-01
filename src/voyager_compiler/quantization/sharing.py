"""Match fake-quants across graphs exported from one model, and share them.

A fake-quant is known by the operand edges it guards -- the parameter it
reads, or the consumers it feeds, each named by module path, op and call --
not by its name: ``prepare_pt2e`` numbers fake-quants in annotation order,
which differs between an inference graph and a training joint graph.  A
decode graph whose KV caches ``split_kv_cache`` split names its edges as
the unsplit graph would (``WHOLE_CACHE_OP``).
"""

from collections import Counter
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import nn
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    ParamAOTInput,
)
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e import FakeQuantizeBase

from voyager_compiler.export_utils import WHOLE_CACHE_OP, get_module_stack
from voyager_compiler.quantization.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.quantization.quantize_pt2e import _is_written_buffer
from voyager_compiler.quantization.training import _VIEWS, _fake_quantize
from voyager_compiler.shape_prop import fetch_attr

# Attributes of a fake-quant that do not set what it computes.
_FLAGS = ("training", "fake_quant_enabled", "observer_enabled")


def _module_path(node: Node, prefix: str) -> str:
    """The innermost module path of ``node``, with ``prefix`` stripped.

    Args:
        node: Node of an exported or joint graph.
        prefix: Module path of the exported model within the exported
            object; ``""`` when the model is the exported object.

    Returns:
        The path relative to the model; ``""`` for a node of the model's
        own forward or of no module.
    """
    stack = get_module_stack(node)
    path = list(stack.values())[-1][0] if stack else ""
    if path == prefix:
        return ""
    if prefix and path.startswith(prefix + "."):
        return path[len(prefix) + 1 :]
    return path


def _call_indices(
    graph: GraphModule, prefix: str
) -> Dict[Node, Tuple[object, int]]:
    """Each forward node -> the op it counts as and how many earlier forward
    nodes count as the same op from the same module.

    A node counts as its target, or as the op ``meta[WHOLE_CACHE_OP]``
    names: on a split KV cache's decode graph, the op the node stands for
    in the unsplit graph.  A node the split added (None there) is not
    counted.

    Args:
        graph: Exported or joint graph.
        prefix: Passed to ``_module_path``.

    Returns:
        ``(op, index)`` of every node counted, which leaves out a joint
        graph's backward nodes.
    """
    counts = Counter()
    indices = {}
    for node in graph.graph.nodes:
        op = node.meta.get(WHOLE_CACHE_OP, node.target)
        if node.meta.get("partitioner_tag") == "is_backward" or op is None:
            continue
        key = (_module_path(node, prefix), op)
        indices[node] = (op, counts[key])
        counts[key] += 1
    return indices


def _consumers(node: Node) -> Iterator[Tuple[Node, int]]:
    """Yield ``(consumer, argument index)`` for every read of ``node``, the
    reads of its views counted as reads of it."""
    for user in node.users:
        if user.target in _VIEWS:
            yield from _consumers(user)
        else:
            for index, arg in enumerate(user.args):
                if arg is node:
                    yield user, index


def _stored_tensor(
    source: Node, graph: GraphModule, root: Optional[nn.Module]
) -> Optional[torch.Tensor]:
    """The parameter or buffer ``source`` reads, if it reads one the graph
    does not write.

    Args:
        source: The node a fake-quant reads, past its views.
        graph: The graph, which holds a ``get_attr``'s tensor.
        root: The module whose tensors a joint graph's placeholders name;
            None for a graph that holds its own.

    Returns:
        The tensor, or None when ``source`` is a computed value, a model
        input or a buffer the graph writes, such as a split KV cache's main
        half.
    """
    if source.op == "get_attr":
        if _is_written_buffer(graph, source):
            return None
        return fetch_attr(graph, source.target)
    desc = source.meta.get("desc")
    if isinstance(desc, (ParamAOTInput, BufferAOTInput)):
        return fetch_attr(root, desc.target)
    return None


def _fake_quant_calls(graph: GraphModule) -> Iterator[Tuple[str, Node, Node]]:
    """Yield ``(name, call, operand)`` for each fake-quant call of ``graph``:
    a ``call_module`` in a prepared graph, a ``_fake_quantize`` call in a
    joint graph.  A derived observer, whose scale follows the fake-quants
    it derives from, is left out."""
    for node in graph.graph.nodes:
        if node.op == "call_module":
            name, operand = node.target, node.args[0]
        elif node.op == "call_function" and node.target is _fake_quantize:
            name, operand = node.args[0].target, node.args[1]
        else:
            continue
        module = graph.get_submodule(name)
        if isinstance(module, FakeQuantizeBase) and not isinstance(
            module, _DerivedObserverOrFakeQuantize
        ):
            yield name, node, operand


def fake_quant_edges(
    graph: GraphModule, prefix: str, root: Optional[nn.Module] = None
) -> Dict[str, FrozenSet[tuple]]:
    """Each fake-quant of ``graph``, by the name it is held under, -> the
    set of operand edges it guards.

    A fake-quant on a parameter, or on a buffer the graph does not write,
    guards ``("weight", data_ptr, shape)`` of the tensor.  One on the
    output of an op annotated with an output qspec guards ``("out", path,
    op, call_index)`` of that op.  Any other guards ``("act", path, op,
    call_index, argument index)`` for each read of its output, through
    views, by a counted forward op.  ``path`` is the op's innermost module
    path relative to the model, and ``op`` and ``call_index`` are what
    ``_call_indices`` counts, so the edges name the same operand in every
    export of the model.  A gradient fake-quant, read by backward ops
    alone, guards no edge.

    Args:
        graph: Graph ``prepare_pt2e`` prepared, or the joint graph
            ``prepare_training`` captured.
        prefix: Module path of the model within the exported object, such
            as ``"model"`` for a wrapper holding it as ``self.model``; ``""``
            when the model was exported itself.
        root: For a joint graph, the module ``prepare_training`` prepared,
            whose parameters the graph's placeholders name; None for a
            graph holding its own parameters.

    Returns:
        ``name -> edges``.
    """
    indices = _call_indices(graph, prefix)
    edges = {}
    for name, call, operand in _fake_quant_calls(graph):
        source = operand
        while source.target in _VIEWS:
            source = source.args[0]
        tensor = _stored_tensor(source, graph, root)
        annotation = operand.meta.get("quantization_annotation")
        if tensor is not None:
            edges[name] = frozenset(
                {("weight", tensor.data_ptr(), tuple(tensor.shape))}
            )
        elif annotation is not None and annotation.output_qspec is not None:
            path = _module_path(operand, prefix)
            edges[name] = frozenset({("out", path, *indices[operand])})
        else:
            edges[name] = frozenset(
                (
                    "act",
                    _module_path(consumer, prefix),
                    *indices[consumer],
                    index,
                )
                for consumer, index in _consumers(call)
                if consumer in indices
            )
    return edges


def _configuration(fake_quant: nn.Module) -> Tuple[type, dict]:
    """What ``fake_quant`` computes: its class and the attributes that set
    it, which are all of ``vars`` but the tensors, the enable flags and the
    module internals."""
    settings = {
        name: value
        for name, value in vars(fake_quant).items()
        if not name.startswith("_")
        and name not in _FLAGS
        and not isinstance(value, torch.Tensor)
    }
    return type(fake_quant), settings


def share_fake_quants(
    graphs: Sequence[GraphModule],
    prefixes: Sequence[str],
    holders: Optional[Mapping[GraphModule, Sequence[nn.Module]]] = None,
    roots: Optional[Mapping[GraphModule, nn.Module]] = None,
) -> List[Dict[str, str]]:
    """Make the fake-quants that guard one operand one object across graphs.

    The first graph's fake-quants are kept.  In each later graph, a
    fake-quant whose edges (``fake_quant_edges``) equal a kept one's and
    whose configuration equals it is replaced by the kept one, under its
    own name, on the graph and on every holder listed for the graph.  One
    whose configuration differs, or whose edges match none, stays as it is.

    Args:
        graphs: Graphs exported from one model, each prepared.
        prefixes: Each graph's model path, as ``fake_quant_edges`` takes.
        holders: Other modules holding a graph's fake-quants under the same
            names -- for a training joint graph, its forward and backward
            graphs and ``model.training_quantizers``; None for none.
        roots: Each joint graph's prepared module, as ``fake_quant_edges``
            takes; None when no graph is a joint graph.

    Returns:
        For each graph after the first, ``name -> "shared"``,
        ``"mismatched"`` or ``"unmatched"``.
    """
    holders = holders or {}
    roots = roots or {}
    first = graphs[0]
    kept = {
        edges: first.get_submodule(name)
        for name, edges in fake_quant_edges(
            first, prefixes[0], roots.get(first)
        ).items()
        if edges
    }
    reports = []
    for graph, prefix in zip(graphs[1:], prefixes[1:]):
        report = {}
        replaced = {}
        for name, edges in fake_quant_edges(
            graph, prefix, roots.get(graph)
        ).items():
            own = graph.get_submodule(name)
            match = kept.get(edges)
            if match is None:
                report[name] = "unmatched"
            elif _configuration(match) != _configuration(own):
                report[name] = "mismatched"
            else:
                report[name] = "shared"
                replaced[id(own)] = match
                for holder in (graph, *holders.get(graph, ())):
                    if getattr(holder, name, None) is own:
                        setattr(holder, name, match)
        # A derived fake-quant, such as a bias's, follows the fake-quants
        # it derives from: the shared ones from here on.
        for holder in (graph, *holders.get(graph, ())):
            for module in holder.modules():
                if isinstance(module, _DerivedObserverOrFakeQuantize):
                    module.obs_or_fqs = [
                        replaced.get(id(m), m) for m in module.obs_or_fqs
                    ]
        reports.append(report)
    return reports
