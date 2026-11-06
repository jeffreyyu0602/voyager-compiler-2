"""The sync pass: every dispatch a ``commit``, every dependency a counter
wait.

Runs after memory planning, on each layer the builders emitted -- its tile
loop(s) with their prologue and epilogue, every op inline and every copy
bare -- and leaves the graph the program the hardware runs: each compute op
wrapped in a ``commit`` whose waits, grouped by unit, the pass derived, and
whose last pass adds 1 to that unit's counter; each copy adding 1 to its
stream's counter; a CPU ``async_wait`` before every copy, scalar read, host
op or zero fill that depends on a dispatch or a copy; every counter waited
at its total when the layer ends.

A producer is a dispatch, posting on the counter of its last pass's unit,
or a copy, posting on its stream's: the loads into one on-chip buffer, or
the stores out of one.  Posts on one counter complete in issue order.  The
pass identifies each wait's producer; the counts come from the program
itself.  Per counter it keeps the number of posts issued so far, and per
buffer slot and producer (a unit or a stream) the count at which the last
one touching it (reading or writing) and the last one writing it posted:
scalars the CPU updates beside each dispatch and copy, carried through the
loops and out of the conds.  Every access, by a dispatch or a copy, waits
``counter >=`` the saved count of every producer it depends on: the last
writer of what it reads and the last toucher of what it overwrites, on
every unit, its own included (passes of one unit may overlap), and on every
stream but its own.  A dispatch takes them as hardware waits in its commit
(an SpMM dispatch's matrix-class waits gate the systolic array and the SpMM
unit alike); a copy, a scalar read and a host op as CPU waits right before
it.  A wait the CPU has already made where the consumer runs is not made
again.
"""

import logging
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.fx import Graph, GraphModule, Node

from voyager_compiler.codegen.node_info import (
    is_gemm_op,
    is_nop,
    runs_on_vector_unit,
)
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _is_compute,
    _viewed_buffer,
)
from voyager_compiler.codegen.transform.bufferize.memory_planning import (
    _slots,
    plan_counters,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    UNITS,
    MemoryLevel,
    commit_hop,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _placeholders,
    _subgraph,
    voyager,
)
from voyager_compiler.shape_prop import set_node_value

logger = logging.getLogger(__name__)

WHILE_LOOP = torch.ops.higher_order.while_loop
COND = torch.ops.higher_order.cond
WITH_SCRATCH = torch.ops.higher_order.with_scratch
_ASYNC_COPY = voyager.async_copy.default
_ASYNC_WAIT = voyager.async_wait.default
_INSERT = voyager.insert.default
_ALLOC = voyager.alloc.default
_ZEROS = voyager.zeros.default
_SUBVIEW = voyager.subview.default
_SCALAR_READ = torch.ops.aten._local_scalar_dense.default
_SPMM = torch.ops.quantized_ops.spmm_csr.default
_REGISTER = int(MemoryLevel.REGISTER)

MATRIX, VECTOR = UNITS
# The unit counters' names in the IR.
_COUNTER_NAMES = {MATRIX: "mu", VECTOR: "vu"}
# Waits one unit's pass can queue.
_WAITS_PER_UNIT = 16
# The state keys: a counter's post count, and per buffer slot, accessor (a
# pass unit or a stream) and posting counter the count at which the last
# access reading-or-writing the slot, and the last writing it, posted on
# that counter (a dispatch's completion is seen only on the counter it
# posts).
_POSTS, _TOUCH, _WRITE = "posts", "touch", "write"


class SyncError(ValueError):
    """A layer the pass cannot synchronize; the message names the layer."""


@dataclass
class _Region:
    """One graph a layer runs in: a loop body or a cond branch.  ``binding``
    maps its placeholders to the values of the enclosing region (the layer's
    own graph for a top-level loop body) and ``hop`` is the node that calls
    it there."""

    gm: GraphModule
    parent: Optional["_Region"]
    hop: Node
    binding: Dict[Node, object]


@dataclass
class _Pass:
    """One pass of a dispatch: its unit (for a copy, its stream), and the
    on-chip buffers it reads and writes, in operand order, each with the
    slot it accesses (an int, a ``(node, region)`` or ``None`` for every
    slot)."""

    unit: object
    reads: List[tuple] = field(default_factory=list)
    writes: List[tuple] = field(default_factory=list)

    def buffers(self, accesses):
        return [b for b, _ in accesses]


@dataclass
class _Dispatch:
    """A dispatchable op: the node, the region it runs in (``None``: the
    layer's graph), the ``insert``s of its results, its passes, the unit of
    its last pass, which posts, and whether its matrix-class pass runs on
    the SpMM unit."""

    node: Node
    region: Optional[_Region]
    inserts: List[Node]
    passes: List[_Pass] = field(default_factory=list)
    post_unit: str = VECTOR
    spmm: bool = False


@dataclass(frozen=True)
class _Stream:
    """A copy stream, one counter: the loads into on-chip ``buffer``
    (``load``), or the stores out of it."""

    buffer: Node
    load: bool


@dataclass
class _Copy:
    """A ``voyager.async_copy``: its node and region (``None`` outside every
    loop), its stream, and its on-chip accesses as the one pass of a
    producer whose unit is the stream."""

    node: Node
    region: Optional[_Region]
    stream: _Stream
    access: _Pass


class _Names:
    """Model-wide unique names for the nodes the pass adds: ``prefix_n``
    with the lowest free ``n`` per prefix, so a name means one thing in
    ``model.txt`` whichever region it is in."""

    def __init__(self, model: GraphModule):
        self.used = set()
        for sub in model.modules():
            if isinstance(sub, GraphModule):
                self.used.update(n.name for n in sub.graph.nodes)
        self.next = {}
        # Which layer first took each region graph (a graph two layers
        # reach is a sharing bug upstream, reported by name).
        self.owners = {}

    def __call__(self, prefix: str) -> str:
        i = self.next.get(prefix, 0)
        while (name := f"{prefix}_{i}") in self.used:
            i += 1
        self.next[prefix] = i + 1
        self.used.add(name)
        return name


def synchronize(model: GraphModule) -> int:
    """Synchronize every layer of ``model`` and assign the counter ids.

    Args:
        model: The bufferized, memory-planned graph, rewritten in place.

    Returns:
        How many counter ids the model uses.

    Raises:
        SyncError: A layer whose dispatches the pass cannot read (a result
            leaving one other than through a store, a wait on a unit a
            dispatch has no pass on, more waits than a unit queues), a
            builder's synchronization (a wait, a copy's counter), a copy in
            no layer, or more counters live at once than ``plan_counters``
            has ids.
    """
    names = _Names(model)
    order = {}
    for sub in model.modules():
        if isinstance(sub, GraphModule):
            order.update((n, i) for i, n in enumerate(sub.graph.nodes))
    # The model and every top-level cond branch, each once: two conds may
    # share a branch module.
    graphs = [model]
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target is COND:
            for handle in node.args[1:3]:
                branch = _subgraph(model, handle.target)
                if branch is not None and branch not in graphs:
                    graphs.append(branch)
    layers = 0
    regions, synced = set(), []
    for gm in graphs:
        for name, nodes in _layers(gm):
            # A layer with no loop, dispatch or copy of its own -- a host
            # op, or a cond whose branches hold layers of their own (walked
            # as graphs above) -- has nothing to synchronize.
            if not any(
                n.op == "call_function"
                and n.target in (WHILE_LOOP, _ASYNC_COPY)
                or _is_dispatch(n)
                for n in nodes
            ):
                continue
            layer = _Layer(gm, nodes, name, names, order)
            layer.synchronize()
            regions |= layer.touched
            if gm not in synced:
                synced.append(gm)
            layers += 1
    # Every graph the layers edited is recompiled once, after them all: a
    # recompile regenerates the module's whole subtree, so one per layer
    # would cost the model's size once per layer.
    for gm in [*regions, *synced]:
        gm.recompile()
    bare = [
        n.name
        for sub in model.modules()
        if isinstance(sub, GraphModule)
        for n in sub.graph.nodes
        if n.op == "call_function"
        and n.target is _ASYNC_COPY
        and _counter_arg(n) is None
    ]
    if bare:
        raise SyncError(f"copies in no layer, with no counter: {bare}")
    counters = plan_counters(model)
    # The program size: the dispatches the program issues, each a commit
    # in the IR (the emitter enforces the chip's program region).
    commits = sum(
        1
        for sub in model.modules()
        if isinstance(sub, GraphModule)
        for n in sub.graph.nodes
        if n.op == "call_function" and n.target is commit_hop
    )
    logger.info(
        "Sync: %d dispatch sites in %d layers, %d counters",
        commits,
        layers,
        counters,
    )
    return counters


def _layers(gm: GraphModule):
    """The layers of ``gm`` as ``(name, nodes)``: each contiguous run of
    top-level nodes sharing a ``meta['scope']``.  A node with no scope -- a
    parameter, a host op -- belongs to no layer."""
    runs, current = [], None
    for node in gm.graph.nodes:
        scope = node.meta.get("scope")
        if scope is None:
            current = None
            continue
        if current is None or current[0] != scope[0]:
            current = (scope[0], [])
            runs.append(current)
        current[1].append(node)
    return runs


def _counter_arg(copy: Node):
    """The counter operand of an ``async_copy`` (``None``: none yet)."""
    if len(copy.args) > 4:
        return copy.args[4]
    return copy.kwargs.get("counter")


def _is_host_op(node: Node) -> bool:
    """An op the SoC emitter runs on the CPU instead of dispatching it (the
    runtime's ``is_host_bookkeeping``): a ``clone``, or an ``add`` of an
    int32 or int64 tensor."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops.aten.clone.default:
        return True
    if node.target not in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
    ):
        return False
    arg = node.args[0]
    value = getattr(arg, "value", None)
    if value is None and isinstance(arg, Node):
        value = arg.meta.get("val")
    return isinstance(value, torch.Tensor) and value.dtype in (
        torch.int32,
        torch.int64,
    )


def _is_dispatch(node: Node) -> bool:
    """A dispatchable op: a compute op (a fused module, a matrix op, a
    vector op) or a ``with_scratch`` region around one, but not a host op
    (``_is_host_op``)."""
    if node.op == "call_function" and node.target is WITH_SCRATCH:
        return True
    if node.op not in ("call_module", "call_function") or is_nop(node):
        return False
    return _is_compute(node) and not _is_host_op(node)


def _is_matrix_op(node: Node) -> bool:
    """An op that runs as a matrix-class pass: an SpMM, or a GEMM the
    toolchain does not map wholly onto the vector unit (the mapper's rule,
    ``runs_on_vector_unit``)."""
    if node.op != "call_function":
        return False
    if node.target is _SPMM:
        return True
    return is_gemm_op(node) and not runs_on_vector_unit(node)


class _Scalars:
    """Scalar expressions in one graph, folded when constant and shared
    when repeated, each new node inserted before ``anchor``.  ``memo`` is
    the graph's: the pass walks a graph in program order, so a node made
    before an earlier anchor is defined at every later one."""

    def __init__(self, names: _Names, graph: Graph, anchor: Node, memo):
        self.names = names
        self.graph = graph
        self.anchor = anchor
        self.memo = memo

    def _op(self, target, a, b, prefix):
        if isinstance(a, int) and isinstance(b, int):
            return target(a, b)
        key = (target, a, b)
        if key in self.memo:
            return self.memo[key]
        with self.graph.inserting_before(self.anchor):
            node = self.graph.call_function(
                target, (a, b), name=self.names(prefix)
            )
        self.memo[key] = node
        return node

    def add(self, a, b):
        if b == 0:
            return a
        if a == 0:
            return b
        return self._op(operator.add, a, b, "add")

    def eq(self, a, b):
        return self._op(operator.eq, a, b, "eq")

    def ite(self, pred, a, b):
        if isinstance(pred, bool):
            return a if pred else b
        if a == b:
            return a
        key = (torch.sym_ite, pred, a, b)
        if key in self.memo:
            return self.memo[key]
        with self.graph.inserting_before(self.anchor):
            node = self.graph.call_function(
                torch.sym_ite, (pred, a, b), name=self.names("sym_ite")
            )
        self.memo[key] = node
        return node


class _Layer:
    """One layer and the pass's rewrite of it (module docstring): how its
    graphs are read (buffers, slots, dispatches and their passes, copies and
    their streams) and rewritten (values threaded into regions, counters,
    carried counts, commits, waits)."""

    def __init__(self, gm, nodes, name, names, order):
        self.gm = gm
        self.nodes = nodes
        self.name = name
        self.names = names
        self.order = order
        self.touched = set()
        self.regions: Dict[GraphModule, _Region] = {}
        self.threaded = {}
        # The counter of each unit and stream, in allocation order.
        self.counters: Dict[object, Node] = {}
        # Each on-chip buffer's streams.
        self.streams: Dict[Node, List[_Stream]] = {}
        # Per graph the CPU waits made in it, as resolved ``(counter,
        # threshold)``: they hold for whatever runs after them there, and
        # in the regions nested there.
        self.cpu_waits: Dict[GraphModule, set] = {}
        # Per graph the scalar expressions the pass made in it (``_Scalars``).
        self.memos: Dict[Graph, dict] = {}
        # What marks a node as the layer's: its scope, and the build the
        # layer came from (``print_layer_table`` groups layers by it).
        self.tags = {
            key: nodes[0].meta[key]
            for key in ("scope", "build_group", "build_key")
            if key in nodes[0].meta
        }
        self.scope = self.tags.get("scope")
        self.sites: Dict[Node, _Dispatch] = {}
        self.copies: Dict[Node, _Copy] = {}
        self.slots: Dict[Node, int] = {}
        self.modified: Dict[Node, set] = {}
        self.modified[None] = self._explore(nodes, None)
        accessors = [d.passes for d in self.sites.values()]
        accessors += [[c.access] for c in self.copies.values()]
        self.written = {
            b
            for passes in accessors
            for p in passes
            for b in p.buffers(p.writes)
        }

    # --- reading ---------------------------------------------------------

    def _fresh(self, parent, name) -> GraphModule:
        """The region ``parent.name``, re-registered as a ``GraphModule`` of
        its own class.  Bufferization splices identical nests as copies
        sharing one class (and so one compiled ``forward``); a region the
        pass edits has to recompile alone."""
        old = parent.get_submodule(str(name))
        owner = self.names.owners.setdefault(id(old.graph), (self.name, name))
        if owner[0] != self.name:
            raise SyncError(
                f"{self.name}: region {name} shares its graph with region "
                f"{owner[1]} of layer {owner[0]}"
            )
        missing = [
            n.target
            for n in old.graph.nodes
            if n.op in ("get_attr", "call_module")
            and not hasattr(old, str(n.target))
        ]
        if missing:
            raise SyncError(
                f"{self.name}: region {name} names {missing}, which it does "
                f"not hold (its graph is owned by "
                f"{type(old.graph.owning_module).__name__}, shared: "
                f"{old.graph.owning_module is not old})"
            )
        fresh = GraphModule(old, old.graph, str(name))
        setattr(parent, str(name), fresh)
        self.touched.add(fresh)
        return fresh

    def _resolve(self, node, region):
        """``node`` of ``region`` as the value it is bound to, up through
        the regions' placeholders: ``(node, region)``, ``region`` ``None``
        at the layer's own graph."""
        while (
            region is not None
            and isinstance(node, Node)
            and node.op == "placeholder"
        ):
            if node not in region.binding:
                return node, region
            node = region.binding[node]
            region = region.parent
        return node, region

    def _buffer(self, node, region) -> Optional[Node]:
        """The buffer ``node`` of ``region`` names, in the layer's graph,
        through views and region placeholders; ``None`` for a scalar."""
        while True:
            node, region = self._resolve(node, region)
            if not isinstance(node, Node):
                return None
            viewed = (
                _viewed_buffer(node) if node.op == "call_function" else None
            )
            if viewed is None:
                return node
            node = viewed

    def _on_chip(self, node, region) -> Optional[Node]:
        """The scratchpad buffer ``node`` names, or ``None``."""
        buffer = self._buffer(node, region)
        if buffer is None or buffer.meta.get("space") != "Scratchpad":
            return None
        return buffer

    def _access(self, node, region):
        """``(buffer, slot)`` for an on-chip operand: the slot an int, a
        ``(node, region)`` naming a runtime slot index, or ``None`` for the
        whole buffer; ``None`` when the operand is not on chip."""
        buffer = self._on_chip(node, region)
        if buffer is None:
            return None
        if _slots(buffer) <= 1:
            return buffer, 0
        rank = len(buffer.value.shape)
        while True:
            node, region = self._resolve(node, region)
            if not isinstance(node, Node) or node.op != "call_function":
                return buffer, None
            if (
                node.target is _SUBVIEW
                and len(node.args[1]) == rank
                and list(node.args[4] if len(node.args) > 4 else []) == [0]
            ):
                slot = node.args[1][0]
                return buffer, (
                    slot if isinstance(slot, int) else (slot, region)
                )
            viewed = _viewed_buffer(node)
            if viewed is None:
                return buffer, None
            node = viewed

    def _inserts(self, node) -> List[Node]:
        """The ``insert``s storing ``node``'s results, through ``getitem``
        and view wrappers."""
        inserts, stack = [], [node]
        while stack:
            n = stack.pop()
            for user in n.users:
                if user.target is _INSERT and user.args[0] is n:
                    inserts.append(user)
                elif user.target is operator.getitem or is_nop(user):
                    stack.append(user)
                else:
                    raise SyncError(
                        f"{self.name}: {user.name} reads {node.name}'s value "
                        f"without a store; a dispatch's result leaves it "
                        f"through voyager.insert"
                    )
        return sorted(inserts, key=self.order.__getitem__)

    def _module_ops(self, sub, operands):
        """The ops of a fused module or a ``with_scratch`` region, each
        with the operands of the enclosing region it reads, nested modules
        flattened; a value computed inside the module is no operand, and a
        view of an operand (a reshape, a same-dtype cast) is that operand."""
        bound = dict(zip(_placeholders(sub), operands))

        def outer(n):
            return bound.get(n) if isinstance(n, Node) else None

        ops = []
        for n in sub.graph.nodes:
            if n.op == "call_module":
                inner = sub.get_submodule(n.target)
                ops += self._module_ops(inner, [outer(a) for a in n.args])
            elif n.op == "call_function" and is_nop(n):
                source = (
                    outer(n.all_input_nodes[0]) if n.all_input_nodes else None
                )
                if source is not None:
                    bound[n] = source
            elif n.op == "call_function":
                ops.append((n, [o for o in map(outer, n.all_input_nodes) if o]))
        return ops

    def _dispatch(self, node, region) -> _Dispatch:
        """``node`` as a dispatch: its stores, and its passes -- a matrix op
        is a matrix-class pass reading its own operands, every other op a
        vector pass, the results written by the pass of the last op, or by
        the vector pass after an SpMM (the vector unit stores its output)."""
        d = _Dispatch(node, region, self._inserts(node))
        if node.op == "call_module":
            sub = region.gm.get_submodule(node.target) if region else None
            sub = sub or self.gm.get_submodule(node.target)
            ops = self._module_ops(sub, list(node.args))
        elif node.target is WITH_SCRATCH:
            owner = region.gm if region else self.gm
            sub = _subgraph(owner, node.args[0].target)
            ops = self._module_ops(sub, list(node.args[1:]))
        else:
            ops = [(node, list(node.all_input_nodes))]
        matrix = next((op for op, _ in ops if _is_matrix_op(op)), None)
        # An SpMM dispatch is an spmm_csr, or a GEMM that fuses one (its CSR
        # operands, the emitter's has_fused_spmm): both run on the SpMM unit.
        d.spmm = matrix is not None and (
            matrix.target is _SPMM or matrix.kwargs.get("A_data") is not None
        )
        tail = matrix is not None and any(
            op is not matrix and self.order[op] > self.order[matrix]
            for op, _ in ops
        )
        passes: Dict[str, _Pass] = {}

        def read(unit, operands):
            p = passes.setdefault(unit, _Pass(unit))
            for operand in operands:
                access = self._access(operand, region)
                if access is not None and access not in p.reads:
                    p.reads.append(access)

        for op, operands in ops:
            read(MATRIX if op is matrix else VECTOR, operands)
        direct = matrix is not None and not tail and not d.spmm
        writer = MATRIX if direct else VECTOR
        p = passes.setdefault(writer, _Pass(writer))
        if node.target is WITH_SCRATCH:
            # The op's regions are its own, read and written by its unit.
            scratch = list(node.kwargs["scratch"].values())
            read(writer, scratch)
            p.writes += [self._access(r, region) for r in scratch]
        for insert in d.inserts:
            access = self._access(insert.args[1], region)
            if access is None:
                raise SyncError(
                    f"{self.name}: {node.name} writes {insert.args[1].name}, "
                    f"which is not on chip"
                )
            if access not in p.writes:
                p.writes.append(access)
        d.passes = [passes[u] for u in UNITS if u in passes]
        d.post_unit = writer
        return d

    def _copy(self, node, region) -> _Copy:
        """A copy and its stream: the loads into the on-chip buffer it
        fills, else the stores out of the one it drains.

        Raises:
            SyncError: The copy already has a counter, or moves between two
                off-chip buffers.
        """
        if _counter_arg(node) is not None:
            raise SyncError(
                f"{self.name}: {node.name} has a counter; the builders "
                f"issue copies bare"
            )
        src = self._access(node.args[0], region)
        dst = self._access(node.args[1], region)
        if dst is None and src is None:
            raise SyncError(
                f"{self.name}: {node.name} copies between two off-chip "
                f"buffers"
            )
        stream = _Stream(dst[0], True) if dst else _Stream(src[0], False)
        streams = self.streams.setdefault(stream.buffer, [])
        if stream not in streams:
            streams.append(stream)
        access = _Pass(stream, [src] if src else [], [dst] if dst else [])
        return _Copy(node, region, stream, access)

    # --- rewriting -------------------------------------------------------

    def _thread(self, value, region: Optional[_Region]):
        """``value`` -- a node of the layer's graph or of a region above
        ``region``, or an int -- as a value of ``region``, passed in through
        the loops' extra operands and the conds' operand lists on the way."""
        if (
            region is None
            or not isinstance(value, Node)
            or value.graph is region.gm.graph
        ):
            return value
        key = (region.gm, value)
        if key in self.threaded:
            return self.threaded[key]
        outer = self._thread(value, region.parent)
        hop = region.hop
        operands = list(hop.args[3]) if len(hop.args) > 3 else []
        hop.args = (*hop.args[:3], tuple(operands + [outer]))
        owner = region.parent.gm if region.parent is not None else self.gm
        if hop.target is WHILE_LOOP:
            graphs = [_subgraph(owner, hop.args[0].target), region.gm]
        else:
            graphs = [_subgraph(owner, h.target) for h in hop.args[1:3]]
        result = None
        for gm in graphs:
            last = _placeholders(gm)[-1]
            with gm.graph.inserting_after(last):
                ph = gm.graph.placeholder(value.name)
            for key_ in ("val", "space", "dtype"):
                if key_ in value.meta:
                    ph.meta[key_] = value.meta[key_]
            if hasattr(value, "value"):
                set_node_value(ph, value.value)
            self.touched.add(gm)
            if gm in self.regions:
                self.regions[gm].binding[ph] = outer
                self.threaded[(gm, value)] = ph
            if gm is region.gm:
                result = ph
        return result

    def _counter(self, agent) -> Node:
        """The layer's counter for ``agent``, a unit or a stream, allocated
        at the register level before the layer's first node on first use."""
        if agent not in self.counters:
            if isinstance(agent, _Stream):
                prefix = "ld" if agent.load else "st"
            else:
                prefix = _COUNTER_NAMES[agent]
            graph = self.gm.graph
            with graph.inserting_before(self.nodes[0]):
                counter = graph.call_function(
                    _ALLOC,
                    ([], torch.int64, _REGISTER),
                    name=self.names(prefix),
                )
                counter.meta["space"] = "Register"
                counter.meta.update(self.tags)
                set_node_value(counter, torch.zeros((), dtype=torch.int64))
                reset = graph.call_function(
                    voyager.zeros.default,
                    (counter,),
                    name=self.names("zeros"),
                )
                reset.meta.update(self.tags)
            self.counters[agent] = counter
        return self.counters[agent]

    def _wrap(self, d: _Dispatch, waits, post) -> None:
        """Replace ``d``'s op and its stores with one ``commit`` over a
        subgraph holding them, waiting ``waits`` per unit and posting
        ``post``.

        Raises:
            SyncError: A wait is on a unit ``d`` runs no pass on.
        """
        units = {p.unit for p in d.passes}
        stray = sorted(set(waits) - units)
        if stray:
            raise SyncError(
                f"{self.name}: {d.node.name} waits on the {stray} unit, but "
                f"its dispatch runs passes on {sorted(units)} only"
            )
        owner = d.region.gm if d.region is not None else self.gm
        graph = owner.graph
        members = {d.node, *d.inserts}
        stack = list(d.inserts)
        while stack:
            n = stack.pop()
            for a in n.all_input_nodes:
                if (
                    a not in members
                    and (a.target is operator.getitem or is_nop(a))
                    and a.args[0] in members
                ):
                    members.add(a)
                    stack.append(a)
        nodes = sorted(members, key=self.order.__getitem__)
        root = torch.nn.Module()
        sub = Graph()
        remap, operands = {}, []
        for n in nodes:
            for a in n.all_input_nodes:
                if a in remap:
                    continue
                if a.op == "get_attr":
                    remap[a] = sub.node_copy(a)
                    setattr(root, str(a.target), getattr(owner, a.target))
                    continue
                ph = sub.placeholder(a.name)
                for key in ("val", "space", "dtype"):
                    if key in a.meta:
                        ph.meta[key] = a.meta[key]
                if hasattr(a, "value"):
                    set_node_value(ph, a.value)
                remap[a] = ph
                operands.append(a)
            remap[n] = sub.node_copy(n, lambda x: remap[x])
            if hasattr(n, "value"):
                set_node_value(remap[n], n.value)
            if n.op == "call_module":
                setattr(root, n.target, owner.get_submodule(n.target))
        sub.output(())
        sub.lint()
        name = self.names("commit_body")
        setattr(owner, name, GraphModule(root, sub, name))
        # At the last store's position: every operand, the stores'
        # destination views included, is defined by then.
        with graph.inserting_before(nodes[-1]):
            body = graph.get_attr(name)
            commit = graph.call_function(
                commit_hop,
                (body, *operands),
                {"dependencies": waits, "post": post},
                name=self.names("commit"),
            )
        commit.meta["scope"] = d.node.meta.get("scope")
        for n in reversed(nodes):
            graph.erase_node(n)
        self.touched.add(owner)

    def _origin(self, value, region):
        """``value`` of ``region`` as the value it is bound to outside every
        region it was threaded into."""
        return self._resolve(value, region)[0]

    def _cpu_waited(self, counter, threshold, region) -> bool:
        """Whether the CPU has already waited ``counter >= threshold`` where
        a node of ``region`` runs: earlier in its graph or in a graph
        enclosing it."""
        key = (self._origin(counter, region), self._origin(threshold, region))
        while True:
            gm = region.gm if region is not None else self.gm
            if key in self.cpu_waits.get(gm, ()):
                return True
            if region is None:
                return False
            region = region.parent

    def _wait(self, anchor, counter, threshold, region) -> None:
        """A CPU ``async_wait(counter, threshold)`` before ``anchor`` of
        ``region``, unless the CPU has waited it there already."""
        if self._cpu_waited(counter, threshold, region):
            return
        graph = self._graph(region)
        with graph.inserting_before(anchor):
            wait = graph.call_function(
                _ASYNC_WAIT, (counter, threshold), name=self.names("async_wait")
            )
        wait.meta.update(self.tags)
        key = (self._origin(counter, region), self._origin(threshold, region))
        gm = region.gm if region is not None else self.gm
        self.cpu_waits.setdefault(gm, set()).add(key)

    def _close(self, totals) -> None:
        """The layer ends quiescent: every counter waited at its total after
        the layer's last node and the counts read off its loops and conds
        there, the streams' first."""
        anchor = self.nodes[-1]
        members = set(self.nodes)
        while (
            anchor.next.op == "call_function"
            and anchor.next.target is operator.getitem
            and anchor.next.args[0] in members
        ):
            anchor = anchor.next
        agents = [a for a in self.counters if isinstance(a, _Stream)]
        agents += [a for a in self.counters if not isinstance(a, _Stream)]
        for agent in agents:
            total = totals.get(agent, 0)
            if total == 0:
                continue
            with self.gm.graph.inserting_after(anchor):
                anchor = self.gm.graph.call_function(
                    _ASYNC_WAIT,
                    (self.counters[agent], total),
                    name=self.names("async_wait"),
                )
            anchor.meta.update(self.tags)

    def _unique(self, seq) -> list:
        unique = []
        for item in seq:
            if item not in unique:
                unique.append(item)
        return unique

    def _check_waits(self, d: _Dispatch, waits) -> None:
        for unit, seq in waits.items():
            if len(seq) > _WAITS_PER_UNIT:
                raise SyncError(
                    f"{self.name}: {d.node.name} queues {len(seq)} waits on "
                    f"the {unit} unit, more than {_WAITS_PER_UNIT}"
                )

    # --- the counts each producer updates -------------------------------

    def _keys(self, passes, post) -> set:
        """The state keys a producer of ``passes`` posting on ``post``'s
        counter updates."""
        keys = {(_POSTS, post)}
        for p in passes:
            for kind, accesses in (
                (_TOUCH, p.reads + p.writes),
                (_WRITE, p.writes),
            ):
                for buffer, slot in accesses:
                    for j in self._slot_range(buffer, slot):
                        keys.add((kind, p.unit, buffer, j, post))
        return keys

    def _slot_range(self, buffer, slot):
        count = max(1, _slots(buffer))
        self.slots[buffer] = count
        if isinstance(slot, int):
            return [slot]
        return list(range(count))

    def _explore(self, nodes, region) -> set:
        """Find every dispatch, copy and loop under ``nodes`` of ``region``
        (``None``: the layer's graph), freshening the regions on the way;
        return the state keys the dispatches and copies among them
        update."""
        keys = set()
        owner = region.gm if region is not None else self.gm
        for node in list(nodes):
            if node.op != "call_function" and node.op != "call_module":
                continue
            if node.target is COND and region is None:
                # A top-level cond's branches are graphs of their own.
                continue
            if node.target is COND:
                operands = list(node.args[3]) if len(node.args) > 3 else []
                for handle in node.args[1:3]:
                    sub = self._fresh(owner, handle.target)
                    child = _Region(
                        sub,
                        region,
                        node,
                        dict(zip(_placeholders(sub), operands)),
                    )
                    self.regions[sub] = child
                    keys |= self._explore(sub.graph.nodes, child)
            elif node.target is WHILE_LOOP:
                body = self._fresh(owner, node.args[1].target)
                self._fresh(owner, node.args[0].target)
                carried = list(node.args[2])
                extra = list(node.args[3]) if len(node.args) > 3 else []
                phs = _placeholders(body)
                child = _Region(
                    body, region, node, dict(zip(phs[len(carried) :], extra))
                )
                self.regions[body] = child
                inner = self._explore(body.graph.nodes, child)
                self.modified[node] = inner
                keys |= inner
            elif node.target is _ASYNC_COPY:
                c = self._copy(node, region)
                self.copies[node] = c
                keys |= self._keys([c.access], c.stream)
            elif node.target is _ASYNC_WAIT:
                raise SyncError(
                    f"{self.name}: {node.name} is a builder's wait; the "
                    f"builders emit no synchronization"
                )
            elif _is_dispatch(node):
                d = self._dispatch(node, region)
                self.sites[node] = d
                keys |= self._keys(d.passes, d.post_unit)
        return keys

    # --- state ---------------------------------------------------------------

    def _slot_index(self, s: _Scalars, slot, region):
        """The runtime slot index of an access as a value of ``region``."""
        node, home = slot
        return self._thread(node, region) if home is not region else node

    def _lookup(self, s, state, kind, agent, buffer, slot, region):
        """The saved counts ``kind`` of accessor ``agent`` (a pass unit or a
        stream) on ``buffer``'s ``slot`` in ``state``, as ``[(posting
        agent, count)]``: one per counter its producers post on (a unit's
        dispatch on either unit's, a stream's copy on its own), and for a
        whole-buffer access one per slot too."""
        count = self.slots.get(buffer, max(1, _slots(buffer)))
        found = []
        for post in UNITS if agent in UNITS else (agent,):
            values = [
                self._thread(
                    state.get((kind, agent, buffer, j, post), 0), region
                )
                for j in range(count)
            ]
            if slot is None:
                found += [(post, v) for v in values]
                continue
            if isinstance(slot, int):
                found.append((post, values[slot]))
                continue
            index = self._slot_index(s, slot, region)
            chosen = values[-1]
            for j in range(count - 2, -1, -1):
                chosen = s.ite(s.eq(index, j), values[j], chosen)
            found.append((post, chosen))
        return found

    def _record(self, s, state, kind, unit, post, buffer, slot, value, region):
        """Set the count ``kind`` of accessor ``unit`` on ``buffer``'s
        ``slot``, posted on ``post``'s counter, to ``value`` in ``state``."""
        count = self.slots.get(buffer, max(1, _slots(buffer)))
        index = None
        if slot is not None and not isinstance(slot, int):
            index = self._slot_index(s, slot, region)
        for j in range(count):
            key = (kind, unit, buffer, j, post)
            if slot is None or slot == j:
                state[key] = value
            elif index is not None:
                old = self._thread(state.get(key, 0), region)
                state[key] = s.ite(s.eq(index, j), value, old)

    def _producers(self, s, state, p: _Pass, region) -> list:
        """The ``(counter, count)`` waits the accesses of ``p`` need: on
        every stream of each buffer but ``p``'s own, then on every unit, the
        saved count of the last writer of what ``p`` reads and of the last
        toucher of what it writes (RAW, WAR, WAW).  A slot ``p`` also
        writes needs only the toucher's count: it is never below the
        writer's."""
        lookups = [(_WRITE, a) for a in p.reads if a not in p.writes]
        lookups += [(_TOUCH, a) for a in p.writes]
        lookups = [
            (kind, access)
            for kind, access in lookups
            if kind == _TOUCH or access[0] in self.written
        ]
        found = []
        for kind, (buffer, slot) in lookups:
            for stream in self.streams.get(buffer, []):
                if stream != p.unit:
                    found += self._lookup(
                        s, state, kind, stream, buffer, slot, region
                    )
        for unit in UNITS:
            for kind, (buffer, slot) in lookups:
                found += self._lookup(
                    s, state, kind, unit, buffer, slot, region
                )
        waits = [
            (self._thread(self._counter(post), region), value)
            for post, value in found
            if value != 0
        ]
        return self._unique(waits)

    def _dependencies(self, s, state, d: _Dispatch, region):
        """``d``'s waits per unit: each pass's ``_producers``, but those the
        CPU has already waited where ``d`` runs."""
        waits = {}
        for p in d.passes:
            seq = [
                (counter, value)
                for counter, value in self._producers(s, state, p, region)
                if not self._cpu_waited(counter, value, region)
            ]
            if seq:
                waits[p.unit] = seq
        return waits

    def _post(self, s, state, passes, post, region) -> None:
        """Issue a producer of ``passes`` posting on ``post``'s counter: its
        ordinal, the counter's count once it completes, becomes the saved
        count of every slot it touches and writes."""
        key = (_POSTS, post)
        ordinal = s.add(self._thread(state.get(key, 0), region), 1)
        state[key] = ordinal
        for p in passes:
            for kind, accesses in (
                (_TOUCH, p.reads + p.writes),
                (_WRITE, p.writes),
            ):
                for buffer, slot in accesses:
                    self._record(
                        s,
                        state,
                        kind,
                        p.unit,
                        post,
                        buffer,
                        slot,
                        ordinal,
                        region,
                    )

    # --- the walk ------------------------------------------------------

    def synchronize(self) -> None:
        state = {}
        self._walk(self.nodes, None, state)
        self._close({a: state.get((_POSTS, a), 0) for a in self.counters})

    def _walk(self, nodes, region, state) -> None:
        for node in list(nodes):
            if node.op not in ("call_function", "call_module"):
                continue
            if node.target is COND and region is None:
                continue
            if node.target is COND:
                self._walk_cond(node, region, state)
            elif node.target is WHILE_LOOP:
                self._walk_loop(node, region, state)
            elif node in self.sites:
                self._walk_dispatch(self.sites[node], state)
            elif node in self.copies:
                self._walk_copy(self.copies[node], state)
            elif node.target is _SCALAR_READ:
                self._walk_cpu(node, region, state)
            elif _is_host_op(node) or node.target is _ZEROS:
                self._walk_cpu(node, region, state)

    def _graph(self, region) -> Graph:
        return (region.gm if region is not None else self.gm).graph

    def _scalars(self, region, anchor) -> _Scalars:
        graph = self._graph(region)
        return _Scalars(
            self.names, graph, anchor, self.memos.setdefault(graph, {})
        )

    def _walk_dispatch(self, d: _Dispatch, state) -> None:
        region = d.region
        s = self._scalars(region, d.node)
        waits = self._dependencies(s, state, d, region)
        self._check_waits(d, waits)
        post = self._thread(self._counter(d.post_unit), region)
        # The ordinals the dispatch leaves behind are computed before the
        # op is wrapped: the scalars sit before the commit, as its own
        # thresholds do.
        self._post(s, state, d.passes, d.post_unit, region)
        self._wrap(d, waits, post)

    def _walk_copy(self, copy: _Copy, state) -> None:
        """A copy waits on the CPU for its producers (``_producers``), then
        is issued on its stream's counter."""
        region = copy.region
        s = self._scalars(region, copy.node)
        for counter, value in self._producers(s, state, copy.access, region):
            self._wait(copy.node, counter, value, region)
        args = list(copy.node.args) + [None] * (5 - len(copy.node.args))
        args[4] = self._thread(self._counter(copy.stream), region)
        copy.node.args = tuple(args)
        copy.node.kwargs = {
            k: v for k, v in copy.node.kwargs.items() if k != "counter"
        }
        self._post(s, state, [copy.access], copy.stream, region)
        self.touched.add(copy.region.gm if copy.region else self.gm)

    def _walk_cpu(self, node, region, state) -> None:
        """A CPU op -- a scalar read, a host op (``_is_host_op``) or a zero
        fill -- waits on the CPU for the producers of the on-chip values it
        reads and overwrites.  It posts nothing: the CPU issues what follows
        only once it is done."""
        if node.target is _SCALAR_READ:
            reads, written = [node.args[0]], []
        elif node.target is _ZEROS:
            # Fills its operand in place; a counter reset is not on chip, so
            # it waits for nothing.
            reads, written = [], [node.args[0]]
        else:
            reads = node.all_input_nodes
            written = [node] + [
                user.args[1]
                for user in node.users
                if user.target is _INSERT and user.args[0] is node
            ]
        p = _Pass(
            None,
            [a for a in (self._access(n, region) for n in reads) if a],
            [a for a in (self._access(n, region) for n in written) if a],
        )
        s = self._scalars(region, node)
        for counter, value in self._producers(s, state, p, region):
            self._wait(node, counter, value, region)

    def _walk_cond(self, node, region, state) -> None:
        """Walk both branches from the state at the cond; a key either
        branch changes is yielded by both and read back through a
        ``getitem``."""
        owner = region.gm if region is not None else self.gm
        branches = [
            self.regions[_subgraph(owner, h.target)] for h in node.args[1:3]
        ]
        states = []
        for branch in branches:
            local = dict(state)
            self._walk(branch.gm.graph.nodes, branch, local)
            states.append(local)
        keys = [
            key
            for key in sorted(set(states[0]) | set(states[1]), key=str)
            if any(st.get(key, 0) != state.get(key, 0) for st in states)
        ]
        if not keys:
            return
        outputs = [n for n in branches[0].gm.graph.nodes if n.op == "output"][0]
        base = len(outputs.args[0])
        for branch, local in zip(branches, states):
            out = [n for n in branch.gm.graph.nodes if n.op == "output"][0]
            values = [self._thread(local.get(k, 0), branch) for k in keys]
            out.args = (tuple(list(out.args[0]) + values),)
            self.touched.add(branch.gm)
        graph = self._graph(region)
        with graph.inserting_after(node):
            for i, key in enumerate(keys):
                state[key] = graph.call_function(
                    operator.getitem,
                    (node, base + i),
                    name=self.names("getitem"),
                )
        self.touched.add(owner)

    def _walk_loop(self, node, region, state) -> None:
        """Walk the body with the keys it modifies carried: each starts from
        the state at the loop, is returned by the body, and is read back
        from the loop's results."""
        owner = region.gm if region is not None else self.gm
        body = _subgraph(owner, node.args[1].target)
        cond_fn = _subgraph(owner, node.args[0].target)
        body_region = self.regions[body]
        keys = sorted(self.modified[node], key=str)
        carried = list(node.args[2])
        base = len(carried)
        inits = [self._thread(state.get(key, 0), region) for key in keys]
        node.args = (*node.args[:2], tuple(carried + inits), *node.args[3:])
        local = dict(state)
        incoming = {}
        for gm in (cond_fn, body):
            phs = _placeholders(gm)
            anchor = phs[base - 1]
            for key in keys:
                with gm.graph.inserting_after(anchor):
                    anchor = gm.graph.placeholder(self.names("ordinal"))
                if gm is body:
                    local[key] = anchor
                    incoming[key] = anchor
            self.touched.add(gm)
        self._walk(body.graph.nodes, body_region, local)
        # A count is carried because a dispatch in the body updates it; one
        # that comes back as it went in would keep its value on every trip.
        stale = [k for k in keys if local.get(k) is incoming[k]]
        if stale:
            raise SyncError(
                f"{self.name}: loop {node.name} carries counts its body never "
                f"updates: {[str(k[:2]) + ' ' + k[2].name for k in stale]}"
            )
        out = [n for n in body.graph.nodes if n.op == "output"][0]
        finals = [self._thread(local.get(key, 0), body_region) for key in keys]
        out.args = (tuple(list(out.args[0]) + finals),)
        graph = self._graph(region)
        with graph.inserting_after(node):
            for i, key in enumerate(keys):
                state[key] = graph.call_function(
                    operator.getitem,
                    (node, base + i),
                    name=self.names("getitem"),
                )
        self.touched.add(owner)
