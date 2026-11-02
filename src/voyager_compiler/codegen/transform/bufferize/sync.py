"""The sync pass: every dispatch a ``commit``, every dependency a hardware
counter wait.

Runs after memory planning, on each layer the builders emitted -- its tile
loop(s) with their prologue and epilogue -- and leaves the graph the
program the hardware runs: each compute op wrapped in a ``commit`` whose
waits, grouped by unit, the pass derived, and whose last pass adds 1 to
that unit's counter; a CPU ``async_wait`` before every copy or scalar read
that depends on a dispatch; the unit counters waited at their totals when
the layer ends.

The pass identifies each wait's producer; the counts come from the
program itself.  Per unit it keeps the number of posts issued so far, and
per buffer slot and unit the count at which the last dispatch touching it
(reading or writing) and the last one writing it posted: scalars the CPU
updates beside each dispatch, carried through the loops and out of the
conds like the builders' own cursors.  A consumer waits ``counter >=``
the saved count of its producer: the other unit's last writer of what it
reads and last toucher of what it overwrites, and its own unit's last
writer of either (a pipelined unit starts a pass before the previous one
has finished writing).  The builders' CPU waits on stream counters move
into the commits they guard.  An SpMM dispatch fetches before it starts,
so it takes CPU waits before its dispatch and posts nothing; whatever
depends on it drains.
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
_DRAIN = voyager.drain.default
_INSERT = voyager.insert.default
_SUBVIEW = voyager.subview.default
_SCALAR_READ = torch.ops.aten._local_scalar_dense.default
_SPMM = torch.ops.quantized_ops.spmm_csr.default
_REGISTER = int(MemoryLevel.REGISTER)

MATRIX, VECTOR = UNITS
# The unit counters' names in the IR.
_COUNTER_NAMES = {MATRIX: "mu", VECTOR: "vu"}
# Waits one unit's pass can queue.
_WAITS_PER_UNIT = 16
# The state keys: a unit's post count, and per buffer slot, pass unit and
# posting counter the count at which the last dispatch reading-or-writing
# the slot on that unit, and the last writing it, posted on that counter (a
# dispatch's completion is seen only on the counter it posts).
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
    """One pass of a dispatch: its unit, and the on-chip buffers it reads
    and writes, in operand order, each with the slot it accesses (an int, a
    ``(node, region)`` or ``None`` for every slot)."""

    unit: str
    reads: List[tuple] = field(default_factory=list)
    writes: List[tuple] = field(default_factory=list)

    def buffers(self, accesses):
        return [b for b, _ in accesses]

    def touches(self, buffer) -> bool:
        return buffer in self.buffers(self.reads + self.writes)


@dataclass
class _Dispatch:
    """A dispatchable op: the node, the region it runs in (``None``: the
    layer's graph), the ``insert``s of its results, its passes, and the unit
    of its last pass, which posts."""

    node: Node
    region: Optional[_Region]
    inserts: List[Node]
    passes: List[_Pass] = field(default_factory=list)
    post_unit: str = VECTOR
    spmm: bool = False


@dataclass
class _Copy:
    """A ``voyager.async_copy``: its node and region (``None`` outside every
    loop), the on-chip buffer it fills or drains with the slot, the counter
    it adds to, and whether it is a load."""

    node: Node
    region: Optional[_Region]
    buffer: Node
    slot: object
    counter: Node
    load: bool


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
            dispatch has no pass on, more waits than a unit queues), or more
            counters live at once than ``plan_counters`` has ids.
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
    for gm in graphs:
        for name, nodes in _layers(gm):
            # A layer with neither a loop nor a dispatch of its own -- a
            # host op, or a cond whose branches hold layers of their own
            # (walked as graphs above) -- has nothing to synchronize.
            if not any(
                n.op == "call_function"
                and n.target is WHILE_LOOP
                or _is_dispatch(n)
                for n in nodes
            ):
                continue
            _Layer(gm, nodes, name, names, order).synchronize()
            layers += 1
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


def _placeholders(gm: GraphModule) -> List[Node]:
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


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


def _is_scalar_node(node: Node) -> bool:
    """A node a wait may sit after without separating it from the op it
    guards: addressing and index arithmetic."""
    if node.op != "call_function":
        return False
    if node.target in (_SUBVIEW, operator.getitem) or is_nop(node):
        return True
    return not isinstance(getattr(node, "value", None), torch.Tensor) and (
        node.meta.get("val") is None
        or not isinstance(node.meta.get("val"), torch.Tensor)
    )


class _Scalars:
    """Scalar expressions in one graph, folded when constant and shared
    when repeated, each new node inserted before ``anchor``."""

    def __init__(self, names: _Names, graph: Graph, anchor: Node):
        self.names = names
        self.graph = graph
        self.anchor = anchor
        self.memo = {}

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
    graphs are read (buffers, slots, dispatches and their passes, copies,
    the builders' waits) and rewritten (values threaded into regions, unit
    counters, carried counts, commits, waits)."""

    def __init__(self, gm, nodes, name, names, order):
        self.gm = gm
        self.nodes = nodes
        self.name = name
        self.names = names
        self.order = order
        self.touched = set()
        self.regions: Dict[GraphModule, _Region] = {}
        self.threaded = {}
        self.unit_counters: Dict[str, Node] = {}
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
        # The builders' CPU waits, each with the region it stands in.
        self.builder_waits: List[tuple] = []
        self.spmm_buffers = set()
        self.slots: Dict[Node, int] = {}
        self.modified: Dict[Node, set] = {}
        self.modified[None] = self._explore(nodes, None)
        self.written = {
            b
            for d in self.sites.values()
            for p in d.passes
            for b in p.buffers(p.writes)
        }
        self.guards = self._guards()

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
        vector pass, the results written by the pass of the last op."""
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
        d.spmm = matrix is not None and matrix.target is _SPMM
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
        writer = MATRIX if matrix is not None and not tail else VECTOR
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
        """Classify a copy by the on-chip buffer it fills or drains."""
        src = self._access(node.args[0], region)
        dst = self._access(node.args[1], region)
        if dst is not None:
            (buffer, slot), load = dst, True
        elif src is not None:
            (buffer, slot), load = src, False
        else:
            raise SyncError(
                f"{self.name}: {node.name} copies between two off-chip "
                f"buffers"
            )
        counter = self._buffer(node.args[4], region)
        return _Copy(node, region, buffer, slot, counter, load)

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

    def _unit_counter(self, unit) -> Node:
        """The layer's counter for ``unit``, allocated at the register level
        before its first node on first use."""
        if unit not in self.unit_counters:
            graph = self.gm.graph
            with graph.inserting_before(self.nodes[0]):
                counter = graph.call_function(
                    voyager.alloc.default,
                    ([], torch.int64, _REGISTER),
                    name=self.names(_COUNTER_NAMES[unit]),
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
            self.unit_counters[unit] = counter
        return self.unit_counters[unit]

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

    def _wait(self, graph, anchor, counter, threshold) -> Node:
        """A CPU ``async_wait(counter, threshold)`` before ``anchor``."""
        with graph.inserting_before(anchor):
            wait = graph.call_function(
                _ASYNC_WAIT, (counter, threshold), name=self.names("async_wait")
            )
        wait.meta.update(self.tags)
        return wait

    def _drain(self, graph, anchor) -> Node:
        """A CPU ``drain()`` before ``anchor``."""
        with graph.inserting_before(anchor):
            drain = graph.call_function(_DRAIN, (), name=self.names("drain"))
        drain.meta.update(self.tags)
        return drain

    def _close(self, totals) -> None:
        """The layer ends quiescent: every unit counter waited at its total
        after the layer's last node."""
        anchor = self.nodes[-1]
        for unit in UNITS:
            total = totals.get(unit, 0)
            if total == 0:
                continue
            with self.gm.graph.inserting_after(anchor):
                anchor = self.gm.graph.call_function(
                    _ASYNC_WAIT,
                    (self._unit_counter(unit), total),
                    name=self.names("async_wait"),
                )
            anchor.meta.update(self.tags)

    def _recompile(self) -> None:
        for gm in self.touched:
            gm.recompile()
        self.gm.recompile()

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

    # --- the counts each dispatch updates -------------------------------

    def _keys(self, d: _Dispatch) -> set:
        """The state keys ``d`` updates."""
        keys = set()
        if d.spmm:
            return keys
        keys.add((_POSTS, d.post_unit))
        for p in d.passes:
            for kind, accesses in (
                (_TOUCH, p.reads + p.writes),
                (_WRITE, p.writes),
            ):
                for buffer, slot in accesses:
                    for j in self._slot_range(buffer, slot):
                        keys.add((kind, p.unit, buffer, j, d.post_unit))
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
        return the state keys the dispatches among them update."""
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
                self.copies[node] = self._copy(node, region)
            elif node.target is _ASYNC_WAIT:
                self.builder_waits.append((node, region))
            elif _is_dispatch(node):
                d = self._dispatch(node, region)
                self.sites[node] = d
                if d.spmm:
                    for p in d.passes:
                        self.spmm_buffers.update(p.buffers(p.reads + p.writes))
                keys |= self._keys(d)
        return keys

    def _guards(self) -> Dict[Node, List[Node]]:
        """The builders' stream waits each dispatch takes over, by dispatch:
        a wait standing right before it, and a wait at the root of a loop
        body for every later dispatch in that body (not in a loop nested in
        it) accessing the buffer the waited counter streams.  A wait on a
        buffer the CPU itself reads (a scalar read of it) stays the CPU's."""
        stream = {c.counter: c.buffer for c in self.copies.values()}
        cpu_read = set()
        for region in [None, *self.regions.values()]:
            graph = (region.gm if region is not None else self.gm).graph
            for n in graph.nodes:
                if n.op == "call_function" and n.target is _SCALAR_READ:
                    access = self._access(n.args[0], region)
                    if access is not None:
                        cpu_read.add(access[0])
        guards = {d: [] for d in self.sites}
        for wait, region in self.builder_waits:
            buffer = stream.get(self._buffer(wait.args[0], region))
            if buffer is None or buffer in cpu_read:
                continue
            for d in self.sites.values():
                if not any(p.touches(buffer) for p in d.passes):
                    continue
                if wait in self._pending_waits(d) or self._after_in_body(
                    d, wait, region
                ):
                    guards[d.node].append(wait)
        return guards

    def _after_in_body(self, d: _Dispatch, wait, region) -> bool:
        """Whether ``wait`` stands at the root of a loop body (``region``)
        and ``d`` runs later in the same iteration: under that body, after
        the wait, and in no loop nested in it."""
        if region is None or region.hop.target is not WHILE_LOOP:
            return False
        node, inner = d.node, d.region
        while inner is not None and inner is not region:
            if inner.hop.target is WHILE_LOOP:
                return False
            node, inner = inner.hop, inner.parent
        return inner is region and self.order[node] > self.order[wait]

    # --- state ---------------------------------------------------------------

    def _slot_index(self, s: _Scalars, slot, region):
        """The runtime slot index of an access as a value of ``region``."""
        node, home = slot
        return self._thread(node, region) if home is not region else node

    def _lookup(self, s, state, kind, unit, buffer, slot, region):
        """The saved counts ``kind`` of pass unit ``unit`` on ``buffer``'s
        ``slot`` in ``state``, as ``[(posting unit, count)]``: one per
        posting counter, and for a whole-buffer access one per slot too."""
        count = self.slots.get(buffer, max(1, _slots(buffer)))
        found = []
        for post in UNITS:
            values = [
                self._thread(
                    state.get((kind, unit, buffer, j, post), 0), region
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
        """Set the count ``kind`` of pass unit ``unit`` on ``buffer``'s
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

    def _dependencies(self, s, state, d: _Dispatch, region):
        """``d``'s waits per unit from ``state`` (the ordinals of the other
        unit's last toucher of what it writes and last writer of what it
        reads, and of its own unit's last writer of either) plus whether it
        must drain, and the builders' stream waits it takes over
        (``_guards``), on the pass accessing the streamed buffer."""
        stream = {c.counter: c.buffer for c in self.copies.values()}
        guards = self.guards.get(d.node, [])
        waits, drain = {}, False
        for p in d.passes:
            other = VECTOR if p.unit == MATRIX else MATRIX
            seq = []
            for buffer in p.buffers(p.reads + p.writes):
                for wait in guards:
                    home = next(r for w, r in self.builder_waits if w is wait)
                    if stream.get(self._buffer(wait.args[0], home)) is buffer:
                        seq.append(
                            (
                                self._thread(wait.args[0], region),
                                self._thread(wait.args[1], region),
                            )
                        )
            # The other unit's last writer of what the pass reads and last
            # toucher of what it writes; this unit's last writer of either
            # (a pipelined unit starts a pass before the previous one has
            # finished writing).
            # A buffer the pass also writes needs only the toucher's count:
            # it is never below the writer's.
            written = set(p.buffers(p.writes))
            reads = [a for a in p.reads if a[0] not in written]
            lookups = [(_WRITE, other, reads), (_TOUCH, other, p.writes)]
            lookups += [(_WRITE, p.unit, p.reads + p.writes)]
            for kind, unit, accesses in lookups:
                for buffer, slot in accesses:
                    if buffer in self.spmm_buffers and not d.spmm:
                        drain = True
                    if buffer not in self.written and kind == _WRITE:
                        continue
                    for post, value in self._lookup(
                        s, state, kind, unit, buffer, slot, region
                    ):
                        if value == 0:
                            continue
                        counter = self._thread(self._unit_counter(post), region)
                        seq.append((counter, value))
            seq = self._unique(seq)
            if seq:
                waits[p.unit] = seq
        return waits, drain

    def _pending_waits(self, d: _Dispatch) -> List[Node]:
        """The builder's CPU waits standing right before ``d`` (only
        addressing and index arithmetic between)."""
        pending, node = [], d.node.prev
        while node is not None and node.op == "call_function":
            if node.target is _ASYNC_WAIT:
                pending.append(node)
            elif not _is_scalar_node(node):
                break
            node = node.prev
        return pending

    # --- the walk ------------------------------------------------------

    def synchronize(self) -> None:
        state = {}
        self._walk(self.nodes, None, state)
        # The builders' waits now in commits leave the CPU's program.
        for wait in {w for ws in self.guards.values() for w in ws}:
            wait.graph.erase_node(wait)
        self._close({u: state.get((_POSTS, u), 0) for u in UNITS})
        self._recompile()

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
                self._walk_read(node, region, state)
            elif _is_host_op(node):
                self._walk_host(node, region, state)

    def _graph(self, region) -> Graph:
        return (region.gm if region is not None else self.gm).graph

    def _walk_dispatch(self, d: _Dispatch, state) -> None:
        region = d.region
        s = _Scalars(self.names, self._graph(region), d.node)
        waits, drain = self._dependencies(s, state, d, region)
        self._check_waits(d, waits)
        if drain:
            self._drain(s.graph, d.node)
        if d.spmm:
            # SpMM fetches before it starts: its waits are the CPU's, and
            # it posts nothing.
            for seq in waits.values():
                for counter, threshold in seq:
                    self._wait(s.graph, d.node, counter, threshold)
            self._wrap(d, {}, None)
            return
        post = self._thread(self._unit_counter(d.post_unit), region)
        # The ordinals the dispatch leaves behind are computed before the
        # op is wrapped: the scalars sit before the commit, as its own
        # thresholds do.
        key = (_POSTS, d.post_unit)
        ordinal = s.add(self._thread(state.get(key, 0), region), 1)
        state[key] = ordinal
        for p in d.passes:
            for kind, accesses in (
                (_TOUCH, p.reads + p.writes),
                (_WRITE, p.writes),
            ):
                # Every count ``_keys`` declares is recorded here: a load
                # overwriting a slot waits on its last toucher, read-only
                # stream slots included.
                for buffer, slot in accesses:
                    self._record(
                        s,
                        state,
                        kind,
                        p.unit,
                        d.post_unit,
                        buffer,
                        slot,
                        ordinal,
                        region,
                    )
        self._wrap(d, waits, post)

    def _walk_copy(self, copy: _Copy, state) -> None:
        """A load waits for the last toucher of the slot it overwrites, a
        store for the last writer of the tile it reads, on every unit."""
        region = copy.region
        if copy.load and copy.buffer not in self.written:
            kind = _TOUCH
        elif copy.load:
            kind = _TOUCH
        else:
            kind = _WRITE
        if not copy.load and copy.buffer not in self.written:
            return
        s = _Scalars(self.names, self._graph(region), copy.node)
        if copy.buffer in self.spmm_buffers:
            self._drain(s.graph, copy.node)
        for unit in UNITS:
            for post, value in self._lookup(
                s, state, kind, unit, copy.buffer, copy.slot, region
            ):
                if value == 0:
                    continue
                counter = self._thread(self._unit_counter(post), region)
                self._wait(s.graph, copy.node, counter, value)

    def _walk_host(self, node, region, state) -> None:
        """A host op waits on the CPU for every unit's last writer of the
        on-chip values it reads and last toucher of those it overwrites.
        It posts nothing: the CPU issues what follows only once it is
        done."""
        accesses = [
            (_WRITE, self._access(arg, region)) for arg in node.all_input_nodes
        ]
        written = [node] + [
            user.args[1]
            for user in node.users
            if user.target is _INSERT and user.args[0] is node
        ]
        accesses += [(_TOUCH, self._access(w, region)) for w in written]
        s = _Scalars(self.names, self._graph(region), node)
        for kind, access in accesses:
            if access is None:
                continue
            buffer, slot = access
            if buffer in self.spmm_buffers:
                self._drain(s.graph, node)
            for unit in UNITS:
                for post, value in self._lookup(
                    s, state, kind, unit, buffer, slot, region
                ):
                    if value == 0:
                        continue
                    counter = self._thread(self._unit_counter(post), region)
                    self._wait(s.graph, node, counter, value)

    def _walk_read(self, node, region, state) -> None:
        """A CPU read of an on-chip value waits for its last writer."""
        access = self._access(node.args[0], region)
        if access is None or access[0] not in self.written:
            return
        buffer, slot = access
        s = _Scalars(self.names, self._graph(region), node)
        if buffer in self.spmm_buffers:
            self._drain(s.graph, node)
        for unit in UNITS:
            for post, value in self._lookup(
                s, state, _WRITE, unit, buffer, slot, region
            ):
                if value == 0:
                    continue
                counter = self._thread(self._unit_counter(post), region)
                self._wait(s.graph, node, counter, value)

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
