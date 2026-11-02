"""Concrete DRAM and Scratchpad addresses for a bufferized FX graph.

Bufferization tags every tensor node with ``meta['space']`` and fixes which
memory each value lives in; this pass only decides addresses, reuse and pool
sizes. A Local buffer gets no address (the target places it), and a Register
``alloc`` is a hardware counter, which gets an id instead.

* **DRAM**: the inputs and weights are laid out one after another with no
  reuse; the intermediate buffers (the DRAM ``voyager.alloc`` activations and
  the tensors the host materializes) are packed after them by
  ``_greedy_best_fit``, so buffers of disjoint lifetimes share addresses.
* **Scratchpad**: every on-chip buffer is an explicit ``voyager.alloc``,
  packed by the same allocator across region boundaries, above
  ``config.scratchpad_offset``. A pipelined alloc of ``num_slots`` slots is
  contiguous: slot ``i`` sits at ``base + i * slot_stride``, so the slot a
  step writes can stay a runtime index. On a banked config a buffer with a
  searched bank partition (``meta['bank_group']``) is placed with its group
  in whole banks, each pipeline slot in its own bank(s).

An address is a ``Segment`` on each buffer root, ``meta['memory']`` (DRAM)
or ``meta['scratchpad']``, plus ``meta['slot_count']`` and
``meta['slot_stride']`` on a pipelined one. A tile has no address of its
own: it is named by the buffer it lives in.
"""

import bisect
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import (
    get_arg_value,
    tensor_alloc_bytes,
)
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _viewed_buffer,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    UNPIPELINED,
    MemoryLevel,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _COND,
    _WHILE_LOOP,
    _collect_codebook_nodes,
    _passed_whole,
    _placeholders,
    _regions,
    _subgraph,
    voyager,
)
from voyager_compiler.shape_prop import fetch_attr

logger = logging.getLogger(__name__)

_ALLOC = voyager.alloc.default
# Position of ``num_slots`` in ``alloc``'s schema.
_SLOTS_ARG = 3

# The hardware's counters, ids ``0 .. NUM_COUNTERS - 1``.
NUM_COUNTERS = 32
# The id kept out of the compiler's assignment: the SoC emitter orders the
# passes inside one op on it (pass ``n`` posts it, pass ``n + 1`` waits),
# since only the mapper knows how an op splits into passes.
INTRA_OP_COUNTER = NUM_COUNTERS - 1
# The ids the compiler hands out, lowest first.
_COUNTER_IDS = [i for i in range(NUM_COUNTERS) if i != INTRA_OP_COUNTER]

# How many of the buffers live at the peak a capacity error lists.
_SHOWN_PEAK_BUFFERS = 12


@dataclass
class Segment:
    """The byte range ``[start, end)`` that ``node`` holds in a memory.

    ``start`` is truncated toward zero and ``end`` rounded up to an integer,
    with a warning when either was not one already.

    Raises:
        ValueError: If ``end`` lies before ``start``.
    """

    start: Union[float, int]
    end: Union[float, int]
    memory_space: Optional[int] = None
    node: Optional[Node] = None

    def __post_init__(self) -> None:
        start = int(self.start)
        end = math.ceil(self.end)
        if start != self.start:
            logger.warning(
                "Segment start %r is not an integer. Rounding to %d.",
                self.start,
                start,
            )
        if end != self.end:
            logger.warning(
                "Segment end %r is not an integer. Rounding up to %d.",
                self.end,
                end,
            )
        if end < start:
            raise ValueError(
                f"Segment end ({end}) is less than start ({start})."
            )
        self.start = start
        self.end = end


@dataclass
class MemoryPlan:
    """The pool sizes ``plan_memory`` used, in bytes."""

    dram_bytes: int
    scratchpad_bytes: int


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------


def _val(node) -> Optional[torch.Tensor]:
    """The tensor ``node`` produces, or ``None`` for a tuple or non-tensor.

    The size comes from ``node.value``, the shape the emitter declares, never
    from ``meta['val']``, which keeps the export-time shape after a pass
    reshapes the node.
    """
    if not isinstance(node, Node):
        return None
    value = getattr(node, "value", None)
    return value if isinstance(value, torch.Tensor) else None


def _slots(node) -> int:
    """The software-pipeline depth of an ``alloc`` (the slots its leading
    dimension holds); ``UNPIPELINED`` for every other node."""
    if (
        not isinstance(node, Node)
        or node.op != "call_function"
        or node.target is not _ALLOC
    ):
        return UNPIPELINED
    return int(get_arg_value(node, _SLOTS_ARG, "num_slots", UNPIPELINED))


def is_counter(node) -> bool:
    """Whether ``node`` declares a hardware counter: a ``voyager.alloc`` at
    the register level."""
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is _ALLOC
        and node.meta.get("space") == "Register"
    )


def _payload_bytes(node: Node, numel: int, config) -> int:
    """The allocated bytes of ``numel`` elements of ``node`` at its logical
    (quantized) dtype, aligned to ``config.bank_width``.

    A Scratchpad tensor also reserves the tail of its last vector beat, since
    the datapath stores in whole beats; the DMA writes DRAM byte-exact.
    """
    dtype = node.meta.get("dtype") or _val(node).dtype
    if node.meta.get("space") == "Scratchpad":
        lanes = config.vector_lanes
    else:
        lanes = None
    return tensor_alloc_bytes(numel, dtype, config.bank_width, lanes)


def _slot_stride(node: Node, config) -> int:
    """The byte distance between adjacent slots of a pipelined ``alloc``: the
    allocated size of one slot's payload, the ``numel // slots`` elements
    behind the leading slot dimension."""
    return _payload_bytes(node, _val(node).numel() // _slots(node), config)


def _nbytes(node: Node, config) -> int:
    """The allocated bytes of ``node``'s tensor; a pipelined ``alloc`` is
    ``slots`` aligned payloads, so every slot starts on an aligned boundary.

    Raises:
        ValueError: If ``node`` produces no tensor.
    """
    value = _val(node)
    if value is None:
        raise ValueError(f"{node} has no sized value to allocate memory for")
    slots = _slots(node)
    if slots:
        return slots * _slot_stride(node, config)
    return _payload_bytes(node, value.numel(), config)


def _align_up(value: int, alignment: int) -> int:
    """``value`` rounded up to a multiple of ``alignment``."""
    return math.ceil(value / alignment) * alignment


# ---------------------------------------------------------------------------
# Greedy best-fit shared-object allocator
# ---------------------------------------------------------------------------


class _Request(NamedTuple):
    """A block to place: ``size`` bytes at a multiple of ``align``, live from
    schedule step ``birth`` to ``death`` inclusive."""

    key: object
    size: int
    birth: int
    death: int
    align: int


def _greedy_best_fit(
    requests: List[_Request],
) -> Tuple[Dict[object, int], int]:
    """Place each block at the lowest aligned offset clear of every placed
    block whose lifetime overlaps its own.

    Blocks are placed largest first (ties: earliest birth, then request
    order), the greedy-by-size offset planner: blocks of disjoint lifetimes
    collapse onto the same low addresses, so the high-water mark approaches
    the peak concurrent demand.

    Args:
        requests: The blocks; each ``key`` is unique.

    Returns:
        ``({key: offset}, high-water mark in bytes)``.
    """
    # The placed blocks as (start, end, birth, death), in address order.
    placed: List[Tuple[int, int, int, int]] = []
    offsets: Dict[object, int] = {}
    total = 0
    by_size = sorted(requests, key=lambda r: (-r.size, r.birth))
    for request in by_size:
        offset = 0
        for start, end, birth, death in placed:
            if birth > request.death or request.birth > death:
                continue
            if _align_up(offset, request.align) + request.size <= start:
                break
            offset = max(offset, end)
        offset = _align_up(offset, request.align)
        end = offset + request.size
        bisect.insort(placed, (offset, end, request.birth, request.death))
        offsets[request.key] = offset
        total = max(total, end)
    return offsets, total


# ---------------------------------------------------------------------------
# DRAM
# ---------------------------------------------------------------------------


def _is_param(node: Node, gm: GraphModule, codebooks: set) -> bool:
    """Whether ``node`` is a weight that owns DRAM storage: a ``get_attr`` of
    a tensor, but not a subgraph handle, a lifted constant, a codebook /
    qmap or a scalar scale (those are passed whole)."""
    if node.op != "get_attr":
        return False
    if _subgraph(gm, node.target) is not None:
        return False
    if str(node.target).startswith("lifted_tensor"):
        return False
    return not _passed_whole(node, codebooks)


def _is_dram_buffer(node: Node) -> bool:
    """Whether ``node`` owns an intermediate DRAM buffer: a DRAM ``alloc``,
    or a tensor the host produces in DRAM (a ``pad`` to the hardware
    unrolling, a copy) that a loop then loads tiles from. A view owns no
    bytes of its own."""
    if node.op != "call_function" or node.meta.get("space") != "DRAM":
        return False
    if node.target is _ALLOC:
        return True
    return _val(node) is not None and _viewed_buffer(node) is None


def shared_dram_layout(
    models: List[GraphModule], config
) -> Dict[str, Tuple[int, int]]:
    """One DRAM range per input name across programs that share their inputs.

    The programs of a training step read and write the same parameters,
    gradients and optimizer state; each input keeps one address in all of
    them, and the ranges are laid out one after another from address 0.

    Args:
        models: Transformed programs, their inputs shape-propagated.
        config: The accelerator, which sets each range's alignment.

    Returns:
        ``{input name: (start, end)}`` in bytes.
    """
    layout = {}
    offset = 0
    for model in models:
        for node in model.graph.nodes:
            if node.op == "placeholder" and node.name not in layout:
                size = _nbytes(node, config)
                layout[node.name] = (offset, offset + size)
                offset += size
    return layout


def planned_buffer_layout(
    planned: GraphModule, model: GraphModule, config
) -> Dict[str, Tuple[int, int]]:
    """The DRAM ranges ``model``'s buffers already hold in ``planned``.

    A ``get_attr`` buffer of ``model`` whose tensor is the very object a
    planned buffer of ``planned`` holds -- a KV cache a prefill and a decode
    graph write -- keeps that buffer's address.  Matching is by tensor
    identity, never by name: a constant each graph holds a copy of, such as
    a codebook, stays private.

    Args:
        planned: A program ``plan_memory`` placed.
        model: A transformed program, not yet planned.
        config: The accelerator, which sets each buffer's size.

    Returns:
        ``{get_attr target: (start, end)}`` in bytes, for ``plan_memory``'s
        ``dram_layout``.

    Raises:
        ValueError: If the two programs size a shared buffer differently.
    """
    segment_of = {}
    for node in planned.graph.find_nodes(op="get_attr"):
        if "memory" in node.meta:
            tensor = fetch_attr(planned, node.target)
            segment_of[id(tensor)] = node.meta["memory"]
    layout = {}
    for node in model.graph.find_nodes(op="get_attr"):
        segment = segment_of.get(id(fetch_attr(model, node.target)))
        if segment is None:
            continue
        size = _nbytes(node, config)
        held = segment.end - segment.start
        if size != held:
            raise ValueError(
                f"{node.target} takes {size} bytes but {segment.node} holds "
                f"the same tensor in {held}"
            )
        layout[node.target] = (segment.start, segment.end)
    return layout


def _plan_dram(
    model: GraphModule,
    buffer_of: Dict[Node, Node],
    positions: Dict[Node, int],
    config,
    dram_layout: Dict[str, Tuple[int, int]],
) -> int:
    """Place every DRAM tensor of ``model`` and write ``meta['memory']``.

    The inputs and weights come first, after every range of ``dram_layout``,
    one after another; one named in ``dram_layout`` (an input by name, a
    ``get_attr`` by target) takes its range there. The intermediate buffers
    follow, packed by ``_greedy_best_fit`` over their lifetimes: from the
    node that defines one to the last top-level node that reads any of its
    names. A buffer inside a ``cond`` branch (a split cache's fold) takes the
    ``cond``'s step.

    Args:
        model: The bufferized graph.
        buffer_of: ``_buffer_identity``'s map from a name to its buffer.
        positions: The step of each top-level node.
        config: The accelerator, which sets each buffer's size.
        dram_layout: Pinned ranges, ``{input name or target: (start, end)}``.

    Returns:
        The DRAM bytes used.
    """
    codebooks = _collect_codebook_nodes(model)
    step = dict(positions)
    persistent: List[Node] = []
    intermediates: List[Node] = []
    for node in model.graph.nodes:
        if node.op == "placeholder" or _is_param(node, model, codebooks):
            if _val(node) is not None:
                persistent.append(node)
        elif _is_dram_buffer(node):
            intermediates.append(node)
        elif node.target is _COND:
            for branch, _operands in _regions(model, node):
                for inner in branch.graph.nodes:
                    if _is_dram_buffer(inner):
                        intermediates.append(inner)
                        step[inner] = step[node]

    # A read through any name of a buffer (a reshape, the getitem of a loop
    # result) keeps it alive; a while_loop reading it keeps it live for the
    # whole loop.
    death = {buffer: step[buffer] for buffer in intermediates}
    for node in model.graph.nodes:
        for operand in node.all_input_nodes:
            root = buffer_of.get(operand, operand)
            if root in death:
                death[root] = max(death[root], step[node])

    offset = max((end for _start, end in dram_layout.values()), default=0)
    for node in persistent:
        key = node.name if node.op == "placeholder" else node.target
        if key in dram_layout:
            start, end = dram_layout[key]
        else:
            start = offset
            end = offset + _nbytes(node, config)
            offset = end
        node.meta["memory"] = Segment(start, end, MemoryLevel.DRAM, node)

    sizes = {buffer: _nbytes(buffer, config) for buffer in intermediates}
    requests = []
    for buffer in intermediates:
        request = _Request(
            buffer, sizes[buffer], step[buffer], death[buffer], 1
        )
        requests.append(request)
    placed, intermediate_bytes = _greedy_best_fit(requests)
    for buffer in intermediates:
        start = offset + placed[buffer]
        end = start + sizes[buffer]
        buffer.meta["memory"] = Segment(start, end, MemoryLevel.DRAM, buffer)
    return offset + intermediate_bytes


# ---------------------------------------------------------------------------
# Schedule and buffer identity
# ---------------------------------------------------------------------------


def _walk(gm: GraphModule):
    """Yield every node of ``gm`` in execution order, each followed by the
    nodes of the regions it runs (``_regions``)."""
    for node in gm.graph.nodes:
        yield node
        for sub, _operands in _regions(gm, node):
            yield from _walk(sub)


def _timestamps(model: GraphModule) -> Dict[Node, int]:
    """The global execution step of every node, in ``_walk`` order, so a
    region's nodes fall inside the span of the node that runs it.  The map
    iterates its nodes in that order."""
    return {node: i for i, node in enumerate(_walk(model))}


def _buffer_identity(model: GraphModule) -> Dict[Node, Node]:
    """The buffer each node names; a node absent from the map names itself.

    A view names the buffer it views (a slot ``subview``, a reshape, the
    ``getitem`` of a loop result). A region's placeholder names the buffer of
    the operand bound to it (``_regions``), and each result of a
    ``while_loop`` body names the carried buffer it returns, written in
    place. Only a tensor is a buffer: a name bound to a scalar (a carried
    cursor or count) names nothing. Every rule maps a later name to an
    earlier one, so one pass in program order resolves them all.

    Args:
        model: The bufferized graph.

    Returns:
        ``{name: buffer root}``.

    Raises:
        ValueError: If a node would name two different buffers.
    """
    buffer_of: Dict[Node, Node] = {}

    def bind(alias: Node, source) -> None:
        # Two carried scalars may share the body's value: neither is bound.
        if _val(source) is None:
            return
        owner = buffer_of.get(source, source)
        if buffer_of.setdefault(alias, owner) is not owner:
            raise ValueError(
                f"'{alias.name}' names two buffers, '{buffer_of[alias].name}' "
                f"and '{owner.name}': they would have to be one"
            )

    def walk(gm: GraphModule) -> None:
        for node in gm.graph.nodes:
            viewed = _viewed_buffer(node)
            if viewed is not None:
                bind(node, viewed)
            for sub, operands in _regions(gm, node):
                for placeholder, operand in zip(_placeholders(sub), operands):
                    bind(placeholder, operand)
                if node.target is _WHILE_LOOP:
                    results = sub.graph.output_node().args[0]
                    if not isinstance(results, (list, tuple)):
                        results = [results]
                    for result, carried in zip(results, node.args[2]):
                        bind(result, carried)
                walk(sub)

    walk(model)
    return buffer_of


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def _plan_counters(order: Dict[Node, int], buffer_of: Dict[Node, Node]) -> int:
    """Give every counter a hardware id, by lifetime.

    A counter (``is_counter``) lives from its ``alloc`` to its last reset,
    wait or post, under every name a region binds it to.  Ids are handed
    out in program order, each the lowest one whose last holder is dead, so
    a kernel reuses the ids of the kernels before it once their epilogues
    have waited every counter at its final count; ``INTRA_OP_COUNTER`` is
    never handed out.  Stamps ``meta['counter']`` on each counter
    ``alloc``.

    Args:
        order: ``_timestamps``' global execution order.
        buffer_of: ``_buffer_identity``'s map from a name to its buffer.

    Returns:
        How many ids were used.

    Raises:
        ValueError: A counter is not a 0-d int64, or more counters are
            live at once than there are ids besides ``INTRA_OP_COUNTER``.
    """
    last_use: Dict[Node, int] = {}
    for node, step in order.items():
        root = buffer_of.get(node, node)
        if not is_counter(root):
            continue
        uses = [order[user] for user in node.users if user in order]
        last_use[root] = max(last_use.get(root, 0), step, *uses)

    # id -> last use of the counter holding it
    holders: Dict[int, int] = {}
    for counter in sorted(last_use, key=order.__getitem__):
        value = _val(counter)
        if value is None or value.dim() != 0 or value.dtype != torch.int64:
            raise ValueError(
                f"{counter.name}: a counter is a 0-d int64 alloc at the "
                f"register level, not {value}"
            )
        born = order[counter]
        free = [i for i in _COUNTER_IDS if holders.get(i, -1) < born]
        if not free:
            raise ValueError(
                f"[plan_memory] more than {NUM_COUNTERS - 1} counters live "
                f"at '{counter.name}' (id {INTRA_OP_COUNTER} is reserved)"
            )
        holders[free[0]] = last_use[counter]
        counter.meta["counter"] = free[0]
    return len(holders)


def plan_counters(model: GraphModule) -> int:
    """Assign the counter ids of ``model`` afresh (``_plan_counters``), from
    its current buffer identity and schedule.

    Args:
        model: The bufferized graph, after a pass that added counters.

    Returns:
        How many ids were used.
    """
    return _plan_counters(_timestamps(model), _buffer_identity(model))


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------


@dataclass
class _Buf:
    """A Scratchpad buffer.

    Attributes:
        size: Its allocated bytes, those of ``largest``.
        birth: The first step a tile of it is defined.
        death: The last step any of its names is read.
        tiles: The names that own its bytes, in schedule order: every
            Scratchpad name except the views.
        largest: The first tile of the largest size.
        group: ``(scope, bank_group)`` of its searched bank group, or
            ``None`` for one the tile search left loose. ``scope`` is the
            per-splice nest tag, so two nests spliced from one cached build
            never share a group.
    """

    size: int
    birth: int
    death: int
    tiles: List[Node]
    largest: Node
    group: Optional[tuple]


def _buffer_lifetimes(
    order: Dict[Node, int], buffer_of: Dict[Node, Node], config
) -> Dict[Node, _Buf]:
    """Every Scratchpad buffer, by root, with its size and lifetime.

    A view names its buffer but neither sizes it nor starts its life (a slot
    ``select`` is one tile of a slot several tiles deep); a use of any name,
    view or not, extends the life.

    Args:
        order: ``_timestamps``' global execution order.
        buffer_of: ``_buffer_identity``'s map from a name to its buffer.
        config: The accelerator, which sets each buffer's size.

    Returns:
        ``{root: _Buf}``, in schedule order of the roots' first names.
    """
    names: Dict[Node, List[Node]] = {}
    for node in order:
        names.setdefault(buffer_of.get(node, node), []).append(node)

    bufs: Dict[Node, _Buf] = {}
    for root, aliases in names.items():
        tiles = []
        for alias in aliases:
            in_scratchpad = alias.meta.get("space") == "Scratchpad"
            if in_scratchpad and _viewed_buffer(alias) is None:
                tiles.append(alias)
        if not tiles:
            continue
        sizes = [_nbytes(tile, config) for tile in tiles]
        size = max(sizes)
        largest = tiles[sizes.index(size)]
        birth = min(order[tile] for tile in tiles)
        death = birth
        for alias in aliases:
            for user in alias.users:
                if user in order:
                    death = max(death, order[user])
        group = None
        for tile in tiles:
            bank_group = tile.meta.get("bank_group")
            if bank_group is not None:
                group = (tile.meta.get("scope"), bank_group)
                break
        bufs[root] = _Buf(size, birth, death, tiles, largest, group)
    return bufs


def _plan_scratchpad(bufs: Dict[Node, _Buf], config) -> Tuple[int, int, int]:
    """Pack every Scratchpad buffer with ``_greedy_best_fit`` and write
    ``meta['scratchpad']`` on each of its tiles.

    Each on-chip buffer is an explicit ``alloc``, so simultaneously-live
    slots are distinct buffers and land at distinct addresses. On a banked
    config the buffers of one bank group are placed together: side by side
    in a slot region rounded up to whole banks, the region repeated once per
    slot of the group's deepest member (a shallower member occupies the
    first region only), the group at a bank-aligned base. Slot ``s`` of
    every member thus sits in its own whole bank(s), and a loose buffer
    disjoint from the group in address is disjoint from it in banks. A
    grouped pipelined tile gets ``meta['bank_group_stride']``, the region's
    size. Addresses start at ``config.scratchpad_offset``.

    Args:
        bufs: ``_buffer_lifetimes``' buffers.
        config: The accelerator.

    Returns:
        ``(bytes used including the reserved offset, blocks placed, bank
        groups)``.
    """
    bank = config.bank_size
    groups: Dict[tuple, List[Node]] = {}
    if bank:
        for root, buf in bufs.items():
            if buf.group is not None:
                groups.setdefault(buf.group, []).append(root)
    grouped = set()
    for roots in groups.values():
        grouped.update(roots)

    requests = []
    for root, buf in bufs.items():
        if root not in grouped:
            requests.append(_Request(root, buf.size, buf.birth, buf.death, 1))

    # Per group: the slot region's size, and each member's (offset in the
    # region, payload of one slot, slots).
    layouts: Dict[tuple, Tuple[int, Dict[Node, Tuple[int, int, int]]]] = {}
    for key, roots in groups.items():
        members = {}
        offset = 0
        depth = 1
        for root in roots:
            largest = bufs[root].largest
            slots = _slots(largest)
            if slots:
                payload = _slot_stride(largest, config)
            else:
                payload = bufs[root].size
                slots = 1
            members[root] = (offset, payload, slots)
            offset += payload
            depth = max(depth, slots)
        region_bytes = _align_up(offset, bank)
        layouts[key] = (region_bytes, members)
        birth = min(bufs[root].birth for root in roots)
        death = max(bufs[root].death for root in roots)
        request = _Request(
            ("bank_group", key), depth * region_bytes, birth, death, bank
        )
        requests.append(request)

    offsets, total = _greedy_best_fit(requests)
    reserved = config.scratchpad_offset

    for root, buf in bufs.items():
        if root in grouped:
            continue
        start = reserved + offsets[root]
        segment = Segment(start, start + buf.size, MemoryLevel.SRAM, root)
        for tile in buf.tiles:
            tile.meta.setdefault("scratchpad", segment)

    for key, (region_bytes, members) in layouts.items():
        base = reserved + offsets[("bank_group", key)]
        for root, (offset, payload, slots) in members.items():
            start = base + offset
            end = start + payload + (slots - 1) * region_bytes
            segment = Segment(start, end, MemoryLevel.SRAM, root)
            for tile in bufs[root].tiles:
                tile.meta.setdefault("scratchpad", segment)
                if _slots(tile):
                    tile.meta["bank_group_stride"] = region_bytes

    return int(total) + reserved, len(requests), len(groups)


def _peak_live_buffers(bufs: Dict[Node, _Buf]) -> Tuple[int, List[_Buf]]:
    """The Scratchpad buffers live at the busiest schedule step.

    Returns:
        ``(their summed bytes, the buffers largest first)``.
    """
    peak_bytes = 0
    peak_live: List[_Buf] = []
    births = sorted({buf.birth for buf in bufs.values()})
    for step in births:
        live = [buf for buf in bufs.values() if buf.birth <= step <= buf.death]
        live_bytes = sum(buf.size for buf in live)
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_live = live
    return peak_bytes, sorted(peak_live, key=lambda buf: -buf.size)


def _capacity_error(bufs: Dict[Node, _Buf], needed: int, config) -> ValueError:
    """The error for a Scratchpad plan of ``needed`` bytes that exceeds
    ``config.scratchpad_size``, listing the largest buffers live at the
    peak."""
    peak_bytes, live = _peak_live_buffers(bufs)
    lines = []
    for buf in live[:_SHOWN_PEAK_BUFFERS]:
        tile = buf.largest
        value = _val(tile)
        if value is not None:
            dtype = tile.meta.get("dtype") or value.dtype
            shape = "x".join(str(d) for d in value.shape)
        else:
            dtype = tile.meta.get("dtype") or "?"
            shape = "?"
        lines.append(
            f"    {tile.name}<{shape}x{dtype}>  {buf.size} B  "
            f"[def={buf.birth} last={buf.death}]"
        )
    hidden = len(live) - _SHOWN_PEAK_BUFFERS
    if hidden > 0:
        lines.append(f"    ... (+{hidden} more)")
    detail = "\n".join(lines)
    reserved = config.scratchpad_offset
    if reserved:
        note = f" ({reserved} B of it reserved by scratchpad_offset)"
    else:
        note = ""
    return ValueError(
        f"[plan_memory] scratchpad plan needs {needed} bytes{note} > "
        f"scratchpad_size {config.scratchpad_size}; peak concurrency "
        f"{peak_bytes} B across {len(live)} live scratchpad buffers:"
        f"\n{detail}"
    )


def _stamp_slots(order: Dict[Node, int], config) -> None:
    """Record each pipelined ``alloc``'s depth and slot pitch,
    ``meta['slot_count']`` and ``meta['slot_stride']``.

    The code generator serializes the slot dimension as ``bank_count`` /
    ``bank_stride_bytes`` rather than as a tensor dimension: slot ``s`` sits
    at ``base + s * slot_stride``, so a runtime slot index stays a runtime
    value. A bank-grouped buffer's slots step by its group's slot region
    (``meta['bank_group_stride']``), any other by its own payload.
    """
    for node in order:
        slots = _slots(node)
        if not slots:
            continue
        stride = node.meta.get("bank_group_stride")
        if not stride:
            stride = _slot_stride(node, config)
        node.meta["slot_count"] = slots
        node.meta["slot_stride"] = stride


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------


class _Live(NamedTuple):
    """A placed buffer for ``_check_overlaps``: live over steps ``[birth,
    death]`` at ``segment``, in bank group ``group`` (``None``: none)."""

    name: str
    birth: int
    death: int
    segment: Segment
    group: Optional[tuple]


def _check_overlaps(arena: str, buffers: List[_Live]) -> None:
    """Warn for every two buffers live at once whose address ranges
    overlap, in ``buffers`` order. Two members of one bank group are exempt:
    their slot-interleaved segments overlap by construction."""
    by_address = sorted(
        range(len(buffers)), key=lambda i: buffers[i].segment.start
    )
    clashes = []
    for rank, i in enumerate(by_address):
        first = buffers[i]
        for later in range(rank + 1, len(by_address)):
            j = by_address[later]
            second = buffers[j]
            # Later buffers start no lower, so none can reach into ``first``.
            if second.segment.start >= first.segment.end:
                break
            if first.group is not None and first.group == second.group:
                continue
            if (
                first.birth <= second.death
                and second.birth <= first.death
                and first.segment.start < second.segment.end
            ):
                clashes.append((min(i, j), max(i, j)))
    for i, j in sorted(clashes):
        first = buffers[i]
        second = buffers[j]
        logger.warning(
            "[MEM_OVERLAP] %s %s [%d,%d) and %s [%d,%d) overlap",
            arena,
            first.name,
            first.segment.start,
            first.segment.end,
            second.name,
            second.segment.start,
            second.segment.end,
        )


def _check_invariants(
    positions: Dict[Node, int], bufs: Dict[Node, _Buf]
) -> None:
    """Warn if two buffers live at once share an address range, per arena.

    A top-level DRAM tensor lives from its node to its last direct reader; a
    Scratchpad buffer over its ``_Buf`` lifetime.
    """
    dram = []
    for node, step in positions.items():
        segment = node.meta.get("memory")
        if (
            segment is None
            or node.meta.get("space") != "DRAM"
            or _val(node) is None
        ):
            continue
        readers = [positions[user] for user in node.users if user in positions]
        last = max([step] + readers)
        dram.append(_Live(node.name, step, last, segment, None))
    _check_overlaps("DRAM", dram)

    scratchpad = []
    for root, buf in bufs.items():
        segments = []
        for tile in buf.tiles:
            if "scratchpad" in tile.meta:
                segments.append(tile.meta["scratchpad"])
        if segments:
            live = _Live(
                root.name, buf.birth, buf.death, segments[0], buf.group
            )
            scratchpad.append(live)
    _check_overlaps("Scratchpad", scratchpad)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plan_memory(
    model: GraphModule,
    config,
    dram_layout: Optional[Dict[str, Tuple[int, int]]] = None,
) -> MemoryPlan:
    """Assign concrete DRAM / Scratchpad addresses to a bufferized FX graph.

    Writes ``meta['memory']`` (DRAM) / ``meta['scratchpad']`` (Scratchpad)
    ``Segment``s on each buffer root, the ``alloc`` that owns the storage,
    plus ``meta['slot_count']`` / ``meta['slot_stride']`` on a pipelined one
    and ``meta['counter']`` (the hardware counter id) on each counter
    ``alloc``. A tile carries no address: it is named by its buffer (and,
    for a slot, a runtime slot index). ``config.scratchpad_size`` is the
    whole physical scratchpad, ping-pong slots included, and the plan is
    checked against it directly; with no size, the Scratchpad is not
    planned.

    Args:
        model: The bufferized graph.
        config: The accelerator.
        dram_layout: The DRAM ranges this program shares with others:
            inputs by name (``shared_dram_layout``) and ``get_attr``
            buffers by target (``planned_buffer_layout``).

    Returns:
        The DRAM and Scratchpad bytes used.

    Raises:
        ValueError: If the Scratchpad plan exceeds ``scratchpad_size``,
            listing the buffers live at the peak: the tile search has to
            pick smaller tiles.
    """
    # Both arenas need these: a buffer dies at the last read of any name.
    buffer_of = _buffer_identity(model)
    order = _timestamps(model)
    positions = {node: i for i, node in enumerate(model.graph.nodes)}

    dram_bytes = _plan_dram(
        model, buffer_of, positions, config, dram_layout or {}
    )
    bufs = _buffer_lifetimes(order, buffer_of, config)
    counters = _plan_counters(order, buffer_of)

    capacity = config.scratchpad_size
    scratchpad_bytes = 0
    blocks = 0
    groups = 0
    if capacity is not None:
        scratchpad_bytes, blocks, groups = _plan_scratchpad(bufs, config)

    _stamp_slots(order, config)
    _check_invariants(positions, bufs)

    if capacity is not None and scratchpad_bytes > capacity:
        raise _capacity_error(bufs, scratchpad_bytes, config)

    logger.info(
        "Memory plan: DRAM=%d bytes, Scratchpad=%d/%s bytes (%d blocks, "
        "%d bank groups), %d counters",
        dram_bytes,
        scratchpad_bytes,
        capacity,
        blocks,
        groups,
        counters,
    )
    return MemoryPlan(dram_bytes, scratchpad_bytes)
