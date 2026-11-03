"""Fold view-like glue ops into ``voyager.subview`` windows.

Runs at the end of ``bufferize_graph``, on the bufferized graph: the nests
and their output allocs exist, and emit resolves every reference through
the graph itself (sub-graph placeholders bind positionally), so rewiring
here reaches everything the builders made.

``aten.select`` / ``aten.slice`` / ``aten.unbind`` (read side): when every
dim before the named one has extent 1, the named bytes are one contiguous
run of the source, so the op is pure addressing -- replace it with a
``voyager.subview`` and emit folds the window into each consumer's
``TensorBoxRef``.  ``unbind`` is a ``select`` at every index of its dim, so
each element becomes its own window.  A select on an extent-1 dim and a
whole-extent slice are already nops (``is_nop``); a non-contiguous one
stays an op.

``aten.split_with_sizes`` / ``aten.chunk`` (read side): each piece is a
window of its size at the running offset along the split dim.  A subview
windows every dim, so the pieces need no extent-1 dims ahead of the split
one.

``aten.cat`` / ``aten.stack`` (write side): when every dim before the join
dim has extent 1, each source occupies one contiguous window of the
result -- ``stack`` joining along a dim its sources do not carry, so its
windows are size 1 there and squeeze it away.  Allocate the result once,
then point each source's storage at its window: a source produced by a
nest has its output alloc *replaced* by the window, so the nest stores
straight into the join buffer -- zero copies.  A nest whose output reaches
the join through a reshape keeps storing at its own shape, so it takes the
window reshaped back to it; a source with no alloc to redirect (a
parameter) is host-copied into its window with ``insert(clone(src),
window)``, the sanctioned buffer-to-buffer move.

``aten.copy_`` (write side): a training program ends by writing each
result into a buffer that outlives it -- a gradient, a parameter, an
optimizer moment.  That is a one-source join whose window is the whole
destination: the producing nest's output alloc is redirected into it, so
the nest stores straight into the destination and the copy goes.

``aten.index_copy_`` (write side): a KV-cache store writes one contiguous
run of positions -- ``arange(start, start + n)``, a constant or with a
runtime ``start``, or a one-element ``full`` -- so it is a ``copy_`` into
the cache's window there, and folds like one, in every graph of the model
(a decode fold stores inside a ``cond`` branch).  A source with no alloc to
redirect is host-copied into the window.
"""

import operator

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value, is_nop
from voyager_compiler.codegen.transform.bufferize.ops import MemoryLevel
from voyager_compiler.shape_prop import fetch_attr, set_node_value

_ALLOC = torch.ops.voyager.alloc.default
_ARANGE = (
    torch.ops.aten.arange.default,
    torch.ops.aten.arange.start,
    torch.ops.aten.arange.start_step,
)
_CAT = torch.ops.aten.cat.default
_CLONE = torch.ops.aten.clone.default
_COPY = torch.ops.aten.copy_.default
_FULL = torch.ops.aten.full.default
_INDEX_COPY = torch.ops.aten.index_copy_.default
_INSERT = torch.ops.voyager.insert.default
_RESHAPE = torch.ops.aten.reshape.default
_SELECT = torch.ops.aten.select.int
_SLICE = torch.ops.aten.slice.Tensor
_SPLITS = (
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.chunk.default,
)
_STACK = torch.ops.aten.stack.default
_SUBVIEW = torch.ops.voyager.subview.default
_UNBIND = torch.ops.aten.unbind.int
_WHILE_LOOP = torch.ops.higher_order.while_loop


def lower_views(model: GraphModule) -> GraphModule:
    """Rewrite foldable ``select`` / ``slice`` / ``unbind`` / ``split`` /
    ``cat`` / ``stack`` / ``copy_`` nodes, and every graph's foldable
    ``index_copy_``, into subview windows.  A window
    carries the logical (quantized) dtype of the buffer it views, so its
    bytes are sized like the buffer's."""
    for node in list(model.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is _SELECT:
            _fold_select(model, node)
        elif node.target is _SLICE:
            _fold_slice(model, node)
        elif node.target is _UNBIND:
            _fold_unbind(model, node)
        elif node.target in _SPLITS:
            _fold_split(model, node)
        elif node.target in (_CAT, _STACK):
            _fold_join(model, node)
        elif node.target is _COPY:
            _fold_copy(model, node)
    for gm in model.modules():
        if not isinstance(gm, GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op == "call_function" and node.target is _INDEX_COPY:
                _fold_index_copy(gm, node)
        gm.graph.lint()
    return model


def _leading_unit_dims(shape, dim) -> bool:
    return all(s == 1 for s in shape[:dim])


def _fold_select(model: GraphModule, node: Node) -> None:
    source, dim, index = node.args
    value = getattr(source, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    index = index + shape[dim] if index < 0 else index
    # An extent-1 dim is a pure rename (``is_nop``); a non-unit dim ahead
    # of the selected one makes the bytes several disjoint runs, which one
    # window cannot express.
    if shape[dim] == 1 or not _leading_unit_dims(shape, dim):
        return
    offsets = [0] * len(shape)
    offsets[dim] = index
    sizes = list(shape)
    sizes[dim] = 1
    graph = model.graph
    with graph.inserting_before(node):
        subview = graph.call_function(
            _SUBVIEW,
            (source, offsets, sizes, [1] * len(shape)),
            {"squeeze_dim": [dim]},
        )
    set_node_value(subview, node.value)
    subview.meta["dtype"] = source.meta.get("dtype")
    node.replace_all_uses_with(subview)
    graph.erase_node(node)


def _fold_unbind(model: GraphModule, node: Node) -> None:
    source = node.args[0]
    dim = get_arg_value(node, 1, "dim", 0)
    value = getattr(source, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    # ``unbind`` is a ``select`` at every index of ``dim``, and carries the
    # same rule: a non-unit dim ahead of it makes each element several
    # disjoint runs, which one window cannot express.
    if not _leading_unit_dims(shape, dim):
        return
    # The elements are reached one at a time; an unbind consumed whole (a
    # list handed to another op) names no single window.
    users = list(node.users)
    if any(
        u.op != "call_function" or u.target is not operator.getitem
        for u in users
    ):
        return

    graph = model.graph
    for user in users:
        index = user.args[1]
        index = index + shape[dim] if index < 0 else index
        offsets = [0] * len(shape)
        offsets[dim] = index
        sizes = list(shape)
        sizes[dim] = 1
        with graph.inserting_before(user):
            subview = graph.call_function(
                _SUBVIEW,
                (source, offsets, sizes, [1] * len(shape)),
                {"squeeze_dim": [dim]},
            )
        set_node_value(subview, user.value)
        subview.meta["dtype"] = source.meta.get("dtype")
        user.replace_all_uses_with(subview)
        graph.erase_node(user)
    graph.erase_node(node)


def _fold_split(model: GraphModule, node: Node) -> None:
    source = node.args[0]
    dim = get_arg_value(node, 2, "dim", 0)
    value = getattr(source, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    # The pieces' own extents, so one rule serves explicit sizes and chunks.
    split_sizes = [piece.shape[dim] for piece in node.value]
    # The pieces are reached one at a time; a split consumed whole (a list
    # handed to another op) names no single window.
    users = list(node.users)
    if any(
        u.op != "call_function" or u.target is not operator.getitem
        for u in users
    ):
        return

    graph = model.graph
    for user in users:
        index = user.args[1]
        index = index + len(split_sizes) if index < 0 else index
        offsets = [0] * len(shape)
        offsets[dim] = sum(split_sizes[:index])
        sizes = list(shape)
        sizes[dim] = split_sizes[index]
        with graph.inserting_before(user):
            subview = graph.call_function(
                _SUBVIEW, (source, offsets, sizes, [1] * len(shape))
            )
        set_node_value(subview, user.value)
        subview.meta["dtype"] = source.meta.get("dtype")
        user.replace_all_uses_with(subview)
        graph.erase_node(user)
    graph.erase_node(node)


def _fold_slice(model: GraphModule, node: Node) -> None:
    source = node.args[0]
    dim = get_arg_value(node, 1, "dim", 0)
    start = get_arg_value(node, 2, "start")
    end = get_arg_value(node, 3, "end")
    step = get_arg_value(node, 4, "step", 1)
    value = getattr(source, "value", None)
    # A runtime bound names a window the reference cannot carry statically,
    # and a step skips bytes one window cannot express.
    if not isinstance(value, torch.Tensor) or step != 1:
        return
    if isinstance(start, Node) or isinstance(end, Node):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    extent = shape[dim]
    start = 0 if start is None else (start + extent if start < 0 else start)
    end = extent if end is None else (end + extent if end < 0 else end)
    start = max(0, min(start, extent))
    end = max(start, min(end, extent))
    # A whole-extent slice is already a nop (``is_prunable_op``); a non-unit
    # dim ahead of the sliced one makes the bytes several disjoint runs,
    # which one window cannot express.
    if (start, end) == (0, extent) or not _leading_unit_dims(shape, dim):
        return
    offsets = [0] * len(shape)
    offsets[dim] = start
    sizes = list(shape)
    sizes[dim] = end - start
    graph = model.graph
    with graph.inserting_before(node):
        subview = graph.call_function(
            _SUBVIEW, (source, offsets, sizes, [1] * len(shape))
        )
    set_node_value(subview, node.value)
    subview.meta["dtype"] = source.meta.get("dtype")
    node.replace_all_uses_with(subview)
    graph.erase_node(node)


def _storage_of(node):
    """The builder-created output ``alloc`` whose bytes ``node`` names, walked
    through result handles and views; ``None`` when there is none to redirect
    (a parameter, a window onto part of a buffer -- one an earlier join
    already placed -- or a shape this walk does not cover)."""
    while isinstance(node, Node):
        if node.op != "call_function":
            return None
        if node.target is _ALLOC:
            return node
        if node.target is _SUBVIEW and (
            node.value.numel() != node.args[0].value.numel()
        ):
            return None
        if node.target is operator.getitem:
            src, index = node.args
            if isinstance(src, Node) and src.target is _WHILE_LOOP:
                carried = list(src.args[2])
                if index < len(carried):
                    node = carried[index]
                    continue
            return None
        if node.target is _SUBVIEW or is_nop(node):
            node = node.args[0]
            continue
        return None
    return None


def _fold_join(model: GraphModule, node: Node) -> None:
    stacked = node.target is _STACK
    sources = list(node.args[0])
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    out_shape = list(value.shape)
    rank = len(out_shape)
    dim = dim + rank if dim < 0 else dim
    if not _leading_unit_dims(out_shape, dim):
        return
    values = [getattr(s, "value", None) for s in sources]
    # ``stack`` joins along a dim its sources do not carry.
    src_rank = rank - 1 if stacked else rank
    if any(
        not isinstance(v, torch.Tensor)
        or len(v.shape) != src_rank
        or v.dtype != value.dtype
        for v in values
    ):
        return
    dtypes = {s.meta.get("dtype") for s in sources}
    if len(dtypes) != 1:
        return
    dtype = dtypes.pop()
    # A shared producer cannot land in two windows at once.
    targets = [_storage_of(s) for s in sources]
    redirected = [t for t in targets if t is not None]
    if len(set(redirected)) != len(redirected):
        return

    graph = model.graph
    ordered = list(graph.nodes)
    order = {n: i for i, n in enumerate(ordered)}
    anchors = [t if t is not None else s for t, s in zip(targets, sources)]
    first = min(anchors, key=lambda n: order[n])

    # The cat's storage, alive from the first producer's stores onward.
    # Space rides positionally: ``annotate_tensor_spaces`` reads ``args[2]``.
    with graph.inserting_before(first):
        buf = graph.call_function(
            _ALLOC, (out_shape, value.dtype, int(MemoryLevel.DRAM), 0)
        )
    set_node_value(buf, value)
    buf.meta["dtype"] = dtype
    if (scope := first.meta.get("scope")) is not None:
        buf.meta["scope"] = scope

    offset = 0
    last = buf
    for s, t, v in zip(sources, targets, values):
        offsets = [0] * rank
        offsets[dim] = offset
        if stacked:
            sizes = list(out_shape)
            sizes[dim] = 1
            squeeze = {"squeeze_dim": [dim]}
            offset += 1
        else:
            sizes = list(v.shape)
            squeeze = {}
            offset += v.shape[dim]
        with graph.inserting_after(last):
            window = graph.call_function(
                _SUBVIEW, (buf, offsets, sizes, [1] * rank), squeeze
            )
        set_node_value(window, v)
        window.meta["dtype"] = dtype
        last = window
        if t is not None:
            last = _redirect(graph, t, window, before=last.next)
        else:
            with graph.inserting_before(node):
                clone = graph.call_function(_CLONE, (s,))
                graph.call_function(_INSERT, (clone, window))
            set_node_value(clone, v)
            clone.meta["dtype"] = dtype

    node.replace_all_uses_with(buf)
    graph.erase_node(node)


def _redirect(graph, storage: Node, window: Node, before: Node) -> Node:
    """Point the output alloc ``storage`` at ``window``, so the nest that
    stores into it stores into the window: reshaped back to the alloc's
    shape, inserted before ``before``, when the two differ.  Returns the node
    the nest now stores into."""
    dest = window
    if tuple(storage.value.shape) != tuple(window.value.shape):
        with graph.inserting_before(before):
            dest = graph.call_function(
                _RESHAPE, (window, list(storage.value.shape))
            )
        set_node_value(dest, storage.value)
        dest.meta["dtype"] = window.meta.get("dtype")
    storage.replace_all_uses_with(dest)
    graph.erase_node(storage)
    return dest


def _names_of(buffer: Node) -> list:
    """``buffer`` and every view that names its bytes, transitively."""
    names, frontier = [buffer], [buffer]
    while frontier:
        node = frontier.pop()
        for user in node.users:
            if (user.target is _SUBVIEW or is_nop(user)) and (
                user.args[0] is node
            ):
                names.append(user)
                frontier.append(user)
    return names


def _space(buffer: Node) -> int:
    """The memory level a buffer root lives in: an ``alloc``'s level
    argument; an input or a weight is in DRAM."""
    if buffer.target is _ALLOC:
        return get_arg_value(buffer, 2, "space", int(MemoryLevel.DRAM))
    return int(MemoryLevel.DRAM)


def _fold_copy(model: GraphModule, node: Node) -> None:
    """Make ``copy_(dst, src)``'s producer store into ``dst``.

    Folds when ``src`` is a nest's output of ``dst``'s shape and dtype, in
    the memory ``dst`` lives in, and nothing reads the buffer ``dst`` views,
    under any name, after that nest starts storing.  The nest may read
    ``dst`` itself, as an update does: it reads each tile before storing
    over it.
    """
    dst, src = node.args
    storage = _storage_of(src)
    value = getattr(src, "value", None)
    dst_value = getattr(dst, "value", None)
    root = dst
    while root.target is _SUBVIEW or is_nop(root):
        root = root.args[0]
    if (
        storage is None
        or _space(storage) != _space(root)
        or not isinstance(value, torch.Tensor)
        or not isinstance(dst_value, torch.Tensor)
        or value.shape != dst_value.shape
        or value.dtype != dst_value.dtype
    ):
        return
    graph = model.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    first_store = min(storage.users, key=order.__getitem__)
    if any(
        order[user] > order[first_store]
        for name in _names_of(root)
        for user in name.users
        if user is not node
    ):
        return
    _redirect(graph, storage, dst, before=first_store)
    node.replace_all_uses_with(dst)
    graph.erase_node(node)
    # The views that named the result for the copy are unread now.
    while src is not storage and is_nop(src) and not src.users:
        source = src.args[0]
        graph.erase_node(src)
        src = source


def _run_start(model: GraphModule, index: Node, length: int):
    """The first position of ``index`` when it names ``length`` consecutive
    ascending positions -- an int, or the scalar node a runtime start is --
    else ``None``.  A folded constant is judged by the one call its
    producer graph computes it with, never by its contents; one with no
    producer names no run."""
    if not isinstance(index, Node):
        return None
    if index.op == "get_attr":
        producer = index.meta.get("producer")
        if not isinstance(producer, GraphModule):
            return None
        (output,) = producer.graph.find_nodes(op="output")
        call = output.args[0]
        if not isinstance(call, Node) or call.op != "call_function":
            return None
        numel = fetch_attr(model, index.target).numel()
        return _call_run_start(call, length, numel)
    if index.op != "call_function":
        return None
    return _call_run_start(index, length, index.value.numel())


def _call_run_start(call: Node, length: int, numel: int):
    """The first position ``call`` names when it is an ``arange`` of step 1
    over ``numel`` == ``length`` positions, or a ``full`` naming one, else
    ``None``."""
    if call.target is _FULL and length == 1:
        return call.args[1]
    if call.target not in _ARANGE:
        return None
    bounds = list(call.args)
    if call.target is torch.ops.aten.arange.default:
        bounds = [0, *bounds]
    step = bounds[2] if len(bounds) > 2 else 1
    if step != 1 or numel != length:
        return None
    return bounds[0]


def _in_layer(node) -> bool:
    """Whether scalar ``node`` can be computed inside a layer: a literal, a
    scalar operand of the graph (a ``cond`` branch's), or arithmetic on
    those.  A scalar read out of a tensor is not: the core reads only
    scratchpad cells, and a layer run alone has no other layer's scalars."""
    if not isinstance(node, Node):
        return True
    if isinstance(getattr(node, "value", None), torch.Tensor):
        return False
    if node.op == "placeholder":
        return True
    return node.op == "call_function" and all(
        _in_layer(arg) for arg in node.all_input_nodes
    )


def _scalar_copy(node, before: Node, scope, copies: dict):
    """``node`` recomputed just before ``before``, in its layer (``scope``):
    scalar arithmetic is copied there with the arithmetic it reads, so a run
    of that layer alone computes it.  A scalar read out of a tensor (an
    ``_local_scalar_dense``), a tensor, a ``get_attr``, a placeholder or a
    literal is read where it is: the core reads only scratchpad cells, so
    such a read stays in the prelude the emitter evaluates."""
    if not isinstance(node, Node) or node.op != "call_function":
        return node
    if any(
        isinstance(getattr(arg, "value", None), torch.Tensor)
        for arg in [node, *node.all_input_nodes]
    ):
        return node
    if node not in copies:
        args = [_scalar_copy(a, before, scope, copies) for a in node.args]
        with before.graph.inserting_before(before):
            copy = before.graph.call_function(node.target, tuple(args))
        copy.meta = dict(node.meta)
        if hasattr(node, "value"):
            set_node_value(copy, node.value)
        if scope is not None:
            copy.meta["scope"] = scope
            # Kept apart from the same scalar computed elsewhere.
            copy.meta["layer_local"] = True
        copies[node] = copy
    return copies[node]


def _names_of_buffer(gm: GraphModule, root: Node) -> list:
    """Every name ``root``'s buffer has in ``gm``: each ``get_attr`` of the
    same target, when it is one, and their views."""
    roots = [root]
    if root.op == "get_attr":
        roots = [
            n
            for n in gm.graph.nodes
            if n.op == "get_attr" and n.target == root.target
        ]
    return [name for r in roots for name in _names_of(r)]


def _fold_index_copy(gm: GraphModule, node: Node) -> None:
    """Make ``index_copy_(dst, dim, index, src)`` store into ``dst``'s window
    at ``index``'s positions, when they are one ascending run.

    The window is a ``subview`` of ``dst``, strided where ``dst`` has
    non-unit dims ahead of ``dim``.  ``src``'s producer stores into it, as
    for ``copy_``, when its output alloc has the window's shape and dtype,
    lives where ``dst`` does, and nothing reads ``dst``'s buffer under any
    name between that producer's first store and the copy; otherwise
    ``src`` is host-copied into the window.  A runtime start is a scalar the
    graph computes, recomputed in the storing layer from the tensors it is
    read from, so that layer run alone has it.
    """
    dst, dim, index, src = node.args
    value = getattr(src, "value", None)
    dst_value = getattr(dst, "value", None)
    if not isinstance(value, torch.Tensor) or not isinstance(
        dst_value, torch.Tensor
    ):
        return
    rank = dst_value.dim()
    dim = dim + rank if dim < 0 else dim
    if value.dim() != rank or any(
        value.shape[d] != dst_value.shape[d] for d in range(rank) if d != dim
    ):
        return
    start = _run_start(gm, index, value.shape[dim])
    # A runtime start the storing layer cannot compute stays a host copy.
    if start is None or not _in_layer(start):
        return

    graph = gm.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    root = dst
    while root.target is _SUBVIEW or is_nop(root):
        root = root.args[0]
    storage = _storage_of(src)
    redirect = (
        storage is not None
        and _space(storage) == _space(root)
        and tuple(storage.value.shape) == tuple(value.shape)
        and value.dtype == dst_value.dtype
    )
    if redirect:
        first_store = min(storage.users, key=order.__getitem__)
        redirect = not any(
            order[first_store] < order[user] < order[node]
            for name in _names_of_buffer(gm, root)
            for user in name.users
        )
    # The window stands in the layer that stores into it, just before its
    # first store, and a runtime start is recomputed there.
    anchor = first_store if redirect else node
    scope = anchor.meta.get("scope")
    start = _scalar_copy(start, anchor, scope, {})
    # A ``get_attr`` belongs to no layer: one standing later moves to the top.
    if dst.op == "get_attr" and order[dst] > order[anchor]:
        head = next(n for n in graph.nodes if n.op != "placeholder")
        if head is not dst:
            head.prepend(dst)

    offsets = [0] * rank
    offsets[dim] = start
    with graph.inserting_before(anchor):
        window = graph.call_function(
            _SUBVIEW, (dst, offsets, list(value.shape), [1] * rank)
        )
    set_node_value(window, value)
    window.meta["dtype"] = dst.meta.get("dtype")
    if scope is not None:
        window.meta["scope"] = scope
    if redirect:
        _redirect(graph, storage, window, before=first_store)
    else:
        with graph.inserting_before(node):
            clone = graph.call_function(_CLONE, (src,))
            graph.call_function(_INSERT, (clone, window))
        set_node_value(clone, value)
        clone.meta["dtype"] = src.meta.get("dtype")
    node.replace_all_uses_with(dst)
    graph.erase_node(node)
    # The views that named the result for the copy are unread now.
    while redirect and src is not storage and is_nop(src) and not src.users:
        source = src.args[0]
        graph.erase_node(src)
        src = source
