"""How the shared builders walk an interstellar mapping's tile grid.

The GEMM / conv builders emit the L3 grid in the mapping's loop order and load
an operand only when its block changes, whatever hardware runs the kernel, so
every runtime model counts tiles, steps and fetches the same way.  The
flash-attention builder ends a causal query block's key loop at
``attention_kv_last``, so a model counts its steps by it too.
"""

import math

import interstellar
from voyager_compiler.codegen.node_info import (
    bound_operands,
    get_anchor_node,
    get_node_to_key_map,
    is_full_reduction,
    is_gemm_op,
    reduced_input_grid,
    require_allocation,
    weight_transforms,
)

le = interstellar.le

# The L3 loops each operand's tile spans.  A tile advances only when one of
# them turns, so an operand whose every entry is a single L3 step reads one
# tile for the whole sweep.
INPUT_DIMS = (le.OX, le.OY, le.IC, le.ON)
WEIGHT_DIMS = (le.OC, le.IC, le.FX, le.FY)
OUTPUT_DIMS = (le.OC, le.OY, le.OX, le.ON)


def tile_extent(mapping, loop, level):
    """Elements along ``loop`` in one tile at ``level``: the blockings
    through that level times the PE-array partition."""
    extent = mapping.loop_partitionings[loop][0]
    for blocking in mapping.loop_blockings[loop][1 : level + 1]:
        extent *= blocking
    return extent


def total_extent(mapping, loop, level):
    """Elements along ``loop`` in one tile at ``level``, as interstellar
    sizes the tile: the loop's blockings times its partitionings through
    ``level``.  Unlike ``tile_extent`` it does not depend on how L0 splits
    into steps and copies, so it holds on the point interstellar checks a
    blocking's fit on as on every mapping of that blocking."""
    blockings = mapping.loop_blockings[loop][: level + 1]
    partitionings = mapping.loop_partitionings[loop][: level + 1]
    return math.prod(blockings) * math.prod(partitionings)


def operand_elements(mapping, level, stride):
    """``(input, output, weight)`` elements of one ``level`` tile, the
    level's copies included: interstellar's ``get_if_size``, ``get_of_size``
    and ``get_fl_size``.  The input spans the window its outputs read at
    ``stride``, ``(stride_h, stride_w)``, and an operand the level's copies
    do not share is counted once per copy."""
    sizes = [total_extent(mapping, loop, level) for loop in range(le.NUM)]
    copies = [
        partitioning[level] for partitioning in mapping.loop_partitionings
    ]
    stride_h, stride_w = stride
    width = sizes[le.FX] + (sizes[le.OX] - 1) * stride_w
    height = sizes[le.FY] + (sizes[le.OY] - 1) * stride_h
    inputs = width * height * sizes[le.IC] * sizes[le.ON] * copies[le.OC]
    outputs = (
        sizes[le.OX]
        * sizes[le.OY]
        * sizes[le.OC]
        * sizes[le.ON]
        * copies[le.IC]
        * copies[le.FX]
        * copies[le.FY]
    )
    weights = (
        sizes[le.FX]
        * sizes[le.FY]
        * sizes[le.IC]
        * sizes[le.OC]
        * copies[le.OX]
        * copies[le.OY]
        * copies[le.ON]
    )
    return inputs, outputs, weights


def l3_steps(mapping):
    """Total L3 (DRAM) grid steps, the IC reduction included: with IC
    innermost at L3 the grid is ``(output tiles) x num_k``, one input and
    weight load each.  Stores are ``num_k`` times fewer.
    """
    blockings = mapping.loop_blockings
    steps = 1
    for i in range(le.NUM):
        steps *= blockings[i][3]
    return steps


def l3_loads(mapping, dims):
    """How many times an operand spanning ``dims`` is fetched over the
    sweep.

    Order the nest outermost to innermost and let ``p`` be the position of
    the innermost loop the operand spans.  Every loop inside ``p`` re-reads
    the tile that is already there, so the operand is fetched once per
    iteration of the loops at or outside ``p``.  Ranks come off the mapping
    (``loop_orders[d][3]``, 0 = innermost), the same order the builders
    emit, so the two cannot disagree.

    A loop that is empty at L3 carries the sentinel rank and a blocking of
    1, so it can only ever multiply in as 1 -- including the case where the
    operand spans nothing tiled, which correctly gives a single fetch.
    """
    orders, blockings = mapping.loop_orders, mapping.loop_blockings
    innermost = min(orders[d][3] for d in dims)
    steps = 1
    for d in range(le.NUM):
        if orders[d][3] >= innermost:
            steps *= blockings[d][3]
    return steps


def batch_loads(mapping, dims, batch, distinct):
    """Fetches of an operand spanning ``dims`` over the whole sweep, the
    ``batch`` loop the builder wraps the mapping in included.

    Diced inside a batch step, the operand re-reads in full on every one: the
    next step restarts the sequence, so its first block differs from the last
    one loaded and the guard never fires.  Held whole, the tile survives into
    the next step and only a change of block costs -- ``distinct`` of them
    over the batch, which is fewer than ``batch`` for an operand a group
    shares and 1 for one they all share.
    """
    per_step = l3_loads(mapping, dims) if dims else 1
    return per_step * (batch if per_step > 1 else distinct)


# The trailing grid dims each operand role is diced by, named as the matrix
# search names them (``matrix_search._GEMM_LOOP``: M is ``OX``, N is ``OC``,
# and the reduction is ``IC``).  ``build_gemm`` loops the batch, then M, then
# the output block, then the reduction innermost -- so a role that misses the
# reduction holds its tile across it, and one that misses the output block as
# well holds it for the entire kernel.  Whatever a fused tail brings of its
# own has no role and is diced by the output block, like the output.
def operand_roles(node) -> dict:
    """Each buffer the kernel loads for an anchor operand -> ``(role,
    operand)``, keyed by the outer FX node the memory planner allocates.

    A fused ``call_module``'s operands take their roles from the anchor, which
    reads them through the prologue's expand / transpose / dequantize; the
    kernel loads the placeholder beneath (``weight_transforms``), bound to an
    outer node (``bound_operands``).  ``operand`` is the anchor-side node,
    which keeps the expand ``operand_spans`` prices reuse by.  What the tail
    or prologue brings of its own has no role and takes a bank of its own.
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor):
        return {}
    submod = node.meta.get("submodule")
    bound = bound_operands(node, submod)
    roles = {}
    for n, role in get_node_to_key_map(anchor, bound).items():
        if n is anchor:
            continue
        loaded = n
        if submod is not None and n.graph is submod.graph:
            source = weight_transforms(n)[0]
            loaded = bound.get(source, source)
        roles[loaded] = (role, n)
    # The output is named on the outer node -- what the shape map keys it by --
    # in place of the anchor's own entry, which is inside the submodule.
    roles[node] = ("output", node)
    return roles


GEMV_GRID_DIMS = {
    "input": (le.OX, le.IC),
    "input_scale": (le.OX, le.IC),
    "weight": (le.OC, le.IC),
    "other": (le.OC, le.IC),
    "weight_scale": (le.OC, le.IC),
    "bias": (le.OC,),
}
OUTPUT_GRID_DIMS = (le.OX, le.OC)
_GEMV_DIM_POS = {le.OX: 0, le.OC: 1, le.IC: 2}


def operand_spans(node, dims, nb):
    """Grid dim -> consecutive steps ``node`` holds one tile for, over the dims
    it is diced by.

    Mirrors the specs ``build_gemm`` builds: the operand's own batch dims
    right-align onto the output's, a size-1 one broadcasting (pinned to block
    0, so it is not diced at all) and a repeated one holding its tile for the
    whole group -- eight KV heads reaching an attention matmul as thirty-two
    are addressed as head ``h // 4``, so four consecutive heads read one tile
    and the expand never runs.  ``dims`` names the trailing dims its role is
    diced by.
    """
    shape = tuple(node.shape)
    own = shape[: max(0, len(shape) - 2)]
    repeat = weight_transforms(node)[2]

    spans = {}
    for j, size in enumerate(own):
        g = nb - len(own) + j
        if g >= 0 and size != 1:
            spans[g] = repeat[j] if repeat else 1
    for d in dims:
        spans[nb + _GEMV_DIM_POS[d]] = 1
    return spans


def block_transfers(spans, grid):
    """DMAs an operand issues over the whole grid.

    Every load is guarded on the tile's block index changing
    (``_BufferedRef.copy_in``), so the operand is fetched once per *run* of
    steps addressing the same tile.  The innermost dim it is diced by sets that
    run: every dim inner to it re-reads the tile, and so does a dim it merely
    repeats over.  Diced by nothing, it is read once and held.
    """
    tiled = [g for g in spans if grid[g] > 1]
    if not tiled:
        return 1
    inner = max(tiled)
    return math.prod(grid[:inner]) * max(1, grid[inner] // spans[inner])


def step_classes(dmas, steps):
    """``(cycles, count)`` per class of step, grouped by what reloads on it.

    A DMA issued ``t`` times over ``steps`` steps lands every ``steps // t`` of
    them, so what a step owes is whatever happens to recur on it.  Those
    periods nest -- each is a suffix-product of the grid times a repeat on a
    batch dim -- so the longest one dividing a step fixes what it owes and the
    class sizes come out by subtraction.  Non-nested periods (a repeat
    somewhere other than a batch dim) have no such structure and are counted
    directly.

    Cheapest class first, and the classes always cover every step: a DMA that
    recurs on all of them may be absent, and then the steps nothing lands on
    are a class of their own owing nothing.

    Args:
        dmas: ``(cycles, transfers)`` per operand over the whole sweep.
        steps: Steps to spread them over.

    Returns:
        ``(cycles, count)`` pairs whose counts sum to ``steps``.
    """
    by_period = {1: 0.0}
    for cycles, transfers in dmas:
        period = steps // transfers
        by_period[period] = by_period.get(period, 0.0) + cycles

    periods = sorted(by_period)
    if any(b % a for a, b in zip(periods, periods[1:])):
        counts = {}
        for s in range(steps):
            owed = sum(c for p, c in by_period.items() if s % p == 0)
            counts[owed] = counts.get(owed, 0) + 1
        return sorted(counts.items())

    classes, owed = [], 0.0
    for j, period in enumerate(periods):
        owed += by_period[period]
        inner = steps // periods[j + 1] if j + 1 < len(periods) else 0
        classes.append((owed, steps // period - inner))
    return classes


def sweep_cycles(dmas, steps, compute):
    """Cycles a double-buffered sweep of ``steps`` grid steps takes.

    A guarded load makes the DRAM stream uneven, so ``step_classes`` sorts the
    steps by what reloads on each.  The two engines run concurrently, so a step
    costs whichever of them is busier.  Comparing the *totals* instead would
    let the DRAM idle of a cheap step pay for the load of an expensive one, and
    no amount of buffering can move time backwards like that.

    A tile is loaded while its predecessor computes, so the two engines are
    offset by a step and the ends do not line up: the first load overlaps
    nothing, the step computing tile 0 has no store to make yet, the one
    computing the last tile has no load left to issue, and the last store
    waits on it.  Those four are priced on their own, in place of the two
    steps the per-step charge would have spent on them.

    Args:
        dmas: ``(cycles, transfers)`` per operand over the whole grid, the
            output store first.
        steps: Grid steps the sweep runs.
        compute: Cycles one tile costs the compute engine.

    Returns:
        Cycles the sweep takes, the exposed ends included.
    """
    store = dmas[0][0]
    prologue = sum(cycles for cycles, _ in dmas[1:])
    if steps == 1:
        return prologue + compute + store

    total = sum(
        count * max(owed, compute) for owed, count in step_classes(dmas, steps)
    )

    # ``recur`` is what a step past the first reloads, ``held`` the same without
    # the store; the test is the period ``step_classes`` bucketed them under,
    # so the slots it priced and the ones unbooked below cancel exactly.
    recur = sum(c for c, t in dmas if steps // t == 1)
    held = sum(c for c, t in dmas[1:] if steps // t == 1)
    return (
        total
        - max(prologue + store, compute)  # step 0 is the prologue: no compute
        - max(recur, compute)  # step 1 has no prior tile to store
        + prologue  # prologue fetch, overlapping nothing
        + max(held, compute)  # compute tile 0 while tile 1 loads; no store
        + max(store, compute)  # compute the last tile; no load left to issue
        + store  # the last store, once that compute ends
    )


def vector_transfers(node, tiled_shapes, tiling):
    """Each buffer a vector op's tile sweep moves, and how often.

    The grid is the tile counts over ``node``'s own output, and each input's
    dims right-align onto it, the way the tile shapes were built
    (``compute_tiled_shape``), or follow ``reduced_input_grid`` for a
    reduction that drops its reduced dims -- so a dim an operand lacks, or
    broadcasts over, leaves its tile in place while that loop turns and
    ``block_transfers`` counts the reuse.  A reduction to one value stores it
    once.

    Args:
        node: The op being tiled -- a vector op, or a fused ``call_module``
            around one.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated input.
        tiling: Per-dim tile *count* over ``node``'s output.

    Returns:
        ``(buffer, tile shape, transfers)`` per buffer, the output first --
        the order ``sweep_cycles`` reads its DMAs in.
    """
    num_tiles = math.prod(tiling)
    out_ndim = len(tiling)
    anchor = get_anchor_node(node) or node
    stores = 1 if is_full_reduction(anchor) else num_tiles
    moves = [(node, tiled_shapes[node], stores)]
    # A reduction that drops its reduced dims reads them whole.
    grid = reduced_input_grid(anchor)
    for n, shp in tiled_shapes.items():
        if n is node:
            continue
        if grid is None:
            offset = out_ndim - len(n.shape)
            dims = [
                offset + j if offset + j >= 0 else None
                for j in range(len(n.shape))
            ]
        else:
            dims = grid[len(grid) - len(n.shape) :]
        spans = {
            g: 1
            for g, size in zip(dims, n.shape)
            if g is not None and size != 1
        }
        moves.append((n, shp, block_transfers(spans, tiling)))
    return moves


def gemv_transfers(node, tiled_shapes, tiling):
    """The grid a matrix-vector GEMM's kernel sweeps and each buffer it moves,
    with how often.

    ``tiling`` covers ``(X, C, K)``, but the kernel loops more: the anchor's
    leading batch dims (attention heads) are one grid step each, outside all
    three, and ``build_gemm`` emits them outermost.  Which grid dims dice an
    operand follows from its role (``GEMV_GRID_DIMS``), and that is what
    ``block_transfers`` needs to count its reuse -- a GQA KV tile shared by
    four query heads is re-read once per head as soon as a dim inner to the
    head loop splits.  The output is diced by every grid dim but the
    reduction, which it accumulates over: it is stored on the step that
    finishes it.

    Args:
        node: The op being tiled -- a fully-connected GEMM, or a fused
            ``call_module`` around one.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated activation input.
        tiling: ``(n_x, n_c, n_k)`` -- tile counts over ``(X, C, K)``, ``n_c``
            being the reduction.

    Returns:
        ``(grid, moves)``: the grid ``build_gemm`` emits, outermost first, and
        ``(buffer, tile shape, transfers)`` per buffer, the output first.
    """
    n_x, n_c, n_k = tiling
    anchor = get_anchor_node(node) or node
    batch = tuple(anchor.value.shape[:-2])
    nb = len(batch)
    # ``tiling`` is in the search's ``(X, C, K)`` order, so its last two
    # swap.
    grid = batch + (n_x, n_k, n_c)
    roles = operand_roles(node)

    # The output's dims need no deriving (``build_gemm``'s
    # ``out_index_map``), and a fused ``quantize_mx`` tail makes its shape a
    # *pair*, which ``operand_spans`` could not read anyway.
    stores = block_transfers({g: 1 for g in range(nb + 2)}, grid)
    moves = [(node, tiled_shapes[node], stores)]
    for n, shape in tiled_shapes.items():
        if n is node or shape is None or not require_allocation(n):
            continue
        # The reuse comes off the anchor-side operand, which carries the
        # expand.
        role, operand = roles.get(n, (None, n))
        dims = GEMV_GRID_DIMS.get(role, OUTPUT_GRID_DIMS)
        transfers = block_transfers(operand_spans(operand, dims, nb), grid)
        moves.append((n, shape, transfers))
    return grid, moves


def attention_kv_last(q_block, tq, tkv, sq):
    """The last key block a causal query block attends: the one holding
    the key of its last row.  The rows of query block ``q_block`` sit at
    ``(q_block * tq) % sq`` within their head of ``sq`` rows (the GQA fold
    stacks heads along the rows, a block never straddling one).
    ``q_block`` may be a loop's SymInt."""
    return ((q_block * tq) % sq + tq - 1) // tkv
