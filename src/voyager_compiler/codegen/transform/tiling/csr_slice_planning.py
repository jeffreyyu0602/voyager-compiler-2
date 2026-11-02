"""Slice-width choice for outlier-CSR producers and their consumers.

An outlier CSR is emitted per K slice, and its column indices are local to
the slice, so every GEMM consuming it must tile its reduction dim at exactly
that width.  One number therefore couples each producer to its consumers.
Searched op by op, each consumer would pick the reduction tile that suits
itself, and they would disagree.  Here every consumer is searched under each
candidate width, and each producer takes the width minimizing its consumers'
summed modeled runtime.  The winners are registered with the tiler, so the
builders' own ``get_tiling`` calls land on them, and stamped on each producer
as ``meta["csr_slice"]``.
"""

import logging
import math
import time

import interstellar
from voyager_compiler.codegen.node_info import (
    csr_consumers,
    csr_quantize_node,
    get_anchor_node,
)
from voyager_compiler.codegen.transform.tiling.matrix_search import (
    search_tilings,
)
from voyager_compiler.codegen.transform.tiling.tiler import TileConstraint

logger = logging.getLogger(__name__)
le = interstellar.le

# Narrowest CSR slice a consumer is handed when its reduction dim allows
# one: a step's fixed costs (gather DMAs, systolic fill and drain) dwarf the
# matrix work below this, whatever the rest of the tile.
DEFAULT_MIN_CSR_SLICE = 256


def _slice_candidates(producer, consumers, min_slice):
    """Slice widths ``producer``'s CSR may be cut at.

    Divisors of the consumers' reduction dim that hold a power of two of
    microscaling blocks, no narrower than ``min_slice`` -- or the whole dim
    when it is narrower than that.
    """
    block = csr_quantize_node(producer).args[3]
    depth = get_anchor_node(consumers[0]).args[0].value.shape[-1]
    floor = min(min_slice, depth)
    widths = []
    blocks = 1
    while blocks * block <= depth:
        width = blocks * block
        if depth % width == 0 and width >= floor:
            widths.append(width)
        blocks *= 2
    return widths or [depth]


def _constraint(width):
    """The constraint a consumer searches under when handed a ``width``-wide
    CSR."""
    return TileConstraint(exact=((le.IC, width),))


def _explicit_extent(anchor):
    """``extent(loop)`` of an anchor whose tiles were given outright
    (``meta["l2_tiling"]``), so the planner can tell which widths it admits;
    ``None`` when it carries none."""
    tiling = anchor.meta.get("l2_tiling")
    if tiling is None:
        return None
    n_n = tiling[1]
    n_k = tiling[2] if len(tiling) > 2 else 1
    tiles = {
        le.OC: anchor.value.shape[-1] // n_n,
        le.IC: anchor.args[0].value.shape[-1] // n_k,
    }
    return tiles.__getitem__


def _search_costs(searches, tiler):
    """Modeled runtime of every ``(node, constraint)`` search, in cycles.

    An anchor tiled outright (``l2_tiling``) prices at zero under the
    constraints its tiles satisfy and needs no tiler; the rest are searched
    together (``search_tilings``), which caches each mapping for the
    builders.  A search that fits nothing prices at infinity, and an
    unranked one, which prices nothing, at zero.

    Returns:
        The runtime per ``(node, constraint)``; a node no search maps is
        left out, which ``op_cost`` prices at zero.
    """
    costs, pending = {}, []
    for node, constraint in searches:
        extent = _explicit_extent(get_anchor_node(node))
        if extent is not None:
            costs[node, constraint] = (
                0.0 if constraint.allows(extent) else math.inf
            )
        elif tiler is None:
            costs[node, constraint] = math.inf
        else:
            pending.append((node, constraint))
    for pair, result in search_tilings(pending, tiler).items():
        if result is None:
            costs[pair] = math.inf
        elif result.cost is None:
            costs[pair] = 0.0
        else:
            costs[pair] = result.cost.latency
    for (node, constraint), cost in costs.items():
        logger.debug("[csr] %s under %s: %.0f", node.name, constraint, cost)
    return costs


def plan_csr_slices(nodes, tiler, min_slice=DEFAULT_MIN_CSR_SLICE):
    """Choose every CSR's slice width with the GEMMs that consume it.

    Args:
        nodes: The graph's nodes, before bufferization.
        tiler: The ``Tiler``; receives the winners' constraints and
            their finished searches.  ``None`` when every consumer is
            tiled outright.
        min_slice: Narrowest slice a consumer is handed (see
            ``DEFAULT_MIN_CSR_SLICE``).

    Raises:
        RuntimeError: No candidate width lets every consumer of some CSR
            fit.
    """
    consumers_of = {}
    for node in nodes:
        if csr_quantize_node(node) is None:
            continue
        consumers = csr_consumers(node)
        if consumers:
            consumers_of[node] = consumers
    if not consumers_of:
        return
    candidates = {
        p: _slice_candidates(p, consumers_of[p], min_slice)
        for p in consumers_of
    }
    producer_of = {c: p for p in consumers_of for c in consumers_of[p]}

    consumers = [n for n in nodes if n in producer_of]
    searches = [
        (node, _constraint(width))
        for node in consumers
        for width in candidates[producer_of[node]]
    ]
    start = time.perf_counter()
    costs = _search_costs(searches, tiler)
    logger.info(
        "[csr] %d constrained searches over %d coupled ops in %.1fs",
        len(searches),
        len(consumers) + len(consumers_of),
        time.perf_counter() - start,
    )

    for producer, users in consumers_of.items():
        best, best_width = math.inf, None
        for width in candidates[producer]:
            total = 0.0
            for user in users:
                total += costs.get((user, _constraint(width)), 0.0)
            logger.debug(
                "[csr] %s emitting %s: consumers %.0f",
                producer.name,
                width,
                total,
            )
            if total < best:
                best, best_width = total, width
        if math.isinf(best):
            raise RuntimeError(
                f"{producer.name}: no CSR slice width in "
                f"{candidates[producer]} fits every op coupled to it"
            )
        if tiler is not None:
            for user in users:
                tiler.constraints[get_anchor_node(user)] = _constraint(
                    best_width
                )
        producer.meta["csr_slice"] = best_width
        logger.info(
            "[csr] %s: slice width %d (subtree runtime %.0f)",
            producer.name,
            best_width,
            best,
        )
