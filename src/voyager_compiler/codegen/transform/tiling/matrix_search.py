"""The tile search of a GEMM / conv.

The bufferization builders (``build_gemm`` / ``build_conv2d``) call
``get_tiling`` on their anchor op to get the per-dim tile *counts* —
preferring the anchor's ``l2_tiling`` meta, else searching.  A reduction count
greater than 1 is what drives the ``PipelinedKernel``'s ``num_k`` accumulation
loop.

The ``Tiler``'s runtime model prepares each op (``RuntimeModel.prepare``): a
GEMM it maps onto a ``MatrixSpace`` is searched by interstellar, through an
adapter that answers interstellar's callbacks from the op's ``fit`` and
``cost``, several at once in forked workers (``prefetch_tilings``); one it
gives a ``GridSpace`` is searched by ``gemv_op_tiling``.  The winner is
stamped as ``meta["tile_result"]``.
"""

import gc
import logging
import math
import multiprocessing
import operator
import os
import time
from dataclasses import dataclass, replace

import interstellar
from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_gemm_op,
)
from voyager_compiler.codegen.transform.tiling.grid_search import gemv_op_tiling
from voyager_compiler.codegen.transform.tiling.lowering import (
    matrix_layer,
    mha_head,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    GridSpace,
    MatrixSpace,
    TileResult,
)
from voyager_compiler.codegen.transform.tiling.tiler import TileConstraint
from voyager_compiler.shape_prop import ShapeProp

logger = logging.getLogger(__name__)
le = interstellar.le


# The non-reduction L3 loops a builder's grid may permute, outermost to
# innermost, in the order it emits when nothing says otherwise.  The reduction
# is always innermost (the kernels accumulate in place) and the gemm batch dims
# are always outermost, so neither appears here.
GEMM_L3_ORDER = ("M", "N")
CONV_L3_ORDER = ("K", "Y", "X")

# The interstellar loop each of those tags maps onto.  A gemm is modelled as a
# 1x1 conv, so M rides on OX and N on OC.
_GEMM_LOOP = {"M": le.OX, "N": le.OC}
_CONV_LOOP = {"K": le.OC, "Y": le.OY, "X": le.OX}


@dataclass
class _Search:
    """One GEMM / conv's interstellar search, prepared and ready to run.

    Every step that reads the FX node happened while it was built, so
    ``_run_search`` needs only ``interstellar`` and the plain values ``op``
    holds.  It must stay that way: ``prefetch_tilings`` runs it in a forked
    worker, where dispatching a torch op deadlocks.

    Attributes:
        name: The anchor node's name, for logging.
        op: The anchor's ``OpModel``; its ``space`` is a ``MatrixSpace``.
        layer: The interstellar ``Layer`` to map.
        constraint: The ``TileConstraint`` the search runs under, or
            ``None``.
        tolerance: How much slower than the fastest a mapping may be and
            still win on energy.
    """

    name: str
    op: object
    layer: object
    constraint: object
    tolerance: float

    @property
    def key(self):
        """What the result depends on, which keys the tiler's cache: the
        op's key and the constraint."""
        return self.op.key, self.constraint


def _prepare_search(node, tiler, constraint):
    """The tile search of a GEMM / conv ``node``, bare or a fused
    ``call_module``, prepared in the parent.

    Propagates shapes through the fused submodule and copies its inputs'
    dtypes onto its placeholders -- what the builders rely on too -- then has
    the runtime model prepare the anchor (``RuntimeModel.prepare``).  A
    fully-connected GEMM skips that preamble: its grid search
    (``gemv_op_tiling``) reads the shapes fusion propagated.  An op
    interstellar maps also gets its ``Layer`` and its :class:`_Search`.

    ``constraint`` (a :class:`TileConstraint`) pins loop extents the search
    vetoes against -- the caller's, merged with the head alignment an MHA
    relayout demands -- and keys the result apart from the free search's.

    Returns:
        ``(op, search)``: the anchor's ``OpModel``, and its ``_Search`` when
        its space is a ``MatrixSpace``, else ``None``.  ``None`` instead when
        no search applies: not a GEMM / conv, an anchor that carries an
        ``l2_tiling``, a depthwise conv, or one the model leaves untiled.
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor) or anchor.meta.get("l2_tiling") is not None:
        return None

    sub_gm = node.meta.get("submodule")
    if sub_gm is not None and not is_fully_connected(anchor):
        ShapeProp(sub_gm).propagate(
            *(n.value.clone() for n in node.all_input_nodes)
        )
        dtypes = [n.meta.get("dtype") for n in node.all_input_nodes]
        phs = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]
        for i, ph in enumerate(phs):
            ph.meta["dtype"] = dtypes[i]

    if is_depthwise_conv(anchor):
        return None
    op = tiler.model.prepare(node)
    if op is None:
        return None
    if not isinstance(op.space, MatrixSpace):
        return op, None

    # A projection GEMM feeding an MHA relayout must tile OC on whole heads
    # else _detect_mha_relayout can't store the tile: that joins whatever
    # constraint the caller registered.
    head = mha_head(node)
    if head is not None:
        constraint = TileConstraint(multiple=((le.OC, head),)).merged(
            constraint
        )

    layer = matrix_layer(anchor)
    logger.info(
        f"[interstellar] {anchor.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil} | "
        f"bs={anchor.kwargs.get('block_size')}"
    )
    return op, _Search(
        anchor.name, op, layer, constraint, tiler.runtime_tolerance
    )


# Interstellar's ``size_fn`` answers for a tile that fits and one that does
# not: no bytes, which every capacity holds, and more than any holds.
_FITS = (0.0, 0.0, 0.0)
_VETO = (math.inf,) * 3


def _level_totals(mapping):
    """Per loop, per level: ``mapping``'s blocking times its partitioning,
    the level's steps and copies together.  The point interstellar checks a
    blocking's fit on and every mapping of that blocking have the same
    totals."""
    return tuple(
        tuple(map(operator.mul, blocking, partitioning))
        for blocking, partitioning in zip(
            mapping.loop_blockings, mapping.loop_partitionings
        )
    )


def _run_search(search):
    """The ``MappingPoint`` interstellar picks for ``search``.

    Interstellar is driven through its own contract.  It sizes the scratchpad
    level (L2) of every blocking before it partitions the blocking and orders
    the loops, through ``layer.size_fn``, which answers ``_FITS`` or ``_VETO``
    from the op's ``fit`` of interstellar's own point -- its capacity check then
    holds exactly when the model's does -- and it prices each full mapping
    through its runtime callback, the op's ``cost``.  The constraint vetoes a
    tile off its L2 extents before ``fit`` is asked.  The fit reads only a
    blocking's per-level totals (``_level_totals``) -- above L0, which no matrix
    space partitions, they are its blockings -- so every partitioning and loop
    order of the blocking takes its answer.  Interstellar prices those right
    after it checks the blocking, so a mapping takes the last fit's answer; one
    whose totals differ is fit afresh.

    Dispatches no torch op, so it is safe in a forked worker (:class:`_Search`).

    Raises:
        ValueError: The schedule does not pin the reduction innermost at L3.
        RuntimeError: No tiling fits on chip.
    """
    op, constraint = search.op, search.constraint
    space = op.space
    # The builders accumulate a split reduction across consecutive grid
    # steps, so the schedule has to pin IC innermost at L3.
    ic_hint = space.schedule.schedule_hint[le.IC][3]
    if ic_hint is None or ic_hint[0] != 0:
        raise ValueError(
            f"{search.name}: the schedule does not pin the L3 reduction "
            "innermost, so the builders could not emit its nest"
        )

    last = None, None  # the blocking fit last: its totals, its Placement

    def fit(mapping):
        """``op.fit`` of ``mapping``, vetoed off the constraint by its L2
        extents, kept as ``last``."""
        nonlocal last
        totals = _level_totals(mapping)
        allowed = (
            constraint is None
            or constraint.allows(lambda loop: math.prod(totals[loop][:3]))
        )
        last = totals, (op.fit(mapping) if allowed else None)
        return last[1]

    def size_fn(point, *_):
        """``Layer.size_fn``: whether ``point``'s blocking fits.  Interstellar
        asks it only at L2, before it partitions the blocking
        (``opt_valid_blocking``), so ``point`` has no loop orders and
        partitionings of 1.  The other arguments are interstellar's
        contract."""
        return _VETO if fit(point) is None else _FITS

    def runtime(architecture, layer, mapping):
        """Interstellar's runtime callback: the latency of a full mapping,
        laid out as its blocking's fit placed it.  ``architecture`` and
        ``layer`` are its contract."""
        totals, placement = last
        if _level_totals(mapping) != totals:
            placement = fit(mapping)
        return op.cost(mapping, placement).latency

    search.layer.size_fn = size_fn
    try:
        result = interstellar.optimizer.opt_optimizer(
            space.architecture,
            search.layer,
            space.schedule,
            runtime,
            verbose=False,
            runtime_tolerance=search.tolerance,
        )
    except AssertionError as e:
        # The optimizer reports "nothing fits" with a bare assert, so match
        # it narrowly: every other AssertionError in interstellar is a real
        # invariant break.
        if "No valid mapping point found" not in str(e):
            raise
        raise RuntimeError(f"{search.name}: no tiling fits on chip") from e
    _, _, mapping, _ = result
    return mapping


def _finish_search(search, mapping):
    """``search``'s winning ``mapping`` as a ``TileResult``, finished in the
    parent: the mapping, the placement its blocking takes, its cost, and
    interstellar's per-level ``(input, output, weight)`` access counts, which
    the ``Tiling`` proto reports."""
    op = search.op
    placement = op.fit(mapping)
    cost = op.cost(mapping, placement)

    blockings = mapping.loop_blockings
    for level in (1, 2, 3):
        logger.info(
            f"[interstellar] {search.name} L{level} tiles: "
            f"IC={blockings[le.IC][level]} OC={blockings[le.OC][level]} "
            f"OX={blockings[le.OX][level]} OY={blockings[le.OY][level]} "
            f"ON={blockings[le.ON][level]}"
        )
    logger.info(
        f"[interstellar] {search.name} estimated runtime: {cost.latency}"
    )
    logger.info(interstellar.utils.format_tiling(mapping))

    _, _, access_list = interstellar.cost_model.get_cost(
        op.space.architecture, mapping, search.layer
    )
    return TileResult(mapping, placement, cost, tuple(access_list))


# Prepared in the parent before forking; a worker reads its job by index so
# only the index crosses the process boundary (the searches are inherited).
_PREFETCH_JOBS = []

# How long the pool may take before the serial path takes the work back; the
# searches carry no deadline of their own.
PREFETCH_TIMEOUT_S = 300.0


def _run_prefetch_job(index):
    """Run one prefetched search in a forked worker.  Returns ``(ok, found)``
    so a failure leaves the entry uncached and is re-raised by the serial path,
    where it carries its normal traceback."""
    try:
        return True, _run_search(_PREFETCH_JOBS[index])
    except Exception:
        return False, None


# ``_search_in_pool``'s result for a search still running at the deadline,
# as opposed to ``None`` for one that raised.
SEARCH_TIMED_OUT = object()


def _search_in_pool(searches):
    """Run ``searches`` concurrently in forked workers.

    Only the search is forked out; each :class:`_Search` was prepared in the
    parent, and a worker reads its own from what it inherited rather than
    having it marshalled.  The deadline is on the pool, not on any one search:
    whatever has finished when it expires is kept.

    Returns:
        One ``_run_search`` result per search, in order -- ``None`` for one
        that raised (nothing fits, or a worker died), ``SEARCH_TIMED_OUT``
        for one that missed the deadline.
    """
    global _PREFETCH_JOBS

    if not searches:
        return []
    _PREFETCH_JOBS = searches
    start = time.perf_counter()
    # Small cap: the runner already forks per design point, so an uncapped
    # pool multiplies to jobs x cpu_count.  VOYAGER_TILING_JOBS overrides.
    workers = min(
        len(searches),
        int(os.environ.get("VOYAGER_TILING_JOBS", "4")),
        os.cpu_count() or 1,
    )
    # Keep the parent heap out of the workers' GC so fork stays copy-on-write.
    gc.freeze()
    pool = None
    # One slot per search; a slot still empty when the deadline expires is
    # left for the serial path.
    results = [SEARCH_TIMED_OUT] * len(searches)
    try:
        context = multiprocessing.get_context("fork")
        pool = context.Pool(workers)
        pending = [
            pool.apply_async(_run_prefetch_job, (index,))
            for index in range(len(searches))
        ]
        deadline = start + PREFETCH_TIMEOUT_S
        for index, job in enumerate(pending):
            # Collected one at a time rather than as one map: past the
            # deadline a zero timeout still hands back every job already
            # finished, so a search that overruns costs only itself.
            try:
                results[index] = job.get(
                    max(0.0, deadline - time.perf_counter())
                )
            except multiprocessing.TimeoutError:
                continue
            except Exception as e:  # a worker died outright
                results[index] = None
                logger.warning("[tiling] prefetched search failed (%s)", e)
    except Exception as e:
        logger.warning(
            "[tiling] parallel prefetch failed (%s); going serial", e
        )
    finally:
        if pool is not None:
            # Terminate, not close: a worker still running on timeout has to
            # go with the pool.
            pool.terminate()
            pool.join()
        gc.unfreeze()
        _PREFETCH_JOBS = []
    return [
        (
            r
            if r is SEARCH_TIMED_OUT
            else (r[1] if r is not None and r[0] else None)
        )
        for r in results
    ]


def prefetch_tilings(nodes, tiler):
    """Map every node's GEMM / conv up front, concurrently, into
    ``tiler.cache``.

    The nodes are read here, in the parent (see :class:`_Search`); only the
    searches go to the pool (``_search_in_pool``), each under the constraint
    ``plan_csr_slices`` may have registered for its anchor.  An op the model
    gives a ``GridSpace`` is searched when it is built.  A key that does
    not match the one ``get_tiling`` recomputes during the build simply
    misses and is redone serially — a stale key costs time, never
    correctness, and the same holds for a search the deadline passes over.
    """
    jobs = {}
    for node in nodes:
        prepared = _prepare_search(
            node, tiler, tiler.constraints.get(get_anchor_node(node))
        )
        if prepared is None:
            continue
        _, search = prepared
        if search is not None and search.key not in tiler.cache:
            jobs[search.key] = search

    if len(jobs) < 2:
        return

    keys = list(jobs)
    searches = [jobs[k] for k in keys]
    start = time.perf_counter()
    results = _search_in_pool(searches)

    cached = 0
    for key, search, found in zip(keys, searches, results):
        if found is None or found is SEARCH_TIMED_OUT:
            continue
        try:
            tiler.cache[key] = _finish_search(search, found)
        except Exception:  # redone, and re-raised, by the serial path
            continue
        cached += 1
    logger.info(
        "[tiling] prefetched %d/%d mappings in %.2fs",
        cached,
        len(searches),
        time.perf_counter() - start,
    )
    if cached < len(searches):
        logger.warning(
            "[tiling] %d mapping(s) did not finish in %.0fs; going serial",
            len(searches) - cached,
            PREFETCH_TIMEOUT_S,
        )


def search_tilings(pairs, tiler):
    """Each ``(node, constraint)`` pair's ``TileResult``, searched together.

    How CSR planning prices a GEMM under each constraint it could be handed.
    A GEMV is searched inline.  The GEMM / conv searches, one per key -- the
    first pair's -- go through the fork pool together when there are two or
    more, and a lone one runs here, serially, as does a search the pool's
    deadline cut off.  Each mapping is finished in the parent and cached, so
    the builders' own ``get_tiling`` calls find it.

    Returns:
        Each pair's result, ``None`` when no tiling fits; a pair whose node
        no search maps (``_prepare_search``) is left out.
    """
    results, pending = {}, {}
    for node, constraint in pairs:
        prepared = _prepare_search(node, tiler, constraint)
        if prepared is None:
            continue
        op, search = prepared
        if search is None:
            try:
                gemv_op_tiling(node, op, constraint)
            except RuntimeError:  # no tiling fits
                results[node, constraint] = None
            else:
                results[node, constraint] = node.meta["tile_result"]
            continue
        users = pending.setdefault(search.key, (search, []))[1]
        users.append((node, constraint))

    def run_serially(search):
        """``_run_search`` in this process, ``None`` when no tiling fits;
        any other error propagates with its traceback."""
        try:
            return _run_search(search)
        except RuntimeError:
            return None

    keys = list(pending)
    searches = [pending[key][0] for key in keys]
    if len(searches) > 1:
        found = _search_in_pool(searches)
    else:
        found = [run_serially(search) for search in searches]
    for key, mapping in zip(keys, found):
        search, users = pending[key]
        if mapping is SEARCH_TIMED_OUT:
            mapping = run_serially(search)
        result = None
        if mapping is not None:
            result = tiler.cache[key] = _finish_search(search, mapping)
        for pair in users:
            results[pair] = result
    return results


def _l3_order_from_mapping(mapping, canonical, loop_of):
    """``canonical`` resorted into the L3 loop order ``mapping`` chose.

    ``loop_orders[d][3]`` counts from the innermost (0), so sorting descending
    puts the outermost first, which is how the builders read a grid.  A loop
    that is empty at L3 carries interstellar's ``le.NUM - 1`` sentinel rather
    than a real rank, so ordering it would be meaningless -- those tags keep
    their canonical slot and only the genuinely tiled ones are permuted.  A
    size-1 grid dim is inert anyway: the scheduler skips it when it looks for
    the innermost tiled dim.
    """
    orders, blockings = mapping.loop_orders, mapping.loop_blockings
    tiled = [t for t in canonical if blockings[loop_of[t]][3] > 1]
    ranked = iter(sorted(tiled, key=lambda t: -orders[loop_of[t]][3]))
    return tuple(next(ranked) if t in tiled else t for t in canonical)


def get_tiling(node, tiler=None):
    """``(counts, l3_order)`` for a GEMM/conv ``node`` (standalone or fused
    ``call_module``); ``counts`` is ``None`` for a node that is not a matrix op
    / is untiled / was skipped.

    ``counts`` holds the per-dim tile counts: conv -> ``(n_y, n_x, n_k, n_c)``;
    gemm -> ``(batch.., n_m, n_n, n_k)`` — the output-spatial / M / N counts
    plus the reduction count last (``n_c`` for conv, ``n_k`` for gemm; the
    builder's ``num_k``).  The builder derives the tile sizes as
    ``full_dim // count``.

    ``l3_order`` permutes the builder's non-reduction grid dims, outermost to
    innermost (a permutation of ``CONV_L3_ORDER`` / ``GEMM_L3_ORDER``);
    ``None`` asks for the canonical order.

    Prefers the anchor's ``l2_tiling`` (the attention builders' explicit
    output-dim factors; the reduction is kept whole / decomposed away, so its
    factor is 1).  ``l2_tiling`` may carry the reduction factor explicitly — a
    3-tuple gemm ``(n_m, n_n, n_k)`` / a 5-tuple conv ``(n_N, n_k, n_y, n_x,
    n_c)`` — to drive a ``num_k > 1`` reduction sweep.  Otherwise the runtime
    model routes the op: one it gives a ``GridSpace`` is searched by
    ``gemv_op_tiling``, which stamps its result on ``node``; everything else
    is searched by interstellar via ``tiler``, caching each search, and the
    result is stamped on the anchor, which the builders copy onto the nest
    they build.  A cache hit keeps the cached tiling and placement, priced by
    this node's own op.  Neither search tiles the leading batch dims (e.g.
    attention heads); the builder loops them, so their counts are the full
    extent (one tile per batch element).
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor):
        return None, None
    is_conv = is_conv2d(anchor)

    # Neither search tiles the leading batch dims (e.g. attention heads); the
    # builder loops them, so emit a full-extent count -- one tile per batch
    # element.
    batch_dims = tuple(anchor.value.shape[: anchor.value.ndim - 2])

    if (tiling := anchor.meta.get("l2_tiling")) is not None:
        logger.debug(f"Found {anchor.name} tiling: {tiling}")
        if is_conv:
            if len(tiling) not in (4, 5):
                raise ValueError(
                    f"{anchor.name} tiling {tiling} must be 4 or 5 elements"
                )
            _, nk, ny, nx, *nc = tiling
            nc = nc[0] if nc else 1
            return (ny, nx, nk, nc), None
        if len(tiling) not in (2, 3):
            raise ValueError(
                f"{anchor.name} tiling {tiling} must be 2 or 3 elements"
            )
        nm, nn, *nk = tiling
        nk = nk[0] if nk else 1
        return batch_dims + (nm, nn, nk), None

    if tiler is None:
        return None, None

    constraint = tiler.constraints.get(anchor)
    prepared = _prepare_search(node, tiler, constraint)
    if prepared is None:
        return None, None
    op, search = prepared
    if isinstance(op.space, GridSpace):
        return batch_dims + gemv_op_tiling(node, op, constraint), None

    if search.key in tiler.cache:
        cached = tiler.cache[search.key]
        result = replace(cached, cost=op.cost(cached.tiling, cached.placement))
        logger.debug(
            "[tiling] %s: mapping cache hit (%d entries)",
            anchor.name,
            len(tiler.cache),
        )
    else:
        logger.info("[tiling] %s: running interstellar", anchor.name)
        t0 = time.perf_counter()
        result = _finish_search(search, _run_search(search))
        logger.info(
            "[tiling] %s: interstellar took %.2fs",
            anchor.name,
            time.perf_counter() - t0,
        )
        tiler.cache[search.key] = result
    anchor.meta["tile_result"] = result

    mapping = result.tiling
    blockings = mapping.loop_blockings  # [loop][3]: the loop's DRAM tiles
    if is_conv:
        order = _l3_order_from_mapping(mapping, CONV_L3_ORDER, _CONV_LOOP)
        loops = (le.OY, le.OX, le.OC, le.IC)
        return tuple(blockings[loop][3] for loop in loops), order
    order = _l3_order_from_mapping(mapping, GEMM_L3_ORDER, _GEMM_LOOP)
    loops = (le.OX, le.OC, le.IC)
    return batch_dims + tuple(blockings[loop][3] for loop in loops), order
