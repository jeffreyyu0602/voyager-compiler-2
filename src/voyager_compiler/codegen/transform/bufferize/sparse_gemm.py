"""A GEMM whose activation carries an outlier CSR.

``linear_mx`` / ``matmul_mx`` fuse a dense low-precision matmul with a sparse
high-precision correction, and the engine takes **one** CSR per call covering
exactly the dense tile's rows. The producer, though, emits its CSR per
``(row block, K slice)`` and packs the blocks end to end (see
``quantize_mx_outlier``), so a row tile spanning ``R`` row blocks finds its
correction in ``R`` runs scattered through that stream. The consumer therefore
concatenates them on chip before the call.

That concatenation is a valid CSR without re-indexing anything. Column indices
are already local to ``tk``, so a block's columns lie in ``[0, tk)`` whichever
rows they came from, and stacking blocks that share a ``k`` just extends the
row list. The row pointers need no work either: the producer wrote them in
slice coordinates, so the tile's pointer range is a contiguous window of one
array, and the engine subtracts ``indptr[0]`` itself. Placing block ``j``'s
data at ``p[j*rows] - p[0]`` is exactly what makes those pointers address it.

Everything else is the dense GEMM: ``_gemm_plan`` derives the grid, the operand
specs and the per-step call, and this module only adds the fetches and swaps
the CSR kwargs for its own scratch.
"""

import contextlib
import math
from dataclasses import replace
from typing import Optional

import torch
from torch._higher_order_ops.while_loop import while_loop

import interstellar
from voyager_compiler.codegen.node_info import (
    gemm_produces_csr,
    is_nop,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    oracle_disabled,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    _bank_group_list,
    _DEFAULT_NUM_SLOTS,
    _gemm_plan,
    _gemm_scratch_and_kernel,
    _split_stream_break,
    _stamp_anchor_meta,
    _stamp_bank_groups,
    PipelinedKernel,
)
from voyager_compiler.codegen.transform.bufferize.quantize_mx_outlier import (
    _Bufs,
    _entry,
    _Geometry,
    _QUANTIZE_MX_OUTLIER,
    _scalar,
    _split_prefix,
    base_table_shape,
    copy_base_table,
    GEOMETRY_META,
    PRODUCER_META,
    tag_base_table,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    BuildContext,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    effect_cond,
    new_counter,
    outline_dps_ops,
    stamp_csr_fill,
    voyager,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import Role
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import set_node_value, ShapeProp

_SRAM = int(MemoryLevel.SRAM)

# Stores ``_store_staged_csr`` issues per staged sub-slice (the CSR's
# indptr, packed data and packed indices, then the dense scale and
# inliers), each adding 1 to the store stream's counter once landed.
_CSR_STORES = 5


class _EpilogueTail(torch.nn.Module):
    """The fused tail of a CSR-producing GEMM, plus its CSR stores.

    Runs the tail's prefix once on the whole accumulator tile, then the
    quantize once per ``tk``-wide sub-slice of it, so the CSR comes out at
    the *consumers'* slice width while this GEMM's own tiling stays whatever
    its search picked.  Wrapping rather than threading extra arguments keeps
    the tail's arity: the reduction kernel applies it on the last reduction
    step and calls it with the group's own operands.  The offset, the grid
    index and the output slots it needs are captured here instead, which is
    also what puts the stores on the step that actually completes an output
    tile.  The stores below own their subgraph — the rolled sub-slice loop
    — rather than relying on the reduction's last-step ``torch.cond``, which
    a single-round reduction does not emit.

    With ``chain_epilogue`` the tail is pure compute, chained on the live
    total inside the anchor's own pass: all five results land in the staging
    tiles as destinations and the stores run bare afterwards, on the same
    round (``_SparseGemm._kernel``).  Otherwise the stores run here,
    interleaved.
    """

    def __init__(self, owner, idx, scratch=None):
        super().__init__()
        self.owner = owner
        self.idx = idx
        self.scratch = scratch

    def _sub_slice(self, tile, j, bound, tiles):
        """Quantize one sub-slice into the staging tiles, and store it.

        Args:
            tile: The sub-slice of the output tile to quantize.
            j: Its index within the column tile, naming its K slice.
            bound: The tail's operand list, which ``epi_q_args`` indexes.
            tiles: The five staging tiles the results land in.
        """
        o = self.owner

        def val(t):
            kind, v = t
            return bound[v] if kind == "arg" else v

        d, i, p, s, q = _QUANTIZE_MX_OUTLIER(
            tile,
            *[val(t) for t in o.epi_q_args],
            **{k: val(t) for k, t in o.epi_q_kwargs.items()},
            indptr_offset=o._offset_at(self.idx, j),
        )
        for res, tile in zip((d, i, p, s, q), tiles):
            voyager.insert(res, tile)
        if not o.chain_epilogue:
            o._store_staged_csr(self.idx, j)
            # Every sub-slice stored so far, this one included, has to
            # land before the staging tiles are written again.
            stored = o._ordinal(self.idx, o.grid_k) * o.n_sub + j + 1
            o._wait_csr_stores(stored)

    def forward(self, acc, *operands):
        o = self.owner
        og = o.out_geom
        bufs = o._bufs
        nb, ones = og.nb, og.ones
        tiles = (
            bufs.store_data_tile,
            bufs.store_index_tile,
            bufs.store_indptr_tile,
            bufs.store_scale_tile,
            bufs.store_inlier_tile,
        )
        bound = (acc,) + operands

        x = acc if o.epi_prefix is None else o.epi_prefix(*bound)
        if o.n_sub == 1:
            self._sub_slice(x, 0, bound, tiles)
            return None

        # TODO: chain the prefix onto the reduction instead.  It is pure
        # compute, so it could run on the live total of the last K step --
        # the ``split`` head path -- and stage only its result, saving the
        # full read and write of the tile below.  It does not today because
        # ``stream_breaking_quantize`` recognizes only ``quantize_mx``, so
        # an ``_EpilogueTail`` is never split and prefix, quantize and
        # stores stay one unchainable unit.
        if o.epi_prefix is not None:
            if x.dtype != self.scratch.dtype:
                x = x.to(self.scratch.dtype)
            voyager.insert(x, self.scratch)

        def cond_fn(j):
            return j < o.n_sub

        def body_fn(j):
            tile = voyager.subview(
                self.scratch,
                [0] * nb + [0, j * og.tk],
                ones + (og.rows, og.tk),
                (1,) * (nb + 2),
                [],
            )
            if tile.dtype != o.plan.out_dtype:
                tile = tile.to(o.plan.out_dtype)
            self._sub_slice(tile, j, bound, tiles)
            return (j + 1,)

        (j_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(j_end >= 0)
        return None


class _SparseGemm(torch.nn.Module):
    """The dense GEMM nest plus the CSR gather.

    The gather cannot live inside the plan's ``gemm_kernel``, which is
    index-blind by design: a block's source address comes from the shared base
    table, indexed by ``(K slice, row block)``. So it runs here, where the grid
    index is in hand, and the kernel below only substitutes the filled scratch
    for the raw operands.

    A row tile's blocks are the same for every step the sweep nests inside
    its row dim, so a whole-K sweep gathers them once, on the row tile's
    first such step, and reads them from the staging slot for the rest
    (``reuse_gathers``).  A split reduction, or a sweep with nothing inside
    the row dim, changes block with every step and gathers per step.

    The nest stores each tile on its own step (``lag`` 0) and waits every
    operand inline: the gathered blocks, the staged CSR's stores and the
    base table each add 1 to a counter of their own per landed copy, and
    the step that reads or overwrites them waits for the count it needs.
    """

    def __init__(
        self,
        plan,
        geom,
        data_dtype,
        *,
        out_geom=None,
        split_k_tail_fusible: bool,
        single_k_tail_fusible: bool = True,
        num_slots: int = _DEFAULT_NUM_SLOTS,
        accumulate_fp32: bool = False,
    ):
        super().__init__()
        self.plan = plan
        self.accumulate_fp32 = accumulate_fp32
        self.data_dtype = data_dtype
        self.geom = geom
        self.out_geom = out_geom
        self.R = plan.tile_m // geom.rows if geom is not None else 1
        self.span = self.R * geom.budget if geom is not None else 0
        self.grid_m, self.grid_n, self.grid_k = plan.grid_dims
        # Grid dims the sweep nests inside the row dim, ahead of the
        # reduction, and the steps they span: how long a row tile lasts.
        self.inner_dims = [d for d in range(self.grid_k) if d > self.grid_m]
        self.row_steps = math.prod(plan.grid[d] for d in self.inner_dims)
        self.reuse_gathers = (
            geom is not None and plan.num_k == 1 and self.row_steps > 1
        )
        if out_geom is not None:
            # Every output is stored by hand, the dense pair by the same
            # route as the CSR, so the output grid dices none of them.
            plan.out_specs = list(plan.out_specs)[5:]
            # The quantize runs per ``tk``-wide sub-slice of the accumulator
            # tile (see ``_EpilogueTail``): split the tail at the quantize and
            # bind the op's own arguments to their positions in the tail's
            # operand list, so the per-slice calls can be issued directly.
            self.n_sub = plan.tile_n // out_geom.tk
            qnode = next(
                n
                for n in plan.fused_gm.graph.nodes
                if n.op == "call_function" and n.target is _QUANTIZE_MX_OUTLIER
            )
            self.epi_prefix = _split_prefix(plan.fused_gm, qnode)
            phs = [
                n for n in plan.fused_gm.graph.nodes if n.op == "placeholder"
            ]
            position = {p: i for i, p in enumerate(phs)}

            def resolve(a):
                if isinstance(a, torch.fx.Node):
                    return ("arg", position[a])
                if isinstance(a, (list, tuple)):
                    return ("lit", list(a))
                return ("lit", a)

            self.epi_q_args = [resolve(a) for a in qnode.args[1:]]
            self.epi_q_kwargs = {k: resolve(v) for k, v in qnode.kwargs.items()}
            # A fitted budget reaches the per-sub-slice calls through their
            # ``max_pct``; an unfitted geometry leaves the op's own
            # arguments untouched.
            if out_geom.op_max_pct != out_geom.max_pct:
                if len(qnode.args) > 9:
                    self.epi_q_args = self.epi_q_args[:8]
                self.epi_q_kwargs["max_pct"] = ("lit", out_geom.op_max_pct)
        self.chain_epilogue = (
            out_geom is not None
            and self.n_sub == 1
            and (split_k_tail_fusible or plan.num_k == 1)
        )
        self.chain_fused_tail = split_k_tail_fusible and out_geom is None
        # Classified once: the traced per-step kernel rebuild cannot walk
        # a graph.
        self.tail_split = _split_stream_break(plan.fused_gm)
        # A tail the drain cannot hold in one pass runs over the staged
        # tile.  So does a multi-sub-slice epilogue, which reads windows of
        # its tile at an index the rolled loop supplies at runtime, and an
        # unchained single-slice one, which runs its stores interleaved.
        # ``staged`` is what puts the tile in scratch: its sole effect is to
        # disqualify ``_single_pass_kernel``, which allocates none.
        self.stage_tile = not single_k_tail_fusible or (
            out_geom is not None and (self.n_sub > 1 or not self.chain_epilogue)
        )
        # The K accumulator, when the reduction needs one.  The kernel built
        # alongside it is rebuilt per step (with the grid index captured) and
        # discarded here; only the specs are needed before the loop exists.
        self.scratch_specs = self._scratch_and_kernel(plan.fused_gm)[0]
        self.inner = PipelinedKernel(
            self._kernel,
            grid=plan.grid,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            scratch_specs=self.scratch_specs,
            num_slots=num_slots,
        )

    # --- kernel assembly ----------------------------------------------------

    def _scratch_and_kernel(self, tail):
        """The dense path's scratch and kernel, over a ``gemm_kernel`` that
        reads the gathered CSR, in its staging buffers, instead of the raw
        operands.

        Args:
            tail: The fused tail to apply — the plan's own, or its
                ``_EpilogueTail`` wrapper when a step's tail also stores a
                CSR. Rebuilt per step because a HOP body may not mutate an
                outer Python object, so the stores have to run where the
                results exist.

        Returns:
            ``(scratch_specs, kernel)``, the kernel matching the arity of
            the dense path's.
        """
        plan = self.plan
        num_operands = len(plan.in_specs)

        indptr_idx = plan.kw_idx["A_indptr"]
        indices_idx = plan.kw_idx["A_indices"]
        data_idx = plan.kw_idx["A_data"]

        def gemm_kernel(in_tiles, first):
            tiles = list(in_tiles[:num_operands])
            if self.geom is not None:
                tiles[data_idx] = self._bufs.gather_data
                tiles[indices_idx] = self._bufs.gather_indices
                # One call consumes one K slice, so the slice axis is
                # addressing, not data: the op wants a plain [*batch, tm+1]
                # pointer array.
                ptr = in_tiles[indptr_idx]
                tiles[indptr_idx] = ptr.reshape(
                    tuple(ptr.shape[:-2]) + (ptr.shape[-1],)
                )
            return plan.gemm_kernel(tuple(tiles), first)

        return _gemm_scratch_and_kernel(
            gemm_kernel,
            tail,
            reduction_dim=self.grid_k,
            num_k=plan.num_k,
            acc_shape=plan.acc_shape,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            fused_idx=list(plan.fused_idx),
            anchor=plan.anchor,
            accumulate_fp32=self.accumulate_fp32,
            lag=0,
            split=self.tail_split if tail is plan.fused_gm else None,
            single_k_tail_fusible=not self.stage_tile,
            split_k_tail_fusible=self.chain_epilogue or self.chain_fused_tail,
        )

    def _gather_rolled(self, idx, slots):
        """Gather a row tile's ``R`` blocks, in a ``while_loop`` over them.

        Unrolled this would be ``R`` copies of the same block fetch, one
        per row block; the loop spells it once.  The pointer tile's first
        entry, which every block subtracts, is picked before the loop.
        Every copy adds 1 to the gather stream's counter and is waited for
        inline: copy ``b`` of block ``j`` is the stream's ``2R`` copies per
        gather before this one, plus ``2j + b + 1``.  A trip reads both of
        its own block's bounds rather than carrying the boundary it shares
        with the next one, so the loop carries nothing but its index, at
        one extra scalar read per block.

        Args:
            idx: The grid coordinate.
            slots: The operand tiles, positionally.
        """
        plan, g, bufs = self.plan, self.geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[i] for i in range(nb)]
        m_c, k = idx[self.grid_m], idx[self.grid_k]

        p = slots[plan.kw_idx["A_indptr"]].reshape(ones + (plan.tile_m + 1,))
        lo = _scalar(_entry(p, 0, nb, ones))
        pairs = (
            (slots[plan.kw_idx["A_data"]], bufs.gather_data),
            (slots[plan.kw_idx["A_indices"]], bufs.gather_indices),
        )
        gathers_before = self._ordinal(
            idx, self.grid_m + 1 if self.reuse_gathers else len(plan.grid)
        )
        landed_before = 2 * self.R * gathers_before

        def cond_fn(j):
            return j < self.R

        def body_fn(j):
            entry = voyager.subview(
                bufs.base_table,
                [0] * g.dropped + bidx + [k, m_c * self.R + j],
                (1,) * g.dropped + ones + (1, 1),
                (1,) * (g.dropped + nb + 2),
                [],
            )
            at = _scalar(entry)
            torch._check(at >= 0)
            at_j = _scalar(_entry(p, j * g.rows, nb, ones))
            start = at_j - lo
            torch._check(start >= 0)
            count = _scalar(_entry(p, (j + 1) * g.rows, nb, ones)) - at_j
            torch._check(count >= 0)

            for b, (src, dst) in enumerate(pairs):
                window = voyager.subview(
                    dst,
                    [0] * nb + [start],
                    ones + (g.budget,),
                    (1,) * (nb + 1),
                    [],
                )
                voyager.async_copy(
                    src,
                    window,
                    bidx + [at],
                    ones + (g.budget,),
                    bufs.gather_counter,
                    None,
                    [1] * (nb + 1),
                    count=[1] * nb + [count],
                )
                voyager.async_wait(
                    bufs.gather_counter, landed_before + 2 * j + b + 1
                )
            return (j + 1,)

        # An unused ``while_loop`` is pruned, taking the body's copies with
        # it, so the trip count has to be consumed.
        (j_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(j_end >= 0)

    def _offset_at(self, idx, j):
        """Sub-slice ``j``'s running nonzero count, the op's
        ``indptr_offset``."""
        g, bufs = self.out_geom, self._bufs
        nb = g.nb
        off = _scalar(
            voyager.subview(
                bufs.slice_nnz,
                [idx[x] for x in range(nb)]
                + [idx[self.grid_n] * self.n_sub + j],
                g.ones + (1,),
                (1,) * (nb + 1),
                [],
            )
        )
        torch._check(off >= 0)
        return off

    def _wait_csr_stores(self, stored):
        """The CPU waits until the first ``stored`` staged sub-slice stores
        have landed."""
        voyager.async_wait(self._bufs.store_counter, _CSR_STORES * stored)

    def _ordinal(self, idx, dims):
        """``idx``'s position in the sweep, counted over its first ``dims``
        grid dims: the whole grid counts steps, ``grid_k`` (the reduction,
        last) output tiles, ``grid_m + 1`` row tiles."""
        ordinal = 0
        for d in range(dims):
            ordinal = ordinal * self.plan.grid[d] + idx[d]
        return ordinal

    def _store_staged_csr(self, idx, j):
        """Store one staged sub-slice: its CSR block and its dense pair.

        The same layout the row-swept producer writes, one block at a time:
        the row pointers land in their slice's continuous array, the data and
        indices at the stream position ``base`` names, the scale and inliers
        at the sub-slice's own place in the grid, and the block's position
        goes into the shared table for its consumer.  The ``_CSR_STORES``
        copies add to the store stream's counter unwaited; the caller waits
        (``_wait_csr_stores``) -- the chained mode before the next finalize
        reuses the staging tiles and after the sweep, the interleaved mode
        inline (``_EpilogueTail._sub_slice``).
        """
        g, bufs = self.out_geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[x] for x in range(nb)]
        m = idx[self.grid_m]
        # The CSR's K slice is sub-slice ``j`` of this GEMM's column tile.
        k = idx[self.grid_n] * self.n_sub + j

        d, i, p, s, q = (
            bufs.store_data_tile,
            bufs.store_index_tile,
            bufs.store_indptr_tile,
            bufs.store_scale_tile,
            bufs.store_inlier_tile,
        )

        voyager.async_copy(
            p.reshape(ones + (1, g.rows + 1)),
            bufs.csr_indptr,
            bidx + [k, m],
            ones + (1, g.rows + 1),
            bufs.store_counter,
            None,
            [1] * nb + [1, g.rows],
        )

        last = _entry(p, g.rows, nb, ones)
        nnz = _scalar(last) - _scalar(_entry(p, 0, nb, ones))
        torch._check(nnz >= 0)
        base_ref = voyager.subview(
            bufs.stream_pos, bidx + [0], ones + (1,), (1,) * (nb + 1), []
        )
        at = _scalar(base_ref)
        torch._check(at >= 0)

        for src, dst in ((d, bufs.csr_data), (i, bufs.csr_indices)):
            voyager.async_copy(
                src,
                dst,
                bidx + [at],
                ones + (g.budget,),
                bufs.store_counter,
                None,
                [1] * (nb + 1),
                count=[1] * nb + [nnz],
            )

        for tile, dst, width in (
            (s, bufs.scale, g.tk // g.block_size),
            (q, bufs.inliers, g.tk),
        ):
            voyager.async_copy(
                tile,
                dst,
                bidx + [m, k],
                ones + (g.rows, width),
                bufs.store_counter,
                None,
                [1] * nb + [g.rows, width],
            )

        voyager.insert(
            base_ref.reshape(ones + (1, 1)).clone(),
            voyager.subview(
                bufs.out_base_table,
                bidx + [k, m],
                ones + (1, 1),
                (1,) * (nb + 2),
                [],
            ),
        )
        voyager.insert(base_ref + nnz, base_ref)
        voyager.insert(
            last.clone(),
            voyager.subview(
                bufs.slice_nnz, bidx + [k], ones + (1,), (1,) * (nb + 1), []
            ),
        )

    def _row_tile_start(self, idx):
        """Whether ``idx`` is the first step of its row tile: every dim
        inside the row dim at zero."""
        first = idx[self.inner_dims[0]] == 0
        for d in self.inner_dims[1:]:
            first = first & (idx[d] == 0)
        return first

    def _kernel(self, idx, *slots):
        if self.reuse_gathers:
            effect_cond(
                self._row_tile_start(idx),
                lambda: self._gather_rolled(idx, slots),
            )
        elif self.geom is not None:
            self._gather_rolled(idx, slots)
        tail = self.plan.fused_gm
        if self.out_geom is not None:
            # The scheduler passes ``*in_slots, *out_slots, *scratch``, and
            # this nest declares no scheduler-managed output.
            scratch = slots[len(self.plan.in_specs) :]
            tail = _EpilogueTail(self, idx, scratch[0] if scratch else None)
        if self.chain_epilogue:
            # The finalizes before this tile's stored their CSRs from the
            # staging tiles; their DMAs ran under the rounds between, so
            # the wait costs nothing, and the tiles are free to be written
            # again.  The last round's stores are waited for in ``forward``.
            self._wait_csr_stores(self._ordinal(idx, self.grid_k))
        self._scratch_and_kernel(tail)[1](idx, *slots)
        if self.chain_epilogue:
            # The chained pass staged this tile's CSR; its DMAs and
            # stream bookkeeping run bare, on the round that completed
            # the tile.
            effect_cond(
                idx[self.grid_k] == self.plan.num_k - 1,
                lambda: self._store_staged_csr(idx, 0),
            )

    # --- entry --------------------------------------------------------------

    def forward(self, *inputs):
        """Allocate every buffer this nest owns, then run its loop.

        The SRAM allocations below are ORDER-SENSITIVE. ``build_sparse_gemm``
        stamps their bank groups positionally, by zipping them against its
        own ``hand_groups`` list, so adding, removing or reordering one
        here silently misassigns banks unless that list is changed to
        match. Counters and DRAM allocations are skipped and may move
        freely.
        """
        bufs = _Bufs()
        g = self.geom
        if g is not None:
            # Names the producer's table rather than adding one:
            # ``merge_base_tables`` collapses the group onto its alloc.
            bufs.base_table_dram = voyager.alloc(
                base_table_shape(g), torch.int32
            )
            bufs.gather_data = voyager.alloc(
                list(g.ones) + [self.span], self.data_dtype, _SRAM
            )
            bufs.gather_indices = voyager.alloc(
                list(g.ones) + [self.span], torch.int32, _SRAM
            )
            bufs.base_table = voyager.alloc(
                base_table_shape(g), torch.int32, _SRAM
            )
            bufs.gather_counter = new_counter()
            bufs.table_counter = new_counter()

        og = self.out_geom
        if og is not None:
            vals = self.out_values
            bufs.csr_data = voyager.alloc(
                list(og.batch) + [og.stream], vals[0].dtype
            )
            bufs.csr_indices = voyager.alloc(
                list(og.batch) + [og.stream], torch.int32
            )
            bufs.csr_indptr = voyager.alloc(
                list(og.batch) + [og.n_k, og.M + 1], torch.int32
            )
            bufs.scale = voyager.alloc(
                list(og.batch) + [og.M, og.K // og.block_size], vals[3].dtype
            )
            bufs.inliers = voyager.alloc(
                list(og.batch) + [og.M, og.K], vals[4].dtype
            )
            bufs.out_base_table_dram = voyager.alloc(
                base_table_shape(og), torch.int32
            )
            bufs.slice_nnz = voyager.alloc(
                list(og.batch) + [og.n_k], torch.int32, _SRAM
            )
            bufs.stream_pos = voyager.alloc(
                list(og.batch) + [1], torch.int32, _SRAM
            )
            bufs.out_base_table = voyager.alloc(
                base_table_shape(og), torch.int32, _SRAM
            )
            bufs.store_counter = new_counter()
            bufs.out_table_counter = new_counter()
            # Per-sub-slice staging: a compute result reaches DRAM through
            # a scratchpad tile, never straight from the op.
            bufs.store_data_tile = voyager.alloc(
                list(og.ones) + [og.budget], vals[0].dtype, _SRAM
            )
            bufs.store_index_tile = voyager.alloc(
                list(og.ones) + [og.budget], torch.int32, _SRAM
            )
            bufs.store_indptr_tile = voyager.alloc(
                list(og.ones) + [og.rows + 1], torch.int32, _SRAM
            )
            bufs.store_scale_tile = voyager.alloc(
                list(og.ones) + [og.rows, og.tk // og.block_size],
                vals[3].dtype,
                _SRAM,
            )
            bufs.store_inlier_tile = voyager.alloc(
                list(og.ones) + [og.rows, og.tk], vals[4].dtype, _SRAM
            )
        self._bufs = bufs
        if g is not None:
            # The gather reads the table's scalars to address its copies, so
            # the whole table lands on chip before the loop runs.
            copy_base_table(
                bufs.base_table_dram,
                bufs.base_table,
                base_table_shape(g),
                bufs.table_counter,
            )
        dense = self.inner(*inputs)
        if og is None:
            return dense
        # Every sub-slice's stores have landed before the consumer's nest
        # reads the CSR: a tile stores on its last round, so the steps the
        # loop ran completed ``steps // num_k`` tiles.
        tiles = self.inner.steps_run // self.plan.num_k
        self._wait_csr_stores(tiles * self.n_sub)
        copy_base_table(
            bufs.out_base_table,
            bufs.out_base_table_dram,
            base_table_shape(og),
            bufs.out_table_counter,
        )
        return (
            bufs.csr_data,
            bufs.csr_indices,
            bufs.csr_indptr,
            bufs.scale,
            bufs.inliers,
        )


def _epilogue_geometry(node, plan) -> _Geometry:
    """The CSR geometry a GEMM epilogue produces.

    The quantize sees the GEMM's output tile, so the row block *is* the row
    tile.  The K slice is the width ``plan_csr_slices`` settled on with the
    consumers (``meta["csr_slice"]``), which the column tile is cut into.
    """
    vals = node.value
    inliers = vals[4]
    batch = tuple(inliers.shape[:-2])
    M, K = inliers.shape[-2], inliers.shape[-1]
    submod = node.meta["submodule"]
    qnode = next(
        n for n in submod.graph.nodes if n.target is _QUANTIZE_MX_OUTLIER
    )
    bs = qnode.args[3]
    max_pct = qnode.args[9] if len(qnode.args) > 9 else 0.01
    tk = node.meta.get("csr_slice")
    if tk is None:
        raise ValueError(
            f"{node.name}: no CSR slice width was planned for this epilogue "
            f"(plan_csr_slices runs before bufferization)"
        )
    if plan.tile_n % tk:
        raise ValueError(
            f"{node.name}: column tile {plan.tile_n} is not a whole number "
            f"of the planned {tk}-wide CSR slices"
        )
    return _Geometry(
        batch=batch,
        M=M,
        K=K,
        rows=plan.tile_m,
        tk=tk,
        block_size=bs,
        budget=int(plan.tile_m * tk * max_pct),
        max_pct=max_pct,
    )


def _retarget_csr_views(node) -> None:
    """Let a CSR reach a lower-rank consumer without losing its slicing.

    A reshape can sit between producer and consumer — the boundary between a
    decoder layer and ``lm_head`` drops the leading batch dim — and it carries
    no meta of its own. Dropping the batch dim is what the consumer wants;
    flattening the K-slice axis away with it is not, since the pointer array
    holds one array per slice. So the pointer view is retargeted to keep that
    axis, and every CSR view inherits the producer's geometry.
    """
    for arg in list(node.args) + list(node.kwargs.values()):
        if not isinstance(arg, torch.fx.Node) or not is_nop(arg):
            continue
        src = arg
        while is_nop(src) and src.all_input_nodes:
            src = src.all_input_nodes[0]
        geom = src.meta.get(GEOMETRY_META)
        if geom is None:
            continue
        arg.meta[GEOMETRY_META] = geom
        arg.meta[PRODUCER_META] = src.meta.get(PRODUCER_META)
        value = getattr(arg, "value", None)
        if (
            value is None
            or value.dtype != torch.int32
            or value.shape[-1] != geom.M + 1
            or value.ndim >= src.value.ndim
        ):
            continue
        shape = list(value.shape[:-1]) + [geom.n_k, value.shape[-1]]
        arg.args = (arg.args[0], shape)
        set_node_value(arg, src.value.reshape(shape))


def _csr_geometry(node):
    """This GEMM's ``A_indptr`` operand and the geometry its producer stamped.

    Identified by shape rather than by the anchor's kwargs, which on a fused
    group name submodule placeholders that carry no meta.

    Returns:
        ``(operand, geometry)``, or ``(None, None)``.
    """
    for operand in node.all_input_nodes:
        geom = operand.meta.get(GEOMETRY_META)
        value = getattr(operand, "value", None)
        if geom is None or not isinstance(value, torch.Tensor):
            continue
        if value.dtype == torch.int32 and value.shape[-1] == geom.M + 1:
            # A reshape on the way in may have dropped batch dims the producer
            # had; the consumer indexes only the ones it still sees.
            batch = tuple(value.shape[:-2])
            if batch != geom.batch:
                geom = replace(
                    geom,
                    batch=batch,
                    dropped=len(geom.batch) - len(batch),
                )
            return operand, geom
    return None, None


@contextlib.contextmanager
def _slice_axis_hidden(node):
    """Hide the K-slice axis of this GEMM's ``A_indptr`` operand.

    The producer's buffer is ``[*batch, n_k, M+1]`` -- one cumulative pointer
    array per K slice -- but a single ``linear_mx`` consumes exactly one slice,
    so what the op is handed is ``[*batch, M+1]``. Shape derivation runs the
    submodule on the operand *values*, and ``spmm_csr`` reads every leading dim
    as batch, so an extra one there gives the GEMM's output a rank it does not
    have.

    Only derivation sees the shorter value: the fetch still addresses the whole
    buffer, picking its slice off the reduction grid dim, so the axis is put
    back on the way out.

    Yields:
        The ``(node, full_value)`` pair that was hidden, or ``None``.
    """
    hidden = None
    for operand in node.all_input_nodes:
        geom = operand.meta.get(GEOMETRY_META)
        value = getattr(operand, "value", None)
        if geom is None or not isinstance(value, torch.Tensor):
            continue
        if (
            value.dtype == torch.int32
            and value.ndim == geom.nb + 2
            and value.shape[-1] == geom.M + 1
        ):
            hidden = (operand, value)
            operand.value = value[..., 0, :]
            operand.shape = tuple(operand.value.shape)
            break
    try:
        yield hidden
    finally:
        if hidden is not None:
            operand, value = hidden
            operand.value = value
            operand.shape = tuple(value.shape)


def build_sparse_gemm(
    node: torch.fx.Node, ctx: BuildContext
) -> Optional[torch.fx.GraphModule]:
    """Bufferize a GEMM whose activation carries an outlier CSR.

    The nest stores each tile on its own step, whatever ``ctx.lag`` says:
    its gather and staged stores are addressed by values read out of
    buffers at run time, which the ``PipelinedKernel`` lag does not cover.

    Args:
        node: The sparse ``linear_mx`` / ``matmul_mx``, bare or fused.
        ctx: The ``BuildContext``; ``num_slots`` is the software-pipeline
            depth for the dense operands.

    Returns:
        The bufferized gm, or ``None`` when the node has no CSR or the
        producer's geometry never reached it.
    """
    produces = gemm_produces_csr(node)
    # The CSR's geometry is fixed by its producer, and the GEMM has to split K
    # the same way, so read it before deriving anything and hand the derivation
    # that slice count instead of the tile search's.
    _retarget_csr_views(node)
    ptr, geom = _csr_geometry(node)
    if geom is None and not produces:
        return None

    with _slice_axis_hidden(node) as hidden:
        plan = _gemm_plan(
            node,
            ctx.tiler,
            k_tiles=geom.n_k if geom is not None else None,
            accumulate_fp32=ctx.accumulate_fp32,
        )
    if plan is None:
        return None
    operands = list(node.all_input_nodes)
    if hidden is not None:
        # Derivation ran against one slice; the nest fetches from the whole
        # buffer, so hand the loop the operand it will actually address.
        inputs = list(plan.inputs)
        inputs[plan.kw_idx["A_indptr"]] = hidden[1].clone()
        plan.inputs = tuple(inputs)

    data_dtype = None
    if geom is not None:
        if plan.tile_k != geom.tk:
            raise ValueError(
                f"{node.name}: k tile {plan.tile_k} does not match the CSR's "
                f"slice width {geom.tk}; column indices are local to a slice "
                f"and the engine cannot rebase them"
            )
        if plan.tile_m % geom.rows:
            raise ValueError(
                f"{node.name}: row tile {plan.tile_m} is not a whole number "
                f"of the producer's {geom.rows}-row blocks"
            )
        data_dtype = operands[plan.kw_idx["A_data"]].value.dtype

    out_geom = _epilogue_geometry(node, plan) if produces else None
    pattern = _SparseGemm(
        plan,
        geom,
        data_dtype,
        out_geom=out_geom,
        split_k_tail_fusible=node.meta.get("split_k_tail_fusible", False),
        single_k_tail_fusible=node.meta.get("single_k_tail_fusible", True),
        num_slots=ctx.num_slots,
        accumulate_fp32=ctx.accumulate_fp32,
    )
    if out_geom is not None:
        pattern.out_values = list(node.value)
    with _lenient_verifier():
        gm = export_model(pattern, plan.inputs)
    gm = _finalize_exported_gm(gm)

    # The hand-made allocs join the groups of the roles they serve, which
    # only an interstellar mapping places: under a GEMV's result the whole
    # nest stays loose.
    result = plan.anchor.meta.get("tile_result")
    groups = None
    if result is not None and isinstance(
        result.tiling, interstellar.MappingPoint
    ):
        groups = result.placement.groups
    if groups is not None:

        def group_of(role):
            return next(
                (i for i, group in enumerate(groups) if role in group),
                None,
            )

        # One entry per SRAM alloc ``_SparseGemm.forward`` makes, in its
        # order -- keep the two in step.
        hand_groups = []
        if geom is not None:
            # gather_data, gather_indices, base_table
            hand_groups += [group_of(Role.INPUT)] * 3
        if out_geom is not None:
            # slice_nnz, stream_pos, out_base_table and the five
            # store-staging tiles
            hand_groups += [group_of(Role.OUTPUT)] * 8
        _stamp_bank_groups(
            gm,
            hand_groups
            + _bank_group_list(
                node, plan.in_specs, plan.out_specs, pattern.scratch_specs
            ),
        )
    # The rolled gather nests a second ``while_loop`` inside the step,
    # over the ``R`` row blocks it concatenates.
    extents = [[(0, pattern.inner.num_steps, 1)]]
    if geom is not None:
        extents.append([(0, pattern.R, 1)])
    _tag_loop_extents(gm, extents)
    if geom is not None:
        tag_base_table(gm, ptr.meta[PRODUCER_META], base_table_shape(geom))
    # A gather moves a block's own count, at the occupancy its producer
    # stamped.  A nest that also stores a CSR (the epilogue producer) is
    # left at the budget.
    if geom is not None and out_geom is None:
        stamp_csr_fill(gm, geom.fill)
    if out_geom is not None:
        tag_base_table(gm, node.name, base_table_shape(out_geom))
        gm.meta[GEOMETRY_META] = out_geom
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*plan.inputs)
    if plan.outline_dps:
        outline_dps_ops(gm)
    _stamp_anchor_meta(gm, plan.anchor)
    return gm
