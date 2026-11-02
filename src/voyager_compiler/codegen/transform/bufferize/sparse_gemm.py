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
from voyager_compiler.codegen.node_info import is_nop
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
    _scalar,
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
    outline_dps_ops,
    stamp_csr_fill,
    voyager,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import Role
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import set_node_value, ShapeProp

_SRAM = int(MemoryLevel.SRAM)


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

    The stores trail their compute by ``lag`` steps (``PipelinedKernel``).
    The gather staging has one slot, so a gather overwrites it only once the
    SpMM before it has read it; the sync pass derives that wait.
    """

    def __init__(
        self,
        plan,
        geom,
        data_dtype,
        *,
        lag: int,
        split_k_tail_fusible: bool,
        single_k_tail_fusible: bool = True,
        num_slots: int = _DEFAULT_NUM_SLOTS,
        accumulate_fp32: bool = False,
    ):
        super().__init__()
        self.plan = plan
        self.lag = lag
        self.accumulate_fp32 = accumulate_fp32
        self.data_dtype = data_dtype
        self.geom = geom
        self.R = plan.tile_m // geom.rows
        self.span = self.R * geom.budget
        self.grid_m, self.grid_n, self.grid_k = plan.grid_dims
        # Grid dims the sweep nests inside the row dim, ahead of the
        # reduction, and the steps they span: how long a row tile lasts.
        self.inner_dims = [d for d in range(self.grid_k) if d > self.grid_m]
        self.row_steps = math.prod(plan.grid[d] for d in self.inner_dims)
        self.reuse_gathers = plan.num_k == 1 and self.row_steps > 1
        self.split_k_tail_fusible = split_k_tail_fusible
        self.single_k_tail_fusible = single_k_tail_fusible
        # The K accumulator, when the reduction needs one, and the dense
        # per-step kernel.
        self.scratch_specs, self.dense_kernel = self._scratch_and_kernel()
        self.inner = PipelinedKernel(
            self._kernel,
            grid=plan.grid,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            scratch_specs=self.scratch_specs,
            num_slots=num_slots,
            lag=lag,
        )

    # --- kernel assembly ----------------------------------------------------

    def _scratch_and_kernel(self):
        """The dense path's scratch and kernel, over a ``gemm_kernel`` that
        reads the gathered CSR, in its staging buffers, instead of the raw
        operands.

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
            tiles[data_idx] = self._bufs.gather_data
            tiles[indices_idx] = self._bufs.gather_indices
            # One call consumes one K slice, so the slice axis is addressing,
            # not data: the op wants a plain [*batch, tm+1] pointer array.
            ptr = in_tiles[indptr_idx]
            tiles[indptr_idx] = ptr.reshape(
                tuple(ptr.shape[:-2]) + (ptr.shape[-1],)
            )
            return plan.gemm_kernel(tuple(tiles), first)

        return _gemm_scratch_and_kernel(
            gemm_kernel,
            plan.fused_gm,
            reduction_dim=self.grid_k,
            num_k=plan.num_k,
            acc_shape=plan.acc_shape,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            fused_idx=list(plan.fused_idx),
            anchor=plan.anchor,
            accumulate_fp32=self.accumulate_fp32,
            lag=self.lag,
            split=_split_stream_break(plan.fused_gm),
            single_k_tail_fusible=self.single_k_tail_fusible,
            split_k_tail_fusible=self.split_k_tail_fusible,
        )

    def _gather_rolled(self, idx, slots):
        """Gather a row tile's ``R`` blocks, in a ``while_loop`` over them.

        Unrolled this would be ``R`` copies of the same block fetch, one
        per row block; the loop spells it once.  The pointer tile's first
        entry, which every block subtracts, is picked before the loop.  A
        trip reads both of its own block's bounds rather than carrying the
        boundary it shares with the next one, so the loop carries nothing
        but its index, at one extra scalar read per block.

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
                    None,
                    None,
                    [1] * (nb + 1),
                    count=[1] * nb + [count],
                )
            return (j + 1,)

        # An unused ``while_loop`` is pruned, taking the body's copies with
        # it, so the trip count has to be consumed.
        (j_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(j_end >= 0)

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
        else:
            self._gather_rolled(idx, slots)
        self.dense_kernel(idx, *slots)

    # --- entry --------------------------------------------------------------

    def forward(self, *inputs):
        """Allocate every buffer this nest owns, then run its loop.

        The SRAM allocations below are ORDER-SENSITIVE. ``build_sparse_gemm``
        stamps their bank groups positionally, by zipping them against its
        own ``hand_groups`` list, so adding, removing or reordering one
        here silently misassigns banks unless that list is changed to
        match. DRAM allocations are skipped and may move freely.
        """
        bufs = _Bufs()
        g = self.geom
        # Names the producer's table rather than adding one:
        # ``merge_base_tables`` collapses the group onto its alloc.
        bufs.base_table_dram = voyager.alloc(base_table_shape(g), torch.int32)
        bufs.gather_data = voyager.alloc(
            list(g.ones) + [self.span], self.data_dtype, _SRAM
        )
        bufs.gather_indices = voyager.alloc(
            list(g.ones) + [self.span], torch.int32, _SRAM
        )
        bufs.base_table = voyager.alloc(base_table_shape(g), torch.int32, _SRAM)
        self._bufs = bufs
        # The gather reads the table's scalars to address its copies, so the
        # whole table lands on chip before the loop runs.
        copy_base_table(
            bufs.base_table_dram, bufs.base_table, base_table_shape(g)
        )
        return self.inner(*inputs)


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

    Args:
        node: The sparse ``linear_mx`` / ``matmul_mx``, bare or fused.
        ctx: The ``BuildContext``; ``num_slots`` is the software-pipeline
            depth for the dense operands and ``lag`` how many steps a
            tile's store trails its compute.

    Returns:
        The bufferized gm, or ``None`` when the producer's geometry never
        reached the node.
    """
    # The CSR's geometry is fixed by its producer, and the GEMM has to split K
    # the same way, so read it before deriving anything and hand the derivation
    # that slice count instead of the tile search's.
    _retarget_csr_views(node)
    ptr, geom = _csr_geometry(node)
    if geom is None:
        return None

    with _slice_axis_hidden(node) as hidden:
        plan = _gemm_plan(
            node,
            ctx.tiler,
            k_tiles=geom.n_k,
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

    pattern = _SparseGemm(
        plan,
        geom,
        data_dtype,
        lag=ctx.lag,
        split_k_tail_fusible=node.meta.get("split_k_tail_fusible", False),
        single_k_tail_fusible=node.meta.get("single_k_tail_fusible", True),
        num_slots=ctx.num_slots,
        accumulate_fp32=ctx.accumulate_fp32,
    )
    with _lenient_verifier():
        gm = export_model(pattern, plan.inputs)
    gm = _finalize_exported_gm(gm)

    # The hand-made allocs join the groups of the roles they serve, which
    # only an interstellar mapping places: under a GEMV's result the whole
    # nest stays loose.
    result = plan.anchor.meta.get("tile_result")
    if result is not None and isinstance(
        result.tiling, interstellar.MappingPoint
    ):
        groups = result.placement.groups
        if groups is not None:
            input_group = next(
                (i for i, group in enumerate(groups) if Role.INPUT in group),
                None,
            )
            # One entry per SRAM alloc ``_SparseGemm.forward`` makes, in its
            # order (gather_data, gather_indices, base_table) -- keep the two
            # in step.
            hand_groups = [input_group] * 3
            _stamp_bank_groups(
                gm,
                hand_groups
                + _bank_group_list(
                    node, plan.in_specs, plan.out_specs, pattern.scratch_specs
                ),
            )
    # The rolled gather nests a second ``while_loop`` inside the step,
    # over the ``R`` row blocks it concatenates.
    _tag_loop_extents(
        gm, [[(0, pattern.inner.num_steps, 1)], [(0, pattern.R, 1)]]
    )
    tag_base_table(gm, ptr.meta[PRODUCER_META], base_table_shape(geom))
    # A gather moves a block's own count, at the occupancy its producer
    # stamped.
    stamp_csr_fill(gm, geom.fill)
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*plan.inputs)
    if plan.outline_dps:
        outline_dps_ops(gm)
    _stamp_anchor_meta(gm, plan.anchor)
    return gm
