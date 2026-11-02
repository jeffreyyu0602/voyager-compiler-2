"""FA3-style flash-attention bufferization builder (cross-sweep pipelined).

A dedicated scheduler (``_FA3Pipeline``) that overlaps the systolic (matrix)
array with the vector unit: each iteration runs the current block's QKᵀ matmul
and the previous block's P@V matmul on the matrix unit while the vector unit
softmaxes, a one-step skew fused across Q-tile / head boundaries.  The pipeline
is peeled — a prologue, ``num_steps - 1`` uniform loop iterations, an epilogue.
Every op is emitted inline and every copy bare: the DMA is asynchronous, and
the sync pass derives each wait, an op's on the block it reads having landed
or on the slot it writes being free.

The loop walks the live (Q tile, KV block) pairs: every pair under an
explicit mask; under ``is_causal`` only the blocks up to each Q tile's last
row, the rest being fully masked.  It carries the sweep (the Q tile's index
over ``[*kv_batch, num_q_blocks]``) and ``kv``, the block within the sweep,
and advances them each step past the sweep's last block (``_kv_last``).
Per iteration (``N = num_kv_blocks``), in program order:

  [A] S = (Q @ Kᵀ)·scale (+ mask) -> s_buf[step % 2], once Q, K and the
      mask have landed.  Under ``is_causal`` the mask is added only where the KV
      block crosses the Q tile's diagonal (``_crosses``).  An explicit mask
      broadcast along every dim the walk steps over is one tile, loaded
      once in the prologue (``mask_once``).
  [B] pv_buf = p_buf[(step-1) % 2] @ V_prev, once V_prev has landed
  [D] softmax chain (rowmax, m, alpha, P, rowsum, l).  The running max ``m``
      alternates between two slots by step parity: step ``t`` reads slot ``t
      % 2`` and writes slot ``(t + 1) % 2``, so the previous max needs no
      copy.  A step starting a tile reads a third slot instead, which the
      prologue copies a -inf constant into, as its fresh max.  The sum ``l``
      is double-buffered by sweep parity so the tile that just ended keeps
      its own for [E]; a tile's first block writes it (l = rowsum), later
      blocks accumulate into it.
  [E] kv == 0 only: o += pv_buf; wait for the previous output store;
      (o / l[(sweep + 1) % 2]) -> o_slots; store o_slots to the PREVIOUS
      tile's DRAM rows (the finalize belongs to the tile that just ended).
      When that tile had one block, o holds nothing of it and the finalize
      reads pv_buf instead.
  [F] kv >= 1 only (after [D]): the deferred rescale fused with the
      accumulate, o = alpha·(o + pv_buf), or o = alpha·pv_buf at kv == 1,
      the tile's first P@V.

No state is reset or zeroed: each running value is written by its tile's
first block, so the kernel needs no fill.  A subclass whose steps keep their
state in place instead (``zeroes_state``) resets it at each tile ([C],
``_reset``) and zeroes ``o`` after each finalize.

The probabilities ``p_buf`` are ``s_buf`` itself -- [D] exponentiates the
scores in place -- unless the softmax runs at a different dtype than the
operands (``accumulate_fp32`` over bf16 inputs), when P is written to its
own buffer at V's dtype so the matrix unit reads it as an operand.

Microscaling operands (an ``sdpa_mx`` node): Q, K and V are codes with
block scales, and each scale streams through slots of its own beside its
data, K's transposed by the DMA like K.  Both products are ``matmul_mx``.
[D] exponentiates S in place, sums the exact exponentials into ``l``, then
one more vector pass quantizes them along the keys with the query's
parameters into P's codes (``p_buf``) and block scales (``p_scale``).

``is_causal`` (no mask tensor): [A] adds a mask tile read from
``causal_mask``, a constant table of every tile a live pair can need --
under ``bool_mask`` int1 codes (0 keeps, -1 drops) that a per-tensor scale
dequantizes to the fill in the same pass, as the eager path's quantized
mask is added, else the fill itself at the accumulation dtype.  A
pair's tile depends only on how far its Q tile's first row sits past its KV
block's first key, ``(q·tq) mod sq - kv·tkv``, a multiple of ``gcd(tq,
tkv)``; the table holds one tile per multiple from ``gcd - tq`` up to ``sq -
tq``, a block wholly below the diagonal reading an all-true entry.  A Q tile
must not straddle a head under the GQA fold (``tq`` divides ``sq``), so a
row's position in its head is ``(q·tq) mod sq + r``.

A decode step over a split KV cache (``split_kv_cache``) attends one more
block per sweep: the chunk of the cache being filled, kept in full
precision beside the quantized main cache and handed to the node as
``key_residual`` / ``value_residual`` with ``query_residual`` (the query,
unquantized) and ``residual_mask`` (the chunk's slice of the mask; the main
mask drops the chunk's window).  An online softmax is additive over key
blocks, so the residual is folded in at the sweep's boundary, after [E] has
landed the sweep's last P@V and before the finalize: its QKᵀ, the softmax
chain over the running state the sweep leaves -- its max still in the slot
the boundary step's [D] read past -- and P@V, then ``o = alpha·o + pv``.  Both
products are plain matmuls, ``R`` (the residual's positions) wide, and the
residual's P is not quantized.  The residual's tiles stream through slots
of their own, one per sweep parity: a sweep's residual is fetched on its
last step and consumed at the next sweep's boundary (or the epilogue),
which under a single KV block is the same iteration as the next fetch.

A head narrower than the PE array is padded on chip to the width the
products need (``pad_head``): Q, K, V and their block scales are
held at the padded head and the products run at it, DRAM keeping the real
head.  The pad is zeros.  A column-padded tile (Q, V, their scales, the
residual's Q and V) is zero-filled by its own DMA (``async_copy``'s
``pad``, the read clipped at the source's edge); a transposed tile (Kᵀ, its
scales, the residual's Kᵀ) is row-padded, so its DMA lands the block in the
slot's top rows and the pad rows are zeroed once, before the first load.
P@V's pad columns come out zero and are never read: the accumulate, rescale
and finalize run on the product's real columns, so ``o`` and the output
tile keep the real head.
"""

import logging
import math
from types import SimpleNamespace
from typing import Optional

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.transform.bufferize.attention import (
    _MASK_FILL,
    _fuse_passes,
    fold_mask_tensor,
    plan_gqa_fold,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    oracle_disabled,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import get_slot
from voyager_compiler.codegen.transform.bufferize.utils import (
    BuildContext,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    effect_cond,
    voyager,
)
from voyager_compiler.codegen.transform.tiling import (
    attention_crosses,
    attention_kv_last,
    attention_mask_dyn,
    attention_op_tiling,
    pad_head,
)
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import ShapeProp

_SRAM = int(MemoryLevel.SRAM)
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
# The optional positional operands of an attention node, by position.
_OPTIONAL_OPERANDS = {
    3: "attn_mask",
    6: "query_residual",
    7: "key_residual",
    8: "value_residual",
    9: "residual_mask",
}
_PRODUCT_OPS = (
    torch.ops.aten.matmul.default,
    torch.ops.quantized_ops.matmul_mx.default,
)

logger = logging.getLogger(__name__)


def _unravel(flat, basis):
    """Row-major coordinates of Python-int ``flat`` in ``basis`` (the
    build-time counterpart of ``voyager.delinearize_index``)."""
    out = [0] * len(basis)
    for d in range(len(basis) - 1, -1, -1):
        out[d] = flat % basis[d]
        flat //= basis[d]
    return out


def _causal_mask_table(tq, tkv, sq):
    """The bool mask tiles ``is_causal`` attention reads (module
    docstring): entry ``k`` keeps key ``c`` of row ``r`` when ``c <= r +
    delta`` for ``delta = gcd(tq, tkv)·(k + 1) - tq``."""
    g = math.gcd(tq, tkv)
    deltas = torch.arange(sq // g) * g + (g - tq)
    rows = torch.arange(tq).view(1, tq, 1)
    cols = torch.arange(tkv).view(1, 1, tkv)
    return (cols <= rows + deltas.view(-1, 1, 1)).unsqueeze(1)


class _FA3Pipeline(torch.nn.Module):
    """The FA3 scheduler: prologue → uniform ``while_loop`` → epilogue
    (see the module docstring for the schedule).  Traced whole by
    ``torch.export``; every slot index outside the loop is a Python int.

    A target whose kernel computes the steps its own way subclasses it,
    replacing the ops it lowers (``ops``) and how it reads their query and
    key (``logical_operands``), its streams' tiles (``q_tile``,
    ``out_tile``, ``fold_query``, ``out_shape``), its buffers
    (``_alloc_state``), its loads and store (``_load_q``, ``_load_key``,
    ``_store_out``), its steps -- [A] ``_matmul_qk``, [B] ``_matmul_pv``,
    [C] ``_reset``, [D] ``_softmax_step``, [E]'s ``_land_pv`` and
    ``_finalize``, [F] ``_accumulate_pv`` -- and the fusion of its ops
    (``fuse_ops``), and hands the class to ``build_attention_fa3``.

    Attributes:
        ops: The attention ops the kernel lowers.
        runs_mapped_products: The kernel runs QKᵀ and P@V as the products
            the tile search mapped, so the builder stamps their results on
            them (``_stamp_product_tilings``).
        q_tile: The query tile the Q stream holds, ``(rows, head)``.
        out_tile: The output tile, ``(rows, head)``.
        fold_query: ``forward`` views the query as ``q_fold_shape``.
    """

    ops = (
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.quantized_ops.sdpa_mx.default,
    )
    runs_mapped_products = True
    # The kernel starts each Q tile from reset state (``_reset``, ``o``
    # zeroed) and accumulates every block into it, rather than writing it
    # on the tile's first block.
    zeroes_state = False

    def __init__(
        self,
        *,
        grid,
        num_kv_blocks,
        tq,
        tkv,
        head_dim,
        head_pad,
        scale,
        has_mask,
        mask_dyn,
        is_causal,
        bool_mask,
        out_shape,
        out_dtype,
        acc_dtype,
        g_head,
        sq_orig,
        q_fold_shape,
        out_unfold_shape,
        mask_broadcast,
        mask_fold_shape,
        operand_index,
        block_size,
        probs_quant_max,
        force_scale_power_of_two,
        placement,
    ):
        super().__init__()
        # Where the tile search placed the kernel's on-chip buffers, which
        # a subclass allocates by; ``None`` without a search.
        self.placement = placement
        # ``forward`` takes the node's operands in ``all_input_nodes`` order;
        # ``operand_index`` names each position.  ``block_size`` is the MX
        # block along the contraction axes, ``None`` for unquantized
        # operands; the rest is what P's ``quantize_mx`` takes.
        self.operand_index = operand_index
        self.block_size = block_size
        self.probs_quant_max = probs_quant_max
        self.force_scale_power_of_two = force_scale_power_of_two
        # GQA fold: the ``g_head`` query heads sharing a KV head are folded
        # into the query rows (see ``build_attention_fa3``), so the pipeline
        # runs plain MHA over the KV-head batch and K/V load once per KV head.
        # ``forward`` reshapes q / mask in and the output back out.
        self.g_head = g_head
        self.sq_orig = sq_orig
        self.fold_query = g_head > 1
        self.q_fold_shape = tuple(q_fold_shape)
        self.out_unfold_shape = tuple(out_unfold_shape)
        self.mask_broadcast = mask_broadcast
        self.mask_fold_shape = (
            tuple(mask_fold_shape) if mask_fold_shape is not None else None
        )
        self.grid = tuple(grid)  # (*kv_batch, num_q_blocks, num_kv_blocks)
        self.N = num_kv_blocks
        self.nb = len(grid) - 2
        self.gq, self.gkv = self.nb, self.nb + 1
        self.tq, self.tkv, self.d = tq, tkv, head_dim
        # The head the tiles hold on chip (module docstring); DRAM keeps
        # ``head_dim``.
        self.d_pad = head_pad
        self.q_tile = (tq, head_pad)
        self.out_tile = (tq, head_dim)
        self.scale = scale
        self.has_mask = has_mask
        # Mask: list of (dram_dim, grid_dim) pairs for its dynamic dims
        # (broadcast batch dims are pinned static-0 and never appear).  An
        # explicit mask with none is one tile every step reads, loaded once.
        self.mask_dyn = mask_dyn
        self.mask_once = has_mask and not is_causal and not mask_dyn
        self.out_shape = tuple(out_shape)
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        # Q / K / V share the DRAM layout [*batch, seq, head_dim]: the
        # dynamic dims are the >1-block batch dims plus (for the >1 case)
        # the sequence dim; ``head_dim`` is always loaded whole.
        self.batch_dyn = [i for i in range(self.nb) if grid[i] > 1]
        # The walk (module docstring): ``sweeps`` over ``grid[:-1]``, each
        # running the KV blocks up to ``_kv_last``; under ``is_causal`` the
        # mask tiles come from ``causal_mask``.
        self.causal = is_causal
        self.sweeps = math.prod(self.grid[:-1])
        # The causal table's tiles are added to the scores (module
        # docstring): int1 codes with the fill as their scale, or the fill.
        self.mask_scaled = is_causal and bool_mask
        if is_causal:
            self.causal_gcd = math.gcd(tq, tkv)
            keep = _causal_mask_table(tq, tkv, sq_orig)
            if bool_mask:
                codes = torch.where(keep, 0.0, -1.0).to(acc_dtype)
                self.register_buffer("causal_mask", codes)
                self.register_buffer(
                    "mask_scale", torch.tensor(-_MASK_FILL, dtype=acc_dtype)
                )
            else:
                self.register_buffer(
                    "causal_mask",
                    torch.where(keep, 0.0, _MASK_FILL).to(acc_dtype),
                )
        self.num_steps = math.prod(self.grid[: self.gq]) * sum(
            self._kv_last(q) + 1 for q in range(self.grid[self.gq])
        )
        # The running max a tile starts from, copied once into ``m``'s
        # third slot (``_alloc_state``).
        self.register_buffer(
            "max_fill",
            torch.full((1,) * self.nb + (tq, 1), _MASK_FILL, dtype=acc_dtype),
        )

    def register_pad_rows(self, operands):
        """Register the zero tiles that clear the pad rows of the row-padded
        Kᵀ slots (a head narrower than the products, module docstring), one
        per slot shape, copied in once by the prologue.  ``operands`` are
        ``forward``'s."""
        if self.d_pad == self.d:
            return
        unit = (1,) * self.nb
        index = self.operand_index
        key = operands[index["key"]]
        self.register_buffer(
            "key_pad_rows",
            torch.zeros(unit + (self.d_pad, self.tkv), dtype=key.dtype),
        )
        bs = self.block_size
        if bs is not None and math.ceil(self.d_pad / bs) != math.ceil(
            self.d / bs
        ):
            scale = operands[index["key_scale"]]
            rows = math.ceil(self.d_pad / bs)
            self.register_buffer(
                "key_scale_pad_rows",
                torch.zeros(unit + (rows, self.tkv), dtype=scale.dtype),
            )
        if "key_residual" in index:
            residual = operands[index["key_residual"]]
            length = residual.shape[-2]
            self.register_buffer(
                "residual_key_pad_rows",
                torch.zeros(unit + (self.d_pad, length), dtype=residual.dtype),
            )

    @staticmethod
    def logical_operands(query, key):
        """The query and key as ``aten.scaled_dot_product_attention``
        reads them, ``[*batch, tokens, head_dim]``, from the attention's
        ``query`` and ``key`` as stored."""
        return query, key

    @staticmethod
    def fuse_ops(gm):
        """Collapse the built kernel ``gm``'s compute into the passes its
        hardware runs (``_fuse_passes``)."""
        _fuse_passes(gm)

    def _kv_last(self, q_block):
        """The last live KV block of Q tile ``q_block`` (an int, or the
        loop's SymInt)."""
        if not self.causal:
            return self.N - 1
        return attention_kv_last(q_block, self.tq, self.tkv, self.sq_orig)

    # --- DMA helpers (all take explicit coords; slots may be SymInts) ---

    def _block_address(self, coords, grid_dim, block_count):
        """The (dims, indices) block address of a [*batch, seq, d] DRAM
        tensor at grid point ``coords`` (the dynamically indexed dims and
        their block indices; the sequence dim is driven by ``grid_dim``,
        ``head_dim`` is always loaded whole).  The custom-schedule
        counterpart of ``_BufferedRef._block_address``."""
        dims = list(self.batch_dyn)
        idx = [coords[i] for i in self.batch_dyn]
        if block_count > 1:
            dims.append(self.nb)  # the sequence dim of the DRAM tensor
            idx.append(coords[grid_dim])
        return dims, idx

    def _load_sweep(self, src, slots, slot, coords, tile, pad=False):
        """DMA ``src``'s ``tile`` at ``coords``' query block into
        ``slots[slot]``.  ``pad`` has the DMA zero-fill what ``tile`` extends
        past the source's edge (a column-padded head)."""
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        voyager.async_copy(
            src,
            get_slot(slots, slot),
            idx,
            (1,) * self.nb + tuple(tile),
            None,
            dims,
            pad=[0] * (self.nb + len(tile)) if pad else None,
        )

    def _load_step(
        self,
        src,
        slots,
        slot,
        coords,
        tile,
        transposed,
        pad=False,
        rows=None,
    ):
        """DMA ``src``'s ``tile`` at ``coords``' key block into
        ``slots[slot]``.  ``tile`` is the DRAM block in the buffer's own
        order; ``transposed`` has the DMA ``.mT`` it into the slot (the Kᵀ
        tiles, as ``PipelinedKernel._load_tile`` does).  ``pad`` has the
        DMA zero-fill what ``tile`` extends past the source's edge (a
        column-padded head); ``rows`` lands the block in the slot's top
        ``rows`` rows (a row-padded head, whose pad rows are zeroed
        once)."""
        dims, idx = self._block_address(coords, self.gkv, self.grid[self.gkv])
        dst = get_slot(slots, slot)
        if rows is not None:
            dst = self._rows(dst, rows)
        voyager.async_copy(
            src,
            dst,
            idx,
            (1,) * self.nb + tuple(tile),
            None,
            dims,
            transposed=transposed,
            pad=[0] * (self.nb + len(tile)) if pad else None,
        )

    def _load_mask(self, mask, slots, slot, coords):
        """DMA the mask tile of the pair at ``coords`` into
        ``slots[slot]``: the explicit mask's block, or under ``is_causal``
        the ``causal_mask`` entry for the pair's offset (module
        docstring)."""
        if self.causal:
            q, kv = coords[self.gq], coords[self.gkv]
            offset = (q * self.tq) % self.sq_orig + self.tq - kv * self.tkv
            dims, idx = [0], [offset // self.causal_gcd - 1]
        else:
            dims = [d for d, _ in self.mask_dyn]
            idx = [coords[g] for _, g in self.mask_dyn]
        munit = (1,) * (mask.ndim - 2)
        voyager.async_copy(
            mask,
            get_slot(slots, slot),
            idx,
            munit + (self.tq, self.tkv),
            None,
            dims,
        )

    def _store_out(self, tile, out, coords):
        """DMA the finished output ``tile`` of the sweep at ``coords`` into
        ``out``."""
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        unit = (1,) * self.nb
        voyager.async_copy(tile, out, idx, unit + (self.tq, self.d), None, dims)

    def _finalize(self, b, l, o):
        """The finished output tile, ``o / l`` at the output dtype, into
        the output slot.  ``b`` holds the buffers (``_alloc``), ``l`` is
        the tile's running sum and ``o`` its output (``_finish``)."""
        voyager.insert((o / l).to(self.out_dtype), get_slot(b.out_slots, 0))

    @staticmethod
    def _rows(tile, rows):
        """The top ``rows`` rows of an SRAM ``tile``, as a subview."""
        shape = list(tile.shape)
        return voyager.subview(
            tile,
            [0] * len(shape),
            shape[:-2] + [rows, shape[-1]],
            [1] * len(shape),
        )

    @staticmethod
    def _cols(tile, cols):
        """The first ``cols`` columns of an SRAM ``tile``, as a subview."""
        shape = list(tile.shape)
        return voyager.subview(
            tile, [0] * len(shape), shape[:-1] + [cols], [1] * len(shape)
        )

    def _matmul(self, *tiles):
        """The matrix unit's product of two tiles, ``a @ b``: ``(a, b)``, or
        under MX ``(a, a_scale, b, b_scale, a_code, b_code)`` -- each
        operand's codes, then its block scales, then the codebooks, which
        a format without one (``fp4``) leaves out."""
        if self.block_size is None:
            a, b = tiles
            return torch.matmul(a, b)
        a, a_scale, b, b_scale, *codes = tiles
        a_code, b_code = codes or (None, None)
        return torch.ops.quantized_ops.matmul_mx(
            a,
            b,
            input_scale=a_scale,
            weight_scale=b_scale,
            block_size=self.block_size,
            input_code=a_code,
            weight_code=b_code,
        )

    def _matmul_qk(self, b, operands, slot, masked=True):
        """[A]: S = (Q @ Kᵀ)·scale (+ mask) -> the scores slot ``slot``.
        ``b`` holds the buffers (``_alloc``); ``operands`` are the product's
        tiles
        (``_matmul``), then the mask tile when there is one and, for int1
        codes, its scale, which a dequantize in the same pass turns into
        the additive fill.  The mask is added unless ``masked`` is false --
        a causal pair whose block lies wholly below the diagonal
        (``_crosses``), whose mask tile is all zeros.  The op branches on
        ``has_mask`` and ``masked`` at build time, so the traced graph
        carries no runtime cond."""
        tiles = list(operands)
        if self.mask_scaled:
            *tiles, mask, mask_scale = tiles
            if masked:
                mask = torch.ops.quantized_ops.dequantize(mask, mask_scale)
        elif self.has_mask:
            *tiles, mask = tiles
        out = self._matmul(*tiles) * self.scale
        if self.has_mask and masked:
            out = out + mask
        voyager.insert(out, get_slot(b.s_buf, slot))

    def _crosses(self, coords):
        """Whether the mask of the pair at ``coords`` can drop a key: an
        explicit mask's always, a causal one's by ``attention_crosses`` (an
        int, or the loop's SymBool)."""
        if not self.causal:
            return True
        q, kv = coords[self.gq], coords[self.gkv]
        return attention_crosses(q, kv, self.tq, self.tkv, self.sq_orig)

    def _matmul_pv(self, b, operands, slot):
        """[B]: pv_buf = P @ V.  ``b`` holds the buffers (``_alloc``);
        ``operands`` are the product's tiles (``_matmul``).  ``slot`` is the
        parity of the step whose P it multiplies, for a kernel keeping two
        results; the single ``pv_buf`` takes no slot."""
        voyager.insert(self._matmul(*operands), get_slot(b.opv, 1))

    def _softmax(
        self,
        s_slot,
        p_slot,
        m_old,
        m_new,
        l,
        row_tmp,
        alpha,
        p_scale,
        probs,
        first,
    ):
        """[D]'s chain: rowmax / m / alpha / P / rowsum / l (the baseline
        passes 2-7), from the running max ``m_old`` to ``m_new``, a region
        of its own.  ``first`` (a bool, or the loop's SymBool) marks a
        tile's first block, which writes ``l`` as the rowsum; a later block
        adds the rowsum to ``alpha·l``.  Unquantized, P lands in ``p_slot``
        -- ``s_slot`` itself unless the probabilities have their own buffer
        -- and the rowsum reads it back, so ``l`` sums the P the matmul
        will see.  Under MX, S is exponentiated in place and summed exact,
        then one more pass quantizes it along the keys with the query's
        parameters -- ``probs``: the lookup table, the scale codebook and
        the midpoints -- into ``p_slot``'s codes and ``p_scale``."""
        mx = probs is not None
        voyager.insert(torch.amax(s_slot, dim=-1, keepdim=True), row_tmp)
        voyager.insert(torch.maximum(m_old, row_tmp), m_new)
        voyager.insert(torch.exp(m_old - m_new), alpha)
        exp_slot = s_slot if mx else p_slot
        voyager.insert(torch.exp(s_slot - m_new), exp_slot)

        def start():
            voyager.insert(torch.sum(exp_slot, dim=-1, keepdim=True), l)

        def accumulate():
            voyager.insert(torch.sum(exp_slot, dim=-1, keepdim=True), row_tmp)
            voyager.insert(alpha * l + row_tmp, l)

        if first is True:
            start()
        elif first is False:
            accumulate()
        else:
            effect_cond(first, start, accumulate)
        if mx:
            qmap, scale_qmap, code = probs
            scale, codes = torch.ops.quantized_ops.quantize_mx(
                s_slot,
                qmap,
                [-1],
                self.block_size,
                self.probs_quant_max,
                self.force_scale_power_of_two,
                scale_qmap,
                code,
            )
            voyager.insert(scale, p_scale)
            voyager.insert(codes, p_slot)

    def _softmax_step(self, b, slot, sweep_slot, probs, kv):
        """[D] on the scores of step parity ``slot``, into the running sum
        of sweep parity ``sweep_slot`` (``_softmax``), the running max read
        from ``m``'s slot ``slot`` -- the -inf slot 2 at ``kv`` 0, a new Q
        tile, whose block writes ``l`` -- and written to the other.  ``b``
        holds the buffers (``_alloc``); ``probs`` is what P's
        ``quantize_mx`` takes under MX, else ``None``; ``kv`` is the step's
        block in its tile, the loop's SymInt, or ``None`` for the
        prologue's step 0, a first block."""
        if kv is None:
            fresh, first = 1, True
        else:
            # 1 at a tile's first block, else 0 (``kv`` is below ``N``).
            fresh, first = 1 - (kv + self.N - 1) // self.N, kv == 0
        read = slot + (2 - slot) * fresh
        torch._check(read < 3)
        self._softmax(
            get_slot(b.s_buf, slot),
            get_slot(b.p_buf, slot),
            get_slot(b.m, read),
            get_slot(b.m, 1 - slot),
            get_slot(b.l, sweep_slot),
            b.row_tmp,
            b.alpha,
            get_slot(b.p_scale, slot) if probs is not None else None,
            probs,
            first,
        )

    def _opv(self, b, slot):
        """The real columns of ``b.opv``'s slot ``slot``: ``o`` (0) or P@V's
        result (1).  Taken where it is read -- a view captured beside its
        buffer would alias a ``while_loop`` input."""
        tile = get_slot(b.opv, slot)
        if self.d_pad != self.d:
            return self._cols(tile, self.d)
        return tile

    def _o(self, b):
        """The running output ``o`` (``_opv``)."""
        return self._opv(b, 0)

    def _pv_cols(self, b):
        """What the accumulate reads of P@V's result (``_opv``)."""
        return self._opv(b, 1)

    def _land_pv(self, b, slot):
        """[E]'s first part: add a Q tile's last P@V into ``o``.  ``b``
        holds the buffers (``_alloc``); ``slot`` is the parity of the step
        whose P it multiplied, for a kernel keeping two results
        (``_matmul_pv``)."""
        o = self._o(b)
        voyager.insert(o + self._pv_cols(b), o)

    def _accumulate_pv(self, b, slot):
        """[F]: the deferred rescale fused with the P@V accumulate, ``o =
        alpha·(o + pv)``.  ``b`` holds the buffers (``_alloc``); ``slot`` is
        the parity of the step whose P it multiplied, for a kernel keeping
        two results (``_matmul_pv``)."""
        o = self._o(b)
        voyager.insert(b.alpha * (o + self._pv_cols(b)), o)

    def _start_pv(self, b):
        """[F] at a tile's second step: its first P@V rescaled, ``o =
        alpha·pv``, written rather than added to ``o``.  ``b`` holds the
        buffers (``_alloc``)."""
        voyager.insert(b.alpha * self._pv_cols(b), self._o(b))

    def _finish(self, b, res, l, m, res_tiles, pv_slot, src, land):
        """[E]'s compute for the tile that ended: land its last P@V into
        ``o`` (``_land_pv``), fold in its residual and finalize its output,
        ``b.opv``'s slot ``src``, divided by ``l``, into the output slot.
        The output is ``o`` (``src`` 0, after the land), or for a tile of
        one block, which has nothing in ``o``, its P@V (``src`` 1, no land
        needed).  Every slot but ``src`` comes in as a view.

        Args:
            b: The buffers (``_alloc``).
            res: The residual's (``_residual_setup``), or ``None``.
            l: The tile's running sum, which it divides by.
            m: The running max the tile left (the residual's start).
            res_tiles: The residual's tiles of the tile's sweep
                (``_residual_tiles``), or ``None``.
            pv_slot: The parity of the step whose P the last P@V
                multiplied.
            src: ``b.opv``'s slot holding the output, an int or a SymInt.
            land: Whether to land the last P@V into ``o``.
        """
        if land:
            self._land_pv(b, pv_slot)
        o = self._opv(b, src)
        if res is not None:
            self._residual_step(res, b, res_tiles, l, o, m)
            o = self._o(b)
        self._finalize(b, l, o)

    def _stream(self, src, tile):
        """Two SRAM slots of ``tile`` for ``src``'s blocks."""
        unit = (1,) * self.nb
        return voyager.alloc([*unit, *tile], src.dtype, _SRAM, 2)

    def _alloc(self, b):
        """Allocate the kernel's buffers into ``b``, which holds its
        operands as folded (``forward``): the output, the slots the DMA
        streams fill -- Q, K and V, each with its block scales' under MX,
        and the mask's (one of a mask loaded once, ``mask_once``) -- the
        output slot and what the steps compute in (``_alloc_state``)."""
        unit = (1,) * self.nb
        tq, tkv, d_pad = self.tq, self.tkv, self.d_pad
        bs = self.block_size
        b.out = voyager.alloc(self.out_shape, self.out_dtype)
        b.q_slots = self._stream(b.q, self.q_tile)
        b.k_slots = self._stream(b.k, (d_pad, tkv))
        b.v_slots = self._stream(b.v, (tkv, d_pad))
        if bs is not None:
            # A block wider than the head is one truncated block.
            d_blocks_pad = math.ceil(d_pad / bs)
            b.q_scale_slots = self._stream(b.q_scale, (tq, d_blocks_pad))
            b.k_scale_slots = self._stream(b.k_scale, (d_blocks_pad, tkv))
            b.v_scale_slots = self._stream(b.v_scale, (tkv // bs, d_pad))
        if self.has_mask:
            munit = (1,) * (b.mask.ndim - 2)
            m_count = 1 if self.mask_once else 2
            b.m_slots = voyager.alloc(
                [*munit, tq, tkv], b.mask.dtype, _SRAM, m_count
            )
        # The output slot is single-slotted (written once per Q tile).
        b.out_slots = voyager.alloc(
            [*unit, *self.out_tile], b.out.dtype, _SRAM, 1
        )
        self._alloc_state(b)

    def _alloc_state(self, b):
        """Allocate into ``b`` what the steps compute in: the running
        softmax state -- the max in three slots, the sum per sweep parity,
        the rest single-buffered (module docstring) -- the scores per step
        parity, P's buffer
        (``p_buf``, with its block scales ``p_scale`` under MX), P@V's
        result and the per-row temporaries."""
        unit = (1,) * self.nb
        tq, tkv, d = self.tq, self.tkv, self.d
        acc = self.acc_dtype
        b.m = voyager.alloc([*unit, tq, 1], acc, _SRAM, 3)
        voyager.async_copy(
            self.max_fill,
            get_slot(b.m, 2),
            [0] * (self.nb + 2),
            list(self.max_fill.shape),
        )
        b.l = voyager.alloc([*unit, tq, 1], acc, _SRAM, 2)
        # ``o`` (slot 0) and P@V's result (slot 1) share a buffer, so a
        # finalize picks its source by slot (``_finish``).
        b.opv = voyager.alloc([*unit, tq, self.d_pad], acc, _SRAM, 2)
        b.s_buf = voyager.alloc([*unit, tq, tkv], acc, _SRAM, 2)
        # P: under MX its codes and block scales (module docstring); else S
        # exponentiated in place, unless the softmax runs at a dtype the
        # matrix unit cannot take as an operand.
        b.p_scale = None
        if self.block_size is not None:
            b.p_buf = voyager.alloc([*unit, tq, tkv], b.q.dtype, _SRAM, 2)
            b.p_scale = voyager.alloc(
                [*unit, tq, tkv // self.block_size],
                b.q_scale.dtype,
                _SRAM,
                2,
            )
        elif acc != b.v.dtype:
            b.p_buf = voyager.alloc([*unit, tq, tkv], b.v.dtype, _SRAM, 2)
        else:
            b.p_buf = b.s_buf
        b.row_tmp = voyager.alloc([*unit, tq, 1], acc, _SRAM)
        b.alpha = voyager.alloc([*unit, tq, 1], acc, _SRAM)

    def _load_q(self, b, slot, coords):
        """DMA the query tile of the sweep at ``coords`` into slot ``slot``
        (``_load_sweep``), column-padded to the head; under MX its block
        scales beside it.  ``b`` holds the buffers (``_alloc``)."""
        d, tq, d_pad = self.d, self.tq, self.d_pad
        self._load_sweep(
            b.q,
            b.q_slots,
            slot,
            coords,
            (tq, d_pad),
            pad=d_pad != d,
        )
        if self.block_size is not None:
            d_blocks = math.ceil(d / self.block_size)
            d_blocks_pad = math.ceil(d_pad / self.block_size)
            self._load_sweep(
                b.q_scale,
                b.q_scale_slots,
                slot,
                coords,
                (tq, d_blocks_pad),
                pad=d_blocks_pad != d_blocks,
            )

    def _load_key(self, b, slot, coords):
        """DMA the key block at ``coords`` into Kᵀ slot ``slot``, transposed
        by the DMA (``_load_step``); a row-padded head's lands in the
        slot's top rows.  ``b`` holds the buffers (``_alloc``)."""
        d = self.d
        self._load_step(
            b.k,
            b.k_slots,
            slot,
            coords,
            (self.tkv, d),
            True,
            rows=d if self.d_pad != d else None,
        )

    def _residual_setup(self, operands, acc, q):
        """The residual's operands and on-chip state, or ``None`` without
        one.  ``q`` is the main query as folded: an unquantized attention
        reads it as the residual query too (one operand serving both
        names), and the kernel must not take two views of it."""
        index = self.operand_index
        if "key_residual" not in index:
            return None
        k, v, mask = (
            operands[index[n]]
            for n in ("key_residual", "value_residual", "residual_mask")
        )
        length = k.shape[-2]
        if index["query_residual"] != index["query"]:
            q = operands[index["query_residual"]]
            if self.g_head > 1:
                q = q.reshape(self.q_fold_shape)
        if self.g_head > 1:
            mask = fold_mask_tensor(
                mask,
                mask_broadcast=self.mask_broadcast,
                sq_orig=self.sq_orig,
                g_head=self.g_head,
                mask_fold_shape=(*self.mask_fold_shape[:-1], length),
            )
        unit = (1,) * self.nb
        tq, d_pad = self.tq, self.d_pad

        q_slots = self._stream(q, (tq, d_pad))
        k_slots = self._stream(k, (d_pad, length))
        v_slots = self._stream(v, (length, d_pad))
        munit = (1,) * (mask.ndim - 2)
        m_slots = voyager.alloc([*munit, tq, length], mask.dtype, _SRAM, 2)
        s = voyager.alloc([*unit, tq, length], acc, _SRAM)
        p = s
        if acc != v.dtype:
            p = voyager.alloc([*unit, tq, length], v.dtype, _SRAM)
        return SimpleNamespace(
            q=q,
            k=k,
            v=v,
            mask=mask,
            length=length,
            q_slots=q_slots,
            k_slots=k_slots,
            v_slots=v_slots,
            m_slots=m_slots,
            s=s,
            p=p,
            pv=voyager.alloc([*unit, tq, d_pad], acc, _SRAM),
            m=voyager.alloc([*unit, tq, 1], acc, _SRAM),
        )

    def _residual_prefetch(self, res, slot, coords):
        """DMA the residual's tiles of the sweep at ``coords`` into slot
        ``slot``: the query tile like Q's, K and V whole (K transposed),
        and the mask's block, whole along the keys."""
        unit = (1,) * self.nb
        tq, d, d_pad, length = self.tq, self.d, self.d_pad, res.length
        pad = [0] * (self.nb + 2) if d_pad != d else None
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        voyager.async_copy(
            res.q,
            get_slot(res.q_slots, slot),
            idx,
            unit + (tq, d_pad),
            None,
            dims,
            pad=pad,
        )
        dims, idx = self._block_address(coords, self.gkv, 1)
        k_tile = get_slot(res.k_slots, slot)
        if d_pad != d:
            k_tile = self._rows(k_tile, d)
        voyager.async_copy(
            res.k,
            k_tile,
            idx,
            unit + (length, d),
            None,
            dims,
            transposed=True,
        )
        voyager.async_copy(
            res.v,
            get_slot(res.v_slots, slot),
            idx,
            unit + (length, d_pad),
            None,
            dims,
            pad=pad,
        )
        dyn = [(j, g) for j, g in self.mask_dyn if g != self.gkv]
        munit = (1,) * (res.mask.ndim - 2)
        voyager.async_copy(
            res.mask,
            get_slot(res.m_slots, slot),
            [coords[g] for _, g in dyn],
            munit + (tq, length),
            None,
            [j for j, _ in dyn],
        )

    def _residual_tiles(self, res, slot):
        """The residual's query, key, mask and value tiles in its slots of
        sweep parity ``slot``."""
        return tuple(
            get_slot(slots, slot)
            for slots in (res.q_slots, res.k_slots, res.m_slots, res.v_slots)
        )

    def _residual_step(self, res, b, tiles, l, o, m):
        """Fold the residual block of the sweep that just ended, its
        ``tiles`` (``_residual_tiles``), into its state: S = (q @ Kᵀ)·scale
        + mask, the softmax chain from the max ``m`` the sweep left (its new
        max into ``res.m``) into its sum ``l``, P @ V, then the running
        output ``alpha·o + pv``, ``o`` the sweep's output (``_finish``).
        ``b`` holds the buffers (``_alloc``)."""
        q, k, mask, v = tiles
        voyager.insert(torch.matmul(q, k) * self.scale + mask, res.s)
        row_tmp, alpha = b.row_tmp, b.alpha
        self._softmax(
            res.s, res.p, m, res.m, l, row_tmp, alpha, None, None, False
        )
        voyager.insert(torch.matmul(res.p, v), res.pv)
        pv = self._cols(res.pv, self.d) if self.d_pad != self.d else res.pv
        voyager.insert(alpha * o + pv, self._o(b))

    def _fold_mask(self, mask):
        return fold_mask_tensor(
            mask,
            mask_broadcast=self.mask_broadcast,
            sq_orig=self.sq_orig,
            g_head=self.g_head,
            mask_fold_shape=self.mask_fold_shape,
        )

    def forward(self, *operands):
        index = self.operand_index
        q, k, v = (operands[index[n]] for n in ("query", "key", "value"))
        mask = operands[index["attn_mask"]] if "attn_mask" in index else None
        if self.causal:
            mask = self.causal_mask
        bs = self.block_size
        mx = bs is not None
        q_scale = k_scale = v_scale = probs = None
        codes = []
        if mx:
            q_scale, k_scale, v_scale = (
                operands[index[n]]
                for n in ("query_scale", "key_scale", "value_scale")
            )
            codes = [
                operands[index[n]]
                for n in ("input_code", "weight_code")
                if n in index
            ]
            probs = tuple(
                operands[index[n]] if n in index else None
                for n in ("probs_qmap", "probs_scale_qmap", "probs_code")
            )
        # GQA fold: reshape q (and mask) so the shared query heads become extra
        # query rows; the output is reshaped back at the end.
        if self.fold_query:
            q = q.reshape(self.q_fold_shape)
            if mx:
                q_scale = q_scale.reshape(
                    (*self.q_fold_shape[:-1], q_scale.shape[-1])
                )
        if mask is not None and not self.causal and self.g_head > 1:
            mask = self._fold_mask(mask)
        grid = self.grid
        num_steps = self.num_steps
        tkv, d, d_pad = self.tkv, self.d, self.d_pad
        padded = d_pad != d
        # The head's block scales: a block wider than the head is one
        # truncated block.  ``d_blocks`` is DRAM's, ``d_blocks_pad`` the
        # tile's.
        d_blocks = math.ceil(d / bs) if mx else None
        d_blocks_pad = math.ceil(d_pad / bs) if mx else None
        blocks_padded = mx and d_blocks_pad != d_blocks
        b = SimpleNamespace(
            q=q,
            k=k,
            v=v,
            mask=mask,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        self._alloc(b)
        res = self._residual_setup(operands, self.acc_dtype, q)

        # The K and V DMA streams, each with its scales under MX.  K's DRAM
        # block (and its scales') lands in the Kᵀ tile the product reads
        # (``_load_key``); the mask block rides with K, unless every step
        # reads the same one (``mask_once``).
        def load_k(slot, coords):
            self._load_key(b, slot, coords)
            if mx:
                self._load_step(
                    k_scale,
                    b.k_scale_slots,
                    slot,
                    coords,
                    (tkv, d_blocks),
                    True,
                    rows=d_blocks if blocks_padded else None,
                )
            if self.has_mask and not self.mask_once:
                self._load_mask(mask, b.m_slots, slot, coords)

        def load_v(slot, coords):
            self._load_step(
                v,
                b.v_slots,
                slot,
                coords,
                (tkv, d_pad),
                False,
                pad=padded,
            )
            if mx:
                self._load_step(
                    v_scale,
                    b.v_scale_slots,
                    slot,
                    coords,
                    (tkv // bs, d_pad),
                    False,
                    pad=padded,
                )

        # The tiles each product reads at the given slots.
        def qk_operands(q_slot, k_slot):
            q_tile = get_slot(b.q_slots, q_slot)
            k_tile = get_slot(b.k_slots, k_slot)
            if mx:
                tiles = [
                    q_tile,
                    get_slot(b.q_scale_slots, q_slot),
                    k_tile,
                    get_slot(b.k_scale_slots, k_slot),
                    *codes,
                ]
            else:
                tiles = [q_tile, k_tile]
            if self.has_mask:
                m_slot = 0 if self.mask_once else k_slot
                tiles.append(get_slot(b.m_slots, m_slot))
            if self.mask_scaled:
                tiles.append(self.mask_scale)
            return tiles

        def pv_operands(p_slot, v_slot):
            p_tile = get_slot(b.p_buf, p_slot)
            v_tile = get_slot(b.v_slots, v_slot)
            if mx:
                tiles = [
                    p_tile,
                    get_slot(b.p_scale, p_slot),
                    v_tile,
                    get_slot(b.v_scale_slots, v_slot),
                    *codes,
                ]
            else:
                tiles = [p_tile, v_tile]
            return tiles

        # ---- prologue: prime the DMA and run step 0's [A] + [D] --------
        # The walk's first two steps: the second is sweep 0's next block,
        # or sweep 1's first when sweep 0 has a single block.
        sweeps, basis = self.sweeps, grid[:-1]
        c0 = _unravel(0, grid)
        sweep1 = int(self._kv_last(0) == 0)
        c1 = (*_unravel(sweep1, basis), 1 - sweep1)
        # The row-padded Kᵀ tiles' pad rows, zeroed once before any load
        # lands in the slots (module docstring), by a copy of a zero tile.
        row_padded = [(b.k_slots, "key_pad_rows")] if padded else []
        if blocks_padded:
            row_padded.append((b.k_scale_slots, "key_scale_pad_rows"))
        if padded and res is not None:
            row_padded.append((res.k_slots, "residual_key_pad_rows"))
        for slots, name in row_padded:
            zeros = getattr(self, name)
            for slot in range(2):
                voyager.async_copy(
                    zeros,
                    get_slot(slots, slot),
                    [0] * zeros.dim(),
                    list(zeros.shape),
                )
        self._load_q(b, 0, c0)
        if self.mask_once:
            self._load_mask(mask, b.m_slots, 0, c0)
        load_k(0, c0)
        if num_steps > 1:
            load_k(1, c1)
            if sweep1:
                # Step 0 is also its sweep's LAST step, so the uniform
                # pattern's end-of-sweep Q prefetch belongs here too.
                self._load_q(b, 1, c1)
        if sweep1 and res is not None:
            self._residual_prefetch(res, 0, c0)
        # V's stream runs one step behind K's: block 0 lands in the slot
        # step 1 reads (slot 1); nothing is fetched for step 0, which
        # consumes no V.
        load_v(1 % 2, c0)

        if self.zeroes_state:
            self._reset(b, 0)
            o = self._o(b)
            voyager.insert(torch.zeros_like(o), o)
        self._matmul_qk(b, qk_operands(0, 0), 0, self._crosses(c0))
        self._softmax_step(b, 0, 0, probs, None)

        # ---- the uniform loop: t = 1 .. num_steps - 1 -------------------
        # Carried beside the step: the (sweep, kv) pair it runs, which the
        # walk advances (no step count maps back to its pair).
        def cond_fn(step, sweep, kv):
            return step < num_steps

        def body_fn(step, sweep, kv):
            cur = (*voyager.delinearize_index(sweep, basis), kv)
            last = self._kv_last(cur[self.gq])
            nxt_sweep = sweep + (kv + 1) // (last + 1)
            nxt_kv = (kv + 1) % (last + 1)
            nxt = (*voyager.delinearize_index(nxt_sweep, basis), nxt_kv)
            # The sweep that just ended, whose output a boundary stores;
            # wrapped so the index stays in range on sweep 0, whose own
            # boundary was the prologue's.
            prev = voyager.delinearize_index(
                (sweep + sweeps - 1) % sweeps, basis
            )
            cur_slot = step % 2
            nxt_slot = (step + 1) % 2
            torch._check(cur_slot < 2)
            torch._check(nxt_slot < 2)

            # DMA phase: K (and mask) prefetch the next step's block,
            # gated off on the last iteration; V fetches the CURRENT
            # step's block into the slot the next step reads (the lag);
            # Q prefetches the next sweep's tile at each sweep's end.
            def k_fetch():
                load_k(nxt_slot, nxt)
                return 1

            torch.cond(step + 1 < num_steps, k_fetch, lambda: 0)
            load_v(nxt_slot, cur)

            next_sweep_slot = nxt_sweep % 2
            torch._check(next_sweep_slot < 2)

            def q_fetch():
                self._load_q(b, next_sweep_slot, nxt)
                return 1

            torch.cond(
                (kv == last) & (step + 1 < num_steps), q_fetch, lambda: 0
            )

            sweep_slot = sweep % 2
            torch._check(sweep_slot < 2)
            if res is not None:
                # The sweep's residual, consumed at the next boundary.
                def res_fetch():
                    self._residual_prefetch(res, sweep_slot, cur)
                    return 1

                torch.cond(kv == last, res_fetch, lambda: 0)

            # [A] current block's scores on the matrix unit, once its K (and
            # mask) block and its sweep's Q tile have landed.  V is not
            # waited here — it feeds [B], so [A] issues while V's DMA is in
            # flight.  The sweep's parity picks its Q slot and its ``l``.
            def qk(masked):
                def run():
                    tiles = qk_operands(sweep_slot, cur_slot)
                    self._matmul_qk(b, tiles, cur_slot, masked)
                    return 1

                return run

            if self.causal:
                torch.cond(self._crosses(cur), qk(True), qk(False))
            else:
                qk(True)()

            # [B] the lagged P@V on the matrix unit: previous step's
            # probabilities × its V block (in this step's V read slot) into
            # pv_buf.
            prev_slot = (step - 1) % 2
            torch._check(prev_slot < 2)
            tiles = pv_operands(prev_slot, cur_slot)
            self._matmul_pv(b, tiles, prev_slot)

            # [C] Q-tile boundary, first half, under ``zeroes_state``: fresh
            # state for the new tile.  The tile that just ended keeps its
            # sum in the other l slot for the finalize, which waits until
            # after [D].
            def reset():
                self._reset(b, sweep_slot)
                return 1

            if self.zeroes_state:
                torch.cond(kv == 0, reset, lambda: 0)

            # [D] softmax of the current block on the vector unit — runs
            # (synchronously) while the matrix [B] matmul is still in flight.
            self._softmax_step(b, cur_slot, sweep_slot, probs, kv)

            # [E] Q-tile boundary, second half: land the previous tile's last
            # P@V into o, finalize (o / its l) to ITS rows, zero o for the
            # new tile under ``zeroes_state``.
            prev_sweep_slot = (sweep + 1) % 2
            torch._check(prev_sweep_slot < 2)

            def finish(single):
                def run():
                    self._finish(
                        b,
                        res,
                        get_slot(b.l, prev_sweep_slot),
                        get_slot(b.m, cur_slot),
                        (
                            self._residual_tiles(res, prev_sweep_slot)
                            if res is not None
                            else None
                        ),
                        prev_slot,
                        1 if single else 0,
                        not single,
                    )
                    self._store_out(get_slot(b.out_slots, 0), b.out, prev)
                    if self.zeroes_state:
                        o = self._o(b)
                        voyager.insert(torch.zeros_like(o), o)
                    return 1

                return run

            # Whether the tile that ended had one block, which decides
            # where its output is (``_finish``).  A SymBool predicate
            # cannot be captured into an outer cond, so a runtime one
            # splits the boundary into two flat conds.
            boundary = kv == 0
            prev_last = 1 if self.zeroes_state else self._kv_last(prev[self.gq])
            if self.causal and not self.zeroes_state:
                single = boundary & (prev_last == 0)
                torch.cond(single, finish(True), lambda: 0)
                torch.cond(boundary & (prev_last > 0), finish(False), lambda: 0)
            else:
                torch.cond(boundary, finish(prev_last == 0), lambda: 0)

            # [F] deferred rescale fused with the P@V accumulate: o = alpha·(o
            # + pv), after the softmax (kv >= 1 only); at kv == 1 the tile's
            # first P@V, o = alpha·pv, unless under ``zeroes_state``.
            def rescale():
                self._accumulate_pv(b, prev_slot)
                return 1

            def start():
                self._start_pv(b)
                return 1

            if self.zeroes_state:
                torch.cond(kv > 0, rescale, lambda: 0)
            else:
                torch.cond(kv == 1, start, lambda: 0)
                torch.cond(kv > 1, rescale, lambda: 0)
            return (step + 1, nxt_sweep, nxt_kv)

        steps, sweep, kv = while_loop(cond_fn, body_fn, (1, sweep1, 1 - sweep1))

        # ---- epilogue: the final tile's leftover P@V + finalize --------
        # It reads only what the loop returns, so it is right for however
        # many steps ran (a ``MAX_TILES`` cut): the last step ran in the
        # sweep the loop left, or in the one before when it ended a sweep
        # (``kv`` back at 0; ``kv <= N``, so the floor is 1 iff ``kv > 0``).
        last_sweep = sweep - 1 + (kv + self.N - 1) // self.N
        c_last = (*voyager.delinearize_index(last_sweep, basis), 0)
        v_slot = steps % 2
        pv_slot = (steps - 1) % 2
        last_sweep_slot = last_sweep % 2
        torch._check(v_slot < 2)
        torch._check(pv_slot < 2)
        torch._check(last_sweep_slot < 2)
        # [B] the final block's P@V on the matrix unit, landed into o.
        self._matmul_pv(b, pv_operands(pv_slot, v_slot), pv_slot)
        # The output is in ``o`` after the land, or in P@V's result when
        # the tile had one block: its last step's block, ``kv - 1`` or the
        # tile's last block when the step ended its sweep, is then 0.  The
        # slot is picked by arithmetic, as no cond can take the loop's
        # results; a land on a tile of one block computes from a stale
        # ``o`` and is not read.
        src = 0
        if not self.zeroes_state:
            tile_last = self._kv_last(c_last[self.gq])
            ended = 1 - (kv + self.N - 1) // self.N
            last_kv = kv - 1 + ended * (tile_last + 1)
            src = 1 - (last_kv + self.N - 1) // self.N
            torch._check(src < 2)
        self._finish(
            b,
            res,
            get_slot(b.l, last_sweep_slot),
            get_slot(b.m, v_slot),
            (
                self._residual_tiles(res, last_sweep_slot)
                if res is not None
                else None
            ),
            pv_slot,
            src,
            True,
        )
        # Then the final tile goes out.
        self._store_out(get_slot(b.out_slots, 0), b.out, c_last)
        out = b.out
        if self.g_head > 1:
            out = out.reshape(self.out_unfold_shape)
        return out


def _operand_index(node):
    """Position of each named operand -- query, key, value, attn_mask and
    the MX kwargs -- among ``node.all_input_nodes``, the order
    ``_FA3Pipeline.forward`` receives them."""
    operands = {
        "query": node.args[0],
        "key": node.args[1],
        "value": node.args[2],
    }
    for position, name in _OPTIONAL_OPERANDS.items():
        operand = get_arg_value(node, position, name, None)
        if isinstance(operand, torch.fx.Node):
            operands[name] = operand
    operands.update(
        (name, n)
        for name, n in node.kwargs.items()
        if isinstance(n, torch.fx.Node)
    )
    inputs = list(node.all_input_nodes)
    return {name: inputs.index(n) for name, n in operands.items()}


def _stamp_probs_dtypes(gm, dtypes):
    """Stamp ``dtypes`` -- P's ``(scale, codes)`` logical dtypes, the
    query's -- on every ``quantize_mx`` in ``gm`` and its nested graphs:
    no node of the original graph carries them for the bufferizer's
    propagation to read."""
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is _QUANTIZE_MX:
            n.meta["dtype"] = dtypes
        elif n.op in ("get_attr", "call_module"):
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _stamp_probs_dtypes(sub, dtypes)


def _stamp_product_tilings(gm, products, keys):
    """Copy each product's search result (``attention_op_tiling``) onto
    every kernel of ``gm`` that runs one -- the bare product op or the
    fused ``call_module`` around it, at every nesting level -- the way the
    GEMM builder stamps its nest, so the emitter and the reporting model
    see the mapping the kernel runs.  ``keys`` names each product by
    ``(under MX, contraction, output width)``, which tells the scores from
    the context and the main products from a residual's."""
    named = dict(gm.named_modules())
    for n in gm.graph.nodes:
        sub = named.get(str(n.target)) if n.op != "call_function" else None
        if n.op == "get_attr" and isinstance(sub, torch.fx.GraphModule):
            _stamp_product_tilings(sub, products, keys)
            continue
        if n.op == "call_function" and n.target in _PRODUCT_OPS:
            anchor = n
        elif isinstance(sub, torch.fx.GraphModule):
            anchor = next(
                (x for x in sub.graph.nodes if x.target in _PRODUCT_OPS), None
            )
        else:
            anchor = None
        if anchor is None:
            continue
        which = keys[
            (
                anchor.target is _PRODUCT_OPS[1],
                anchor.args[0].value.shape[-1],
                anchor.value.shape[-1],
            )
        ]
        n.meta["tile_result"] = products[which]


def build_attention_fa3(
    node: torch.fx.Node,
    ctx: BuildContext,
    *,
    accumulate_fp32: bool = False,
    pipeline=_FA3Pipeline,
    **options,
) -> Optional[torch.fx.GraphModule]:
    """FA3-style pipeline builder for an
    ``aten.scaled_dot_product_attention`` node, or its ``sdpa_mx`` twin
    whose operands are MX codes with block scales, scheduled by the
    ``_FA3Pipeline`` class ``pipeline``, which may lower other ops
    (``ops``) and take ``options`` beside the shared arguments.

    Returns the bufferized ``GraphModule`` (prologue + rolled
    ``while_loop`` + epilogue over ``voyager.*`` primitives, see the
    module docstring), or ``None`` when uncovered (unsupported rank, an
    unfoldable GQA layout, no tiling and no ``ctx.tiler`` to choose one).
    GQA is supported by folding the head group into the query rows (below);
    ``dropout_p`` is ignored (identity at inference).  ``accumulate_fp32``
    runs the softmax and the output accumulator in fp32 instead of the
    output dtype, at the cost of a separate probabilities buffer at V's
    dtype when the two differ.  The block counts
    ``(num_q_blocks, num_kv_blocks)`` come from ``node.meta['l2_tiling']``
    when set, else ``attention_op_tiling`` chooses them for the op
    ``ctx.tiler``'s model prepares (``RuntimeModel.prepare_attention``),
    whose space also sets the head the tiles are padded to (``pad_head``); a
    node tiled by a padding tiler is built with that tiler, so the kernel's
    products match the ones it mapped.
    The kernel adds its mask: an explicit ``attn_mask`` must be additive,
    and under ``is_causal`` the kernel's own table is int1 codes plus a
    scale (``ctx.bool_mask``) or the additive fill at the accumulation dtype.
    """
    if node.op != "call_function" or node.target not in pipeline.ops:
        return None

    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
    mask_node = get_arg_value(node, 3, "attn_mask", None)
    dropout_p = get_arg_value(node, 4, "dropout_p", 0.0)
    is_causal = get_arg_value(node, 5, "is_causal", False)
    scale = node.kwargs.get("scale", None)
    if dropout_p:
        logger.warning(
            "%s: dropout_p=%s ignored (attention dropout is identity at "
            "inference; lowering the dropout-free kernel)",
            node.name,
            dropout_p,
        )
    if is_causal and isinstance(mask_node, torch.fx.Node):
        raise ValueError(f"{node.name}: is_causal with an explicit attn_mask")

    block_size = node.kwargs.get("block_size")  # None: unquantized

    # The operands in logical order; ``out`` as stored.
    q, k = pipeline.logical_operands(q_node.value.clone(), k_node.value.clone())
    v = v_node.value.clone()
    out = node.value
    if q.ndim < 2 or q.ndim != k.ndim or q.ndim != v.ndim:
        return None

    Sq, d = q.shape[-2], q.shape[-1]
    Skv = k.shape[-2]
    if is_causal and Skv < Sq:
        raise ValueError(
            f"{node.name}: is_causal over fewer keys ({Skv}) than queries "
            f"({Sq})"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
    op = None
    head_pad = d
    if ctx.tiler is not None:
        op = ctx.tiler.model.prepare_attention(node, acc_dtype, ctx.bool_mask)
        head_pad = pad_head(d, op.space)

    # GQA folds the head group into the query rows so K/V load once per KV head
    # and reuse across the group (see ``plan_gqa_fold``); ``forward`` applies
    # the q / mask / output reshapes.  The fold only needs the query block
    # count to check divisibility, and a count the tiler picks divides by
    # construction, so an unset tiling plans at one block.
    mask_val = mask_node.value if isinstance(mask_node, torch.fx.Node) else None
    tiling = node.meta.get("l2_tiling")
    plan = plan_gqa_fold(
        q, k, v, mask_val, out.shape, tiling[0] if tiling else 1
    )
    if plan is None:
        return None
    kbatch, g_head = plan.kbatch, plan.g_head
    if tiling is None:
        if ctx.tiler is None:
            return None
        tiling = attention_op_tiling(
            node,
            op,
            ctx.tiler,
            sq_eff=plan.sq_eff,
            kv_batch=kbatch,
        )
    num_q_blocks, num_kv_blocks = tiling
    tq, tkv = plan.sq_eff // num_q_blocks, Skv // num_kv_blocks
    if is_causal and Sq % tq:
        raise ValueError(
            f"{node.name}: a {tq}-row query tile straddles a head of {Sq} "
            "rows under is_causal"
        )
    grid = kbatch + (num_q_blocks, num_kv_blocks)

    # Mask [*mbatch, Sq, Skv]: its dynamic dims (``attention_mask_dyn``).
    # Under the fold the mask's batch dims are the folded ones.
    mask_dyn = []
    if isinstance(mask_node, torch.fx.Node):
        mask = mask_node.value
        if mask.dtype == torch.bool:
            raise ValueError(
                f"{node.name}: a bool attn_mask is not lowered; the kernel "
                "adds its mask, so pass an additive one"
            )
        mb = tuple(mask.shape[:-2])
        if g_head > 1:
            mb = plan.mask_fold_shape[:-2]
        mask_dyn = attention_mask_dyn(mb, grid)
    else:
        mask_node = None

    result = node.meta.get("tile_result")
    pattern = pipeline(
        grid=grid,
        num_kv_blocks=num_kv_blocks,
        tq=tq,
        tkv=tkv,
        head_dim=d,
        head_pad=head_pad,
        scale=float(scale),
        has_mask=mask_node is not None or is_causal,
        mask_dyn=mask_dyn,
        is_causal=is_causal,
        bool_mask=ctx.bool_mask,
        g_head=g_head,
        sq_orig=Sq,
        q_fold_shape=plan.q_fold_shape,
        out_unfold_shape=plan.out_unfold_shape,
        mask_broadcast=plan.mask_broadcast,
        mask_fold_shape=plan.mask_fold_shape,
        out_shape=plan.out_fold_shape,
        out_dtype=out.dtype,
        acc_dtype=acc_dtype,
        operand_index=_operand_index(node),
        block_size=block_size,
        probs_quant_max=node.kwargs.get("probs_quant_max"),
        force_scale_power_of_two=node.kwargs.get(
            "force_scale_power_of_two", False
        ),
        placement=result.placement if result is not None else None,
        **options,
    )

    # ``all_input_nodes`` is the order ``forward`` takes its operands
    # (``_operand_index``).
    inputs = tuple(n.value.clone() for n in node.all_input_nodes)
    pattern.register_pad_rows(inputs)
    with _lenient_verifier():
        gm = export_model(pattern, inputs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(1, pattern.num_steps, 1)]])
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*inputs)
    pattern.fuse_ops(gm)
    if block_size is not None:
        _stamp_probs_dtypes(
            gm,
            (
                node.kwargs["query_scale"].meta.get("dtype"),
                q_node.meta.get("dtype"),
            ),
        )
    if is_causal and ctx.bool_mask:
        # The table's int1 codes ride a float buffer; its tiles are sized
        # and emitted as int1 (the bufferizer carries the dtype to the slot).
        for n in gm.graph.nodes:
            if n.op == "get_attr" and str(n.target).endswith("causal_mask"):
                n.meta["dtype"] = "int1"
    if result is not None and pattern.runs_mapped_products:
        mx = block_size is not None
        keys = {
            (mx, head_pad, tkv): "scores",
            (mx, tkv, head_pad): "context",
        }
        residual = get_arg_value(node, 7, "key_residual", None)
        if residual is not None:
            length = residual.value.shape[-2]
            keys.setdefault((False, head_pad, length), "residual_scores")
            keys.setdefault((False, length, head_pad), "residual_context")
        _stamp_product_tilings(gm, result.tiling.products, keys)
    return gm
