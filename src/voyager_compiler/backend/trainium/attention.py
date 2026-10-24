"""Trainium's flash attention: the fit and cost of a K-major flash
attention (``TrainiumAttention``), and its builder,
``TrainiumFlashAttention``: the shared FA3 schedule
(``bufferize/attention_v3.py``) with each step written out as the
NeuronCore runs it, so the translator only translates.

A K-major flash attention (``sdpa_kmajor``) keeps the shared walk, DMA streams
and semaphores; ``TrainiumFlashAttention`` replaces its buffers and steps.  Its
query and key are stored ``[*batch, head_dim, tokens]``, so their tiles ``Qᵀ``
and ``Kᵀ`` are plain DMAs of the blocks QKᵀ reads.  Under the GQA fold the query
is viewed ``[*kv_batch, g_head·head_dim, tokens]``, a Q tile one head's
``head_dim`` rows by ``tq`` of its tokens, so ``tq`` divides the query length.
A query of one token -- a decode step's -- holds a row-major query's bytes, so
it folds like one, a Q tile spanning the head group's rows.  Given
``transpose_output`` the output is stored ``[*batch, head_dim, tokens]``, cut
like the query.  A Q tile is cut into blocks of 128 rows (the partitions) and
its scores into PSUM banks of 512 float32 columns, each a slot of a ``Local``
box, the boxes the tile search placed (``TrainiumAttention``):

- [A] per block and bank: QKᵀ into the bank, then on a step that adds a
  mask the mask added onto it there by a matmul through an identity
  (``matmul_accumulate``), the identity scaled by ``1 / scale`` and made on
  chip by the first such step (``scaled_identity``).
- [D] per block: the scores copied out of PSUM into the SBUF scores scaled,
  with each row's max (``tensor_scalar_reduce``) -- a bank at a time into a
  column each, then their max, when the scores span several banks -- the
  bias ``-max(m, row max)`` (``tensor_scalar``) and the exponentials in
  place with each row's sum (``activation_reduce``).  Then, over every
  block's columns at once, alpha, ``m`` and ``l``.  A per-row value is held
  in float32 columns ``[128, blocks]``, column ``b`` block ``b``'s, the type
  and layout the engines take a per-row scalar in.
- [B] per block: P transposed on TensorE into PSUM a group of 128-key blocks
  at a time, each group copied out into Pᵀ in SBUF, then P·V into the
  block's PSUM slot.
- [E] per block: P·V landed into ``o``, then ``o / l`` into the output tile,
  transposed on TensorE through PSUM when the output is stored transposed.
- [F] per block: ``o = alpha·(o + pv)``, P·V read from PSUM.

A one-token query (a decode step's) lands as stored and is transposed on
TensorE into Qᵀ through PSUM at its load.  The kernel's ops are plain aten
and ``quantized_ops`` ops, fused into the instructions they are by
``INSTRUCTION_PATTERNS``.
"""

import functools
import math
from collections import Counter
from typing import Optional

import torch

from voyager_compiler.codegen.node_info import (
    attention_shapes,
    get_arg_value,
    query_kmajor,
    writes_transposed,
)
from voyager_compiler.codegen.transform import fuse_patterns
from voyager_compiler.codegen.transform.bufferize.attention_v3 import (
    _FA3Pipeline,
    build_attention_fa3,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    commit,
    with_scratch,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import get_slot
from voyager_compiler.codegen.transform.bufferize.utils import (
    BuildContext,
    outline_dps_ops,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    get_dtype_width,
    node_dtype_bits,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    OpModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    attention_crosses,
    attention_kv_last,
    attention_mask_dyn,
)

from .gemm import alloc_placed, window
from .patterns import INSTRUCTION_PATTERNS
from .utils import (
    DIV,
    F32,
    LOCAL,
    PARTITIONS,
    ROW_PASS_BITS,
    SDPA_KMAJOR,
    SRAM,
    TrainiumPlacement,
    chain_tiles,
    extents,
    fits_psum,
    group_regions,
    half_bank_blocks,
    sbuf_bytes,
    sbuf_placed_bytes,
    scratch_regions,
    widest_bits,
)

voyager = torch.ops.voyager

# The op groups the translator computes [F] and [E] in, each with its
# scratch regions (``attention_scratch``).
RESCALE_GROUP, FINALIZE_GROUP = "rescale", "finalize"


def attention_scratch(head):
    """The scratch regions of ``TrainiumFlashAttention``'s kernel of head
    ``head``: [F]'s ``o + pv`` before the scaling by alpha, a float32 column
    (``RESCALE_GROUP``), and [E]'s reciprocals of ``l``, a float32 column
    (``FINALIZE_GROUP``)."""
    rescale = (
        (torch.ops.aten.add.Tensor, ("tile", "tile")),
        (torch.ops.aten.mul.Tensor, ("column32", "value")),
    )
    finalize = ((DIV, ("tile", "column32")),)
    return scratch_regions(
        RESCALE_GROUP, chain_tiles(rescale, head)
    ) + scratch_regions(FINALIZE_GROUP, chain_tiles(finalize, head))


class TrainiumAttention(OpModel):
    """A K-major flash attention (``sdpa_kmajor``) under ``TrainiumModel``:
    the SBUF buffers and PSUM boxes of its kernel
    (``TrainiumFlashAttention``), and its runtime.

    A step runs on its query tile 128 rows at a time.  TensorE runs QKᵀ, the
    query tile stationary and the key tile moving 512 columns at a time,
    accumulates the mask onto it in PSUM by an identity matmul on a step that
    adds one -- under ``is_causal`` only a pair crossing the diagonal
    (``attention_crosses``) -- transposes P one 128-key block at a time and runs
    P·V, each Pᵀ block stationary and the value moving.  Per 128 rows VectorE
    evicts the scores scaled with their row max a PSUM bank at a time, takes
    the banks' max over several and ``-max(M, R)``; ScalarE takes the
    exponentials with their row sum and the sum's read-out.  Pᵀ's copies out
    of PSUM, half a bank per copy, go to VectorE or ScalarE as the step
    compile places them, which balances the two.  Over every 128 rows'
    columns at once VectorE takes alpha's exponent, ``M`` and ``l = alpha·l +
    rowsum`` and ScalarE alpha.  A step past a sweep's first rescales the
    output, ``o = alpha·(o + pv)``, in two VectorE passes, P·V's result read
    from PSUM.  A sweep's first finalizes the tile before it, ``o += pv``
    and ``o / l`` on VectorE, then, for an output stored transposed, Oᵀ's
    TensorE transposes and their copies out of PSUM; its resets of m, l and
    o are GpSimd fills, short beside a step, so they are not charged.  Every
    pass over the scores or the output reads float32.  A query of one token (not
    ``query_kmajor``, a decode step's) lands row-major, its tile spanning the
    head group's rows, and a sweep's first transposes it on TensorE and copies
    it out of PSUM.

    Judged in the parent, so it keeps its node.

    Attributes:
        node: The attention node.
        model: The ``TrainiumModel``.
        space: ``TrainiumModel.attention_space``.
        key: ``None``: an attention search shares no result.
        head_dim: The head.
        bits: ``(query, key, value, output)`` element widths.
        acc_bits: The width of the softmax state, the scores and the output
            accumulator.
        mask_bits: The width of the mask tiles: the causal table's -- int1
            codes under ``bool_mask``, else the additive fill at the
            accumulation dtype -- or an explicit mask's; 0 without one.
        mask_batch: An explicit mask's batch dims, else ``None``.
        own_probs: The softmax runs at a dtype other than V's, so P has a
            buffer of its own at V's.
        causal: Only the pairs up to each query tile's last row are steps.
        query_kmajor: The query is stored ``[head_dim, tokens]`` over more
            than one token (``query_kmajor``).
        transposed_output: The output is stored ``[head_dim, tokens]``
            (``writes_transposed``).
        dtypes: ``(query, value, output)`` element types.
        acc_dtype: The type of the scores, ``o`` and the identity.
    """

    key = None

    def __init__(self, node, model, acc_dtype, bool_mask):
        self.node = node
        self.model = model
        self.space = model.attention_space
        query, key, value = node.args[0], node.args[1], node.args[2]
        self.head_dim = attention_shapes(node)[0][-1]
        self.bits = tuple(widest_bits(n) for n in (query, key, value, node))
        self.acc_bits = get_dtype_width(acc_dtype)
        self.causal = get_arg_value(node, 5, "is_causal", False)
        self.mask_batch = None
        if self.causal:
            self.mask_bits = 1 if bool_mask else self.acc_bits
        else:
            mask = get_arg_value(node, 3, "attn_mask", None)
            self.mask_bits = node_dtype_bits(mask, 0)
            if mask is not None:
                self.mask_batch = tuple(mask.shape[:-2])
        self.own_probs = acc_dtype != value.value.dtype
        self.query_kmajor = query_kmajor(node)
        self.transposed_output = writes_transposed(node)
        self.dtypes = (query.value.dtype, value.value.dtype, node.value.dtype)
        self.acc_dtype = acc_dtype

    def _mask_once(self, grid):
        """Whether the kernel loads an explicit mask once under ``grid``: the
        walk moves through none of its dims (``attention_mask_dyn``)."""
        if self.mask_batch is None:
            return False
        return not attention_mask_dyn(self.mask_batch, grid)

    def _out_tile(self, tq):
        """The output tile as stored, rows on the partitions."""
        d = self.head_dim
        return (d, tq) if self.transposed_output else (tq, d)

    def buffers(self, tiling):
        """The buffers ``TrainiumFlashAttention`` allocates beside the shared
        FA3 kernel's streams under ``tiling`` (``TrainiumPlacement.buffers``),
        by the role it names them by.

        In SBUF: the softmax state as float32 columns ``[rows, blocks]``
        (``m``, the running sum ``l`` per sweep parity, the row max or sum
        ``row_tmp``, ``alpha``, and per step parity the ``bias``, alpha's
        exponent ``delta`` and ``decayed``, alpha·l); ``o``; the scores
        ``s_buf`` per step parity, and P's own buffer ``p_buf`` when the
        softmax runs at a dtype other than V's; over several banks, their
        row maxima ``bank_max`` per step parity; the ``identity`` a mask is
        added through; Pᵀ (``probs_t``), two slots; a one-token query's
        tile as stored (``q_rows``) per sweep parity; and the finished tile
        an output stored transposed is transposed from (``out_rows``).  In
        PSUM: per step parity, a whole bank of scores per 128 query rows and
        512 key columns (``scores``) and P·V's result per 128 query rows
        (``context``); P's transposes, two tiles of half a bank of 128-key
        blocks (``transposes``); Oᵀ for an output stored transposed
        (``output_t``); and Qᵀ per sweep parity for a one-token query
        (``query_t``)."""
        config = self.model.config
        p, free = config.partitions, config.moving_free_max
        tq, tkv = tiling.tile_sizes
        d = self.head_dim
        q_dtype, v_dtype, out_dtype = self.dtypes
        q_bits, _, v_bits, out_bits = self.bits
        acc, acc_bits = self.acc_dtype, self.acc_bits
        unit = (1,) * (len(tiling.grid) - 2)
        rows, blocks = min(tq, p), math.ceil(tq / p)
        banks = math.ceil(tkv / free)
        keys = min(tkv, half_bank_blocks(config, v_bits) * p)
        columns = (*unit, rows, blocks)
        buffers = [
            ("m", columns, F32, 32, 1, SRAM),
            ("l", columns, F32, 32, 2, SRAM),
            ("o", (*unit, tq, d), acc, acc_bits, 1, SRAM),
            ("s_buf", (*unit, tq, tkv), acc, acc_bits, 2, SRAM),
        ]
        if self.own_probs:
            shape = (*unit, tq, tkv)
            buffers.append(("p_buf", shape, v_dtype, v_bits, 2, SRAM))
        buffers += [
            (role, columns, F32, 32, slots, SRAM)
            for role, slots in (
                ("row_tmp", 1),
                ("alpha", 1),
                ("bias", 2),
                ("delta", 2),
                ("decayed", 2),
            )
        ]
        if banks > 1:
            buffers.append(("bank_max", (rows, banks), F32, 32, 2, SRAM))
        if self.mask_bits:
            buffers.append(("identity", (p, p), acc, acc_bits, 1, SRAM))
        buffers.append(("probs_t", (tkv, rows), v_dtype, v_bits, 2, SRAM))
        if not self.query_kmajor:
            shape = (*unit, tq, d)
            buffers.append(("q_rows", shape, q_dtype, q_bits, 2, SRAM))
        buffers += [
            ("scores", (rows, free), F32, 32, 2 * blocks * banks, LOCAL),
            ("context", (rows, d), F32, 32, 2 * blocks, LOCAL),
            ("transposes", (keys, rows), v_dtype, v_bits, 2, LOCAL),
        ]
        if self.transposed_output:
            shape = (*unit, tq, d)
            buffers.append(("out_rows", shape, out_dtype, out_bits, 1, SRAM))
            shape = (d, tq)
            buffers.append(("output_t", shape, out_dtype, out_bits, 1, LOCAL))
        if not self.query_kmajor:
            buffers.append(("query_t", (d, tq), q_dtype, q_bits, 2, LOCAL))
        return tuple(buffers)

    def fit(self, tiling):
        """The kernel's ``TrainiumPlacement`` under ``tiling`` -- its
        buffers (``buffers``) and the regions the translator computes [E]
        and [F] in (``attention_scratch``) -- when they fit PSUM and SBUF,
        else ``None``.

        SBUF holds the shared FA3 kernel's streams as the memory plan places
        them -- two slots each of the Qᵀ, Kᵀ, V and mask tiles, one of a
        mask loaded once (``_mask_once``) and one of the output tile --
        beside the placement's.
        """
        config = self.model.config
        p = config.partitions
        tq, tkv = tiling.tile_sizes
        d = self.head_dim
        q_bits, k_bits, v_bits, out_bits = self.bits
        buffers, scratch = self.buffers(tiling), attention_scratch(d)
        if not fits_psum(config, buffers, scratch):
            return None
        needed = (
            2 * sbuf_bytes(d, tq, q_bits, p)
            + 2 * sbuf_bytes(d, tkv, k_bits, p)
            + 2 * sbuf_bytes(tkv, d, v_bits, p)
            + sbuf_bytes(*self._out_tile(tq), out_bits, p)
            + sbuf_placed_bytes(config, buffers, scratch)
        )
        if self.mask_bits:
            slots = 1 if self._mask_once(tiling.grid) else 2
            needed += slots * sbuf_bytes(tq, tkv, self.mask_bits, p)
        if needed > config.usable_scratchpad_size:
            return None
        return TrainiumPlacement(buffers=buffers, scratch=scratch)

    def cost(self, tiling, placement):
        """The attention in TensorE cycles, and its DRAM bytes: each step
        costs its busiest engine, a sweep's first its finalize too, and the
        key, value and mask tiles' DMAs on every step -- a mask loaded once
        (``_mask_once``) once -- and the query and output tiles' once per
        sweep overlap the compute around them (``sweep_ns``).  A step takes
        at least its softmax's chain: a 128-row block's passes, each waiting
        on the one before, then the columns' passes, which wait on every
        block's.  Only the other blocks' passes and the other engines' work
        run beside the chain, so a step of one block -- a decode step's --
        takes the whole chain.  ``placement`` is the contract's."""
        model = self.model
        config = model.config
        p, free = config.partitions, config.moving_free_max
        tq, tkv = tiling.tile_sizes
        d = self.head_dim
        q_bits, k_bits, v_bits, out_bits = self.bits
        grid = tiling.grid
        sweeps = math.prod(grid[:-1])
        if self.causal:
            sq = attention_shapes(self.node)[0][-2]
            pairs = [
                (q, kv)
                for q in range(grid[-2])
                for kv in range(attention_kv_last(q, tq, tkv, sq) + 1)
            ]
            steps = math.prod(grid[:-2]) * len(pairs)
            masked = math.prod(grid[:-2]) * sum(
                attention_crosses(q, kv, tq, tkv, sq) for q, kv in pairs
            )
        else:
            steps = math.prod(grid)
            masked = steps if self.mask_bits else 0
        rows = extents(tq, p)

        def tensor_ns(widths):
            """TensorE instructions moving ``widths`` columns each."""
            overhead = config.tensor_overhead_cycles
            return sum(w + overhead for w in widths) / config.frequency

        def copies_ns(engine, blocks, cols, dtype, per_copy):
            """``blocks`` transposed blocks of ``cols`` columns of ``dtype``
            copied out of PSUM, ``per_copy`` to a copy."""
            bits = 8 * dtype.itemsize
            return sum(
                model.engine_ns(engine, n * cols, bits)
                for n in extents(blocks, per_copy)
            )

        def passes(engines, cols):
            return model.passes_busy(
                [(e, 1.0, ROW_PASS_BITS) for e in engines], tq, cols
            )

        def balanced(busy):
            """``busy`` with the step's copies out of PSUM, which the step
            compile places on VectorE or ScalarE, split between the two
            so the busier one is least busy."""
            vector, scalar = busy["vector"], busy["scalar"]
            to_scalar = min(max(0.0, (vector + copies - scalar) / 2), copies)
            return busy + Counter(vector=copies - to_scalar, scalar=to_scalar)

        def engine_ns(engine, cols):
            return model.engine_ns(engine, cols, ROW_PASS_BITS)

        kv_blocks, head_blocks = math.ceil(tkv / p), math.ceil(d / p)
        step = Counter()
        # The moving columns of QKᵀ's matmuls, P's transposes and P·V's, and
        # on a masked step's share the mask's identity matmuls.
        step["tensor"] = tensor_ns(
            len(rows) * head_blocks * extents(tkv, free)
            + kv_blocks * rows
            + len(rows) * kv_blocks * extents(d, free)
        )
        step["tensor"] += (
            masked / steps * tensor_ns(len(rows) * extents(tkv, free))
        )
        # Per 128 rows: the scores evicted scaled with their row max a bank
        # at a time, over several banks the banks' max, and -max(M, R) on
        # VectorE; the exponentials with their row sum, and the sum's
        # read-out, on ScalarE.  Then over the columns of every 128 rows,
        # alpha's exponent, M and l's two on VectorE and alpha on ScalarE.
        banks = extents(tkv, free)
        block = Counter(
            vector=sum(engine_ns("vector", w) for w in banks)
            + (engine_ns("vector", len(banks)) if len(banks) > 1 else 0.0)
            + engine_ns("vector", 1),
            scalar=engine_ns("scalar", tkv) + engine_ns("scalar", 1),
        )
        columns = Counter(
            vector=4 * engine_ns("vector", len(rows)),
            scalar=engine_ns("scalar", len(rows)),
        )
        step.update({e: len(rows) * ns for e, ns in block.items()})
        step.update(columns)
        # Each pass waits on the one before: a block's, and the columns'
        # on every block's.  Only another block's passes run beside them.
        chain = sum(block.values()) + sum(columns.values())
        q_dtype, v_dtype, out_dtype = self.dtypes
        copies = sum(
            copies_ns(
                "scalar",
                kv_blocks,
                r,
                v_dtype,
                half_bank_blocks(config, v_bits),
            )
            for r in rows
        )
        # The output stores' issue, on ScalarE's queue.
        step["scalar"] += model.store_issue_ns(sweeps, steps)
        rescale = step + passes(("vector",) * 2, d)
        boundary = step + passes(("vector",) * 2, d)
        if self.transposed_output:
            boundary["tensor"] += head_blocks * tensor_ns(rows)
            boundary["vector"] += head_blocks * copies_ns(
                "vector", len(rows), min(tq, p), out_dtype, len(rows)
            )
        if not self.query_kmajor:
            # The query's transposes, one copy out of PSUM.
            boundary["tensor"] += head_blocks * tensor_ns(rows)
            boundary["vector"] += head_blocks * copies_ns(
                "vector", len(rows), min(tq, p), q_dtype, len(rows)
            )
        interior = max(*balanced(rescale).values(), chain)
        boundary_ns = max(*balanced(boundary).values(), chain)

        out_tile = self._out_tile(tq)
        moves = [
            (out_tile, out_bits, sweeps),
            ((d, tq) if self.query_kmajor else (tq, d), q_bits, sweeps),
            ((d, tkv), k_bits, steps),
            ((tkv, d), v_bits, steps),
        ]
        if self.mask_bits:
            loads = 1 if self._mask_once(grid) else steps
            moves.append(((tq, tkv), self.mask_bits, loads))
        dmas = [
            (model.dma_ns(*tile, bits, load=i > 0), n)
            for i, (tile, bits, n) in enumerate(moves)
        ]
        latency = model.sweep_ns(dmas, steps, interior)
        latency += sweeps * (boundary_ns - interior)
        traffic = sum(
            n * math.ceil(math.prod(tile) * bits / 8) for tile, bits, n in moves
        )
        return Cost(latency * config.frequency, traffic)


# A GEMM into PSUM whose activation is stored contraction-major, ``[K, M]``:
# its product summed and returned in float32.
KMAJOR_MATMUL = functools.partial(
    torch.ops.quantized_ops.matmul.default,
    layout="cx,ck->xk",
    out_dtype=torch.float32,
)


def _block(tile, starts, sizes):
    """The ``sizes`` window at ``starts`` of ``tile``'s last two dims, its
    leading dims (of one element each) dropped."""
    lead = tile.dim() - 2
    return window(
        tile, [0] * lead + list(starts), [1] * lead + list(sizes), range(lead)
    )


class TrainiumFlashAttention(_FA3Pipeline):
    """The FA3 schedule as a NeuronCore runs it (module docstring).

    Attributes:
        query_kmajor: The query is stored ``[head_dim, tokens]`` over more
            than one token; a one-token query is read as the row-major one
            it equals.
        transpose_output: The output is stored ``[head_dim, tokens]``.
        rows: Query rows of a block, at most the partitions.
        blocks: Blocks of a Q tile.
        banks: ``(column, columns)`` of each PSUM bank the scores of a block
            span.
        groups: ``(start, stop)`` of each group of keys P's transposes put
            side by side in a PSUM tile.
        shapes: The shape of each buffer the tile search placed, by role
            (``TrainiumAttention.buffers``).
    """

    ops = (SDPA_KMAJOR,)
    runs_mapped_products = False

    def __init__(self, *, query_kmajor, transpose_output, **kwargs):
        super().__init__(**kwargs)
        if self.mask_scaled:
            raise NotImplementedError("an int1 causal table on Trainium")
        if (query_kmajor or transpose_output) and self.sq_orig % self.tq:
            raise ValueError(
                f"a {self.tq}-row query tile straddles a head of "
                f"{self.sq_orig} rows under a K-major query or output"
            )
        self.query_kmajor = query_kmajor
        self.transpose_output = transpose_output
        # The K-major query and output fold the head group's rows, a
        # one-token query its row-major bytes (module docstring).
        folded = (*self.grid[:-2], self.g_head * self.d, self.sq_orig)
        self.fold_query = self.g_head > 1 or not query_kmajor
        if query_kmajor:
            self.q_fold_shape = folded
        self.q_tile = (self.d, self.tq)
        if transpose_output:
            self.out_shape = folded
            self.out_tile = (self.d, self.tq)
        self.rows = min(self.tq, PARTITIONS)
        self.blocks = math.ceil(self.tq / PARTITIONS)
        self.shapes = {
            role: shape for role, shape, *_ in self.placement.buffers
        }
        width = self.shapes["scores"][-1]
        self.banks = [
            (j, min(width, self.tkv - j)) for j in range(0, self.tkv, width)
        ]
        # The identity a mask is added through holds 1 / scale at the
        # accumulation dtype, the mask's (``_matmul_qk``).
        self.inverse_scale = torch.tensor(
            1 / self.scale, dtype=self.acc_dtype
        ).item()
        depth = self.shapes["transposes"][0]
        self.groups = [
            (lo, min(self.tkv, lo + depth)) for lo in range(0, self.tkv, depth)
        ]

    @staticmethod
    def logical_operands(query, key):
        """The query and key as ``aten.scaled_dot_product_attention``
        reads them, from their K-major storage."""
        return query.mT, key.mT

    @staticmethod
    def fuse_ops(gm):
        """Fuse the kernel's ops into the instructions they are
        (``INSTRUCTION_PATTERNS``), and every other chain ending in a store
        into one op."""
        fuse_patterns(gm, INSTRUCTION_PATTERNS)
        outline_dps_ops(gm)

    def _column(self, buf, block):
        """Block ``block``'s column of the per-row values ``buf``."""
        return _block(buf, (0, block), (self.rows, 1))

    def _rows_of(self, tile, block, cols):
        """Block ``block``'s rows of ``tile``, its first ``cols`` columns."""
        return _block(tile, (block * PARTITIONS, 0), (self.rows, cols))

    def _alloc_state(self, b):
        """Allocate into ``b`` the buffers the tile search placed, each as
        the attribute its role names (``TrainiumAttention.buffers``) -- P is
        the scores themselves unless it has a buffer of its own -- the
        regions the translator computes [E] and [F] in, by name, and the two
        compute semaphores."""
        for role, shape, dtype, _, slots, space in self.placement.buffers:
            setattr(b, role, alloc_placed(shape, dtype, slots, space))
        if "p_buf" not in self.shapes:
            b.p_buf = b.s_buf
        b.identity_made = False
        b.scratch = {
            name: alloc_placed(*placed)
            for name, *placed in self.placement.scratch
        }
        b.sem_scores = voyager.zeros([1], torch.int64)
        b.sem_pv = voyager.zeros([1], torch.int64)

    def _query_address(self, coords, kmajor):
        """The block address of the query tile at ``coords`` in the folded
        query, or in the output, which is cut like it: stored ``[*kv_batch,
        g_head·d, sq]`` when ``kmajor``, the tile's head in its group the
        block row and the tile within the head the block column (module
        docstring)."""
        if not kmajor:
            return self._block_address(coords, self.gq, self.grid[self.gq])
        dims, idx = self._block_address(coords, self.gq, 1)
        q, per_head = coords[self.gq], self.sq_orig // self.tq
        if self.g_head > 1:
            dims.append(self.nb)
            idx.append(q // per_head)
        if per_head > 1:
            dims.append(self.nb + 1)
            idx.append(q % per_head if self.g_head > 1 else q)
        return dims, idx

    def _load_query(self, src, slots, slot, sem, coords, tile, waits=0):
        """DMA the query ``src``'s ``tile`` at ``coords`` into
        ``slots[slot]``, as stored.  As in ``_FA3Pipeline._load_sweep``,
        its semaphore posts once per live step of the sweep, and once more
        for each of the ``waits`` the kernel makes on it itself."""
        dims, idx = self._query_address(coords, self.query_kmajor)
        voyager.async_copy(
            src,
            get_slot(slots, slot),
            idx,
            (1,) * self.nb + tuple(tile),
            get_slot(sem, slot),
            dims,
            post_count=self._kv_last(coords[self.gq]) + (1 + waits),
        )

    def _load_key(self, b, slot, coords):
        """DMA the key block at ``coords`` into Kᵀ slot ``slot`` as stored,
        ``[*batch, d, seq]``: its sequence dim follows the head's.  ``b``
        holds the buffers (``_alloc_state``)."""
        dims, idx = self._block_address(coords, self.gkv, self.grid[self.gkv])
        if self.grid[self.gkv] > 1:
            dims[-1] += 1
        voyager.async_copy(
            b.k,
            get_slot(b.k_slots, slot),
            idx,
            (1,) * self.nb + (self.d, self.tkv),
            get_slot(b.k_sem, slot),
            dims,
        )

    def _store_out(self, tile, out, sem, coords):
        """DMA the finished output ``tile`` of the sweep at ``coords`` into
        ``out``, cut like the query when stored transposed
        (``_query_address``)."""
        dims, idx = self._query_address(coords, self.transpose_output)
        voyager.async_copy(
            tile,
            out,
            idx,
            (1,) * self.nb + tuple(tile.shape[-2:]),
            get_slot(sem, 0),
            dims,
        )

    def _load_q(self, b, slot, coords):
        """DMA the query tile of the sweep at ``coords`` into slot
        ``slot``: a K-major query's block as stored, a one-token query's
        into ``q_rows`` as stored, then, once landed, transposed on TensorE
        into PSUM and copied out into Qᵀ."""
        tq, d = self.tq, self.d
        if self.query_kmajor:
            self._load_query(b.q, b.q_slots, slot, b.q_sem, coords, (d, tq))
            return
        self._load_query(b.q, b.q_rows, slot, b.q_sem, coords, (tq, d), 1)
        voyager.async_wait(get_slot(b.q_sem, slot))
        stored = _block(get_slot(b.q_rows, slot), (0, 0), (tq, d))
        transposed = get_slot(b.query_t, slot)
        voyager.insert(torch.transpose(stored, 0, 1), transposed)
        q_t = _block(get_slot(b.q_slots, slot), (0, 0), (d, tq))
        voyager.insert(transposed.clone(), q_t)

    def _matmul_qk(self, b, operands, slot, deps, masked=True):
        """[A]: per block and bank, QKᵀ into the scores' bank of step
        parity ``slot``, then, when the step adds a mask, the mask added
        onto it there by a matmul through an identity of ``1 / scale``
        (``inverse_scale``) -- the scores are scaled when evicted,
        so the mask is added about once, within its rounding, and the fill
        dropping a key overflows float32 to -inf there.  The first step
        adding a mask makes the identity.

        Args:
            b: The buffers (``_alloc_state``).
            operands: The Qᵀ and Kᵀ tiles, then the mask tile if any.
            slot: The step's parity.
            deps: The load semaphores the commit waits on; it posts
                ``sem_scores``.
            masked: The step adds the mask (``_FA3Pipeline._crosses``).
        """
        adds_mask = masked and self.has_mask
        make_identity = adds_mask and not b.identity_made
        if make_identity:
            b.identity_made = True
        identity = [b.identity] if adds_mask else []
        rows, d = self.rows, self.d

        def body(q, k, *args):
            *tiles, scores, slot = args
            mask, ident = tiles if adds_mask else (None, None)
            for block in range(self.blocks):
                q_block = _block(q, (0, block * PARTITIONS), (d, rows))
                for i, (j, w) in enumerate(self.banks):
                    bank = self._bank(scores, slot, block, i)
                    bank = _block(bank, (0, 0), (rows, w))
                    k_cols = _block(k, (0, j), (d, w))
                    voyager.insert(KMAJOR_MATMUL(q_block, k_cols), bank)
                    if not adds_mask:
                        continue
                    if make_identity and block == i == 0:
                        eye = torch.eye(PARTITIONS, dtype=ident.dtype)
                        voyager.insert(eye * self.inverse_scale, ident)
                    added = KMAJOR_MATMUL(
                        _block(ident, (0, 0), (rows, rows)),
                        _block(mask, (block * PARTITIONS, j), (rows, w)),
                    )
                    voyager.insert(bank + added, bank)

        commit(
            body,
            [*operands, *identity, b.scores, slot],
            dependencies=deps,
            post=b.sem_scores,
        )

    def _softmax_step(self, b, slot, sweep_slot, probs):
        """[D] on the scores of step parity ``slot`` (module docstring),
        into the running sum of sweep parity ``sweep_slot``.  ``b`` holds
        the buffers (``_alloc_state``); ``probs`` is the contract's, ``None``: P
        is not quantized."""
        voyager.async_wait(b.sem_scores)
        scores, probs_tile = get_slot(b.s_buf, slot), get_slot(b.p_buf, slot)
        bias = get_slot(b.bias, slot)
        for block in range(self.blocks):
            row_max = self._column(b.row_tmp, block)
            if len(self.banks) > 1:
                bank_max = get_slot(b.bank_max, slot)
            for i, (j, w) in enumerate(self.banks):
                bank = self._bank(b.scores, slot, block, i)
                scaled = (_block(bank, (0, 0), (self.rows, w)) * self.scale).to(
                    scores.dtype
                )
                largest = torch.amax(scaled, dim=-1, keepdim=True)
                voyager.insert(
                    scaled,
                    _block(scores, (block * PARTITIONS, j), (self.rows, w)),
                )
                if len(self.banks) > 1:
                    voyager.insert(
                        largest, _block(bank_max, (0, i), (self.rows, 1))
                    )
                else:
                    voyager.insert(largest, row_max)
            if len(self.banks) > 1:
                voyager.insert(
                    torch.amax(bank_max, dim=-1, keepdim=True), row_max
                )
            column = self._column(bias, block)
            voyager.insert(
                torch.maximum(self._column(b.m, block), row_max) * -1.0,
                column,
            )
            exps = torch.exp(self._rows_of(scores, block, self.tkv) + column)
            sums = torch.sum(exps, dim=-1, keepdim=True)
            voyager.insert(exps, self._rows_of(probs_tile, block, self.tkv))
            voyager.insert(sums, row_max)
        # Over every block's columns at once: alpha = exp(m - max(m, row
        # max)), m = max(m, row max) and l = alpha·l + row sum.
        delta, decayed = get_slot(b.delta, slot), get_slot(b.decayed, slot)
        total = get_slot(b.l, sweep_slot)
        voyager.insert(b.m + bias, delta)
        voyager.insert(torch.exp(delta), b.alpha)
        voyager.insert(bias * -1.0, b.m)
        voyager.insert(b.alpha * total, decayed)
        voyager.insert(decayed + b.row_tmp, total)

    def _matmul_pv(self, b, operands, deps, slot):
        """[B]: per block, P transposed on TensorE into PSUM a group of keys at
        a time, each group copied out into Pᵀ, then P·V into the block's PSUM
        slot of step parity ``slot``.  ``b`` holds the buffers
        (``_alloc_state``), ``operands`` are the P and V tiles and ``deps`` the
        V load semaphores the commit waits on; it posts ``sem_pv``."""
        rows, d, tkv = self.rows, self.d, self.tkv
        groups = len(self.groups)

        def body(p, v, probs_t, transposes, context, slot):
            values = _block(v, (0, 0), (tkv, d))
            for block in range(self.blocks):
                # Consecutive blocks, of this step and the next, rotate
                # through Pᵀ's slots and the transposes' tiles.
                order = slot * self.blocks + block
                kmajor = get_slot(probs_t, order % 2)
                probs = self._rows_of(p, block, tkv)
                for g, (lo, hi) in enumerate(self.groups):
                    tile = get_slot(transposes, (order * groups + g) % 2)
                    tile = _block(tile, (0, 0), (hi - lo, rows))
                    keys = _block(probs, (0, lo), (rows, hi - lo))
                    voyager.insert(torch.transpose(keys, 0, 1), tile)
                    voyager.insert(
                        tile.clone(), _block(kmajor, (lo, 0), (hi - lo, rows))
                    )
                result = self._context(context, slot, block)
                voyager.insert(KMAJOR_MATMUL(kmajor, values), result)

        commit(
            body,
            [*operands, b.probs_t, b.transposes, b.context, slot],
            dependencies=deps,
            post=b.sem_pv,
        )

    def _bank(self, scores, slot, block, i):
        """Block ``block``'s bank ``i`` of the PSUM ``scores``, of step
        parity ``slot``."""
        index = (slot * self.blocks + block) * len(self.banks) + i
        return get_slot(scores, index)

    def _context(self, context, slot, block):
        """Block ``block``'s P·V in the PSUM ``context``, of step parity
        ``slot``."""
        return get_slot(context, slot * self.blocks + block)

    def _land_pv(self, b, slot):
        """[E]'s first part: wait for a Q tile's last P·V and add each block's
        into ``o`` from PSUM.  ``b`` holds the buffers (``_alloc_state``);
        ``slot`` is the parity of the step whose P it multiplied."""
        voyager.async_wait(b.sem_pv)
        for block in range(self.blocks):
            o = self._rows_of(b.o, block, self.d)
            voyager.insert(o + self._context(b.context, slot, block), o)

    def _accumulate_pv(self, b, slot):
        """[F]: per block, ``o = alpha·(o + pv)``, P·V read from PSUM,
        beside its ``rescale`` scratch.  ``b`` holds the buffers
        (``_alloc_state``); ``slot`` is the parity of the step whose P it
        multiplied."""
        voyager.async_wait(b.sem_pv)
        scratch = group_regions(b.scratch, RESCALE_GROUP)
        for block in range(self.blocks):
            o = self._rows_of(b.o, block, self.d)
            alpha = self._column(b.alpha, block)
            context = self._context(b.context, slot, block)
            rescaled = with_scratch(
                lambda o, alpha, pv: alpha * (o + pv),
                [o, alpha, context],
                scratch,
            )
            voyager.insert(rescaled, o)

    def _finalize(self, b, l):
        """Per block, ``o / l`` into the output tile, stored at its dtype,
        beside its ``finalize`` scratch; under ``transpose_output`` into
        ``out_rows`` first, then transposed on TensorE into PSUM and copied out
        into the output slot.  ``b`` holds the buffers (``_alloc_state``) and
        ``l`` is the tile's running sum."""
        out = get_slot(b.out_slots, 0)
        rows = b.out_rows if self.transpose_output else out
        scratch = group_regions(b.scratch, FINALIZE_GROUP)
        for block in range(self.blocks):
            o = self._rows_of(b.o, block, self.d)
            finished = with_scratch(
                lambda o, l: o / l, [o, self._column(l, block)], scratch
            )
            voyager.insert(finished, self._rows_of(rows, block, self.d))
        if not self.transpose_output:
            return
        tq, d = self.tq, self.d
        transposed = b.output_t
        voyager.insert(
            torch.transpose(_block(rows, (0, 0), (tq, d)), 0, 1), transposed
        )
        voyager.insert(transposed.clone(), _block(out, (0, 0), (d, tq)))


def build_attention(
    node: torch.fx.Node, ctx: BuildContext
) -> Optional[torch.fx.GraphModule]:
    """Trainium's builder of a K-major flash attention: the shared FA3
    builder run with ``TrainiumFlashAttention``."""
    return build_attention_fa3(
        node,
        ctx,
        pipeline=TrainiumFlashAttention,
        query_kmajor=query_kmajor(node),
        transpose_output=writes_transposed(node),
    )
