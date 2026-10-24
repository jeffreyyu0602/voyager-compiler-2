"""Trainium's GEMM: the fit and cost of a GEMM (``TrainiumGemm``) and of a
matrix-vector one (``TrainiumGemv``), and the builder writing a dense GEMM's
steps as TensorE runs them (``build_gemm``).

A GEMM step ``[rows, depth] x [depth, cols]`` transposes its activation tile
on TensorE one 128 x 128 block at a time and copies the blocks out of PSUM
half a bank at a time (``half_bank_blocks``), runs one ``nc_matmul`` per row
chunk, contraction chunk and moving-operand slice into PSUM, and evicts each
finished slice to SBUF (``gemm_step_busy``); a reduction split over steps
keeps its running sum in PSUM across them, so it is split only when its
output tile fits there beside the kernel's other PSUM boxes
(``gemm_buffers``, ``fits_psum``).  An activation stored K-major
(``activation_k_major``) needs no transposes, and a K-major GEMM writing its
output transposed (layout ``"cx,ck->kx"``) holds ``cols`` on the PSUM
partitions: the weight is stationary and the activation moves.  An MX GEMM
step runs one ``nc_matmul_mx`` per 512-K chunk with no transposes.

A ``linear`` or ``matmul`` (an MX one too, and a decode step's GEMV) is
planned the shared way -- ``_gemm_plan`` gives its grid, its operands' tiles
and its fused tail, and the synchronous ``PipelinedKernel`` moves the tiles
-- but each step's work is written out as the instructions run it, so the
translator only translates.  The output tile, as PSUM holds it, is cut into
slices of at most 128 rows (the partitions) and 512 columns (a bank of
float32), and each slice is one GEMM writing PSUM, a ``Local`` box, its
product summed in float32 (``out_dtype``).  Every buffer the kernel uses,
in SBUF and PSUM, and the tail's scratch are the ones the tile search
placed (``TrainiumGemm.fit``, ``gemm_buffers``), which the builder
allocates as placed:

- With the reduction in one step, the slices' GEMMs write the
  ``ACCUMULATORS`` rotating PSUM tiles in turn, and each slice is evicted at
  once: the fused tail run on it (``SliceTail``), or a copy, writes the
  output tile from PSUM.
- With the reduction split over steps, the output tile's sum stays in PSUM,
  a bank per slice: the first step's GEMMs write it, each later one's add to
  it (the ``matmul_accumulate`` pattern), and the last step evicts each
  slice at once, or, when the output tile is the sum itself (the tail
  computes and moves nothing), every slice after the step's GEMMs.
- A row-major activation is transposed on TensorE 128 rows at a time: its
  blocks of 128 columns, as many as a transposition tile holds, side by
  side into that PSUM tile, which is copied out whole into an SBUF tile the
  slices' GEMMs read contraction-major.  A prologue computing the
  activation runs on the 128 rows into an SBUF tile first.
- A tail ending in a row op (a norm, a softmax) evicts the slices through
  the ops before it, then runs the row op on the output tile's 128 rows in
  place; one ending in a transpose of the last two dims evicts them into a
  staging tile, then transposes the finished tile on TensorE through PSUM
  into the output (``transpose_through``).
- An operand stored transposed is loaded as stored into a staging tile and
  transposed on TensorE through PSUM into its slot where the pipeline loads
  it (``load_transposed``, its spec's ``load``).

The instruction patterns then fuse the ops each instruction computes
(``fuse_patterns``), and every other chain of compute ending in a store
becomes one op (``outline_dps_ops``).

The kernel is traced, and dynamo, which traces a reduction's conditions,
reads no FX node and runs no numpy: what a slice reads and writes is worked
out when the steps are built.
"""

import functools
import itertools
import math
import operator
from collections import Counter
from typing import Optional
from dataclasses import dataclass, replace

import numpy as np
import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    activation_prologue,
    activation_transforms,
    attention_shapes,
    bound_operands,
    gemm_layout,
    get_anchor_node,
    get_arg_value,
    is_fully_connected,
    is_gemm_op,
    is_nop,
    reads_kmajor,
    swaps_last_two_dims,
    tail_targets,
    weight_transforms,
    writes_transposed,
)
from voyager_compiler.codegen.transform import fuse_patterns
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _is_compute,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    commit,
    with_scratch,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    _apply_relayout,
    _gemm_plan,
    _stamp_anchor_meta,
    build_pipelined_buffers,
    get_slot,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    BuildContext,
    _ScratchSpec,
    outline_dps_ops,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    activation_k_major,
    fused_operand_specs,
    gemm_batch,
    gemv_operand_shapes,
    get_dtype_width,
    layer_key,
    node_dtype_bits,
    outlier_stats,
    output_dtype,
    tail_keeps_shape,
    weight_batch,
    weight_k_major,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    MatrixSpace,
    OpModel,
    RuntimeModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    INPUT_DIMS,
    OUTPUT_DIMS,
    WEIGHT_DIMS,
    batch_loads,
    gemv_transfers,
    l3_steps,
    tile_extent,
    total_extent,
)

from .patterns import INSTRUCTION_PATTERNS
from .utils import (
    ACCUMULATORS,
    F32,
    LOCAL,
    MX_PACK,
    PARTITIONS,
    QUANTIZE_MX,
    ROW_OPS,
    SDPA_KMAJOR,
    SRAM,
    TRANSPOSE_TILES,
    TrainiumGrid,
    TrainiumPlacement,
    chain_tiles,
    contraction_runs,
    extents,
    fits_psum,
    footprints,
    group_ops,
    group_regions,
    group_tiles,
    half_bank_blocks,
    moved_bytes,
    multiplies_mx,
    output_buffers,
    pair_distance,
    pass_engines,
    psum_bytes,
    quantizable,
    row_op_tiles,
    rows_cols,
    sbuf_bytes,
    sbuf_placed_bytes,
    scratch_regions,
    whole_chunks,
    widest_bits,
)

le = interstellar.le
voyager = torch.ops.voyager

# The SBUF tiles a GEMM step computes or transposes its row-major activation
# into rotate through this many slots (``gemm_buffers``).
ACTIVATION_SLOTS = 2
# The roles of the buffers ``gemm_buffers`` places, by which Trainium's GEMM
# builder reads them: the slices' PSUM tiles, a row-major activation's
# transposition tiles in PSUM and the SBUF tile they are copied into, a
# prologue's SBUF tile, and the staging tile of a tail transposing the
# finished tile and its transposition tiles in PSUM.
RESULTS = "results"
TRANSPOSES = "transposes"
TRANSPOSED = "transposed"
PROLOGUE = "prologue"
STAGED = "staged"
STAGED_TRANSPOSES = "staged_transposes"
# A transposing load's PSUM tiles, the activation's and the weight's, and
# the staging slots of each by its role.
ACTIVATION_LOAD = "activation_load"
WEIGHT_LOAD = "weight_load"
LOAD_STAGING = {
    ACTIVATION_LOAD: "activation_staging",
    WEIGHT_LOAD: "weight_staging",
}
# The op groups the translator computes a GEMM's kernel in, each with its
# scratch regions (``gemm_scratch``): the eviction's chain on each slice,
# the tail's row op on the output tile's rows, the prologue, and a slice's
# product on the step adding the bias.
EVICT_GROUP, ROW_GROUP, PROLOGUE_GROUP = "evict", "row", "prologue"
PRODUCT_GROUP = "product"


def transposed_loads(node):
    """``(role, operand)`` of each operand of GEMM ``node``, bare or fused,
    that lands transposed from its stored layout (``activation_transforms``,
    ``weight_transforms``), its role ``ACTIVATION_LOAD`` or
    ``WEIGHT_LOAD``: the node a fused GEMM's placeholder stands for."""
    anchor = get_anchor_node(node)
    bound = bound_operands(node, node.meta.get("submodule"))
    act, act_loaded = activation_transforms(anchor.args[0])
    weight, weight_loaded = weight_transforms(anchor.args[1])[:2]
    return [
        (role, bound.get(n, n))
        for role, n, loaded in (
            (ACTIVATION_LOAD, act, act_loaded),
            (WEIGHT_LOAD, weight, weight_loaded),
        )
        if loaded
    ]


def first_load_exposed(node):
    """Whether GEMM ``node``'s first loads wait in the open rather than run
    under the kernel before it (``TrainiumModel.sweep_ns``).  The step
    compile starts a kernel's loads under a matrix-matrix GEMM or a flash
    attention of several queries before it, which leave the DMA idle, the
    pieces of an activation as that kernel writes them.  After a vector op
    -- a norm, whose output a GEMM reads whole -- or a matrix-vector GEMM or
    attention, which keep the DMA busy, and first in the graph they find
    nothing to overlap.

    Read off the graph the tile search sees and kept on ``node`` (its
    ``"first_load_exposed"`` meta): the bufferizer asks again with the
    kernels before it already rewritten into loops.
    """
    if "first_load_exposed" not in node.meta:
        before = node.prev
        while before.op != "root" and not _is_compute(before):
            before = before.prev
        exposed = before.op == "root"
        if not exposed:
            anchor = get_anchor_node(before) or before
            if anchor.target is SDPA_KMAJOR:
                exposed = attention_shapes(anchor)[2][-2] == 1
            else:
                exposed = not is_gemm_op(anchor) or is_fully_connected(anchor)
        node.meta["first_load_exposed"] = exposed
    return node.meta["first_load_exposed"]


def lands_on_tensor_engine(elements, rows):
    """Whether a tile loaded transposed, ``elements`` elements of ``rows``
    stored rows each, is transposed by TensorE after a load as stored:
    TensorE reads a block from partition 0 only, so the rows of a tile of
    several elements must fill whole 128-row runs; else the DMA transposes
    it, element by element."""
    return elements == 1 or rows % PARTITIONS == 0


def gemm_operands(node):
    """What Trainium's GEMM builder transposes or computes beside GEMM
    ``node``'s operands, bare or fused: ``(activation, prologue, staged,
    loads)``.

    ``activation`` is ``(dtype, bits)`` of a row-major activation, which it
    transposes on TensorE, ``None`` for an MX or K-major one
    (``reads_kmajor``), whose contraction is on the partitions already;
    ``prologue`` says a prologue computes the activation; ``staged`` is
    ``(dtype, bits)`` of the result when the tail ends in a transpose of
    its last two dims, else ``None``; ``loads`` is ``(role, dtype, bits)``
    of each operand that lands transposed from its stored layout
    (``transposed_loads``).  A width is the stored one, read off the node a
    fused GEMM's placeholder stands for.
    """
    anchor = get_anchor_node(node)
    submod = node.meta.get("submodule")
    activation = None
    if not (multiplies_mx(node) or reads_kmajor(anchor)):
        bound = bound_operands(node, submod)
        act = activation_transforms(anchor.args[0])[0]
        bits = node_dtype_bits(bound.get(act, act))
        activation = (anchor.args[0].value.dtype, bits)
    loads = tuple(
        (role, n.value.dtype, node_dtype_bits(n))
        for role, n in transposed_loads(node)
    )
    staged = None
    if submod is not None:
        ops = [n for n in submod.graph.nodes if n.op == "call_function"]
        tail = ops[ops.index(anchor) + 1 :]
        if tail and swaps_last_two_dims(tail[-1]):
            dtype = anchor.value.dtype
            staged = (dtype, get_dtype_width(dtype))
    prologue = bool(activation_prologue(node))
    return activation, prologue, staged, loads


def multiplied_bits(node):
    """``(activation, weight)`` element widths of the tiles TensorE
    multiplies in GEMM ``node``, bare or fused: each operand as stored,
    beneath a transpose or repeat its load folds in
    (``activation_transforms``, ``weight_transforms``), read off the node a
    fused GEMM's placeholder stands for; a weight dequantized on chip at its
    dequantized width."""
    anchor = get_anchor_node(node)
    bound = bound_operands(node, node.meta.get("submodule"))
    act = activation_transforms(anchor.args[0])[0]
    weight, _, _, dequant = weight_transforms(anchor.args[1])
    if dequant is not None:
        weight = anchor.args[1]
    return tuple(widest_bits(bound.get(n, n)) for n in (act, weight))


def pairing(bits, act_step, weight_step):
    """Chunks apart a GEMM step's ``double_row`` matmuls pair two 128-deep
    chunks of its contraction (``pair_distance``, ``contraction_runs``),
    its contraction-major tiles holding consecutive chunks ``act_step`` and
    ``weight_step`` elements apart; ``None`` unless its operands are FP8
    (``bits``, the step's widths)."""
    if max(bits[:2]) > 8:
        return None
    return pair_distance((act_step, weight_step))


def gemm_buffers(config, operands, scratch, tile, staging):
    """The buffers Trainium's GEMM builder allocates for a step
    (``TrainiumPlacement.buffers``), or ``None`` when its PSUM does not fit
    (``fits_psum``) beside the kernel's ``scratch``.

    In PSUM:

    - ``results``: the matmuls' results -- a split reduction's running sum
      on a bank per 128 rows and 512 columns of each batch element, else
      ``ACCUMULATORS`` rotating tiles as wide as the columns, up to a bank
      (a whole bank for a K-major GEMM, whose slices may run over the rows);
    - ``transposes``: a row-major activation's TensorE transposes, two
      tiles of half a bank (``half_bank_blocks``), its blocks of 128
      contraction columns stacked;
    - ``staged_transposes``: the finished tile's, for a tail ending in a
      transpose, two tiles of half a bank, its blocks of 128 rows side by
      side (``transpose_through``);
    - each transposing load's (``activation_load``, ``weight_load``):
      ``TRANSPOSE_TILES`` tiles of as many 128-row blocks side by side
      (``transpose_through``) as an even share of what the others leave
      holds, up to a bank, and the step does not fit when a share holds no
      block.

    In SBUF: a row-major activation's ``transposed`` tile ``[depth, 128]``,
    after a ``prologue`` tile ``[128, depth]`` it is computed in, each
    ``ACTIVATION_SLOTS``; the ``staged`` output tile a tail transposes, and
    each transposing load's staging slots (``activation_staging``,
    ``weight_staging``).

    Args:
        config: The ``TrainiumConfig``.
        operands: The GEMM's ``gemm_operands``.
        scratch: The kernel's scratch regions (``gemm_scratch``).
        tile: ``(batch, rows, cols, depth, split)``: the step's output tile
            as PSUM holds it, ``batch`` elements of ``rows x cols``, its
            contraction and whether the reduction is split over steps.
        staging: The stored tile and the slots of each transposing load
            TensorE transposes (``lands_on_tensor_engine``), by its role.
    """
    p, free = config.partitions, config.moving_free_max
    activation, prologue, staged, loads = operands
    batch, rows, cols, depth, split = tile
    if split:
        banks = batch * math.ceil(rows / p) * math.ceil(cols / free)
        buffers = [(RESULTS, (p, free), F32, 32, banks, LOCAL)]
    else:
        width = free if activation is None else min(cols, free)
        buffers = [(RESULTS, (p, width), F32, 32, ACCUMULATORS, LOCAL)]
    chunk = min(p, rows)
    if activation is not None:
        dtype, bits = activation
        blocks = half_bank_blocks(config, bits)
        buffers += [
            (TRANSPOSES, (blocks * p, p), dtype, bits, 2, LOCAL),
            (TRANSPOSED, (depth, chunk), dtype, bits, ACTIVATION_SLOTS, SRAM),
        ]
        if prologue:
            shape = (chunk, depth)
            buffers.append(
                (PROLOGUE, shape, dtype, bits, ACTIVATION_SLOTS, SRAM)
            )
    if staged is not None:
        dtype, bits = staged
        blocks = half_bank_blocks(config, bits)
        buffers += [
            (STAGED_TRANSPOSES, (p, blocks * p), dtype, bits, 2, LOCAL),
            (STAGED, (batch * rows, cols), dtype, bits, 1, SRAM),
        ]
    used = [
        (shape, bits, slots)
        for shape, bits, slots, space in footprints(buffers, scratch)
        if space == LOCAL
    ]
    share = (config.psum_bytes - psum_bytes(config, used)) // max(len(loads), 1)
    for role, dtype, bits in loads:
        if role not in staging:
            continue
        block = p * max(bits, 16) // 8
        blocks = min(config.psum_bank_bytes, share // TRANSPOSE_TILES) // block
        if blocks < 1:
            return None
        stored, slots = staging[role]
        buffers += [
            (role, (p, blocks * p), dtype, bits, TRANSPOSE_TILES, LOCAL),
            (LOAD_STAGING[role], stored, dtype, bits, slots, SRAM),
        ]
    buffers = tuple(buffers)
    return buffers if fits_psum(config, buffers, scratch) else None


def gemm_groups(node):
    """The groups of ops the translator runs in GEMM ``node``'s kernel,
    bare or fused (``group_ops``): ``(eviction, prologue)`` -- the tail its
    builder runs on each slice, then its row op on the output tile's rows,
    a last op transposing the result's last two dims left to TensorE; and
    the prologue computing the activation."""
    tail = tail_nodes(node)
    if (
        tail
        and not any(op.target in ROW_OPS for op in tail)
        and swaps_last_two_dims(tail[-1])
    ):
        tail = tail[:-1]
    return group_ops(tail), group_ops(list(activation_prologue(node)))


def tail_nodes(node):
    """The ops GEMM ``node``'s fused tail runs after its anchor, in order;
    none for a bare GEMM."""
    submod = node.meta.get("submodule")
    if submod is None:
        return []
    ops = [n for n in submod.graph.nodes if n.op == "call_function"]
    return ops[ops.index(get_anchor_node(node)) + 1 :]


def evicts_by_copy(tail):
    """Whether a slice leaves PSUM by a copy under the fused ``tail`` ops:
    when no op ahead of its row op, or of a last op transposing the last
    two dims, does arithmetic."""
    stop = next((i for i, n in enumerate(tail) if n.target in ROW_OPS), None)
    if stop is None and tail and swaps_last_two_dims(tail[-1]):
        stop = len(tail) - 1
    return all(is_nop(n) or n.target in RELAYOUTS for n in tail[:stop])


def gemm_scratch(config, groups, cols, depth, bias):
    """The scratch regions of a GEMM's kernel whose output tile PSUM holds
    ``cols`` wide, contracting ``depth`` (``scratch_regions``): the
    eviction's chain on each 512-column slice (``EVICT_GROUP``), its row op
    on the tile's rows (``ROW_GROUP``), the prologue's group over the
    activation's rows (``PROLOGUE_GROUP``), and with a bias, of dtype
    ``bias`` (else ``None``), a ones row of a slice's columns, which a
    slice's first GEMM adds the bias by: a matmul of it by the bias,
    contracting one row (``PRODUCT_GROUP``).  ``groups`` is the GEMM's
    ``gemm_groups``."""
    (chain, row_op, _), prologue = groups
    evict = set()
    for width in extents(cols, config.moving_free_max):
        evict |= chain_tiles(chain, width)
    row = row_op_tiles(row_op, cols, ()) if row_op is not None else ()
    ones = (
        {("one", config.moving_free_max, bias, 1, SRAM)}
        if bias is not None
        else ()
    )
    return (
        scratch_regions(EVICT_GROUP, evict)
        + scratch_regions(ROW_GROUP, row)
        + scratch_regions(PROLOGUE_GROUP, group_tiles(prologue, depth))
        + scratch_regions(PRODUCT_GROUP, ones)
    )


def gemm_step_busy(
    model,
    rows,
    depth,
    cols,
    bits,
    weight_k_major,
    tail_engines=(),
    tail_quantizes_mx=False,
    num_k=1,
    mx=False,
    prologue_engines=(),
    act_k_major=False,
    *,
    tail_copies,
    store_issue=0.0,
    distance,
):
    """Each engine's busy time on one GEMM step, ``[rows, depth] x [depth,
    cols]``.

    TensorE transposes each 128 x 128 block of an activation stored
    ``[rows, depth]`` -- and of a weight stored ``[cols, depth]`` -- and
    multiplies the activation into PSUM one moving slice at a time; VectorE
    copies the transposed blocks out of PSUM, an activation's half a bank
    of them at a time (``half_bank_blocks``, a short last block alone) and
    a weight's one at a time, and evicts the finished slices.  Given a
    ``distance``, the matmuls contract two 128-deep chunks that many apart
    in one ``double_row`` matmul at ``fp8_matmul_speedup`` times the 16-bit
    rate, a chunk left unpaired at the 16-bit rate (``contraction_runs``).
    FP32 operands multiply at ``fp32_matmul_rate`` of the 16-bit rate and
    transpose at ``fp32_transpose_rate``.  MX operands are stored
    contraction-major, so they need no transposes, and each
    ``nc_matmul_mx`` contracts 512.  The tail's passes run over an output
    tile once its reduction ends, so each of its ``num_k`` steps carries a
    share of them; they read the result in PSUM or the float32 scratch
    tiles after it, so they run at 32-bit width.  The prologue's passes run
    over the activation tile on every step.  A reduction split over steps
    keeps its running sum in PSUM (``gemm_buffers``), so an output tile is
    read out of PSUM once: by the tail's first pass, or by a copy
    (``evicts_by_copy``).

    Args:
        model: The ``TrainiumModel``.
        rows: Rows of the activation and output tiles.
        depth: The contraction.
        cols: Columns of the weight and output tiles.
        bits: ``(activation, weight, output)`` element widths.
        weight_k_major: The weight tile lands ``[depth, cols]``, the
            contraction on the partitions.
        tail_engines: The engine of each pass the tail's arithmetic
            makes over an output tile, in order (``pass_engines``).
        tail_quantizes_mx: The tail runs a ``quantize_mx``.
        num_k: Steps of the reduction.
        mx: The operands are MX.
        prologue_engines: The engine of each pass the prologue's
            arithmetic makes over an activation tile, in order
            (``pass_engines``).
        act_k_major: The activation tile lands ``[depth, rows]``, the
            contraction on the partitions.
        tail_copies: Each finished slice is copied out of PSUM
            (``evicts_by_copy``).
        store_issue: ScalarE's time issuing the step's share of the output
            stores (``TrainiumModel.store_issue_ns``).
        distance: Chunks apart the step's matmuls pair them, ``None`` for
            none (``pairing``).
    """
    config = model.config
    act_bits, weight_bits, _ = bits
    overhead = config.tensor_overhead_cycles
    widths = extents(cols, config.moving_free_max)
    tensor_cycles = 0.0
    busy = Counter()
    doubles = 0
    if mx:
        rate = 1.0
        singles = math.ceil(depth / (config.partitions * MX_PACK))
        transposes = []
    else:
        width = max(act_bits, weight_bits)
        rate = 1.0 if width <= 16 else config.fp32_matmul_rate
        depth_chunks = math.ceil(depth / config.partitions)
        runs = contraction_runs(depth, distance)
        doubles = sum(len(chunks) == 2 for chunks in runs)
        singles = len(runs) - doubles
        transposes = []
        if not act_k_major:
            per_copy = half_bank_blocks(config, act_bits)
            transposes += [
                (r, act_bits, per_copy)
                for r in extents(rows, config.partitions)
            ]
        if not weight_k_major:
            transposes += [
                (c, weight_bits, 1) for c in extents(cols, config.partitions)
            ]
    whole, short = divmod(depth, config.partitions)
    for extent, element_bits, per_copy in transposes:
        if element_bits > 16:
            extent_cycles = extent / config.fp32_transpose_rate
        else:
            extent_cycles = extent
        tensor_cycles += depth_chunks * (extent_cycles + overhead)
        copies = extents(whole, per_copy) + [1] * (short > 0)
        busy["vector"] += sum(
            model.engine_ns("vector", n * extent, element_bits) for n in copies
        )
    evictions = 1 / num_k if tail_copies else 0
    for _ in extents(rows, config.pe_array_size[1]):
        tensor_cycles += singles * sum(w / rate + overhead for w in widths)
        tensor_cycles += doubles * sum(
            2 * w / config.fp8_matmul_speedup + overhead for w in widths
        )
        busy["vector"] += evictions * sum(
            model.engine_ns("vector", w, 32) for w in widths
        )
    busy["tensor"] = tensor_cycles / config.frequency
    tail = [(e, 1.0, 32) for e in tail_engines]
    tail_busy = model.passes_busy(tail, rows, cols)
    if tail_quantizes_mx:
        tail_busy.update(model.quantize_mx_busy(rows, cols))
    for engine, ns in tail_busy.items():
        busy[engine] += ns / num_k
    prologue = [(e, 1.0, act_bits) for e in prologue_engines]
    busy.update(model.passes_busy(prologue, rows, depth))
    busy["scalar"] += store_issue
    return busy


@dataclass(frozen=True)
class TrainiumGemm(OpModel):
    """A GEMM under ``TrainiumModel``: the SBUF fit of each tile interstellar
    tries and the runtime of each mapping it finds.

    Judged in a forked search worker, so it holds plain values only, read
    off the graph when it is prepared (``from_node``), and dispatches no
    torch op.

    Attributes:
        space: ``TrainiumModel.matrix_space``.
        key: What the search is cached under, the same facts on every
            backend: the layer (``layer_key``), its output dtype, its fused
            tail's operands and ops (by name), whether that tail takes passes
            of its own and the scratch regions they may take, its outlier
            CSRs, its prologue's ops (by name) and operands, how its
            activation and output are stored, what it keeps in PSUM and its
            bias's dtype.
        model: The ``TrainiumModel``, which prices each step.
        bits: ``(activation, weight, output)`` element widths.
        output_scale_bits: The output's block-scale width, 0 unless it is
            MX.
        bias_bits: The bias's element width, 0 without one.
        bias_dtype: The bias's dtype, ``None`` without one.
        block_size: The operands' MX block, 1 unless they are MX.
        batch: Batch elements the builder loops outside the mapping.
        weight_batch: Distinct weight tiles over that loop (a GQA repeat
            reads one tile for several elements).
        weight_k_major: The weight tile lands ``[depth, cols]``, the
            contraction on the partitions.
        act_k_major: The activation tile lands ``[depth, rows]``
            (``activation_k_major``).
        act_loaded: The activation lands transposed from its stored layout
            (``activation_transforms``), so its load moves the stored tile.
        weight_loaded: The weight lands transposed from its stored layout
            (``weight_transforms``), so its load moves the stored tile.
        out_transposed: The output tile is stored ``[cols, rows]``, its
            columns on the partitions (``writes_transposed``).
        operands: What its builder transposes or computes beside its
            operands (``gemm_operands``).
        groups: The groups of ops the translator runs in its kernel
            (``gemm_groups``).
        fused_specs: ``(dims, bits)`` per operand the fused tail adds
            (``fused_operand_specs``).
        tail_engines: The engine of each pass the tail's arithmetic makes
            over an output tile, in order (``pass_engines``).
        tail_quantizes_mx: The tail runs a ``quantize_mx``.
        tail_copies: Each finished slice is copied out of PSUM
            (``evicts_by_copy``).
        prologue_specs: ``(dims, bits)`` per operand the prologue computing
            the activation reads (``activation_prologue``), on the
            activation's loop dims; with any, they load in its place.
        prologue_engines: The engine of each pass the prologue's arithmetic
            makes over an activation tile, in order (``pass_engines``).
        whole_depth: The prologue runs a row op, so the contraction is never
            split.
        row_width: The output's last dim when the tail runs a row op, which
            an output tile then spans whole; 0 otherwise.
        first_load_exposed: Its first loads overlap no kernel before it
            (``first_load_exposed``).
    """

    space: MatrixSpace
    key: tuple
    model: RuntimeModel
    bits: tuple
    output_scale_bits: int
    bias_bits: int
    bias_dtype: Optional[torch.dtype]
    block_size: int
    batch: int
    weight_batch: int
    weight_k_major: bool
    act_k_major: bool
    act_loaded: bool
    weight_loaded: bool
    out_transposed: bool
    operands: tuple
    groups: tuple
    fused_specs: tuple
    tail_engines: tuple
    tail_quantizes_mx: bool
    tail_copies: bool
    prologue_specs: tuple
    prologue_engines: tuple
    whole_depth: bool
    row_width: int
    first_load_exposed: bool

    @classmethod
    def from_node(cls, node, model):
        """The ``TrainiumGemm`` of a GEMM ``node``, bare or fused, under
        ``model``."""
        anchor = get_anchor_node(node)
        out_dtype = output_dtype(node)
        mx_out = isinstance(out_dtype, (list, tuple))
        values_dtype = out_dtype[-1] if mx_out else out_dtype
        out_bits = (
            get_dtype_width(values_dtype)
            if values_dtype
            else node_dtype_bits(anchor)
        )
        fused_specs = tuple(fused_operand_specs(node, anchor))
        has_tail = node.meta.get("submodule") is not None
        tail_ops = tail_targets(node)
        tail_copies = evicts_by_copy(tail_nodes(node))
        prologue = activation_prologue(node)
        prologue_ops = tuple(n.target for n in prologue)
        operands = gemm_operands(node)
        groups = gemm_groups(node)
        exposed = first_load_exposed(node)
        bias = get_arg_value(anchor, 2, "bias", None)
        bias_dtype = None if bias is None else bias.value.dtype
        # A prologue operand tiles along the activation's loop dims it
        # spans, right-aligned on them, its broadcast dims dropped.
        act_dims = [le.ON] * (anchor.args[0].value.ndim - 2)
        act_dims += [le.IC, le.OX] if reads_kmajor(anchor) else [le.OX, le.IC]
        placeholders = dict.fromkeys(
            p
            for op in prologue
            for p in op.all_input_nodes
            if p.op == "placeholder" and p.value.numel() != 1
        )
        prologue_specs = tuple(
            (
                tuple(
                    d
                    for d, size in zip(act_dims[-len(p.shape) :], p.shape)
                    if size > 1
                ),
                node_dtype_bits(p),
            )
            for p in placeholders
        )
        key = layer_key(anchor) + (
            tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
            fused_specs,
            has_tail,
            tuple(str(target) for target in tail_ops),
            tail_keeps_shape(node),
            tail_copies,
            *outlier_stats(node),
            tuple(str(target) for target in prologue_ops),
            prologue_specs,
            activation_k_major(anchor),
            writes_transposed(anchor),
            operands,
            groups,
            exposed,
            bias_dtype,
        )
        return cls(
            space=model.matrix_space,
            key=key,
            model=model,
            bits=(*multiplied_bits(node), out_bits),
            output_scale_bits=get_dtype_width(out_dtype[-2]) if mx_out else 0,
            bias_bits=node_dtype_bits(bias, 0),
            bias_dtype=bias_dtype,
            block_size=anchor.kwargs.get("block_size") or 1,
            batch=gemm_batch(anchor),
            weight_batch=weight_batch(anchor),
            weight_k_major=weight_k_major(anchor),
            act_k_major=activation_k_major(anchor),
            act_loaded=activation_transforms(anchor.args[0])[1],
            weight_loaded=weight_transforms(anchor.args[1])[1],
            out_transposed=writes_transposed(anchor),
            operands=operands,
            groups=groups,
            fused_specs=fused_specs,
            tail_engines=pass_engines(tail_ops),
            tail_quantizes_mx=QUANTIZE_MX in tail_ops,
            tail_copies=tail_copies,
            prologue_specs=prologue_specs,
            prologue_engines=pass_engines(prologue_ops),
            whole_depth=any(t in ROW_OPS for t in prologue_ops),
            row_width=(
                anchor.value.shape[-1]
                if any(t in ROW_OPS for t in tail_ops)
                else 0
            ),
            first_load_exposed=exposed,
        )

    @staticmethod
    def _tile(extent):
        """``(rows, depth, cols)`` of the tile ``extent(loop)`` describes."""
        rows = extent(le.OX) * extent(le.OY) * extent(le.ON)
        return rows, extent(le.IC), extent(le.OC)

    @staticmethod
    def _prologue_tile(dims, rows, depth):
        """``(rows, cols)`` of a prologue operand's tile spanning ``dims``
        beside a ``rows x depth`` activation tile."""
        return (rows if le.OX in dims else 1, depth if le.IC in dims else 1)

    @staticmethod
    def _act_tile(rows, depth, on, k_major):
        """``(rows, cols)`` of an activation tile, rows on the partitions:
        ``[depth, rows]`` per batch element when it is ``k_major``, ``rows``
        counting the ``on`` batch elements."""
        if k_major:
            return on * depth, rows // on
        return rows, depth

    def _out_tile(self, rows, cols, on, dims=OUTPUT_DIMS):
        """``(rows, cols)`` an output-shaped tile spanning ``dims`` is stored
        in, rows on the partitions: ``[cols, rows]`` per batch element for a
        transposed output, ``rows`` counting the ``on`` batch elements."""
        tile_rows = rows if le.OX in dims else 1
        tile_cols = cols if le.OC in dims else 1
        if self.out_transposed:
            batch = on if le.ON in dims and le.OX in dims else 1
            return batch * tile_cols, tile_rows // batch
        return tile_rows, tile_cols

    def _step_ns(self, rows, depth, cols, on, **kwargs):
        """A ``[rows, depth] x [depth, cols]`` step's busiest engine
        (``gemm_step_busy``), ``rows`` counting the ``on`` batch elements.
        The matmuls read the activation ``[depth, rows]`` per batch element
        when it lands K-major -- its stored layout, flipped by a transposing
        load -- else TensorE's transposes of it, ``[depth, 128]`` of a batch
        element at a time (``gemm_buffers``).  A transposed output holds
        ``cols`` on the PSUM partitions, so the step runs ``[cols, depth] x
        [depth, rows]``, the weight and the activation trading places."""
        bits = self.bits
        k_major = self.act_k_major != self.act_loaded
        act_step = rows // on if k_major else min(PARTITIONS, rows // on)
        distance = pairing(bits, act_step, cols)
        act_k_major, weight_k_major = self.act_k_major, self.weight_k_major
        if self.out_transposed:
            rows, cols = cols, rows
            bits = (bits[1], bits[0], bits[2])
            act_k_major, weight_k_major = weight_k_major, act_k_major
        busy = gemm_step_busy(
            self.model,
            rows,
            depth,
            cols,
            bits,
            weight_k_major,
            act_k_major=act_k_major,
            distance=distance,
            **kwargs,
        )
        return max(busy.values())

    def fit(self, mapping):
        """The kernel's ``TrainiumPlacement`` -- the buffers its builder
        allocates (``gemm_buffers``) and the regions the translator computes
        its tail and prologue in (``gemm_scratch``) -- when the L2 tile fits
        SBUF, else ``None``.

        The tile's SBUF bytes are the activation's, the weight's with its
        bias and the output's with the fused tail's operands, each in its
        pipeline slots, and the placement's, and they must fit the usable
        scratchpad; a prologue's operands take the activation's place.  MX
        operands and an MX output take their codes and scales in the MX
        layout.  A tile the translator cannot lower fits at no size: one
        whose PSUM does not fit (``gemm_buffers``), one splitting the
        contraction under a prologue row op, one not in the MX layout's
        whole chunks (``whole_chunks``, ``quantizable``), or one splitting
        the output's rows under a tail row op.
        """
        model = self.model
        config = model.config
        p = config.partitions
        act_bits, weight_bits, out_bits = self.bits

        def slots(dims, distinct):
            tiles = distinct * math.prod(
                mapping.loop_blockings[d][3] for d in dims
            )
            return config.num_slots if tiles > 1 else 1

        rows, depth, cols = self._tile(
            functools.partial(total_extent, mapping, level=2)
        )
        mx_in = self.block_size > 1
        mx_out = self.output_scale_bits > 0
        split_k = mapping.loop_blockings[le.IC][3] > 1
        on = total_extent(mapping, le.ON, level=2)
        if (mx_in and not whole_chunks(depth, p)) or (
            mx_out and not quantizable(cols, p)
        ):
            return None
        psum_rows, psum_cols = self._out_tile(rows, cols, on)
        if (self.whole_depth and split_k) or (
            self.row_width and psum_cols != self.row_width
        ):
            return None
        weight_tile = (depth, cols) if self.weight_k_major else (cols, depth)
        # An operand stored transposed lands as stored in staging slots,
        # unless the DMA transposes it.
        staging = {}
        stored_rows = rows // on if self.act_k_major else depth
        if lands_on_tensor_engine(on, stored_rows):
            staging[ACTIVATION_LOAD] = (
                self._act_tile(rows, depth, on, self.act_k_major),
                slots(INPUT_DIMS, self.batch),
            )
        weight_elements = on if self.weight_batch > 1 else 1
        if lands_on_tensor_engine(weight_elements, cols):
            staging[WEIGHT_LOAD] = (
                weight_tile,
                slots(WEIGHT_DIMS, self.weight_batch),
            )
        scratch = gemm_scratch(
            config, self.groups, psum_cols, depth, self.bias_dtype
        )
        tile = (on, psum_rows // on, psum_cols, depth, split_k)
        buffers = gemm_buffers(config, self.operands, scratch, tile, staging)
        if buffers is None:
            return None
        if mx_in:
            act = model.mx_sbuf(depth, rows, False)
            act += model.mx_sbuf(depth, rows, True)
            weight = model.mx_sbuf(depth, cols, False)
            weight += model.mx_sbuf(depth, cols, True)
        else:
            act_tile = self._act_tile(rows, depth, on, self.act_k_major)
            act = sbuf_bytes(*act_tile, act_bits, p)
            weight = sbuf_bytes(*weight_tile, weight_bits, p)
        if mx_out:
            out = model.mx_sbuf(cols, rows, False)
            out += model.mx_sbuf(cols, rows, True)
        else:
            out = sbuf_bytes(psum_rows, psum_cols, out_bits, p)
        if self.prologue_specs:
            act = sum(
                slots(dims, self.batch)
                * sbuf_bytes(*self._prologue_tile(dims, rows, depth), bits, p)
                for dims, bits in self.prologue_specs
            )
        else:
            act *= slots(INPUT_DIMS, self.batch)
        weight += sbuf_bytes(1, cols, self.bias_bits, p)
        weight *= slots(WEIGHT_DIMS, self.weight_batch)
        out *= slots(OUTPUT_DIMS, self.batch)
        for dims, bits in self.fused_specs:
            out += slots(dims, self.batch) * sbuf_bytes(
                *self._out_tile(rows, cols, on, dims), bits, p
            )
        placed = sbuf_placed_bytes(config, buffers, scratch)
        if act + out + weight + placed > config.usable_scratchpad_size:
            return None
        return TrainiumPlacement(buffers=buffers, scratch=scratch)

    def cost(self, mapping, placement):
        """The whole GEMM in TensorE cycles -- its L3 sweep, each step's
        compute overlapping the DMAs of the operands that change, its first
        loads paid only when exposed (``first_load_exposed``) -- and, as
        ``compute``, one step of the GEMM alone, its prologue and tail left
        out.
        ``placement`` is the contract's."""
        model = self.model
        frequency = model.config.frequency
        act_bits, weight_bits, out_bits = self.bits
        rows, depth, cols = self._tile(
            functools.partial(tile_extent, mapping, level=2)
        )
        on = tile_extent(mapping, le.ON, level=2)
        num_k = mapping.loop_blockings[le.IC][3]
        steps = l3_steps(mapping) * self.batch
        mx_in = self.block_size > 1
        compute = self._step_ns(
            rows,
            depth,
            cols,
            on,
            tail_engines=self.tail_engines,
            tail_quantizes_mx=self.tail_quantizes_mx,
            num_k=num_k,
            mx=mx_in,
            prologue_engines=self.prologue_engines,
            tail_copies=self.tail_copies,
            store_issue=model.store_issue_ns(steps // num_k, steps),
        )
        if mx_in:
            act_ns = model.mx_dma_ns(depth, rows, False)
            act_ns += model.mx_dma_ns(depth, rows, True)
            weight_ns = model.mx_dma_ns(depth, cols, False)
            weight_ns += model.mx_dma_ns(depth, cols, True)
        else:
            # A load moves a tile as stored, before it lands transposed.
            weight_tile = (
                (depth, cols)
                if self.weight_k_major != self.weight_loaded
                else (cols, depth)
            )
            act_tile = self._act_tile(
                rows, depth, on, self.act_k_major != self.act_loaded
            )
            act_ns = model.dma_ns(*act_tile, act_bits, load=True)
            weight_ns = model.dma_ns(*weight_tile, weight_bits, load=True)
        if self.bias_bits:
            weight_ns += model.dma_ns(1, cols, self.bias_bits, load=True)
        if self.output_scale_bits:
            out_ns = model.mx_dma_ns(cols, rows, False)
            out_ns += model.mx_dma_ns(cols, rows, True)
        else:
            out_ns = model.dma_ns(*self._out_tile(rows, cols, on), out_bits)
        if self.prologue_specs:
            act_dmas = [
                (
                    model.dma_ns(
                        *self._prologue_tile(dims, rows, depth),
                        bits,
                        load=True,
                    ),
                    batch_loads(
                        mapping,
                        dims,
                        self.batch,
                        self.batch if le.ON in dims else 1,
                    ),
                )
                for dims, bits in self.prologue_specs
            ]
        else:
            act_dmas = [
                (
                    act_ns,
                    batch_loads(mapping, INPUT_DIMS, self.batch, self.batch),
                )
            ]
        dmas = [
            (out_ns, steps // num_k),
            *act_dmas,
            (
                weight_ns,
                batch_loads(
                    mapping, WEIGHT_DIMS, self.batch, self.weight_batch
                ),
            ),
        ]
        for dims, bits in self.fused_specs:
            distinct = self.batch if le.ON in dims else 1
            dmas.append(
                (
                    model.dma_ns(
                        *self._out_tile(rows, cols, on, dims), bits, load=True
                    ),
                    batch_loads(mapping, dims, self.batch, distinct),
                )
            )
        latency = model.sweep_ns(
            dmas, steps, compute, self.first_load_exposed, held=True
        )
        gemm = self._step_ns(rows, depth, cols, on, mx=mx_in, tail_copies=True)
        return Cost(latency * frequency, compute=gemm * frequency)


class TrainiumGemv(TrainiumGrid):
    """A matrix-vector GEMM: an ``(X, C, K)`` tile is a GEMM step of ``X``
    rows contracting ``C`` into ``K`` columns.

    Attributes:
        bits: ``(activation, weight, output)`` element widths.
        weight_k_major: The weight tile lands ``[depth, cols]``, the
            contraction on the partitions.
        tail_engines: The engine of each pass the tail's arithmetic makes
            over an output tile, in order (``pass_engines``).
        tail_quantizes_mx: The tail runs a ``quantize_mx``.
        multiplies_mx: The GEMM takes MX operands.
        operands: What its builder transposes or computes beside its
            operands (``gemm_operands``).
        loaded: The role of each operand that lands transposed from its
            stored layout (``transposed_loads``).
        groups: The groups of ops the translator runs in its kernel
            (``gemm_groups``).
        first_load_exposed: Its first loads overlap no kernel before it
            (``first_load_exposed``).
        bias_dtype: The bias's dtype, ``None`` without one.
    """

    def __init__(self, node, model):
        super().__init__(node, model)
        anchor = get_anchor_node(node)
        tail_ops = tail_targets(node)
        self.bits = (*multiplied_bits(node), widest_bits(node))
        self.weight_k_major = weight_k_major(anchor)
        self.tail_engines = pass_engines(tail_ops)
        self.tail_quantizes_mx = QUANTIZE_MX in tail_ops
        self.multiplies_mx = multiplies_mx(node)
        self.operands = gemm_operands(node)
        self.loaded = {n: role for role, n in transposed_loads(node)}
        self.groups = gemm_groups(node)
        self.first_load_exposed = first_load_exposed(node)
        bias = get_arg_value(anchor, 2, "bias", None)
        self.bias_dtype = None if bias is None else bias.value.dtype

    def placement(self, tiling, shapes, moves):
        """The kernel's ``TrainiumPlacement`` for an ``(X, C, K)`` tile of
        an ``X x K`` output, its tiles cut into ``shapes`` and moved as
        ``moves`` (``transfers``): the buffers its builder allocates
        (``gemm_buffers``) -- its reduction split when the grid cuts ``C``,
        an operand stored transposed staged in its slots -- and the regions
        the translator computes its tail and prologue in (``gemm_scratch``),
        ``None`` when they do not fit PSUM."""
        config = self.model.config
        rows, depth, cols = tiling.tile_sizes
        staging = {
            self.loaded[n]: (
                (*shape[:-2], shape[-1], shape[-2]),
                config.num_slots if count > 1 else 1,
            )
            for n, shape, count in moves
            if n in self.loaded
            and lands_on_tensor_engine(math.prod(shape[:-2]), shape[-1])
        }
        scratch = gemm_scratch(
            config, self.groups, cols, depth, self.bias_dtype
        )
        tile = (1, rows, cols, depth, tiling.grid[1] > 1)
        buffers = gemm_buffers(config, self.operands, scratch, tile, staging)
        if buffers is None:
            return None
        return TrainiumPlacement(buffers=buffers, scratch=scratch)

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``gemv_operand_shapes``)."""
        return gemv_operand_shapes(self.node, tiling)

    def transfers(self, shapes, tiling):
        """Each buffer the sweep moves, with its tile in ``shapes`` and how
        often (``gemv_transfers``)."""
        return gemv_transfers(self.node, shapes, tiling.grid)[1]

    def cost(self, tiling, placement):
        """The tile sweep in TensorE cycles, and its DRAM bytes, its first
        loads paid only when exposed (``first_load_exposed``).  The weight
        stream has a DMA queue to itself: the step compile moves the
        one-row activation and output tiles on queues of their own
        (``TrainiumModel.sweep_ns``'s ``apart``).  A step reads its weight
        tile piece by piece as the pieces land, so once the last lands it
        has left the matmuls that piece feeds and the output tile's
        evictions, each slice's after its last matmul.  ``placement`` is
        the contract's."""
        model = self.model
        config = model.config
        rows, depth, cols = tiling.tile_sizes
        shapes = self.operand_shapes(tiling)
        mx = self._mx_tiles(shapes)
        grid, moves = gemv_transfers(self.node, shapes, tiling.grid)
        steps = math.prod(grid)
        stores = sum(c for n, _, c in moves if n is self.node)
        busy = gemm_step_busy(
            model,
            rows,
            depth,
            cols,
            self.bits,
            self.weight_k_major,
            self.tail_engines,
            self.tail_quantizes_mx,
            tiling.grid[1],
            self.multiplies_mx,
            tail_copies=not self.tail_engines,
            # TensorE transposes the activation ``[depth, 128]`` at a time.
            distance=pairing(self.bits, min(PARTITIONS, rows), cols),
            store_issue=model.store_issue_ns(stores, steps),
        )
        dmas = [
            (model.buffer_dma_ns(n, s, mx, n is not self.node), c)
            for n, s, c in moves
        ]
        apart = {
            i
            for i, (n, s, _) in enumerate(moves)
            if all(rows_cols(b)[0] == 1 for b, _ in output_buffers(s, n))
        }
        if len(apart) == len(moves):
            # One-row tiles alone share one queue.
            apart = set()
        # The weight: the largest tile the steps load.
        weight, shape, _ = max(
            (m for i, m in enumerate(moves) if i and i not in apart),
            key=lambda m: math.prod(m[1]),
        )
        share = 1.0
        if weight not in mx:
            (tile, bits), *_ = output_buffers(shape, weight)
            share = model.last_piece_share(*rows_cols(tile), bits)
        widths = extents(cols, config.moving_free_max)
        row_chunks = math.ceil(rows / config.partitions)
        slices = row_chunks * len(widths)
        if self.tail_engines:
            tail = [(e, 1.0, 32) for e in self.tail_engines]
            finish = max(model.passes_busy(tail, rows, cols).values())
        else:
            finish = row_chunks * sum(
                model.engine_ns("vector", w, 32) for w in widths
            )
        if self.tail_quantizes_mx:
            finish += max(model.quantize_mx_busy(rows, cols).values())
        last = max(share * busy["tensor"], finish - finish / slices)
        latency = model.sweep_ns(
            dmas,
            steps,
            max(busy.values()),
            self.first_load_exposed,
            apart,
            last + finish / slices,
            held=True,
        )
        return Cost(latency * config.frequency, moved_bytes(moves))


# The GEMM op a GEMM's slices call instead of its own: a row-major
# activation is transposed into a contraction-major tile, which
# ``aten.matmul`` cannot say it reads.
SLICE_OPS = {
    torch.ops.aten.matmul.default: torch.ops.quantized_ops.matmul.default
}
MX_GEMMS = (
    torch.ops.quantized_ops.linear_mx.default,
    torch.ops.quantized_ops.matmul_mx.default,
)
# The relayouts a slice's tail re-maps its region through.
RELAYOUTS = (torch.ops.aten.transpose.int, torch.ops.aten.permute.default)
# The buffers of a transposing load, which the pipeline makes for its loads
# (``load_transposed``) rather than ``GemmSteps``.
LOAD_BUFFERS = (*LOAD_STAGING, *LOAD_STAGING.values())
# What a row op's argument is (``GemmSteps.row_op``): the output tile's rows
# it runs on, an operand of the tail, or a constant.
ROWS, OPERAND, CONSTANT = "rows", "operand", "constant"


def alloc_placed(shape, dtype, slots, space):
    """A buffer as a ``TrainiumPlacement`` places it: ``slots`` slots, one
    without a slot dim (``PipelinedKernel._allocate``)."""
    if slots > 1:
        return voyager.alloc(shape, dtype, space, slots)
    return voyager.alloc(shape, dtype, space)


def window(tensor, starts, sizes, squeeze=()):
    """The ``sizes`` window of ``tensor`` at ``starts``, the dims ``squeeze``
    dropped."""
    return voyager.subview(
        tensor,
        list(starts),
        list(sizes),
        [1] * len(starts),
        squeeze_dim=list(squeeze),
    )


def _region_window(tensor, region):
    """The window of ``tensor`` that ``region`` -- ``(start, size)`` per
    dim -- names."""
    return window(tensor, [s for s, _ in region], [n for _, n in region])


def _matrix(tile, coords, starts, sizes):
    """The ``sizes`` window at ``starts`` of ``tile``'s last two dims in the
    output tile's batch element ``coords`` (a broadcast dim at 0), its batch
    dims dropped."""
    lead = tile.dim() - 2
    batch = [
        c if n > 1 else 0
        for c, n in zip(coords[len(coords) - lead :], tile.shape[:lead])
    ]
    return window(
        tile, batch + list(starts), [1] * lead + list(sizes), range(lead)
    )


def _aligned(tile, region):
    """``tile``'s window over ``region``, a region of a shape ``tile``
    broadcasts to: its dims right-aligned on it, a dim of 1 whole."""
    lead = len(region) - tile.dim()
    return _region_window(
        tile,
        [
            (0, 1) if n == 1 else region[lead + d]
            for d, n in enumerate(tile.shape)
        ],
    )


def _reshaped(region, shape, new_shape):
    """``region`` -- ``(start, size)`` per dim of ``shape`` -- as a region of
    ``new_shape``, the same elements in row-major order.

    Raises:
        NotImplementedError: The elements are no box of ``new_shape``.
    """
    starts, sizes = zip(*region)
    box = np.indices(sizes).reshape(len(sizes), -1) + np.array(starts)[:, None]
    coords = np.unravel_index(np.ravel_multi_index(box, shape), new_shape)
    low = [int(c.min()) for c in coords]
    high = [int(c.max()) + 1 for c in coords]
    if math.prod(h - lo for lo, h in zip(low, high)) != box.shape[1]:
        raise NotImplementedError(f"a slice of {shape} viewed as {new_shape}")
    return tuple((lo, h - lo) for lo, h in zip(low, high))


def _scales_region(quantize, region):
    """The region of the scales ``quantize_mx`` node ``quantize`` gives on
    the slice ``region`` of its input: a block of the last dim an element.

    Raises:
        NotImplementedError: The slice cuts a block.
    """
    block = get_arg_value(quantize, 3, "block_size")
    *lead, (start, size) = region
    if start % block or size % block:
        raise NotImplementedError(f"{quantize.name}: part of a block")
    return (*lead, (start // block, size // block))


def transpose_through(source, dst, local, ordinal):
    """Transpose the last two dims of tile ``source`` into ``dst`` on TensorE
    through the PSUM box ``local``, ``[tiles, partitions, width]``: per
    element of their leading dims and 128 of its columns, its rows ``width``
    at a time (128-row blocks side by side) into the next of ``local``'s
    tiles, then copied out of it into ``dst``.  Call ``ordinal`` of a
    sequence starts on the tile after the previous call's, so consecutive
    calls rotate through the tiles."""
    count, parts, width = local.shape
    *lead, rows, cols = source.shape
    unit = [1] * len(lead)
    squeeze = list(range(len(lead)))
    runs = [
        (coords, j, i)
        for coords in itertools.product(*(range(n) for n in lead))
        for j in range(0, cols, parts)
        for i in range(0, rows, width)
    ]
    for n, (coords, j, i) in enumerate(runs):
        c, r = min(parts, cols - j), min(width, rows - i)
        tile = get_slot(local, (ordinal * len(runs) + n) % count)
        tile = window(tile, (0, 0), (c, r))
        block = window(source, [*coords, i, j], [*unit, r, c], squeeze)
        voyager.insert(torch.transpose(block, 0, 1), tile)
        out = window(dst, [*coords, j, i], [*unit, c, r], squeeze)
        voyager.insert(tile.clone(), out)


def transposed_load_buffers(staging, psum):
    """The buffers ``load_transposed`` copies an operand's tile through
    (``_InputSpec.load_buffers``), as the placement places them
    (``_ScratchSpec``): the count and the ``staging`` slots of the tile as
    stored, a semaphore each, and the ``psum`` box."""
    return (
        staging.num_slots,
        alloc_placed(
            staging.shape, staging.dtype, staging.num_slots, staging.space
        ),
        voyager.zeros([], torch.int64, num_slots=staging.num_slots),
        alloc_placed(psum.shape, psum.dtype, psum.num_slots, psum.space),
    )


def load_transposed(buffers, slot, src, dst, indices, sizes, sem, **copy):
    """Load an operand stored transposed into its slot ``dst``
    (``_InputSpec.load``): its tile as stored into staging slot ``slot``
    by DMA, then TensorE transposing it into ``dst`` through the PSUM box
    (``transpose_through``), a ``voyager.commit`` waiting for the DMA and
    posting the operand's semaphore ``sem`` -- which then says ``dst``
    holds the transposed tile.

    Args:
        buffers: ``transposed_load_buffers``'.
        slot: The operand's slot.
        src: The operand in DRAM.
        dst: The slot's tile.
        indices: The DMA's block indices, of the tile as stored.
        sizes: The tile as stored.
        sem: The slot's load semaphore.
        **copy: The rest of ``voyager.async_copy``'s arguments; its
            ``transposed`` is the contract's, true.

    Raises:
        NotImplementedError: The load posts more than once, as the async
            pipeline's reused blocks do: a commit posts once.
    """
    if copy.pop("post_count") != 1:
        raise NotImplementedError("a transposed load posting more than once")
    copy.pop("transposed")
    slots, staging, staged, psum = buffers
    if slots > 1:
        stage = get_slot(staging.view(slots, *sizes), slot)
    else:
        stage = staging.view(sizes)
    stage_sem = get_slot(staged, slot)
    voyager.async_copy(src, stage, indices, sizes, stage_sem, **copy)
    commit(
        transpose_through,
        [stage, dst, psum, slot],
        dependencies=[stage_sem],
        post=sem,
    )


@dataclass(frozen=True)
class SliceTail:
    """A GEMM's fused tail run on one slice of its output tile.

    Attributes:
        module: The tail's ops on the slice, ``[result, *operands] ->
            values``: its views at the slice's sizes, every other op as it
            is, up to the op it stops at.
        operands: The region of each operand tile the slice reads, by the
            tail's operand order; ``None`` for one it does not read.
        results: The region of the output tile each value lands in.
    """

    module: torch.fx.GraphModule
    operands: tuple
    results: tuple

    @classmethod
    def build(cls, tail, region, stop):
        """The ``SliceTail`` of ``tail`` on the slice ``region`` -- ``(start,
        size)`` per dim -- of the GEMM's result, stopping at node ``stop``
        (returning its computed input) or at the end.

        Every value carries its region of the tile-wide value the tail's
        node computes: an elementwise op its first computed input's, which
        each operand it takes is read over (``_aligned``); a view its input's
        re-mapped (``_reshaped``); a relayout its input's permuted; the
        scales ``quantize_mx`` gives its input's, a block of the last dim
        to an element (``_scales_region``).

        Raises:
            NotImplementedError: An op reads no value the tail computes, the
                slice reads an operand over two regions, or it cuts a block
                of ``quantize_mx``'s.
        """
        placeholders = [n for n in tail.graph.nodes if n.op == "placeholder"]
        graph = torch.fx.Graph()
        remap = {p: graph.placeholder(p.name) for p in placeholders}
        regions = {placeholders[0]: region}
        operands = {}
        results = None
        for node in tail.graph.nodes:
            if node.op == "output":
                results = node.args[0]
                if not isinstance(results, (tuple, list)):
                    results = [results]
                break
            if node.op != "call_function":
                continue
            computed = [n for n in node.all_input_nodes if n in regions]
            if not computed:
                raise NotImplementedError(f"{node.name}: reads no slice")
            source = computed[0]
            if node is stop:
                results = [source]
                break
            if is_nop(node):
                regions[node] = _reshaped(
                    regions[source], tuple(source.shape), tuple(node.shape)
                )
                sizes = [size for _, size in regions[node]]
                remap[node] = graph.call_function(
                    torch.ops.aten.view.default, (remap[source], sizes)
                )
                continue
            if node.target in RELAYOUTS:
                (regions[node],) = _apply_relayout(node, regions[source])
            elif (
                node.target is operator.getitem
                and source.target is QUANTIZE_MX
                and node.args[1] == 0
            ):
                regions[node] = _scales_region(source, regions[source])
            else:
                regions[node] = regions[source]
                for p in node.all_input_nodes:
                    if p in placeholders[1:]:
                        read = operands.setdefault(p, regions[source])
                        if read != regions[source]:
                            raise NotImplementedError(f"{p.name}: two regions")
            remap[node] = graph.node_copy(node, remap.__getitem__)
        values = []
        for n in results:
            if n.target is QUANTIZE_MX:
                # Its results, the scales and the codes, each a value.
                scales = graph.call_function(operator.getitem, (remap[n], 0))
                codes = graph.call_function(operator.getitem, (remap[n], 1))
                values += [
                    (scales, _scales_region(n, regions[n])),
                    (codes, regions[n]),
                ]
            else:
                values.append((remap[n], regions[n]))
        graph.output(tuple(value for value, _ in values))
        return cls(
            torch.fx.GraphModule(torch.nn.Module(), graph),
            tuple(operands.get(p) for p in placeholders[1:]),
            tuple(where for _, where in values),
        )


class GemmSteps:
    """A GEMM's steps as TensorE runs them: the kernel
    ``build_pipelined_buffers`` calls on each step, and the tiles it writes
    beside its operands'.

    Attributes:
        plan: The GEMM's ``_GemmPlan``.
        op: The GEMM op each slice calls.
        kwargs: The GEMM's arguments other than tensors, with each slice's
            ``layout``: the GEMM's own, or ``"cx,ck->xk"`` reading a transposed
            row-major activation.
        mx: The GEMM's operands are MX.
        buffers: The buffers the tile search placed, by role
            (``gemm_buffers``), as ``_ScratchSpec``.
        scratch: The regions the translator computes the tail, the
            prologue and a biased product in, as placed (``gemm_scratch``),
            by name.
        transposes: The PSUM box a row-major activation is transposed into,
            else ``None``.
        groups: ``(start, stop)`` of each run of the contraction a
            transposition tile holds: as many whole 128-deep blocks as fit,
            a short last block alone.
        prologue: A prologue computes the activation.
        row_chunks: ``(batch coordinates, row, rows)`` of each 128 rows of
            the output tile as PSUM holds it (the output's columns when it
            is stored transposed), in order.
        columns: ``(column, columns)`` of each run of 512 of its columns.
        tails: The ``SliceTail`` of each slice, by ``(row chunk, column
            run)``; empty without a tail.
        row_op: ``(target, args, kwargs)`` of a tail's row op, run on each
            row chunk of the output tile in place after its slices -- each
            argument ``(ROWS,)``, the rows, ``(OPERAND, k)``, the tail's
            operand ``k``, or ``(CONSTANT, value)`` -- or ``None``.
        staged: The tail's last op transposes the finished tile's last two
            dims, from the staging tile into the output.
        copies: The tail computes nothing before ``row_op`` or ``staged``:
            a slice is copied out of PSUM.
        deferred: The tail computes and moves nothing: a split reduction's
            last step evicts the slices after all its GEMMs.
        results: The PSUM box the slices' GEMMs write.
        scratch_specs: The tiles the kernel writes, ``names`` their roles.
        names: Each tile's role, ``RESULTS`` to ``STAGED_TRANSPOSES``.
        out_specs: The outputs' specs: the plan's, or, after ``staged``,
            the output tile stored as the transposed result tile it is.
        in_specs: The plan's operand specs, an operand stored transposed's
            loaded through TensorE (``load_transposed``).
    """

    def __init__(self, node, plan, model):
        anchor = plan.anchor
        layout = gemm_layout(anchor)
        if layout.weight != "ck":
            raise NotImplementedError(f"{anchor}: a weight stored {layout}")
        config = model.config
        placement = (
            node.meta.get("tile_result") or anchor.meta["tile_result"]
        ).placement
        self.buffers = {
            role: _ScratchSpec(shape, dtype, slots, space)
            for role, shape, dtype, _, slots, space in placement.buffers
        }
        self.scratch = {
            name: _ScratchSpec(*region) for name, *region in placement.scratch
        }
        self.plan = plan
        self.transposes = self.buffers.get(TRANSPOSES)
        self.prologue = PROLOGUE in self.buffers
        if self.prologue and self.transposes is None:
            raise NotImplementedError(f"{anchor}: a prologue read as stored")
        if self.transposes is not None and plan.out_transposed:
            raise NotImplementedError(f"{anchor}: a row-major transposed GEMM")
        self.op = SLICE_OPS.get(anchor.target, anchor.target)
        self.mx = anchor.target in MX_GEMMS
        self.kwargs = {
            k: v
            for k, v in anchor.kwargs.items()
            if not isinstance(v, torch.fx.Node)
        }
        self.kwargs["layout"] = "cx,ck->xk" if self.transposes else str(layout)
        # Each slice's product lands in a PSUM tile, summed in its dtype.
        self.kwargs["out_dtype"] = self.buffers[RESULTS].dtype
        p, free = config.partitions, config.moving_free_max
        *batch, rows, cols = plan.acc_shape
        self.row_chunks = [
            (tuple(int(c) for c in np.unravel_index(e, batch)), i, r)
            for e in range(math.prod(batch))
            for i, r in ((i, min(p, rows - i)) for i in range(0, rows, p))
        ]
        self.columns = [(j, min(free, cols - j)) for j in range(0, cols, free)]
        self._plan_tail(plan.fused_gm, plan.num_k > 1)
        self.results = self.buffers[RESULTS]
        if self.transposes is not None:
            self._plan_transposes(config)
        # The tiles the kernel writes: the placement's but a transposing
        # load's, which the pipeline makes, then the scratch.
        self.names = [role for role in self.buffers if role not in LOAD_BUFFERS]
        self.names += list(self.scratch)
        self.scratch_specs = [
            self.buffers.get(name) or self.scratch[name] for name in self.names
        ]
        self.out_specs = plan.out_specs
        if self.staged:
            # The plan tiles a transposing tail's output as the result.
            gm, gn, _ = plan.grid_dims
            dims = (gm, gn) if plan.out_transposed else (gn, gm)
            (spec,) = plan.out_specs
            self.out_specs = [
                replace(
                    spec,
                    tile_sizes=tuple(batch) + (cols, rows),
                    index_map=tuple(range(len(batch))) + dims,
                )
            ]
        roles = {n: role for role, n in transposed_loads(node)}
        self.in_specs = [
            self._load_through_psum(roles.get(n), spec)
            for n, spec in zip(node.all_input_nodes, plan.in_specs)
        ]

    def _load_through_psum(self, role, spec):
        """An operand's ``spec``, loading it through TensorE
        (``load_transposed``) with the staging slots and the PSUM box the
        placement gives its transposing load's ``role``; as it is when the
        placement gives that role none -- the DMA transposing it -- or the
        operand is no transposing load (``role`` ``None``) or loads whole
        (no ``spec``)."""
        if spec is None or role not in self.buffers:
            return spec
        buffers = functools.partial(
            transposed_load_buffers,
            self.buffers[LOAD_STAGING[role]],
            self.buffers[role],
        )
        return replace(spec, load=load_transposed, load_buffers=buffers)

    def _plan_tail(self, tail, split):
        """What evicting a slice runs (``tails``, ``row_op``, ``staged``,
        ``copies``, ``deferred``) for the fused ``tail``, or a copy when it
        is ``None``; ``split`` says the reduction is split over steps."""
        ops = []
        if tail is not None:
            ops = [n for n in tail.graph.nodes if n.op == "call_function"]
        stop = next((n for n in ops if n.target in ROW_OPS), None)
        self.row_op, self.staged = None, False
        if stop is not None:
            placeholders = [
                n for n in tail.graph.nodes if n.op == "placeholder"
            ]

            def template(arg):
                if arg is stop.args[0]:
                    return (ROWS,)
                if isinstance(arg, torch.fx.Node):
                    return (OPERAND, placeholders.index(arg) - 1)
                if isinstance(arg, (list, tuple)):
                    return (CONSTANT, list(arg))
                return (CONSTANT, arg)

            self.row_op = (
                stop.target,
                [template(a) for a in stop.args],
                {k: template(a) for k, a in stop.kwargs.items()},
            )
        elif ops and swaps_last_two_dims(ops[-1]):
            stop = ops[-1]
            self.staged = True
        self.copies = evicts_by_copy(ops)
        self.deferred = split and all(is_nop(n) for n in ops)
        self.tails = {}
        if tail is None:
            return
        for c, (coords, i, r) in enumerate(self.row_chunks):
            for t, (j, w) in enumerate(self.columns):
                region = tuple((x, 1) for x in coords) + ((i, r), (j, w))
                self.tails[c, t] = SliceTail.build(tail, region, stop)

    def _plan_transposes(self, config):
        """The runs of the contraction (``groups``) a row-major activation
        is transposed in: as many 128-deep blocks as a transposition tile in
        PSUM holds stacked, a short last block alone."""
        p = config.partitions
        depth = self.plan.tile_k
        per_tile = self.transposes.shape[0] // p
        whole = depth // p * p
        self.groups = [
            (lo, min(whole, lo + per_tile * p))
            for lo in range(0, whole, per_tile * p)
        ]
        if depth % p:
            self.groups.append((whole, depth))

    def kernel(self, grid_index, *args):
        """One step, ``build_pipelined_buffers``' kernel: called with the
        step's grid coordinates, its operands' tiles, its outputs' and the
        tiles of ``scratch_specs``.  A split reduction's first step writes
        PSUM, and its last evicts it."""
        plan = self.plan
        inputs, outputs = len(plan.inputs), len(self.out_specs)
        tiles = list(args[:inputs])
        outs = args[inputs : inputs + outputs]
        boxes = dict(zip(self.names, args[inputs + outputs :]))
        if self.staged:
            # The staging tile is placed flat; the tail writes it as the
            # output tile PSUM holds.
            boxes[STAGED] = boxes[STAGED].view(plan.acc_shape)
        step = 0
        for coord, extent in zip(grid_index, plan.grid):
            step = step * extent + coord
        if plan.num_k == 1:
            self._step(tiles, outs, boxes, step, first=True, last=True)
            return
        k = grid_index[plan.grid_dims[2]]

        def branch(first, last):
            def run():
                self._step(tiles, outs, boxes, step, first, last)
                return 0

            return run

        def later():
            torch.cond(
                k == plan.num_k - 1,
                branch(False, True),
                branch(False, False),
            )
            return 0

        torch.cond(k == 0, branch(True, False), later)

    def _step(self, tiles, outs, boxes, step, first, last):
        """A step's work: per row chunk, the activation transposed when
        row-major and the slices' GEMMs into PSUM -- writing it on the
        ``first`` step, adding to it after -- and on the ``last`` step the
        evictions and the tail's whole-tile op."""
        plan = self.plan
        weight = plan.weight(tiles)
        activation = None if self.transposes else plan.activation(tiles)
        operands = [tiles[i] for i in plan.fused_idx]
        evict = last and not self.deferred
        for c, chunk in enumerate(self.row_chunks):
            if self.transposes is not None:
                activation = self._transposed(tiles, boxes, step, c)
            for t, (j, w) in enumerate(self.columns):
                psum = self._psum(boxes, step, c, t)
                product = self._product(
                    tiles, boxes, activation, weight, chunk, j, w, first
                )
                voyager.insert(product if first else product + psum, psum)
                if evict:
                    self._evict(operands, outs, boxes, psum, c, t)
            if evict and self.row_op is not None:
                self._row_op(operands, outs, boxes, chunk)
        if not last:
            return
        if self.deferred:
            for c in range(len(self.row_chunks)):
                for t in range(len(self.columns)):
                    psum = self._psum(boxes, step, c, t)
                    self._evict(operands, outs, boxes, psum, c, t)
        if self.staged:
            transpose_through(
                boxes[STAGED], outs[0], boxes[STAGED_TRANSPOSES], step
            )

    def _psum(self, boxes, step, c, t):
        """The PSUM tile slice ``t`` of row chunk ``c`` writes: its bank of a
        split reduction's sum, or the next of the rotating accumulators."""
        slices = len(self.row_chunks) * len(self.columns)
        index = c * len(self.columns) + t
        if self.plan.num_k == 1:
            index = (step * slices + index) % self.results.num_slots
        rows, cols = self.row_chunks[c][2], self.columns[t][1]
        # One bank is allocated unslotted (``PipelinedKernel._allocate``).
        tile = boxes[RESULTS]
        if self.results.num_slots > 1:
            tile = get_slot(tile, index)
        return window(tile, (0, 0), (rows, cols))

    def _product(self, tiles, boxes, activation, weight, chunk, j, w, first):
        """The GEMM of one slice, ``chunk``'s rows and columns ``j .. j +
        w``: of the transposed row-major activation, or of the part of the
        activation tile the slice reads, by the weight's; the bias on the
        ``first`` step, beside its ``product`` scratch in ``boxes``."""
        plan = self.plan
        coords, i, r = chunk
        depth = plan.tile_k
        kwargs = dict(self.kwargs)
        kwargs.update({name: tiles[k] for name, k in plan.kw_idx.items()})
        if self.transposes is not None:
            act = activation
            weight = _matrix(weight, coords, (0, j), (depth, w))
        elif plan.out_transposed:
            act = _matrix(activation, coords, (0, j), (depth, w))
            weight = _matrix(weight, coords, (0, i), (depth, r))
        elif self.mx:
            # An MX activation keeps its rows, its scales a column per
            # block of the contraction, a weight's a row per block.
            blocks = depth // kwargs["block_size"]
            act = _matrix(activation, coords, (i, 0), (r, depth))
            weight = _matrix(weight, coords, (0, j), (depth, w))
            kwargs["input_scale"] = _matrix(
                kwargs["input_scale"], coords, (i, 0), (r, blocks)
            )
            kwargs["weight_scale"] = _matrix(
                kwargs["weight_scale"], coords, (0, j), (blocks, w)
            )
        else:
            act = _matrix(activation, coords, (0, i), (depth, r))
            weight = _matrix(weight, coords, (0, j), (depth, w))
        if plan.bias_idx is None:
            return self.op(act, weight, **kwargs)
        bias = None
        if first:
            start, size = (i, r) if plan.out_transposed else (j, w)
            bias = window(tiles[plan.bias_idx], (start,), (size,))
            return self._beside_scratch(
                PRODUCT_GROUP,
                lambda a, b, c: self.op(a, b, c, **kwargs),
                [act, weight, bias],
                boxes,
            )
        return self.op(act, weight, bias, **kwargs)

    def _transposed(self, tiles, boxes, step, c):
        """Row chunk ``c`` of the activation transposed: its 128 rows (the
        prologue's result on them, written to its tile first), a group of
        blocks at a time into the next transposition tile in PSUM, each
        tile copied out whole into the SBUF tile returned
        ``[contraction, rows]``."""
        plan = self.plan
        coords, i, r = self.row_chunks[c]
        depth = plan.tile_k
        ordinal = step * len(self.row_chunks) + c
        if self.prologue:
            region = tuple((x, 1) for x in coords) + ((i, r), (0, depth))
            sliced = list(tiles)
            for k in plan.activation_idx:
                if plan.in_specs[k] is not None:
                    sliced[k] = _aligned(tiles[k], region)
            slots = self.buffers[PROLOGUE].num_slots
            pa = get_slot(boxes[PROLOGUE], ordinal % slots)
            source = window(pa, (0, 0), (r, depth))
            positions = plan.activation_idx

            def prologue(*tiles):
                full = [None] * len(sliced)
                for k, tile in zip(positions, tiles):
                    full[k] = tile
                return plan.activation(full)

            computed = self._beside_scratch(
                PROLOGUE_GROUP,
                prologue,
                [sliced[k] for k in positions],
                boxes,
            )
            voyager.insert(computed.view(r, depth), source)
        else:
            activation = plan.activation(tiles)
            source = _matrix(activation, coords, (i, 0), (r, depth))
        slots = self.buffers[TRANSPOSED].num_slots
        at = get_slot(boxes[TRANSPOSED], ordinal % slots)
        for g, (lo, hi) in enumerate(self.groups):
            tile = (ordinal * len(self.groups) + g) % self.transposes.num_slots
            psum = window(
                get_slot(boxes[TRANSPOSES], tile), (0, 0), (hi - lo, r)
            )
            block = window(source, (0, lo), (r, hi - lo))
            voyager.insert(torch.transpose(block, 0, 1), psum)
            voyager.insert(psum.clone(), window(at, (lo, 0), (hi - lo, r)))
        return window(at, (0, 0), (depth, r))

    def _evict(self, operands, outs, boxes, psum, c, t):
        """Evict slice ``t`` of row chunk ``c`` from ``psum``: through the
        tail (``tails``) up to its row op or transpose, or copied, into the
        output tiles, or the staging tile."""
        coords, i, r = self.row_chunks[c]
        j, w = self.columns[t]
        value = psum.clone() if self.copies else psum
        value = value.view([1] * len(coords) + [r, w])
        results = [(value, tuple((x, 1) for x in coords) + ((i, r), (j, w)))]
        if self.tails:
            tail = self.tails[c, t]
            windows = [
                self._operand(operands, k, region)
                for k, region in enumerate(tail.operands)
            ]
            evicted = self._beside_scratch(
                EVICT_GROUP, tail.module, [value, *windows], boxes
            )
            results = zip(evicted, tail.results)
        destinations = [boxes[STAGED]] if self.staged else outs
        for (result, region), out in zip(results, destinations):
            voyager.insert(result, _region_window(out, region))

    def _operand(self, operands, k, region):
        """The tail's operand ``k`` over ``region``: that part of its tile,
        or the tile whole when the slice does not read it (no ``region``)
        or it has no spec -- a scalar, a codebook."""
        tile = operands[k]
        if region is None or self.plan.in_specs[self.plan.fused_idx[k]] is None:
            return tile
        return _aligned(tile, region)

    def _row_op(self, operands, outs, boxes, chunk):
        """The tail's row op on ``chunk``'s rows of the output tile, in
        place, beside its ``row`` scratch in ``boxes``."""
        coords, i, r = chunk
        region = tuple((x, 1) for x in coords) + (
            (i, r),
            (0, self.plan.acc_shape[-1]),
        )
        rows = _region_window(outs[0], region)
        target, args, kwargs = self.row_op
        templates = [*args, *kwargs.values()]
        tensors = [
            rows if arg[0] == ROWS else self._operand(operands, arg[1], region)
            for arg in templates
            if arg[0] != CONSTANT
        ]

        def row_op(*tensors):
            values = iter(tensors)
            filled = [
                arg[1] if arg[0] == CONSTANT else next(values)
                for arg in templates
            ]
            return target(
                *filled[: len(args)], **dict(zip(kwargs, filled[len(args) :]))
            )

        voyager.insert(
            self._beside_scratch(ROW_GROUP, row_op, tensors, boxes), rows
        )

    def _beside_scratch(self, group, compute, operands, boxes):
        """``compute(*operands)`` beside the scratch regions in ``boxes`` of
        the op group ``group`` (``gemm_scratch``), handed to it by name
        (``with_scratch``); bare when the group has none."""
        named = {
            name: boxes[name] for name in group_regions(self.scratch, group)
        }
        if not named:
            return compute(*operands)
        return with_scratch(compute, operands, named)


def build_gemm(
    node: torch.fx.Node, ctx: BuildContext
) -> Optional[torch.fx.GraphModule]:
    """Trainium's builder of a GEMM node, bare or fused: its steps as
    ``GemmSteps`` writes them, their ops fused by the instruction patterns.
    ``None`` when the node has no tiling."""
    plan = _gemm_plan(node, ctx.tiler)
    if plan is None:
        return None
    steps = GemmSteps(node, plan, ctx.tiler.model)
    gm = build_pipelined_buffers(
        steps.kernel,
        plan.grid,
        steps.in_specs,
        steps.out_specs,
        plan.inputs,
        scratch_specs=steps.scratch_specs,
        num_slots=ctx.num_slots,
    )
    fuse_patterns(gm, INSTRUCTION_PATTERNS)
    outline_dps_ops(gm)
    _stamp_anchor_meta(gm, plan.anchor)
    return gm
