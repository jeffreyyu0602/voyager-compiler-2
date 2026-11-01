"""The Triton kernels behind the CUDA calls of ``quantization.ops``.

``mx_fake_quant`` fake-quantizes a CUDA tensor in blocks along one axis or
two as ``quantization::mx_fake_quant``'s reference does -- block amax,
block scale (a power of two, or divided by a global scale and quantized
through a scale table, if given), division, rounding (to nearest through a
lookup table, or stochastically between levels), rescale, and with a random
Hadamard transform the rotation before and after -- repeating its float32
sums and its roundings to the input's dtype op for op, so the output is
bit-identical to it.  ``hadamard_amax`` takes the rotation's global amax.
``bin_blocks`` bins block-normalized values into weighted moments in one
pass where the reference launches about twenty kernels; its bins are the
reference's and its sums differ only in the order they are added.  The
``*_supported`` predicates say which calls the kernels compute; the ops
run the reference on the others.  ``mx_fake_quant`` and ``hadamard_amax``
run inside ``triton_op`` bodies, which ``torch.compile`` traces, so they
launch through ``wrap_triton``; ``bin_blocks`` backs a plain custom op,
which compile calls opaquely, so it launches its kernel directly.
``ops`` imports this module only when triton, which ships with the CUDA
builds of torch, is installed.
"""

import functools
import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton
from triton.language.extra import libdevice

from voyager_compiler.ops.quantized import (
    HADAMARD_SIZE,
    PHILOX_ROUNDS,
    QMAP_SIZE,
)
from voyager_compiler.quantization.mx_utils import FP32_MIN_NORMAL

#: Elements one Philox draw serves.
DRAW_SIZE = 16
#: Rows and columns of the tile one program fake-quantizes or rotates, each
#: widened to a whole block when blocks run along it, and its warps.
TILE_ROWS, TILE_COLS, NUM_WARPS = 16, 64, 2
#: Tile elements per warp, past which a wider tile takes more warps.
WARP_ELEMENTS = 512
#: Elements one ``_bin_kernel`` program bins.
TILE = 2048
#: Copies of each bin ``bin_blocks`` spreads its adds over.  Neighbouring
#: elements add into different copies, so the many elements of one bin --
#: a ReLU output's zeros -- do not queue on one address; the copies are
#: summed after.
BIN_COPIES = 512

_DRAW_SIZE = tl.constexpr(DRAW_SIZE)
_HADAMARD_SIZE = tl.constexpr(HADAMARD_SIZE)
_PHILOX_ROUNDS = tl.constexpr(PHILOX_ROUNDS)
_MIN_NORMAL = tl.constexpr(FP32_MIN_NORMAL)


@triton.jit
def _round(x, DTYPE: tl.constexpr):
    """``x`` rounded to ``DTYPE``, as float32."""
    return x.to(DTYPE).to(tl.float32)


@triton.jit
def _lookup(table, x):
    """The entry of a table for each of ``x``'s bfloat16 bit patterns, as
    ``vmap`` reads it -- rounded to bfloat16 -- as float32."""
    bits = x.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    return _round(tl.load(table + (bits & 0xFFFF)), tl.bfloat16)


@triton.jit
def _exp2(e):
    """``2 ** e`` for an integral float32 ``e``, exact through the
    subnormals as torch's is, from its bits: libdevice's ``exp2`` flushes
    them to zero."""
    n = tl.minimum(tl.maximum(e, -160.0), 160.0).to(tl.int32)
    normal = (n + 127) << 23
    subnormal = 1 << tl.maximum(n + 149, 0)
    bits = tl.where(n >= -126, normal, tl.where(n >= -149, subnormal, 0))
    bits = tl.where(n >= 128, 0x7F800000, bits)
    return tl.where(e == e, bits.to(tl.float32, bitcast=True), e)


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _philox_byte(seed, counter, lane):
    """Byte ``lane % 4`` of word ``lane // 4`` of the Philox4x32 draw with
    ``counter`` under ``seed``, ``_random_bytes``' eight bits for the element
    at ``16 * counter + lane``; ``counter`` and ``lane`` broadcast."""
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.zeros_like(c0)
    w0, w1, w2, w3 = tl.philox(seed, c0, c1, zero, zero, _PHILOX_ROUNDS)
    word = tl.where(
        lane < 8, tl.where(lane < 4, w0, w1), tl.where(lane < 12, w2, w3)
    )
    return (word >> ((lane % 4) * 8).to(tl.uint32)) & 0xFF


@triton.jit
def _walk_position(program, row_tiles, col_tiles, batch1, batch2):
    """The batch index ``(b0, b1, b2)``, row tile and column tile of a
    program: the programs run over the column tiles, then the row tiles,
    then the batch, its last axis fastest."""
    col_tile = program % col_tiles
    row_tile = program // col_tiles % row_tiles
    batch = program // col_tiles // row_tiles
    b2 = batch % batch2
    b1 = batch // batch2 % batch1
    b0 = batch // batch2 // batch1
    return b0, b1, b2, row_tile, col_tile


@triton.jit
def _tile_positions(
    row_tile,
    col_tile,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    TILE_BLOCK_ROWS: tl.constexpr,
    TILE_BLOCK_COLS: tl.constexpr,
):
    """Each element's row and column in a tile, its block's first, and its
    lane in the block: ``(r, c, block_r, block_c, lane_r, lane_c)``.

    The tile is ``[rows, blocks, lanes]`` for blocks along the columns,
    ``[blocks, lanes, cols]`` along the rows and ``[blocks, lanes, blocks,
    lanes]`` along both -- no axis one long, which Triton spreads threads
    over, repeating each element's work.
    """
    ROWS: tl.constexpr = TILE_BLOCK_ROWS * BLOCK_ROWS
    COLS: tl.constexpr = TILE_BLOCK_COLS * BLOCK_COLS
    first_r = row_tile * ROWS
    first_c = col_tile * COLS
    if BLOCK_ROWS == 1:
        r = first_r + tl.arange(0, ROWS)[:, None, None]
        block_c = tl.arange(0, TILE_BLOCK_COLS)[None, :, None] * BLOCK_COLS
        block_c += first_c
        lane_c = tl.arange(0, BLOCK_COLS)[None, None, :]
        c, block_r, lane_r = block_c + lane_c, r, 0
    elif BLOCK_COLS == 1:
        block_r = tl.arange(0, TILE_BLOCK_ROWS)[:, None, None] * BLOCK_ROWS
        block_r += first_r
        lane_r = tl.arange(0, BLOCK_ROWS)[None, :, None]
        c = first_c + tl.arange(0, COLS)[None, None, :]
        r, block_c, lane_c = block_r + lane_r, c, 0
    else:
        block_r = tl.arange(0, TILE_BLOCK_ROWS)[:, None, None, None]
        block_r = first_r + block_r * BLOCK_ROWS
        lane_r = tl.arange(0, BLOCK_ROWS)[None, :, None, None]
        block_c = tl.arange(0, TILE_BLOCK_COLS)[None, None, :, None]
        block_c = first_c + block_c * BLOCK_COLS
        lane_c = tl.arange(0, BLOCK_COLS)[None, None, None, :]
        r, c = block_r + lane_r, block_c + lane_c
    return r, c, block_r, block_c, lane_r, lane_c


@triton.jit
def _block_amax(tile, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    """Each block's largest magnitude in a tile ``_tile_positions`` laid
    out, NaN winning, kept in its axes."""
    amax = tl.abs(tile)
    if BLOCK_ROWS == 1:
        amax = tl.reduce(amax, 2, _max_with_nan, keep_dims=True)
    elif BLOCK_COLS == 1:
        amax = tl.reduce(amax, 1, _max_with_nan, keep_dims=True)
    else:
        amax = tl.reduce(amax, 3, _max_with_nan, keep_dims=True)
        amax = tl.reduce(amax, 1, _max_with_nan, keep_dims=True)
    return amax


@triton.jit
def _offset(b0, b1, b2, r, c, s_b0, s_b1, s_b2, s_row, s_col):
    """The offset of element ``(r, c)`` of batch index ``(b0, b1, b2)``
    under the strides ``s_*``."""
    return b0 * s_b0 + b1 * s_b1 + b2 * s_b2 + r * s_row + c * s_col


@triton.jit
def _rotate(
    tile,
    h,
    ALONG_COLS: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """``hadamard_rotate`` of a float32 tile holding a ``[ROWS, COLS]``
    matrix in row-major order, along its columns if ``ALONG_COLS``, else its
    rows: each run of ``HADAMARD_SIZE`` times the matrix at ``h`` in
    float32, or its transpose if ``TRANSPOSE``, rounded to ``DTYPE``.

    An IEEE float32 ``tl.dot`` sums each element's products one by one in
    order, from zero, as cuBLAS's float32 GEMM does.

    Returns:
        The rotated tile as float32, in ``tile``'s shape.
    """
    i = tl.arange(0, _HADAMARD_SIZE)[:, None]
    j = tl.arange(0, _HADAMARD_SIZE)[None, :]
    # Element (i, j) of the matrix, and of its transpose.
    matrix, transpose = h + i * _HADAMARD_SIZE + j, h + j * _HADAMARD_SIZE + i
    if ALONG_COLS:
        runs = tl.reshape(tile, (ROWS * COLS // _HADAMARD_SIZE, _HADAMARD_SIZE))
        factor = tl.load(transpose if TRANSPOSE else matrix).to(tl.float32)
        rotated = tl.dot(runs, factor, input_precision="ieee")
    else:
        # The runs are columns, multiplied by the matrix's transpose from
        # the left.
        RUNS: tl.constexpr = ROWS // _HADAMARD_SIZE
        runs = tl.reshape(tile, (RUNS, _HADAMARD_SIZE, COLS))
        factor = tl.load(matrix if TRANSPOSE else transpose).to(tl.float32)
        factor = tl.broadcast_to(
            factor[None, :, :], (RUNS, _HADAMARD_SIZE, _HADAMARD_SIZE)
        )
        rotated = tl.dot(factor, runs, input_precision="ieee")
    return tl.reshape(_round(rotated, DTYPE), tile.shape)


@triton.jit
def _round_stochastically(
    v, byte, levels, NUM_LEVELS: tl.constexpr, SEARCH_STEPS: tl.constexpr
):
    """``_stochastic_round`` of float32 ``v`` between ``NUM_LEVELS``
    ascending float32 ``levels`` by its random ``byte``, as float32."""
    v = tl.maximum(v, tl.load(levels), propagate_nan=tl.PropagateNan.ALL)
    v = tl.minimum(
        v, tl.load(levels + NUM_LEVELS - 1), propagate_nan=tl.PropagateNan.ALL
    )
    # The levels below v, found as ``searchsorted`` finds them; a NaN sorts
    # after every level.
    below = tl.zeros(v.shape, tl.int32)
    for k in tl.static_range(SEARCH_STEPS):
        step = 1 << (SEARCH_STEPS - 1 - k)
        probe = tl.load(levels + tl.minimum(below + step, NUM_LEVELS) - 1)
        below = tl.where(probe < v, below + step, below)
    below = tl.where(v != v, NUM_LEVELS - 1, below)
    high = tl.load(levels + below)
    low = tl.where(high == v, high, tl.load(levels + tl.maximum(below - 1, 0)))
    near = tl.where(v < 0, high, low)
    far = tl.where(v < 0, low, high)
    noise = byte.to(tl.float32) * (1.0 / 256.0)
    # The product is rounded before the sum, as torch's separate ops round
    # it, not fused into one multiply-add.
    jitter = libdevice.mul_rn(noise, high - low)
    away = tl.abs(v) + jitter >= tl.abs(far)
    return tl.where(away, far, near)


@triton.jit
def _mx_fake_quant_kernel(
    x,
    out,
    table,
    scale_qmap,
    global_scale,
    seed,
    hadamard,
    quant_max,
    quant_max_exponent,
    rows,
    cols,
    row_tiles,
    col_tiles,
    batch1,
    batch2,
    x_b0,
    x_b1,
    x_b2,
    x_row,
    x_col,
    out_b0,
    out_b1,
    out_b2,
    out_row,
    out_col,
    flat_b0,
    flat_b1,
    flat_b2,
    flat_row,
    flat_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    TILE_BLOCK_ROWS: tl.constexpr,
    TILE_BLOCK_COLS: tl.constexpr,
    ROTATE_COLS: tl.constexpr,
    POWER_OF_TWO: tl.constexpr,
    DRAW_PER_BLOCK: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Fake-quantize one tile of ``x`` into ``out``, as ``_Walk`` reads
    them: a ``[rows, cols]`` plane for each batch index, the batch's last
    two axes ``batch1`` and ``batch2`` long, addressed through the strides
    ``x_*`` and ``out_*``.

    A block is ``BLOCK_ROWS`` x ``BLOCK_COLS`` elements, powers of two -- 1
    along an axis the blocks do not run along -- and a tile
    ``TILE_BLOCK_ROWS`` x ``TILE_BLOCK_COLS`` blocks, ``row_tiles`` x
    ``col_tiles`` per plane.  ``table`` is the lookup table nearest rounding
    reads, or with a ``seed`` the ``NUM_LEVELS`` levels stochastic rounding
    searches in ``SEARCH_STEPS`` halvings.  The random bytes follow the
    tensor's flat row-major order, its strides ``flat_*``, whatever its
    layout.  ``DRAW_PER_BLOCK``: each block is 16 consecutive elements of
    that order starting at a multiple of 16, one Philox draw.
    With a ``hadamard`` matrix the tile is rotated along its columns if
    ``ROTATE_COLS``, else its rows, before the block amax and rotated back
    by the transpose before the store.  The values round to ``x``'s dtype
    wherever the reference's do.
    """
    DTYPE: tl.constexpr = x.dtype.element_ty
    ROWS: tl.constexpr = TILE_BLOCK_ROWS * BLOCK_ROWS
    COLS: tl.constexpr = TILE_BLOCK_COLS * BLOCK_COLS
    b0, b1, b2, row_tile, col_tile = _walk_position(
        tl.program_id(0).to(tl.int64), row_tiles, col_tiles, batch1, batch2
    )
    r, c, block_r, block_c, lane_r, lane_c = _tile_positions(
        row_tile,
        col_tile,
        BLOCK_ROWS,
        BLOCK_COLS,
        TILE_BLOCK_ROWS,
        TILE_BLOCK_COLS,
    )
    mask = (r < rows) & (c < cols)
    offset = _offset(b0, b1, b2, r, c, x_b0, x_b1, x_b2, x_row, x_col)
    tile = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
    if hadamard is not None:
        tile = tl.reshape(
            _rotate(
                tl.reshape(tile, (ROWS, COLS)),
                hadamard,
                ROTATE_COLS,
                False,
                ROWS,
                COLS,
                DTYPE,
            ),
            tile.shape,
        )

    # The block scale as ``calculate_mx_qparam`` forms it: a power of two
    # from the amax's exponent, or the amax over quant_max -- a multiply by
    # its float32 reciprocal, as eager CUDA divides by a scalar -- divided
    # by the global scale and looked up in the scale table.  The elements
    # are divided by it times the global scale.
    amax = _block_amax(tile, BLOCK_ROWS, BLOCK_COLS)
    if global_scale is not None:
        t = _round(tl.load(global_scale).to(tl.float32), DTYPE)
    if POWER_OF_TWO:
        amax = _round(tl.where(amax == 0.0, _MIN_NORMAL, amax), DTYPE)
        exponent = libdevice.floor(_round(libdevice.log2(amax), DTYPE))
        scale = _round(_exp2(exponent - quant_max_exponent), DTYPE)
    else:
        # torch.compile passes a float as float64.
        reciprocal = tl.math.div_rn(1.0, tl.cast(quant_max, tl.float32))
        scale = _round(amax * reciprocal, DTYPE)
        if global_scale is not None:
            scale = _round(tl.math.div_rn(scale, t), DTYPE)
        if scale_qmap is not None:
            scale = _round(_lookup(scale_qmap, scale), DTYPE)
    scale = tl.where(scale > 0.0, scale, 1.0)
    element_scale = scale
    if global_scale is not None:
        element_scale = _round(scale * t, DTYPE)
    v = _round(tl.math.div_rn(tile, element_scale), DTYPE)

    if seed is not None:
        # Element i in flat order takes byte i % 16 of draw i // 16.
        if DRAW_PER_BLOCK:
            flat = _offset(
                b0,
                b1,
                b2,
                block_r,
                block_c,
                flat_b0,
                flat_b1,
                flat_b2,
                flat_row,
                flat_col,
            )
            # The block runs along one axis; the other's one lane is 0.
            lane = lane_r + lane_c
        else:
            flat = _offset(
                b0, b1, b2, r, c, flat_b0, flat_b1, flat_b2, flat_row, flat_col
            )
            lane = (flat % _DRAW_SIZE).to(tl.int32)
        byte = _philox_byte(tl.load(seed), flat // _DRAW_SIZE, lane)
        q = _round_stochastically(v, byte, table, NUM_LEVELS, SEARCH_STEPS)
    else:
        q = _lookup(table, v)
    output = _round(_round(q, DTYPE) * scale, DTYPE)
    if global_scale is not None:
        output = _round(output * t, DTYPE)
    if hadamard is not None:
        output = tl.reshape(
            _rotate(
                tl.reshape(output, (ROWS, COLS)),
                hadamard,
                ROTATE_COLS,
                True,
                ROWS,
                COLS,
                DTYPE,
            ),
            output.shape,
        )
    offset = _offset(b0, b1, b2, r, c, out_b0, out_b1, out_b2, out_row, out_col)
    tl.store(out + offset, output.to(DTYPE), mask=mask)


@triton.jit
def _hadamard_amax_kernel(
    x,
    hadamard,
    partial,
    rows,
    cols,
    row_tiles,
    col_tiles,
    batch1,
    batch2,
    x_b0,
    x_b1,
    x_b2,
    x_row,
    x_col,
    ALONG_COLS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """The largest magnitude of one ``[TILE_ROWS, TILE_COLS]`` tile of
    ``x``, read as ``_mx_fake_quant_kernel`` reads it, its runs along the
    columns if ``ALONG_COLS``, else along the rows, rotated by
    ``hadamard``; into ``partial``, one element per program."""
    program = tl.program_id(0).to(tl.int64)
    b0, b1, b2, row_tile, col_tile = _walk_position(
        program, row_tiles, col_tiles, batch1, batch2
    )
    r = row_tile * TILE_ROWS + tl.arange(0, TILE_ROWS)[:, None]
    c = col_tile * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    offset = _offset(b0, b1, b2, r, c, x_b0, x_b1, x_b2, x_row, x_col)
    tile = tl.load(x + offset, mask=(r < rows) & (c < cols), other=0.0).to(
        tl.float32
    )
    tile = _rotate(
        tile,
        hadamard,
        ALONG_COLS,
        False,
        TILE_ROWS,
        TILE_COLS,
        x.dtype.element_ty,
    )
    amax = tl.reduce(tl.abs(tile), None, _max_with_nan)
    tl.store(partial + program, amax.to(partial.dtype.element_ty))


@triton.jit
def _bin_kernel(
    x_ptr,
    energy_ptr,
    edges_ptr,
    scale_qmap_ptr,
    global_scale_ptr,
    row_cells_ptr,
    column_cells_ptr,
    copies_ptr,
    block_rows,
    blocks,
    length,
    inner,
    inner_tiles,
    cells,
    reciprocal,
    quant_max_exponent,
    BINS: tl.constexpr,
    EDGES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    COPIES: tl.constexpr,
    WEIGHTED: tl.constexpr,
    POWER_OF_TWO: tl.constexpr,
    SCALE_QMAP: tl.constexpr,
    GLOBAL_SCALE: tl.constexpr,
    GRID: tl.constexpr,
):
    """Bin one tile: ``ROWS`` block rows by ``COLUMNS`` inner columns.

    The operand is read as ``[outer, length, inner]``, blocked along
    ``length``: block row ``r`` is block ``r % blocks`` of outer index
    ``r // blocks``.  Each of the tile's ``ROWS * COLUMNS`` blocks is one
    row of ``BLOCK`` elements below.  With a grid, a block's cell is its
    outer index's share of the cell index plus its column's.
    """
    tile = tl.program_id(0)
    block = tl.arange(0, ROWS * COLUMNS)
    row = tile // inner_tiles * ROWS + block // COLUMNS
    column = tile % inner_tiles * COLUMNS + block % COLUMNS
    position = (row % blocks)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    mask = (
        (row < block_rows)[:, None]
        & (column < inner)[:, None]
        & (position < length)
    )
    outer = (row // blocks).to(tl.int64)[:, None]
    offset = (outer * length + position) * inner + column[:, None]
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    cell = 0
    if GRID:
        cell = tl.load(
            row_cells_ptr + row // blocks, mask=row < block_rows, other=0
        ) + tl.load(column_cells_ptr + column, mask=column < inner, other=0)
        cell = cell.to(tl.int64)[:, None]

    # The block scale as calculate_mx_qparam forms it on a GPU: a power of
    # two from the maximum's exponent, or the maximum over quant_max -- a
    # multiply by its float32 reciprocal, as eager CUDA divides by a scalar
    # -- divided by the global scale and looked up in the scale codebook as
    # vmap looks it up.  The scale the elements are divided by is that one
    # times the global scale, as quantize_mx forms it.
    amax = tl.max(tl.abs(x), axis=1)
    if GLOBAL_SCALE:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
    if POWER_OF_TWO:
        amax = tl.where(amax == 0, _MIN_NORMAL, amax)
        exponent = libdevice.floor(libdevice.log2(amax)) - quant_max_exponent
        scale = libdevice.exp2(exponent)
    else:
        scale = amax * reciprocal
        if GLOBAL_SCALE:
            scale = tl.math.div_rn(scale, global_scale)
        if SCALE_QMAP:
            pattern = scale.to(tl.bfloat16).to(tl.int16, bitcast=True)
            entry = pattern.to(tl.int32) & 0xFFFF
            scale = tl.load(scale_qmap_ptr + entry).to(tl.float32)
    scale = tl.where(scale > 0, scale, 1.0)
    if GLOBAL_SCALE:
        scale = scale * global_scale
    value = tl.math.div_rn(x, tl.broadcast_to(scale[:, None], x.shape))

    # A value's bin is the number of interior edges at or below it, clamped
    # as the reference clamps; the edges are padded with +inf to EDGES, a
    # power of two, so the search halves a range of known size.
    index = tl.zeros(x.shape, dtype=tl.int32)
    step = EDGES // 2
    for _ in tl.static_range(SEARCH_STEPS):
        edge = tl.load(edges_ptr + index + step - 1)
        index = tl.where(edge <= value, index + step, index)
        step = step // 2
    index = tl.minimum(index, BINS - 1)

    # Each element is weighted by its block's squared scale as well, in the
    # energy's precision as the reference multiplies, before widening.
    square = tl.broadcast_to((scale * scale)[:, None], x.shape)
    if WEIGHTED:
        weight = tl.load(
            energy_ptr + cell * length + position, mask=mask, other=0.0
        )
        weight = weight * square
    else:
        weight = square
    weight = weight.to(tl.float64)
    value = value.to(tl.float64)
    weighted = weight * value

    # Neighbouring elements add into different copies of the bins, so a run
    # of equal values -- a ReLU's zeros, each block's maximum -- does not
    # queue on one address.
    element = block[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    copy = ((element + tile) % COPIES).to(tl.int64)
    target = copies_ptr + (copy * cells + cell) * (3 * BINS) + index
    tl.atomic_add(target, weight, mask=mask, sem="relaxed")
    # An exact zero adds nothing to the two value-weighted sums.
    nonzero = mask & (value != 0)
    tl.atomic_add(target + BINS, weighted, mask=nonzero, sem="relaxed")
    tl.atomic_add(
        target + 2 * BINS, weighted * value, mask=nonzero, sem="relaxed"
    )


class _Walk(NamedTuple):
    """How the kernels walk a tensor in place: tiles over a ``rows`` x
    ``cols`` plane for each index of a batch of three axes.

    The columns are the tensor's fastest axis in memory where the blocks
    allow, so a tile's loads are contiguous; the rows the blocks' other
    axis, or the longest.  An axis that continues the rows or the columns in
    memory, in the output and in row-major order merges into them when it
    holds no blocks; the other axes are the batch, ``batch`` their lengths,
    padded with 1.  ``row_axis`` and ``col_axis`` are the innermost tensor
    axes of the rows and the columns, ``row_axis`` None for a 1-D tensor.
    ``strides`` holds, for the input, the output and the row-major order,
    the strides of the three batch axes, the rows and the columns.
    """

    rows: int
    cols: int
    batch: Tuple[int, int, int]
    row_axis: Optional[int]
    col_axis: int
    strides: Tuple[Tuple[int, ...], ...]


def _walk(input: torch.Tensor, axes: List[int]) -> Optional[_Walk]:
    """How the kernels walk ``input``, blocked along ``axes``, and its
    output, laid out as ``torch.empty_like(input)`` lays it out: as
    ``input`` if that is dense, else row-major.

    Returns:
        The walk, or None for blocks along more than two axes or more than
        three batch axes left.
    """
    ndim, shape = input.ndim, input.shape
    axes = sorted(axis % ndim for axis in axes)
    if len(axes) > 2:
        return None
    flat = [math.prod(shape[d + 1 :]) for d in range(ndim)]
    # Dense: the axes, ordered by stride, tile the memory exactly.
    span, dense = 1, True
    for d in sorted(range(ndim), key=input.stride):
        if shape[d] > 1:
            dense &= input.stride(d) == span
            span *= shape[d]
    layouts = (input.stride(), input.stride() if dense else flat, flat)

    def speed(d):
        # A broadcast or one-long axis moves through no memory.
        stride = input.stride(d)
        return (stride == 0 or shape[d] == 1, stride, -d)

    # Sorted rather than min or max with a key: torch.compile traces the
    # former only, and a graph break here splits every fake-quant's forward.
    fastest = sorted(range(ndim), key=speed)[0]
    if len(axes) == 2:
        col_axis, row_axis = sorted(axes, key=speed)
    elif fastest == axes[0]:
        # The rows need not be contiguous, only long enough to fill tiles.
        col_axis = fastest
        others = sorted(
            (d for d in range(ndim) if d != fastest),
            key=lambda d: (shape[d], d),
        )
        row_axis = others[-1] if others else None
    else:
        col_axis, row_axis = fastest, axes[0]
    groups = {}
    for axis in (row_axis, col_axis):
        if axis is not None:
            groups[axis] = [shape[axis], axis not in axes]
    batch = [d for d in range(ndim) if d not in groups and shape[d] > 1]
    merged = True
    while merged:
        merged = False
        for axis, (length, open_) in groups.items():
            for d in batch:
                if open_ and all(
                    layout[d] == length * layout[axis] for layout in layouts
                ):
                    groups[axis][0] *= shape[d]
                    batch.remove(d)
                    merged = True
                    break
    if len(batch) > 3:
        return None
    batch = [None] * (3 - len(batch)) + batch

    def strides(layout):
        return tuple(
            0 if d is None else layout[d] for d in (*batch, row_axis, col_axis)
        )

    return _Walk(
        groups[row_axis][0] if row_axis is not None else 1,
        groups[col_axis][0],
        tuple(1 if d is None else shape[d] for d in batch),
        row_axis,
        col_axis,
        tuple(strides(layout) for layout in layouts),
    )


def _is_lookup_table(table: torch.Tensor) -> bool:
    """Whether ``table`` holds one entry per bfloat16 bit pattern."""
    return table.is_floating_point() and table.shape == (QMAP_SIZE,)


def mx_fake_quant_supported(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: List[int],
    block_size: int,
    quant_max: float,
    power_of_two: bool,
    scale_qmap: Optional[torch.Tensor],
    seed: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    hadamard: Optional[torch.Tensor],
    hadamard_axis: Optional[int],
) -> bool:
    """Whether ``mx_fake_quant`` computes a call of the op.

    It takes a non-empty bfloat16, float16 or float32 CUDA tensor in any
    layout ``_walk`` walks, in blocks of a power-of-two size along one axis
    or two; a lookup table and scale table, or levels stochastic rounding
    searches; every tensor on the input's device; and a rotation, by a
    contiguous matrix, along the walk's rows or columns and one of the last
    two axes -- the ones ``hadamard_rotate`` takes -- whole runs of
    ``HADAMARD_SIZE`` that ``hadamard_rotate`` pads nothing.

    Args:
        The op's arguments; ``quant_max`` and ``power_of_two`` do not
        decide it.
    """
    tensors = [qmap, scale_qmap, seed, global_scale, hadamard]
    walk = _walk(input, axes)
    if seed is not None:
        table_fits = qmap.dtype is torch.float32 and qmap.dim() == 1
    else:
        table_fits = _is_lookup_table(qmap)
    return (
        table_fits
        and input.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input.numel() > 0
        and block_size & (block_size - 1) == 0
        and walk is not None
        and (scale_qmap is None or _is_lookup_table(scale_qmap))
        and all(t is None or t.device == input.device for t in tensors)
        and (
            hadamard is None
            or hadamard.is_contiguous()
            and hadamard_axis % input.ndim >= input.ndim - 2
            and hadamard_axis % input.ndim in (walk.row_axis, walk.col_axis)
            and input.shape[hadamard_axis] % HADAMARD_SIZE == 0
        )
    )


def mx_fake_quant(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: List[int],
    block_size: int,
    quant_max: float,
    power_of_two: bool,
    scale_qmap: Optional[torch.Tensor],
    seed: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    hadamard: Optional[torch.Tensor],
    hadamard_axis: Optional[int],
) -> torch.Tensor:
    """``quantization::mx_fake_quant`` of a call ``mx_fake_quant_supported``
    takes, in one kernel.

    Args:
        The op's arguments.

    Returns:
        The fake-quantized tensor, laid out as ``input``.
    """
    output = torch.empty_like(input)
    walk = _walk(input, axes)
    axes = [axis % input.ndim for axis in axes]
    block_rows = block_size if walk.row_axis in axes else 1
    block_cols = block_size if walk.col_axis in axes else 1
    tile_block_rows = max(TILE_ROWS // block_rows, 1)
    tile_block_cols = max(TILE_COLS // block_cols, 1)
    row_tiles = triton.cdiv(walk.rows, tile_block_rows * block_rows)
    col_tiles = triton.cdiv(walk.cols, tile_block_cols * block_cols)
    elements = tile_block_rows * block_rows * tile_block_cols * block_cols
    # Each block is one draw if it is 16 elements along the tensor's last
    # axis, whose length is a multiple of 16.
    draw_per_block = (
        seed is not None
        and axes == [input.ndim - 1]
        and block_size == DRAW_SIZE
        and input.shape[-1] % DRAW_SIZE == 0
    )
    num_levels = qmap.numel() if seed is not None else 1
    grid = (math.prod(walk.batch) * row_tiles * col_tiles,)
    wrap_triton(_mx_fake_quant_kernel)[grid](
        input,
        output,
        qmap,
        scale_qmap,
        global_scale,
        seed,
        hadamard,
        quant_max,
        float(math.floor(math.log2(quant_max))),
        walk.rows,
        walk.cols,
        row_tiles,
        col_tiles,
        *walk.batch[1:],
        *walk.strides[0],
        *walk.strides[1],
        *walk.strides[2],
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        TILE_BLOCK_ROWS=tile_block_rows,
        TILE_BLOCK_COLS=tile_block_cols,
        ROTATE_COLS=(
            hadamard is not None and hadamard_axis % input.ndim == walk.col_axis
        ),
        POWER_OF_TWO=power_of_two,
        DRAW_PER_BLOCK=draw_per_block,
        NUM_LEVELS=num_levels,
        SEARCH_STEPS=(num_levels - 1).bit_length(),
        num_warps=min(max(NUM_WARPS, elements // WARP_ELEMENTS), 16),
    )
    return output


def hadamard_amax_supported(input: torch.Tensor, axis: int) -> bool:
    """Whether ``hadamard_amax`` computes a call of the op exactly: a CUDA
    ``input``, not float64, in a layout ``_walk`` walks with ``axis``, one
    of its last two, as its rows or columns, rotated along that axis, whole
    runs of ``HADAMARD_SIZE``."""
    return (
        input.is_cuda
        and input.dtype is not torch.float64
        and input.numel() > 0
        and axis % input.ndim >= input.ndim - 2
        and _walk(input, [axis]) is not None
        and input.shape[axis] % HADAMARD_SIZE == 0
    )


def hadamard_amax(
    input: torch.Tensor, axis: int, hadamard: torch.Tensor
) -> torch.Tensor:
    """``quantization::hadamard_amax`` of a call ``hadamard_amax_supported``
    takes: one kernel's per-tile maxima, then their max.

    Args:
        The op's arguments.

    Returns:
        A 0-d tensor in ``input``'s dtype.
    """
    walk = _walk(input, [axis])
    row_tiles = triton.cdiv(walk.rows, TILE_ROWS)
    col_tiles = triton.cdiv(walk.cols, TILE_COLS)
    grid = (math.prod(walk.batch) * row_tiles * col_tiles,)
    partial = input.new_empty(grid)
    wrap_triton(_hadamard_amax_kernel)[grid](
        input,
        # The kernel reads it row-major.
        hadamard.contiguous(),
        partial,
        walk.rows,
        walk.cols,
        row_tiles,
        col_tiles,
        *walk.batch[1:],
        *walk.strides[0],
        ALONG_COLS=axis % input.ndim == walk.col_axis,
        TILE_ROWS=TILE_ROWS,
        TILE_COLS=TILE_COLS,
        num_warps=NUM_WARPS,
    )
    return partial.amax()


@functools.lru_cache(maxsize=None)
def _cell_maps(shape, axis, counts, device):
    """Each outer row's and each inner column's share of a block's cell.

    The tensor reads as ``[outer, length, inner]``, with ``axis`` split
    into its ``counts[axis]`` runs and the run the last outer index.  A
    cell's row-major index is a sum of one term per axis, so the outer
    axes' terms and the inner axes' add.

    Args:
        shape: The tensor's shape.
        axis: The axis its blocks run along.
        counts: Runs along each axis.
        device: Where the maps live.

    Returns:
        ``(rows, columns)``: int32 shares over the outer rows and the inner
        columns.
    """
    strides = [math.prod(counts[d + 1 :]) for d in range(len(shape))]

    def terms(axes):
        total = torch.zeros((), dtype=torch.int64)
        for d in axes:
            run = torch.arange(shape[d]) // (shape[d] // counts[d])
            total = total[..., None] + run * strides[d]
        return total

    rows = terms(range(axis))[..., None]
    rows = rows + torch.arange(counts[axis]) * strides[axis]
    columns = terms(range(axis + 1, len(shape)))
    return (
        rows.flatten().int().to(device),
        columns.flatten().int().to(device),
    )


def bin_blocks(
    tensor: torch.Tensor,
    axis: int,
    block_size: int,
    quant_max: float,
    energy: Optional[torch.Tensor],
    edges: torch.Tensor,
    sums: List[torch.Tensor],
    counts: List[int],
    power_of_two: bool,
    scale_qmap: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
) -> None:
    """``quantization::bin_blocks`` in one kernel, for any CUDA call.

    Args:
        The op's arguments.
    """
    tensor = tensor.contiguous()
    axis %= tensor.dim()
    runs = counts[axis]
    length = tensor.shape[axis] // runs
    outer = math.prod(tensor.shape[:axis]) * runs
    inner = math.prod(tensor.shape[axis + 1 :])
    cells = len(sums)
    row_cells = column_cells = edges
    if cells > 1:
        row_cells, column_cells = _cell_maps(
            tuple(tensor.shape), axis, tuple(counts), tensor.device
        )
    blocks = triton.cdiv(length, block_size)
    columns = min(triton.next_power_of_2(inner), 64)
    rows = max(TILE // (block_size * columns), 1)
    inner_tiles = triton.cdiv(inner, columns)
    bins = sums[0].shape[1]
    scratch = torch.zeros(
        BIN_COPIES, cells, 3, bins, dtype=torch.float64, device=tensor.device
    )
    _bin_kernel[(triton.cdiv(outer * blocks, rows) * inner_tiles,)](
        tensor,
        edges if energy is None else energy.contiguous(),
        edges,
        edges if scale_qmap is None else scale_qmap,
        edges if global_scale is None else global_scale,
        row_cells,
        column_cells,
        scratch,
        outer * blocks,
        blocks,
        length,
        inner,
        inner_tiles,
        cells,
        (torch.tensor(1.0) / quant_max).item(),
        float(math.floor(math.log2(quant_max))),
        BINS=bins,
        EDGES=edges.numel(),
        SEARCH_STEPS=edges.numel().bit_length() - 1,
        BLOCK=block_size,
        ROWS=rows,
        COLUMNS=columns,
        COPIES=BIN_COPIES,
        WEIGHTED=energy is not None,
        POWER_OF_TWO=power_of_two,
        SCALE_QMAP=scale_qmap is not None,
        GLOBAL_SCALE=global_scale is not None,
        GRID=cells > 1,
    )
    torch._foreach_add_(sums, list(scratch.sum(0)))
