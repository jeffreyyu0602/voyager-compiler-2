"""A Triton kernel that bins block-normalized values into weighted moments.

``codebook_optimizer`` reduces each operand to three float64 sums per bin
-- Σw, Σw·v and Σw·v² over the values ``v = x / blockscale`` -- and on a GPU
the eager ops that get there launch about twenty kernels per operand.
``bin_blocks`` does the whole pass in one: each program loads a tile of
blocks where the operand lies, normalizes them, finds each value's bin and
adds its three terms.  Every step is the one the eager path takes, in the
same precision, so the bins are identical and the sums differ only in the
order they are added.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from voyager_compiler.quantization.mx_utils import FP32_MIN_NORMAL

#: Elements one program bins.
TILE = 2048

_MIN_NORMAL = tl.constexpr(FP32_MIN_NORMAL)


@triton.jit
def _bin_kernel(
    x_ptr,
    energy_ptr,
    edges_ptr,
    scale_qmap_ptr,
    global_scale_ptr,
    copies_ptr,
    block_rows,
    blocks,
    length,
    inner,
    inner_tiles,
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
):
    """Bin one tile: ``ROWS`` block rows by ``COLUMNS`` inner columns.

    The operand is read as ``[outer, length, inner]``, blocked along
    ``length``: block row ``r`` is block ``r % blocks`` of outer index
    ``r // blocks``.  Each of the tile's ``ROWS * COLUMNS`` blocks is one
    row of ``BLOCK`` elements below.
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
    # as Histogram.add clamps; the edges are padded with +inf to EDGES, a
    # power of two, so the search halves a range of known size.
    index = tl.zeros(x.shape, dtype=tl.int32)
    step = EDGES // 2
    for _ in tl.static_range(SEARCH_STEPS):
        edge = tl.load(edges_ptr + index + step - 1)
        index = tl.where(edge <= value, index + step, index)
        step = step // 2
    index = tl.minimum(index, BINS - 1)

    # Each element is weighted by its block's squared scale as well, in the
    # energy's precision as the eager path multiplies, before widening.
    square = tl.broadcast_to((scale * scale)[:, None], x.shape)
    if WEIGHTED:
        weight = tl.load(energy_ptr + position, mask=mask, other=0.0)
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
    target = copies_ptr + copy * (3 * BINS) + index
    tl.atomic_add(target, weight, mask=mask, sem="relaxed")
    # An exact zero adds nothing to the two value-weighted sums.
    nonzero = mask & (value != 0)
    tl.atomic_add(target + BINS, weighted, mask=nonzero, sem="relaxed")
    tl.atomic_add(
        target + 2 * BINS, weighted * value, mask=nonzero, sem="relaxed"
    )


def bin_blocks(
    tensor,
    axis,
    block_size,
    quant_max,
    energy,
    edges,
    moments,
    copies,
    power_of_two,
    scale_qmap,
    global_scale,
):
    """Add ``tensor``'s values, normalized by their blocks, to ``moments``.

    A block's scale is the one ``calculate_mx_qparam`` takes, its last block
    along ``axis`` padded short.  Each element weighs its energy times its
    block's squared scale.

    Args:
        tensor: The operand, on a GPU.
        axis: The axis its microscaling blocks run along.
        block_size: Elements per block, a power of two.
        quant_max: The value each block's largest magnitude normalizes to.
        energy: Weight per position along ``axis``, or None to weight every
            element by its block's squared scale alone.
        edges: The interior bin edges as float32, padded with +inf to a
            power of two above their count.
        moments: The ``[3, bins]`` float64 sums the terms are added to.
        copies: Copies of the bins the adds spread over.
        power_of_two: Floor each block scale to a power of two.
        scale_qmap: The 65536-entry table the block scales are quantized
            through, or None.
        global_scale: The tensor's global scale, a 0-d tensor on the GPU,
            or None.
    """
    tensor = tensor.contiguous()
    axis %= tensor.dim()
    length = tensor.shape[axis]
    outer = math.prod(tensor.shape[:axis])
    inner = math.prod(tensor.shape[axis + 1 :])
    blocks = triton.cdiv(length, block_size)
    columns = min(triton.next_power_of_2(inner), 64)
    rows = max(TILE // (block_size * columns), 1)
    inner_tiles = triton.cdiv(inner, columns)
    bins = moments.shape[1]
    scratch = torch.zeros(
        copies, 3, bins, dtype=torch.float64, device=tensor.device
    )
    _bin_kernel[(triton.cdiv(outer * blocks, rows) * inner_tiles,)](
        tensor,
        edges if energy is None else energy.contiguous(),
        edges,
        edges if scale_qmap is None else scale_qmap,
        edges if global_scale is None else global_scale,
        scratch,
        outer * blocks,
        blocks,
        length,
        inner,
        inner_tiles,
        (torch.tensor(1.0) / quant_max).item(),
        float(math.floor(math.log2(quant_max))),
        BINS=bins,
        EDGES=edges.numel(),
        SEARCH_STEPS=edges.numel().bit_length() - 1,
        BLOCK=block_size,
        ROWS=rows,
        COLUMNS=columns,
        COPIES=copies,
        WEIGHTED=energy is not None,
        POWER_OF_TWO=power_of_two,
        SCALE_QMAP=scale_qmap is not None,
        GLOBAL_SCALE=global_scale is not None,
    )
    moments += scratch.sum(0)
