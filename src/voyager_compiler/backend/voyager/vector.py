"""Voyager's vector unit: a vector op, a matrix-vector GEMM or a pool cut
into a grid of tiles (``VoyagerVector``, ``VoyagerGemv``, ``VoyagerPool``),
its operand tiles packed into scratchpad banks, and the DRAM-aware latency
and traffic model that ranks the tilings.

``vector_tile_latency`` and ``gemv_tile_latency`` score one candidate tiling
of an op that runs on Voyager's vector unit -- an elementwise / reduction op,
or a matrix-vector GEMM -- so the grid search can rank the tiles that fit
rather than take the largest (``VoyagerVector.cost``, ``VoyagerGemv.cost``).
Both price the transfers the shared schedule makes
(``traversal.vector_transfers`` / ``gemv_transfers``) and overlap them the way
``PipelinedKernel`` does (``traversal.sweep_cycles``).  The scratchpad bank
walks here price a GEMM's streams too (``gemm.py``).

Kept independent of the ``reporting`` package: the tile search runs during
``compile()``'s bufferization, before any reporting.  The dependency runs the
other way -- ``reporting/cost.op_utilization`` calls
``vector_op_utilization`` for its vector branch, so this is the single copy of
that formula (and of ``OP_PASSES``).
"""

import functools
import itertools
import logging
import math

import torch

from voyager_compiler.codegen.node_info import (
    LAYER_NORM_OPS,
    SOFTMAX_OPS,
    _pair,
    bound_operands,
    get_anchor_node,
    get_arg_value,
    is_full_reduction,
    is_fully_connected,
    is_pooling,
    reduction_scratch,
    require_allocation,
    stream_breaking_quantize,
    tensor_alloc_bytes,
    weight_transforms,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    gemv_operand_shapes,
    node_dtype_bits,
    operand_bytes,
    pool_operand_shapes,
    vector_operand_shapes,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    KernelInput,
    OpModel,
    Placement,
    Role,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    gemv_transfers,
    operand_roles,
    sweep_cycles,
    vector_transfers,
)
from voyager_compiler.ops.layout import (
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    unproject,
)

logger = logging.getLogger(__name__)

# Passes an op makes over its data, each as what it streams in and what it
# writes per element: IN the op's input, MID the intermediate a normalization
# stages between passes, OUT its output, None a reduction that writes one
# value per row.  Single source of truth: reporting/cost.py imports this via
# ``vector_op_utilization``.
IN, MID, OUT = "in", "mid", "out"
_LAYER_NORM_PASSES = [(IN, None), (IN, None), (IN, MID), (MID, OUT)]
_SOFTMAX_PASSES = [(IN, None), (IN, None), (IN, OUT)]
OP_PASSES = {
    **dict.fromkeys(LAYER_NORM_OPS, _LAYER_NORM_PASSES),
    **dict.fromkeys(SOFTMAX_OPS, _SOFTMAX_PASSES),
}

# Cycles a pass costs the vector unit whatever it holds: params, instruction
# issue and pipeline fill/drain.  Fixed per pass, so a tile too small to
# amortise it is dominated by overhead.  96 on every layer_norm and softmax
# pass the Sphinx RTL measured.
KERNEL_LAUNCH_OVERHEAD = 96

# Cycles a read stream loses each time consecutive requests move to another
# scratchpad bank.  The SoC's read masters take their responses in order, so
# the fabric holds a request to a new bank until every outstanding response
# from the previous one has returned -- one round trip, measured 8 cycles on
# the Sphinx SoC (requests resume once the in-flight beats drain).
BANK_SWITCH_CYCLES = 8


def _record_writes(record_bits: int, beat: int) -> float:
    """Write cycles per record of ``record_bits`` on a ``beat``-byte bus,
    averaged over the alignment cycle: one per beat-sized word, two where
    the record starts mid-beat and the bus splits the word."""
    nbytes = math.ceil(record_bits / 8)
    words = math.ceil(nbytes / beat)
    period = math.lcm(nbytes, beat) // nbytes
    split = sum(1 for g in range(period) if (g * nbytes) % beat)
    return words * (period + split) / period


def _input_bits(node, anchor) -> int:
    """Element width of the buffer the anchor's primary input is fetched
    from: through a fused dequantize prologue to the operand the call binds,
    so a packed input is sized as loaded, not as decoded."""
    src = anchor.args[0]
    if src.target is torch.ops.quantized_ops.dequantize.default:
        src = src.args[0]
    submodule = node.meta.get("submodule")
    if submodule is not None and src.graph is submodule.graph:
        src = bound_operands(node, submodule).get(src, src)
    bits = node_dtype_bits(src)
    return max(bits) if isinstance(bits, list) else bits


def _output_writes(node, lanes: int, beat: int) -> float:
    """Write cycles per lane group of ``node``'s outputs: the data output's
    record, plus one per block scale a ``quantize_mx`` tail writes beside
    it."""
    widths = node_dtype_bits(node)
    if not isinstance(widths, list):
        return _record_writes(lanes * widths, beat)
    values = getattr(node, "value", None)
    if values is None:
        values = node.meta.get("val")
    data = max(range(len(widths)), key=lambda i: values[i].numel())
    return _record_writes(lanes * widths[data], beat) + len(widths) - 1


def pool_taps(anchor) -> int:
    """Input lane groups a pool fetches per output group: its window's taps
    (test/toolchain/Pooling.h runs the fetch reduce_count = kH * kW times per
    output).  An adaptive pool's window is its input over its output."""
    if "adaptive" in str(anchor.target):
        return math.prod(anchor.args[0].shape) // math.prod(anchor.shape)
    kernel = get_arg_value(anchor, 1, "kernel_size")
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
    return kh * kw


def _pool_window(anchor, in_hw, out_hw):
    """``(kernel, stride, dilation)`` pairs of a pool whose input and output
    span ``in_hw`` and ``out_hw``.  An adaptive pool's window and stride are
    its input extent over its output's."""
    if "adaptive" in str(anchor.target):
        kernel = (in_hw[0] // out_hw[0], in_hw[1] // out_hw[1])
        return kernel, kernel, (1, 1)
    kernel = _pair(get_arg_value(anchor, 1, "kernel_size"))
    stride = get_arg_value(anchor, 2, "stride", [])
    stride = _pair(stride) if stride else kernel
    dilation = (1, 1)
    if "max_pool" in str(anchor.target):
        dilation = _pair(get_arg_value(anchor, 4, "dilation", 1))
    return kernel, stride, dilation


def _element_strides(shape):
    """Element strides of a contiguous tensor of ``shape``, per dimension."""
    strides = []
    for dim in range(len(shape)):
        strides.append(math.prod(shape[dim + 1 :]))
    return tuple(strides)


def _pool_fetch_ranges(tile, strides, out_hw, window, chunk, bits):
    """Byte ranges a pool tile's fetches read, in the order the vector unit
    issues them: output position, then channel chunk, then the window row by
    row (``get_pool2d_tiling`` in test/common/Tiling.cc).  ``tile`` and
    ``strides`` are per logical ``(N, C, H, W)`` axis, the strides in
    elements of the tile's own layout.  A chunk is one read where channels
    are innermost (NHWC, the layout the hardware runs); in any other layout
    its channels are read one by one."""
    batch, channels, _, _ = tile
    s_n, s_c, s_h, s_w = strides
    h_out, w_out = out_hw
    (kh, kw), (sh, sw), (dh, dw) = window
    for n, oy, ox, c0, ky, kx in itertools.product(
        range(batch),
        range(h_out),
        range(w_out),
        range(0, channels, chunk),
        range(kh),
        range(kw),
    ):
        y = oy * sh + ky * dh
        x = ox * sw + kx * dw
        first = n * s_n + c0 * s_c + y * s_h + x * s_w
        width = min(chunk, channels - c0)
        if s_c == 1:
            reads = [(first, width)]
        else:
            reads = [(first + c * s_c, 1) for c in range(width)]
        for element, count in reads:
            lo = element * bits // 8
            yield lo, lo + math.ceil(count * bits / 8)


@functools.lru_cache(maxsize=None)
def _pool_bank_switches(tile, strides, out_hw, window, chunk, bits, bank_size):
    """Bank switches of ``_pool_fetch_ranges``, cached: the tiling search
    and the report price the same tile geometries again and again."""
    ranges = _pool_fetch_ranges(tile, strides, out_hw, window, chunk, bits)
    switches, _, _ = bank_walk(ranges, bank_size)
    return switches


def pool_bank_switch_cycles(node, anchor, in_tile, config):
    """Cycles a pool tile loses to scratchpad bank switches.  The vector unit
    fetches ``config.accumulator_lanes`` channels at a time and walks a
    chunk's whole window before the next chunk (``_pool_fetch_ranges``), so a
    window that straddles a bank boundary switches banks twice per chunk.
    ``in_tile`` is the input tile in the op's layout -- NHWC for the layout
    twin, NCHW for the aten op -- and starts on a bank, where the planner
    puts it.  A switch costs the ``BANK_SWITCH_CYCLES`` round trip less the
    fetch's beats after its first, which the drain overlaps: 7 on Sphinx, as
    measured.  0 without banking."""
    if not config.bank_size:
        return 0
    nhwc = anchor.target in NHWC_OP_VARIANTS.values()
    dims = NCHW_TO_NHWC if nhwc else None
    tile = unproject(tuple(in_tile), dims)
    strides = unproject(_element_strides(in_tile), dims)
    out_hw = unproject(tuple(anchor.shape), dims)[2:]
    kernel, stride, dilation = _pool_window(anchor, tile[2:], out_hw)
    h_out = (tile[2] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    w_out = (tile[3] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    bits = _input_bits(node, anchor)
    chunk = config.accumulator_lanes
    switches = _pool_bank_switches(
        tile,
        strides,
        (h_out, w_out),
        (kernel, stride, dilation),
        chunk,
        bits,
        config.bank_size,
    )
    beat = max(1, round(config.bytes_per_cycle))
    fetch_beats = math.ceil(chunk * bits / 8 / beat)
    per_switch = max(1, BANK_SWITCH_CYCLES + 1 - fetch_beats)
    return per_switch * switches


def gemv_weight_bits(node, anchor) -> int:
    """Element width of the buffer a fully-connected GEMM streams its weight
    from (``weight_transforms``): a packed cache reaches the bank packed,
    whatever a fused prologue decodes it into."""
    submodule = node.meta.get("submodule")
    bound = bound_operands(node, submodule)
    weight = anchor.args[1]
    if submodule is not None and weight.graph is submodule.graph:
        weight = weight_transforms(weight)[0]
    bits = node_dtype_bits(bound.get(weight, weight))
    return max(bits) if isinstance(bits, list) else bits


def _logical_dtype(node):
    """The compiler's tracked storage dtype of ``node`` if it has one, else
    the dtype it was traced with."""
    dtype = node.meta.get("dtype")
    if dtype is not None:
        return dtype
    value = getattr(node, "value", None)
    if value is None:
        value = node.meta.get("val")
    return value.dtype


def gemv_lanes(anchor, config) -> int:
    """Elements per cycle of the unit a fully-connected GEMM streams through:
    the vector unit's lanes when its input is bfloat16 -- the toolchain maps
    such an op onto the vector pipeline -- else the matrix-vector unit's."""
    if str(_logical_dtype(anchor.args[0])).endswith("bfloat16"):
        return config.vector_lanes
    return config.matrix_vector_lanes


def bank_walk(ranges, bank_size):
    """Scratchpad bank switches a read stream makes walking ``ranges``, byte
    ``(lo, hi)`` pairs in fetch order: one whenever a range starts in a bank
    other than the one the previous range ended in, plus the boundaries a
    range straddles.  Returns ``(switches, first bank, last bank)``."""
    switches = 0
    first = last = None
    for lo, hi in ranges:
        lo_bank, hi_bank = int(lo // bank_size), int((hi - 1) // bank_size)
        switches += hi_bank - lo_bank
        if last is not None and lo_bank != last:
            switches += 1
        if first is None:
            first = lo_bank
        last = hi_bank
    return switches, first, last


@functools.lru_cache(maxsize=16384)
def strided_bank_walk(loops, width, bank_size, offset=0):
    """Bank walk of an affine request nest, without expanding every fetch.

    ``loops`` contains ``(count, byte_stride)`` pairs, outermost first;
    each leaf fetches ``width`` bytes. Preserve rewinds, broadcasts and
    strides that skip banks. A monotone nest can collapse to one range
    only when consecutive requests cannot skip an entire bank. Cache the
    summaries because tiling candidates repeatedly use the same geometry.
    """
    span = width
    monotone = True
    for count, stride in reversed(loops):
        if count > 1:
            monotone &= span <= stride < span + bank_size
            span += (count - 1) * stride
    first = int(offset // bank_size)
    last = int((offset + span - 1) // bank_size)
    if first == last or monotone:
        return last - first, first, last

    count, stride = loops[0]
    if stride == 0:
        switches, first, last = strided_bank_walk(
            loops[1:], width, bank_size, offset
        )
        return count * switches + (count - 1) * (first != last), first, last

    switches = 0
    first = last = None
    for i in range(count):
        inner, start, end = strided_bank_walk(
            loops[1:], width, bank_size, offset + i * stride
        )
        switches += inner + (last is not None and last != start)
        if first is None:
            first = start
        last = end
    return switches, first, last


def gemv_bank_switch_cycles(rows, reduction, weight_bits, chunk, config):
    """Cycles a matrix-vector tile loses to scratchpad bank switches,
    ``BANK_SWITCH_CYCLES`` each.  The unit walks one ``chunk``-element piece
    of every row of a ``pe_array_size[0]``-row block before the next chunk,
    so when the row stride of a ``reduction``-wide tile spreads a block's
    rows over several banks, every chunk column crosses them again.  The
    tile buffer starts on a bank, where the planner puts it.  0 without
    banking."""
    if not config.bank_size:
        return 0
    block = config.pe_array_size[0]
    row_bytes = reduction * weight_bits / 8
    chunk_bytes = chunk * weight_bits / 8
    ranges = []
    for k in range(0, rows, block):
        for c in range(math.ceil(reduction / chunk)):
            for r in range(k, min(k + block, rows)):
                lo = r * row_bytes + c * chunk_bytes
                ranges.append((lo, lo + chunk_bytes))
    switches, _, _ = bank_walk(ranges, config.bank_size)
    return BANK_SWITCH_CYCLES * switches


def vector_op_utilization(node, config, ideal_cycles=None, tile=None):
    """Fraction of peak a vector ``node`` sustains, bound by the bus.

    Peak is one ``config.vector_lanes``-wide lane group per cycle.  Each
    pass over the data (``OP_PASSES``; one for everything else) costs a lane
    group the greater of its read cycles, the bus beats its widest streamed
    operand is fetched in, and its write cycles, at ``config.bytes_per_cycle``
    per beat: a reduction pass writes one value per row, so it runs at the
    read rate, and a record that ends mid-beat is split by the bus, so a
    sub-byte output can take more write cycles than read cycles.  A
    fully-connected GEMM streams its weight once per output through the
    unit ``gemv_lanes`` names, a lane group being that unit's width, and is
    sized by the buffer the kernel loads (``weight_transforms``), not by
    what a fused prologue decodes it into.  A pool fetches one lane group
    per window tap, the taps being its callers' work, and its output leaves
    on a port of its own behind the fetch.  ``tile`` is the tile being
    priced, which ``node`` cannot name while a tiling is still being
    searched, and prices the scratchpad bank switches of the op's stream: a
    fully-connected GEMM's ``(rows, reduction)`` weight tile
    (``gemv_bank_switch_cycles``), a pool's input tile
    (``pool_bank_switch_cycles``); other ops ignore it.  Given
    ``ideal_cycles``, the per-pass ``KERNEL_LAUNCH_OVERHEAD`` and the
    switches are folded in so ``ideal_cycles / result`` is the tile's whole
    cost.  The single copy of the formula:
    ``reporting/cost.op_utilization`` calls it for its vector branch.
    """
    anchor = get_anchor_node(node) or node
    lanes = config.vector_lanes
    bytes_per_cycle = config.bytes_per_cycle
    beat = max(1, round(bytes_per_cycle))
    switch_cycles = 0
    cycles_per_group = 0.0
    if is_fully_connected(anchor):
        lanes = gemv_lanes(anchor, config)
        reads = {IN: gemv_weight_bits(node, anchor)}
        profile = [(IN, None)]
        rows, reduction = tile
        switch_cycles = gemv_bank_switch_cycles(
            rows, reduction, reads[IN], lanes, config
        )
    elif is_pooling(anchor):
        # One lane group fetched per window tap (test/toolchain/Pooling.h);
        # the output is written on another port while the fetch runs.
        profile = []
        cycles_per_group = math.ceil(
            lanes * _input_bits(node, anchor) / 8 / bytes_per_cycle
        )
        if tile is not None:
            switch_cycles = pool_bank_switch_cycles(node, anchor, tile, config)
    elif anchor.target in OP_PASSES:
        profile = OP_PASSES[anchor.target]
        # ``native_layer_norm`` also returns its mean and rstd: the staged
        # intermediate is its normalized output, the first.
        mid = node_dtype_bits(anchor)
        reads = {
            IN: _input_bits(node, anchor),
            MID: mid[0] if isinstance(mid, list) else mid,
        }
    else:
        widths = [
            node_dtype_bits(n)
            for n in node.all_input_nodes
            if require_allocation(n)
        ]
        reads = {IN: max(widths, default=16)}
        profile = [(IN, OUT)]
    for read, write in profile:
        read_cycles = math.ceil(lanes * reads[read] / 8 / bytes_per_cycle)
        if write is None:
            write_cycles = 0.0
        elif write is MID:
            write_cycles = _record_writes(lanes * reads[MID], beat)
        else:
            write_cycles = _output_writes(node, lanes, beat)
        cycles_per_group += max(read_cycles, write_cycles)
    util = min(1.0, 1.0 / cycles_per_group)
    if not ideal_cycles:
        return util
    overhead = max(1, len(profile)) * KERNEL_LAUNCH_OVERHEAD + switch_cycles
    return ideal_cycles / (ideal_cycles / util + overhead)


def _tile_elems(shape):
    """Element count of a tile shape (largest output of a multi-output node)."""
    if shape and isinstance(shape[0], (tuple, list)):
        return max(math.prod(s) for s in shape)
    return math.prod(shape)


def transfer_cost(shape, n_bytes, lat, bpc):
    """Cycles one transfer of a tiled operand costs: an access latency per
    tensor -- a ``quantize_mx`` pair is two, its tile of scales paying a whole
    latency for a few hundred bytes -- plus the bytes at the DRAM rate."""
    multi = bool(shape) and isinstance(shape[0], (tuple, list))
    return (len(shape) if multi else 1) * lat + n_bytes / bpc


def vector_tile_latency(node, tiled_shapes, tiling, config):
    """Latency and DRAM traffic of a vector op under a candidate tiling.

    The grid is the tile counts over ``node``'s own output, and each input's
    dims right-align onto it, the way the tile shapes were built
    (``compute_tiled_shape``), or follow ``reduced_input_grid`` for a
    reduction that drops its reduced dims -- so a dim an operand lacks, or
    broadcasts over, leaves its tile in place while that loop turns and
    ``block_transfers`` prices the reuse.  A tile costs its element count
    spread over the lanes and de-rated by ``vector_op_utilization``, sized by
    the larger of the output tile and the widest input tile.  Double-buffered
    the two engines overlap (``sweep_cycles``); single-buffered load, compute
    and store run back to back.

    Args:
        node: The op to cost -- a vector op, or a fused ``call_module`` around
            one.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated activation input (resident params are absent).
        tiling: Per-dim tile *count* over ``node``'s output.
        config (AcceleratorConfig): The hardware description.

    Returns:
        ``(cycles, DRAM bytes)``; the bytes break a latency tie (``pick``).
    """
    num_tiles = math.prod(tiling)

    out_shape = tiled_shapes[node]

    in_elems = max(
        (_tile_elems(s) for n, s in tiled_shapes.items() if n is not node),
        default=0,
    )
    anchor = get_anchor_node(node) or node
    tile = None
    if is_pooling(anchor):
        tile_ops = _tile_elems(out_shape) * pool_taps(anchor)
        # The halo the kernel loads, keyed the way ``pool_operand_shapes``
        # keys it.
        bound = bound_operands(node, node.meta.get("submodule"))
        halo = bound.get(anchor.args[0], anchor.args[0])
        tile = tiled_shapes.get(halo)
    else:
        tile_ops = max(_tile_elems(out_shape), in_elems)
    lanes = config.vector_lanes
    bpc = config.bytes_per_cycle
    ideal = math.ceil(tile_ops / lanes)
    util = vector_op_utilization(node, config, ideal, tile)
    compute = math.ceil(ideal / util)

    lat = config.access_latency_cycles
    dmas, traffic = [], 0
    for n, shape, transfers in vector_transfers(node, tiled_shapes, tiling):
        n_bytes = operand_bytes(shape, n)
        dmas.append((transfer_cost(shape, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes

    if config.double_buffered_l2:
        latency = sweep_cycles(dmas, num_tiles, compute)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += num_tiles * compute
    return latency, traffic


def gemv_compute_cycles(node, tile_sizes, config):
    """The vector unit's cycles for one ``(X, C, K)`` tile of a
    matrix-vector GEMM: its MACs over the unit's lanes (``gemv_lanes``),
    de-rated by ``vector_op_utilization``."""
    lanes = gemv_lanes(get_anchor_node(node), config)
    ideal = math.ceil(math.prod(tile_sizes) / lanes)
    _, c_tile, k_tile = tile_sizes
    util = vector_op_utilization(node, config, ideal, (k_tile, c_tile))
    return math.ceil(ideal / util)


def gemv_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Latency, DRAM traffic and one tile's compute of a matrix-vector GEMM
    under a tiling.

    ``tiling`` covers ``(X, C, K)``, but the kernel loops more: the anchor's
    leading batch dims (attention heads) are one grid step each, outside all
    three, and ``build_gemm`` emits them outermost.  Which grid dims dice an
    operand follows from its role (``GEMV_GRID_DIMS``), and that is what
    ``block_transfers`` needs to price its reuse -- a GQA KV tile shared by
    four query heads is re-read once per head as soon as a dim inner to the
    head loop splits.

    A tile costs its MACs spread over the width of the unit the GEMV runs
    on (``gemv_lanes``) and de-rated by ``vector_op_utilization``, the
    weight stream's bank switches included -- the same charge
    ``reporting/cost.op_info`` gives the op, so the two models cannot
    disagree.  Double-buffered the
    DRAM engine and the vector unit overlap (``sweep_cycles``);
    single-buffered they run back to back.

    Args:
        node: The op to cost -- a fully-connected GEMM, or a fused
            ``call_module`` around one.
        tile_sizes: The ``(X, C, K)`` tile; its MAC count sizes the compute.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated activation input.
        tiling: ``(n_x, n_c, n_k)`` -- tile counts over ``(X, C, K)``, ``n_c``
            being the reduction.
        config (AcceleratorConfig): The hardware description.

    Returns:
        ``(cycles, DRAM bytes, compute)``: the bytes break a latency tie
        (``pick``), and ``compute`` is one tile's ``gemv_compute_cycles``,
        what an attention product charges for a step.
    """
    grid, moves = gemv_transfers(node, tiled_shapes, tiling)
    lat = config.access_latency_cycles
    bpc = config.bytes_per_cycle
    dmas, traffic = [], 0
    for n, shape, transfers in moves:
        # Bytes come off the loaded buffer (its packed dtype).
        n_bytes = operand_bytes(shape, n)
        dmas.append((transfer_cost(shape, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes

    compute = gemv_compute_cycles(node, tile_sizes, config)

    steps = math.prod(grid)
    if config.double_buffered_l2:
        latency = sweep_cycles(dmas, steps, compute)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += steps * compute
    return latency, traffic, compute


# One bank per operand *group*, grouped by the read bandwidth each needs.  A
# GEMV streams its weight -- a fresh element per MAC, nothing reused -- so it
# takes a bank of its own.  The input and its scales are reused across
# ``vector_lanes`` outputs, so one fetch feeds a whole lane group and they need
# a fraction of the weight's bandwidth: nothing like a GEMM, where the input
# streams too.  The weight scales do matter, one per block, but the output is
# only written on the step that finishes a tile, so the two rarely contend.
GEMV_BANK_GROUPS = (
    ("input", "input_scale"),
    ("weight", "other"),
    ("output", "weight_scale", "bias"),
    ("A_data", "A_indices", "A_indptr"),
)


def tensor_bytes(node, shape, config):
    """Bytes one tile of ``node`` occupies on chip, each output sized by
    ``tensor_alloc_bytes`` under ``config``'s bank width and vector lanes --
    so a tile the search accepts is exactly what ``plan_memory`` will
    allocate for it.

    Args:
        node: The operand the tile belongs to.
        shape: Its tiled shape -- one per output for a multi-output op.
        config (AcceleratorConfig): The hardware description.
    """
    lanes = config.vector_lanes
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get("dtype") or val.dtype
        return tensor_alloc_bytes(
            math.prod(shape), dtype, config.bank_width, lanes
        )

    if isinstance(val, (tuple, list)):
        numel = [math.prod(s) for s in shape]
        dtypes = node.meta.get("dtype") or [None for _ in val]
        sizes = [
            tensor_alloc_bytes(n, dt or t.dtype, config.bank_width, lanes)
            for t, n, dt in zip(val, numel, dtypes)
        ]
        return sum(sizes)

    logger.warning(f"Node {node} has a non-tensor output")
    return None


class VoyagerVector(OpModel):
    """A vector op on Voyager's vector unit -- elementwise, a reduction, a
    relayout -- bare or a fused ``call_module``: its operand tiles packed
    into scratchpad banks (``fit``), priced by ``vector_tile_latency``.
    Searched in the parent, so it keeps the node.

    Attributes:
        node: The op.
        space: Its ``GridSpace``.
        key: ``None``: a grid search shares no result.
        config (AcceleratorConfig): The hardware description.
    """

    def __init__(self, node, space, config):
        self.node = node
        self.space = space
        self.key = None
        self.config = config
        self.anchor = get_anchor_node(node)
        # The role each loaded buffer plays for a GEMM anchor, which picks its
        # bank (``GEMV_BANK_GROUPS``).
        self._roles = {n: role for n, (role, _) in operand_roles(node).items()}
        # Each buffer's member in a placement: an input by its position, and
        # the output.
        self._members = {
            n: KernelInput(i) for i, n in enumerate(node.all_input_nodes)
        }
        self._members[node] = Role.OUTPUT
        # Whether a buffer is allocated walks its users' graphs, so each is
        # asked once.
        self._allocated = functools.cache(require_allocation)
        # A reduction to one value keeps each tile's result beside the running
        # one (``_running_max_kernel``).
        self._running_bytes = 0
        if is_full_reduction(self.anchor):
            self._running_bytes = tensor_alloc_bytes(
                1,
                self.anchor.value.dtype,
                config.bank_width,
                config.vector_lanes,
            )

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``vector_operand_shapes``)."""
        return vector_operand_shapes(self.node, tiling)

    def fit(self, tiling):
        """The bank partition of ``tiling``'s tile, at the least extra
        sharing that lets one slot of it fit a ``num_slots`` share of the
        scratchpad, else ``None``.

        A ping-ponged buffer holds one slot in each of its ``num_slots``
        banks, so a slot is checked against ``usable_scratchpad_size //
        num_slots``.  Every group takes whole banks, since a bank cannot be
        split, and the two smallest merge while the groups outnumber the
        banks a slot may spread over.  Each tier merges one group further,
        up to ``len(GEMV_BANK_GROUPS) - 1``, lowering the floor ``G`` groups
        put on the footprint, ``G * bank_size``, at which no tile fits
        however small; a merge never adds a bank, so a tile that fits at a
        tier fits at every later one.  Without banks nothing merges.

        Returns:
            Each bank's members (``_bank_groups``) at the lowest tier that
            fits, with no groups without banks; ``None`` when no tier fits.
        """
        config = self.config
        sized = self._bank_groups(self.operand_shapes(tiling))
        budget = config.usable_scratchpad_size // config.num_slots
        if not config.bank_size:
            needed = sum(size for size, _ in sized)
            return Placement() if needed <= budget else None
        target = config.usable_banks // config.num_slots
        for tier in range(len(GEMV_BANK_GROUPS)):
            while len(sized) > max(1, target - tier):
                sized.sort(key=lambda group: group[0])
                (size0, members0), (size1, members1) = sized[0], sized[1]
                sized = [(size0 + size1, members0 + members1)] + sized[2:]
            needed = sum(
                math.ceil(size / config.bank_size) * config.bank_size
                for size, _ in sized
            )
            if needed <= budget:
                groups = tuple(frozenset(members) for _, members in sized)
                return Placement(groups, tier=tier)
        return None

    def _bank_groups(self, shapes):
        """The banks one candidate tile takes before any merge: ``(bytes,
        members)`` per bank.

        The layout is ``GEMV_BANK_GROUPS`` by the role a buffer plays for a
        GEMM anchor, then a bank of its own for each buffer those do not
        name -- a ``where`` lays out as ``("input",) | ("other", ...) |
        ("output", ...) | ("condition",)`` -- then one shared by the regions
        no FX node names (``_reserved_bytes``).  Each buffer is sized the way
        the planner will allocate it (``tensor_bytes``), and a bank holding
        nothing allocated is left out.  A buffer that is not the kernel's
        input or output names no member, though its bytes count.

        Args:
            shapes: Each buffer's tile (``operand_shapes``), ``self.node``
                standing for the output.
        """
        node, config = self.node, self.config
        groups = [
            [n for n in shapes if self._roles.get(n) in group]
            for group in GEMV_BANK_GROUPS
        ]
        grouped = {n for group in groups for n in group}
        groups += [[n] for n in shapes if n not in grouped]

        sized = []
        for group in groups:
            total = 0
            for n in group:
                shape = shapes[n]
                if shape is None or (n is not node and not self._allocated(n)):
                    continue
                total += tensor_bytes(n, shape, config)
            if total:
                members = [
                    self._members[n] for n in group if n in self._members
                ]
                sized.append((total, members))

        # No DMA moves the reserved regions, so they share one bank rather
        # than taking one apiece.
        reserved = self._reserved_bytes(shapes)
        if reserved:
            sized.append((reserved, [Role.SCRATCH]))
        return sized

    def _reserved_bytes(self, shapes):
        """Bytes of the regions the kernel keeps beside its buffers that no
        FX node names: a batched reduction's intermediates
        (``reduction_scratch``) and a reduction to one value's running
        result."""
        config = self.config
        reserved = sum(
            tensor_alloc_bytes(
                math.prod(shape), dtype, config.bank_width, config.vector_lanes
            )
            for _, shape, dtype in reduction_scratch(
                self.node, shapes[self.node], config.vector_lanes
            )
        )
        return reserved + self._running_bytes

    def cost(self, tiling, placement):
        """``vector_tile_latency`` of ``tiling``; ``placement`` is the
        contract's."""
        latency, traffic = vector_tile_latency(
            self.node,
            self.operand_shapes(tiling),
            tiling.grid,
            self.config,
        )
        return Cost(latency, traffic)


class VoyagerGemv(VoyagerVector):
    """A matrix-vector GEMM (one row per batch element) on Voyager's vector
    unit, bare or fused: an ``(X, C, K)`` tile grid packed into banks like a
    vector op's, plus the accumulator a fused tail reduces into, and priced
    by ``gemv_tile_latency``."""

    def __init__(self, node, space, config):
        super().__init__(node, space, config)
        sub_gm = node.meta.get("submodule")
        ops = []
        if sub_gm is not None:
            ops = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        # A submodule holding only the anchor's prologue leaves the builder
        # no tail to stage.
        self._has_tail = bool(ops) and ops[-1] is not self.anchor
        self._input_buffers = [
            n for n, role in self._roles.items() if role == "input"
        ]
        self._reduction_size = self.anchor.args[0].shape[-1]
        # A quantize that breaks the stream, or a pipeline with no stage left
        # for the tail, stages the finished tile even for a single round.
        breaks_stream = stream_breaking_quantize(sub_gm) is not None
        fusible = node.meta.get("single_k_tail_fusible", True)
        self._staged = breaks_stream or not fusible

    def _reserved_bytes(self, shapes):
        """``VoyagerVector._reserved_bytes`` plus the accumulator a fused
        tail reduces into, the one ``_ScratchSpec`` the builder allocates for
        the whole kernel when the reduction is split across K, and when a
        stream-breaking tail stages the finished tile even for a single round
        (``_gemm_scratch_and_kernel``).  It is sized off the shape-preserving
        output tile -- a ``quantize_mx`` tail returns ``(scale, value)`` --
        in the anchor's own dtype, the partial sum it holds."""
        reserved = super()._reserved_bytes(shapes)
        if not self._has_tail:
            return reserved
        split_k = any(
            n in shapes and shapes[n][-1] < self._reduction_size
            for n in self._input_buffers
        )
        if not (split_k or self._staged):
            return reserved
        out_tile = shapes[self.node]
        if isinstance(out_tile[0], (tuple, list)):
            out_tile = out_tile[-1]
        return reserved + tensor_alloc_bytes(
            math.prod(out_tile),
            self.anchor.value.dtype,
            self.config.bank_width,
            self.config.vector_lanes,
        )

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``gemv_operand_shapes``)."""
        return gemv_operand_shapes(self.node, tiling)

    def cost(self, tiling, placement):
        """``gemv_tile_latency`` of ``tiling``, its ``compute`` the one
        tile's ``gemv_compute_cycles``; ``placement`` is the contract's."""
        return Cost(
            *gemv_tile_latency(
                self.node,
                tiling.tile_sizes,
                self.operand_shapes(tiling),
                tiling.grid,
                self.config,
            )
        )


class VoyagerPool(VoyagerVector):
    """A max or adaptive pool on Voyager's vector unit, bare or fused: its
    halo and output tiles (``pool_operand_shapes``) packed into banks like a
    vector op's.  Its space is unranked, so the largest tile that fits wins
    unpriced; ``vector_tile_latency`` would price a pool's halo."""

    def operand_shapes(self, tiling):
        """Each buffer the kernel holds -> its tile under ``tiling``
        (``pool_operand_shapes``)."""
        return pool_operand_shapes(self.node, tiling)
