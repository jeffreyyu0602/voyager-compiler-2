"""Trainium's runtime and fit model, ``TrainiumModel``.

What the shared tile search asks about one NeuronCore, answered from its
``TrainiumConfig``.  A tile lives in SBUF with its rows on the partitions and
its last dim free.  ``TrainiumModel.prepare`` hands each op to its op class,
whose fit places the kernel's buffers and whose cost prices its steps the way
the builders and the translator run them: ``TrainiumGemm`` and
``TrainiumGemv`` (``gemm.py``), ``TrainiumAttention`` (``attention.py``),
``TrainiumVector`` and ``TrainiumPool`` (``vector.py``).  The model prices
what they share:

- one VectorE or ScalarE instruction, an op's passes over a tile and a
  ``quantize_mx``, which transposes its tile on TensorE 128 columns at a
  time before quantizing each chunk;
- a DMA, which moves a tile in as few ``dma_copy`` instructions as access
  patterns walk (``_dma_count``), a load of ``STREAM_BYTES`` or more in one
  per piece of its 128-row blocks (``stream_blocks``): each copy the longer
  of its issue and its rows' bytes, packet overhead included, at the core's
  DMA bandwidth -- half of it for a copy within half the partitions --
  plus the rings' time between copies; a store issues from ScalarE's queue;
  a vector op's tile lies as the translator lays it out
  (``TrainiumVector``);
- an MX (microscaling) tile, stored contraction-major: 4 consecutive
  contraction elements per 32-bit word, a 512-element chunk per 128
  partitions, chunks side by side on the free dim, and one E8M0 scale byte
  per 32 elements in a tile of the same layout, filled only on each
  quadrant's first 4 partitions;
- a sweep: the engines run concurrently, so a step costs its busiest
  engine, and its slots' refills overlap the compute around it, each with
  its first copy's issue to spare (``sweep_ns``); a kernel's first loads
  overlap the kernel before it unless they are exposed
  (``gemm.first_load_exposed``); a GEMV's one-row tiles move beside its
  weight stream, and its last step's compute runs behind the last piece
  of its weight tile (``TrainiumGemv.cost``).

Times are nanoseconds here and TensorE cycles to the search.
"""

import math
from collections import Counter

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    is_conv2d,
    is_fully_connected,
    is_gemm_op,
    is_pooling,
    reads_kmajor,
)

# The bufferizer's own rules for which nodes become kernels and which reuse
# another's build.
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _bufferize_key,
    _is_compute,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    AttentionSpace,
    GridSpace,
    MatrixSpace,
    RuntimeModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import step_classes

from .attention import TrainiumAttention
from .gemm import TrainiumGemm, TrainiumGemv
from .utils import (
    HOISTED_SBUF,
    MX_PACK,
    QUADRANT,
    QUADRANT_SCALES,
    extents,
    output_buffers,
    rows_cols,
    sbuf_bytes,
)
from .vector import TrainiumPool, TrainiumVector

MAX_POOL2D = torch.ops.aten.max_pool2d.default


# A load of at least ``STREAM_BYTES`` moves in pieces of whole 128-row
# blocks: each piece the fewest blocks that hold ``PIECE_BYTES``, the last
# one the blocks left over, so an op reading a block waits for its piece
# alone; a shorter load lasts about a transfer's fixed latency.
STREAM_BYTES = 1 << 20
PIECE_BYTES = 256 << 10


def stream_blocks(blocks, block_bytes):
    """The 128-row blocks, ``block_bytes`` each, that one ``dma_copy`` of a
    load of ``blocks`` moves: all of them, or for a load of
    ``STREAM_BYTES`` or more the fewest that hold ``PIECE_BYTES``."""
    if blocks * block_bytes >= STREAM_BYTES:
        return math.ceil(PIECE_BYTES / block_bytes)
    return blocks


def _dma_count(tile, full, partitions):
    """``dma_copy`` instructions the translator moves a ``tile`` of a
    tensor ``full`` in: one when its rows lie evenly apart across every
    partition and row block, else one per run of evenly spaced rows -- rows
    along the second-to-last dim, continuing into the dim above while the
    dims below it are whole -- or per partition block."""
    rows = math.prod(tile[:-1])
    run = tile[-2] if len(tile) > 1 else 1
    full = (1,) * (len(tile) - len(full)) + tuple(full)[-len(tile) :]
    for d in range(len(tile) - 2, 0, -1):
        if tile[d] != full[d]:
            break
        run *= tile[d - 1]
    if run >= rows or run % partitions == 0:
        return 1
    if run < partitions:
        return math.ceil(rows / run)
    return math.ceil(rows / partitions)


class TrainiumModel(RuntimeModel):
    """One NeuronCore: TensorE, VectorE and ScalarE beside SBUF and PSUM,
    and DMA to HBM.

    NKI places SBUF and PSUM buffers itself, so a tiling that fits takes a
    ``TrainiumPlacement`` of the buffers its kernel computes in, at no
    address.

    Attributes:
        matrix_space: How a GEMM is mapped: a 128 x 128 array, SBUF as one
            byte pool and HBM.  L1 holds no tile of its own -- the GEMM
            builder cuts each step into instructions itself -- so its
            blockings are pinned to 1 and the search maps the PE partition,
            the SBUF tile and the HBM grid.  The reduction stays innermost at
            L3, where the builders accumulate it.
        grid_space: How a vector op or a matrix-vector GEMM is cut: lanes of
            1, since a tile's rows go on the partitions and its last dim is
            free, and every tiling that fits is priced.
        attention_space: How a flash attention is cut: query and key tiles
            in whole partitions, the head unpadded, since it lies on the
            partitions QKᵀ contracts over or is P·V's moving free dim.

    Both weigh a near-tie in latency against bytes share for share
    (``relative_ties``): a step pays a kernel's latency, so a slower tiling
    wins only by saving a larger share of bytes than it adds of time.
    """

    def __init__(self, config, accumulate_fp32):
        super().__init__(config, accumulate_fp32)
        p, q = config.pe_array_size
        architecture = interstellar.Resource(
            buf_capacity_list=[
                [1, 1, 1],
                # Unbounded: L1 only passes the PE tile through.
                [math.inf, math.inf, math.inf],
                [config.usable_scratchpad_size],
                [config.dram_size * 1024**3],  # GB -> bytes
            ],
            buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100], [1000]],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            para_count_list=[p * q, 1, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            mac_capacity=0,
            partition_mode=[0, 0, 0, 0],
            invalid_underutilized=False,
            bank_size_list=[None, None, None, None],
        )
        unit = {"blocking_size": 1, "partitioning_size": 1}
        hints = {
            loop: {"level1": dict(unit)} for loop in ("OX", "OY", "OC", "ON")
        }
        hints["IC"] = {
            "level0": {"order": 1, "partitioning_size": p},
            "level1": dict(unit),
            "level3": {"order": 0},
        }
        hints["OC"]["level0"] = {"order": 0, "partitioning_size": q}
        for loop in ("FX", "FY"):
            hints[loop] = {f"level{i}": dict(unit) for i in range(4)}
        schedule_data = interstellar.extract_input.extract_schedule_info(
            {"schedule_hint": hints}, 4
        )
        schedule = interstellar.Schedule(
            schedule_data["schedule_hint"], schedule_data["partition_loops"]
        )
        self.matrix_space = MatrixSpace(architecture, schedule)
        self.grid_space = GridSpace(lanes=1, ranked=True, relative_ties=True)
        self.attention_space = AttentionSpace(
            unit=config.partitions, head_multiple=1, relative_ties=True
        )

    def prepare(self, node):
        """The model of ``node``, bare or fused: a ``TrainiumGemm`` for a
        GEMM, a ``TrainiumGemv`` for a matrix-vector one, which the grid
        search tiles, a ``TrainiumPool`` for an NCHW 2-D max pool and a
        ``TrainiumVector`` for a vector op.

        Raises:
            NotImplementedError: ``node`` is a convolution or another pool,
                which do not lower to Trainium.
        """
        anchor = get_anchor_node(node)
        if anchor.target is MAX_POOL2D:
            return TrainiumPool(node, self)
        if is_conv2d(anchor) or is_pooling(anchor):
            raise NotImplementedError(f"{anchor}: not lowered on Trainium")
        if not is_gemm_op(anchor):
            return TrainiumVector(node, self)
        if is_fully_connected(anchor):
            return TrainiumGemv(node, self)
        return TrainiumGemm.from_node(node, self)

    def prepare_attention(self, node, acc_dtype, bool_mask):
        """A ``TrainiumAttention`` for the flash-attention ``node``.

        Raises:
            NotImplementedError: ``node`` is not K-major (``sdpa_kmajor``):
                a row-major or MX attention does not lower to Trainium.
        """
        if not reads_kmajor(node):
            raise NotImplementedError(f"{node}: not lowered on Trainium")
        return TrainiumAttention(node, self, acc_dtype, bool_mask)

    def engine_ns(self, engine, elements, bits):
        """One instruction on ``engine`` over ``elements`` per partition, its
        widest operand ``bits`` wide."""
        config = self.config
        if engine == "scalar":
            rate, ghz = config.scalar_elements, config.scalar_ghz
            fixed = config.scalar_overhead_ns
        else:
            rate = (
                config.vector_elements_16
                if bits <= 16
                else config.vector_elements_32
            )
            ghz, fixed = config.vector_ghz, config.vector_overhead_ns
        return math.ceil(elements / rate) / ghz + fixed

    def dma_ns(self, rows, cols, bits, count=1, load=False):
        """A ``rows x cols`` tile of ``bits``-wide elements moved in
        ``count`` ``dma_copy`` instructions; a ``load`` moved in one is cut
        into one per piece of its 128-row blocks (``stream_blocks``).  The
        rings move a copy's rows while the next copy issues, so each copy
        takes the longer of its issue and its rows' bytes, packet overhead
        included, at the core's DMA bandwidth -- half of it for a copy on
        ``dma_half_rate_partitions`` partitions or fewer -- plus the rings'
        time between copies: ``dma_copy_ns``, and ``dma_copy_idle_ns`` for a
        copy no longer than its issue, falling off in proportion as it is
        longer."""
        config = self.config
        row_bytes = math.ceil(cols * bits / 8)
        parts = min(rows, config.partitions)
        if load and count == 1:
            blocks = math.ceil(rows / parts)
            count = math.ceil(blocks / stream_blocks(blocks, parts * row_bytes))
        bandwidth = config.dram_bandwidth
        if parts <= config.dma_half_rate_partitions:
            bandwidth /= 2
        moved = rows * (row_bytes + config.dma_row_overhead_bytes)
        issue = config.dma_issue_ns
        copy = moved / bandwidth / count
        idle = config.dma_copy_idle_ns * min(1.0, issue / copy)
        return count * max(issue, copy + config.dma_copy_ns + idle)

    def store_issue_ns(self, stores, steps):
        """ScalarE's time per step issuing ``stores`` output stores over a
        sweep of ``steps`` steps: the step compile issues a store's
        ``dma_copy`` from ScalarE's queue, ``dma_store_issue_ns`` each,
        while a load's issue takes a queue of its own (``dma_ns``).  The
        last store's issue is the sweep's end (``sweep_ns``)."""
        return self.config.dma_store_issue_ns * (stores - 1) / steps

    def _mx_layout(self, depth, other):
        """``(partitions, words)`` of an MX tile contracting ``depth`` beside
        ``other`` elements: a word per 4 contraction elements, the first 512
        on the partitions and every further chunk of 512 beside them."""
        p = self.config.partitions
        chunks = math.ceil(depth / (p * MX_PACK))
        return min(p, math.ceil(depth / MX_PACK)), chunks * other

    def mx_sbuf(self, depth, other, scales):
        """SBUF bytes of an MX codes tile, or of its ``scales``: one byte per
        word's place, though only each quadrant's first 4 partitions hold
        one."""
        partitions, words = self._mx_layout(depth, other)
        bits = 8 if scales else 32
        return sbuf_bytes(partitions, words, bits, self.config.partitions)

    def mx_dma_ns(self, depth, other, scales):
        """One transfer of an MX codes tile in one ``dma_copy``, or of its
        ``scales`` in one per quadrant."""
        partitions, words = self._mx_layout(depth, other)
        if not scales:
            return self.dma_ns(partitions, words, 32)
        quadrants = math.ceil(partitions / QUADRANT)
        return quadrants * self.dma_ns(QUADRANT_SCALES, words, 8)

    def buffer_sbuf(self, n, shape, mx, parts=1, spread=False):
        """SBUF bytes of buffer ``n``'s tile ``shape``: one array per output,
        rows on the partitions -- cut evenly across ``parts`` of them when
        ``spread`` (``TrainiumVector.row_parts``) -- or the MX layout ``mx``
        (``TrainiumGrid._mx_tiles``) gives it."""
        if n in mx:
            return sum(self.mx_sbuf(*array) for array in mx[n])
        p = self.config.partitions
        total = 0
        for s, bits in output_buffers(shape, n):
            rows, cols = rows_cols(s)
            if spread:
                rows, cols = parts, rows * cols // parts
            total += sbuf_bytes(rows, cols, bits, p)
        return total

    def buffer_dma_ns(self, n, shape, mx, load, parts=1, spread=False):
        """One transfer of buffer ``n``'s tile ``shape``, a ``load`` or a
        store: each output's ``dma_copy`` instructions (``_dma_count``), or
        the MX layout's transfers.  A tile lies on ``parts`` partitions, as
        the translator lays out a vector op's (``TrainiumVector.row_parts``):
        cut evenly across them in one ``dma_copy`` when ``spread``, else a
        load of fewer rows copied onto each: one row in one ``dma_copy``,
        several loaded once per copy."""
        if n in mx:
            return sum(self.mx_dma_ns(*array) for array in mx[n])
        values = n.value if isinstance(n.value, (tuple, list)) else [n.value]
        p = self.config.partitions
        total = 0.0
        for (s, bits), value in zip(output_buffers(shape, n), values):
            rows, cols = rows_cols(s)
            count = _dma_count(tuple(s), value.shape, p)
            copies = 1
            if spread:
                rows, cols, count = parts, rows * cols // parts, 1
            elif rows == 1 and load:
                rows = parts
            elif rows < parts and load:
                copies = parts // rows
            total += copies * self.dma_ns(rows, cols, bits, count, load)
        return total

    def quantize_mx_busy(self, rows, cols):
        """Each engine's busy time quantizing a ``rows x cols`` tile, rows on
        the partitions, to MX along its columns: per 128 rows and 512
        columns, TensorE transposes every 4th column into PSUM 4 times,
        VectorE copies each out into a staging tile and runs one
        ``quantize_mx`` over it."""
        config = self.config
        overhead = config.tensor_overhead_cycles
        chunks = math.ceil(cols / (config.partitions * MX_PACK))
        busy = Counter()
        for r in extents(rows, config.partitions):
            transposes = MX_PACK * chunks
            busy["tensor"] += transposes * (r + overhead) / config.frequency
            busy["vector"] += transposes * self.engine_ns("vector", r, 16)
            busy["vector"] += chunks * self.engine_ns("vector", MX_PACK * r, 16)
        return busy

    def passes_busy(self, passes, rows, cols):
        """Each engine's busy time making ``passes`` -- ``(engine, share,
        bits)`` -- over a ``rows x cols`` tile, 128 rows per instruction."""
        row_chunks = math.ceil(rows / self.config.partitions)
        busy = Counter()
        for engine, share, bits in passes:
            elements = math.ceil(cols * share)
            busy[engine] += row_chunks * self.engine_ns(engine, elements, bits)
        return busy

    def step_ns(self, owed, compute, refill):
        """A double-buffered step of ``compute`` that reloads ``owed`` of
        DMA.  A slot's refill starts as the step before it frees the slot
        and moves no row for ``refill`` after, so it hides under the step's
        compute only with that latency to spare: the step takes ``compute``
        up to ``compute - refill`` of DMA, ``owed`` from ``compute + refill``
        on, when the queue runs ahead of the compute, and the mean of the
        two and the latency between."""
        if not owed:
            return compute
        return max(compute, owed, (compute + owed + refill) / 2)

    def last_piece_share(self, rows, cols, bits):
        """The share of a load of a ``rows x cols`` tile of ``bits``-wide
        elements that its last ``dma_copy`` moves (``stream_blocks``): what
        an op reading the tile as its pieces land has left once the last
        one does."""
        p = self.config.partitions
        blocks = math.ceil(rows / p)
        per = stream_blocks(blocks, p * math.ceil(cols * bits / 8))
        return (blocks - (math.ceil(blocks / per) - 1) * per) / blocks

    def sweep_ns(
        self,
        dmas,
        steps,
        compute,
        first_load=True,
        apart=(),
        last=None,
        held=False,
    ):
        """A double-buffered sweep of ``steps`` grid steps of ``compute``
        each, moving ``dmas`` -- ``(ns, transfers)`` per buffer, the output
        first.  ``step_classes`` sorts the steps by what each reloads, each
        priced by ``step_ns``, and the ends are priced on their own as
        ``sweep_cycles`` prices them: the first loads overlap no compute of
        the kernel's, the first compute stores nothing and the last loads
        nothing, and the last store waits on it.  The first loads and the
        first DMA's latency are paid when ``first_load``; else the loads run
        under the kernel before (``gemm.first_load_exposed``) for as long as
        the DMA takes to fill the SBUF a kernel leaves them
        (``HOISTED_SBUF``), and only the rest is paid.  The kernel's fixed
        cost comes on top.

        ``apart`` holds the positions in ``dmas`` of the buffers that move
        on DMA queues of their own (a GEMV's one-row activation and output,
        ``TrainiumGemv.cost``): the steps owe only the others, the first
        loads take the longer of the two queues, and the sweep lasts at
        least as long as their transfers.  ``last`` is the compute a step
        has left once its last load lands, when it reads its tiles piece by
        piece as they land (``TrainiumGemv.cost``): a sweep the DMA binds
        ends that compute and ``dram_access_latency`` after its last load
        lands, one the compute binds ends a whole step's compute later, and
        one in between ends as much later as the compute falls behind the
        loads.

        A slot's refill issues its first copy (``dma_issue_ns``) when the
        step reading the slot frees it.  ``held`` is a step whose matmuls
        read the tiles it reloads until its compute ends (a GEMM's), so the
        refill's rows also start to move only ``dram_access_latency`` after
        that; otherwise the issue alone counts."""
        config = self.config
        latency = config.dram_access_latency
        refill = config.dma_issue_ns + (latency if held else 0.0)
        store = dmas[0][0]
        queue = [
            (0.0 if i in apart else ns, t) for i, (ns, t) in enumerate(dmas)
        ]
        loads = sum(ns for ns, _ in queue[1:])
        first = max(
            loads,
            sum(ns for i, (ns, _) in enumerate(dmas) if i and i in apart),
        )
        fixed = config.kernel_overhead_ns
        if first_load:
            fixed += first + latency
        else:
            hoisted = config.partitions * HOISTED_SBUF / config.dram_bandwidth
            fixed += max(0.0, first - hoisted)
        if steps == 1:
            return fixed + compute + store
        total = sum(
            count * self.step_ns(owed, compute, refill)
            for owed, count in step_classes(queue, steps)
        )
        # What a step past the first reloads, and the same without the
        # store, as ``sweep_cycles`` counts them.
        recur = sum(ns for ns, t in queue if steps // t == 1)
        held = sum(ns for ns, t in queue[1:] if steps // t == 1)
        if last is None:
            end = self.step_ns(queue[0][0], compute, refill)
        else:
            end = max(last + latency, min(compute, 2 * compute - held))
        sweep = (
            total
            - self.step_ns(loads + queue[0][0], compute, refill)
            - self.step_ns(recur, compute, refill)
            + self.step_ns(held, compute, refill)
            + end
            + store
            + fixed
        )
        if not apart:
            return sweep
        beside = sum(ns * t for i, (ns, t) in enumerate(dmas) if i in apart)
        return max(sweep, beside + latency)


def kernel_nodes(gm):
    """The nodes the bufferizer turns into kernels, one each and in order,
    with the key under which it reuses the build of an identical one.  Taken
    before ``compile``, which replaces them."""
    return [(n, _bufferize_key(n)) for n in gm.graph.nodes if _is_compute(n)]


def predicted_seconds(nodes, config):
    """The latency the tile search predicted for each kernel of ``nodes``
    (``kernel_nodes``): the cost of its winning tiling, stamped as
    ``meta["tile_result"]`` on a GEMM's anchor and on a GEMV or vector op
    itself.  A kernel whose build was reused takes the prediction of the one
    it reuses.

    Returns:
        Seconds per kernel, in order.
    """
    priced, predicted = {}, []
    for node, key in nodes:
        anchor = get_anchor_node(node) or node
        result = anchor.meta.get("tile_result") or node.meta.get("tile_result")
        if result is not None:
            seconds = result.cost.latency / (config.frequency * 1e9)
            priced.setdefault(key, seconds)
        predicted.append(priced[key])
    return predicted
