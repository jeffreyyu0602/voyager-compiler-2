"""A GEMM / conv on Voyager's PE array, mapped by interstellar:
``VoyagerGemm`` packs a tile into the scratchpad banks and its
``RuntimeCalculator`` prices a tiling.
"""

import math
from dataclasses import dataclass
from functools import partial

import interstellar
from voyager_compiler.codegen.node_info import (
    csr_quantize_node,
    get_anchor_node,
    get_arg_value,
    tail_targets,
    weight_transforms,
)
from voyager_compiler.codegen.transform.tiling.lowering import (
    fused_operand_specs,
    gemm_batch,
    get_dtype_width,
    layer_key,
    matrix_layer,
    node_dtype_bits,
    outlier_stats,
    output_dtype,
    tail_extra_passes,
    tail_keeps_shape,
    weight_batch,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    Cost,
    Fused,
    MatrixSpace,
    OpModel,
    Placement,
    Role,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    INPUT_DIMS,
    OUTPUT_DIMS,
    WEIGHT_DIMS,
    batch_loads,
    l3_loads,
    l3_steps,
    operand_elements,
    step_classes,
    sweep_cycles,
    tile_extent,
    total_extent,
)

from .vector import BANK_SWITCH_CYCLES, strided_bank_walk

le = interstellar.le

# Finished output vectors the matrix -> vector path holds before a
# single-buffered accumulator's array feels the tail's drain rate: the
# matrix processor's output FIFO (8), the vector pipeline's input FIFO (9)
# and the stages between.  Measured on the E4M3x16x16 SoC as 20 (MobileBERT
# GEMMs) to 40 (ResNet18) vectors; either way the error is under ~60 cycles
# per L3 step.
OUTPUT_SLACK = 24

# Cycles the SpMM unit spends per row of each PE-array-wide pass on top of
# the row's outliers: it drains and restarts its accumulator ring between
# rows.  Measured on the Sphinx SoC: 7.3-7.5.
SPMM_ROW_CYCLES = 8

# Block-scale rows the SpMM unit's weight-scale buffer holds
# (``DoubleBuffer<32>`` in ``SpMMUnit.h``, fixed in the Sphinx silicon).  A
# streaming weight tile takes one row per K block per inner OC pass, and the
# hardware wraps the address rather than checking it.
SPMM_SCALE_ROWS = 32


@dataclass(frozen=True)
class RuntimeCalculator:
    """Runtime cost model for a 4-level hierarchy (PE / L1 / L2 / DRAM).

    A mapping is priced in cycles as its L3 grid sweep.  With a
    double-buffered L2 each step costs the slower of its DRAM transfers and
    its compute, otherwise their sum, and the sweep is framed by the
    un-overlapped first load and last store; a split reduction adds its
    accumulate steps and the tail pass that finishes each output tile.  A
    DRAM transfer costs its bytes at ``dram_bandwidth`` plus one access
    latency, block scales being a transfer of their own, and the batch loop
    outside the mapping shares a weight tile among the elements of one
    group.  Energy is interstellar's cost model, not this one.

    The compute of an L2 block is the busier of the matrix unit and the
    scratchpad bus.  The matrix unit charges the systolic passes, each weight
    tile costing the longer of its loading and the rows streamed through it,
    plus the back-pressure of a single-buffered accumulator against
    ``OUTPUT_SLACK`` of buffering.  The bus charges the words every operand
    role moves, summed per bank group of the placement with the busiest bank
    setting the pace: input and weight rows in whole beats, packed as the
    toolchain packs them, with the read interface's lost cycle whenever an
    aligned request follows an unaligned one; block scales one word each;
    the output, the fused tail operands, the bias and the reduction scratch;
    and the round trip a stream pays when it changes bank on its
    bank-aligned tile buffer.  A weight tile held across the spatial loops
    is fetched once.  The SpMM unit adds a turnaround per row visit and the
    outlier rows it gathers at the block's K depth, most of them a bank
    switch when the weight tile spans several banks.  A tail pass of its own
    costs the vector unit's lane rate or its bank words, whichever is
    slower.  The ramps of a sweep, the first buffer fill, the systolic skew
    and the last drain, are spread over the ops the sweep dispatches.

    Not modeled are the per-op costs (parameter load, deserialisation, the
    start/done handshake, the drain between uncommitted ops, the host's
    dispatch time), so an op that runs alone pays its ramps in full where
    the spreading charges a share; the cycle-exact datapath, the systolic
    skew being a constant of the array dims; a stream-breaking
    ``quantize_mx`` tail, charged as a staged region rather than a drained
    pass; a conv input tile's halo; more than one port width, every operand
    moving at ``sram_bandwidth`` over one bus per bank; the block scales'
    bank switches; and the outlier density of an individual tile, priced at
    the layer's average.

    Attributes:
        input_dtype_width: Bits of an input element.
        weight_dtype_width: Bits of a weight element.
        output_dtype_width: Bits of a finished output element.
        accum_dtype_width: Bits of a partial sum: 32 when accumulating in
            float32, else the anchor's own dtype's.
        double_buffered_accum_buffer: The PE array's accumulator can park a
            finished tile while it computes the next.
        sram_bandwidth: Bits a scratchpad bank's port moves per cycle.
        dram_bandwidth: DRAM bytes per cycle.
        dram_access_latency_cycles: Cycles a DRAM transfer waits for its
            first byte.
        double_buffered_l2: Scratchpad tiles ping-pong, so a step's DMAs
            overlap its compute.
        outlier_rate: The consumed outlier CSR's measured outlier rate, 0
            without one.
        batch: Batch elements the builder loops outside the mapping.
        weight_batch: Distinct weight tiles over that loop (a GQA repeat
            reads one tile for several elements).
        has_tail: The group fuses ops after the anchor.
        single_k_tail_extra_pass: The tail needs a pass of its own after an
            unsplit reduction.
        split_k_tail_extra_pass: The tail needs a pass of its own after a
            split reduction.
        tail_keeps_shape: The tail's output has the anchor's shape, so an
            extra pass can run in place.
        tail_specs: ``(dims, bits)`` per fused tail operand
            (``fused_operand_specs``).
        input_scale_width: Bits of an input block scale, 0 without one.
        weight_scale_width: Bits of a weight block scale, 0 without one.
        output_scale_width: Bits of an output block scale, 0 without one.
        scale_block_size: Values one block scale covers, 1 without one.
        bias_width: Bits of a bias element, 0 without a bias.
        stride: The conv's ``(height, width)`` stride, ``(1, 1)`` for a
            GEMM.
        bank_size: Bytes of a scratchpad bank, ``None`` without banks.
        weight_transposed: The weight is stored transposed.
    """

    input_dtype_width: int
    weight_dtype_width: int
    output_dtype_width: int
    accum_dtype_width: int
    double_buffered_accum_buffer: bool
    sram_bandwidth: int
    dram_bandwidth: float
    dram_access_latency_cycles: float
    double_buffered_l2: bool
    outlier_rate: float
    batch: int
    weight_batch: int
    has_tail: bool
    single_k_tail_extra_pass: bool
    split_k_tail_extra_pass: bool
    tail_keeps_shape: bool
    tail_specs: tuple
    input_scale_width: int
    weight_scale_width: int
    output_scale_width: int
    scale_block_size: int
    bias_width: int
    stride: tuple
    bank_size: int | None
    weight_transposed: bool

    def tail_tile_sizes(self, mapping):
        """DRAM bytes each fused tail operand streams for one output tile --
        one transfer apiece.  An operand is tiled along the output dims it is
        not broadcast over, so its tile is the output tile's extent there."""
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings
        sizes = []
        for dims, bits in self.tail_specs:
            count = 1
            for d in dims:
                count *= blockings[d][1] * blockings[d][2] * partitionings[d][0]
            sizes.append(count * bits / 8)
        return sizes

    def _bank_cycles(self, words, bank_groups):
        """Cycles the busiest scratchpad bank spends moving ``words`` -- bus
        words per operand role -- when the roles sharing a bank
        (``bank_groups``: the placement's groups, ``None`` = nothing shares)
        queue on its single port.  A role the placement does not name keeps a
        port of its own.
        """
        placed = set()
        busiest = 0
        for roles in bank_groups or ():
            busiest = max(busiest, sum(words.get(role, 0) for role in roles))
            placed.update(roles)
        loose = [count for role, count in words.items() if role not in placed]
        return max([busiest, *loose])

    def _request_words(self, requests, row_elems, bits, loop_bound, pitch):
        """Bus words ``requests`` fetches of one ``row_elems``-element row
        take.  A request is served in whole beats, so a row that is not a
        multiple of the port costs more than its bytes.  The controller packs
        the rows that fill whole beats into one request when the L1 loop
        that walks them, of ``loop_bound`` steps, divides into that many
        (the toolchain's ``get_packing_factor``); a ``loop_bound`` of 0 never
        packs.  Consecutive requests are ``pitch`` bytes apart, the row
        pitch of the tile buffer.  The SoC's read interface flushes the tail
        of a request that started inside a beat in a cycle of its own and
        holds the next request's first beat when that one starts on a beat
        boundary, so a pitch that is not a multiple of the beat cycles the
        requests through the beat offsets and costs one cycle per aligned
        request that follows an unaligned one."""
        if not bits or requests <= 0:
            return 0
        row_bits = row_elems * bits
        pf = math.lcm(row_bits, self.sram_bandwidth) // row_bits
        rows_per_request = pf if loop_bound and loop_bound % pf == 0 else 1
        request_words = math.ceil(
            rows_per_request * row_bits / self.sram_bandwidth
        )
        count = math.ceil(requests / rows_per_request)
        beat = self.sram_bandwidth // 8
        misalignment = rows_per_request * pitch % beat
        flushes = 0
        if misalignment:
            flushes = count * math.gcd(misalignment, beat) / beat
        return count * request_words + flushes

    def _bus_words(self, count, bits, bandwidth=None):
        """Bus words ``count`` elements of ``bits`` occupy at the bank's full
        width, or at ``bandwidth`` bytes per cycle."""
        if not bits or count <= 0:
            return 0
        return math.ceil(
            count * bits / 8 / (bandwidth or self.sram_bandwidth / 8)
        )

    _extent = staticmethod(tile_extent)
    _l3_blocks = staticmethod(l3_steps)
    _l3_loads = staticmethod(l3_loads)

    def _load_words(self, mapping):
        """Bus words one L1 tile's every-round operands take, per role: the
        input and its block scales, the weight and its scales.  The input
        and the weight arrive one PE-array row per request
        (``_request_words``), packed several rows per request when the L1
        loop that walks them allows it -- never for a transposed weight;
        the weight scales one PE-array row of scales per request, never
        packed.  An input scale is delivered one
        per bus word however narrow it is; every outlier in the input tile
        gathers one more weight row on top of the dense weight tile."""
        ext = lambda loop: self._extent(mapping, loop, 1)
        rows = ext(le.OX) * ext(le.OY) * ext(le.ON)
        depth = ext(le.IC)
        taps = ext(le.FX) * ext(le.FY)
        gathered_rows = rows * depth * self.outlier_rate
        blockings = mapping.loop_blockings
        ic_unroll = mapping.loop_partitionings[le.IC][0]
        oc_unroll = mapping.loop_partitionings[le.OC][0]
        weight_loop = 0 if self.weight_transposed else blockings[le.OC][1]
        # The tile buffers are [rows, IC] for the input and [IC, OC] for the
        # weight and its scales: a request walks one row of each.
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        input_pitch = ic3 * self.input_dtype_width // 8
        weight_pitch = oc3 * self.weight_dtype_width // 8
        scale_pitch = oc3 * self.weight_scale_width // 8
        words = {
            Role.INPUT: self._request_words(
                rows * depth / ic_unroll,
                ic_unroll,
                self.input_dtype_width,
                blockings[le.IC][1],
                input_pitch,
            ),
            Role.WEIGHT: self._request_words(
                (depth * taps + gathered_rows) * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_dtype_width,
                weight_loop,
                weight_pitch,
            ),
            Role.WEIGHT_SCALE: self._request_words(
                depth * taps / self.scale_block_size * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_scale_width,
                0,
                scale_pitch,
            ),
        }
        if self.input_scale_width:
            words[Role.INPUT_SCALE] = math.ceil(
                rows * depth / self.scale_block_size
            )
        return words

    def _tail_words(self, mapping, level):
        """Bus words the tail's operands take for one tile at ``level`` (1 =
        an L1 output tile, 2 = the whole L3 output tile), per role: the
        finished output with its block scales -- each scale leaves in a bus
        word of its own, however narrow, as the input scales arrive -- and
        each fused operand over the output dims it is tiled along.  The tail
        rides on the array's output one vector (a row of ``oc_dim`` values)
        at a time and fetches a fused operand in one unpacked request per
        vector, so an operand narrower than a bus word still costs a word
        per vector."""
        ext = lambda loop: self._extent(mapping, loop, level)
        out = ext(le.OC) * ext(le.OY) * ext(le.OX) * ext(le.ON)
        words = {Role.OUTPUT: self._bus_words(out, self.output_dtype_width)}
        if self.output_scale_width:
            words[Role.OUTPUT] += math.ceil(out / self.scale_block_size)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        oc2 = self._extent(mapping, le.OC, 2)
        for i, (dims, bits) in enumerate(self.tail_specs):
            if le.OC in dims:
                rows = math.prod(ext(dim) for dim in dims if dim != le.OC)
                vectors = rows * math.ceil(ext(le.OC) / oc_dim)
                words[Fused(i)] = self._request_words(
                    vectors, oc_dim, bits, 0, math.ceil(oc2 * bits / 8)
                )
            else:
                words[Fused(i)] = self._bus_words(
                    math.prod(ext(dim) for dim in dims), bits
                )
        return words

    def _stream_switches(self, mapping, key_loops, walk_of, held_loops=()):
        """Compose L1 bank walks in the emitted L2 loop order.

        ``walk_of(idx)`` returns ``(switches, first_bank, last_bank)`` for
        one L1 request nest. Key loops change its addresses; other loops
        repeat it, unless the controller holds the operand across them.
        Fold repeats at their actual nesting depth, including loops between
        two key loops, so both rewinds and inter-block transitions survive.
        """
        blockings, orders = mapping.loop_blockings, mapping.loop_orders
        nest = sorted(
            (
                i
                for i in range(le.NUM)
                if blockings[i][2] > 1 and i not in held_loops
            ),
            key=lambda i: -orders[i][2],
        )

        def walk(depth, idx):
            if depth == len(nest):
                return walk_of(idx)
            loop = nest[depth]
            count = blockings[loop][2]
            if loop not in key_loops:
                inner, start, end = walk(depth + 1, idx)
                return count * inner + (count - 1) * (start != end), start, end
            total = 0
            first = last = None
            for index in range(count):
                idx[loop] = index
                inner, start, end = walk(depth + 1, idx)
                total += inner + (last is not None and start != last)
                if first is None:
                    first = start
                last = end
            return total, first, last

        return walk(0, {})[0]

    def _bank_switch_cycles(self, mapping):
        """Cycles one L3 step's input and weight streams lose to bank
        switches, per role (``BANK_SWITCH_CYCLES`` each).

        Follow the mapping's L1 input order and the weight FY/FX/IC/OC scan.
        Packing combines adjacent channel groups into a request exactly
        when the toolchain permits it, matching _request_words. Buffers
        start on banks; input scales are not included.
        A weight tile held across spatial loops is not refetched by them.
        """
        if not self.bank_size:
            return {}
        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        ic1 = self._extent(mapping, le.IC, 1)
        oc1 = self._extent(mapping, le.OC, 1)
        fy, fx = blockings[le.FY][1], blockings[le.FX][1]
        oy1, ox1 = blockings[le.OY][1], blockings[le.OX][1]
        hs, ws = self.stride
        y_in = (oy1 * blockings[le.OY][2] - 1) * hs + fy
        x_in = (ox1 * blockings[le.OX][2] - 1) * ws + fx
        pitch_in = ic3 * self.input_dtype_width / 8
        ic_dim = mapping.loop_partitionings[le.IC][0]
        oc_dim = mapping.loop_partitionings[le.OC][0]

        def packed_width(lanes, bits, count):
            row_bits = lanes * bits
            factor = math.lcm(row_bits, self.sram_bandwidth) // row_bits
            return lanes * (factor if count and count % factor == 0 else 1)

        input_chunk = packed_width(
            ic_dim, self.input_dtype_width, blockings[le.IC][1]
        )
        input_order = sorted((le.IC, le.OY, le.OX), key=lambda i: -orders[i][1])

        def input_walk(idx):
            y0 = idx.get(le.OY, 0) * oy1 * hs
            x0 = idx.get(le.OX, 0) * ox1 * ws
            # Filter taps expand the spatial fetch; the controller disables
            # its separate L1 FX/FY/OC loops. Clip the last halo to the tile.
            sy, sx = (hs if fy == 1 else 1), (ws if fx == 1 else 1)
            ny = min(
                oy1 if fy == 1 else oy1 * hs + fy - 1, (y_in - 1 - y0) // sy + 1
            )
            nx = min(
                ox1 if fx == 1 else ox1 * ws + fx - 1, (x_in - 1 - x0) // sx + 1
            )
            width = input_chunk * self.input_dtype_width / 8
            scans = {
                le.IC: (ic1 // input_chunk, width),
                le.OY: (ny, sy * x_in * pitch_in),
                le.OX: (nx, sx * pitch_in),
            }
            loops = tuple(scans[i] for i in input_order)
            offset = (y0 * x_in + x0) * pitch_in
            offset += idx.get(le.IC, 0) * ic1 * self.input_dtype_width / 8
            return strided_bank_walk(loops, width, self.bank_size, offset)

        pitch_w = oc3 * self.weight_dtype_width / 8
        beat_w = oc1 * self.weight_dtype_width / 8
        weight_chunk = packed_width(
            oc_dim,
            self.weight_dtype_width,
            0 if self.weight_transposed else blockings[le.OC][1],
        )

        def weight_walk(idx):
            c0 = idx.get(le.IC, 0) * ic1
            k_off = idx.get(le.OC, 0) * beat_w
            width = weight_chunk * self.weight_dtype_width / 8
            loops = (
                (fy * fx, ic3 * pitch_w),
                (ic1, pitch_w),
                (oc1 // weight_chunk, width),
            )
            return strided_bank_walk(
                loops, width, self.bank_size, c0 * pitch_w + k_off
            )

        held = ()
        if blockings[le.IC][2] == 1:
            held = tuple(
                loop
                for loop in (le.OX, le.OY)
                if orders[loop][2] < orders[le.OC][2]
            )
        return {
            Role.INPUT: BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OX, le.OY, le.IC), input_walk),
            Role.WEIGHT: BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OC, le.IC), weight_walk, held),
        }

    def _tail_bank_switch_cycles(self, mapping, tiled):
        """Fused-operand read latency, charged only on a finishing pass.

        A riding tail uses MatrixOps' filtered L2/L1 output loops. A
        separate vector pass scans the finished output tile in storage
        order. Broadcast dimensions have zero address stride. Multi-beat
        requests overlap part of the bank-switch drain (two beats lose
        seven cycles, versus eight for one), as in pool_bank_switch_cycles.
        """
        if not self.bank_size:
            return {}
        blockings, orders = mapping.loop_blockings, mapping.loop_orders
        dims_out = (le.ON, le.OY, le.OX, le.OC)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        l1_order = sorted(dims_out, key=lambda i: -orders[i][1])
        result = {}
        for i, (dims, bits) in enumerate(self.tail_specs):
            strides = {}
            pitch = bits / 8
            for dim in reversed(dims_out):
                strides[dim] = pitch if dim in dims else 0
                if dim in dims:
                    pitch *= self._extent(mapping, dim, 2)
            lanes = oc_dim if le.OC in dims else 1
            width = math.ceil(lanes * bits / 8)

            def tile_walk(idx):
                offset = sum(
                    idx.get(d, 0) * self._extent(mapping, d, 1) * strides[d]
                    for d in dims_out
                )
                loops = tuple(
                    (
                        blockings[d][1],
                        strides[d] * mapping.loop_partitionings[d][0],
                    )
                    for d in l1_order
                )
                return strided_bank_walk(loops, width, self.bank_size, offset)

            if tiled:
                switches = self._stream_switches(
                    mapping, dims, tile_walk, (le.IC, le.FX, le.FY)
                )
            else:
                loops = tuple(
                    (
                        blockings[d][1] * blockings[d][2],
                        strides[d] * mapping.loop_partitionings[d][0],
                    )
                    for d in dims_out
                )
                switches = strided_bank_walk(loops, width, self.bank_size)[0]
            beats = math.ceil(width * 8 / self.sram_bandwidth)
            result[Fused(i)] = switches * max(1, BANK_SWITCH_CYCLES + 1 - beats)
        return result

    def _tail_stall(self, mapping, bank_groups, words, compute, bank):
        """Cycles the array loses to the tail's burst on a bank that feeds
        one of its every-round buffers.

        Those buffers ping-pong per L1 sweep (``blockings[IC][2]`` sweeps
        to a block), so each sweep's fetch must land inside the sweep before
        it.  The tail's words on such a bank do not spread over the block:
        they arrive together, and the bank's round-robin port lets them
        through at the tail's beats per request round for every grant of a
        stream that is always pending, or in the sweep's free cycles when
        the stream idles between its requests.  The sweeps the burst lands
        on are priced one by one, and what they cost beyond the block's
        price is the stall.  A tail buffer meets the stream's bank only on
        the steps whose ping-pong slots agree: every step when the stream
        is refetched with it, else every other.
        """
        if not bank_groups:
            return 0
        sweeps = mapping.loop_blockings[le.IC][2]
        steps = self._l3_blocks(mapping)
        if sweeps == 1 or steps == 1:
            return 0
        sweep_compute = compute / sweeps
        vectors = 1
        for loop in [le.OC, le.OY, le.OX]:
            vectors *= mapping.loop_blockings[loop][1]
        stream_dims = {
            Role.INPUT: INPUT_DIMS,
            Role.INPUT_SCALE: INPUT_DIMS,
            Role.WEIGHT: WEIGHT_DIMS,
            Role.WEIGHT_SCALE: WEIGHT_DIMS,
        }
        stall = 0.0
        for roles in bank_groups:
            streams = [role for role in roles if role in stream_dims]
            tails = []
            for role in roles:
                is_tail = role is Role.OUTPUT or isinstance(role, Fused)
                if is_tail and words.get(role, 0):
                    tails.append(role)
            if not streams or not tails:
                continue
            others = [
                role
                for role in roles
                if role not in streams and role not in tails
            ]
            stream_words = [words.get(role, 0) / sweeps for role in streams]
            busiest = max(stream_words)
            pending = sum(stream_words)
            spread = sum(words.get(role, 0) for role in others) / sweeps
            tail_words = sum(words[role] for role in tails)
            # The output's beats and its scales leave on two requesters,
            # every beat and every scale a request of its own; a fused
            # operand is fetched once per output vector.
            oc_dim = mapping.loop_partitionings[le.OC][0]
            requests = []
            for role in tails:
                if role is Role.OUTPUT:
                    out = vectors * oc_dim
                    stores = self._bus_words(out, self.output_dtype_width)
                    requests.append(stores)
                    if self.output_scale_width:
                        requests.append(math.ceil(out / self.scale_block_size))
                elif le.OC in self.tail_specs[role.index][0]:
                    requests.append(vectors)
                else:
                    requests.append(words[role])
            beats_per_grant = tail_words / max(requests)
            remaining = tail_words
            priced = 0.0
            for sweep in range(sweeps):
                free = max(0.0, sweep_compute - pending - spread)
                landed = min(remaining, max(beats_per_grant * busiest, free))
                if sweep == sweeps - 1:
                    landed = remaining
                remaining -= landed
                priced += max(sweep_compute, pending + spread + landed)
            refetched = [
                self._l3_loads(mapping, stream_dims[role]) == steps
                for role in streams
            ]
            fraction = 1.0 if any(refetched) else 0.5
            excess = max(0.0, priced - max(compute, bank))
            stall = max(stall, fraction * excess)
        return stall

    def matrix_cycles(self, mapping, bank_groups):
        """Cycles of one L3 grid step: the L2 sweep of weight-reuse tiles,
        each costing the busier of the matrix unit -- its systolic passes,
        plus the back-pressure a single-buffered accumulator takes while the
        tail drains each finished tile -- and the scratchpad bank with the
        most to move for it -- the every-round operands of its L1 sub-tiles,
        the accumulator read back and rewritten while the reduction is
        split, the finished tile and the tail's operands when it is not,
        the round trips a stream idles for when it changes bank, each
        summed with whatever shares its bank -- plus the stall the tail's
        burst inflicts where it shares a bank with one of the array's
        every-round buffers (``_tail_stall``) -- and, for an outlier GEMM,
        the SpMM unit, which must deliver the block's sparse correction
        before the vector pipeline releases any of its rows: per 64-column
        pass it walks every row of the block, paying ``SPMM_ROW_CYCLES`` of
        turnaround plus that row's outliers, each gather a bank switch when
        the weight tile spans several banks -- plus the once-per-sweep
        overhead (buffer fill, systolic skew, the last parked tile's drain)
        spread over the ops a double-buffered L2 overlaps it with.  Also
        the reporting model's per-tile utilization denominator.

        Args:
            mapping: The interstellar mapping to price.
            bank_groups: Its placement's bank groups (``Placement.groups``),
                or ``None`` when nothing shares a bank.
        """
        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing ---
        sa_weight_loading_time = partitionings[le.IC][0]

        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        weight_reuse_tile_size = 1
        for i in range(le.NUM):
            if orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= blockings[i][1]
        weight_reuse_tile_time = max(
            sa_weight_loading_time, weight_reuse_tile_size
        )

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        num_remaining_l1_tiles *= blockings[le.IC][2]
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        # --- the finished L1 output tile: its vectors, and the bus beats the
        # tail spends on each -- storing it, and reading a fused operand
        # alongside when one is wider than the port ---
        num_k = blockings[le.IC][3]
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        oc_dim = partitionings[le.OC][0]
        output_width = (
            self.accum_dtype_width if num_k > 1 else self.output_dtype_width
        )
        store_cycles = math.ceil(output_width * oc_dim / self.sram_bandwidth)
        if num_k == 1 and self.output_scale_width:
            # The block scales leave on a requester of their own, one beat
            # per vector, on the output's bank.
            store_cycles += 1
        vector_beats = store_cycles
        if num_k == 1 and not self.single_k_tail_extra_pass:
            for dims, bits in self.tail_specs:
                vector_beats = max(
                    vector_beats,
                    math.ceil(
                        bits
                        * (oc_dim if le.OC in dims else 1)
                        / self.sram_bandwidth
                    ),
                )

        # Without a bank to park the finished tile in, its vectors leave the
        # array one per step during the last reduction pass of each weight
        # tile -- as one burst of the whole tile when the OC passes are
        # adjacent (no filter loops at L1), else as OC1 bursts of one spatial
        # tile -- and the tail drains them at ``vector_beats`` apiece.  The
        # path absorbs ``OUTPUT_SLACK`` of them; past that the array runs at
        # the tail's pace for the rest of the burst.  A double-buffered
        # accumulator parks a tile whose tail moves more than a bus word per
        # vector on some port (the toolchain's ``should_use_direct_path``);
        # a narrower tail rides the pass and pays like a single-buffered one.
        parked = self.double_buffered_accum_buffer
        if parked:
            widths = [output_width * oc_dim]
            for dims, bits in self.tail_specs:
                widths.append(bits * (oc_dim if le.OC in dims else 1))
            parked = max(widths) > self.sram_bandwidth
        if not parked:
            burst_vectors = weight_reuse_tile_size
            burst_cycles = weight_reuse_tile_time
            bursts = blockings[le.OC][1]
            if blockings[le.FX][1] * blockings[le.FY][1] == 1:
                burst_vectors *= bursts
                burst_cycles *= bursts
                bursts = 1
            computation_l1_time += bursts * max(
                0, burst_vectors * vector_beats - burst_cycles - OUTPUT_SLACK
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        # --- bus traffic of one L2 output block: the loads of its L1
        # sub-tiles, then what the vector unit moves for the block itself ---
        loads = self._load_words(mapping)
        words = {
            role: count * blockings[le.IC][2] for role, count in loads.items()
        }
        # With the whole reduction inside the block, a weight tile whose L2
        # loop is outside the spatial ones is fetched once and held across
        # them (the input is refetched every block), so its words are spread
        # over the blocks that reuse it.
        if blockings[le.IC][2] == 1:
            held = 1
            for loop in [le.OX, le.OY]:
                if orders[loop][2] < orders[le.OC][2]:
                    held *= blockings[loop][2]
            words[Role.WEIGHT] /= held
            words[Role.WEIGHT_SCALE] /= held
        # The bias is read once per output tile: spread over its rounds.
        words[Role.BIAS] = (
            self._bus_words(self._extent(mapping, le.OC, 1), self.bias_width)
            / num_k
        )
        output_elems = 1
        for loop in [le.OC, le.OY, le.OX, le.ON]:
            output_elems *= self._extent(mapping, loop, 1)
        if num_k > 1:
            # A split reduction reads the running partial back through the
            # same single-ported bank it writes the new one to.
            words[Role.SCRATCH] = 2 * self._bus_words(
                output_elems, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass and not self.tail_keeps_shape:
            # A staged single round parks the finished tile in scratch for
            # the tail's own pass (``vector_cycles``) to read back.
            words[Role.SCRATCH] = self._bus_words(
                output_elems, self.output_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # An in-place pass reads the tile back from its output slot and
            # rewrites it: two bank visits beyond a riding tail's.
            words.update(self._tail_words(mapping, 1))
            words[Role.OUTPUT] += 2 * self._bus_words(
                output_elems, self.output_dtype_width
            )
        else:
            words.update(self._tail_words(mapping, 1))
        # A stream that changes bank idles its port for a round trip each
        # time; spread the step's switches over its output blocks.
        switches = self._bank_switch_cycles(mapping)
        if num_k == 1 and (
            not self.single_k_tail_extra_pass or self.tail_keeps_shape
        ):
            switches.update(
                self._tail_bank_switch_cycles(
                    mapping, tiled=not self.single_k_tail_extra_pass
                )
            )
        for role, cycles in switches.items():
            words[role] += cycles / l2_blocks
        # The matrix unit and the vector unit are pipelined -- one drains a
        # tile while the other computes the next -- so a block costs the
        # busier of the two.
        bank = self._bank_cycles(words, bank_groups)
        block_time = max(computation_l1_time, bank)
        # The tail's words do not spread over the block: they land as a
        # burst on a few of its sweeps, and on a bank that feeds one of the
        # array's ping-pong buffers they can outrun the sweeps they land on.
        block_time += self._tail_stall(
            mapping, bank_groups, words, computation_l1_time, bank
        )

        # The SpMM unit runs the block alongside and the vector pipeline
        # waits for its correction on every output vector, so a block also
        # costs its pace: per PE-array-wide pass, every row's turnaround
        # plus its gathered weight rows.
        if self.outlier_rate:
            rows = 1
            for loop in [le.OX, le.OY, le.ON]:
                rows *= self._extent(mapping, loop, 1)
            k_block = self._extent(mapping, le.IC, 2)
            passes = blockings[le.OC][1]
            visits = passes * rows
            gathers = visits * k_block * self.outlier_rate
            # Gathers hit random K rows of the weight tile; when it spans
            # several banks most consecutive gathers change bank and pay the
            # read path's round trip, as the streams above do.
            switch = 0.0
            if self.bank_size:
                weight_tile_bytes = (
                    self._extent(mapping, le.OC, 2)
                    * k_block
                    * blockings[le.FX][1]
                    * blockings[le.FY][1]
                    * self.weight_dtype_width
                    / 8
                )
                banks = max(1, math.ceil(weight_tile_bytes / self.bank_size))
                switch = BANK_SWITCH_CYCLES * (1 - 1 / banks)
            spmm_block_time = visits * SPMM_ROW_CYCLES + gathers * (1 + switch)
            block_time = max(block_time, spmm_block_time)

        # The first tile's loads overlap nothing; the last parked tile's drain
        # is a whole vector pass, while a single-buffered accumulator's drain
        # is already in its blocks' own time.
        buffer_fill = self._bank_cycles(loads, bank_groups)
        skew = partitionings[le.IC][0] + partitionings[le.OC][0] - 2
        drain = output_size * vector_beats if parked else 0
        overhead = buffer_fill + skew + drain
        steady = l2_blocks * block_time

        if not self.double_buffered_l2:
            return steady + overhead

        # Every op the sweep dispatches -- the L3 steps and the batch elements
        # looped outside the mapping -- overlaps the ramps of its neighbours.
        steps = self._l3_blocks(mapping) if num_k == 1 else num_k
        return steady + overhead / (steps * self.batch)

    def vector_cycles(self, mapping, bank_groups):
        """Vector-unit cycles to finish one L3 output tile.  Charged on the grid
        step that ends a K sweep, after that step's accumulation.

        The busier of the unit's own rate -- one lane group per cycle, sized
        by the widest element the tail touches (the partial sum it reads, not
        the narrower value a ``quantize_mx`` writes) at ``dram_bandwidth``,
        which is what ``vector_op_utilization`` charges for the same tail in
        the reporting model -- and the busiest bank: the accumulator read
        back, the finished tile and its scales written, each tail operand
        read, summed wherever the placement puts them together.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        widths = [self.output_dtype_width, self.accum_dtype_width]
        widths += [bits for _, bits in self.tail_specs]
        lane_bytes = max(widths) / 8 * oc_dim
        lanes = output_size * math.ceil(lane_bytes / self.dram_bandwidth)
        words = self._tail_words(mapping, 2)
        for role, cycles in self._tail_bank_switch_cycles(
            mapping, tiled=False
        ).items():
            words[role] += cycles
        if blockings[le.IC][3] > 1 or (
            self.single_k_tail_extra_pass and not self.tail_keeps_shape
        ):
            words[Role.SCRATCH] = self._bus_words(
                output_size * oc_dim, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # The in-place pass reads the finished tile from its output slot.
            words[Role.OUTPUT] += self._bus_words(
                output_size * oc_dim, self.output_dtype_width
            )
        return max(lanes, self._bank_cycles(words, bank_groups))

    def cost(self, mapping, placement):
        """The op's latency under ``mapping`` laid out as ``placement``, the
        DRAM bytes its sweep moves, and one grid step's ``matrix_cycles``.

        Args:
            mapping: The interstellar mapping to price, its loop orders
                included.
            placement: The ``Placement`` its tile takes: the bank groups the
                bus is priced per and the regions the reduction scratch
                takes.
        """
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        # Elements of one L3 tile: levels 0-2 only, since [3] is the grid trip
        # count, not part of the tile.
        input_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
        )
        weight_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
        )
        output_elems = (
            partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
        )

        lat = self.dram_access_latency_cycles

        def transfer(*sizes):
            """Cycles to move each of ``sizes`` as its own DMA: one fixed
            access latency apiece plus the bytes.  A microscaling operand's
            block scales are such a DMA -- a few hundred bytes, a whole
            latency."""
            sizes = [s for s in sizes if s]
            return len(sizes) * lat + sum(sizes) / self.dram_bandwidth

        input_sizes = (
            input_elems * self.input_dtype_width / 8,
            input_elems / self.scale_block_size * self.input_scale_width / 8,
        )
        weight_sizes = (
            weight_elems * self.weight_dtype_width / 8,
            weight_elems / self.scale_block_size * self.weight_scale_width / 8,
        )
        output_sizes = (
            output_elems * self.output_dtype_width / 8,
            output_elems / self.scale_block_size * self.output_scale_width / 8,
        )
        input_load = transfer(*input_sizes)
        weight_load = transfer(*weight_sizes)
        store = transfer(*output_sizes)

        # A tail operand spans output dims alone, so its count already runs
        # over the output steps -- the only ones that read it.
        tail_sizes = self.tail_tile_sizes(mapping)
        tail_dmas = [
            (
                transfer(size),
                batch_loads(
                    mapping,
                    dims,
                    self.batch,
                    self.batch if le.ON in dims else 1,
                ),
            )
            for (dims, _), size in zip(self.tail_specs, tail_sizes)
        ]

        bank_groups, scratch_slots = placement.groups, placement.scratch_slots
        matrix_cycles = self.matrix_cycles(mapping, bank_groups)
        vector_cycles = (
            self.vector_cycles(mapping, bank_groups) if self.has_tail else 0
        )

        input_steps = batch_loads(mapping, INPUT_DIMS, self.batch, self.batch)
        weight_steps = batch_loads(
            mapping, WEIGHT_DIMS, self.batch, self.weight_batch
        )

        # The mapping covers one batch element; the builder loops the rest.
        l3_blocks = self._l3_blocks(mapping) * self.batch
        num_k = blockings[le.IC][3]
        output_tiles = l3_blocks // num_k

        # Traffic the sweep moves, for a caller ranking by DRAM rather than
        # by time.
        dram_bytes = (
            input_steps * sum(input_sizes)
            + weight_steps * sum(weight_sizes)
            + output_tiles * sum(output_sizes)
            + sum(t * s for (_, t), s in zip(tail_dmas, tail_sizes))
        )

        dmas = [
            (store, output_tiles),
            (input_load, input_steps),
            (weight_load, weight_steps),
            *tail_dmas,
        ]

        if not self.double_buffered_l2:
            total_time = l3_blocks * matrix_cycles + sum(t * c for c, t in dmas)
            if num_k > 1 or self.single_k_tail_extra_pass:
                total_time += output_tiles * vector_cycles
            return Cost(total_time, dram_bytes, matrix_cycles)

        if num_k == 1:
            # Every step finishes a tile: one schedule covers the sweep.  A
            # riding tail drains inside the matrix pass, and an in-place one
            # overlaps the next tile's (its bank words are in the block);
            # a staged one is a pass of its own, serial on the single
            # scratch region.
            step = matrix_cycles
            if self.single_k_tail_extra_pass and not self.tail_keeps_shape:
                step += vector_cycles
            total_time = sweep_cycles(dmas, l3_blocks, step)
            return Cost(total_time, dram_bytes, matrix_cycles)

        load = input_load + weight_load
        # The sweep's last step has no tile after it to prefetch, so it costs
        # compute alone: hold it out of the count and let the epilogue charge
        # it, with the tail and store that drain behind it.
        accum_steps = l3_blocks - 2 * (output_tiles - 1) - 1
        classes = step_classes(tail_dmas, output_tiles)
        # Hold out the first output step in the same way -- the prologue fetches
        # its tail, with nothing running yet to hide it behind.  Taking what
        # that step owed leaves every tail fetch counted exactly once.
        first_tail = classes[-1][0]
        classes[-1] = (classes[-1][0], classes[-1][1] - 1)

        total_time = load + first_tail + accum_steps * max(load, matrix_cycles)
        # One window per remaining tile, spanning two grid steps: the matrix
        # unit finishes this tile and starts the next, while DRAM fits that
        # tile's tail read, its store and the next prefetch into the same span.
        # The busier side sets the price, and only the tail differs from one
        # window to the next -- hence one price per class.
        for tail, count in classes:
            prefetch = load + tail
            if self.split_k_tail_extra_pass and scratch_slots == 1:
                # The bare pass holds the control stream through both the
                # matrix and the vector pass, so the window's loads are
                # issued only then and nothing hides them.
                total_time += count * (
                    matrix_cycles + vector_cycles + prefetch + matrix_cycles
                )
                continue
            compute = max(matrix_cycles, prefetch) + matrix_cycles
            dma = max(matrix_cycles + vector_cycles, prefetch) + store + load
            total_time += count * max(compute, dma)
        total_time += matrix_cycles + vector_cycles + store
        return Cost(total_time, dram_bytes, matrix_cycles)


@dataclass(frozen=True)
class VoyagerGemm(RuntimeCalculator, OpModel):
    """A GEMM / conv on Voyager's PE array, mapped by interstellar: whether a
    tile fits the scratchpad banks (``fit``) and what a tiling costs (the
    ``RuntimeCalculator``'s ``cost``), both from widths and flags read once
    off the node (``from_node``).  Plain values only, so its search runs in
    a forked worker.

    Attributes:
        space: The ``MatrixSpace`` interstellar searches.
        key: What the answers depend on beyond the hardware: the layer
            (``layer_key``), its output dtype, the fused tail and its extra
            passes, the scratch regions and the outlier rates.
        anchor_dtype_width: Bits of the anchor's own dtype, which a partial
            sum and a parked finished tile take on chip.
        output_csr_data_width: Bits of an outlier value the CSR the fused
            tail emits stages, 0 without one.
        output_csr_index_width: Bits of that outlier's column index, 0
            without one.
        csr_data_width: Bits of an outlier value the consumed CSR gathers, 0
            without one.
        csr_index_width: Bits of that outlier's column index, 0 without one.
        consumes_csr: The GEMM consumes an outlier CSR, so its weight tile
            streams through the SpMM unit.
        outlier_pct: Fraction of the activation the consumed CSR holds.
        out_outlier_pct: Fraction of the output the emitted CSR may hold.
        scratch_regions: Regions a split reduction's read-back tail may give
            its scratch.
        accumulate_fp32: A split reduction accumulates in float32.
        num_slots: Banks a ping-ponged buffer takes, one per slot.
        bank_width: Bytes of a scratchpad store word, ``None`` without one.
        vector_lanes: Values one store beat writes.
        usable_scratchpad_size: Bytes the plan may spend on chip.
    """

    space: MatrixSpace
    key: tuple
    anchor_dtype_width: int
    output_csr_data_width: int
    output_csr_index_width: int
    csr_data_width: int
    csr_index_width: int
    consumes_csr: bool
    outlier_pct: float
    out_outlier_pct: float
    scratch_regions: int
    accumulate_fp32: bool
    num_slots: int
    bank_width: int | None
    vector_lanes: int
    usable_scratchpad_size: int

    @classmethod
    def from_node(cls, node, model):
        """The ``VoyagerGemm`` of ``node``, a GEMM / conv, bare or a fused
        ``call_module``, on ``model``'s accelerator.

        Args:
            node: The node being tiled.
            model: The ``VoyagerModel``.
        """
        config = model.config
        anchor = get_anchor_node(node)
        out_dtype = output_dtype(node)
        fused_specs = tuple(fused_operand_specs(node, anchor))
        has_tail = node.meta.get("submodule") is not None
        single_k_extra_pass, split_k_extra_pass = tail_extra_passes(node)
        keeps_shape = tail_keeps_shape(node)
        # A split reduction's extra pass reads the tile back from scratch; a
        # second region lets it ride the finalize commit, except in a
        # CSR-producing nest, whose bare stores pin one (``_SparseGemm``).
        scratch_regions = 1
        if split_k_extra_pass and csr_quantize_node(node) is None:
            scratch_regions = 2
        outlier_pct, out_outlier_pct, outlier_rate = outlier_stats(node)
        if isinstance(out_dtype, (list, tuple)):
            value_dtype = out_dtype[-1]
            output_scale_width = get_dtype_width(out_dtype[-2])
        else:
            value_dtype = out_dtype
            output_scale_width = 0
        # A CSR-producing tail returns (data, indices, indptr, scale,
        # inliers); a staged entry is an outlier value and its column index,
        # each its own buffer.
        emits_csr = isinstance(out_dtype, (list, tuple)) and len(out_dtype) == 5
        input_width = node_dtype_bits(anchor.args[0])
        anchor_width = get_dtype_width(anchor.value.dtype)
        # A scratchpad bank moves one store word per cycle, ``bank_width``
        # bytes: a port the hardware fixes, not one that widens with the
        # element.  So a row of elements wider than the port's lanes (int6
        # attention operands on the 4-bit NF4 port) takes more than one beat.
        # Without a bank width the port is one input row per cycle: the
        # ``min(unroll)`` elements the array's narrow side consumes.
        sram_bandwidth = (
            config.bank_width * 8
            if config.bank_width
            else min(config.pe_array_size) * input_width
        )
        layer = matrix_layer(anchor)
        key = layer_key(anchor) + (
            tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
            fused_specs,
            has_tail,
            tuple(str(target) for target in tail_targets(node)),
            single_k_extra_pass,
            split_k_extra_pass,
            keeps_shape,
            scratch_regions,
            outlier_pct,
            out_outlier_pct,
            outlier_rate,
        )
        return cls(
            input_dtype_width=input_width,
            weight_dtype_width=node_dtype_bits(anchor.args[1]),
            output_dtype_width=(
                get_dtype_width(value_dtype)
                if value_dtype
                else node_dtype_bits(anchor)
            ),
            accum_dtype_width=32 if model.accumulate_fp32 else anchor_width,
            double_buffered_accum_buffer=config.double_buffered_accum_buffer,
            sram_bandwidth=sram_bandwidth,
            dram_bandwidth=config.bytes_per_cycle,
            dram_access_latency_cycles=config.access_latency_cycles,
            double_buffered_l2=config.double_buffered_l2,
            outlier_rate=outlier_rate,
            batch=gemm_batch(anchor),
            weight_batch=weight_batch(anchor),
            has_tail=has_tail,
            single_k_tail_extra_pass=single_k_extra_pass,
            split_k_tail_extra_pass=split_k_extra_pass,
            tail_keeps_shape=keeps_shape,
            tail_specs=fused_specs,
            input_scale_width=node_dtype_bits(
                anchor.kwargs.get("input_scale"), 0
            ),
            weight_scale_width=node_dtype_bits(
                anchor.kwargs.get("weight_scale"), 0
            ),
            output_scale_width=output_scale_width,
            scale_block_size=anchor.kwargs.get("block_size") or 1,
            bias_width=node_dtype_bits(
                get_arg_value(anchor, 2, "bias", None), 0
            ),
            stride=(layer.hstd, layer.wstd),
            bank_size=config.bank_size,
            weight_transposed=weight_transforms(anchor.args[1])[1],
            space=model.matrix_space,
            key=key,
            anchor_dtype_width=anchor_width,
            output_csr_data_width=(
                get_dtype_width(out_dtype[0]) if emits_csr else 0
            ),
            output_csr_index_width=(
                get_dtype_width(out_dtype[1]) if emits_csr else 0
            ),
            csr_data_width=node_dtype_bits(anchor.kwargs.get("A_data"), 0),
            csr_index_width=node_dtype_bits(anchor.kwargs.get("A_indices"), 0),
            consumes_csr=anchor.kwargs.get("A_data") is not None,
            outlier_pct=outlier_pct,
            out_outlier_pct=out_outlier_pct,
            scratch_regions=scratch_regions,
            accumulate_fp32=model.accumulate_fp32,
            num_slots=config.num_slots,
            bank_width=config.bank_width,
            vector_lanes=config.vector_lanes,
            usable_scratchpad_size=config.usable_scratchpad_size,
        )

    def fit(self, mapping):
        """The placement of ``mapping``'s scratchpad (L2) tile, or ``None``
        when it does not fit the usable scratchpad.

        Every operand is sized the way ``tensor_alloc_bytes`` sizes it for
        the planner -- payload, the slack a final store beat overshoots by,
        aligned to ``bank_width`` -- block scales (one per
        ``scale_block_size`` values), the bias and the fused tail operands
        included, and each operand of a shared bank is padded before they
        are summed, because the planner lays them out one after another.
        Operands are grouped one per bank ideally:

            input+csr | input_scale | weight+weight_scale+bias
                      | output+output_scale+csr staging | each fused operand

        An outlier CSR the GEMM consumes rides the input's bank -- both are
        DMA-written and compute-read at the same depth -- and the CSR a fused
        tail emits stages in the output's, beside the dense pair it stores by
        the same route.  Each group is ping-ponged unless a single tile
        serves the whole sweep, its slots in ``num_slots`` separate banks, so
        a load overlaps the compute reading the other slot.

        A split reduction with a fused tail, or accumulating in float32, also
        accumulates into a scratch buffer the builders allocate exactly once
        (``_ScratchSpec``), charged whole bank-aligned regions on top; so is
        the finished tile a tail that needs a pass of its own parks even for
        a single round, unless it keeps the tile's shape and runs that pass
        in place on the output slot.  A split reduction whose tail reads that
        tile back from scratch takes ``scratch_regions`` regions -- a second
        lets the read-back pass ride the finalize commit -- when they fit
        beside the groups, and one otherwise.

        A bank cannot be split between groups, so each group rounds up to
        whole banks, which floors a tile at ``num_slots * len(groups)``
        banks however small it is.  While the groups' banks exceed what the
        scratch leaves, the two smallest merge, so a tile is admitted with
        the least sharing that fits it; only whole groups merge, never a
        group's own slots.  Without banks the groups are summed as they are.

        Reads only the tile's extents through L2 (``total_extent``) and the
        blockings above the PE array, which no matrix space partitions, never
        a loop order or how the PE level splits into steps and copies.

        Returns:
            The groups' members, one frozenset per bank group and the
            scratch's last, with the regions the scratch takes;
            ``Placement()`` without banks.  ``None`` when the tile does not
            fit, or its block scales overflow the SpMM unit's buffer.
        """
        blockings = mapping.loop_blockings
        extent = partial(total_extent, mapping, level=2)
        # TEMPORARY WORKAROUND: veto a tile whose block scales overflow the
        # SpMM unit's fixed 32-row buffer (the RTL wraps and applies the
        # wrong scales).  Drop once the depth is an accelerator parameter.
        # The weight tile takes a row per K block (its L1 and L2 IC
        # blockings; the PE level is the array) per L1 OC pass, ``C * K0`` in
        # the toolchain's ``SpMM.h``.
        scale_rows = (
            blockings[le.IC][1] * blockings[le.IC][2] * blockings[le.OC][1]
        )
        if self.consumes_csr and scale_rows > SPMM_SCALE_ROWS:
            return None

        if_count, of_count, fl_count = operand_elements(mapping, 2, self.stride)
        # The output is a partial sum in the anchor's dtype while IC is split
        # across L3 steps; only the final value carries an output scale.
        split = blockings[le.IC][3] > 1
        out_bits = self.anchor_dtype_width if split else self.output_dtype_width
        out_scale = (
            0.0
            if split
            else self._scale_bytes(of_count, self.output_scale_width)
        )
        bias = self._alloc_bytes(extent(le.OC), self.bias_width)
        # A CSR the tail emits stages in the output's bank before its packed
        # stores: a block's (value, index) pairs and its row pointers.  The
        # quantize sees the output tile, so the budget follows it.
        staged_csr = 0.0
        if self.output_csr_data_width:
            rows = of_count / max(1.0, extent(le.OC))
            staged = of_count * self.out_outlier_pct
            staged_csr = (
                self._alloc_bytes(staged, self.output_csr_data_width)
                + self._alloc_bytes(staged, self.output_csr_index_width)
                + self._alloc_bytes(rows + 1, 32)
            )

        def slots(dims, distinct):
            """Banks an operand spanning ``dims`` needs: a second holds the
            next tile, so an operand with one tile over the whole sweep needs
            one.  The mapping covers a single batch element -- the builder
            loops the rest -- so the sweep's tile count is the L3 trips times
            ``distinct``, the operand's distinct tiles over that batch loop
            (``batch_loads`` counts fetches the same way).  The builders'
            ``PipelinedKernel._num_slots`` resolves the depth from the full
            grid, so this never charges less than is allocated."""
            tiles = distinct
            for d in dims:
                tiles *= blockings[d][3]
            return self.num_slots if tiles > 1 else 1

        # The gathered CSR shares the input's bank: a block's (value, index)
        # pairs, at the fraction of the activation the stream is sized for.
        # Its staging ping-pongs exactly when the input does -- a sweep that
        # gathers once keeps one slot (``_SparseGemm.forward``).
        gathered = if_count * self.outlier_pct
        groups = [
            (
                self._alloc_bytes(if_count, self.input_dtype_width)
                + self._alloc_bytes(gathered, self.csr_data_width)
                + self._alloc_bytes(gathered, self.csr_index_width),
                slots(INPUT_DIMS, self.batch),
                frozenset({Role.INPUT, Role.CSR}),
            ),
            (
                self._scale_bytes(if_count, self.input_scale_width),
                slots(INPUT_DIMS, self.batch),
                frozenset({Role.INPUT_SCALE}),
            ),
            (
                self._alloc_bytes(fl_count, self.weight_dtype_width)
                + self._scale_bytes(fl_count, self.weight_scale_width)
                + bias,
                slots(WEIGHT_DIMS, self.weight_batch),
                frozenset({Role.WEIGHT, Role.WEIGHT_SCALE, Role.BIAS}),
            ),
            (
                self._alloc_bytes(of_count, out_bits) + out_scale + staged_csr,
                slots(OUTPUT_DIMS, self.batch),
                frozenset({Role.OUTPUT}),
            ),
        ]
        for i, (dims, bits) in enumerate(self.tail_specs):
            elements = math.prod(extent(d) for d in dims)
            distinct = self.batch if le.ON in dims else 1
            groups.append(
                (
                    self._alloc_bytes(elements, bits),
                    slots(dims, distinct),
                    frozenset({Fused(i)}),
                )
            )

        # An absent operand (no scale, no bias) occupies no bank.
        groups = [group for group in groups if group[0] > 0]

        # A reduction split across L3 steps accumulates into a scratch buffer
        # the builders allocate once for the whole kernel, not per ping-pong
        # half -- so it is charged outside ``num_slots``.  One accumulating
        # in float32 always has it, since the output slot holds the narrower
        # result.  A shape-changing extra pass parks even a single round's
        # tile.
        if split and (self.has_tail or self.accumulate_fp32):
            scratch = self._alloc_bytes(of_count, self.accum_dtype_width)
        elif (
            self.has_tail
            and self.single_k_tail_extra_pass
            and not self.tail_keeps_shape
        ):
            scratch = self._alloc_bytes(of_count, self.anchor_dtype_width)
        else:
            scratch = 0.0
        # Only a split reduction's read-back tail has a second region to
        # gain; a single staged round keeps one.
        regions = self.scratch_regions if scratch and split else 1

        if not self.bank_size:
            needed = sum(slot_count * size for size, slot_count, _ in groups)
            if needed + regions * scratch > self.usable_scratchpad_size:
                return None
            return Placement()

        bank_size = self.bank_size
        num_banks = self.usable_scratchpad_size // bank_size
        scratch_banks = math.ceil(scratch / bank_size) if scratch else 0

        def banks(groups):
            return sum(
                slot_count * math.ceil(size / bank_size)
                for size, slot_count, _ in groups
            )

        def merge_to_fit(budget):
            """The groups with their two smallest merged until their banks
            fit ``budget``, and whether they do -- one group may still not."""
            merged = list(groups)
            while len(merged) > 1 and banks(merged) > budget:
                merged.sort(key=lambda group: group[0])
                size0, slot_count0, members0 = merged[0]
                size1, slot_count1, members1 = merged[1]
                # One buffer holding both has to ping-pong if either of them
                # does.
                merged = [
                    (
                        size0 + size1,
                        max(slot_count0, slot_count1),
                        members0 | members1,
                    )
                ] + merged[2:]
            return merged, banks(merged) <= budget

        # Banks left for the ping-ponged groups, after the scratch regions.
        # A second scratch region is kept only when the groups still fit
        # beside it.
        while True:
            budget = num_banks - regions * scratch_banks
            merged, fits = merge_to_fit(budget)
            if fits or regions == 1:
                break
            regions -= 1

        needed = sum(
            slot_count * math.ceil(size / bank_size) * bank_size
            for size, slot_count, _ in merged
        )
        if scratch:
            needed += regions * math.ceil(scratch / bank_size) * bank_size
        if needed > self.usable_scratchpad_size:
            return None
        partition = [members for _, _, members in merged]
        if scratch:
            partition.append(frozenset({Role.SCRATCH}))
        return Placement(tuple(partition), scratch_slots=regions)

    def _align(self, size):
        """``size`` rounded up to a whole ``bank_width`` store word."""
        if not self.bank_width:
            return size
        return math.ceil(size / self.bank_width) * self.bank_width

    def _alloc_bytes(self, count, bits):
        """Bytes an on-chip buffer of ``count`` ``bits``-wide elements takes.

        Mirrors ``tensor_alloc_bytes``, which is what the memory planner
        allocates through: the payload, plus the slack a final store beat of
        ``vector_lanes`` values overshoots the payload by, aligned to a whole
        store word.
        """
        if not bits or count <= 0:
            return 0.0
        size = math.ceil(count * bits / 8.0)
        if self.bank_width and self.vector_lanes:
            beat = math.ceil(self.vector_lanes * bits / 8.0)
            size += self._align(beat) - beat
        return float(self._align(size))

    def _scale_bytes(self, count, bits):
        """Bytes the scales of ``count`` values take: one per block."""
        return self._alloc_bytes(count / self.scale_block_size, bits)
