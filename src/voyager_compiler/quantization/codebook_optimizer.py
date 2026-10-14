"""Codebook optimization: fit a lookup table's entries to a tensor.

A lookup-table dtype ``lut<I>_to_<E>`` stores ``I``-bit indices into
``2**I`` entries of dtype ``E``, so the stored element keeps its width and
only the levels it decodes to change.  It applies not to a tensor's raw
values but to ``x / blockscale`` under microscaling, so every routine below
works in that normalized space.  The levels are drawn from a set of
candidates -- every value the entry dtype holds in the range, or an even
spread across it when the entries keep the model's dtype -- and nothing here
assumes their number or their spacing.

``optimal_codebooks`` fits the levels to measured samples: the exact
weighted-MSE minimum over each accumulated ``Histogram``, by dynamic
programming.  Weighting each sample by its channel's downstream energy is
what makes it minimize layer-output error rather than tensor MSE.  Squared
error alone is blind to zero and will spend that level elsewhere, which
under block scaling forces a block's many small values to at least one step
off zero -- pass ``pin_zero`` to forbid it.
"""

import contextlib
import enum
import functools
import itertools
import json
import logging
import math
import os
from typing import NamedTuple

import torch

from voyager_compiler.export_utils import get_node_name_to_scope
from voyager_compiler.ops import calculate_mx_qparam, expand
from voyager_compiler.quantization.fake_quantize import (
    MXFakeQuantize,
    _rotate,
    calculate_global_scale,
    entry_levels,
)
from voyager_compiler.quantization.training import (
    forward_gemm_operands,
    gemm_partners,
    operand_gradient,
)
from voyager_compiler.shape_prop import fetch_attr

logger = logging.getLogger(__name__)

EPS = 1e-12

#: Elements one ``Histogram.add`` converts to float64 at a time.  Widening
#: an operand and forming the two products is the accumulator's whole
#: memory cost, and a model's largest weight runs to half a billion
#: elements, so the pass is taken in slices rather than whole.
ACCUMULATE_CHUNK = 1 << 25

#: Copies of each bin ``bin_blocks`` spreads its adds over on a GPU.
#: Neighbouring elements add into different copies, so the many elements of
#: one bin -- a ReLU output's zeros -- do not queue on one address; the
#: copies are summed after.
BIN_COPIES = 512

#: Float64 elements one round of a batched solve holds, per device type: a
#: chunk of histograms times the candidate pairs each searches.  On a CPU
#: about 16 MiB, so a round stays in cache; a round is bandwidth-bound
#: there, and a larger chunk is slower.  On a GPU 128 MiB, which with the
#: temporaries the gap tables are built through peaks near 1 GiB.
SOLVE_CHUNK = {"cpu": 1 << 21, "cuda": 1 << 24}


@functools.lru_cache(maxsize=None)
def _bin_grid(levels, device):
    """Return the bins over a set of candidates, built once per device.

    Args:
        levels: The candidate levels, ascending, as a tuple.
        device: Where the tensors live.

    Returns:
        ``(edges, midpoint_bins)``: the ascending float64 bin boundaries,
        and ``midpoint_bins[a, b]``, the bin the midpoint of candidates
        ``a`` and ``b`` starts.
    """
    if device.type != "cpu":
        cpu = _bin_grid(levels, torch.device("cpu"))
        return tuple(tensor.to(device) for tensor in cpu)
    level = torch.tensor(levels, dtype=torch.float64)
    midpoints = (level[:, None] + level[None, :]) / 2
    edges = midpoints.reshape(-1).unique()
    midpoint_bins = torch.searchsorted(edges, midpoints, right=True) - 1
    return edges, midpoint_bins.clamp_(0, edges.numel() - 1)


@functools.lru_cache(maxsize=None)
def _search_edges(levels, device):
    """Return the interior bin edges as ``bin_blocks`` searches them.

    Each edge is the smallest float32 at or above the exact one.  The values
    searched are float32, so a value reaches that edge exactly when it
    reaches the exact one, and is binned as ``Histogram.add`` bins it.

    Args:
        levels: The candidate levels, ascending, as a tuple.
        device: Where the tensor lives.

    Returns:
        The edges between the first and last, as float32, padded with +inf
        to a power of two above their count.
    """
    edges = _bin_grid(levels, torch.device("cpu"))[0][1:-1]
    inner = edges.float()
    up = torch.nextafter(inner, torch.full_like(inner, torch.inf))
    inner = torch.where(inner.double() < edges, up, inner)
    pad = torch.full(
        ((1 << inner.numel().bit_length()) - inner.numel(),), torch.inf
    )
    return torch.cat([inner, pad]).to(device)


class Histogram:
    """Weighted moments of one tensor's normalized values, per bin.

    The exact solver reduces whatever it is handed to three sums per bin --
    total weight, weighted sum, weighted sum of squares -- and reads nothing
    else.  Accumulating those directly over every element is therefore not
    an approximation of sampling but a replacement for it: no draw to vary
    between runs, no bound on how much data one fit may see, and a few
    kilobytes of state per tensor instead of gigabytes of retained samples.

    A value rounds to its nearest level, so it switches level halfway
    between two.  The bins have an edge at the midpoint of every pair of
    candidates, which leaves no bin straddling a switch wherever the solver
    puts the levels, and so makes its cost exact on any grid.  Those are
    the only edges, since the solver reads nothing between them.  The bins
    depend on the candidates alone, so histograms over the same ones share
    them.

    Args:
        levels: The candidate levels, ascending.

    Attributes:
        levels: The candidate levels, ascending, as a float64 tensor.
        quant_min: The lowest candidate, and the low edge.
        quant_max: The highest candidate, and the high edge.
        bins: How many bins the midpoint edges make.
    """

    def __init__(self, levels):
        self.levels = torch.as_tensor(levels, dtype=torch.float64).cpu()
        self.quant_min = float(self.levels[0])
        self.quant_max = float(self.levels[-1])
        self._candidates = tuple(self.levels.tolist())
        self.bins = self.edges().numel() - 1
        self.weight = None
        self.total = None
        self.square = None

    def edges(self, device="cpu"):
        """Return the bin boundaries, on ``device``."""
        return _bin_grid(self._candidates, torch.device(device))[0]

    def midpoint_bins(self, device="cpu"):
        """Return ``[a, b]``, the bin candidates a and b's midpoint starts."""
        return _bin_grid(self._candidates, torch.device(device))[1]

    def search_edges(self, device):
        """Return the interior edges as ``bin_blocks`` searches them."""
        return _search_edges(self._candidates, torch.device(device))

    def sums(self, device):
        """Return the ``[3, bins]`` sums, created on ``device`` at first use."""
        if self.weight is None:
            self.moments = torch.zeros(
                3, self.bins, dtype=torch.float64, device=device
            )
            self.weight, self.total, self.square = self.moments
        return self.moments

    def add(self, values, weights):
        """Accumulate one tensor's normalized values into the bins.

        Args:
            values: Flat tensor of ``x / blockscale``.
            weights: Per-element weight, broadcastable to ``values``.
        """
        self.sums(values.device)
        edges = self.edges(values.device)
        values = values.reshape(-1)
        weights = weights.reshape(-1).expand_as(values)
        for start in range(0, values.numel(), ACCUMULATE_CHUNK):
            # Float64 so that a value is binned against the exact midpoints,
            # and so that the order the bins are summed in -- which on a GPU
            # is decided by the scheduler -- cannot move a level.  That same
            # indifference to order is what lets the pass be sliced.
            piece = values[start : start + ACCUMULATE_CHUNK].double()
            share = weights[start : start + ACCUMULATE_CHUNK].double()
            index = (torch.bucketize(piece, edges, right=True) - 1).clamp_(
                0, self.bins - 1
            )
            self.weight.index_add_(0, index, share)
            self.total.index_add_(0, index, share * piece)
            self.square.index_add_(0, index, share * piece * piece)

    def cumulative(self):
        """Return the three running sums the segment cost is read from.

        Each is led by a zero and stays on the device the sums are on.
        """
        return tuple(
            torch.nn.functional.pad(moment.cumsum(0), (1, 0))
            for moment in (self.weight, self.total, self.square)
        )


#: Candidates across the range when the entries keep the model's dtype and
#: so have no grid of their own.  513 spaces them a 256th of the top value
#: apart, bf16's own spacing just below a power-of-two top, so each is exact
#: in bf16 and none merge; an odd count keeps an exact zero for ``pin_zero``.
FLOAT_RESOLUTION = 513


#: Tables one round of ``codebook_qmaps`` builds.  Each holds a 65536-entry
#: index while it is built, about 1 MiB with its temporaries, so a round
#: stays near 128 MiB however many tables a refit installs.
QMAP_CHUNK = 128


def codebook_qmaps(tables, device=None):
    """Build the 65536-entry bf16 lookup table a fake-quant consumes, per table.

    A value maps to its nearest entry, and one halfway between two to the
    lower, so the entry it takes is the number of midpoints below it.  A NaN
    bit pattern maps to the lowest entry.

    Args:
        tables: Codebooks of the same size, each ascending.
        device: Where to build the tables.

    Returns:
        A ``[tables, 65536]`` bf16 tensor mapping every bf16 bit pattern to
        each table's nearest entry.
    """
    codebooks = torch.tensor(tables, dtype=torch.float32, device=device)
    midpoints = (codebooks[:, :-1] + codebooks[:, 1:]) / 2
    every = (
        torch.arange(2**16, dtype=torch.int16, device=device)
        .view(torch.bfloat16)
        .float()
    )
    every = torch.where(every.isnan(), -torch.inf, every)
    qmaps = []
    for start in range(0, len(codebooks), QMAP_CHUNK):
        chunk = midpoints[start : start + QMAP_CHUNK]
        values = every.expand(len(chunk), -1).contiguous()
        index = torch.searchsorted(chunk, values)
        entries = codebooks[start : start + QMAP_CHUNK].gather(1, index)
        qmaps.append(entries.to(torch.bfloat16))
    return torch.cat(qmaps)


def optimal_codebooks(histograms, k, pin_zero=True):
    """Return each histogram's globally optimal codebook.

    Levels are ``k`` distinct candidates from the histograms' ``levels``,
    the lowest and highest among them.  Every value between two consecutive
    levels rounds to the nearer of the two, so the weighted squared error is
    a sum of one cost per gap that depends on the gap's two ends alone.  The
    exact minimizer is therefore a shortest path over the levels, found by
    dynamic programming -- no initialization, no local minima.  The
    histograms have a bin edge at every candidate, so a gap's cost splits
    between its ends without error.  Pinning zero forbids every gap that
    straddles it, so each round of the search is two independent blocks,
    the levels below zero and those above.  The histograms are searched
    together, in chunks of ``SOLVE_CHUNK`` elements, on the device their
    sums are on.

    Args:
        histograms: Accumulated weighted moments of tensors'
            ``x / blockscale`` values, all over the same candidates.  A
            range starting at zero already pins the zero level, so
            ``pin_zero`` has nothing left to do and is ignored.
        k: Number of levels.
        pin_zero: Force the level ``0`` into each codebook.  Minimizing
            squared error alone almost always spends that level
            elsewhere, which leaves a block's many small values unable
            to quantize to zero.

    Returns:
        One optimal codebook per histogram, each ``k`` ascending distinct
        levels, integers when every candidate is one.

    Raises:
        ValueError: Fewer candidates than levels asked for, so no codebook
            of this size exists on this grid.
        RuntimeError: A shortest path did not reconstruct to ``k``
            distinct levels, which is a defect in the search rather than a
            property of the data.
    """
    levels = histograms[0].levels
    n = levels.numel()
    if k > n:
        raise ValueError(
            f"a {k}-level codebook cannot be drawn from the {n} candidates "
            f"in [{histograms[0].quant_min}, {histograms[0].quant_max}]"
        )
    # A codebook of integers reads as the entries the hardware stores.
    integral = torch.equal(levels, levels.round())
    if k <= 2:
        # Both endpoints are pinned, so there is nothing left to choose.
        ends = levels[[0, -1]][:k]
        ends = (ends.long() if integral else ends).tolist()
        return [list(ends) for _ in histograms]
    zero = int(levels.abs().argmin())
    # Pinning is meaningless when zero is already the lowest level, since
    # that one is always chosen, or when the grid holds no exact zero.
    if pin_zero and zero != 0 and levels[zero] == 0:
        # (upper levels, lower levels) each block's gaps run between.
        blocks = [
            (slice(0, zero + 1), slice(0, zero + 1)),
            (slice(zero + 1, n), slice(zero, n)),
        ]
    else:
        blocks = [(slice(0, n), slice(0, n))]
    pairs = sum(
        (upper.stop - upper.start) * (lower.stop - lower.start)
        for upper, lower in blocks
    )
    device = histograms[0].weight.device
    chunk = max(1, SOLVE_CHUNK[device.type] // pairs)
    paths = torch.cat(
        [
            _shortest_paths(histograms[start : start + chunk], k, blocks)
            for start in range(0, len(histograms), chunk)
        ]
    )
    if not (paths.diff(dim=1) > 0).all():
        raise RuntimeError(
            f"the search returned a {k}-level codebook with a repeated level"
        )
    codebooks = levels.to(device)[paths]
    return (codebooks.long() if integral else codebooks).tolist()


def _shortest_paths(histograms, k, blocks):
    """The candidates each histogram's optimal codebook takes, by index.

    Every tensor below carries one leading row per histogram, so the
    histograms are searched together on the device their sums are on.

    Args:
        histograms: Histograms over the same candidates.
        k: Number of levels.
        blocks: ``(upper, lower)`` slices of candidates, covering every
            candidate once as an upper level: a gap may run from a lower
            level to an upper one only within a block.

    Returns:
        ``[histograms, k]`` ascending candidate indices.
    """
    device = histograms[0].weight.device
    cum_w, cum_s, cum_q = (
        torch.stack(moment)
        for moment in zip(*(histogram.cumulative() for histogram in histograms))
    )
    level = histograms[0].levels.to(device)
    at = histograms[0].midpoint_bins(device)
    # The bin each candidate starts: the midpoint of it with itself.
    own = at.diagonal()
    n = level.numel()
    rows = torch.arange(len(histograms), device=device)

    def segment_cost(lo_bin, hi_bin, level):
        span_w = cum_w[:, hi_bin] - cum_w[:, lo_bin]
        span_s = cum_s[:, hi_bin] - cum_s[:, lo_bin]
        span_q = cum_q[:, hi_bin] - cum_q[:, lo_bin]
        return span_q - 2 * level * span_s + level * level * span_w

    # gap[:, b, a]: cost of the samples between levels a < b, each rounded
    # to the nearer of the two.  The lower level runs along the last axis, so
    # each round's minimum over it reads contiguous memory.
    gaps = []
    for upper, lower in blocks:
        middle = at[upper, lower]
        to_lower = segment_cost(own[None, lower], middle, level[None, lower])
        to_upper = segment_cost(middle, own[upper, None], level[upper, None])
        ascending = level[None, lower] < level[upper, None]
        gaps.append(torch.where(ascending, to_lower + to_upper, torch.inf))

    # state[:, c]: cost of the cheapest codebook so far whose highest level
    # is c, counting the samples up to c.
    state = torch.full(
        (len(histograms), n), torch.inf, dtype=torch.float64, device=device
    )
    state[:, 0] = segment_cost(0, int(own[0]), level[0])
    back = []
    for _ in range(k - 1):
        reached = torch.empty_like(state)
        choice = torch.empty_like(state, dtype=torch.long)
        for (upper, lower), gap in zip(blocks, gaps):
            cost, source = (state[:, None, lower] + gap).min(dim=2)
            reached[:, upper] = cost
            choice[:, upper] = source + lower.start
        state = reached
        back.append(choice)

    # The highest candidate ends every codebook, so the samples above it
    # cost the same on every path and are left out.
    cur = torch.full_like(rows, n - 1)
    chosen = [cur]
    for choice in reversed(back):
        cur = choice[rows, cur]
        chosen.append(cur)
    return torch.stack(chosen[::-1], dim=1)


def _contraction_last(tensor, ch_axis):
    """Return the tensor with its contraction axis last.

    Args:
        tensor: The operand being quantized.
        ch_axis: Axis the microscaling blocks run along, as the spec gives
            it -- an int, or a one-element sequence.

    Returns:
        The tensor, transposed if its blocks do not already run along the
        last axis.
    """
    axis = ch_axis[0] if isinstance(ch_axis, (tuple, list)) else ch_axis
    if axis in (-1, tensor.dim() - 1):
        return tensor
    return tensor.transpose(axis, -1)


def _channel_energy(tensor, ch_axis):
    """Return the mean square of each channel along the contraction axis.

    This is what turns a tensor-MSE fit into an output-error fit: an error
    on channel ``j`` reaches the output scaled by the *other* operand's
    magnitude on ``j``, so samples are weighted by that.

    Args:
        tensor: The partner operand.
        ch_axis: Axis its microscaling blocks run along.

    Returns:
        One non-negative number per channel.
    """
    flat = _contraction_last(tensor.detach().float(), ch_axis)
    return flat.reshape(-1, flat.shape[-1]).pow(2).mean(0)


#: Ops that contract their first two arguments against each other, mapped
#: to the axis each of the two positions contracts along.  Only there is a
#: codebook's job defined: the block runs along the contraction axis, and
#: the error a level makes reaches the output scaled by the other operand
#: on the shared axis.  The quantizer also annotates elementwise ops -- a
#: residual add, a softmax -- whose operands have neither.  Any remaining
#: argument is a bias or a stride, not an operand.
CONTRACTED_OPS = {
    torch.ops.aten.linear.default: (-1, -1),
    torch.ops.aten.matmul.default: (-1, -2),
    torch.ops.aten.conv1d.default: (1, 1),
    torch.ops.aten.conv2d.default: (1, 1),
}


def _quantized_operands(model, skip):
    """Find every tensor a prepared model quantizes through a codebook.

    Enumerating the fake-quants is what makes the search exhaustive: a
    module whose dtype keeps its levels in a table is one this can fit, and
    no traversal of compute nodes can miss one.  The op reading the tensor
    is consulted only to name it and to find the partner whose channel
    energy weights the fit.

    Args:
        model: Graph module from ``prepare_pt2e``.
        skip: Substrings of a name that exclude it.

    Returns:
        ``name -> (fake-quant module, operand node, partner node)``.
        Operands sharing a fake-quant appear once, under the first name
        found: the table belongs to the module, so they cannot hold
        different ones.
    """
    modules = dict(model.named_modules())
    scopes = get_node_name_to_scope(model)
    nodes = list(model.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}

    # Which call of an op within its module a node is -- what tells Q @ K^T
    # from P @ V, neither of which has a weight to be named after.
    order, seen = {}, {}
    for node in nodes:
        if node.target in CONTRACTED_OPS:
            scope = scopes.get(node.name, ("", None, 0))[0]
            order[node] = seen.get((scope, node.target), 0)
            seen[(scope, node.target)] = order[node] + 1

    found, shared, loose = {}, set(), []
    for node in nodes:
        if node.op != "call_module":
            continue
        module = modules.get(node.target)
        if not getattr(module, "is_codebook_quantization", False):
            continue
        consumers = [
            user
            for user in sorted(node.users, key=position.__getitem__)
            if user.target in CONTRACTED_OPS and node in user.args[:2]
        ]
        if not consumers:
            loose.append(node.target)
            continue
        consumer = consumers[0]
        pair = consumer.args[:2]
        index = pair.index(node)
        scope = scopes.get(consumer.name, ("", None, 0))[0]
        name = _operand_name(scope, consumer, order[consumer], index)
        if any(excluded in name for excluded in skip) or module in shared:
            continue
        shared.add(module)
        # One tensor can feed several ops -- a normalized hidden state
        # reaches every projection reading it -- and its error reaches the
        # output through all of them, so every partner is kept.
        partners = tuple(
            user.args[:2][1 - user.args[:2].index(node)] for user in consumers
        )
        found[name] = (module, node, partners)

    if loose:
        logger.warning(
            "%d codebooks sit on a tensor no contracted op reads, so there "
            "is no contraction axis to block along and no partner to weight "
            "by, and they are left at their seed: %s%s",
            len(loose),
            ", ".join(loose[:3]),
            " ..." if len(loose) > 3 else "",
        )
    return found


#: How the two operands of an attention matmul are named, by which matmul
#: in the module it is and which side of it: Q @ K^T then P @ V.
ATTENTION_ROLES = {(0, 0): "q", (0, 1): "k", (1, 0): "p", (1, 1): "v"}


def _operand_name(scope, consumer, order, position):
    """Name one quantized operand, stably across runs.

    A linear's or conv's operands are the activation and the weight, so they
    take the module's own path.  A matmul of two activations has no weight to
    name them after, so it takes the enclosing module's path and its
    position.

    Args:
        scope: Module path export recorded for the consuming node.
        consumer: The contracted op reading the operand.
        order: Which call of that op within the module it is.
        position: Which of the consumer's two operands this is, 0 or 1.

    Returns:
        The operand's name, of the form ``<module path>|<role>``.
    """
    if consumer.target is not torch.ops.aten.matmul.default:
        return f"{scope}|{'act' if position == 0 else 'wgt'}"
    role = ATTENTION_ROLES.get((order, position))
    return f"{scope}|{role or f'mm{order}.{position}'}"


def _global_scale(module, tensor):
    """Return ``tensor``'s global scale as ``module``'s forward computes it.

    It is taken over the whole operand, so a caller that splits the operand
    into grid cells computes it once and hands the same scale to every cell.

    Args:
        module: The fake-quant.
        tensor: The whole operand, as ``module.prepare`` returns it.

    Returns:
        A 0-d tensor, or None when the module has no global scale.
    """
    if not getattr(module, "global_scale", False):
        return None
    return calculate_global_scale(tensor, module.quant_max, module.scale_max)


def _accumulate(
    tensor,
    module,
    energies,
    histograms,
    counts,
    global_scale,
    quantized_scale=True,
):
    """Bin one operand's normalized values into each grid cell's histogram,
    weighted by its partner.

    Every element is counted, so what the fit sees is the tensor itself
    rather than a draw from it.  Each element is also weighted by its
    block's squared scale: the levels are fitted in normalized space, but
    deployment multiplies the residual back by the block scale, so a
    block's share of the output error grows with that scale squared.  An
    outlier arrives as a zero and is binned as one: the pinned zero level
    quantizes it without error, so it adds nothing to any codebook's cost.
    On a GPU the whole operand, every cell of it, runs as one kernel,
    ``bin_blocks``, to the same bins.

    Args:
        tensor: The operand as its fake-quant quantizes it -- rotated, its
            outliers zeroed -- as ``module.prepare`` returns it.
        module: That fake-quant, which knows the block geometry.
        energies: Each cell's partner energy per channel, or None for a
            flat fit.
        histograms: Each cell's histogram, in ``_grid_cells`` order.
        counts: Cells along each axis, as ``_resolve_counts`` gives them.
        global_scale: The whole operand's global scale, from
            ``_global_scale``, or None.  A quantized scale divides it out
            before the scale codebook and multiplies it back after, as
            deployment does; the exact scale has no use for it.
        quantized_scale: Normalize by the scale deployment can store --
            the block maximum pushed through the scale codebook, and its
            power-of-two floor when the module forces one -- rather than
            the exact ``amax / quant_max``.  The decoder divides by the
            former, so the latter places the levels against values it
            never sees.
    """
    cells = _grid_cells(tensor.detach(), counts)
    first = _contraction_last(cells[0][1], module.ch_axis)
    if first.numel() < module.block_size:
        return
    channels = first.shape[-1]
    if energies is not None and any(e.numel() != channels for e in energies):
        raise ValueError(
            f"the partner energy has {energies[0].numel()} channels but the "
            f"operand contracts over {channels}; a table contracted over "
            f"tokens needs every step padded to one length"
        )
    if not quantized_scale:
        global_scale = None
    power_of_two = quantized_scale and getattr(module, "power_2_scale", False)
    scale_qmap = (
        getattr(module, "scale_qmap", None) if quantized_scale else None
    )
    if tensor.is_cuda:
        # Deferred: triton ships only with the CUDA builds of torch.
        from voyager_compiler.quantization.histogram_kernel import bin_blocks

        axis = module.ch_axis
        bin_blocks(
            tensor.detach(),
            axis[0] if isinstance(axis, (tuple, list)) else axis,
            module.block_size,
            module.quant_max,
            None if energies is None else torch.stack(energies),
            histograms[0].search_edges(tensor.device),
            [histogram.sums(tensor.device) for histogram in histograms],
            counts,
            BIN_COPIES,
            power_of_two,
            scale_qmap,
            global_scale,
        )
        return
    if energies is None:
        energies = [None] * len(cells)
    for (_, slab), energy, histogram in zip(cells, energies, histograms):
        ordered = _contraction_last(slab, module.ch_axis).float()
        # One scale serves both the normalization and the squared-scale
        # weight, so the two halves of the objective refer to the same
        # number -- the one deployment computes: its blocks pad out along
        # the contraction axis rather than running past it.
        scale = calculate_mx_qparam(
            ordered,
            axes=-1,
            block_size=module.block_size,
            quant_max=module.quant_max,
            force_scale_power_of_two=power_of_two,
            scale_qmap=scale_qmap,
            global_scale=global_scale,
        )
        if global_scale is not None:
            scale = scale * global_scale
        scale = expand(scale, ordered.shape, module.block_size)
        normalized = (ordered / scale).reshape(-1)
        if energy is None:
            weights = torch.ones_like(normalized)
        else:
            # The contraction axis is last, so an element's channel is its
            # position modulo the channel count -- which is what tiling the
            # energy across the flattened tensor spells.
            weights = energy.repeat(normalized.numel() // energy.numel())
        weights = weights * (scale**2).reshape(-1)
        histogram.add(normalized, weights)


def _resolve_counts(counts, shape):
    """Align a granularity tuple to a shape and resolve its ``-1`` entries.

    The tuple aligns to the trailing axes the way ``Tensor.expand``'s
    does, so a caller names only the axes it means to split.

    Args:
        counts: Tables along each trailing axis -- ``-1`` for one per
            item, ``1`` to share one table across the whole axis.
        shape: The operand's shape.

    Returns:
        One count per axis of ``shape``, every ``-1`` resolved.

    Raises:
        ValueError: More counts than axes, a count below ``-1`` or zero,
            or a count that does not divide the axis it splits.
    """
    if len(counts) > len(shape):
        raise ValueError(
            f"granularity {tuple(counts)} names {len(counts)} axes, but "
            f"the tensor reaching it has {len(shape)}"
        )
    resolved = [1] * (len(shape) - len(counts)) + list(counts)
    for axis, count in enumerate(resolved):
        if count == -1:
            resolved[axis] = shape[axis]
        elif count < 1:
            raise ValueError(
                f"granularity {tuple(counts)} asks for {count} tables "
                f"along axis {axis}; use -1 for one per item"
            )
        elif shape[axis] % count:
            raise ValueError(
                f"granularity {tuple(counts)} splits axis {axis} into "
                f"{count}, which does not divide its {shape[axis]} items"
            )
    return tuple(resolved)


def _grid_cells(tensor, counts):
    """Split a tensor into the grid its granularity asks for.

    Args:
        tensor: The operand.
        counts: One resolved table count per axis.

    Returns:
        ``(index, slab)`` per cell in row-major order, where ``index``
        gives the cell's position along every axis.
    """
    cells = [((0,) * len(counts), tensor)]
    for axis, count in enumerate(counts):
        if count == 1:
            continue
        cells = [
            (index[:axis] + (position,) + index[axis + 1 :], piece)
            for index, slab in cells
            for position, piece in enumerate(slab.chunk(count, axis))
        ]
    return cells


def leading_split(counts):
    """Return the leading axis a granularity splits, or None.

    Counts align to the trailing axes, so anything split before the last
    two is a leading axis, along which a partner's energy has to be
    sliced to match.  Which it is settles from the tuple alone, without
    the operand's rank.

    Args:
        counts: A granularity tuple, as the caller wrote it.

    Returns:
        The split leading axis as an offset from the end, or None when
        the tuple splits only the last two axes.

    Raises:
        ValueError: More than one leading axis is split.  The partner
            energy is measured along a single axis, so the fit has no
            way to weight a cell split across two.
    """
    lead = [
        axis - len(counts)
        for axis, count in enumerate(counts[:-2])
        if count != 1
    ]
    if not lead:
        return None
    if len(lead) > 1:
        raise ValueError(
            f"granularity {tuple(counts)} splits {len(lead)} leading "
            f"axes; the partner energy is measured along one"
        )
    return lead[0]


def _candidate_levels(module, nonnegative, resolution):
    """The levels a fake-quant's table is fitted from, ascending.

    An entry dtype offers its own values.  Entries that keep the model's
    dtype are bfloat16, so the candidates are ``resolution`` bfloat16
    values spread across the range, mirrored about zero when it is signed
    so that it holds an exact 0.  A tensor that never goes negative -- a
    softmax output -- keeps only the non-negative candidates, so that no
    level is spent on values that cannot occur.

    Args:
        module: The fake-quant.
        nonnegative: Whether the tensor it quantizes never goes negative.
        resolution: Candidates across the range for entries that keep the
            model's dtype.

    Returns:
        The candidates as a float64 tensor.
    """
    low = 0.0 if nonnegative else -module.quant_max
    if module.code_dtype is None:
        if low < 0:
            half = torch.linspace(0, module.quant_max, resolution // 2 + 1)
            spread = torch.cat([-half.flip(0)[:-1], half])
        else:
            spread = torch.linspace(0, module.quant_max, resolution)
        candidates = spread.bfloat16().float().unique()
    else:
        candidates = entry_levels(module.code_dtype)
    candidates = candidates.double()
    return candidates[(candidates >= low) & (candidates <= module.quant_max)]


def install_codebook(module, levels, counts=None):
    """Write fitted levels into a fake-quant, in place.

    The levels replace the fake-quant's quantization map outright, since
    that map is what ``quantize`` reads and what ``convert_pt2e`` hands to
    the graph.  One table per tensor stays a lookup table, so nothing
    downstream sees a change.  A grid of tables cannot be a lookup table
    -- an entry per bfloat16 bit pattern has no room for a second index --
    and is installed as a ``[*counts, k]`` codebook, which ``quantize``
    searches with each element's position picking its table.

    Args:
        module: The fake-quant the levels belong to.
        levels: One table of entries, or one table per grid cell in
            row-major order when ``counts`` is given.
        counts: How many tables along each operand axis, for a grid of
            them.  None installs ``levels`` as the tensor's one table.
    """
    device = module.qmap.device
    if counts is None:
        module.qmap = codebook_qmaps([levels], device)[0]
    else:
        module.qmap = torch.tensor(
            levels, dtype=torch.float32, device=device
        ).reshape(*counts, -1)


#: The op a weight's tap wraps.  A conv's weight contracts over its
#: window as well as its input channels, so the same reduction would not
#: hold, and its weight keeps the partner-energy weighting.
TAPPED_OP = torch.ops.aten.linear.default


def _tap_output(call, note):
    """Wrap an op so its output gradient reaches ``note``, per token.

    A module hook cannot see this: fake-quants sit on the op's operands,
    and the gradient wanted is the one w.r.t. its *output*.  Pointing the
    node at a wrapper is the least invasive way to reach it -- the graph
    keeps its shape, and only the callable one node holds changes.

    Args:
        call: What the node calls today.
        note: Called during the backward as ``note(activation, grad)``
            with the op's first operand and the gradient of the loss
            w.r.t. its output, both still carrying their token axis.

    Returns:
        A stand-in with the same signature, for the node to point at.
    """

    def tapped(*args, **kwargs):
        output = call(*args, **kwargs)
        if output.requires_grad:
            output.register_hook(lambda grad: note(args[0], grad))
        return output

    return tapped


def _accumulate_fisher(model, calibration, operands, grid, tap_weights):
    """Return per-channel loss-gradient Fisher weights, by a backward pass.

    Weights each operand by the sum over calibration of its squared
    gradient of the final loss, per channel -- the Gauss-Newton diagonal,
    the operand's full downstream sensitivity.  Squaring happens per
    token and the squares are then summed, which is what makes it a
    curvature surrogate rather than a gradient magnitude.

    An activation's gradient arrives with its token axis intact, so its
    diagonal is read straight off the fake-quant's backward hook.  A
    parameter's does not: ``dL/dW[o, i]`` is already
    ``sum_t g[t, o] x[t, i]``, contracted by the op's own backward before
    any hook runs, and squaring that gives the square of a sum where the
    diagonal needs the sum of squares.  Summed over the output channels
    the wanted quantity factors,

        sum_o sum_t (g[t, o] x[t, i])^2 = sum_t x[t, i]^2 ||g_t||^2

    so a weight is served by tapping its op's output gradient, reducing
    it to one norm per token, and pairing that with the activation the
    backward is holding anyway.  A weight whose op is not a linear, or
    whose output never needs a gradient, is left out and keeps the
    partner-energy weighting -- the same diagonal under a uniform
    per-token weight.

    Args:
        model: Prepared graph module.
        calibration: Calibration inputs, each yielding a loss.
        operands: ``name -> (module, node, partners)`` from the caller.
        grid: ``name -> granularity tuple`` for the tensors fitted more
            than one table, whose diagonal is resolved per cell here.
        tap_weights: Also weight the parameters, by the taps described
            above.  A parameter's exact diagonal carries a per-token
            ``||g_t||^2`` factor that a few high-loss calibration tokens
            dominate -- a property of the calibration set rather than the
            weight -- so ``fisher_activations`` leaves them on partner
            energy.

    Returns:
        ``name -> per-channel Fisher weight``, ``[cells, channels]`` for a
        tensor in ``grid``.  An operand absent from it reached no
        per-token gradient and keeps the partner-energy weighting.
    """
    fisher = {}

    def per_channel(grad, ch_axis):
        ordered = _contraction_last(grad, ch_axis)
        return ordered.reshape(-1, ordered.shape[-1]).pow(2).sum(0)

    def note_grad(name, module, grad):
        if grad is None:
            return
        grad = _as_quantized(module, grad.detach().float())
        if name in grid:
            per = torch.stack(
                [
                    per_channel(slab, module.ch_axis)
                    for _, slab in _grid_cells(
                        grad, _resolve_counts(grid[name], grad.shape)
                    )
                ]
            )
        else:
            per = per_channel(grad, module.ch_axis)
        fisher[name] = per if name not in fisher else fisher[name] + per

    def note_weight(name, module, activation, grad):
        rows = activation.detach().reshape(-1, activation.shape[-1]).float()
        rows = _as_quantized(module, rows)
        grads = grad.detach().reshape(-1, grad.shape[-1]).float().pow(2)
        if name in grid:
            # The weight is ``[out, in]`` -- one axis from each factor --
            # so a cell's diagonal pairs its output band's norms with its
            # input band's columns, in the same row-major cell order the
            # collection indexes.
            counts = _resolve_counts(
                grid[name], (grad.shape[-1], rows.shape[-1])
            )
            per = torch.stack(
                [
                    (band.pow(2) * norm.sum(-1, keepdim=True)).sum(0)
                    for norm in grads.chunk(counts[0], -1)
                    for band in rows.chunk(counts[1], -1)
                ]
            )
        else:
            per = (rows.pow(2) * grads.sum(-1, keepdim=True)).sum(0)
        fisher[name] = per if name not in fisher else fisher[name] + per

    handles, tapped = [], {}
    for name, (module, node, _) in operands.items():
        if getattr(node.args[0], "op", None) != "get_attr":
            handles.append(
                module.register_full_backward_hook(
                    lambda module, gi, go, name=name: note_grad(
                        name, module, gi[0]
                    )
                )
            )
            continue
        if not tap_weights:
            continue
        # A weight can feed several ops -- the same Linear called twice --
        # and its diagonal is the sum over all of them, so every use is
        # tapped and ``note_weight`` adds them.
        for consumer in [
            user for user in node.users if user.target is TAPPED_OP
        ]:
            tapped[consumer] = consumer.target
            consumer.target = _tap_output(
                consumer.target,
                lambda activation, grad, name=name, module=module: note_weight(
                    name, module, activation, grad
                ),
            )
    if tapped:
        model.recompile()

    # A full backward fills a gradient for every parameter and this pass
    # reads none of them, so restricting it to one anchor parameter looks
    # free.  It is not: ``inputs`` prunes the graph to what that anchor
    # needs, and an operand off that path stops being weighted at all --
    # anchoring on Q's weight drops K and V, whose gradients the path to Q
    # never visits.  The wasted gradients are the price of the coverage.
    was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for entry in calibration:
            output = model(*entry)
            loss = output[0] if isinstance(output, tuple) else output.loss
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        torch.set_grad_enabled(was_enabled)
        for handle in handles:
            handle.remove()
        for consumer, target in tapped.items():
            consumer.target = target
        if tapped:
            model.recompile()

    missed = sorted(set(operands) - set(fisher))
    logger.info(
        "%d operands weighted by their loss-gradient Fisher%s",
        len(fisher),
        (
            f"; {len(missed)} reached no per-token gradient and keep the "
            f"partner-energy weighting: {_preview(missed)}"
            if missed
            else ""
        ),
    )
    return fisher


class Weighting(enum.Enum):
    """Which sensitivity weights each sample in a codebook fit.

    ``PARTNER`` weights a sample by its channel's energy in the operand it
    contracts against, which is what turns a tensor-MSE fit into a
    layer-output-error fit.  The two Fisher modes replace that with the
    operand's own Gauss-Newton diagonal, measured by a backward pass, so
    the fit minimizes the whole model's output error rather than the
    immediate layer's -- an operand no per-token gradient reaches falls
    back to ``PARTNER``.  ``FISHER_ALL`` extends the diagonal to the
    parameters by tapping each linear's output gradient; a parameter's
    exact diagonal is dominated by a few high-loss calibration tokens, so
    the default leaves parameters on partner energy.  ``NONE`` weights
    nothing and fits plain tensor MSE.
    """

    NONE = "none"
    PARTNER = "partner"
    FISHER_ACTIVATIONS = "fisher_activations"
    FISHER_ALL = "fisher_all"


def fit_codebooks(
    model,
    calibration,
    skip=(),
    weighting=Weighting.PARTNER,
    weight_by=(),
    granularity=None,
    pin_zero=True,
    quantized_inputs=False,
    quantized_scale=True,
    dump=None,
    max_levels=None,
    resolution=FLOAT_RESOLUTION,
):
    """Fit a codebook to every tensor a prepared model quantizes, in place.

    Block size, contraction axis and range come from the fake-quant the
    quantizer attached to each tensor, and the fitted levels are written
    back into it.  The fit runs in four stages:

    1. Plan (``_plan_fit``): from the graph alone, which tensors are
       fitted, which partners weight each, and where each tensor is read.
    2. Measure (``_measuring``): a pass records every tensor's channel
       energy and every fitted tensor's sign and shape.  Under a Fisher
       weighting it rides the Fisher pass, which runs every input forward
       anyway.
    3. Collect (``_new_fits``, ``_collecting``): a second pass bins every
       element of each fitted tensor, weighted by its partners' energy or
       its own Fisher diagonal, so the fit minimizes output error rather
       than tensor MSE.  Nothing is sampled, so a fit is reproducible and
       sees the whole tensor.
    4. Solve (``_solve_fits``): each histogram's exact optimum becomes a
       table, and the tables are installed.

    Args:
        model: Graph module from ``prepare_pt2e``, before ``convert_pt2e``.
        calibration: Entries of positional arguments matching the graph's
            placeholders.
        skip: Substrings of operand names to leave alone.
        weighting: Which sensitivity weights each sample; see
            ``Weighting``.  ``PARTNER`` is the partner's channel energy,
            the two Fisher modes the operand's own loss-gradient diagonal,
            ``NONE`` no weighting at all.
        weight_by: Substrings picking which partners weight a tensor read
            by several ops; empty adds every partner's energy.  Only a
            partner that is itself quantized has a name to be picked by.
        granularity: ``name substring -> counts``, how many codebooks a
            tensor is fitted along each axis.  The tuple aligns to the
            trailing axes: ``-1`` gives one table per item of that axis,
            ``n`` splits it into ``n``, ``1`` shares one across it.
            ``(-1, 1, 1)`` is one table per attention head, ``(28, 16)``
            a grid over a weight's last two axes.
        pin_zero: Force a zero level.
        quantized_inputs: Sample with the fake-quants live.
        quantized_scale: Normalize by quantized block scale.
        dump: Path to write the fitted tables to, as JSON.
        max_levels: Fit at most this many levels per table.  None fills the
            index, and a table never holds more levels than its grid.
        resolution: Candidates across the range for a table whose entries
            keep the model's dtype, and so have no grid of their own.

    Returns:
        ``name -> the fitted levels`` for every tensor fitted: one table
        as a list of levels, a grid of them as
        ``{"granularity": counts, "levels": tables}``.
    """
    plan = _plan_fit(model, skip, weight_by, granularity)
    logger.info(
        "fitting %d tensors over %d calibration inputs, weighting a shared "
        "tensor by %s",
        len(plan.operands),
        len(calibration),
        (
            f"its {', '.join(weight_by)} partner"
            if weight_by
            else "every partner it feeds"
        ),
    )
    if quantized_inputs:
        logger.warning(
            "sampling with the fake-quants live: fitting against one "
            "configuration and deploying another measured worse"
        )
    if weighting is Weighting.NONE:
        logger.warning(
            "sampling unweighted: the fit minimizes tensor MSE rather than "
            "layer-output error, which recovers less than half as much"
        )
    if plan.grid:
        logger.info(
            "fitting %d of them more than one codebook: %s",
            len(plan.grid),
            _preview(sorted(plan.grid)),
        )

    live = {
        module: module.fake_quant_enabled
        for module in model.modules()
        if hasattr(module, "qmap")
    }
    if not quantized_inputs:
        for module in live:
            module.disable_fake_quant()

    fisher = {}
    with _measuring(model, plan) as measured:
        if weighting in (Weighting.FISHER_ACTIVATIONS, Weighting.FISHER_ALL):
            fisher = _accumulate_fisher(
                model,
                calibration,
                plan.operands,
                plan.grid,
                tap_weights=weighting is Weighting.FISHER_ALL,
            )
        else:
            _run_calibration(model, calibration, "measure")

    fits = _new_fits(plan, measured, fisher, weighting, resolution)
    with _collecting(model, plan, fits, quantized_scale):
        _run_calibration(model, calibration, "collect")

    for module, enabled in live.items():
        module.enable_fake_quant(enabled)

    _log_concentration("fisher", fisher)
    _log_concentration("partner energy", measured.energies)
    tables = _solve_fits(plan.operands, fits, pin_zero, max_levels)
    _install_tables(plan.operands, tables)
    if dump is not None:
        with open(dump, "w") as handle:
            json.dump(tables, handle, indent=2, sort_keys=True)
        logger.info("wrote %d codebooks to %s", len(tables), dump)
    return tables


class _Plan(NamedTuple):
    """What a fit reads, decided from the graph before any input runs.

    A tensor whose energy is measured goes by a key: a fitted tensor by its
    name, any other partner by its node's name.

    Attributes:
        operands: ``name -> (fake-quant, node, partner nodes)`` for every
            tensor fitted, from ``_quantized_operands``.
        sources: ``key -> (node, axis, fake-quant)`` for every tensor
            measured: the node yielding it, the axis it is contracted over,
            and the fake-quant whose rotation it takes, None for a partner
            no fake-quant reads.
        partners: ``name -> keys`` of the partners whose energy weights
            each fitted tensor.
        grid: ``name -> granularity`` for each tensor fitted more than one
            table.
        leading: ``key -> axis`` for each tensor measured per slice along a
            leading axis: one whose tables split that axis, and its
            partners.
    """

    operands: dict
    sources: dict
    partners: dict
    grid: dict
    leading: dict


def _plan_fit(model, skip, weight_by, granularity):
    """Decide from the graph what a fit reads and what weights each tensor.

    A fitted tensor is read where its fake-quant reads it.  So is a partner
    that is not fitted but has a fake-quant, and a partner with none is the
    tensor its contracted op reads; either is measured along the axis that
    op contracts it over.

    Args:
        model: Graph module from ``prepare_pt2e``.
        skip: Substrings of operand names to leave alone.
        weight_by: Substrings picking the partners that weight a tensor
            read by several ops; see ``fit_codebooks``.
        granularity: ``name substring -> counts``; see ``fit_codebooks``.

    Returns:
        The ``_Plan``.

    Raises:
        ValueError: A granularity splits more than one leading axis.
    """
    operands = _quantized_operands(model, skip)
    named = {node: name for name, (_, node, _) in operands.items()}

    sources = {
        name: (node.args[0], module.ch_axis, module)
        for name, (module, node, _) in operands.items()
    }
    partners = {}
    for name, (_, node, group) in operands.items():
        # Only a fitted partner has a name for ``weight_by`` to pick; when
        # it picks none, every partner weights the tensor.
        picked = [
            partner
            for partner in group
            if any(pick in named.get(partner, "") for pick in weight_by)
        ]
        chosen = picked or group
        partners[name] = [
            named.get(partner, partner.name) for partner in chosen
        ]
        # A partner that is not fitted is read where its own fake-quant
        # reads it, or as the tensor itself when it has none.
        for partner in chosen:
            if partner in named or partner.name in sources:
                continue
            consumer = next(
                user
                for user in node.users
                if user.target in CONTRACTED_OPS
                and node in user.args[:2]
                and partner in user.args[:2]
            )
            axis = CONTRACTED_OPS[consumer.target][
                consumer.args[:2].index(partner)
            ]
            if partner.op == "call_module":
                fake_quant = model.get_submodule(partner.target)
                sources[partner.name] = (partner.args[0], axis, fake_quant)
            else:
                sources[partner.name] = (partner, axis, None)

    grid = {
        name: tuple(counts)
        for pattern, counts in (granularity or {}).items()
        for name in operands
        if pattern in name
    }
    leading = {}
    for name, counts in grid.items():
        axis = leading_split(counts)
        if axis is not None:
            leading[name] = axis
    # A split tensor's error reaches the output through its partners' own
    # slices of that axis, so they are measured per slice too.
    for name, axis in list(leading.items()):
        for other in partners[name]:
            leading[other] = axis
    return _Plan(operands, sources, partners, grid, leading)


class _Measurements(NamedTuple):
    """What the measuring pass records, filled while it runs.

    Attributes:
        energies: ``key -> energy per channel``, summed over the pass, for
            every partner; ``[slices, channels]`` for a key measured per
            slice.
        floors: ``name -> lowest value`` of each fitted tensor.
        shapes: ``name -> shape`` of each fitted tensor.
    """

    energies: dict
    floors: dict
    shapes: dict


@contextlib.contextmanager
def _measuring(model, plan):
    """Measure every tensor the plan names during the passes run inside.

    Each tensor is rotated as its fake-quant rotates it before it is
    measured.

    Args:
        model: The prepared graph module.
        plan: What the fit reads, from ``_plan_fit``.

    Yields:
        The ``_Measurements``, filled as the passes run.
    """
    measured = _Measurements({}, {}, {})
    # Only a partner's energy weights anything.
    partners = {key for keys in plan.partners.values() for key in keys}

    @torch.no_grad()
    def measure(key, tensor):
        _, axis, fake_quant = plan.sources[key]
        tensor = _as_quantized(fake_quant, tensor)
        if key in partners:
            slices = plan.leading.get(key)
            if slices is None:
                energy = _channel_energy(tensor, axis)
            else:
                energy = torch.stack(
                    [
                        _channel_energy(piece, axis)
                        for piece in tensor.unbind(slices)
                    ]
                )
            held = measured.energies.get(key)
            measured.energies[key] = energy if held is None else held + energy
        if key in plan.operands:
            lowest = tensor.min()
            held = measured.floors.get(key)
            measured.floors[key] = (
                lowest if held is None else torch.minimum(held, lowest)
            )
            measured.shapes.setdefault(key, tensor.shape)

    reads = [(key, node) for key, (node, _, _) in plan.sources.items()]
    with _reading(model, measure, reads):
        yield measured


class _Fit(NamedTuple):
    """One fitted tensor's grid of histograms, as the collecting pass bins it.

    Attributes:
        counts: Tables along each axis, resolved against its shape.
        histograms: Each cell's histogram, in ``_grid_cells`` order.
        energies: Each cell's weight per channel, or None for a flat fit.
    """

    counts: tuple
    histograms: list
    energies: list | None


def _new_fits(plan, measured, fisher, weighting, resolution):
    """Set up each measured tensor's histograms and the weight of each cell.

    The grid resolves against the shape the measuring pass saw, and the
    candidates against the sign it saw.  A cell is weighted by the
    tensor's own Fisher diagonal when the Fisher pass reached it, or else
    by its partners' summed energy cut to the cell, unless the fit is
    unweighted.

    Args:
        plan: What the fit reads, from ``_plan_fit``.
        measured: What the measuring pass recorded.
        fisher: ``name -> Fisher diagonal`` from ``_accumulate_fisher``,
            empty under any other weighting.
        weighting: The ``Weighting`` asked for.
        resolution: Candidates across the range for entries that keep the
            model's dtype.

    Returns:
        ``name -> _Fit``.
    """
    fits = {}
    for name, shape in measured.shapes.items():
        module = plan.operands[name][0]
        if name in plan.grid:
            counts = _resolve_counts(plan.grid[name], shape)
        else:
            counts = (1,) * len(shape)
        candidates = _candidate_levels(
            module, measured.floors[name].item() >= 0, resolution
        )
        histograms = [Histogram(candidates) for _ in range(math.prod(counts))]
        shares = [
            measured.energies[key]
            for key in plan.partners[name]
            if key in measured.energies
        ]
        if name in fisher:
            # A tensor with a grid has its diagonal measured per cell.
            own = fisher[name]
            energies = list(own) if own.dim() == 2 else [own] * len(histograms)
        elif weighting is Weighting.NONE or not shares:
            energies = None
        else:
            energies = _partner_cells(
                sum(shares), counts, plan.leading.get(name), module.ch_axis
            )
        fits[name] = _Fit(counts, histograms, energies)
    return fits


def _partner_cells(energy, counts, axis, ch_axis):
    """Cut a fitted tensor's summed partner energy to each of its cells.

    An energy measured per slice gives a cell the mean of the slices its
    stretch of ``axis`` covers, or of all of them when ``axis`` is None.
    The energy runs along the contraction axis, so only a split along that
    axis gives each cell its own share of the channels.

    Args:
        energy: Energy per channel, or ``[slices, channels]`` measured per
            slice along ``axis``.
        counts: The tensor's resolved grid.
        axis: The leading axis the slices run along, or None.
        ch_axis: The tensor's contraction axis, as its fake-quant gives it.

    Returns:
        One energy per cell, in ``_grid_cells`` order.
    """
    channel = ch_axis[0] if isinstance(ch_axis, (tuple, list)) else ch_axis
    shares = []
    for index in itertools.product(*(range(count) for count in counts)):
        share = energy
        if share.dim() == 2:
            if axis is None:
                share = share.mean(0)
            else:
                start = index[axis] * len(share) // counts[axis]
                stop = (index[axis] + 1) * len(share) // counts[axis]
                share = share[start : max(stop, start + 1)].mean(0)
        if counts[channel] > 1:
            share = share.chunk(counts[channel])[index[channel]]
        shares.append(share)
    return shares


@contextlib.contextmanager
def _collecting(model, plan, fits, quantized_scale):
    """Bin each fitted tensor into its cells during the passes run inside.

    Args:
        model: The prepared graph module.
        plan: What the fit reads, from ``_plan_fit``.
        fits: ``name -> _Fit`` from ``_new_fits``.
        quantized_scale: Normalize by the scale deployment stores; see
            ``_accumulate``.
    """

    @torch.no_grad()
    def collect(name, tensor):
        module = plan.operands[name][0]
        fit = fits[name]
        prepared = module.prepare(tensor)
        _accumulate(
            prepared,
            module,
            fit.energies,
            fit.histograms,
            fit.counts,
            _global_scale(module, prepared),
            quantized_scale,
        )

    reads = [(name, plan.sources[name][0]) for name in fits]
    with _reading(model, collect, reads):
        yield


def _solve_fits(operands, fits, pin_zero, max_levels):
    """Solve each fitted tensor's histograms for their optimal tables.

    Args:
        operands: ``name -> (fake-quant, node, partner nodes)``.
        fits: ``name -> _Fit``, binned.
        pin_zero: Force a zero level.
        max_levels: Fit at most this many levels per table, or None.

    Returns:
        ``name -> levels``, as ``fit_codebooks`` returns them.
    """
    tables, unsigned = {}, []
    for name, (module, _, _) in operands.items():
        fit = fits.get(name)
        if fit is None or fit.histograms[0].weight is None:
            logger.warning(
                "%s: nothing accumulated, so it keeps the table it was "
                "seeded with",
                name,
            )
            continue
        first = fit.histograms[0]
        if first.quant_min == 0.0:
            unsigned.append(name)
        # The index addresses 2**index_bits entries, and the grid can hold
        # fewer.
        k = min(2**module.index_bits, first.levels.numel())
        if max_levels is not None:
            k = min(k, max_levels)
        fitted = optimal_codebooks(fit.histograms, k=k, pin_zero=pin_zero)
        # A codebook carries one axis per tensor axis, so the grid the fit
        # asked for is the shape the decoder dispatches on.
        if all(count == 1 for count in fit.counts):
            tables[name] = fitted[0]
        else:
            tables[name] = {"granularity": list(fit.counts), "levels": fitted}
        logger.debug("%s: %s", name, tables[name])

    logger.info("fitted %d of %d tensors", len(tables), len(operands))
    if unsigned:
        logger.info(
            "%d never went negative, so their levels span [0, quant_max]: "
            "%s",
            len(unsigned),
            _preview(unsigned),
        )
    return tables


def _install_tables(operands, tables):
    """Install tables into their fake-quants, in place.

    Args:
        operands: ``name -> (fake-quant, node, partner nodes)``.
        tables: ``name -> levels``, as ``fit_codebooks`` returns them.
    """
    for name, entry in tables.items():
        module = operands[name][0]
        if isinstance(entry, dict):
            install_codebook(module, entry["levels"], entry["granularity"])
        else:
            install_codebook(module, entry)


@contextlib.contextmanager
def _reading(model, visit, reads):
    """Hand each read's tensors to ``visit`` during the passes run inside.

    A parameter is the same tensor on every calibration input, so reading
    it once is reading it every time: the moments scale together and the
    solver's minimum stays where it was.  A read of one parameter is made
    once, off the model, on entry.  Any other read is a call inserted after
    the last node yielding its tensors, on every pass; the calls are erased
    on exit, which leaves the graph as it was.

    Args:
        model: The graph module.
        visit: Called as ``visit(key, *tensors)`` with each read's tensors.
        reads: ``(key, *nodes)`` per read: the nodes yielding its tensors.
    """
    graph = model.graph
    order = {node: index for index, node in enumerate(graph.nodes)}
    inserted = []
    try:
        for key, *nodes in reads:
            if len(nodes) == 1 and nodes[0].op == "get_attr":
                visit(key, fetch_attr(model, nodes[0].target))
                continue
            with graph.inserting_after(max(nodes, key=order.__getitem__)):
                inserted.append(graph.call_function(visit, (key, *nodes)))
        model.recompile()
        yield
    finally:
        for read in inserted:
            graph.erase_node(read)
        model.recompile()


def _run_calibration(model, calibration, label):
    """Run the calibration set forward, logging progress every tenth.

    Args:
        model: The prepared graph module.
        calibration: Entries of positional arguments for it.
        label: The pass's name, for the progress messages.
    """
    step = max(1, len(calibration) // 10)
    with torch.no_grad():
        for index, entry in enumerate(calibration):
            model(*entry)
            if (index + 1) % step == 0:
                logger.info(
                    "%s pass: %d/%d windows",
                    label,
                    index + 1,
                    len(calibration),
                )


def _log_concentration(label, weights):
    """Log how concentrated each tensor's weighting is, as max over median.

    Past roughly a hundredfold the levels stop serving the bulk of a tensor
    and serve its few heaviest channels instead.

    Args:
        label: What the weights are, for the message.
        weights: ``key -> weight per channel``.
    """
    spread = [
        (one.reshape(-1).max() / one.reshape(-1).median().clamp_min(EPS)).item()
        for one in weights.values()
        if one.numel() > 1
    ]
    if not spread:
        return
    spread = torch.tensor(spread)
    logger.info(
        "%s concentration over %d tensors (max/median per tensor): median "
        "%.3g, p90 %.3g, worst %.3g",
        label,
        len(spread),
        spread.median(),
        spread.quantile(0.9),
        spread.max(),
    )


def _preview(names):
    """Join the first three names, adding `` ...`` when there are more."""
    return ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")


def load_codebooks(model, tables, skip=()):
    """Install already-fitted codebooks into a prepared model, in place.

    Args:
        model: Graph module from ``prepare_pt2e``.
        tables: ``name -> levels``, or the path of a JSON file holding it,
            as ``fit_codebooks`` writes with ``dump``.  One table is a
            list of levels; a grid of them is
            ``{"granularity": counts, "levels": ...}``, one count per
            operand axis.
        skip: Substrings of operand names to leave alone.

    Returns:
        The names installed.

    Raises:
        KeyError: The tables name a tensor this model does not quantize,
            which means they were fitted against a different scheme.
    """
    if isinstance(tables, (str, os.PathLike)):
        with open(tables) as handle:
            tables = json.load(handle)
    tables = {
        name: entry
        for name, entry in tables.items()
        if not any(excluded in name for excluded in skip)
    }

    operands = _quantized_operands(model, skip)
    unknown = sorted(set(tables) - set(operands))
    if unknown:
        raise KeyError(
            f"{len(unknown)} codebooks do not match anything this model "
            f"quantizes, starting with {unknown[0]}"
        )
    _install_tables(operands, tables)
    logger.info("installed %d codebooks", len(tables))

    unfitted = sorted(set(operands) - set(tables))
    if unfitted:
        logger.warning(
            "%d tensors this model quantizes have no codebook and keep the "
            "table they were seeded with, starting with %s",
            len(unfitted),
            unfitted[0],
        )
    return sorted(tables)


@contextlib.contextmanager
def _pre_hooks(modules, hook):
    """Hold ``hook(name, module, inputs)`` before each of ``modules`` runs.

    Args:
        modules: ``name -> module``.
        hook: Called with the module's name ahead of its own arguments.
    """
    handles = [
        module.register_forward_pre_hook(functools.partial(hook, name))
        for name, module in modules.items()
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _as_quantized(module, tensor):
    """``tensor`` as ``module`` quantizes it: rotated first if it rotates."""
    if getattr(module, "rht_axis", None) is None:
        return tensor
    return _rotate(tensor, module.rht_axis, module.rht_matrix)


def _training_operand_names(model, operands):
    """Name each forward GEMM operand of a training graph as
    ``fit_codebooks`` names an inference graph's.

    A GEMM's scope is the innermost module the export recorded for it.

    Args:
        model: Model ``prepare_training`` prepared.
        operands: ``name -> [(GEMM, position)]`` from
            ``forward_gemm_operands``.

    Returns:
        ``fake-quant name -> operand name``, after the first GEMM each
        fake-quant feeds.
    """
    scopes, order, seen = {}, {}, {}
    for gemm in model.forward.graphs["joint"].graph.nodes:
        if gemm.meta.get("gemm_kind") != "fprop":
            continue
        scopes[gemm] = list(gemm.meta["nn_module_stack"].values())[-1][0]
        key = (scopes[gemm], gemm.target)
        order[gemm] = seen.get(key, 0)
        seen[key] = order[gemm] + 1
    return {
        name: _operand_name(scopes[gemm], gemm, order[gemm], position)
        for name, ((gemm, position), *_) in operands.items()
    }


class CodebookRefitter:
    """Refit a training model's forward lookup tables to what it now sees.

    A window of training steps runs under ``measure`` and then
    ``accumulate``.  The first records each operand's channel energy and
    whether it goes negative, and under ``fisher_activations`` each forward
    GEMM activation's loss-gradient diagonal, read off the backward graph.
    The rest bin each forward table's operand, weighted by its own diagonal
    when it has one and by its partners' energy otherwise, as
    ``fit_codebooks`` bins a calibration set.  ``refit`` then installs the
    table each histogram's optimum gives, one per cell for a table with a
    granularity.  The gradients' tables are refitted too when asked,
    weighted by their partners' energy.  A partner without microscaling
    blocks to measure along leaves its operand unweighted.

    Args:
        model: Model ``prepare_training`` prepared with lookup-table specs.
        pin_zero: Force a zero level into every table.
        resolution: Candidates across the range for entries that keep the
            model's dtype.
        weighting: ``Weighting.PARTNER`` or ``Weighting.FISHER_ACTIVATIONS``.
        granularity: ``name substring -> counts``, how many tables a forward
            table is fitted along each axis, as ``fit_codebooks`` takes it.
            A forward GEMM's operand goes by the name ``fit_codebooks``
            gives it -- ``<module path>|q`` is attention's query -- and any
            other table by its fake-quant's name.
        refit_gradients: Refit the gradients' tables as well.

    Raises:
        ValueError: ``weighting`` is neither of those.
    """

    def __init__(
        self,
        model,
        pin_zero=True,
        resolution=FLOAT_RESOLUTION,
        weighting=Weighting.PARTNER,
        granularity=None,
        refit_gradients=False,
    ):
        if weighting not in (Weighting.PARTNER, Weighting.FISHER_ACTIVATIONS):
            raise ValueError(
                "a refit weights by partner energy or fisher_activations, "
                f"not {weighting.value}"
            )
        quantizers = model.training_quantizers
        self.modules = dict(quantizers.named_children())
        self.tables = [
            name
            for name, module in self.modules.items()
            if isinstance(module, MXFakeQuantize)
            and module.is_codebook_quantization
            and (refit_gradients or name not in quantizers.gradients)
        ]
        partners = gemm_partners(model)
        self.partners = {
            name: [
                partner
                for partner in sorted(partners[name])
                if isinstance(self.modules[partner], MXFakeQuantize)
            ]
            for name in self.tables
        }
        operands = forward_gemm_operands(model)
        labels = _training_operand_names(model, operands)
        self.grid = {
            name: tuple(counts)
            for pattern, counts in (granularity or {}).items()
            for name in self.tables
            if pattern in labels.get(name, name)
        }
        self.leading = {}
        for name, counts in self.grid.items():
            axis = leading_split(counts)
            if axis is not None:
                self.leading[name] = axis
        # A split table's error reaches the output through its partners' own
        # slices of that axis, so they are measured per slice too.
        for name, axis in list(self.leading.items()):
            for other in self.partners[name]:
                self.leading[other] = axis

        # Under ``fisher_activations``, each forward activation's gradient:
        # the outputs of the dgrads of the GEMMs it feeds, in the backward
        # graph.
        self.backward = model.forward.graphs["backward"]
        self.gradients = []
        if weighting is Weighting.FISHER_ACTIVATIONS:
            nodes = {node.name: node for node in self.backward.graph.nodes}
            for name in self.tables:
                dgrads = [
                    operand_gradient(gemm, position)
                    for gemm, position in operands.get(name, [])
                    if gemm.meta["operand_types"][position] == "activation"
                ]
                dgrads = [nodes[dgrad.name] for dgrad in dgrads if dgrad]
                if dgrads:
                    self.gradients.append((name, *dgrads))

        self.pin_zero = pin_zero
        self.resolution = resolution
        self.energies = {}
        self.nonnegative = {}
        self.fisher = {}
        self.counts = {}
        self.histograms = {}

    @contextlib.contextmanager
    def measure(self):
        """Start a window: record each operand's channel energy and sign,
        and under ``fisher_activations`` each forward activation's
        loss-gradient diagonal.

        Hooks the forward tables and their partners, and reads the
        activations' gradients off the backward graph, for the step it
        wraps.  Each sign stays on the device until the step ends, and one
        read then brings them all back.
        """
        self.energies, self.nonnegative, self.fisher = {}, {}, {}
        self.counts, self.histograms = {}, {}
        names = set(self.tables).union(*self.partners.values())
        gradients = (
            _reading(self.backward, self._record_gradient, self.gradients)
            if self.gradients
            else contextlib.nullcontext()
        )
        with (
            _pre_hooks({n: self.modules[n] for n in names}, self._record),
            gradients,
        ):
            yield
        signs = torch.stack(list(self.nonnegative.values())).tolist()
        self.nonnegative = dict(zip(self.nonnegative, signs))

    def accumulate(self):
        """Hook the tables to bin their operands, after ``measure`` in the
        same window."""
        tables = {name: self.modules[name] for name in self.tables}
        return _pre_hooks(tables, self._bin)

    def refit(self):
        """Install the table each binned histogram's optimum gives.

        Histograms over the same candidates, fitted to the same number of
        levels, are solved together, and their lookup tables built together.
        A table with a grid is solved per cell and installed as a codebook.

        Returns:
            ``name -> fitted levels``, a grid's as
            ``{"granularity": counts, "levels": tables}``.
        """
        groups, fitted = {}, {}
        for name, histograms in self.histograms.items():
            module = self.modules[name]
            k = min(2**module.index_bits, histograms[0].levels.numel())
            if name in self.grid:
                levels = optimal_codebooks(histograms, k, self.pin_zero)
                install_codebook(module, levels, self.counts[name])
                fitted[name] = {
                    "granularity": list(self.counts[name]),
                    "levels": levels,
                }
            else:
                key = (tuple(histograms[0].levels.tolist()), k)
                groups.setdefault(key, []).append(name)
        for (_, k), names in groups.items():
            codebooks = optimal_codebooks(
                [self.histograms[name][0] for name in names], k, self.pin_zero
            )
            fitted.update(zip(names, codebooks))
            device = self.modules[names[0]].qmap.device
            for name, qmap in zip(names, codebook_qmaps(codebooks, device)):
                self.modules[name].qmap = qmap
        return fitted

    def _cells(self, name, shape):
        """Return a forward table's grid against its operand's shape."""
        if name in self.grid:
            return _resolve_counts(self.grid[name], shape)
        return (1,) * len(shape)

    def _record(self, name, module, inputs):
        """Add one operand's channel energy and sign to the window's."""
        tensor = _as_quantized(module, inputs[0])
        slices = self.leading.get(name)
        if slices is None:
            energy = _channel_energy(tensor, module.ch_axis)
        else:
            energy = torch.stack(
                [
                    _channel_energy(piece, module.ch_axis)
                    for piece in tensor.unbind(slices)
                ]
            )
        self.energies[name] = self.energies.get(name, 0) + energy
        nonnegative = (tensor >= 0).all()
        held = self.nonnegative.get(name)
        self.nonnegative[name] = (
            nonnegative if held is None else held & nonnegative
        )

    @torch.no_grad()
    def _record_gradient(self, name, *gradients):
        """Add one forward activation's Fisher diagonal to the window's.

        Its gradient is the sum of those the GEMMs it feeds pass back, and
        the diagonal is that squared and summed over every position but the
        contraction axis, per cell of its grid.
        """
        module = self.modules[name]
        gradient = _as_quantized(module, sum(gradients).float())
        cells = [
            _contraction_last(cell, module.ch_axis).flatten(0, -2)
            for _, cell in _grid_cells(
                gradient, self._cells(name, gradient.shape)
            )
        ]
        diagonal = torch.stack([cell.pow(2).sum(0) for cell in cells])
        held = self.fisher.get(name)
        self.fisher[name] = diagonal if held is None else held + diagonal

    def _bin(self, name, module, inputs):
        """Bin one forward table's operand into its cells, weighted by its
        own Fisher diagonal or else its partners' energy."""
        operand = module.prepare(inputs[0])
        if name not in self.histograms:
            self.counts[name] = self._cells(name, operand.shape)
            candidates = _candidate_levels(
                module, self.nonnegative[name], self.resolution
            )
            self.histograms[name] = [
                Histogram(candidates)
                for _ in range(math.prod(self.counts[name]))
            ]
        counts = self.counts[name]
        if name in self.fisher:
            energies = list(self.fisher[name])
        else:
            shares = [
                self.energies[partner]
                for partner in self.partners[name]
                if partner in self.energies
            ]
            energies = (
                _partner_cells(
                    sum(shares),
                    counts,
                    self.leading.get(name),
                    module.ch_axis,
                )
                if shares
                else None
            )
        _accumulate(
            operand,
            module,
            energies,
            self.histograms[name],
            counts,
            _global_scale(module, operand),
        )


def refit_window(refitter, step, every, collect):
    """The context a training step runs in, to refit every ``every`` steps.

    The ``collect`` steps before each refit form its window: the first
    measures and the rest accumulate.  Every other step runs unhooked.

    Args:
        refitter: The model's ``CodebookRefitter``.
        step: Steps completed before this one.
        every: Steps between refits.
        collect: Steps each window spans, at least two.

    Returns:
        A context manager.
    """
    position = step % every
    start = every - collect
    if position == start:
        return refitter.measure()
    if position > start:
        return refitter.accumulate()
    return contextlib.nullcontext()
