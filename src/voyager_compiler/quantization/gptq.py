"""GPTQ against the deployed microscaling grid.

Ordinary GPTQ quantizes one column at a time by handing ``W[:, i]`` to a
fake-quant.  Under microscaling that blocks along the wrong axis -- a column
runs over *output* channels while the deployed quantizer blocks over *input*
channels -- so the compensation targets errors the hardware never makes.
Here each block of 64 input channels takes its scale at block start from the
current, already-compensated weights, and that block's columns are quantized
against it.

Nothing about the format changes: same index width, same codebook, same
block size.  Only the stored weight values move.  ``gptq`` drives the whole
thing from a prepared model and a calibration set; ``compensate_weight`` is
the single-weight step it is built from.
"""

import logging
import re
import time

import torch
from torch.fx import Interpreter

from voyager_compiler.export_utils import get_node_name_to_scope
from voyager_compiler.ops import (
    calculate_mx_qparam,
    compiled_on_gpu,
    quantize,
    vmap,
)
from voyager_compiler.quantization.codebook_optimizer import EPS
from voyager_compiler.shape_prop import fetch_attr

logger = logging.getLogger(__name__)


def hessian_inverse_chol(hessian, tokens, damping):
    """Upper Cholesky factor of inv(H), damped and retried on failure.

    Args:
        hessian: Accumulated ``X^T X`` for one linear.
        tokens: Rows that went into it, for the mean.
        damping: Fraction of the mean diagonal added to the diagonal.

    Returns:
        The upper triangular factor GPTQ divides its errors by.

    Raises:
        torch.linalg.LinAlgError: Still not positive-definite at 10^4x
            damping, which means far too little calibration data.
    """
    hessian = hessian / max(tokens, 1)
    damp = damping * hessian.diagonal().mean().clamp_min(EPS)
    applied = 0
    for escalation in (1, 10, 100, 1000, 10000):
        # The division above already made this a private copy, so the
        # diagonal takes the increment in place rather than through an
        # identity matrix the size of the Hessian.
        hessian.diagonal().add_((escalation - applied) * damp)
        applied = escalation
        try:
            chol = torch.linalg.cholesky(hessian)
            if escalation > 1:
                logger.warning(
                    "Hessian needed %dx the requested damping to invert, so "
                    "it is rank-deficient: give it more calibration tokens "
                    "than the layer has input channels",
                    escalation,
                )
            return torch.linalg.cholesky(
                torch.cholesky_inverse(chol), upper=True
            )
        except torch.linalg.LinAlgError:
            continue
    raise torch.linalg.LinAlgError("Hessian not positive-definite")


# Scale factors tried below `amax / quant_max`.  fp8_e5m3 carries three
# mantissa bits, so consecutive representable scales are 7-12% apart and a
# finer grid would collapse onto the same values.
SHRINK_GRID = (1.0, 0.93, 0.86, 0.79, 0.72)


@compiled_on_gpu(dynamic=False)
def block_scale(block, weights, qmap, scale_qmap, quant_max, shrink):
    """Choose each row's scale for one microscaling block.

    The scale is normally ``amax / quant_max``, which spends the whole
    codebook range on the block's largest value.  When one value dominates
    its 63 neighbours that costs them resolution, and clipping it can be the
    better trade -- GPTQ then compensates the clipped error into the columns
    it has not reached yet, which is why the search belongs inside the loop
    rather than in a preprocessing pass.  Shrinking keeps the deployed scale
    exact: the clipped value still lands on ``quant_max``, so recomputing the
    scale from the stored weights returns the same number.  Compiled whole,
    so the candidates are tried in a few fused kernels.

    Args:
        block: The block's weights, ``[out, 64]``.
        weights: Per-column Hessian diagonal, the salience each column's
            error is charged at.
        qmap: The lookup table or codebook the weight is quantized into.
        scale_qmap: The lookup table the scale is quantized into.
        quant_max: Largest magnitude ``qmap`` represents.
        shrink: Search below the maximum, instead of taking it.

    Returns:
        One scale per output row.
    """
    base = (
        calculate_mx_qparam(
            block,
            axes=-1,
            block_size=block.shape[-1],
            quant_max=quant_max,
            scale_qmap=scale_qmap,
        )
        .reshape(-1)
        .clamp_min(EPS)
    )
    if not shrink:
        return base

    chosen, lowest = base, None
    for factor in SHRINK_GRID:
        scale = (
            base
            if factor == 1.0
            else vmap((base * factor).bfloat16(), scale_qmap)
            .float()
            .clamp_min(EPS)
        )
        quantized = (
            quantize(block, scale[:, None], qmap=qmap).float() * scale[:, None]
        )
        error = ((block - quantized).pow(2) * weights).sum(dim=1)
        if lowest is None:
            chosen, lowest = scale, error
            continue
        better = error < lowest
        chosen = torch.where(better, scale, chosen)
        lowest = torch.where(better, error, lowest)
    return chosen


def salience_order(hessian, columns, within_block, block_size):
    """Return the column permutation putting the most salient blocks first.

    GPTQ compensates a column's rounding error into the columns it has not
    reached yet, so whatever it quantizes last is left with no budget.
    Ordinary act-order sorts individual columns by ``diag(H)``, which
    scatters the microscaling blocks; the hardware blocks over 64
    *contiguous* input channels, so the permutation has to move whole blocks
    to keep the deployed layout intact.

    Args:
        hessian: Accumulated ``X^T X`` for one linear.
        columns: The linear's input width.
        block_size: Input channels sharing one scale.
        within_block: Also order the columns inside each block by their
            own salience.  A block's scale is a max over its channels, so it
            does not depend on their order, and the permutation is inverted
            before the weights are stored -- which is what makes ordering
            inside a block free.  Column salience spans three orders of
            magnitude where block-mean salience spans one, so most of the
            spread lives here.

    Returns:
        A permutation of ``range(columns)`` under which every 64 consecutive
        positions hold exactly one original block, or None when the width is
        not a whole number of blocks and the reordering would split one.
    """
    if columns % block_size:
        return None
    blocks = columns // block_size
    diagonal = hessian.diagonal().reshape(blocks, block_size)
    block_order = torch.argsort(diagonal.mean(dim=1), descending=True)
    if within_block:
        rank = torch.argsort(diagonal, dim=1, descending=True)
    else:
        rank = torch.arange(block_size, device=hessian.device).expand(
            blocks, block_size
        )
    base = torch.arange(blocks, device=hessian.device).unsqueeze(1)
    return (base * block_size + rank)[block_order].reshape(-1)


@compiled_on_gpu(dynamic=False)
def quantize_block(block, scale, hinv, qmap):
    """Quantize one block column by column, compensating the columns after.

    Compiled whole, so each column step runs as one fused kernel rather
    than a dozen launches.

    Args:
        block: The block's weights, ``[out, n]``, as compensated so far.
        scale: Each row's scale for the block.
        hinv: The block's diagonal tile of the upper Cholesky factor of the
            inverse Hessian, ``[n, n]``.
        qmap: The lookup table or codebook the weight is quantized into.

    Returns:
        ``(quantized, deltas)``: the block on the grid, and each column's
        error over its pivot, which the caller carries into the columns
        after the block.
    """
    quantized, deltas = [], []
    for column in range(block.shape[1]):
        values = block[:, column]
        # The codebook names every axis of the weight, so a column
        # enters as the column it is rather than a bare vector.
        level = quantize(values[:, None], scale[:, None], qmap=qmap)[:, 0]
        level = level * scale
        delta = (values - level) / hinv[column, column].clamp_min(EPS)
        # The factor is upper triangular, so the columns already done take
        # nothing from this update.
        block = block - delta[:, None] * hinv[column]
        quantized.append(level)
        deltas.append(delta)
    return torch.stack(quantized, dim=1), torch.stack(deltas, dim=1)


def compensate_weight(
    weight,
    hessian,
    tokens,
    fake_quant,
    damping,
    act_order=True,
    within_block=True,
    shrink=False,
):
    """Return the GPTQ-compensated reconstruction on the deployed MX grid.

    Args:
        weight: The linear's weight ``[out, in]``.
        hessian: Accumulated ``X^T X`` for this linear.
        tokens: Rows that went into the Hessian, for the mean.
        fake_quant: The fake-quant the quantizer attached to this weight.
            It names the grid the compensation has to land on -- codebook,
            scale codebook, block size, range and blocking axis -- and they
            have to come from one module, since mixing a codebook with
            another tensor's geometry solves for a grid nothing deploys.
        damping: Fraction of the mean diagonal added to the diagonal.
        act_order: Quantize the most salient blocks first.
        within_block: Also order columns by salience inside each block.
        shrink: Search below `amax / quant_max` for each block's scale.

    Returns:
        The compensated weight in its original column order, every value
        already on the grid.

    Raises:
        ValueError: The weight's blocks do not run along its input
            channels, which is the axis this walks.
    """
    block_size = fake_quant.block_size
    axis = fake_quant.ch_axis
    axis = axis[0] if isinstance(axis, (tuple, list)) else axis
    if axis not in (-1, weight.dim() - 1):
        raise ValueError(
            f"weight blocks run along axis {axis}, but the compensation "
            f"walks the input channels of a {tuple(weight.shape)} weight"
        )
    perm = (
        salience_order(hessian, weight.shape[1], within_block, block_size)
        if act_order
        else None
    )
    if act_order and perm is None:
        logger.warning(
            "%d input channels is not a whole number of %d-channel blocks, "
            "so the salience ordering would split one and is skipped",
            weight.shape[1],
            block_size,
        )
    if perm is not None:
        hessian = hessian[perm][:, perm]
        weight = weight[:, perm]
    salience = hessian.diagonal().clone()
    hinv_chol = hessian_inverse_chol(hessian, tokens, damping)

    work = weight.float().clone()
    out = torch.zeros_like(work)
    columns = weight.shape[1]
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        scale = block_scale(
            work[:, start:end],
            salience[start:end],
            fake_quant.qmap,
            fake_quant.scale_qmap,
            fake_quant.quant_max,
            shrink,
        )
        out[:, start:end], deltas = quantize_block(
            work[:, start:end],
            scale,
            hinv_chol[start:end, start:end],
            fake_quant.qmap,
        )
        if end < weight.shape[1]:
            work[:, end:] -= deltas @ hinv_chol[start:end, end:]
    if perm is None:
        return out
    restored = torch.empty_like(out)
    restored[:, perm] = out
    return restored


#: Ops carrying a weight GPTQ can compensate.  An attention matmul is not
#: one of them: both its operands are activations, so there is nothing to
#: rewrite, and an elementwise op against a parameter -- a norm's scale --
#: has no Hessian to speak of.
COMPENSATED_OPS = (torch.ops.aten.linear.default,)


def _weight_of(node, modules):
    """Return the weight operand of a compute node, and its fake-quant.

    Args:
        node: A candidate compute node.
        modules: The graph module's named modules.

    Returns:
        ``(get_attr node, fake-quant module)``, or ``(None, None)`` when the
        node carries no weight, or carries one the quantizer left alone.
    """
    if node.op != "call_function" or node.target not in COMPENSATED_OPS:
        return None, None
    operand = node.args[1]
    if getattr(operand, "op", None) != "call_module":
        return None, None
    quantizer = modules[operand.target]
    operand = operand.args[0]
    if getattr(operand, "op", None) != "get_attr":
        return None, None
    return operand, quantizer


def _live_at(nodes, index):
    """Return the values a run resuming at a cut has to be handed.

    Those are the activations defined before the cut and still used at or
    after it.  Parameters are excluded -- they are read once into a table
    the walk shares, rather than carried per input -- and placeholders are
    kept unconditionally because nothing can recompute them.

    Args:
        nodes: The graph's nodes, in order.
        index: Cut position in ``nodes``.

    Returns:
        The set of nodes to carry across the cut.
    """
    position = {node: i for i, node in enumerate(nodes)}
    # Placeholders are kept whatever the cut, including one before them:
    # they sit at the head of the graph, so a cut at the start leaves them
    # ahead of it, and nothing can recompute an input.
    carry = {node for node in nodes if node.op == "placeholder"}
    for node in nodes[:index]:
        if node.op in ("get_attr", "placeholder"):
            continue
        if any(position.get(use, index) >= index for use in node.users):
            carry.add(node)
    return carry


def _advance(interpreter, stretch, keep):
    """Run the nodes of a stretch the environment does not hold yet.

    Each value is dropped once its last user has run, as
    ``Interpreter.run`` does, so a stretch holds only what is still live
    rather than every intermediate it made.

    Args:
        interpreter: The interpreter whose ``env`` the stretch reads and
            fills.
        stretch: The nodes to run, in order.
        keep: Values to hold past their last use, for the caller to read
            after the stretch.
    """
    for node in stretch:
        if node in interpreter.env:
            continue
        interpreter.env[node] = interpreter.run_node(node)
        for value in interpreter.user_to_last_uses.get(node, ()):
            if value not in keep:
                del interpreter.env[value]


#: Module path of one decoder layer, or of the norm after its attention,
#: where the walk parks.  The MLP fold parks after the attention it has
#: just replayed with its compensated weights, so the next layer's pass
#: replays the MLP alone.  A bare top-level name does not match, so
#: ``lm_head`` resumes from the last layer rather than opening a spot at its
#: own first node.
PARKING = re.compile(r".*\.\d+(?:\.post_attention_layernorm)?")

#: Module path of a layer's attention or MLP half, the unit the walk folds.
FOLDS = re.compile(r".*\.\d+(?:\.\w+)?|[^.]+")


def _blocks(model, nodes, pattern):
    """Return the position where each block ``pattern`` names starts.

    Export records the module path every node came from, so a pattern over
    that path names an architectural unit.

    Args:
        model: The graph module being walked.
        nodes: Its nodes, in order.
        pattern: Compiled regex naming a block from a module path.

    Returns:
        Ascending positions, one per block, empty when the graph has no
        repeated structure to read.
    """
    scopes = get_node_name_to_scope(model)
    starts, seen = [], set()
    for index, node in enumerate(nodes):
        # A compute node records ``(path, type, order)``; a parameter records
        # a list of them, and carries no path worth reading.
        entry = scopes.get(node.name) or ("",)
        path = entry[0] if isinstance(entry[0], str) else ""
        match = pattern.match(path)
        if match is None:
            continue
        block = match.group(0)
        if block not in seen:
            seen.add(block)
            starts.append(index)
    return starts


def _carried_bytes(environment):
    """Return the bytes one input's carried values occupy.

    Measured from the tensors themselves rather than from graph metadata,
    which a non-tensor placeholder does not carry.

    Args:
        environment: The pruned environment for one calibration input.

    Returns:
        Total bytes of the activations in it.
    """
    return sum(
        value.numel() * value.element_size()
        for value in environment.values()
        if isinstance(value, torch.Tensor)
    )


#: Device memory held back from the activation cache, for the compensation
#: working tensors -- lm_head's alone are about 4 GiB -- and for headroom
#: against fragmentation.  What is left is what the cache may hold; past it
#: the cache moves to host memory, so this trades transfer time for device
#: memory, never recomputation.
CACHE_RESERVE = 12 << 30

#: Calibration windows one forward carries when the graph's batch axis is
#: dynamic.  At this size the forward already keeps the GPU busy, so a larger
#: batch only holds more memory.
BATCH = 8


def _cache_budget(device, reserve):
    """Return the bytes of activation cache a device can hold.

    Args:
        device: Where the walk runs.
        reserve: Bytes to leave free for everything else.

    Returns:
        The budget, or infinity where there is no device memory to run out
        of and the cache can simply stay put.
    """
    if device.type != "cuda":
        return float("inf")
    return max(0, torch.cuda.mem_get_info(device)[0] - reserve)


@torch.no_grad()
def gptq(
    model,
    calibration,
    skip=(),
    damping=0.01,
    act_order=True,
    within_block=True,
    shrink=False,
    folded=True,
    budget=None,
    reserve=CACHE_RESERVE,
    fast_hessian=True,
):
    """Compensate every quantized weight in a prepared model, in place.

    Walks the graph once, in node order.  On reaching a weight it
    accumulates that weight's Hessian from the activation about to enter it
    -- already fake-quantized, so the compensation is solved against the
    operand the hardware will see -- then rewrites the weight before moving
    past it.  Weights sharing an activation are compensated together, which
    is exact: their Hessians are equal and none of them feeds another.

    Where the walk *parks* is not where it stops.  Held state sits where
    only the residual stream crosses -- between decoder layers, and between
    a layer's attention and its MLP -- and the stretch from there to each
    weight is re-run.  Parking at the weight instead holds two attention
    score matrices per input, which for a long calibration set does not fit
    on a device.  Re-running a stretch is arithmetic and, under
    microscaling, changes nothing: the scale is computed from the input
    every time.

    The codebook, the scale codebook and the range all come from the
    fake-quant the quantizer already attached to each weight, so nothing has
    to be told which tables are being deployed.

    Args:
        model: Graph module from ``prepare_pt2e``, before ``convert_pt2e``.
        calibration: The calibration set; each entry is a tuple of
            positional arguments matching the graph's placeholders.  When
            the graph was exported with a dynamic batch axis, ``BATCH``
            entries are stacked along it into each forward.
        skip: Substrings of weight names to leave alone.
        damping: Fraction of the mean Hessian diagonal added before
            inverting.
        act_order: Quantize the most salient blocks of each weight first.
        within_block: Also order columns by salience inside each block.
        shrink: Search below ``amax / quant_max`` for each block's scale.
        folded: Hold a compensated weight back until the walk leaves the
            stretch it was compensated in, so the weights sharing that
            stretch are all solved against activations the *original*
            weights produced -- what a block-folded GPTQ does.  Turning it
            off makes every weight see its predecessors' compensation,
            which is the sequential variant.  Neither is more correct;
            folded measured 0.027 better on Llama-3.1-8B at 4 bits, and is
            the default for that reason alone.
        budget: Bytes of cached activation to keep on the device.  Defaults
            to whatever the device has free beyond ``reserve``.  A cache
            larger than this moves to host memory instead; the frontier
            advances either way.
        reserve: Device memory to leave free, when ``budget`` is derived.
        fast_hessian: Multiply each Hessian's ``X^T X`` on the tensor cores
            in the activation's own dtype, with an fp32 result.  For a bf16
            activation every product is still exact and only the summation
            changes, to tensor-core accumulation; False casts to fp32 first,
            the slower exact reference.

    Returns:
        The same model, its weights compensated.
    """
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    placeholders = [node for node in nodes if node.op == "placeholder"]
    device = next(model.parameters()).device
    calibration = list(calibration)
    windows = len(calibration)
    # A dynamic batch axis lets one forward carry several windows, paying
    # the walk's per-node cost once for all of them.  An argument that is
    # not a tensor -- a flag -- is the same in every entry, so the batch
    # takes it from its first.
    example = placeholders[0].meta["val"]
    batch = BATCH if isinstance(example.shape[0], torch.SymInt) else 1
    calibration = [
        tuple(
            torch.cat(values) if torch.is_tensor(values[0]) else values[0]
            for values in zip(*calibration[start : start + batch])
        )
        for start in range(0, windows, batch)
    ]
    if budget is None:
        budget = _cache_budget(device, reserve)

    # Every input's run reads the same parameters, and compensation writes
    # through `data`, so one table of them serves the whole walk.
    parameters = {
        node: fetch_attr(model, node.target)
        for node in nodes
        if node.op == "get_attr"
    }

    operands, position = {}, {}
    for index, node in enumerate(nodes):
        weight, fake_quant = _weight_of(node, modules)
        if weight is None or any(name in weight.target for name in skip):
            continue
        operands.setdefault(node.args[0], []).append((weight, fake_quant))
        position.setdefault(node.args[0], index)

    # Park and fold at the model's own boundaries, at two granularities.
    # Only the residual stream crosses a decoder layer or the norm after its
    # attention, so parking there holds one hidden state; a fold boundary
    # sits on its own first weight, whose fake-quant would then be carried
    # across the cut instead of recomputed from the compensated value.  A
    # graph with no repeated structure parks at each weight: correct, but it
    # holds a mid-layer live set that a long calibration set cannot fit on a
    # device.
    cuts = list(position.values())
    parks = [s for s in _blocks(model, nodes, PARKING) if s <= max(cuts)]
    folds = [s for s in _blocks(model, nodes, FOLDS) if s <= max(cuts)]
    checkpoints = [0] + (parks or cuts)
    boundaries = [0] + (folds or cuts)
    total = sum(len(weights) for weights in operands.values())
    logger.info(
        "compensating %d weights in %d groups over %d calibration inputs "
        "in batches of %d, parking the cache at %d %s and folding at %d",
        total,
        len(operands),
        windows,
        batch,
        len(checkpoints),
        "residual-stream boundaries" if parks else "weights",
        len(boundaries),
    )

    # One pass per fold, resuming from the parking spot before it.  Weights
    # sharing a fold do not see each other's compensation, so one pass can
    # accumulate all their Hessians; a sequential run cannot batch, since
    # each weight has to see the last one.  Several folds share a parking
    # spot, and re-running the stretch between them changes nothing.
    spans, previous = [], None
    for activation, weights in operands.items():
        until = position[activation]
        fold = max(point for point in boundaries if point <= until)
        if folded and spans and fold == previous:
            spans[-1][1].append((activation, until, weights))
        else:
            target = max(point for point in checkpoints if point <= until)
            spans.append((target, [(activation, until, weights)]))
        previous = fold
    logger.info("%d groups batched into %d passes", len(operands), len(spans))

    started, done = time.monotonic(), 0
    frontier, offload, pending = 0, False, []
    cache = [dict(zip(placeholders, entry)) for entry in calibration]
    # The walk calls ``run_node`` itself, which reads nothing but ``env``,
    # and frees values by the interpreter's last-use table, which depends on
    # the graph alone, so one interpreter serves every batch of every pass.
    interpreter = Interpreter(model)
    for target, members in spans:
        until = max(cut for _, cut, _ in members)
        # One pass is one fold, so what the pass before it held commits now.
        for parameter, value in pending:
            parameter.data.copy_(value)
        pending.clear()
        # A node reading parameters alone -- a weight's fake-quant -- gives
        # every batch the same value, and the weights change only between
        # passes, so the stretch's ones run once here rather than per batch.
        shared = dict(parameters)
        interpreter.env = shared
        for node in nodes[frontier:until]:
            inputs = node.all_input_nodes
            if inputs and all(arg.op == "get_attr" for arg in inputs):
                shared[node] = interpreter.run_node(node)
        carry = _live_at(nodes, target)
        keep = {activation for activation, _, _ in members}
        hessians = {cut: None for _, cut, _ in members}
        rows = 0
        for index, environment in enumerate(cache):
            interpreter.env = dict(shared)
            interpreter.env.update(
                (
                    node,
                    (
                        value.to(device)
                        if offload and torch.is_tensor(value)
                        else value
                    ),
                )
                for node, value in environment.items()
            )
            # Advance to the checkpoint first, so what the cache holds is
            # exactly the live set there, then carry on to the weight.
            # Anything the resumed run was handed is not recomputed.
            _advance(interpreter, nodes[frontier:target], keep)
            carried = {
                node: value
                for node, value in interpreter.env.items()
                if node in carry
            }
            _advance(interpreter, nodes[target:until], keep)
            # Every member's operand is defined before its own cut, and the
            # pass ran to the last of them, so they are all in hand.
            for activation, cut, _ in members:
                inputs = interpreter.env[activation]
                inputs = inputs.reshape(-1, inputs.shape[-1])
                if hessians[cut] is None:
                    hessians[cut] = torch.zeros(
                        inputs.shape[-1],
                        inputs.shape[-1],
                        device=inputs.device,
                        dtype=torch.float32,
                    )
                # Only CUDA multiplies one dtype into another's result.
                if fast_hessian and inputs.is_cuda:
                    torch.addmm(
                        hessians[cut],
                        inputs.T,
                        inputs,
                        out_dtype=torch.float32,
                        out=hessians[cut],
                    )
                else:
                    inputs = inputs.float()
                    hessians[cut].addmm_(inputs.T, inputs)
                if cut == until:
                    rows += inputs.shape[0]
                del inputs
            # Overwriting the slot in place releases the environment this
            # input arrived with, so one cache is live at a time, not two.
            interpreter.env = {}
            share = _carried_bytes(carried)
            if not offload and share * len(cache) > budget:
                # The live set grows and shrinks along the graph, so a cache
                # that fits at one weight may not at the next.  It only ever
                # moves to host memory, never back, so it cannot thrash.
                offload = True
                logger.info(
                    "carrying %.1f MiB per input at node %d, which is past "
                    "the %.1f GiB budget; moving the cache to host memory",
                    share / 2**20,
                    until,
                    budget / 2**30,
                )
                for stale in range(index):
                    cache[stale] = {
                        node: (value.cpu() if torch.is_tensor(value) else value)
                        for node, value in cache[stale].items()
                    }
            cache[index] = (
                {
                    node: value.cpu() if torch.is_tensor(value) else value
                    for node, value in carried.items()
                }
                if offload
                else carried
            )
        for _, cut, weights in members:
            for weight, fake_quant in weights:
                parameter = parameters[weight]
                value = compensate_weight(
                    parameter.data.float(),
                    hessians[cut],
                    rows,
                    fake_quant,
                    damping,
                    act_order=act_order,
                    within_block=within_block,
                    shrink=shrink,
                ).to(parameter.dtype)
                if folded:
                    pending.append((parameter, value))
                else:
                    parameter.data.copy_(value)
                done += 1
                logger.info(
                    "[%3d/%d] %s  %.0fs",
                    done,
                    total,
                    weight.target,
                    time.monotonic() - started,
                )
        hessians.clear()
        frontier = target
    for parameter, value in pending:
        parameter.data.copy_(value)
    return model
