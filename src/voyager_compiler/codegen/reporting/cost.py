"""Latency (cycles) and DRAM-traffic (bytes) model.

Compute cycles are driven by the **anchor only** — a fused ``call_module``'s
anchor (GEMM/conv) and its pointwise tail run pipelined, so the anchor
dominates.  DRAM bytes apply to ``async_copy`` only (``insert`` is zero-time
DPS bookkeeping).  Byte counts use the physical storage dtype via
``dtype_byte_size`` (sub-byte / quantized aware), so packed formats report
their real footprint.
"""

import math
from typing import Tuple

import torch
from torch.fx import GraphModule, Node

import interstellar

# Reporting estimates with Voyager's formulas for now; they are to be made
# backend-general.
from voyager_compiler.backend.voyager.vector import (
    gemv_lanes,
    pool_taps,
    vector_op_utilization,
)
from voyager_compiler.codegen.node_info import (
    dtype_byte_size,
    get_anchor_node,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_gemm_op,
    is_nop,
    is_pooling,
)
from voyager_compiler.codegen.reporting.model import OpInfo
from voyager_compiler.codegen.transform.bufferize.sync import _is_matrix_op
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    PassTiming,
)
from voyager_compiler.hardware_config import AcceleratorConfig

# interstellar's loop ids, named as the IR's ``LoopIndex`` names them.
_LOOP_NAMES = ("FX", "FY", "OX", "OY", "OC", "IC", "ON")
# The mapping levels a nest is told apart by: L1 and L2, the levels the IR's
# ``Tiling`` records (level 0 is the PE).
_NEST_LEVELS = (1, 2)


def _val(node):
    """The node's propagated tensor value: ``.value`` at the top level, or the
    exported ``meta['val']`` inside loop / cond bodies (mirrors codegen).

    For a multi-output node whose value is a tuple (e.g. ``quantize_mx`` ->
    ``(scale, quantized)``), return its largest tensor — the quantized tile,
    not the small per-block scale — so it is sized like the op it represents."""
    if not isinstance(node, Node):
        return None
    for v in (getattr(node, "value", None), node.meta.get("val")):
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (tuple, list)):
            tensors = [t for t in v if isinstance(t, torch.Tensor)]
            if tensors:
                return max(tensors, key=lambda t: t.numel())
    return None


def _shape(node) -> Tuple[int, ...]:
    """Every node ``_shape`` is asked about is a node of interest (an anchor, a
    tile buffer, a DMA operand) that must carry a shape, so a missing value is
    an error, not a silent ``None`` (``_val`` already reduces a multi-output
    node to its representative tensor)."""
    v = _val(node)
    if v is None:
        raise ValueError(f"{node} has no shape (a sized value is required)")
    return tuple(int(d) for d in v.shape)


def _dtype(node):
    """Logical (quantized) dtype string if present, else the physical dtype."""
    dt = node.meta.get("dtype") if isinstance(node, Node) else None
    if dt is not None:
        return dt
    return _val(node).dtype


def _dtypes(out_node, *in_nodes) -> str:
    """``in,...->out`` logical dtypes, for the op's calibration signature."""
    ins = ",".join(str(_dtype(n)) for n in in_nodes if isinstance(n, Node))
    return f"{ins}->{_dtype(out_node)}"


# --------------------------------------------------------------------------
# DRAM traffic
# --------------------------------------------------------------------------


def tile_bytes(buffer_node: Node, sizes) -> int:
    """Physical bytes of a ``sizes`` tile of ``buffer_node`` (rounded up; a
    sub-byte dtype gives a fractional bytes-per-element)."""
    n = math.prod(sizes)
    return math.ceil(n * dtype_byte_size(_dtype(buffer_node)))


def dram_cycles(n_bytes: int, cost: AcceleratorConfig) -> int:
    # GB/s == bytes/ns, so n_bytes / dram_bandwidth is the transfer time in ns;
    # add the access latency (ns) and convert to cycles with frequency (GHz).
    time_ns = cost.dram_access_latency + n_bytes / cost.dram_bandwidth
    return math.ceil(time_ns * cost.frequency)


# --------------------------------------------------------------------------
# Utilization
# --------------------------------------------------------------------------


def op_utilization(
    node: Node,
    units: Tuple[str, ...],
    ideal_cycles: int,
    cost: AcceleratorConfig,
) -> float:
    """The utilization to charge ``node``.

    A matrix op interstellar mapped carries the search's result
    (``meta["tile_result"]``, stamped on it by the builder), whose
    ``cost.compute`` is one grid step's ``PassTiming``, whose ``steady`` is
    the array's span of the step, so its utilization is the ideal-to-actual
    ratio: the padding alone, the ramps being the schedule's.  A backend
    with no ``PassTiming`` gives the step's cycles as one float instead.

    Everything else runs on the vector unit, charged by
    ``vector_op_utilization`` (shared with the vector L2-tiling cost model)
    at ``cost.scratchpad_bytes_per_cycle`` SRAM bandwidth -- including the
    fully-connected case, which it sizes by the streamed weight and its bank
    switches over the anchor's shape, the tile once bufferized, and a pool,
    whose fetch it walks for bank switches over the anchor's input tile.
    ``ideal_cycles`` is one tile's, so it also folds in the per-launch
    overhead the tiling model charges, and the two cannot disagree on what a
    tile costs.
    """
    result = node.meta.get("tile_result")
    if (
        "mma" in units
        and result is not None
        and isinstance(result.tiling, interstellar.MappingPoint)
        and ideal_cycles > 0
    ):
        step = result.cost.compute
        if isinstance(step, PassTiming):
            step = step.steady
        return min(1.0, ideal_cycles / step)
    anchor = get_anchor_node(node) or node
    tile = None
    if is_fully_connected(anchor):
        tile = (_shape(anchor)[-1], _shape(anchor.args[0])[-1])
    elif is_pooling(anchor):
        tile = _shape(anchor.args[0])
    return vector_op_utilization(node, cost, ideal_cycles, tile)


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


def tiling_text(node: Node) -> str:
    """The tiling the search chose for ``node``, as the Operations sheet's
    and the Op calibration sheet's ``Tiling`` cell: the reduction scratch's
    slot count, then a matrix op's L1 and L2 loops with a blocking above 1
    in loop order (``slots 1; L1 OX128 IC8; L2 OC8 OX2``), or a grid's tile
    sizes and counts.  Blank for an op no search tiled."""
    result = node.meta.get("tile_result")
    if result is None:
        return ""
    parts = [f"slots {result.placement.scratch_slots}"]
    mapping = result.tiling
    if isinstance(mapping, interstellar.MappingPoint):
        for level in _NEST_LEVELS:
            ranked = sorted(
                range(len(_LOOP_NAMES)),
                key=lambda loop: mapping.loop_orders[loop][level],
            )
            loops = [
                f"{_LOOP_NAMES[loop]}{mapping.loop_blockings[loop][level]}"
                for loop in ranked
                if mapping.loop_blockings[loop][level] > 1
            ]
            parts.append(" ".join([f"L{level}", *loops]))
    else:
        tile = "x".join(str(t) for t in mapping.tile_sizes)
        grid = "x".join(str(g) for g in mapping.grid)
        parts.append(f"tile {tile} grid {grid}")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------


def _module_ops(gm: GraphModule) -> list:
    """The ops of a fused module in program order, nested modules flattened
    and views left out, as the sync pass lists a dispatch's ops."""
    ops = []
    for n in gm.graph.nodes:
        if n.op == "call_module":
            ops += _module_ops(gm.get_submodule(n.target))
        elif n.op == "call_function" and not is_nop(n):
            ops.append(n)
    return ops


def op_passes(node: Node) -> dict:
    """The passes the sync pass gives compute ``node``, as ``OpInfo``
    fields: a matrix-class pass when one of its ops is a matrix op
    (``_is_matrix_op``), on the SpMM unit for a bare SpMM, on the
    matrix-vector unit for a fully-connected GEMM, else on the systolic
    array (``engine``); a vector pass when it has any other op or its
    matrix op is an SpMM (``vector_pass``); and the pass that posts, the
    matrix-class one only when no op follows the matrix op and it is no
    SpMM (``posts_on``).  A tiled op on the array also carries its search's
    ``PassTiming`` (``timing``)."""
    sub = node.meta.get("submodule")
    ops = _module_ops(sub) if isinstance(sub, GraphModule) else [node]
    matrix = next((op for op in ops if _is_matrix_op(op)), None)
    if matrix is None:
        return {"engine": "", "vector_pass": True, "posts_on": "vector"}
    spmm = not is_gemm_op(matrix) or matrix.kwargs.get("A_data") is not None
    if not is_gemm_op(matrix):
        engine = "spmm"
    elif is_fully_connected(matrix):
        engine = "mvm"
    else:
        engine = "mma"
    tail = ops.index(matrix) < len(ops) - 1
    result = node.meta.get("tile_result")
    return {
        "engine": engine,
        "vector_pass": len(ops) > 1 or spmm,
        "posts_on": "vector" if tail or spmm else "matrix",
        "spmm": spmm,
        "timing": (
            result.cost.compute
            if engine == "mma"
            and result is not None
            and result.cost is not None
            and isinstance(result.cost.compute, PassTiming)
            else None
        ),
    }


# --------------------------------------------------------------------------
# Compute cycles
# --------------------------------------------------------------------------


def op_info(node: Node, cost: AcceleratorConfig) -> OpInfo:
    """Classify a compute node and compute its ideal (100%-utilization) cycle
    count.  GEMM/conv use the ``unroll[0]*unroll[1]`` systolic throughput;
    vector ops use the ``unroll[1]`` lane count.

    All shapes come from the *anchor* -- the real GEMM/conv/vector op (inside
    the submodule for a fused ``call_module``, the node itself when bare).  Its
    inner nodes are shape-propagated at fusion, so ``anchor.args[i]`` is
    unambiguously the i-th operand (a fused wrapper's args may interleave
    scales / codes / bias).
    """
    anchor = get_anchor_node(node)
    mma_macs = max(1, cost.pe_array_size[0] * cost.pe_array_size[1])
    vec_lanes = max(1, cost.vector_lanes)
    out = _shape(anchor)

    # A fused submodule is priced as driving the vector unit too (its tail
    # runs on the VU, even a bare reshape); a bare matrix op as occupying
    # only the matrix unit.
    fused = isinstance(node.meta.get("submodule"), GraphModule)
    matrix_units = ("mma", "vector") if fused else ("mma",)
    # The lanes its passes occupy: the matrix-class unit, the SpMM unit
    # beside the array, the vector unit.
    passes = op_passes(node)
    lanes = [passes["engine"]] if passes["engine"] else []
    if passes.get("spmm") and passes["engine"] == "mma":
        lanes.append("spmm")
    if passes["vector_pass"]:
        lanes.append("vector")
    lanes = tuple(lanes)

    if is_conv2d(anchor):
        w = _shape(anchor.args[1])
        nhwc = anchor.kwargs.get("layout") == "nhwc"
        if nhwc and not is_depthwise_conv(anchor):  # HWIO = [kh, kw, C, K]
            kh, kw, c = w[0], w[1], w[2]
        else:  # OIHW = [K, C, kh, kw]
            c, kh, kw = w[1], w[2], w[3]
        macs = math.prod(out) * c * kh * kw
        ideal = math.ceil(macs / mma_macs)
        return OpInfo(
            node.name,
            "conv",
            ideal,
            {
                "macs": macs,
                "input": _shape(anchor.args[0]),
                "weight": w,
                "output": out,
                "dtypes": _dtypes(anchor, anchor.args[0], anchor.args[1]),
                "tiling": tiling_text(node),
            },
            units=lanes,
            utilization=op_utilization(node, matrix_units, ideal, cost),
            **passes,
        )

    if is_gemm_op(anchor):
        # MACs = prod(out) * K; each output element costs K = in[-1] MACs
        # (folds batch dims; works for linear / matmul / bmm).
        inp = _shape(anchor.args[0])
        macs = math.prod(out) * inp[-1]
        # A fully-connected (matrix-vector) GEMM streams through the
        # matrix-vector or the vector unit (``gemv_lanes``), not the array.
        fc = is_fully_connected(anchor)
        divisor = gemv_lanes(anchor, cost) if fc else mma_macs
        ideal = math.ceil(macs / divisor)
        units = ("vector",) if fc else matrix_units
        return OpInfo(
            node.name,
            "gemm",
            ideal,
            {
                "macs": macs,
                "input": inp,
                "weight": _shape(anchor.args[1]),
                "output": out,
                "dtypes": _dtypes(anchor, anchor.args[0], anchor.args[1]),
                "tiling": tiling_text(node),
            },
            units=lanes,
            utilization=op_utilization(node, units, ideal, cost),
            **passes,
        )

    # Vector op: work is the larger of input / output element count -- a
    # reduction reads its whole input to make a smaller output.  Use the
    # anchor's primary input (all_input_nodes[0]) to skip qmap / codebook
    # operands.
    in_nodes = anchor.all_input_nodes
    in_shape = _shape(in_nodes[0]) if in_nodes else ()
    ops = max(math.prod(out), math.prod(in_shape))
    if is_pooling(anchor):
        # A pool's work is one fetched lane group per window tap and output.
        ops = math.prod(out) * pool_taps(anchor)
    ideal = math.ceil(ops / vec_lanes)
    return OpInfo(
        node.name,
        "vector",
        ideal,
        {
            "ops": ops,
            "input": in_shape,
            "output": out,
            "dtypes": _dtypes(anchor, *in_nodes[:1]),
            "tiling": tiling_text(node),
        },
        units=lanes,
        utilization=op_utilization(node, ("vector",), ideal, cost),
        **passes,
    )
