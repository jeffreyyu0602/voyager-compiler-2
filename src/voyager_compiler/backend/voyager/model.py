"""Voyager's runtime and capacity model, ``VoyagerModel``.

What the shared tile search asks about a Voyager accelerator, answered from
its ``AcceleratorConfig``.  ``VoyagerModel.prepare`` gives each op its model:
a GEMM / conv is a ``VoyagerGemm`` (``gemm.py``), mapped onto the interstellar
memory hierarchy, its tile packed into scratchpad banks and a tiling priced by
the ``RuntimeCalculator``; a vector op, GEMV or pool is a ``VoyagerVector``,
``VoyagerGemv`` or ``VoyagerPool`` (``vector.py``), its operand tiles packed
into banks.  ``VoyagerAttention`` (``attention.py``) fits and prices a flash
attention.
"""

import math

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    get_arg_value,
    is_conv2d,
    is_fully_connected,
    is_gemm_op,
    is_pooling,
)
from voyager_compiler.codegen.transform.tiling.lowering import matrix_layer
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    AttentionSpace,
    GridSpace,
    MatrixSpace,
    RuntimeModel,
)

from .attention import VoyagerAttention
from .gemm import VoyagerGemm
from .vector import VoyagerGemv, VoyagerPool, VoyagerVector


def _matrix_space(config):
    """The 4-level (PE / L1 / L2 / DRAM) interstellar architecture a GEMM /
    conv is mapped onto, and its schedule hints.

    ``config.pe_array_size`` is ``(ic_dim, oc_dim)``.  The DRAM transfer
    accounting is in absolute bytes (bandwidth as ``config.bytes_per_cycle``
    is read when a tiling is priced), so it is not normalized by any element
    width.

    The L0/L1 capacities are slot arrays: one fixed-width slot per element
    (the max dtype in a mixed-precision design; narrower dtypes are padded
    into a full slot), so they are element / slot counts and the fit check
    is dtype-independent.  The flat L2/L3 byte pools let sub-byte operands
    pack, so those stay in bytes.

    L2 is planned exactly the way ``plan_memory`` allocates it: the capacity
    is ``usable_scratchpad_size``, the scratchpad above whatever
    ``scratchpad_offset`` reserves, and a ping-ponged source is charged one
    bank per slot (``VoyagerGemm.fit``) rather than the capacity being
    halved, which could not express a buffer allocated only once -- the
    reduction scratch.

    ``config`` carries physical units (GB); the interstellar model wants
    bytes, so ``dram_size`` is scaled to bytes here.
    """
    ic_dim, oc_dim = config.pe_array_size

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                config.input_buffer_size * ic_dim,
                config.accum_buffer_size * oc_dim,
                config.weight_buffer_size * oc_dim,
            ],
            [config.usable_scratchpad_size],
            [config.dram_size * 1024**3],  # GB -> bytes
        ],
        buf_access_cost_list=[
            [1, 1, 1],
            [10, 10, 10],
            [100],
            [1000],
        ],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
        para_count_list=[ic_dim * oc_dim, 1, 1, 1],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        mac_capacity=0,
        partition_mode=[0, 0, 0, 0],
        invalid_underutilized=False,
        bank_size_list=[None, None, config.bank_size, None],
    )

    # L1 IC is outermost; the inner order is pinned to FY > FX > OY > OX.
    # OX/OY innermost is never slower -- the L1 sweep costs
    # ``max(loading, reused_tile) * remaining``, monotone in the reused
    # tile -- and the arrangement of the loops above them ties on both
    # runtime and energy, so FX/FY are fixed to the representative the
    # search's first-seen tie-break picked anyway.  FX=2/FY=3 assumes a
    # square kernel (equal FX/FY blocking); a non-square model may
    # prefer them swapped.
    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": ic_dim},
                "level1": {"order": -1},
                "level2": {"order": 0},
                "level3": {"order": 0},
            },
            "OC": {
                "level0": {"order": 0, "partitioning_size": oc_dim},
            },
            "OX": {
                "level1": {"order": 0},
            },
            "OY": {
                "level1": {"order": 1},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 2},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 3},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 4
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return MatrixSpace(architecture, schedule)


# The regions the vector unit keeps beside a batched reduction's tile between
# its passes: its statistics (one value per row), then its tile-sized
# regions.  A softmax holds a max and a sum, a layer_norm a mean and a
# variance plus the normalized tile, an rms_norm a mean square.
_SCRATCH_REGIONS = {
    torch.ops.aten.softmax.int: (("max", "sum"), ()),
    torch.ops.aten.layer_norm.default: (("mean", "variance"), ("normalized",)),
    torch.ops.aten.rms_norm.default: (("mean_square",), ()),
}


def _reduced_dims(node, ndim: int) -> set:
    """The dims ``node`` reduces over, as non-negative indices."""
    if node.target is torch.ops.aten.softmax.int:
        return {get_arg_value(node, 1, "dim", -1) % ndim}
    normalized_shape = get_arg_value(node, 1, "normalized_shape") or ()
    return set(range(ndim - len(normalized_shape), ndim))


class VoyagerModel(RuntimeModel):
    """A Voyager accelerator: a weight-stationary PE array behind a banked
    scratchpad, a vector unit ``config.vector_lanes`` wide, one DRAM
    channel.

    Attributes:
        matrix_space: The interstellar hierarchy a GEMM / conv is mapped
            onto (``_matrix_space``).
        grid_space: A vector op's or a GEMV's tilings: whole lane groups of
            the vector unit, priced when DRAM has a bandwidth to price their
            transfers at, else the largest that fits.
        pool_space: A pool's: whole lane groups, the largest that fits.
        attention_space: A flash attention's: query and key tiles in
            multiples of the wider PE-array dim, the head padded to a
            multiple of both, since the scores product contracts over it and
            the context product emits it.
    """

    def __init__(self, config, accumulate_fp32):
        super().__init__(config, accumulate_fp32)
        self.matrix_space = _matrix_space(config)
        self.grid_space = GridSpace(
            config.vector_lanes, ranked=config.dram_bandwidth is not None
        )
        self.pool_space = GridSpace(config.vector_lanes, ranked=False)
        self.attention_space = AttentionSpace(
            unit=max(config.pe_array_size),
            head_multiple=math.lcm(*config.pe_array_size),
        )

    def prepare(self, node):
        """``node``'s op model: a GEMV runs on the vector unit
        (``VoyagerGemv``), any other GEMM / conv is mapped onto the PE array
        (``VoyagerGemm``), and a pool or any other op tiles the vector
        unit's lanes (``VoyagerPool``, ``VoyagerVector``).  A conv over three
        input channels, an image network's first layer, stays untiled
        (``None``)."""
        anchor = get_anchor_node(node)
        if is_gemm_op(anchor):
            if is_fully_connected(anchor):
                return VoyagerGemv(node, self.grid_space, self)
            if is_conv2d(anchor) and matrix_layer(anchor).nifm == 3:
                return None
            return VoyagerGemm.from_node(node, self)
        if is_pooling(anchor):
            return VoyagerPool(node, self.pool_space, self)
        return VoyagerVector(node, self.grid_space, self)

    def prepare_attention(self, node, acc_dtype, bool_mask):
        """A ``VoyagerAttention`` for the flash-attention ``node``."""
        return VoyagerAttention(
            node, self.attention_space, self.config, acc_dtype, bool_mask
        )

    def kernel_scratch(self, node, out_tile):
        """The regions the vector unit keeps beside an op's tiles that no FX
        node names: a batched reduction's (a norm's, a softmax's) between its
        passes, and none for any other op.

        A statistic is one value per row, held across a lane group; a
        tile-sized region (layer_norm's normalized output) is the whole tile.
        A fused group is scanned whole: its reduction need not be its anchor.

        Args:
            node: The op being bufferized or tiled, bare or fused.
            out_tile: One tile of ``node``'s output, or one per output, the
                last naming the iteration space.

        Returns:
            ``[(name, shape, dtype), ...]``, one per region, in the physical
            dtype of the op's input.
        """
        if not out_tile:
            return []
        if isinstance(out_tile[0], (tuple, list)):
            out_tile = out_tile[-1]
        tile = tuple(out_tile)

        submod = node.meta.get("submodule")
        ops = submod.graph.nodes if submod is not None else [node]
        scratch = []
        for op in ops:
            if op.target not in _SCRATCH_REGIONS:
                continue
            stat_names, tile_names = _SCRATCH_REGIONS[op.target]
            reduced = _reduced_dims(op, len(tile))
            stat = list(tile)
            for i in reduced:
                stat[i] = 1
            stat[max(reduced)] = self.config.vector_lanes
            dtype = get_arg_value(op, 0, "input").value.dtype
            scratch += [(name, tuple(stat), dtype) for name in stat_names]
            scratch += [(name, tile, dtype) for name in tile_names]
        return scratch
