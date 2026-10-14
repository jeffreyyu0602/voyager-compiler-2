"""The runtime model a backend plugs into the shared tile search.

The shared search helpers -- interstellar for GEMM / conv, the grid
enumeration for vector, GEMV and pool ops, the attention search -- generate
candidate tilings and pick the winner.  What depends on the hardware comes
from the backend's :class:`RuntimeModel`: for each op it prepares an
:class:`OpModel`, whose ``space`` says which candidates exist and whose
``fit`` and ``cost`` judge each one.  A GEMM / conv's tiling is
interstellar's own ``MappingPoint``, read in place; a grid op's is a
:class:`GridTiling` and a flash attention's an :class:`AttentionTiling`.  The
winner is stamped on the node as a :class:`TileResult`, which the builders,
the memory planner, emit and reporting read.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """A buffer of a tiled op, by the part it plays: an operand of a GEMM /
    conv anchor, the op's output, or the scratch its kernel keeps."""

    INPUT = "input"
    CSR = "csr"
    INPUT_SCALE = "input_scale"
    WEIGHT = "weight"
    WEIGHT_SCALE = "weight_scale"
    BIAS = "bias"
    OUTPUT = "output"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class Fused:
    """The ``index``-th operand a GEMM / conv's fused tail adds
    (``fused_operand_nodes``)."""

    index: int


@dataclass(frozen=True)
class KernelInput:
    """The ``index``-th of a grid op's ``all_input_nodes``."""

    index: int


@dataclass(frozen=True)
class Placement:
    """Where a tiling's buffers go on chip.

    Attributes:
        groups: The bank groups, each a frozenset of the :class:`Role`,
            :class:`Fused` and :class:`KernelInput` members sharing its banks;
            ``None`` when the model imposes no grouping.
        scratch_slots: The regions the reduction scratch is allocated with.
        tier: How much extra bank sharing the tiling needed to fit, 0 for
            none; the grid search keeps the lowest.
    """

    groups: tuple | None = None
    scratch_slots: int = 1
    tier: int = 0


@dataclass(frozen=True)
class Cost:
    """What a tiling costs, in the backend's cycles.

    Attributes:
        latency: The whole op.
        dram_bytes: Its DRAM traffic, which breaks near-ties in latency;
            ``None`` when the model does not count it.
        compute: One step of its grid, which attention's products and
            reporting charge; ``None`` when nothing reads it.
    """

    latency: float
    dram_bytes: float | None = None
    compute: float | None = None


@dataclass(frozen=True)
class MatrixSpace:
    """The GEMM / conv tilings interstellar searches; its candidates are
    interstellar ``MappingPoint``s.

    Attributes:
        architecture: The interstellar ``Resource``: the memory levels, their
            capacities and the PE array.
        schedule: The interstellar ``Schedule``: the loops and orders the
            search is pinned to.
    """

    architecture: object
    schedule: object


@dataclass(frozen=True)
class GridSpace:
    """The vector, GEMV and pool tilings the grid enumeration searches.

    Attributes:
        lanes: The multiple a tile's innermost dim takes.
        ranked: Price every tiling that fits; else the first that fits, in
            the enumeration's order, wins unpriced.
    """

    lanes: int
    ranked: bool


@dataclass(frozen=True)
class AttentionSpace:
    """The flash-attention tilings the attention search enumerates.

    Attributes:
        unit: The multiple a query / key tile takes.
        head_multiple: The kernel holds the head dim padded to a multiple of
            this (:func:`pad_head`).
    """

    unit: int
    head_multiple: int


@dataclass(frozen=True)
class GridTiling:
    """A vector, GEMV or pool op cut into a grid of tiles, per dim it tiles.

    Attributes:
        tile_sizes: The elements one tile spans.
        grid: The tiles along the dim.
    """

    tile_sizes: tuple
    grid: tuple


@dataclass(frozen=True)
class AttentionTiling:
    """A flash attention cut into query and key tiles.

    Attributes:
        tile_sizes: ``(tq, tkv)``, the query and key rows of one tile.
        grid: The key / value batch dims, then the query and key block
            counts.
        products: Each product's :class:`TileResult`, ``"scores"`` and
            ``"context"``; ``None`` while the search checks the fit, before
            it maps them.
    """

    tile_sizes: tuple
    grid: tuple
    products: dict | None = None


@dataclass(frozen=True)
class TileResult:
    """A search's winner, stamped as ``meta["tile_result"]``.

    Attributes:
        tiling: The winning tiling: interstellar's ``MappingPoint`` for a
            GEMM / conv, else a :class:`GridTiling` or
            :class:`AttentionTiling`.
        placement: Its :class:`Placement`.
        cost: Its :class:`Cost`; ``None`` from an unranked grid search.
        access_list: Interstellar's per-level ``(input, output, weight)``
            access counts, which emit reports; ``None`` off the matrix
            search.
    """

    tiling: object
    placement: Placement
    cost: Cost | None
    access_list: tuple | None = None


class OpModel(ABC):
    """A backend's model of one op: which tilings exist, whether one fits
    and what it costs.

    A matrix op's model runs in a forked search worker, so it holds plain
    values only -- no FX node, op object or tensor -- and dispatches no torch
    op.

    Attributes:
        space: How the shared search generates the op's candidates, which
            also picks the search: a :class:`MatrixSpace`,
            :class:`GridSpace` or :class:`AttentionSpace`.
        key: What the answers depend on: matrix ops with equal keys share one
            search; ``None`` never shares.
    """

    space: object
    key: object

    @abstractmethod
    def fit(self, tiling):
        """The :class:`Placement` ``tiling`` takes on chip, or ``None`` when
        it does not fit.

        A matrix op's fit is asked at interstellar's fit check, on a
        ``MappingPoint`` with no loop orders whose L0 blocking counts that
        level's steps and copies together (its partitionings all 1), and its
        answer is reused for every partitioning and loop order of that
        blocking.  So it reads only each level's total extent (blocking x
        partitioning) and the blockings above L0, never the loop orders or
        how L0 splits into steps and copies.
        """

    @abstractmethod
    def cost(self, tiling, placement):
        """The :class:`Cost` of ``tiling`` laid out as ``placement``, the
        answer ``fit`` gave."""


class RuntimeModel(ABC):
    """A backend's hardware, as the shared tile search sees it.

    Built once per compile from the backend's config
    (``config.runtime_model``), it answers in that backend's cycles.  Times
    rank candidates against each other, so only their ratios matter to the
    search; a reported latency is cycles at the backend's clock.

    Attributes:
        config: The hardware description the model was built from.
        accumulate_fp32: A GEMM split along its reduction accumulates in
            float32.
    """

    def __init__(self, config, accumulate_fp32):
        self.config = config
        self.accumulate_fp32 = accumulate_fp32

    @abstractmethod
    def prepare(self, node):
        """The :class:`OpModel` of ``node`` -- a GEMM / conv, vector, GEMV or
        pool op, bare or fused -- or ``None`` to leave it untiled.

        Raises:
            NotImplementedError: The backend does not lower ``node``.
        """

    def prepare_attention(self, node, acc_dtype, bool_mask):
        """The :class:`OpModel` of a flash-attention ``node``.

        Args:
            node: The attention node.
            acc_dtype: The dtype the kernel keeps its softmax state in.
            bool_mask: The causal mask is int1 codes plus a scale, rather
                than an additive fill.

        Raises:
            NotImplementedError: The backend does not run fused attention.
        """
        raise NotImplementedError(f"{type(self).__name__}: fused attention")

    def kernel_scratch(self, node, out_tile):
        """The on-chip regions a backend's kernel keeps beside an op's tiles
        that no FX node names, which the builders declare and the fit
        charges.  None by default: a kernel that keeps its intermediates in
        its own temporaries needs no region in the IR.

        Args:
            node: The op being bufferized or tiled, bare or fused.
            out_tile: One tile of ``node``'s output, or one per output.

        Returns:
            ``[(name, shape, dtype), ...]``, one per region.
        """
        return []


def pad_head(head_dim, space):
    """``head_dim`` rounded up to the multiple an :class:`AttentionSpace`
    holds the head at."""
    return math.ceil(head_dim / space.head_multiple) * space.head_multiple
