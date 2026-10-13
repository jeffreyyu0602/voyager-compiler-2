"""What every tile search shares: the ``Tiler`` threaded down to each
builder, holding the backend's runtime model and the run's search state
(``build_tiler``), the ``TileConstraint`` a search may run under, and how a
search picks its winner among the candidates it priced (``pick``).
"""

from dataclasses import dataclass, field
from typing import Tuple

from voyager_compiler.codegen.transform.tiling.runtime_model import RuntimeModel

# How much longer than the best modeled runtime a tiling may take and still be
# chosen; the least-traffic one among those wins.  0.0 = only the fastest.
# The matrix search hands it to interstellar, which picks the least-energy
# mapping among those.
DEFAULT_RUNTIME_TOLERANCE = 0.02


@dataclass(frozen=True)
class TileConstraint:
    """Loop extents a search must land on at the scratchpad level, keyed by
    interstellar loop (``le.IC``, ``le.OC``, ...).  ``exact`` pins a loop's
    tile to one extent, ``multiple`` requires it to be a whole number of the
    value; both are ``((loop, value), ...)`` so a constraint can key the
    tiling cache.  A GEMM feeding an MHA relayout takes a ``le.OC``
    multiple of the head; a CSR consumer an exact ``le.IC`` (the slice
    width) and a CSR producer a ``le.OC`` multiple of it (its column tile
    is cut into slices), which is how ``plan_csr_slices`` searches the two
    under one width.
    """

    exact: Tuple[Tuple[int, int], ...] = ()
    multiple: Tuple[Tuple[int, int], ...] = ()

    def allows(self, extent) -> bool:
        """Whether tile extents ``extent(loop)`` satisfy every entry."""
        return all(extent(loop) == v for loop, v in self.exact) and all(
            extent(loop) % v == 0 for loop, v in self.multiple
        )

    def merged(self, other):
        """This constraint and ``other`` (``None`` = nothing) together."""
        if other is None:
            return self
        return TileConstraint(
            tuple(sorted(self.exact + other.exact)),
            tuple(sorted(self.multiple + other.multiple)),
        )


@dataclass
class Tiler:
    """The runtime model and the run's search state, built once and shared
    by every builder so each can tile its op on demand.

    Attributes:
        model: The backend's ``RuntimeModel`` (``config.runtime_model``).
        runtime_tolerance: How much slower than the fastest a mapping may be
            and still win on traffic.
        cache: Each GEMM / conv search's ``TileResult``, by its
            ``_Search.key``.
        constraints: The ``TileConstraint`` a joint optimization settled on
            for an anchor, which its search then runs under.
    """

    model: RuntimeModel
    runtime_tolerance: float
    cache: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)

    @property
    def config(self):
        """The hardware description the model was built from."""
        return self.model.config


def build_tiler(
    config,
    runtime_tolerance=DEFAULT_RUNTIME_TOLERANCE,
    accumulate_fp32=False,
):
    """A ``Tiler`` for ``config``'s hardware: the runtime model its
    backend describes.

    Args:
        config: The hardware description; ``config.runtime_model`` builds
            its backend's model.
        runtime_tolerance: How much slower than the fastest a mapping may be
            and still win on traffic.
        accumulate_fp32: Size and price a GEMM split along K with a float32
            accumulator.
    """
    return Tiler(
        model=config.runtime_model(accumulate_fp32),
        runtime_tolerance=runtime_tolerance,
    )


def pick(results, tolerance):
    """The winner among priced ``results`` (``TileResult``): the one moving
    the fewest DRAM bytes among those within ``(1 + tolerance)`` of the best
    latency, then the faster, then the first.

    Latency alone is not enough: a compute-bound op is equally fast however
    its operands are diced, so the residue the model leaves behind would buy
    any amount of traffic for a rounding error.  This is how the interstellar
    tiler picks a mapping too (``mapping_point_generator``, with energy in
    place of bytes).
    """
    budget = min(result.cost.latency for result in results) * (1.0 + tolerance)
    return min(
        (result for result in results if result.cost.latency <= budget),
        key=lambda result: (result.cost.dram_bytes, result.cost.latency),
    )
