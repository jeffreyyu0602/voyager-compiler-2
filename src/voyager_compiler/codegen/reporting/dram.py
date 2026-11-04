"""DRAMSim3 prices for the estimator's DMAs (``--dram_sim``).

A DMA's request addresses come from the memory plan: its DRAM buffer's
``meta['memory']`` start plus the block it moves, laid out row-major in the
buffer's shape. Each contiguous run of the block becomes the requests it
touches, in address order, and ``DramSystem.simulate`` prices that stream.
Prices are cached by pattern: two DMAs whose streams differ only by a whole
number of address-map periods (``period_bytes``, which changes only their
rows) cost the same.
"""

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np
from torch.fx import Node

from voyager_compiler.codegen.node_info import dtype_byte_size
from voyager_compiler.codegen.reporting.cost import _dtype, _val
from voyager_compiler.dram import DramSystem


class DmaPrice(NamedTuple):
    """One DMA's cost: accelerator ``cycles``, activate / read / write
    ``energy_pj``, the ranks' ``standby_pj`` while it runs, and its
    ``requests`` and ``row_hits``."""

    cycles: int
    energy_pj: float
    standby_pj: float
    requests: int
    row_hits: int


def _dram_start(root: Node) -> int:
    """The byte address of DRAM buffer ``root``; 0 for a one-element tensor,
    which the program passes by value and the memory plan gives none.

    Raises:
        ValueError: If the memory plan gave any other tensor no address.
    """
    segment = root.meta.get("memory") if isinstance(root, Node) else None
    if segment is None and _val(root) is not None and _val(root).numel() == 1:
        return 0
    if segment is None:
        raise ValueError(
            f"{root} has no DRAM address: --dram_sim needs plan_memory first"
        )
    return int(segment.start)


def _lines(starts_bits: np.ndarray, run_bits: int, line: int) -> np.ndarray:
    """The ``line``-byte request addresses runs of ``run_bits`` bits starting
    at ``starts_bits`` touch, in order, a request shared by adjacent runs
    once."""
    line_bits = 8 * line
    first = starts_bits // line_bits
    last = (starts_bits + run_bits - 1) // line_bits
    counts = last - first + 1
    offsets = np.arange(counts.sum()) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    lines = np.repeat(first, counts) + offsets
    keep = np.ones(len(lines), dtype=bool)
    keep[1:] = lines[1:] != lines[:-1]
    return lines[keep].astype(np.uint64) * np.uint64(line)


class DramPricer:
    """Prices the estimator's DMAs on one DRAMSim3 configuration.

    Args:
        config_file: The DRAMSim3 ``.ini``.
        frequency: The accelerator clock in GHz, to convert to its cycles.
    """

    def __init__(self, config_file: str, frequency: float):
        self.system = DramSystem(config_file)
        self.frequency = frequency
        self._cache = {}

    def _key_base(self, start_bits: int) -> int:
        period_bits = 8 * self.system.period_bytes
        return start_bits % period_bits if period_bits else start_bits

    def _price(self, key, start_bits: int, make_lines, is_write) -> DmaPrice:
        """The cached price of ``key``; on a miss, simulate ``make_lines``'s
        addresses shifted to ``start_bits``' place in its period."""
        price = self._cache.get(key)
        if price is None:
            shift = (start_bits - self._key_base(start_bits)) // 8
            addresses = make_lines() - np.uint64(shift)
            cost = self.system.simulate(addresses, is_write)
            price = self._cache[key] = DmaPrice(
                cycles=math.ceil(cost.ns * self.frequency),
                energy_pj=cost.energy_pj,
                standby_pj=cost.standby_pj,
                requests=cost.requests,
                row_hits=cost.row_hits,
            )
        return price

    def block(
        self,
        root: Node,
        view_offset: int,
        strides: Sequence[int],
        offsets: Sequence[int],
        sizes: Sequence[int],
        is_write: bool,
        fill: Optional[float] = None,
    ) -> DmaPrice:
        """Price moving the ``sizes`` block at ``offsets`` of a strided view
        of DRAM buffer ``root``.

        Args:
            root: The DRAM buffer (its address and element dtype).
            view_offset: The view's first element, in ``root``'s elements.
            strides: The view's stride along each dim, in elements.
            offsets: The block's first element along each dim.
            sizes: The block's extent along each dim.
            is_write: A store (else a load).
            fill: The fraction of the block's requests that move (a CSR
                payload), the first ones in address order; ``None`` for all.
        """
        bits = round(dtype_byte_size(_dtype(root)) * 8)
        first = view_offset + sum(o * s for o, s in zip(offsets, strides))
        start_bits = 8 * _dram_start(root) + first * bits
        # A size-1 dim moves nothing along it; drop it.
        kept = [d for d, n in enumerate(sizes) if n != 1]
        sizes = [sizes[d] for d in kept]
        strides = [strides[d] for d in kept]
        # The trailing dims whose block is contiguous form one run.
        k = len(sizes)
        if sizes and strides[-1] == 1:
            k -= 1
            while k > 0 and strides[k - 1] == strides[k] * sizes[k]:
                k -= 1
        run_elems = math.prod(sizes[k:])
        key = (
            "block",
            is_write,
            self._key_base(start_bits),
            tuple(strides[:k]),
            tuple(sizes),
            bits,
            fill,
        )

        def make_lines():
            grids = np.meshgrid(
                *[np.arange(n) * s for n, s in zip(sizes[:k], strides[:k])],
                indexing="ij",
            )
            runs = sum(g.ravel() for g in grids) if grids else np.zeros(1)
            starts = start_bits + np.asarray(runs, dtype=np.int64) * bits
            lines = _lines(starts, run_elems * bits, self.system.request_bytes)
            if fill is not None:
                lines = lines[: math.ceil(len(lines) * fill)]
            return lines

        return self._price(key, start_bits, make_lines, is_write)

    def stream(self, root: Node, n_bytes: int, is_write: bool) -> DmaPrice:
        """Price moving ``n_bytes`` contiguous bytes from the start of DRAM
        buffer ``root`` -- a whole-tensor materialization's side."""
        start_bits = 8 * _dram_start(root)
        key = ("stream", is_write, self._key_base(start_bits), n_bytes)

        def make_lines():
            return _lines(
                np.array([start_bits]), 8 * n_bytes, self.system.request_bytes
            )

        return self._price(key, start_bits, make_lines, is_write)

    def run_energy_pj(
        self, cycles: int, busy_cycles: int, standby_pj: float
    ) -> dict:
        """Refresh and standby energy over a run of ``cycles`` accelerator
        cycles, ``busy_cycles`` of them with a DMA running, whose DMAs drew
        ``standby_pj`` of standby."""
        ns = cycles / self.frequency
        idle_ns = max(0, cycles - busy_cycles) / self.frequency
        return {
            "refresh": self.system.refresh_energy_pj(ns),
            "background": standby_pj + self.system.idle_energy_pj(idle_ns),
        }
