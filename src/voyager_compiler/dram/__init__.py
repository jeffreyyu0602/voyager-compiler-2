"""DRAMSim3-backed DRAM timing and energy for one DMA at a time.

``DramSystem`` wraps one DRAMSim3 ``.ini`` (see ``configs/``) through the C
shim ``build.sh`` compiles. ``simulate`` runs one DMA's request addresses on
a fresh memory system, all banks precharged, with refresh held off: the DMA's
time is stretched by refresh's average share of the channel instead
(``refresh_fraction``), so a DMA's price never depends on when it runs.
A DMA's standby energy is measured in its simulation, its refresh stretch
in precharge standby; refresh energy and the standby between DMAs belong
to the whole run: ``refresh_energy_pj`` and ``idle_energy_pj``.  This
package knows nothing of FX graphs.
"""

import ctypes
import os
import re
import tempfile
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBRARY = os.path.join(_HERE, "build", "libvoyager_dramsim3.so")
CONFIG_DIR = os.path.join(_HERE, "configs")

# vds_config's output order (shim.cc).
_CONFIG_FIELDS = (
    "period_bytes",
    "tck",
    "request_bytes",
    "channels",
    "ranks",
    "trefi",
    "trfc",
    "burst_cycle",
    "act_energy",
    "read_energy",
    "write_energy",
    "ref_energy",
    "refb_energy",
    "act_stb_energy",
    "pre_stb_energy",
    "sref_energy",
)
# vds_counters' output order (shim.cc).
_COUNTERS = (
    "cycles",
    "act",
    "pre",
    "read",
    "write",
    "read_row_hits",
    "write_row_hits",
    "ref",
    "refb",
    "rank_active_cycles",
    "all_bank_idle_cycles",
    "sref_cycles",
)
# A refresh interval no DMA simulation reaches (DRAMSim3 reads an int).
_NO_REFRESH_TREFI = 2**31 - 1

_lib = None


def _load():
    """The shim library, loaded once.

    Raises:
        FileNotFoundError: If ``build.sh`` has not been run.
    """
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIBRARY):
        raise FileNotFoundError(
            f"{_LIBRARY} is missing: run {os.path.join(_HERE, 'build.sh')}"
        )
    lib = ctypes.CDLL(_LIBRARY)
    lib.vds_open.restype = ctypes.c_void_p
    lib.vds_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.vds_close.argtypes = [ctypes.c_void_p]
    lib.vds_run.restype = ctypes.c_uint64
    lib.vds_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.vds_tick.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.vds_counters.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vds_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    _lib = lib
    return lib


@dataclass(frozen=True)
class DmaCost:
    """What one DMA costs the DRAM.

    Attributes:
        ns: Time from its first request to its last data, refresh's average
            share included.
        energy_pj: Activate, read and write energy (no refresh or standby).
        standby_pj: Standby energy of every rank while it runs.
        requests: Requests (``request_bytes`` each) it issued.
        row_hits: Requests that found their row open.
    """

    ns: float
    energy_pj: float
    standby_pj: float
    requests: int
    row_hits: int


class _Handle:
    """One open DRAMSim3 memory system, closed on exit."""

    def __init__(self, config_file: str, output_dir: str):
        self.lib = _load()
        self.ptr = self.lib.vds_open(config_file.encode(), output_dir.encode())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lib.vds_close(self.ptr)

    def run(self, addresses: np.ndarray, is_write: bool) -> int:
        addresses = np.ascontiguousarray(addresses, dtype=np.uint64)
        pointer = addresses.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.vds_run(
            self.ptr, pointer, len(addresses), int(is_write)
        )

    def tick(self, cycles: int) -> None:
        self.lib.vds_tick(self.ptr, cycles)

    def counters(self) -> dict:
        out = (ctypes.c_uint64 * len(_COUNTERS))()
        self.lib.vds_counters(self.ptr, out)
        return dict(zip(_COUNTERS, out))

    def config(self) -> dict:
        out = (ctypes.c_double * len(_CONFIG_FIELDS))()
        self.lib.vds_config(self.ptr, out)
        return dict(zip(_CONFIG_FIELDS, out))


class DramSystem:
    """A DRAMSim3 configuration, priced one DMA at a time.

    Args:
        config_file: A DRAMSim3 ``.ini``; it must set ``tREFI``.
    """

    def __init__(self, config_file: str):
        self.config_file = os.path.abspath(config_file)
        with open(self.config_file) as f:
            text = f.read()
        if not re.search(r"^\s*tREFI\s*=", text, re.MULTILINE):
            raise ValueError(f"{config_file} does not set tREFI")
        self._tmp = tempfile.TemporaryDirectory(prefix="voyager_dramsim3_")
        self.output_dir = self._tmp.name
        self._no_refresh = os.path.join(self.output_dir, "no_refresh.ini")
        with open(self._no_refresh, "w") as f:
            f.write(
                re.sub(
                    r"^(\s*tREFI\s*=).*$",
                    rf"\g<1> {_NO_REFRESH_TREFI}",
                    text,
                    flags=re.MULTILINE,
                )
            )
        with _Handle(self.config_file, self.output_dir) as handle:
            c = handle.config()
        self.tck_ns = c["tck"]
        self.request_bytes = int(c["request_bytes"])
        self.channels = int(c["channels"])
        self.ranks = int(c["ranks"])
        self.period_bytes = int(c["period_bytes"])
        self._c = c

    @property
    def peak_gbps(self) -> float:
        """Peak data bandwidth: one request per burst on every channel."""
        burst_ns = self._c["burst_cycle"] * self.tck_ns
        return self.channels * self.request_bytes / burst_ns

    @property
    def refresh_fraction(self) -> float:
        """Share of each rank's time spent refreshing (``tRFC / tREFI``)."""
        return self._c["trfc"] / self._c["trefi"]

    def _pj(self, mw_cycles: float) -> float:
        # DRAMSim3's increments are V * mA * cycles; times tCK (ns) is pJ.
        return mw_cycles * self.tck_ns

    def simulate(self, addresses, is_write: bool) -> DmaCost:
        """Run one DMA on a fresh memory system.

        Args:
            addresses: Its request-aligned byte addresses, in issue order.
            is_write: A store (else a load).

        Returns:
            The DMA's ``DmaCost``.
        """
        with _Handle(self._no_refresh, self.output_dir) as handle:
            cycles = handle.run(np.asarray(addresses), is_write)
            n = handle.counters()
        if cycles == 2**64 - 1:
            raise RuntimeError(
                f"DRAMSim3 stopped making progress on a {len(addresses)}-"
                f"request {'store' if is_write else 'load'}"
            )
        c = self._c
        energy = (
            n["act"] * c["act_energy"]
            + n["read"] * c["read_energy"]
            + n["write"] * c["write_energy"]
        )
        ns = cycles * self.tck_ns / (1 - self.refresh_fraction)
        standby = (
            n["rank_active_cycles"] * c["act_stb_energy"]
            + n["all_bank_idle_cycles"] * c["pre_stb_energy"]
            + n["sref_cycles"] * c["sref_energy"]
        )
        stretch_ns = ns - cycles * self.tck_ns
        return DmaCost(
            ns=ns,
            energy_pj=self._pj(energy),
            standby_pj=self._pj(standby) + self.idle_energy_pj(stretch_ns),
            requests=len(addresses),
            row_hits=n["read_row_hits"] + n["write_row_hits"],
        )

    def refresh_energy_pj(self, ns: float) -> float:
        """Refresh energy of every rank over ``ns`` nanoseconds."""
        refreshes = ns / (self._c["trefi"] * self.tck_ns)
        ranks = self.channels * self.ranks
        return self._pj(refreshes * ranks * self._c["ref_energy"])

    def idle_energy_pj(self, ns: float) -> float:
        """Standby energy of every rank over ``ns`` nanoseconds with no DMA
        running: precharge standby, since the next refresh closes the rows a
        DMA left open, within one tREFI."""
        ranks = self.channels * self.ranks
        return self._pj(ranks * ns / self.tck_ns * self._c["pre_stb_energy"])
