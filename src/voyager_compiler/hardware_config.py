"""The accelerator hardware description.

``AcceleratorConfig`` bundles every hardware knob the compiler needs — the PE
array, the on-chip L1 systolic buffers, the L2 scratchpad, DRAM, and the clock —
into one frozen object that ``transform()`` / ``compile()`` and their callees
pass around, instead of threading a dozen loose, drift-prone arguments through
every layer.

Physical units: ``dram_bandwidth`` is GB/s, ``dram_access_latency`` ns,
``frequency`` GHz, so bytes/cycle is ``dram_bandwidth / frequency`` and
per-transfer latency in cycles is ``dram_access_latency * frequency``.
``dram_energy_per_bit`` is pJ/bit, which ``dram_energy_per_byte`` turns into
J/B for the reporting energy columns.  The reporting model reads this object
directly as its cost knobs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

DEFAULT_PE_ARRAY_SIZE = (32, 32)
DEFAULT_FREQUENCY_GHZ = 1.0
DEFAULT_INPUT_BUFFER_SIZE = 1024
DEFAULT_WEIGHT_BUFFER_SIZE = 1024
DEFAULT_ACCUM_BUFFER_SIZE = 1024
DEFAULT_SCRATCHPAD_OFFSET = 0
DEFAULT_DOUBLE_BUFFERED_L2 = True
DEFAULT_DRAM_SIZE_GB = 16.0
DEFAULT_DRAM_BANDWIDTH_GBS = 64.0
DEFAULT_DRAM_ACCESS_LATENCY_NS = 100.0
DEFAULT_DRAM_ENERGY_PJ_PER_BIT = 6.25


@dataclass(frozen=True)
class AcceleratorConfig:
    """Compiler-visible hardware parameters for Voyager accelerators.

    Voyager couples two programmable compute engines: a weight-stationary
    2-D systolic array for convolution and GEMM, and a multi-stage vector
    unit for elementwise operations, reductions, nonlinear activations,
    normalization, and quantization. During a matrix tile, weights remain
    in the processing elements, activations flow horizontally, and partial
    sums propagate vertically. The matrix output can stream directly into
    the vector unit for operator fusion or pass through a double-buffered
    accumulation buffer to decouple the two engines.

    The systolic array uses dedicated input, weight, and accumulation
    buffers backed by a banked L2 scratchpad. This configuration describes
    the accelerator's compute parallelism, clock frequency, and on-chip
    memory hierarchy. The optional DRAM capacity, bandwidth, and latency
    parameters are later system-modeling extensions, rather than parameters
    of the Voyager accelerator template described in the paper.
    """

    # Compute
    pe_array_size: Tuple[int, int] = DEFAULT_PE_ARRAY_SIZE
    vector_lanes: Optional[int] = None  # None -> pe_array_size[1]
    matrix_vector_lanes: Optional[int] = None  # None -> pe_array_size[1]
    accumulator_lanes: Optional[int] = None
    frequency: float = DEFAULT_FREQUENCY_GHZ  # accelerator clock
    # L1 systolic buffers (# elements)
    input_buffer_size: Optional[int] = DEFAULT_INPUT_BUFFER_SIZE
    weight_buffer_size: Optional[int] = DEFAULT_WEIGHT_BUFFER_SIZE
    accum_buffer_size: Optional[int] = DEFAULT_ACCUM_BUFFER_SIZE
    double_buffered_accum_buffer: bool = False
    # L2 scratchpad
    scratchpad_size: Optional[int] = None
    scratchpad_offset: int = DEFAULT_SCRATCHPAD_OFFSET  # bytes at the base
    scratchpad_banks: Optional[int] = None
    scratchpad_word_bytes: Optional[int] = None
    double_buffered_l2: bool = DEFAULT_DOUBLE_BUFFERED_L2
    # L3 DRAM
    dram_size: Optional[float] = DEFAULT_DRAM_SIZE_GB
    dram_bandwidth: Optional[float] = DEFAULT_DRAM_BANDWIDTH_GBS
    dram_access_latency: Optional[float] = DEFAULT_DRAM_ACCESS_LATENCY_NS
    dram_energy_per_bit: float = DEFAULT_DRAM_ENERGY_PJ_PER_BIT  # pJ/bit

    def __post_init__(self):
        """Fill the unit widths left unset, then reject a reservation the
        rest of the compiler could not honour.

        An unset vector or matrix-vector width is the PE array's columns
        (left unset without a PE array), an unset accumulator width the
        vector unit's.  The reservation check is inert at the default offset
        of 0, so a config that names no scratchpad at all still constructs.
        """
        # The config is frozen: the defaults are set once, here.
        if self.pe_array_size is not None:
            cols = self.pe_array_size[1]
            if self.vector_lanes is None:
                object.__setattr__(self, "vector_lanes", cols)
            if self.matrix_vector_lanes is None:
                object.__setattr__(self, "matrix_vector_lanes", cols)
        if self.accumulator_lanes is None:
            object.__setattr__(self, "accumulator_lanes", self.vector_lanes)
        if self.scratchpad_offset < 0:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is negative"
            )
        if not self.scratchpad_offset:
            return
        if self.scratchpad_size is None:
            raise ValueError(
                "scratchpad_offset needs a scratchpad_size to reserve from"
            )
        if self.scratchpad_offset >= self.scratchpad_size:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} leaves nothing of "
                f"scratchpad_size {self.scratchpad_size}"
            )
        # A reservation that split a bank would put the planner's bank-aligned
        # groups and the tile search's bank budget on different geometries.
        bank = self.scratchpad_bank_bytes
        if bank is not None and self.scratchpad_offset % bank:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is not a multiple "
                f"of the {bank} B bank size"
            )

    @property
    def dram_energy_per_byte(self) -> float:
        """DRAM access energy in joules per byte, from the pJ/bit figure."""
        return self.dram_energy_per_bit * 8 * 1e-12

    @property
    def dram_bytes_per_cycle(self) -> float:
        return self.dram_bandwidth / self.frequency

    @property
    def dram_latency_cycles(self) -> float:
        return self.dram_access_latency * self.frequency

    @property
    def scratchpad_bytes_per_cycle(self) -> float:
        """Bytes one scratchpad bank port moves per cycle: one
        ``scratchpad_word_bytes`` word, or the DRAM rate for a config without
        a word size."""
        if self.scratchpad_word_bytes:
            return self.scratchpad_word_bytes
        return self.dram_bytes_per_cycle

    @property
    def num_slots(self) -> int:
        """Banks one buffer occupies: two when it is ping-ponged, else one."""
        return 2 if self.double_buffered_l2 else 1

    @property
    def scratchpad_bank_bytes(self) -> Optional[int]:
        if self.scratchpad_banks is None:
            return None
        return self.scratchpad_size // self.scratchpad_banks

    @property
    def usable_scratchpad_size(self) -> Optional[int]:
        """What the plan may spend: the SRAM above ``scratchpad_offset``."""
        if self.scratchpad_size is None:
            return None
        return self.scratchpad_size - self.scratchpad_offset

    @property
    def usable_scratchpad_banks(self) -> Optional[int]:
        """Banks above the reservation.

        Counted off ``scratchpad_banks`` rather than divided out of the
        usable bytes, so it is exactly ``scratchpad_banks`` at an offset of 0
        even when the banks do not divide the SRAM evenly.
        """
        if self.scratchpad_banks is None or self.scratchpad_bank_bytes is None:
            return None
        return (
            self.scratchpad_banks
            - self.scratchpad_offset // self.scratchpad_bank_bytes
        )

    def runtime_model(self, accumulate_fp32):
        """The runtime model the tile search fits and prices this
        accelerator's tilings with: a ``VoyagerModel``.

        Args:
            accumulate_fp32: A GEMM split along its reduction accumulates in
                float32.
        """
        # Imported here to keep the config a leaf that loads nothing of the
        # compiler: cli_args reads its defaults before the compiler loads.
        from voyager_compiler.backend.voyager import VoyagerModel

        return VoyagerModel(self, accumulate_fp32)

    def backend(self):
        """This target's hooks into ``compile()``: a ``Backend`` with the
        shared builders."""
        # Imported here to keep the config a leaf, as in ``runtime_model``.
        from voyager_compiler.backend.backend import Backend
        from voyager_compiler.codegen.transform.bufferize import (
            SHARED_BUILDERS,
        )

        return Backend(SHARED_BUILDERS)

    @classmethod
    def from_args(cls, args) -> "AcceleratorConfig":
        """Build the config from parsed CLI args (``add_compile_args``)."""
        return cls(
            pe_array_size=args.pe_array_size,
            vector_lanes=args.vector_unit_width,
            matrix_vector_lanes=args.matrix_vector_unit_width,
            accumulator_lanes=args.accumulator_width,
            frequency=args.frequency,
            input_buffer_size=args.input_buffer_size,
            weight_buffer_size=args.weight_buffer_size,
            accum_buffer_size=args.accum_buffer_size,
            double_buffered_accum_buffer=args.double_buffered_accum_buffer,
            scratchpad_size=args.scratchpad_size,
            scratchpad_offset=args.scratchpad_offset,
            scratchpad_banks=args.scratchpad_banks,
            scratchpad_word_bytes=args.scratchpad_word_bytes,
            double_buffered_l2=args.double_buffered_l2,
            dram_size=args.dram_size,
            dram_bandwidth=args.dram_bandwidth,
            dram_access_latency=args.dram_access_latency,
            dram_energy_per_bit=args.dram_energy_per_bit,
        )
