"""One Trainium NeuronCore, described to the shared compiler.

``TrainiumConfig`` is an ``AcceleratorConfig`` -- the fields the shared passes,
tile search and memory planner read -- plus what Trainium's runtime model
(``TrainiumModel``) needs: the other engines' clocks and rates, the matmul's
moving-operand limit and DMA overheads.  ``TRN2`` and ``TRN3`` are AWS's
documented figures for one core, and the Trn2 profiles' where those are
missing or differ:

- NKI architecture guides (``nki/guides/architecture/trainium2_arch``):
  SBUF 128 partitions x 224 KiB (28 MiB) on NeuronCore-v3, TensorE 128 x 128
  at 2.4 GHz with FP8 at twice the BF16 rate and FP32 at a quarter of it (20
  vs 79 dense TFLOPS), a moving operand of at most 512 columns (one 2 KiB
  PSUM bank of float32, 8 banks a partition), VectorE at 0.96 GHz and
  ScalarE at 1.2 GHz.
- A Trn2 device profile: per partition, VectorE moves 2 16-bit elements a
  cycle and 1 when any operand is 32-bit (a PSUM accumulator is float32),
  ScalarE 1, and an instruction costs about 160 ns on VectorE and 270 ns on
  ScalarE beyond its elements.
- NKI DMA bandwidth guide: 16 DMA engines per core at 23 B/ns (TRN2) or
  33 B/ns (TRN3), about 1300 ns from semaphore to transfer start, and
  transfers under 2 KiB per partition bound by per-packet overhead.
- Trn2 profiles: back to back, a matmul or transpose streams its moving
  columns plus ``tensor_overhead_cycles``; GpSimd issues a ``dma_copy`` in
  about 650 ns; each of the 16 DMA rings moves about 27 B/ns, a row of a
  transfer costing ``dma_row_overhead_bytes`` more and a copy
  ``dma_copy_ns`` more, and idles between copies (``dma_copy_idle_ns``); a
  copy on 64 partitions or fewer moves at about half the rate (190-199
  GB/s against 438 for 342 and 171 KB copies on 32 and 64 partitions);
  a store issues from ScalarE's queue, in about 550 ns of ScalarE's time
  (a prefill step's attention, ``o_proj`` and ``gate_proj``).
- A step's compile log: neuronx-cc pins ``STEP_PINNED_SBUF`` bytes of each
  partition for its own DMA scratch, and starts a kernel's loads under the
  kernel before it, spilling them when the two do not fit
  (``HOISTED_SBUF``).

The shared compiler sees a 128 x 128 array clocked at TensorE's rate, SBUF as
an unbanked scratchpad less what the step keeps (``scratchpad_offset``), since
every buffer a kernel uses is in its tile result's placement and the plan,
and lanes of 1, since a tile's rows go on the partitions and its last dim is
free.
"""

from dataclasses import dataclass

from voyager_compiler.backend.backend import Backend
from voyager_compiler.codegen.node_info import is_linear, is_matmul
from voyager_compiler.codegen.transform.bufferize import (
    SHARED_BUILDERS,
    BuilderRule,
)
from voyager_compiler.hardware_config import AcceleratorConfig

from .attention import SDPA_KMAJOR, build_attention
from .gemm import build_gemm
from .model import TrainiumModel
from .utils import HOISTED_SBUF, PARTITIONS

# Bytes of each SBUF partition a step's neuronx-cc compile pins for itself:
# its dynamic DMA scratch (``--dynamic-dma-scratch-size-per-partition``,
# 16 KiB) and 8 more.  A kernel's buffers beside them would be spilled.
STEP_PINNED_SBUF = 16392

# A dense GEMM -- ``linear``, ``matmul`` and their MX forms -- and a K-major
# flash attention lower by Trainium's own builders.
GEMM_RULE = BuilderRule(
    "trainium_gemm",
    lambda node, anchor: is_linear(anchor) or is_matmul(anchor),
    build_gemm,
)
ATTENTION_RULE = BuilderRule(
    "trainium_attention",
    lambda node, anchor: anchor.target is SDPA_KMAJOR,
    build_attention,
)


@dataclass(frozen=True)
class TrainiumConfig(AcceleratorConfig):
    """One NeuronCore of a Trainium chip.

    ``pe_array_size`` is TensorE, its rows the SBUF partitions; ``frequency``
    is TensorE's clock; ``dram_bandwidth`` (GB/s, so bytes per ns) is the 16
    DMA engines together; ``dram_access_latency`` is a DMA's start latency.

    Attributes:
        target: The ``neuronx-cc --target`` this core is compiled for.
        psum_banks: PSUM banks per partition, every one the kernel's: NKI
            reserves one only to lower a ``dma_transpose``, which the
            translator never emits.
        psum_bank_bytes: Bytes per partition of one PSUM bank, the most one
            matmul writes.
        vector_ghz: VectorE clock.
        scalar_ghz: ScalarE clock.
        vector_elements_16: Elements per partition VectorE moves a cycle at
            16-bit width.
        vector_elements_32: The same when any operand is 32-bit.
        scalar_elements: Elements per partition ScalarE moves a cycle.
        tensor_overhead_cycles: Fixed TensorE cycles each matmul or
            transpose costs on top of its columns.
        vector_overhead_ns: Fixed time each VectorE instruction costs on
            top of its elements.
        scalar_overhead_ns: The same on ScalarE.
        fp8_matmul_speedup: TensorE's FP8 rate over its 16-bit rate.
        fp32_matmul_rate: TensorE's FP32 matmul rate over its 16-bit rate.
        fp32_transpose_rate: TensorE's FP32 transpose rate over its 16-bit
            rate.
        dma_issue_ns: Time to issue one ``dma_copy``.
        dma_row_overhead_bytes: Per-row packet overhead, so a row of ``b``
            bytes moves at the rate of ``b + dma_row_overhead_bytes``.
        dma_copy_ns: Ring time each ``dma_copy`` adds beyond its rows.
        dma_copy_idle_ns: Further ring idle each ``dma_copy`` adds when its
            rows take ``dma_issue_ns`` or less, falling off in proportion as
            they take longer.
        dma_half_rate_partitions: A ``dma_copy`` on this many partitions or
            fewer moves at half the DMA bandwidth.
        dma_store_issue_ns: ScalarE's time issuing one store's
            ``dma_copy``.
        kernel_overhead_ns: Fixed time per kernel, beyond its sweep.
    """

    target: str = "trn2"
    psum_banks: int = 8
    psum_bank_bytes: int = 2048
    vector_ghz: float = 0.96
    scalar_ghz: float = 1.2
    vector_elements_16: int = 2
    vector_elements_32: int = 1
    scalar_elements: int = 1
    tensor_overhead_cycles: int = 6
    vector_overhead_ns: float = 160.0
    scalar_overhead_ns: float = 270.0
    fp8_matmul_speedup: float = 2.0
    fp32_matmul_rate: float = 0.25
    fp32_transpose_rate: float = 0.5
    dma_issue_ns: float = 650.0
    dma_row_overhead_bytes: int = 200
    dma_copy_ns: float = 31.0
    dma_copy_idle_ns: float = 80.0
    dma_half_rate_partitions: int = 64
    dma_store_issue_ns: float = 550.0
    kernel_overhead_ns: float = 0.0

    @property
    def partitions(self) -> int:
        """SBUF partitions: TensorE's rows, the matmul contraction."""
        return self.pe_array_size[0]

    @property
    def moving_free_max(self) -> int:
        """Columns of a matmul's moving operand: a PSUM bank of float32."""
        return self.psum_bank_bytes // 4

    @property
    def psum_bytes(self) -> int:
        """PSUM bytes per partition."""
        return self.psum_banks * self.psum_bank_bytes

    def runtime_model(self, accumulate_fp32):
        """This core's ``TrainiumModel``.

        Args:
            accumulate_fp32: A GEMM split along its reduction accumulates in
                float32.
        """
        return TrainiumModel(self, accumulate_fp32)

    def backend(self):
        """Trainium's hooks into ``compile()``: its GEMM and attention
        builders (``gemm.py``, ``attention.py``) ahead of the shared
        builders."""
        return Backend((GEMM_RULE, ATTENTION_RULE, *SHARED_BUILDERS))


def _core(target, sbuf_bytes, dram_bandwidth):
    """A core of ``target``: ``sbuf_bytes`` of SBUF, less ``STEP_PINNED_SBUF``
    and ``HOISTED_SBUF`` of each partition, and ``dram_bandwidth`` GB/s of
    DMA."""
    return TrainiumConfig(
        target=target,
        pe_array_size=(PARTITIONS, 128),
        vector_unit_width=1,
        frequency=2.4,
        scratchpad_size=sbuf_bytes,
        scratchpad_offset=PARTITIONS * (STEP_PINNED_SBUF + HOISTED_SBUF),
        double_buffered_l2=True,
        async_pipeline=False,
        dram_size=12.0,
        dram_bandwidth=dram_bandwidth,
        dram_access_latency=1300.0,
    )


# The ``transform`` layouts: every GEMM weight contraction-major (``"ck"``,
# [in, out]; K is the output dim), so a weight tile lands with the
# contraction on the partitions TensorE contracts over, the activations
# between matrix-matrix GEMMs contraction-major (``"cx"``) for the same
# reason, and feature maps NCHW, so a pool tile has H on the partitions and W
# on the free axis.
LAYOUT = {
    "tensor_layout": "nchw",
    "gemm_layout": "cx,ck->xk",
    "gemv_layout": "xc,ck->xk",
}

TRN2 = _core("trn2", 28 << 20, 16 * 27.0)
TRN3 = _core("trn3", 30 << 20, 16 * 33.0)
