"""One Trainium NeuronCore, described to the shared compiler.

``TrainiumConfig`` is an ``AcceleratorConfig`` -- the fields the shared passes,
tile search and memory planner read -- plus what Trainium's runtime model
(``TrainiumModel``) needs: the other engines' clocks and rates, the matmul's
moving-operand limit and DMA overheads.  ``TRN2`` and ``TRN3`` are AWS's
documented figures for one core:

- NKI architecture guides (``nki/guides/architecture/trainium2_arch``):
  SBUF 128 partitions x 224 KiB (28 MiB) on NeuronCore-v3, TensorE 128 x 128
  at 2.4 GHz with FP8 at twice the BF16 rate and FP32 at a quarter of it (20
  vs 79 dense TFLOPS), a moving operand of at most 512 columns (one 2 KiB
  PSUM bank of float32), VectorE at 0.96 GHz moving 512 16-bit (256 wider)
  elements a cycle, ScalarE at 1.2 GHz moving 128.
- NKI DMA bandwidth guide: 16 DMA engines per core at 23 B/ns (TRN2) or
  33 B/ns (TRN3), about 1300 ns from semaphore to transfer start, and
  transfers under 2 KiB per partition bound by per-packet overhead.
- About 600 ns for each descriptor-generated DMA instruction (TRN2 guide).

The shared compiler sees a 128 x 128 array clocked at TensorE's rate, SBUF as
an unbanked scratchpad whose first 4 MiB (``scratchpad_offset``) are kept for
the translator's own temporaries -- transposed activation chunks and evicted
PSUM slices -- and lanes of 1, since a tile's rows go on the partitions and
its last dim is free.
"""

from dataclasses import dataclass

from voyager_compiler.hardware_config import AcceleratorConfig

from .model import TrainiumModel


@dataclass(frozen=True)
class TrainiumConfig(AcceleratorConfig):
    """One NeuronCore of a Trainium chip.

    ``pe_array_size`` is TensorE, its rows the SBUF partitions; ``frequency``
    is TensorE's clock; ``dram_bandwidth`` (GB/s, so bytes per ns) is the 16
    DMA engines together; ``dram_access_latency`` is a DMA's start latency.

    Attributes:
        target: The ``neuronx-cc --target`` this core is compiled for.
        moving_free_max: Columns of a matmul's moving operand.
        vector_ghz: VectorE clock.
        scalar_ghz: ScalarE clock.
        vector_elements_16: Elements per partition VectorE moves a cycle at
            16-bit width.
        vector_elements_32: The same at 32-bit width.
        scalar_elements: Elements per partition ScalarE moves a cycle.
        instruction_overhead_cycles: Fixed cycles each engine instruction
            costs on top of its elements.
        fp8_matmul_speedup: TensorE's FP8 rate over its 16-bit rate.
        fp32_matmul_rate: TensorE's FP32 matmul rate over its 16-bit rate.
        fp32_transpose_rate: TensorE's FP32 transpose rate over its 16-bit
            rate.
        dma_issue_ns: Time to issue one ``dma_copy``.
        dma_row_overhead_bytes: Per-partition packet overhead, so a row of
            ``b`` bytes moves at the rate of ``b + dma_row_overhead_bytes``.
        kernel_overhead_ns: Fixed time per kernel, beyond its sweep.
    """

    target: str = "trn2"
    moving_free_max: int = 512
    vector_ghz: float = 0.96
    scalar_ghz: float = 1.2
    vector_elements_16: int = 4
    vector_elements_32: int = 2
    scalar_elements: int = 1
    instruction_overhead_cycles: int = 64
    fp8_matmul_speedup: float = 2.0
    fp32_matmul_rate: float = 0.25
    fp32_transpose_rate: float = 0.5
    dma_issue_ns: float = 600.0
    dma_row_overhead_bytes: int = 512
    kernel_overhead_ns: float = 0.0

    @property
    def partitions(self) -> int:
        """SBUF partitions: TensorE's rows, the matmul contraction."""
        return self.pe_array_size[0]

    def runtime_model(self, accumulate_fp32):
        """This core's ``TrainiumModel``.

        Args:
            accumulate_fp32: A GEMM split along its reduction accumulates in
                float32.
        """
        return TrainiumModel(self, accumulate_fp32)


def _core(target, sbuf_bytes, dram_bandwidth):
    """A core of ``target``: ``sbuf_bytes`` of SBUF and ``dram_bandwidth``
    GB/s of DMA."""
    return TrainiumConfig(
        target=target,
        pe_array_size=(128, 128),
        vector_unit_width=1,
        frequency=2.4,
        scratchpad_size=sbuf_bytes,
        scratchpad_offset=4 << 20,
        double_buffered_l2=True,
        dram_size=12.0,
        dram_bandwidth=dram_bandwidth,
        dram_access_latency=1300.0,
    )


# The ``transform`` layouts: every GEMM weight contraction-major (``"ck"``,
# [in, out]; in the kc/ck naming K is the output dim), so a weight tile lands
# with the contraction on the partitions TensorE contracts over, and feature
# maps NCHW, so a pool tile has H on the partitions and W on the free axis.
LAYOUT = {
    "tensor_layout": "nchw",
    "gemm_weight_layout": "ck",
    "gemv_weight_layout": "ck",
}

TRN2 = _core("trn2", 28 << 20, 16 * 23.0)
TRN3 = _core("trn3", 30 << 20, 16 * 33.0)
