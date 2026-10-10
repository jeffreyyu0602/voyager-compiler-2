"""Quantization: choose a spec per node, observe, then bake in quant/dequant.

Everything runs through PT2E (``quantize_pt2e``), configured with the
comma-separated spec strings ``QuantizationSpec.from_str`` parses.
Training -- QAT, or with the backward pass quantized too -- is set up by
``training``.
"""

from voyager_compiler.quantization.fake_quantize import (
    DirectCastFakeQuantize,
    FusedAmaxObsFakeQuantize,
    GroupWiseAffineFakeQuantize,
    MXFakeQuantize,
    fake_quantize_class,
    get_quantization_map,
)
from voyager_compiler.quantization.gptq import (
    CACHE_RESERVE,
    compensate_weight,
    gptq,
)
from voyager_compiler.quantization.codebook_optimizer import (
    CodebookRefitter,
    Weighting,
    Histogram,
    codebook_qmaps,
    fit_codebooks,
    load_codebooks,
    optimal_codebooks,
    refit_window,
)
from voyager_compiler.quantization.qspec import QScheme, parse_codebook_dtype
from voyager_compiler.quantization.quantize_pt2e import (
    convert_pt2e,
    derive_bias_qparams_fn,
    fold_conv_bn_qat,
    freeze_cache_reads,
    freeze_weights,
    get_default_quantizer,
    prepare_pt2e,
    prepare_qat_pt2e,
    set_batch_norm_training,
    set_training,
    sink_obs_or_fq,
    swap_matmul_inputs,
)
from voyager_compiler.quantization.training import (
    TrainingQuantizers,
    capture_training,
    disable_observers,
    gemm_partners,
    gradient_program,
    prepare_from_args,
    prepare_training,
    update_program,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

__all__ = [
    "CACHE_RESERVE",
    "CodebookRefitter",
    "Weighting",
    "Histogram",
    "DerivedQuantizationSpec",
    "DirectCastFakeQuantize",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "TrainingQuantizers",
    "capture_training",
    "codebook_qmaps",
    "compensate_weight",
    "convert_pt2e",
    "derive_bias_qparams_fn",
    "disable_observers",
    "fake_quantize_class",
    "fit_codebooks",
    "fold_conv_bn_qat",
    "freeze_cache_reads",
    "freeze_weights",
    "gemm_partners",
    "get_default_quantizer",
    "get_quantization_map",
    "gptq",
    "gradient_program",
    "load_codebooks",
    "optimal_codebooks",
    "parse_codebook_dtype",
    "prepare_from_args",
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "prepare_training",
    "refit_window",
    "set_batch_norm_training",
    "set_training",
    "sink_obs_or_fq",
    "swap_matmul_inputs",
    "update_program",
]
