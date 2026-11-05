"""The hardware-lowering passes, in the order ``transform()`` runs them."""

from voyager_compiler.codegen.transform.data_layout import (
    eliminate_reshape_with_no_effect,
    normalize_activation_layout,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
)
from voyager_compiler.codegen.transform.operator_fusion import (
    fuse_operator,
    fuse_patterns,
)
from voyager_compiler.codegen.transform.padding import (
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
)
from voyager_compiler.codegen.transform.quant_folding import (
    fuse_quantize_dequantize_with_producer,
    quantizes_kv_cache,
    sink_cache_folds,
    split_kv_cache,
)
from voyager_compiler.codegen.transform.rewrites import (
    deduplicate_nodes,
    extract_input_preprocessor,
    fold_constant_generators,
    inline_autocast_modules,
    match_and_fuse_rms_norm,
    normalize_attention_operands,
    remove_fp32_casts,
    remove_prunable_ops,
    remove_zero_attention_mask,
    replace_conv2d_with_im2col,
    replace_interpolate,
)
from voyager_compiler.codegen.transform.scalarize import (
    scalarize_single_elements,
)

__all__ = [
    "deduplicate_nodes",
    "eliminate_reshape_with_no_effect",
    "extract_input_preprocessor",
    "fold_constant_generators",
    "fuse_operator",
    "fuse_patterns",
    "fuse_quantize_dequantize_with_producer",
    "inline_autocast_modules",
    "match_and_fuse_rms_norm",
    "normalize_activation_layout",
    "normalize_attention_operands",
    "normalize_conv2d_layout",
    "normalize_gemm_weight_layout",
    "pad_matrix_op_dimensions",
    "pad_vector_op_dimensions",
    "pad_vit_embeddings_output",
    "quantizes_kv_cache",
    "remove_fp32_casts",
    "remove_prunable_ops",
    "remove_zero_attention_mask",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "scalarize_single_elements",
    "sink_cache_folds",
    "split_kv_cache",
]
