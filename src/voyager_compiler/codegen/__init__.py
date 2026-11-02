"""Hardware lowering: the ``transform()`` passes and the bufferized backend.

Every pass ``transform()`` runs is exported from here, in the order the
pipeline applies them.
"""

from voyager_compiler.codegen.subgraph import rename_nodes_with_param_names
from voyager_compiler.codegen.transform.bufferize import gen_compute_graph
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
    CacheFormat,
    fuse_quantize_dequantize_with_producer,
    move_folds_out,
    quantizes_kv_cache,
    sink_cache_folds,
    split_kv_cache,
    split_token_blocked_caches,
    store_prompt_chunks,
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
    specialize_symbolic_sizes,
)
from voyager_compiler.codegen.transform.scalarize import (
    scalarize_index_arithmetic,
)
from voyager_compiler.codegen.transform.tiling.grid_search import (
    gemv_op_tiling,
    pool_op_tiling,
    vector_op_tiling,
)

__all__ = [
    "CacheFormat",
    "deduplicate_nodes",
    "eliminate_reshape_with_no_effect",
    "extract_input_preprocessor",
    "fold_constant_generators",
    "fuse_operator",
    "fuse_patterns",
    "fuse_quantize_dequantize_with_producer",
    "gen_compute_graph",
    "inline_autocast_modules",
    "match_and_fuse_rms_norm",
    "move_folds_out",
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
    "rename_nodes_with_param_names",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "scalarize_index_arithmetic",
    "sink_cache_folds",
    "specialize_symbolic_sizes",
    "split_kv_cache",
    "split_token_blocked_caches",
    "store_prompt_chunks",
    "gemv_op_tiling",
    "vector_op_tiling",
    "pool_op_tiling",
]
