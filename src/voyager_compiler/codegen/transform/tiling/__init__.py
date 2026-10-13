"""Choosing tile sizes: the tile searches every backend shares.

``matrix_search`` maps a GEMM / conv through ``interstellar``,
``grid_search`` tiles a vector, GEMV or pool op and ``attention_search`` a
flash attention; ``csr_slice_planning`` chooses outlier-CSR slice widths
jointly for the GEMMs they couple.  Each asks the backend's ``RuntimeModel``
(``runtime_model``) to prepare each op, then whether a candidate fits and
what it costs, and stamps the winner as ``meta["tile_result"]``; ``tiler``
holds what they share.  ``lowering`` and ``traversal`` hold the graph facts
and the builders' walks every model reads; the models themselves live with
their backends (``voyager_compiler.backend``).  Nothing here imports
``bufferize`` or a backend — the dependencies run one way, so the builders
can ask for a tiling without a cycle.
"""

from voyager_compiler.codegen.transform.tiling.attention_search import (
    attention_op_tiling,
)
from voyager_compiler.codegen.transform.tiling.csr_slice_planning import (
    DEFAULT_MIN_CSR_SLICE,
    plan_csr_slices,
)
from voyager_compiler.codegen.transform.tiling.grid_search import (
    gemv_op_tiling,
    pool_op_tiling,
    vector_op_tiling,
)
from voyager_compiler.codegen.transform.tiling.matrix_search import (
    CONV_L3_ORDER,
    GEMM_L3_ORDER,
    get_tiling,
    prefetch_tilings,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    AttentionSpace,
    AttentionTiling,
    Cost,
    Fused,
    GridSpace,
    GridTiling,
    KernelInput,
    MatrixSpace,
    OpModel,
    Placement,
    Role,
    RuntimeModel,
    TileResult,
    pad_head,
)
from voyager_compiler.codegen.transform.tiling.tiler import (
    DEFAULT_RUNTIME_TOLERANCE,
    TileConstraint,
    Tiler,
    build_tiler,
)
from voyager_compiler.codegen.transform.tiling.traversal import (
    attention_kv_last,
)

__all__ = [
    "AttentionSpace",
    "AttentionTiling",
    "CONV_L3_ORDER",
    "Cost",
    "DEFAULT_MIN_CSR_SLICE",
    "DEFAULT_RUNTIME_TOLERANCE",
    "Fused",
    "GEMM_L3_ORDER",
    "GridSpace",
    "GridTiling",
    "KernelInput",
    "MatrixSpace",
    "OpModel",
    "Placement",
    "Role",
    "RuntimeModel",
    "TileConstraint",
    "TileResult",
    "Tiler",
    "attention_kv_last",
    "attention_op_tiling",
    "build_tiler",
    "gemv_op_tiling",
    "get_tiling",
    "pad_head",
    "plan_csr_slices",
    "pool_op_tiling",
    "prefetch_tilings",
    "vector_op_tiling",
]
