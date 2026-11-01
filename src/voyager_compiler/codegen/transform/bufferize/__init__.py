"""
Bufferized FX lowering.

Rewrites tiled FX nodes into an explicit, executable *bufferized FX graph* that
contains tile loops (``torch.ops.higher_order.while_loop``) and explicit memory
primitives in the ``voyager`` torch.library namespace (``voyager.alloc``,
``voyager.async_copy``, ``voyager.insert``), then generates protobuf /
graphviz / text from that graph.

Modules: ``ops`` (voyager.* primitives), ``utils`` (shared builder helpers +
layout projections), ``pipeline`` (the unified Pallas-style software-pipelining
scheduler and its GEMM / conv2d / pointwise / pool builders), ``builders``
(which builder lowers each node), ``bufferization`` (the rewrite pass),
``memory_planning`` (addresses), ``sync`` (the dispatches' commits and
counter waits), ``emit`` (loop-aware output).
"""

import torch
from torch.fx.node import has_side_effect

# Registers the voyager.* torch.library ops.
from voyager_compiler.codegen.transform.bufferize import ops  # noqa: F401
from voyager_compiler.codegen.transform.bufferize.ops import MemoryLevel
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    annotate_tensor_spaces,
    bufferize_graph,
)
from voyager_compiler.codegen.transform.bufferize.builders import (
    SHARED_BUILDERS,
    BuilderRule,
)
from voyager_compiler.codegen.transform.bufferize.emit import (
    flush_tensor_files,
    gen_code_bufferized,
    gen_compute_graph,
    print_bufferized_graph,
    print_layer_table,
)
from voyager_compiler.codegen.transform.bufferize.memory_planning import (
    MemoryPlan,
    plan_memory,
    planned_buffer_layout,
    shared_dram_layout,
)
from voyager_compiler.codegen.transform.bufferize.sync import (
    SyncError,
    synchronize,
)
from voyager_compiler.codegen.transform.bufferize.utils import BuildContext

# Mark the in-place / DMA primitives side-effecting so DCE never drops them.
has_side_effect(torch.ops.higher_order.cond)
has_side_effect(torch.ops.voyager.insert.default)
has_side_effect(torch.ops.voyager.async_copy.default)
has_side_effect(torch.ops.voyager.async_wait.default)
has_side_effect(torch.ops.voyager.zeros.default)
has_side_effect(torch.ops.voyager.ones.default)
has_side_effect(torch.ops.higher_order.commit)
has_side_effect(torch.ops.higher_order.with_scratch)

__all__ = [
    "bufferize_graph",
    "annotate_tensor_spaces",
    "BuildContext",
    "BuilderRule",
    "SHARED_BUILDERS",
    "print_layer_table",
    "flush_tensor_files",
    "gen_code_bufferized",
    "gen_compute_graph",
    "print_bufferized_graph",
    "plan_memory",
    "planned_buffer_layout",
    "MemoryLevel",
    "MemoryPlan",
    "shared_dram_layout",
    "synchronize",
    "SyncError",
]
