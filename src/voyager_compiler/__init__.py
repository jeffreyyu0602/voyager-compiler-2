"""Lower a quantized PyTorch model onto a Voyager-generated accelerator.

``transform()`` runs the hardware-lowering passes over an exported FX graph;
``compile()`` bufferizes, plans memory and emits the ``voyager`` IR.  Both
operate in place and leave the graph executable, so every stage can be checked
numerically against the original.
"""

import copy
import os
from typing import Callable, Optional, Tuple

import torch
from google.protobuf import text_format
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

# Defines torch.ops.quantized_ops.* (and the hardware-layout twins).  This must
# come first: the imports below reference those targets, and the op namespace
# resolves lazily, so an unregistered target fails at call time rather than
# here.
from voyager_compiler import ops as _register_ops  # noqa: F401
from voyager_compiler.cli_args import (
    add_compile_args,
    add_experiment_args,
    add_quantization_args,
)
from voyager_compiler.codegen import (
    deduplicate_nodes,
    extract_input_preprocessor,
    fold_constant_generators,
    fuse_operator,
    fuse_quantize_dequantize_with_producer,
    gen_compute_graph,
    inline_autocast_modules,
    match_and_fuse_rms_norm,
    move_folds_out,
    normalize_activation_layout,
    normalize_attention_operands,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
    quantizes_kv_cache,
    remove_fp32_casts,
    remove_prunable_ops,
    remove_zero_attention_mask,
    rename_nodes_with_param_names,
    replace_conv2d_with_im2col,
    replace_interpolate,
    scalarize_index_arithmetic,
    specialize_symbolic_sizes,
    split_kv_cache,
    split_token_blocked_caches,
    store_prompt_chunks,
)
from voyager_compiler.codegen.transform.bufferize import (
    bufferize_graph,
    flush_tensor_files,
    gen_code_bufferized,
    plan_memory,
    planned_buffer_layout,
    print_bufferized_graph,
    print_layer_table,
    shared_dram_layout,
    synchronize,
)
from voyager_compiler.codegen.transform.tiling import (
    DEFAULT_RUNTIME_TOLERANCE,
    build_tiler,
)
from voyager_compiler.export_utils import (
    export_model,
    get_aten_graph_module,
    get_conv_bn_layers,
    get_node_name_to_scope,
    merge_sibling_linears,
    print_node_scope_tabular,
)
from voyager_compiler.hardware_config import AcceleratorConfig
from voyager_compiler.modeling import (
    dispatch_model,
    get_device_map,
    insert_align_device_nodes,
)
from voyager_compiler.ops.layout import (
    DEFAULT_GEMM_LAYOUT,
    DEFAULT_TENSOR_LAYOUT,
    GEMM_LAYOUTS,
    GEMV_LAYOUTS,
    TENSOR_LAYOUTS,
    GemmLayout,
)
from voyager_compiler.quantization import (
    DerivedQuantizationSpec,
    DirectCastFakeQuantize,
    Example,
    FusedAmaxObsFakeQuantize,
    GroupWiseAffineFakeQuantize,
    MXFakeQuantize,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    QuantizedLM,
    QuantizedModel,
    TrainingQuantizers,
    TrainingStep,
    capture_training,
    convert_pt2e,
    derive_bias_qparams_fn,
    disable_observers,
    fold_conv_bn_qat,
    get_default_quantizer,
    gradient_program,
    prepare_for_qat,
    prepare_pt2e,
    prepare_qat_pt2e,
    prepare_training,
    set_batch_norm_training,
    set_training,
    sink_obs_or_fq,
    update_program,
)
from voyager_compiler.quantization.dtypes import (
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.shape_prop import (
    ShapeProp,
    constant_value,
    fake_like,
    fetch_attr,
    propagate_shape,
)
from voyager_compiler.utils import with_execution_context

__all__ = [
    "AcceleratorConfig",
    "DerivedQuantizationSpec",
    "DirectCastFakeQuantize",
    "Example",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "OpMatcher",
    "Pattern",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "QuantizedLM",
    "QuantizedModel",
    "ShapeProp",
    "TrainingQuantizers",
    "TrainingStep",
    "add_compile_args",
    "add_experiment_args",
    "add_quantization_args",
    "capture_training",
    "compile",
    "convert_pt2e",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "disable_observers",
    "dispatch_model",
    "export_model",
    "extract_input_preprocessor",
    "fetch_attr",
    "fold_conv_bn_qat",
    "fuse_operator",
    "get_aten_graph_module",
    "get_conv_bn_layers",
    "get_default_quantizer",
    "get_device_map",
    "get_node_name_to_scope",
    "gradient_program",
    "insert_align_device_nodes",
    "lower",
    "match_and_fuse_rms_norm",
    "merge_sibling_linears",
    "normalize_attention_operands",
    "pad_vit_embeddings_output",
    "planned_buffer_layout",
    "prepare_for_qat",
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "prepare_training",
    "print_node_scope_tabular",
    "propagate_shape",
    "quantize_to_nf",
    "quantize_to_posit",
    "quantizes_kv_cache",
    "remove_fp32_casts",
    "remove_zero_attention_mask",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "scalarize_index_arithmetic",
    "set_batch_norm_training",
    "set_training",
    "shared_dram_layout",
    "sink_obs_or_fq",
    "specialize_symbolic_sizes",
    "split_kv_cache",
    "split_token_blocked_caches",
    "store_prompt_chunks",
    "transform",
    "update_program",
    "with_execution_context",
]


class qscheme: ...


# Defined in voyager_compiler/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE


def _get_op_overload(op_name: str):
    all_overloads = []
    for lib in [torch.ops.aten, torch.ops.quantized_ops]:
        # Also check inplace version of the op (e.g., "add_" for "add")
        for name in [op_name, f"{op_name}_"]:
            if (packet := getattr(lib, name, None)) is None:
                continue
            all_overloads.extend(
                [getattr(packet, name) for name in packet.overloads()]
            )
    return all_overloads


class OpMatcher:
    """One op of a fusion pattern: its targets and a check of its operands.

    Args:
        *ops: The op names it matches, every overload of each, in
            ``aten`` and ``quantized_ops``.
        predicate: A check of the op node, e.g. of its operands; ``None``
            for none.
        keeps_input: Whether the op's input, the op before it in the
            chain, is read outside the group too: the instruction-level
            fusion (``fuse_patterns``) fuses the chain only then, returning
            it as a result of the group -- a reduction an instruction makes
            of the tile it stores.
    """

    targets: Tuple[torch._ops.OpOverload]
    predicate: Optional[Callable[[Node], bool]] = None

    def __init__(self, *ops, predicate=None, keeps_input=False):
        self.predicate = predicate
        self.keeps_input = keeps_input

        # Resolve symbolic ops
        targets = []
        for op in ops:
            targets.extend(_get_op_overload(op))

        # Freeze resolved targets
        self.targets = tuple(targets)

    def matches(self, node: Node) -> bool:
        if node.target not in self.targets:
            return False

        return self.predicate(node) if self.predicate else True


class Pattern(list):
    """A fusion pattern: the ``OpMatcher`` of each op along a producer ->
    consumer chain, in order, and an optional name.

    ``transform()`` fuses the model's own ops by unnamed patterns (a plain
    list is one too).  A named pattern is an instruction a backend's builder
    fuses the ops it built into (``fuse_patterns``); the fused node records
    the name as ``meta['pattern']``, and ``model.txt`` carries it.

    Args:
        *ops: The ops' matchers, in chain order.
        name: The instruction form the pattern is; ``None`` for none.
    """

    def __init__(self, *ops, name=None):
        super().__init__(ops)
        self.name = name


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    config=None,
    tensor_layout=DEFAULT_TENSOR_LAYOUT,
    gemm_layout=DEFAULT_GEMM_LAYOUT,
    gemv_layout=DEFAULT_GEMM_LAYOUT,
    skip_op_fusion=False,
    fuse_reshape=True,
    keep_fp32=True,
    cache_formats=None,
    host_scalars=True,
):
    """Lower ``model`` in place through the graph-level passes.

    The passes run on fake copies of the example inputs and read only their
    shapes (see ``shape_prop``).

    Args:
        model: The exported graph, rewritten in place.
        example_args: Example positional inputs.
        example_kwargs: Example keyword inputs; ``None`` for none.
        patterns: The fusion patterns ``fuse_operator`` applies, each a list
            of ``OpMatcher`` from anchor to tail.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware and skips padding.
        tensor_layout: The memory order of the 4-D feature maps that conv2d,
            the 2-D pools and the ops connected to them read and write:
            ``"nchw"`` (channels first, PyTorch's order) or ``"nhwc"``
            (channels last, with HWIO conv weights). GEMM operands are not
            affected; they follow ``gemm_layout`` and ``gemv_layout``.
        gemm_layout: The layout of a matrix-matrix GEMM in einsum form, in
            C (contraction), K (output features) and X (tokens):
            ``"xc,kc->xk"`` (PyTorch's), ``"xc,ck->xk"`` (the weight
            [contraction, out]) or ``"cx,ck->xk"`` (also the activations
            between GEMMs [features, tokens] wherever that transposes less,
            for a Trainium TensorE; see ``normalize_activation_layout``).
        gemv_layout: The layout of a matrix-vector GEMM (a single activation
            row): ``"xc,kc->xk"`` or ``"xc,ck->xk"``.
        skip_op_fusion: Skip operator fusion.
        fuse_reshape: Let fusion fold a trailing reshape or permute into
            its producer's output.
        keep_fp32: ``False`` computes in 16-bit float where the model casts
            up to float32 (``remove_fp32_casts``).
        cache_formats: For a prefill graph filling a decode graph's split
            KV caches, the transformed decode graph's
            ``meta["cache_formats"]``: the prefill then stores the prompt's
            whole chunks in the decode's quantized format, into the
            decode's own buffers (``store_prompt_chunks``).  None for any
            other graph.
        host_scalars: Turn index arithmetic on one-element tensors into
            control-processor scalars (``scalarize_index_arithmetic``), as
            the Voyager program runs it.  False keeps it in device tensors,
            so a GPU step reads nothing back to the host and can replay as
            a CUDA graph.

    Returns:
        ``model``.

    Raises:
        ValueError: ``model`` writes a split cache's residual but no
            ``cache_formats`` say how its chunks are stored.
    """
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware) skips padding and tiling.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    if not keep_fp32:
        remove_fp32_casts(model)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*map(fake_like, flatten_args))

    specialize_symbolic_sizes(model)
    # A decode cache whose blocks run along the positions is split here, so
    # its index arithmetic is folded and scalarized with the graph's own.
    lanes = config.vector_lanes if config.pe_array_size is not None else 1
    split_token_blocked_caches(model, lanes)
    fold_constant_generators(model)
    inline_autocast_modules(model)
    remove_prunable_ops(model)
    normalize_attention_operands(model)
    if host_scalars:
        scalarize_index_arithmetic(model)
    fuse_quantize_dequantize_with_producer(model)
    if cache_formats is not None:
        store_prompt_chunks(model, cache_formats)
    elif not model.meta.get("cache_formats") and any(
        n.target is torch.ops.aten.index_copy_.default
        and n.args[0].op == "get_attr"
        and n.args[0].target.endswith("_residual")
        for n in model.graph.nodes
    ):
        # A prefill writes a residual but folds no chunk of its own.
        raise ValueError(
            "the graph writes a split KV cache's residual: pass the decode "
            "graph's cache_formats so its chunks are stored in that format"
        )

    if config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, config.pe_array_size)

    assert tensor_layout in TENSOR_LAYOUTS, tensor_layout
    if tensor_layout == "nhwc":
        normalize_conv2d_layout(model)

    assert gemm_layout in GEMM_LAYOUTS, gemm_layout
    assert gemv_layout in GEMV_LAYOUTS, gemv_layout
    gemm = GemmLayout.parse(gemm_layout)
    gemv_weight = GemmLayout.parse(gemv_layout).weight
    normalize_gemm_weight_layout(model, gemm.weight, gemv_weight)
    if gemm.activation == "cx":
        normalize_activation_layout(model)

    if config.pe_array_size is not None:
        pad_vector_op_dimensions(model, config.vector_lanes)

    if not skip_op_fusion:
        fuse_operator(model, patterns, fuse_reshape)

    rename_nodes_with_param_names(model)
    deduplicate_nodes(model)

    return model


def lower(lm: QuantizedLM, replay: bool = True) -> None:
    """Convert ``lm`` and lower its inference graphs (``transform``, no
    hardware), so ``prefill`` and ``generate`` run them on the device they
    live on, the cache stored as the compiled programs store it.

    The decode graph is lowered in place, its index arithmetic kept on the
    device and its cache folds moved into ``lm.fold_step``
    (``move_folds_out``), so every step runs the same ops and, on a GPU,
    can replay as a CUDA graph.  Its stored cache formats are what each
    prefill writes the prompt in.  The prefill is lowered once per prompt
    length, the first time a prompt runs it: a copy of the converted graph
    holding its weights and cache, not copies of them.  Each lowered
    graph's deferred constants are computed and installed.

    Args:
        lm: A prepared, unfrozen model; converted and lowered in place.
        replay: On a GPU, replay the decode step as a CUDA graph.
    """
    lm.convert()
    device = lm.device

    def install_constants(graph):
        for node in graph.graph.nodes:
            if node.op == "get_attr":
                constant_value(graph, node)

    formats = None
    if "decode" in lm.graphs:
        decode = lm.graphs["decode"]
        end = lm.context_length + lm.decode_tokens
        step = {
            "input_ids": torch.zeros(
                1, lm.decode_tokens, dtype=torch.long, device=device
            ),
            "cache_position": torch.arange(
                lm.context_length, end, device=device
            ),
        }
        transform(decode, (), step, host_scalars=False)
        install_constants(decode)
        formats = decode.meta.get("cache_formats") or None
        lm.fold_step = move_folds_out(decode)
    if "prefill" in lm.graphs:
        prefill = lm.graphs["prefill"]
        lowered = {}

        def prompt_graph(length):
            if length not in lowered:
                graph = GraphModule(prefill, copy.deepcopy(prefill.graph))
                prompt = {
                    "input_ids": torch.zeros(
                        1, length, dtype=torch.long, device=device
                    ),
                    "cache_position": torch.arange(length, device=device),
                }
                transform(graph, (), prompt, cache_formats=formats)
                install_constants(graph)
                lowered[length] = graph
            return lowered[length]

        lm.prompt_graph = prompt_graph
    lm.replay = replay


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    config=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    runtime_tolerance=None,
    dram_layout=None,
    accumulate_fp32=False,
    bool_mask=False,
):
    """Bufferize ``model``, plan its memory, synchronize its dispatches
    and write its ``voyager`` IR.

    Writes ``model.txt`` (the IR as text), ``layers.txt`` (a table of its
    layers), ``bufferized_pre_sync.log`` (the bufferized graph before the
    sync pass) and ``<output_file>.svg`` (the compute graph) to
    ``output_dir``.

    Args:
        model: A graph ``transform`` lowered, bufferized in place.
        example_args: The inputs ``model`` runs on. With ``dump_tensors``,
            one it writes, such as a gradient buffer, is overwritten;
            otherwise only their shapes and dtypes are read.
        example_kwargs: Keyword inputs; ``None`` for none.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware.
        output_dir: The directory the files are written to.
        output_file: The compute-graph drawing's basename.
        dump_tensors: Also write every tensor's values under
            ``output_dir/tensor_files``.
        runtime_tolerance: How much longer than the fastest modeled tiling
            a tiling may run and still be chosen, as a fraction; ``None``
            takes ``DEFAULT_RUNTIME_TOLERANCE``.
        dram_layout: The DRAM ranges ``shared_dram_layout`` agreed for the
            inputs ``model`` shares with other programs, such as a training
            step's gradient and update programs, or that
            ``planned_buffer_layout`` found for the ``get_attr`` buffers
            it shares with an already planned program, such as a decode
            graph's KV cache; ``None`` places every input freely.
        accumulate_fp32: Accumulate a GEMM split along K in float32 rather
            than its output dtype.
        bool_mask: Store an ``is_causal`` flash attention's mask table as
            int1 codes plus a scale, for hardware whose vector unit reads
            int1 (MXNF4); off, as the additive fill at the accumulation
            dtype.

    Returns:
        The ``voyager`` IR ``Model`` written to ``model.txt``.
    """
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*map(fake_like, flatten_args))

    gen_compute_graph(model, os.path.join(output_dir, output_file))

    tolerance = (
        DEFAULT_RUNTIME_TOLERANCE
        if runtime_tolerance is None
        else runtime_tolerance
    )
    tiler = build_tiler(
        config, runtime_tolerance=tolerance, accumulate_fp32=accumulate_fp32
    )
    backend = config.backend()
    bufferize_graph(
        model,
        pipelined=config.double_buffered_l2,
        tiler=tiler,
        async_pipeline=config.async_pipeline,
        bool_mask=bool_mask,
        builders=backend.builders,
    )
    plan_memory(model, config, dram_layout)
    # The builders' graph, as planned, goes to a file of its own; the log
    # shows the graph the sync pass leaves, so the two read side by side.
    with open(os.path.join(output_dir, "bufferized_pre_sync.log"), "w") as f:
        f.write(print_bufferized_graph(model, to_string=True))
    # The sync pass: every dispatch a commit, every cross-unit dependency a
    # counter wait.  A target whose own toolchain orders the work skips it.
    if backend.synchronizes:
        synchronize(model)
        print("Bufferized graph after synchronize:")
    else:
        print("Bufferized graph (no sync pass for this backend):")
    print_bufferized_graph(model)

    # Without a dump the emitter reads only shapes and dtypes, so it runs on
    # fake inputs. The deferred constants are installed before it, so the
    # emitter reads them, and every node computed from them alone, as real
    # values, and the callers that run or save the graph find them in place.
    for sub in model.modules():
        if isinstance(sub, torch.fx.GraphModule):
            for node in sub.graph.find_nodes(op="get_attr"):
                constant_value(sub, node)
    path = os.path.join(output_dir, "tensor_files")
    params = gen_code_bufferized(
        model,
        flatten_args if dump_tensors else list(map(fake_like, flatten_args)),
        path if dump_tensors else None,
    )

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))
    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write(print_layer_table(model, params, to_string=True))

    flush_tensor_files()
    return params
