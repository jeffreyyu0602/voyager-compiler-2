"""A model quantized once for inference and training.

``QuantizedModel`` exports a module twice, into a prepared inference graph
and the joint training graph ``prepare_training`` captures, and shares the
fake-quants between them, so the scales one graph calibrates are the ones
the other quantizes with.  ``QuantizedLM`` does the same for a causal LM
whose inference runs as a prefill graph and a static-cache decode graph,
behind an HF-style ``generate``.  Either builds only the phases asked of it.
"""

import logging
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from unittest.mock import patch

import torch
from torch import nn
from torch.fx import GraphModule
from torchao.quantization.pt2e import FakeQuantizeBase
from transformers import masking_utils
from transformers.cache_utils import StaticLayer
from transformers.integrations.executorch import (
    TorchExportableModuleWithStaticCache,
)

from voyager_compiler.export_utils import export_model
from voyager_compiler.quantization.quantize_pt2e import (
    convert_pt2e,
    freeze_cache_reads,
    freeze_weights,
    prepare_pt2e,
    sink_obs_or_fq,
)
from voyager_compiler.quantization.sharing import share_fake_quants
from voyager_compiler.quantization.training import (
    DEFAULT_BACKWARD_OVERRIDE,
    prepare_training,
)

logger = logging.getLogger(__name__)

# The prefill graph takes prompts of any multiple of this many tokens: the
# microscaling block along the sequence axis.
SEQ_BLOCK = 64
# A decode graph's KV-cache buffers, split or whole.
_CACHE_BUFFER = re.compile(r"(key|value)_cache_\d+(_residual)?")
_RESIDUAL = "_residual"

Rewrite = Callable[[GraphModule, str], None]


@dataclass
class Example:
    """Inputs a graph is exported with.

    Args:
        args: Positional example inputs.
        kwargs: Keyword example inputs, or None for none.
        dynamic_shapes: Dynamic dimensions of the inputs, keyed by the
            model's parameter names, or None for static shapes.
    """

    args: Tuple[Any, ...]
    kwargs: Optional[Dict[str, Any]] = None
    dynamic_shapes: Optional[Dict[str, Any]] = None


def _quantizer(quantizer, phase: str):
    """The quantizer of ``phase``: one of a mapping by phase, else the one
    quantizer given for every phase."""
    return quantizer[phase] if isinstance(quantizer, Mapping) else quantizer


def _check_phases(
    phases: Sequence[str], known: Sequence[str], training: Optional[Example]
) -> None:
    """Raise unless ``phases`` names one or more of ``known``, with the
    training example given when it names ``"training"``."""
    unknown = set(phases) - set(known)
    if not phases or unknown:
        raise ValueError(f"phases must be some of {known}, got {tuple(phases)}")
    if "training" in phases and training is None:
        raise ValueError("the training phase needs a training example")


def _reads_cache_quantized(graph: GraphModule) -> bool:
    """Whether a fake-quant of the prepared decode ``graph`` reads a KV
    cache -- through its write and single-operand views -- so the cache is
    stored quantized."""
    for node in graph.graph.nodes:
        if node.op != "call_module" or not isinstance(
            graph.get_submodule(node.target), FakeQuantizeBase
        ):
            continue
        source = node.args[0]
        while source.op != "get_attr" and (
            source.target is torch.ops.aten.index_copy_.default
            or len(source.all_input_nodes) == 1
        ):
            source = source.all_input_nodes[0]
        if source.op == "get_attr" and _CACHE_BUFFER.fullmatch(source.target):
            return True
    return False


def _prepare(
    graph: GraphModule, phase: str, quantizer, rewrite: Optional[Rewrite]
) -> GraphModule:
    """Rewrite ``graph`` for ``phase`` if a rewrite is given, prepare it with
    the phase's quantizer and sink its weight fake-quants to their first
    use."""
    if rewrite is not None:
        rewrite(graph, phase)
    graph = prepare_pt2e(graph, _quantizer(quantizer, phase))
    return sink_obs_or_fq(graph)


class QuantizedModel(nn.Module):
    """A module prepared for quantized inference and quantized training.

    The module is exported in eval mode into the inference graph and in
    train mode into the joint, forward and backward graphs of
    ``prepare_training``; every graph reads the module's own parameters,
    and the module's forward runs the training graphs from then on.  The
    fake-quants of the inference graph are shared with the training graphs
    wherever both quantize the same operand the same way
    (``share_fake_quants``), so calibrating one calibrates the other.
    ``forward`` runs the captured training forward: in train mode with the
    backward recorded, in eval mode under the caller's grad setting.
    ``eval()`` fixes every scale and ``train()`` lets them move again;
    ``convert`` bakes the inference graph, after which only ``infer`` runs.
    ``phases`` builds the inference graph or the training graphs alone.
    The state dict holds the module's own tensors and every fake-quant: the
    training graphs' under ``model.training_quantizers`` and each inference
    graph's under ``inference_quantizers.<graph>``.

    Args:
        model: Module whose forward returns the loss, or an output with a
            ``loss`` attribute, on the training example.
        quantizer: Annotates every export, or a mapping from phase
            (``"inference"``, ``"training"``) to the quantizer of that
            phase.
        inference: Example inputs of the inference graph; None when it is
            not built.
        training: Example inputs of the training graphs; None when they
            are not built.
        phases: The phases to build: ``"inference"``, ``"training"`` or
            both.
        rewrite: Called with each exported graph and its phase name before
            it is prepared, for rewrites that must precede quantization;
            None for none.
        backward_override: As ``prepare_training`` takes.

    Raises:
        ValueError: ``phases`` names an unknown phase, or none, or one
            whose example is missing.
    """

    # Module path of the model within each inference graph.
    PREFIX = ""
    # The inference graphs' keys in ``graphs``.
    INFERENCE = ("inference",)
    PHASES = INFERENCE + ("training",)

    def __init__(
        self,
        model: nn.Module,
        quantizer,
        inference: Optional[Example] = None,
        training: Optional[Example] = None,
        *,
        phases: Sequence[str] = PHASES,
        rewrite: Optional[Rewrite] = None,
        backward_override: str = DEFAULT_BACKWARD_OVERRIDE,
    ) -> None:
        super().__init__()
        _check_phases(phases, self.PHASES, training)
        graphs = {}
        if "inference" in phases:
            if inference is None:
                raise ValueError("the inference phase needs an example")
            graph = export_model(
                model.eval(),
                inference.args,
                inference.kwargs,
                dynamic_shapes=inference.dynamic_shapes,
            )
            graphs["inference"] = _prepare(
                graph, "inference", quantizer, rewrite
            )
        self._capture(
            model, graphs, quantizer, phases, training, backward_override
        )

    def _capture(
        self,
        model: nn.Module,
        graphs: Dict[str, GraphModule],
        quantizer,
        phases: Sequence[str],
        training: Optional[Example],
        backward_override: str,
    ) -> None:
        """Capture the training graphs of ``model`` if ``phases`` asks for
        them, share the fake-quants of the prepared inference ``graphs``
        and the training graphs, and end in eval mode with every observer
        still on."""
        self.converted = False
        self.model = model
        self.phases = tuple(phases)
        self.graphs = dict(graphs)
        shared = list(graphs.values())
        prefixes = [self.PREFIX] * len(graphs)
        holders, roots = {}, {}
        if "training" in phases:
            prepare_training(
                model.train(),
                _quantizer(quantizer, "training"),
                training.args,
                training.kwargs,
                training.dynamic_shapes,
                backward_override=backward_override,
            )
            self.graphs.update(model.forward.graphs)
            joint = self.graphs["joint"]
            shared.append(joint)
            prefixes.append("")
            holders[joint] = [
                self.graphs["forward"],
                self.graphs["backward"],
                self.training_quantizers,
            ]
            roots[joint] = model
        self.reports = share_fake_quants(shared, prefixes, holders, roots)
        self.inference_quantizers = nn.ModuleDict(
            {
                phase: nn.ModuleDict(
                    {
                        name: module
                        for name, module in graph.named_children()
                        if isinstance(module, FakeQuantizeBase)
                    }
                )
                for phase, graph in graphs.items()
            }
        )
        nn.Module.train(self, False)

    @property
    def training_quantizers(self) -> nn.Module:
        """The training graphs' fake-quants, as ``prepare_training`` holds
        them."""
        return self.model.training_quantizers

    def fake_quants(self) -> Iterator[FakeQuantizeBase]:
        """Yield every fake-quant of the graphs, a shared one once, and those
        ``inference_quantizers`` holds beyond them: a frozen graph's reads of
        its caches run outside it."""
        seen = set()
        holders = [*self.graphs.values(), self.inference_quantizers]
        for holder in holders:
            for module in holder.modules():
                if (
                    isinstance(module, FakeQuantizeBase)
                    and id(module) not in seen
                ):
                    seen.add(id(module))
                    yield module

    def enable_observers(self) -> None:
        """Let every scale follow the values it sees."""
        for fake_quant in self.fake_quants():
            fake_quant.enable_observer()

    def disable_observers(self) -> None:
        """Fix every scale except the gradients', which keep moving."""
        for fake_quant in self.fake_quants():
            fake_quant.disable_observer()
        if "training" not in self.phases:
            return
        for name in self.training_quantizers.gradients:
            getattr(self.training_quantizers, name).enable_observer()

    def train(self, mode: bool = True) -> "QuantizedModel":
        """Switch modes as ``nn.Module.train`` does, with the observers on
        in train mode and off in eval mode.

        Raises:
            RuntimeError: Train mode was asked of a converted model.
        """
        if mode and self.converted:
            raise RuntimeError("A converted model has its weights baked")
        nn.Module.train(self, mode)
        if mode:
            self.enable_observers()
        else:
            self.disable_observers()
        return self

    def forward(self, *args, **kwargs):
        """Run the captured training forward.

        Raises:
            RuntimeError: The model was converted, or built without its
                training graphs.
        """
        if self.converted:
            raise RuntimeError("A converted model has its weights baked")
        if "training" not in self.phases:
            raise RuntimeError("The model was built without training graphs")
        return self.model(*args, **kwargs)

    def infer(self, *args, **kwargs):
        """Run the inference graph."""
        return self.graphs["inference"](*args, **kwargs)

    def inference_graphs(self) -> Dict[str, GraphModule]:
        """The inference graphs the model was built with, by phase."""
        return {k: self.graphs[k] for k in self.INFERENCE if k in self.graphs}

    def freeze(self) -> None:
        """Bake each weight's fake-quantization into the inference graphs
        (``freeze_weights``), for evaluation with the scales fixed.  A
        weight every inference graph quantizes alike is written in place,
        so the training graphs read it quantized and their fake-quant of it
        is switched off; ``convert`` rejects the frozen graphs.

        Raises:
            RuntimeError: The model was built without inference graphs.
            ValueError: The model is frozen already.
        """
        graphs = self.inference_graphs()
        if not graphs:
            raise RuntimeError("The model was built without inference graphs")
        if any(graph.meta.get("frozen_weights") for graph in graphs.values()):
            raise ValueError("The model is frozen already")
        for module in freeze_weights(*graphs.values()):
            module.disable_fake_quant()
        for phase, graph in graphs.items():
            kept = {id(m) for m in graph.modules()}
            quantizers = self.inference_quantizers[phase]
            for name in [n for n, m in quantizers.items() if id(m) not in kept]:
                del quantizers[name]

    def convert(
        self,
        output_dtype: Optional[str] = None,
        mismatched_weights: str = "copy",
    ) -> None:
        """Bake the inference graphs (``convert_pt2e``) and drop their
        fake-quants.  The weights they share with the training graphs are
        quantized in place, so ``forward`` and ``train`` raise from here on.

        Args:
            output_dtype: As ``convert_pt2e`` takes.
            mismatched_weights: As ``convert_pt2e`` takes.

        Raises:
            RuntimeError: The model was built without inference graphs.
        """
        graphs = list(self.inference_graphs().values())
        if not graphs:
            raise RuntimeError("The model was built without inference graphs")
        convert_pt2e(
            graphs,
            output_dtype,
            mismatched_weights=mismatched_weights,
        )
        self.inference_quantizers.clear()
        self.converted = True


class _PromptLayer(StaticLayer):
    """A static cache layer taking a prompt written from position 0: the
    write lands in the cache at the positions ``0 .. length``, but the
    attention reads the fresh keys and values, so the mask is the prompt's
    own causal one and, as the layer is not compileable, sdpa may run
    ``is_causal`` without it.

    Args:
        layer: The initialized layer whose tensors this one takes over.
    """

    is_compileable = False

    def __init__(self, layer: StaticLayer) -> None:
        super().__init__(layer.max_cache_len)
        vars(self).update(vars(layer))

    def update(self, key_states, value_states, *args, **kwargs):
        length = key_states.shape[-2]
        positions = torch.arange(length, device=self.device)
        self.cumulative_length.add_(length)
        self.keys.index_copy_(2, positions, key_states)
        self.values.index_copy_(2, positions, value_states)
        return key_states, value_states

    def get_mask_sizes(self, query_length):
        return query_length, 0


class _StaticCacheLM(TorchExportableModuleWithStaticCache):
    """A causal LM over a static KV cache, returning the logits of the last
    ``logits_to_keep`` positions, every position's for 0.  One position
    lets the head lower as a GEMV."""

    logits_to_keep = 1

    def forward(self, input_ids, cache_position):
        for layer in self.static_cache.layers:
            layer.cumulative_length.copy_(cache_position[0])
        return self.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=self.static_cache,
            use_cache=True,
            logits_to_keep=self.logits_to_keep,
        ).logits

    @property
    def slides(self) -> bool:
        """Whether a layer attends through a sliding window."""
        config = self.model.config.get_text_config()
        return "sliding_attention" in (
            getattr(config, "layer_types", None) or ()
        )

    @contextmanager
    def prompt_layers(self):
        """Run with every cache layer a ``_PromptLayer``, for the export of
        a prompt whose attention reads the fresh keys and values.  Under
        sdpa, transformers builds a mask whenever it traces; here the
        prompt attends to itself alone, so the mask is skipped for
        ``is_causal``.

        Raises:
            NotImplementedError: A layer attends through a sliding window,
                which the prompt's own mask does not apply.
        """
        if self.slides:
            raise NotImplementedError(
                "a prompt attending its fresh keys and values is not lowered "
                "for a sliding-window layer"
            )

        def skip(padding_mask, query_length, kv_length, *unused):
            return query_length == kv_length

        layers = self.static_cache.layers
        self.static_cache.layers = [_PromptLayer(layer) for layer in layers]
        try:
            with patch.object(masking_utils, "_ignore_causal_mask_sdpa", skip):
                yield
        finally:
            self.static_cache.layers = layers


class QuantizedLM(QuantizedModel):
    """A Hugging Face causal LM quantized for prefill, decode and training.

    Inference runs on two graphs over one static KV cache of ``cache_len``
    slots: ``"prefill"`` takes a prompt of any multiple of ``SEQ_BLOCK``
    tokens up to ``context_length`` and ``"decode"`` one token at a cache
    position; the prefill returns the last ``logits_to_keep`` positions'
    logits, the decode the last position's.  ``generate`` continues a
    prompt greedily on them.  The decode graph is exported and rewritten
    first.  The prefill is exported attending its fresh keys and values,
    not the whole cache, which also lets the compiled prefill store the
    prompt in the decode's quantized cache format; a model with
    sliding-window layers attends the cache unless it is stored
    quantized, which such a model does not lower.  When the rewrite
    split the decode's caches (``split_kv_cache``) into completed chunks
    and a full-precision residual, the decode quantizer must leave the
    attention's residual halves unquantized, and the prefill must split
    its cache writes alike: it then writes the prompt's whole chunks into
    the main caches and the tail into the residuals, which both graphs
    hold as one tensor.

    The inference graphs take one of four forms, along two paths from the
    model as built -- prepared to frozen, and prepared to converted to
    lowered; a form is never left, and ``convert`` rejects a frozen model:

    - prepared, as built: every quantization is a fake-quant, calibrating
      while its observer is on;
    - frozen (``freeze``), for fast fake-quant evaluation: each weight's
      fake-quant and the reads of a split cache's main halves are computed
      once, the activations' every step;
    - converted (``convert``): real quantize and dequantize ops, each
      weight stored as its codes and scales -- the graphs the compiler's
      ``transform`` and ``compile`` take;
    - lowered (``voyager_compiler.lower``): converted, then ``transform``
      with no hardware, so the KV cache is stored and read as the compiled
      programs store and read it.

    A frozen or lowered model on a GPU replays its decode step as a CUDA
    graph (``replay``).  ``prompt_graph``, when set, gives the prefill graph
    each prompt length runs instead of ``graphs["prefill"]``;
    ``fold_step``, when set, completes the chunks of a split cache after
    each decode step, for a decode graph that does not: both are how a
    lowered model runs.

    Args:
        model: The causal LM, with eager attention for the training lowering.
            Its ``generation_config`` is set to a static cache of
            ``cache_len`` slots.
        quantizer: Annotates every export, or a mapping from phase
            (``"prefill"``, ``"decode"``, ``"training"``) to the quantizer
            of that phase.
        training: Example inputs of the training graphs; None when they
            are not built.
        context_length: Longest prompt the prefill graph takes, rounded
            down to a multiple of ``SEQ_BLOCK``.
        cache_len: Slots in the KV cache.
        phases: The phases to build: some of ``"prefill"``, ``"decode"``
            and ``"training"``.  ``generate`` needs both inference graphs.
        logits_to_keep: Positions whose logits the prefill graph returns,
            as Hugging Face's argument of that name: the last ones, or
            every one for 0.
        rewrite: Called with each exported graph and its phase name before
            it is prepared; None for none.
        decode_tokens: Positions a decode step writes at once: one, or the
            draft tokens of a speculative-decoding verification step, which
            ``generate`` and ``step`` do not drive.
        backward_override: As ``prepare_training`` takes.

    Raises:
        ValueError: ``phases`` names an unknown phase, or none, or the
            training phase without its example.
    """

    PREFIX = "model"
    INFERENCE = ("prefill", "decode")
    PHASES = INFERENCE + ("training",)

    def __init__(
        self,
        model: nn.Module,
        quantizer,
        training: Optional[Example] = None,
        *,
        context_length: int,
        cache_len: int,
        phases: Sequence[str] = PHASES,
        logits_to_keep: int = 1,
        rewrite: Optional[Rewrite] = None,
        decode_tokens: int = 1,
        backward_override: str = DEFAULT_BACKWARD_OVERRIDE,
    ) -> None:
        nn.Module.__init__(self)
        _check_phases(phases, self.PHASES, training)
        model.eval()
        model.generation_config.update(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": cache_len},
        )
        cache = _StaticCacheLM(model)
        device = model.device
        graphs = {}
        if "decode" in phases:
            cache.logits_to_keep = 1
            decode = export_model(
                cache,
                (),
                {
                    "input_ids": torch.zeros(
                        1, decode_tokens, dtype=torch.long, device=device
                    ),
                    "cache_position": torch.arange(
                        context_length,
                        context_length + decode_tokens,
                        device=device,
                    ),
                },
                strict=True,
            )
            graphs["decode"] = _prepare(decode, "decode", quantizer, rewrite)
        # The residuals of a split decode cache, which the prefill shares.
        residuals = {
            name: buffer
            for name, buffer in graphs.get(
                "decode", nn.Module()
            ).named_buffers()
            if name.endswith(_RESIDUAL)
        }
        # The prompt attends its fresh keys and values rather than the
        # whole cache, as it must when the cache is stored quantized, split
        # or whole; a sliding window, which that mask does not apply,
        # attends the cache otherwise.
        prompt = (
            bool(residuals)
            or ("decode" in graphs and _reads_cache_quantized(graphs["decode"]))
            or not cache.slides
        )
        if "prefill" in phases:
            chunks = context_length // SEQ_BLOCK
            dynamic = None
            if chunks > 1:
                chunk = torch.export.Dim("chunk", min=1, max=chunks)
                dynamic = {
                    "input_ids": {1: chunk * SEQ_BLOCK},
                    "cache_position": {0: chunk * SEQ_BLOCK},
                }
            cache.logits_to_keep = logits_to_keep
            with cache.prompt_layers() if prompt else nullcontext():
                prefill = export_model(
                    cache,
                    (),
                    {
                        "input_ids": torch.zeros(
                            1,
                            chunks * SEQ_BLOCK,
                            dtype=torch.long,
                            device=device,
                        ),
                        "cache_position": torch.arange(
                            chunks * SEQ_BLOCK, device=device
                        ),
                    },
                    dynamic_shapes=dynamic,
                    strict=True,
                )
            prefill = _prepare(prefill, "prefill", quantizer, rewrite)
            for name, residual in residuals.items():
                if not hasattr(prefill, name):
                    raise ValueError(
                        f"the prefill graph does not write {name}: split its "
                        "caches as the decode graph's are split"
                    )
                delattr(prefill, name)
                prefill.register_buffer(name, residual)
            graphs = {"prefill": prefill, **graphs}
        self.context_length = context_length
        self.cache_len = cache_len
        self.decode_tokens = decode_tokens
        # The longest chunk of a split cache, read off the residual buffers;
        # None for whole caches.
        self.split = max((r.shape[2] for r in residuals.values()), default=None)
        eos = model.generation_config.eos_token_id
        self.eos_token_ids = [eos] if isinstance(eos, int) else list(eos or [])
        self._refresh = None
        # The prefill graph for a prompt length, when one graph does not
        # take every length; None to run ``graphs["prefill"]``.
        self.prompt_graph = None
        # Called with each decode step's position when the decode graph
        # leaves its cache folds to the caller; None when it does not.
        self.fold_step = None
        self._replay = False
        self.cuda_graph = None
        self.samples = 0
        self.prefill_seconds = 0.0
        self.tokens = 0
        self.decode_seconds = 0.0
        self._capture(
            model, graphs, quantizer, phases, training, backward_override
        )

    @property
    def config(self):
        """The LM's configuration."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Where the LM and its graphs live."""
        return self.model.device

    @property
    def replay(self) -> bool:
        """Whether ``step`` replays the decode step as a CUDA graph."""
        return self._replay

    @replay.setter
    def replay(self, on: bool) -> None:
        # A CUDA graph exists only on a GPU; a capture of the old step is
        # stale.
        self._replay = on and self.device.type == "cuda"
        self.cuda_graph = None

    @staticmethod
    def _residuals(
        decode: GraphModule,
    ) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Yield ``(name, main, residual)`` for each split cache of
        ``decode``, ``name`` being the main buffer's."""
        for name, residual in decode.named_buffers():
            if name.endswith(_RESIDUAL):
                main = name[: -len(_RESIDUAL)]
                yield main, decode.get_buffer(main), residual

    def freeze(self, replay: bool = True) -> None:
        """Freeze the weights (``QuantizedModel.freeze``), then what the
        decode graph computes from a split cache's main halves
        (``freeze_cache_reads``).  From then on ``prefill`` and ``fold``
        write the main halves and recompute those reads.

        Args:
            replay: On a GPU, replay the decode step as a CUDA graph.

        Raises:
            RuntimeError: The model was built without inference graphs.
            ValueError: The model is frozen already.
        """
        super().freeze()
        if self.split is not None:
            decode = self.graphs["decode"]
            caches = [name for name, _, _ in self._residuals(decode)]
            self._refresh = freeze_cache_reads(decode, caches)
        self.replay = replay

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Write a prompt into the emptied cache and return the logits of
        its last token.  The prefill graph takes the longest multiple of
        ``SEQ_BLOCK`` it accepts, writing a split cache's whole chunks to
        the main caches and the tail to the residuals; the decode graph
        takes the rest token by token.  Frozen reads of the main caches are
        refreshed.

        Raises:
            RuntimeError: The model was built without a graph the prompt
                needs: the prefill for a whole block, the decode for the
                rest.
        """
        length = input_ids.shape[1]
        done = min(length, self.context_length) // SEQ_BLOCK * SEQ_BLOCK
        needed = []
        if done:
            needed.append("prefill")
        if done < length:
            needed.append("decode")
        missing = [phase for phase in needed if phase not in self.graphs]
        if missing:
            raise RuntimeError(
                f"a {length}-token prompt needs the {' and '.join(missing)} "
                "graph, which the model was built without"
            )
        for graph in self.inference_graphs().values():
            for name, buffer in graph.named_buffers():
                if _CACHE_BUFFER.fullmatch(name):
                    buffer.zero_()
        logits = None
        if done:
            prefill = (
                self.graphs["prefill"]
                if self.prompt_graph is None
                else self.prompt_graph(done)
            )
            logits = prefill(
                input_ids=input_ids[:, :done],
                cache_position=torch.arange(done, device=self.device),
            )[0, -1]
        if self._refresh is not None:
            self._refresh()
        for position in range(done, length):
            logits = self.step(input_ids[0, position].item(), position)
        return logits

    def fold(self, position: int) -> None:
        """Copy each residual whose chunk ``position`` completes into that
        chunk of its main cache, the write the decode graph leaves to the
        caller once its main-cache reads are frozen, and recompute those
        reads."""
        folded = False
        for _, main, residual in self._residuals(self.graphs["decode"]):
            length = residual.shape[2]
            if position % length == length - 1:
                start = position - length + 1
                main[:, :, start : start + length] = residual
                folded = True
        if folded:
            self._refresh()

    def capture(self, token: int, position: int) -> None:
        """Capture one decode step as a CUDA graph.  The warmup steps it
        needs write ``token`` at ``position``, as the captured step and its
        first replay do."""
        decode = self.graphs["decode"]
        self.static_ids = torch.tensor([[token]], device=self.device)
        self.static_pos = torch.tensor([position], device=self.device)
        # Capture on the graphs' own device: ``torch.cuda.graph`` otherwise
        # captures the current device's stream, which is GPU 0 by default.
        with torch.cuda.device(self.device):
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    decode(
                        input_ids=self.static_ids,
                        cache_position=self.static_pos,
                    )
            torch.cuda.current_stream(self.device).wait_stream(side)
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.cuda_graph, stream=torch.cuda.Stream(self.device)
            ):
                out = decode(
                    input_ids=self.static_ids, cache_position=self.static_pos
                )
        self.static_logits = out[0, -1]
        logger.info("captured the decode step as a CUDA graph")

    def step(self, token: int, position: int) -> torch.Tensor:
        """Write ``token`` at ``position`` and return the next logits.  On
        a split cache with frozen reads, a chunk ``position`` completes is
        folded after the step.

        Raises:
            RuntimeError: The model was built without its decode graph.
            NotImplementedError: The decode graph writes several tokens.
        """
        if "decode" not in self.graphs:
            raise RuntimeError("The model was built without a decode graph")
        if self.decode_tokens != 1:
            raise NotImplementedError(
                f"the decode graph writes {self.decode_tokens} tokens a step"
            )
        if self.device.type != "cuda" or not self.replay:
            logits = self.graphs["decode"](
                input_ids=torch.tensor([[token]], device=self.device),
                cache_position=torch.tensor([position], device=self.device),
            )[0, -1]
        else:
            if self.cuda_graph is None:
                self.capture(token, position)
            self.static_ids.fill_(token)
            self.static_pos.fill_(position)
            with torch.cuda.device(self.device):
                self.cuda_graph.replay()
            logits = self.static_logits
        if self.fold_step is not None:
            self.fold_step(position)
        elif self._refresh is not None:
            self.fold(position)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        eos_token_id=None,
        **unused,
    ) -> torch.Tensor:
        """Greedily continue ``input_ids`` and return prompt plus generation,
        as ``model.generate`` does.

        Args:
            input_ids: ``[1, prompt]`` token ids.
            max_new_tokens: Generation budget.
            min_new_tokens: Steps during which the stop tokens are suppressed.
            eos_token_id: Stop token or tokens; the model's by default.
            **unused: Sampling flags, which greedy decoding has no use for.

        Raises:
            ValueError: The prompt and the budget do not fit the cache.
            NotImplementedError: The decode graph writes several tokens.
        """
        if self.decode_tokens != 1:
            raise NotImplementedError(
                f"the decode graph writes {self.decode_tokens} tokens a step"
            )
        stops = self.eos_token_ids if eos_token_id is None else eos_token_id
        stops = [stops] if isinstance(stops, int) else list(stops)
        length = input_ids.shape[1]
        if length + max_new_tokens > self.cache_len:
            raise ValueError(
                f"{length} prompt + {max_new_tokens} new tokens exceed the "
                f"{self.cache_len}-slot cache"
            )
        started = time.perf_counter()
        next_logits = self.prefill(input_ids)
        # Wait for the device, so the prefill time is the prefill's.
        next_logits.max().item()
        prefilled = time.perf_counter()
        generated = []
        for step in range(max_new_tokens):
            if step < min_new_tokens:
                next_logits = next_logits.clone()
                next_logits[stops] = float("-inf")
            token = next_logits.argmax().item()
            generated.append(token)
            if token in stops or step == max_new_tokens - 1:
                break
            next_logits = self.step(token, length + step)
        self.samples += 1
        self.prefill_seconds += prefilled - started
        self.tokens += len(generated)
        self.decode_seconds += time.perf_counter() - prefilled
        if self.samples % 20 == 0:
            logger.info(
                "%d samples: prefill %.2f s each, decode %.0f ms per token "
                "over %d tokens",
                self.samples,
                self.prefill_seconds / self.samples,
                1000 * self.decode_seconds / max(self.tokens, 1),
                self.tokens,
            )
        return torch.cat(
            [input_ids, torch.tensor([generated], device=input_ids.device)], 1
        )
