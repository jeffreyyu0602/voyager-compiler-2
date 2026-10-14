"""Run a Voyager-quantized graph on a GPU with real low-precision GEMMs.

``lower()`` takes the graph ``convert_pt2e`` produces, before any Voyager
pass, and swaps each FP8 linear for ``torch._scaled_mm``, each FP8 matmul of
two activations for ``torch._scaled_grouped_mm`` and each NVFP4 linear for a
block-scaled ``F.scaled_mm``, on real FP8 and FP4 codes; every other node
keeps running as written, fake-quant included.  ``check_sites()`` checks each
swap against the converted graph on the graph's own inputs.  See README.md
for what is shared with Voyager.
"""

import operator
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.fx import GraphModule, Interpreter, Node
from torch.utils._pytree import tree_flatten

from voyager_compiler.codegen.node_info import get_arg_value, is_gemm_op
from voyager_compiler.export_utils import create_getattr_from_value
from voyager_compiler.shape_prop import fetch_attr

from . import nvfp4
from .fp8 import (
    BATCHED_GEMM_MAJORS,
    Site,
    fp8_bmm,
    match_site,
    quantize_fp8,
    site_operands,
)
from .nvfp4 import pack_nvfp4, quantize_nvfp4, unpack_nvfp4

__all__ = [
    "check_sites",
    "find_sites",
    "fp8_unsupported",
    "lower",
    "nvfp4_unsupported",
]


def _missing_kernels(device: torch.device) -> Optional[str]:
    """Why this torch build has no kernels that run on ``device``'s
    architecture, or None.  Code built for the same major and a minor no
    higher runs on it, as a cubin (``sm_XY``) or as PTX (``compute_XY``)."""
    major, minor = torch.cuda.get_device_capability(device)
    runnable = {
        f"{kind}_{major}{lower}"
        for kind in ("sm", "compute")
        for lower in range(minor + 1)
    }
    if runnable.isdisjoint(torch.cuda.get_arch_list()):
        return f"this torch build has no sm_{major}{minor} kernels"
    return None


def fp8_unsupported(device: torch.device) -> Optional[str]:
    """Why ``device`` has no FP8 GEMM on Voyager's FP8 grids, or None."""
    if device.type != "cuda":
        return f"{device} is not a GPU"
    if torch.version.hip is not None:
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if "gfx950" in arch:
            return None
        if "gfx94" in arch:
            return f"{arch}'s FP8 is E4M3FNUZ, not Voyager's fp8_e4m3 grid"
        return f"{arch} has no FP8 GEMM"
    if torch.cuda.get_device_capability(device) < (8, 9):
        return "FP8 GEMMs need compute capability 8.9 or later"
    return _missing_kernels(device)


def nvfp4_unsupported(device: torch.device) -> Optional[str]:
    """Why ``device`` has no NVFP4 GEMM, or None."""
    if device.type != "cuda":
        return f"{device} is not a GPU"
    if torch.version.hip is not None:
        return "NVFP4 GEMMs need CUDA"
    major, _ = torch.cuda.get_device_capability(device)
    if major not in nvfp4.BLOCK_SCALED_MAJORS:
        return "block-scaled GEMMs need compute capability 10.x or 12.x"
    cuda = torch.version.cuda
    if tuple(map(int, cuda.split("."))) < nvfp4.BLOCK_SCALED_CUDA:
        return f"block-scaled GEMMs need CUDA 12.8 or later, not {cuda}"
    return _missing_kernels(device)


def _unsupported(device: torch.device, kernel: Callable) -> Optional[str]:
    """Why ``device`` cannot run ``kernel``, a site's real GEMM, or None."""
    if kernel is nvfp4.NVFP4_LINEAR:
        return nvfp4_unsupported(device)
    reason = fp8_unsupported(device)
    if reason is None and kernel is fp8_bmm and torch.version.hip is None:
        major, _ = torch.cuda.get_device_capability(device)
        if major not in BATCHED_GEMM_MAJORS:
            reason = "batched FP8 GEMMs need compute capability 9.x or 10.x"
    return reason


def find_sites(
    model: GraphModule,
) -> Tuple[List[Union[Site, nvfp4.Site]], Dict[str, str]]:
    """The FP8 and NVFP4 sites of a converted graph that its GPU runs.

    Args:
        model: The converted graph.

    Returns:
        ``(sites, skipped)``: the sites in graph order, and per other GEMM
        node, by name, why it is not one or why the GPU cannot run it.
    """
    sites, skipped = [], {}
    for node in model.graph.nodes:
        if node.op != "call_function" or not is_gemm_op(node):
            continue
        if node.target is nvfp4.LINEAR_MX:
            site = nvfp4.match_site(model, node)
        else:
            site = match_site(model, node)
        if isinstance(site, str):
            skipped[node.name] = site
            continue
        # The graph's constants are on the device it runs on.
        reason = _unsupported(next(model.buffers()).device, site.kernel)
        if reason is None:
            sites.append(site)
        else:
            skipped[node.name] = reason
    return sites, skipped


def _delete_attr(model: GraphModule, target: str) -> None:
    owner, _, name = target.rpartition(".")
    delattr(model.get_submodule(owner) if owner else model, name)


def _erase_unused(model: GraphModule, nodes: List[Node]) -> None:
    """Erase each of ``nodes`` -- the operands the rewrite stopped reading --
    left with no users and no side effect, then likewise their inputs, and
    the attribute behind each constant no other node reads."""
    graph = model.graph
    pending, erased = list(nodes), set()
    while pending:
        node = pending.pop()
        if node in erased or node.users or node.is_impure():
            continue
        pending.extend(node.all_input_nodes)
        graph.erase_node(node)
        erased.add(node)
        if node.op == "get_attr":
            if not graph.find_nodes(op="get_attr", target=node.target):
                _delete_attr(model, node.target)


def _insert_fp8_gemm(
    model: GraphModule, site: Site, casts: Dict[Node, Node]
) -> Node:
    """Insert the casts of ``site``'s activations no earlier site inserted,
    and its real GEMM before its output.

    Args:
        model: The converted graph.
        site: One of its FP8 sites.
        casts: Each activation ``quantize`` cast so far -> its cast, updated.

    Returns:
        The GEMM.
    """
    graph = model.graph
    for quantize, dtype in zip(site.quantizes, site.dtypes):
        if quantize not in casts:
            with graph.inserting_after(quantize):
                casts[quantize] = graph.call_function(
                    quantize_fp8, (*quantize.args[:2], dtype)
                )
    with graph.inserting_before(site.output):
        constants = []
        for name, value in site_operands(model, site).items():
            if value is not None:
                value = create_getattr_from_value(
                    model, graph, f"{site.gemm.name}_fp8_{name}", value
                )
            constants.append(value)
        return graph.call_function(
            site.kernel,
            (*(casts[q] for q in site.quantizes), *constants),
        )


def _insert_nvfp4_gemm(
    model: GraphModule,
    site: nvfp4.Site,
    casts: Dict[Node, Tuple[Node, ...]],
) -> Node:
    """Insert the cast of ``site``'s activation unless an earlier site did,
    and its real GEMM before it.

    The cast computes the activation's global scale itself, and ``alpha``
    reads it from there.

    Args:
        model: The converted graph.
        site: One of its NVFP4 sites.
        casts: Each activation ``quantize_mx`` cast so far -> the cast's
            codes, scales and global scale, updated.

    Returns:
        The GEMM.
    """
    graph = model.graph
    quantize = site.quantizes[0]
    if quantize not in casts:
        with graph.inserting_before(quantize):
            cast = graph.call_function(quantize_nvfp4, (quantize.args[0],))
            casts[quantize] = tuple(
                graph.call_function(operator.getitem, (cast, i))
                for i in range(3)
            )
    codes, scales, global_scale = casts[quantize]
    site.alpha.replace_input_with(
        get_arg_value(quantize, 9, "global_scale"), global_scale
    )
    packed = pack_nvfp4(
        fetch_attr(model, site.weight.target),
        fetch_attr(model, site.weight_scale.target),
    )
    with graph.inserting_before(site.gemm):
        weight, weight_scale = (
            create_getattr_from_value(
                model, graph, f"{site.gemm.name}_nvfp4_{name}", value
            )
            for name, value in zip(("weight", "weight_scale"), packed)
        )
        alpha, one = nvfp4.insert_gemm_scalars(
            model, site.alpha, packed[0].device
        )
        return graph.call_function(
            site.kernel,
            (codes, scales, weight, weight_scale, alpha, one, site.bias),
        )


def lower(model: GraphModule) -> Dict[str, str]:
    """Swap each FP8 and NVFP4 GEMM of a converted graph for a real one, in
    place.

    Each activation ``quantize`` gains one cast to FP8 (``quantize_fp8``),
    and each NVFP4 ``quantize_mx`` one cast to packed codes and scales
    (``quantize_nvfp4``), that every GEMM it fed reads; each GEMM, with the
    dequantize after it, becomes one call of its site's kernel.  A linear's
    weight codes are cast to FP8, or packed, once and their bfloat16 copies
    dropped.  A quantize that also feeds other ops stays for them.

    Args:
        model: A graph ``convert_pt2e`` produced, on the GPU it runs on.

    Returns:
        Per GEMM node, by name: ``"fp8"``, ``"nvfp4"``, or why it was left
        as it is.
    """
    sites, report = find_sites(model)
    if not sites:
        return report

    graph = model.graph
    casts, unused = {}, []
    for site in sites:
        if isinstance(site, nvfp4.Site):
            gemm = _insert_nvfp4_gemm(model, site, casts)
            report[site.gemm.name] = "nvfp4"
        else:
            gemm = _insert_fp8_gemm(model, site, casts)
            report[site.gemm.name] = "fp8"
        site.output.replace_all_uses_with(gemm)
        replaced = dict.fromkeys((site.output, site.gemm))
        for node in replaced:
            unused.extend(n for n in node.all_input_nodes if n not in replaced)
            graph.erase_node(node)

    _erase_unused(model, unused)
    graph.lint()
    model.recompile()
    return report


class _SiteCheck(Interpreter):
    """Runs a converted graph and, beside each site, the real GEMM on the
    activations the site reads."""

    def __init__(
        self, model: GraphModule, sites: List[Union[Site, nvfp4.Site]]
    ):
        super().__init__(model)
        self.sites = {site.output: site for site in sites}
        self.dtypes = {
            quantize: dtype
            for site in sites
            if isinstance(site, Site)
            for quantize, dtype in zip(site.quantizes, site.dtypes)
        }
        self.nvfp4_quantizes = {
            site.quantizes[0] for site in sites if isinstance(site, nvfp4.Site)
        }
        self.readers = Counter(q for site in sites for q in site.quantizes)
        self.casts = {}
        self.mismatches = {}
        self.codes = {}
        self.results = {}

    def run_node(self, node: Node):
        result = super().run_node(node)
        self._check_cast(node, result)
        if node in self.sites:
            self._check_gemm(self.sites[node], result)
        return result

    def _check_cast(self, node: Node, result) -> None:
        """At an activation's ``quantize`` or ``quantize_mx``, cast the
        activation as the real path does, keep the cast for the GEMMs that
        read it and count the values it picks that differ from ``result``,
        the graph's."""
        if node in self.dtypes:
            x, scale = (self.env[a] for a in node.args[:2])
            cast = quantize_fp8(x, scale, self.dtypes[node])
            back = cast.to(result.dtype)
            same = (back == result) | (back.isnan() & result.isnan())
            self.casts[node] = cast
            self.mismatches[node] = int((~same).sum())
            self.codes[node] = result.numel()
        elif node in self.nvfp4_quantizes:
            codes, scales, global_scale = quantize_nvfp4(self.env[node.args[0]])
            values, block_scales = unpack_nvfp4(codes, scales)
            graph_scales, graph_codes = result
            graph_global_scale = get_arg_value(node, 9, "global_scale")
            # Each value the GEMM reads beside the graph's.
            pairs = [
                (values, graph_codes),
                (block_scales, graph_scales),
                (global_scale, self.env[graph_global_scale]),
            ]
            self.casts[node] = (codes, scales)
            self.mismatches[node] = sum(int((a != b).sum()) for a, b in pairs)
            self.codes[node] = sum(b.numel() for _, b in pairs)

    def _check_gemm(self, site: Union[Site, nvfp4.Site], result) -> None:
        """At a site's output, run its real GEMM on its activations' casts
        and record its error against ``result``, the graph's, and the code
        mismatches of those casts, which are dropped after their last
        reader."""
        quantizes = site.quantizes
        if isinstance(site, nvfp4.Site):
            weight = pack_nvfp4(
                self.env[site.weight], self.env[site.weight_scale]
            )
            alpha = self.env[site.alpha].float()
            bias = None if site.bias is None else self.env[site.bias]
            real = site.kernel(
                *self.casts[quantizes[0]],
                *weight,
                alpha,
                torch.ones_like(alpha),
                bias,
            )
        else:
            real = site.kernel(
                *(self.casts[q] for q in quantizes),
                *site_operands(self.module, site).values(),
            )
        reference = result.float()
        error = (real.float() - reference).abs().max()
        self.results[site.gemm.name] = {
            "code_mismatches": sum(self.mismatches[q] for q in quantizes),
            "codes": sum(self.codes[q] for q in quantizes),
            "error": float(error / reference.abs().max().clamp_min(1e-30)),
        }
        for quantize in quantizes:
            self.readers[quantize] -= 1
            if not self.readers[quantize]:
                del self.casts[quantize]


def check_sites(model: GraphModule, args, kwargs=None):
    """Run a converted graph and check each site's real GEMM against it.

    At each site the real path -- ``quantize_fp8`` or ``quantize_nvfp4`` on
    each activation, then the site's kernel -- runs on the activations the
    converted graph's ``quantize`` and ``quantize_mx`` nodes read, so a
    site's error is its own, not carried in from earlier sites.

    Args:
        model: The converted graph, before ``lower``.
        args: Positional inputs.
        kwargs: Keyword inputs; None for none.

    Returns:
        ``(results, output)``: per site, by GEMM name, how many of the values
        its activations' casts pick -- FP8 codes; NVFP4 codes, block scales
        and global scales -- differ from the graph's (``code_mismatches``,
        out of ``codes``) and the largest output difference over the largest
        output magnitude (``error``); and the graph's output.
    """
    sites, _ = find_sites(model)
    checker = _SiteCheck(model, sites)
    flat_args, _ = tree_flatten((args, kwargs or {}))
    output = checker.run(*flat_args)
    return checker.results, output
