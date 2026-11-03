"""Two programs that write one buffer keep it at one DRAM address.

A decode graph is planned alone; a prefill graph then places each tensor
it shares with decode -- matched by tensor identity, never by name -- at
decode's address (``planned_buffer_layout``), while its own tensors and
its activation arena land elsewhere.
"""

import pytest
import torch
from google.protobuf import text_format
from torch import nn

import voyager_compiler
from voyager_compiler import export_model, planned_buffer_layout
from voyager_compiler.codegen import voyager_ir_pb2
from voyager_compiler.hardware_config import (
    DEFAULT_DRAM_ACCESS_LATENCY_NS,
    DEFAULT_DRAM_BANDWIDTH_GBS,
    AcceleratorConfig,
)

GRAD_ENABLED = False

CONFIG = AcceleratorConfig(
    pe_array_size=(16, 16),
    input_buffer_size=1024,
    weight_buffer_size=1024,
    accum_buffer_size=1024,
    scratchpad_size=1 << 22,
    scratchpad_banks=4,
    scratchpad_word_bytes=64,
    dram_size=16.0,
    dram_bandwidth=DEFAULT_DRAM_BANDWIDTH_GBS,
    dram_access_latency=DEFAULT_DRAM_ACCESS_LATENCY_NS,
)


class Writer(nn.Module):
    """A linear whose rows are written into a cache buffer, registered
    under ``cache_name``, plus a private constant named ``qmap``."""

    def __init__(self, cache_name, cache, qmap):
        super().__init__()
        self.cache_name = cache_name
        self.register_buffer(cache_name, cache)
        self.register_buffer("qmap", qmap)
        self.fc = nn.Linear(64, 64, bias=False)

    def forward(self, x, pos):
        y = self.fc(x)
        getattr(self, self.cache_name).index_copy_(0, pos, y)
        return y + self.qmap


def transformed(cache_name, cache, qmap, rows):
    """A ``Writer`` exported on ``rows`` rows and lowered by ``transform``,
    with its example inputs."""
    torch.manual_seed(rows)
    model = Writer(cache_name, cache, qmap).bfloat16()
    args = (torch.randn(rows, 64).bfloat16(), torch.arange(rows))
    graph = export_model(model, args)
    voyager_compiler.transform(graph, args, config=CONFIG, keep_fp32=False)
    return graph, args


def dram_ranges(graph):
    """``{node name: (start, end)}`` of every DRAM buffer root planned."""
    return {
        n.name: (n.meta["memory"].start, n.meta["memory"].end)
        for n in graph.graph.nodes
        if "memory" in n.meta
    }


def box_addresses(path):
    """``{node: address}`` of the parameters in a ``model.txt``."""
    ir = text_format.Parse(open(path).read(), voyager_ir_pb2.Model())
    return {box.node: box.memory.address for box in ir.parameters}


def test_shared_buffer_keeps_decode_address(tmp_path):
    cache = torch.zeros(128, 64, dtype=torch.bfloat16)
    decode, decode_args = transformed(
        "cache", cache, torch.ones(64, dtype=torch.bfloat16), 32
    )
    prefill, prefill_args = transformed(
        "kv", cache, torch.full((64,), 2.0, dtype=torch.bfloat16), 64
    )
    voyager_compiler.compile(
        decode,
        decode_args,
        config=CONFIG,
        output_dir=str(tmp_path / "decode"),
        dump_tensors=False,
    )
    before = dram_ranges(decode)

    layout = planned_buffer_layout(decode, prefill, CONFIG)
    # The cache is matched by identity under another name; the constant
    # each program holds a copy of under one name, and the weight, are not.
    assert layout == {"kv": before["cache"]}

    voyager_compiler.compile(
        prefill,
        prefill_args,
        config=CONFIG,
        output_dir=str(tmp_path / "prefill"),
        dump_tensors=False,
        dram_layout=layout,
    )
    after = dram_ranges(prefill)
    start, end = before["cache"]
    assert after["kv"] == (start, end)
    for name, (lo, hi) in after.items():
        if name != "kv":
            assert hi <= start or end <= lo, (name, (lo, hi))
    assert dram_ranges(decode) == before

    decode_boxes = box_addresses(tmp_path / "decode" / "model.txt")
    prefill_boxes = box_addresses(tmp_path / "prefill" / "model.txt")
    assert prefill_boxes["kv"] == decode_boxes["cache"] == start
    assert prefill_boxes["qmap"] != decode_boxes["qmap"]


def test_shared_buffer_sized_differently_raises(tmp_path):
    cache = torch.zeros(128, 64, dtype=torch.bfloat16)
    decode, decode_args = transformed(
        "cache", cache, torch.ones(64, dtype=torch.bfloat16), 32
    )
    voyager_compiler.compile(
        decode,
        decode_args,
        config=CONFIG,
        output_dir=str(tmp_path / "decode"),
        dump_tensors=False,
    )
    prefill, _ = transformed(
        "cache", cache, torch.ones(64, dtype=torch.bfloat16), 64
    )
    # Prefill holds the cache at a narrower logical dtype than decode.
    for node in prefill.graph.find_nodes(op="get_attr"):
        if node.target == "cache":
            node.meta["dtype"] = torch.int8
    with pytest.raises(ValueError, match="cache"):
        planned_buffer_layout(decode, prefill, CONFIG)
