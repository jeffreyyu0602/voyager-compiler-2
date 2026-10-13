"""Voyager's runtime model: what the shared tile search asks about a Voyager
accelerator -- whether a tile fits its banked scratchpad and what a tiling
costs -- answered from its ``AcceleratorConfig`` (``VoyagerModel``).
"""

from .model import VoyagerModel

__all__ = ["VoyagerModel"]
