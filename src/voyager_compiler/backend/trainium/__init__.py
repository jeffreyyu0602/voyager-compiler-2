"""Voyager's schedule on AWS Trainium.

The shared compiler tiles, bufferizes and plans a model for one NeuronCore
(``TrainiumConfig``, priced by ``TrainiumModel``), and the translator turns
the ``model.txt`` it writes into NKI kernels without deciding anything.
"""

from .config import LAYOUT, TRN2, TRN3, TrainiumConfig
from .model import TrainiumModel, kernel_nodes, predicted_seconds
from .translate import immediates, record, translate

__all__ = [
    "LAYOUT",
    "TRN2",
    "TRN3",
    "TrainiumConfig",
    "TrainiumModel",
    "immediates",
    "kernel_nodes",
    "predicted_seconds",
    "record",
    "translate",
]
