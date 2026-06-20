"""Dataset implementations and the shared base class."""

from .base import BaseDataset, Example
from .toy import ToyDataset

__all__ = ["BaseDataset", "Example", "ToyDataset"]
