"""
Base model class for all forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None

    @abstractmethod
    def build(self) -> Any:
        """Build the model architecture."""
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model."""
        pass

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        if isinstance(self.model, nn.Module):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0