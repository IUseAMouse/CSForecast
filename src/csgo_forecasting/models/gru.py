"""
GRU model for time series forecasting.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from .base import BaseModel


class GRU(nn.Module):
    """GRU network architecture."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        bias: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize GRU model.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_classes: Output dimension
            num_layers: Number of GRU layers
            bias: Whether to use bias
            device: Device to run the model on
        """
        super(GRU, self).__init__()
        
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.device = device

        self.gru_cells = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        ).to(device)
        
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        ).to(self.device)

        _, h_out = self.gru_cells(x, h0)
        out = self.fc(h_out)
        
        return out[-1]  # Only keep the state from last layer


class GRUModel(BaseModel):
    """GRU model wrapper."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 5,
        device: str = "cuda",
    ):
        """
        Initialize GRU model.

        Args:
            input_size: Input sequence length
            hidden_size: Hidden dimension
            output_size: Output sequence length
            num_layers: Number of GRU layers
            device: Device to run on
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.model = self.build()

    def build(self) -> GRU:
        """Build GRU model."""
        return GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_classes=self.output_size,
            num_layers=self.num_layers,
            device=self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)[0]

    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)