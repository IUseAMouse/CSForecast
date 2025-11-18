"""
LSTM model for time series forecasting.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from .base import BaseModel


class LSTM(nn.Module):
    """LSTM network architecture."""

    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        device: str = "cuda",
    ):
        """
        Initialize LSTM model.

        Args:
            num_classes: Output dimension
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            device: Device to run the model on
        """
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        out_logits = self.fc(h_out)

        return out_logits


class LSTMModel(BaseModel):
    """LSTM model wrapper."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Input sequence length
            hidden_size: Hidden dimension
            output_size: Output sequence length
            num_layers: Number of LSTM layers
            device: Device to run on
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.model = self.build()

    def build(self) -> LSTM:
        """Build LSTM model."""
        return LSTM(
            num_classes=self.output_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
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