"""
Transformer model for time series forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle

from .base import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Gère le cas où d_model est impair
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    """Transformer architecture for time series."""

    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        feature_size: int = 250,
        dropout: float = 0.2,
        nhead: int = 10,
    ):
        """
        Initialize Transformer model.

        Args:
            num_layers: Number of transformer encoder layers
            num_classes: Output dimension
            feature_size: Feature dimension
            dropout: Dropout rate
            nhead: Number of attention heads
        """
        super(TransAm, self).__init__()
        
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(nn.Linear(int(feature_size), num_classes))
        
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for layer in self.decoder:
            layer.bias.data.zero_()
            layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Input tensor

        Returns:
            Output predictions
        """
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = torch.sum(output, dim=1)
        output = self.decoder(output)

        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate attention mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


class TransformerModel(BaseModel):
    """Transformer model wrapper."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        nhead: int = 10,
        device: str = "cuda",
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Input feature size
            output_size: Output dimension
            num_layers: Number of transformer layers
            dropout: Dropout rate
            nhead: Number of attention heads
            device: Device to run on
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.nhead = nhead
        self.device = device
        self.model = self.build()

    def build(self) -> TransAm:
        """Build Transformer model."""
        return TransAm(
            num_layers=self.num_layers,
            num_classes=self.output_size,
            feature_size=self.input_size,
            dropout=self.dropout,
            nhead=self.nhead,
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)