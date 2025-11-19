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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Seq_Len, Batch, d_model)
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    """Transformer architecture for time series."""
    def __init__(
        self, 
        feature_size: int = 1, 
        d_model: int = 64, 
        num_layers: int = 1, 
        dropout: float = 0.1, 
        nhead: int = 4, 
        num_classes: int = 1
    ):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        # 1. Projection d'entrée : on passe de 1 feature (rating) à d_model (ex: 64)
        # C'est crucial pour que le Transformer "comprenne" la donnée sans exploser les calculs
        self.input_embedding = nn.Linear(feature_size, d_model)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 3. Decoder final
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape arrive comme : (Batch, Seq, Feature=1)
        
        # Permutation pour le Transformer PyTorch : (Seq, Batch, Feature)
        src = src.permute(1, 0, 2) 
        
        # Projection vers d_model
        src = self.input_embedding(src)
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        # On prend le dernier état temporel pour la prédiction
        output = output[-1, :, :] # (Batch, d_model)
        
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerModel(BaseModel):
    """Transformer model wrapper."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        nhead: int = 4,
        d_model: int = 64, # Nouveau paramètre important
        device: str = "cuda",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.nhead = nhead
        self.d_model = d_model
        self.device = device
        
        # Appel de la méthode build
        self.model = self.build()

    def build(self) -> TransAm:
        """Build Transformer model."""
        return TransAm(
            feature_size=self.input_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            dropout=self.dropout,
            nhead=self.nhead,
            num_classes=self.output_size
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