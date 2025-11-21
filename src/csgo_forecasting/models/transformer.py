import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import math
from typing import Optional, Tuple

from .base import BaseModel


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes the input to mean 0 and std 1 per instance, applies the model,
    and denormalizes the output. Helps significantly with non-stationary data.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: (Batch, n_head, seq_len, head_dim)
        # Only return elements corresponding to current sequence
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (Batch, n_head, seq_len, head_dim)
    # cos, sin: (1, 1, seq_len, head_dim) 
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, max_len: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.n_head = nhead
        self.head_dim = d_model // nhead
        
        self.c_attn = nn.Linear(d_model, 3 * d_model) 
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.size() # Batch, Time, Channels
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # (B, T, n_head, head_dim) -> transpose -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Attention Scaled Dot Product
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble heads
        
        return self.resid_dropout(self.c_proj(y))

class TransformerBlock(nn.Module):
    """Standart Transformer Block: Pre-LayerNorm, Attention, FeedForward."""
    def __init__(self, d_model: int, nhead: int, dropout: float, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-Norm architecture 
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransAm(nn.Module):
    def __init__(
        self, 
        feature_size: int = 1, 
        d_model: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        nhead: int = 4, 
        num_classes: int = 1,
        max_len: int = 5000
    ):
        super(TransAm, self).__init__()
        self.model_type = 'TransformerRoPE_RevIN'
        
        self.revin = RevIN(feature_size)
        self.input_embedding = nn.Linear(feature_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dropout, max_len)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        std = 0.02
        nn.init.normal_(self.input_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=std)
        nn.init.zeros_(self.decoder.bias)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Shape is (Batch, Sequence, d_model)
        src = self.revin(src, mode='norm')
        
        x = self.input_embedding(src)
        x = self.dropout(x)
        
        B, T, C = x.shape
        mask = self._generate_square_subsequent_mask(T, x.device)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        
        output = x[:, -1, :] # (Batch, d_model)
        output = self.decoder(output) # (Batch, Output_Dim)
        
        output = output.unsqueeze(1) 
        output = self.revin(output, mode='denorm')
        output = output.squeeze(1)
        
        return output


class TransformerModel(BaseModel):
    """Transformer model wrapper updated with RoPE and RevIN."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 2, # Augmenté un peu car le modèle est plus robuste
        dropout: float = 0.2,
        nhead: int = 4,
        d_model: int = 64,
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
        
        self.model = self.build()

    def build(self) -> TransAm:
        """Build Transformer model."""
        return TransAm(
            feature_size=self.input_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            dropout=self.dropout,
            nhead=self.nhead,
            num_classes=self.output_size,
            max_len=5000 # Buffer pour RoPE
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Le modèle attend (Batch, Seq, Feature)
        # Si l'entrée est (Batch, Feature, Seq), il faut permuter.
        # Assumons que le DataLoader envoie (Batch, Seq, Feature) comme discuté précédemment.
        return self.model(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 1e-3) -> None:
        # Implémentation simple d'un fit loop pour l'exemple, 
        # à adapter selon ta classe BaseModel originale
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        self.model.train()
        X = X.to(self.device)
        y = y.to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)