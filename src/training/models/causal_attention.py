"""Module that manages the self-attention with rotary embeddings."""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

class CausalAttention(nn.Module):
    """Causal self-attention with rotary embeddings."""

    def __init__(self, **config: dict):
        """Initialize torch layers"""

        super().__init__()

        # Extract the parameters from the config
        self.n_embed: int = config['n_embedding']
        bias: bool = config['bias']
        self.n_head: int = config['n_head']
        self.dropout: float = config['dropout']

        # Initialize the linear projections
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=bias)
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=bias)

        # Initialize the regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Initialize the rotary positional embedding
        self.rotary_emb = RotaryEmbedding(dim=self.n_embed // self.n_head)

        # Prevent standard deviation creep
        self.init_weight_normalization_flag = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the attention module."""

        B, T, C = x.size()

        # Compute the query, key and values for all heads in the batch
        query, key, value = self.c_attn(x).split(self.n_embed, dim=2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        query = self.rotary_emb.rotate_queries_or_keys(query)
        key = self.rotary_emb.rotate_queries_or_keys(key)

        # Manual implementation of the attention module
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # Normalization
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of the interesting tokens
        y = att @ value
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        return self.resid_dropout(self.c_proj(y))
