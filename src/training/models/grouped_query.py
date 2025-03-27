"""Module that manages the grouped query attention module"""

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from torchtune.modules import MultiHeadAttention


class GroupedQuery(nn.Module):
    """Grouped query attention module."""

    def __init__(self, **config: dict):
        """Initialize torch layers"""

        super().__init__()

        # Extract the parameters from the config
        self.n_embd: int = config['n_embedding']
        self.bias: bool = config['bias']
        self.n_head: int = config['n_head']
        self.dropout: float = config['dropout']
        self.kv_heads: int = config['kv_heads']
        self.block_size: int = config['block_size']

        self.kv_factor = self.n_head // self.kv_heads
        self.kv_factor = self.n_embd // self.kv_factor

        # Initialize linear projections
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.init_weight_normalization_flag = True
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.k_proj = nn.Linear(self.n_embd, self.kv_factor, bias=self.bias)
        self.v_proj = nn.Linear(self.n_embd, self.kv_factor, bias=self.bias)

        # Initialize the rotary embedding and grouped attention
        self.rotary_emb = RotaryEmbedding(dim=self.n_embd // self.n_head)
        self.query_attention = MultiHeadAttention(
            embed_dim=self.n_head,
            num_heads=self.n_head,
            num_kv_heads=self.kv_heads,
            head_dim=self.n_embd // self.n_head,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            output_proj=self.c_proj,
            attn_dropout = self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the grouped query attention."""

        # batch size, sequence length, and embedding dim
        B, T, C = x.shape

        # re-assemble all head outputs side by side
        y = self.query_attention(x, x, )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y
