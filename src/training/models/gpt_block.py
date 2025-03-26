"""Module that manages the GPT-2 transformer block"""

import torch.nn as nn
from torch import Tensor
from src.training.models.grouped_query import GroupedQuery
from src.training.models.causal_attention import CausalAttention

class GPTBlock(nn.Module):
    def __init__(self, **config: dict):
        """Initialize the torch layers."""

        super().__init__()

        # Extract the parameters from the config
        self.embed: int = config['n_embedding']
        self.bias: bool = config['bias']
        self.n_head: int = config['n_head']
        self.dropout: float = config['dropout']
        self.grouped_query: bool = config['grouped']

        # Initialize layer normalization and attention layer
        self.norm_1 = nn.LayerNorm(self.embed, bias=self.bias)
        self.norm_2 = nn.LayerNorm(self.embed, bias=self.bias)

        if self.grouped_query:
            self.attention = GroupedQuery(**config)
        else:
            self.attention = CausalAttention(**config)

        # Initialize the layers in the feedforward network
        self.fully = nn.Linear(self.embed, 4 * self.embed, bias=self.bias)
        self.project = nn.Linear(4 * self.embed, self.embed, bias=self.bias)

        self.dropout = nn.Dropout(self.dropout)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the transformer block."""

        x = self.norm_1(x)
        z = x + self.attention(x, x, x)[0]
        x = self.fully(self.norm_2(z))
        x = self.project(self.gelu(x))
        return z + self.dropout(x)
