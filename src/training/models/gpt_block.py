"""Module that manages the GPT-2 transformer block"""


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from src.training.models.attention import MultiHeadAttention

class MLP(nn.Module):
    def __init__(self, embed_dim, factor=4):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, embed_dim * factor)
        self.fc2 = nn.Linear(embed_dim * factor, embed_dim)

        kaiming_normal_(self.fc.weight, nonlinearity="relu")
        kaiming_normal_(self.fc2.weight, nonlinearity="linear")

        self.fc = self.fc.half()
        self.fc2 = self.fc2.half()

    def forward(self, x):
        h = F.gelu(self.fc(x.half()).float())
        return self.fc2(h.half())


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GPTBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x