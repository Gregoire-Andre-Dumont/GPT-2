"""Module that manages the attention layers."""

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_dim, drop=0.1):
        super().__init__()
        self.temperature = np.power(key_dim, 0.5)
        self.dropout = nn.Dropout(drop)

    def forward(self, q, k, v):
        energies = (torch.bmm(q, k.transpose(1, 2))) / self.temperature

        seq_len = energies.size(1)
        mask = (torch.tril(torch.ones(seq_len, seq_len)) == 0).to(energies.device)
        energies.masked_fill_(mask, -np.inf)

        attention = self.dropout(F.softmax(energies, dim=2))
        context = torch.bmm(attention, v).squeeze(1)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = num_heads
        self.attn = ScaledDotProductAttention(embed_dim)

        self.query_trans = nn.Linear(embed_dim, embed_dim)
        self.keys_trans = nn.Linear(embed_dim, embed_dim)
        self.value_trans = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        kaiming_normal_(self.query_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.keys_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.value_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.projection.weight, nonlinearity="linear")

        self.keys_trans = self.keys_trans.half()
        self.value_trans = self.value_trans.half()
        self.projection = self.projection.half()

    def split_heads(self, x):
        bs, seq_len, _ = x.size()
        # result: (head*batch_size) x seq_len x new features
        return x.view(bs, seq_len, self.n_heads, -1).permute(2, 0, 1, 3).reshape(self.n_heads * bs, seq_len, -1)

    def merge_heads(self, x):
        _, seq_len, features_size = x.size()
        x = x.view(self.n_heads, -1, seq_len, features_size)
        bs = x.size(1)
        # batch_size x seq_len x heads x features
        return x.permute(1, 2, 0, 3).reshape(bs, seq_len, -1)

    def forward(self, x):
        q = self.split_heads(self.query_trans(x))
        k = self.split_heads(self.keys_trans(x.half()).float())
        v = self.split_heads(self.value_trans(x.half()).float())
        a, _ = self.attn(q, k, v)
        a = self.merge_heads(a)
        return self.projection(a.half())