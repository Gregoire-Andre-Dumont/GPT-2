import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from src.training.models.gpt_block import GPTBlock
from src.config.gpt_config import GPTConfig


def get_positional_encoding(n_positions, n_embd):
    def angle_defn(pos, i, d_model_size):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model_size))
        return pos * angle_rates

    angle_rads = angle_defn(np.arange(n_positions)[:, np.newaxis], np.arange(n_embd)[np.newaxis, :], n_embd)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = torch.tensor(np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...], dtype=torch.float)
    return pos_encoding


class GPT2Model(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super(GPT2Model, self).__init__()


        self.wte = nn.Embedding(cfg.n_tokens, cfg.n_embeddings, padding_idx=0).half()
        self.register_buffer('positional_encoding', get_positional_encoding(cfg.n_positions, cfg.n_embd))
        self.blocks = nn.ModuleList([GPTBlock(self.n_embd, self.n_head) for _ in range(self.n_layer)])

        self.decoder_norm = nn.LayerNorm(self.n_embd)
        self.decoder = nn.Linear(self.n_embd, self.n_tokens, bias=False)
        kaiming_normal_(self.decoder.weight, nonlinearity="linear")

    def forward(self, input_ids):
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.positional_encoding[:, :input_ids.size(1), :]
        hidden_states = (inputs_embeds + position_embeds).float()

        for block in self.blocks:
            hidden_states = block(hidden_states)

        decoded = self.decoder(self.decoder_norm(hidden_states))
        return F.log_softmax(decoded, dim=-1)