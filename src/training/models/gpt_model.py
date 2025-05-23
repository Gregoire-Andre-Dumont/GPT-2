"""Module that manages the GPT-2 torch model."""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.training.models.gpt_block import GPTBlock

class GPT(nn.Module):
    def __init__(self, **config: dict):
        """Initialize the torch layers."""

        super().__init__()

        # Extract the parameters from the config
        self.n_embed: int = config['n_embedding']
        self.bias: bool = config['bias']
        self.n_head: int = config['n_head']
        self.dropout: float = config['dropout']

        self.block_size: int = config['block_size']
        self.vocab_size: int = config['vocab_size']
        self.n_layer: int = config['n_layer']

        # Initialize the embedding layers
        self.wte = nn.Embedding(self.vocab_size, self.n_embed)
        self.wpe = nn.Embedding(self.block_size, self.n_embed)

        # Initialize the dropout and layer normalization
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.n_embed, bias=self.bias)

        # Initialize the GPT-2 blocks and head
        self.blocks = nn.ModuleList([GPTBlock(**config) for _ in range(self.n_layer)])
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        # Initialize all the weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def _init_weights(self, module):
        """Initialize the weights of each layer."""

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None):
        """Forward pass of the GPT-2 model."""

        # Extract the batch and sequence size
        batch, length = inputs.size()

        # Check whether the sequence size is valid
        assert length <= self.block_size, f"Cannot forward sequence of length {length}"
        pos = torch.arange(0, length, dtype=torch.long, device=inputs.device)

        # Compute the token and position embeddings
        tok_embeddings = self.wte(inputs)
        pos_embeddings = self.wpe(pos)
        outputs = self.drop(tok_embeddings + pos_embeddings)

        # Perform the transformer blocks
        for block in self.blocks:
            outputs = block(outputs)
        x = self.norm(outputs)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            return logits, loss

        # Using -1 to preserve the time dimension
        return self.lm_head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx