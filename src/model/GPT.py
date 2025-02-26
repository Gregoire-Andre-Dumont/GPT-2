import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers=6, n_heads=6, dropout = 0.2):
        super().__init__()
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        # Transformer encoder as a stack of standard encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd ,nhead=n_heads,dropout = dropout,norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final linear layer to project hidden states to vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Optional: move eval logging outside of forward; here we just keep a counter
        self.eval_counter = 0

    def forward(self, idx, targets=None):
      B, T = idx.shape

      token_emb = self.token_embedding(idx)  # (B, T, n_embd)
      pos_ids = torch.arange(T, device=idx.device)
      pos_emb = self.position_embedding(pos_ids)  # (T, n_embd)
      x = token_emb + pos_emb  # (B, T, n_embd)

      x = x.transpose(0, 1)  # (T, B, n_embd)

      # Create a causal mask: True for positions that should be masked
      mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
      x = self.transformer(x, mask=mask)

      x = x.transpose(0, 1)  # Back to (B, T, n_embd)
      logits = self.lm_head(x)  # (B, T, vocab_size)

      loss = None
      if targets is not None:
          logits_flat = logits.reshape(-1, logits.size(-1))
          targets_flat = targets.reshape(-1)
          loss = F.cross_entropy(logits_flat, targets_flat)
          self.eval_counter += 1

      return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size=None):
        # Optionally, set block_size to the model's maximum context length if not provided.
        block_size = block_size or self.position_embedding.num_embeddings
        self.eval()  # Ensure dropout and other train-specific layers are in eval mode
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx