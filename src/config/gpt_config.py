"""Module that manages the parameters of the GPT-2 model."""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Manage the parameters of the GPT-2 model."""

    n_layers: int = 12
    n_embeddings: int = 768 + 128
    n_head: int = 12 + 2
    n_tokens: int = 512
    n_positions: int = 512