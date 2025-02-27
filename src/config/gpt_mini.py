"""Module that manages the configuration of GPT."""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Manages the configuration of GPT."""
    
    vocab_size: int = 1500
    n_embd: int = 300