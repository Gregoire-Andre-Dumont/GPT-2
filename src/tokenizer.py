"""Module that manages pre-trained tokenizers from hugging face."""

from dataclasses import dataclass
from typing import Any
from transformers import AutoTokenizer
import numpy.typing as npt

@dataclass
class Tokenizer:
    """pre-trained tokenizers from hugging face.

    :param token: hugging face class
    :param token_name: name of the tokenizer."""

    token: Any | None = None
    token_name: str | None = None

    def __post_init__(self):
        """Load the tokenizer from hugging face."""

        self.tokenizer = self.token.from_pretrained(self.token_name)

    def encode(self, prompt: str) -> npt.NDArray:
        """Encode the input prompt into token ids."""

        return self.tokenizer(prompt, return_tensors="pt").input_ids

    def decode(self, generated_ids: npt.NDArray) -> str:
        """Decode the generated ids into a string."""

        return self.tokenizer.decode(generated_ids)