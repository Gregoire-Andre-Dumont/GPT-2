"""Main torch dataset for pre-training the GPT-2 model."""

import tiktoken
import warnings
import torch
from torch import Tensor
import random
from dataclasses import dataclass
from torch.utils.data import Dataset
from src.training.augments.base_augment import BaseAugment

# Suppress specific warning
warnings.filterwarnings("ignore", message=".*indexing errors.*")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MainDataset(Dataset):
    """Main torch dataset for pre-training GPT-2.

    :param text_data: string containing the text data.
    :param augment: whether to use data augmentation.
    :param p_bert: probability of augmenting a token.
    :param p_augment: probability of augmenting a sequence.
    :param block_size: number of tokens in each sample."""

    text_data: str | None = None
    augment: bool | None = None
    p_bert: float | None = None
    p_augment: float | None = None
    block_size : int | None = None
    augmentation: BaseAugment | None = None

    def initialize(self, text: str, augment: bool):
        """Initialize the dataloader with the text."""

        # Load the necessary tokenizers and models
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.tokenized_data = self.tokenizer.encode(text)

        # Augment the data if necessary
        if augment:
            self.augmented = self.augmentation.augment_main(text)
            self.augmented = self.tokenizer.encode(self.augmented)


    def __len__(self) -> int:
        """Return the length of the dataset."""

        return len(self.tokenized_data) // self.block_size - 1

    def __getitem__(self, index: int) -> tuple:
        """Extract the sequence and the target."""

        # Randomly extract a section of the text
        start_idx = index * self.block_size
        start_idx = start_idx + random.randint(0, self.block_size - 1)

        # Extract the input and target sequences
        input = self.tokenized_data[start_idx: start_idx + self.block_size]
        target = self.tokenized_data[start_idx + 1:start_idx + self.block_size + 1]

        return Tensor(input), Tensor(target)