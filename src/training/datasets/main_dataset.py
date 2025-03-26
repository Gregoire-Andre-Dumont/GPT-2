"""Main torch dataset for pre-training the GPT-2 model."""

from torch import Tensor
import random
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from transformers import BertForMaskedLM, BertTokenizer

@dataclass
class MainDataset(Dataset):
    """Main torch dataset for pre-training GPT-2.

    :param text_data: string containing the text data.
    :param augment: whether to use data augmentation.
    :param p_bert: probability of bert data augmentation.
    :param block_size: number of tokens in each sample."""

    text_data: str | None = None
    augment: bool | None = None
    p_bert: float | None = None
    block_size : int | None = None

    def initialize(self, text: str, augment: bool):
        """Initialize the dataloader with the text."""

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_model = BertForMaskedLM.from_pretrained("bert-large-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

        self.text = self.augment_data(text) if augment else text
        self.text_data = self.tokenizer(self.text)


    def augment_data(self, text: str) -> str:
        """Augment the text using masked language modeling."""













@dataclass
class MainDatasetss(Dataset):
    """Main torch dataset for language models.

    :param data: processed tokens with shape (1, length)
    :param data_augment: Whether to apply data augmentation."""

    data: npt.NDArray[np.uint64] | None = None
    data_augment: bool | None = None
    block_size: int = 512

    def __len__(self) -> int:
        """Return the length of the dataset."""

        return self.data.shape[0] // self.block_size - 1

    def __getitem__(self, index: int) -> tuple:
        """Extract the sequence and the target."""

        start_index = index * self.block_size
        idx = start_index + random.randint(0, self.block_size - 1)
        input = self.data[idx: idx + self.block_size]
        target = self.data[idx+1:idx + self.block_size+1]

        return Tensor(input), Tensor(target)