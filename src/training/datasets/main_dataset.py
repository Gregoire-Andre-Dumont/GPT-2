"""Main torch dataset for pre-training the GPT-2 model."""

import warnings
import torch
from torch import Tensor
import random
import numpy as np
from dataclasses import dataclass
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm

# Suppress specific warning
warnings.filterwarnings("ignore", message=".*indexing errors.*")

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

    def initialize(self, text: str, augment: bool):
        """Initialize the dataloader with the text."""

        # Load the necessary tokenizers and models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.bert_model = BertForMaskedLM.from_pretrained("bert-large-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

        # Encode and save the text
        self.text_data = text
        self.augment = augment
        self.tokenized_data = self.tokenizer.encode(text, add_special_tokens=True)

    def data_augment(self, text: str) -> str:
        """Data augmentation with mask language modeling."""

        tokenized_text = np.array(text)

        # Randomly mask tokens in the text
        mask = np.random.rand(len(tokenized_text)) < self.p_bert
        masked_text = np.where(mask, "[MASK]", tokenized_text).tolist()
        mask_indices = np.where(mask)[0].tolist()

        # Convert to BERT input format
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(masked_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Predict masked tokens
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor)
            predictions = outputs[0]

        # Extract predicted token indices for all masked positions at once
        predicted_indices = torch.argmax(predictions[0, mask_indices], dim=1).tolist()
        predicted_tokens = self.bert_tokenizer.convert_ids_to_tokens(predicted_indices)

        # Replace MASK tokens with predictions
        augmented_tokens = tokenized_text.copy()
        for idx, mask_pos in enumerate(mask_indices):
            augmented_tokens[mask_pos] = predicted_tokens[idx]

        # Convert back to text
        return self.bert_tokenizer.convert_tokens_to_string(augmented_tokens)

    def pad_sequence(self, sequence: list[int]):
        """Pad the token sequence to the block size."""

        token_id = self.tokenizer.pad_token_id
        padding_size = self.block_size - len(sequence)
        return sequence + [token_id] * padding_size

    def __len__(self) -> int:
        """Return the length of the dataset."""

        return len(self.tokenized_data) // self.block_size - 1

    def __getitem__(self, index: int) -> tuple:
        """Extract the sequence and the target."""

        # Randomly extract a section of the text
        start_index = index * self.block_size
        idx = start_index + random.randint(0, self.block_size - 1)

        # Extract the input and target sequences
        input = self.tokenized_data[idx: idx + self.block_size]
        target = self.tokenized_data[idx + 1:idx + self.block_size + 1]

        # Check whether we have to perform augmentation
        if random.random() < self.p_augment and self.augment:
            sequence = self.tokenizer.decode(input)
            sequence = self.bert_tokenizer.tokenize(sequence)

            # Augment the input with BERT
            augmented = self.data_augment(sequence[:512])
            augmented = self.tokenizer.encode(sequence, truncation=True, max_length=512)

            # Extract the input and output
            input = self.pad_sequence(augmented[:-1])
            target = self.pad_sequence(augmented[1:])


        return Tensor(input), Tensor(target)
