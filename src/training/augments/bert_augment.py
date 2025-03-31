"""Module that manages the data augmentation with BERT."""

import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from transformers import BertForMaskedLM, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class BertAugment:
    """Data augmentation with mask language modeling.
    :param p_bert: probability of augmenting tokens."""

    p_bert: float = 0.0
    block_size: int = 512

    def __init__(self):
        """Load Bert model and tokenizer."""

        self.model = BertForMaskedLM.from_pretrained("bert-large-uncased").to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    def mask_text(self, text: str):
        """Randomly mask tokens in the text."""

        # Process the text using the tokenizer
        sequence = np.array(self.tokenizer.tokenize(text))

        # Randomly mask tokens in the text
        mask = np.random.rand(len(sequence)) < self.p_bert
        masked_text = np.where(mask, "[MASK]", sequence)
        masked_indices = np.where(mask)[0]

        return masked_text, masked_indices


    def augment_text(self, text: str) -> str:
        """Augment the text with mask language modeling."""

        # Randomly mask tokens in the text
        masked_text, masked_indices = self.mask_text(text)

        # Convert to BERT input format
        tokens = self.tokenizer.convert_tokens_to_ids(masked_text)
        tokens = torch.tensor([tokens]).to(device)

        # Predict masked tokens
        with torch.no_grad():
            outputs = self.model(tokens)
            predictions = outputs[0]

        # Extract predicted token indices for all masked positions at once
        predicted_indices = torch.argmax(predictions[0, masked_indices], dim=1)
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices)

        # Replace MASK tokens with predictions
        augmented_tokens = tokens.copy()
        for idx, mask_pos in enumerate(masked_indices):
            augmented_tokens[mask_pos] = predicted_tokens[idx]

        # Convert back to text
        return self.tokenizer.convert_tokens_to_string(augmented_tokens)

    def augment_main(self, text: str) -> str:
        """Augmentation for the main dataset."""

        augmented = ""
        indices = np.arange(0, len(text), self.block_size)

        # print a progress bar
        for idx in tqdm(indices, desc="Augmenting"):
            batch = text[idx:idx + self.block_size]
            augmented += self.augment_text(batch)

        return augmented

