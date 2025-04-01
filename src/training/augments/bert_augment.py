"""Module that manages the data augmentation with BERT."""

import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from transformers import BertForMaskedLM, BertTokenizer
from src.training.augments.base_augment import BaseAugment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class BertAugment(BaseAugment):
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

        # Predict the masked tokens
        tokens = self.tokenizer.convert_tokens_to_ids(masked_text)
        tokens = torch.tensor([tokens]).to(device)

        with torch.no_grad():
            outputs = self.model(tokens)
            predictions = outputs.logits

            predicted_ids = torch.argmax(predictions[0, masked_indices], dim=1)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids)

        # Replace MASK tokens with predictions
        for idx, mask_pos in enumerate(masked_indices):
            masked_text[mask_pos] = predicted_tokens[idx]

        return self.tokenizer.convert_tokens_to_string(masked_text)

    def augment_main(self, text: str) -> str:
        """Augmentation for the main dataset."""

        augmented_texts = []
        indices = np.arange(0, len(text), self.block_size)

        for idx in tqdm(indices, total=len(indices), desc="Augmenting"):
            batch = text[idx:idx + self.block_size]
            augmented_texts.append(self.augment_text(batch))

        return "".join(augmented_texts)