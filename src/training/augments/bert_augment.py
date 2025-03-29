"""Module that manages the data augmentation with BERT."""
from dataclasses import dataclass

@dataclass
class BertAugment:
    """Data augmentation with mask language modeling.
    :param p_bert: probability of augmenting tokens."""

    p_bert: float = 0.0

    def augment_text(self, text: str) -> str:
        """Augment the text with mask language modeling."""

        pass