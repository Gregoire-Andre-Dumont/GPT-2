"""Module that manages the base data augmentation."""

from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class BaseAugment:
    """Base class for all data augmentations."""

    def augment_main(self, text: str) -> str:
        """Data augmentation for the main dataset"""

        return text

    def augment_summarize(self, data: DataFrame) -> DataFrame:
        """Data augmentation for the summarize dataset"""

        return data

