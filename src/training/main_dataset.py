import torch

from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

@dataclass
class MainDataset(Dataset):
    """Main torch dataset for language models.

    :param data: processed tokens with shape (x, 1024)
    :param data_augment: Whether to apply data augmentation."""

    data: npt.NDArray[np.uint64] | None = None
    data_augment: bool | None = None

    def __len__(self) -> int:
        """Return the length of the dataset."""

        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple:
        """Extract the sequence and the target."""

        # Check whether we are training
        sequence = self.data[index]
        length = sequence.shape[0]

        if self.data_augment:
            # Extract a random subsequence
            index = np.random.randint(0, length)
            return sequence[:index], sequence[index]

        return sequence[:-1], sequence[-1]