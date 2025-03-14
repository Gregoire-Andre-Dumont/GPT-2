from torch import Tensor
import random
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

@dataclass
class MainDataset(Dataset):
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
        idx = start_index + random.randint(0, 1020)
        input = self.data[idx: idx + self.block_size]
        target = self.data[idx+1:idx + self.block_size+1]

        return Tensor(input), Tensor(target)