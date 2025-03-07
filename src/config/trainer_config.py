"""Module that manages the parameters of the torch trainer."""

from torch import Tensor, nn
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    """Manage the parameters of the GPT-2 model.

    :param model: the torch model to train.
    :param criterion: the torch loss function.
    :param model_name: the name of the model.

    :param save_checkpoint: whether to save intermediate models.
    :param init_from: whether to train from scratch or checkpoint.
    :param epochs: number of epochs for training.
    :param patience: stopping training after no improvement.
    :param batch_size: batch size for training.

    :param learning_rate: learning rate for training."""

    criterion: nn.Module
    model: nn.Module | None = None
    model_name: str = "GPT2"

    save_checkpoint: bool = True
    init_from: str = 'scratch'
    epochs: int = 20
    patience: int = 5
    batch_size: int = 16

    # torch optimizer parameters
    learning_rate: float = 6e-4
    beta1: float = 0.90
    beta2: float = 0.95
    grad_clip: float = 1.0

    # cosine scheduler parameters
    t_init: int = 40
    warmup_t: int = 1
    warmup_lr: float = 1e-5