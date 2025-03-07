"""Module that allows for the training of torch models."""

from torch import Tensor, nn
from dataclasses import dataclass
from torch.optim import Adam

from timm.scheduler.cosine_lr import CosineLRScheduler


@dataclass
class TorchTrainer(logger):
    """Abstract class for torch trainers.

    :param model: the torch model to train.
    :param criterion: the torch loss function.
    :param model_name: the name of the model.

    :param save_checkpoint: whether to save intermediate models.
    :param init_from: whether to train from scratch or checkpoint.
    :param epochs: number of epochs for training.
    :param patience: stopping training after no improvement.
    :param batch_size: batch size for training.

    :param learning_rate: learning rate for training."""

    model: nn.Module
    criterion: nn.Module
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


    def __post_init__(self):
        """Initialize the model and other parameters."""

        # Set up the torch optimizer
        parameters = {'lr': self.learning_rate, 'betas': (self.beta1, self.beta2)}
        self.optimizer = Adam(self.model.parameters(), *parameters)

        # Initialize the scheduler for learning rate
        parameters = {"t_initial": self.t_init, "warmup_lr_init": self.warmup_lr}
        self.scheduler = CosineLRScheduler(warmup_t=self.warmup_lr, *parameters)



        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_to_terminal(f"Setting device: {self.device}")

        # If multiple GPUs are available, distribute batch size over the GPUs
        if torch.cuda.device_count() > 1:
            self.log_to_terminal(f"Using {torch.cuda.device_count()} GPUs")
            self.model = _CustomDataParallel(self.model)

        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

        # Mixed precision
        if self.use_mixed_precision:
            self.log_to_terminal("Using mixed precision training.")
            self.scaler = torch.GradScaler(device=self.device.type)
            torch.set_float32_matmul_precision("high")

        # Check validity of model_name
        if " " in self.model_name:
            raise ValueError("Spaces in model_name not allowed")

        super().__post_init__()