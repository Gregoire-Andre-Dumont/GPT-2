"""Module that allows for the training of language models."""

import torch
import logging

import numpy as np
from torch import Tensor, nn
from dataclasses import dataclass
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from timm.scheduler.cosine_lr import CosineLRScheduler


@dataclass
class TorchTrainer:
    """Torch trainer for large language models.

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

    # Path to the torch model
    model_path: str | None = None

    def __post_init__(self):
        """Initialize the model and other parameters."""

        # Custom logger for the terminal
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)

        # Set up the torch optimizer
        parameters = {'lr': self.learning_rate, 'betas': (self.beta1, self.beta2)}
        self.optimizer = Adam(self.model.parameters(), *parameters)

        # Initialize the scheduler for learning rate
        parameters = {"t_initial": self.t_init, "warmup_lr_init": self.warmup_lr}
        self.scheduler = CosineLRScheduler(warmup_t=self.warmup_lr, *parameters)

        # Set the torch model to correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"setting the device to {self.device}")
        self.model = self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

    def custom_train(self, x: DataLoader, y: DataLoader):
        """Train the large language model using torch loaders."""

        # Predict if the models is already trained
        if self._model_exists():
            self.logger.info(f"Model exists in {self.model_path}")
            return self.predict_after_train(x, y)

    def initialize_path(self, config: str):
        """Create the path of the model using a hydra config."""




    def custom_trains(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments.
            - train_indices: The indices to train on.
            - validation_indices: The indices to validate on.
            - save_model: Whether to save the model.
            - fold: Fold number if running cv.
        :return: The input and output of the system.
        """
        train_indices = train_args.get("train_indices")
        if train_indices is None:
            raise ValueError("train_indices not provided")
        validation_indices = train_args.get("validation_indices")
        if validation_indices is None:
            raise ValueError("validation_indices not provided")
        save_model = train_args.get("save_model", True)
        self._fold = train_args.get("fold", -1)

        self.save_model_to_disk = save_model

        # Create datasets
        train_dataset, validation_dataset = self.create_datasets(
            x,
            y,
            train_indices,
            validation_indices,
        )

        # Create dataloaders
        train_loader, validation_loader = self.create_dataloaders(train_dataset, validation_dataset)

        # Check if a trained model exists
        if self._model_exists():
            self.log_to_terminal(
                f"Model exists in {self.get_model_path()}. Loading model...",
            )
            self._load_model()

            # Return the predictions
            return self.predict_after_train(
                x,
                y,
                train_dataset,
                validation_dataset,
                train_indices,
                validation_indices,
            )

        # Log the model being trained
        self.log_to_terminal(f"Training model: {self.model.__class__.__name__}")

        # Resume from checkpoint if enabled and checkpoint exists
        start_epoch = 0
        if self.checkpointing_resume_if_exists:
            saved_checkpoints = list(Path(self.trained_models_directory).glob(f"{self.get_hash()}_checkpoint_*.pt"))
            if len(saved_checkpoints) > 0:
                self.log_to_terminal("Resuming training from checkpoint")
                epochs = [int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints]
                self._load_model(saved_checkpoints[np.argmax(epochs)])
                start_epoch = max(epochs) + 1

        # Train the model
        self.log_to_terminal(f"Training model for {self.epochs} epochs{', starting at epoch ' + str(start_epoch) if start_epoch > 0 else ''}")

        train_losses: list[float] = []
        val_losses: list[float] = []

        self.lowest_val_loss = np.inf
        if len(validation_loader) == 0:
            self.log_to_warning(
                f"Doing train full, model will be trained for {self.epochs} epochs",
            )

        self._training_loop(
            train_loader,
            validation_loader,
            train_losses,
            val_losses,
            self._fold,
            start_epoch,
        )
        self.log_to_terminal(
            f"Done training the model: {self.model.__class__.__name__}",
        )

        # Revert to the best model
        if self.best_model_state_dict:
            self.log_to_terminal(
                f"Reverting to model with best validation loss {self.lowest_val_loss}",
            )
            self.model.load_state_dict(self.best_model_state_dict)

        if save_model:
            self._save_model()

        return self.predict_after_train(
            x,
            y,
            train_dataset,
            validation_dataset,
            train_indices,
            validation_indices,
        )