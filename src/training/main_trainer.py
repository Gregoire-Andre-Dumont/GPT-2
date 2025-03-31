"""Module that manages the torch trainer for training LLMs."""

import gc
import copy
import wandb
import torch
import logging
from joblib import hash
from typing import Any

import numpy as np
from torch import Tensor, nn
from dataclasses import dataclass, field
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from timm.scheduler.cosine_lr import CosineLRScheduler

@dataclass
class MainTrainer:
    """Torch trainer for large language models.

    :param model: the torch model to train.
    :param criterion: the torch loss function.
    :param model_name: the name of the model.

    :param save_model: whether to save the model on the disk.
    :param epochs: number of epochs for training.
    :param patience: stopping training after no improvement.
    :param learning_rate: learning rate for ADAM.
    :param scheduler_param: parameters of cosine scheduler."""

    criterion: nn.Module | None = None
    model: nn.Module | None = None
    model_name: str = "GPT2"
    save_model: bool = True

    epochs: int = 20
    patience: int = 5
    learning_rate: float = 6e-4
    scheduler_param: dict | None = None

    def __post_init__(self):
        """Initialize the model and other parameters."""

        # Custom logger for the terminal
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)

        # Initialize the optimizer and scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CosineLRScheduler(optimizer=self.optimizer, **self.scheduler_param)

        # Set the torch model to correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"setting the device to {self.device}")
        self.model = self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf
        self.best_model_state_dict: dict[Any, Any] = {}

        # Mixed precision trainer
        self.logger.info("Using mixed precision training.")
        self.scaler = torch.GradScaler(device=self.device.type)
        torch.set_float32_matmul_precision("high")

    def custom_train(self, train_loader: DataLoader, valid_loader: DataLoader):
        """Train and evaluate the large language model.

        :param train_loader: loader for the training data.
        :param valid_loader: loader for the validation data."""

        # Train the large language model
        self.logger.info(f"Train the model for {self.epochs} epochs")
        valid_score = self._training_loop(train_loader, valid_loader)

        # Revert to the model with the best validation loss
        self.logger.info(f"Done training the model {self.model_name}")

        if self.best_model_state_dict:
            self.logger.info(f"model with best validation loss {self.lowest_val_loss}")
            self.model.load_state_dict(self.best_model_state_dict)

        # save the model if necessary
        self._save_model() if self.save_model else None
        return valid_score

    def create_path(self, config: dict):
        """Create the model path using the config hash."""

        self.model_path = Path(f"tm/{hash(config)}.pt")

    def _load_model(self) -> bool:
        """Load the model from the model directory tm."""

        self.logger.info(f"Loading the model from {self.model_path}")
        model = torch.load(self.model_path)
        self.model.load_state_dict(model.state_dict())

    def predict_score(self, loader: DataLoader):
        """Predict the model performance on the given loader."""

        self.logger.info(f"Running inference on the test loader")
        losses = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(loader, unit="batch", disable=False):
                X_batch = X_batch.to(self.device, dtype=torch.int64)
                y_batch = y_batch.to(self.device, dtype=torch.int64)

                _, loss = self.model(X_batch, y_batch)
                losses.append(loss.item())

        self.logger.info("Done predicting!")
        return sum(losses) / len(losses)

    def _training_loop(self, train_loader: DataLoader, valid_loader: DataLoader):
        """Training loop for the large language model.

        :param train_loader: Dataloader for the training data
        :param valid_loader: Dataloader for the validation data (can be empty)"""

        self.external_define_metric(f"Training Loss", "epoch")
        self.external_define_metric(f"Validation Loss", "epoch")

        # Initialize the training and validation losses
        train_losses: list[float] = []
        val_losses: list[float] = []

        # Set the scheduler to the correct epoch
        if self.scheduler is not None:
            self.scheduler.step(epoch=0)

        for epoch in range(self.epochs):
            # Compute the training loss
            train_loss = self.train_one_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            # Compute the validation loss
            self.last_val_loss = self.val_one_epoch(valid_loader, epoch)
            val_losses.append(self.last_val_loss)

            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step(epoch=epoch + 1)

            # Check whether wandb is initialized
            if wandb.run:
                # Log the training and validation loss to wandb
                wandb.log({"Validation Loss": val_losses[-1], "epoch": epoch})
                wandb.log({"Training Loss": train_loss, "epoch": epoch})

                # plot the train/val loss against each other
                plot = wandb.plot.line_series(
                    xs=list(range(epoch + 1)),
                    ys=[train_losses, val_losses],
                    keys=[f"Train", f"Validation"],
                    title=f"Training/Validation",
                    xname="Epoch")

                wandb.log({'title': plot}, commit=False)

            # Store the best model based on validation loss
            if self.last_val_loss < self.lowest_val_loss:
                self.lowest_val_loss = self.last_val_loss
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    self.logger.info("Early stopping!")
                    return sum(val_losses) / len(val_losses)
        return sum(val_losses) / len(val_losses)

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train the model for one epoch and compute its loss.

        :param loader: Dataloader for the training data.
        :param epoch: Current epoch number."""

        losses = []
        self.model.train()

        # Progress bar for training
        learning_rate = self.optimizer.param_groups[0]['lr']
        desc = f"Epoch {epoch} Train ({learning_rate:0.8f})"
        progress_bar = tqdm(loader, unit="batch", desc=desc)

        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(self.device, dtype=torch.int64)
            y_batch = y_batch.to(self.device, dtype=torch.int64)

            with torch.autocast(self.device.type):
                # Forward and backward pass
                _, loss = self.model(X_batch, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Print the progress bar
            losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            progress_bar.set_postfix(loss=mean_loss)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def val_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Compute the validation loss of the model for one epoch.

        :param loader: Dataloader for the validation data.
        :param epoch: current epoch number."""

        losses = []
        self.model.eval()
        progress = tqdm(loader, unit="batch")

        with torch.no_grad():
            for X_batch, y_batch in progress:
                X_batch = X_batch.to(self.device, dtype=torch.int64)
                y_batch = y_batch.to(self.device, dtype=torch.int64)

                # Forward pass of the model
                _, loss = self.model(X_batch, y_batch)
                losses.append(loss.item())

                # Print the progress bar
                progress.set_description(desc=f"Epoch {epoch} Valid")
                progress.set_postfix(loss=sum(losses) / len(losses))
            return sum(losses) / len(losses)

    def _save_model(self):
        """Save the model in the model directory."""

        self.logger.info(f"Saving model to {self.model_path}")
        Path(self.model_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, self.model_path)


    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define a metric in an external service.

        :param metric: The metric to define
        :param metric_type: The type of the metric"""

        if wandb.run:
            wandb.define_metric(metric, step_metric=metric_type)