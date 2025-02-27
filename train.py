"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""

import torch
import hydra
from omegaconf import DictConfig
from src.setup.setup_data import setup_train_data

@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train the GPT2 model with the text8 dataset."""

    # Load and encode the train and validation
    train, val = setup_train_data(cfg)

    # Load and Initialize the model
    a = 1


if __name__ == "__main__":
    run_train()