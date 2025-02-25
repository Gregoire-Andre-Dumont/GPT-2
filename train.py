"""train is the main script for training and fine-tuning GPT 2 for text8."""

import os
import hydra
import warnings
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train the GPT 2 model with a train-test split. Hydra loads the config file."""




if __name__ == "__main__":
    run_train()