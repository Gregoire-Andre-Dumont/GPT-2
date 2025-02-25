"""train is the main script for training and fine-tuning GPT 2 for text8."""

import hydra
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train the GPT 2 model with a train-test split. Hydra loads the config file."""

    # Load the training and validation texts
    train = pd.read_parquet(cfg.train_path)['text']
    validation = pd.read_parquet(cfg.val_path)['text']

    # Check whether the tokenizer is already trained
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer.train(train[0][:100], 500)
    a = 1




if __name__ == "__main__":
    run_train()