"""train is the main script for training and fine-tuning GPT 2 for text8."""

import hydra
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train the GPT 2 model with a train-test split. Hydra loads the config file."""

    # Load the training and validation texts
    train = pd.read_parquet(cfg.train_path)['text']
    validation = pd.read_parquet(cfg.val_path)['text']

    # Load the pre-trained tokenizer from hugging face
    tokenizer = AutoTokenizer.from_pretrained(cfg.token_name)

    # Process the train and test prompts
    train = tokenizer(train[0], return_tensors="np").input_ids
    test = tokenizer(validation[0], return_tensors="np").input_ids






if __name__ == "__main__":
    run_train()