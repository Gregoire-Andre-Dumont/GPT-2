"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""

import hydra
from omegaconf import DictConfig
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train the GPT2 model with the text8 dataset."""

    # Load the train and validation text
    train = pd.read_parquet(cfg.train_path)['text']
    validation = pd.read_parquet(cfg.val_path)['text']

    # Encode both texts using the chosen tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    train = tokenizer.encode(train[0][:60])
    validation = tokenizer.encode(validation[0][:60])

    # Initialize the torch trainer and model
    trainer = hydra.utils.instantiate(cfg.trainer)
    b = 1






if __name__ == "__main__":
    run_train()