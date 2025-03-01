"""Loads and encodes the text data using a pre-trained tokenizer."""

import pandas as pd
from transformers import AutoTokenizer
from omegaconf import DictConfig

def setup_train_data(cfg: DictConfig):
    """Load and encode the text data using a pre-trained tokenizer.
    :param cfg: dictionary containing the paths and tokenizer name."""

    # Load the train and validation text
    train = pd.read_parquet(cfg.train_path)['text']
    validation = pd.read_parquet(cfg.val_path)['text']

    # Load the pre-trained tokenizer from hugging face
    tokenizer = AutoTokenizer.from_pretrained(cfg.token_name)

    # Encode both texts using a pre-trained tokenizer
    train = tokenizer(train[0][:60], return_tensors="pt")
    validation = tokenizer(validation[0][:60], return_tensors="pt")

    return train.input_ids, validation.input_ids