"""Module that encodes and decodes the text data using a pre-trained tokenizer."""

import pandas as pd
from transformers import AutoTokenizer
from omegaconf import DictConfig

def encode_data(cfg: DictConfig):
    """Load and encode the text data using a pre-trained tokenizer."""

    # Load the pre-trained tokenizer from hugging face
    tokenizer = AutoTokenizer.from_pretrained(cfg.token_name)

    # Load the train, validation and test texts
    train = pd.read_parquet(cfg.train_path)['text']
    validation = pd.read_parquet(cfg.val_path)['text']
    test = pd.read_parquet(cfg.val_path)['text']

    # Reduce the dataset size for development
    train_size = int(len(train) * cfg.train_size)
    val_size = int(len(validation) * cfg.test_size)

    validation = validation[:val_size]
    train = train[:train_size]

    # Encode the text data using the tokenizer
    train = tokenizer(train[0], return_tensors="pt")
    validation = tokenizer(validation[0], return_tensors="pt")
    test = tokenizer(test[0], return_tensors="pt")

    return train.input_ids, validation.input_ids, test.input_ids