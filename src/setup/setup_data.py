"""Module that encodes and decodes the text using Byte-Pair."""

from transformers import GPT2Tokenizer
import pandas as pd
import numpy as np

def encode_data(path: str, development: int | bool):
    """Encode the text data using a pre-trained encoder.

    :param path: parquet file containing the text.
    :param development: Reduce the text size for development."""

    # Load the pre-trained tokenizer from hugging face
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load and reduce the text data if necessary
    data = pd.read_parquet(path)['text'][0]
    data = data[:development] if development else data

    # Encode the text data using the tokenizer
    data = tokenizer(data, return_tensors='np')
    return np.squeeze(data.input_ids)


