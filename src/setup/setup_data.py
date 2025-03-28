"""Module that creates the dataloaders for pre-training, summarization, and question answering."""

from transformers import GPT2Tokenizer
import pandas as pd
from torch.utils.data import DataLoader

def setup_pre_training(cfg, path: str, augment: bool, loader: DataLoader):
    """Create the dataloader for pre-training the GPT2 model.

    :param cfg: Contains the config parameters.
    :param path: parquet file containing the text.
    :param augment: whether to use data augmentation or not.
    :param loader: dataloader without the text."""

    # Extract the text from the dataframe
    data = pd.read_parquet(path)['text'][0]
    size = cfg.development
    data = data[:size] if size else data

    # Initialize the dataloader with the text data
    loader.dataset.initialize(data, augment)
    return loader

