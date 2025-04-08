"""Module that creates the dataloaders for pre-training, summarization, and question answering."""

import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader

def setup_pre_training(cfg, path: str, augment: bool, loader: DataLoader):
    """Create the dataloader for pre-training the GPT2 model.

    :param cfg: Contains the config parameters.
    :param path: parquet file containing the text.
    :param augment: whether to use data augmentation or not.
    :param loader: dataloader without the text."""

    # Extract the text from the dataframe
    data = pd.read_csv(path)['text'][0]
    size = cfg.development
    data = data[:size] if size else data

    # Remove unnecessary characters

    # Initialize the dataloader with the text data
    loader.dataset.initialize(data, augment)
    return loader

def setup_fine_tuning(cfg, data: DataFrame, augment: bool, loader: DataLoader):
    """Create the dataloader for fine-tuning the GPT2 model.

    :param cfg: Contains the config parameters.
    :param data: Contains the texts and classes.
    :param augment: whether to use data augmentation or not.
    :param loader: dataloader without the input and output."""

    # Extract the texts and classes



