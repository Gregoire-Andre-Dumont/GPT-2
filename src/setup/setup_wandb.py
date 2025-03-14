"""File containing functions related to setting up Weights and Biases."""

import logging
from pathlib import Path
import wandb
from typing import Any
from omegaconf import DictConfig, OmegaConf

# Custom logger for the terminal
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

def setup_wandb(cfg: DictConfig, output_dir: Path) -> Any:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object created with hydra
    :param output_dir: The directory to the Hydra outputs."""

    logger.debug("Initializing Weights & Biases")
    config = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        config=replace_list_with_dict(config),
        project="GPT-2",
        entity="gregoire-andre-c-dumont ",
        job_type="train",
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=output_dir,
        reinit=True)

    if cfg.wandb_log_config:
        # Extract the path to the train config
        model_name = OmegaConf.load("conf/train.yaml").defaults[0].model
        model_path = f"conf/model/{model_name}.yaml"

        # Store the training config as an artifact
        artifact = wandb.Artifact("train_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"

        artifact.add_file(str(config_path), "config.yaml")
        artifact.add_file("conf/train.yaml")
        artifact.add_file(model_path)

        wandb.log_artifact(artifact)
        logger.info("Done initializing Weights & Biases")
    return run


def replace_list_with_dict(o: object) -> object:
    """Recursively replace lists with integer index dicts.

    This is necessary for wandb to properly show any parameters in the config that are contained in a list.

    :param o: Initially the dict, or any object recursively inside it.
    :return: Integer index dict."""
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = replace_list_with_dict(v)
    elif isinstance(o, list):
        o = {i: replace_list_with_dict(v) for i, v in enumerate(o)}
    return o