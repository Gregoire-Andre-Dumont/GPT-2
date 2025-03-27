"""Main script for training the model and will take in the raw data and output a trained model."""
import wandb
import logging
import warnings
import os
import hydra
import coloredlogs
from omegaconf import DictConfig
from pathlib import Path

from src.setup.setup_wandb import setup_wandb
from src.setup.setup_data import setup_pre_training
from copy import deepcopy
from src.setup.setup_seed import set_torch_seed
from src.utils.separator import section_separator

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"



@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train or fine-tune the GPT-2 model using the hydra configuration."""
    section_separator("Q3 - Machine Learning: a bayesian perspective")

    # Custom logger for the terminal
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    # Set the torch random seed
    coloredlogs.install()
    set_torch_seed()

    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb_enabled:
        setup_wandb(cfg, output_dir)

    # Load and initialize the dataloaders
    section_separator("Load and initialize the torch dataloaders")
    loader = hydra.utils.instantiate(cfg.model.dataloader)

    train_loader = setup_pre_training(cfg, cfg.train_path, True, deepcopy(loader))
    valid_loader = setup_pre_training(cfg, cfg.valid_path, False, deepcopy(loader))
    test_loader = setup_pre_training(cfg, cfg.valid_path, False, deepcopy(loader))

    logger.info(f"Training size: {len(train_loader)}")
    logger.info(f"Validation size: {len(valid_loader)}")
    logger.info(f"Test size: {len(test_loader)}")

    # Initialize the torch trainer and model
    section_separator("Initialize the torch model")
    trainer = hydra.utils.instantiate(cfg.model.torch_trainer)
    trainer.create_path(cfg.model.torch_trainer)

    # Train or fine-tune the GPT-2 model
    section_separator("Train and validate the GPT-2 model")
    valid_loss = trainer.custom_train(train_loader, valid_loader)
    test_loss = trainer.predict_score(test_loader)


    if wandb.run:
        wandb.log({"Validation Score": valid_loss})
        wandb.log({"Test Score": test_loss})
    wandb.finish()

if __name__ == "__main__":
    run_train()