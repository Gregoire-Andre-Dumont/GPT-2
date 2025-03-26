"""Main script for training the model and will take in the raw data and output a trained model."""
import wandb
import logging
import hydra
import coloredlogs
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader

from src.setup.setup_wandb import setup_wandb
from src.setup.setup_data import encode_data
from src.setup.setup_seed import set_torch_seed
from src.utils.separator import section_separator
from src.training.datasets.main_dataset import MainDataset


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

    # Load and encode train and validation
    section_separator("Load and tokenize the data")
    train = encode_data(cfg.train_path, cfg.development)
    valid = encode_data(cfg.valid_path, cfg.development)

    logger.info(f"Training size: {len(train)}")
    logger.info(f"Validation size: {len(valid)}")

    # Initialize the torch trainer and model
    section_separator("Initialize the torch model")
    trainer = hydra.utils.instantiate(cfg.model.torch_trainer)
    trainer.create_path(cfg.model.torch_trainer)

    # Initialize the train and validation datasets
    logger.info("Create the train and validation datasets")
    train_dataset = MainDataset(data=train, data_augment=True)
    valid_dataset = MainDataset(data=valid, data_augment=False)

    logger.info("Create the train and validation dataloaders")
    train_dataloader = DataLoader(train_dataset, batch_size=trainer.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=trainer.batch_size, shuffle=False)

    # Train or fine-tune the GPT-2 model
    section_separator("Train or fine-tune the GPT-2 model")
    valid_loss = trainer.custom_train(train_dataloader, valid_dataloader)

    if wandb.run:
        wandb.log({"Validation Score": valid_loss})
    wandb.finish()

if __name__ == "__main__":
    run_train()