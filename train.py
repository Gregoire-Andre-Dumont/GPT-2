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
from src.training.main_dataset import MainDataset


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

    # Get the output directory from the runtime configuration
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_config.runtime.output_dir)

    if cfg.wandb_enabled:
        setup_wandb(cfg, output_dir)

    # Load and encode train and validation
    section_separator("Load and tokenize the data")
    train = encode_data(cfg.train_path, cfg.development)
    valid = encode_data(cfg.valid_path, cfg.development)
    test = encode_data(cfg.test_path, cfg.development)

    logger.info(f"Training size: {len(train)}")
    logger.info(f"Validation size: {len(valid)}")
    logger.info(f"Test size: {len(test)}")

    # Initialize the torch model and trainer
    section_separator("Train and evaluate the model")
    trainer = hydra.utils.instantiate(cfg.model.torch_trainer)
    trainer.create_path(cfg.model.torch_trainer)

    logger.info('Create train and validation dataloader')
    train_dataset = MainDataset(data=train, data_augment=True)
    valid_dataset = MainDataset(data=valid, data_augment=False)
    test_dataset = MainDataset(data=test, data_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=trainer.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=trainer.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False)

    # Check whether the model is already trained
    if trainer.model_exist():
        trainer.custom_train(train_loader, valid_loader)
    else:
        # Otherwise, load the saved model
        trainer._load_model()

    logger.info('Predict on the validation and test data.')
    valid_loss = trainer.predict_on_valid(valid_loader)
    test_loss = trainer.predict_on_test(test_loader)

    logger.info(f"Validation loss: {valid_loss}")
    logger.info(f"Test loss: {test_loss}")

    if wandb.run:
        wandb.log({"Validation loss": valid_loss,
                   "Test loss": test_loss})
    wandb.finish()

if __name__ == "__main__":
    run_train()