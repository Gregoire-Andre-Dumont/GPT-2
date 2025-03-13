"""Main script for training the model and will take in the raw data and output a trained model."""
import logging
import hydra
import coloredlogs
from omegaconf import DictConfig

from src.setup.setup_data import encode_data
from src.setup.setup_seed import set_torch_seed
from src.utils.separator import section_separator


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

    # Load and encode train and validation
    section_separator("Load and tokenize the data")
    train = encode_data(cfg.train_path, cfg.development)
    valid = encode_data(cfg.valid_path, cfg.development)

    logger.info(f"Training size: {len(train)}")
    logger.info(f"Validation size: {len(valid)}")
    logger.info(f"Sequence length: {1024}")

    ff = hydra.utils.instantiate(cfg.model.bb)


    a = 1



if __name__ == "__main__":
    run_train()