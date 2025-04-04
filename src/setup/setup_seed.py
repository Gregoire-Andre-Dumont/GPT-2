"""Set seed for reproducibility."""
import torch

def set_torch_seed(seed: int = 42) -> None:
    """Set torch seed for reproducibility.
    :param seed: seed to set"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False