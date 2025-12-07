import os
import random

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    """Sets the random seed across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
