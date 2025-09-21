import gc
import random

import numpy as np
import torch


def clear_memory():
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
