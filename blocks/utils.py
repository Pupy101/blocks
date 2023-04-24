import random
from typing import Optional

import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module, requires_grad: Optional[bool] = None) -> int:
    count = 0
    for params in model.parameters():
        if requires_grad is not None and params.requires_grad != requires_grad:
            continue
        count += params.numel()
    return count


def freeze_weight(model: nn.Module) -> None:
    for weight in model.parameters():
        weight.requires_grad = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
