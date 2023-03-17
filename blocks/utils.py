from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(_.numel() for _ in model.parameters())
