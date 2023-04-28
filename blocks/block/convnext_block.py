from random import random
from typing import Callable

from torch import Tensor, nn
from torch.nn import functional as F

from blocks.layer import DWConv2d, LayerNorm2d

from .mlp import MLP2d


class ConvNeXtBlock2d(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        droupout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self.drop_path = drop_path
        self.conv = DWConv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3)
        self.norm = LayerNorm2d(channels=dim)
        self.mlp = MLP2d(in_channels=dim, hidden_channels=dim * 4, dropout=droupout, activation=activation)

    def forward(self, batch: Tensor) -> Tensor:
        if self.training and random() < self.drop_path:
            return batch
        in_batch = batch
        batch = self.conv(batch)
        batch = self.norm(batch)
        batch = self.mlp(batch)
        return in_batch + batch
