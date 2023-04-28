from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from blocks.block import MLP2d
from blocks.tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "in_channels": [8, 16],
    "hidden_channels": [256],
    "out_channels": [16, 32, None],
    "dropout": [0.0, 0.5],
    "activation": [F.relu, F.gelu],
    "batch_size": [4],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    in_channels: int,
    hidden_channels: int,
    out_channels: Optional[int],
    dropout: float,
    activation: Callable[[Tensor], Tensor],
    batch_size: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    mlp = MLP2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
        activation=activation,
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch = mlp.forward(input_batch)
    out_channels = out_channels or in_channels
    assert tuple(output_batch.shape) == (batch_size, out_channels, heigth, width)
