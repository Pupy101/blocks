from typing import Any, Callable, Dict, List

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from blocks.block import ConvNeXtBlock2d
from blocks.tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "dim": [8, 16],
    "drop_path": [0.1, 0.4],
    "droupout": [0.1, 0.4],
    "activation": [F.relu, F.gelu],
    "batch_size": [4],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    dim: int,
    drop_path: float,
    droupout: float,
    activation: Callable[[Tensor], Tensor],
    batch_size: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    block = ConvNeXtBlock2d(dim=dim, drop_path=drop_path, droupout=droupout, activation=activation).to(device)
    input_batch = torch.rand(batch_size, dim, heigth, width).to(device)
    with torch.no_grad():
        output_batch = block.forward(input_batch)
    assert tuple(output_batch.shape) == (batch_size, dim, heigth, width)
