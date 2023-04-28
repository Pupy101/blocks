from typing import Any, Dict, List

import pytest
import torch

from blocks.block import SEBlock2d
from blocks.tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "batch_size": [4],
    "in_channels": [8, 16],
    "squeeze_channels": [16, 32, 64],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    batch_size: int,
    in_channels: int,
    squeeze_channels: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    block = SEBlock2d(in_channels=in_channels, squeeze_channels=squeeze_channels).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch = block.forward(input_batch)
    assert tuple(output_batch.shape) == (batch_size, in_channels, heigth, width)
