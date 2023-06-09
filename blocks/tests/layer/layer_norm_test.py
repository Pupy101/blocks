from typing import Any, Dict, List

import pytest
import torch

from blocks.layer import LayerNorm2d
from blocks.tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "batch_size": [4],
    "channels": [3, 16],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    batch_size: int,
    channels: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    norm = LayerNorm2d(channels=channels).to(device)
    input_batch = torch.rand(batch_size, channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch = norm.forward(input_batch)
    assert tuple(output_batch.shape) == (batch_size, channels, heigth, width)
    assert torch.allclose(
        torch.mean(output_batch, dim=(2, 3)),
        torch.zeros(batch_size, channels).to(device),
        atol=1e-3,
    )
    assert torch.allclose(
        torch.std(output_batch, dim=(2, 3)),
        torch.ones(batch_size, channels).to(device),
        atol=1e-3,
    )
