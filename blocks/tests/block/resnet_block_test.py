from typing import Any, Dict, List, Type, Union

import pytest
import torch

from blocks.block import ResNetBlock2d, ResNetDWBlock2d
from blocks.tests.utils import compute_conv_size, create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "block_type": [ResNetBlock2d, ResNetDWBlock2d],
    "in_channels": [8, 16],
    "out_channels": [16, 32],
    "stride": [1, 2],
    "batch_size": [4],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    block_type: Union[Type[ResNetBlock2d], Type[ResNetDWBlock2d]],
    in_channels: int,
    out_channels: int,
    stride: int,
    batch_size: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    block = block_type(in_channels=in_channels, out_channels=out_channels, stride=stride).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch = block.forward(input_batch)
    kwargs = {"kernel_size": 3, "padding": 1, "stride": stride}
    out_heigth = compute_conv_size(heigth, **kwargs)
    out_width = compute_conv_size(width, **kwargs)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
