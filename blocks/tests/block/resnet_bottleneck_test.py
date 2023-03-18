from typing import Any, Dict, List, Tuple, Type, Union

import pytest
import torch

from blocks.block import ResNetBottleneck2d, ResNetDWBottleneck2d
from blocks.tests.utils import compute_conv_size, create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "block_type": [ResNetBottleneck2d, ResNetDWBottleneck2d],
    "in_channels": [8, 16],
    "out_channels": [(32, 32, 32), (32, 128, 32), (32, 16, 32)],
    "stride": [1, 2],
    "expansion": [2, 4],
    "batch_size": [2, 4],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    block_type: Union[Type[ResNetBottleneck2d], Type[ResNetDWBottleneck2d]],
    in_channels: int,
    out_channels: Tuple[int, int, int],
    stride: int,
    expansion: int,
    batch_size: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    block = block_type(
        in_channels=in_channels, out_channels=out_channels, stride=stride, expansion=expansion
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch: torch.Tensor = block(input_batch)
    kwargs = dict(kernel_size=3, padding=1, stride=stride)
    out_heigth = compute_conv_size(heigth, **kwargs)
    out_width = compute_conv_size(width, **kwargs)
    assert tuple(output_batch.shape) == (
        batch_size,
        out_channels[-1] * expansion,
        out_heigth,
        out_width,
    )
