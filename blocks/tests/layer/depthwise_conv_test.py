from typing import Any, Dict, List

import pytest
import torch

from blocks.layer import DWConv2d
from blocks.tests.utils import compute_conv_size, create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "batch_size": [4],
    "in_channels": [3, 8],
    "out_channels": [8, 16],
    "kernel_size": [3, 2],
    "stride": [1, 2],
    "padding": [0, 1],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(  # pylint: disable=too-many-arguments
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    layer = DWConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch = layer.forward(input_batch)
    kwargs = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
    out_heigth = compute_conv_size(size=heigth, **kwargs)
    out_width = compute_conv_size(size=width, **kwargs)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
