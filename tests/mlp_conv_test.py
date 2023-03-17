from typing import Any, Dict, List

import pytest
import torch

from blocks import MLPConv2d
from tests.utils import compute_conv_size, create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "batch_size": [4],
    "in_channels": [8, 16],
    "out_channels": [16, 32],
    "hidden_channels": [256],
    "kernel_size": [2, 3],
    "padding": [0, 1],
    "droupout": [0.0, 0.5],
    "heigth": [64, 112],
    "width": [64, 112],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    kernel_size: int,
    padding: int,
    droupout: float,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    conv = MLPConv2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        droupout=droupout,
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
    with torch.no_grad():
        output_batch: torch.Tensor = conv(input_batch)
    kwargs = dict(kernel_size=kernel_size, padding=padding)
    # 2 times compute size because apply 2 times convolution in MLPConv2d
    out_heigth = compute_conv_size(size=compute_conv_size(size=heigth, **kwargs), **kwargs)
    out_width = compute_conv_size(size=compute_conv_size(size=width, **kwargs), **kwargs)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
