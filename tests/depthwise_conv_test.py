import pytest
import torch

from blocks import DepthwiseConv2d


@pytest.mark.parametrize(
    "kernel_size,stride,padding,batch_size,in_channels,in_heigth,in_width,out_channels,out_heigth,out_width",
    [
        [3, 1, 1, 4, 3, 64, 64, 8, 64, 64],
    ],
)
def test(
    kernel_size: int,
    stride: int,
    padding: int,
    batch_size: int,
    in_channels: int,
    in_heigth: int,
    in_width: int,
    out_channels: int,
    out_heigth: int,
    out_width: int,
    device: torch.device,
) -> None:
    conv = DepthwiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, in_heigth, in_width).to(device)
    with torch.no_grad():
        output_batch: torch.Tensor = conv(input_batch)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
