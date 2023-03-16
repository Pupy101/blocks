import pytest
import torch

from blocks import MLPConv2d


@pytest.mark.parametrize(
    "batch_size,hidden_channels,in_channels,in_heigth,in_width,out_channels,out_heigth,out_width,droupout",
    [
        [4, 64, 3, 64, 64, 8, 64, 64, 0.5],
    ],
)
def test(
    batch_size: int,
    hidden_channels: int,
    in_channels: int,
    in_heigth: int,
    in_width: int,
    out_channels: int,
    out_heigth: int,
    out_width: int,
    droupout: float,
    device: torch.device,
) -> None:
    conv = MLPConv2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        droupout=droupout,
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, in_heigth, in_width).to(device)
    with torch.no_grad():
        output_batch: torch.Tensor = conv(input_batch)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
