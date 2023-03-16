import pytest
import torch

from blocks import CrossAttentionConv2d


@pytest.mark.parametrize(
    "num_heads,batch_size,in_channels,in_heigth,in_width,out_channels,out_heigth,out_width",
    [
        [4, 4, 16, 64, 64, 64, 64, 64],
    ],
)
def test(
    num_heads: int,
    batch_size: int,
    in_channels: int,
    in_heigth: int,
    in_width: int,
    out_channels: int,
    out_heigth: int,
    out_width: int,
    device: torch.device,
) -> None:
    conv = CrossAttentionConv2d(
        in_channels=in_channels, out_channels=out_channels, num_heads=num_heads
    ).to(device)
    input_batch = torch.rand(batch_size, in_channels, in_heigth, in_width).to(device)
    with torch.no_grad():
        output_batch: torch.Tensor = conv(input_batch)
    assert tuple(output_batch.shape) == (batch_size, out_channels, out_heigth, out_width)
