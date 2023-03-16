import pytest
import torch

from blocks import LayerNorm2d


@pytest.mark.parametrize(
    "batch_size,channels,heigth,width",
    [
        [4, 3, 112, 112],
    ],
)
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
        output_batch: torch.Tensor = norm(input_batch)
    assert tuple(output_batch.shape) == (batch_size, channels, heigth, width)
