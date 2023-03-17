from contextlib import nullcontext
from typing import Any, Dict, List

import pytest
import torch

from blocks import CrossAttentionConv2d
from tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "batch_size": [4],
    "num_heads": [4, 12],
    "in_channels": [12, 32],
    "out_channels": [32, 64],
    "heigth": [32, 64],
    "width": [32, 64],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    num_heads: int,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    raise_assert = bool(out_channels % num_heads)
    context = pytest.raises(Exception) if raise_assert else nullcontext()
    with context:
        conv = CrossAttentionConv2d(
            in_channels=in_channels, out_channels=out_channels, num_heads=num_heads
        ).to(device)
    if not raise_assert:
        input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
        with torch.no_grad():
            output_batch: torch.Tensor = conv(input_batch)
        assert tuple(output_batch.shape) == (batch_size, out_channels, heigth, width)
