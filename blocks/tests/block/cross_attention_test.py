from contextlib import nullcontext
from typing import Any, Dict, List

import pytest
import torch

from blocks.block import CrossAttention2d
from blocks.tests.utils import create_product_parametrize

PARAMS: Dict[str, List[Any]] = {
    "in_channels": [12, 32],
    "out_channels": [32, 64],
    "num_heads": [4, 12],
    "batch_size": [4],
    "heigth": [32, 64],
    "width": [32, 64],
}


@pytest.mark.parametrize(*create_product_parametrize(PARAMS))
def test(
    in_channels: int,
    out_channels: int,
    num_heads: int,
    batch_size: int,
    heigth: int,
    width: int,
    device: torch.device,
) -> None:
    raise_assert = bool(out_channels % num_heads)
    context = pytest.raises(Exception) if raise_assert else nullcontext()
    model_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "num_heads": num_heads}
    with context:
        attention = CrossAttention2d(**model_kwargs).to(device)
    if not raise_assert:
        input_batch = torch.rand(batch_size, in_channels, heigth, width).to(device)
        with torch.no_grad():
            output_batch = attention.forward(input_batch)
        assert tuple(output_batch.shape) == (batch_size, out_channels, heigth, width)
