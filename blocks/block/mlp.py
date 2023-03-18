from typing import Callable

from torch import Tensor, nn


class MLP2d(nn.Module):
    def __init__(  # pylint:disable=too-many-arguments
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        droupout: float,
        activation: Callable[[Tensor], Tensor] = nn.GELU(),
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.linear1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = activation
        self.linear2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.droupout = nn.Dropout2d(droupout)

    def forward(self, batch: Tensor) -> Tensor:
        batch = self.linear1(batch)
        batch = self.activation(batch)
        batch = self.linear2(batch)
        batch = self.droupout(batch)
        return batch
