from typing import Callable

from torch import Tensor, nn


class SqueezeExcitation2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        activation: Callable[[Tensor], Tensor] = nn.GELU(),
        scale_activation: Callable[[Tensor], Tensor] = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1
        )
        self.activation = activation
        self.scale_activation = scale_activation

    def scale_batch(self, batch: Tensor) -> Tensor:
        scale = self.avgpool(batch)
        scale = self.conv1(scale)
        scale = self.activation(scale)
        scale = self.conv2(scale)
        return self.scale_activation(scale)

    def forward(self, batch: Tensor) -> Tensor:
        scale = self.scale_batch(batch)
        return scale * batch
