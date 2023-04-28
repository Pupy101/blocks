from typing import Callable

from torch import Tensor, nn
from torch.nn import functional as F


class SEBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        scale_activation: Callable[[Tensor], Tensor] = F.sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
        self.activation = activation
        self.expansion_conv = nn.Conv2d(in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1)
        self.scale_activation = scale_activation

    def scale_batch(self, batch: Tensor) -> Tensor:
        scale = self.avgpool(batch)
        scale = self.squeeze_conv(scale)
        scale = self.activation(scale)
        scale = self.expansion_conv(scale)
        return self.scale_activation(scale)

    def forward(self, batch: Tensor) -> Tensor:
        scale = self.scale_batch(batch)
        return scale * batch
