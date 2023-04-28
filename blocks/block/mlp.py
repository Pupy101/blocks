from typing import Callable, Optional

from torch import Tensor, nn
from torch.nn import functional as F


class MLP2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.expansion_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, padding=0)
        self.activation = activation
        self.squeeze_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, batch: Tensor) -> Tensor:
        batch = self.expansion_conv(batch)
        batch = self.activation(batch)
        batch = self.squeeze_conv(batch)
        batch = self.dropout(batch)
        return batch
