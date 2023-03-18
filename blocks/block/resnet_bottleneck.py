from typing import Optional, Tuple, Union

from torch import Tensor, nn

from blocks.layer import DWConv2d


class BaseResNetBottleneck(nn.Module):
    conv1: Union[nn.Conv2d, DWConv2d]
    bn1: nn.BatchNorm2d
    conv2: Union[nn.Conv2d, DWConv2d]
    bn2: nn.BatchNorm2d
    conv3: Union[nn.Conv2d, DWConv2d]
    bn3: nn.BatchNorm2d
    relu: nn.ReLU
    downsample: Optional[nn.Sequential]

    def forward(self, batch: Tensor) -> Tensor:
        identity = batch

        output = self.conv1(batch)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            identity = self.downsample(batch)

        output += identity
        output = self.relu(output)

        return output


class ResNetBottleneck2d(BaseResNetBottleneck):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        in_channels: int,
        out_channels: Tuple[int, int, int],
        stride: int = 1,
        expansion: int = 4,
    ):
        super().__init__()
        out_channels1, out_channels2, out_channels3 = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels1)

        self.conv2 = nn.Conv2d(
            out_channels1, out_channels2, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels2)

        self.conv3 = nn.Conv2d(out_channels2, out_channels3 * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels3 * expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or in_channels != out_channels3 * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels3 * expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels3 * expansion),
            )
        else:
            self.downsample = None


class ResNetDWBottleneck2d(BaseResNetBottleneck):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        in_channels: int,
        out_channels: Tuple[int, int, int],
        stride: int = 1,
        expansion: int = 4,
    ):
        super().__init__()
        out_channels1, out_channels2, out_channels3 = out_channels

        self.conv1 = DWConv2d(in_channels, out_channels1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels1)

        self.conv2 = DWConv2d(
            out_channels1, out_channels2, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels2)

        self.conv3 = DWConv2d(out_channels2, out_channels3 * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels3 * expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or in_channels != out_channels3 * expansion:
            self.downsample = nn.Sequential(
                DWConv2d(
                    in_channels, out_channels3 * expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels3 * expansion),
            )
        else:
            self.downsample = None
