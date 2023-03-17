# Module with PyTorch custom blocks

## __Supported__

## *Computer vision:*

### Blocks:

1. `CrossAttention2d`
2. `MLP2d`
3. `ResNetBlock2d`
4. `ResNetDWBlock2d`
5. `ResNetBottleneck2d`
6. `ResNetDWBottleneck2d`
7. `SqueezeExcitationBlock2d`

### Layers:

1. `DepthwiseConv2d`
2. `LayerNorm2d`

### Some examples:
```python
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor, nn

from blocks.block import ResNetBlock2d, ResNetBottleneck2d


@dataclass
class AsDictMixin:
    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResNetBlockConfig(AsDictMixin):
    in_channels: int
    out_channels: int
    stride: int = 1


@dataclass
class ResNetBottleneckConfig(AsDictMixin):
    in_channels: int
    out_channels: Tuple[int, int, int]
    stride: int = 1
    expansion: int = 4


@dataclass
class ResNetConfig:
    conv_1: List[Union[ResNetBlockConfig, ResNetBottleneckConfig]]
    conv_2: List[Union[ResNetBlockConfig, ResNetBottleneckConfig]]
    conv_3: List[Union[ResNetBlockConfig, ResNetBottleneckConfig]]
    conv_4: List[Union[ResNetBlockConfig, ResNetBottleneckConfig]]
    count_classes: int


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(configs=config.conv_1)
        self.layer2 = self._make_layer(configs=config.conv_2)
        self.layer3 = self._make_layer(configs=config.conv_3)
        self.layer4 = self._make_layer(configs=config.conv_4)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        last_config = config.conv_4[-1]
        if isinstance(last_config, ResNetBottleneckConfig):
            in_features = last_config.out_channels[-1]
        elif isinstance(last_config, ResNetBlockConfig):
            in_features = last_config.out_channels
        else:
            raise ValueError(f"Strange config: {config}")
        self.fc = nn.Linear(in_features=in_features, out_features=config.count_classes)

    @staticmethod
    def _make_layer(
        configs: List[Union[ResNetBlockConfig, ResNetBottleneckConfig]]
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for config in configs:
            if isinstance(config, ResNetBlockConfig):
                layers.append(ResNetBlock2d(**config.asdict()))
            elif isinstance(config, ResNetBlockConfig):
                layers.append(ResNetBottleneck2d(**config.asdict()))
            else:
                raise ValueError(f"Strange config: {config}")
        return nn.Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        output = self.maxpool(self.bn1(self.conv1(batch)))

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.fc(self.avgpool(output))

        return output


resnet18config = ResNetConfig(
    conv_1=[
        ResNetBlockConfig(in_channels=64, out_channels=64),
        ResNetBlockConfig(in_channels=64, out_channels=64),
    ],
    conv_2=[
        ResNetBlockConfig(in_channels=64, out_channels=128, stride=2),
        ResNetBlockConfig(in_channels=128, out_channels=128),
    ],
    conv_3=[
        ResNetBlockConfig(in_channels=128, out_channels=256, stride=2),
        ResNetBlockConfig(in_channels=256, out_channels=256),
    ],
    conv_4=[
        ResNetBlockConfig(in_channels=256, out_channels=512, stride=2),
        ResNetBlockConfig(in_channels=512, out_channels=512),
    ],
    count_classes=1000,
)


resnet18 = ResNet(config=resnet18config)

```