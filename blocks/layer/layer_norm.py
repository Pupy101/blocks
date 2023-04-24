from torch import Tensor, nn, ones, zeros


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(ones(1, channels, 1, 1))
        self.beta = nn.Parameter(zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, batch: Tensor) -> Tensor:
        mean = batch.mean(dim=(2, 3), keepdim=True)
        std = batch.std(dim=(2, 3), keepdim=True, unbiased=False)
        batch = (batch - mean) / (std + self.eps)
        return batch * self.gamma + self.beta
