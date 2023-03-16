from torch import Tensor, einsum, nn, rand  # pylint: disable=E0611


class CrossAttentionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        assert (
            not out_channels % num_heads
        ), f"out_channels({out_channels}) must be divided entirely num_heads ({num_heads})"

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pos_emb = nn.Parameter(rand(1, num_heads, out_channels // num_heads, 1))
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, batch: Tensor) -> Tensor:
        batch_size, _, heigth, width = batch.shape
        num_heads = self.num_heads
        head_shape = self.out_channels // self.num_heads

        query = self.conv_q(batch).view(batch_size, num_heads, head_shape, heigth * width)
        key = self.conv_k(batch).view(batch_size, num_heads, head_shape, heigth * width)
        value = self.conv_v(batch).view(batch_size, num_heads, head_shape, heigth * width)

        pos_emb = self.pos_emb.repeat(batch_size, 1, 1, heigth * width)
        query_with_pos = query + pos_emb

        attn_energy = einsum("bnci,bncj->bnij", query_with_pos, key) / head_shape**0.5
        attn_weights = nn.functional.softmax(attn_energy, dim=-1)

        output = einsum("bnij,bncj->bnci", attn_weights, value)
        output = output.reshape(batch_size, self.out_channels, heigth, width)
        output = self.out_conv(output)
        return output
