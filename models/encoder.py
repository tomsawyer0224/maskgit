import torch
import torch.nn as nn
from .components import DownBlock, AttnDownBlock


class Encoder(nn.Module):
    def __init__(
        self,
        image_channel: int = 3,
        image_size: int = 256,
        down_block_types: list[str] = [
            "DownBlock",
            "DownBlock",
            "AttnDownBlock",
            "DownBlock",
        ],
        block_out_channels: list[int] = [64, 128, 256, 512],
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
    ):
        super().__init__()

        # assert set(down_block_types) == set(['DownBlock', 'AttnDownBlock']), \
        assert all(
            db in ["DownBlock", "AttnDownBlock"] for db in down_block_types
        ), f'down_block_types should contain "DownBlock" or "AttnDownBlock"'
        assert len(block_out_channels) == len(
            down_block_types
        ), f"block_out_channels and down_block_types should be the same length"
        depth = len(block_out_channels)
        assert (
            image_size % (2 ** (depth - 1)) == 0
        ), f"image_size should be divisible by {2**(depth-1)}"
        assert (
            block_out_channels[0] % num_groups == 0
        ), f"block_out_channels[0] should be divisible by num_groups"

        self.conv_in = nn.Conv2d(
            in_channels=image_channel,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        encoder = []
        in_channels = block_out_channels[0]
        for i in range(depth):
            resize = i < depth - 1
            down_block = eval(down_block_types[i])(
                in_channels=in_channels,
                out_channels=block_out_channels[i],
                num_resblocks=num_resblocks,
                num_groups=num_groups,
                use_conv=use_conv,
                resize=resize,
            )
            in_channels = block_out_channels[i]
            encoder.append(down_block)
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.encoder(x)
        return x
