import torch
import torch.nn as nn
from .components import UpBlock, AttnUpBlock

class Decoder(nn.Module):
    def __init__(
        self,
        image_channel: int = 3,
        image_size: int = 256,
        up_block_types: list[str] = ['UpBlock', 'AttnUpBlock', 'UpBlock', 'UpBlock'],
        block_out_channels: list[int] = [512,256,128,64],
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True
    ):
        super().__init__()
        
        #assert set(up_block_types) == set(['UpBlock', 'AttnUpBlock']), \
        assert all(ub in ['UpBlock', 'AttnUpBlock'] for ub in up_block_types), \
        f'up_block_types should contain "UpBlock" or "AttnUpBlock"'
        assert len(block_out_channels) == len(up_block_types), \
        f'block_out_channels and down_block_types should be the same length'
        depth = len(block_out_channels)
        assert image_size % (2**(depth-1)) == 0, \
        f'image_size should be divisible by {2**(depth-1)}'
        assert block_out_channels[-1] % num_groups == 0, \
        f'block_out_channels[0] should be divisible by num_groups'

        decoder = []
        in_channels = block_out_channels[0]
        for i in range(depth):
            resize = i > 0
            up_block = eval(up_block_types[i])(
                in_channels = in_channels,
                out_channels = block_out_channels[i],
                num_resblocks = num_resblocks,
                num_groups = num_groups,
                use_conv = use_conv,
                resize = resize
            )
            in_channels = block_out_channels[i]
            decoder.append(up_block)
        self.decoder = nn.Sequential(*decoder)

        self.norm = nn.GroupNorm(
            num_groups = num_groups, 
            num_channels = block_out_channels[-1]
        )
        self.decoder_out = nn.Conv2d(
                in_channels = block_out_channels[-1],
                out_channels = image_channel,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = self.norm(x)
        x = self.decoder_out(x)
        return x



