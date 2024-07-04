import torch
import torch.nn as nn
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups,num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        else:
            self.shortcut = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return x + self.block(x)
        else:
            return self.shortcut(x) + self.block(x)

class Attention(nn.Module):
    def __init__(self, in_channels: int, num_groups:int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups = num_groups,
            num_channels = in_channels
        )

        self.to_q = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.to_k = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.to_v = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            batch_first=True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')

        attn_score = self.mha(q,k,v)[0] # (b, h*w, c)
        attn_score = rearrange(attn_score, 'b (h w) c -> b c h w', h = h)

        return x + attn_score
    
class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, use_conv: bool = True, num_groups: int = 32):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.norm = nn.GroupNorm(
                    num_groups = num_groups,
                    num_channels = in_channels
                )
            self.down = nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = 3,
                    stride = 2,
                    padding = 0
                )
        else:
            self.down = nn.MaxPool2d(kernel_size = 2, stride = 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.norm(x)
            x = nn.functional.pad(
                input = x, pad = (0,1,0,1), mode = 'constant', value = 0.0
            )
            return self.down(x)
        else:
            return self.down(x)
class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, use_conv: bool = True, num_groups: int = 32):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')

        self.use_conv = use_conv
        if use_conv:
            self.norm = nn.GroupNorm(
                    num_groups = num_groups,
                    num_channels = in_channels
                )
            self.conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.use_conv:
            x = self.norm(x)
            return self.conv(x)
        else:
            return x

class GeneralBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
        block_type: str = None,
        with_attention: bool = False
    ):
        super().__init__()
        resblocks = []
        for i in range(num_resblocks):
            if i > 0:
                in_channels = out_channels
            resblocks.append(
                ResnetBlock(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    num_groups = num_groups
                )
            )
        self.resblocks = nn.ModuleList(resblocks)
        
        self.with_attention = with_attention
        if with_attention:
            attnblocks = []
            for i in range(num_resblocks):
                attnblocks.append(
                    Attention(in_channels = out_channels, num_groups = num_groups)
                )
            self.attnblocks = nn.ModuleList(attnblocks)
        
        assert block_type in ['down', 'up', None],\
        f'block_type should be "down" or "up"'
        self.block_type = block_type
        if block_type == 'down':
            self.resize = Downsample2D(
                in_channels = out_channels, 
                use_conv = use_conv, 
                num_groups = num_groups
            )
        elif block_type == 'up': # up
            self.resize = Upsample2D(
                in_channels = out_channels, 
                use_conv = use_conv,
                num_groups = num_groups
            )
        else:
            self.resize = None
    def forward(self, x: torch.Tensor):
        if self.with_attention:
            for res, attn in zip(self.resblocks, self.attnblocks):
                x = res(x)
                x = attn(x)
        else:
            for res in self.resblocks:
                x = res(x)
        if self.block_type is None:
            return x
        else:
            return self.resize(x)

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
        resize: bool = True
    ) -> None:
        super().__init__()
        block_type = 'down' if resize else None
        self.down_block = GeneralBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            num_resblocks = num_resblocks,
            num_groups = num_groups,
            use_conv = use_conv,
            block_type = block_type,
            with_attention = False
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_block(x)
        return x
class AttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
        resize: bool = True
    ) -> None:
        super().__init__()
        block_type = 'down' if resize else None
        self.attn_down_block = GeneralBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            num_resblocks = num_resblocks,
            num_groups = num_groups,
            use_conv = use_conv,
            block_type = block_type,
            with_attention = True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_down_block(x)
        return x
class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
        resize: bool = True
    ) -> None:
        super().__init__()
        block_type = 'up' if resize else None
        self.up_block = GeneralBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            num_resblocks = num_resblocks,
            num_groups = num_groups,
            use_conv = use_conv,
            block_type = block_type,
            with_attention = False
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_block(x)
        return x
class AttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        num_groups: int = 32,
        use_conv: bool = True,
        resize: bool = True
    ) -> None:
        super().__init__()
        block_type = 'up' if resize else None
        self.attn_up_block = GeneralBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            num_resblocks = num_resblocks,
            num_groups = num_groups,
            use_conv = use_conv,
            block_type = block_type,
            with_attention = True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_up_block(x)
        return x


