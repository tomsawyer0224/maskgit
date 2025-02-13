import sys

if not "." in sys.path:
    sys.path.append(".")
import unittest
import torch
from models import components


class TestComponents(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.in_channels = 32
        self.out_channels = 64
        self.H = 64
        self.W = 64
        self.x = torch.randn(self.B, self.in_channels, self.H, self.W)

    def test_GeneralBlock(self):
        print("---test_GeneralBlock---")
        GB = components.GeneralBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_resblocks=1,
            num_groups=32,
            use_conv=False,
            block_type=None,
            with_attention=True,
        )
        out = GB(self.x)
        print(f"x.shape: {self.x.shape}")
        print(f"out.shape: {out.shape}")
        print()

        print("---test_DownBlock---")
        down = components.DownBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
            resize=False,
        )
        out = down(self.x)
        print(f"x.shape: {self.x.shape}")
        print(f"out.shape: {out.shape}")
        print()

        print("---test_AttnDownBlock---")
        attn_down = components.AttnDownBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
            resize=False,
        )
        out = attn_down(self.x)
        print(f"x.shape: {self.x.shape}")
        print(f"out.shape: {out.shape}")
        print()

        print("---test_UpBlock---")
        up = components.UpBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
            resize=False,
        )
        out = up(self.x)
        print(f"x.shape: {self.x.shape}")
        print(f"out.shape: {out.shape}")
        print()

        print("---test_AttnUpBlock---")
        attn_up = components.AttnUpBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
            resize=False,
        )
        out = attn_up(self.x)
        print(f"x.shape: {self.x.shape}")
        print(f"out.shape: {out.shape}")
        print()


if __name__ == "__main__":
    unittest.main()
