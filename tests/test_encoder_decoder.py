import sys

if not "." in sys.path:
    sys.path.append(".")
import unittest
import torch
from models import encoder, decoder


class TestEncoder(unittest.TestCase):
    def setUp(self):
        pass

    def test_encoder_decoder(self):
        print("---test_encoder_decoder---")
        enc = encoder.Encoder(
            image_channel=3,
            image_size=256,
            down_block_types=["DownBlock", "DownBlock", "AttnDownBlock", "DownBlock"],
            block_out_channels=[64, 128, 256, 512],
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
        )
        dec = decoder.Decoder(
            image_channel=3,
            image_size=256,
            up_block_types=["UpBlock", "AttnUpBlock", "UpBlock", "UpBlock"],
            block_out_channels=[512, 256, 128, 64],
            num_resblocks=2,
            num_groups=32,
            use_conv=True,
        )

        x = torch.randn(4, 3, 64, 64)
        enc_out = enc(x)
        dec_out = dec(enc_out)
        print(f"x.shape: {x.shape}")
        print(f"enc_out.shape: {enc_out.shape}")
        print(f"dec_out.shape: {dec_out.shape}")
        print(
            f"dec_out range: ({round(dec_out.min().item(),2)}, {round(dec_out.max().item(),2)})"
        )
        print()


if __name__ == "__main__":
    unittest.main()
