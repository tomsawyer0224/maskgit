import sys

if not "." in sys.path:
    sys.path.append(".")
import torch
import torch.nn as nn
import unittest

from models import VQGAN_Wrapper, Transformer_Wrapper


class TestPretrainedVQGAN(unittest.TestCase):
    def setUp(self):

        self.vqgan_pretrained = VQGAN_Wrapper.from_pretrained(
            "./results/vqgan/checkpoints/epoch=4-step=20.ckpt"
        )
        self.trans = Transformer_Wrapper.from_pretrained(
            "./results/transformer/checkpoints/epoch=4-step=10.ckpt"
        )

    def test_quantize(self):
        torch.manual_seed(42)
        print("---test_quantize---")
        x = torch.randn(4, 3, 32, 32)
        quantized, indices, _ = self.vqgan_pretrained.encode(x)
        print(f"quantized.shape: {quantized.shape}")
        print(f"indices.shape: {indices.shape}")
        print(indices.min(), indices.max(), torch.mean(indices.float()))
        print()

    def test_transformer_pretrained(self):
        print("---test_transformer_pretrained---")
        x = torch.randint(0, 10, (4, 64))
        output = self.trans(x)
        for k, v in output.items():
            print(f"{k}: {v.shape}")
        print()


if __name__ == "__main__":
    unittest.main()
