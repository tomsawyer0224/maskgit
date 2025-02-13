import sys

if not "." in sys.path:
    sys.path.append(".")
import unittest
import torch
import torch.nn as nn
from models import vector_quantizer


class TestVectorQuantizer(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 512
        self.vq = vector_quantizer.VectorQuantizer(
            codebook_size=1024, latent_dim=self.latent_dim, commitment_weight=1.0
        )

    def test_VectorQuantizer(self):
        print("---test_forward---")
        z = torch.randn(4, self.latent_dim, 8, 8)
        quantized, indices, codebook_loss = self.vq(z)
        print(f"z.shape: {z.shape}")
        print(f"quantized.shape :{quantized.shape}, quantized.dtype :{quantized.dtype}")
        print(f"indices.shape: {indices.shape}, indices.dtype: {indices.dtype}")
        print(f"codebook_loss: {codebook_loss}, {codebook_loss.grad_fn}")
        print()

        print("---test_get_quantized_from_indices---")
        indices = torch.randint(0, 1024, (8, 64))
        quantized = self.vq.get_quantized_from_indices(indices)
        print(f"indices.shape: {indices.shape}")
        print(f"quantized.shape: {quantized.shape}")
        print()


if __name__ == "__main__":
    unittest.main()
