import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch
import torch.nn as nn
from models import discriminator

class TestNLayerDiscriminator(unittest.TestCase):
    def setUp(self):
        self.disc = discriminator.NLayerDiscriminator(
            image_channel = 3,
            ndf = 64,
            n_layers = 3,
            norm_layer = nn.BatchNorm2d
        )
    def test_NLayerDiscriminator(self):
        print('---test_NLayerDiscriminator---')
        x = torch.randn(4,3,128,128)
        disc_out = self.disc(x)
        print(f'x.shape: {x.shape}')
        print(f'disc_out.shape: {disc_out.shape}')
        print()
if __name__=="__main__":
    unittest.main()


