import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch
import torch.nn as nn
from models import vqgan

class Test_VQGAN(unittest.TestCase):
    def setUp(self):
        encoder_config = dict(
            name = 'Encoder',
            image_channel = 3,
            image_size = 256,
            down_block_types = ['DownBlock', 'DownBlock', 'AttnDownBlock', 'DownBlock'],
            block_out_channels = [64,128,256,512],
            num_resblocks = 2,
            num_groups = 32,
            use_conv = True,    
        )
        decoder_config = dict(
            name = 'Decoder',
            image_channel = 3,
            image_size = 256,
            up_block_types = ['UpBlock', 'AttnUpBlock', 'UpBlock', 'UpBlock'],
            block_out_channels = [512,256,128,64],
            num_resblocks = 2,
            num_groups = 32,
            use_conv = True,
        )
        vq_config = dict(
            name = 'VectorQuantizer',
            codebook_size = 512,
            latent_dim = 256,
            commitment_weight = 0.25
        )
        discriminator_config = dict(
            name = 'NLayerDiscriminator',
            image_channel = 3, 
            ndf = 64, 
            n_layers = 3,
            norm_layer = nn.BatchNorm2d
        )
        loss_config = {
            'discriminator_weight': 0.8,
            'discriminator_start': 10000,
            'discriminator_factor': 1.0,
            'codebook_weight': 1.0,
            'reconstruction_weight': 1.0,
            'perceptual_weight': 1.0,
        }

        self.VQGAN_model = vqgan.VQGAN(
            encoder_config = encoder_config,
            decoder_config = decoder_config,
            vq_config = vq_config,
            discriminator_config = discriminator_config,
            loss_config = loss_config
        )
    def test_encode_decode(self):
        print('---test_encode---')
        x = torch.randn(4,3,128,128)
        quantized, indices, codebook_loss = self.VQGAN_model.encode(x)
        print(f'x.shape: {x.shape}')
        print(f'quantized.shape: {quantized.shape}')
        print(f'indices.shape: {indices.shape}')
        print(f'codebook_loss: {codebook_loss}, {codebook_loss.grad_fn}')
        
        print(f'---test_decode---')
        x_hat = self.VQGAN_model.decode(quantized)
        print(f'x_hat.shape: {x_hat.shape}')
        print()
        print(f'---test_indices_to_image---')
        indices1 = torch.randint_like(indices,indices.max())
        decoded_image = self.VQGAN_model.indices_to_image(indices = indices1)
        print(f'indices1.shape: {indices1.shape}')
        print(f'decoded_image: {decoded_image.shape}')
        print()
    def test_forward_loss_fn(self):
        print('---test_forward_and_loss_fn---')
        x = torch.randn(4,3,64,64)
        outputs = self.VQGAN_model(x)
        print('--outputs--')
        for k, v in outputs.items():
            print(f'{k}: {v.shape}')
        print()
        print('--losses--')
        disc_factor = self.VQGAN_model.adopt_disc_factor(1.0,10,5)
        print(f'disc_factor: {disc_factor}')
        losses = self.VQGAN_model.loss_fn(**outputs)
        for k, v in losses.items():
            print(f'{k}: {v}')
        print()


if __name__=="__main__":
    unittest.main()
