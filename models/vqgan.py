import sys

if not "." in sys.path:
    sys.path.append(".")
import torch
import torch.nn as nn
from einops import rearrange

from .encoder import Encoder
from .decoder import Decoder
from loss.lpips import LPIPS

from .discriminator import NLayerDiscriminator
from .vector_quantizer import VectorQuantizer, VectorQuantizer2


class VQGAN(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, any],
        decoder_config: dict[str, any],
        vq_config: dict[str, any],
        discriminator_config: dict[str, any],
        loss_config: dict[str, any],
    ) -> None:
        """

        args:
            encoder_config, decoder_config, vq_config, discriminator_config:
                the dict configs with extra parameter "name" = class name
            loss_config = {
                'discriminator_weight': 0.8,
                'discriminator_start': 10000,
                'discriminator_factor': 1.0,
                'codebook_weight': 1.0,
                'reconstruction_weight': 1.0,
                'perceptual_weight': 1.0,
            }
        """
        super().__init__()

        enc_block_out_channels = encoder_config["block_out_channels"]
        dec_block_out_channels = decoder_config["block_out_channels"]
        assert len(enc_block_out_channels) == len(
            enc_block_out_channels
        ), f"encoder and decoder should be the same number of blocks, number of block out channels"

        encoder_name = encoder_config["name"]
        encoder_config = {k: v for k, v in encoder_config.items() if k != "name"}
        self.encoder = eval(encoder_name)(**encoder_config)

        self.vq_in = nn.Sequential(
            nn.GroupNorm(
                num_groups=encoder_config["num_groups"],
                num_channels=encoder_config["block_out_channels"][-1],
            ),
            nn.Conv2d(
                in_channels=encoder_config["block_out_channels"][-1],
                out_channels=vq_config["latent_dim"],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        vq_name = vq_config["name"]
        vq_config = {k: v for k, v in vq_config.items() if k != "name"}
        self.vq = eval(vq_name)(**vq_config)

        self.vq_out = nn.Sequential(
            nn.GroupNorm(
                num_groups=decoder_config["num_groups"],
                num_channels=vq_config["latent_dim"],
            ),
            nn.Conv2d(
                in_channels=vq_config["latent_dim"],
                out_channels=decoder_config["block_out_channels"][0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder_name = decoder_config["name"]
        decoder_config = {k: v for k, v in decoder_config.items() if k != "name"}
        self.decoder = eval(decoder_name)(**decoder_config)

        discriminator_name = discriminator_config["name"]
        discriminator_config = {
            k: v for k, v in discriminator_config.items() if k != "name"
        }
        self.discriminator = eval(discriminator_name)(**discriminator_config)
        self.disc_factor = 0.0

        self.loss_config = loss_config
        self.lpips_loss = LPIPS().eval()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        args:
            x: (B,C,H,W)
        returns:
            tuple (quantized, indices, codebook_loss)
        """
        z = self.encoder(x)
        z = self.vq_in(z)
        quantized, indices, codebook_loss = self.vq(z)
        return quantized, indices, codebook_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        args:
            quantized: quantized image of shape (B,latent_dim,h,w)
        returns:
            decoded image: (B,C,H,W)
        """
        z_q = self.vq_out(quantized)
        decoded = self.decoder(z_q)
        return decoded

    @torch.no_grad
    def indices_to_image(self, indices: torch.LongTensor) -> torch.Tensor:
        self.eval()
        quantized = self.vq.get_quantized_from_indices(indices=indices)
        decoded_image = self.decode(quantized=quantized)
        decoded_image = decoded_image.clip(-1.0, 1.0)
        decoded_image = (decoded_image + 1.0) / 2.0
        self.train()
        return decoded_image

    def forward(self, x: torch.Tensor, **kwargs):
        quantized, indices, codebook_loss = self.encode(x)
        # pass through discriminator
        decoded_image = self.decode(quantized)
        real_logit = self.discriminator(x)
        fake_logit = self.discriminator(decoded_image)
        return {
            "real_image": x,
            "decoded_image": decoded_image,
            "real_logit": real_logit,
            "fake_logit": fake_logit,
            "codebook_loss": codebook_loss,
        }

    def reconstruct(self, x: torch.Tensor):
        # self.eval()
        with torch.no_grad():
            outputs = self(x)
            decoded_image = outputs["decoded_image"]
        decoded_image = decoded_image.clip(-1.0, 1.0)
        decoded_image = (decoded_image + 1.0) / 2.0
        # self.train()
        return decoded_image

    def adopt_disc_factor(self, disc_factor, global_step, threshold):
        if global_step < threshold:
            self.disc_factor = 0.0
        else:
            self.disc_factor = disc_factor
        return self.disc_factor

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.decoder_out
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[
            0
        ]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        lmd = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        lmd = torch.clamp(lmd, 0, 1e4).detach()
        return self.loss_config["discriminator_weight"] * lmd

    def hinge_d_loss(self, real_logit, fake_logit):
        real_loss = torch.mean(nn.functional.relu(1.0 - real_logit))
        fake_loss = torch.mean(nn.functional.relu(1.0 + fake_logit))
        d_loss = 0.5 * (real_loss + fake_loss)
        return d_loss

    def loss_fn(self, real_image, decoded_image, real_logit, fake_logit, codebook_loss):
        perceptual_loss = self.lpips_loss(real_image, decoded_image)
        reconstruction_loss = torch.abs(real_image - decoded_image)
        nll_loss = (
            self.loss_config["perceptual_weight"] * perceptual_loss
            + self.loss_config["reconstruction_weight"] * reconstruction_loss
        )
        nll_loss = nll_loss.mean()
        g_loss = -torch.mean(fake_logit)

        if self.training:
            lmd = self.calculate_lambda(nll_loss, g_loss)
        else:
            lmd = 0.0
        vq_loss = (
            nll_loss
            + self.loss_config["codebook_weight"] * codebook_loss
            + self.disc_factor * lmd * g_loss
        )
        gan_loss = self.disc_factor * self.hinge_d_loss(real_logit, fake_logit)

        return {
            "perceptual_loss": perceptual_loss.mean(),
            "reconstruction_loss": reconstruction_loss.mean(),
            "nll_loss": nll_loss,
            "g_loss": g_loss,
            "lmd": lmd,
            "codebook_loss": codebook_loss,
            "vq_loss": vq_loss,
            "gan_loss": gan_loss,
        }
