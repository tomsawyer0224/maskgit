import torch
import torch.nn as nn
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    - Codebook mapping: takes in an encoded image and maps each vector onto
    its closets codebook vector.
    - Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(
        self,
        codebook_size: int = 512,
        latent_dim: int = 256,
        commitment_weight: int = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight

        self.embedding = nn.Embedding(
            num_embeddings=self.codebook_size, embedding_dim=self.latent_dim
        )
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        z_flattened = rearrange(z, "b c h w -> (b h w) c")

        # distances from z to e
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        # d -> (b*h*w, codebook_size)

        min_encoding_indices = torch.argmin(d, dim=1)
        indices = rearrange(min_encoding_indices, "(b hw) -> b hw", b=b)
        # indices -> (b, h*w)
        z_q = self.embedding(indices)  # -> (b, h*w, c)
        z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)

        codebook_loss = torch.mean(
            (z_q - z.detach()) ** 2
        ) + self.commitment_weight * torch.mean((z_q.detach() - z) ** 2)
        """
        codebook_loss = torch.mean((z_q.detach() - z) ** 2) + \
                        self.commitment_weight * torch.mean((z_q - z.detach()) ** 2)
        """

        # preserve gradients
        quantized = (
            z + (z_q - z).detach()
        )  # moving average instead of hard codebook remapping

        return quantized, indices, codebook_loss

    def get_quantized_from_indices(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        args:
            indices: int tensor of shape (b, h*w)
            shape: 4-tuple (b,c,h,w)
        returns:
            quantized of shape (b,c,h,w)
        """
        b, hw = indices.shape
        h = w = int(hw**0.5)
        device = self.embedding.weight.device
        indices = indices.to(device)
        z_q = self.embedding(indices)  # -> (b, h*w, c)
        quantized = rearrange(z_q, "b (h w) c -> b c h w", h=h)  # -> (b,c,h,w)
        return quantized


class VectorQuantizer2(nn.Module):
    """
    - Codebook mapping: takes in an encoded image and maps each vector onto
    its closets codebook vector.
    - Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(
        self,
        codebook_size: int = 512,
        latent_dim: int = 256,
        commitment_weight: int = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight

        self.embedding = nn.Embedding(
            num_embeddings=self.codebook_size, embedding_dim=self.latent_dim
        )
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        # distances from z to embeddings e (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        codebook_loss = self.commitment_weight * torch.mean(
            (z_q.detach() - z) ** 2
        ) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        quantized = z_q.permute(0, 3, 1, 2).contiguous()

        indices = min_encoding_indices.view(z.shape[:-1])

        return quantized, indices, codebook_loss

    def get_quantized_from_indice(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        args:
            indices: int tensor of shape (b, h*w)
            shape: 4-tuple (b,c,h,w)
        returns:
            quantized of shape (b,c,h,w)
        """
        b, hw = indices.shape
        h = w = int(hw**0.5)
        device = self.embedding.weight.device
        indices = indices.to(device)
        z_q = self.embedding(indices)  # -> (b, h*w, c)
        quantized = rearrange(z_q, "b (h w) c -> b c h w", h=h)  # -> (b,c,h,w)
        return quantized
