import inspect

import torch
from anemoi.training.utils.debug_hydra import instantiate_debug
from torch import nn

from vector_quantize_pytorch import VectorQuantize


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Improved Initialization
        if self.embedding_dim % 2 == 0:
            nn.init.orthogonal_(self.embedding.weight)
        else:
            # If embedding_dim is not even, fallback to Xavier Uniform
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)

        # Alternatively, you can use uniform initialization with a specific range
        # bound = 1 / math.sqrt(self.embedding_dim)
        # nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, z):

        torch._assert(
            z.shape[-1] == self.embedding_dim,
            f"Input shape {z.shape[-1]} does not match embedding dimension {self.embedding_dim}",
        )

        z_flattened = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = torch.index_select(self.embedding.weight, 0, encoding_indices.view(-1))
        quantized = quantized.view(z.size())
        return quantized, encoding_indices


class VectorQuantizerLR(nn.Module):
    # Lucid Rains VQ implementations
    def __init__(self, vq_config):
        super().__init__()
        # input_dim = input_dim if input_dim is not None else embedding_dim

        if hasattr(vq_config, "codebook_size") and isinstance(vq_config.codebook_size, list):
            vq_config.codebook_size = tuple(vq_config.codebook_size)

        self.vq = instantiate_debug(vq_config, manual_in_place_optimizer_update=True)
        # Remember to add use_ddp is True when doing  multi node

        self.has_returns_loss_breakdown_arg = "return_loss_breakdown" in inspect.signature(self.vq.forward).parameters

    def forward(self, z, sample_codebook_temp=1.0, stochastic_sample_codes=False, freeze_codebook=False):

        # NOTE: stochastic sampling/non sampling from the latent space is not supported for the VectorQuantizerLR when trained with nonsampling / stochastic sampling
        if self.has_returns_loss_breakdown_arg:
            quantized, indices, loss, loss_breakdown = self.vq(
                z,
                freeze_codebook=freeze_codebook,
                return_loss_breakdown=True,
            )
            map_loss_breakdown = {k: getattr(loss_breakdown, k) for k in loss_breakdown._fields}
            map_loss_breakdown["loss"] = loss

        else:
            quantized, all_indices, latent_loss = self.vq(
                z,
                freeze_codebook=freeze_codebook,
            )
            indices = all_indices[:, :, 0]  # When using Grouped Residual VQ getting the first layers indexes
            map_loss_breakdown = {"loss": torch.sum(latent_loss)}  # This has already been scaled by commitment_weight

        return quantized, indices, map_loss_breakdown
