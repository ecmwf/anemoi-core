from __future__ import annotations

import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class GridViTNodeEncoder(nn.Module):
    """Very small ViT-style encoder for flattened gridded nodes.

    The goal is to produce per-node embeddings (one per grid cell). We use
    patch tokens and broadcast each patch token to its pixels.
    """

    def __init__(
        self,
        *,
        xdim: int,
        ydim: int,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 10,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)

        # Allow non-divisible sizes by padding to the next multiple of patch_size.
        self.pad_h = (-self.ydim) % self.patch_size
        self.pad_w = (-self.xdim) % self.patch_size
        self.h_padded = self.ydim + self.pad_h
        self.w_padded = self.xdim + self.pad_w

        self.nh = self.h_padded // self.patch_size
        self.nw = self.w_padded // self.patch_size
        self.n_patches = self.nh * self.nw

        self.proj = nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, T, E, G, C] where G = xdim*ydim.

        Returns
        -------
        torch.Tensor
            Shape [B, E, G, D].
        """
        B, T, E, G, C = x.shape
        if G != self.xdim * self.ydim:
            raise ValueError(f"Expected grid={self.xdim*self.ydim} got {G}")
        if C != self.in_channels:
            raise ValueError(f"Expected channels={self.in_channels} got {C}")

        # collapse time into channels (simple, fast)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, E, T, G, C]
        x = x.reshape(B * E, T * C, self.ydim, self.xdim)  # [B*E, TC, H, W]

        if self.pad_h or self.pad_w:
            # Pad (left,right,top,bottom) for 2D spatial dims; keep channels intact.
            import torch.nn.functional as F

            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="replicate")

        # patchify
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B*E, TC, nh, nw, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B*E, nh, nw, TC, p, p]
        patches = patches.view(B * E, self.n_patches, (T * C) * p * p)

        tokens = self.proj(patches) + self.pos  # [B*E, n_patches, D]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # broadcast patch tokens back to pixels
        tokens_2d = tokens.view(B * E, self.nh, self.nw, self.embed_dim)
        tokens_up = tokens_2d.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)  # [B*E, Hpad, Wpad, D]
        # Trim padding back to original shape
        tokens_up = tokens_up[:, : self.ydim, : self.xdim, :]
        node_emb = tokens_up.contiguous().view(B, E, G, self.embed_dim)
        return node_emb


class PointMLPNodeEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        # Allow in_dim=0 in config and infer at first forward.
        d = int(in_dim)
        for _ in range(depth - 1):
            if d <= 0:
                layers += [nn.LazyLinear(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            else:
                layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E, G, C] -> collapse time into channels like ViT
        B, T, E, G, C = x.shape
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, E, G, T * C)
        out = self.net(x)
        return out
