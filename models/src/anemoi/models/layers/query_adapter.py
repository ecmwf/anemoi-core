# (C) Copyright 2026 Anemoi contributors.

"""Value and metadata adapters for the experimental query model."""

from __future__ import annotations

import torch
from torch import nn

CONTINUOUS_METADATA = (
    "log_pressure",
    "pressure_applies",
    "pressure_known",
    "height_m",
    "height_applies",
    "height_known",
    "time_offset_hours",
    "input_cadence_hours",
    "input_cadence_known",
    "output_cadence_hours",
    "output_cadence_known",
    "interval_start_hours",
    "interval_end_hours",
    "interval_applies",
    "grid_spacing_km",
    "grid_spacing_known",
    "spatial_support_km",
    "spatial_support_known",
    "level_surface",
    "level_pressure",
    "level_height",
    "level_layer",
    "level_unknown",
    "processing_instantaneous",
    "processing_mean",
    "processing_accumulation",
    "processing_other",
)


class QueryValueAdapter(nn.Module):
    """Embed each value jointly with its metadata before aggregation.

    The caller owns canonicalisation.  This avoids fixed variable-level channel
    mappings and preserves value/field association through a nonlinear map.
    """

    def __init__(
        self,
        metadata_dim: int,
        hidden_dim: int,
        num_variables: int,
        num_provenances: int,
        node_chunk_size: int,
    ) -> None:
        super().__init__()
        self.variable_embedding = nn.Embedding(num_variables, hidden_dim)
        self.provenance_embedding = nn.Embedding(num_provenances, hidden_dim)
        self.node_chunk_size = node_chunk_size
        self.net = nn.Sequential(
            nn.Linear(metadata_dim + 1 + 2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool a variable-sized set of fields independently at every node."""
        variable_embedding = self.variable_embedding(variable_ids)[:, None, :, :]
        provenance_embedding = self.provenance_embedding(provenance_ids)[:, None, :, :]
        pooled_chunks = []
        for start in range(0, values.shape[1], self.node_chunk_size):
            stop = min(start + self.node_chunk_size, values.shape[1])
            nodes = stop - start
            chunk_mask = mask[:, start:stop]
            chunk_metadata = (
                metadata[:, None, :, :].expand(-1, nodes, -1, -1) if metadata.ndim == 3 else metadata[:, start:stop]
            )
            encoded = self.net(
                torch.cat(
                    (
                        values[:, start:stop].unsqueeze(-1),
                        chunk_metadata,
                        variable_embedding.expand(-1, nodes, -1, -1),
                        provenance_embedding.expand(-1, nodes, -1, -1),
                    ),
                    dim=-1,
                )
            )
            scores = self.score(encoded).squeeze(-1).masked_fill(~chunk_mask, -torch.inf)
            all_missing = ~chunk_mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_missing, 0)
            weights = torch.softmax(scores, dim=-1).masked_fill(~chunk_mask, 0)
            pooled = (encoded * weights.unsqueeze(-1)).sum(dim=-2)
            pooled_chunks.append(
                torch.cat(
                    (pooled, chunk_mask.any(dim=-1, keepdim=True).to(pooled.dtype)),
                    dim=-1,
                )
            )
        return torch.cat(pooled_chunks, dim=1)


class QueryMetadataAdapter(nn.Module):
    """Embed a requested field without tying it to a training geometry."""

    def __init__(
        self,
        metadata_dim: int,
        hidden_dim: int,
        variable_embedding: nn.Embedding,
        provenance_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.variable_embedding = variable_embedding
        self.provenance_embedding = provenance_embedding
        self.net = nn.Sequential(
            nn.Linear(metadata_dim + 2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat(
                (
                    metadata,
                    self.variable_embedding(variable_ids),
                    self.provenance_embedding(provenance_ids),
                ),
                dim=-1,
            )
        )
