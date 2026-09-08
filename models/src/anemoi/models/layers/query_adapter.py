# (C) Copyright 2026 Anemoi contributors.

"""Value and metadata adapters for the experimental query model."""

from __future__ import annotations

import torch
from torch import nn

CONTINUOUS_METADATA = (
    "log_pressure",
    "pressure_applies",
    "pressure_known",
    "model_level",
    "model_level_applies",
    "model_level_known",
    "height_m",
    "height_applies",
    "height_known",
    "time_offset_hours",
    "input_cadence_hours",
    "input_cadence_known",
    "output_frequency_hours",
    "output_frequency_known",
    "temporal_aggregation_window_start_hours",
    "temporal_aggregation_window_end_hours",
    "temporal_aggregation_window_applies",
    "grid_spacing_km",
    "grid_spacing_known",
    "spatial_support_km",
    "spatial_support_known",
    "level_surface",
    "level_pressure",
    "level_model",
    "level_height",
    "level_layer",
    "level_unknown",
    "aggregation_type_instantaneous",
    "aggregation_type_mean",
    "aggregation_type_accumulation",
    "aggregation_type_other",
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
        num_units: int,
        node_chunk_size: int,
    ) -> None:
        super().__init__()
        self.variable_embedding = nn.Embedding(num_variables, hidden_dim)
        self.provenance_embedding = nn.Embedding(num_provenances, hidden_dim)
        self.unit_embedding = nn.Embedding(num_units, hidden_dim)
        self.node_chunk_size = node_chunk_size
        self.net = nn.Sequential(
            nn.Linear(metadata_dim + 1 + 3 * hidden_dim, hidden_dim),
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
        unit_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool a variable-sized set of fields independently at every node."""
        variable_embedding = self.variable_embedding(variable_ids)[:, None, :, :]
        provenance_embedding = self.provenance_embedding(provenance_ids)[:, None, :, :]
        unit_embedding = self.unit_embedding(unit_ids)[:, None, :, :]
        pooled_chunks = []
        for start in range(0, values.shape[1], self.node_chunk_size):
            stop = min(start + self.node_chunk_size, values.shape[1])
            nodes = stop - start
            chunk_mask = mask[:, start:stop]
            chunk_metadata = (
                metadata[:, None, :, :].expand(-1, nodes, -1, -1)
                if metadata.ndim == 3
                else metadata[:, start:stop]
            )
            encoded = self.net(
                torch.cat(
                    (
                        values[:, start:stop].unsqueeze(-1),
                        chunk_metadata,
                        variable_embedding.expand(-1, nodes, -1, -1),
                        provenance_embedding.expand(-1, nodes, -1, -1),
                        unit_embedding.expand(-1, nodes, -1, -1),
                    ),
                    dim=-1,
                )
            )
            scores = (
                self.score(encoded).squeeze(-1).masked_fill(~chunk_mask, -torch.inf)
            )
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

    @torch.no_grad()
    def diagnostic_stages(
        self,
        values: torch.Tensor,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        mask: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expose bounded, detached stages without installing forward hooks."""
        values = values[:, node_indices]
        mask = mask[:, node_indices]
        nodes = values.shape[1]
        continuous = (
            metadata[:, None].expand(-1, nodes, -1, -1)
            if metadata.ndim == 3
            else metadata[:, node_indices]
        )
        variable = self.variable_embedding(variable_ids)[:, None].expand(
            -1, nodes, -1, -1
        )
        provenance = self.provenance_embedding(provenance_ids)[:, None].expand(
            -1, nodes, -1, -1
        )
        unit = self.unit_embedding(unit_ids)[:, None].expand(-1, nodes, -1, -1)
        joint_input = torch.cat(
            (values.unsqueeze(-1), continuous, variable, provenance, unit), dim=-1
        )
        joint = self.net(joint_input)
        scores = self.score(joint).squeeze(-1).masked_fill(~mask, -torch.inf)
        all_missing = ~mask.any(dim=-1, keepdim=True)
        weights = torch.softmax(scores.masked_fill(all_missing, 0), dim=-1).masked_fill(
            ~mask, 0
        )
        pooled = (joint * weights.unsqueeze(-1)).sum(dim=-2)
        return {
            "continuous": continuous.detach(),
            "variable": variable.detach(),
            "provenance": provenance.detach(),
            "unit": unit.detach(),
            "joint_input": joint_input.detach(),
            "joint": joint.detach(),
            "pooled": pooled.detach(),
            "coverage": mask.any(dim=-1).detach(),
        }


class QueryMetadataAdapter(nn.Module):
    """Embed a requested field without tying it to a training geometry."""

    def __init__(
        self,
        metadata_dim: int,
        hidden_dim: int,
        variable_embedding: nn.Embedding,
        provenance_embedding: nn.Embedding,
        unit_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.variable_embedding = variable_embedding
        self.provenance_embedding = provenance_embedding
        self.unit_embedding = unit_embedding
        self.net = nn.Sequential(
            nn.Linear(metadata_dim + 3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat(
                (
                    metadata,
                    self.variable_embedding(variable_ids),
                    self.provenance_embedding(provenance_ids),
                    self.unit_embedding(unit_ids),
                ),
                dim=-1,
            )
        )

    @torch.no_grad()
    def diagnostic_stages(
        self,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the actual categorical, continuous, and projected query spaces."""
        variable = self.variable_embedding(variable_ids)
        provenance = self.provenance_embedding(provenance_ids)
        unit = self.unit_embedding(unit_ids)
        joint_input = torch.cat((metadata, variable, provenance, unit), dim=-1)
        return {
            "continuous": metadata.detach(),
            "variable": variable.detach(),
            "provenance": provenance.detach(),
            "unit": unit.detach(),
            "joint_input": joint_input.detach(),
            "final": self.net(joint_input).detach(),
        }
