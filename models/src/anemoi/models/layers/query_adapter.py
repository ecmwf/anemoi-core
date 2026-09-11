# (C) Copyright 2026 Anemoi contributors.

"""Value and metadata adapters for the experimental query model."""

from __future__ import annotations

import math

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
    "bbox_center_latitude_sin",
    "bbox_center_latitude_cos",
    "bbox_center_longitude_sin",
    "bbox_center_longitude_cos",
    "bbox_latitude_span",
    "bbox_longitude_span",
    "bbox_known",
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

TIME_OFFSET_NORMALIZATION_HOURS = 168.0
_TIME_OFFSET_INDEX = CONTINUOUS_METADATA.index("time_offset_hours")


def encode_bbox_metadata(
    bbox: tuple[float, float, float, float] | list[float] | None,
) -> dict[str, float]:
    """Encode a degree bbox without a longitude discontinuity at the dateline."""
    if bbox is None:
        return {
            "bbox_center_latitude_sin": 0.0,
            "bbox_center_latitude_cos": 0.0,
            "bbox_center_longitude_sin": 0.0,
            "bbox_center_longitude_cos": 0.0,
            "bbox_latitude_span": 0.0,
            "bbox_longitude_span": 0.0,
            "bbox_known": 0.0,
        }
    west, south, east, north = (float(value) for value in bbox)
    latitude = math.radians((south + north) / 2)
    longitude = math.radians((west + east) / 2)
    return {
        "bbox_center_latitude_sin": math.sin(latitude),
        "bbox_center_latitude_cos": math.cos(latitude),
        "bbox_center_longitude_sin": math.sin(longitude),
        "bbox_center_longitude_cos": math.cos(longitude),
        "bbox_latitude_span": (north - south) / 180,
        "bbox_longitude_span": (east - west) / 360,
        "bbox_known": 1.0,
    }


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
        num_grids: int,
        lead_time_fourier_features: int,
        lead_time_min_period_hours: float,
        lead_time_max_period_hours: float,
    ) -> None:
        super().__init__()
        self.variable_embedding = variable_embedding
        self.provenance_embedding = provenance_embedding
        self.unit_embedding = unit_embedding
        self.grid_embedding = nn.Embedding(num_grids, hidden_dim)
        if lead_time_min_period_hours > lead_time_max_period_hours:
            raise ValueError(
                "lead_time_min_period_hours must not exceed lead_time_max_period_hours."
            )
        periods = torch.logspace(
            math.log10(lead_time_min_period_hours),
            math.log10(lead_time_max_period_hours),
            lead_time_fourier_features,
        )
        self.register_buffer("lead_time_periods_hours", periods, persistent=True)
        self.lead_time_encoder = nn.Sequential(
            nn.Linear(1 + 2 * lead_time_fourier_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(metadata_dim - 1 + 5 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _lead_time_features(
        self, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate continuous lead time from metadata and encode it smoothly."""
        normalized = metadata[..., _TIME_OFFSET_INDEX : _TIME_OFFSET_INDEX + 1]
        lead_time_hours = normalized * TIME_OFFSET_NORMALIZATION_HOURS
        phase = 2 * math.pi * lead_time_hours / self.lead_time_periods_hours
        features = torch.cat((normalized, torch.sin(phase), torch.cos(phase)), dim=-1)
        remaining = torch.cat(
            (
                metadata[..., :_TIME_OFFSET_INDEX],
                metadata[..., _TIME_OFFSET_INDEX + 1 :],
            ),
            dim=-1,
        )
        return remaining, lead_time_hours, self.lead_time_encoder(features)

    def forward(
        self,
        metadata: torch.Tensor,
        variable_ids: torch.Tensor,
        provenance_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        grid_ids: torch.Tensor,
    ) -> torch.Tensor:
        remaining_metadata, _, lead_time_embedding = self._lead_time_features(metadata)
        return self.net(
            torch.cat(
                (
                    remaining_metadata,
                    self.variable_embedding(variable_ids),
                    self.provenance_embedding(provenance_ids),
                    self.unit_embedding(unit_ids),
                    self.grid_embedding(grid_ids),
                    lead_time_embedding,
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
        grid_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the actual categorical, continuous, and projected query spaces."""
        variable = self.variable_embedding(variable_ids)
        provenance = self.provenance_embedding(provenance_ids)
        unit = self.unit_embedding(unit_ids)
        grid = self.grid_embedding(grid_ids)
        remaining_metadata, lead_time_hours, lead_time_embedding = (
            self._lead_time_features(metadata)
        )
        joint_input = torch.cat(
            (remaining_metadata, variable, provenance, unit, grid, lead_time_embedding),
            dim=-1,
        )
        return {
            "continuous": metadata.detach(),
            "lead_time_hours": lead_time_hours.detach(),
            "lead_time_embedding": lead_time_embedding.detach(),
            "variable": variable.detach(),
            "provenance": provenance.detach(),
            "unit": unit.detach(),
            "grid": grid.detach(),
            "joint_input": joint_input.detach(),
            "final": self.net(joint_input).detach(),
        }
