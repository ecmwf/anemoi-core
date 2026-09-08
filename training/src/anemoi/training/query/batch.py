# (C) Copyright 2026 Anemoi contributors.

"""Tensor batch exchanged by query-aware readers, tasks and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QueryInput:
    """All available fields on one native source grid."""

    values: torch.Tensor
    metadata: torch.Tensor
    variable_ids: torch.Tensor
    provenance_ids: torch.Tensor
    unit_ids: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> QueryInput:
        return QueryInput(
            **{name: value.to(device, non_blocking=non_blocking) for name, value in self.__dict__.items()},
        )


@dataclass
class QueryBatch:
    """One metadata-defined task and its variable-sized source set.

    Metadata is numeric and has explicit applicability/missingness indicators;
    a physical zero is consequently never used as an unknown sentinel.
    """

    inputs: dict[str, QueryInput]
    query_metadata: torch.Tensor
    query_variable_id: torch.Tensor
    query_provenance_id: torch.Tensor
    query_unit_id: torch.Tensor
    target_dataset: str
    query: dict[str, Any]
    output_coordinates: torch.Tensor | None = None
    target: torch.Tensor | None = None
    target_mask: torch.Tensor | None = None
    loss_weight: torch.Tensor | None = None
    # Human-readable, CPU-only facts recorded by the sampler. The model never
    # consumes this mapping; diagnostics use it instead of reverse engineering
    # field order, valid times, units, and normalization from tensors.
    diagnostic_context: dict[str, Any] | None = None

    def to(self, device: torch.device, non_blocking: bool = False) -> QueryBatch:
        values = {}
        for name, value in self.__dict__.items():
            if name == "inputs":
                values[name] = {
                    source: source_input.to(device, non_blocking=non_blocking) for source, source_input in value.items()
                }
            elif isinstance(value, torch.Tensor):
                values[name] = value.to(device, non_blocking=non_blocking)
            else:
                values[name] = value
        return QueryBatch(**values)
