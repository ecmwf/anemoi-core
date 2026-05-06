# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.transport.random_fields import randn_like_with_grid_sharding
from anemoi.models.transport.random_fields import randn_with_grid_sharding
from anemoi.models.transport.settings import TransportSourceSettings

TRANSPORT_SOURCE_KINDS = frozenset({"zero", "gaussian", "reference_state"})
TransportSourceFactory = Callable[[], dict[str, torch.Tensor]]


def reference_state_sampling_source(
    x: dict[str, torch.Tensor],
    *,
    data_indices: dict[str, Any],
    n_step_output: int,
) -> dict[str, torch.Tensor]:
    """Use the latest input state as the source field, selecting model-output variables."""
    sources = {}
    for dataset_name, x_data in x.items():
        output_names = data_indices[dataset_name].model.output.ordered_names
        try:
            input_positions = data_indices[dataset_name].model.input.positions_for_names(output_names)
        except ValueError as exc:
            msg = (
                "reference_state transport sources require all model-output variables "
                f"to be available in the model input for dataset '{dataset_name}'. "
                "Choose a non-reference source when this is not true."
            )
            raise ValueError(msg) from exc
        input_idx = torch.as_tensor(input_positions, device=x_data.device, dtype=torch.long)
        source = x_data[:, -1:, :, :, :].index_select(-1, input_idx)
        if n_step_output > 1:
            source = source.expand(-1, n_step_output, -1, -1, -1)
        sources[dataset_name] = source
    return sources


@dataclass(frozen=True)
class TransportSourceSpec:
    """Shape, device, and dtype used to create a source tensor."""

    shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    grid_shard_shapes: list[int] | None = None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, grid_shard_shapes: list[int] | None = None) -> TransportSourceSpec:
        return cls(
            shape=tuple(tensor.shape),
            device=tensor.device,
            dtype=tensor.dtype,
            grid_shard_shapes=grid_shard_shapes,
        )


def sampling_source_specs(
    x: dict[str, torch.Tensor],
    *,
    n_step_output: int,
    num_output_channels: dict[str, int],
    grid_shard_shapes: dict[str, Optional[list[int]]] | None = None,
) -> dict[str, TransportSourceSpec]:
    """Infer source tensor shapes from the sampling input batch."""
    return {
        dataset_name: TransportSourceSpec(
            shape=(
                x_data.shape[0],
                n_step_output,
                x_data.shape[2],
                x_data.shape[-2],
                num_output_channels[dataset_name],
            ),
            device=x_data.device,
            dtype=x_data.dtype,
            grid_shard_shapes=grid_shard_shapes.get(dataset_name) if grid_shard_shapes is not None else None,
        )
        for dataset_name, x_data in x.items()
    }


@dataclass(frozen=True)
class TransportSourceRequest:
    """Information needed to build a source field for training or sampling."""

    specs: dict[str, TransportSourceSpec]
    default_kind: str
    source_factories: dict[str, TransportSourceFactory] = field(default_factory=dict)
    model_comm_group: Optional[ProcessGroup] = None
    allowed_kinds: frozenset[str] | None = None
    error_context: str = "transport source"

    def __post_init__(self) -> None:
        factories = {
            "zero": lambda: self._zero(self.specs),
            "gaussian": lambda: self._gaussian(self.specs, model_comm_group=self.model_comm_group),
        }
        factories.update(self.source_factories)
        object.__setattr__(self, "source_factories", factories)

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        *,
        default_kind: str,
        source_factories: dict[str, TransportSourceFactory] | None = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list[int]]] | None = None,
        allowed_kinds: frozenset[str] | None = None,
        error_context: str = "transport source",
    ) -> TransportSourceRequest:
        return cls(
            specs={
                name: TransportSourceSpec.from_tensor(
                    tensor,
                    grid_shard_shapes=grid_shard_shapes.get(name) if grid_shard_shapes is not None else None,
                )
                for name, tensor in tensors.items()
            },
            default_kind=default_kind,
            source_factories={} if source_factories is None else source_factories,
            model_comm_group=model_comm_group,
            allowed_kinds=allowed_kinds,
            error_context=error_context,
        )

    @staticmethod
    def _zero(specs: dict[str, TransportSourceSpec]) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(spec.shape, device=spec.device, dtype=spec.dtype) for name, spec in specs.items()}

    @staticmethod
    def _gaussian(
        specs: dict[str, TransportSourceSpec],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> dict[str, torch.Tensor]:
        return {
            name: randn_with_grid_sharding(
                spec.shape,
                device=spec.device,
                dtype=spec.dtype,
                model_comm_group=model_comm_group,
                grid_shard_shapes=spec.grid_shard_shapes,
            )
            for name, spec in specs.items()
        }


class TransportSourceBuilder:
    """Build source fields such as Gaussian noise, zeros, or the latest input state."""

    def __init__(self, settings: TransportSourceSettings | None = None) -> None:
        self.settings = settings or TransportSourceSettings()

    @classmethod
    def from_config(cls, config: Any) -> TransportSourceBuilder:
        return cls(TransportSourceSettings.from_config(config))

    @property
    def kind(self) -> str:
        return self.settings.kind

    @property
    def scale(self) -> float:
        return float(self.settings.scale)

    @property
    def noise_scale(self) -> float:
        return float(getattr(self.settings, "noise_scale", 0.0))

    def resolve_kind(self, default_kind: str) -> str:
        return default_kind if self.kind == "default" else self.kind

    def build(self, request: TransportSourceRequest) -> dict[str, torch.Tensor]:
        kind = self.resolve_kind(request.default_kind)
        allowed_kinds = request.allowed_kinds or (TRANSPORT_SOURCE_KINDS | frozenset(request.source_factories))
        if kind not in allowed_kinds:
            msg = f"Transport source kind '{kind}' is not valid for {request.error_context}."
            raise ValueError(msg)

        source_factory = self._source_factory(kind, request)
        return self._postprocess_source(self._scale_source(source_factory()), request)

    def _source_factory(self, kind: str, request: TransportSourceRequest) -> TransportSourceFactory:
        source_factory = request.source_factories.get(kind)
        if source_factory is not None:
            return source_factory

        msg = f"Transport source kind '{kind}' requires a source factory."
        raise ValueError(msg)

    def _scale_source(self, sources: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.scale != 1.0:
            return {name: self.scale * source for name, source in sources.items()}
        return sources

    def _postprocess_source(
        self,
        sources: dict[str, torch.Tensor],
        request: TransportSourceRequest,
    ) -> dict[str, torch.Tensor]:
        noise_scale = self.noise_scale
        if noise_scale != 0.0:
            sources = {
                name: source
                + noise_scale
                * randn_like_with_grid_sharding(
                    source,
                    model_comm_group=request.model_comm_group,
                    grid_shard_shapes=request.specs[name].grid_shard_shapes,
                )
                for name, source in sources.items()
            }
        return sources
