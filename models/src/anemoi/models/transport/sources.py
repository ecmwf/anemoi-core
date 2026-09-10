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

from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.transport.data_helpers import Data
from anemoi.models.transport.data_helpers import add_data
from anemoi.models.transport.data_helpers import data_device
from anemoi.models.transport.data_helpers import data_dtype
from anemoi.models.transport.data_helpers import is_sparse_data
from anemoi.models.transport.data_helpers import randn_like_data
from anemoi.models.transport.data_helpers import scale_data
from anemoi.models.transport.random_fields import randn_with_grid_sharding
from anemoi.models.transport.settings import TransportSourceSettings

TRANSPORT_SOURCE_KINDS = frozenset({"zero", "gaussian", "reference_state"})
TransportSourceFactory = Callable[[], dict[str, Data]]


def reference_state_sampling_source(
    x: dict[str, Data],
    *,
    data_indices: dict[str, Any],
    n_step_output: int,
) -> dict[str, Data]:
    """Use the latest input state as the source field, selecting model-output variables."""
    sources = {}
    for dataset_name, x_data in x.items():
        if is_sparse_data(x_data):
            msg = (
                "reference_state transport sources are not implemented for sparse observation datasets. "
                f"Choose a non-reference source for dataset '{dataset_name}'."
            )
            raise NotImplementedError(msg)
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
    """Shape, device, and dtype used to create source data."""

    shape: tuple[int, ...] | list[tuple[int, ...]]
    device: torch.device
    dtype: torch.dtype
    grid_shard_sizes: ShardSizes = None
    is_sparse: bool = False

    @classmethod
    def from_data(cls, data: Data, grid_shard_sizes: ShardSizes = None) -> TransportSourceSpec:
        if is_sparse_data(data):
            if not data:
                msg = "Cannot infer transport source shape from an empty sparse data list."
                raise ValueError(msg)
            if grid_shard_sizes is not None:
                msg = "Grid sharding is not supported for sparse transport source data."
                raise NotImplementedError(msg)
            return cls(
                shape=[tuple(sample.shape) for sample in data],
                device=data_device(data),
                dtype=data_dtype(data),
                grid_shard_sizes=grid_shard_sizes,
                is_sparse=True,
            )
        return cls(
            shape=tuple(data.shape),
            device=data.device,
            dtype=data.dtype,
            grid_shard_sizes=grid_shard_sizes,
        )


def sampling_source_specs(
    target_template: dict[str, Data],
    *,
    num_output_channels: dict[str, int],
    grid_shard_sizes: DatasetShardSizes | None = None,
) -> dict[str, TransportSourceSpec]:
    """Infer source tensor shapes from the sampling output template."""
    specs = {}
    for dataset_name, target_data in target_template.items():
        dataset_grid_shard_sizes = grid_shard_sizes.get(dataset_name) if grid_shard_sizes is not None else None
        if is_sparse_data(target_data):
            if dataset_grid_shard_sizes is not None:
                msg = "Grid sharding is not supported for sparse transport source data."
                raise NotImplementedError(msg)
            specs[dataset_name] = TransportSourceSpec(
                shape=[(*sample.shape[:-1], num_output_channels[dataset_name]) for sample in target_data],
                device=data_device(target_data),
                dtype=data_dtype(target_data),
                grid_shard_sizes=dataset_grid_shard_sizes,
                is_sparse=True,
            )
            continue

        specs[dataset_name] = TransportSourceSpec(
            shape=(
                target_data.shape[0],
                target_data.shape[1],
                target_data.shape[2],
                target_data.shape[-2],
                num_output_channels[dataset_name],
            ),
            device=target_data.device,
            dtype=target_data.dtype,
            grid_shard_sizes=dataset_grid_shard_sizes,
        )
    return specs


@dataclass(frozen=True)
class TransportSourceRequest:
    """Information needed to build a source field for training or sampling."""

    specs: dict[str, TransportSourceSpec]
    default_kind: str
    custom_source_factories: dict[str, TransportSourceFactory] = field(default_factory=dict)
    model_comm_group: Optional[ProcessGroup] = None
    allowed_kinds: frozenset[str] | None = None
    error_context: str = "transport source"

    def source_factories(self) -> dict[str, TransportSourceFactory]:
        return {
            "zero": lambda: self._zero(self.specs),
            "gaussian": lambda: self._gaussian(self.specs, model_comm_group=self.model_comm_group),
            **self.custom_source_factories,
        }

    @classmethod
    def from_data(
        cls,
        data: dict[str, Data],
        *,
        default_kind: str,
        custom_source_factories: dict[str, TransportSourceFactory] | None = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        allowed_kinds: frozenset[str] | None = None,
        error_context: str = "transport source",
    ) -> TransportSourceRequest:
        return cls(
            specs={
                name: TransportSourceSpec.from_data(
                    dataset_data,
                    grid_shard_sizes=grid_shard_sizes.get(name) if grid_shard_sizes is not None else None,
                )
                for name, dataset_data in data.items()
            },
            default_kind=default_kind,
            custom_source_factories={} if custom_source_factories is None else custom_source_factories,
            model_comm_group=model_comm_group,
            allowed_kinds=allowed_kinds,
            error_context=error_context,
        )

    @staticmethod
    def _zero(specs: dict[str, TransportSourceSpec]) -> dict[str, Data]:
        sources: dict[str, Data] = {}
        for name, spec in specs.items():
            if spec.is_sparse:
                assert isinstance(spec.shape, list)
                sources[name] = [torch.zeros(shape, device=spec.device, dtype=spec.dtype) for shape in spec.shape]
            else:
                assert isinstance(spec.shape, tuple)
                sources[name] = torch.zeros(spec.shape, device=spec.device, dtype=spec.dtype)
        return sources

    @staticmethod
    def _gaussian(
        specs: dict[str, TransportSourceSpec],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> dict[str, Data]:
        sources: dict[str, Data] = {}
        for name, spec in specs.items():
            if spec.is_sparse:
                if spec.grid_shard_sizes is not None:
                    msg = "Grid sharding is not supported for sparse transport source data."
                    raise NotImplementedError(msg)
                assert isinstance(spec.shape, list)
                sources[name] = [torch.randn(shape, device=spec.device, dtype=spec.dtype) for shape in spec.shape]
            else:
                assert isinstance(spec.shape, tuple)
                sources[name] = randn_with_grid_sharding(
                    spec.shape,
                    device=spec.device,
                    dtype=spec.dtype,
                    model_comm_group=model_comm_group,
                    grid_shard_sizes=spec.grid_shard_sizes,
                )
        return sources


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
        return float(self.settings.noise_scale)

    def resolve_kind(self, default_kind: str) -> str:
        return default_kind if self.kind == "default" else self.kind

    def build(self, request: TransportSourceRequest) -> dict[str, Data]:
        kind = self.resolve_kind(request.default_kind)
        source_factories = request.source_factories()
        allowed_kinds = request.allowed_kinds or (TRANSPORT_SOURCE_KINDS | frozenset(source_factories))
        if kind not in allowed_kinds:
            msg = f"Transport source kind '{kind}' is not valid for {request.error_context}."
            raise ValueError(msg)

        source_factory = self._source_factory(kind, source_factories)
        return self._postprocess_source(self._scale_source(source_factory()), request)

    def _source_factory(
        self,
        kind: str,
        source_factories: dict[str, TransportSourceFactory],
    ) -> TransportSourceFactory:
        source_factory = source_factories.get(kind)
        if source_factory is not None:
            return source_factory

        msg = f"Transport source kind '{kind}' requires a source factory."
        raise ValueError(msg)

    def _scale_source(self, sources: dict[str, Data]) -> dict[str, Data]:
        if self.scale != 1.0:
            return {name: scale_data(source, self.scale) for name, source in sources.items()}
        return sources

    def _postprocess_source(
        self,
        sources: dict[str, Data],
        request: TransportSourceRequest,
    ) -> dict[str, Data]:
        noise_scale = self.noise_scale
        if noise_scale != 0.0:
            sources = {
                name: add_data(
                    source,
                    scale_data(
                        randn_like_data(
                            source,
                            model_comm_group=request.model_comm_group,
                            grid_shard_sizes=request.specs[name].grid_shard_sizes,
                        ),
                        noise_scale,
                    ),
                )
                for name, source in sources.items()
            }
        return sources
