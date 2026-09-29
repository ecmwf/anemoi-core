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
from typing import TypeAlias

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.transport.settings import TransportSourceSettings

# Plain per-dataset payload: one stacked tensor for gridded datasets or one tensor per sample
# (a list) for sparse observation datasets. Custom source factories return these; the builder
# wraps them in sources described by the request's templates.
Data: TypeAlias = torch.Tensor | list[torch.Tensor]

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
        if isinstance(x_data, list):
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
class TransportSourceRequest:
    """Information needed to build a source field for training or sampling.

    ``templates`` describes the field to build, one source per dataset: structure, per-sample
    node counts, variables, device, dtype and grid shard sizes. Only its payload shapes are
    read, so a zero-stride template payload is enough.
    """

    templates: Batch
    default_kind: str
    custom_source_factories: dict[str, TransportSourceFactory] = field(default_factory=dict)
    model_comm_group: Optional[ProcessGroup] = None
    allowed_kinds: frozenset[str] | None = None
    error_context: str = "transport source"

    def source_factories(self) -> dict[str, Callable[[], Batch]]:
        custom = {
            kind: (lambda factory=factory: self.templates.with_data(factory()))
            for kind, factory in self.custom_source_factories.items()
        }
        return {
            # contiguous: template payloads may be zero-stride views
            "zero": lambda: self.templates.map_data(
                lambda data: torch.zeros_like(data, memory_format=torch.contiguous_format)
            ),
            "gaussian": lambda: self._gaussian(),
            **custom,
        }

    def _gaussian(self) -> Batch:
        return self.templates.with_sources(
            {name: template.randn_like(self.model_comm_group) for name, template in self.templates.items()},
        )


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

    def build(self, request: TransportSourceRequest) -> Batch:
        """Return the source field, described like ``request.templates``."""
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
        source_factories: dict[str, Callable[[], Batch]],
    ) -> Callable[[], Batch]:
        source_factory = source_factories.get(kind)
        if source_factory is not None:
            return source_factory

        msg = f"Transport source kind '{kind}' requires a source factory."
        raise ValueError(msg)

    def _scale_source(self, sources: Batch) -> Batch:
        if self.scale == 1.0:
            return sources
        return sources.map_data(lambda data: data * self.scale)

    def _postprocess_source(self, sources: Batch, request: TransportSourceRequest) -> Batch:
        noise_scale = self.noise_scale
        if noise_scale == 0.0:
            return sources

        noise = sources.with_sources(
            {name: source.randn_like(request.model_comm_group) for name, source in sources.items()},
        )
        return sources.zip_map_data(lambda data, noise_data: data + noise_data * noise_scale, noise)
