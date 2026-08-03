# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import einops
import torch
from torch import Tensor

from anemoi.models.distributed.graph import shard_tensor

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.distributed.shapes import ShardSizes
    from anemoi.models.models.base import BaseGraphModel

LOGGER = logging.getLogger(__name__)

TARGET_FEATURE_REGISTRY: dict[str, type[DecodingTargetFeature]] = {}


def register_target_feature(name: str):
    """Register a :class:`DecodingTargetFeature` subclass under ``name``."""

    def decorator(cls: type[DecodingTargetFeature]) -> type[DecodingTargetFeature]:
        if name in TARGET_FEATURE_REGISTRY:
            raise ValueError(f"Target feature '{name}' is already registered.")
        TARGET_FEATURE_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


class DecodingTargetFeature(ABC):
    """Base class for a decoder target feature.

    A feature is bound once to its ``(model, dataset_name)`` context at construction time
    (in ``BaseGraphModel._build_decoder_routing``), so its ``dim`` and ``tensor`` do not need
    the context threaded through on every call.

    Attributes
    ----------
    needs_sharding : bool
        Whether this feature must be explicitly sharded:
        - `True` for features derived from full-size node attributes
        - `False` for features already sharded via the input tensor or the encoder
    name : str
        Registry key, set by :func:`register_target_feature`.
    """

    needs_sharding: bool = False
    name: str = ""

    def __init__(self, model: BaseGraphModel, datasets_names: list[str]) -> None:
        self.model = model
        self.datasets_names = datasets_names

    def validate(self) -> None:
        """Check build-time preconditions and raise if the feature is misconfigured.

        Called once at construction so misconfiguration fails at model init rather than on the
        first batch. No-op by default.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Feature width contributed to the decoder target input."""

    @abstractmethod
    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        """Compute the (unsharded) feature tensor of shape ``(batch*ensemble*grid, dim)``."""

    def tensor(
        self,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        grid_shard_sizes: ShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> Tensor:
        """Return the feature tensor, sharding it iff the feature is defined on full-size nodes."""
        assert (
            dataset_name is not None
        ), f"dataset_name must be provided to {self.__class__.__name__}.tensor() for sharding and validation."
        out = self._compute(x_input_data, x_encoded_data, batch_size=batch_size, dataset_name=dataset_name)
        if self.needs_sharding and grid_shard_sizes is not None:
            out = shard_tensor(out, 0, grid_shard_sizes, model_comm_group)
        return out


@register_target_feature("coordinates")
class CoordinatesFeature(DecodingTargetFeature):
    """Sin/cos encoded lat-lon coordinates."""

    needs_sharding = True

    def validate(self) -> None:
        num_coords_dim = {}
        for dataset_name in self.datasets_names:
            num_coords_dim[dataset_name] = getattr(self.model.node_attributes, f"latlons_{dataset_name}").shape[1]

        assert len(set(num_coords_dim.values())) == 1, (
            f"Coordinates feature must have the same dimension across all datasets encoded with the same encoder. "
            f"Found dimensions: {num_coords_dim}"
        )

    @cached_property
    def dim(self) -> int:
        return getattr(self.model.node_attributes, f"latlons_{self.datasets_names[0]}").shape[1]

    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        coords = getattr(self.model.node_attributes, f"latlons_{dataset_name}")
        return einops.repeat(coords, "e f -> (repeat e) f", repeat=batch_size)


@register_target_feature("forcings")
class InputForcingsFeature(DecodingTargetFeature):
    """Forcing variables over the output timestep window."""

    needs_sharding = False

    def validate(self) -> None:
        num_trainable_params = {}
        for dataset_name in self.datasets_names:
            num_trainable_params[dataset_name] = self.model.num_input_channels_forcings[dataset_name]

        assert len(set(num_trainable_params.values())) == 1, (
            f"Forcings feature must have the same dimension across all datasets encoded with the same encoder. "
            f"Found dimensions: {num_trainable_params}"
        )

    @cached_property
    def dim(self) -> int:
        return self.model.n_step_output * self.model.num_input_channels_forcings[self.datasets_names[0]]

    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        indices = self.model._forcing_input_idx[dataset_name]
        x_forcing = x_input_data[:, : self.model.n_step_output, ..., indices]
        return einops.rearrange(x_forcing, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")


@register_target_feature("prognostics")
class PrognosticsFeature(DecodingTargetFeature):
    """Prognostic variables over the input timestep window."""

    needs_sharding = False

    def validate(self) -> None:
        num_trainable_params = {}
        for dataset_name in self.datasets_names:
            num_trainable_params[dataset_name] = self.model.num_input_channels_prognostic[dataset_name]

        assert len(set(num_trainable_params.values())) == 1, (
            f"Prognostics feature must have the same dimension across all datasets encoded with the same encoder. "
            f"Found dimensions: {num_trainable_params}"
        )

    @cached_property
    def dim(self) -> int:
        return self.model.n_step_input * self.model.num_input_channels_prognostic[self.datasets_names[0]]

    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        indices = self.model._internal_input_idx[dataset_name]
        x_prog = x_input_data[:, : self.model.n_step_input, ..., indices]
        return einops.rearrange(x_prog, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")


@register_target_feature("trainable_parameters")
class TrainableParametersFeature(DecodingTargetFeature):
    """Learnable per-node parameters."""

    needs_sharding = True

    def validate(self) -> None:
        num_trainable_params = {}
        for dataset_name in self.datasets_names:
            if self.model.node_attributes.trainable_tensors[dataset_name].trainable is None:
                decoder_name = self.model.dataset2decoder[dataset_name]
                error_msg = (
                    f"No trainable parameters configured for dataset '{dataset_name}'. "
                    f"Set trainable_parameters.data > 0 (for dataset '{dataset_name}') "
                    f"or remove '{self.name}' from decoder.{decoder_name}.input_target_features."
                )
                raise ValueError(error_msg)
            num_trainable_params[dataset_name] = self.model.node_attributes.num_trainable_parameters[dataset_name]

        assert len(set(num_trainable_params.values())) == 1, (
            f"Trainable parameters feature must have the same dimension across all datasets encoded with the"
            f" same encoder. Found dimensions: {num_trainable_params}"
        )

    @cached_property
    def dim(self) -> int:
        return self.model.node_attributes.num_trainable_parameters[self.datasets_names[0]]

    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        trainable = self.model.node_attributes.trainable_tensors[dataset_name].trainable
        return einops.repeat(trainable, "e f -> (repeat e) f", repeat=batch_size)


@register_target_feature("encoded_data")
class EncodedDataFeature(DecodingTargetFeature):
    """Encoder-updated data tensor (full encoder output on data nodes)."""

    needs_sharding = False

    def validate(self) -> None:
        input_dims = {}
        for dataset_name in self.datasets_names:
            if dataset_name not in self.model.input_datasets:
                decoder_name = self.model.dataset2decoder[dataset_name]
                raise ValueError(
                    f'"{self.name}" requires dataset "{dataset_name}" to have an encoder. '
                    f"Update decoder.{decoder_name}.input_target_features."
                )
            input_dims[dataset_name] = self.model.input_dim[dataset_name]
        assert len(set(input_dims.values())) == 1, (
            f"Encoded data feature must have the same dimension across all datasets encoded with the same encoder. "
            f"Found dimensions: {input_dims}"
        )

    @cached_property
    def dim(self) -> int:
        return self.model.input_dim[self.datasets_names[0]]

    def _compute(
        self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int, dataset_name: str
    ) -> Tensor:
        if x_encoded_data is None:
            raise ValueError(f"'{self.name}' requires the encoder output for dataset '{dataset_name}'.")

        return x_encoded_data


class CompositeTargetFeature(DecodingTargetFeature):
    """Per-dataset aggregate of decoder target features.

    Wraps the ordered list of features declared for a decoder and exposes the same
    :class:`DecodingTargetFeature` interface, so callers use ``.dim`` and ``.tensor(...)`` without
    knowing whether they hold a single feature or many. Concatenation and per-feature sharding are
    owned here.
    """

    def __init__(self, model: BaseGraphModel, datasets_names: list[str], feature_names: list[str]) -> None:
        assert (
            feature_names
        ), f"Decoder for datasets {', '.join(datasets_names)} must declare at least one target feature."
        self.features = [TARGET_FEATURE_REGISTRY[name](model, datasets_names) for name in feature_names]

    def validate(self) -> None:
        """Assert the decoder's target features are valid and consistent across all its datasets.

        Must run after ``BaseGraphModel._calculate_shapes_and_indices`` because the dimension check
        below reads per-dataset shapes.
        """
        for feature in self.features:
            feature.validate()

    @cached_property
    def dim(self) -> int:
        return sum(feature.dim for feature in self.features)

    def _compute(self, x_input_data: Tensor, x_encoded_data: Tensor | None, batch_size: int) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} shards per child feature, use tensor().")

    def tensor(
        self,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        grid_shard_sizes: ShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> Tensor:
        parts = [
            feature.tensor(
                x_input_data, x_encoded_data, batch_size, grid_shard_sizes, model_comm_group, dataset_name=dataset_name
            )
            for feature in self.features
        ]

        if len(parts) == 1:
            return parts[0]

        return torch.cat(parts, dim=-1)


def create_decoding_target_features(
    features: list[str],
    decoder_datasets_names: list[str],
    model: BaseGraphModel,
) -> DecodingTargetFeature:
    """Create a dict of :class:`DecodingTargetFeature` instances from a list of registry keys."""
    invalid = set(features) - VALID_TARGET_FEATURES
    assert not invalid, (
        f"Decoder has invalid input_target_features: {invalid}. " f"Valid options: {sorted(VALID_TARGET_FEATURES)}"
    )

    return CompositeTargetFeature(model, datasets_names=decoder_datasets_names, feature_names=features)


VALID_TARGET_FEATURES = set(TARGET_FEATURE_REGISTRY)
