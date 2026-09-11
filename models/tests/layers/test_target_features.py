# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from anemoi.models.layers.target_features import TARGET_FEATURE_REGISTRY
from anemoi.models.layers.target_features import CompositeTargetFeature
from anemoi.models.layers.target_features import DecodingTargetFeature
from anemoi.models.layers.target_features import create_decoding_target_features
from anemoi.models.layers.target_features import register_target_feature


@dataclass
class DatasetSpec:
    """Per-dataset description of the fake model."""

    coord_dim: int = 4  # sin/cos encoded lat-lon
    forcing_idx: list[int] = field(default_factory=lambda: [0, 3])
    prognostic_idx: list[int] = field(default_factory=lambda: [1, 2, 4])
    num_trainable: int = 3
    input_dim: int = 11


@dataclass
class FakeModelConfig:
    """Configuration of the `BaseGraphModel` stand-in the target features are bound to."""

    specs: dict[str, DatasetSpec] = field(default_factory=lambda: {"data": DatasetSpec()})
    num_nodes: int = 5
    n_step_input: int = 2
    num_vars: int = 7
    input_datasets: list[str] | None = None  # defaults to every dataset in `specs`

    def build(self) -> SimpleNamespace:
        """Build a stand-in for `BaseGraphModel` exposing only what the target features read."""
        node_attributes = SimpleNamespace(trainable_tensors={}, num_trainable_parameters={})
        model = SimpleNamespace(
            node_attributes=node_attributes,
            n_step_input=self.n_step_input,
            num_input_channels_forcings={},
            num_input_channels_prognostic={},
            _forcing_input_idx={},
            _internal_input_idx={},
            input_dim={},
            input_datasets=list(self.specs) if self.input_datasets is None else self.input_datasets,
            dataset2decoder={name: f"decoder_{name}" for name in self.specs},
        )

        for name, spec in self.specs.items():
            setattr(node_attributes, f"latlons_{name}", torch.rand((self.num_nodes, spec.coord_dim)))
            trainable = torch.rand(self.num_nodes, spec.num_trainable) if spec.num_trainable > 0 else None
            node_attributes.trainable_tensors[name] = SimpleNamespace(trainable=trainable)
            node_attributes.num_trainable_parameters[name] = spec.num_trainable

            model._forcing_input_idx[name] = torch.tensor(spec.forcing_idx)
            model._internal_input_idx[name] = torch.tensor(spec.prognostic_idx)
            model.num_input_channels_forcings[name] = len(spec.forcing_idx)
            model.num_input_channels_prognostic[name] = len(spec.prognostic_idx)
            model.input_dim[name] = spec.input_dim

        return model


class TargetFeatureTestCase:
    """Fixtures shared by the target feature test classes."""

    BATCH_SIZE: int = 2
    ENSEMBLE_SIZE: int = 1
    DATASET: str = "data"

    @pytest.fixture
    def model_init(self) -> FakeModelConfig:
        return FakeModelConfig()

    @pytest.fixture
    def spec(self, model_init) -> DatasetSpec:
        return model_init.specs[self.DATASET]

    @pytest.fixture
    def model(self, model_init) -> SimpleNamespace:
        return model_init.build()

    @pytest.fixture
    def x_input_data(self, model_init) -> Tensor:
        """Input tensor of shape (batch, time, ensemble, grid, vars)."""
        shape = (
            self.BATCH_SIZE,
            model_init.n_step_input,
            self.ENSEMBLE_SIZE,
            model_init.num_nodes,
            model_init.num_vars,
        )
        return torch.rand(shape, dtype=torch.float32)

    @pytest.fixture
    def x_encoded_data(self, model_init, spec) -> Tensor:
        """Encoder output on the data nodes, already sharded and flattened by the encoder."""
        encoded_shape = (self.BATCH_SIZE * self.ENSEMBLE_SIZE * model_init.num_nodes, spec.input_dim)
        return torch.rand(encoded_shape, dtype=torch.float32)


class TestTargetFeatures(TargetFeatureTestCase):
    """Test the width each target feature contributes to the decoder target input."""

    @pytest.mark.parametrize(
        ("feature_name", "expected_dim"),
        [
            ("coordinates", 4),  # spec.coord_dim
            ("forcings", 2 * 2),  # n_step_input * len(spec.forcing_idx)
            ("prognostics", 2 * 3),  # n_step_input * len(spec.prognostic_idx)
            ("trainable_parameters", 3),  # spec.num_trainable
            ("encoded_data", 11),  # spec.input_dim
        ],
    )
    def test_target_feature(
        self,
        feature_name: str,
        expected_dim: int,
        model: SimpleNamespace,
        model_init: FakeModelConfig,
        x_input_data: Tensor,
        x_encoded_data: Tensor,
    ) -> None:
        """Test that the target feature has the expected dimension and produces the correct tensor shape."""
        feature = create_decoding_target_features([feature_name], [self.DATASET], model)

        # Test feature dimension
        assert feature.dim == expected_dim

        # Test feature shape
        out = feature.tensor(x_input_data, x_encoded_data, batch_size=self.BATCH_SIZE, dataset_name=self.DATASET)
        ensemble_size, num_nodes, n_step_input = self.ENSEMBLE_SIZE, model_init.num_nodes, model_init.n_step_input
        var_idx = model_init.specs[self.DATASET].forcing_idx
        num_rows = self.BATCH_SIZE * ensemble_size * num_nodes
        assert out.shape == (num_rows, feature.dim)

        out = feature.tensor(x_input_data, x_encoded_data, batch_size=self.BATCH_SIZE, dataset_name=self.DATASET)
        if feature_name == "forcings":
            # Test forcing specific shape and content
            assert out.shape == (self.BATCH_SIZE * ensemble_size * num_nodes, feature.dim)
            for batch in range(self.BATCH_SIZE):
                for ens in range(ensemble_size):
                    for node in range(num_nodes):
                        row = (batch * ensemble_size + ens) * num_nodes + node
                        expected = torch.cat(
                            [x_input_data[batch, step, ens, node, var_idx] for step in range(n_step_input)],
                        )
                        torch.testing.assert_close(out[row], expected)
        elif feature_name == "encoded_data":
            # Test encoded_data is passed through correctly
            assert out is x_encoded_data
            with pytest.raises(ValueError, match="requires the encoder output for dataset 'data'"):
                feature.tensor(x_input_data, None, batch_size=self.BATCH_SIZE, dataset_name=self.DATASET)
        elif feature_name == "coordinates":
            # Test coordinates feature requires dataset_name to be provided
            with pytest.raises(AssertionError, match="dataset_name must be provided"):
                feature.tensor(x_input_data, None, batch_size=self.BATCH_SIZE)

    def test_composite_dim_is_the_sum_of_its_features(self, model: SimpleNamespace) -> None:
        """Test that the composite feature's dimension is the sum of its child features."""
        feature_names = ["coordinates", "forcings", "trainable_parameters"]
        composite = create_decoding_target_features(feature_names, [self.DATASET], model)

        expected_dim = sum(create_decoding_target_features([name], [self.DATASET], model).dim for name in feature_names)
        assert composite.dim == expected_dim

    def test_composite_concatenates_in_declared_order(
        self,
        model_init: FakeModelConfig,
        spec: DatasetSpec,
        model: SimpleNamespace,
        x_input_data: Tensor,
    ) -> None:
        """Test that a composite feature concatenates its child features in the declared order."""
        composite = create_decoding_target_features(["coordinates", "trainable_parameters"], [self.DATASET], model)

        out = composite.tensor(x_input_data, None, batch_size=self.BATCH_SIZE, dataset_name=self.DATASET)

        num_nodes = model_init.num_nodes
        assert out.shape == (self.BATCH_SIZE * num_nodes, composite.dim)
        torch.testing.assert_close(out[:num_nodes, : spec.coord_dim], model.node_attributes.latlons_data)
        torch.testing.assert_close(
            out[:num_nodes, spec.coord_dim :],
            model.node_attributes.trainable_tensors[self.DATASET].trainable,
        )


class TestTargetFeatureValidate(TargetFeatureTestCase):
    """Test the build-time checks of the target features."""

    def test_validate_trainable_parameters_requires_configured_parameters(
        self,
        model_init: FakeModelConfig,
    ) -> None:
        """Test that validating trainable parameters requires configured parameters."""
        model_init.specs[self.DATASET] = replace(model_init.specs[self.DATASET], num_trainable=0)
        feature = create_decoding_target_features(["trainable_parameters"], [self.DATASET], model_init.build())

        with pytest.raises(ValueError, match="No trainable parameters configured for dataset 'data'"):
            feature.validate()

    def test_validate_encoded_data_requires_an_encoded_dataset(self, model_init: FakeModelConfig) -> None:
        """Test that validating an encoded data feature requires an encoded dataset."""
        model_init = replace(model_init, input_datasets=[])
        feature = create_decoding_target_features(["encoded_data"], [self.DATASET], model_init.build())

        with pytest.raises(ValueError, match=f'requires dataset "{self.DATASET}" to have an encoder'):
            feature.validate()

    def test_composite_validate_propagates_child_failures(self, model_init: FakeModelConfig) -> None:
        """Test that a composite feature propagates validation failures from its child features."""
        model_init.specs[self.DATASET] = replace(model_init.specs[self.DATASET], num_trainable=0)
        composite = create_decoding_target_features(
            ["coordinates", "trainable_parameters"],
            [self.DATASET],
            model_init.build(),
        )

        with pytest.raises(ValueError, match="No trainable parameters configured"):
            composite.validate()

    def test_composite_requires_at_least_one_feature(self, model: SimpleNamespace) -> None:
        """Test that creating a composite feature with no child features raises an assertion error."""
        with pytest.raises(AssertionError, match="must declare at least one target feature"):
            create_decoding_target_features([], [self.DATASET], model)


class TestTargetFeatureRegistry(TargetFeatureTestCase):
    """Test the target feature registry and the decoder-side factory."""

    def test_register_target_feature_sets_name_and_registers(self) -> None:
        """Test that registering a new target feature sets its name and registers it correctly."""

        @register_target_feature("new_feature")
        class NewFeature(DecodingTargetFeature):
            @property
            def dim(self) -> int:
                return 1

            def _compute(self, *args) -> Tensor:
                return torch.zeros(1, 1)

        assert TARGET_FEATURE_REGISTRY["new_feature"] is NewFeature
        assert NewFeature.name == "new_feature"

    def test_register_target_feature_rejects_duplicate_name(self) -> None:
        """Test that registering a target feature with a duplicate name raises a ValueError."""
        with pytest.raises(ValueError, match="already registered"):

            @register_target_feature("coordinates")
            class Duplicate(DecodingTargetFeature):
                @property
                def dim(self) -> int:
                    return 1

                def _compute(self, *args) -> Tensor:
                    return torch.zeros(1, 1)

    def test_create_decoding_target_features_builds_a_composite(self, model: SimpleNamespace) -> None:
        """Test that creating decoding target features with valid names builds a composite feature."""
        feature_names = ["coordinates", "forcings"]

        composite = create_decoding_target_features(feature_names, [self.DATASET], model)

        assert isinstance(composite, CompositeTargetFeature)
        assert [feature.name for feature in composite.features] == feature_names

    def test_create_decoding_target_features_rejects_unknown_features(self, model: SimpleNamespace) -> None:
        """Test that creating decoding target features with unknown names raises an AssertionError."""
        with pytest.raises(AssertionError, match="invalid target_node_features: {'not_a_feature'}"):
            create_decoding_target_features(["coordinates", "not_a_feature"], [self.DATASET], model)
