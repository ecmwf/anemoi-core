# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
from pytest_mock import MockFixture
from torch.utils.data import ConcatDataset

from anemoi.training.data.loading import CrossDatasetLoadingPolicy
from anemoi.training.data.multidataset import MultiDataset
from anemoi.transform.variables import Variable


class TestMultiDomain:
    """Test cross-dataset sampling with independently indexed domains."""

    @pytest.fixture
    def data_readers(self, mocker: MockFixture) -> dict:
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.grid_size = 5
        mock_dataset_a.num_sequences = 1
        mock_dataset_a.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_a.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_a.compute_anchors.return_value = np.array(
            [[0, 0], *[[0, index] for index in range(11, 24)]],
        )

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.frequency = "3h"
        mock_dataset_b.grid_size = 8
        mock_dataset_b.num_sequences = 1
        mock_dataset_b.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_b.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_b.compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        return {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}

    @pytest.fixture
    def multi_domain(self, data_readers: dict) -> ConcatDataset:
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}
        return CrossDatasetLoadingPolicy().build_dataset(data_readers, relative_date_indices)

    def test_independent_domain_anchors(self, multi_domain: ConcatDataset) -> None:
        domains = dict(zip(("dataset_a", "dataset_b"), multi_domain.datasets, strict=True))

        assert isinstance(domains["dataset_a"], MultiDataset)
        assert np.array_equal(domains["dataset_a"].anchors[:, 1], [0, *range(11, 24)])
        assert np.array_equal(domains["dataset_b"].anchors, [[0, 0], [0, 1], [0, 2], [0, 3]])

    def test_relative_date_indices_refresh_domain_boundaries(
        self,
        data_readers: dict,
        multi_domain: ConcatDataset,
    ) -> None:
        data_readers["dataset_a"].compute_anchors.return_value = np.array([[0, 0], [0, 1]])
        data_readers["dataset_b"].compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2]])

        multi_domain.set_relative_date_indices({"dataset_a": [0, 1], "dataset_b": [0, 1]})

        assert multi_domain.cumulative_sizes == [2, 5]
        assert set(multi_domain[4]) == {"dataset_b"}

        sampler = CrossDatasetLoadingPolicy().build_batch_sampler(
            multi_domain,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
            batch_size=2,
            drop_last=True,
        )
        assert list(sampler) == [[0, 1], [2, 3], [4]]

    def test_empty_domain_raises(self, data_readers: dict) -> None:
        data_readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)

        with pytest.raises(ValueError, match="No valid anchors found for data reader 'dataset_b'"):
            CrossDatasetLoadingPolicy().build_dataset(
                data_readers,
                {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]},
            )

    def test_check_datasets_units(self, data_readers: dict) -> None:
        data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "K"}),
        }
        policy = CrossDatasetLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            policy.check_dataset_compatibility(dataset)

    def test_check_datasets_units_accepts_compatibility_options(self, data_readers: dict) -> None:
        data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "km/h"}),
        }
        policy = CrossDatasetLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        policy.check_dataset_compatibility(dataset, ignore_units=True)
