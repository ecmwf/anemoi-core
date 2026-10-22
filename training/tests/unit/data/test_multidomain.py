# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest
import torch
from pytest_mock import MockFixture
from torch.utils.data import ConcatDataset

from anemoi.training.data.loading import CrossDatasetLoadingPolicy
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.tasks.temporal_downscaler import TemporalDownscaler
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

    def test_temporal_downscaler_offsets_are_loaded_from_one_domain(self, mocker: MockFixture) -> None:
        task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
        readers = {}
        for name, frequency in (("data", datetime.timedelta(hours=1)), ("datb", datetime.timedelta(hours=2))):
            reader = mocker.MagicMock()
            reader.frequency = frequency
            reader.num_sequences = 1
            reader.compute_anchors.return_value = np.array([[0, 0]])
            reader.data.typed_variables = {}
            reader.get_sample.return_value = torch.zeros(4, 1, 1, 1)
            readers[name] = reader

        relative_date_indices = compute_relative_date_indices(task, readers)
        assert relative_date_indices == {"data": [0, 2, 4, 6], "datb": [0, 1, 2, 3]}

        dataset = CrossDatasetLoadingPolicy().build_dataset(readers, relative_date_indices)
        sample = dataset[0]

        readers["data"].get_sample.assert_called_once_with(0, slice(0, 8, 2), slice(None))
        readers["datb"].get_sample.assert_not_called()

        batch = {"data": sample["data"].unsqueeze(0)}
        data_indices = {"data": mocker.MagicMock()}
        data_indices["data"].data.input.full = slice(None)
        assert set(task.get_inputs(batch, data_indices)) == {"data"}
        assert set(task.get_targets(batch)) == {"data"}
