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

from anemoi.training.data.multidomain import MultiDomainDataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.data.sampler import CrossDatasetSampler
from anemoi.training.tasks.temporal_downscaler import TemporalDownscaler
from anemoi.transform.variables import Variable


class TestMultiDomain:
    """Test MultiDomainDataset instantiation and properties."""

    @pytest.fixture
    def multi_domain(self, mocker: MockFixture) -> MultiDomainDataset:
        """Fixture to provide a MultiDomainDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = {7, 8, 9, 10}
        mock_dataset_a.dates = list(range(30))
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.grid_size = 5
        mock_dataset_a.num_sequences = 1
        mock_dataset_a.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_a.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_a.compute_anchors.return_value = np.array(
            [[0, 0], *[[0, index] for index in range(11, 24)]],
        )

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = set()
        mock_dataset_b.dates = list(range(20, 60))
        mock_dataset_b.frequency = "3h"
        mock_dataset_b.grid_size = 8
        mock_dataset_b.num_sequences = 1
        mock_dataset_b.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_b.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_b.compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])

        data_readers = {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDomainDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_sharding(self, multi_domain: MultiDomainDataset) -> None:
        """Test that sharding logic correctly partitions the dataset."""
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        expected_indices = {
            "dataset_a": np.array([0, 1, 2, 3, 4, 5, 6]),
            "dataset_b": np.array([0, 1]),
        }
        for key in expected_indices:
            assert np.array_equal(multi_domain.chunk_index_range[key], expected_indices[key])

    def test_valid_date_indices(self, multi_domain: MultiDomainDataset) -> None:
        """Test that valid_date_indices returns a dictionary of indices from all datasets.

        relative_date_indices = [0, 1, 2]

        Each reader supplies valid ``(sequence, position)`` anchors. The
        dataset keeps a one-dimensional index for shuffling and sharding.
        """
        expected_indices = {"dataset_a": np.arange(14), "dataset_b": np.arange(4)}
        for key in expected_indices:
            assert np.array_equal(multi_domain.valid_date_indices[key], expected_indices[key])

        assert np.array_equal(multi_domain.anchors["dataset_a"][:, 1], [0, *range(11, 24)])
        assert np.array_equal(multi_domain.anchors["dataset_b"], [[0, 0], [0, 1], [0, 2], [0, 3]])

    def test_domains_do_not_need_shared_anchors(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_a"].compute_anchors.return_value = np.array([[0, 10]])
        multi_domain.data_readers["dataset_b"].compute_anchors.return_value = np.array([[0, 20]])

        dataset = MultiDomainDataset(
            data_readers=multi_domain.data_readers,
            relative_date_indices=multi_domain.relative_date_indices,
        )

        assert np.array_equal(dataset.anchors["dataset_a"], [[0, 10]])
        assert np.array_equal(dataset.anchors["dataset_b"], [[0, 20]])

    def test_frequency_is_shared_across_domains(self, multi_domain: MultiDomainDataset) -> None:
        assert multi_domain.frequency == "3h"

    def test_empty_domain_raises(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)

        with pytest.raises(ValueError, match="No valid anchors found for data reader 'dataset_b'"):
            MultiDomainDataset(
                data_readers=multi_domain.data_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_set_epoch_empty_domain_raises(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)

        with pytest.raises(ValueError, match="No valid anchors found for data reader 'dataset_b'"):
            multi_domain.set_epoch(1, relative_date_indices=multi_domain.relative_date_indices)

    def test_per_worker_init_creates_domain_specific_worker_state(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        assert set(multi_domain.n_samples_per_worker) == {"dataset_a", "dataset_b"}
        assert set(multi_domain.chunk_index_range) == {"dataset_a", "dataset_b"}

        assert isinstance(multi_domain.chunk_index_range["dataset_a"], np.ndarray)
        assert isinstance(multi_domain.chunk_index_range["dataset_b"], np.ndarray)

    def test_worker_shards_do_not_overlap_per_domain(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.per_worker_init(n_workers=2, worker_id=0)
        worker_0_ranges = {k: v.copy() for k, v in multi_domain.chunk_index_range.items()}

        multi_domain.per_worker_init(n_workers=2, worker_id=1)
        worker_1_ranges = {k: v.copy() for k, v in multi_domain.chunk_index_range.items()}

        for domain in multi_domain.dataset_names:
            assert set(worker_0_ranges[domain]).isdisjoint(set(worker_1_ranges[domain]))

    def test_sampler_dispatches_to_requested_domain(self, multi_domain: MultiDomainDataset) -> None:
        sampler = CrossDatasetSampler(multi_domain)
        sampler.sample(("dataset_a", 0))

        multi_domain.data_readers["dataset_a"].get_sample.assert_called_once()
        multi_domain.data_readers["dataset_b"].get_sample.assert_not_called()

        sampler.sample(("dataset_b", 2))
        multi_domain.data_readers["dataset_b"].get_sample.assert_called_once()

    def test_mixing_native_grid_and_trajectory_datasets_raises(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].num_sequences = 2

        with pytest.raises(ValueError, match="same IterableDataset is unsupported"):
            MultiDomainDataset(
                data_readers=multi_domain.data_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_check_datasets_units_runs_during_initialization(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "km/h"}),
        }

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            MultiDomainDataset(
                data_readers=multi_domain.data_readers,
                relative_date_indices=multi_domain.relative_date_indices,
            )

    def test_check_datasets_units_accepts_compatibility_options(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "km/h"}),
        }

        MultiDomainDataset(
            data_readers=multi_domain.data_readers,
            relative_date_indices=multi_domain.relative_date_indices,
            check_variables_compatibility={"ignore_units": True},
        )

    def test_check_datasets_units_raises_error_for_incompatible_units(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_a"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "m/s"}),
        }
        multi_domain.data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "km/h"}),
        }
        with pytest.raises(
            ValueError,
            match="Variable compatibility check failed for domain1 'dataset_a' and domain2 'dataset_b'",
        ):
            multi_domain._check_datasets_units()

    def test_check_datasets_units_passes_for_compatible_units(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_a"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "m/s"}),
        }
        multi_domain.data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "m/s"}),
            "2t": Variable.from_dict("2t", {"units": "K"}),
        }

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_no_dataset_has_metadata(self, multi_domain: MultiDomainDataset) -> None:
        multi_domain.data_readers["dataset_a"].data.typed_variables = {}
        multi_domain.data_readers["dataset_b"].data.typed_variables = {}

        assert multi_domain._check_datasets_units() is None

    def test_check_datasets_units_skips_when_only_one_dataset_has_metadata(
        self,
        multi_domain: MultiDomainDataset,
    ) -> None:
        multi_domain.data_readers["dataset_a"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "m/s"}),
        }
        multi_domain.data_readers["dataset_b"].data.typed_variables = {}
        assert (
            multi_domain._check_datasets_units() is None
        ), "Should skip units check when only one dataset has variable metadata"

    def test_temporal_downscaler_offsets_are_loaded_from_one_domain(self, mocker: MockFixture) -> None:
        task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
        readers = {}
        for name, frequency in (("sg_1", datetime.timedelta(hours=1)), ("sg_2", datetime.timedelta(hours=2))):
            reader = mocker.MagicMock()
            reader.frequency = frequency
            reader.num_sequences = 1
            reader.compute_anchors.return_value = np.array([[0, 0]])
            reader.data.typed_variables = {}
            reader.get_sample.return_value = torch.zeros(4, 1, 1, 1)
            readers[name] = reader

        relative_date_indices = compute_relative_date_indices(task, readers)
        assert relative_date_indices == {"sg_1": [0, 2, 4, 6], "sg_2": [0, 1, 2, 3]}

        dataset = MultiDomainDataset(
            data_readers=readers,
            relative_date_indices=relative_date_indices,
            shuffle=False,
        )
        dataset.per_worker_init(n_workers=1, worker_id=0)
        sample = next(iter(dataset))

        readers["sg_1"].get_sample.assert_called_once_with(0, slice(0, 8, 2), slice(None))
        readers["sg_2"].get_sample.assert_not_called()

        batch = {"sg_1": sample["sg_1"].unsqueeze(0)}
        data_indices = {"sg_1": mocker.MagicMock()}
        data_indices["sg_1"].data.input.full = slice(None)
        assert set(task.get_inputs(batch, data_indices)) == {"sg_1"}
        assert set(task.get_targets(batch)) == {"sg_1"}
