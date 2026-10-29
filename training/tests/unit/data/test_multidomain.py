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
from pytest_mock import MockFixture

from anemoi.training.data.loading import CrossDataset
from anemoi.training.data.loading import CrossDatasetLoadingPolicy
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.usable_indices import ReaderAnchors


def _mock_reader(mocker: MockFixture, positions: list[int], units: str = "m/s") -> MockFixture:
    """Mock a single-sequence reader whose valid anchors are ``positions``."""
    reader = mocker.MagicMock()
    reader.frequency = datetime.timedelta(hours=3)
    reader.sampling = None
    reader.metadata = {"variables_metadata": {"10u": {"units": units}}}
    reader.measure_from_reference.side_effect = lambda sample, *_args: sample
    _set_positions(reader, positions)
    return reader


def _set_positions(reader: MockFixture, positions: list[int]) -> None:
    positions = np.asarray(positions, dtype=np.int64)
    reader.valid_anchors.side_effect = lambda _relative_indices: ReaderAnchors(
        positions.astype("datetime64[s]"),
        np.zeros_like(positions),
        positions,
    )


class TestMultiDomain:
    """Test cross-dataset sampling with independently indexed domains."""

    @pytest.fixture
    def data_readers(self, mocker: MockFixture) -> dict:
        return {
            "dataset_a": _mock_reader(mocker, [0, *range(11, 24)]),
            "dataset_b": _mock_reader(mocker, [0, 1, 2, 3]),
        }

    @pytest.fixture
    def multi_domain(self, data_readers: dict) -> CrossDataset:
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}
        return CrossDatasetLoadingPolicy().build_dataset(data_readers, relative_date_indices)

    def test_independent_domain_anchors(self, multi_domain: CrossDataset) -> None:
        domains = dict(zip(("dataset_a", "dataset_b"), multi_domain.datasets, strict=True))

        assert isinstance(domains["dataset_a"], MultiDataset)
        assert list(domains["dataset_a"].data_readers) == ["dataset_a"]
        assert list(domains["dataset_b"].data_readers) == ["dataset_b"]
        assert np.array_equal(domains["dataset_a"].anchors.rows["dataset_a"][:, 1], [0, *range(11, 24)])
        assert np.array_equal(domains["dataset_b"].anchors.rows["dataset_b"][:, 1], [0, 1, 2, 3])

    def test_samples_contain_a_single_dataset(self, multi_domain: CrossDataset) -> None:
        assert set(multi_domain[0]) == {"dataset_a"}
        assert set(multi_domain[len(multi_domain.datasets[0])]) == {"dataset_b"}

    def test_metadata_is_collected_from_all_readers(self, multi_domain: CrossDataset, data_readers: dict) -> None:
        assert multi_domain.data_readers == data_readers
        assert set(multi_domain.metadata) == {"dataset_a", "dataset_b"}

    def test_relative_date_indices_refresh_domain_boundaries(
        self,
        data_readers: dict,
        multi_domain: CrossDataset,
    ) -> None:
        _set_positions(data_readers["dataset_a"], [0, 1])
        _set_positions(data_readers["dataset_b"], [0, 1, 2])

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
        _set_positions(data_readers["dataset_b"], [])

        with pytest.raises(ValueError, match="No valid anchors found for data reader 'dataset_b'"):
            CrossDatasetLoadingPolicy().build_dataset(
                data_readers,
                {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]},
            )

    def test_check_datasets_units(self, data_readers: dict) -> None:
        data_readers["dataset_b"].metadata = {"variables_metadata": {"10u": {"units": "K"}}}
        policy = CrossDatasetLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            policy.check_dataset_compatibility(dataset)

    def test_check_datasets_units_accepts_compatibility_options(self, data_readers: dict) -> None:
        data_readers["dataset_b"].metadata = {"variables_metadata": {"10u": {"units": "km/h"}}}
        policy = CrossDatasetLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        policy.check_dataset_compatibility(dataset, ignore_units=True)
