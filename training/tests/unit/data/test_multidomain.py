# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockFixture
from torch.utils.data import ConcatDataset

from anemoi.models.data import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.loading import MultiDomainLoadingPolicy
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.tasks import TemporalDownscaler
from anemoi.transform.variables import Variable
from tests.batch_builders import build_batch


def _reader_anchors(rows: list[list[int]]) -> ReaderAnchors:
    """Valid anchors of a single-sequence reader from its ``(sequence, position)`` rows.

    Each domain is its own ``MultiDataset``, so the anchor times only need to be unique.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    positions = rows[:, 1]
    return ReaderAnchors(positions.astype("datetime64[s]"), rows[:, 0], positions)


def _mock_reader(mocker: MockFixture, rows: list[list[int]]) -> MagicMock:
    reader = mocker.MagicMock()
    reader.sampling = None
    reader.valid_anchors.return_value = _reader_anchors(rows)
    # Gridded samples carry no point times, so measuring them from the reference changes nothing.
    reader.measure_from_reference.side_effect = lambda sample, *_args: sample
    return reader


class TestMultiDomain:
    """Test multi-domain sampling with independently indexed domains."""

    @pytest.fixture
    def data_readers(self, mocker: MockFixture) -> dict:
        mock_dataset_a = _mock_reader(mocker, [[0, 0], *[[0, index] for index in range(11, 24)]])
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.grid_size = 5
        mock_dataset_a.num_sequences = 1
        mock_dataset_a.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_a.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_a.name_to_index = {"10u": 0}

        mock_dataset_b = _mock_reader(mocker, [[0, 0], [0, 1], [0, 2], [0, 3]])
        mock_dataset_b.frequency = "3h"
        mock_dataset_b.grid_size = 8
        mock_dataset_b.num_sequences = 1
        mock_dataset_b.metadata = {"variables_metadata": {"10u": {"units": "m/s"}}}
        mock_dataset_b.data.typed_variables = {"10u": Variable.from_dict("10u", {"units": "m/s"})}
        mock_dataset_b.name_to_index = {"10u": 0}
        return {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}

    @pytest.fixture
    def multi_domain(self, data_readers: dict) -> ConcatDataset:
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}
        return MultiDomainLoadingPolicy().build_dataset(data_readers, relative_date_indices)

    def test_exposes_the_dataset_metadata_of_all_domains(self, data_readers: dict, multi_domain: ConcatDataset) -> None:
        """The trainer and datamodule read these per-dataset mappings, as from a single MultiDataset."""
        for name, reader in data_readers.items():
            reader.variables = [f"{name}_var"]
            reader.resolution = f"{name}_res"

        assert multi_domain.dataset_names == ["dataset_a", "dataset_b"]
        assert multi_domain.variables == {"dataset_a": ["dataset_a_var"], "dataset_b": ["dataset_b_var"]}
        assert multi_domain.resolution == {"dataset_a": "dataset_a_res", "dataset_b": "dataset_b_res"}
        assert multi_domain.name_to_index == {"dataset_a": {"10u": 0}, "dataset_b": {"10u": 0}}
        assert set(multi_domain.metadata) == set(multi_domain.statistics) == {"dataset_a", "dataset_b"}

    def test_reader_group_info_reaches_every_reader(self, data_readers: dict, multi_domain: ConcatDataset) -> None:
        multi_domain.set_reader_group_info(1, 4)

        for reader in data_readers.values():
            reader.set_reader_group_info.assert_called_once_with(1, 4)

    def test_independent_domain_anchors(self, multi_domain: ConcatDataset) -> None:
        domains = dict(zip(("dataset_a", "dataset_b"), multi_domain.datasets, strict=True))

        assert isinstance(domains["dataset_a"], MultiDataset)
        assert np.array_equal(domains["dataset_a"].anchors.rows["dataset_a"][:, 1], [0, *range(11, 24)])
        assert np.array_equal(domains["dataset_b"].anchors.rows["dataset_b"], [[0, 0], [0, 1], [0, 2], [0, 3]])

    def test_relative_date_indices_refresh_domain_boundaries(
        self,
        data_readers: dict,
        multi_domain: ConcatDataset,
    ) -> None:
        data_readers["dataset_a"].valid_anchors.return_value = _reader_anchors([[0, 0], [0, 1]])
        data_readers["dataset_b"].valid_anchors.return_value = _reader_anchors([[0, 0], [0, 1], [0, 2]])

        multi_domain.set_relative_date_indices({"dataset_a": [0, 1], "dataset_b": [0, 1]})

        assert multi_domain.cumulative_sizes == [2, 5]
        assert set(multi_domain[4]) == {"dataset_b"}

        sampler = MultiDomainLoadingPolicy().build_batch_sampler(
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
        data_readers["dataset_b"].valid_anchors.return_value = _reader_anchors([])

        with pytest.raises(ValueError, match="No valid anchors found for data reader 'dataset_b'"):
            MultiDomainLoadingPolicy().build_dataset(
                data_readers,
                {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]},
            )

    @pytest.mark.parametrize(
        "name_to_index",
        [{"2t": 0, "10u": 1}, {"2t": 0, "tp": 1, "10u": 2}, {"10u": 0}],
        ids=["order", "order_of_shared_variables", "missing_variable"],
    )
    def test_check_datasets_variables_order(self, data_readers: dict, name_to_index: dict) -> None:
        data_readers["dataset_a"].name_to_index = {"10u": 0, "2t": 1}
        data_readers["dataset_b"].name_to_index = name_to_index
        policy = MultiDomainLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        with pytest.raises(ValueError, match="same variables in the same order"):
            policy.check_dataset_compatibility(dataset)

    def test_check_datasets_units(self, data_readers: dict) -> None:
        data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "K"}),
        }
        policy = MultiDomainLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        with pytest.raises(ValueError, match="Variable compatibility check failed"):
            policy.check_dataset_compatibility(dataset)

    def test_check_datasets_units_accepts_compatibility_options(self, data_readers: dict) -> None:
        data_readers["dataset_b"].data.typed_variables = {
            "10u": Variable.from_dict("10u", {"units": "km/h"}),
        }
        policy = MultiDomainLoadingPolicy()
        dataset = policy.build_dataset(data_readers, {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]})

        policy.check_dataset_compatibility(dataset, ignore_units=True)


def test_temporal_downscaler_offsets_are_loaded_from_one_domain(mocker: MockFixture) -> None:
    task = TemporalDownscaler(input_timestep="6h", output_timestep="2h")
    readers = {}
    for name, frequency in (("data", datetime.timedelta(hours=1)), ("data_2", datetime.timedelta(hours=2))):
        reader = _mock_reader(mocker, [[0, 0]])
        reader.frequency = frequency
        reader.num_sequences = 1
        reader.data.typed_variables = {}
        reader.get_sample.return_value = torch.zeros(4, 1, 1, 1)
        readers[name] = reader

    relative_date_indices = compute_relative_date_indices(task, readers)
    assert relative_date_indices == {"data": [0, 2, 4, 6], "data_2": [0, 1, 2, 3]}

    dataset = MultiDomainLoadingPolicy().build_dataset(readers, relative_date_indices)
    sample = dataset[0]

    readers["data"].get_sample.assert_called_once_with(0, slice(0, 8, 2))
    readers["data_2"].get_sample.assert_not_called()

    assert set(sample) == {"data"}
    batch = build_batch(
        data={"data": sample["data"].unsqueeze(0)},
        coordinates={"data": torch.zeros(1, 2)},
        layouts={"data": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={"data": ["v"]},
    )
    data_indices = {"data": IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), {"v": 0})}
    assert set(task.get_inputs(batch, data_indices).dataset_names) == {"data"}
    targets, _template, _forcing = task.get_targets(batch, data_indices)
    assert set(targets.dataset_names) == {"data"}
