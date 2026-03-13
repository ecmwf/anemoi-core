# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for SparseObservationDataset, sparse collation, and transfer_batch_to_device."""

import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.dataset import SparseObservationDataset
from anemoi.training.data.dataset import create_dataset
from anemoi.training.data.multidataset import collate_sparse
from anemoi.training.data.multidataset import multidataset_collator_func

# ---------------------------------------------------------------------------
# The real microwave observation dataset used in multi_obs.yaml
# ---------------------------------------------------------------------------
MW_DATASET_NAME = "observations-file-1994-2025-combined-mw-humidity-o96-v2-from-dop-try-2"
MW_FREQUENCY = "6h"
MW_WINDOW = "(-6h, 0]"
MW_DROP = ["spatial_index", "zenith"]

# Short date range to keep tests fast
MW_START = 2020
MW_END = 2020


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def sparse_obs_dataset() -> SparseObservationDataset:
    """Load the real microwave humidity observation dataset."""
    return SparseObservationDataset(
        dataset={
            "dataset": MW_DATASET_NAME,
            "frequency": MW_FREQUENCY,
            "window": MW_WINDOW,
            "drop": MW_DROP,
        },
        start=MW_START,
        end=MW_END,
    )


@pytest.fixture(scope="module")
def sparse_obs_sample(sparse_obs_dataset: SparseObservationDataset) -> tuple[torch.Tensor, dict]:
    """Get a single sample from the observation dataset."""
    return sparse_obs_dataset.get_sample(time_indices=0)


# ===================================================================
# SparseObservationDataset: instantiation & properties
# ===================================================================


class TestSparseObservationDatasetInstantiation:
    """Test SparseObservationDataset instantiation using the real microwave dataset."""

    @pytest.mark.data_dependent
    def test_basic_instantiation(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """Test that the dataset can be opened and core attributes are accessible."""
        assert sparse_obs_dataset.data is not None

    @pytest.mark.data_dependent
    def test_has_trajectories_is_false(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        assert sparse_obs_dataset.has_trajectories is False

    @pytest.mark.data_dependent
    def test_missing_is_empty(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        assert sparse_obs_dataset.missing == set()

    @pytest.mark.data_dependent
    def test_dates(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        dates = sparse_obs_dataset.dates
        assert isinstance(dates, np.ndarray)
        assert len(dates) > 0

    @pytest.mark.data_dependent
    def test_variables(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        variables = sparse_obs_dataset.variables
        assert isinstance(variables, list)
        assert len(variables) > 0

    @pytest.mark.data_dependent
    def test_frequency(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        freq = sparse_obs_dataset.frequency
        assert isinstance(freq, datetime.timedelta)

    @pytest.mark.data_dependent
    def test_name_to_index(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        n2i = sparse_obs_dataset.name_to_index
        assert isinstance(n2i, dict)
        assert len(n2i) == len(sparse_obs_dataset.variables)

    @pytest.mark.data_dependent
    def test_dropped_variables_not_present(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """Variables listed in MW_DROP should have been excluded."""
        for dropped in MW_DROP:
            assert dropped not in sparse_obs_dataset.variables

    @pytest.mark.data_dependent
    def test_window_attribute(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """The underlying anemoi-datasets object should expose a window attribute."""
        assert hasattr(sparse_obs_dataset.data, "window")

    @pytest.mark.data_dependent
    def test_tree_repr(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """tree() should return a Rich Tree without errors."""
        tree = sparse_obs_dataset.tree()
        assert tree is not None
        text = repr(sparse_obs_dataset)
        assert "SparseObservationDataset" in text


# ===================================================================
# SparseObservationDataset: get_sample
# ===================================================================


class TestSparseObservationGetSample:
    """Test the (Tensor, metadata) tuple returned by get_sample."""

    @pytest.mark.data_dependent
    def test_returns_tuple(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        assert isinstance(sparse_obs_sample, tuple)
        assert len(sparse_obs_sample) == 2

    @pytest.mark.data_dependent
    def test_tensor_shape(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        """First element is a tensor with shape (1, n_obs, n_vars) — 1 = dummy ensemble dim."""
        tensor, _ = sparse_obs_sample
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1  # dummy ensemble dimension

    @pytest.mark.data_dependent
    def test_metadata_keys(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        _, meta = sparse_obs_sample
        assert isinstance(meta, dict)
        for key in ("latitudes", "longitudes", "timedeltas", "boundaries"):
            assert key in meta, f"Missing metadata key: {key}"

    @pytest.mark.data_dependent
    def test_metadata_latitudes_longitudes_are_tensors(
        self, sparse_obs_sample: tuple[torch.Tensor, dict]
    ) -> None:
        _, meta = sparse_obs_sample
        assert isinstance(meta["latitudes"], torch.Tensor)
        assert isinstance(meta["longitudes"], torch.Tensor)
        assert meta["latitudes"].ndim == 1
        assert meta["longitudes"].ndim == 1

    @pytest.mark.data_dependent
    def test_metadata_timedeltas_is_float_tensor(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        _, meta = sparse_obs_sample
        assert isinstance(meta["timedeltas"], torch.Tensor)
        assert meta["timedeltas"].dtype == torch.float32

    @pytest.mark.data_dependent
    def test_metadata_obs_count_consistent(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        """The number of observations in the tensor and metadata should be consistent."""
        tensor, meta = sparse_obs_sample
        n_obs = tensor.shape[1]
        assert meta["latitudes"].shape[0] == n_obs
        assert meta["longitudes"].shape[0] == n_obs

    @pytest.mark.data_dependent
    def test_get_sample_with_slice(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """get_sample accepts a slice for time_indices."""
        tensor, meta = sparse_obs_dataset.get_sample(time_indices=slice(0, 2))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 3
        assert isinstance(meta, dict)

    @pytest.mark.data_dependent
    def test_grid_shard_indices_is_ignored(self, sparse_obs_dataset: SparseObservationDataset) -> None:
        """grid_shard_indices is accepted but currently ignored for sparse obs."""
        sample_no_shard = sparse_obs_dataset.get_sample(time_indices=0)
        sample_with_shard = sparse_obs_dataset.get_sample(time_indices=0, grid_shard_indices=slice(0, 10))
        # Both should produce the same tensor since sharding is a no-op
        # equal_nan=True because observation data may contain NaN values
        torch.testing.assert_close(sample_no_shard[0], sample_with_shard[0], equal_nan=True)

    @pytest.mark.data_dependent
    def test_latitude_longitude_ranges(self, sparse_obs_sample: tuple[torch.Tensor, dict]) -> None:
        """Sanity-check that lat/lon values fall within plausible geographic ranges."""
        _, meta = sparse_obs_sample
        lats = meta["latitudes"]
        lons = meta["longitudes"]
        if lats.numel() > 0:
            assert lats.min() >= -90.0, "Latitude below -90"
            assert lats.max() <= 90.0, "Latitude above 90"
        if lons.numel() > 0:
            assert lons.min() >= -360.0, "Longitude below -360"
            assert lons.max() <= 360.0, "Longitude above 360"


# ===================================================================
# create_dataset factory dispatch
# ===================================================================


class TestCreateDatasetFactory:
    """Test that create_dataset dispatches to the correct reader based on `tabular`."""

    @pytest.mark.data_dependent
    def test_tabular_true_creates_sparse_dataset(self) -> None:
        cfg = {
            "dataset_config": {
                "dataset": MW_DATASET_NAME,
                "frequency": MW_FREQUENCY,
                "window": MW_WINDOW,
                "drop": MW_DROP,
            },
            "start": MW_START,
            "end": MW_END,
            "trajectory": None,
            "tabular": True,
        }
        dataset = create_dataset(cfg)
        assert isinstance(dataset, SparseObservationDataset)

    @patch("anemoi.training.data.dataset.open_dataset")
    def test_tabular_false_creates_native_grid(self, mock_open: MagicMock) -> None:
        """Without tabular flag, create_dataset returns NativeGridDataset."""
        mock_ds = MagicMock()
        mock_ds.dates = np.array([np.datetime64("2020-01-01")])
        mock_ds.variables = ["t2m"]
        mock_ds.frequency = datetime.timedelta(hours=6)
        mock_ds.name_to_index = {"t2m": 0}
        mock_ds.grids = (100,)
        mock_open.return_value = mock_ds

        cfg = {
            "dataset_config": {"dataset": "fake-gridded-dataset"},
            "start": None,
            "end": None,
            "trajectory": None,
        }
        dataset = create_dataset(cfg)
        assert isinstance(dataset, NativeGridDataset)

    @patch("anemoi.training.data.dataset.open_dataset")
    def test_tabular_absent_defaults_to_native_grid(self, mock_open: MagicMock) -> None:
        """When tabular key is absent entirely, create_dataset defaults to NativeGridDataset."""
        mock_ds = MagicMock()
        mock_ds.dates = np.array([np.datetime64("2020-01-01")])
        mock_ds.variables = ["t2m"]
        mock_ds.frequency = datetime.timedelta(hours=6)
        mock_ds.name_to_index = {"t2m": 0}
        mock_ds.grids = (100,)
        mock_open.return_value = mock_ds

        cfg = {
            "dataset_config": {"dataset": "fake-gridded-dataset"},
            "start": None,
            "end": None,
        }
        dataset = create_dataset(cfg)
        assert isinstance(dataset, NativeGridDataset)


# ===================================================================
# Collation functions
# ===================================================================


def _make_sparse_sample(n_obs: int, n_vars: int) -> tuple[torch.Tensor, dict]:
    """Build a synthetic sparse observation sample."""
    return (
        torch.randn(1, n_obs, n_vars),
        {
            "latitudes": torch.randn(n_obs),
            "longitudes": torch.randn(n_obs),
            "timedeltas": torch.randn(n_obs),
            "boundaries": [slice(0, n_obs)],
        },
    )


class TestCollation:
    """Test collate_sparse and multidataset_collator_func."""

    def test_collate_sparse_basic(self) -> None:
        """collate_sparse groups tensors into a list and metadata into per-key lists."""
        samples = [_make_sparse_sample(10, 5), _make_sparse_sample(8, 5)]
        tensors, meta = collate_sparse(samples)

        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert tensors[0].shape == (1, 10, 5)
        assert tensors[1].shape == (1, 8, 5)

        for key in ("latitudes", "longitudes", "timedeltas", "boundaries"):
            assert key in meta
            assert len(meta[key]) == 2

    def test_collate_sparse_single_sample(self) -> None:
        """Collation works with a single-element batch."""
        samples = [_make_sparse_sample(20, 3)]
        tensors, meta = collate_sparse(samples)
        assert len(tensors) == 1
        assert len(meta["latitudes"]) == 1

    def test_multidataset_collator_gridded_only(self) -> None:
        """Gridded-only batch: tensors are stacked along a new batch dimension."""
        batch = [
            {"era5": torch.randn(3, 1, 100, 5)},
            {"era5": torch.randn(3, 1, 100, 5)},
        ]
        collated = multidataset_collator_func(batch)

        assert "era5" in collated
        assert isinstance(collated["era5"], torch.Tensor)
        assert collated["era5"].shape[0] == 2  # batch size

    def test_multidataset_collator_sparse_only(self) -> None:
        """Sparse-only batch: tuples are collated via collate_sparse."""
        batch = [
            {"combined_mw": _make_sparse_sample(15, 4)},
            {"combined_mw": _make_sparse_sample(12, 4)},
        ]
        collated = multidataset_collator_func(batch)

        assert "combined_mw" in collated
        tensors, meta = collated["combined_mw"]
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert "latitudes" in meta

    def test_multidataset_collator_mixed_batch(self) -> None:
        """Mixed batch with one gridded key and one sparse key."""
        batch = [
            {"era5": torch.randn(3, 1, 100, 5), "combined_mw": _make_sparse_sample(15, 4)},
            {"era5": torch.randn(3, 1, 100, 5), "combined_mw": _make_sparse_sample(12, 4)},
        ]
        collated = multidataset_collator_func(batch)

        # Gridded: stacked tensor
        assert isinstance(collated["era5"], torch.Tensor)
        assert collated["era5"].shape[0] == 2

        # Sparse: (list[Tensor], dict)
        tensors, meta = collated["combined_mw"]
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert isinstance(meta, dict)


# ===================================================================
# transfer_batch_to_device
# ===================================================================


def _transfer_batch_to_device(batch, device, _dataloader_idx=0):
    """Standalone reimplementation of BaseGraphModule.transfer_batch_to_device for testing.

    This avoids needing to instantiate the full Lightning module.
    Mirrors the logic in training/src/anemoi/training/train/tasks/base.py lines 785-809.
    """
    transferred_batch = {}
    for dataset_name, dataset_batch in batch.items():
        if isinstance(dataset_batch, torch.Tensor):
            transferred_batch[dataset_name] = dataset_batch.to(device, non_blocking=True)
        else:
            tensors, meta = dataset_batch
            transferred_batch[dataset_name] = (
                [t.to(device, non_blocking=True) for t in tensors],
                {
                    "latitudes": [t.to(device, non_blocking=True) for t in meta["latitudes"]],
                    "longitudes": [t.to(device, non_blocking=True) for t in meta["longitudes"]],
                    "timedeltas": [t.to(device, non_blocking=True) for t in meta["timedeltas"]],
                    "boundaries": meta["boundaries"],
                },
            )
    return transferred_batch


class TestTransferBatchToDevice:
    """Test the transfer_batch_to_device logic for gridded and sparse batches."""

    device = torch.device("cpu")

    def test_gridded_tensor_is_transferred(self) -> None:
        """A plain tensor (gridded data) should be moved to the target device."""
        batch = {"era5": torch.randn(2, 3, 1, 100, 5)}
        result = _transfer_batch_to_device(batch, self.device)

        assert "era5" in result
        assert isinstance(result["era5"], torch.Tensor)
        assert result["era5"].device == self.device
        assert result["era5"].shape == (2, 3, 1, 100, 5)

    def test_sparse_tuple_structure_preserved(self) -> None:
        """Sparse obs batch should remain as (List[Tensor], Dict) after transfer."""
        sparse_tensors = [torch.randn(1, 15, 4), torch.randn(1, 12, 4)]
        sparse_meta = {
            "latitudes": [torch.randn(15), torch.randn(12)],
            "longitudes": [torch.randn(15), torch.randn(12)],
            "timedeltas": [torch.randn(15), torch.randn(12)],
            "boundaries": [[slice(0, 15)], [slice(0, 12)]],
        }
        batch = {"combined_mw": (sparse_tensors, sparse_meta)}
        result = _transfer_batch_to_device(batch, self.device)

        tensors, meta = result["combined_mw"]
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert isinstance(meta, dict)

    def test_sparse_data_tensors_on_device(self) -> None:
        """All data tensors in the sparse batch should be on the target device."""
        sparse_tensors = [torch.randn(1, 10, 3)]
        sparse_meta = {
            "latitudes": [torch.randn(10)],
            "longitudes": [torch.randn(10)],
            "timedeltas": [torch.randn(10)],
            "boundaries": [[slice(0, 10)]],
        }
        batch = {"obs": (sparse_tensors, sparse_meta)}
        result = _transfer_batch_to_device(batch, self.device)

        tensors, meta = result["obs"]
        assert tensors[0].device == self.device
        assert meta["latitudes"][0].device == self.device
        assert meta["longitudes"][0].device == self.device
        assert meta["timedeltas"][0].device == self.device

    def test_sparse_boundaries_not_transferred(self) -> None:
        """Boundaries are plain Python objects (slices) and should not be modified."""
        boundaries = [[slice(0, 10)], [slice(0, 8)]]
        sparse_tensors = [torch.randn(1, 10, 3), torch.randn(1, 8, 3)]
        sparse_meta = {
            "latitudes": [torch.randn(10), torch.randn(8)],
            "longitudes": [torch.randn(10), torch.randn(8)],
            "timedeltas": [torch.randn(10), torch.randn(8)],
            "boundaries": boundaries,
        }
        batch = {"obs": (sparse_tensors, sparse_meta)}
        result = _transfer_batch_to_device(batch, self.device)

        _, meta = result["obs"]
        assert meta["boundaries"] is boundaries  # exact same object

    def test_mixed_gridded_and_sparse(self) -> None:
        """A batch containing both gridded and sparse datasets is handled correctly."""
        gridded = torch.randn(2, 3, 1, 100, 5)
        sparse_tensors = [torch.randn(1, 15, 4), torch.randn(1, 12, 4)]
        sparse_meta = {
            "latitudes": [torch.randn(15), torch.randn(12)],
            "longitudes": [torch.randn(15), torch.randn(12)],
            "timedeltas": [torch.randn(15), torch.randn(12)],
            "boundaries": [[slice(0, 15)], [slice(0, 12)]],
        }
        batch = {
            "era5": gridded,
            "combined_mw": (sparse_tensors, sparse_meta),
        }
        result = _transfer_batch_to_device(batch, self.device)

        # Gridded
        assert isinstance(result["era5"], torch.Tensor)
        assert result["era5"].shape == gridded.shape

        # Sparse
        tensors, meta = result["combined_mw"]
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert isinstance(meta, dict)
        assert "boundaries" in meta

    def test_tensor_values_preserved(self) -> None:
        """Data values should be identical after transfer (CPU → CPU)."""
        original_gridded = torch.randn(1, 2, 1, 50, 3)
        original_sparse = torch.randn(1, 20, 4)
        original_lats = torch.randn(20)

        batch = {
            "era5": original_gridded.clone(),
            "obs": (
                [original_sparse.clone()],
                {
                    "latitudes": [original_lats.clone()],
                    "longitudes": [torch.randn(20)],
                    "timedeltas": [torch.randn(20)],
                    "boundaries": [[slice(0, 20)]],
                },
            ),
        }
        result = _transfer_batch_to_device(batch, self.device)

        torch.testing.assert_close(result["era5"], original_gridded)
        torch.testing.assert_close(result["obs"][0][0], original_sparse)
        torch.testing.assert_close(result["obs"][1]["latitudes"][0], original_lats)

    def test_empty_sparse_batch(self) -> None:
        """Edge case: sparse batch with zero observations."""
        sparse_tensors = [torch.randn(1, 0, 3)]
        sparse_meta = {
            "latitudes": [torch.randn(0)],
            "longitudes": [torch.randn(0)],
            "timedeltas": [torch.randn(0)],
            "boundaries": [[]],
        }
        batch = {"obs": (sparse_tensors, sparse_meta)}
        result = _transfer_batch_to_device(batch, self.device)

        tensors, meta = result["obs"]
        assert tensors[0].shape == (1, 0, 3)
        assert meta["latitudes"][0].shape == (0,)
