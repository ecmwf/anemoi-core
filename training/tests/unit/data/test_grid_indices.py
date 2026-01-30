# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.training.data.grid_indices import FullGrid
from anemoi.training.data.grid_indices import MaskedGrid


class TestFullGrid:
    """Test the FullGrid class."""

    @pytest.fixture
    def full_grid(self) -> FullGrid:
        """Create a FullGrid instance."""
        return FullGrid()

    def test_full_grid_get_shard_indices_with_none(self, full_grid: FullGrid) -> None:
        """Test get_shard_indices with None returns None."""
        assert full_grid.supporting_arrays == {}
        assert full_grid.get_shard_indices(None) is None

    def test_full_grid_get_shard_indices_with_array(self, full_grid: FullGrid) -> None:
        """Test get_shard_indices with array returns the same array."""
        indices_array = np.array([0, 1, 2, 3, 4])
        result_array = full_grid.get_shard_indices(indices_array)
        np.testing.assert_array_equal(result_array, indices_array)

    def test_full_grid_get_shard_indices_with_slice(self, full_grid: FullGrid) -> None:
        """Test get_shard_indices with slice returns the same slice."""
        indices_slice = slice(0, 10, 2)
        result_slice = full_grid.get_shard_indices(indices_slice)
        assert result_slice == indices_slice

    def test_full_grid_preserves_dtype(self, full_grid: FullGrid) -> None:
        """Test that get_shard_indices preserves the dtype of the input array."""
        # Preservers dtype
        indices_int32 = np.array([0, 1, 2], dtype=np.int32)
        result_int32 = full_grid.get_shard_indices(indices_int32)
        assert result_int32.dtype == np.int32


class TestMaskedGrid:
    """Test the MaskedGrid class."""

    @pytest.fixture
    def simple_grid_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create simple grid data for testing."""
        # Create a simple 5x5 lat-lon grid
        coords_1d = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        lat_grid, lon_grid = np.meshgrid(coords_1d, coords_1d, indexing="ij")
        latitudes = lat_grid.flatten()
        longitudes = lon_grid.flatten()

        # Create a mask: only the center point and its immediate neighbors
        mask = np.zeros(25, dtype=bool)
        mask[12] = True  # center point, (20, 20)

        return latitudes, longitudes, mask

    @pytest.fixture
    def masked_grid(self, simple_grid_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> MaskedGrid:
        """Create a MaskedGrid instance."""
        latitudes, longitudes, mask = simple_grid_data
        return MaskedGrid(
            latitudes=latitudes,
            longitudes=longitudes,
            mask=mask,
            mask_radius_km=1800.0,
        )

    def test_masked_grid_instantiation(self, simple_grid_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test that MaskedGrid can be instantiated."""
        latitudes, longitudes, mask = simple_grid_data
        grid = MaskedGrid(
            latitudes=latitudes,
            longitudes=longitudes,
            mask=mask,
            mask_radius_km=500.0,
        )

        assert grid is not None

    def test_masked_grid(self, masked_grid: MaskedGrid) -> None:
        """Test MaskedGrid properties."""
        assert masked_grid.coords_3d.shape == (len(masked_grid.latitudes), 3)
        np.testing.assert_allclose(np.linalg.norm(masked_grid.coords_3d, axis=1), 1.0, rtol=1e-10)
        assert masked_grid.grid_indices.dtype == np.int64
        assert "grid_indices" in masked_grid.supporting_arrays
        assert isinstance(masked_grid.supporting_arrays["grid_indices"], np.ndarray)
        assert set(masked_grid.grid_indices) == {6, 7, 8, 11, 12, 13, 16, 17, 18}

    def test_masked_grid_get_shard_indices_with_array(self, masked_grid: MaskedGrid) -> None:
        """Test get_shard_indices with array returns indexed grid indices."""
        shard_indices = np.array([0, 1, 2])
        result = masked_grid.get_shard_indices(shard_indices)
        expected = masked_grid.grid_indices[shard_indices]
        np.testing.assert_array_equal(result, expected)
