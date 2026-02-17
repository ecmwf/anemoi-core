import numpy as np
import pytest

from anemoi.training.diagnostics.focus_area import BoundingBoxSpatialMask
from anemoi.training.diagnostics.focus_area import NodeAttributeSpatialMask
from anemoi.training.diagnostics.focus_area import NoOpSpatialMask
from anemoi.training.diagnostics.focus_area import build_spatial_mask

_rng = np.random.default_rng()


def test_noop_mask() -> None:
    latlons = _rng.random((5, 2))
    field = _rng.random((5, 3))
    mask = NoOpSpatialMask()
    out_latlons, out_field = mask.apply({}, latlons, field)
    np.testing.assert_array_equal(out_latlons, latlons)
    np.testing.assert_array_equal(out_field, field)


def test_node_attribute_mask() -> None:
    latlons = np.array([[0, 0], [1, 1], [2, 2]])
    graph_data = {"data": {"mask": np.array([[0], [2]])}}
    field = _rng.random((3, 4))
    mask = NodeAttributeSpatialMask("mask")
    out_latlons, out_field = mask.apply(graph_data, latlons, field)
    np.testing.assert_array_equal(out_latlons[:, 0], [0, 2])
    assert out_field.shape[0] == 2


def test_node_attribute_missing() -> None:
    mask = NodeAttributeSpatialMask("missing")
    with pytest.raises(AssertionError):
        mask.compute_mask({"data": {}}, np.zeros((1, 2)))


def test_bbox_mask() -> None:
    latlons = np.array([[0, 0], [5, 5], [10, 10]])
    field = _rng.random((3, 2))
    mask = BoundingBoxSpatialMask((1, 1, 6, 6))
    out_latlons, out_field = mask.apply({}, latlons, field)
    np.testing.assert_array_equal(out_latlons[:, 0], [5])
    assert out_field.shape[0] == 1


def test_bbox_invalid() -> None:
    with pytest.raises(AssertionError):
        BoundingBoxSpatialMask((10, 0, 5, 5))


def test_build_spatial_mask_factory() -> None:
    assert isinstance(build_spatial_mask(), NoOpSpatialMask)
    assert isinstance(build_spatial_mask(node_attribute_name="a"), NodeAttributeSpatialMask)
    assert isinstance(build_spatial_mask(latlon_bbox=(0, 0, 1, 1)), BoundingBoxSpatialMask)
