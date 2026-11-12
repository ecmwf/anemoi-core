# (C) Copyright 2026 Anemoi contributors.

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import CutOutNumpyAttribute


def _write_source(tmp_path, name, coordinates, values):
    coordinate_path = tmp_path / f"{name}_coordinates.npz"
    values_path = tmp_path / f"{name}_values.npy"
    np.savez(
        coordinate_path,
        latitudes=np.asarray([point[0] for point in coordinates]),
        longitudes=np.asarray([point[1] for point in coordinates]),
    )
    np.save(values_path, np.asarray(values, dtype=np.float32))
    return values_path, coordinate_path


def test_cutout_numpy_attribute_joins_regional_and_global_values(tmp_path):
    regional_values, regional_coordinates = _write_source(
        tmp_path,
        "regional",
        [(50.0, 10.0), (51.0, 11.0)],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    global_values, global_coordinates = _write_source(
        tmp_path,
        "global",
        [(-20.0, 100.0), (0.0, 0.0)],
        [[5.0, 6.0], [7.0, 8.0]],
    )
    graph = HeteroData()
    graph["data"].x = torch.deg2rad(torch.tensor([[0.0, 0.0], [51.0, 11.0], [-20.0, 100.0], [50.0, 10.0]]))
    graph["data"].cutout_mask = torch.tensor([False, True, False, True])

    result = CutOutNumpyAttribute(
        regional_values=regional_values,
        regional_coordinates=regional_coordinates,
        global_values=global_values,
        global_coordinates=global_coordinates,
    ).compute(graph, "data")

    torch.testing.assert_close(
        result,
        torch.tensor([[7.0, 8.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0]]),
    )


def test_cutout_numpy_attribute_accepts_postprocessed_column_mask(tmp_path):
    regional_values, regional_coordinates = _write_source(
        tmp_path,
        "regional",
        [(51.0, 11.0)],
        [[3.0, 4.0]],
    )
    global_values, global_coordinates = _write_source(
        tmp_path,
        "global",
        [(0.0, 0.0)],
        [[7.0, 8.0]],
    )
    graph = HeteroData()
    graph["data"].x = torch.deg2rad(torch.tensor([[0.0, 0.0], [51.0, 11.0]]))
    # Boolean node attributes are postprocessed to (num_nodes, 1) by the graph builder.
    graph["data"].cutout_mask = torch.tensor([[False], [True]])

    result = CutOutNumpyAttribute(
        regional_values=regional_values,
        regional_coordinates=regional_coordinates,
        global_values=global_values,
        global_coordinates=global_coordinates,
    ).compute(graph, "data")

    torch.testing.assert_close(result, torch.tensor([[7.0, 8.0], [3.0, 4.0]]))


def test_cutout_numpy_attribute_requires_mask(tmp_path):
    values, coordinates = _write_source(tmp_path, "source", [(0.0, 0.0)], [[1.0]])
    graph = HeteroData()
    graph["data"].x = torch.zeros((1, 2))

    attribute = CutOutNumpyAttribute(values, coordinates, values, coordinates)
    try:
        attribute.compute(graph, "data")
    except ValueError as error:
        assert "must be built before" in str(error)
    else:
        raise AssertionError("Expected a missing-mask error.")
