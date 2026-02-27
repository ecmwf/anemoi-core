# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.nodes import RegionalTriMeshNodes


def _reference_graph(coords_rad: np.ndarray) -> HeteroData:
    graph = HeteroData()
    graph["ref"].x = torch.tensor(coords_rad, dtype=torch.float32)
    return graph


def test_register_nodes_builds_staggered_triangular_rows():
    graph = _reference_graph(
        np.array(
            [
                [0.00, 0.00],
                [0.00, 0.08],
                [0.06, 0.00],
                [0.06, 0.08],
            ],
            dtype=np.float64,
        )
    )

    builder = RegionalTriMeshNodes(reference_node_name="ref", spacing_km=20.0, margin_km=0.0, name="hidden")
    graph = builder.register_nodes(graph)

    coords = graph["hidden"].x.detach().cpu().numpy().astype(np.float64)
    lat_values = np.sort(np.unique(np.round(coords[:, 0], decimals=7)))

    assert lat_values.size >= 2

    lat_step = float(np.median(np.diff(lat_values)))
    expected_lat_step = (np.sqrt(3.0) / 2.0) * (builder.spacing_km / EARTH_RADIUS)
    assert lat_step == pytest.approx(expected_lat_step, rel=0.05)

    row0 = np.sort(coords[np.isclose(coords[:, 0], lat_values[0], atol=1e-6), 1])
    row1 = np.sort(coords[np.isclose(coords[:, 0], lat_values[1], atol=1e-6), 1])

    assert row0.size >= 2
    assert row1.size >= 2

    lon_step = float(np.median(np.diff(row0)))
    phase = ((row1[0] - row0[0]) / lon_step) % 1.0

    assert phase == pytest.approx(0.5, abs=0.1)


def test_register_nodes_uses_mask_for_domain_bounds():
    graph = _reference_graph(
        np.array(
            [
                [0.00, 0.00],
                [0.01, 0.02],
                [0.90, 0.90],  # outlier that should be ignored by mask
            ],
            dtype=np.float64,
        )
    )
    graph["ref"]["domain_mask"] = torch.tensor([[True], [True], [False]])

    builder = RegionalTriMeshNodes(
        reference_node_name="ref",
        spacing_km=10.0,
        margin_km=0.0,
        mask_attr_name="domain_mask",
        name="hidden",
    )
    graph = builder.register_nodes(graph)

    coords = graph["hidden"].x.detach().cpu().numpy()

    assert coords.shape[0] > 0
    assert float(coords[:, 0].max()) < 0.2
    assert float(coords[:, 1].max()) < 0.2
