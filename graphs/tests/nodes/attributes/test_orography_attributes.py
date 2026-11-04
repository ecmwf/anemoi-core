# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import OrogElevationAttribute
from anemoi.graphs.nodes.attributes import OrogGradientAttribute
from anemoi.graphs.nodes.attributes import OrogGradientWeights


@pytest.fixture
def alpine_graph() -> HeteroData:
    """Regular 0.1 deg grid of nodes over the synthetic mountain of ``orography_file``."""
    lat_g, lon_g = np.meshgrid(np.arange(44.5, 47.5, 0.1), np.arange(6.5, 9.5, 0.1), indexing="ij")
    graph = HeteroData()
    graph["data"].x = torch.tensor(np.deg2rad(np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)), dtype=torch.float32)
    return graph


def test_elevation_attribute(alpine_graph, orography_file):
    raw = OrogElevationAttribute(orography_file, norm=None).compute(alpine_graph, "data")
    assert raw.shape == (alpine_graph["data"].num_nodes, 1)
    assert raw.dtype == torch.float32
    assert 2900 < raw.max() <= 3000

    normed = OrogElevationAttribute(orography_file).compute(alpine_graph, "data")
    assert normed.max() == pytest.approx(1.0)
    torch.testing.assert_close(normed, raw / raw.max())


def test_gradient_attribute_range(alpine_graph, orography_file):
    gradient = OrogGradientAttribute(orography_file, percentile_clip=90.0).compute(alpine_graph, "data")
    assert gradient.shape == (alpine_graph["data"].num_nodes, 1)
    assert gradient.min() >= 0.0 and gradient.max() == pytest.approx(1.0)
    # At least the steepest 10% of the nodes are clipped to 1.
    assert (gradient == 1.0).float().mean() >= 0.09


def test_gradient_attribute_flat(alpine_graph, flat_orography_file):
    gradient = OrogGradientAttribute(flat_orography_file).compute(alpine_graph, "data")
    assert torch.all(gradient == 0.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 3.0])
def test_gradient_weights(alpine_graph, orography_file, alpha):
    gradient = OrogGradientAttribute(orography_file).compute(alpine_graph, "data")
    weights = OrogGradientWeights(orography_file, alpha=alpha).compute(alpine_graph, "data")
    torch.testing.assert_close(weights, 1.0 + alpha * gradient)
    assert weights.min() >= 1.0 and weights.max() <= 1.0 + alpha + 1e-6


def test_gradient_weights_flat(alpine_graph, flat_orography_file):
    weights = OrogGradientWeights(flat_orography_file, alpha=2.0).compute(alpine_graph, "data")
    assert torch.all(weights == 1.0)


@pytest.mark.parametrize(
    "attr_cls, kwargs",
    [
        (OrogGradientWeights, {"alpha": -1.0}),
        (OrogGradientWeights, {"percentile_clip": 110.0}),
        (OrogGradientAttribute, {"percentile_clip": 0.0}),
    ],
)
def test_invalid_parameters(attr_cls, kwargs):
    with pytest.raises(AssertionError):
        attr_cls(orography_path="unused.nc", **kwargs)
