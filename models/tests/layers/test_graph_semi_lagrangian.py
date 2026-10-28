# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.adr.layers import GraphSpatialMixer
from anemoi.models.layers.edge_stencil import EdgeDepthwiseConv
from anemoi.models.layers.geometry import MeshGeometry
from anemoi.models.layers.layout import GraphLayout
from anemoi.models.layers.semi_lagrangian import KNNSampler
from anemoi.models.layers.semi_lagrangian import departure_from_local_angles
from anemoi.models.layers.semi_lagrangian import knn_on_sphere
from anemoi.models.layers.semi_lagrangian import latlon_to_unit_vector
from anemoi.models.layers.semi_lagrangian import typical_node_spacing
from anemoi.models.layers.utils import load_layer_kernels
from tests.layers.graph_mesh import expand_edges
from tests.layers.graph_mesh import fibonacci_nodes
from tests.layers.graph_mesh import hidden_nodes
from tests.layers.graph_mesh import knn_edges


def _smooth_field(points: torch.Tensor) -> torch.Tensor:
    """A smooth function on the sphere, written in terms of 3D position so it has no pole artefacts."""
    x, y, z = points.unbind(-1)
    return x + 0.5 * y * z + 0.3 * z**2 - 0.2 * x * y


def _geometry(lat: torch.Tensor, lon: torch.Tensor, **kwargs) -> MeshGeometry:
    return MeshGeometry(hidden_nodes=hidden_nodes(lat, lon), edge_dim=4, **kwargs)


def _layout(lat: torch.Tensor, lon: torch.Tensor, batch_size: int = 1, **kwargs) -> GraphLayout:
    """Layout on one GPU, with edges from the three nearest nodes."""
    num_nodes = lat.shape[0]
    edge_attr, edge_index = expand_edges(*knn_edges(lat, lon, 3), num_nodes, batch_size)
    return _geometry(lat, lon, **kwargs).bind(
        torch.zeros(batch_size * num_nodes, 1),
        batch_size,
        GraphShardInfo(nodes=[batch_size * num_nodes]),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )


def test_knn_on_sphere_finds_the_nearest_points():
    torch.manual_seed(0)
    points = torch.nn.functional.normalize(torch.randn(300, 3), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    expected = torch.cdist(queries, points).topk(5, largest=False).indices
    # Small chunks, so that the search runs over several of them.
    torch.testing.assert_close(knn_on_sphere(points, queries, 5, chunk_elements=1000), expected)


def test_typical_node_spacing_of_an_even_mesh():
    lat, lon = fibonacci_nodes(2000)
    spacing = typical_node_spacing(latlon_to_unit_vector(lat, lon).float())
    # About the side of the area per node.
    assert spacing == pytest.approx(math.sqrt(4 * math.pi / 2000), rel=0.15)


def test_layout_round_trip_single_process():
    lat, lon = fibonacci_nodes(40)
    layout = _layout(lat, lon, batch_size=2)
    x = torch.randn(2 * 40, 3, 5)
    field = layout.to_field(x)
    assert field.shape == (2, 3, 5, 40)
    torch.testing.assert_close(field[1, 2, :, 7], x[40 + 7, 2])
    torch.testing.assert_close(layout.from_field(field, 3), x)


@pytest.mark.parametrize("cartesian_displacement", [False, True])
def test_zero_displacement_is_identity(cartesian_displacement):
    lat, lon = fibonacci_nodes(500)
    heads = 3
    layer = KNNSampler(num_heads=heads, cartesian_displacement=cartesian_displacement)
    values = torch.randn(2 * 500, heads, 4)
    displacement = torch.zeros(2 * 500, heads, layer.displacement_dim)
    moved = layer.sample(values, displacement, _layout(lat, lon, batch_size=2))
    # A departure point on a node gives the neighbours a weight of order smoothing^2 relative to
    # the node itself.
    torch.testing.assert_close(moved, values, atol=2e-4, rtol=0)


def test_sampling_reads_from_departure_points():
    """Values are read at the departure points of displacements up to about 20 degrees.

    The weighted average of the nearest nodes is first-order accurate, so the error scales with
    the node spacing (about 4 degrees here).
    """
    torch.manual_seed(0)
    lat, lon = fibonacci_nodes(2000)
    heads = 3
    layer = KNNSampler(num_heads=heads)
    position = latlon_to_unit_vector(lat, lon)
    values = _smooth_field(position)[:, None, None].expand(-1, heads, 1).float().contiguous()

    displacement = (torch.rand(2000, heads, 2, dtype=torch.float64) * 2 - 1) * 0.35
    departure = latlon_to_unit_vector(*departure_from_local_angles(lat[:, None], lon[:, None], displacement))

    moved = layer.sample(values, displacement.float(), _layout(lat, lon)).squeeze(-1)
    error = (moved - _smooth_field(departure).float()).abs()
    assert error.mean() < 1.5e-2
    assert error.max() < 6e-2


@pytest.mark.parametrize("component", ["east", "north"])
def test_cartesian_and_local_displacements_agree_along_one_direction(component):
    lat, lon = fibonacci_nodes(300)
    local = KNNSampler(num_heads=1)
    cartesian = KNNSampler(num_heads=1, cartesian_displacement=True)
    east_dir = torch.stack([-lon.sin(), lon.cos(), torch.zeros_like(lon)], dim=-1)
    north_dir = torch.stack([-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()], dim=-1)

    angle = 0.3
    local_displacement = torch.zeros(300, 1, 2)
    local_displacement[..., 0 if component == "east" else 1] = angle
    vector = (angle * (east_dir if component == "east" else north_dir)).float().view(-1, 1, 3)

    layout = _layout(lat, lon)
    torch.testing.assert_close(
        cartesian.departure_points(vector, layout),
        local.departure_points(local_displacement, layout),
        atol=1e-5,
        rtol=0,
    )


def test_gradient_reaches_the_displacements():
    torch.manual_seed(0)
    lat, lon = fibonacci_nodes(300)
    layer = KNNSampler(num_heads=2)
    values = _smooth_field(latlon_to_unit_vector(lat, lon))[:, None, None].expand(-1, 2, 1).float().contiguous()
    displacement = ((torch.rand(300, 2, 2) * 2 - 1) * 0.2).requires_grad_()
    layer.sample(values, displacement, _layout(lat, lon)).sum().backward()
    assert displacement.grad.isfinite().all()
    assert displacement.grad.abs().sum() > 0


def test_graph_low_rank_bias_is_shared_over_the_batch_and_single_valued_at_the_poles():
    torch.manual_seed(0)
    # Two points on each pole at different longitudes, and two elsewhere.
    lat = torch.tensor([math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.3, -0.7], dtype=torch.float64)
    lon = torch.tensor([0.0, 2.0, 1.0, 4.0, 1.5, 3.0], dtype=torch.float64)
    geometry = _geometry(lat, lon, bias_rank=3, bias_base_maps=2, bias_lat_degree=4, bias_lon_wavenumber=3)
    bias = geometry.position_bias(5, load_layer_kernels())
    with torch.no_grad():
        for parameter in (bias.coefficients, bias.lat_profiles, bias.lon_profiles):
            parameter.normal_()

    out = bias(torch.zeros(2 * 6, 5), _layout(lat, lon, batch_size=2)).view(2, 6, 5)
    torch.testing.assert_close(out[0], out[1])
    torch.testing.assert_close(out[0, 0], out[0, 1], atol=1e-5, rtol=0)
    torch.testing.assert_close(out[0, 2], out[0, 3], atol=1e-5, rtol=0)
    assert (out[0, 4] - out[0, 5]).abs().sum() > 0


def test_graph_spatial_mixer_rejects_unknown_block():
    with pytest.raises(ValueError, match="'edge_conv', 'transformer' or 'gnn'"):
        GraphSpatialMixer(8, 8, edge_dim=4, layer_kernels=load_layer_kernels(), block="mlp")


def test_edge_depthwise_conv_is_the_weighted_mean_of_the_neighbours():
    """Each node's new value is (1 / neighbours) * sum_j Linear(edge_attr_ij) * x_j, channel by channel."""
    torch.manual_seed(0)
    num_nodes, num_neighbours, channels = 40, 3, 5
    lat, lon = fibonacci_nodes(num_nodes)
    edge_attr, edge_index = knn_edges(lat, lon, num_neighbours)
    layout = _layout(lat, lon)
    edge_attr, edge_index = layout.edge_attr, layout.edge_index

    stencil = EdgeDepthwiseConv(channels, edge_attr.shape[1], load_layer_kernels())
    torch.nn.init.normal_(stencil.weight.weight)
    x = torch.randn(num_nodes, channels)

    weight = stencil.weight(edge_attr)
    expected = torch.zeros_like(x)
    for edge, (source, destination) in enumerate(edge_index.T.tolist()):
        expected[destination] += weight[edge] * x[source]
    expected /= num_neighbours  # every node has the same number of neighbours here

    torch.testing.assert_close(stencil(x, layout), expected, atol=1e-5, rtol=1e-5)


def test_edge_depthwise_conv_keeps_the_channels_apart():
    """Changing one channel of the input changes only that channel of the output."""
    torch.manual_seed(0)
    lat, lon = fibonacci_nodes(40)
    layout = _layout(lat, lon)
    stencil = EdgeDepthwiseConv(5, 4, load_layer_kernels())
    torch.nn.init.normal_(stencil.weight.weight)

    x = torch.randn(40, 5)
    changed = x.clone()
    changed[:, 2] += 1.0
    difference = (stencil(changed, layout) - stencil(x, layout)).abs().sum(0)

    assert difference[2] > 0
    torch.testing.assert_close(difference[[0, 1, 3, 4]], torch.zeros(4), atol=1e-6, rtol=0)
