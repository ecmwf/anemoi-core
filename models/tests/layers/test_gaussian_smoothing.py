# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Gaussian smoothing over a neighbour list, and the neighbour list provider."""

import copy

import pytest
import torch

from anemoi.graphs.builders import build_node_to_node_projection_subgraph
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.utils import haversine_distance
from anemoi.models.layers import gaussian_smoothing
from anemoi.models.layers.gaussian_smoothing import gaussian_smooth
from anemoi.models.layers.gaussian_smoothing import squared_neighbour_distances
from anemoi.models.layers.graph_provider import NeighbourListProvider
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from tests.grids import octahedral_graph

# The graph builder puts the neighbour tables on the GPU when there is one; these tests run the
# torch backend on the CPU, so they ask for the tables there.
CPU = torch.device("cpu")


def dense_smoothing_matrix(neighbours: torch.Tensor, unit_vectors: torch.Tensor, sigma: float) -> torch.Tensor:
    """The smoothing as a dense float64 matrix: row ``n`` holds point ``n``'s normalised weights."""
    num_points = neighbours.shape[0]
    squared_distances = squared_neighbour_distances(unit_vectors, neighbours, slice(None), torch.float64)
    weights = torch.exp(-squared_distances / (2 * sigma**2))
    weights = weights / weights.sum(dim=1, keepdim=True)
    matrix = torch.zeros(num_points, num_points, dtype=torch.float64)
    matrix.scatter_add_(1, neighbours.long(), weights)
    return matrix


@pytest.fixture(scope="module")
def neighbour_list() -> NeighbourListProvider:
    return NeighbourListProvider(octahedral_graph(6), "data", num_neighbours=12)


def test_provider_tables(neighbour_list: NeighbourListProvider) -> None:
    graph = octahedral_graph(6)
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    num_points = graph["data"].num_nodes

    assert neighbours.shape == (num_points, 12) and neighbours.dtype == torch.int32
    assert unit_vectors.shape == (num_points, 3) and unit_vectors.dtype == torch.float32
    torch.testing.assert_close(unit_vectors.norm(dim=1), torch.ones(num_points), rtol=0, atol=1e-6)

    # Each point is one of its own neighbours.
    assert bool((neighbours == torch.arange(num_points, dtype=torch.int32)[:, None]).any(dim=1).all())

    squared_distances = squared_neighbour_distances(unit_vectors, neighbours, slice(None), torch.float64)

    # The distances worked out from the unit vectors are the great-circle distances of the graph.
    coords = graph["data"].x.double()
    targets = torch.arange(num_points).repeat_interleave(12)
    expected = haversine_distance(coords[neighbours.reshape(-1).long()], coords[targets]) ** 2
    torch.testing.assert_close(squared_distances.reshape(-1), expected, rtol=1e-5, atol=1e-10)


def test_provider_has_the_neighbours_of_the_smoother_graph(neighbour_list: NeighbourListProvider) -> None:
    neighbours, _ = neighbour_list.get_neighbours(device=CPU)
    smoother = {"num_nearest_neighbours": 12, "sigma": 0.1}
    edges = build_node_to_node_projection_subgraph(octahedral_graph(6), "data", "data", smoother)[
        ("data", "to", "data")
    ]
    for point in range(neighbours.shape[0]):
        sources = edges.edge_index[0, edges.edge_index[1] == point]
        assert set(sources.tolist()) == set(neighbours[point].tolist())


def test_provider_shares_its_tables_when_deep_copied(neighbour_list: NeighbourListProvider) -> None:
    copied = copy.deepcopy(neighbour_list)
    assert copied.neighbours is neighbour_list.neighbours
    assert copied.unit_vectors is neighbour_list.unit_vectors


def test_provider_reach_is_the_median_distance_to_the_farthest_neighbour(
    neighbour_list: NeighbourListProvider,
) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    squared_distances = squared_neighbour_distances(unit_vectors, neighbours, slice(None), torch.float64)
    expected = float(squared_distances.amax(dim=1).sqrt().median())

    assert neighbour_list.typical_reach(num_samples=neighbours.shape[0]) == pytest.approx(expected)


def test_provider_needs_enough_nodes() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        NeighbourListProvider(octahedral_graph(1), "data", num_neighbours=1000)


@pytest.mark.parametrize("sigma", [0.02, 0.1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_torch_smoothing_matches_the_dense_matrix(neighbour_list, sigma: float, dtype: torch.dtype) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    matrix = dense_smoothing_matrix(neighbours, unit_vectors, sigma)
    x = torch.randn(2, 3, neighbours.shape[0], 5, dtype=dtype, generator=torch.Generator().manual_seed(0))

    out = gaussian_smooth(x, neighbours, unit_vectors, sigma, backend="torch")

    expected = torch.einsum("nm,btmc->btnc", matrix, x.double())
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(out.double(), expected, rtol=tolerance, atol=tolerance)
    assert out.dtype == dtype and out.shape == x.shape


def test_torch_smoothing_gradient_is_the_transpose(neighbour_list) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    matrix = dense_smoothing_matrix(neighbours, unit_vectors, 0.05)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(4, neighbours.shape[0], 3, dtype=torch.float64, generator=generator, requires_grad=True)
    v = torch.randn(4, neighbours.shape[0], 3, dtype=torch.float64, generator=generator)

    (gaussian_smooth(x, neighbours, unit_vectors, 0.05, backend="torch") * v).sum().backward()

    torch.testing.assert_close(x.grad, torch.einsum("mn,bmc->bnc", matrix, v), rtol=1e-12, atol=1e-12)


def test_torch_smoothing_passes_gradcheck() -> None:
    neighbours, unit_vectors = NeighbourListProvider(octahedral_graph(2), "data", num_neighbours=5).get_neighbours(
        device=CPU
    )
    x = torch.randn(2, neighbours.shape[0], 3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda field: gaussian_smooth(field, neighbours, unit_vectors, 0.1, backend="torch"), (x,)
    )


def test_torch_smoothing_gives_the_same_result_in_bands(neighbour_list, monkeypatch) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    x = torch.randn(2, neighbours.shape[0], 4, dtype=torch.float64, requires_grad=True)
    whole = gaussian_smooth(x, neighbours, unit_vectors, 0.05, backend="torch")
    (grad_whole,) = torch.autograd.grad(whole.sum(), x)

    # Few enough gathered values at once to work through the points in many bands.
    monkeypatch.setattr(gaussian_smoothing, "_TORCH_GATHER_ELEMENTS", 2 * 12 * 4 * 7)
    banded = gaussian_smooth(x, neighbours, unit_vectors, 0.05, backend="torch")
    (grad_banded,) = torch.autograd.grad(banded.sum(), x)

    torch.testing.assert_close(banded, whole, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(grad_banded, grad_whole, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("sigma", [0.03, 0.08])
def test_smoothing_matches_the_smoother_matrix(sigma: float) -> None:
    """With the same neighbours and width, the result equals the Gaussian smoothing matrix."""
    graph = octahedral_graph(6)
    neighbours, unit_vectors = NeighbourListProvider(graph, "data", num_neighbours=16).get_neighbours(device=CPU)
    subgraph = build_node_to_node_projection_subgraph(
        graph, "data", "data", {"num_nearest_neighbours": 16, "sigma": sigma}
    )
    provider = ProjectionGraphProvider(
        graph=subgraph,
        edges_name=("data", "to", "data"),
        edge_weight_attribute=DEFAULT_EDGE_WEIGHT_ATTRIBUTE,
        backend="torch",
    )
    x = torch.randn(2, 1, neighbours.shape[0], 6)

    expected = SparseProjector()(x, provider)
    out = gaussian_smooth(x, neighbours, unit_vectors, sigma, backend="torch")

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_smoothing_checks_its_arguments(neighbour_list) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    x = torch.randn(1, neighbours.shape[0], 2)
    with pytest.raises(ValueError, match="backend"):
        gaussian_smooth(x, neighbours, unit_vectors, 0.1, backend="numpy")
    with pytest.raises(ValueError, match="sigma"):
        gaussian_smooth(x, neighbours, unit_vectors, 0.0, backend="torch")
    with pytest.raises(ValueError, match="points"):
        gaussian_smooth(x[:, :-1], neighbours, unit_vectors, 0.1, backend="torch")
    with pytest.raises(ValueError, match="GPU"):
        gaussian_smooth(x, neighbours, unit_vectors, 0.1, backend="triton")


def test_torch_smoothing_of_a_field_with_zero_channels(neighbour_list: NeighbourListProvider) -> None:
    neighbours, unit_vectors = neighbour_list.get_neighbours(device=CPU)
    x = torch.randn(2, 3, neighbours.shape[0], 0, requires_grad=True)

    out = gaussian_smooth(x, neighbours, unit_vectors, 0.1, backend="torch")
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad.shape == x.shape
