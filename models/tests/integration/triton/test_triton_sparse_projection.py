# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The Triton sparse projection matches the CSR matrix, in values and gradients."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.triton.utils import is_triton_available
from tests.grids import octahedral_coords


@pytest.fixture(autouse=True)
def needs_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_triton_available():
        pytest.skip("Triton not available")


def gaussian_provider(source: int, target: int, num_neighbours: int, backend: str) -> ProjectionGraphProvider:
    """Gaussian weights over the nearest points, from O``source`` to O``target``."""
    graph = HeteroData()
    for name, n in (("source", source), ("target", target)):
        graph[name].x = octahedral_coords(n)
        graph[name].num_nodes = graph[name].x.shape[0]
    config = {"num_nearest_neighbours": num_neighbours, "sigma": 2.0 / source, "target_node_name": "target"}
    return ProjectionGraphProvider.from_config(config, graph_data=graph, data_node_name="source", backend=backend)


def uneven_provider(
    num_sources: int, num_targets: int, seed: int, backend: str, longest_row: int = 40
) -> ProjectionGraphProvider:
    """Random weights with rows of 0 to ``longest_row`` entries."""
    rng = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, longest_row + 1, (num_targets,), generator=rng)
    targets = torch.repeat_interleave(torch.arange(num_targets), lengths)
    sources = torch.cat([torch.randperm(num_sources, generator=rng)[:length] for length in lengths.tolist()])
    graph = HeteroData()
    graph["source"].num_nodes = num_sources
    graph["target"].num_nodes = num_targets
    graph[("source", "to", "target")].edge_index = torch.stack([sources, targets])
    graph[("source", "to", "target")].weight = torch.rand(len(sources), generator=rng)
    return ProjectionGraphProvider(
        graph=graph,
        edges_name=("source", "to", "target"),
        edge_weight_attribute="weight",
        row_normalize=True,
        backend=backend,
    )


PROVIDERS = {
    "gaussian_o8_o12_k3": lambda backend: gaussian_provider(8, 12, 3, backend),
    "gaussian_o16_o40_k32": lambda backend: gaussian_provider(16, 40, 32, backend),
    "gaussian_o24_o48_k17": lambda backend: gaussian_provider(24, 48, 17, backend),
    "gaussian_o16_o40_k96": lambda backend: gaussian_provider(16, 40, 96, backend),
    "uneven_rows": lambda backend: uneven_provider(3000, 2000, seed=0, backend=backend),
    "uneven_short_rows": lambda backend: uneven_provider(3000, 2000, seed=2, backend=backend, longest_row=20),
}


def forward_backward(project, x: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_(True)
    out = project(x)
    out.backward(grad)
    return out.detach(), x.grad


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("channels", [5, 70, 130])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_matches_the_matrix(provider_name: str, channels: int, dtype: torch.dtype) -> None:
    provider = PROVIDERS[provider_name]("triton")
    num_targets, num_sources = provider.shape
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, 3, num_sources, channels, device="cuda", dtype=dtype, generator=generator)
    grad = torch.randn(2, 3, num_targets, channels, device="cuda", dtype=dtype, generator=generator)
    dense = PROVIDERS[provider_name]("torch").get_edges().to_dense().to(device="cuda", dtype=dtype)

    out_t, grad_t = forward_backward(lambda x: SparseProjector()(x, provider), x, grad)
    out_r, grad_r = forward_backward(lambda x: torch.einsum("mn,...nc->...mc", dense, x), x, grad)

    # Different summation order, so compare to the dtype's rounding.
    tolerance = 16 * torch.finfo(dtype).eps
    torch.testing.assert_close(out_t, out_r, rtol=tolerance, atol=tolerance * out_r.abs().max().item())
    torch.testing.assert_close(grad_t, grad_r, rtol=tolerance, atol=tolerance * grad_r.abs().max().item())


def test_triton_spreads_nan_only_through_stored_entries() -> None:
    # The padding of short rows points at source 0 with weight 0.
    triton_provider = uneven_provider(300, 200, seed=1, backend="triton")
    torch_provider = uneven_provider(300, 200, seed=1, backend="torch")
    num_targets, num_sources = triton_provider.shape
    matrix = torch_provider.get_edges()
    row_lengths = matrix.crow_indices().diff()
    reads_source_0 = matrix.to_dense()[:, 0] != 0
    padded_rows = torch.nonzero((row_lengths > 0) & (row_lengths < row_lengths.max()) & ~reads_source_0).flatten()

    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, num_sources, 8, device="cuda", generator=generator)
    x[:, 0] = float("nan")
    grad = torch.randn(2, num_targets, 8, device="cuda", generator=generator)
    grad[:, padded_rows[0]] = float("nan")

    out_t, grad_t = forward_backward(lambda x: SparseProjector()(x, triton_provider), x, grad)
    out_r, grad_r = forward_backward(lambda x: SparseProjector()(x, torch_provider), x, grad)

    assert bool(out_t[:, reads_source_0].isnan().all())
    assert not bool(out_t[:, ~reads_source_0].isnan().any())
    torch.testing.assert_close(out_t, out_r, rtol=1e-5, atol=1e-5, equal_nan=True)
    torch.testing.assert_close(grad_t, grad_r, rtol=1e-5, atol=1e-5, equal_nan=True)
    assert not bool(grad_t[:, 0].isnan().any())


def test_triton_matches_torch_backend_in_chunks() -> None:
    provider = gaussian_provider(16, 40, 32, "triton")
    x = torch.randn(6, provider.shape[1], 40, device="cuda")

    expected = SparseProjector()(x, gaussian_provider(16, 40, 32, "torch"))
    out = SparseProjector(num_chunks=4)(x, provider)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_triton_keeps_the_input_dtype_and_reads_float32_weights() -> None:
    provider = gaussian_provider(16, 40, 32, "triton")
    x = torch.randn(2, provider.shape[1], 40, device="cuda")

    expected = SparseProjector()(x, provider)
    out = SparseProjector()(x.to(torch.bfloat16), provider)

    assert out.dtype == torch.bfloat16
    assert provider.get_table()[1].dtype == torch.float32
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_triton_projection_of_a_field_with_zero_channels() -> None:
    provider = gaussian_provider(16, 40, 32, "triton")
    x = torch.randn(2, provider.shape[1], 0, device="cuda", requires_grad=True)

    out = SparseProjector()(x, provider)
    out.sum().backward()

    assert out.shape == (2, provider.shape[0], 0)
    assert x.grad.shape == x.shape
