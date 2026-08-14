# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from importlib.util import find_spec

import torch
from sklearn.neighbors import NearestNeighbors

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian

LOGGER = logging.getLogger(__name__)

FORCE_CPU_ENV_VAR = "ANEMOI_GRAPHS_FORCE_CPU"
DISABLE_PYG_LIB_ENV_VAR = "ANEMOI_GRAPHS_DISABLE_PYG_LIB"


def get_distributed_device() -> torch.device:
    """Get the device that graph building should use on this rank.
    Also makes the current CUDA device match the returned device.

    Set ``ANEMOI_GRAPHS_FORCE_CPU=1`` to build graphs on the CPU instead.

    Returns
    -------
    torch.device
        The device to build the graph on.
    """
    if os.environ.get(FORCE_CPU_ENV_VAR):
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        LOGGER.warning(
            "SLURM_LOCALID=%d but only %d CUDA device(s) are visible; building the graph on "
            "cuda:%d. Check that the number of tasks per node matches the number of GPUs.",
            local_rank,
            device_count,
            local_rank % device_count,
        )
        local_rank = local_rank % device_count

    # Keep the current device in sync with where the data will live - see docstring.
    torch.cuda.set_device(local_rank)

    return torch.device(f"cuda:{local_rank}")


@contextmanager
def cuda_device_of(device: torch.device | str | None) -> Iterator[None]:
    """Temporarily make the current CUDA device the one ``device`` refers to.

    Defence in depth for kernels that lack their own device guard (see get_distributed_device).
    A no-op for CPU tensors and when no device is given, so it is safe to wrap call sites unconditionally.
    """
    device = torch.device(device) if device is not None else None
    if device is None or device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield


def pyg_lib_available() -> bool:
    """Whether the pyg-lib accelerated neighbour-search kernels should be used.

    Set ANEMOI_GRAPHS_DISABLE_PYG_LIB=1 to fall back to the scikit-learn implementation.
    """
    if os.environ.get(DISABLE_PYG_LIB_ENV_VAR):
        return False

    return find_spec("pyg_lib") is not None


def get_nearest_neighbour(coords_rad: torch.Tensor, mask: torch.Tensor | None = None) -> NearestNeighbors:
    """Get NearestNeighbour object fitted to coordinates.

    Parameters
    ----------
    coords_rad : torch.Tensor
        corrdinates in radians
    mask : torch.Tensor, optional
        mask to remove nodes, by default None

    Returns
    -------
    NearestNeighbors
        fitted NearestNeighbour object
    """
    assert mask is None or mask.shape == (
        coords_rad.shape[0],
        1,
    ), "Mask must have the same shape as the number of nodes."

    if isinstance(coords_rad, torch.Tensor):
        coords_rad = coords_rad.detach().cpu()

    nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)

    nearest_neighbour.fit(coords_rad)

    return nearest_neighbour


def get_grid_reference_distance(
    coords_rad: torch.Tensor, mask: torch.Tensor | None = None, use_cartesian: bool = True
) -> float:
    """Get the reference distance of the grid.

    It is the maximum distance of a node in the mesh with respect to its nearest neighbour.

    Parameters
    ----------
    coords_rad : torch.Tensor
        corrdinates in radians
    mask : torch.Tensor, optional
        mask to remove nodes, by default None
    use_cartesian : bool, optional
        Whether to convert coordinates to Cartesian before computing distances. Defaults to True.

    Returns
    -------
    float
        The reference distance of the grid.
    """
    points = latlon_rad_to_cartesian(coords_rad) if use_cartesian else coords_rad
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu()
    nearest_neighbours = get_nearest_neighbour(points, mask)
    dists, _ = nearest_neighbours.kneighbors(points, n_neighbors=2, return_distance=True)
    return dists[dists > 0].max()


def concat_edges(edge_indices1: torch.Tensor, edge_indices2: torch.Tensor) -> torch.Tensor:
    """Concat edges

    Parameters
    ----------
    edge_indices1: torch.Tensor
        Edge indices of the first set of edges. Shape: (2, num_edges1)
    edge_indices2: torch.Tensor
        Edge indices of the second set of edges. Shape: (2, num_edges2)

    Returns
    -------
    torch.Tensor
        Concatenated edge indices.
    """
    return torch.unique(torch.cat([edge_indices1, edge_indices2], axis=1), dim=1)


def intersect_edges(edge_indices1: torch.Tensor, edge_indices2: torch.Tensor) -> torch.Tensor:
    """Intersect two sets of edges, keeping only edges present in both.

    Parameters
    ----------
    edge_indices1 : torch.Tensor
        Edge indices of the first set of edges. Shape: (2, num_edges1).
    edge_indices2 : torch.Tensor
        Edge indices of the second set of edges. Shape: (2, num_edges2).

    Returns
    -------
    torch.Tensor
        The edges (columns) that appear in both inputs, in the column order of
        ``edge_indices1``. Shape: (2, num_mutual_edges). Assumes non-negative
        indices (always true for node indices).
    """
    if edge_indices1.numel() == 0 or edge_indices2.numel() == 0:
        return torch.empty((2, 0), dtype=torch.int64)

    edge_indices1 = edge_indices1.to(torch.int64)
    edge_indices2 = edge_indices2.to(torch.int64)

    # Encode each (row0, row1) column as a single integer so membership can be
    # tested with torch.isin. The stride must exceed every row-1 index.
    stride = max(int(edge_indices1[1].max()), int(edge_indices2[1].max())) + 1
    keys1 = edge_indices1[0] * stride + edge_indices1[1]
    keys2 = edge_indices2[0] * stride + edge_indices2[1]

    mask = torch.isin(keys1, keys2)
    return edge_indices1[:, mask]


def haversine_distance(source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Haversine distance.

    Parameters
    ----------
    source_coords : torch.Tensor of shape (N, 2)
        Source coordinates in radians.
    target_coords : torch.Tensor of shape (N, 2)
        Destination coordinates in radians.

    Returns
    -------
    torch.Tensor of shape (N,)
        Haversine distance between source and destination coordinates.
    """
    dlat = target_coords[:, 0] - source_coords[:, 0]
    dlon = target_coords[:, 1] - source_coords[:, 1]
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(source_coords[:, 0]) * torch.cos(target_coords[:, 0]) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return c


class NodesAxis(Enum):
    SOURCE = 0
    TARGET = 1


def get_edge_attributes(config: dict, source_name: str, target_name: str) -> dict:
    """Get edge attributes out of a graph config

    Parameters
    ----------
    config : dict
        The graph configuration.
    source_name : str
        Name of source nodes of edges to be considered
    target_name : str
        Name of target nodes of edges to be considered
    Returns
    -------
    dict
        Dictionary of the form {attribute_name: attribute}
    """
    attrs = {}
    for edges_config in config.get("edges", {}):
        if edges_config["source_name"] == source_name and edges_config["target_name"] == target_name:
            attrs.update(edges_config["attributes"])
    return attrs
