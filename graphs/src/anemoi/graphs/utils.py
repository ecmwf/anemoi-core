# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from enum import Enum

import torch
from sklearn.neighbors import NearestNeighbors

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian


def get_distributed_device() -> torch.device:
    """Get the distributed device.

    Returns
    -------
    torch.device
        The distributed device.
    """
    if torch.cuda.is_available():
        import os

        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = "cpu"

    return device


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

    nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)

    nearest_neighbour.fit(coords_rad)

    return nearest_neighbour


def get_grid_reference_distance(coords_rad: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Get the reference distance of the grid.

    It is the maximum distance of a node in the mesh with respect to its nearest neighbour.

    Parameters
    ----------
    coords_rad : torch.Tensor
        corrdinates in radians
    mask : torch.Tensor, optional
        mask to remove nodes, by default None

    Returns
    -------
    float
        The reference distance of the grid.
    """
    xyz = latlon_rad_to_cartesian(coords_rad)
    nearest_neighbours = get_nearest_neighbour(xyz, mask)
    dists, _ = nearest_neighbours.kneighbors(xyz, n_neighbors=2, return_distance=True)
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
