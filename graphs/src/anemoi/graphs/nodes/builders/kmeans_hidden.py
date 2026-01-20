# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class KMeansHiddenNodes(BaseNodeBuilder):
    """Create a shared hidden mesh using K-Means over lat/lon node coordinates.

    Notes
    -----
    - Expects the *data* nodes to already exist in the graph (i.e. this builder
      should appear **after** your dataset node builder in the graph config).
    - Coordinates are expected in `graph[data_name].x` as (N, 2) in radians.

    Parameters
    ----------
    n_hidden : int
        Number of hidden nodes to create.
    data_names : list[str]
        Node types to draw coordinates from. Default: ['data'].
    seed : int
        RNG seed.
    batch_size : int
        MiniBatchKMeans batch size.
    max_iter : int
        MiniBatchKMeans max iterations.
    """

    def __init__(
        self,
        *,
        name: str,
        n_hidden: int = 2048,
        data_names: list[str] | None = None,
        seed: int = 0,
        batch_size: int = 8192,
        max_iter: int = 200,
    ) -> None:
        super().__init__(name=name)
        self.n_hidden = int(n_hidden)
        self.data_names = data_names or ["data"]
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)

    def _iter_coords(self, graph: HeteroData) -> Iterable[np.ndarray]:
        for node_name in self.data_names:
            if node_name not in graph.node_types:
                continue
            if "x" not in graph[node_name]:
                continue
            x = graph[node_name].x
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if x.ndim != 2 or x.shape[-1] != 2:
                continue
            yield x

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        coords = np.concatenate(list(self._iter_coords(graph)), axis=0)
        if coords.size == 0:
            raise RuntimeError(
                "KMeansHiddenNodes could not find any coordinates in graph[*].x. "
                "Make sure your dataset node builder ran first."
            )

        # Use a stable feature space for clustering: (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))
        lat = coords[:, 0]
        lon = coords[:, 1]
        xyz = np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
            axis=1,
        ).astype(np.float32)

        n_clusters = min(self.n_hidden, xyz.shape[0])
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.seed,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            n_init="auto",
        )
        km.fit(xyz)
        centers_xyz = km.cluster_centers_.astype(np.float32)

        # Convert back to (lat, lon) in radians
        x, y, z = centers_xyz[:, 0], centers_xyz[:, 1], centers_xyz[:, 2]
        lat_c = np.arctan2(z, np.sqrt(x * x + y * y))
        lon_c = np.arctan2(y, x)
        centers_ll = np.stack([lat_c, lon_c], axis=1).astype(np.float32)

        graph[self.name].x = torch.from_numpy(centers_ll)
        graph[self.name]._builder = self.__class__.__name__
        graph[self.name]._n_hidden = int(n_clusters)
        LOGGER.info("Created %d hidden nodes via KMeans.", n_clusters)
        return graph
