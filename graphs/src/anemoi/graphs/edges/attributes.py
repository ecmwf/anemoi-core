# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch_geometric.data.storage import NodeStorage
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter

from anemoi.graphs.edges.directional import compute_directions
from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import NodesAxis
from anemoi.graphs.utils import get_distributed_device
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeAttributeBuilder(MessagePassing, NormaliserMixin, ABC):
    """Base class for edge attribute builders."""

    node_attr_name: str = None
    norm_by_group: bool = False

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.norm = norm
        self.dtype = dtype
        self.device = get_distributed_device()
        if self.node_attr_name is None:
            error_msg = f"Class {self.__class__.__name__} must define 'node_attr_name' either as a class attribute or in __init__"
            raise TypeError(error_msg)

    def subset_node_information(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> PairTensor:
        if self.node_attr_name in source_nodes:
            source_nodes_data = source_nodes[self.node_attr_name].to(self.device)
        else:
            source_nodes_data = None
            LOGGER.warning(
                "The attribute %s is not in the source nodes.", self.node_attr_name
            )

        if self.node_attr_name in target_nodes:
            target_nodes_data = target_nodes[self.node_attr_name].to(self.device)
        else:
            target_nodes_data = None
            LOGGER.warning(
                "The attribute %s is not in the target nodes.", self.node_attr_name
            )

        return source_nodes_data, target_nodes_data

    def forward(
        self, x: tuple[NodeStorage, NodeStorage], edge_index: Adj, size: Size = None
    ) -> torch.Tensor:
        x = self.subset_node_information(*x)
        return self.propagate(edge_index.to(self.device), x=x, size=size)

    @abstractmethod
    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor: ...

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_features = self.compute(x_i, x_j)

        if edge_features.ndim == 1:
            edge_features = edge_features.unsqueeze(-1)

        return edge_features

    def aggregate(
        self, edge_features: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None
    ) -> torch.Tensor:
        return self.normalise(edge_features, index, dim_size)


class BasePositionalBuilder(BaseEdgeAttributeBuilder, ABC):
    node_attr_name: str = "x"
    _idx_lat: int = 0
    _idx_lon: int = 1


class EdgeLength(BasePositionalBuilder):
    """Computes edge length for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance(x_i, x_j)
        return edge_length


class EdgeDirection(BasePositionalBuilder):
    """Computes edge direction for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_dirs = compute_directions(source_coords=x_j, target_coords=x_i)
        return edge_dirs


class DirectionalHarmonics(EdgeDirection):
    """Computes directional harmonics from edge directions.

    Builds directional harmonics [sin(mψ), cos(mψ)]_{m=1..order} from per-edge
    2D directions (dx, dy). Returns shape [N, 2*order].

    Attributes
    ----------
    order : int
        The maximum order of harmonics to compute.
    norm : str | None
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".

    Methods
    -------
    compute(x_i, x_j)
        Compute directional harmonics from edge directions.
    """

    def __init__(
        self, order: int = 3, norm: str | None = None, dtype: str = "float32"
    ) -> None:
        self.order = order
        super().__init__(norm=norm, dtype=dtype)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Get the 2D direction vectors [dx, dy]
        edge_dirs = compute_directions(source_coords=x_j, target_coords=x_i)

        # Compute the angle ψ from the direction vectors
        psi = torch.atan2(edge_dirs[:, 1], edge_dirs[:, 0])  # atan2(dy, dx)

        # Build harmonics: [sin(ψ), cos(ψ), sin(2ψ), cos(2ψ), ..., sin(order*ψ), cos(order*ψ)]
        harmonics = []
        for m in range(1, self.order + 1):
            harmonics.append(torch.sin(m * psi))
            harmonics.append(torch.cos(m * psi))

        # Stack into shape [N, 2*order]
        return torch.stack(harmonics, dim=1)


class Azimuth(BasePositionalBuilder):
    """Compute the azimuth of the edge.

    Attributes
    ----------
    norm : str | None
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".
    invert : bool
        Whether to invert the edge lengths, i.e. 1 - edge_length. Defaults to False.

    Methods
    -------
    compute(x_i, x_j)
        Compute edge lengths attributes.

    References
    ----------
    - https://www.movable-type.co.uk/scripts/latlong.html
    """

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Forward bearing. x_i, x_j must be radians.
        a11 = torch.cos(x_i[:, self._idx_lat]) * torch.sin(x_j[:, self._idx_lat])
        a12 = (
            torch.sin(x_i[:, self._idx_lat])
            * torch.cos(x_j[:, self._idx_lat])
            * torch.cos(x_j[..., self._idx_lon] - x_i[..., self._idx_lon])
        )
        a1 = a11 - a12
        a2 = torch.sin(x_j[..., self._idx_lon] - x_i[..., self._idx_lon]) * torch.cos(
            x_j[:, self._idx_lat]
        )
        edge_dirs = torch.atan2(a2, a1)

        return edge_dirs


class BaseBooleanEdgeAttributeBuilder(BaseEdgeAttributeBuilder, ABC):
    """Base class for boolean edge attributes."""

    def __init__(self) -> None:
        super().__init__(norm=None, dtype="bool")


class BaseEdgeAttributeFromNodeBuilder(BaseBooleanEdgeAttributeBuilder, ABC):
    """Base class for propagating an attribute from the nodes to the edges."""

    nodes_axis: NodesAxis | None = None

    def __init__(self, node_attr_name: str) -> None:
        self.node_attr_name = node_attr_name
        super().__init__()
        if self.nodes_axis is None:
            raise AttributeError(
                f"{self.__class__.__name__} class must set 'nodes_axis' attribute."
            )

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        node_attr = (x_j, x_i)[self.nodes_axis.value]
        assert (
            node_attr is not None
        ), f"The node attribute specified for {self.node_attr_name} cannot be found in the nodes."
        return node_attr


class AttributeFromSourceNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the source node to the edge."""

    nodes_axis = NodesAxis.SOURCE


class AttributeFromTargetNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the target node to the edge."""

    nodes_axis = NodesAxis.TARGET


class RadialBasisFeatures(EdgeLength):
    """Radial basis features from edge distances using Gaussian RBFs.

    Computes Gaussian radial basis function features from normalized great-circle distances:
    phi_r = [exp(-((α - c)/σ)²) for c in centers], where α = r_ij / r_scale.

    Provides RBF features via per-node adaptive scaling.
    By default, each destination node's edges are normalized by that node's maximum edge length.
    RBF features are normalized per target node per RBF center: within each RBF center,
    all edges pointing to the same target node have values that sum to 1 (L1 norm).

    Parameters
    ----------
    r_scale : float | None, optional
        Global scale factor for normalizing distances. Default is None.
        If None: Use per-node adaptive scaling (max edge length per destination node).
        If float: Use global scale for all nodes.
    centers : list of float, optional
        RBF center positions along normalized distance axis [0, 1].
        Default is [0.0, 0.25, 0.5, 0.75, 1.0].
    sigma : float, optional
        Width (standard deviation) of Gaussian RBF functions. Default is 0.2.
        Controls how localized each basis function is around its center.
    epsilon : float, optional
        Small constant to avoid division by zero. Default is 1e-10.
    dtype : str, optional
        Data type for computations. Default is "float32".

    Note
    ----
    RBF features are normalized per target node per RBF center.
    Within each RBF center, all edges to the same target node sum to 1.

    Methods
    -------
    compute(x_i, x_j)
        Compute raw edge distances (RBF computation happens in aggregate).
    aggregate(edge_features, index, ptr, dim_size)
        Compute RBF features with adaptive scaling and per-target-node normalization.

    Examples
    --------
    # Default: per-node adaptive scaling with grouped normalization
    rbf = RadialBasisFeatures()

    # To use global scale
    rbf_global = RadialBasisFeatures(r_scale=1.0)

    # Custom RBF centers and width
    rbf_custom = RadialBasisFeatures(centers=[0.0, 0.33, 0.67, 1.0], sigma=0.15)

    Notes
    -----
    - Closer edges → higher values at low-distance centers (0.0, 0.25)
    - Farther edges → higher values at high-distance centers (0.75, 1.0)
    """

    norm_by_group: bool = True  # normalise the RBF features per destination node

    def __init__(
        self,
        r_scale: float | None = None,
        centers: list[float] | None = None,
        sigma: float = 0.2,
        norm: str = "l1",
        epsilon: float = 1e-10,
        dtype: str = "float32",
    ) -> None:
        self.epsilon = epsilon
        self.r_scale = r_scale

        if self.r_scale is not None and self.r_scale < self.epsilon:
            LOGGER.warning(
                "r_scale (%f) is too small (< epsilon=%f). Clamping to epsilon to avoid division by zero.",
                self.r_scale,
                self.epsilon,
            )
            self.r_scale = self.epsilon

        self.centers = centers if centers is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

        # Normalize centers if using global scaling
        if self.r_scale is not None:
            self.centers = [c / self.r_scale for c in self.centers]

        # Check that centers are in the range [0, 1]
        assert all(
            0.0 <= c <= 1.0 for c in self.centers
        ), f"RBF centers must be in range [0, 1] (or [0, r_scale] if r_scale is set). Got centers: {centers}, r_scale: {r_scale}"

        self.sigma = sigma
        super().__init__(norm=norm, dtype=dtype)

    def aggregate(
        self, edge_features: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None
    ) -> torch.Tensor:
        """Aggregate edge features with per-node scaling and per-target-node normalization.

        Parameters
        ----------
        edge_features : torch.Tensor
            Raw edge distances, shape [num_edges] or [num_edges, 1]
        index : torch.Tensor
            Destination node index for each edge
        ptr : optional
            CSR pointer (not used)
        dim_size : int, optional
            Number of destination nodes

        Returns
        -------
        torch.Tensor
            RBF features, shape [num_edges, num_centers].
            Normalized per target node per RBF center .
        """
        # Ensure edge_features is 1D
        if edge_features.ndim == 2:
            edge_features = edge_features.squeeze(-1)

        # Compute scale factor per destination node
        if self.r_scale is None:
            # Per-node max edge length scaling
            max_dists = scatter(
                edge_features, index.long(), dim=0, dim_size=dim_size, reduce="max"
            )

            # Clamp to epsilon to avoid division by zero
            max_dists = torch.clamp(max_dists, min=self.epsilon)

            # Broadcast to each edge
            scales = max_dists[index]
            alpha = edge_features / scales  # Normalized distance [0, 1]
        else:
            # Global scaling
            scales = torch.full_like(edge_features, self.r_scale)
            alpha = edge_features / scales  # Scaled distance [0, max_edge/r_scale]

        # Compute Gaussian RBF for each center
        rbf_features = []
        for center in self.centers:
            rbf = torch.exp(-(((alpha - center) / self.sigma) ** 2))
            rbf_features.append(rbf)

        rbf_features = torch.stack(rbf_features, dim=1)

        # Within each RBF center, normalise edges to the same target node
        rbf_features = self.normalise(rbf_features, index, dim_size)

        return rbf_features


class GaussianDistanceWeights(EdgeLength):
    """Gaussian distance weights."""

    norm_by_group: bool = True  # normalise the gaussian weights by target node

    def __init__(self, sigma: float = 1.0, norm: str = "l1", **kwargs) -> None:
        self.sigma = sigma
        super().__init__(norm=norm)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        dists = super().compute(x_i, x_j)
        gaussian_weights = torch.exp(-(dists**2) / (2 * self.sigma**2))
        return gaussian_weights


class RelativeEdgeLength(EdgeLength):
    """Relative edge length using local mesh spacing.

    For each processor edge j -> i:

        r_ij = log(l_ij / sqrt(h_i * h_j))

    where:
        l_ij = great-circle edge length
        h_i  = mean length of the K nearest distinct incoming
               processor-edge neighbors of node i

    Uses raw haversine distances, independently of EdgeLength
    normalization.

    Intended for homogeneous hidden-to-hidden processor graphs.

    Output:
        Tensor of shape [num_edges, 1].
    """

    def __init__(
        self,
        num_local_neighbors: int = 5,
        chunk_size: int = 250_000,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm=None, dtype=dtype)

        if num_local_neighbors < 1:
            raise ValueError("num_local_neighbors must be >= 1")

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.num_local_neighbors = num_local_neighbors
        self.chunk_size = chunk_size

    @torch.no_grad()
    def forward(
        self,
        x: tuple[NodeStorage, NodeStorage],
        edge_index: Adj,
        size: Size = None,
    ) -> torch.Tensor:

        source_nodes, target_nodes = x

        if source_nodes is not target_nodes:
            raise ValueError(
                "RelativeEdgeLength currently supports "
                "homogeneous hidden-to-hidden graphs only."
            )

        # --------------------------------------------------
        # 1. Extract coordinates and processor connectivity.
        # --------------------------------------------------

        device = edge_index.device

        coords = source_nodes.x.to(
            device=device,
            dtype=torch.float64,
        )

        edge_index = edge_index.to(
            device=device,
            dtype=torch.long,
        )

        src, dst = edge_index

        num_nodes = coords.shape[0]
        num_edges = src.numel()

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "Expected coordinates [num_nodes, 2] "
                "containing latitude and longitude in radians."
            )

        if num_edges == 0:
            raise ValueError("Processor graph has no edges.")

        if (
            src.min() < 0
            or dst.min() < 0
            or src.max() >= num_nodes
            or dst.max() >= num_nodes
        ):
            raise ValueError("Invalid processor edge indices.")

        # --------------------------------------------------
        # 2. Calculate physical edge lengths.
        # --------------------------------------------------

        lengths = torch.empty(
            num_edges,
            device=device,
            dtype=torch.float64,
        )

        for begin in range(0, num_edges, self.chunk_size):

            end = min(begin + self.chunk_size, num_edges)

            s = src[begin:end]
            d = dst[begin:end]

            lengths[begin:end] = haversine_distance(
                coords[s],
                coords[d],
            )

        # Haversine distances are in radians.
        # Since we only use ratios, Earth radius cancels.

        valid_edges = (src != dst) & torch.isfinite(lengths) & (lengths > 0)

        if not valid_edges.any():
            raise ValueError("No valid non-self processor edges available.")

        # --------------------------------------------------
        # 3. Calculate local node spacing h_i.
        #
        # Select the K shortest DISTINCT incoming neighbors.
        # This avoids repeatedly selecting duplicate edges.
        # --------------------------------------------------

        eligible = valid_edges.clone()

        neighbor_sum = torch.zeros(
            num_nodes,
            dtype=torch.float64,
            device=device,
        )

        neighbor_count = torch.zeros(
            num_nodes,
            dtype=torch.float64,
            device=device,
        )

        edge_ids = torch.arange(
            num_edges,
            dtype=torch.long,
            device=device,
        )

        for _ in range(self.num_local_neighbors):

            if not eligible.any():
                break

            # Shortest eligible edge for every destination.
            minimum_length = torch.full(
                (num_nodes,),
                float("inf"),
                dtype=torch.float64,
                device=device,
            )

            masked_lengths = lengths.masked_fill(
                ~eligible,
                float("inf"),
            )

            minimum_length.scatter_reduce_(
                0,
                dst,
                masked_lengths,
                reduce="amin",
                include_self=True,
            )

            # Identify shortest edges.
            is_minimum = eligible & (lengths == minimum_length[dst])

            # Deterministically select one edge per node.
            candidate_ids = torch.where(
                is_minimum,
                edge_ids,
                num_edges,
            )

            best_id = torch.full(
                (num_nodes,),
                num_edges,
                dtype=torch.long,
                device=device,
            )

            best_id.scatter_reduce_(
                0,
                dst,
                candidate_ids,
                reduce="amin",
                include_self=True,
            )

            has_neighbor = best_id != num_edges

            if not has_neighbor.any():
                break

            selected = best_id[has_neighbor]

            selected_dst = dst[selected]
            selected_src = src[selected]

            neighbor_sum.index_add_(
                0,
                selected_dst,
                lengths[selected],
            )

            neighbor_count.index_add_(
                0,
                selected_dst,
                torch.ones_like(lengths[selected]),
            )

            # Exclude all edges from the selected source
            # for each destination in subsequent iterations.
            selected_sources = torch.full(
                (num_nodes,),
                -1,
                dtype=torch.long,
                device=device,
            )

            selected_sources[has_neighbor] = selected_src

            eligible &= src != selected_sources[dst]

        # Mean distance of up to K local neighbors.
        h = neighbor_sum / neighbor_count.clamp_min(1)

        # Ensure spacing exists for every used endpoint.
        used_nodes = torch.unique(
            torch.cat(
                [
                    src[valid_edges],
                    dst[valid_edges],
                ]
            )
        )

        if (neighbor_count[used_nodes] == 0).any():
            raise ValueError(
                "Some processor-edge endpoints have no "
                "incoming neighbors. Check graph connectivity."
            )

        # --------------------------------------------------
        # 4. Compute relative edge lengths.
        #
        # log(l_ij / sqrt(h_i * h_j))
        # = log(l_ij) - 0.5 * (log(h_i) + log(h_j))
        # --------------------------------------------------

        log_h = torch.zeros_like(h)

        has_spacing = neighbor_count > 0

        log_h[has_spacing] = h[has_spacing].log()

        relative_length = torch.zeros(
            num_edges,
            dtype=torch.float64,
            device=device,
        )

        s = src[valid_edges]
        d = dst[valid_edges]

        relative_length[valid_edges] = lengths[valid_edges].log() - 0.5 * (
            log_h[s] + log_h[d]
        )

        # Self-edges and zero-length edges receive zero.
        # They are excluded from the local-spacing estimate.

        if not torch.isfinite(relative_length).all():
            raise ValueError("Non-finite relative edge lengths produced.")

        LOGGER.info(
            "RelativeEdgeLength: %d nodes, %d edges, " "K=%d, range=[%.4f, %.4f]",
            num_nodes,
            num_edges,
            self.num_local_neighbors,
            relative_length.min().item(),
            relative_length.max().item(),
        )

        return relative_length.unsqueeze(-1).to(
            dtype=getattr(torch, self.dtype),
        )
