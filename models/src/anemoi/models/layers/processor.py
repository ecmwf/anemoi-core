# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.halo import HaloInfo
from anemoi.models.distributed.halo import build_halo_info
from anemoi.models.distributed.halo import cache_specs as halo_cache_specs
from anemoi.models.distributed.khop_edges import ANEMOI_DEBUG_SHARDING
from anemoi.models.distributed.khop_edges import build_graph_partition_from_shard_info
from anemoi.models.distributed.khop_edges import ensure_edges_are_dst_sorted
from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.block import FlowersProcessorBlock
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.block import PointWiseMLPProcessorBlock
from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.mlp import MLPImplementation
from anemoi.models.layers.semi_lagrangian import EARTH_ROTATION_RATE
from anemoi.models.layers.semi_lagrangian import latlon_grid_shape
from anemoi.models.layers.utils import compute_mlp_hidden_dim
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.layers.utils import maybe_checkpoint
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)


class NoOpProcessor(nn.Module):
    """No-op processor, used for ablations."""

    def __init__(self, **kwargs) -> None:
        if len(kwargs) > 0:
            LOGGER.warning(
                f"{self.__class__.__name__} does not use any of the following provided kwargs: {list(kwargs.keys())}"
            )
        super().__init__()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x


class BaseProcessor(nn.Module, ABC):
    """Base Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseProcessor.

        Parameters
        ----------
        num_layers : int
            Number of processor layers.
        num_channels : int
            Number of channels, i.e. feature dimension of the processor state.
        num_chunks: int
            Number of chunks of the processor. The num_chunks and num_layers, defines how many layers are grouped together for checkpointing, i.e. chunk_size = num_layers/ num_chunks.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        gradient_checkpointing : bool
            Whether to enable gradient checkpointing, by default True
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        **kwargs : dict
            Additional keyword arguments
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_chunks = num_chunks
        self.chunk_size = num_layers // num_chunks
        self.num_channels = num_channels
        self.gradient_checkpointing = gradient_checkpointing

        self.layer_factory = load_layer_kernels(layer_kernels)

        self._has_dropout = kwargs.get("dropout_p", 0.0) > 0 if "dropout_p" in kwargs else False

        assert (
            num_layers % num_chunks == 0
        ), f"Number of processor layers ({num_layers}) has to be divisible by the number of processor chunks ({num_chunks})."

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def build_layers(self, layer_class, *layer_args, **layer_kwargs) -> None:
        """Build Layers."""
        self.proc = nn.ModuleList(
            [
                layer_class(
                    *layer_args,
                    **layer_kwargs,
                )
                for _ in range(self.num_layers)
            ],
        )

    def run_layer_chunk(self, chunk_start: int, data: tuple, *args, **kwargs) -> tuple:
        for layer_id in range(chunk_start, chunk_start + self.chunk_size):
            data = self.proc[layer_id](*data, *args, **kwargs)

        return data

    def run_layers(self, data: tuple, *args, **kwargs) -> tuple:
        """Run Layers with optional checkpoints around chunks."""
        for chunk_start in range(0, self.num_layers, self.chunk_size):
            data = maybe_checkpoint(
                self.run_layer_chunk,
                self.gradient_checkpointing,
                chunk_start,
                data,
                *args,
                **kwargs,
            )

        return data

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Example forward pass."""

        if (model_comm_group := kwargs.get("model_comm_group", None)) is not None:
            assert (
                model_comm_group.size() == 1 or not self._has_dropout
            ), f"Dropout is not supported when model is sharded across {model_comm_group.size()} GPUs"

        x = self.run_layers((x,), *args, **kwargs)
        return x


class PointWiseMLPProcessor(BaseProcessor):
    """Point-wise MLP Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        mlp_hidden_ratio: float,
        cpu_offload: bool = False,
        dropout_p: float = 0.0,
        layer_kernels: DotDict,
        **kwargs,
    ):
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
            **kwargs,
        )

        self.build_layers(
            PointWiseMLPProcessorBlock,
            num_channels=num_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), f"Only batch size of 1 is supported when model is sharded accross {model_comm_group.size()} GPUs"

        (x,) = self.run_layers((x,), shard_info, batch_size, model_comm_group, **kwargs)

        return x


class TransformerProcessor(BaseProcessor):
    """Transformer Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: float,
        attn_channels: Optional[int] = None,
        qk_norm=False,
        dropout_p: float = 0.0,
        attention_implementation: str = "triton_attention",
        mlp_implementation: MLPImplementation = "mlp",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        window_size: Optional[int] = None,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: float
            Ratio of mlp hidden dimension to embedding dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v projections. If None,
            defaults to num_channels. This allows reducing the number of
            channels used for the attention computation without changing the
            width of the surrounding MLPs.
        qk_norm: bool, optional
            Normalize query and key, by default False
        dropout_p: float, optional
            Dropout probability used for multi-head self attention, default 0.1
        attention_implementation: str
            A predefined string which selects which underlying attention
            implementation, by default "triton_attention"
        mlp_implementation: MLPImplementation
            Implementation of feed-forward blocks in processor layers.
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        use_alibi_slopes : bool
            Use aLiBI option, only used for flash attention, by default False
        window_size: int, optional
            1/2 size of shifted window for attention computation, by default None
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            window_size=window_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
            **kwargs,
        )

        self.build_layers(
            TransformerProcessorBlock,
            num_channels=num_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            attn_channels=attn_channels,
            num_heads=num_heads,
            qk_norm=qk_norm,
            window_size=window_size,
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            mlp_implementation=mlp_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        (x,) = self.run_layers((x,), shard_info, batch_size, model_comm_group=model_comm_group, **kwargs)

        return x


class ADRProcessor(BaseProcessor):
    """Advection-diffusion-reaction processor on a regular latitude-longitude hidden grid (PARADIS).

    The hidden nodes must be the cell centres of a regular latitude-longitude grid, stored row by
    row from north to south, with longitudes running east from zero, as built by
    ``anemoi.graphs.nodes.RegularLatLonNodes``. The grid size is read from the node coordinates.

    The layers split one model time step into equal sub-steps.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        node_coordinates: Tensor,
        timestep: str,
        advection_channels: int = 768,
        num_heads: Optional[int] = None,
        velocity_hidden_dim: int = 384,
        reaction_hidden_dim: int = 896,
        reaction_num_layers: int = 4,
        kernel_size: int = 5,
        interpolation: str = "bicubic",
        bias_rank: int = 128,
        bias_base_maps: int = 8,
        cartesian_displacement: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize ADRProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        node_coordinates : Tensor
            Latitude and longitude of the hidden nodes in radians, shape (num_nodes, 2)
        timestep : str
            Model time step, e.g. "6h". Sets how far the learned velocities move the state.
        advection_channels : int
            Number of channels moved by the advection step
        num_heads : int, optional
            Number of velocity fields; each moves advection_channels // num_heads channels.
            By default every moved channel has its own velocity field.
        velocity_hidden_dim : int
            Hidden dimension of the velocity network
        reaction_hidden_dim : int
            Hidden dimension of the reaction MLP
        reaction_num_layers : int
            Number of linear layers in the reaction MLP, at least 2
        kernel_size : int
            Size of the square stencils of the spatial mixers, must be odd
        interpolation : str
            "bicubic" or "bilinear", by default "bicubic"
        bias_rank : int
            Number of latitude-longitude profile pairs in each learned bias field
        bias_base_maps : int
            Number of base maps each learned bias field is mixed from
        cartesian_displacement : bool
            Predict velocities as 3D vectors instead of local east and north components (the
            paper's choice, default). 3D vectors do not turn round across the poles.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        nlat, nlon = latlon_grid_shape(node_coordinates)
        self.build_layers(
            ADRProcessorBlock,
            num_channels=num_channels,
            advection_channels=advection_channels,
            num_heads=num_heads if num_heads is not None else advection_channels,
            velocity_hidden_dim=velocity_hidden_dim,
            reaction_hidden_dim=reaction_hidden_dim,
            reaction_num_layers=reaction_num_layers,
            nlat=nlat,
            nlon=nlon,
            time_step=frequency_to_seconds(timestep) * EARTH_ROTATION_RATE / num_layers,
            layer_kernels=self.layer_factory,
            kernel_size=kernel_size,
            interpolation=interpolation,
            bias_rank=bias_rank,
            bias_base_maps=bias_base_maps,
            cartesian_displacement=cartesian_displacement,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        (x,) = self.run_layers((x,), shard_info, batch_size, model_comm_group=model_comm_group, **kwargs)

        return x


class FlowersProcessor(BaseProcessor):
    """Processor of FLOWERS warp blocks on a regular latitude-longitude hidden grid.

    The blocks are stacked at a single resolution. The hidden nodes must be the cell centres of a
    regular latitude-longitude grid, stored row by row from north to south, with longitudes
    running east from zero, as built by ``anemoi.graphs.nodes.RegularLatLonNodes``. The grid
    size is read from the node coordinates.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        node_coordinates: Tensor,
        num_heads: Optional[int] = None,
        mlp_hidden_ratio: float = 4.0,
        mlp_implementation: MLPImplementation = "mlp",
        block_style: str = "pre_norm",
        interpolation: str = "bilinear",
        cartesian_displacement: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize FlowersProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        node_coordinates : Tensor
            Latitude and longitude of the hidden nodes in radians, shape (num_nodes, 2)
        num_heads : int, optional
            Number of displacement fields per block; each moves num_channels // num_heads
            channels. By default 4 channels per head, as in FLOWERS.
        mlp_hidden_ratio : float
            Ratio of the MLP hidden dimension to num_channels, used by the "pre_norm" block
        mlp_implementation : MLPImplementation
            Implementation of the MLP in the "pre_norm" block
        block_style : str
            "pre_norm" (residual warp and MLP, each after a layer norm) or "flowers" (the
            original FLOWERS block)
        interpolation : str
            "bilinear" or "bicubic", by default "bilinear" as in FLOWERS
        cartesian_displacement : bool
            Predict displacements as 3D vectors instead of local east and north angles (default).
            3D vectors do not turn round across the poles.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        nlat, nlon = latlon_grid_shape(node_coordinates)
        self.build_layers(
            FlowersProcessorBlock,
            num_channels=num_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            num_heads=num_heads if num_heads is not None else num_channels // 4,
            nlat=nlat,
            nlon=nlon,
            layer_kernels=self.layer_factory,
            block_style=block_style,
            interpolation=interpolation,
            mlp_implementation=mlp_implementation,
            cartesian_displacement=cartesian_displacement,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        (x,) = self.run_layers((x,), shard_info, batch_size, model_comm_group=model_comm_group, **kwargs)

        return x


class GNNProcessor(BaseProcessor):
    """GNN Processor."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int,
        num_chunks: int,
        mlp_extra_layers: int,
        edge_dim: int,
        mlp_hidden_ratio: float = 1.0,
        mlp_implementation: MLPImplementation = "mlp",
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        mlp_extra_layers : int
            Number of extra layers in MLP
        edge_dim : int
            Edge feature dimension
        mlp_hidden_ratio : float
            Ratio of MLP hidden dimension to num_channels. Default 1.0 preserves existing behaviour.
        mlp_implementation: MLPImplementation
            Implementation of feed-forward blocks in processor layers.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml

        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        kwargs_build = {
            "mlp_extra_layers": mlp_extra_layers,
            "mlp_hidden_ratio": mlp_hidden_ratio,
            "mlp_implementation": mlp_implementation,
            "layer_kernels": self.layer_factory,
            "edge_dim": None,
        }

        self.build_layers(
            GraphConvProcessorBlock,
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs_build,
        )

        kwargs_build["edge_dim"] = edge_dim  # Edge dim for first layer
        self.proc[0] = GraphConvProcessorBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs_build,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        edges_are_dst_sorted: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """Run the GNN processor.

        Parameters
        ----------
        x : Tensor
            Node features.
        batch_size : int
            Batch size.
        shard_info : GraphShardInfo
            Shard metadata for node and edge tensors.
        edge_attr : Tensor
            Edge attributes.
        edge_index : Adj
            Edge indices.
        model_comm_group : ProcessGroup, optional
            Model communication group.
        edges_are_dst_sorted : bool, optional
            Whether `edge_index` and `edge_attr` are already ordered by destination node.
            Edges from graph providers already are. Pass False for custom full-graph
            edges that are not ordered this way. If edges are already sharded, each rank
            is expected to already have the right edges for its local destination nodes.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments passed to processor blocks.

        Returns
        -------
        Tensor
            Processed node features.
        """
        if not shard_info.edges_are_sharded():
            # Edges not pre-sharded, do 1-hop sorting and sharding here
            target_nodes = sum(shard_info.nodes)
            edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(
                edge_attr,
                edge_index,
                target_nodes,
                target_nodes,
                model_comm_group,
                edges_are_dst_sorted=edges_are_dst_sorted,
            )
            shard_info = GraphShardInfo(nodes=shard_info.nodes, edges=edge_shard_sizes)

        x, edge_attr = self.run_layers((x, edge_attr), edge_index, shard_info, model_comm_group, **kwargs)

        return x


class GraphTransformerProcessor(BaseProcessor):
    """Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: float,
        edge_dim: int,
        attn_channels: Optional[int] = None,
        qk_norm: bool = False,
        mlp_implementation: MLPImplementation = "mlp",
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        shard_strategy: str = "edges",
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: float
            Ratio of mlp hidden dimension to embedding dimension
        edge_dim : int
            Edge feature dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v and edge projections. If
            None, defaults to num_channels. This allows reducing the number
            of channels used for the attention computation without changing
            the width of the surrounding MLPs.
        qk_norm: bool, optional
            Normalize query and key, by default False
        mlp_implementation: MLPImplementation
            Implementation of feed-forward blocks in processor layers.
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        shard_strategy: str, by default "edges"
            Strategy to shard tensors, options are "edges" and "heads"
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        assert shard_strategy in ["edges", "heads"], (
            f"Invalid shard strategy '{shard_strategy}' for {self.__class__.__name__}. "
            f"Supported strategies are 'edges' and 'heads'."
        )
        self.shard_strategy = shard_strategy
        self._cached_halo_info = None
        self._cached_halo_cache_specs = None

        self.build_layers(
            GraphTransformerProcessorBlock,
            in_channels=num_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            out_channels=num_channels,
            attn_channels=attn_channels,
            num_heads=num_heads,
            layer_kernels=self.layer_factory,
            qk_norm=qk_norm,
            mlp_implementation=mlp_implementation,
            shard_strategy=shard_strategy,
            graph_attention_backend=graph_attention_backend,
            edge_dim=edge_dim,
            edge_pre_mlp=edge_pre_mlp,
        )

        self.offload_layers(cpu_offload)

    def _get_or_build_cached_halo_info(
        self,
        x: Tensor,
        edge_index: Adj,
        shard_info: GraphShardInfo,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup],
    ) -> Optional[HaloInfo]:
        """Return one halo plan shared by all processor layers.

        The plan is kept for as long as the shard sizes stay the same, so the
        processor graph must not change between calls.
        """
        if self.shard_strategy != "edges" or not model_is_distributed(model_comm_group):
            return None

        if batch_size != 1:
            raise ValueError(
                "GraphTransformerProcessor halo exchange requires batch_size=1 when model sharding is enabled."
            )
        if not shard_info.nodes_are_sharded():
            raise ValueError(
                "GraphTransformerProcessor halo exchange requires sharded nodes when model sharding is enabled."
            )
        assert shard_info.edges_are_sharded(), "Halo strategy requires edges to be sharded"

        cache_specs = halo_cache_specs(shard_info, model_comm_group)
        if self._cached_halo_info is not None and self._cached_halo_cache_specs == cache_specs:
            return self._cached_halo_info

        LOGGER.info(f"Building halo info for {self.__class__.__name__} with shard strategy 'edges'")

        bipartite_shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_info.nodes,
            dst_nodes=shard_info.nodes,
            edges=shard_info.edges,
        )
        partition = build_graph_partition_from_shard_info(
            edge_index,
            (x, x),
            bipartite_shard_info,
            model_comm_group,
        )
        halo_info = build_halo_info(
            partition,
            edge_index,
            model_comm_group,
            shard_info.edges,
            debug=ANEMOI_DEBUG_SHARDING,
        )

        self._cached_halo_info = halo_info
        self._cached_halo_cache_specs = cache_specs
        return halo_info

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Tensor,
        edge_index: Adj,
        model_comm_group: Optional[ProcessGroup] = None,
        edges_are_dst_sorted: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """Run the graph-transformer processor.

        Parameters
        ----------
        x : Tensor
            Node features.
        batch_size : int
            Batch size.
        shard_info : GraphShardInfo
            Shard metadata for node and edge tensors.
        edge_attr : Tensor
            Edge attributes.
        edge_index : Adj
            Edge indices.
        model_comm_group : ProcessGroup, optional
            Model communication group.
        edges_are_dst_sorted : bool, optional
            Whether `edge_index` and `edge_attr` are already ordered by destination node.
            Edges from graph providers already are. Pass False for custom full-graph
            edges that are not ordered this way. If edges are already sharded, each rank
            is expected to already have the right edges for its local destination nodes.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments passed to processor blocks.

        Returns
        -------
        Tensor
            Processed node features.
        """
        size = sum(shard_info.nodes) if shard_info.nodes_are_sharded() else x.size(0)
        edge_attr, edge_index = ensure_edges_are_dst_sorted(
            edge_attr,
            edge_index,
            num_dst=size,
            edges_are_sharded=shard_info.edges_are_sharded(),
            model_comm_group=model_comm_group,
            edges_are_dst_sorted=edges_are_dst_sorted,
        )

        if not shard_info.edges_are_sharded():  # ensure edges are sharded
            edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(
                edge_attr, edge_index, size, size, model_comm_group, edges_are_dst_sorted=edges_are_dst_sorted
            )
            shard_info = GraphShardInfo(nodes=shard_info.nodes, edges=edge_shard_sizes)

        # Heads sharding needs full edge_index (nodes are full, only heads are sharded)
        halo_info = None
        if self.shard_strategy == "heads":
            edge_index = gather_tensor(edge_index, 1, shard_info.edges, model_comm_group)
        else:  # shard strategy "edges" w/ halo-exchange
            halo_info = self._get_or_build_cached_halo_info(
                x,
                edge_index,
                shard_info,
                batch_size,
                model_comm_group,
            )

        x, edge_attr = self.run_layers(
            data=(x, edge_attr),
            edge_index=edge_index,
            shard_info=shard_info,
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            halo_info=halo_info,
            edges_are_dst_sorted=True,  # ensured by ensure_edges_are_dst_sorted above
            **kwargs,
        )

        return x
