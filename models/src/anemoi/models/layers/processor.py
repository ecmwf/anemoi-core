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

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data.storage import NodeStorage
from torch_geometric.typing import Adj

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.halo import HaloInfoCache
from anemoi.models.distributed.khop_edges import ensure_edges_are_dst_sorted
from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import ADRProcessorBlock
from anemoi.models.layers.block import FlowersProcessorBlock
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.block import PointWiseMLPProcessorBlock
from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.mlp import MLPImplementation
from anemoi.models.layers.utils import compute_mlp_hidden_dim
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.layers.utils import maybe_checkpoint
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

# Rotation rate of the Earth in radians per second.
EARTH_ROTATION_RATE = 7.29212e-5


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


class SemiLagrangianProcessor(BaseProcessor):
    """Shared parts of the processors that run on a ``LatentGeometry``.

    The geometry is built once from the graph and shared by every block, so the node coordinates
    are held once rather than once per layer. Each forward binds it to the batch, and on a mesh to
    the hidden-to-hidden edges, which gives the layout the blocks work in; the state is put into
    that layout on the way in and taken back out on the way out.
    """

    def __init__(
        self,
        *,
        geometry: DictConfig,
        hidden_nodes: NodeStorage,
        edge_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.geometry = instantiate(geometry, hidden_nodes=hidden_nodes, edge_dim=edge_dim)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        edges_are_dst_sorted: bool = True,
        cond: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        """Run the layers on the geometry.

        x has shape (batch * local nodes, channels), as the model hands it over, and comes back
        the same way. The edges are only used by a mesh geometry; ``edges_are_dst_sorted`` says
        whether they already are ordered by destination node.
        """
        layout = self.geometry.bind(
            x,
            batch_size,
            shard_info,
            model_comm_group,
            edge_attr=edge_attr,
            edge_index=edge_index,
            edges_are_dst_sorted=edges_are_dst_sorted,
        )
        if cond is not None:
            kwargs["cond"] = layout.to_state(cond)
        (x,) = self.run_layers(
            (layout.to_state(x),), shard_info, batch_size, model_comm_group=model_comm_group, layout=layout, **kwargs
        )
        return layout.from_state(x)


class ADRProcessor(SemiLagrangianProcessor):
    """Advection-diffusion-reaction processor (PARADIS), on any hidden nodes.

    Each layer runs an advection, a diffusion and a reaction step, each built from its own config
    (see ``anemoi.models.layers.adr``). A step set to None is left out, and the layers split one
    model time step into equal sub-steps.

    The ``geometry`` config decides what the hidden nodes are and what the spatial parts of the
    steps look like: ``RegularLatLonGeometry`` for a regular latitude-longitude grid, as in the
    paper, or ``MeshGeometry`` for any other mesh, which mixes neighbours over the
    hidden-to-hidden edges of the graph. The steps themselves are the same either way.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        geometry: DictConfig,
        hidden_nodes: NodeStorage,
        timestep: str,
        advection: Optional[DictConfig],
        diffusion: Optional[DictConfig],
        reaction: Optional[DictConfig],
        edge_dim: Optional[int] = None,
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
        geometry : DictConfig
            Config of the hidden nodes, e.g.
            ``anemoi.models.layers.geometry.RegularLatLonGeometry``. It is given the hidden nodes
            of the graph and the dimension of the hidden-to-hidden edges.
        hidden_nodes : NodeStorage
            The hidden nodes of the graph, with their coordinates and any layout attribute
        timestep : str
            Model time step, e.g. "6h". Sets how far the learned velocities move the state.
        advection : DictConfig, optional
            Config of the advection step, e.g. ``anemoi.models.layers.adr.Advection``.
            None leaves the step out.
        diffusion : DictConfig, optional
            Config of the diffusion step, e.g. ``anemoi.models.layers.adr.Diffusion``.
            None leaves the step out.
        reaction : DictConfig, optional
            Config of the reaction step, e.g. ``anemoi.models.layers.adr.Reaction``.
            None leaves the step out.
        edge_dim : int, optional
            Number of features of the hidden-to-hidden edges, used by a mesh geometry
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            geometry=geometry,
            hidden_nodes=hidden_nodes,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        time_step = frequency_to_seconds(timestep) * EARTH_ROTATION_RATE / num_layers
        self.build_layers(
            ADRProcessorBlock,
            num_channels=num_channels,
            geometry=self.geometry,
            time_step=time_step,
            layer_kernels=self.layer_factory,
            advection=advection,
            diffusion=diffusion,
            reaction=reaction,
        )
        self.offload_layers(cpu_offload)


class FlowersProcessor(SemiLagrangianProcessor):
    """Processor of FLOWERS warp blocks, stacked at a single resolution, on any hidden nodes.

    As in ``ADRProcessor``, the ``geometry`` config decides what the hidden nodes are and how a
    value is read at a departure point.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        geometry: DictConfig,
        hidden_nodes: NodeStorage,
        num_heads: Optional[int] = None,
        mlp_hidden_ratio: float = 4.0,
        mlp_implementation: MLPImplementation = "mlp",
        block_style: str = "pre_norm",
        cartesian_displacement: bool = False,
        edge_dim: Optional[int] = None,
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
        geometry : DictConfig
            Config of the hidden nodes, e.g.
            ``anemoi.models.layers.geometry.RegularLatLonGeometry``
        hidden_nodes : NodeStorage
            The hidden nodes of the graph, with their coordinates and any layout attribute
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
        cartesian_displacement : bool
            Predict displacements as 3D vectors instead of local east and north angles (default).
            3D vectors do not turn round across the poles.
        edge_dim : int, optional
            Number of features of the hidden-to-hidden edges, used by a mesh geometry
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            geometry=geometry,
            hidden_nodes=hidden_nodes,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            **kwargs,
        )

        self.build_layers(
            FlowersProcessorBlock,
            num_channels=num_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            num_heads=num_heads if num_heads is not None else num_channels // 4,
            geometry=self.geometry,
            layer_kernels=self.layer_factory,
            block_style=block_style,
            mlp_implementation=mlp_implementation,
            cartesian_displacement=cartesian_displacement,
        )

        self.offload_layers(cpu_offload)


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


class GraphTransformerProcessor(BaseProcessor, HaloInfoCache):
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
