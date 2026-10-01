# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import BaseGraphProvider
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import NoOpProcessor
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ForwardContext:
    """Per-call state shared by the encode, process and decode stages of :class:`AnemoiModelEncProcDec`.

    A context is created once per forward pass by ``_init_forward_context`` and threaded through
    ``_encode``, ``_process`` and ``_decode``. Subclasses extend it with the model-specific inputs
    their stages consume (e.g. the forecast step of the ensemble model or the corrupted target of
    the transport models).

    Attributes
    ----------
    batch_size : int
        Number of samples in the batch (first input dimension).
    ensemble_size : int
        Number of ensemble members per sample (third input dimension).
    model_comm_group : ProcessGroup | None
        Model communication group for sharded execution.
    grid_shard_sizes : DatasetShardSizes | None
        Per-dataset shard sizes of the input grid. ``None`` means the inputs are replicated.
    in_out_sharded : dict[str, bool]
        Whether the input, and hence the output, of each dataset is sharded across the model comm group.
    x_hidden : dict[str, Tensor]
        Trainable node attributes of every hidden level, replicated over batch and ensemble and sharded.
    shard_sizes_hidden : dict[str, ShardSizes]
        Shard sizes of ``x_hidden`` per hidden level.
    x_data_latent : dict[str, Tensor]
        Encoder-updated data-node features per input dataset. Filled by ``_encode``.
    x_skip : dict[str, Tensor | None]
        Residual skip tensor per input dataset. Filled by ``_encode``.
    shard_sizes_data : dict[str, ShardSizes]
        Shard sizes of the data nodes per input dataset. Filled by ``_encode``.
    encoder_kwargs : dict[str, dict]
        Extra keyword arguments passed to the encoder of each dataset (e.g. ``cond``).
    processor_kwargs : dict
        Extra keyword arguments passed to the processor.
    decoder_kwargs : dict[str, dict]
        Extra keyword arguments passed to the decoder of each dataset.
    """

    batch_size: int
    ensemble_size: int
    model_comm_group: ProcessGroup | None
    grid_shard_sizes: DatasetShardSizes | None
    in_out_sharded: dict[str, bool]
    x_hidden: dict[str, Tensor]
    shard_sizes_hidden: dict[str, ShardSizes]
    x_data_latent: dict[str, Tensor] = field(default_factory=dict)
    x_skip: dict[str, Tensor | None] = field(default_factory=dict)
    shard_sizes_data: dict[str, ShardSizes] = field(default_factory=dict)
    encoder_kwargs: dict[str, dict] = field(default_factory=dict)
    processor_kwargs: dict = field(default_factory=dict)
    decoder_kwargs: dict[str, dict] = field(default_factory=dict)

    @property
    def batch_ens_size(self) -> int:
        """Number of graph replicas seen by mappers and processors.

        Batch and ensemble dimensions are flattened together with the grid dimension, so every
        graph operation is replicated ``batch_size * ensemble_size`` times.
        """
        return self.batch_size * self.ensemble_size

    def dataset_shard_sizes(self, dataset_name: str) -> ShardSizes:
        """Grid shard sizes of one dataset, ``None`` if it is replicated."""
        if self.grid_shard_sizes is None:
            return None
        return self.grid_shard_sizes[dataset_name]


class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network.

    The network is built and run in three stages that subclasses override independently:

    * **encode** (``_build_encoders`` / ``_encode``): every input dataset is assembled with
      ``_assemble_input`` and mapped onto the first hidden level, ``_hidden_names[0]``; the
      per-dataset latents are combined by the latent aggregator.
    * **process** (``_build_processor`` / ``_process``): the latent is transformed on the deepest
      hidden level, ``_hidden_names[-1]``. The hierarchical model wraps this stage in a U-Net over
      the hidden levels; the ensemble model injects noise before it.
    * **decode** (``_build_decoders`` / ``_decode``): the processed latent is mapped back onto every
      target dataset and finalised with ``_assemble_output``.

    Per-call state (sizes, sharding, hidden latents, encoder by-products and extra mapper/processor
    keyword arguments) is carried in a :class:`ForwardContext` built by ``_init_forward_context``.
    """

    # ----------------------------------------------------------------------------------------------
    # Network construction
    # ----------------------------------------------------------------------------------------------

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        self._build_encoders(model_config)
        self._build_latent_aggregator(model_config.latent_aggregator)
        self.hidden_dims = self._calculate_hidden_dims()
        self._build_processor(model_config)
        self._build_decoders(model_config)

    def _calculate_hidden_dims(self) -> dict[str, int]:
        """Number of latent channels at each hidden level."""
        assert len(self._hidden_names) == 1, (
            f"{type(self).__name__} supports a single hidden level, got {self._hidden_names}. "
            "Use AnemoiModelEncProcDecHierarchical for several hidden levels."
        )
        return {self._hidden_names[0]: self.latent_aggregator.hidden_dim}

    def _build_graph_provider(
        self, src_nodes_name: str, dst_nodes_name: str, mapper_config: DotDict
    ) -> BaseGraphProvider:
        """Create the graph provider for the edges ``src_nodes_name -> dst_nodes_name`` of one mapper or processor."""
        return create_graph_provider(
            graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
            edge_attributes=mapper_config.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[src_nodes_name],
            dst_size=self.node_attributes.num_nodes[dst_nodes_name],
            trainable_size=mapper_config.get("trainable_size", 0),
        )

    @staticmethod
    def _shared_dim(dims: dict[str, int], dataset_names: list[str], mapper: str, dim_name: str) -> int:
        """Return the dimension shared by all ``dataset_names``, asserting they agree."""
        values = [dims[d] for d in dataset_names]
        assert all(
            value == values[0] for value in values
        ), f"All datasets for {mapper} must have the same {dim_name} dimension, but got {values}."
        return values[0]

    def _build_encoders(self, model_config: DotDict) -> None:
        """Build the data -> hidden graph providers and encoders."""
        hidden_name = self._hidden_names[0]

        self.encoder_graph_provider = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.input_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the input as it doesn't have a corresponding encoder."
                )
                continue

            encoder_config = model_config.encoders[self.dataset2encoder[dataset_name]]
            self.encoder_graph_provider[dataset_name] = self._build_graph_provider(
                dataset_name, hidden_name, encoder_config.mapper
            )

        self.encoder = nn.ModuleDict()
        for encoder_name, encoder_config in model_config.encoders.items():
            self.encoder[encoder_name] = instantiate(
                encoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self._shared_dim(
                    self.input_dim, self.encoder2datasets[encoder_name], f"encoder {encoder_name}", "input"
                ),
                in_channels_dst=self.input_dim_latent,
                edge_dim=self.encoder_graph_provider[encoder_config.source_datasets[0]].edge_dim,
            )

    def _build_processor(self, model_config: DotDict, **processor_kwargs) -> None:
        """Build the graph provider and processor acting on the deepest hidden level.

        ``processor_kwargs`` override entries of ``model_config.processor`` (e.g. ``num_channels``).
        """
        hidden_name = self._hidden_names[-1]

        self.processor_graph_provider = self._build_graph_provider(hidden_name, hidden_name, model_config.processor)
        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            edge_dim=self.processor_graph_provider.edge_dim,
            **processor_kwargs,
        )

        assert (
            isinstance(self.processor, NoOpProcessor) or self.processor.num_channels == self.hidden_dims[hidden_name]
        ), (
            f"Processor number of channels ({self.processor.num_channels}) must match the latent channels"
            f" on '{hidden_name}' ({self.hidden_dims[hidden_name]})."
        )

    def _build_decoders(self, model_config: DotDict) -> None:
        """Build the hidden -> data graph providers and decoders."""
        hidden_name = self._hidden_names[0]

        self.decoder_graph_provider = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.target_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the output as it doesn't have a corresponding decoder."
                )
                continue

            decoder_config = model_config.decoders[self.dataset2decoder[dataset_name]]
            self.decoder_graph_provider[dataset_name] = self._build_graph_provider(
                hidden_name, dataset_name, decoder_config.mapper
            )

        self.decoder = nn.ModuleDict()
        for decoder_name, decoder_config in model_config.decoders.items():
            datasets = self.decoder2datasets[decoder_name]
            self.decoder[decoder_name] = instantiate(
                decoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[hidden_name],
                in_channels_dst=self._shared_dim(self.target_dim, datasets, f"decoder {decoder_name}", "target"),
                out_channels_dst=self._shared_dim(self.output_dim, datasets, f"decoder {decoder_name}", "output"),
                edge_dim=self.decoder_graph_provider[decoder_config.target_datasets[0]].edge_dim,
            )

    # ----------------------------------------------------------------------------------------------
    # Per-dataset tensor assembly
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def _flatten_nodes(x: Tensor) -> Tensor:
        """Flatten ``(batch, time, ensemble, grid, vars)`` to ``((batch ensemble grid), (time vars))``."""
        return einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")

    def _data_node_attributes(self, dataset_name: str, ctx: ForwardContext) -> Tensor:
        """Node attributes of one dataset, replicated over batch and ensemble and sharded like its grid."""
        node_attributes = self.node_attributes(dataset_name, batch_size=ctx.batch_ens_size)
        grid_shard_sizes = ctx.dataset_shard_sizes(dataset_name)
        if grid_shard_sizes is not None:
            node_attributes = shard_tensor(node_attributes, 0, grid_shard_sizes, ctx.model_comm_group)
        return node_attributes

    def _assemble_input(
        self,
        x: Tensor,
        ctx: ForwardContext,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor | None, ShardSizes]:
        """Assemble the encoder source features and the residual skip for a single dataset.

        Flattens the raw input over ``(batch, ensemble, grid)`` and ``(time, vars)``, concatenates
        the per-node attributes on the feature dimension, and computes the residual skip tensor.

        Returns
        -------
        tuple[Tensor, Tensor | None, ShardSizes]
            ``(x_data_latent, x_skip, grid_shard_sizes)`` where ``x_data_latent`` is the encoder
            source input, and ``x_skip`` is the residual to add to the decoder output.
        """
        grid_shard_sizes = ctx.dataset_shard_sizes(dataset_name)

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=ctx.model_comm_group,
            n_step_output=self.n_step_output,
        )

        # add data positional info (lat/lon) on the feature dimension
        x_data_latent = torch.cat((self._flatten_nodes(x), self._data_node_attributes(dataset_name, ctx)), dim=-1)

        return x_data_latent, x_skip, grid_shard_sizes

    def _assemble_targets(
        self,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        ctx: ForwardContext,
        dataset_name: str,
    ) -> tuple[Tensor, ShardSizes]:
        """Assemble the decoder destination features for a single dataset.

        Concatenates the feature blocks listed in ``decoders_target_input`` for this dataset's
        decoder into the per-node vector fed to the decoder as ``x_dst``.

        Returns
        -------
        tuple[Tensor, ShardSizes]
            ``(x_target_latent, grid_shard_sizes)`` where ``x_target_latent`` has width
            ``target_dim[dataset_name]``.
        """
        grid_shard_sizes = ctx.dataset_shard_sizes(dataset_name)

        x_target_latent = self.decoders_target_input[self.dataset2decoder[dataset_name]].tensor(
            x_input_data,
            x_encoded_data,
            batch_size=ctx.batch_ens_size,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=ctx.model_comm_group,
            dataset_name=dataset_name,
        )

        return x_target_latent, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: Tensor,
        x_skip: Tensor | None,
        ctx: ForwardContext,
        dtype: torch.dtype,
        dataset_name: str,
    ) -> Tensor:
        """Reshape decoder output, add the prognostic residual and apply output boundings.

        Rearranges the flat decoder output back to ``(batch, time, ensemble, grid, vars)``, adds
        ``x_skip`` on the prognostic channels, and applies the per-dataset ``boundings`` in
        config order.
        """
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                batch=ctx.batch_size,
                ensemble=ctx.ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        if x_skip is not None:
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            assert (
                x_skip.shape[1] == x_out.shape[1]
            ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
            x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    # ----------------------------------------------------------------------------------------------
    # Graph operations
    # ----------------------------------------------------------------------------------------------

    def _run_mapper(
        self,
        mapper: nn.Module,
        graph_provider: BaseGraphProvider,
        x: tuple[Tensor, Tensor],
        *,
        src_shard_sizes: ShardSizes,
        dst_shard_sizes: ShardSizes,
        ctx: ForwardContext,
        keep_x_dst_sharded: bool,
        **mapper_kwargs,
    ):
        """Fetch the edges of ``graph_provider`` and apply ``mapper`` to ``x = (x_src, x_dst)``."""
        edge_attr, edge_index, edge_shard_sizes = graph_provider.get_edges(
            batch_size=ctx.batch_ens_size,
            model_comm_group=ctx.model_comm_group,
        )
        shard_info = BipartiteGraphShardInfo(
            src_nodes=src_shard_sizes,
            dst_nodes=dst_shard_sizes,
            edges=edge_shard_sizes,
        )
        return mapper(
            x,
            batch_size=ctx.batch_ens_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=ctx.model_comm_group,
            keep_x_dst_sharded=keep_x_dst_sharded,
            **mapper_kwargs,
        )

    def _run_processor(
        self,
        processor: nn.Module,
        graph_provider: BaseGraphProvider,
        x: Tensor,
        node_shard_sizes: ShardSizes,
        ctx: ForwardContext,
        **processor_kwargs,
    ) -> Tensor:
        """Fetch the edges of ``graph_provider`` and apply ``processor`` to the node features ``x``."""
        edge_attr, edge_index, edge_shard_sizes = graph_provider.get_edges(
            batch_size=ctx.batch_ens_size,
            model_comm_group=ctx.model_comm_group,
        )
        return processor(
            x,
            batch_size=ctx.batch_ens_size,
            shard_info=GraphShardInfo(nodes=node_shard_sizes, edges=edge_shard_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=ctx.model_comm_group,
            **processor_kwargs,
        )

    # ----------------------------------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------------------------------

    def _init_forward_context(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
        **kwargs,
    ) -> ForwardContext:
        """Validate the inputs and build the per-call context, including the sharded hidden latents.

        ``kwargs`` are the model-specific forward arguments; they are ignored here and consumed by
        subclasses that extend the context.
        """
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Trainable parameters of every hidden level: the initial latent state, pre-sharded
        batch_ens_size = batch_size * ensemble_size
        x_hidden: dict[str, Tensor] = {}
        shard_sizes_hidden: dict[str, ShardSizes] = {}
        for hidden_name in self._hidden_names:
            x_hidden_full = self.node_attributes(hidden_name, batch_size=batch_ens_size)
            shard_sizes_hidden[hidden_name] = get_shard_sizes(x_hidden_full, 0, model_comm_group)
            x_hidden[hidden_name] = shard_tensor(x_hidden_full, 0, shard_sizes_hidden[hidden_name], model_comm_group)

        return ForwardContext(
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            in_out_sharded=in_out_sharded,
            x_hidden=x_hidden,
            shard_sizes_hidden=shard_sizes_hidden,
        )

    def _encode(self, x: dict[str, Tensor], ctx: ForwardContext) -> Tensor:
        """Encode every input dataset onto the first hidden level and aggregate the latents.

        Stores the encoder by-products (updated data features, residual skips and data shard sizes)
        on ``ctx`` for the decode stage.
        """
        hidden_name = self._hidden_names[0]

        dataset_latents = {}
        for dataset_name in x.keys():
            if dataset_name not in self.input_datasets:
                continue

            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(x[dataset_name], ctx, dataset_name)
            ctx.x_skip[dataset_name] = x_skip
            ctx.shard_sizes_data[dataset_name] = shard_sizes_data

            x_data_latent, x_latent = self._run_mapper(
                self.encoder[self.dataset2encoder[dataset_name]],
                self.encoder_graph_provider[dataset_name],
                (x_data_latent, ctx.x_hidden[hidden_name]),
                src_shard_sizes=shard_sizes_data,  # None if not sharded
                dst_shard_sizes=ctx.shard_sizes_hidden[hidden_name],
                ctx=ctx,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                **ctx.encoder_kwargs.get(dataset_name, {}),
            )
            ctx.x_data_latent[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        return self.latent_aggregator(ctx.x_hidden[hidden_name], dataset_latents)

    def _prepare_processor_input(self, x_latent: Tensor, ctx: ForwardContext) -> tuple[Tensor, dict]:
        """Return the tensor fed to the processor and the processor keyword arguments.

        Hook for models that transform the latent right before the processor (e.g. noise injection
        in the ensemble model). The latent skip connection always uses the untransformed latent.
        """
        return x_latent, ctx.processor_kwargs

    def _process(self, x_latent: Tensor, ctx: ForwardContext) -> Tensor:
        """Run the processor on the deepest hidden level, with the latent skip connection."""
        hidden_name = self._hidden_names[-1]

        x_latent_in, processor_kwargs = self._prepare_processor_input(x_latent, ctx)
        x_latent_proc = self._run_processor(
            self.processor,
            self.processor_graph_provider,
            x_latent_in,
            ctx.shard_sizes_hidden[hidden_name],
            ctx,
            **processor_kwargs,
        )

        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        return x_latent_proc

    def _decode(self, x: dict[str, Tensor], x_latent: Tensor, ctx: ForwardContext) -> dict[str, Tensor]:
        """Decode the processed latent of the first hidden level onto every target dataset."""
        hidden_name = self._hidden_names[0]

        x_out_dict = {}
        for dataset_name in self.target_datasets:
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                ctx.x_data_latent.get(dataset_name, None),
                ctx,
                dataset_name,
            )

            x_out = self._run_mapper(
                self.decoder[self.dataset2decoder[dataset_name]],
                self.decoder_graph_provider[dataset_name],
                (x_latent, x_target_latent),
                src_shard_sizes=ctx.shard_sizes_hidden[hidden_name],
                dst_shard_sizes=shard_sizes_target,  # None if not sharded
                ctx=ctx,
                keep_x_dst_sharded=ctx.in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
                **ctx.decoder_kwargs.get(dataset_name, {}),
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                ctx.x_skip.get(dataset_name, None),
                ctx,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )

        return x_out_dict

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        **kwargs
            Model-specific inputs, forwarded to ``_init_forward_context``.

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        ctx = self._init_forward_context(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            **kwargs,
        )
        x_latent = self._encode(x, ctx)
        x_latent = self._process(x_latent, ctx)
        return self._decode(x, x_latent, ctx)

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
