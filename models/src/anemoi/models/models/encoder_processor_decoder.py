# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import random
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
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class GatedLatentFusion(nn.Module):
    """Gated fusion block to incorporate an encoder latent into the running latent.

    NOT attention — there is no Q/K/V or softmax over positions. Both tensors
    are already node-aligned (same N nodes, same D dims), so we use concatenation
    + MLP instead. This is simpler, cheaper, and sufficient for pointwise fusion.

    Uses a learned sigmoid gate (Flamingo-style) so the block can learn to be a
    no-op — critical for optional encoders that may be absent at inference time.

    Applied independently per node with full weight sharing across all nodes.
    Parameter count depends only on hidden_dim (D), not on number of nodes (N).

    Both inputs and output have shape [N, D].
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Separate norms because latent (running accumulation) and encoder output
        # (fresh from encoder) have different scale/distribution.
        self.norm_latent = nn.LayerNorm(hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        # Gate: single linear → sigmoid. Only needs to learn "how much" to incorporate.
        self.to_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        # Value: 2-layer MLP with SiLU. Needs more capacity to learn "what" to add.
        self.to_value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, latent: Tensor, encoder_output: Tensor) -> Tensor:
        """Fold encoder_output into the running latent.

        Parameters
        ----------
        latent : Tensor
            Running latent of shape [N, D].
        encoder_output : Tensor
            Encoder output to incorporate, shape [N, D].

        Returns
        -------
        Tensor
            Updated latent of shape [N, D].
        """
        ln = self.norm_latent(latent)
        en = self.norm_input(encoder_output)
        combined = torch.cat([ln, en], dim=-1)  # [N, 2D]
        gate = self.to_gate(combined)  # [N, D] in (0, 1)
        value = self.to_value(combined)  # [N, D]
        # Gated residual: if gate → 0, block is a no-op (safe for missing encoders)
        return latent + gate * value


class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network."""

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )

            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.input_dim_latent,
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # Latent fusion strategy: "gated" (default) or "sum".
        # Config key: model.latent_fusion
        self.latent_fusion_method = str(model_config.model.get("latent_fusion", "sum")).lower()
        assert self.latent_fusion_method in {"gated", "sum"}, (
            "model.latent_fusion must be one of {'gated', 'sum'}, got "
            f"'{self.latent_fusion_method}'"
        )
        LOGGER.info(f"Using latent fusion method: {self.latent_fusion_method.upper()}")
        if self.latent_fusion_method == "gated":
            # Gated fusion blocks to combine encoder latents sequentially (Perceiver-style flow).
            # The first dataset's latent is the initial latent; each subsequent encoder
            # has its own fusion block. Optional encoders can be skipped — gate learns no-op.
            # One block per encoder (~1.3M params each for D=512).
            self.latent_fusion = torch.nn.ModuleDict()
            for dataset_name in self.dataset_names[1:]:
                self.latent_fusion[dataset_name] = GatedLatentFusion(
                    hidden_dim=self.num_channels,
                )

        # Principal dataset is defined in config as model.principal_dataset.
        # Dropout probabilities are handled in anemoi-training and only a list of
        # dropped dataset names is passed into forward.
        self.principal_dataset_name = model_config.model.get("principal_dataset", self.dataset_names[0])
        assert self.principal_dataset_name in self.dataset_names, (
            "Configured principal dataset given in config file '%s' not in dataset_names=%s."
            % (self.principal_dataset_name, self.dataset_names)
        )


        # Decoder hidden -> data
        self.decoder_graph_provider = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(self._graph_name_hidden, "to", dataset_name)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                dst_size=self.node_attributes.num_nodes[dataset_name],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.num_channels,
                in_channels_dst=self.target_dim[dataset_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.output_dim[dataset_name],
                edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
            )

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )

        return x_data_latent, x_skip, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str,
    ):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
        assert (
            x_skip.shape[1] == x_out.shape[1]
        ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
        x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def _assert_valid_sharding(
        self,
        batch_size: int,
        ensemble_size: int,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

            assert (
                model_comm_group.size() == 1 or ensemble_size == 1
            ), "Ensemble size per device must be 1 when model is sharded across GPUs"

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        dropped_dataset_names: list[str] | set[str] | None = None,
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

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(x.keys())

        # Dataset dropout selection must be passed in from the training step
        primary_dataset = self.principal_dataset_name

        if dropped_dataset_names is None and dropped_dataset_names is not None:
            dropped_dataset_names = dropped_dataset_names
        dropped_dataset_names = set() if dropped_dataset_names is None else set(dropped_dataset_names)
        dropped_dataset_names.discard(primary_dataset)
        LOGGER.info(f"predict_step dropped_dataset_names: {dropped_dataset_names}")

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Process each active dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data,  # None if not sharded (in_out_sharded=False)
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset — always run to keep graph static for DDP.
            # If dropped, multiply latent by 0 so it contributes nothing.
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            keep_scale = 0.0 if dataset_name in dropped_dataset_names else 1.0
            x_latent = x_latent * keep_scale
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent

        # Fuse encoder latents sequentially: first encoder is the base latent,
        # subsequent encoders are folded in via gated fusion blocks (no attention).
        # Dropped datasets contribute zero latent — fusion block still runs (static graph)
        # but learns no-op. Order randomized during training to prevent order dependence.
        x_latent = dataset_latents[primary_dataset]
        remaining = [name for name in self.dataset_names if name != primary_dataset and name in dataset_latents]
        if self.training:
            random.shuffle(remaining)
        else:
            if dropped_dataset_names:
                LOGGER.info(f"Evaluation with dropped datasets: {dropped_dataset_names}")
        if self.latent_fusion_method == "sum":
            # x_latent = sum(dataset_latents[name] for name in remaining)
            x_latent = sum(dataset_latents.values())
        else:
            for dataset_name in remaining:
                x_latent = self.latent_fusion[dataset_name](x_latent, dataset_latents[dataset_name])

        # Processor
        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent = x_latent_proc + x_latent

        # Decoder — always run all decoders (static graph for DDP).
        # For dropped datasets, decoder still runs but output is replaced with input
        # so the loss sees zero tendency → no gradient signal for that dataset.
        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent, x_data_latent_dict[dataset_name]),
                batch_size=batch_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out = self._assemble_output(
                x_out, x_skip_dict[dataset_name], batch_size, ensemble_size, x[dataset_name].dtype, dataset_name
            )

            if dataset_name in dropped_dataset_names:
                # Replace output with NaN — NaN-aware loss will ignore this dataset.
                # Multiply by 0 and add NaN to keep decoder params in the graph.
                x_out = x_out * 0 + float("nan")

            x_out_dict[dataset_name] = x_out

        return x_out_dict

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
