# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Deterministic local downscaler (fine-scale epic, Track E stage 1, 2026-09-02).

The local high-resolution branch ALONE on the output grid, trained with the lane's weighted
mean-squared error on the normalised residual ``y - interp(x)``. It has two uses:

1. its held-out fine-band skill, per band and per terrain class, is the strongest available
   lower bound on how much of the fine band is predictable from the driver and the static
   fields (the "predictability ceiling" the epic asks for);
2. frozen, it can serve as the deterministic first stage of a residual-diffusion design, the
   diffusion model then modelling only ``y - prior``.

It reuses the diffusion downscaler's residual machinery (interpolation connection,
``compute_residuals``, ``add_interp_to_state``, ``_before_sampling``/``_after_sampling``) and
replaces the encoder-processor-decoder trunk by ``LocalHresBranch`` with plain layer norms
(there is no noise level to condition on). Configured by ``model.model.local_downscaler``.
"""

from __future__ import annotations

import logging
from typing import Optional

import einops
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def plain_layer_kernels() -> DotDict:
    """Unconditioned kernels: plain LayerNorm (the deterministic model has no noise level)."""
    return DotDict(
        {
            "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
            "QueryNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
            "KeyNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
        }
    )


class AnemoiLocalDownscaler(AnemoiD2ModelEncProcDec):
    """Deterministic residual downscaler made of the local branch only."""

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Interpolated driver + high-resolution forcings + node attributes; no noised target."""
        num_channels_in_lres = len(self.data_indices["in_lres"].model.input)
        num_channels_in_hres = len(self.data_indices["in_hres"].model.input)
        return (
            self.multi_step * num_channels_in_lres
            + self.multi_step * num_channels_in_hres
            + self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        )

    def _build_networks(self, model_config: DotDict) -> None:
        from anemoi.models.layers.graph_provider import create_graph_provider
        from anemoi.models.layers.hres_branch import CONFIG_KEYS
        from anemoi.models.layers.hres_branch import LocalHresBranch

        cfg = model_config["model"]["model"].get("local_downscaler", None) or {}
        dataset_name = self._decoder_datasets[0]
        edge_key = (self._graph_name_data, "to", self._graph_name_data)
        graph = self._graph_data[dataset_name]
        if edge_key not in graph.edge_types:
            raise ValueError(
                f"AnemoiLocalDownscaler requires a {edge_key} edge set in the '{dataset_name}' graph; "
                "augment the graph file first (graphs/scripts/augment_data_knn_edges.py)"
            )
        num_data = self.node_attributes[dataset_name].num_nodes[self._graph_name_data]
        self.hres_branch_graph_provider = create_graph_provider(
            graph=graph[edge_key],
            edge_attributes=list(cfg.get("sub_graph_edge_attributes", ["edge_length", "edge_dirs"])),
            src_size=num_data,
            dst_size=num_data,
            trainable_size=0,
        )
        kwargs = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
        kwargs.setdefault("layer_kernels", plain_layer_kernels())
        kwargs.setdefault("zero_init_head", True)
        kwargs["detach_inputs"] = False
        self.hres_branch = LocalHresBranch(
            in_features=self.input_dim[dataset_name],
            out_features=self.num_output_channels[dataset_name],
            edge_dim=self.hres_branch_graph_provider.edge_dim,
            cond_dim=self.noise_cond_dim,
            **kwargs,
        )
        # No encoder / processor / decoder: keep the attribute names the parent expects, empty.
        self.encoder_graph_provider = nn.ModuleDict()
        self.encoder = nn.ModuleDict()
        self.decoder_graph_provider = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        LOGGER.info(
            "AnemoiLocalDownscaler: input %d -> output %d channels, %d data->data edges, %d parameters",
            self.input_dim[dataset_name], self.num_output_channels[dataset_name],
            int(graph[edge_key].edge_index.shape[1]), sum(p.numel() for p in self.hres_branch.parameters()),
        )

    def _build_hres_branch(self, model_config: DotDict) -> None:
        """The branch is built in _build_networks; the diffusion hook must not build a second one."""
        return

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y_noised: Optional[dict[str, torch.Tensor]] = None,
        sigma: Optional[dict[str, torch.Tensor]] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Deterministic forward: {"out_hres": normalised residual estimate}. y_noised and sigma are ignored."""
        dataset_name = "out_hres"
        x_in_lres = x["in_lres"]
        x_in_hres = x["in_hres"]
        batch_size = x_in_lres.shape[0]
        ensemble_size = x_in_lres.shape[2]
        bse = batch_size * ensemble_size
        in_out_sharded = self._resolve_in_out_sharded(dataset_names=[dataset_name], grid_shard_sizes=grid_shard_sizes)
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=bse)
        grid_shard_sizes_data = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
        if grid_shard_sizes_data is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes_data, model_comm_group)

        h_in = torch.cat(
            (
                einops.rearrange(x_in_lres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(x_in_hres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        out = self.hres_branch(
            h_in,
            None,
            graph_provider=self.hres_branch_graph_provider,
            batch_size=bse,
            node_shard_sizes=grid_shard_sizes_data,
            model_comm_group=model_comm_group,
            cond=None,
            inputs_sharded=bool(in_out_sharded[dataset_name]),
        )
        out = einops.rearrange(
            out, "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
            batch=batch_size, ensemble=ensemble_size, time=1,
        ).to(x_in_lres.dtype)
        return {dataset_name: out}

    def fwd_with_preconditioning(self, *args, **kwargs):  # noqa: D102
        raise NotImplementedError("AnemoiLocalDownscaler is deterministic; use forward()")

    def sample(self, *args, **kwargs):  # noqa: D102
        raise NotImplementedError("AnemoiLocalDownscaler is deterministic; use predict_step()")

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        post_processors: dict[str, nn.Module],
        n_step_input: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        pre_processors_tendencies: Optional[nn.Module] = None,
        post_processors_tendencies: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Deterministic prediction: interpolate, normalise, one forward, denormalise and add the driver back."""
        with torch.no_grad():
            for dataset_name, dataset_tensor in batch.items():
                assert len(dataset_tensor.shape) == 5, f"{dataset_name}: expected 5-D, got {dataset_tensor.shape}"
            before_sampling_data, grid_shard_sizes = self._before_sampling(
                batch, pre_processors, n_step_input, model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies, **kwargs,
            )
            x_in_lres_upsampled, x_in_hres = before_sampling_data[0], before_sampling_data[1]
            out = self(
                {"in_lres": x_in_lres_upsampled, "in_hres": x_in_hres},
                model_comm_group=model_comm_group, grid_shard_sizes=grid_shard_sizes,
            )["out_hres"].to(x_in_lres_upsampled.dtype)
            out = self._after_sampling(
                out, post_processors, before_sampling_data, model_comm_group, grid_shard_sizes, gather_out,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies, **kwargs,
            )
        return out
