# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import einops
import torch
from torch import Tensor

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.decoder_conditioning import DecoderConditioning
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing.qc_flags import QCFeaturizer
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class Nowcaster(AnemoiModelEncProcDec):
    """Unified QC-aware interpolator with decoder-only conditioning."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config = DotDict(model_config)

        # Boundary/input times are defined by the training task; keep for metadata.
        self.input_times = model_config.training.explicit_times.input
        self.output_times = model_config.training.explicit_times.target

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

        self.latent_skip = bool(model_config.model.get("latent_skip", False))

        # Optional union-mesh decoding (single decoder: hidden -> union mesh).
        # When enabled with decode_only=True, the model will decode once to the
        # union mesh and slice back to per-dataset tensors for downstream loss.
        self._union_cfg = DotDict(getattr(model_config.model, "union_mesh", {}))
        self._union_enabled = bool(self._union_cfg.get("enabled", False))
        self._union_name = str(self._union_cfg.get("name", "mesh"))
        self._union_return_mesh = bool(self._union_cfg.get("return_mesh", True))
        self._union_slices = dict(self._union_cfg.get("mesh_slices", None))
        self._union_infer_slices = False
        if self._union_slices is None:
            self._union_infer_slices = True

        self.union_decoder_graph_provider = None
        self.union_decoder = None
        if self._union_enabled:
            first_dataset_name = next(iter(self._graph_data.keys()))
            self.union_decoder_graph_provider = create_graph_provider(
                graph=self._graph_data[first_dataset_name][(self._graph_name_hidden, "to", self._union_name)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes[first_dataset_name].num_nodes[self._graph_name_hidden],
                dst_size=self.node_attributes[first_dataset_name].num_nodes[self._union_name],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            # Instantiate a decoder for hidden -> union mesh. Defaults to the same decoder config.
            out_channels_dst = int(
                self._union_cfg.get("out_channels", 0) or model_config.model.get("out_channels", 0) or 0,
            )
            if out_channels_dst <= 0:
                # Fall back to first dataset output dim if not provided.
                out_channels_dst = int(self.output_dim[first_dataset_name])

            from hydra.utils import instantiate

            self.union_decoder = instantiate(
                model_config.model.decoder,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=self.node_attributes[first_dataset_name].attr_ndims[self._union_name],
                hidden_dim=self.num_channels,
                out_channels_dst=out_channels_dst,
                edge_dim=self.union_decoder_graph_provider.edge_dim,
            )

        qc_cfg = getattr(model_config.data, "qc", None)
        self._qc_enabled = qc_cfg is not None
        self.qc_var = "qc_flags"
        self.qc: dict[str, QCFeaturizer] = {}
        if self._qc_enabled:
            # qc_cfg may be a dict keyed by dataset, or a single config.
            if isinstance(qc_cfg, dict) and "invalid_mask" not in qc_cfg:
                qc_by_ds = qc_cfg
            else:
                qc_by_ds = dict.fromkeys(self.input_dim.keys(), qc_cfg)

            for ds_name, cfg in qc_by_ds.items():
                cfg = DotDict(cfg)
                self.qc_var = str(cfg.get("qc_var", "qc_flags"))
                self.qc[ds_name] = QCFeaturizer(
                    invalid_mask=int(cfg.get("invalid_mask", 0)),
                    decoded_bits=list(cfg.get("decoded_bits", [])) or None,
                    embedding_dim=(
                        int(cfg.get("embedding", {}).get("dim", 0))
                        if isinstance(cfg.get("embedding", None), (dict, DotDict))
                        else int(cfg.get("embedding_dim", 0) or 0)
                    ),
                    embedding_bits=(
                        list(cfg.get("embedding", {}).get("bits", []))
                        if isinstance(cfg.get("embedding", None), (dict, DotDict))
                        else None
                    ),
                )

        dec_cfg = DotDict(getattr(model_config.model, "decoder_conditioning", {}))
        xdst_dim_by_ds = {ds: self._calculate_xdst_dim(ds) for ds in self.input_dim}
        latent_dim = int(self.num_channels)
        mesh_xdst_dim = None
        if self._union_enabled:
            first_ds = next(iter(self._graph_data.keys()))
            mesh_xdst_dim = int(self.node_attributes[first_ds].attr_ndims[self._union_name])

        self.cond = DecoderConditioning(
            dec_cfg,
            xdst_dim_by_ds=xdst_dim_by_ds,
            latent_dim=latent_dim,
            mesh_xdst_dim=mesh_xdst_dim,
        )

    def _calculate_xdst_dim(self, dataset_name: str) -> int:
        # Base input channels (excluding node attrs).
        base = len(self.input_times) * self.num_input_channels[dataset_name]
        # If QC is enabled, we may remove qc_flags from base and add derived channels.
        qc_extra = 0
        if self._qc_enabled and dataset_name in self.qc:
            featurizer = self.qc[dataset_name]
            # bit/emb dims not stored directly; infer from modules.
            if featurizer.decode is not None:
                qc_extra += int(featurizer.decode.bits.numel())
            qc_extra += 1  # valid
            if featurizer.emb is not None:
                qc_extra += int(featurizer.emb.emb.embedding_dim)
            # qc_flags itself is removed from numeric channels when qc is enabled.
            base -= 1

        node_attr = self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        return base + len(self.input_times) * qc_extra + node_attr

    def _append_qc_features(self, x: Tensor, *, dataset_name: str) -> Tensor:
        """Append QC features at encoding time."""
        if not self._qc_enabled or dataset_name not in self.qc:
            return x

        name_to_index = getattr(self.data_indices[dataset_name].model.input, "name_to_index", None)
        if name_to_index is None or self.qc_var not in name_to_index:
            msg = f"QC enabled but '{self.qc_var}' not found in model input variables for dataset '{dataset_name}'."
            raise KeyError(
                msg,
            )

        qc_idx = int(name_to_index[self.qc_var])
        qc_flags = x[..., qc_idx]
        feats = self.qc[dataset_name](qc_flags)

        # Remove qc_flags from numeric channels
        x_noqc = torch.cat([x[..., :qc_idx], x[..., qc_idx + 1 :]], dim=-1)

        extra = [feats.valid_mask.unsqueeze(-1)]
        if feats.bits is not None:
            extra.append(feats.bits)
        if feats.embedding is not None:
            extra.append(feats.embedding)
        extra_t = torch.cat(extra, dim=-1).to(x_noqc.dtype)

        return torch.cat([x_noqc, extra_t], dim=-1)

    def _assemble_xdst_base(
        self,
        x: Tensor,
        batch_size: int,
        *,
        grid_shard_shapes: dict | None,
        model_comm_group: ProcessGroup | None,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Assemble x_dst base features (time-stacked obs + node attributes)."""
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_size)
        grid_shard_shapes_ds = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        # Residual (for prognostic variables only).
        x_skip = self.residual[dataset_name](
            x,
            grid_shard_shapes=grid_shard_shapes_ds,
            model_comm_group=model_comm_group,
            n_step_output=1,  # single-step residual; multi-out handled by stacking
        )

        if grid_shard_shapes_ds is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data,
                0,
                shard_shapes_dim=grid_shard_shapes_ds,
                model_comm_group=model_comm_group,
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        x_aug = self._append_qc_features(x, dataset_name=dataset_name)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x_aug, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )

        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent,
            0,
            shard_shapes_dim=grid_shard_shapes_ds,
            model_comm_group=model_comm_group,
        )

        return x_data_latent, x_skip, shard_shapes_data

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        decoder_context: dict[str, dict[str, Tensor]] | None = None,
        model_comm_group: ProcessGroup | None = None,
        grid_shard_shapes: list | None = None,
    ) -> dict[str, Tensor]:
        ds0 = next(iter(self._graph_data.keys()))
        dataset_names = list(x.keys())

        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, sharded, model_comm_group)

        # ---- encode each dataset -> hidden, then sum ----
        latents = []
        shard_shapes_hidden = None
        for ds in dataset_names:
            x_dst, _, shard_shapes_data = self._assemble_xdst_base(
                x[ds],
                batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=ds,
            )
            x_hidden = self.node_attributes[ds](self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden = get_shard_shapes(x_hidden, 0, model_comm_group)

            enc_edge_attr, enc_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[ds].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            _, h = self.encoder[ds](
                (x_dst, x_hidden),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                edge_attr=enc_edge_attr,
                edge_index=enc_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=sharded,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=True,
                edge_shard_shapes=enc_edge_shard_shapes,
            )
            latents.append(h)

        h = sum(latents)

        proc_edge_attr, proc_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        h = self.processor(
            h,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )
        if self.latent_skip:
            h = h + sum(latents)

        # ---- decode on union mesh with decoder-only time conditioning ----
        mesh_attrs = self.node_attributes[ds0](self._union_name, batch_size=batch_size)
        shard_shapes_mesh = get_shard_shapes(mesh_attrs, 0, model_comm_group)

        cond = decoder_context[self._union_name]["cond"]
        t_out = int(cond.shape[1])

        dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.union_decoder_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        outs = []
        for t in range(t_out):
            cond_flat = einops.rearrange(cond[:, t], "b e g c -> (b e g) c")
            h_t, mesh_t = self.cond.apply(
                h,
                mesh_attrs,
                cond_flat=cond_flat,
                dataset_name=self._union_name,
                is_union_mesh=True,
            )

            y = self.union_decoder(
                (h_t, mesh_t),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_mesh),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=False,
                edge_shard_shapes=dec_edge_shard_shapes,
            )
            y = einops.rearrange(y, "(b e g) v -> b e g v", b=batch_size, e=ensemble_size).to(dtype=x[ds0].dtype)
            outs.append(y)

        mesh_pred = torch.stack(outs, dim=1)
        return {self._union_name: mesh_pred}
