# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""QC-aware observation-guided interpolator with decoder-only conditioning.

This model implements the contract described in the provided design note:
  - Encode only boundary-time information (obs at input/boundary times)
  - QC-aware encoding (valid mask + optional bits + optional embedding)
  - Decoder-only conditioning on target-time context (time embedding +
    future-known target forcings)
  - Multi-output implemented as a decoder scan over target times

The implementation is written to fit the Anemoi encoder–processor–decoder
pattern and to minimise changes to the rest of the stack.

Notes
-----
* The model expects input tensors shaped like Anemoi models:
    x[dataset] : [B, T_in, E, G, C_in]
  where G is the flattened grid/node dimension.
* For decoder-only conditioning, pass decoder_context[dataset]["cond"]:
    cond : [B, T_out, E, G, C_cond]
  The task is responsible for constructing cond from time + target forcings.
"""

from __future__ import annotations

import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.decoder_conditioning import build_decoder_conditioner
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing.qc_flags import QCFeaturizer
from anemoi.utils.config import DotDict

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

        # QC configuration (per dataset)
        qc_cfg = getattr(model_config.data, "qc", None)
        self._qc_enabled = qc_cfg is not None
        self.qc_var = "qc_flags"
        self.qc: dict[str, QCFeaturizer] = {}
        if self._qc_enabled:
            # qc_cfg may be a dict keyed by dataset, or a single config.
            if isinstance(qc_cfg, dict) and "invalid_mask" not in qc_cfg:
                qc_by_ds = qc_cfg
            else:
                qc_by_ds = {name: qc_cfg for name in self.input_dim.keys()}

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

        # Decoder-only conditioning configuration
        dec_cfg = DotDict(getattr(model_config.model, "decoder_conditioning", {}))
        self._cond_enabled = bool(dec_cfg)
        self._cond_key = str(dec_cfg.get("key", "cond"))  # decoder_context[ds][key]
        self._cond_method = str(dec_cfg.get("method", "film"))
        self._cond_apply_to = str(dec_cfg.get("apply_to", "x_dst"))  # x_dst | latent | both
        self._cond_dim = int(dec_cfg.get("cond_dim", 0))

        # Build per-dataset conditioning adapter for x_dst (data-node query features).
        self._cond_xdst: dict[str, torch.nn.Module] = {}
        if self._cond_enabled and self._cond_apply_to in {"x_dst", "both"}:
            for ds_name in self.input_dim.keys():
                # x_dst feature dimension is: (T_in * C_in_aug) + node_attr_dim
                xdst_dim = self._calculate_xdst_dim(ds_name)
                self._cond_xdst[ds_name] = build_decoder_conditioner(
                    method=self._cond_method,
                    x_dim=xdst_dim,
                    cond_dim=self._cond_dim,
                    cfg=dec_cfg,
                )

        self._cond_latent: Optional[torch.nn.Module] = None
        if self._cond_enabled and self._cond_apply_to in {"latent", "both"}:
            # Latent dimension depends on processor/hidden graph; use model_config.model.dim if present.
            latent_dim = int(getattr(model_config.model, "hidden_dim", 0) or getattr(model_config.model, "dim", 0) or 0)
            if latent_dim <= 0:
                LOGGER.warning(
                    "decoder_conditioning.apply_to includes 'latent' but latent_dim is unknown; skipping latent conditioning"
                )
            else:
                self._cond_latent = build_decoder_conditioner(
                    method=self._cond_method,
                    x_dim=latent_dim,
                    cond_dim=self._cond_dim,
                    cfg=dec_cfg,
                )

    def _calculate_xdst_dim(self, dataset_name: str) -> int:
        # Base input channels (excluding node attrs).
        base = len(self.input_times) * self.num_input_channels[dataset_name]
        # If QC is enabled, we may remove qc_flags from base and add derived channels.
        qc_extra = 0
        if self._qc_enabled and dataset_name in self.qc:
            # Derived: valid(1) + bits(K) + emb(D)
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
        """Append QC features at encoding time.

        x: [B, T_in, E, G, C_in]
        returns x_aug: [B, T_in, E, G, C_in_aug]
        """
        if not self._qc_enabled or dataset_name not in self.qc:
            return x

        name_to_index = getattr(self.data_indices[dataset_name].model.input, "name_to_index", None)
        if name_to_index is None or self.qc_var not in name_to_index:
            raise KeyError(
                f"QC enabled but '{self.qc_var}' not found in model input variables for dataset '{dataset_name}'."
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
        model_comm_group: Optional[ProcessGroup],
        dataset_name: str,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
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
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes_ds, model_comm_group=model_comm_group
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
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes_ds, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def forward(
        self,
        x: dict[str, Tensor],
        target_forcing: dict[str, Tensor] | None = None,
        *,
        decoder_context: Optional[dict[str, dict[str, Tensor]]] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
          Boundary-time inputs (already sliced by the task): [B,T_in,E,G,C]
        decoder_context:
          Optional per-dataset dict containing conditioning tensors for target times.
          Expected key: self._cond_key (default "cond") with shape [B,T_out,E,G,C_cond].

        Returns
        -------
        Dict[str, Tensor]
          y_pred per dataset with shape [B,T_out,E,G,C_out]
        """
        dataset_names = list(x.keys())
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Encode each dataset, then sum latents.
        dataset_latents = {}
        x_skip_dict = {}
        x_dst_base_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}
        for dataset_name in dataset_names:
            x_dst_base, x_skip, shard_shapes_data = self._assemble_xdst_base(
                x[dataset_name],
                batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            x_dst_base_dict[dataset_name] = x_dst_base
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            enc_edge_attr, enc_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )

            _, x_latent = self.encoder[dataset_name](
                (x_dst_base, x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden_dict[dataset_name]),
                edge_attr=enc_edge_attr,
                edge_index=enc_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=True,
                edge_shard_shapes=enc_edge_shard_shapes,
            )
            dataset_latents[dataset_name] = x_latent

        x_latent = sum(dataset_latents.values())

        # Processor
        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        proc_edge_attr, proc_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Determine T_out from decoder_context if present, else default to len(output_times).
        t_out = len(self.output_times)
        if decoder_context is not None:
            for ds in dataset_names:
                if ds in decoder_context and self._cond_key in decoder_context[ds]:
                    t_out = int(decoder_context[ds][self._cond_key].shape[1])
                    break

        # Decode: scan over target times, applying conditioning to x_dst (and optionally latent).
        y_pred: dict[str, Tensor] = {}
        for dataset_name in dataset_names:
            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )

            cond = None
            if self._cond_enabled and decoder_context is not None and dataset_name in decoder_context:
                cond = decoder_context[dataset_name].get(self._cond_key, None)
            if cond is not None and cond.shape[-1] != self._cond_dim:
                raise ValueError(
                    f"decoder_context[{dataset_name}][{self._cond_key}] last dim {cond.shape[-1]} != cond_dim {self._cond_dim}"
                )

            outs = []
            for t in range(t_out):
                x_dst = x_dst_base_dict[dataset_name]
                x_lat = x_latent_proc

                if cond is not None:
                    # cond_t: [B,E,G,C] -> flatten to match x_dst [B*E*G, D]
                    cond_t = cond[:, t]
                    cond_flat = einops.rearrange(cond_t, "batch ensemble grid c -> (batch ensemble grid) c")

                    if self._cond_latent is not None:
                        x_lat = self._cond_latent(x_lat, cond_flat)  # expects matching first dim when unsharded

                    if dataset_name in self._cond_xdst:
                        x_dst = self._cond_xdst[dataset_name](x_dst, cond_flat)

                x_out = self.decoder[dataset_name](
                    (x_lat, x_dst),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hidden, shard_shapes_data_dict[dataset_name]),
                    edge_attr=dec_edge_attr,
                    edge_index=dec_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=True,
                    x_dst_is_sharded=in_out_sharded,
                    keep_x_dst_sharded=in_out_sharded,
                    edge_shard_shapes=dec_edge_shard_shapes,
                )

                # x_out is (B*E*G, (time vars)) where time==1 for a single decode.
                x_out = einops.rearrange(
                    x_out,
                    "(batch ensemble grid) (vars) -> batch ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                ).to(dtype=x[dataset_name].dtype)

                # Apply residual + bounds for this time.
                if x_skip_dict[dataset_name] is not None:
                    # residual is shaped (B, time=1, E, G, vars)
                    x_out[..., self._internal_output_idx[dataset_name]] += x_skip_dict[dataset_name].squeeze(1)[
                        ..., self._internal_input_idx[dataset_name]
                    ]

                for bounding in self.boundings[dataset_name]:
                    x_out = bounding(x_out.unsqueeze(1)).squeeze(1)

                outs.append(x_out)

            y_pred[dataset_name] = torch.stack(outs, dim=1)  # [B,T_out,E,G,vars]

        return y_pred


# Backwards-compatible alias (some configs refer to ObsInterpolatorMultiOut)
ObsInterpolatorMultiOut = Nowcaster
