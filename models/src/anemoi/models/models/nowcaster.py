# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.decoder_conditioning import build_decoder_conditioner
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.preprocessing.qc_flags import QCDecodeBits
from anemoi.models.preprocessing.qc_flags import QCFeaturizer
from anemoi.models.preprocessing.qc_flags import QCPackedEmbedding
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class Nowcaster(AnemoiModelEncProcDec):
    """Single-dataset nowcaster with decoder-side future-variable conditioning."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config = DotDict(model_config)
        if len(data_indices) != 1:
            raise ValueError("Nowcaster currently supports exactly one dataset.")

        dec_cfg = DotDict(getattr(model_config.model, 'decoder_conditioning', {}))
        self._cond_key = str(dec_cfg.get('key', 'cond'))

        qc_cfg = getattr(model_config.data, 'qc', None)
        self._qc_enabled = qc_cfg is not None
        self.qc: dict[str, QCFeaturizer] = {}
        self.qc_var: dict[str, str] = {}
        if self._qc_enabled:
            for ds_name, cfg in qc_cfg.items():
                cfg = DotDict(cfg)
                self.qc_var[str(ds_name)] = str(cfg.get('qc_var', 'qc_flags'))
                self.qc[str(ds_name)] = QCFeaturizer(method=cfg.get('method', 'decode_bits'), qc_cfg=cfg)

        raw_known_future = model_config.training.get('known_future_variables', None)
        self.known_future_variables = []
        if raw_known_future is not None:
            self.known_future_variables = list(OmegaConf.to_container(raw_known_future, resolve=True))

        self.num_channels = model_config.model.num_channels
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

        self.dataset_name = next(iter(self.dataset_names))
        self.qc = nn.ModuleDict(self.qc)

        cond_dim = 1 + len(self.known_future_variables)
        method = str(dec_cfg.get('method', 'film'))
        self._cond_xdst = nn.ModuleDict(
            {
                self.dataset_name: build_decoder_conditioner(
                    method=method,
                    x_dim=int(self.input_dim[self.dataset_name]),
                    cond_dim=cond_dim,
                    cfg=dec_cfg,
                ),
            },
        )

    def _append_qc_features(self, x: torch.Tensor, *, dataset_name: str) -> torch.Tensor:
        if not self._qc_enabled or dataset_name not in self.qc:
            return x

        name_to_index = getattr(self.data_indices[dataset_name].model.input, 'name_to_index', None)
        if name_to_index is None or self.qc_var[dataset_name] not in name_to_index:
            raise KeyError(
                f"QC enabled but '{self.qc_var[dataset_name]}' not found in model input variables for dataset '{dataset_name}'."
            )

        qc_idx = int(name_to_index[self.qc_var[dataset_name]])
        qc_flags = x[..., qc_idx]
        feats = self.qc[dataset_name](qc_flags)
        x_noqc = torch.cat([x[..., :qc_idx], x[..., qc_idx + 1 :]], dim=-1)
        return torch.cat([x_noqc, feats], dim=-1)

    def _calculate_output_dim(self, dataset_name: str) -> int:
        return self.num_output_channels[dataset_name]

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base = self.n_step_input * self.num_input_channels[dataset_name]
        qc_extra = 0
        if self._qc_enabled and dataset_name in self.qc:
            featurizer = self.qc[dataset_name]
            if hasattr(featurizer, 'mask'):
                qc_extra += 1
            if isinstance(featurizer.feat, QCDecodeBits):
                qc_extra += int(len(featurizer.feat.bits))
            elif isinstance(featurizer.feat, QCPackedEmbedding):
                qc_extra += int(featurizer.feat.emb.num_embeddings)
            base -= self.n_step_input

        node_attr = self.node_attributes.attr_ndims[dataset_name]
        return base + self.n_step_input * qc_extra + node_attr

    def forward(
        self,
        x: dict[str, torch.Tensor],
        *,
        decoder_context: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        decode_dataset_names: Optional[tuple[str, ...] | list[str]] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, list] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(x) != 1:
            raise ValueError('Nowcaster forward expects exactly one dataset in the input batch.')

        dataset_name = next(iter(x.keys()))
        if dataset_name != self.dataset_name:
            raise KeyError(f"Unexpected dataset '{dataset_name}', expected '{self.dataset_name}'.")

        if decode_dataset_names is not None:
            decode_list = list(decode_dataset_names)
            if decode_list != [dataset_name]:
                raise ValueError(
                    f"Nowcaster supports exactly one decoded dataset '{dataset_name}', got {decode_list}."
                )

        batch_size = int(x[dataset_name].shape[0])
        ensemble_size = int(x[dataset_name].shape[2])

        dataset_grid_shard_shapes = None if grid_shard_shapes is None else grid_shard_shapes.get(dataset_name)
        in_out_sharded = dataset_grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        x_in = x[dataset_name]
        if x_in.is_floating_point() and x_hidden_latent.is_floating_point() and x_in.dtype != x_hidden_latent.dtype:
            x_in = x_in.to(dtype=x_hidden_latent.dtype)
        x_aug = self._append_qc_features(x_in, dataset_name=dataset_name)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x_aug,
            batch_size=batch_size,
            grid_shard_shapes={dataset_name: dataset_grid_shard_shapes},
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )

        encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[dataset_name].get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        x_data_latent, x_latent = self.encoder[dataset_name](
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            edge_attr=encoder_edge_attr,
            edge_index=encoder_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
            edge_shard_shapes=enc_edge_shard_shapes,
        )

        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )
        x_latent_proc = x_latent_proc + x_latent

        cond = None
        if decoder_context is not None and dataset_name in decoder_context:
            cond = decoder_context[dataset_name].get(self._cond_key, None)
        t_out = int(cond.shape[1]) if cond is not None else int(self.n_step_output)

        decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[dataset_name].get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        outs = []
        for t in range(t_out):
            x_dst = x_data_latent
            if cond is not None:
                cond_t = cond[:, t]
                cond_flat = einops.rearrange(cond_t, 'batch ensemble grid c -> (batch ensemble grid) c')
                if cond_flat.device != x_dst.device:
                    cond_flat = cond_flat.to(x_dst.device)
                if cond_flat.is_floating_point() and x_dst.is_floating_point() and cond_flat.dtype != x_dst.dtype:
                    cond_flat = cond_flat.to(dtype=x_dst.dtype)
                x_dst = self._cond_xdst[dataset_name](x_dst, cond_flat)

            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_dst),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
                edge_shard_shapes=dec_edge_shard_shapes,
            )
            x_out = einops.rearrange(
                x_out,
                '(batch ensemble grid) (time vars) -> batch time ensemble grid vars',
                batch=batch_size,
                ensemble=ensemble_size,
                time=1,
            ).to(x[dataset_name].dtype).clone()
            x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]
            for bounding in self.boundings[dataset_name]:
                x_out = bounding(x_out)
            outs.append(x_out)

        return {dataset_name: torch.cat(outs, dim=1)}
