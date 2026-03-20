# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import ast
import logging
from typing import Optional

import einops
import torch
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
    """Message passing interpolating graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        model_config = DotDict(model_config)
        dec_cfg = DotDict(getattr(model_config.model, "decoder_conditioning", {}))
        self._cond_key = str(dec_cfg.get("key", "cond"))
        qc_cfg = getattr(model_config.data, "qc", None)
        self._qc_enabled = qc_cfg is not None
        self.qc: dict[str, QCFeaturizer] = {}
        self.qc_var: dict[str, str] = {}
        if self._qc_enabled:
            for ds_name, cfg in qc_cfg.items():
                cfg = DotDict(cfg)
                self.qc_var[ds_name] = str(cfg.get("qc_var", "qc_flags"))
                self.qc[str(ds_name)] = QCFeaturizer(method=cfg.get("method", "decode_bits"), qc_cfg=cfg)
        datasets = list(model_config.training.scalers.datasets)
        if model_config.training.get("known_future_variables", None) is not None:
            LOGGER.info("Ignoring known_future_variables for Nowcaster: decoder conditioning is time-fraction-only.")
        self.known_future_variables = {str(dataset_name): [] for dataset_name in datasets}
        self.num_channels = model_config.model.num_channels
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )
        # Register QC featurizers so they follow model device/dtype moves.
        self.qc = nn.ModuleDict(self.qc)
        self._cond_xdst = nn.ModuleDict()
        for ds_name, xdst_dim in self.input_dim.items():
            self._cond_xdst[str(ds_name)] = build_decoder_conditioner(
                method="film",
                x_dim=xdst_dim,
                cond_dim=1,
                cfg=dec_cfg,
            )

    def _append_qc_features(self, x: torch.Tensor, *, dataset_name: str) -> torch.Tensor:
        """Append QC features at encoding time.

        x: [B, T_in, E, G, C_in]
        returns x_aug: [B, T_in, E, G, C_in_aug]
        """
        if not self._qc_enabled or dataset_name not in self.qc:
            return x

        name_to_index = getattr(self.data_indices[dataset_name].model.input, "name_to_index", None)
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
            # Derived: bits(K) | emb(D)
            featurizer = self.qc[dataset_name]
            if hasattr(featurizer, "mask"):  # valid data mask
                qc_extra += 1
            if isinstance(featurizer.feat, QCDecodeBits):
                qc_extra += int(len(featurizer.feat.bits))
            elif isinstance(featurizer.feat, QCPackedEmbedding):
                qc_extra += int(featurizer.feat.emb.num_embeddings)
            # qc_flags itself is removed from numeric channels when qc is enabled.
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
        decode_datasets = list(decode_dataset_names) if decode_dataset_names is not None else dataset_names
        unknown_decode_datasets = [name for name in decode_datasets if name not in dataset_names]
        if unknown_decode_datasets:
            raise KeyError(f"decode_dataset_names contains unknown datasets: {unknown_decode_datasets}")

        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = {dataset_name: False for dataset_name in dataset_names}
        if grid_shard_shapes is not None:
            for dataset_name, shard_shapes in grid_shard_shapes.items():
                in_out_sharded[dataset_name] = shard_shapes is not None
                self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        x_dst_base_dict = {}
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)
        for dataset_name in dataset_names:
            x_in = x[dataset_name]
            if (
                x_in.is_floating_point()
                and x_hidden_latent.is_floating_point()
                and x_in.dtype != x_hidden_latent.dtype
            ):
                x_in = x_in.to(dtype=x_hidden_latent.dtype)
            x_aug = self._append_qc_features(x_in, dataset_name=dataset_name)
            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x_aug,
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_dst_base_dict[dataset_name] = x_data_latent
            x_skip_dict[dataset_name] = x_skip
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
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
                x_src_is_sharded=in_out_sharded[dataset_name],  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                edge_shard_shapes=enc_edge_shard_shapes,
            )
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent
        x_latent = sum(dataset_latents.values())
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
        t_out = self.n_step_output
        if decoder_context is not None:
            for ds in decode_datasets:
                if ds in decoder_context:
                    t_out = int(decoder_context[ds][self._cond_key].shape[1])
                    break

        # Decode: run once per provider mesh, then map to each target dataset.
        provider_by_dataset = getattr(self, "decoder_provider_by_dataset", {})
        decode_groups: dict[str, list[str]] = {}
        for dataset_name in decode_datasets:
            provider_name = str(provider_by_dataset.get(dataset_name, dataset_name))
            decode_groups.setdefault(provider_name, []).append(dataset_name)

        provider_x_dst_base: dict[str, torch.Tensor] = {}
        provider_shard_shapes: dict[str, list] = {}
        provider_dst_sharded: dict[str, bool] = {}
        for provider_name in decode_groups:
            if provider_name in x_dst_base_dict:
                provider_x_dst_base[provider_name] = x_dst_base_dict[provider_name]
                provider_shard_shapes[provider_name] = shard_shapes_data_dict[provider_name]
                provider_dst_sharded[provider_name] = in_out_sharded[provider_name]
                continue

            x_dst_provider = self.node_attributes(provider_name, batch_size=batch_size)
            if x_dst_provider.device != x_hidden_latent.device:
                x_dst_provider = x_dst_provider.to(x_hidden_latent.device)
            if (
                x_dst_provider.is_floating_point()
                and x_hidden_latent.is_floating_point()
                and x_dst_provider.dtype != x_hidden_latent.dtype
            ):
                x_dst_provider = x_dst_provider.to(dtype=x_hidden_latent.dtype)
            provider_x_dst_base[provider_name] = x_dst_provider
            provider_shard_shapes[provider_name] = get_shard_shapes(x_dst_provider, 0, model_comm_group)
            provider_dst_sharded[provider_name] = False

        provider_slices_cache: dict[str, dict[str, tuple[int, int]] | None] = {}

        def _get_provider_slices(provider_name: str) -> dict[str, tuple[int, int]] | None:
            if provider_name in provider_slices_cache:
                return provider_slices_cache[provider_name]

            if provider_name not in self._graph_data.node_types:
                provider_slices_cache[provider_name] = None
                return None

            slices_value = self._graph_data[provider_name].get("_slices", None)
            if slices_value is None:
                provider_slices_cache[provider_name] = None
                return None

            if isinstance(slices_value, str):
                parsed = ast.literal_eval(slices_value)
            else:
                parsed = slices_value
            provider_slices_cache[provider_name] = {
                str(key): (int(value[0]), int(value[1])) for key, value in dict(parsed).items()
            }
            return provider_slices_cache[provider_name]

        y_pred = {}
        for provider_name, provider_targets in decode_groups.items():
            exemplar_dataset = provider_targets[0]
            exemplar_output_dim = int(self.output_dim[exemplar_dataset])
            for dataset_name in provider_targets[1:]:
                if int(self.output_dim[dataset_name]) != exemplar_output_dim:
                    raise ValueError(
                        "Datasets mapped to the same decoder provider must have equal output dimensions: "
                        f"{exemplar_dataset}={self.output_dim[exemplar_dataset]}, "
                        f"{dataset_name}={self.output_dim[dataset_name]}"
                    )

            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                provider_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            cond = None
            if decoder_context is not None and exemplar_dataset in decoder_context:
                cond = decoder_context[exemplar_dataset].get(self._cond_key, None)

            outs = []
            for t in range(t_out):
                x_dst = provider_x_dst_base[provider_name]
                x_lat = x_latent_proc
                if cond is not None:
                    cond_t = cond[:, t]
                    cond_flat = einops.rearrange(cond_t, "batch ensemble grid c -> (batch ensemble grid) c")

                    if exemplar_dataset in self._cond_xdst:
                        if cond_flat is not None:
                            if cond_flat.device != x_dst.device:
                                cond_flat = cond_flat.to(x_dst.device)
                            if (
                                cond_flat.is_floating_point()
                                and x_dst.is_floating_point()
                                and cond_flat.dtype != x_dst.dtype
                            ):
                                cond_flat = cond_flat.to(dtype=x_dst.dtype)
                            x_dst = self._cond_xdst[str(exemplar_dataset)](x_dst, cond_flat)

                x_out = self.decoder[provider_name](
                    (x_lat, x_dst),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hidden, provider_shard_shapes[provider_name]),
                    edge_attr=dec_edge_attr,
                    edge_index=dec_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=True,
                    x_dst_is_sharded=provider_dst_sharded[provider_name],
                    keep_x_dst_sharded=provider_dst_sharded[provider_name],
                    edge_shard_shapes=dec_edge_shard_shapes,
                )
                x_out = einops.rearrange(
                    x_out,
                    "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                    time=1,
                ).to(x[exemplar_dataset].dtype).clone()
                outs.append(x_out)

            provider_concat = torch.cat(outs, dim=1)
            provider_slices = _get_provider_slices(provider_name)
            for dataset_name in provider_targets:
                if provider_name == dataset_name:
                    concat = provider_concat.clone()
                else:
                    if provider_slices is None or dataset_name not in provider_slices:
                        raise KeyError(
                            f"Provider '{provider_name}' has no slice for dataset '{dataset_name}'. "
                            "For union decoding, provider nodes must expose '_slices' with dataset ranges."
                        )
                    start, end = provider_slices[dataset_name]
                    concat = provider_concat[..., start:end, :].clone()

                if concat.shape[3] != x_skip_dict[dataset_name].shape[3]:
                    raise ValueError(
                        f"Decoded grid size for '{dataset_name}' ({concat.shape[3]}) does not match "
                        f"residual input grid size ({x_skip_dict[dataset_name].shape[3]})."
                    )

                concat[..., self._internal_output_idx[dataset_name]] += x_skip_dict[dataset_name][
                    ..., self._internal_input_idx[dataset_name]
                ]

                for bounding in self.boundings[dataset_name]:
                    # bounding performed in the order specified in the config file
                    concat = bounding(concat)

                y_pred[dataset_name] = concat

        return y_pred
