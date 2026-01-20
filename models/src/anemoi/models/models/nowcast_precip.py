# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""Union-mesh multi-dataset nowcasting prototype.

This is a *working* prototype for OPERA + Nordic radar + Netatmo multi-source
nowcasting with:
  - per-source QC flag decoding (torch) -> valid + embedding (+ optional bits)
  - per-source node encoders (ViT for gridded sources, MLP for point sources)
  - graph encoder (data -> hidden) per dataset, shared hidden processor
  - single graph decoder (hidden -> union mesh)
  - slicing union mesh back to per-dataset tensors (for masked loss downstream)

The design keeps all graph node sets flattened (with .pos coordinates). Gridded
sources are reshaped only inside the node encoders using provided xdim/ydim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models.base import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def decode_bits_torch(flags: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Decode packed integer QC flags into bit indicators.

    Parameters
    ----------
    flags : torch.Tensor
        Packed integer flags.
    num_bits : int
        Number of bits.

    Returns
    -------
    torch.Tensor
        Tensor of shape (*flags.shape, num_bits) with 0/1 values.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be > 0")
    # Ensure integer
    if not flags.dtype.is_floating_point and flags.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        flags = flags.to(torch.int64)
    if flags.dtype.is_floating_point:
        flags = flags.to(torch.int64)
    shifts = torch.arange(num_bits, device=flags.device, dtype=torch.int64)
    # (..., 1) >> (num_bits) -> (..., num_bits)
    bits = ((flags.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    return bits


@dataclass
class QCScheme:
    num_bits: int
    embed_dim: int
    invalid_mask: int = 0  # if 0, defaults to flags==0
    include_bits: bool = False
    add_valid: bool = True


class QCTorchFeaturizer(nn.Module):
    """Create (valid + embedding (+ optional bits)) from packed qc flags."""

    def __init__(self, scheme: QCScheme) -> None:
        super().__init__()
        self.scheme = scheme
        self.embedding = nn.Embedding(2**scheme.num_bits, scheme.embed_dim)

    def forward(self, qc_flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # qc_flags: [B, T, E, G] or [B, T, E, N]
        flags_int = qc_flags.to(torch.int64)

        if self.scheme.invalid_mask != 0:
            valid = (flags_int & int(self.scheme.invalid_mask)) == 0
        else:
            valid = flags_int == 0
        valid_f = valid.to(torch.float32)

        # categorical embedding of full bitmask
        # clamp to embedding table range (defensive)
        max_id = (2**self.scheme.num_bits) - 1
        ids = torch.clamp(flags_int, 0, max_id)
        emb = self.embedding(ids)  # [..., D]

        bits = None
        if self.scheme.include_bits:
            bits = decode_bits_torch(flags_int, self.scheme.num_bits).to(torch.float32)
        return valid_f, emb, bits


class GridViTNodeEncoder(nn.Module):
    """Very small ViT-style encoder for flattened gridded nodes.

    The goal is to produce per-node embeddings (one per grid cell). We use
    patch tokens and broadcast each patch token to its pixels.
    """

    def __init__(
        self,
        *,
        xdim: int,
        ydim: int,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 10,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)

        # Allow non-divisible sizes by padding to the next multiple of patch_size.
        self.pad_h = (-self.ydim) % self.patch_size
        self.pad_w = (-self.xdim) % self.patch_size
        self.h_padded = self.ydim + self.pad_h
        self.w_padded = self.xdim + self.pad_w

        self.nh = self.h_padded // self.patch_size
        self.nw = self.w_padded // self.patch_size
        self.n_patches = self.nh * self.nw

        self.proj = nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, T, E, G, C] where G = xdim*ydim.

        Returns
        -------
        torch.Tensor
            Shape [B, E, G, D].
        """
        B, T, E, G, C = x.shape
        if G != self.xdim * self.ydim:
            raise ValueError(f"Expected grid={self.xdim*self.ydim} got {G}")
        if C != self.in_channels:
            raise ValueError(f"Expected channels={self.in_channels} got {C}")

        # collapse time into channels (simple, fast)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, E, T, G, C]
        x = x.reshape(B * E, T * C, self.ydim, self.xdim)  # [B*E, TC, H, W]

        if self.pad_h or self.pad_w:
            # Pad (left,right,top,bottom) for 2D spatial dims; keep channels intact.
            import torch.nn.functional as F

            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="replicate")

        # patchify
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B*E, TC, nh, nw, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B*E, nh, nw, TC, p, p]
        patches = patches.view(B * E, self.n_patches, (T * C) * p * p)

        tokens = self.proj(patches) + self.pos  # [B*E, n_patches, D]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # broadcast patch tokens back to pixels
        tokens_2d = tokens.view(B * E, self.nh, self.nw, self.embed_dim)
        tokens_up = tokens_2d.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)  # [B*E, Hpad, Wpad, D]
        # Trim padding back to original shape
        tokens_up = tokens_up[:, : self.ydim, : self.xdim, :]
        node_emb = tokens_up.contiguous().view(B, E, G, self.embed_dim)
        return node_emb


class PointMLPNodeEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        # Allow in_dim=0 in config and infer at first forward.
        d = int(in_dim)
        for _ in range(depth - 1):
            if d <= 0:
                layers += [nn.LazyLinear(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            else:
                layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E, G, C] -> collapse time into channels like ViT
        B, T, E, G, C = x.shape
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, E, G, T * C)
        out = self.net(x)
        return out


class NowcastPrecipUnionMeshModel(BaseGraphModel):
    """Union-mesh model with per-source QC featurization + per-source node encoders."""

    def _build_networks(self, model_config: DotDict) -> None:
        # ----- QC featurizers + node encoders -----
        self.qc = nn.ModuleDict()
        self.node_encoder = nn.ModuleDict()

        # model.model.sources is a dict keyed by dataset_name
        sources_cfg = model_config.model.get("sources", {})
        if not sources_cfg:
            LOGGER.warning("nowcast_precip: model.model.sources not set; defaulting to identity node encoders")

        for dataset_name in self._graph_data.keys():
            scfg = sources_cfg.get(dataset_name, {})
            qc_cfg = scfg.get("qc", None)
            if qc_cfg is not None:
                scheme = QCScheme(
                    num_bits=int(qc_cfg.get("num_bits")),
                    embed_dim=int(qc_cfg.get("embed_dim", 8)),
                    invalid_mask=int(qc_cfg.get("invalid_mask", 0)),
                    include_bits=bool(qc_cfg.get("include_bits", False)),
                    add_valid=bool(qc_cfg.get("add_valid", True)),
                )
                self.qc[dataset_name] = QCTorchFeaturizer(scheme)

            enc_kind = (scfg.get("node_encoder") or {}).get("kind", "identity")
            if enc_kind == "vit":
                enc_cfg = scfg.get("node_encoder")
                # in_channels here is *original vars* + qc extras per timestep
                # we will compute the true in_channels in forward (based on qc settings)
                self.node_encoder[dataset_name] = GridViTNodeEncoder(
                    xdim=int(enc_cfg["xdim"]),
                    ydim=int(enc_cfg["ydim"]),
                    in_channels=int(enc_cfg["in_channels"]),
                    embed_dim=int(enc_cfg.get("embed_dim", 128)),
                    patch_size=int(enc_cfg.get("patch_size", 10)),
                    depth=int(enc_cfg.get("depth", 4)),
                    num_heads=int(enc_cfg.get("num_heads", 8)),
                    mlp_ratio=float(enc_cfg.get("mlp_ratio", 4.0)),
                    dropout=float(enc_cfg.get("dropout", 0.0)),
                )
            elif enc_kind == "mlp":
                enc_cfg = scfg.get("node_encoder")
                self.node_encoder[dataset_name] = PointMLPNodeEncoder(
                    in_dim=int(enc_cfg["in_dim"]),
                    embed_dim=int(enc_cfg.get("embed_dim", 64)),
                    hidden_dim=int(enc_cfg.get("hidden_dim", 128)),
                    depth=int(enc_cfg.get("depth", 2)),
                    dropout=float(enc_cfg.get("dropout", 0.0)),
                )
            else:
                self.node_encoder[dataset_name] = nn.Identity()

        # ----- Per-dataset encoder: data -> hidden -----
        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[dataset_name][(self._graph_name_data, "to", self._graph_name_hidden)],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                dst_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )
            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,
                in_channels_src=self._latent_input_dim(dataset_name, model_config),
                in_channels_dst=self.node_attributes[dataset_name].attr_ndims[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        # ----- Shared processor: hidden -> hidden (from first dataset) -----
        first_dataset_name = next(iter(self._graph_data.keys()))
        processor_graph = self._graph_data[first_dataset_name][(self._graph_name_hidden, "to", self._graph_name_hidden)]
        processor_grid_size = self.node_attributes[first_dataset_name].num_nodes[self._graph_name_hidden]

        self.processor_graph_provider = create_graph_provider(
            graph=processor_graph,
            edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=processor_grid_size,
            dst_size=processor_grid_size,
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # ----- Single decoder: hidden -> mesh (shared, from first dataset graph) -----
        decoder_graph = self._graph_data[first_dataset_name][(self._graph_name_hidden, "to", "mesh")]
        mesh_size = self.node_attributes[first_dataset_name].num_nodes["mesh"]

        self.decoder_graph_provider = create_graph_provider(
            graph=decoder_graph,
            edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
            src_size=processor_grid_size,
            dst_size=mesh_size,
            trainable_size=model_config.model.decoder.get("trainable_size", 0),
        )

        out_channels = int(model_config.model.get("out_channels", 1))
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,
            in_channels_src=self.num_channels,
            in_channels_dst=self.node_attributes[first_dataset_name].attr_ndims["mesh"],
            hidden_dim=self.num_channels,
            out_channels_dst=out_channels,
            edge_dim=self.decoder_graph_provider.edge_dim,
        )

        # Slices to map mesh -> dataset outputs (optional; inferred if missing)
        self.mesh_slices = dict(model_config.model.get("mesh_slices", {}))

    def _latent_input_dim(self, dataset_name: str, model_config: DotDict) -> int:
        """Compute the feature dim fed into the graph encoder for a dataset."""
        # Start from raw input dim used by BaseGraphModel (time*vars for that dataset)
        base = int(self.input_dim[dataset_name])

        # If we use node_encoder (vit/mlp), we replace raw time*vars with encoder embed dim.
        scfg = model_config.model.get("sources", {}).get(dataset_name, {})
        enc_cfg = scfg.get("node_encoder", {})
        kind = enc_cfg.get("kind", "identity")
        if kind == "vit":
            return int(enc_cfg.get("embed_dim", 128)) + int(
                self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
            )
        if kind == "mlp":
            return int(enc_cfg.get("embed_dim", 64)) + int(
                self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
            )

        # Identity: keep base time*vars + node attrs
        return base + int(self.node_attributes[dataset_name].attr_ndims[self._graph_name_data])

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_shapes: dict | None,
        model_comm_group=None,
        dataset_name: str = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        assert dataset_name is not None

        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        x_skip = self.residual[dataset_name](x, grid_shard_shapes=grid_shard_shapes, model_comm_group=model_comm_group)

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data,
                0,
                shard_shapes_dim=grid_shard_shapes,
                model_comm_group=model_comm_group,
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # x comes as [B, T, E, G, V]
        if x.ndim != 5:
            raise ValueError(f"Expected x to have 5 dims [B,T,E,G,V], got {x.shape}")

        # Split out qc flags if present as last channel named in config.
        # In this prototype we assume qc flags are included in x vars by the dataset config.
        # If not present, qc featurizer is skipped.
        scfg = self.model_config.model.get("sources", {}).get(dataset_name, {})
        qc_var_index = scfg.get("qc", {}).get("qc_var_index", None)
        qc_features = None
        if dataset_name in self.qc and qc_var_index is not None:
            idx = int(qc_var_index)
            if idx < 0:
                idx = x.shape[-1] + idx
            qc_flags = x[..., idx]
            valid, emb, bits = self.qc[dataset_name](qc_flags)
            parts = []
            if self.qc[dataset_name].scheme.add_valid:
                parts.append(valid.unsqueeze(-1))
            parts.append(emb)
            if bits is not None:
                parts.append(bits)
            qc_features = torch.cat(parts, dim=-1)  # [B,T,E,G,Fqc]
            # remove qc channel from x vars
            x = torch.cat([x[..., :idx], x[..., idx + 1 :]], dim=-1)

        # Append qc features to each timestep as additional channels for node encoder
        if qc_features is not None:
            x = torch.cat([x, qc_features], dim=-1)

        # Node encoder may produce [B,E,G,D]
        node_enc = self.node_encoder.get(dataset_name, nn.Identity())
        if not isinstance(node_enc, nn.Identity):
            node_lat = node_enc(x)  # [B,E,G,D]
            x_data_latent = einops.rearrange(node_lat, "B E G D -> (B E G) D")
        else:
            x_data_latent = einops.rearrange(x, "B T E G V -> (B E G) (T V)")

        x_data_latent = torch.cat((x_data_latent, node_attributes_data), dim=-1)

        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )
        return x_data_latent, x_skip, shard_shapes_data

    def forward(
        self,
        x: Dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Dict[str, list] | None = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        dataset_names = list(x.keys())
        batch_sizes = [x[n].shape[0] for n in dataset_names]
        ensemble_sizes = [x[n].shape[2] for n in dataset_names]
        assert all(bs == batch_sizes[0] for bs in batch_sizes)
        assert all(es == ensemble_sizes[0] for es in ensemble_sizes)
        batch_size = batch_sizes[0]
        ensemble_size = ensemble_sizes[0]
        dtype = next(iter(x.values())).dtype

        # Encode each dataset into hidden space and sum
        hidden_latent = None
        for dataset_name in dataset_names:
            x_data_latent, _x_skip, shard_shapes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            enc_edges = self.encoder_graph_provider[dataset_name](shard_shapes_data=shard_shapes_data)
            h = self.encoder[dataset_name](x_data_latent, enc_edges)
            hidden_latent = h if hidden_latent is None else hidden_latent + h

        # Process on hidden graph
        proc_edges = self.processor_graph_provider(shard_shapes_data=None)
        hidden_latent = self.processor(hidden_latent, proc_edges)

        # Decode to union mesh
        dec_edges = self.decoder_graph_provider(shard_shapes_data=None)
        mesh_out = self.decoder(hidden_latent, dec_edges)
        mesh_out = einops.rearrange(
            mesh_out,
            "(B E G) V -> B E G V",
            B=batch_size,
            E=ensemble_size,
        ).to(dtype=dtype)

        # Infer slices if needed
        inferred_slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for dataset_name in dataset_names:
            n = int(self._graph_data[dataset_name][self._graph_name_data].num_nodes)
            inferred_slices[dataset_name] = (offset, offset + n)
            offset += n

        out: Dict[str, Tensor] = {}
        for dataset_name in dataset_names:
            a, b = self.mesh_slices.get(dataset_name, (0, 0))
            if (a, b) == (0, 0):
                a, b = inferred_slices[dataset_name]
            out[dataset_name] = mesh_out[:, :, a:b, :]
        return out
