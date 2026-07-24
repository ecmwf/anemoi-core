# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-time corrector networks applied to model predictions for the loss.

Correctors are training-only modules held on the Lightning module (not part of
the inference model). They read observation-metadata "corrector" variables and
predict an additive correction to the raw model output before the loss.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch import nn

from anemoi.models.distributed.graph import sync_tensor

LOGGER = logging.getLogger(__name__)


class CorrectorMLP(nn.Module):
    """Small MLP that computes an additive correction for a group of output variables.

    Takes the concatenation of target output variables and corrector variables
    and predicts an additive correction. Output layer is zero-initialised so
    training starts with no correction.
    """

    def __init__(self, n_target: int, n_corrector: int, hidden_dim: int, n_out: int | None = None) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_target + n_corrector, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, n_out if n_out is not None else n_target)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, y_subset: torch.Tensor, corrector_vars: torch.Tensor) -> torch.Tensor:
        """Compute additive correction.

        Parameters
        ----------
        y_subset : torch.Tensor
            Target output variables for this group, shape (..., n_target).
        corrector_vars : torch.Tensor
            Corrector variables for this group, shape (..., n_corrector).

        Returns
        -------
        torch.Tensor
            Additive correction, shape (..., n_target).
        """
        x = torch.cat([y_subset, corrector_vars], dim=-1)
        return self.out(self.act(self.hidden(x)))


class CorrectorGNN(nn.Module):
    """Graph-based corrector with spatial neighbor aggregation.

    Uses message passing on data-to-data edges to gather spatial context
    before computing an additive correction. This helps channels where the
    satellite viewing geometry causes the radiation path to cross multiple
    grid cells (e.g. high-peaking limb-sounding channels at large VZA).

    Architecture per layer:
        1. Concatenate ``[x_src, x_dst, edge_attr]`` per edge
        2. MLP to produce edge messages
        3. Scatter-add messages to destination nodes
        4. Combine node state with aggregated messages via MLP + residual

    The final output layer is zero-initialised so training starts with no
    correction, identical to :class:`CorrectorMLP`.

    Parameters
    ----------
    n_target : int
        Number of output channels to correct.
    n_corrector : int
        Number of corrector input variables.
    hidden_dim : int
        Hidden dimension for all internal layers.
    edge_index : torch.Tensor
        Data-to-data edge indices, shape ``(2, num_edges)``.
    edge_attr : torch.Tensor
        Edge features (e.g. edge_length, edge_dirs), shape ``(num_edges, edge_dim)``.
    num_nodes : int
        Number of data-grid nodes (for scatter operations).
    num_layers : int
        Number of message-passing rounds. More layers extend the
        spatial receptive field (1 layer ~ 1-hop neighbors).
    n_out : int, optional
        Width of the output head. Defaults to ``n_target``.
    """

    def __init__(
        self,
        n_target: int,
        n_corrector: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        num_layers: int = 1,
        n_out: int | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Project (y_pred, corrector_vars) -> hidden_dim per node
        self.input_proj = nn.Linear(n_target + n_corrector, hidden_dim)
        self.act = nn.GELU()

        # Message-passing layers
        edge_dim = edge_attr.shape[-1]
        self.msg_mlps = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            )
            self.combine_mlps.append(nn.Linear(2 * hidden_dim, hidden_dim))

        # Output head (zero-initialized for no-correction start)
        self.out = nn.Linear(hidden_dim, n_out if n_out is not None else n_target)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Store graph structure as persistent buffers
        self.register_buffer("edge_index", edge_index, persistent=True)
        self.register_buffer("edge_attr", edge_attr, persistent=True)

        # Cache of per-shard edge partitions, keyed by (node_offset, g_local, device)
        self._local_edge_cache: dict = {}

    def forward(
        self,
        y_subset: torch.Tensor,
        corrector_vars: torch.Tensor,
        *,
        model_comm_group: dist.ProcessGroup | None = None,
        grid_shard_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute spatially-aware additive correction.

        When the grid is sharded across a model communication group, pass
        ``model_comm_group`` and ``grid_shard_sizes`` (per-rank grid sizes).
        Edges are partitioned by destination shard (same convention as the
        mappers' ``shard_strategy: edges``): each rank processes only the edges
        whose destination it owns, reading remote source-node features via
        ``sync_tensor`` (gather in forward, all-reduce + split in backward) so
        gradient contributions across shards are accumulated correctly.

        Parameters
        ----------
        y_subset : torch.Tensor
            Target output variables for this group, shape ``(..., G, n_target)``
            where G is the number of (local) grid points.
        corrector_vars : torch.Tensor
            Corrector variables for this group, shape ``(..., G, n_corrector)``.
        model_comm_group : ProcessGroup, optional
            Model communication group when the grid dimension is sharded.
        grid_shard_sizes : list[int], optional
            Grid points per rank, required when ``model_comm_group`` has more than one rank.

        Returns
        -------
        torch.Tensor
            Additive correction, shape ``(..., G, n_target)``.
        """
        leading_shape = y_subset.shape[:-2]  # e.g. (batch,) or (batch, time, ens)
        g_local = y_subset.shape[-2]

        x = torch.cat([y_subset, corrector_vars], dim=-1)
        x = self.act(self.input_proj(x))  # (..., g_local, H)

        # Flatten leading dims for scatter: (B, g_local, H)
        x_flat = x.reshape(-1, g_local, x.shape[-1])
        b, _, h = x_flat.shape

        sharded = model_comm_group is not None and dist.get_world_size(model_comm_group) > 1
        if sharded:
            assert grid_shard_sizes is not None, (
                "CorrectorGNN: grid_shard_sizes is required when the grid is sharded "
                "across a model communication group."
            )
            rank = dist.get_rank(model_comm_group)
            node_offset = int(sum(grid_shard_sizes[:rank]))
            g = int(sum(grid_shard_sizes))
            assert g_local == int(grid_shard_sizes[rank]), (
                f"CorrectorGNN: local grid size {g_local} does not match "
                f"grid_shard_sizes[{rank}]={grid_shard_sizes[rank]}."
            )
        else:
            node_offset = 0
            g = g_local

        assert g == self.num_nodes, (
            f"CorrectorGNN: got {g} grid points but the data-to-data graph has "
            f"{self.num_nodes} nodes. If the grid is sharded, pass model_comm_group "
            f"and grid_shard_sizes."
        )

        # Edges owned by this rank: destination inside the local shard.
        # src indexes the full grid, dst is shifted to local coordinates.
        src, dst, ea = self._local_edges(node_offset, g_local)
        e_local = src.shape[0]

        # Expand for batch: offset src by full-grid size, dst by local size
        batch_idx = torch.arange(b, device=x_flat.device)
        src_b = (src.unsqueeze(0) + (batch_idx * g).view(b, 1)).reshape(-1)  # (B*e_local,)
        dst_b = (dst.unsqueeze(0) + (batch_idx * g_local).view(b, 1)).reshape(-1)  # (B*e_local,)
        ea_b = ea.unsqueeze(0).expand(b, -1, -1).reshape(b * e_local, -1)  # (B*e_local, edge_dim)

        # Flatten local nodes: (B*g_local, H)
        x_nodes = x_flat.reshape(b * g_local, h)

        for layer_idx in range(self.num_layers):
            if sharded:
                # Full-grid source features for this layer's messages
                x_src_nodes = sync_tensor(
                    x_nodes.reshape(b, g_local, h),
                    1,
                    grid_shard_sizes,
                    model_comm_group,
                ).reshape(b * g, h)
            else:
                x_src_nodes = x_nodes

            # Build edge messages: [x_src, x_dst, edge_attr]
            msgs = torch.cat([x_src_nodes[src_b], x_nodes[dst_b], ea_b], dim=-1)
            msgs = self.msg_mlps[layer_idx](msgs)  # (B*e_local, H)

            # Scatter-add to (local) destination nodes
            agg = torch.zeros_like(x_nodes)
            agg.scatter_add_(0, dst_b.unsqueeze(-1).expand_as(msgs), msgs)

            # Combine self + aggregated neighbors with residual
            x_nodes = x_nodes + self.act(self.combine_mlps[layer_idx](torch.cat([x_nodes, agg], dim=-1)))

        # Output correction
        correction = self.out(x_nodes)
        return correction.reshape(*leading_shape, g_local, -1)

    def _local_edges(self, node_offset: int, g_local: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return edges whose destination lies in ``[node_offset, node_offset + g_local)``.

        Parameters
        ----------
        node_offset : int
            Offset of this rank's shard in the full grid.
        g_local : int
            Number of grid points owned by this rank.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(src, dst_local, edge_attr)`` where ``src`` indexes the full grid
            and ``dst_local`` is shifted into local shard coordinates. The
            unsharded case (offset 0, full grid) returns the complete edge set.
            Results are cached per shard layout and device.
        """
        device = self.edge_index.device
        key = (node_offset, g_local, str(device))
        cached = self._local_edge_cache.get(key)
        if cached is None:
            dst = self.edge_index[1]
            mask = (dst >= node_offset) & (dst < node_offset + g_local)
            cached = (
                self.edge_index[0, mask],
                dst[mask] - node_offset,
                self.edge_attr[mask],
            )
            self._local_edge_cache[key] = cached
        return cached


class InstrumentCorrectorMLPs(nn.Module):
    """Per-instrument corrector networks.

    Holds a separate corrector network per instrument group. Each group's
    network receives only that group's corrector variables and corrects only
    that group's output channels.

    The corrector backend is selected via ``corrector_type``:

    - ``"mlp"`` (default): pointwise :class:`CorrectorMLP`.
    - ``"gnn"``: spatially-aware :class:`CorrectorGNN` using data-to-data
      graph edges for message passing, useful for channels affected by
      slant-path viewing geometry.

    Parameters
    ----------
    instrument_groups : dict[str, dict]
        Mapping of group_name to group config. Each group config has:
        - "corrector_variables": list[str] — corrector variable names for this group
        - "channels": list[str] | None — explicit output channels, or None for
          prefix matching (group_name used as prefix against output variable names)
    all_corrector_names : list[str]
        All corrector variable names (from data_indices), in tensor order.
    output_name_to_index : dict[str, int]
        Model output variable name to tensor index mapping.
    hidden_dim : int
        Hidden layer dimension for all group networks.
    corrector_type : str
        Backend type: ``"mlp"`` or ``"gnn"``.
    edge_index : torch.Tensor | None
        Data-to-data edge indices for GNN backend, shape ``(2, num_edges)``.
    edge_attr : torch.Tensor | None
        Data-to-data edge features for GNN backend, shape ``(num_edges, edge_dim)``.
    num_nodes : int | None
        Number of data-grid nodes for GNN backend.
    num_gnn_layers : int
        Number of message-passing rounds for GNN backend.
    """

    def __init__(
        self,
        instrument_groups: dict[str, dict],
        all_corrector_names: list[str],
        output_name_to_index: dict[str, int],
        hidden_dim: int = 64,
        corrector_type: str = "mlp",
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        num_nodes: int | None = None,
        num_gnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.mlps = nn.ModuleDict()
        self.corrector_type = corrector_type

        if corrector_type == "gnn" and (edge_index is None or edge_attr is None or num_nodes is None):
            msg = "CorrectorGNN requires edge_index, edge_attr, and num_nodes from data-to-data graph edges."
            raise ValueError(msg)

        # Map corrector variable names to their position in the corrector tensor
        corrector_name_to_pos = {name: i for i, name in enumerate(all_corrector_names)}

        for group_name, group_cfg in instrument_groups.items():
            group_corrector_vars = group_cfg["corrector_variables"]

            # Find positions of this group's corrector vars in the full corrector tensor
            corrector_positions = []
            for v in group_corrector_vars:
                if v not in corrector_name_to_pos:
                    LOGGER.warning(
                        "Corrector variable '%s' for group '%s' not found in data indices, skipping",
                        v,
                        group_name,
                    )
                    continue
                corrector_positions.append(corrector_name_to_pos[v])

            if not corrector_positions:
                LOGGER.warning("No valid corrector variables for group '%s', skipping", group_name)
                continue

            # Determine target output channels for this group
            explicit_channels = group_cfg.get("channels")
            if explicit_channels is not None:
                target_indices = [output_name_to_index[ch] for ch in explicit_channels if ch in output_name_to_index]
            else:
                # Prefix matching: group_name is the prefix
                prefix = group_name + "_"
                target_indices = sorted(idx for name, idx in output_name_to_index.items() if name.startswith(prefix))

            if not target_indices:
                LOGGER.warning(
                    "No output channels matched for group '%s' (prefix='%s_'), skipping",
                    group_name,
                    group_name,
                )
                continue

            self.register_buffer(
                f"_corrector_idx_{group_name}",
                torch.tensor(corrector_positions, dtype=torch.long),
            )
            self.register_buffer(
                f"_target_idx_{group_name}",
                torch.tensor(sorted(target_indices), dtype=torch.long),
            )

            if corrector_type == "gnn":
                self.mlps[group_name] = CorrectorGNN(
                    n_target=len(target_indices),
                    n_corrector=len(corrector_positions),
                    hidden_dim=hidden_dim,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_nodes,
                    num_layers=num_gnn_layers,
                )
            else:
                self.mlps[group_name] = CorrectorMLP(
                    n_target=len(target_indices),
                    n_corrector=len(corrector_positions),
                    hidden_dim=hidden_dim,
                )

            LOGGER.info(
                "Corrector group '%s' (type=%s): %d corrector vars -> %d output channels (hidden=%d%s)",
                group_name,
                corrector_type,
                len(corrector_positions),
                len(target_indices),
                hidden_dim,
                f", gnn_layers={num_gnn_layers}" if corrector_type == "gnn" else "",
            )

    def forward(
        self,
        y_pred: torch.Tensor,
        corrector_vars: torch.Tensor,
        *,
        model_comm_group: dist.ProcessGroup | None = None,
        grid_shard_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Apply per-instrument corrections.

        Parameters
        ----------
        y_pred : torch.Tensor
            Raw model output, shape (..., n_output).
        corrector_vars : torch.Tensor
            All corrector variables, shape (..., n_corrector).
        model_comm_group : ProcessGroup, optional
            Model communication group when the grid dimension is sharded.
            Only used by the GNN backend (pointwise MLPs are shard-safe).
        grid_shard_sizes : list[int], optional
            Grid points per rank, required by the GNN backend when sharded.

        Returns
        -------
        torch.Tensor
            Corrected output, shape (..., n_output).
        """
        y_out = y_pred.clone()
        for group_name, mlp in self.mlps.items():
            corrector_idx = getattr(self, f"_corrector_idx_{group_name}")
            target_idx = getattr(self, f"_target_idx_{group_name}")

            group_corrector = corrector_vars[..., corrector_idx]
            y_subset = y_out[..., target_idx]
            if isinstance(mlp, CorrectorGNN):
                correction = mlp(
                    y_subset,
                    group_corrector,
                    model_comm_group=model_comm_group,
                    grid_shard_sizes=grid_shard_sizes,
                )
            else:
                correction = mlp(y_subset, group_corrector)
            y_out[..., target_idx] = y_subset + correction
        return y_out
