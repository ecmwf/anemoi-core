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
from typing import TYPE_CHECKING

import einops
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch import nn
from torch_geometric.utils import is_undirected

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.utils import model_is_distributed

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from anemoi.models.layers.graph_provider import StaticGraphProvider
    from anemoi.models.layers.processor import BaseProcessor

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


class ProcessorCorrector(nn.Module):
    """Embed instrument predictions and metadata, then predict an additive correction.

    The processor receives graph tensors from the dataset's shared graph provider.
    Its layer kernels also build the input and output projections. The output head
    is zero-initialised so training starts with no correction.
    """

    def __init__(self, n_target: int, n_corrector: int, processor: BaseProcessor) -> None:
        super().__init__()
        self.processor = processor
        kernels = processor.layer_factory
        self.input_proj = kernels.Linear(n_target + n_corrector, processor.num_channels)
        self.act = kernels.Activation()
        self.out = kernels.Linear(processor.num_channels, n_target)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        y_subset: torch.Tensor,
        corrector_vars: torch.Tensor,
        *,
        graph_batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        model_comm_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Correct predictions in ``(batch, time, ensemble, grid, variables)`` layout.

        Each output time and ensemble member is an independent graph instance.
        ``graph_batch_size`` is their combined batch size used by the graph provider.
        """
        batch_size, time_size, ensemble_size, _, _ = y_subset.shape

        # Assemble node inputs. Output times are corrected independently.
        x = torch.cat([y_subset, corrector_vars], dim=-1)
        x = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        x = self.act(self.input_proj(x))

        # Processor
        x = self.processor(
            x,
            batch_size=graph_batch_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )

        # Assemble output corrections in the model's prediction layout.
        return einops.rearrange(
            self.out(x),
            "(batch time ensemble grid) vars -> batch time ensemble grid vars",
            batch=batch_size,
            time=time_size,
            ensemble=ensemble_size,
        )


class InstrumentCorrectors(nn.Module):
    """Per-instrument correctors with a shared data-to-data graph provider.

    ``corrector_type="mlp"`` selects pointwise MLPs. ``"processor"`` builds a
    separate processor from ``processor_config`` for each instrument, while
    ``graph_provider`` supplies the same edges and features to all instruments.
    Instrument groups select metadata via ``corrector_variables`` and outputs
    via explicit ``channels`` or the group's name as a channel prefix.
    """

    def __init__(
        self,
        instrument_groups: dict[str, dict],
        all_corrector_names: list[str],
        output_name_to_index: dict[str, int],
        hidden_dim: int = 64,
        corrector_type: str = "mlp",
        processor_config: DictConfig | None = None,
        graph_provider: StaticGraphProvider | None = None,
    ) -> None:
        super().__init__()
        self.correctors = nn.ModuleDict()
        self.corrector_type = corrector_type

        if corrector_type not in {"mlp", "processor"}:
            msg = f"Unknown corrector type: {corrector_type!r}."
            raise ValueError(msg)
        if corrector_type == "processor" and (processor_config is None or graph_provider is None):
            msg = "Processor correctors require processor_config and a data-to-data graph_provider."
            raise ValueError(msg)
        self.graph_provider = graph_provider
        self.num_nodes = int(graph_provider.edge_inc[1, 0]) if graph_provider is not None else None
        self._requires_symmetric_graph = (
            corrector_type == "processor"
            and processor_config._target_ == "anemoi.models.layers.processor.GraphTransformerProcessor"
            and processor_config.get("shard_strategy", "edges") == "edges"
        )
        self._graph_is_symmetric = (
            is_undirected(graph_provider.edge_index_base, num_nodes=self.num_nodes)
            if self._requires_symmetric_graph
            else True
        )

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

            if corrector_type == "processor":
                processor = instantiate(
                    processor_config,
                    _recursive_=False,
                    num_channels=hidden_dim,
                    edge_dim=graph_provider.edge_dim,
                )
                self.correctors[group_name] = ProcessorCorrector(
                    n_target=len(target_indices),
                    n_corrector=len(corrector_positions),
                    processor=processor,
                )
            else:
                self.correctors[group_name] = CorrectorMLP(
                    n_target=len(target_indices),
                    n_corrector=len(corrector_positions),
                    hidden_dim=hidden_dim,
                )

            LOGGER.info(
                "Corrector group '%s' (type=%s): %d corrector vars -> %d output channels (hidden=%d)",
                group_name,
                corrector_type,
                len(corrector_positions),
                len(target_indices),
                hidden_dim,
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
            Raw model output, shape (batch, time, ensemble, grid, n_output).
        corrector_vars : torch.Tensor
            Corrector metadata, shape (batch, time, ensemble, grid, n_corrector).
        model_comm_group : ProcessGroup, optional
            Model communication group when the grid dimension is sharded.
            Only used by processor correctors (pointwise MLPs are shard-safe).
        grid_shard_sizes : list[int], optional
            Grid points per rank, required by processor correctors when sharded.

        Returns
        -------
        torch.Tensor
            Corrected output, shape (batch, time, ensemble, grid, n_output).
        """
        graph_kwargs = {}
        if self.corrector_type == "processor" and self.correctors:
            batch_size, time_size, ensemble_size, _, _ = y_pred.shape
            graph_batch_size = batch_size * time_size * ensemble_size
            node_shard_sizes = self._node_shard_sizes(y_pred, graph_batch_size, model_comm_group, grid_shard_sizes)
            edge_attr, edge_index, edge_shard_sizes = self.graph_provider.get_edges(
                batch_size=graph_batch_size,
                model_comm_group=model_comm_group,
            )
            graph_kwargs = {
                "graph_batch_size": graph_batch_size,
                "shard_info": GraphShardInfo(nodes=node_shard_sizes, edges=edge_shard_sizes),
                "edge_attr": edge_attr,
                "edge_index": edge_index,
                "model_comm_group": model_comm_group,
            }

        y_out = y_pred.clone()
        for group_name, corrector in self.correctors.items():
            corrector_idx = getattr(self, f"_corrector_idx_{group_name}")
            target_idx = getattr(self, f"_target_idx_{group_name}")

            group_corrector = corrector_vars[..., corrector_idx]
            y_subset = y_out[..., target_idx]
            correction = corrector(y_subset, group_corrector, **graph_kwargs)
            y_out[..., target_idx] = y_subset + correction
        return y_out

    def _node_shard_sizes(
        self,
        y_pred: torch.Tensor,
        graph_batch_size: int,
        model_comm_group: dist.ProcessGroup | None,
        grid_shard_sizes: list[int] | None,
    ) -> list[int]:
        """Validate node layout against the graph provider's balanced partitions.

        Batch, time, and ensemble axes represent independent graph instances.
        Model sharding requires one instance per rank, as in the main model's GT processor.
        """
        local_nodes = y_pred.shape[-2]
        if model_is_distributed(model_comm_group):
            if not self._graph_is_symmetric:
                msg = (
                    "GT correctors with shard_strategy='edges' require bidirectional graph connectivity "
                    "when model sharding is enabled. Use shard_strategy='heads' for directed graphs."
                )
                raise ValueError(msg)
            if graph_batch_size != 1:
                msg = "Processor correctors require batch_size=1 (one graph instance) when model sharding is enabled."
                raise ValueError(msg)
            expected_sizes = get_balanced_partition_sizes(self.num_nodes, model_comm_group.size())
            if grid_shard_sizes != expected_sizes:
                msg = f"Processor corrector grid_shard_sizes must match graph provider partitions {expected_sizes}."
                raise ValueError(msg)
            expected_local_nodes = expected_sizes[model_comm_group.rank()]
        else:
            expected_sizes = [graph_batch_size * self.num_nodes]
            expected_local_nodes = self.num_nodes
        if local_nodes != expected_local_nodes:
            msg = f"Processor corrector expected {expected_local_nodes} local grid nodes, got {local_nodes}."
            raise ValueError(msg)
        return expected_sizes
