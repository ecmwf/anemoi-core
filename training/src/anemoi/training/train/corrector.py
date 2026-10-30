# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-time corrector networks applied to model predictions for the loss.

Correctors are training-only modules held on the Lightning module (not part of the inference
model), configured by ``training.corrector`` and usable with any training method. They read the
observation-metadata ``corrector`` variables of a dataset at the target time and predict an
additive correction to the raw model output before the loss; the input state always advances
with the raw prediction.
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

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.categorical import build_categorical_embeddings
from anemoi.models.layers.graph_provider import create_graph_provider

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data import Batch
    from anemoi.models.data.sources import Source
    from anemoi.models.data_indices.collection import IndexCollection
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

    Corrector variables listed in ``categorical_specs`` (``{variable: {codes, ...}}``,
    normally ``model.categorical_embeddings[dataset]``) are embedded with tables owned
    by the group's corrector, after the group's continuous variables.
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
        categorical_specs: dict[str, dict] | None = None,
        categorical_embedding_dim: int | None = None,
        categorical_unknown_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.correctors = nn.ModuleDict()
        self.categorical = nn.ModuleDict()
        categorical_specs = categorical_specs or {}
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

            corrector_positions, categorical_positions = self._corrector_positions(
                group_name,
                group_corrector_vars,
                corrector_name_to_pos,
                categorical_specs,
            )
            if not corrector_positions and not categorical_positions:
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

            n_corrector = len(corrector_positions)
            if categorical_positions:
                self.categorical[group_name] = build_categorical_embeddings(
                    {v: categorical_specs[v] for v in categorical_positions},
                    embedding_dim=categorical_embedding_dim,
                    unknown_prob=categorical_unknown_prob,
                )
                self.register_buffer(
                    f"_categorical_idx_{group_name}",
                    torch.tensor(list(categorical_positions.values()), dtype=torch.long),
                )
                n_corrector += sum(e.embedding_dim for e in self.categorical[group_name].values())

            if corrector_type == "processor":
                processor = instantiate(
                    processor_config,
                    _recursive_=False,
                    num_channels=hidden_dim,
                    edge_dim=graph_provider.edge_dim,
                )
                self.correctors[group_name] = ProcessorCorrector(
                    n_target=len(target_indices),
                    n_corrector=n_corrector,
                    processor=processor,
                )
            else:
                self.correctors[group_name] = CorrectorMLP(
                    n_target=len(target_indices),
                    n_corrector=n_corrector,
                    hidden_dim=hidden_dim,
                )

            LOGGER.info(
                "Corrector group '%s' (type=%s): %d corrector vars (%d embedded) -> %d output channels (hidden=%d)",
                group_name,
                corrector_type,
                len(corrector_positions) + len(categorical_positions),
                len(categorical_positions),
                len(target_indices),
                hidden_dim,
            )

    @staticmethod
    def _corrector_positions(
        group_name: str,
        group_corrector_vars: list[str],
        corrector_name_to_pos: dict[str, int],
        categorical_specs: dict[str, dict],
    ) -> tuple[list[int], dict[str, int]]:
        """Positions of a group's corrector vars in the full corrector tensor.

        Returns the continuous positions and a ``{name: position}`` mapping of the
        categorical (embedded) ones, skipping variables absent from the data indices.
        """
        corrector_positions = []
        categorical_positions = {}
        for v in group_corrector_vars:
            if v not in corrector_name_to_pos:
                LOGGER.warning(
                    "Corrector variable '%s' for group '%s' not found in data indices, skipping",
                    v,
                    group_name,
                )
                continue
            if v in categorical_specs:
                categorical_positions[v] = corrector_name_to_pos[v]
            else:
                corrector_positions.append(corrector_name_to_pos[v])
        return corrector_positions, categorical_positions

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
            if group_name in self.categorical:
                group_corrector = self._append_embeddings(group_name, group_corrector, corrector_vars)
            y_subset = y_out[..., target_idx]
            correction = corrector(y_subset, group_corrector, **graph_kwargs)
            y_out[..., target_idx] = y_subset + correction
        return y_out

    def _append_embeddings(
        self,
        group_name: str,
        group_corrector: torch.Tensor,
        corrector_vars: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the group's categorical embeddings after its continuous variables."""
        categorical_idx = getattr(self, f"_categorical_idx_{group_name}")
        parts = [group_corrector]
        for i, embedding in enumerate(self.categorical[group_name].values()):
            parts.append(embedding(corrector_vars[..., categorical_idx[i]]).to(dtype=group_corrector.dtype))
        return torch.cat(parts, dim=-1)

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
                batch_size, time_size, ensemble_size, _, _ = y_pred.shape
                msg = (
                    "Processor correctors require exactly one graph instance when model sharding is enabled, "
                    f"but got batch={batch_size} x time={time_size} x ensemble={ensemble_size} = {graph_batch_size}. "
                    "Reduce training.batch_size, task.multistep_output or training.ensemble_size_per_device to 1 "
                    "(ensemble members can instead come from system.hardware.num_gpus_per_ensemble)."
                )
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


def build_correctors(
    corrector_config: DictConfig | None,
    data_indices: dict[str, IndexCollection],
    dataset_names: list[str],
    graph_data: HeteroData,
    categorical_config: DictConfig | None = None,
) -> nn.ModuleDict:
    """Build the per-instrument correctors of every dataset with ``corrector`` variables.

    Parameters
    ----------
    corrector_config : DictConfig | None
        The ``training.corrector`` section: ``instrument_groups`` (each with its corrector
        variables and output channels), the backend ``type`` (``"mlp"`` or ``"processor"``),
        ``hidden_dim``, the ``processor`` configuration and categorical-embedding overrides.
        ``None`` builds no corrector.
    data_indices : dict[str, IndexCollection]
        Data indices per dataset.
    dataset_names : list[str]
        Datasets that get a corrector when they have ``corrector`` variables (the loss datasets).
    graph_data : HeteroData
        The model graph; processor correctors take its data-to-data edges.
    categorical_config : DictConfig | None
        ``model.categorical_embeddings``: the corrector embeds the categorical metadata with its
        own tables over the same vocabularies.

    Returns
    -------
    nn.ModuleDict
        One :class:`InstrumentCorrectors` per corrected dataset; empty without correctors.
    """
    correctors = nn.ModuleDict()
    if corrector_config is None or corrector_config.get("instrument_groups") is None:
        return correctors

    corrector_type = corrector_config.get("type", "mlp")
    processor_config = corrector_config.get("processor")
    instrument_groups = {
        name: {
            "corrector_variables": list(group.get("corrector_variables", [])),
            "channels": None if group.get("channels") is None else list(group.get("channels")),
        }
        for name, group in corrector_config.instrument_groups.items()
    }
    categorical_config = categorical_config or {}

    fused = uses_fused_dataset_graph(graph_data, list(data_indices))
    for dataset_name in dataset_names:
        corrector_indices = data_indices[dataset_name].data.input.corrector
        if len(corrector_indices) == 0:
            continue
        index_to_name = {v: k for k, v in data_indices[dataset_name].data.input.name_to_index.items()}

        graph_provider = None
        if corrector_type == "processor":
            node_name = dataset_name if fused else DEFAULT_DATASET_NAME
            data_data_key = (node_name, "to", node_name)
            if data_data_key not in graph_data.edge_types:
                msg = (
                    f"Corrector type 'processor' requires data-to-data edges in the graph (key {data_data_key}), "
                    f"but they were not found for dataset '{dataset_name}'. Add a data->data edge builder "
                    "(e.g. KNNEdges) to your graph config."
                )
                raise ValueError(msg)
            num_nodes = graph_data[node_name].num_nodes
            graph_provider = create_graph_provider(
                graph=graph_data[data_data_key],
                edge_attribute_names=processor_config.sub_graph_edge_attributes,
                src_size=num_nodes,
                dst_size=num_nodes,
                trainable_size=processor_config.get("trainable_size", 0),
            )

        correctors[dataset_name] = InstrumentCorrectors(
            instrument_groups=instrument_groups,
            all_corrector_names=[index_to_name[int(idx)] for idx in corrector_indices],
            output_name_to_index=data_indices[dataset_name].model.output.name_to_index,
            hidden_dim=corrector_config.get("hidden_dim", 64),
            corrector_type=corrector_type,
            processor_config=processor_config,
            graph_provider=graph_provider,
            categorical_specs=dict(categorical_config.get(dataset_name) or {}),
            categorical_embedding_dim=corrector_config.get("categorical_embedding_dim"),
            categorical_unknown_prob=corrector_config.get("categorical_unknown_prob"),
        )
    return correctors


def apply_correctors(
    correctors: nn.ModuleDict,
    y_pred: Batch,
    y: Batch,
    data_indices: dict[str, IndexCollection],
    model_comm_group: dist.ProcessGroup | None = None,
) -> Batch:
    """Apply the correctors to the raw predictions, for the loss only.

    Parameters
    ----------
    correctors : nn.ModuleDict
        :class:`InstrumentCorrectors` per corrected dataset.
    y_pred : Batch
        Raw predictions, in the model output variables.
    y : Batch
        Pre-processed targets, in the full data variables; the corrector variables are read from
        them at the target time. Missing metadata (NaN) reads as 0, i.e. "no observation here".
    data_indices : dict[str, IndexCollection]
        Data indices per dataset.
    model_comm_group : ProcessGroup, optional
        Model communication group, used by processor correctors when the grid is sharded.

    Returns
    -------
    Batch
        The predictions with the corrected datasets replaced.
    """
    if len(correctors) == 0:
        return y_pred

    corrected = y_pred
    for dataset_name, corrector in correctors.items():
        if dataset_name not in y_pred:
            continue
        prediction: Source = y_pred[dataset_name]
        target: Source = y[dataset_name]
        corrector_idx = data_indices[dataset_name].data.input.corrector

        def correct(
            pred: torch.Tensor,
            metadata: torch.Tensor,
            shard_sizes: list[int] | None = None,
            corrector: InstrumentCorrectors = corrector,
            corrector_idx: torch.Tensor = corrector_idx,
        ) -> torch.Tensor:
            metadata = metadata[..., corrector_idx.to(metadata.device)].nan_to_num(0.0).to(pred.dtype)
            # The target carries a single member while the prediction may carry several; the
            # observations are the same for all of them.
            if metadata.shape[:-1] != pred.shape[:-1]:
                metadata = metadata.expand(*pred.shape[:-1], -1)
            return corrector(pred, metadata, model_comm_group=model_comm_group, grid_shard_sizes=shard_sizes)

        if prediction.is_tabular:
            if corrector.corrector_type == "processor":
                msg = f"Processor correctors need a gridded dataset; '{dataset_name}' is tabular. Use type 'mlp'."
                raise ValueError(msg)
            data = [correct(p, t) for p, t in zip(prediction.data, target.data, strict=True)]
        else:
            data = correct(prediction.data, target.data, prediction.shard_sizes)
        corrected = corrected.replace(dataset_name, prediction.clone(data=data))
    return corrected
