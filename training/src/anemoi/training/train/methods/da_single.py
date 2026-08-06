# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Deterministic training method with data-assimilation cycling and correctors.

Pairs with :class:`anemoi.training.tasks.da_forecaster.DAForecaster`. Extends
:class:`SingleTraining` so that:

- DA-cycle steps (``is_da``) are weighted by ``training.da_loss_weight`` (and
  skipped entirely when that weight is zero), while forecast steps use weight 1.
- The total loss is averaged over the forecast steps only.
- A training-only per-instrument corrector network is applied to predictions
  before the loss (never for state advancement).
"""

import logging

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.training.train.methods.corrector import InstrumentCorrectors
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class DASingleTraining(SingleTraining):
    """Deterministic training method for the DA forecaster with correctors."""

    def __init__(self, *, graph_data: HeteroData, **kwargs) -> None:
        super().__init__(graph_data=graph_data, **kwargs)
        self.corrector = torch.nn.ModuleDict()
        self._init_correctors(graph_data)
        # Cache of DATA_FULL -> model-output column indices, keyed by dataset name,
        # used to slice targets down to output variables before the loss checkpoint.
        self._model_output_idx_cache: dict[str, torch.Tensor] = {}

    def _init_correctors(self, graph_data: HeteroData) -> None:
        """Build per-instrument corrector networks for each dataset with corrector variables.

        Reads the ``training.corrector`` config section which defines instrument
        groups (each with their own corrector variables and output channels) and
        the corrector backend type (``"mlp"`` or ``"gnn"``). For ``type: gnn``,
        data-to-data graph edges are extracted from the training graph.

        Parameters
        ----------
        graph_data : HeteroData
            The training graph, used to resolve data-to-data edges for the GNN backend.
        """
        corrector_config = getattr(self.config.training, "corrector", None)
        if corrector_config is None:
            return

        hidden_dim = getattr(corrector_config, "hidden_dim", 64)
        corrector_type = getattr(corrector_config, "type", "mlp")
        num_gnn_layers = getattr(corrector_config, "num_gnn_layers", 1)
        edge_attribute_names = list(getattr(corrector_config, "edge_attributes", ["edge_length", "edge_dirs"]))
        raw_groups = getattr(corrector_config, "instrument_groups", None)
        if raw_groups is None:
            return

        instrument_groups = {}
        for group_name in raw_groups:
            group = raw_groups[group_name]
            corrector_variables = list(getattr(group, "corrector_variables", []))
            channels = getattr(group, "channels", None)
            instrument_groups[group_name] = {
                "corrector_variables": corrector_variables,
                "channels": list(channels) if channels is not None else None,
            }

        fused = uses_fused_dataset_graph(graph_data, self.dataset_names)
        for dataset_name in self.target_dataset_names:
            corrector_indices = self.data_indices[dataset_name].data.input.corrector
            if len(corrector_indices) == 0:
                continue

            name_to_index = self.data_indices[dataset_name].data.input.name_to_index
            index_to_name = {v: k for k, v in name_to_index.items()}
            all_corrector_names = [index_to_name[int(idx)] for idx in corrector_indices]

            output_name_to_index = self.data_indices[dataset_name].model.output.name_to_index

            edge_index = None
            edge_attr = None
            num_nodes = None
            if corrector_type == "gnn":
                node_name = dataset_name if fused else DEFAULT_DATASET_NAME
                data_data_key = (node_name, "to", node_name)
                if data_data_key not in graph_data.edge_types:
                    msg = (
                        f"Corrector type 'gnn' requires data-to-data edges in the graph "
                        f"(key {data_data_key}), but they were not found for dataset '{dataset_name}'. "
                        f"Add a data->data edge builder (e.g. KNNEdges) to your graph config."
                    )
                    raise ValueError(msg)
                data_data_graph = graph_data[data_data_key]
                edge_index = data_data_graph.edge_index
                edge_attr = torch.cat([data_data_graph[attr] for attr in edge_attribute_names], dim=-1)
                num_nodes = graph_data[node_name].num_nodes
                LOGGER.info(
                    "Corrector GNN for '%s': %d nodes, %d edges, %d edge features, %d layers",
                    dataset_name,
                    num_nodes,
                    edge_index.shape[1],
                    edge_attr.shape[-1],
                    num_gnn_layers,
                )

            self.corrector[dataset_name] = InstrumentCorrectors(
                instrument_groups=instrument_groups,
                all_corrector_names=all_corrector_names,
                output_name_to_index=output_name_to_index,
                hidden_dim=hidden_dim,
                corrector_type=corrector_type,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                num_gnn_layers=num_gnn_layers,
            )

    def _apply_corrector(
        self,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply the corrector networks to raw predictions for the loss only.

        Corrector variables are read from the DATA_FULL target ``y`` at the
        target time. Datasets without a corrector network pass through unchanged.

        Parameters
        ----------
        y_pred : dict[str, torch.Tensor]
            Raw model output per dataset (MODEL_OUTPUT layout).
        y : dict[str, torch.Tensor]
            Targets per dataset (DATA_FULL layout) at the current step.

        Returns
        -------
        dict[str, torch.Tensor]
            Corrector-adjusted predictions per dataset.
        """
        if len(self.corrector) == 0:
            return y_pred

        y_for_loss = {}
        for dataset_name, pred in y_pred.items():
            # nn.ModuleDict has no .get(), so SIM401's suggestion does not apply here.
            corrector = self.corrector[dataset_name] if dataset_name in self.corrector else None  # noqa: SIM401
            if corrector is None:
                y_for_loss[dataset_name] = pred
                continue
            corrector_idx = self.data_indices[dataset_name].data.input.corrector.to(device=pred.device)
            corrector_vars = y[dataset_name].index_select(-1, corrector_idx)
            y_for_loss[dataset_name] = corrector(
                pred,
                corrector_vars,
                model_comm_group=self.model_comm_group,
                grid_shard_sizes=self.grid_shard_sizes[dataset_name],
            )
        return y_for_loss

    def _model_output_idx(self, dataset_name: str, device: torch.device) -> torch.Tensor:
        """Return (and cache) the DATA_FULL column indices of the model-output variables.

        Selecting these columns from a DATA_FULL target yields the model-output
        variables in model-output order, so the sliced tensor pairs with
        ``target_layout=IndexSpace.MODEL_OUTPUT`` (identity mapping) in the loss.
        """
        idx = self._model_output_idx_cache.get(dataset_name)
        if idx is None or idx.device != device:
            positions = self.data_indices[dataset_name].model_output_positions_in_data_full
            idx = torch.as_tensor(positions, dtype=torch.long, device=device)
            self._model_output_idx_cache[dataset_name] = idx
        return idx

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Training / validation step with DA cycling and corrector-adjusted loss."""
        loss = torch.zeros(1, dtype=next(iter(batch.values())).dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)

        task_steps = self.task.steps("validation" if validation_mode else "training")
        n_forecast = max(1, sum(1 for step in task_steps if not step.get("is_da", False)))

        for task_kwargs in task_steps:
            is_da = task_kwargs.get("is_da", False)
            rollout_step = task_kwargs["rollout_step"]
            weight = self.task.da_loss_weight if is_da else 1.0

            decoder_forcings = self.task.build_decoder_forcings(batch, data_indices=self.data_indices, **task_kwargs)
            forward_kwargs = {} if decoder_forcings is None else {"decoder_forcings": decoder_forcings}
            y_pred = self(x, **forward_kwargs)
            y = self.task.get_targets(batch, **task_kwargs)

            if weight > 0:
                # Corrector needs the full DATA_FULL target (reads input.corrector columns).
                y_for_loss = self._apply_corrector(y_pred, y)
                # Slice the target to model-output variables BEFORE the checkpoint so the
                # activation checkpoint saves only the reduced target rather than every
                # DATA_FULL channel (forcings/obs/corrector) for each DA + rollout step.
                y_target = {name: t.index_select(-1, self._model_output_idx(name, t.device)) for name, t in y.items()}
                loss_next, metrics_next, _ = checkpoint(
                    self.compute_loss_metrics,
                    y_for_loss,
                    y_target,
                    rollout_step=rollout_step,
                    validation_mode=validation_mode,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.MODEL_OUTPUT,
                    use_reentrant=False,
                )
                if loss_next is not None:
                    loss = loss + weight * loss_next
                metrics.update(metrics_next)

            # Advance with the RAW prediction, never the corrected tensor.
            x = self.task.advance_input(
                x,
                y_pred,
                batch,
                **task_kwargs,
                data_indices=self.data_indices,
                output_mask=self.output_mask,
                grid_shard_slice=self.grid_shard_slice,
            )

            y_preds.append(y_pred)

        loss *= 1.0 / n_forecast
        return TrainingStepOutput(loss=loss, metrics=metrics, predictions=y_preds)
