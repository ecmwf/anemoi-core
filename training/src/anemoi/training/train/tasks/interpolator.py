# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter

import torch
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        if len(config.training.target_forcing.data) >= 1:
            self.target_forcing_indices = itemgetter(*config.training.target_forcing.data)(
                data_indices.data.input.name_to_index,
            )
            if isinstance(self.target_forcing_indices, int):
                self.target_forcing_indices = [self.target_forcing_indices]
        else:
            self.target_forcing_indices = []

        self.use_time_fraction = config.training.target_forcing.time_fraction

        # ────────────────────────────────────────────────────────────────
        # Accumulation-handling configuration (optional)
        # ────────────────────────────────────────────────────────────────
        # Training YAML may define:
        # training:
        #   accumulated_vars:
        #     <varname>:
        #       input_accum_period : <int>
        #       output_accum_period: <int>
        #       energy_conservation: <bool>
        self.accumulated_vars_cfg = getattr(config.training, "accumulated_vars", {})
        self.accum_energy_indices: list[int] = []
        for vname, vcfg in self.accumulated_vars_cfg.items():
            if vcfg.get("energy_conservation", False) and vname in data_indices.data.output.name_to_index:
                # Check that the input accumulation period covers the full time period for the interpolated variables
                assert vcfg.get("input_accum_period", 0) < vcfg.get(
                    "output_accum_period",
                    0,
                ), "Output accumulation period must be smaller than input accumulation period"
                assert (vcfg.get("input_accum_period") % vcfg.get("output_accum_period", 0)) == (
                    config.training.explicit_times.input[-1] - config.training.explicit_times.input[0]
                ), "Input accumulation period must be a multiple of the output accumulation period"
                self.accum_energy_indices.append(data_indices.data.output.name_to_index[vname])

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.rollout = 1

    def _step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        del batch_idx

        # -----------------------------------------------------------------
        # Split input if a second “accum-targets” batch is supplied
        # -----------------------------------------------------------------
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            batch, batch_accum_targets = batch  # unpack
        else:
            batch_accum_targets = None

        total_loss = torch.zeros(1, dtype=batch.dtype, device=self.device)
        metrics: dict[str, torch.Tensor] = {}

        # caches
        y_preds_raw: list[torch.Tensor] = []  # logits-level predictions
        y_trues: list[torch.Tensor] = []  # ground-truth tensors (full dim)

        batch = self.model.pre_processors(batch)
        if batch_accum_targets is not None:
            # ensure identical preprocessing
            batch_accum_targets = self.model.pre_processors(batch_accum_targets)

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.update_scalers(callback=AvailableCallbacks.ON_TRAINING_START)
            self.is_first_step = False
        self.update_scalers(callback=AvailableCallbacks.ON_BATCH_START)

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, time, ens, latlon, nvar)

        num_tfi = len(self.target_forcing_indices)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            num_tfi + self.use_time_fraction,
            device=self.device,
            dtype=batch.dtype,
        )
        # ─────────────────────────────────────────────────────────
        # 1) FORWARD PASSES : fill caches, no loss yet
        # ─────────────────────────────────────────────────────────
        for interp_step in self.interp_times:
            # Build the “target-forcing” tensor for this interpolation step
            if num_tfi >= 1:
                target_forcing[..., :num_tfi] = batch[:, self.imap[interp_step], :, :, self.target_forcing_indices]
            if self.use_time_fraction:
                target_forcing[..., -1] = (interp_step - self.boundary_times[-2]) / (
                    self.boundary_times[-1] - self.boundary_times[-2]
                )

            y_pred = self(x_bound, target_forcing)
            # -------------------------------------------------------------
            # Build ground-truth tensor “y”
            # For energy-conserved vars, optionally overwrite with
            # high-frequency targets from `batch_accum_targets`
            # -------------------------------------------------------------
            y = batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full].clone()

            if batch_accum_targets is not None and self.accum_energy_indices:

                y[..., self.accum_energy_indices] = batch_accum_targets[
                    :,
                    self.imap[interp_step],
                    :,
                    :,
                    self.accum_energy_indices,
                ]

            # collect caches
            y_preds_raw.append(y_pred)
            y_trues.append(y)

        # ─────────────────────────────────────────────────────────
        # 2) POST-PROCESS LOGITS → hourly deltas (energy-conserving)
        # ─────────────────────────────────────────────────────────
        if self.accum_energy_indices:
            import torch.nn.functional as F

            # (T, …, V_acc)
            logits = torch.stack(
                [y_pred[..., self.accum_energy_indices] for y_pred in y_preds_raw],
                dim=0,
            )  # (T,B,E,G,V_acc)
            zeros = torch.zeros_like(logits[:1])
            weights = F.softmax(torch.cat([logits, zeros], dim=0), dim=0)[:-1]  # drop pinned slot

            # Total N-hour accumulation at RHS boundary
            a_total = batch[:, self.imap[self.boundary_times[-1]], :, :, self.accum_energy_indices]  # (B,E,G,V_acc)
            a_total = a_total.unsqueeze(0)  # (1,B,E,G,V_acc)

            deltas = weights * a_total  # (T,B,E,G,V_acc)

            # Replace logits with δAₖ in the cached predictions
            for k, y_pred in enumerate(y_preds_raw):
                y_preds_raw[k] = y_pred.clone()
                y_preds_raw[k][..., self.accum_energy_indices] = deltas[k]

        # ─────────────────────────────────────────────────────────
        # 3) LOSS / METRIC LOOP  (now using adjusted predictions)
        # ─────────────────────────────────────────────────────────
        for idx, (y_pred_adj, y_true) in enumerate(zip(y_preds_raw, y_trues)):
            interp_step = self.interp_times[idx]

            # In the *absence* of a high-resolution target batch we mask out
            # the accumulated variables so that they do not contribute to the loss.
            if self.accum_energy_indices and batch_accum_targets is None:
                y_pred_masked = y_pred_adj.clone()
                y_pred_masked[..., self.accum_energy_indices] = y_true[..., self.accum_energy_indices]
                y_pred_adj = y_pred_masked

            loss_step, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred_adj,
                y_true,
                interp_step - 1,
                training_mode=True,
                validation_mode=validation_mode,
                use_reentrant=False,
            )
            total_loss += loss_step
            metrics.update(metrics_next)

        total_loss *= 1.0 / len(self.interp_times)
        return total_loss, metrics, y_preds_raw

    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            target_forcing=target_forcing,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
