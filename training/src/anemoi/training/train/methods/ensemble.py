# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import contextlib
import logging
import os
import random
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapterWrapper
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from omegaconf import DictConfig
    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.training.train.training_task.base import BaseTask

LOGGER = logging.getLogger(__name__)

# Opt-in GPU memory log through a rollout, to find how many rollout steps fit. Set ANEMOI_LOG_ROLLOUT_MEMORY=K and
# every rank logs, for the first K batches of each (training or validation, rollout length), the memory after each
# rollout step's forward and loss and, for training, after the backward pass and at the end of the batch (after the
# optimiser step, on the batches that take one). Off (and free) when unset or 0, and a no-op without CUDA.
ROLLOUT_MEMORY_ENV = "ANEMOI_LOG_ROLLOUT_MEMORY"
_rollout_memory_batches_logged: dict[tuple[str, int], int] = {}


def _claim_rollout_memory_log(mode: str, n_steps: int) -> bool:
    """Whether to log this batch: the first K of each (mode, rollout length) in this process."""
    try:
        limit = int(os.environ.get(ROLLOUT_MEMORY_ENV, "0"))
    except ValueError:
        return False
    if limit <= 0 or not torch.cuda.is_available():
        return False
    logged = _rollout_memory_batches_logged.get((mode, n_steps), 0)
    if logged >= limit:
        return False
    _rollout_memory_batches_logged[(mode, n_steps)] = logged + 1
    return True


def _log_rollout_memory(rank: int, label: str, device: torch.device) -> None:
    """One line: the GPU's allocated, peak allocated (since the batch started) and reserved memory, in GiB.

    It reads ``device`` (the model's GPU), not the current CUDA device, so the numbers stay right if something has
    switched the current device.
    """
    gib = 2**30
    LOGGER.info(
        "rollout memory | rank %d | %s | allocated %.2f GiB | peak allocated %.2f GiB"
        " | reserved %.2f GiB | device %.2f GiB",
        rank,
        label,
        torch.cuda.memory_allocated(device) / gib,
        torch.cuda.max_memory_allocated(device) / gib,
        torch.cuda.memory_reserved(device) / gib,
        torch.cuda.get_device_properties(device).total_memory / gib,
    )


class EnsembleTraining(BaseTrainingModule):
    """Graph neural network forecaster for ensembles for PyTorch Lightning."""

    # Ensemble communication groups are wired up later by DDPEnsGroupStrategy. These
    # class-level single-device defaults mean the ensemble layout is always readable,
    # even before the strategy has run.
    ens_comm_group = None
    ens_comm_group_id = 0
    ens_comm_group_rank = 0
    ens_comm_num_groups = 1
    ens_comm_group_size = 1
    # Set from the run seed in __init__; None lets the input noise follow torch's seed.
    input_noise_seed: int | None = None
    # training.input_dataset_dropout, set in __init__ ({dataset: probability}); none by default.
    input_dataset_dropout: Mapping[str, float] = MappingProxyType({})
    # training.rollout_windows, set in __init__: (rollout steps per window, weight per window); None = one window.
    rollout_windows: tuple[int, tuple[float, ...]] | None = None
    # With rollout windows, the first optimiser steps check that every rank holds the same parameters.
    _parameter_sync_checks_left = 3

    def __init__(
        self,
        *,
        config: DictConfig,
        task: BaseTask,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        task : BaseTask
            Training task
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__(
            config=config,
            task=task,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        # num_gpus_per_ensemble >= 1 and num_gpus_per_ensemble >= num_gpus_per_model (as per the DDP strategy)
        self.model_comm_group_size = config.system.hardware.num_gpus_per_model
        num_gpus_per_model = config.system.hardware.num_gpus_per_model
        num_gpus_per_ensemble = config.system.hardware.num_gpus_per_ensemble

        assert num_gpus_per_ensemble % num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            f"{num_gpus_per_ensemble} mod {num_gpus_per_model} != 0.\
            If you would like to run in deterministic mode, please use aifs-train"
        )

        self.effective_lr = (
            config.system.hardware.num_nodes
            * config.system.hardware.num_gpus_per_node
            * config.training.optimization.lr
            / num_gpus_per_ensemble
        )
        LOGGER.info(
            "Base (config) learning rate: %e -- Effective learning rate: %e",
            config.training.optimization.lr,
            self.effective_lr,
        )

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = self.nens_per_device * num_gpus_per_ensemble // num_gpus_per_model
        LOGGER.info("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        # The input noise draws from its own generators; seeding them from the run seed keeps
        # one seed (ANEMOI_BASE_SEED) in control of all randomness, identical on every rank.
        self.input_noise_seed = derive_seed(get_base_seed(), SeedContext.INPUT_NOISE)

        # Input datasets left out of some training samples (dataset name -> probability). Only datasets the
        # model encodes at the first rollout step alone (first_step_only) qualify: without them in training,
        # the steps after the first, where the model never has them, would be inputs it has not seen.
        self.input_dataset_dropout = self._validated_input_dataset_dropout(
            config.training.get("input_dataset_dropout", None),
            getattr(self.model.model, "first_step_only_datasets", frozenset()),
        )
        if self.input_dataset_dropout:
            LOGGER.info("Input datasets left out of training samples with probability %s", self.input_dataset_dropout)

        # The rollout cut into windows, each backpropagated on its own (training.rollout_windows).
        self.rollout_windows = self._validated_rollout_windows(
            config.training.get("rollout_windows", None),
            config.training.accum_grad_batches,
        )
        if self.rollout_windows is not None:
            rollout = getattr(self.task, "rollout", None)
            if rollout is not None and rollout.epoch_increment > 0 and rollout.maximum > rollout.step:
                msg = "training.rollout_windows needs a fixed rollout (task.rollout: start = maximum)."
                raise ValueError(msg)
            weights = self._rollout_window_step_weights(self.task.num_steps)
            # Several backward passes per optimiser step: Lightning's manual optimisation (training_step below).
            self.automatic_optimization = False
            LOGGER.info(
                "Rollout windows: %d steps in windows of %d, gradients cut between windows; weight per step %s",
                self.task.num_steps,
                self.rollout_windows[0],
                [round(w, 5) for w in weights],
            )

        # lazy init ensemble group info, will be set by the DDPEnsGroupStrategy.
        # Defaults are the single-device values used by SingleDeviceStrategy,
        # which does not set up communication groups. A ``None`` process group
        # makes the ensemble gather a no-op (see gather_tensor).
        self.ens_comm_group = None
        self.ens_comm_group_id = 0
        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_size = 1

        self.ens_comm_subgroup = None
        self.ens_comm_subgroup_id = 0
        self.ens_comm_subgroup_rank = 0
        self.ens_comm_subgroup_num_groups = 1
        self.ens_comm_subgroup_size = 1

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup: ProcessGroup,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

    @property
    def ensemble_member_offset(self) -> int:
        """Global index of this rank's first ensemble member.

        Ranks inside an ensemble group are laid out in blocks of
        ``model_comm_group_size``; every rank of a model group holds the same
        members, so they must agree on the offset.
        """
        return (self.ens_comm_group_rank // self.model_comm_group_size) * self.nens_per_device

    @property
    def ensemble_members_total(self) -> int:
        """Number of ensemble members across the whole ensemble group."""
        return self.nens_per_device * (self.ens_comm_group_size // self.model_comm_group_size)

    @property
    def plot_adapter(self) -> EnsemblePlotAdapterWrapper:
        """Wrap the task's plot adapter with ensemble handling."""
        if not hasattr(self, "_ensemble_plot_adapter"):
            self._ensemble_plot_adapter = EnsemblePlotAdapterWrapper(self.task._plot_adapter)
        return self._ensemble_plot_adapter

    @staticmethod
    def _validated_input_dataset_dropout(dropout: dict | None, first_step_only: set[str] | frozenset[str]) -> dict:
        """training.input_dataset_dropout as {dataset: probability}, checked against the first_step_only datasets."""
        dropout = {name: float(probability) for name, probability in (dropout or {}).items()}
        for name, probability in dropout.items():
            if name not in first_step_only:
                msg = (
                    f"training.input_dataset_dropout: '{name}' is not a first_step_only input dataset of the "
                    f"model (those are {sorted(first_step_only)})."
                )
                raise ValueError(msg)
            if not 0.0 <= probability <= 1.0:
                msg = (
                    f"training.input_dataset_dropout: the probability for '{name}' must be in [0, 1], "
                    f"got {probability}."
                )
                raise ValueError(msg)
        return dropout

    def _dropped_input_datasets(self) -> tuple[str, ...]:
        """first_step_only input datasets to leave out of this training sample (training.input_dataset_dropout).

        The draw depends only on the run seed and the global step, so every rank (and every member of an
        ensemble spread over several GPUs) makes the same choice. The model then zeroes their latents at the
        first step; later rollout steps never encode them anyway.
        """
        dropped = []
        for index, (name, probability) in enumerate(sorted(self.input_dataset_dropout.items())):
            draw = random.Random(  # noqa: S311 (a reproducible training draw, not cryptography)
                derive_seed(get_base_seed(), SeedContext.INPUT_DATASET_DROPOUT, int(self.global_step), index),
            ).random()
            if draw < probability:
                dropped.append(name)
        return tuple(dropped)

    @staticmethod
    def _validated_rollout_windows(
        windows: dict | None,
        accum_grad_batches: int,
    ) -> tuple[int, tuple[float, ...]] | None:
        """training.rollout_windows as (rollout steps per window, weight per window), or None when it is not set.

        ``steps`` is the window length; ``weights`` (optional, one per window) weigh the windows' losses and are
        normalised to sum to one (equal weights by default).
        """
        if not windows:
            return None
        steps = int(windows["steps"])
        weights = tuple(float(weight) for weight in (windows.get("weights") or ()))
        if steps < 1:
            msg = f"training.rollout_windows.steps must be at least 1, got {steps}."
            raise ValueError(msg)
        if any(weight < 0 for weight in weights) or (weights and sum(weights) <= 0):
            msg = f"training.rollout_windows.weights must be non-negative with a positive sum, got {list(weights)}."
            raise ValueError(msg)
        if accum_grad_batches != 1:
            msg = (
                "training.rollout_windows makes one optimiser step per batch (manual optimisation): set "
                f"training.accum_grad_batches to 1, not {accum_grad_batches}."
            )
            raise ValueError(msg)
        return steps, weights

    def _rollout_window_step_weights(self, num_steps: int) -> list[float]:
        """The weight of each of ``num_steps`` rollout steps in the loss: its window's weight, shared by its steps.

        The weights sum to one, so equal window weights give the plain mean over the steps (the loss without
        windows).
        """
        window_steps, weights = self.rollout_windows
        if num_steps % window_steps:
            msg = f"The rollout ({num_steps} steps) must be a whole number of rollout windows of {window_steps} steps."
            raise ValueError(msg)
        num_windows = num_steps // window_steps
        weights = weights or (1.0,) * num_windows
        if len(weights) != num_windows:
            msg = (
                f"training.rollout_windows.weights has {len(weights)} weights, for {num_windows} windows "
                f"({num_steps} steps in windows of {window_steps})."
            )
            raise ValueError(msg)
        total = sum(weights)
        return [weights[i // window_steps] / (total * window_steps) for i in range(num_steps)]

    def _windowed_rollout_backward(
        self,
        batch: dict[str, torch.Tensor],
        backward: Callable[[torch.Tensor, bool], None],
    ) -> torch.Tensor:
        """Run the training rollout window by window, calling ``backward(window_loss, is_last_window)`` for each.

        Each window's loss is its steps' losses times their weights (``_rollout_window_step_weights``), so the
        gradients added up over the windows are those of the weighted loss over the whole rollout, with the
        gradient cut at every window boundary: the next window starts from the values of the state, not from the
        graph that made them. Only one window's graph is in memory at a time. Steps keep their absolute index, so
        the noise follows the lead time, and noise modules carry their own state from one window to the next.
        Returns the weighted loss over the rollout (detached).
        """
        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        x = self._expand_ens_dim(x)
        dropped = self._dropped_input_datasets()

        task_steps = self.task.steps("training")
        step_weights = self._rollout_window_step_weights(len(task_steps))
        window_steps = self.rollout_windows[0]
        num_windows = len(task_steps) // window_steps

        log_memory = _claim_rollout_memory_log("training", len(task_steps))
        label = f"training, {len(task_steps)} steps in {num_windows} windows"
        self._memory_log_training_label = label if log_memory else None
        rank = getattr(self, "global_rank", 0)
        if log_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
            _log_rollout_memory(rank, f"{label} | start of batch", self.device)

        total = None
        for window in range(num_windows):
            window_loss = None
            for i in range(window * window_steps, (window + 1) * window_steps):
                task_step_kwargs = task_steps[i]
                y_pred = self(x, dropped_input_datasets=dropped, **task_step_kwargs)
                y = self.task.get_targets(batch, **task_step_kwargs)
                loss_next, _, _ = checkpoint(
                    self.compute_loss_metrics,
                    y_pred,
                    y,
                    **task_step_kwargs,
                    validation_mode=False,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.DATA_FULL,
                    use_reentrant=False,
                )
                weighted = step_weights[i] * loss_next
                window_loss = weighted if window_loss is None else window_loss + weighted
                if i < len(task_steps) - 1:
                    x = self.task.advance_input(
                        x,
                        y_pred,
                        batch,
                        **task_step_kwargs,
                        data_indices=self.data_indices,
                        output_mask=self.output_mask,
                        grid_shard_slice=self.grid_shard_slice,
                    )
                if log_memory:
                    _log_rollout_memory(rank, f"{label} | step {i + 1} after loss", self.device)

            backward(window_loss, window == num_windows - 1)
            window_loss = window_loss.detach()
            total = window_loss if total is None else total + window_loss
            # The gradient cut.
            x = {name: tensor.detach() for name, tensor in x.items()}

        return total

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Without rollout windows, the usual step. With them, one optimiser step after a backward pass per window.

        The gradients of the windows are added up locally (DDP's gradient synchronisation is blocked for all but
        the last window's backward pass), clipped as training.gradient_clip sets, and the optimiser and the
        step-interval learning-rate schedulers then step once.
        """
        if self.rollout_windows is None:
            return super().training_step(batch, batch_idx)
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        batch_size = next(iter(batch.values())).shape[0]

        optimizer = self.optimizers()
        optimizer.zero_grad()
        block_sync = getattr(self.trainer.strategy, "block_backward_sync", contextlib.nullcontext)

        def backward(loss: torch.Tensor, last_window: bool) -> None:
            with contextlib.nullcontext() if last_window else block_sync():
                self.manual_backward(loss)

        train_loss = self._windowed_rollout_backward(batch, backward)

        clip = self.config.training.gradient_clip
        if clip.val:
            self.clip_gradients(optimizer, gradient_clip_val=clip.val, gradient_clip_algorithm=clip.algorithm)
        optimizer.step()
        for scheduler_config in self.trainer.lr_scheduler_configs:
            if scheduler_config.interval != "step":
                msg = "training.rollout_windows supports learning-rate schedulers with interval 'step' only."
                raise NotImplementedError(msg)
            self.lr_scheduler_step(scheduler_config.scheduler, None)
        self._check_parameter_sync()

        self.log(
            "train_" + self._get_loss_name() + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.task.log_extra(logger=self.log, logger_enabled=self.logger_enabled)
        return train_loss

    def _check_parameter_sync(self) -> None:
        """After each of the first optimiser steps with rollout windows, check every rank holds the same parameters.

        Ranks hold copies of one model; if the gradients of the windows were not synchronised, the copies would
        drift apart silently. A checksum of all parameters is compared across ranks, and a difference stops
        training.
        """
        if self._parameter_sync_checks_left <= 0:
            return
        self._parameter_sync_checks_left -= 1
        distributed = torch.distributed
        if not (distributed.is_available() and distributed.is_initialized()) or distributed.get_world_size() == 1:
            return
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            checksum = torch.stack(
                [
                    sum(p.double().sum() for p in parameters),
                    sum(p.double().square().sum() for p in parameters),
                ],
            )
            gathered = [torch.zeros_like(checksum) for _ in range(distributed.get_world_size())]
            distributed.all_gather(gathered, checksum)
        if any(not torch.equal(other, gathered[0]) for other in gathered[1:]):
            msg = (
                f"After optimiser step {self.global_step} the ranks hold different parameters "
                f"({[g.tolist() for g in gathered]}): the windows' gradients were not synchronised."
            )
            raise RuntimeError(msg)
        LOGGER.info(
            "Rollout windows: parameters identical on all %d ranks after optimiser step %d.",
            distributed.get_world_size(),
            self.global_step,
        )

    def _combine_loss_and_metrics(
        self,
        step_losses: list[torch.Tensor],
        step_metrics: list[dict[str, torch.Tensor]],
        predictions: list[dict[str, torch.Tensor]],
    ) -> TrainingStepOutput:
        """As BaseTrainingModule's, but with rollout windows the loss weighs the steps as training does.

        So ``val_*_loss`` is the same weighted loss as ``train_*_loss``.
        """
        if self.rollout_windows is None:
            return super()._combine_loss_and_metrics(step_losses, step_metrics, predictions)
        num_loss_steps = self.task.num_steps
        weights = self._rollout_window_step_weights(num_loss_steps)
        loss = sum(weight * step_loss for weight, step_loss in zip(weights, step_losses[:num_loss_steps], strict=True))
        loss_names = {self._dataset_loss_metric_name(name) for name in self.target_dataset_names}
        metrics: dict[str, torch.Tensor] = {}
        for i, metrics_next in enumerate(step_metrics):
            for name, value in metrics_next.items():
                if name not in loss_names:
                    metrics[name] = value
                elif i < num_loss_steps:
                    metrics[name] = metrics.get(name, 0.0) + weights[i] * value
        return TrainingStepOutput(loss=loss, metrics=metrics, predictions=predictions)

    def _expand_ens_dim(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expand the ensemble dimension in the input batch by stacking the data nens_per_device times."""
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch.tile(1, 1, self.nens_per_device, 1, 1)
            LOGGER.debug("SHAPE: x[%s].shape = %s", dataset_name, list(x[dataset_name].shape))

        return x

    def compute_dataset_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        rollout_step: int | None = None,
        validation_mode: bool = False,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor]:

        y_pred_ens = gather_tensor(
            y_pred.clone(),  # for bwd because we checkpoint this region
            dim=TensorDim.ENSEMBLE_DIM,
            sizes=[y_pred.size(TensorDim.ENSEMBLE_DIM)] * self.ens_comm_subgroup_size,
            mgroup=self.ens_comm_subgroup,
        )

        y_pred_ens_full, y_full, grid_shard_slice = self._prepare_tensors_for_loss(
            y_pred_ens,
            y,
            validation_mode=validation_mode,
            dataset_name=dataset_name,
        )

        # torch.compile performance change
        # mark pred_filtered and target_filtered as dynamic shapes
        # (they change based on *_indices)
        # Marking them as dynamic prevents torch from recompiling
        # everytime the *_indices change
        # Tensors must be marked as dynamic before being passed to the compiled function
        dynamic_indices = True  # TODO(cathal): set as true only for validation
        if dynamic_indices:
            torch._dynamo.mark_dynamic(y_pred_ens_full, -1)

        loss = self._compute_loss(
            y_pred_ens_full,
            y_full,
            grid_shard_slice=grid_shard_slice,
            dataset_name=dataset_name,
            pred_layout=pred_layout,
            target_layout=target_layout,
        )

        # Compute metrics if in validation mode
        metrics_next = {}
        if validation_mode:
            metrics_next = self._compute_metrics(
                y_pred_ens_full,
                y_full,
                rollout_step=rollout_step,
                dataset_name=dataset_name,
                grid_shard_slice=grid_shard_slice,
                pred_layout=pred_layout,
                target_layout=target_layout,
            )

        return loss, metrics_next, y_pred_ens

    def forward(self, x: dict[str, torch.Tensor], rollout_step: int | None = None, **kwargs) -> dict[str, torch.Tensor]:
        """Forward method.

        This method calls the model's forward method with the appropriate
        communication group and sharding information.
        """
        if rollout_step is not None:
            kwargs["fcstep"] = rollout_step
        else:
            kwargs["fcstep"] = 0  # TODO(Mario,Simon): set the conditioning on the step optional

        # Global ensemble indices, so noise realisations depend on which member a
        # tensor slice *is*, not on how members happen to be spread over devices.
        kwargs.setdefault("ensemble_member_offset", self.ensemble_member_offset)
        kwargs.setdefault("ensemble_members_total", self.ensemble_members_total)
        kwargs.setdefault("ensemble_group_id", self.ens_comm_group_id)
        kwargs.setdefault("input_noise_seed", self.input_noise_seed)

        return self.model(
            x,
            model_comm_group=self.model_comm_group,
            grid_shard_sizes=self.grid_shard_sizes,
            **kwargs,
        )

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> TrainingStepOutput:
        """Training / validation step."""
        step_losses, step_metrics, y_preds = [], [], []

        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        x = self._expand_ens_dim(x)
        dropped = () if validation_mode else self._dropped_input_datasets()

        task_steps = self.task.steps("training" if not validation_mode else "validation")
        mode = "validation" if validation_mode else "training"
        log_memory = _claim_rollout_memory_log(mode, len(task_steps))
        self._memory_log_training_label = (
            f"training, {len(task_steps)} steps" if log_memory and not validation_mode else None
        )
        rank = getattr(self, "global_rank", 0)
        if log_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
            _log_rollout_memory(rank, f"{mode}, {len(task_steps)} steps | start of batch", self.device)
        for i, task_step_kwargs in enumerate(task_steps):
            y_pred = self(x, dropped_input_datasets=dropped, **task_step_kwargs)
            if log_memory:
                _log_rollout_memory(rank, f"{mode}, {len(task_steps)} steps | step {i + 1} after forward", self.device)

            y = self.task.get_targets(batch, **task_step_kwargs)

            loss_next, metrics_next, y_preds_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                **task_step_kwargs,
                validation_mode=validation_mode,
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
                use_reentrant=False,
            )

            # Advance input state for each dataset if another step follows
            if i < len(task_steps) - 1:
                x = self.task.advance_input(
                    x,
                    y_pred,
                    batch,
                    **task_step_kwargs,
                    data_indices=self.data_indices,
                    output_mask=self.output_mask,
                    grid_shard_slice=self.grid_shard_slice,
                )

            step_losses.append(loss_next)
            step_metrics.append(metrics_next)
            y_preds.append(y_preds_next)
            if log_memory:
                _log_rollout_memory(rank, f"{mode}, {len(task_steps)} steps | step {i + 1} after loss", self.device)

        return self._combine_loss_and_metrics(step_losses, step_metrics, y_preds)

    def on_after_backward(self) -> None:
        """Log the memory after the backward pass of a batch whose rollout memory is being logged."""
        super().on_after_backward()
        label = getattr(self, "_memory_log_training_label", None)
        if label is not None:
            _log_rollout_memory(getattr(self, "global_rank", 0), f"{label} | after backward", self.device)

    def on_train_batch_end(self, outputs: object, batch: object, batch_idx: int) -> None:
        """Log the memory at the end of a logged batch: its peak then includes the optimiser step, if one ran."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        label = getattr(self, "_memory_log_training_label", None)
        if label is not None:
            self._memory_log_training_label = None
            _log_rollout_memory(getattr(self, "global_rank", 0), f"{label} | end of batch", self.device)
