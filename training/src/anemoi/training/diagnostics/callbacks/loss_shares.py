# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback recording every variable's share of the training loss during training."""

import fnmatch
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback

from anemoi.training.losses.combined import CombinedLoss

LOGGER = logging.getLogger(__name__)

PRESSURE_LEVEL_PARAMS = ("z", "t", "u", "v", "q", "w")
SURFACE_VARIABLES = ("2t", "2d", "10u", "10v", "msl", "sp", "sst", "skt", "tcw")
SINGLE_VARIABLES = ("cf", "tp", "cris_cf")

# Columns of a recorded row before the three per-variable vectors.
_HEAD = ("global_step", "rollout_step", "is_da", "ret")


def variable_group(
    name: str,
    groups: Mapping[str, Sequence[str]] | None = None,
    pressure_level_params: Sequence[str] = PRESSURE_LEVEL_PARAMS,
    surface_variables: Sequence[str] = SURFACE_VARIABLES,
    single_variables: Sequence[str] = SINGLE_VARIABLES,
) -> str:
    """Return the loss-share group of an output variable.

    The first of ``groups`` with an ``fnmatch`` pattern matching ``name`` wins. Otherwise
    pressure-level variables group by parameter (``t_850 -> t``), the surface variables form
    ``surface``, the single variables keep their own name, and instrument channels group by
    instrument (``iasi_921 -> iasi``, ``georing_vis064 -> georing``).

    Parameters
    ----------
    name : str
        Output variable name.
    groups : Mapping[str, Sequence[str]], optional
        Explicit groups: ``{group: [patterns]}``, checked in order.
    pressure_level_params : Sequence[str]
        Parameters whose ``<param>_<level>`` variables group by parameter.
    surface_variables : Sequence[str]
        Variables grouped as ``surface``.
    single_variables : Sequence[str]
        Variables that form a group of their own.

    Returns
    -------
    str
        Group name.
    """
    for group, patterns in (groups or {}).items():
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            return group
    head, _, tail = name.rpartition("_")
    if head in pressure_level_params and tail.isdigit():
        return head
    if name in surface_variables:
        return "surface"
    if name in single_variables:
        return name
    return name.split("_", 1)[0]


def gather_rows(rows: torch.Tensor, keep: bool, device: torch.device) -> torch.Tensor:
    """Gather the rows of every rank with ``keep`` set onto all ranks.

    Ranks may hold different numbers of rows; they are padded to the largest count
    for the collective and the padding is dropped again.

    Parameters
    ----------
    rows : torch.Tensor
        Local rows, shape ``(n_rows, n_cols)``.
    keep : bool
        Whether this rank's rows are kept (one rank per model group).
    device : torch.device
        Device for the collectives (the rank's CUDA device under NCCL).

    Returns
    -------
    torch.Tensor
        The kept rows of all ranks, in rank order, on the CPU.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return rows.cpu() if keep else rows[:0].cpu()
    n_local = torch.tensor([rows.shape[0] if keep else 0], device=device, dtype=torch.long)
    counts = [torch.zeros_like(n_local) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, n_local)
    counts = [int(c) for c in counts]
    n_max = max(counts)
    if n_max == 0:
        return rows[:0].cpu()
    padded = torch.zeros((n_max, rows.shape[1]), dtype=rows.dtype, device=device)
    if keep and rows.shape[0]:
        padded[: rows.shape[0]] = rows.to(device)
    gathered = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:c] for g, c in zip(gathered, counts, strict=True)]).cpu()


def leaf_weights(loss: torch.nn.Module) -> list[float]:
    """Return the weight of each leaf of ``loss`` in its total, in ``iter_leaf_losses`` order.

    Weights multiply through nested :class:`CombinedLoss` and pass through loss wrappers.

    Parameters
    ----------
    loss : torch.nn.Module
        A training loss (possibly wrapped or combined).

    Returns
    -------
    list[float]
        One weight per leaf.
    """
    if isinstance(loss, CombinedLoss):
        return [
            float(weight) * w
            for sub_loss, weight in zip(loss.losses, loss.loss_weights, strict=True)
            for w in leaf_weights(sub_loss)
        ]
    inner = getattr(loss, "loss", None)
    if isinstance(inner, torch.nn.Module):
        return leaf_weights(inner)
    return [1.0] * len(list(loss.iter_leaf_losses()))


class LossShareMonitor(Callback):
    """Record each variable's contribution to the training loss, on every step.

    The monitored leaf of ``dataset_name``'s training loss (``leaf``) is wrapped so that every
    call made during the forward pass also evaluates the per-variable loss vector
    (``squash=False``) three ways:

    - ``full``     : all scalers, i.e. what the optimiser sees. The mean over variables is
                     the leaf's loss.
    - ``no_gv``    : without ``general_variable``. Re-weighting is exact on this basis:
                     the loss with weights ``w`` is ``w * no_gv``.
    - ``no_gv_st`` : also without ``stdev_tendency`` (area-weighted error in normalised
                     units; availability-diluted for sparse observations).

    The other leaves of the loss are recorded as scalars. The vectors are already reduced
    over the model-parallel group, so one rank per model group contributes rows. Recomputation
    of the loss inside activation checkpointing (during backward) is not recorded twice.
    Validation calls are recorded too, sanity checking is not. Each row carries the step's
    ``rollout_step`` and ``is_da`` (DA cycle of a DA forecaster).

    Every ``every_n_steps`` optimiser steps, at the end of every training epoch (rollout
    changes there) and at the end of every validation epoch, rank 0 writes the rows of all
    model groups to ``<output_dir>/<split>_s<first>-<last>/loss_shares_raw.npz`` plus
    ``meta.json``, and logs each group's share of the window-mean loss as
    ``loss_share/<split>/<group>``, over the forecast rows (all rows if there are none).
    Groups follow :func:`variable_group` with ``groups`` and the variable lists, or are single
    variables with ``per_variable`` (observation datasets, whose names do not follow the
    gridded conventions).

    Parameters
    ----------
    output_dir : str
        Directory for the window files.
    dataset_name : str
        Dataset whose loss is monitored.
    every_n_steps : int
        Training window length in optimiser steps.
    record_validation : bool
        Also record the validation loss calls.
    per_variable : bool
        Log one share per variable instead of per group.
    leaf : str | int | None
        The leaf loss to monitor: its index in ``iter_leaf_losses`` order, or its class or
        ``name``. ``None`` requires the loss to have a single leaf.
    groups : Mapping[str, Sequence[str]] | None
        Explicit groups ``{group: [fnmatch patterns]}``, checked before the default rules.
    pressure_level_params : Sequence[str]
        Parameters whose ``<param>_<level>`` variables group by parameter.
    surface_variables : Sequence[str]
        Variables grouped as ``surface``.
    single_variables : Sequence[str]
        Variables that form a group of their own.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "data",
        every_n_steps: int = 250,
        record_validation: bool = True,
        per_variable: bool = False,
        leaf: str | int | None = None,
        groups: Mapping[str, Sequence[str]] | None = None,
        pressure_level_params: Sequence[str] = PRESSURE_LEVEL_PARAMS,
        surface_variables: Sequence[str] = SURFACE_VARIABLES,
        single_variables: Sequence[str] = SINGLE_VARIABLES,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.every_n_steps = every_n_steps
        self.record_validation = record_validation
        self.per_variable = per_variable
        self.leaf = leaf
        self.groups = {group: list(patterns) for group, patterns in (groups or {}).items()}
        self.pressure_level_params = tuple(pressure_level_params)
        self.surface_variables = tuple(surface_variables)
        self.single_variables = tuple(single_variables)

        self._installed = False
        self._split: str | None = None
        self._rows: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._others: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        self._first_step: int | None = None
        self._names: list[str] = []
        self._leaf_name = ""
        self._leaf_weight = 1.0
        self._other_leaves: list[tuple[str, float]] = []
        self._step = 0
        self._rollout_step: int = -1
        self._is_da = False

    def group_of(self, name: str) -> str:
        """Return the loss-share group of an output variable.

        Parameters
        ----------
        name : str
            Output variable name.

        Returns
        -------
        str
            Group name (the variable itself with ``per_variable``).
        """
        if self.per_variable:
            return name
        return variable_group(
            name,
            self.groups,
            pressure_level_params=self.pressure_level_params,
            surface_variables=self.surface_variables,
            single_variables=self.single_variables,
        )

    # ------------------------------------------------------------------ setup

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Wrap the loss leaves of ``dataset_name``.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method holding ``loss``.
        stage : str
            Lightning stage.
        """
        del trainer, stage
        if self._installed:
            return
        loss = pl_module.loss[self.dataset_name]
        leaves = list(loss.iter_leaf_losses())
        weights = leaf_weights(loss)
        index = self._select_leaf(leaves)
        main = leaves[index]
        self._leaf_name = type(main).__name__
        self._leaf_weight = weights[index]
        others = [(i, leaf) for i, leaf in enumerate(leaves) if i != index]
        self._other_leaves = [(type(leaf).__name__, weights[i]) for i, leaf in others]

        self._wrap_main(main)
        for position, (_, leaf) in enumerate(others):
            self._wrap_other(leaf, position)
        self._wrap_compute_loss_metrics(pl_module)
        self._installed = True
        LOGGER.info(
            "LossShareMonitor: %s on '%s', windows of %d steps -> %s",
            self._leaf_name,
            self.dataset_name,
            self.every_n_steps,
            self.output_dir,
        )

    def _select_leaf(self, leaves: list[torch.nn.Module]) -> int:
        """Index of the monitored leaf among ``leaves``."""
        names = [type(leaf).__name__ for leaf in leaves]
        if self.leaf is None:
            if len(leaves) != 1:
                msg = f"LossShareMonitor: the loss has several leaves {names}; choose one with `leaf`."
                raise ValueError(msg)
            return 0
        if isinstance(self.leaf, int):
            if not -len(leaves) <= self.leaf < len(leaves):
                msg = f"LossShareMonitor: leaf index {self.leaf} is out of range for the leaves {names}."
                raise ValueError(msg)
            return self.leaf % len(leaves)
        matches = [i for i, leaf in enumerate(leaves) if self.leaf in (type(leaf).__name__, leaf.name)]
        if len(matches) != 1:
            msg = (
                f"LossShareMonitor: leaf {self.leaf!r} matches {len(matches)} of the leaves {names}; it must match one."
            )
            raise ValueError(msg)
        return matches[0]

    def _recording(self) -> bool:
        # The graph task id is -1 outside backward; inside it, the checkpointed loss is a recompute.
        return self._split is not None and torch._C._current_graph_task_id() == -1

    def _wrap_main(self, leaf: torch.nn.Module) -> None:
        orig = leaf.forward

        def forward(pred: Any, target: Any, squash: bool = True, **kwargs) -> torch.Tensor:
            ret = orig(pred, target, squash, **kwargs)
            if not self._recording():
                return ret
            kw = {k: v for k, v in kwargs.items() if k not in ("without_scalers", "squash_mode")}
            with torch.no_grad():
                vectors = [
                    orig(pred, target, False, **kw),
                    orig(pred, target, False, without_scalers=["general_variable"], **kw),
                    orig(pred, target, False, without_scalers=["general_variable", "stdev_tendency"], **kw),
                ]
                names = list(getattr(pred, "variables", []))
                if not self._names:
                    self._names = names
                elif names and names != self._names:
                    msg = f"LossShareMonitor: the monitored leaf saw variables {names}, before {self._names}."
                    raise ValueError(msg)
                # Kept on the device until the window is written, so recording never syncs the host.
                head = torch.tensor([self._step, self._rollout_step, float(self._is_da)], dtype=torch.float32)
                parts = [
                    head.to(ret.device),
                    ret.detach().float().mean().reshape(1),
                    *(v.detach().float().reshape(-1) for v in vectors),
                ]
                self._rows[self._split].append(torch.cat(parts))
            return ret

        leaf.forward = forward

    def _wrap_other(self, leaf: torch.nn.Module, position: int) -> None:
        orig = leaf.forward

        def forward(*args, **kwargs) -> torch.Tensor:
            ret = orig(*args, **kwargs)
            if self._recording():
                self._others[self._split][position].append(ret.detach().float().mean().reshape(1))
            return ret

        leaf.forward = forward

    def _wrap_compute_loss_metrics(self, pl_module: pl.LightningModule) -> None:
        orig = pl_module.compute_loss_metrics

        def compute_loss_metrics(*args, rollout_step: int | None = None, **kwargs) -> Any:
            self._rollout_step = -1 if rollout_step is None else int(rollout_step)
            self._is_da = bool(kwargs.get("is_da", False))
            return orig(*args, rollout_step=rollout_step, **kwargs)

        pl_module.compute_loss_metrics = compute_loss_metrics

    # ------------------------------------------------------------------ training

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record the coming training step.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        batch : Any
            The batch.
        batch_idx : int
            Batch index.
        """
        del pl_module, batch, batch_idx
        self._split = "train"
        self._step = trainer.global_step
        if self._first_step is None:
            self._first_step = trainer.global_step

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Write the training window when it is complete.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        outputs : Any
            Step outputs.
        batch : Any
            The batch.
        batch_idx : int
            Batch index.
        """
        del outputs, batch, batch_idx
        self._split = None
        if trainer.global_step % self.every_n_steps == 0:
            self._flush(trainer, pl_module, "train")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write the partial training window, so no window spans a rollout change.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        """
        self._flush(trainer, pl_module, "train")

    # ------------------------------------------------------------------ validation

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record the coming validation step, outside sanity checking.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        batch : Any
            The batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        del pl_module, batch, batch_idx, dataloader_idx
        if self.record_validation and not trainer.sanity_checking:
            self._split = "valid"
            self._step = trainer.global_step

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Stop recording until the next step.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        outputs : Any
            Step outputs.
        batch : Any
            The batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        self._split = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write the validation window.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        """
        if self.record_validation and not trainer.sanity_checking:
            self._flush(trainer, pl_module, "valid")

    # ------------------------------------------------------------------ output

    def _flush(self, trainer: pl.Trainer, pl_module: pl.LightningModule, split: str) -> None:
        """Gather the window's rows from one rank per model group and write them on rank 0."""
        n_vars = len(self._names)
        n_head = len(_HEAD)
        device = pl_module.device
        rows = self._rows.pop(split, [])
        others = self._others.pop(split, {})
        local = torch.stack(rows) if rows else torch.zeros((0, n_head + 3 * n_vars), device=device)
        keep = getattr(pl_module, "model_comm_group_rank", 0) == 0
        table = gather_rows(local, keep, device).numpy()
        other_values = []
        for position in range(len(self._other_leaves)):
            values = others.get(position, [])
            other_local = torch.cat(values).reshape(-1, 1) if values else torch.zeros((0, 1), device=device)
            other_values.append(gather_rows(other_local, keep, device).numpy().reshape(-1))
        first = self._first_step if split == "train" else trainer.global_step
        if split == "train":
            self._first_step = None
        if trainer.global_rank != 0 or table.shape[0] == 0:
            return

        step, rollout_step, is_da, ret = (table[:, k] for k in range(n_head))
        full, no_gv, no_gv_st = (table[:, n_head + k * n_vars : n_head + (k + 1) * n_vars] for k in range(3))
        last = int(step.max())
        out = self.output_dir / f"{split}_s{first:06d}-{last:06d}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "loss_shares_raw.npz",
            names=np.array(self._names),
            full=full,
            no_gv=no_gv,
            no_gv_st=no_gv_st,
            ret=ret,
            step=rollout_step,
            is_da=is_da.astype(bool),
            global_step=step,
            **{f"other{i}": values for i, values in enumerate(other_values)},
        )
        mean_step_loss = self._leaf_weight * float(ret.mean()) + sum(
            weight * float(values.mean())
            for (_, weight), values in zip(self._other_leaves, other_values, strict=True)
            if values.size
        )
        meta = {
            "split": split,
            "dataset": self.dataset_name,
            "ckpt_global_step": last,
            "first_step": int(first),
            "last_step": last,
            "epoch": trainer.current_epoch,
            "rollout_steps": sorted({int(r) for r in rollout_step}),
            "n_rows": int(table.shape[0]),
            "n_da_rows": int(is_da.sum()),
            "n_vars": n_vars,
            "model_groups": getattr(pl_module, "model_comm_num_groups", 1),
            "leaf": self._leaf_name,
            "leaf_weight": self._leaf_weight,
            "other_leaves": [{"name": name, "weight": weight} for name, weight in self._other_leaves],
            "mean_step_loss": mean_step_loss,
            "lr": [float(g["lr"]) for opt in trainer.optimizers for g in opt.param_groups][:1],
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        forecast = is_da == 0
        self._log_shares(trainer, split, last, full[forecast] if forecast.any() else full)

    def shares(self, full: np.ndarray) -> dict[str, float]:
        """Each group's share of the mean loss.

        Parameters
        ----------
        full : np.ndarray
            Per-variable losses, ``(n_rows, n_vars)`` in ``self._names`` order.

        Returns
        -------
        dict[str, float]
            Share per group, summing to 1.
        """
        mean = full.astype(np.float64).mean(0)
        total = mean.sum()
        shares: dict[str, float] = defaultdict(float)
        for name, value in zip(self._names, mean, strict=True):
            shares[self.group_of(name)] += value / total
        return dict(shares)

    def _log_shares(self, trainer: pl.Trainer, split: str, step: int, full: np.ndarray) -> None:
        shares = self.shares(full)
        metrics = {f"loss_share/{split}/{g}": s for g, s in shares.items()}
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)
        top = " ".join(f"{g}={s * 100:.1f}%" for g, s in sorted(shares.items(), key=lambda x: -x[1])[:10])
        LOGGER.info("LossShareMonitor %s step %d (%d rows): %s", split, step, full.shape[0], top)
