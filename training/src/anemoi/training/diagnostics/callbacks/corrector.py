# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Callback logging the training health of the per-instrument training-time correctors."""

import logging
import math
from functools import partial
from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from torch import nn

from anemoi.training.optimizers.AdEMAMix import linear_warmup_scheduler

if TYPE_CHECKING:
    from anemoi.models.data.sources import Source

LOGGER = logging.getLogger(__name__)

DST_MLP_SUFFIX = "node_dst_mlp.mlp.0"


class CorrectorDiagnostics(Callback):
    """Log, per instrument group, why a corrector does or does not receive gradient.

    Every ``every_n_steps`` training steps this records, for each group of
    ``InstrumentCorrectors`` of ``dataset_name``:

    - ``obs_frac``, ``obs_point_frac``, ``obs_sample_frac``: the fraction of finite target
      values in the group's channels at the loss-carrying steps (``task.step_loss_weight``
      above 0), the fraction of grid points (tabular: rows) with at least one finite channel,
      and the fraction of samples with any observation. The loss is NaN-masked, so the
      corrector gets gradient only from these.
    - ``meta_nonzero_frac``: the fraction of non-zero corrector metadata. Missing metadata
      reads as 0.
    - ``input_proj_rms``, ``input_proj_sat_frac``: the input projection's pre-activation
      RMS and the fraction below ``saturation_threshold`` (GELU gradient below about 1%).
    - ``proc<i>_dst_mlp0_{mean,std,sat_frac}``: the same for the first ``node_dst_mlp``
      linear of each processor block (modules named ``...<dst_mlp_suffix>``).
    - ``correction_rms``, ``correction_absmax``: the size of the additive correction.
    - ``grad_norm``, ``grad_zero_frac``, ``proc<i>_dst_mlp0_grad_rms``: gradients after the
      data-parallel reduction, before clipping and the optimizer step.
    - ``param_rms``, ``proc<i>_dst_mlp0_param_rms``: parameter size.
    - ``upd_fast_rms``, ``upd_slow_rms``, ``upd_absmax``, ``sqrt_v_floor_frac``: the RMS of
      the Adam-style term ``m_fast / sqrt(v)`` and of the AdEMAMix slow term
      ``alpha * m_slow / sqrt(v)``, the largest per-element update, and the fraction of
      elements whose ``sqrt(v_hat)`` is below ``2 * eps``. ``upd_slow_rms`` is 0 for AdamW.

    The main model is reported as group ``_model`` for the gradient, parameter and update
    metrics. Values are averaged (``absmax``: maximum) over all ranks, written straight to
    the loggers as ``corrector_diag/<group>/<metric>`` and summarised in the job log.

    Parameters
    ----------
    dataset_name : str
        Dataset whose correctors are monitored.
    groups : list[str] | None
        Instrument groups to monitor. ``None`` monitors every group.
    every_n_steps : int
        Logging frequency in optimizer steps.
    saturation_threshold : float
        Pre-activation value below which a GELU unit counts as saturated.
    include_model : bool
        Also report gradient, parameter and update metrics for the main model.
    dst_mlp_suffix : str
        Name suffix of the processor-block linears reported as ``proc<i>_dst_mlp0``.
    """

    def __init__(
        self,
        dataset_name: str = "data",
        groups: list[str] | None = None,
        every_n_steps: int = 25,
        saturation_threshold: float = -3.0,
        include_model: bool = True,
        dst_mlp_suffix: str = DST_MLP_SUFFIX,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.groups = None if groups is None else list(groups)
        self.every_n_steps = max(1, int(every_n_steps))
        self.saturation_threshold = float(saturation_threshold)
        self.include_model = include_model
        self.dst_mlp_suffix = dst_mlp_suffix

        self._active = False
        self._handles: list[Any] = []
        self._correctors: dict[str, nn.Module] = {}
        self._dst_mlp: dict[str, list[nn.Module]] = {}
        self._target_names: dict[str, list[str]] = {}
        self._keys: list[str] = []
        self._max_keys: set[str] = set()
        self._values: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ setup

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Resolve the correctors, register forward hooks and fix the metric keys.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method holding ``corrector``.
        stage : str
            Lightning stage.
        """
        del trainer, stage
        if self._handles:
            return
        corrector_dict = getattr(pl_module, "corrector", None)
        if corrector_dict is None or self.dataset_name not in corrector_dict:
            LOGGER.warning("CorrectorDiagnostics: no corrector for dataset '%s', disabled.", self.dataset_name)
            return
        instrument_correctors = corrector_dict[self.dataset_name]
        groups = self.groups if self.groups is not None else list(instrument_correctors.correctors.keys())
        indices = pl_module.data_indices[self.dataset_name]
        output_names = indices.model.output.ordered_names

        for group in groups:
            if group not in instrument_correctors.correctors:
                LOGGER.warning("CorrectorDiagnostics: group '%s' has no corrector, skipped.", group)
                continue
            corrector = instrument_correctors.correctors[group]
            self._correctors[group] = corrector
            target_idx = getattr(instrument_correctors, f"_target_idx_{group}").tolist()
            self._target_names[group] = [output_names[i] for i in target_idx]
            self._dst_mlp[group] = [m for n, m in corrector.named_modules() if n.endswith(self.dst_mlp_suffix)]
            self._register_hooks(group, corrector)

        self._keys, self._max_keys = self._metric_keys()
        LOGGER.info(
            "CorrectorDiagnostics: monitoring groups %s of dataset '%s' every %d steps (%d metrics).",
            list(self._correctors),
            self.dataset_name,
            self.every_n_steps,
            len(self._keys),
        )

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Remove the forward hooks.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        stage : str
            Lightning stage.
        """
        del trainer, pl_module, stage
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _metric_keys(self) -> tuple[list[str], set[str]]:
        """Return a fixed, rank-independent metric key list and the keys reduced by max."""
        keys, max_keys = [], set()
        per_layer = ("dst_mlp0_mean", "dst_mlp0_std", "dst_mlp0_sat_frac", "dst_mlp0_grad_rms", "dst_mlp0_param_rms")
        optim = ("grad_norm", "grad_zero_frac", "param_rms", "upd_fast_rms", "upd_slow_rms", "upd_absmax")
        for group, layers in self._dst_mlp.items():
            keys += [
                f"{group}/{m}"
                for m in (
                    "obs_frac",
                    "obs_point_frac",
                    "obs_sample_frac",
                    "meta_nonzero_frac",
                    "input_proj_rms",
                    "input_proj_sat_frac",
                    "correction_rms",
                    "correction_absmax",
                    "sqrt_v_floor_frac",
                    *optim,
                )
            ]
            keys += [f"{group}/proc{i}_{m}" for i in range(len(layers)) for m in per_layer]
        if self.include_model:
            keys += [f"_model/{m}" for m in (*optim, "sqrt_v_floor_frac")]
        max_keys = {k for k in keys if k.endswith(("_absmax", "grad_norm"))}
        return keys, max_keys

    # ------------------------------------------------------------------ hooks

    def _record(self, key: str, value: torch.Tensor | float) -> None:
        self._values.setdefault(key, []).append(float(value))

    def _recording(self, module: nn.Module) -> bool:
        # Validation runs without grad; only the training forward that feeds the loss is recorded.
        return self._active and module.training and torch.is_grad_enabled()

    def _register_hooks(self, group: str, corrector: nn.Module) -> None:
        self._handles.append(corrector.register_forward_pre_hook(partial(self._corrector_pre, group), with_kwargs=True))
        self._handles.append(corrector.register_forward_hook(partial(self._corrector_post, group), with_kwargs=True))
        if hasattr(corrector, "input_proj"):
            self._handles.append(corrector.input_proj.register_forward_hook(partial(self._input_proj_post, group)))
        for i, layer in enumerate(self._dst_mlp[group]):
            self._handles.append(layer.register_forward_hook(partial(self._dst_mlp_post, f"{group}/proc{i}")))

    def _corrector_pre(self, group: str, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if not self._recording(module):
            return
        corrector_vars = args[1] if len(args) > 1 else kwargs["corrector_vars"]
        self._record(f"{group}/meta_nonzero_frac", (corrector_vars != 0).float().mean())

    def _corrector_post(self, group: str, module: nn.Module, _args: tuple, _kwargs: dict, out: torch.Tensor) -> None:
        if not self._recording(module):
            return
        out = out.detach().float()
        self._record(f"{group}/correction_rms", out.pow(2).mean().sqrt())
        self._record(f"{group}/correction_absmax", out.abs().max())

    def _input_proj_post(self, group: str, module: nn.Module, _args: tuple, out: torch.Tensor) -> None:
        if not self._recording(module):
            return
        out = out.detach().float()
        self._record(f"{group}/input_proj_rms", out.pow(2).mean().sqrt())
        self._record(f"{group}/input_proj_sat_frac", (out < self.saturation_threshold).float().mean())

    def _dst_mlp_post(self, prefix: str, module: nn.Module, _args: tuple, out: torch.Tensor) -> None:
        if not self._recording(module):
            return
        out = out.detach().float()
        self._record(f"{prefix}_dst_mlp0_mean", out.mean())
        self._record(f"{prefix}_dst_mlp0_std", out.std())
        self._record(f"{prefix}_dst_mlp0_sat_frac", (out < self.saturation_threshold).float().mean())

    # ------------------------------------------------------------------ batch

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Arm the hooks on logging steps and record observation availability.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method; its task selects the loss-carrying target steps.
        batch : Batch
            The batch.
        batch_idx : int
            Index of the batch, not used.
        """
        del batch_idx
        self._active = bool(self._keys) and trainer.global_step % self.every_n_steps == 0
        if not self._active:
            return
        self._values = {}
        task = pl_module.task
        with torch.no_grad():
            for step_kwargs in task.steps("training"):
                if task.step_loss_weight(**step_kwargs) <= 0:
                    continue
                targets, _, _ = task.get_targets(batch, data_indices=pl_module.data_indices, **step_kwargs)
                for group, names in self._target_names.items():
                    for metric, value in _availability(targets[self.dataset_name], names).items():
                        self._record(f"{group}/{metric}", value)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Record gradient, parameter and optimizer-update statistics, then log.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer.
        pl_module : pl.LightningModule
            The training method.
        optimizer : torch.optim.Optimizer
            The optimizer about to step; its state holds the moments of the previous step.
        """
        if not self._active:
            return
        self._active = False
        with torch.no_grad():
            for group, corrector in self._correctors.items():
                self._optim_stats(group, list(corrector.parameters()), optimizer)
                for i, layer in enumerate(self._dst_mlp[group]):
                    w = layer.weight
                    self._record(f"{group}/proc{i}_dst_mlp0_param_rms", w.float().pow(2).mean().sqrt())
                    if w.grad is not None:
                        self._record(f"{group}/proc{i}_dst_mlp0_grad_rms", w.grad.float().pow(2).mean().sqrt())
            if self.include_model and hasattr(pl_module, "model"):
                self._optim_stats("_model", list(pl_module.model.parameters()), optimizer)
        self._log(trainer)

    def _optim_stats(self, group: str, params: list[nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
        """Accumulate gradient, parameter and update statistics over ``params``."""
        hyper = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        sq = dict.fromkeys(("grad", "param", "fast", "slow"), 0.0)
        n = n_zero = n_floor = n_state = 0
        absmax = 0.0
        for p in params:
            if not p.requires_grad:
                continue
            n += p.numel()
            sq["param"] += p.float().pow(2).sum().item()
            if p.grad is not None:
                sq["grad"] += p.grad.float().pow(2).sum().item()
                n_zero += (p.grad == 0).sum().item()
            else:
                n_zero += p.numel()
            state, hp = optimizer.state.get(p), hyper.get(id(p))
            if not state or hp is None or "exp_avg_sq" not in state:
                continue
            step = int(state["step"])
            beta1, beta2 = hp["betas"][0], hp["betas"][1]
            eps = hp["eps"]
            denom = state["exp_avg_sq"].float().sqrt() / math.sqrt(1 - beta2**step)
            n_floor += (denom < 2 * eps).sum().item()
            denom = denom + eps
            m_fast = state.get("exp_avg_fast", state.get("exp_avg"))
            upd = m_fast.float() / (1 - beta1**step) / denom
            sq["fast"] += upd.pow(2).sum().item()
            if "exp_avg_slow" in state:
                alpha = hp["alpha"]
                if hp.get("alpha_warmup") is not None:
                    alpha = linear_warmup_scheduler(step, alpha_end=alpha, alpha_start=0, warmup=hp["alpha_warmup"])
                slow = alpha * state["exp_avg_slow"].float() / denom
                sq["slow"] += slow.pow(2).sum().item()
                upd = upd + slow
            absmax = max(absmax, upd.abs().max().item())
            n_state += p.numel()
        if n == 0:
            return
        self._record(f"{group}/grad_norm", math.sqrt(sq["grad"]))
        self._record(f"{group}/grad_zero_frac", n_zero / n)
        self._record(f"{group}/param_rms", math.sqrt(sq["param"] / n))
        if n_state:
            self._record(f"{group}/upd_fast_rms", math.sqrt(sq["fast"] / n_state))
            self._record(f"{group}/upd_slow_rms", math.sqrt(sq["slow"] / n_state))
            self._record(f"{group}/upd_absmax", absmax)
            self._record(f"{group}/sqrt_v_floor_frac", n_floor / n_state)

    # ------------------------------------------------------------------ logging

    def _log(self, trainer: pl.Trainer) -> None:
        """Reduce the recorded values over all ranks and write them to the loggers."""
        device = trainer.strategy.root_device
        local = torch.full((len(self._keys),), float("nan"), dtype=torch.float64, device=device)
        for j, key in enumerate(self._keys):
            vals = self._values.get(key)
            if vals:
                local[j] = max(vals) if key in self._max_keys else sum(vals) / len(vals)
        is_max = torch.tensor([k in self._max_keys for k in self._keys], device=device)
        finite = torch.isfinite(local)
        total = torch.where(finite & ~is_max, local, torch.zeros_like(local))
        count = (finite & ~is_max).to(torch.float64)
        peak = torch.where(finite & is_max, local, torch.full_like(local, -math.inf))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
            dist.all_reduce(count)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        reduced = torch.where(is_max, peak, total / count.clamp(min=1)).tolist()
        valid = torch.where(is_max, peak > -math.inf, count > 0).tolist()
        self._values = {}
        if trainer.global_rank != 0:
            return
        metrics = {f"corrector_diag/{k}": v for k, v, ok in zip(self._keys, reduced, valid, strict=True) if ok}
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)
        for group in (*self._correctors, *(("_model",) if self.include_model else ())):
            summary = " ".join(
                f"{k.split('/', 1)[1]}={v:.3g}"
                for k, v in metrics.items()
                if k.startswith(f"corrector_diag/{group}/") and "/proc" not in k
            )
            LOGGER.info("CorrectorDiagnostics step %d [%s] %s", trainer.global_step, group, summary)


def _availability(source: "Source", names: list[str]) -> dict[str, float]:
    """Fractions of finite values of ``names`` in a target source: values, points and samples.

    A point is a grid point (gridded) or a row (tabular) with at least one finite channel; a
    sample is one with any finite value.
    """
    positions = [source.name_to_index[name] for name in names if name in source.name_to_index]
    tensors = source.data if isinstance(source.data, list) else [source.data]
    n_values = n_finite = n_points = n_finite_points = 0
    samples = []
    for tensor in tensors:
        finite = torch.isfinite(tensor.movedim(source.layout.axis("variables", ndim=tensor.ndim), -1)[..., positions])
        n_values += finite.numel()
        n_finite += int(finite.sum())
        points = finite.any(dim=-1)
        n_points += points.numel()
        n_finite_points += int(points.sum())
        if source.is_tabular:
            samples.append(bool(finite.any()))
        else:
            batch_axis = source.layout.axis("batch", ndim=tensor.ndim)
            samples += finite.movedim(batch_axis, 0).flatten(1).any(dim=1).tolist()
    return {
        "obs_frac": n_finite / max(n_values, 1),
        "obs_point_frac": n_finite_points / max(n_points, 1),
        "obs_sample_frac": sum(samples) / max(len(samples), 1),
    }
