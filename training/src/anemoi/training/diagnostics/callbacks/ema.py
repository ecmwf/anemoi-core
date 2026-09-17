# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exponential moving average (EMA) of the model weights.

The callback keeps one shadow copy of the model's floating-point parameters per requested
half-life.  Every optimiser step the shadow is pulled a little towards the raw weights, so the
shadow is a smoothed version of the training trajectory.  Whenever the checkpoint callback writes
an inference checkpoint, one extra inference checkpoint is written per half-life, holding the
shadow weights instead of the raw ones.

The callback is inert unless ``training.ema.enabled`` is true, and it never modifies the raw
inference checkpoint or the Lightning checkpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

import pytorch_lightning as pl
import torch

from anemoi.training.diagnostics.callbacks.checkpoint import get_inference_checkpoint_filepath
from anemoi.training.diagnostics.callbacks.checkpoint import save_inference_checkpoint
from anemoi.training.diagnostics.callbacks.checkpoint import torch_drop_down

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

# EDM-style half-life ramp: the effective half-life at optimiser step t is
# min(H, max(1, WARMUP_RATIO_DEFAULT * t)), so the average never reaches further back than a fixed
# fraction of the steps taken so far and is not anchored to a stale initial copy.
WARMUP_RATIO_DEFAULT = 0.2

DEFAULT_HALFLIFE_STEPS = (2000, 10000)


def _cfg(config: DictConfig, key: str, default: Any) -> Any:
    """Read ``training.ema.<key>`` from the config, falling back to ``default``."""
    training = getattr(config, "training", None)
    ema = getattr(training, "ema", None) if training is not None else None
    if ema is None:
        return default
    value = getattr(ema, key, default)
    return default if value is None else value


class EMAWeights(pl.Callback):
    """Maintain one exponential moving average of the model weights per half-life.

    Parameters
    ----------
    config : DictConfig
        The full job configuration.  Only the ``training.ema`` block is read.

    Notes
    -----
    Under DDP the parameters are identical on every rank, so every rank keeps its own shadow
    copies.  That is the simplest correct option: no collective is ever needed, resume works the
    same way on every rank, and the export path is guarded to global rank zero anyway.  The price
    is one extra copy of the model per half-life per rank.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        self.enabled: bool = bool(_cfg(config, "enabled", False))
        halflives = _cfg(config, "halflife_steps", DEFAULT_HALFLIFE_STEPS)
        self.halflife_steps: list[int] = [int(h) for h in halflives]
        self.warmup: bool = bool(_cfg(config, "warmup", True))
        self.warmup_ratio: float = float(_cfg(config, "warmup_ratio", WARMUP_RATIO_DEFAULT))
        self.device: str = str(_cfg(config, "device", "cuda"))
        self.update_every: int = int(_cfg(config, "update_every", 1))
        self.log_every: int = int(_cfg(config, "log_every", 1000))

        if self.enabled:
            if not self.halflife_steps:
                msg = "training.ema.enabled is true but training.ema.halflife_steps is empty."
                raise ValueError(msg)
            if min(self.halflife_steps) <= 0:
                msg = f"training.ema.halflife_steps must all be positive, got {self.halflife_steps}."
                raise ValueError(msg)
            if self.warmup and self.warmup_ratio <= 0:
                msg = f"training.ema.warmup_ratio must be positive, got {self.warmup_ratio}."
                raise ValueError(msg)
            if self.update_every < 1:
                msg = f"training.ema.update_every must be >= 1, got {self.update_every}."
                raise ValueError(msg)

        # shadow[halflife][parameter name] -> float32 tensor
        self._shadows: dict[int, dict[str, torch.Tensor]] = {}
        # foreach-able (shadow, parameter) pairs, and the slow per-parameter remainder
        self._fast: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._slow: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._num_updates: int = 0
        self._last_seen_global_step: int = -1
        self._pending_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _averaged_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
        """Return the floating-point parameters, which are the only ones that get averaged."""
        return [(name, p) for name, p in model.named_parameters() if p.is_floating_point()]

    def _shadow_device(self, param: torch.Tensor) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")
        return param.device

    def _initialise(self, model: torch.nn.Module, *, source: str) -> None:
        """Create the shadow copies from the model's current weights."""
        self._shadows = {}
        for halflife in self.halflife_steps:
            self._shadows[halflife] = {
                name: p.detach().to(device=self._shadow_device(p), dtype=torch.float32).clone()
                for name, p in self._averaged_parameters(model)
            }
        self._rebuild_views(model)
        n_params = sum(p.numel() for _, p in self._averaged_parameters(model))
        LOGGER.info(
            "EMAWeights: %s %d shadow copies of %d averaged parameters (%.3f GB each, float32) from %s.",
            "initialised" if source != "checkpoint" else "restored",
            len(self.halflife_steps),
            n_params,
            n_params * 4 / 1024**3,
            source,
        )

    def _rebuild_views(self, model: torch.nn.Module) -> None:
        """Split the (shadow, parameter) pairs into a foreach-able fast path and a slow path."""
        self._fast = {}
        self._slow = {}
        params = dict(self._averaged_parameters(model))
        for halflife, shadow in self._shadows.items():
            fast_shadow: list[torch.Tensor] = []
            fast_param: list[torch.Tensor] = []
            slow: list[tuple[torch.Tensor, torch.Tensor]] = []
            for name, tensor in shadow.items():
                param = params.get(name)
                if param is None:
                    continue
                if param.dtype == tensor.dtype and param.device == tensor.device:
                    fast_shadow.append(tensor)
                    fast_param.append(param.detach())
                else:
                    slow.append((tensor, param))
            self._fast[halflife] = (fast_shadow, fast_param)
            self._slow[halflife] = slow

    def effective_halflife(self, halflife: int, step: int) -> float:
        """The half-life actually applied at optimiser step ``step``.

        With warmup on it is ``min(H, max(1, warmup_ratio * step))`` (Karras et al. 2022, the EMA
        ramp-up): early in training the average only reaches back a fixed fraction of the steps taken,
        so it is not anchored to a stale initial copy, and once ``warmup_ratio * step`` exceeds ``H`` the
        nominal half-life applies.  Without warmup the nominal half-life applies from the first step.
        """
        if not self.warmup:
            return float(halflife)
        return min(float(halflife), max(1.0, self.warmup_ratio * step))

    def decay_for(self, halflife: int, step: int) -> float:
        """The decay applied by one update at optimiser step ``step`` for the given half-life.

        The per-step decay for an effective half-life ``h`` is ``0.5 ** (1 / h)``.  When several
        optimiser steps pass between updates the decay is raised to ``update_every`` so the half-life
        stays expressed in optimiser steps.
        """
        decay = 0.5 ** (1.0 / self.effective_halflife(halflife, step))
        return decay**self.update_every

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        """Pull every shadow copy one update towards the model's current weights."""
        for halflife in self.halflife_steps:
            weight = 1.0 - self.decay_for(halflife, step)
            fast_shadow, fast_param = self._fast[halflife]
            if fast_shadow:
                torch._foreach_lerp_(fast_shadow, fast_param, weight)
            for tensor, param in self._slow[halflife]:
                tensor.lerp_(param.detach().to(device=tensor.device, dtype=tensor.dtype), weight)
        self._num_updates += 1

    @torch.no_grad()
    def relative_distance(self, model: torch.nn.Module, halflife: int) -> float:
        """RMS relative distance between the shadow weights and the raw weights."""
        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        shadow = self._shadows[halflife]
        for name, param in self._averaged_parameters(model):
            tensor = shadow.get(name)
            if tensor is None:
                continue
            raw = param.detach().to(device=tensor.device, dtype=torch.float32)
            num += (tensor - raw).pow(2).sum().double().cpu()
            den += raw.pow(2).sum().double().cpu()
        if den.item() == 0.0:
            return 0.0
        return float((num / den).sqrt().item())

    # ------------------------------------------------------------------ lightning hooks

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not self.enabled:
            return
        model = torch_drop_down(trainer)
        if self._pending_state is not None:
            self._adopt_pending_state(model)
            return
        if getattr(trainer, "ckpt_path", None):
            LOGGER.warning(
                "EMAWeights: resuming from %s but the checkpoint carries no EMA state. "
                "Re-initialising the shadow copies from the current weights; the averages "
                "restart from this point.",
                trainer.ckpt_path,
            )
        self._initialise(model, source="the current weights")
        self._last_seen_global_step = trainer.global_step

    def _adopt_pending_state(self, model: torch.nn.Module) -> None:
        """Move the tensors loaded from a checkpoint onto the right devices."""
        state = self._pending_state
        self._pending_state = None
        shadows: dict[int, dict[str, torch.Tensor]] = {}
        params = dict(self._averaged_parameters(model))
        missing: list[str] = []
        for halflife, tensors in state["shadows"].items():
            shadows[halflife] = {}
            for name, tensor in tensors.items():
                param = params.get(name)
                if param is None:
                    missing.append(name)
                    continue
                shadows[halflife][name] = tensor.to(device=self._shadow_device(param), dtype=torch.float32)
            # any parameter the checkpoint did not carry starts from the current weights
            for name, param in params.items():
                if name not in shadows[halflife]:
                    shadows[halflife][name] = param.detach().to(
                        device=self._shadow_device(param),
                        dtype=torch.float32,
                    )
        self._shadows = shadows
        self._num_updates = int(state.get("num_updates", 0))
        self._last_seen_global_step = int(state.get("last_seen_global_step", -1))
        self._rebuild_views(model)
        if missing:
            LOGGER.warning(
                "EMAWeights: the checkpoint carried %d shadow entries the model does not have "
                "(first: %s); they were dropped.",
                len(missing),
                missing[0],
            )
        LOGGER.info(
            "EMAWeights: restored EMA state from the checkpoint — %d updates already applied, "
            "half-lives %s. The shadow copies were NOT re-initialised.",
            self._num_updates,
            sorted(self._shadows),
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if not self.enabled or not self._shadows:
            return
        step = trainer.global_step
        # global_step advances exactly once per optimiser step, so this respects accum_grad_batches
        # without having to reproduce Lightning's accumulation bookkeeping.
        if step == self._last_seen_global_step:
            return
        self._last_seen_global_step = step
        if self.update_every > 1 and step % self.update_every != 0:
            return

        model = torch_drop_down(trainer)
        self.update(model, step)

        if self.log_every > 0 and step % self.log_every == 0 and trainer.is_global_zero:
            distances = ", ".join(
                f"h{h}(eff {self.effective_halflife(h, step):.0f})={self.relative_distance(model, h):.3e}"
                for h in self.halflife_steps
            )
            LOGGER.info("EMAWeights: step %d, RMS relative distance EMA vs raw: %s", step, distances)

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "num_updates": self._num_updates,
            "last_seen_global_step": self._last_seen_global_step,
            "halflife_steps": list(self.halflife_steps),
            "warmup": self.warmup,
            "warmup_ratio": self.warmup_ratio,
            "update_every": self.update_every,
            "shadows": {
                str(halflife): {name: tensor.detach().to("cpu").clone() for name, tensor in tensors.items()}
                for halflife, tensors in self._shadows.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.enabled:
            return
        shadows = state_dict.get("shadows") if state_dict else None
        if not shadows:
            LOGGER.warning(
                "EMAWeights: the checkpoint contains no EMA shadow weights. The shadow copies will "
                "be re-initialised from the model weights at the start of training.",
            )
            self._pending_state = None
            return

        loaded = {int(halflife): tensors for halflife, tensors in shadows.items()}
        wanted = set(self.halflife_steps)
        if set(loaded) != wanted:
            LOGGER.warning(
                "EMAWeights: the checkpoint carries half-lives %s but the config asks for %s. "
                "Half-lives present in both are restored; the rest are initialised from the "
                "current weights.",
                sorted(loaded),
                sorted(wanted),
            )
        kept = {halflife: loaded[halflife] for halflife in self.halflife_steps if halflife in loaded}
        if not kept:
            LOGGER.warning(
                "EMAWeights: no requested half-life was found in the checkpoint; re-initialising.",
            )
            self._pending_state = None
            return
        self._pending_state = {
            "shadows": kept,
            "num_updates": state_dict.get("num_updates", 0),
            "last_seen_global_step": state_dict.get("last_seen_global_step", -1),
        }

    # ------------------------------------------------------------------ export

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del pl_module, checkpoint
        if not self.enabled or not self._shadows:
            return
        if not trainer.is_global_zero:
            return

        lightning_filepath = self._find_lightning_filepath(trainer)
        if lightning_filepath is None:
            LOGGER.debug(
                "EMAWeights: trainer.save_checkpoint was not called from an AnemoiCheckpoint; "
                "no EMA inference checkpoint written.",
            )
            return

        model = torch_drop_down(trainer)
        for halflife in self.halflife_steps:
            filepath = get_inference_checkpoint_filepath(lightning_filepath, prefix=f"inference-ema{halflife}-")
            try:
                self._export_one(model, halflife, filepath)
            except Exception:
                LOGGER.exception("EMAWeights: failed to write the EMA inference checkpoint %s.", filepath)

    @staticmethod
    def _find_lightning_filepath(trainer: pl.Trainer) -> str | None:
        """Find the Lightning checkpoint path the AnemoiCheckpoint callback is currently writing."""
        from anemoi.training.diagnostics.callbacks.checkpoint import AnemoiCheckpoint

        for callback in trainer.callbacks:
            if isinstance(callback, AnemoiCheckpoint):
                filepath = getattr(callback, "saving_lightning_checkpoint_filepath", None)
                if filepath is not None:
                    return filepath
        return None

    @torch.no_grad()
    def _export_one(self, model: torch.nn.Module, halflife: int, filepath: str) -> None:
        """Write one EMA inference checkpoint, leaving the model's weights exactly as found."""
        shadow = self._shadows[halflife]
        raw: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for name, param in self._averaged_parameters(model):
                tensor = shadow.get(name)
                if tensor is None:
                    continue
                raw.append((param, param.detach().clone()))
                param.detach().copy_(tensor.to(device=param.device, dtype=param.dtype))
            save_inference_checkpoint(
                model,
                filepath,
                extra_metadata={
                    "ema_halflife_steps": halflife,
                    "ema_num_updates": self._num_updates,
                    "ema_warmup": self.warmup,
                },
            )
            LOGGER.info("EMAWeights: wrote the half-life %d inference checkpoint to %s.", halflife, filepath)
        finally:
            for param, values in raw:
                param.detach().copy_(values)
