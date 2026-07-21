# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
from typing import NamedTuple
from typing import Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from packaging.version import Version
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import WeightAveraging as _PLWeightAveraging
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.optim.swa_utils import AveragedModel as _TorchAveragedModel
from torch.optim.swa_utils import get_ema_avg_fn
from torch.optim.swa_utils import get_swa_avg_fn

from anemoi.utils.parametrisation import HydraParametrisation

LOGGER = logging.getLogger(__name__)

MIN_PL_VERSION = "2.6.0"

_PARAMETRISATION = HydraParametrisation()


class _UpdateModelPlan(NamedTuple):
    """Tensors partitioned for updating averaged-model.

    ``avg_self`` and ``avg_model`` are parallel lists of (averaged-model, source-model)
    tensors that get averaged together. ``sync_pairs`` are (averaged-model, source-model).
    """

    avg_self: list[torch.Tensor]
    avg_model: list[torch.Tensor]
    sync_pairs: list[tuple[torch.Tensor, torch.Tensor]]


class AveragedModel(_TorchAveragedModel):
    """Variant of `torch.optim.swa_utils.AveragedModel` that pairs by name.

    The torch :meth:`AveragedModel.update_parameters` iterates parameters and buffers
    positionally via ``zip``. That breaks in anemoi for two reasons:

    1. Updating loss scalers (e.g. ``NaNMaskScaler``) call ``ScaleTensor.update_scaler``
       every batch, which pops and re-registers buffers — shuffling the buffer order in
       the live model while the averaged model's snapshot keeps the original order.
    2. Imputers (e.g. ``ConstantImputer``) register scratch buffers whose shapes change
       on the first forward pass.

    This subclass matches by name and skips entries whose shapes don't agree.

    """

    def update_parameters(self, model: torch.nn.Module) -> None:
        avg_self, avg_model, sync_pairs = self._collect_pairs(model)
        if self.n_averaged > 0:
            self._apply_averaging(avg_self, avg_model)
        for b_avg, b_model in sync_pairs:
            b_avg.copy_(b_model)
        self.n_averaged += 1

    def _collect_pairs(self, model: torch.nn.Module) -> _UpdateModelPlan:
        """Partition tensors into (averaged, sync-only) pairs, matched by name.

        Parameters and floating-point buffers (when ``use_buffers`` is True) go into the
        averaging set. Non-float buffers and all buffers under ``use_buffers=False`` are
        sync'd from source — averaging bool/int tensors is meaningless and would crash
        SWA's foreach subtraction path.

        Also performs the ``n_averaged == 0`` bootstrap copy on the averaging set.
        """
        avg_params = dict(self.module.named_parameters())
        avg_buffers = dict(self.module.named_buffers())

        avg_self: list[torch.Tensor] = []
        avg_model: list[torch.Tensor] = []
        sync_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for name, p_model in model.named_parameters():
            p_avg = avg_params.get(name)
            if p_avg is None or p_avg.shape != p_model.shape:
                continue
            p_model_ = p_model.detach().to(p_avg.device)
            p_avg_det = p_avg.detach()
            avg_self.append(p_avg_det)
            avg_model.append(p_model_)
            if self.n_averaged == 0:
                p_avg_det.copy_(p_model_)

        for name, b_model in model.named_buffers():
            b_avg = avg_buffers.get(name)
            if b_avg is None or b_avg.shape != b_model.shape:
                continue
            b_model_ = b_model.detach().to(b_avg.device)
            b_avg_det = b_avg.detach()
            if self.use_buffers and b_avg.is_floating_point():
                avg_self.append(b_avg_det)
                avg_model.append(b_model_)
                if self.n_averaged == 0:
                    b_avg_det.copy_(b_model_)
            else:
                sync_pairs.append((b_avg_det, b_model_))

        return _UpdateModelPlan(avg_self, avg_model, sync_pairs)

    def _apply_averaging(self, avg_self: list[torch.Tensor], avg_model: list[torch.Tensor]) -> None:
        """Apply the averaging step to the averaging set, pairwise."""
        if self.multi_avg_fn is not None:
            msg = "anemoi AveragedModel supports a per-tensor avg_fn only, not multi_avg_fn."
            raise NotImplementedError(msg)
        avg_fn = self.avg_fn if self.avg_fn is not None else get_swa_avg_fn()
        for p_avg, p_model in zip(avg_self, avg_model, strict=True):
            n_averaged = self.n_averaged.to(p_avg.device)
            p_avg.copy_(avg_fn(p_avg, p_model, n_averaged))


class WeightAveraging(_PLWeightAveraging):
    """Replaces :class:`pytorch_lightning.callbacks.WeightAveraging` with name-based matching.

    Uses the :class:`AveragedModel` (from anemoi) instead of torch
    ``AveragedModel`` and overrides the swap/copy hooks to pair tensors by name.
    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:  # noqa: ARG002
        if stage != "fit":
            return
        device = self._device or pl_module.device
        if is_overridden("configure_model", pl_module):
            rank_zero_warn(
                "You're using the WeightAveraging callback with a model that overrides the "
                "configure_model callback. WeightAveraging doesn't support sharding model layers, "
                "so you may run out of memory.",
            )
            pl_module.configure_model()
        self._average_model = AveragedModel(
            model=pl_module,
            device=device,
            use_buffers=self._use_buffers,
            **self._kwargs,
        )

    def _swap_models(self, pl_module: "pl.LightningModule") -> None:
        assert self._average_model is not None
        avg_params = dict(self._average_model.module.named_parameters())
        for name, current_param in pl_module.named_parameters():
            avg_param = avg_params.get(name)
            if avg_param is None or avg_param.shape != current_param.shape:
                continue
            tmp = avg_param.data.clone()
            avg_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

        avg_buffers = dict(self._average_model.module.named_buffers())
        for name, current_buf in pl_module.named_buffers():
            avg_buf = avg_buffers.get(name)
            if avg_buf is None or avg_buf.shape != current_buf.shape:
                continue
            tmp = avg_buf.data.clone()
            avg_buf.data.copy_(current_buf.data)
            current_buf.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: "pl.LightningModule") -> None:
        assert self._average_model is not None
        avg_params = dict(self._average_model.module.named_parameters())
        for name, current_param in pl_module.named_parameters():
            avg_param = avg_params.get(name)
            if avg_param is None or avg_param.shape != current_param.shape:
                continue
            current_param.data.copy_(avg_param.data)

        avg_buffers = dict(self._average_model.module.named_buffers())
        for name, current_buf in pl_module.named_buffers():
            avg_buf = avg_buffers.get(name)
            if avg_buf is None or avg_buf.shape != current_buf.shape:
                continue
            current_buf.data.copy_(avg_buf.data)


class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average weight averaging.

    Mirrors :class:`pytorch_lightning.callbacks.EMAWeightAveraging`.
    """

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = True,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            **kwargs,
            avg_fn=get_ema_avg_fn(decay=decay),
        )
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch


class SWAWeightAveraging(WeightAveraging):
    """Stochastic Weight Averaging (running mean).

    Mirrors :class:`pytorch_lightning.callbacks.SWAWeightAveraging` but with name-based matching.
    Uses the default running-mean ``avg_fn`` from torch's ``AveragedModel``.
    """

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = True,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers, **kwargs)
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch


def _get_weight_averaging_callback(weight_averaging_config: DictConfig | None) -> list[Callback]:
    """Get weight averaging callback from the config.

    Example config (recommended):
        weight_averaging:
            _target_: anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging
            decay: 0.999

    Stock ``pytorch_lightning.callbacks.*WeightAveraging`` classes also work but
    are unsafe with anemoi imputers and updating loss scalers.

    Parameters
    ----------
    weight_averaging_config : DictConfig | None
        Weight averaging configuration (``config.training.weight_averaging``),
        or ``None`` if not configured.

    Returns
    -------
    list[Callback]
        List containing the weight averaging callback, or empty list if not configured.
    """
    if weight_averaging_config is None:
        LOGGER.debug("No weight averaging configured. Skipping.")
        return []
    if not isinstance(weight_averaging_config, dict | DictConfig):
        LOGGER.warning(
            "training.weight_averaging has unexpected type %s; expected a dict with '_target_'. Skipping.",
            type(weight_averaging_config).__name__,
        )
        return []
    if "_target_" not in weight_averaging_config:
        LOGGER.warning("training.weight_averaging is set but has no '_target_' field. Skipping.")
        return []

    if Version(pl.__version__) < Version(MIN_PL_VERSION):
        msg = (
            f"Weight averaging callback {weight_averaging_config['_target_']!r} requires "
            f"pytorch_lightning>={MIN_PL_VERSION}, but found {pl.__version__}. "
            f"Please upgrade pytorch_lightning to use this callback."
        )
        raise RuntimeError(msg)

    callback = _PARAMETRISATION.create_module(weight_averaging_config)
    LOGGER.info("Loaded weight averaging callback: %s", weight_averaging_config["_target_"])

    if isinstance(callback, _PLWeightAveraging) and not isinstance(callback, WeightAveraging):
        LOGGER.warning(
            "Configured weight-averaging callback %r is from stock pytorch_lightning. "
            "This is known to crash or silently mis-pair tensors when used with "
            "imputers (e.g. ConstantImputer) or updating loss scalers (e.g. NaNMaskScaler). "
            "Use anemoi.training.diagnostics.callbacks.weight_averaging."
            "EMAWeightAveraging (or SWAWeightAveraging) instead.",
            type(callback).__name__,
        )

    return [callback]
