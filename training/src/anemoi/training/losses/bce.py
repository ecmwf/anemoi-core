# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import TensorLayout
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.base import _tabular_shard_group

if TYPE_CHECKING:
    from anemoi.models.data import Source

LOGGER = logging.getLogger(__name__)


class BinaryCrossEntropyLoss(FunctionalLoss):
    """Binary cross-entropy with an optional focal term, for predicted probabilities."""

    name: str = "bce"

    def __init__(
        self,
        focal_gamma: float = 0.0,
        alpha: float | None = None,
        pos_weight: float = 1.0,
        eps: float = 1e-6,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        r"""Binary cross-entropy (BCE) loss with an optional focal term.

        The prediction is a probability in [0, 1], for example the output of a sigmoid bounding,
        not a logit. With target :math:`y` and clamped prediction :math:`p`, the pointwise loss is

        .. math::

            L = -w_+ \, y \, (1 - p)^\gamma \log p - w_- \, (1 - y) \, p^\gamma \log (1 - p)

        with :math:`w_+ = \alpha \cdot \mathrm{pos\_weight}` and :math:`w_- = 1 - \alpha`
        (both :math:`1` apart from ``pos_weight`` when ``alpha`` is not set).
        With the defaults (``focal_gamma=0``, ``alpha=None``, ``pos_weight=1``) this is the plain BCE,
        equal to ``torch.nn.functional.binary_cross_entropy`` for predictions in :math:`[\epsilon, 1 - \epsilon]`.

        Warnings
        --------
        Only the plain BCE (``focal_gamma=0``, ``alpha=None``, ``pos_weight=1``) is a proper scoring rule,
        whose minimiser is the calibrated probability. With ``focal_gamma > 0``, ``alpha`` or
        ``pos_weight != 1`` the trained output is a score that ranks points, not a calibrated probability:
        choose a decision threshold on validation data and do not read the output as a probability.

        The loss is computed in float32 with autocast disabled, using explicit logarithms (the fused
        ``torch.nn.functional.binary_cross_entropy`` is not allowed under CUDA autocast).
        The prediction is clamped to :math:`[\epsilon, 1 - \epsilon]`, so saturated predictions (exactly 0 or 1)
        give a finite loss, and a zero gradient on the clamped side.

        Points where the target is NaN are always excluded, whatever ``ignore_nans``, because the BCE of a
        zero-filled point is not zero. Excluded points contribute nothing to the loss and are also removed from
        its normalisation: the scaled loss is divided by the fraction of scaler weight that belongs to valid
        points, so the result is the weighted mean over the valid points only.
        With ``ignore_nans=True``, points where the prediction is NaN are excluded as well, otherwise a NaN
        prediction propagates to the loss.

        Parameters
        ----------
        focal_gamma : float, optional
            Focusing parameter :math:`\gamma \ge 0` of the focal loss, by default 0.0 (plain BCE).
        alpha : float | None, optional
            Class balance in [0, 1]: weight :math:`\alpha` for positives and :math:`1 - \alpha` for negatives.
            By default None (no class balancing).
        pos_weight : float, optional
            Extra factor > 0 on the positive term, by default 1.0.
        eps : float, optional
            Clamp of the prediction to [eps, 1 - eps], by default 1e-6.
        ignore_nans : bool, optional
            Also exclude points where the prediction is NaN, by default False.
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if focal_gamma < 0:
            msg = f"focal_gamma must be >= 0, got {focal_gamma}."
            raise ValueError(msg)
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in [0, 1], got {alpha}."
            raise ValueError(msg)
        if pos_weight <= 0:
            msg = f"pos_weight must be > 0, got {pos_weight}."
            raise ValueError(msg)
        if not 0.0 < eps < 0.5:
            msg = f"eps must be in (0, 0.5), got {eps}."
            raise ValueError(msg)
        self.focal_gamma = float(focal_gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.pos_weight = float(pos_weight)
        self.eps = float(eps)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the pointwise BCE or focal loss.

        NaN values are not handled here, see ``_forward_impl``.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, shape (bs, time, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target in [0, 1], shape (bs, time, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Pointwise loss in float32
        """
        with torch.autocast(device_type=pred.device.type, enabled=False):
            p = pred.float().clamp(self.eps, 1.0 - self.eps)
            y = target.float()

            pos = -torch.log(p)
            neg = -torch.log1p(-p)
            if self.focal_gamma != 0.0:
                pos = pos * (1.0 - p) ** self.focal_gamma
                neg = neg * p**self.focal_gamma

            w_pos = self.pos_weight if self.alpha is None else self.pos_weight * self.alpha
            w_neg = 1.0 if self.alpha is None else 1.0 - self.alpha
            return w_pos * y * pos + w_neg * (1.0 - y) * neg

    def _apply_pairwise(
        self,
        pred: "Source",
        target: "Source",
        func: Any,
        *,
        group: ProcessGroup | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the loss to aligned sources, refusing tabular targets split across ``group``."""
        if target.is_tabular and _tabular_shard_group(target, group) is not None:
            msg = "BinaryCrossEntropyLoss does not support tabular targets that are split across ranks."
            raise NotImplementedError(msg)
        return super()._apply_pairwise(pred, target, func, group=group, **kwargs)

    def _forward_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        layout: TensorLayout,
        weights: torch.Tensor | None = None,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        valid_counts: torch.Tensor | None = None,  # noqa: ARG002
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss over the valid (non-NaN) points.

        Unlike the inherited implementation, the NaN mask is taken from ``pred`` and ``target`` before
        any value is replaced, and the loss is renormalised over the valid points. The renormalisation
        replaces the tabular ``valid_counts`` of the base class, so these are not used.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, laid out as ``layout``.
        target : torch.Tensor
            Target tensor, laid out as ``layout``.
        layout : TensorLayout
            Layout describing the logical axes of ``pred`` and ``target``.
        weights : torch.Tensor | None, optional
            Weights multiplying the pointwise loss before the scalers, by default None.
        squash : bool, optional
            Reduce the variable dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            List of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None.
        group : ProcessGroup, optional
            Distributed group, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        valid_counts : torch.Tensor | None, optional
            Unused, see above.
        **_kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Weighted loss.
        """
        if squash_mode not in ("avg", "sum"):
            msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
            raise ValueError(msg)

        group = group if grid_shard_slice is not None else None

        with torch.autocast(device_type=pred.device.type, enabled=False):
            # The mask must be taken before values are replaced: a zero-filled point is not neutral for BCE.
            valid = ~torch.isnan(target)
            if self.ignore_nans:
                valid &= ~torch.isnan(pred)
            pred = pred.float().masked_fill(~valid, 0.5)
            target = target.float().masked_fill(~valid, 0.0)

            out = torch.where(valid, self.calculate_difference(pred, target), 0.0)
            if weights is not None:
                out = out * weights

            def scaled_reduce(x: torch.Tensor) -> torch.Tensor:
                x = self.scale(
                    x,
                    scaler_indices,
                    layout=layout,
                    without_scalers=without_scalers,
                    grid_shard_slice=grid_shard_slice,
                )
                return self.reduce(x, layout=layout, squash=False, group=group)

            loss = scaled_reduce(out)
            with torch.no_grad():
                valid_weight = scaled_reduce(valid.to(out.dtype))
                total_weight = scaled_reduce(torch.ones_like(out))
                # Weighted mean over the valid points only; a variable without valid points gives 0.
                norm = torch.where(
                    valid_weight > 0,
                    total_weight / valid_weight.clamp_min(torch.finfo(out.dtype).tiny),
                    0,
                )
            loss = loss * norm

            if not squash:
                return loss
            return loss.mean() if squash_mode == "avg" else loss.sum()
