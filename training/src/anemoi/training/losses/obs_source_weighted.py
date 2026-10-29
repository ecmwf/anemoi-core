# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MSE with per-element weights by observation source (AMV / aircraft / radiosonde) of the target."""

from __future__ import annotations

import functools
import logging
import re
from typing import TYPE_CHECKING
from typing import ClassVar

import torch
import torch.distributed as dist

from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.mse import MSELoss
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)

SOURCES = ("amv", "aircraft", "sonde")
# Gridded upper-air report types (observations-combined v6-rpt `sonde_reportype`).
DEFAULT_REPORTYPE_CODES = {"amv": [1.0, 1.5], "aircraft": [2.0], "sonde": [2.5, 3.0]}
LEVEL_VARIABLE = re.compile(r"^([A-Za-z0-9]+)_(\d+)$")


class ObsSourceWeightedMSELoss(MSELoss):
    """MSE whose upper-air target elements are weighted by the source that observed them.

    The source of a target element of ``{var}_{level}`` (``var`` in ``variables``) is taken from the
    cell's report-type code when ``reportype_variable`` is given (a loss-only ``target`` variable
    with normaliser ``none``, e.g. ``sonde_reportype``; codes mapped by ``reportype_codes``).
    Where there is no code, it is inferred from what else the cell reports at that time, since each
    source leaves a distinct pattern: radiosondes and dropsondes report ``z`` (on some level),
    aircraft report ``t`` without ``z`` and AMVs report winds only:

    - ``sonde``: any ``z_*`` of the weighted levels present in the column;
    - ``aircraft``: no ``z``, ``t_{level}`` present;
    - ``amv``: neither.

    The inference alone misclassifies wind-only sonde (PILOT-type) levels as AMV, which matters
    before ~2000, so prefer the report-type code. The squared error is multiplied by
    ``weights[source]``. Everything else (other variables,
    surface variables, missing targets) gets weight 1. The weights depend only on the targets, so
    the inputs, and thus what the model sees, are unchanged: down-weighting AMV targets keeps AMVs
    fully available as inputs. A cell holding several sources at one level is classified by what
    the gridding kept there.

    With ``preserve_variable_share`` the weights of each weighted variable are divided by their
    mean over its valid elements in the batch, so the variable's loss share (as set by
    ``general_variable``) is kept and only the mix of sources within it changes.

    The target may come in DATA_FULL layout (single-model step) or DATA_OUTPUT layout
    (``DASingleTraining``); with ``reportype_variable`` it must be a ``target`` variable to be in
    the latter. The loss reweights validation loss too, so validation losses are not comparable
    with runs using plain MSE.

    Example config (inside a ``CombinedLoss``):

    .. code-block:: yaml

        - _target_: anemoi.training.losses.ObsSourceWeightedMSELoss
          scalers: [general_variable, node_weights]
          ignore_nans: true
          variables: [u, v]
          weights: {amv: 0.3, aircraft: 1.0, sonde: 2.0}
          reportype_variable: sonde_reportype  # under data.<dataset>.target, normaliser none
          preserve_variable_share: true
    """

    name: str = "obs_source_mse"
    factory_context_keys: ClassVar = frozenset({LossFactoryContextKey.DATA_INDICES, LossFactoryContextKey.NORMALIZER})

    def __init__(
        self,
        *,
        data_indices: IndexCollection,
        variables: list[str] | tuple[str, ...] = ("u", "v"),
        weights: dict[str, float] | None = None,
        reportype_variable: str | None = None,
        reportype_codes: dict[str, list[float]] | None = None,
        preserve_variable_share: bool = True,
        ignore_nans: bool = True,
        normalizer: object | None = None,
    ) -> None:
        """Initialise the loss.

        Parameters
        ----------
        data_indices : IndexCollection
            Dataset index collection (injected by the loss factory).
        variables : list[str]
            Variable families to weight, e.g. ``[u, v]``; every ``{family}_{level}`` model output is
            weighted. ``t`` and ``z`` may be added (``t`` is then split into sonde/aircraft).
        weights : dict[str, float]
            Weight per source (``amv``, ``aircraft``, ``sonde``); missing sources default to 1.
        reportype_variable : str, optional
            Per-cell report-type code in the DATA_FULL target (normaliser ``none``). Without it the
            source is inferred from the target pattern only.
        reportype_codes : dict[str, list[float]], optional
            Codes per source, by default ``amv: [1, 1.5], aircraft: [2], sonde: [2.5, 3]``.
        preserve_variable_share : bool
            Renormalise each weighted variable's weights to mean 1 over its valid elements.
        ignore_nans : bool
            Mask NaN targets, as for ``MSELoss`` (required for sparse observation targets).
        normalizer : object, optional
            Dataset input normaliser (injected); used to check that ``reportype_variable`` is raw.
        """
        super().__init__(ignore_nans=ignore_nans)
        weights = dict(weights or {})
        unknown = sorted(set(weights) - set(SOURCES))
        if unknown:
            msg = f"{self.__class__.__name__}: unknown sources {unknown}, expected {list(SOURCES)}."
            raise ValueError(msg)
        if any(float(w) < 0 for w in weights.values()):
            msg = f"{self.__class__.__name__}: weights must be non-negative, got {weights}."
            raise ValueError(msg)
        self.source_weights = {source: float(weights.get(source, 1.0)) for source in SOURCES}
        self.families = list(variables)
        self.preserve_variable_share = preserve_variable_share

        model_names = list(data_indices.model.output.ordered_names)
        # VARIABLE scalers such as general_variable are laid out like data.output, which also holds the
        # loss-only `target` variables; keep the model-output positions within it.
        output_pos = data_indices.data.output.name_to_position
        self._model_in_data_output = [output_pos[name] for name in model_names]
        self._data_output_width = len(data_indices.data.output.full)
        full_pos = data_indices.data_full_name_to_position

        weighted, t_names, z_names, missing = [], [], [], []
        for position, name in enumerate(model_names):
            match = LEVEL_VARIABLE.match(name)
            if match is None or match.group(1) not in self.families:
                continue
            level = match.group(2)
            companions = (f"t_{level}", f"z_{level}")
            absent = [c for c in companions if c not in full_pos]
            if absent:
                missing.extend(absent)
                continue
            weighted.append(position)
            t_names.append(companions[0])
            z_names.append(companions[1])
        if missing:
            msg = (
                f"{self.__class__.__name__}: source classification needs t and z on every weighted level; "
                f"missing from the data: {sorted(set(missing))}."
            )
            raise ValueError(msg)
        if not weighted:
            msg = f"{self.__class__.__name__}: no model output matches families {self.families}."
            raise ValueError(msg)
        self.weighted_names = [model_names[p] for p in weighted]

        self._setup_reportype(reportype_variable, reportype_codes, data_indices, normalizer)
        self.register_buffer("weighted_pos", torch.tensor(weighted), persistent=False)
        self._output_width = len(model_names)
        columns = {"target": model_names, "t": t_names, "z": z_names}
        if reportype_variable is not None:
            columns["rpt"] = [reportype_variable]
        self._register_layouts(columns, data_indices)
        self.last_source_fractions: dict[str, float] | None = None
        self._logged = False
        LOGGER.info(
            "%s: weights %s on %d variables (%s), sources from %s, preserve_variable_share=%s",
            self.__class__.__name__,
            self.source_weights,
            len(weighted),
            ", ".join(self.families),
            f"{reportype_variable} {self.reportype_codes}" if reportype_variable else "the target pattern",
            preserve_variable_share,
        )

    def _register_layouts(self, columns: dict[str, list[str]], data_indices: IndexCollection) -> None:
        """Column positions read by the classification, per target layout that carries them all.

        Targets arrive in DATA_FULL (single-model step) or DATA_OUTPUT (``DASingleTraining`` slices them).
        """
        self._layout_widths: dict[IndexSpace, int] = {}
        for layout, positions, width in (
            (IndexSpace.DATA_FULL, data_indices.data_full_name_to_position, len(data_indices.name_to_index)),
            (IndexSpace.DATA_OUTPUT, data_indices.data.output.name_to_position, self._data_output_width),
        ):
            if any(name not in positions for names in columns.values() for name in names):
                continue
            self._layout_widths[layout] = width
            for key, names in columns.items():
                self.register_buffer(
                    f"_{layout.value}_{key}",
                    torch.tensor([positions[n] for n in names]),
                    persistent=False,
                )

    def _setup_reportype(
        self,
        reportype_variable: str | None,
        reportype_codes: dict[str, list[float]] | None,
        data_indices: IndexCollection,
        normalizer: object | None,
    ) -> None:
        self.reportype_variable = reportype_variable
        codes = DEFAULT_REPORTYPE_CODES if reportype_codes is None else dict(reportype_codes)
        unknown = sorted(set(codes) - set(SOURCES))
        if unknown:
            msg = f"{self.__class__.__name__}: unknown sources in reportype_codes {unknown}."
            raise ValueError(msg)
        self.reportype_codes = {source: [float(c) for c in codes.get(source, [])] for source in SOURCES}
        if reportype_variable is None:
            return
        full_pos = data_indices.data_full_name_to_position
        if reportype_variable not in full_pos:
            msg = (
                f"{self.__class__.__name__}: reportype_variable {reportype_variable!r} is not in the data; "
                "select it and declare it under data.<dataset>.target."
            )
            raise ValueError(msg)
        self._check_raw(reportype_variable, data_indices, normalizer)

    @staticmethod
    def _check_raw(name: str, data_indices: IndexCollection, normalizer: object | None) -> None:
        if normalizer is None or not hasattr(normalizer, "_norm_mul"):
            LOGGER.warning("ObsSourceWeightedMSELoss: cannot check that %s is not normalised.", name)
            return
        idx = int(data_indices.name_to_index[name])
        mul, add = float(normalizer._norm_mul[idx]), float(normalizer._norm_add[idx])
        if abs(mul - 1.0) > 1e-6 or abs(add) > 1e-6:
            msg = f"ObsSourceWeightedMSELoss: {name} must keep its raw codes; set its normaliser to 'none'."
            raise ValueError(msg)

    def _to_model_output(self, dimension: int | tuple[int], scaler: torch.Tensor) -> torch.Tensor:
        """Restrict a data.output-wide VARIABLE scaler to the model outputs."""
        dims = (dimension,) if isinstance(dimension, int) else tuple(dimension)
        if TensorDim.VARIABLE not in dims:
            return scaler
        axis = dims.index(TensorDim.VARIABLE)
        width = scaler.shape[axis]
        if width == self._data_output_width != self._output_width:
            index = torch.as_tensor(self._model_in_data_output, device=scaler.device)
            return scaler.index_select(axis, index)
        return scaler

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        super().add_scaler(dimension, self._to_model_output(dimension, scaler), name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        if name in self.scaler.tensors:
            scaler = self._to_model_output(self.scaler.tensors[name][0], scaler)
        super().update_scaler(name, scaler, override=override)

    def _resolve_layout(self, target: torch.Tensor, layout: IndexSpace | str | None) -> IndexSpace:
        if layout is None:
            matches = [lay for lay, width in self._layout_widths.items() if width == target.shape[-1]]
            if len(matches) != 1:
                msg = (
                    f"{self.__class__.__name__}: cannot infer the target layout from width {target.shape[-1]}; "
                    f"supported {({lay.value: w for lay, w in self._layout_widths.items()})}."
                )
                raise ValueError(msg)
            return matches[0]
        layout = IndexSpace(str(layout))
        if self._layout_widths.get(layout) != target.shape[-1]:
            msg = (
                f"{self.__class__.__name__}: target layout {layout.value!r} with width {target.shape[-1]} is not "
                f"supported; supported {({lay.value: w for lay, w in self._layout_widths.items()})}."
            )
            raise ValueError(msg)
        return layout

    def _pos(self, layout: IndexSpace, key: str) -> torch.Tensor:
        return getattr(self, f"_{layout.value}_{key}")

    def source_masks(
        self,
        target: torch.Tensor,
        layout: IndexSpace | str | None = IndexSpace.DATA_FULL,
    ) -> dict[str, torch.Tensor]:
        """Boolean masks ``(..., n_weighted)`` of the valid weighted targets per source."""
        layout = self._resolve_layout(target, layout)
        valid = torch.isfinite(target.index_select(-1, self._pos(layout, "target")[self.weighted_pos]))
        column_z = torch.isfinite(target.index_select(-1, self._pos(layout, "z"))).any(dim=-1, keepdim=True)
        has_t = torch.isfinite(target.index_select(-1, self._pos(layout, "t")))
        inferred = {"sonde": column_z, "aircraft": ~column_z & has_t, "amv": ~column_z & ~has_t}
        if self.reportype_variable is None:
            return {source: valid & inferred[source] for source in SOURCES}

        code = target.index_select(-1, self._pos(layout, "rpt"))  # (..., 1), broadcast over levels
        known = {
            source: torch.isin(code, torch.tensor(codes, dtype=code.dtype, device=code.device))
            for source, codes in self.reportype_codes.items()
        }
        coded = known["amv"] | known["aircraft"] | known["sonde"]
        return {source: valid & torch.where(coded, known[source], inferred[source]) for source in SOURCES}

    def element_weights(
        self,
        target: torch.Tensor,
        group: ProcessGroup | None = None,
        layout: IndexSpace | str | None = IndexSpace.DATA_FULL,
    ) -> torch.Tensor:
        """Weights ``(..., n_weighted)`` for the weighted variables (1 where no target)."""
        masks = self.source_masks(target, layout)
        weights = torch.ones(masks["amv"].shape, dtype=torch.float32, device=target.device)
        for source, mask in masks.items():
            weights = torch.where(mask, self.source_weights[source], weights)

        valid = masks["amv"] | masks["aircraft"] | masks["sonde"]
        lead_dims = tuple(range(valid.dim() - 1))
        counts = torch.stack([masks[s].sum(dim=lead_dims) for s in SOURCES]).to(torch.float32)  # (3, n_weighted)
        if self.preserve_variable_share:
            totals = torch.stack([(weights * valid).sum(dim=lead_dims), valid.sum(dim=lead_dims).to(torch.float32)])
            if group is not None:
                dist.all_reduce(totals, group=group)
                dist.all_reduce(counts, group=group)
            mean = totals[0] / totals[1].clamp_min(1.0)
            weights = torch.where(mean > 0, weights / mean.clamp_min(torch.finfo(torch.float32).tiny), weights)
        elif group is not None:
            dist.all_reduce(counts, group=group)

        total = counts.sum().clamp_min(1.0)
        self.last_source_fractions = {s: float(c) for s, c in zip(SOURCES, counts.sum(dim=1) / total, strict=True)}
        if not self._logged:
            LOGGER.info("%s: first-batch source fractions %s", self.__class__.__name__, self.last_source_fractions)
            self._logged = True
        return weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Source-weighted MSE.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction ``(bs, time, ens, grid, n_model_outputs)`` in MODEL_OUTPUT layout.
        target : torch.Tensor
            Target ``(bs, time, ens, grid, n)`` in DATA_FULL or DATA_OUTPUT layout.
        squash, scaler_indices, without_scalers, grid_shard_slice, group, squash_mode
            As for :class:`MSELoss`.
        pred_layout, target_layout : IndexSpace | str | None
            MODEL_OUTPUT, and DATA_FULL or DATA_OUTPUT (inferred from the width when not given).

        Returns
        -------
        torch.Tensor
            Scalar loss, or one value per model output when ``squash`` is False.
        """
        pred_layout = None if pred_layout is None else IndexSpace(str(pred_layout))
        if pred_layout not in (None, IndexSpace.MODEL_OUTPUT) or pred.shape[-1] != self._output_width:
            msg = f"{self.__class__.__name__} expects pred in MODEL_OUTPUT layout (width {self._output_width})."
            raise ValueError(msg)
        layout = self._resolve_layout(target, target_layout)

        is_sharded = grid_shard_slice is not None
        with torch.no_grad():
            weights = self.element_weights(target, group=group if is_sharded else None, layout=layout)

        pred_m, target_m = self.mask_nans(pred, target.index_select(-1, self._pos(layout, "target")))
        out = self.calculate_difference(pred_m, target_m)
        factor = torch.ones(out.shape, dtype=out.dtype, device=out.device)
        factor = factor.index_copy(-1, self.weighted_pos, weights.to(out.dtype))
        out = out * factor
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if is_sharded else None, squash_mode=squash_mode)
