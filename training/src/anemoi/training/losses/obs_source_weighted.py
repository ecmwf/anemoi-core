# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MSE with per-element weights by the observing system (e.g. AMV / aircraft / radiosonde) of the target."""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import torch
import torch.distributed as dist

from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.base import Squash_mode
from anemoi.training.losses.base import _tabular_shard_group
from anemoi.training.losses.mse import MSELoss
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.models.data import Source
    from anemoi.models.data import TensorLayout
    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)

# Gridded upper-air report types (observations-combined v6-rpt `sonde_reportype`). The order is the
# inference priority: radiosondes and dropsondes report ``z`` (on some level of the column), aircraft
# report ``t`` without ``z``, and AMVs, the fallback, report winds only.
DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "sonde": {"weight": 1.0, "reportype_codes": [2.5, 3.0], "requires": ["z"], "requires_scope": "column"},
    "aircraft": {"weight": 1.0, "reportype_codes": [2.0], "requires": ["t"], "requires_scope": "level"},
    "amv": {"weight": 1.0, "reportype_codes": [1.0, 1.5]},
}
DEFAULT_LEVEL_PATTERN = r"^([A-Za-z0-9]+)_(\d+)$"

RequiresScope = Literal["level", "column"]


@dataclass(frozen=True)
class ObservationSource:
    """One observing system the loss can attribute a target element to.

    Parameters
    ----------
    name : str
        Source name, e.g. ``sonde``.
    weight : float
        Weight of the squared error of the elements it observed.
    reportype_codes : tuple[float, ...]
        Report-type codes identifying the source in ``reportype_variable``.
    requires : tuple[str, ...]
        Variable families the cell must report for the source to be inferred; empty for the fallback.
    requires_scope : {"level", "column"}
        ``level``: ``{family}_{level}`` at the weighted level; ``column``: at any weighted level.
    """

    name: str
    weight: float = 1.0
    reportype_codes: tuple[float, ...] = ()
    requires: tuple[str, ...] = ()
    requires_scope: RequiresScope = "level"

    @property
    def is_fallback(self) -> bool:
        return not self.requires


class ObsSourceWeightedMSELoss(MSELoss):
    """MSE whose upper-air target elements are weighted by the source that observed them.

    The weighted elements are the model outputs ``{family}_{level}`` (matching ``level_pattern``) with
    ``family`` in ``variables``. The source of such an element is taken from the cell's report-type
    code when ``reportype_variable`` is given (a loss-only ``target`` variable with normaliser
    ``none``, e.g. ``sonde_reportype``; codes per source in ``sources``). Where there is no code, it is
    inferred from what else the cell reports at that time: the first source in ``sources`` whose
    ``requires`` families are all present wins, and the one source without ``requires`` takes the rest.
    With the defaults:

    - ``sonde``: any ``z_*`` of the weighted levels present in the column;
    - ``aircraft``: no ``z``, ``t_{level}`` present;
    - ``amv``: neither.

    The inference alone misclassifies wind-only sonde (PILOT-type) levels as AMV, which matters
    before ~2000, so prefer the report-type code. The squared error is multiplied by the source's
    ``weight``. Everything else (other variables, surface variables, missing targets) gets weight 1.
    The weights depend only on the targets, so the inputs, and thus what the model sees, are
    unchanged: down-weighting AMV targets keeps AMVs fully available as inputs. A cell holding several
    sources at one level is classified by what the gridding kept there.

    With ``preserve_variable_share`` the weights of each weighted variable are divided by their mean
    over its valid elements in the batch, so the variable's loss share (as set by ``general_variable``)
    is kept and only the mix of sources within it changes.

    The target may come in DATA_FULL or DATA_OUTPUT layout, gridded or tabular; with
    ``reportype_variable`` it must be a ``target`` variable to be in the latter. The loss reweights the
    validation loss too, so validation losses are not comparable with runs using plain MSE.

    Example config (inside a ``CombinedLoss``):

    .. code-block:: yaml

        - _target_: anemoi.training.losses.ObsSourceWeightedMSELoss
          scalers: [general_variable, node_weights]
          ignore_nans: true
          variables: [u, v]
          reportype_variable: sonde_reportype  # under data.<dataset>.target, normaliser none
          sources:  # order = inference priority; exactly one source without `requires`
            sonde: {weight: 2.0, reportype_codes: [2.5, 3.0], requires: [z], requires_scope: column}
            aircraft: {weight: 1.0, reportype_codes: [2.0], requires: [t]}
            amv: {weight: 0.3, reportype_codes: [1.0, 1.5]}
          preserve_variable_share: true
    """

    name: str = "obs_source_mse"
    factory_context_keys: ClassVar = frozenset({LossFactoryContextKey.DATA_INDICES, LossFactoryContextKey.NORMALIZER})

    def __init__(
        self,
        *,
        data_indices: IndexCollection,
        variables: list[str] | tuple[str, ...] = ("u", "v"),
        sources: dict[str, dict[str, Any]] | None = None,
        weights: dict[str, float] | None = None,
        reportype_variable: str | None = None,
        reportype_codes: dict[str, list[float]] | None = None,
        level_pattern: str = DEFAULT_LEVEL_PATTERN,
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
        sources : dict[str, dict], optional
            Observing systems in inference priority order, each with ``weight``, ``reportype_codes``,
            ``requires`` and ``requires_scope`` (see :class:`ObservationSource`). Exactly one source
            has no ``requires``. Defaults to sonde / aircraft / AMV.
        weights : dict[str, float], optional
            Shorthand overriding the ``weight`` of named sources.
        reportype_variable : str, optional
            Per-cell report-type code in the target (normaliser ``none``). Without it the source is
            inferred from the target pattern only.
        reportype_codes : dict[str, list[float]], optional
            Shorthand overriding the ``reportype_codes`` of named sources.
        level_pattern : str
            Regular expression splitting a variable name into its family and level groups.
        preserve_variable_share : bool
            Renormalise each weighted variable's weights to mean 1 over its valid elements.
        ignore_nans : bool
            Mask NaN targets, as for ``MSELoss`` (required for sparse observation targets).
        normalizer : object, optional
            Dataset input normaliser (injected); used to check that ``reportype_variable`` is raw.
        """
        super().__init__(ignore_nans=ignore_nans)
        self.sources = self._build_sources(sources, weights, reportype_codes)
        self.source_weights = {source.name: source.weight for source in self.sources}
        self.reportype_codes = {source.name: list(source.reportype_codes) for source in self.sources}
        self.families = list(variables)
        self.preserve_variable_share = preserve_variable_share
        self.reportype_variable = reportype_variable

        model_names = list(data_indices.model.output.ordered_names)
        self._output_width = len(model_names)
        companions = sorted({family for source in self.sources for family in source.requires})
        weighted, columns = self._match_weighted(model_names, re.compile(level_pattern), companions, data_indices)
        self.weighted_names = [model_names[p] for p in weighted]
        self.register_buffer("weighted_pos", torch.tensor(weighted, dtype=torch.long), persistent=False)

        if reportype_variable is not None:
            self._check_reportype(reportype_variable, data_indices, normalizer)
            columns["rpt"] = [reportype_variable]
        columns["target"] = model_names
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

    def _build_sources(
        self,
        sources: dict[str, dict[str, Any]] | None,
        weights: dict[str, float] | None,
        reportype_codes: dict[str, list[float]] | None,
    ) -> tuple[ObservationSource, ...]:
        """Validate the configured sources and apply the ``weights`` / ``reportype_codes`` shorthands."""
        specs = copy.deepcopy(
            DEFAULT_SOURCES if sources is None else {name: dict(spec) for name, spec in sources.items()},
        )
        for option, overrides in (("weight", weights), ("reportype_codes", reportype_codes)):
            unknown = sorted(set(overrides or {}) - set(specs))
            if unknown:
                msg = f"{self.__class__.__name__}: unknown sources {unknown}, expected {list(specs)}."
                raise ValueError(msg)
            for name, value in (overrides or {}).items():
                specs[name][option] = value

        built = []
        for name, spec in specs.items():
            unknown_keys = sorted(set(spec) - {"weight", "reportype_codes", "requires", "requires_scope"})
            if unknown_keys:
                msg = f"{self.__class__.__name__}: unknown options {unknown_keys} for source {name!r}."
                raise ValueError(msg)
            source = ObservationSource(
                name=name,
                weight=float(spec.get("weight", 1.0)),
                reportype_codes=tuple(float(c) for c in spec.get("reportype_codes") or ()),
                requires=tuple(spec.get("requires") or ()),
                requires_scope=spec.get("requires_scope", "level"),
            )
            if source.weight < 0:
                msg = f"{self.__class__.__name__}: weights must be non-negative, got {source.weight} for {name!r}."
                raise ValueError(msg)
            if source.requires_scope not in ("level", "column"):
                msg = (
                    f"{self.__class__.__name__}: requires_scope of {name!r} must be 'level' or 'column', "
                    f"got {source.requires_scope!r}."
                )
                raise ValueError(msg)
            built.append(source)

        fallbacks = [source.name for source in built if source.is_fallback]
        if len(fallbacks) != 1:
            msg = (
                f"{self.__class__.__name__}: exactly one source must have no `requires` (the fallback), "
                f"got {fallbacks}."
            )
            raise ValueError(msg)
        codes = [code for source in built for code in source.reportype_codes]
        if len(codes) != len(set(codes)):
            msg = f"{self.__class__.__name__}: report-type codes must identify one source each, got {specs}."
            raise ValueError(msg)
        return tuple(built)

    def _match_weighted(
        self,
        model_names: list[str],
        level_pattern: re.Pattern,
        companions: list[str],
        data_indices: IndexCollection,
    ) -> tuple[list[int], dict[str, list[str]]]:
        """Model-output positions of the weighted variables and, per companion family, the names read at their level."""
        full_pos = data_indices.data_full_name_to_position
        weighted, missing = [], []
        columns: dict[str, list[str]] = {f"req_{family}": [] for family in companions}
        for position, name in enumerate(model_names):
            match = level_pattern.match(name)
            if match is None or match.group(1) not in self.families:
                continue
            level = match.group(2)
            names = [f"{family}_{level}" for family in companions]
            absent = [n for n in names if n not in full_pos]
            if absent:
                missing.extend(absent)
                continue
            weighted.append(position)
            for family, companion in zip(companions, names, strict=True):
                columns[f"req_{family}"].append(companion)
        if missing:
            msg = (
                f"{self.__class__.__name__}: source classification needs {companions} on every weighted level; "
                f"missing from the data: {sorted(set(missing))}."
            )
            raise ValueError(msg)
        if not weighted:
            msg = f"{self.__class__.__name__}: no model output matches families {self.families}."
            raise ValueError(msg)
        return weighted, columns

    def _register_layouts(self, columns: dict[str, list[str]], data_indices: IndexCollection) -> None:
        """Column positions read by the classification, per target layout that carries them all."""
        self._layout_widths: dict[IndexSpace, int] = {}
        for layout, positions, width in (
            (IndexSpace.DATA_FULL, data_indices.data_full_name_to_position, len(data_indices.name_to_index)),
            (IndexSpace.DATA_OUTPUT, data_indices.data.output.name_to_position, len(data_indices.data.output.full)),
        ):
            if any(name not in positions for names in columns.values() for name in names):
                continue
            self._layout_widths[layout] = width
            for key, names in columns.items():
                self.register_buffer(
                    f"_{layout.value}_{key}",
                    torch.tensor([positions[n] for n in names], dtype=torch.long),
                    persistent=False,
                )

    def _check_reportype(self, name: str, data_indices: IndexCollection, normalizer: object | None) -> None:
        if name not in data_indices.data_full_name_to_position:
            msg = (
                f"{self.__class__.__name__}: reportype_variable {name!r} is not in the data; "
                "select it and declare it under data.<dataset>.target."
            )
            raise ValueError(msg)
        if normalizer is None or not hasattr(normalizer, "methods") or not hasattr(normalizer, "default"):
            LOGGER.warning("%s: cannot check that %s is not normalised.", self.__class__.__name__, name)
            return
        method = normalizer.methods.get(name, normalizer.default)
        if method != "none":
            msg = (
                f"{self.__class__.__name__}: {name} must keep its raw codes, but is normalised with {method!r}; "
                "set its normaliser to 'none'."
            )
            raise ValueError(msg)

    def _resolve_layout(self, width: int, layout: IndexSpace | str | None) -> IndexSpace:
        supported = {lay.value: w for lay, w in self._layout_widths.items()}
        if layout is None:
            # data.output is a sorted subset of the full data, so equal widths mean identical index
            # spaces (no forcings): the first match (DATA_FULL) is then as good as any.
            matches = [lay for lay, w in self._layout_widths.items() if w == width]
            if not matches:
                msg = (
                    f"{self.__class__.__name__}: cannot infer the target layout from width {width}; "
                    f"supported {supported}."
                )
                raise ValueError(msg)
            return matches[0]
        layout = IndexSpace(str(layout))
        if self._layout_widths.get(layout) != width:
            msg = (
                f"{self.__class__.__name__}: target layout {layout.value!r} with width {width} is not supported; "
                f"supported {supported}."
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
        """Boolean masks ``(..., n_weighted)`` of the valid weighted targets per source.

        ``target`` has its variables on the last axis, in DATA_FULL or DATA_OUTPUT layout.
        """
        layout = self._resolve_layout(target.shape[-1], layout)
        valid = torch.isfinite(target.index_select(-1, self._pos(layout, "target")[self.weighted_pos]))

        inferred: dict[str, torch.Tensor] = {}
        unclaimed = torch.ones_like(valid)
        for source in self.sources:
            if source.is_fallback:
                continue
            present = unclaimed
            for family in source.requires:
                reported = torch.isfinite(target.index_select(-1, self._pos(layout, f"req_{family}")))
                if source.requires_scope == "column":
                    reported = reported.any(dim=-1, keepdim=True)
                present = present & reported
            inferred[source.name] = present
            unclaimed = unclaimed & ~present
        fallback = next(source.name for source in self.sources if source.is_fallback)
        inferred[fallback] = unclaimed

        if self.reportype_variable is None:
            return {source.name: valid & inferred[source.name] for source in self.sources}

        code = target.index_select(-1, self._pos(layout, "rpt"))  # (..., 1), broadcast over levels
        known = {
            source.name: torch.isin(code, torch.tensor(source.reportype_codes, dtype=code.dtype, device=code.device))
            for source in self.sources
        }
        coded = torch.zeros_like(code, dtype=torch.bool)
        for mask in known.values():
            coded = coded | mask
        return {
            source.name: valid & torch.where(coded, known[source.name], inferred[source.name])
            for source in self.sources
        }

    def _partial_weights(self, target: torch.Tensor, layout: IndexSpace) -> tuple[torch.Tensor, torch.Tensor]:
        """Unnormalised weights ``(..., n_weighted)`` and the per-source / validity sums over the leading axes.

        The sums are ``(n_sources + 2, n_weighted)``: one row per source with its element count, then the
        sum of the weights and the number of valid elements.
        """
        masks = self.source_masks(target, layout)
        weights = torch.ones(next(iter(masks.values())).shape, dtype=torch.float32, device=target.device)
        valid = torch.zeros_like(weights, dtype=torch.bool)
        for source in self.sources:
            weights = torch.where(masks[source.name], source.weight, weights)
            valid = valid | masks[source.name]
        lead_dims = tuple(range(valid.dim() - 1))
        sums = torch.stack(
            [
                *(masks[source.name].sum(dim=lead_dims).to(torch.float32) for source in self.sources),
                (weights * valid).sum(dim=lead_dims),
                valid.sum(dim=lead_dims).to(torch.float32),
            ],
        )
        return weights, sums

    def _normalise_weights(self, weights: list[torch.Tensor], sums: torch.Tensor) -> list[torch.Tensor]:
        """Apply the variable-share renormalisation and record the source fractions."""
        n_sources = len(self.sources)
        if self.preserve_variable_share:
            mean = sums[n_sources] / sums[n_sources + 1].clamp_min(1.0)
            scale = torch.where(mean > 0, mean.clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(mean))
            weights = [w / scale for w in weights]

        counts = sums[:n_sources]
        total = counts.sum().clamp_min(1.0)
        self.last_source_fractions = {
            source.name: float(c) for source, c in zip(self.sources, counts.sum(dim=1) / total, strict=True)
        }
        if not self._logged:
            LOGGER.info("%s: first-batch source fractions %s", self.__class__.__name__, self.last_source_fractions)
            self._logged = True
        return weights

    def element_weights(
        self,
        target: torch.Tensor,
        group: ProcessGroup | None = None,
        layout: IndexSpace | str | None = IndexSpace.DATA_FULL,
    ) -> torch.Tensor:
        """Weights ``(..., n_weighted)`` for the weighted variables (1 where no target).

        ``target`` has its variables on the last axis. With ``group``, the variable shares are
        normalised over the elements of every rank of the group.
        """
        layout = self._resolve_layout(target.shape[-1], layout)
        weights, sums = self._partial_weights(target, layout)
        if group is not None:
            dist.all_reduce(sums, group=group)
        return self._normalise_weights([weights], sums)[0]

    def _source_weights(
        self,
        target: Source,
        layout: IndexSpace,
        group: ProcessGroup | None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Element weights of a gridded target, or one tensor per sample of a tabular one."""
        samples = target.data if target.is_tabular else [target.data]
        parts, total = [], None
        for sample in samples:
            var_axis = target.layout.axis("variables", ndim=sample.ndim)
            weights, sums = self._partial_weights(sample.movedim(var_axis, -1), layout)
            parts.append(weights)
            total = sums if total is None else total + sums
        if group is not None:
            dist.all_reduce(total, group=group)
        parts = self._normalise_weights(parts, total)
        return parts if target.is_tabular else parts[0]

    def _forward_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        layout: TensorLayout,
        element_weights: torch.Tensor | None = None,
        squash: bool = True,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        valid_counts: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Source-weighted MSE of aligned (MODEL_OUTPUT) prediction and target tensors.

        ``element_weights`` holds the weights of the weighted variables with the variables on the last
        axis (see :meth:`element_weights`); the other variables keep weight 1.
        """
        is_sharded = grid_shard_slice is not None
        pred, target = self.mask_nans(pred, target)
        out = self.calculate_difference(pred, target)

        if element_weights is not None:
            var_axis = layout.axis("variables", ndim=out.ndim)
            out = out.movedim(var_axis, -1)
            factor = torch.ones(out.shape, dtype=out.dtype, device=out.device)
            factor[..., self.weighted_pos] = element_weights.to(device=out.device, dtype=out.dtype)
            out = (out * factor).movedim(-1, var_axis)

        out = self.scale(
            out,
            scaler_indices,
            layout=layout,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out,
            layout=layout,
            squash=squash,
            group=group if is_sharded else None,
            squash_mode=squash_mode,
            valid_counts=self._counts_like(valid_counts, pred, layout, scaler_indices),
        )

    def forward(
        self,
        pred: Source,
        target: Source,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Source-weighted MSE.

        Parameters
        ----------
        pred : Source
            Prediction in MODEL_OUTPUT layout.
        target : Source
            Target in DATA_FULL or DATA_OUTPUT layout (inferred from the width when not given).
        squash : bool, optional
            Average the variable dimension, by default True.
        scaler_indices : tuple[int, ...], optional
            Indices to subset the calculated scaler with, by default None.
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude from scaling, by name or dimension. By default None.
        grid_shard_slice : slice, optional
            Slice of the grid if the gridded data comes sharded, by default None.
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None.
        squash_mode : {"avg", "sum"}, optional
            Reduction mode for the variable dimension, by default ``"avg"``.
        pred_layout, target_layout : IndexSpace | str | None
            MODEL_OUTPUT, and DATA_FULL or DATA_OUTPUT.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Scalar loss, or one value per model output when ``squash`` is False.
        """
        pred_layout = None if pred_layout is None else IndexSpace(str(pred_layout))
        if pred_layout not in (None, IndexSpace.MODEL_OUTPUT) or pred.n_variables != self._output_width:
            msg = f"{self.__class__.__name__} expects pred in MODEL_OUTPUT layout (width {self._output_width})."
            raise ValueError(msg)
        layout = self._resolve_layout(target.n_variables, target_layout)

        if target.is_tabular:
            weight_group = _tabular_shard_group(target, group)
        else:
            weight_group = group if grid_shard_slice is not None else None
        with torch.no_grad():
            weights = self._source_weights(target, layout, weight_group)

        target = target.select_variables(self._pos(layout, "target").tolist())
        if target.is_tabular:
            per_sample = {**(kwargs.pop("per_sample_kwargs", None) or {}), "element_weights": weights}
            kwargs["per_sample_kwargs"] = per_sample
        else:
            kwargs["element_weights"] = weights

        return self._apply_pairwise(
            pred,
            target,
            self._evaluate_loss_tensor,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group,
            squash_mode=squash_mode,
            **kwargs,
        )
