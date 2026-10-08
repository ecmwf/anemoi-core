# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

r"""Spatial amplitude maps for input noise, built from ensemble spread fields.

The FourCastNet 3 perturbation (:class:`~anemoi.models.layers.ensemble.SphericalInputNoise`)
has the same amplitude everywhere on the globe. Real analysis uncertainty does
not: it concentrates in fronts, convection, jets and data-sparse regions, and it
was larger in decades with fewer observations. The modules here turn a set of
spread fields -- e.g. the ERA5 EDA standard deviation, joined into the input
dataset as ``std_``-prefixed forcings -- into one dimensionless amplitude map per
noise channel.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter

LOGGER = logging.getLogger(__name__)

# Data normalisers that only rescale a variable. A spread divided by its mean is unit-free
# only if the normaliser does not shift it, which rules out 'mean-std' and 'min-max'.
SCALE_ONLY_NORMALIZERS = ("none", "std", "max")

# The area-mean of a spread field over its climatological mean is about 1 on any sample
# (0.5-2.5 across the ERA5 record). Far outside these bounds the reference is in the wrong
# units, i.e. ``normalizer`` does not match the data config.
REFERENCE_CHECK_BOUNDS = (0.1, 10.0)


def climatological_spread_statistics(
    variables: list[str],
    statistics: dict,
    name_to_index_stats: dict[str, int],
    normalizer: str,
) -> tuple[Tensor, Tensor]:
    r"""Climatological reference and mean square of each spread variable.

    For a spread field :math:`S_v` with dataset mean :math:`\mu_v` and standard
    deviation :math:`s_v` over the statistics period:

    - the **reference** is :math:`\mu_v` in the units the model receives, i.e. after
      the data normaliser, so :math:`S_v / \mathrm{reference}_v` is the spread in
      multiples of its climatological mean whatever the normaliser;
    - the **mean square** of that ratio over all times and points is
      :math:`1 + (s_v / \mu_v)^2`. Dividing by its square root makes the long-term
      mean square of the multiplier about one (a little less once it is smoothed
      and clipped), while keeping its variations in space and time.

    Parameters
    ----------
    variables : list[str]
        Spread variables, as named in the dataset (e.g. ``std_t_850``).
    statistics : dict
        Dataset statistics (``mean``, ``stdev``, ``maximum``), indexed like ``name_to_index_stats``.
    name_to_index_stats : dict[str, int]
        Position of each variable in ``statistics``, i.e. ``data_indices.data.input.name_to_index``.
    normalizer : str
        How the data config normalises these variables: one of :data:`SCALE_ONLY_NORMALIZERS`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Reference and mean square, each of shape ``(variables,)``.
    """
    if normalizer not in SCALE_ONLY_NORMALIZERS:
        raise ValueError(
            f"normalizer '{normalizer}' is not supported for spread fields: a spread over its mean is only "
            f"unit-free if the data normaliser rescales without shifting, i.e. one of {SCALE_ONLY_NORMALIZERS}."
        )
    missing = sorted(set(variables) - set(name_to_index_stats))
    if missing:
        raise ValueError(f"spread variables {missing} have no dataset statistics.")

    index = [name_to_index_stats[name] for name in variables]
    mean = np.asarray(statistics["mean"], dtype=np.float64)[index]
    stdev = np.asarray(statistics["stdev"], dtype=np.float64)[index]
    if np.any(mean <= 0):
        bad = [name for name, value in zip(variables, mean) if value <= 0]
        raise ValueError(f"spread variables {bad} have a non-positive climatological mean.")

    if normalizer == "std":
        mean_in_model_units = mean / stdev
    elif normalizer == "max":
        mean_in_model_units = mean / np.asarray(statistics["maximum"], dtype=np.float64)[index]
    else:
        mean_in_model_units = mean

    mean_square = 1.0 + (stdev / mean) ** 2
    return torch.as_tensor(mean_in_model_units), torch.as_tensor(mean_square)


def strip_level(name: str) -> str:
    """Base variable of a level-suffixed name: ``t_850 -> t``, ``2t -> 2t``, ``10u -> 10u``."""
    head, _, level = name.rpartition("_")
    return head if head and level.isdigit() else name


def resolve_prefixed_variables(name_to_index: dict[str, int], prefix: str) -> list[tuple[int, str]]:
    """``(input position, name)`` of every variable starting with ``prefix``, in input order."""
    return sorted((int(index), name) for name, index in name_to_index.items() if name.startswith(prefix))


class GroupedStdModulation(nn.Module):
    r"""One amplitude map per noise channel from grouped ensemble spread fields.

    For variable group :math:`g` and channel :math:`c` assigned to it:

    1. **Normalise** every spread field to a dimensionless ratio, so fields in
       different units (``q`` in kg/kg, ``z`` in m^2/s^2) are comparable. The
       ratio is :math:`R_v = S_v / \mathrm{ref}_v`, with

       - ``reference`` given, a fixed :math:`\mathrm{ref}_v` per variable,
         typically its climatological mean
         (:func:`climatological_spread_statistics`). The ratio then keeps how
         uncertain the whole analysis is, e.g. that the 1980s were less well
         observed than the 2020s;
       - without it, the field's own area-weighted global mean on that sample,
         so that :math:`\mathrm{ref}_v = \langle S_v \rangle_A`. The ratio keeps
         only where the uncertainty is, and is independent of any per-variable
         scaling the data normaliser applied.
    2. **Group** by averaging, :math:`G_g = \mathrm{mean}_{v \in g} R_v`.
    3. **Smooth** (optional) with a per-channel spectral response, so the
       amplitude varies no faster than the noise it scales and the sampling noise
       of a small ensemble's spread is damped.
    4. **Clip** to ``clip``, in multiples of the reference.
    5. **Rescale**, with one of:

       - with ``mean_square``, divide by a fixed constant per channel, the square
         root of :math:`\mathrm{mean}_{v \in g}\,\overline{R_v^2}`, where each
         ratio's long-term mean square is :math:`\overline{R_v^2}`. The multiplier
         then has a long-term mean square of about one but still varies from
         sample to sample;
       - with ``preserve_total_variance``, renormalise every sample to unit
         area-weighted mean square, :math:`\langle M_c^2 \rangle_A = 1`. Noise of
         unit variance multiplied by the map keeps unit variance over the globe:
         the map moves noise around, never adds or removes it.

    All spatial means are area-weighted with ``area_weights``, so the dense
    polar rows of a reduced grid do not dominate.

    Parameters
    ----------
    groups : dict[str, list[str]]
        Group name to its variables. Entries are matched against the prefixed
        input variables by base name (``t`` matches ``std_t_850``), by exact
        name without the prefix (``t_850``) or by full name (``std_t_850``).
    channel_group : list[str]
        The group modulating each noise channel, one entry per channel.
    name_to_index : dict[str, int]
        The dataset's model-input name to position map, i.e.
        ``data_indices.model.input.name_to_index``.
    area_weights : Tensor
        Per-grid-point weights summing to one, shape ``(points,)``.
    variable_prefix : str, optional
        Prefix identifying the spread variables, ``"std_"`` by default.
    reference : Tensor, optional
        Fixed reference of every spread variable, in model-input units and in the
        order of :attr:`variables`. ``None`` uses each sample's own global mean.
    mean_square : Tensor, optional
        Long-term mean square of every ratio :math:`R_v`, in the order of
        :attr:`variables`. Selects the fixed rescaling; needs ``reference``.
    clip : tuple[float, float], optional
        Bounds applied after smoothing. ``None`` disables clipping.
    preserve_total_variance : bool, optional
        Renormalise each map to unit area-weighted mean square. Not with ``mean_square``.
    spectral_filter : SphericalSpectralFilter, optional
        Transform used for smoothing. Required with ``smooth_response``.
    smooth_response : Tensor, optional
        Per-channel smoothing response, shape ``(channels, filter.lmax)``.
        ``None`` disables smoothing.
    """

    def __init__(
        self,
        *,
        groups: dict[str, list[str]],
        channel_group: list[str],
        name_to_index: dict[str, int],
        area_weights: Tensor,
        variable_prefix: str = "std_",
        reference: Optional[Tensor] = None,
        mean_square: Optional[Tensor] = None,
        clip: Optional[tuple[float, float]] = (0.25, 4.0),
        preserve_total_variance: bool = True,
        spectral_filter: Optional[SphericalSpectralFilter] = None,
        smooth_response: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.variable_prefix = variable_prefix
        self.clip = tuple(float(bound) for bound in clip) if clip is not None else None
        self.preserve_total_variance = preserve_total_variance
        self.groups = {name: list(members) for name, members in groups.items()}
        self.channel_group = list(channel_group)

        if self.clip is not None and not 0.0 <= self.clip[0] < self.clip[1]:
            raise ValueError(f"clip must be (low, high) with 0 <= low < high, got {self.clip}")
        if smooth_response is not None and spectral_filter is None:
            raise ValueError("smooth_response requires a spectral_filter to apply it.")
        if mean_square is not None and reference is None:
            raise ValueError("mean_square is the mean square of the ratio to a fixed reference; give a reference.")
        if mean_square is not None and preserve_total_variance:
            raise ValueError("choose one rescaling: a fixed one (mean_square) or per sample (preserve_total_variance).")

        consumed = resolve_prefixed_variables(name_to_index, variable_prefix)
        if not consumed:
            raise ValueError(
                f"No model input variable starts with '{variable_prefix}'. The spread fields must be joined into "
                "the input dataset and listed under 'forcing' in the data config."
            )
        self.input_idx = [index for index, _ in consumed]
        self.variables = [name for _, name in consumed]

        group_weights = self._group_weights()
        channel_index = self._channel_index()

        self.register_buffer("group_weights", group_weights, persistent=False)
        self.register_buffer("channel_index", channel_index, persistent=False)
        self.register_buffer("area_weights", torch.as_tensor(area_weights, dtype=torch.float32), persistent=False)

        self._reference_checked = reference is None
        if reference is not None:
            reference = torch.as_tensor(reference, dtype=torch.float32)
            if tuple(reference.shape) != (len(self.variables),) or torch.any(reference <= 0):
                raise ValueError(
                    f"reference must hold one positive value per spread variable ({len(self.variables)}), "
                    f"got shape {tuple(reference.shape)}."
                )
            self.register_buffer("reference", reference, persistent=False)
        else:
            self.reference = None
        if mean_square is not None:
            mean_square = torch.as_tensor(mean_square, dtype=torch.float32)
            if tuple(mean_square.shape) != (len(self.variables),) or torch.any(mean_square <= 0):
                raise ValueError(
                    f"mean_square must hold one positive value per spread variable ({len(self.variables)}), "
                    f"got shape {tuple(mean_square.shape)}."
                )
            scale = torch.rsqrt(group_weights @ mean_square)[channel_index]
            self.register_buffer("channel_scale", scale, persistent=False)
        else:
            self.channel_scale = None

        self.spectral_filter = spectral_filter
        if smooth_response is not None:
            if tuple(smooth_response.shape) != (len(self.channel_group), spectral_filter.lmax):
                raise ValueError(
                    f"smooth_response has shape {tuple(smooth_response.shape)}, expected "
                    f"({len(self.channel_group)}, {spectral_filter.lmax})."
                )
            self.register_buffer("smooth_response", smooth_response.to(torch.float32), persistent=False)
        else:
            self.smooth_response = None

    def _group_weights(self) -> Tensor:
        """``(groups, variables)`` averaging matrix: ``1/|g|`` on each group's members."""
        if not self.groups:
            raise ValueError("modulation.groups must define at least one group.")

        stripped = [name[len(self.variable_prefix) :] for name in self.variables]
        weights = torch.zeros(len(self.groups), len(self.variables))
        for row, (group, entries) in enumerate(self.groups.items()):
            entries = set(entries)
            members = [
                column
                for column, (full, short) in enumerate(zip(self.variables, stripped))
                if full in entries or short in entries or strip_level(short) in entries
            ]
            if not members:
                available = sorted({strip_level(short) for short in stripped})
                raise ValueError(
                    f"modulation group '{group}' ({sorted(entries)}) matches no '{self.variable_prefix}' input "
                    f"variable. Available base names: {available}."
                )
            weights[row, members] = 1.0 / len(members)

        unused = sorted({strip_level(stripped[column]) for column in torch.nonzero(weights.sum(dim=0) == 0).flatten()})
        if unused:
            LOGGER.warning(
                "GroupedStdModulation: '%s' variables %s belong to no group; they are read but do not "
                "shape any amplitude map.",
                self.variable_prefix,
                unused,
            )
        return weights

    def _channel_index(self) -> Tensor:
        """Group row for every channel."""
        group_names = list(self.groups)
        unknown = sorted(set(self.channel_group) - set(group_names))
        if unknown:
            raise ValueError(f"modulation.channel_group refers to undefined groups {unknown}; defined: {group_names}.")
        idle = sorted(set(group_names) - set(self.channel_group))
        if idle:
            LOGGER.warning("GroupedStdModulation: groups %s modulate no channel.", idle)
        return torch.as_tensor([group_names.index(group) for group in self.channel_group], dtype=torch.long)

    @property
    def num_channels(self) -> int:
        return len(self.channel_group)

    def group_sizes(self) -> dict[str, int]:
        """Number of spread variables in each group."""
        return {group: int((row > 0).sum()) for group, row in zip(self.groups, self.group_weights)}

    def area_mean(self, field: Tensor) -> Tensor:
        """Area-weighted mean over the trailing points dimension."""
        return torch.einsum("...p,p->...", field, self.area_weights.to(field.dtype))

    def forward(self, std: Tensor) -> Tensor:
        """Build the amplitude maps.

        Parameters
        ----------
        std : Tensor
            Spread fields on the full grid, shape ``(batch, time, points, variables)``,
            with variables in the order of :attr:`input_idx`.

        Returns
        -------
        Tensor
            Amplitude maps of shape ``(batch, time, channels, points)``.
        """
        std = std.to(torch.float32).transpose(-1, -2)  # (batch, time, variables, points)

        if self.reference is None:
            global_mean = self.area_mean(std)
            if not torch.all(global_mean > 0):
                raise ValueError(
                    f"'{self.variable_prefix}' fields must be positive spreads, but some have a non-positive global "
                    "mean. Normalise them with 'std' or 'none', not 'mean-std'."
                )
            ratio = std / global_mean.unsqueeze(-1)
        else:
            ratio = std / self.reference.unsqueeze(-1)
            if not self._reference_checked:
                self._check_reference(ratio)
        group_maps = torch.einsum("btvp,gv->btgp", ratio, self.group_weights)

        if self.smooth_response is not None:
            coeffs = self.spectral_filter.analyse(group_maps)[:, :, self.channel_index]
            modulation = self.spectral_filter.synthesise(coeffs * self.smooth_response.unsqueeze(-1))
        else:
            modulation = group_maps[:, :, self.channel_index]

        if self.clip is not None:
            modulation = modulation.clamp(*self.clip)
        if self.channel_scale is not None:
            modulation = modulation * self.channel_scale.unsqueeze(-1)
        elif self.preserve_total_variance:
            modulation = modulation / torch.sqrt(self.area_mean(modulation**2)).unsqueeze(-1)
        return modulation

    def _check_reference(self, ratio: Tensor) -> None:
        """Fail on the first sample if the fixed reference is in the wrong units.

        Checked once: it synchronises with the device.
        """
        level = self.area_mean(ratio).mean(dim=(0, 1))  # (variables,)
        low, high = REFERENCE_CHECK_BOUNDS
        outside = torch.nonzero((level < low) | (level > high)).flatten().tolist()
        if outside:
            examples = ", ".join(f"{self.variables[i]} {level[i].item():.3g}" for i in outside[:5])
            raise ValueError(
                f"The area mean of {len(outside)} spread fields over their climatological mean is far from 1 "
                f"({examples}; expected {low}-{high}). The reference is probably in the wrong units: is "
                "modulation.normalizer set to how the data config normalises the spread variables?"
            )
        self._reference_checked = True
        LOGGER.info(
            "GroupedStdModulation: spread over its climatological mean on the first sample, area mean %.2f-%.2f.",
            level.min().item(),
            level.max().item(),
        )
