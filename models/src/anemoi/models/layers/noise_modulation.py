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
not: it concentrates in fronts, convection, jets and data-sparse regions. The
modules here turn a set of spread fields -- e.g. the ERA5 EDA standard deviation,
joined into the input dataset as ``std_``-prefixed forcings -- into one
dimensionless amplitude map per noise channel, so a channel can be redistributed
in space without changing how much noise it carries overall.
"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.spectral_transforms import SphericalSpectralFilter

LOGGER = logging.getLogger(__name__)


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

    1. **Normalise** every spread field by its own area-weighted global mean, so
       that :math:`R_v = S_v / \langle S_v \rangle_A`. This makes fields in
       different units (``q`` in kg/kg, ``z`` in m^2/s^2) comparable, and because
       it is a ratio it is independent of any per-variable scaling the data
       normaliser applied.
    2. **Group** by averaging, :math:`G_g = \mathrm{mean}_{v \in g} R_v`, so that
       the group map has :math:`\langle G_g \rangle_A = 1` exactly.
    3. **Smooth** (optional) with a per-channel spectral response, so the
       amplitude varies no faster than the channel it modulates.
    4. **Clip** to ``clip``. The global mean is one, so the bounds read as
       multiples of the global mean amplitude.
    5. **Renormalise** (optional) so :math:`\langle M_c^2 \rangle_A = 1`. Noise of
       unit variance multiplied by :math:`M_c` then keeps unit variance on
       average over the globe: the map moves noise around, never adds or
       removes it.

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
    clip : tuple[float, float], optional
        Bounds applied after smoothing. ``None`` disables clipping.
    preserve_total_variance : bool, optional
        Renormalise each map to unit area-weighted mean square.
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

        global_mean = self.area_mean(std)
        if not torch.all(global_mean > 0):
            raise ValueError(
                f"'{self.variable_prefix}' fields must be positive spreads, but some have a non-positive global "
                "mean. Normalise them with 'std' or 'none', not 'mean-std'."
            )
        ratio = std / global_mean.unsqueeze(-1)
        group_maps = torch.einsum("btvp,gv->btgp", ratio, self.group_weights)

        if self.smooth_response is not None:
            coeffs = self.spectral_filter.analyse(group_maps)[:, :, self.channel_index]
            modulation = self.spectral_filter.synthesise(coeffs * self.smooth_response.unsqueeze(-1))
        else:
            modulation = group_maps[:, :, self.channel_index]

        if self.clip is not None:
            modulation = modulation.clamp(*self.clip)
        if self.preserve_total_variance:
            modulation = modulation / torch.sqrt(self.area_mean(modulation**2)).unsqueeze(-1)
        return modulation
