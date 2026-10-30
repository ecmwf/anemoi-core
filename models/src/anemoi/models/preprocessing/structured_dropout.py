# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Structured dropout of observing systems for data augmentation."""

import fnmatch
import logging
import math
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.spatial_dropout import BaseInputDropout

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

DEFAULT_COORDINATE_VARIABLES = {
    "cos_latitude": "cos_latitude",
    "sin_latitude": "sin_latitude",
    "cos_longitude": "cos_longitude",
    "sin_longitude": "sin_longitude",
}


class StructuredObsDropout(BaseInputDropout):
    """Drops whole observing systems, stations or regions (sets them to NaN) during training.

    Unlike :class:`~anemoi.models.preprocessing.spatial_dropout.RandomSpatialDropout`, which draws
    an independent mask per value, every mask here is shared across all variables of a *group*
    (one observing system, e.g. radiosondes or microwave sounders), all members and all time steps
    of the view (the first ``multi_step`` ones, if set). A withheld observation is therefore not
    recoverable from another level, channel or time at the same cell, which forces the model to
    spread information horizontally and across observing systems.

    Each group combines three independent mask modes (union):

    - ``stream_prob``: per sample, drop the whole group everywhere (instrument outage). At most
      ``max_streams_dropped`` groups are dropped per sample.
    - ``cell_prob``: per (sample, cell), drop the group's whole column (station withholding).
    - ``n_blocks`` / ``block_radius_km``: per sample, drop the group inside ``n_blocks`` discs
      centred on random cells. Cell coordinates are read from the sin/cos latitude/longitude input
      columns; a view without them gets no blocks.

    Group variables are ``fnmatch`` patterns, so ``mwt_*`` also catches the instrument's
    geometry/report-type corrector variables, and a dropped stream looks exactly like a real outage
    at inference. Forcings are protected unless a group sets ``allow_forcing: true``.

    ``dropout_prob`` is a global multiplier on every group probability (default 1.0), so
    :class:`~anemoi.training.diagnostics.callbacks.dropout_scheduler.DropoutScheduler` can schedule
    all groups at once: set the group probabilities at their peak values and schedule the
    multiplier down from ``start_prob: 1.0``.

    The pre-processors run once per view: on the inputs, then on each rollout step's observations
    that advance the input state (the DA-cycle observations of a DA forecaster). Each call draws
    new masks. A tabular view is processed one sample at a time, its rows as the cells.

    Random draws use the global torch RNG on each rank. Under grid sharding each rank of a model
    group draws its own stream and block patterns for its grid shard; cell masks are unaffected.

    Configuration example:
    ```yaml
    obs_dropout:
      _target_: anemoi.models.preprocessing.structured_dropout.StructuredObsDropout
      _convert_: all
      config:
        max_streams_dropped: 2
        groups:
          radiosonde:
            variables: ["z_*", "t_*", "q_*", "u_*", "v_*"]
            cell_prob: 0.3
            stream_prob: 0.05
            n_blocks: 2
            block_radius_km: 1500
          mw_temp:
            variables: ["mwt_*"]
            cell_prob: 0.1
            stream_prob: 0.1
    ```
    """

    _GROUP_KEYS = {"variables", "stream_prob", "cell_prob", "n_blocks", "block_radius_km", "allow_forcing"}
    _parameter_keys = ("dropout_prob", "multi_step", "max_streams_dropped", "groups", "coordinate_variables")

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the structured observation dropout preprocessor.

        Parameters
        ----------
        config : DotDict
            ``groups``, and optional ``multi_step``, ``max_streams_dropped``, ``dropout_prob`` and
            ``coordinate_variables``.
        data_indices : IndexCollection
            Data indices of the dataset.
        statistics : dict
            Not used by this preprocessor.
        """
        super().__init__(config, data_indices, statistics)
        config = config or {}

        self.dropout_prob = float(config.get("dropout_prob", 1.0))
        if self.dropout_prob < 0.0:
            msg = f"dropout_prob is a non-negative multiplier, got {self.dropout_prob}"
            raise ValueError(msg)
        max_streams = config.get("max_streams_dropped", None)
        self.max_streams_dropped = None if max_streams is None else int(max_streams)

        name_to_index = self.data_indices.data.input.name_to_index
        forcing_names = set(self.data_indices.forcing)

        self.group_names: list[str] = []
        self.group_variables: dict[str, list[str]] = {}
        self._stream_prob: list[float] = []
        self._cell_prob: list[float] = []
        self._n_blocks: list[int] = []
        self._block_radius_km: list[float] = []
        claimed: dict[str, str] = {}

        groups = config.get("groups", None) or {}
        if not groups:
            msg = "StructuredObsDropout: 'groups' must define at least one group."
            raise ValueError(msg)

        for group_name, group in groups.items():
            names = self._group_names(group_name, group, name_to_index)
            forcing_hits = sorted(set(names) & forcing_names)
            if forcing_hits and not group.get("allow_forcing", False):
                msg = (
                    f"StructuredObsDropout: group {group_name!r} matches forcing variables {forcing_hits}. "
                    "Narrow the patterns or set allow_forcing: true."
                )
                raise ValueError(msg)
            for name in names:
                if name in claimed:
                    msg = (
                        f"StructuredObsDropout: variable {name!r} is in both groups "
                        f"{claimed[name]!r} and {group_name!r}; groups must be disjoint."
                    )
                    raise ValueError(msg)
                claimed[name] = group_name

            probs = {key: float(group.get(key, 0.0)) for key in ("stream_prob", "cell_prob")}
            for key, value in probs.items():
                if not 0.0 <= value <= 1.0:
                    msg = f"StructuredObsDropout: {key} of group {group_name!r} must be in [0, 1], got {value}."
                    raise ValueError(msg)
            n_blocks = int(group.get("n_blocks", 0))
            radius = float(group.get("block_radius_km", 0.0))
            if n_blocks < 0 or (n_blocks > 0 and radius <= 0.0):
                msg = f"StructuredObsDropout: group {group_name!r} needs n_blocks >= 0 and block_radius_km > 0."
                raise ValueError(msg)

            self.group_names.append(group_name)
            self.group_variables[group_name] = names
            self._stream_prob.append(probs["stream_prob"])
            self._cell_prob.append(probs["cell_prob"])
            self._n_blocks.append(n_blocks)
            self._block_radius_km.append(radius)
            LOGGER.info(
                "StructuredObsDropout: group %r (stream_prob=%.3f, cell_prob=%.3f, n_blocks=%d, radius=%.0f km): %s",
                group_name,
                probs["stream_prob"],
                probs["cell_prob"],
                n_blocks,
                radius,
                names,
            )

        # Union of the dropped variables; DropoutScheduler requires this to be non-empty.
        self.register_buffer(
            "dropout_indices",
            torch.tensor(sorted(name_to_index[n] for n in claimed), dtype=torch.long),
            persistent=False,
        )

        self.coordinate_variables = None
        if any(n > 0 for n in self._n_blocks):
            coordinate_variables = {
                **DEFAULT_COORDINATE_VARIABLES,
                **dict(config.get("coordinate_variables", {}) or {}),
            }
            missing = [v for v in coordinate_variables.values() if v not in name_to_index]
            if missing:
                msg = f"StructuredObsDropout: block dropout needs coordinate variables {missing} in the input."
                raise ValueError(msg)
            self.coordinate_variables = coordinate_variables

    def _group_names(self, group_name: str, group: dict, name_to_index: dict[str, int]) -> list[str]:
        """Input variables matched by a group's patterns, in pattern order."""
        unknown = set(group.keys()) - self._GROUP_KEYS
        if unknown:
            msg = f"StructuredObsDropout: group {group_name!r} has unknown keys {sorted(unknown)}."
            raise ValueError(msg)
        patterns = list(group.get("variables", []))
        if not patterns:
            msg = f"StructuredObsDropout: group {group_name!r} lists no variables."
            raise ValueError(msg)
        names = []
        for pattern in patterns:
            matched = [name for name in name_to_index if fnmatch.fnmatchcase(name, pattern)]
            if not matched:
                msg = f"StructuredObsDropout: pattern {pattern!r} in group {group_name!r} matches no input variable."
                raise ValueError(msg)
            names.extend(name for name in matched if name not in names)
        return names

    def _stream_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a ``(batch, n_groups)`` mask of whole groups dropped per sample."""
        probs = torch.tensor(self._stream_prob, device=device) * self.dropout_prob
        dropped = torch.rand(batch_size, len(probs), device=device) < probs.clamp(max=1.0)
        if self.max_streams_dropped is not None and dropped.sum(dim=1).max() > self.max_streams_dropped:
            # Keep a random subset of at most max_streams_dropped of the drawn groups.
            priority = torch.where(dropped, torch.rand(dropped.shape, device=device), -1.0)
            rank = priority.argsort(dim=1, descending=True).argsort(dim=1)
            dropped = dropped & (rank < self.max_streams_dropped)
        return dropped

    def _unit_vectors(self, x: torch.Tensor, coordinates: dict[str, int]) -> torch.Tensor:
        """Return ``(batch, grid, 3)`` cell unit vectors from the coordinate columns of a gridded view."""
        first = x[:, 0, 0]
        cos_lat, sin_lat = first[..., coordinates["cos_latitude"]], first[..., coordinates["sin_latitude"]]
        cos_lon, sin_lon = first[..., coordinates["cos_longitude"]], first[..., coordinates["sin_longitude"]]
        return torch.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dim=-1).float()

    @staticmethod
    def _block_mask(xyz: torch.Tensor, n_blocks: int, radius_km: float) -> torch.Tensor:
        """Return a ``(batch, grid)`` mask of cells inside ``n_blocks`` random discs."""
        batch_size, grid = xyz.shape[:2]
        centres = torch.randint(grid, (batch_size, n_blocks), device=xyz.device)
        centre_xyz = torch.gather(xyz, 1, centres.unsqueeze(-1).expand(-1, -1, 3))  # (B, n, 3)
        cos_angle = torch.einsum("bgk,bnk->bgn", xyz, centre_xyz)
        return (cos_angle >= math.cos(radius_km / EARTH_RADIUS_KM)).any(dim=-1)

    def transform(
        self,
        x: torch.Tensor,
        name_to_index: Optional[dict[str, int]] = None,
        skip_imputation: bool = False,
        apply_dropout: Optional[bool] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Apply structured dropout.

        Parameters
        ----------
        x : torch.Tensor
            A gridded view ``(batch, time, ensemble, grid, variables)`` or one tabular sample
            ``(ensemble, rows, variables)``. Modified in place.
        name_to_index : dict[str, int], optional
            Variable positions in ``x``.
        skip_imputation : bool, optional
            Set on target views, which are returned unchanged.
        apply_dropout : bool, optional
            Overrides whether the view gets dropout; by default only views that are imputed do.

        Returns
        -------
        torch.Tensor
            ``x`` with the dropped groups set to NaN (training), or unchanged.
        """
        if not self._active(x, skip_imputation, apply_dropout):
            return x
        positions = self._positions(self.group_variables, name_to_index)
        if not any(positions.values()):
            return x

        view = self._as_gridded(x)
        batch_size, grid = view.shape[0], view.shape[-2]
        n_input = self._n_input(view)
        stream = self._stream_mask(batch_size, x.device)
        xyz = None
        if self.coordinate_variables is not None and all(
            name in name_to_index for name in self.coordinate_variables.values()
        ):
            coordinates = {key: name_to_index[name] for key, name in self.coordinate_variables.items()}
            xyz = self._unit_vectors(view, coordinates)

        for g, group_name in enumerate(self.group_names):
            if not positions[group_name]:
                continue
            mask = stream[:, g : g + 1].expand(batch_size, grid)
            cell_prob = min(self._cell_prob[g] * self.dropout_prob, 1.0)
            if cell_prob > 0:
                mask = mask | (torch.rand(batch_size, grid, device=x.device) < cell_prob)
            if self._n_blocks[g] > 0 and xyz is not None:
                mask = mask | self._block_mask(xyz, self._n_blocks[g], self._block_radius_km[g])
            if not mask.any():
                continue
            idx = torch.tensor(positions[group_name], dtype=torch.long, device=x.device)
            # (batch, grid) -> broadcast over time, members and the group's variables.
            mask = mask.view(batch_size, 1, 1, grid, 1)
            sub = view[:, :n_input, ..., idx]
            view[:, :n_input, ..., idx] = sub.masked_fill(mask, torch.nan)

        return x
