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
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

DEFAULT_COORDINATE_VARIABLES = {
    "cos_latitude": "cos_latitude",
    "sin_latitude": "sin_latitude",
    "cos_longitude": "cos_longitude",
    "sin_longitude": "sin_longitude",
}


class StructuredObsDropout(BasePreprocessor):
    """Drops whole observing systems, stations or regions (sets them to NaN) during training.

    Unlike :class:`~anemoi.models.preprocessing.spatial_dropout.RandomSpatialDropout`, which
    draws an independent mask per (time, cell, variable), every mask here is shared across all
    variables of a *group* (one observing system, e.g. radiosondes or microwave sounders) and
    across all of the first ``multi_step`` timesteps. A withheld observation is therefore not
    recoverable from another level, channel or DA time at the same cell, which forces the model
    to spread information horizontally and across observing systems.

    Each group combines three independent mask modes (union):

    - ``stream_prob``: per sample, drop the whole group everywhere (instrument outage). At most
      ``max_streams_dropped`` groups are dropped per sample.
    - ``cell_prob``: per (sample, cell), drop the group's whole column (station withholding).
    - ``n_blocks`` / ``block_radius_km``: per sample, drop the group inside ``n_blocks`` discs
      centred on random cells. Cell coordinates are read from the sin/cos latitude/longitude
      input columns.

    Group variables are ``fnmatch`` patterns, so ``mwt_*`` also catches the instrument's
    geometry/report-type corrector variables, and a dropped stream looks exactly like a real
    outage at inference. Only originally valid values are dropped; forcings are protected unless
    a group sets ``allow_forcing: true``.

    ``dropout_prob`` is a global multiplier on every group probability (default 1.0), so
    :class:`~anemoi.training.diagnostics.callbacks.dropout_scheduler.DropoutScheduler` can
    schedule all groups at once. Set the group probabilities at their peak values and schedule
    the multiplier down from ``start_prob: 1.0``.

    Training only: dropout follows ``self.training`` like ``nn.Dropout``, so it is off in
    validation and inference.

    Random draws use the global torch RNG on each rank. Under grid sharding
    (``num_gpus_per_model > 1``) each rank of a model group draws its own stream and block
    patterns for its grid shard; cell masks are unaffected.

    Configuration example:
    ```yaml
    obs_dropout:
      _target_: anemoi.models.preprocessing.structured_dropout.StructuredObsDropout
      _convert_: all
      config:
        multi_step: 6  # multistep_input + da_cycles * multistep_output
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

    @classmethod
    def _process_config(cls, config) -> tuple:
        """Treat this preprocessor's parameters as special keys, not processing methods."""
        _special_keys = [
            "default",
            "remap",
            "normalizer",
            "method_kwargs",
            "dropout_prob",
            "multi_step",
            "max_streams_dropped",
            "groups",
            "coordinate_variables",
        ]
        config = config or {}
        default = config.get("default", "none")
        remap = config.get("remap", {})
        normalizer = config.get("normalizer", "none")
        method_kwargs = config.get("method_kwargs", {})
        method_config = {k: v for k, v in config.items() if k not in _special_keys and v is not None and v != "none"}
        if method_config:
            LOGGER.warning("%s: Unexpected config keys %s.", cls.__name__, list(method_config.keys()))
        return default, remap, normalizer, method_config, method_kwargs

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
            Configuration with ``groups``, ``multi_step``, ``max_streams_dropped``,
            ``dropout_prob`` and optional ``coordinate_variables``.
        data_indices : IndexCollection
            Data indices for input variables.
        statistics : dict
            Not used by this preprocessor, but required by the base class.
        """
        super().__init__(config, data_indices, statistics)
        config = config or {}

        self.dropout_prob = float(config.get("dropout_prob", 1.0))
        if self.dropout_prob < 0.0:
            msg = f"dropout_prob is a non-negative multiplier, got {self.dropout_prob}"
            raise ValueError(msg)
        self.multi_step = int(config.get("multi_step", 2))
        max_streams = config.get("max_streams_dropped", None)
        self.max_streams_dropped = None if max_streams is None else int(max_streams)

        name_to_index = self.data_indices.data.input.name_to_index
        forcing_names = set(getattr(self.data_indices, "forcing", []) or [])

        self.group_names: list[str] = []
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
                    msg = (
                        f"StructuredObsDropout: pattern {pattern!r} in group {group_name!r} matches no input variable."
                    )
                    raise ValueError(msg)
                names.extend(name for name in matched if name not in names)

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
            self._stream_prob.append(probs["stream_prob"])
            self._cell_prob.append(probs["cell_prob"])
            self._n_blocks.append(n_blocks)
            self._block_radius_km.append(radius)
            self.register_buffer(
                f"_group_indices_{len(self.group_names) - 1}",
                torch.tensor([name_to_index[n] for n in names], dtype=torch.long),
                persistent=False,
            )
            LOGGER.info(
                "StructuredObsDropout: group %r (stream_prob=%.3f, cell_prob=%.3f, n_blocks=%d, radius=%.0f km) "
                "over first %d timesteps: %s",
                group_name,
                probs["stream_prob"],
                probs["cell_prob"],
                n_blocks,
                radius,
                self.multi_step,
                names,
            )

        # Union of dropped variables; DropoutScheduler requires this to be non-empty.
        self.register_buffer(
            "dropout_indices",
            torch.tensor(sorted(name_to_index[n] for n in claimed), dtype=torch.long),
            persistent=False,
        )

        self._coord_idx = None
        if any(n > 0 for n in self._n_blocks):
            coord_names = {**DEFAULT_COORDINATE_VARIABLES, **dict(config.get("coordinate_variables", {}) or {})}
            missing = [v for v in coord_names.values() if v not in name_to_index]
            if missing:
                msg = f"StructuredObsDropout: block dropout needs coordinate variables {missing} in the input."
                raise ValueError(msg)
            self._coord_idx = {key: name_to_index[name] for key, name in coord_names.items()}

    def _group_indices(self, group: int) -> torch.Tensor:
        return getattr(self, f"_group_indices_{group}")

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

    def _unit_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, grid, 3)`` cell unit vectors from the coordinate input columns."""
        c = self._coord_idx
        first = x[:, 0]
        while first.ndim > 3:  # drop ensemble (and any other) dims between time and grid
            first = first[:, 0]
        cos_lat, sin_lat = first[..., c["cos_latitude"]], first[..., c["sin_latitude"]]
        cos_lon, sin_lon = first[..., c["cos_longitude"]], first[..., c["sin_longitude"]]
        return torch.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dim=-1).float()

    def _block_mask(self, xyz: torch.Tensor, n_blocks: int, radius_km: float) -> torch.Tensor:
        """Return a ``(batch, grid)`` mask of cells inside ``n_blocks`` random discs."""
        batch_size, grid = xyz.shape[:2]
        centres = torch.randint(grid, (batch_size, n_blocks), device=xyz.device)
        centre_xyz = torch.gather(xyz, 1, centres.unsqueeze(-1).expand(-1, -1, 3))  # (B, n, 3)
        cos_angle = torch.einsum("bgk,bnk->bgn", xyz, centre_xyz)
        return (cos_angle >= math.cos(radius_km / EARTH_RADIUS_KM)).any(dim=-1)

    def transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Apply structured dropout to the first ``multi_step`` timesteps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, time, ..., grid, variable)``.
        in_place : bool
            Whether to modify the tensor in place (default: True).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            Tensor with dropout applied (training) or unchanged (eval).
        """
        if not self.training or self.dropout_prob == 0 or x.ndim < 4:
            return x if in_place else x.clone()
        if not in_place:
            x = x.clone()

        batch_size, grid = x.shape[0], x.shape[-2]
        n_input = min(self.multi_step, x.shape[1])
        stream = self._stream_mask(batch_size, x.device)
        xyz = self._unit_vectors(x) if self._coord_idx is not None else None

        for g in range(len(self.group_names)):
            mask = stream[:, g : g + 1].expand(batch_size, grid)
            cell_prob = min(self._cell_prob[g] * self.dropout_prob, 1.0)
            if cell_prob > 0:
                mask = mask | (torch.rand(batch_size, grid, device=x.device) < cell_prob)
            if self._n_blocks[g] > 0 and self.dropout_prob > 0:
                mask = mask | self._block_mask(xyz, self._n_blocks[g], self._block_radius_km[g])
            if not mask.any():
                continue
            idx = self._group_indices(g)
            # (batch, grid) -> broadcast over time, any middle dims and the group's variables.
            mask = mask.view(batch_size, *([1] * (x.ndim - 3)), grid, 1)
            sub = x[:, :n_input, ..., idx]
            x[:, :n_input, ..., idx] = sub.masked_fill(mask, torch.nan)

        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """No-op: dropout is not reversible."""
        return x if in_place else x.clone()
