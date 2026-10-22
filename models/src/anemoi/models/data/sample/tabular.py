# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import ClassVar
from typing import Self

import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sample import sample_registry
from anemoi.models.data.sample.base import BaseSample
from anemoi.models.data.sources import TabularSource
from anemoi.models.data.sources import TabularTemplate
from anemoi.models.distributed.shapes import ShardSizes


@sample_registry.register("tabular")
@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class TabularSample(BaseSample):
    """A sample of points that change from sample to sample (e.g. observations).

    ``data`` is ``(ensemble, grid, variables)``. The time windows are stacked along
    the grid axis and described by :attr:`boundaries`.

    Parameters
    ----------
    timedeltas : torch.Tensor
        ``(N,)`` time offset of each point, in seconds.
    boundaries : tuple[slice, ...]
        One slice per time window along the grid axis. This is the only record of
        the sample's time axis.
    shard_sizes : list[ShardSizes], optional
        Read-time sharding descriptor, one per time window.
    """

    timedeltas: torch.Tensor
    boundaries: tuple[slice, ...]
    shard_sizes: list[ShardSizes] | None = None

    source_type: ClassVar[type[TabularSource]] = TabularSource
    template_type: ClassVar[type[TabularTemplate]] = TabularTemplate
    _METADATA_ATTRS: ClassVar[tuple[str, ...]] = ("variables", "statistics")
    _DYNAMIC_ATTRS: ClassVar[tuple[str, ...]] = ("coordinates", "timedeltas", "boundaries", "shard_sizes")

    def __post_init__(self) -> None:
        if self.layout.has_axis("time") or self.layout.has_axis("batch"):
            msg = (
                f"{self.__class__.__name__} requires a layout without time and batch axes; the time windows are "
                f"given by 'boundaries' and the batch is a list. Got {self.layout!r}."
            )
            raise ValueError(msg)

    def with_time_offsets(self, window_offsets: Sequence[float]) -> Self:
        """Return the sample with each window's timedeltas shifted by that window's offset, in seconds.

        Moves timedeltas measured from each window's own date onto a common reference time.
        """
        if len(window_offsets) != len(self.boundaries):
            msg = (
                f"{self.__class__.__name__} has {len(self.boundaries)} time windows, "
                f"but {len(window_offsets)} window offsets were given."
            )
            raise ValueError(msg)
        shift = torch.zeros_like(self.timedeltas)
        for window, offset in zip(self.boundaries, window_offsets, strict=True):
            shift[window] = offset
        return replace(self, timedeltas=self.timedeltas + shift)

    @classmethod
    def _collate_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Return the layout of the collated source, given the per-sample ``layout``."""
        return layout

    @classmethod
    def _collate_data(cls, samples: Sequence[Self]) -> list[torch.Tensor]:
        """Collate the per-sample data tensors into a list of tensors."""
        return [s.data for s in samples]

    @classmethod
    def _collate_dynamic_attrs(cls, samples: Sequence[Self]) -> dict[str, Any]:
        """Keep one value per sample, collapsing unsharded ``shard_sizes`` to ``None``.

        :class:`TabularSource` treats ``shard_sizes is None`` as replicated, so a list of
        ``None`` entries would wrongly read as sharded. A mix of sharded and unsharded
        samples cannot be gathered consistently and is rejected.
        """
        # No zero-argument super(): slots=True rebuilds the class, which breaks it.
        attrs = {attr: [getattr(s, attr) for s in samples] for attr in cls._DYNAMIC_ATTRS}
        shard_sizes = attrs["shard_sizes"]
        n_unsharded = sum(sizes is None for sizes in shard_sizes)
        if n_unsharded == len(shard_sizes):
            attrs["shard_sizes"] = None
        elif n_unsharded:
            msg = f"The batch mixes sharded and unsharded samples ({n_unsharded} of {len(shard_sizes)} unsharded)."
            raise ValueError(msg)
        return attrs

    @classmethod
    def from_validated(
        cls,
        *,
        n_points: int,
        device: torch.device | None,
        timedeltas: Any = None,
        boundaries: Sequence[slice | tuple[int, int]] | None = None,
        **common: Any,
    ) -> Self:
        """Convert the ``timedeltas`` to float32 and the ``(start, stop)`` boundaries to slices."""
        timedeltas, boundaries = cls._validate_time_windows(n_points, device, timedeltas, boundaries)
        return cls(**common, timedeltas=timedeltas, boundaries=boundaries)

    @classmethod
    def template_from_validated(
        cls,
        *,
        n_points: int,
        device: torch.device | None,
        coordinates: torch.Tensor,
        timedeltas: Any = None,
        boundaries: Sequence[slice | tuple[int, int]] | None = None,
        shard_sizes: list[ShardSizes] | None = None,
        batch_size: int = 1,
        time_size: int | None = None,
        **common: Any,
    ) -> TabularTemplate:
        """Build a :class:`TabularTemplate` whose ``batch_size`` samples all share this geometry.

        The geometry describes one sample, as for :meth:`from_validated`. A tabular template
        keeps it per sample and takes its batch and time sizes from those lists, so the
        geometry is repeated ``batch_size`` times. ``time_size`` is not used: the decoded time
        windows are the ``boundaries`` the caller asked for, each point placed by its timedelta.
        """
        del time_size
        timedeltas, boundaries = cls._validate_time_windows(n_points, device, timedeltas, boundaries)
        return cls.template_type(
            **common,
            coordinates=[coordinates] * batch_size,
            timedeltas=[timedeltas] * batch_size,
            boundaries=[boundaries] * batch_size,
            shard_sizes=None if shard_sizes is None else [shard_sizes] * batch_size,
        )

    @staticmethod
    def _validate_time_windows(
        n_points: int,
        device: torch.device | None,
        timedeltas: Any,
        boundaries: Sequence[slice | tuple[int, int]] | None,
    ) -> tuple[torch.Tensor, tuple[slice, ...]]:
        """Return the ``timedeltas`` as float32 and the ``(start, stop)`` boundaries as slices, checked."""
        if timedeltas is None or boundaries is None:
            msg = "tabular samples require timedeltas and boundaries."
            raise ValueError(msg)

        timedeltas = torch.as_tensor(timedeltas, dtype=torch.float32, device=device).reshape(-1)
        if timedeltas.shape[0] != n_points:
            msg = f"{timedeltas.shape[0]} timedeltas were given for {n_points} points."
            raise ValueError(msg)

        boundaries = tuple(b if isinstance(b, slice) else slice(int(b[0]), int(b[1])) for b in boundaries)
        if any(b.stop is not None and b.stop > n_points for b in boundaries):
            msg = f"boundaries {boundaries} extend past the {n_points} points of the sample."
            raise ValueError(msg)

        return timedeltas, boundaries
