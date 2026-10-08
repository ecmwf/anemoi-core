# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The base class of the samples, and the helpers shared by its kinds."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Self

import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import Source


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class SourceSample(ABC):
    """One dataset's contribution to one sample, as produced by a data reader.

    Everything needed to build the collated
    :class:`~anemoi.models.data.sources.Source` travels with the sample, so
    collation needs no side channel. Build samples with
    :func:`~anemoi.models.data.sample.create_source_sample`; each kind is registered
    in :data:`~anemoi.models.data.sample.sample_registry`.

    Parameters
    ----------
    data : torch.Tensor or None
        The sample's data, laid out per :attr:`layout`. ``None`` for a sample that
        only describes its geometry (e.g. an inference target template).
    variables : list[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout
        The **per-sample** layout, without a batch axis.
    latitudes : torch.Tensor
        ``(N,)`` array of latitudes in **radians**.
    longitudes : torch.Tensor
        ``(N,)`` array of longitudes in **radians**.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis, as produced by
        ``anemoi-datasets``.
    """

    data: torch.Tensor | None
    variables: list[str]
    layout: TensorLayout
    latitudes: torch.Tensor
    longitudes: torch.Tensor
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def coordinates(self) -> torch.Tensor:
        """Return the coordinates as a ``(N, 2)`` tensor stacking ``(latitudes, longitudes)`` in radians."""
        return torch.stack([self.latitudes, self.longitudes], dim=-1)

    @classmethod
    @abstractmethod
    def from_validated(cls, *, n_points: int, device: torch.device | None, **kwargs: Any) -> Self:
        """Build a sample from inputs already checked by :func:`create_source_sample`.

        ``kwargs`` holds the common fields, already converted, plus the arguments
        specific to this kind, which the subclass validates and converts here.
        ``n_points`` is the number of points and ``device`` the data's device
        (``None`` without data).
        """
        ...

    @classmethod
    @abstractmethod
    def collated_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Return the layout of the collated source, given the per-sample ``layout``."""
        ...

    @classmethod
    @abstractmethod
    def collate(cls, name: str, samples: Sequence[Self]) -> Source:
        """Collate the samples of one dataset into a batched source."""
        ...


def require_data(name: str, samples: Sequence[SourceSample]) -> None:
    """Reject samples without data: a :class:`Source` always holds data."""
    if any(s.data is None for s in samples):
        msg = f"Dataset {name!r} has samples without data, which cannot be collated into a Source."
        raise ValueError(msg)


def validate_layout_against(name: str, layout: TensorLayout, data: torch.Tensor | list[torch.Tensor]) -> None:
    """Check every non-None axis position is a valid axis of the collated tensor.

    Catches reader-side mistakes early instead of letting them surface as cryptic
    errors deep inside model code.
    """
    ref = data[0] if isinstance(data, list) else data
    ndim = ref.ndim
    for axis_name in TensorLayout.AXES:
        pos = getattr(layout, axis_name)
        if pos is None:
            continue
        if not (-ndim <= pos < ndim):
            msg = (
                f"TensorLayout for dataset {name!r} declares {axis_name}={pos} but the "
                f"collated tensor only has {ndim} dimensions (shape={tuple(ref.shape)}). "
                f"Layout: {layout!r}."
            )
            raise ValueError(msg)
