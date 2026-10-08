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
from typing import Any
from typing import ClassVar
from typing import Self

import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import Source


@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class BaseSample(ABC):
    """One dataset's contribution to one sample, as produced by a data reader.

    Everything needed to build the collated :class:`~anemoi.models.data.sources.Source` travels with the sample, so
    collation needs no side channel. Build samples with :func:`~anemoi.models.data.sample.create_source_sample`; each
    kind is registered in :data:`~anemoi.models.data.sample.source_sample_registry`.

    Subclasses must implement :meth:`from_validated`, :meth:`_collate_layout` and
    :meth:`_collate_data`, and set the class variable :attr:`source_type` to the
    :class:`Source` subclass to collate into. Each field is either metadata, listed in
    :attr:`_METADATA_ATTRS` and taken from the first sample by :meth:`_collate_metadata`,
    or dynamic, listed in :attr:`_DYNAMIC_ATTRS` and kept per sample by
    :meth:`_collate_dynamic_attrs`.

    Parameters
    ----------
    data : torch.Tensor or None
        The sample's data, laid out per :attr:`layout`. ``None`` for a sample that
        only describes its geometry (e.g. an inference target template).
    variables : list[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout
        The **per-sample** layout, without a batch axis.
    coordinates : torch.Tensor
        ``(N, 2)`` ``(latitude, longitude)`` of each point, in **radians**.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis, as produced by ``anemoi-datasets``.
    """

    data: torch.Tensor | None
    variables: list[str]
    layout: TensorLayout
    coordinates: torch.Tensor
    statistics: Mapping[str, Any] | None = None

    @classmethod
    @abstractmethod
    def from_validated(
        cls,
        *,
        n_points: int,
        device: torch.device | None,
        coordinates: torch.Tensor,
        **kwargs: Any,
    ) -> Self:
        """Build a sample from inputs already checked by :func:`create_source_sample`.

        ``coordinates`` is an ``(N, 2)`` tensor of ``(latitude, longitude)`` in radians.
        ``kwargs`` holds the other common fields, already converted, plus the arguments
        specific to this kind, which the subclass validates and converts here. ``n_points``
        is the number of points and ``device`` the data's device (``None`` without data).
        """
        ...

    #: The :class:`Source` subclass this kind collates into.
    source_type: ClassVar[type[Source]]

    #: Fields that describe the dataset rather than the sample: identical across the batch.
    _METADATA_ATTRS: ClassVar[tuple[str, ...]] = ("variables", "statistics")

    #: Fields that vary between samples, kept as one value per sample by :meth:`_collate_dynamic_attrs`.
    _DYNAMIC_ATTRS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def _collate_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Return the layout of the collated source, given the per-sample ``layout``."""
        ...

    @classmethod
    @abstractmethod
    def _collate_data(cls, samples: Sequence[Self]) -> torch.Tensor | list[torch.Tensor]:
        """Collate the per-sample data tensors into a single tensor or a list of tensors."""
        ...

    @classmethod
    def _collate_dynamic_attrs(cls, samples: Sequence[Self]) -> dict[str, Any]:
        """Return the fields in :attr:`_DYNAMIC_ATTRS` as lists of one value per sample."""
        return {attr: [getattr(s, attr) for s in samples] for attr in cls._DYNAMIC_ATTRS}

    @classmethod
    def _collate_metadata(cls, samples: Sequence[Self]) -> dict[str, Any]:
        """Return the dataset-level fields, checking every sample agrees with the first.

        The layout is metadata too: every sample shares it, and :meth:`_collate_layout`
        turns it into the collated layout, whether or not the samples have data.
        """
        first_sample = samples[0]
        metadata = {attr: getattr(first_sample, attr) for attr in cls._METADATA_ATTRS}
        metadata["layout"] = cls._collate_layout(first_sample.layout)
        return metadata

    @classmethod
    def collate(cls, name: str, samples: Sequence[Self]) -> Source:
        """Collate the samples of one dataset into a batched :attr:`source_type`.

        The metadata (variables, statistics, layout) is shared by the samples and taken
        once; the data and the per-sample fields are combined by the kind's hooks.
        """
        metadata = cls._collate_metadata(samples)
        attrs = cls._collate_dynamic_attrs(samples)

        # Collate data
        n_without_data = sum(s.data is None for s in samples)
        if 0 < n_without_data < len(samples):
            msg = (
                f"Dataset {name!r} mixes samples with and without data "
                f"({n_without_data} of {len(samples)} without data)."
            )
            raise ValueError(msg)

        if n_without_data:
            data = None
        else:
            data = cls._collate_data(samples)
            validate_layout_against(name, metadata["layout"], data)

        return cls.source_type(name=name, data=data, **metadata, **attrs)


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
