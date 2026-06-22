# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.distributed import ProcessGroup

from anemoi.models.data.views import SourceView
from anemoi.models.data.views import TensorLayout
from anemoi.models.data.views import create_source_view

LOGGER = logging.getLogger(__name__)

# Key in ``Batch.metadata`` listing dataset names whose coordinate tensors are
# static (allocated once, shared by reference, not transferred per step).
STATIC_COORDS_META_KEY = "static_coords"

# Reserved per-dataset metadata key carrying sparse-obs per-time boundaries
# as a ``list[tuple[slice, ...]]`` (one entry per batch sample). Untouched by
# device transfer / pinning since ``slice`` is not a tensor.
BOUNDARIES_META_KEY = "boundaries"


IndicesType = slice | Sequence[int] | int


def _to_device(value, device, *, non_blocking: bool):
    """Recursively move tensors to ``device``, pass non-tensors through."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [_to_device(v, device, non_blocking=non_blocking) for v in value]
    return value


def _pin_memory(value):
    """Recursively pin tensors, pass non-tensors through. See :func:`_to_device`."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    return value


def _broadcast_to_dict(value, keys: Iterable[str]) -> dict[str, Any]:
    """Broadcast a non-dict value to a dict with the same value for each key."""
    if isinstance(value, dict):
        return value
    return {key: value for key in keys}


@dataclass(frozen=True, slots=True)
class Batch:
    """Typed batch carrying data, coordinates and metadata.

    Parameters
    ----------
    data : dict[str, torch.Tensor | list[torch.Tensor]]
        Per-dataset input. For gridded datasets a single stacked tensor of
        shape ``(batch, time, ensemble, grid, vars)``; for sparse
        observation datasets a ``list[torch.Tensor]`` of length ``batch``,
        one entry per sample with shape ``(ensemble=1, grid_i, vars)``.
    coordinates : dict[str, torch.Tensor | list[torch.Tensor]]
        Per-dataset ``(N, 2)`` coordinate tensor stacking
        ``(latitudes, longitudes)`` in **radians**. For static-grid
        datasets the tensor is shared by reference across the batch
        dimension and has shape ``(grid, 2)``; for dynamic-grid (gridded
        non-static) datasets it has shape ``(batch, grid, 2)``; for
        sparse datasets it is a ``list[torch.Tensor]`` of length ``batch``
        with per-sample shape ``(grid_i, 2)``.
    metadata : dict[str, Any], optional
        Free-form per-batch metadata. Reserved keys:

        * ``"static_coords"`` — :class:`frozenset` of dataset names whose
          coordinates are static. :meth:`to` and :meth:`pin_memory` will
          skip these entries.
        * ``<dataset_name>`` — for sparse datasets, a dict that may carry
          ``"boundaries": list[tuple[slice, ...]]`` (one per batch sample).
          Non-tensor leaves are passed through device transfer untouched.
    grid_sizes : dict[str, int], optional
        Per-dataset full grid sizes (number of grid points before any
        distributed sharding). Populated during collation from sample
        payloads. For static-grid datasets this equals
        ``sum(dataset.grids)``; for observation datasets it equals the
        grid dimension of the data tensor.
    timedeltas : dict[str, torch.Tensor | list[torch.Tensor]], optional
        Per-dataset per-point time-offset tensors (sparse observation
        datasets only) of shape ``(N,)``. Stored separately from
        :attr:`coordinates` so the spatial and temporal axes can be
        consumed independently. For sparse datasets the value is a
        ``list[torch.Tensor]`` of length ``batch``. Transferred to
        device alongside data.
    shard_sizes : dict[str, Any], optional
        Per-dataset sharding descriptors from readers. For gridded readers
        values are a single ``ShardSizes`` over the static grid axis; for
        sparse/tabular readers values are ``list[ShardSizes]`` (one entry
        per window boundary for each sample payload).
    layouts : dict[str, TensorLayout], optional
        Per-dataset :class:`TensorLayout` descriptors mapping logical
        axes (time, ensemble, grid, variables) to physical dimension
        positions. Not transferred to device.
    """

    data: dict[str, torch.Tensor | list[torch.Tensor]]
    coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    grid_sizes: dict[str, int] = field(default_factory=dict)
    timedeltas: dict[str, torch.Tensor | list[torch.Tensor]] = field(default_factory=dict)
    shard_sizes: dict[str, Any] = field(default_factory=dict)
    layouts: dict[str, TensorLayout] = field(default_factory=dict)
    variables: dict[str, list[str]] = field(default_factory=dict)
    statistics: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of samples (batch size) in this batch."""
        batch_sizes = {}
        for name, view in self.data.items():
            if isinstance(view, list):
                batch_sizes[name] = len(view)
            else:
                batch_sizes[name] = view.shape[self.layouts[name].batch]

        assert len(set(batch_sizes.values())) == 1, f"Inconsistent batch sizes across datasets: {batch_sizes}"
        return next(iter(batch_sizes.values()))

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Names of the datasets present in this batch (insertion order)."""
        return tuple(self.data.keys())

    @property
    def device(self) -> torch.device:
        """Device the batch data lives on.

        This is derived from the first dataset's data payload.
        All data tensors are expected to share the same device
        after the call to :meth:`to`.
        """
        if not self.data:
            raise ValueError("Cannot determine device of an empty batch.")
        first = next(iter(self.data.values()))
        tensor = first[0] if isinstance(first, list) else first
        return tensor.device

    @property
    def static_coord_datasets(self) -> frozenset[str]:
        """Dataset names whose coordinate tensors are static."""
        return frozenset(self.metadata.get(STATIC_COORDS_META_KEY, ()))

    def is_static_coords(self, dataset_name: str) -> bool:
        """Return whether ``dataset_name``'s coordinates are static."""
        return dataset_name in self.static_coord_datasets

    def __repr__(self) -> str:
        """Compact summary of per-dataset shapes, layouts and static-coords flag.

        Designed for debug logging — describes each dataset by its data
        shape (or ``list[shape]`` for sparse), its :class:`TensorLayout`
        (or ``<no layout>``) and whether its coordinates are static.
        """
        if not self.data:
            return "Batch(<empty>)"

        lines = ["Batch("]
        for name in self.dataset_names:
            payload = self.data[name]
            if isinstance(payload, list):
                shapes = [tuple(t.shape) for t in payload]
                shape_repr = f"list[{len(payload)}] of shapes={shapes}"
            else:
                shape_repr = f"shape={tuple(payload.shape)}"
            layout_repr = repr(self.layouts[name]) if name in self.layouts else "<no layout>"
            static_repr = " static_coords" if self.is_static_coords(name) else ""
            lines.append(f"  {name}: {shape_repr} layout={layout_repr}{static_repr}")
        lines.append(")")
        return "\n".join(lines)

    def __getitem__(self, dataset_name: str) -> "SourceView":
        """Return a per-dataset view bundling data, coordinates, layout and metadata."""
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        layout = self.layouts.get(dataset_name)
        if layout is None:
            msg = (
                f"Dataset {dataset_name!r} has no TensorLayout in this batch; "
                "layout is required for view-based access. "
                "Set batch.layouts[name] or pass 'layout' in the sample payload during collation."
            )
            raise ValueError(msg)

        per_dataset_meta = (
            self.metadata.get(dataset_name) if isinstance(self.metadata.get(dataset_name), dict) else None
        )
        boundaries = per_dataset_meta.get(BOUNDARIES_META_KEY) if per_dataset_meta else None
        return create_source_view(
            name=dataset_name,
            data=self.data[dataset_name],
            variables=self.variables.get(dataset_name),
            statistics=self.statistics[dataset_name],
            coordinates=self.coordinates.get(dataset_name),
            is_static=self.is_static_coords(dataset_name),
            timedeltas=self.timedeltas.get(dataset_name),
            layout=layout,
            boundaries=boundaries,
            shard_sizes=self.shard_sizes.get(dataset_name),
        )

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self.data

    def __len__(self) -> int:
        return len(self.dataset_names)

    def keys(self):  # noqa: D401 - mapping protocol
        """Return the dataset names (mapping protocol)."""
        return self.data.keys()

    def values(self):  # noqa: D401 - mapping protocol
        """Return the data tensors (mapping protocol)."""
        return (self[dataset_name] for dataset_name in self.dataset_names)

    def items(self):  # noqa: D401 - mapping protocol
        """Return ``(name, tensor)`` pairs (mapping protocol)."""
        return ((dataset_name, self[dataset_name]) for dataset_name in self.dataset_names)

    def to(self, device: torch.device | str, *, non_blocking: bool = True) -> "Batch":
        """Move the batch to ``device``.

        Static coordinate tensors (those whose dataset name is listed in
        ``metadata["static_coords"]``) are skipped: they are expected to have
        been moved to the model device once, at training start, and never
        transferred again. Non-tensor metadata leaves (e.g. ``boundaries``
        ``slice`` objects for sparse observations) are passed through
        untouched.

        Returns a new :class:`Batch`; the receiver is not mutated.
        """
        new_data = {name: _to_device(tensor, device, non_blocking=non_blocking) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name, value in self.coordinates.items():
            if name in static:
                # Skip the H2D transfer; the tensors are already on device.
                new_coordinates[name] = value
                continue
            new_coordinates[name] = _to_device(value, device, non_blocking=non_blocking)

        new_timedeltas = {
            name: _to_device(value, device, non_blocking=non_blocking) for name, value in self.timedeltas.items()
        }

        return Batch(
            data=new_data,
            coordinates=new_coordinates,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            shard_sizes=self.shard_sizes,
            layouts=self.layouts,
            variables=self.variables,
            statistics=self.statistics,
        )

    def pin_memory(self) -> "Batch":
        """Pin host memory for non-static tensors. Static coords are left untouched."""
        new_data = {name: _pin_memory(tensor) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for name, value in self.coordinates.items():
            if name in static:
                new_coordinates[name] = value
                continue
            new_coordinates[name] = _pin_memory(value)

        new_timedeltas = {name: _pin_memory(value) for name, value in self.timedeltas.items()}

        return Batch(
            data=new_data,
            coordinates=new_coordinates,
            metadata=self.metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            shard_sizes=self.shard_sizes,
            layouts=self.layouts,
            variables=self.variables,
            statistics=self.statistics,
        )

    def with_data(self, new_data: dict[str, torch.Tensor | list[torch.Tensor]]) -> "Batch":
        """Return a new :class:`Batch` with ``data`` replaced by ``new_data``.

        ``coordinates``, ``timedeltas`` and ``metadata`` are shared by
        reference with the receiver. This preserves static-coord identity
        (no extra H2D, no copy) and is the recommended way for tasks to
        slice or transform the data tensors while keeping the coordinate
        envelope intact.

        Parameters
        ----------
        new_data : dict[str, torch.Tensor | list[torch.Tensor]]
            Replacement data tensors. Must cover the same dataset names
            as ``self.data``.

        Returns
        -------
        Batch
            A new frozen :class:`Batch` sharing ``self.coordinates``,
            ``self.timedeltas`` and ``self.metadata`` by reference.
        """
        if set(new_data.keys()) == set(self.data.keys()):
            return Batch(
                data=new_data,
                coordinates=self.coordinates,
                metadata=self.metadata,
                grid_sizes=self.grid_sizes,
                timedeltas=self.timedeltas,
                shard_sizes=self.shard_sizes,
                layouts=self.layouts,
                variables=self.variables,
                statistics=self.statistics,
            )

        new_data_keys = set(new_data.keys())
        metadata_static_coords = self.metadata.get(STATIC_COORDS_META_KEY, frozenset())
        metadata_static_coords &= new_data_keys
        return Batch(
            new_data,
            coordinates={name: self.coordinates[name] for name in new_data_keys},
            metadata={STATIC_COORDS_META_KEY: metadata_static_coords}
            | {name: self.metadata[name] for name in new_data_keys if name in self.metadata},
            grid_sizes={name: self.grid_sizes[name] for name in new_data_keys},
            timedeltas={name: self.timedeltas[name] for name in new_data_keys if name in self.timedeltas},
            shard_sizes={name: self.shard_sizes[name] for name in new_data_keys if name in self.shard_sizes},
            layouts={name: self.layouts[name] for name in new_data_keys},
            variables={name: self.variables[name] for name in new_data_keys},
            statistics=self.statistics,
        )

    def _update_source(self, source_name: str, source_view: SourceView) -> "Batch":
        """Return a new batch with one dataset replaced from a ``SourceView``."""
        new_data = {**self.data, source_name: source_view.data}

        new_variables = dict(self.variables)
        if source_view.variables is not None:
            new_variables[source_name] = source_view.variables
        else:
            new_variables.pop(source_name, None)

        new_coordinates = dict(self.coordinates)
        if source_view.coordinates is None:
            new_coordinates.pop(source_name, None)
        else:
            new_coordinates[source_name] = source_view.coordinates

        new_timedeltas = dict(self.timedeltas)
        if source_view.timedeltas is None:
            new_timedeltas.pop(source_name, None)
        else:
            new_timedeltas[source_name] = source_view.timedeltas

        new_metadata = dict(self.metadata)
        if source_view.boundaries is not None:
            per_dataset_meta = new_metadata.get(source_name)
            per_dataset_meta = dict(per_dataset_meta) if isinstance(per_dataset_meta, dict) else {}
            per_dataset_meta[BOUNDARIES_META_KEY] = source_view.boundaries
            new_metadata[source_name] = per_dataset_meta

        return Batch(
            data=new_data,
            coordinates=new_coordinates,
            metadata=new_metadata,
            grid_sizes=self.grid_sizes,
            timedeltas=new_timedeltas,
            shard_sizes=self.shard_sizes,
            layouts=self.layouts,
            variables=new_variables,
            statistics=self.statistics,
        )

    def select(self, **kwargs) -> "Batch":
        """Return a new :class:`Batch` with per-dataset selection applied."""
        per_source_indices = defaultdict(dict)
        for dim, indices in kwargs.items():
            # if indices is not a dict, broadcast the same indexing to every dataset.
            indices_dict = _broadcast_to_dict(indices, self.dataset_names)
            for source_name, idx in indices_dict.items():
                per_source_indices[source_name][dim] = idx

        batch = self
        for source_name, per_source_idx in per_source_indices.items():
            selected_source = batch[source_name].select(**per_source_idx)
            batch = batch._update_source(source_name, selected_source)

        return batch

    @staticmethod
    def collate(
        samples: list[dict[str, dict[str, Any]]] | dict[str, dict[str, Any]],
        *,
        static_coord_datasets: Iterable[str] = (),
    ) -> "Batch":
        """Collate a list of per-sample dicts into a :class:`Batch`.

        Each sample must be a mapping ``{dataset_name: payload}`` where
        ``payload`` is a mapping with a required ``"data"`` key, an
        optional ``"coordinates"`` tensor (shape ``(N, 2)`` stacking
        latitudes and longitudes in radians), an optional ``"timedeltas"``
        tensor (sparse only) and an optional ``"metadata"`` mapping.

        Two payload shapes are supported and dispatched on the presence of
        ``BOUNDARIES_META_KEY`` (``"boundaries"``) inside
        ``payload["metadata"]``:

        * **Gridded** — no ``"boundaries"`` metadata. ``payload["data"]`` is
          a :class:`torch.Tensor` of uniform shape across samples; data and
          (non-static) coordinates are stacked along a new leading batch
          dimension via :func:`torch.utils.data.default_collate`. Datasets
          listed in ``static_coord_datasets`` reuse the first sample's
          ``coordinates`` tensor by reference (no stacking, no copy).
        * **Sparse** — ``payload["metadata"]["boundaries"]`` is present (set
          by :meth:`anemoi.training.data.data_reader.ObservationDataReader._unpack_sample`).
          ``payload["data"]`` is a per-sample :class:`torch.Tensor` of
          shape ``(E=1, N_i, V)`` whose ``N_i`` varies between samples.
          ``data[name]``, ``coordinates[name]`` and ``timedeltas[name]``
          each become a ``list[torch.Tensor]`` of length ``B``;
          per-sample ``payload["metadata"]`` is collected into
          ``Batch.metadata[name]`` with each leaf gathered into a list of
          length ``B`` (so ``"boundaries"`` becomes
          ``list[tuple[slice, ...]]``). Sparse datasets must not appear in
          ``static_coord_datasets`` — see
          :attr:`anemoi.training.data.data_reader.BaseAnemoiReader.is_static_grid`.
        """
        if isinstance(samples, dict):
            samples = [samples]

        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        static = frozenset(static_coord_datasets)

        # Discover the dataset names from the first sample; assume consistent.
        first = samples[0]
        dataset_names = tuple(first.keys())

        collated_data: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_coordinates: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_timedeltas: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_shard_sizes: dict[str, Any] = {}
        collated_variables: dict[str, list[str]] = {}
        collated_statistics: dict[str, np.ndarray] = {}
        per_dataset_metadata: dict[str, dict[str, list[Any]]] = {}

        for name in dataset_names:
            first_payload = first[name]
            sample_meta = first_payload.get("metadata") or {}
            # Sparse observation samples are tagged by the presence of the
            # ``boundaries`` key in their per-sample metadata (set by
            # ``ObservationDataReader._unpack_sample``). Gridded samples
            # never carry it, so this is a self-describing dispatch.
            is_sparse = BOUNDARIES_META_KEY in sample_meta

            if "variables" in first_payload:
                collated_variables[name] = first_payload["variables"]
                # Assumed that all samples have the same variables

            if "statistics" in first_payload:
                collated_statistics[name] = first_payload["statistics"]
                # Assumed that all samples have the same variables and statistics

            if "grid_shard_sizes" in first_payload:
                collated_shard_sizes[name] = first_payload["grid_shard_sizes"]

            if is_sparse:
                if name in static:
                    msg = (
                        f"Dataset {name!r} produced a sparse (boundaries-tagged) "
                        "data payload but is listed in static_coord_datasets; "
                        "sparse datasets must have is_static_grid=False."
                    )
                    raise ValueError(msg)

                # data / coordinates / timedeltas: list[Tensor] of length B,
                # no stacking — sparse samples have varying N_i.
                collated_data[name] = [sample[name]["data"] for sample in samples]

                if "coordinates" in first_payload:
                    collated_coordinates[name] = [sample[name]["coordinates"] for sample in samples]

                if "timedeltas" in first_payload:
                    collated_timedeltas[name] = [sample[name]["timedeltas"] for sample in samples]

                if "shard_sizes" in first_payload:
                    collated_shard_sizes[name] = [sample[name]["shard_sizes"] for sample in samples]

                # metadata: each leaf becomes a list of length B (one per sample).
                meta_keys = tuple(sample_meta.keys())
                per_dataset_metadata[name] = {
                    key: [sample[name].get("metadata", {}).get(key) for sample in samples] for key in meta_keys
                }
                continue

            # Gridded path: stack data via default_collate.
            collated_data[name] = default_collate([sample[name]["data"] for sample in samples])

            if "coordinates" in first_payload:
                if name in static:
                    # Use the first sample's coordinates tensor by reference;
                    # do NOT copy or repeat across the batch dimension.
                    collated_coordinates[name] = first_payload["coordinates"]
                else:
                    collated_coordinates[name] = default_collate(
                        [sample[name]["coordinates"] for sample in samples],
                    )

            if "timedeltas" in first_payload:
                # Gridded datasets normally don't carry ``timedeltas``
                # if they do, stack them along a new batch dimension
                # just like the other tensors.
                collated_timedeltas[name] = default_collate([sample[name]["timedeltas"] for sample in samples])

        # Grid sizes: extract from sample payloads (all samples in a batch
        # must agree since default_collate requires matching tensor shapes).
        collated_grid_sizes: dict[str, int] = {}
        for name in dataset_names:
            payload = first[name]
            if "grid_size" in payload:
                collated_grid_sizes[name] = payload["grid_size"]

        # Layouts: per-dataset TensorLayout from sample payloads. For
        # gridded datasets the data is stacked along a new leading batch
        # axis, so we shift the layout via ``with_batch_dim()``. For
        # sparse datasets the per-sample tensors are kept as a
        # ``list[Tensor]`` (no new tensor axis is added — the batch
        # dimension is the list itself), so the per-sample layout is
        # stored unchanged.
        collated_layouts: dict[str, TensorLayout] = {}
        for name in dataset_names:
            payload = first[name]
            if "layout" not in payload:
                continue
            sample_layout = payload["layout"]
            if isinstance(collated_data[name], list):
                collated_layouts[name] = sample_layout
            else:
                collated_layouts[name] = sample_layout.with_batch_dim()

        # Sanity-check: every non-None axis position in the layout must be
        # a valid axis of the (possibly per-sample) collated data tensor.
        # This catches reader-side mistakes early instead of letting them
        # surface as cryptic errors deep inside model code.
        for name, layout in collated_layouts.items():
            payload_data = collated_data[name]
            ref = payload_data[0] if isinstance(payload_data, list) else payload_data
            ndim = ref.ndim
            for axis_name in ("batch", "time", "ensemble", "grid", "variables"):
                pos = getattr(layout, axis_name)
                if pos is None:
                    continue
                if not (-ndim <= pos < ndim):
                    msg = (
                        f"TensorLayout for dataset {name!r} declares "
                        f"{axis_name}={pos} but the collated tensor only has "
                        f"{ndim} dimensions (shape={tuple(ref.shape)}). "
                        f"Layout: {layout!r}."
                    )
                    raise ValueError(msg)

        metadata: dict[str, Any] = {}
        if static:
            metadata[STATIC_COORDS_META_KEY] = frozenset(static)

        metadata.update(per_dataset_metadata)

        batch = Batch(
            data=collated_data,
            coordinates=collated_coordinates,
            metadata=metadata,
            grid_sizes=collated_grid_sizes,
            timedeltas=collated_timedeltas,
            shard_sizes=collated_shard_sizes,
            layouts=collated_layouts,
            variables=collated_variables,
            statistics=collated_statistics,
        )
        LOGGER.debug("Batch.collate produced:\n%r", batch)
        return batch

    def allgather(self, group: ProcessGroup | None) -> "Batch":
        """Allgather the batch across the given process group.

        This is a collective operation that synchronizes all processes in
        ``group``. All processes must call this method with the same group
        and have batches of the same size and dataset structure.

        Parameters
        ----------
        group : ProcessGroup or None
            The process group to allgather across.

        Returns
        -------
        Batch
            A new :class:`Batch` with allgathered data.
        """
        batch = self
        for dataset in self.dataset_names:
            view = self[dataset]
            gathered_view = view.allgather(group=group)
            batch = batch._update_source(dataset, gathered_view)

        return batch
