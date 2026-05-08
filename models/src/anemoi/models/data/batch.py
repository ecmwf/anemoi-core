# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Typed batch envelope carrying data, coordinates and metadata.

A :class:`Batch` is the object that flows from the dataloader to the model.
It bundles, per dataset:

* ``data`` — either a stacked tensor with shape
  ``(batch, time, ensemble, grid, vars)`` (gridded datasets) or a
  ``list[torch.Tensor]`` of length ``batch`` with per-sample shape
  ``(ensemble=1, grid_i, vars)`` (sparse observation datasets, where
  ``grid_i`` varies per sample).
* ``coords`` — per-dataset coordinate tensors (e.g. ``latitudes``,
  ``longitudes``, and for sparse readers also ``timedeltas``) in **radians**.
  For sparse datasets each coord key holds a ``list[torch.Tensor]`` of
  length ``batch``, mirroring the structure of ``data``.
* ``metadata`` — extension point for per-batch / per-dataset information
  (masks, timestamps, sparse-obs ``boundaries``, ...).

Coordinate tensors for static-grid datasets are shared by reference across
the batch dimension to avoid per-worker / per-step copies. The set of static
dataset names is stored under ``metadata["static_coords"]`` and consulted by
:meth:`Batch.to` to skip redundant host-to-device transfers.

Sparse observation datasets carry their per-time split information under
``metadata[dataset_name]["boundaries"]`` as a ``list[tuple[slice, ...]]``
(one entry per batch sample). These are Python ``slice`` objects and are
intentionally **not** transferred to device by :meth:`Batch.to`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch.utils.data import default_collate

# Key in ``Batch.metadata`` listing dataset names whose coordinate tensors are
# static (allocated once, shared by reference, not transferred per step).
STATIC_COORDS_META_KEY = "static_coords"

# Reserved per-dataset metadata key carrying sparse-obs per-time boundaries
# as a ``list[tuple[slice, ...]]`` (one entry per batch sample). Untouched by
# device transfer / pinning since ``slice`` is not a tensor.
BOUNDARIES_META_KEY = "boundaries"


def _to_device(value, device, *, non_blocking: bool):
    """Recursively move tensors to ``device``; pass non-tensors through.

    Handles the union types that flow through a sparse-aware :class:`Batch`:
    plain :class:`torch.Tensor`, ``list[torch.Tensor]`` (sparse per-sample
    payloads), and any non-tensor leaf (e.g. ``slice`` objects in the
    ``boundaries`` metadata) which is returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [_to_device(v, device, non_blocking=non_blocking) for v in value]
    return value


def _pin_memory(value):
    """Recursively pin tensors; pass non-tensors through. See :func:`_to_device`."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    return value


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
    coords : dict[str, dict[str, torch.Tensor | list[torch.Tensor]]]
        Per-dataset coordinate tensors keyed by name (e.g. ``"latitudes"``,
        ``"longitudes"``, ``"timedeltas"``) in **radians**. For static-grid
        datasets the tensors are shared by reference across the batch
        dimension and have shape ``(grid, ...)``; for dynamic-grid (gridded
        non-static) datasets they have a leading batch (and optionally
        time) dimension; for sparse datasets each value is a
        ``list[torch.Tensor]`` of length ``batch``.
    metadata : dict[str, Any], optional
        Free-form per-batch metadata. Reserved keys:

        * ``"static_coords"`` — :class:`frozenset` of dataset names whose
          coordinates are static. :meth:`to` and :meth:`pin_memory` will
          skip these entries.
        * ``<dataset_name>`` — for sparse datasets, a dict that may carry
          ``"boundaries": list[tuple[slice, ...]]`` (one per batch sample).
          Non-tensor leaves are passed through device transfer untouched.
    """

    data: dict[str, torch.Tensor | list[torch.Tensor]]
    coords: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- access

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Names of the datasets present in this batch (insertion order)."""
        return tuple(self.data.keys())

    @property
    def static_coord_datasets(self) -> frozenset[str]:
        """Dataset names whose coordinate tensors are static."""
        return frozenset(self.metadata.get(STATIC_COORDS_META_KEY, ()))

    def is_static_coords(self, dataset_name: str) -> bool:
        """Return whether ``dataset_name``'s coordinates are static."""
        return dataset_name in self.static_coord_datasets

    def __getitem__(self, dataset_name: str) -> torch.Tensor | list[torch.Tensor]:
        """Return the data payload for ``dataset_name`` (mapping behaviour).

        Returns a stacked :class:`torch.Tensor` for gridded datasets and a
        ``list[torch.Tensor]`` of length ``batch`` for sparse observation
        datasets. For the richer per-dataset view that bundles data,
        coords and the static flag, use :meth:`view`.
        """
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        return self.data[dataset_name]

    def view(self, dataset_name: str) -> "DatasetView":
        """Return a per-dataset view bundling data, coords and the static flag."""
        if dataset_name not in self.data:
            msg = f"Dataset {dataset_name!r} not found in batch (have {list(self.data)})."
            raise KeyError(msg)
        return DatasetView(
            name=dataset_name,
            data=self.data[dataset_name],
            coords=self.coords.get(dataset_name, {}),
            is_static=self.is_static_coords(dataset_name),
        )

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self.data

    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):  # noqa: D401 - mapping protocol
        """Return the dataset names (mapping protocol)."""
        return self.data.keys()

    def values(self):  # noqa: D401 - mapping protocol
        """Return the data tensors (mapping protocol)."""
        return self.data.values()

    def items(self):  # noqa: D401 - mapping protocol
        """Return ``(name, tensor)`` pairs (mapping protocol)."""
        return self.data.items()

    # -------------------------------------------------------- device transfer

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
        new_coords: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] = {}
        for name, per_dataset in self.coords.items():
            if name in static:
                # Skip the H2D transfer; the tensors are already on device.
                new_coords[name] = per_dataset
                continue
            new_coords[name] = {key: _to_device(value, device, non_blocking=non_blocking) for key, value in per_dataset.items()}

        return Batch(data=new_data, coords=new_coords, metadata=self.metadata)

    def pin_memory(self) -> "Batch":
        """Pin host memory for non-static tensors. Static coords are left untouched."""
        new_data = {name: _pin_memory(tensor) for name, tensor in self.data.items()}

        static = self.static_coord_datasets
        new_coords: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] = {}
        for name, per_dataset in self.coords.items():
            if name in static:
                new_coords[name] = per_dataset
                continue
            new_coords[name] = {key: _pin_memory(value) for key, value in per_dataset.items()}

        return Batch(data=new_data, coords=new_coords, metadata=self.metadata)

    # ------------------------------------------------------------- transform

    def with_data(self, new_data: dict[str, torch.Tensor]) -> "Batch":
        """Return a new :class:`Batch` with ``data`` replaced by ``new_data``.

        ``coords`` and ``metadata`` are shared by reference with the
        receiver. This preserves static-coord identity (no extra H2D, no
        copy) and is the recommended way for tasks to slice or transform
        the data tensors while keeping the coordinate envelope intact.

        Parameters
        ----------
        new_data : dict[str, torch.Tensor]
            Replacement data tensors. Must cover the same dataset names
            as ``self.data``.

        Returns
        -------
        Batch
            A new frozen :class:`Batch` sharing ``self.coords`` and
            ``self.metadata`` by reference.
        """
        if set(new_data.keys()) != set(self.data.keys()):
            msg = (
                "with_data: new_data keys must match the existing dataset names "
                f"(got {sorted(new_data)}, expected {sorted(self.data)})."
            )
            raise ValueError(msg)
        return Batch(data=new_data, coords=self.coords, metadata=self.metadata)

    def node_coords(self, dataset_name: str) -> torch.Tensor | None:
        """Return per-node ``(num_nodes, 2)`` lat/lon coords for ``dataset_name``.

        Stacks ``coords[dataset_name]["latitudes"]`` and ``["longitudes"]``
        along a trailing dimension. Coordinates are expected in **radians**
        (see the module docstring) and to share the layout used at graph
        registration time, so the result can be fed directly to
        :meth:`anemoi.models.layers.graph.NamedNodesAttributes.forward`
        via the ``coords=`` kwarg.

        Returns ``None`` when the dataset has no ``coords`` entry or is
        missing one of ``latitudes``/``longitudes`` — the model layer
        then falls back to its registered static buffer (preserving
        full backward compatibility with checkpoints trained before
        the typed-batch refactor).
        """
        per_dataset = self.coords.get(dataset_name)
        if not per_dataset:
            return None
        latitudes = per_dataset.get("latitudes")
        longitudes = per_dataset.get("longitudes")
        if latitudes is None or longitudes is None:
            return None
        return torch.stack([latitudes, longitudes], dim=-1)

    # ----------------------------------------------------------- construction

    @staticmethod
    def collate(
        samples: list[dict[str, dict[str, Any]]],
        *,
        static_coord_datasets: Iterable[str] = (),
    ) -> "Batch":
        """Collate a list of per-sample dicts into a :class:`Batch`.

        Each sample must be a mapping ``{dataset_name: payload}`` where
        ``payload`` is a mapping with a required ``"data"`` key, an
        optional ``"coords"`` mapping and an optional ``"metadata"``
        mapping.

        Two payload shapes are supported and dispatched on the presence of
        ``BOUNDARIES_META_KEY`` (``"boundaries"``) inside
        ``payload["metadata"]``:

        * **Gridded** — no ``"boundaries"`` metadata. ``payload["data"]`` is
          a :class:`torch.Tensor` of uniform shape across samples; data and
          (non-static) coords are stacked along a new leading batch
          dimension via :func:`torch.utils.data.default_collate`. Datasets
          listed in ``static_coord_datasets`` reuse the first sample's
          coords by reference (no stacking, no copy).
        * **Sparse** — ``payload["metadata"]["boundaries"]`` is present (set
          by :meth:`anemoi.training.data.data_reader.ObservationDataReader._unpack_sample`).
          ``payload["data"]`` is a per-sample :class:`torch.Tensor` of
          shape ``(E=1, N_i, V)`` whose ``N_i`` varies between samples.
          ``data[name]`` becomes a ``list[torch.Tensor]`` of length ``B``;
          each coord key becomes a ``list[torch.Tensor]`` of length ``B``;
          per-sample ``payload["metadata"]`` is collected into
          ``Batch.metadata[name]`` with each leaf gathered into a list of
          length ``B`` (so ``"boundaries"`` becomes
          ``list[tuple[slice, ...]]``). Sparse datasets must not appear in
          ``static_coord_datasets`` — see
          :attr:`anemoi.training.data.data_reader.BaseAnemoiReader.is_static_grid`.
        """
        if not samples:
            msg = "Cannot collate an empty list of samples."
            raise ValueError(msg)

        static = frozenset(static_coord_datasets)

        # Discover the dataset names from the first sample; assume consistent.
        first = samples[0]
        dataset_names = tuple(first.keys())

        collated_data: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        collated_coords: dict[str, dict[str, torch.Tensor | list[torch.Tensor]]] = {}
        per_dataset_metadata: dict[str, dict[str, list[Any]]] = {}

        for name in dataset_names:
            first_payload = first[name]
            sample_meta = first_payload.get("metadata") or {}
            # Sparse observation samples are tagged by the presence of the
            # ``boundaries`` key in their per-sample metadata (set by
            # ``ObservationDataReader._unpack_sample``). Gridded samples
            # never carry it, so this is a self-describing dispatch.
            is_sparse = BOUNDARIES_META_KEY in sample_meta

            if is_sparse:
                if name in static:
                    msg = (
                        f"Dataset {name!r} produced a sparse (boundaries-tagged) "
                        "data payload but is listed in static_coord_datasets; "
                        "sparse datasets must have is_static_grid=False."
                    )
                    raise ValueError(msg)

                # data: list[Tensor] of length B, no stacking.
                collated_data[name] = [sample[name]["data"] for sample in samples]

                # coords: each key becomes list[Tensor] of length B.
                if "coords" in first_payload:
                    coord_keys = tuple(first_payload["coords"].keys())
                    collated_coords[name] = {
                        key: [sample[name]["coords"][key] for sample in samples] for key in coord_keys
                    }

                # metadata: each leaf becomes a list of length B (one per sample).
                meta_keys = tuple(sample_meta.keys())
                per_dataset_metadata[name] = {
                    key: [sample[name].get("metadata", {}).get(key) for sample in samples] for key in meta_keys
                }
                continue

            # Gridded path: stack data via default_collate.
            collated_data[name] = default_collate([sample[name]["data"] for sample in samples])

            if "coords" not in first_payload:
                continue
            if name in static:
                # Use the first sample's coords by reference; do NOT copy or
                # repeat across the batch dimension.
                collated_coords[name] = first_payload["coords"]
            else:
                collated_coords[name] = default_collate(
                    [sample[name]["coords"] for sample in samples],
                )

        metadata: dict[str, Any] = {}
        if static:
            metadata[STATIC_COORDS_META_KEY] = frozenset(static)
        metadata.update(per_dataset_metadata)

        return Batch(data=collated_data, coords=collated_coords, metadata=metadata)


@dataclass(frozen=True, slots=True)
class DatasetView:
    """Per-dataset view returned by :meth:`Batch.__getitem__`."""

    name: str
    data: torch.Tensor | list[torch.Tensor]
    coords: dict[str, torch.Tensor | list[torch.Tensor]]
    is_static: bool
