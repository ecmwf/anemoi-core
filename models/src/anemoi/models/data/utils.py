# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import Source
from anemoi.models.data.sources import TabularSource
import torch

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any


def _apply_pairwise_tabular(
    pred: TabularSource,
    target: TabularSource,
    func: Callable[..., torch.Tensor],
    *,
    per_sample_kwargs: dict[str, Sequence[Any]] | None = None,
    **kwargs
) -> torch.Tensor:
    if not isinstance(target, TabularSource):
        msg = f"Other source must be a TabularSource; got {type(target).__name__}."
        raise TypeError(msg)

    if pred.layout != target.layout:
        msg = f"Both sources must have the same layout; got {pred.layout!r} and {target.layout!r}."
        raise ValueError(msg)

    if len(pred.data) != len(target.data):
        msg = f"Both sources must have the same number of samples; got {len(pred.data)} and {len(target.data)}."
        raise ValueError(msg)

    per_sample_kwargs = {} if per_sample_kwargs is None else per_sample_kwargs
    if kwargs.keys() & per_sample_kwargs.keys():
        raise ValueError("Loss arguments cannot be both shared and per-sample.")

    for name, values in per_sample_kwargs.items():
        if len(values) != len(pred.data):
            raise ValueError(f"Loss argument {name!r} requires one value per sample ({len(pred.data)}).")

    losses = []
    non_empty = []
    for i, (pred_sample, target_sample) in enumerate(zip(pred.data, target.data)):
        # Every axis but the ensemble one must line up for this to work.
        if pred.layout.ensemble is None:
            pred_shape = tuple(pred_sample.shape)
            target_shape = tuple(target_sample.shape)
        else:
            ensemble_axis = pred.layout.axis("ensemble", ndim=pred_sample.ndim)
            pred_shape = tuple(size for dim, size in enumerate(pred_sample.shape) if dim != ensemble_axis)
            target_shape = tuple(size for dim, size in enumerate(target_sample.shape) if dim != ensemble_axis)

        assert pred_shape == target_shape, (
            f"Sample {i} of both views must have the same shape apart from the ensemble axis; "
            f"got {tuple(pred_sample.shape)} and {tuple(target_sample.shape)}."
        )
        assert torch.equal(pred.coordinates[i], target.coordinates[i]), (
            f"Sample {i} of both views must have the same coordinates; "
            f"got {pred.coordinates[i]} and {target.coordinates[i]}."
        )
        sample_kwargs = kwargs | {name: values[i] for name, values in per_sample_kwargs.items()}
        losses.append(
            func(
                pred_sample,
                target_sample,
                layout=pred.layout,
                statistics=pred.statistics,
                name_to_index=pred.name_to_index,
                **sample_kwargs,
            ),
        )
        # A fully-empty worker contributes a graph-connected zero without
        # reducing the mean for non-empty workers.
        non_empty.append(pred_sample.shape[pred.layout.grid] > 0)

    if not losses:
        msg = "Cannot apply a loss to an empty sparse source view."
        raise ValueError(msg)

    stacked = torch.stack(losses)
    return stacked.sum(dim=0) / max(sum(non_empty), 1)


def _apply_pairwise_gridded(
    pred: GriddedSource,
    target: GriddedSource,
    func: Callable[..., torch.Tensor],
    *,
    per_sample_kwargs: dict[str, Sequence[Any]] | None = None,
    **kwargs,
) -> torch.Tensor:
    if not isinstance(target, GriddedSource):
        msg = f"Other source must be a GriddedSource; got {type(target).__name__}."
        raise TypeError(msg)

    if per_sample_kwargs is not None:
        raise ValueError("Gridded losses take batched arguments; per_sample_kwargs is only for tabular sources.")

    if pred.layout != target.layout:
        msg = f"Both sources must have the same layout; got {pred.layout!r} and {target.layout!r}."
        raise ValueError(msg)

    assert torch.equal(pred.coordinates, target.coordinates), "Both views must have the same coordinates."
    return func(
        pred.data,
        target.data,
        layout=pred.layout,
        statistics=pred.statistics,
        name_to_index=pred.name_to_index,
        **kwargs,
    )


def apply_pairwise(pred: Source, target: Source, func: Callable, *args: Any, **kwargs) -> torch.Tensor:
    """Apply a function to two aligned source views.
    """
    if isinstance(pred, GriddedSource):
        return _apply_pairwise_gridded(pred, target, func, *args, **kwargs)

    if isinstance(pred, TabularSource):
        return _apply_pairwise_tabular(pred, target, func, *args, **kwargs)

    msg = f"Pairwise losses do not support source type {type(pred).__name__}."
    raise TypeError(msg)
