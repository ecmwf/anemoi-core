# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.transport.random_fields import randn_like_with_grid_sharding

Data: TypeAlias = torch.Tensor | list[torch.Tensor]


def is_sparse_data(data: Data) -> bool:
    """Return whether data is represented as per-sample sparse tensors."""
    return isinstance(data, list)


def data_device(data: Data) -> torch.device:
    """Return the device of dense or sparse data."""
    return data[0].device if isinstance(data, list) else data.device


def data_dtype(data: Data) -> torch.dtype:
    """Return the dtype of dense or sparse data."""
    return data[0].dtype if isinstance(data, list) else data.dtype


def condition_shape(data: Data) -> tuple[int, int, int, int, int]:
    """Return a 5D condition-compatible shape for transport condition sampling."""
    if isinstance(data, list):
        if not data:
            msg = "Cannot infer condition shape from an empty sparse data list."
            raise ValueError(msg)
        if data[0].ndim <= 2:
            return len(data), 1, 1, 1, 1
        return len(data), 1, data[0].shape[0], 1, 1

    if data.ndim != 5:
        msg = f"Expected dense transport data to be 5D, got shape {tuple(data.shape)}."
        raise ValueError(msg)
    return tuple(data.shape)


def condition_shapes(data: dict[str, Data]) -> dict[str, tuple[int, int, int, int, int]]:
    """Return per-dataset condition-compatible shapes."""
    return {name: condition_shape(dataset_data) for name, dataset_data in data.items()}


def first_data_device(data: dict[str, Data]) -> torch.device:
    """Return the device of the first dataset data entry."""
    if not data:
        msg = "Cannot infer device from an empty data dictionary."
        raise ValueError(msg)
    return data_device(next(iter(data.values())))


def map_data(data: Data, fn: Callable[[torch.Tensor], torch.Tensor]) -> Data:
    """Apply ``fn`` to dense data or each sparse sample."""
    if isinstance(data, list):
        return [fn(sample) for sample in data]
    return fn(data)


def zip_map_data(left: Data, right: Data, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Data:
    """Apply ``fn`` pairwise to dense data or sparse sample lists."""
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            msg = "Cannot combine dense and sparse transport data."
            raise TypeError(msg)
        if len(left) != len(right):
            msg = f"Sparse transport data lists must have the same length, got {len(left)} and {len(right)}."
            raise ValueError(msg)
        return [fn(left_sample, right_sample) for left_sample, right_sample in zip(left, right, strict=True)]
    return fn(left, right)


def zeros_like_data(data: Data) -> Data:
    """Create zero data with the same dense/sparse structure."""
    return map_data(data, torch.zeros_like)


def randn_like_data(
    data: Data,
    *,
    model_comm_group=None,
    grid_shard_sizes: ShardSizes = None,
) -> Data:
    """Create Gaussian data with the same dense/sparse structure."""
    if isinstance(data, list):
        if grid_shard_sizes is not None:
            msg = "Grid sharding is not supported for sparse transport data."
            raise NotImplementedError(msg)
        return [torch.randn_like(sample) for sample in data]

    return randn_like_with_grid_sharding(
        data,
        model_comm_group=model_comm_group,
        grid_shard_sizes=grid_shard_sizes,
    )


def scale_data(data: Data, scale: float | torch.Tensor) -> Data:
    """Multiply dense data or each sparse sample by ``scale``."""
    return map_data(data, lambda sample: sample * scale)


def add_data(left: Data, right: Data) -> Data:
    """Add dense data or sparse sample lists."""
    return zip_map_data(left, right, lambda left_sample, right_sample: left_sample + right_sample)


def _sample_scalar_for_data(data_sample: torch.Tensor, scalar: torch.Tensor, sample_index: int) -> torch.Tensor:
    """Select the scalar condition for one dense-batch or sparse sample."""
    if scalar.ndim != 5:
        msg = f"Expected transport condition to be 5D, got shape {tuple(scalar.shape)}."
        raise ValueError(msg)

    sample_scalar = scalar[sample_index, 0, :, 0, 0]
    if data_sample.ndim <= 2:
        if sample_scalar.numel() != 1:
            msg = "Sparse observation data without an ensemble axis requires ensemble_size == 1."
            raise NotImplementedError(msg)
        return sample_scalar.reshape(())

    view_shape = [sample_scalar.shape[0]] + [1] * (data_sample.ndim - 1)
    return sample_scalar.reshape(view_shape)


def apply_batch_scalar_data(
    data: Data,
    scalar: torch.Tensor,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Data:
    """Apply a per-batch/per-ensemble scalar condition to dense or sparse data."""
    if isinstance(data, list):
        if scalar.shape[0] != len(data):
            msg = f"Condition batch size {scalar.shape[0]} does not match sparse data length {len(data)}."
            raise ValueError(msg)
        return [
            fn(sample, _sample_scalar_for_data(sample, scalar, sample_index))
            for sample_index, sample in enumerate(data)
        ]
    return fn(data, scalar)


def zip_map_batch_scalar_data(
    data: Data,
    *others: Data,
    scalar: torch.Tensor,
    fn: Callable[..., torch.Tensor],
) -> Data:
    """Apply ``fn`` to one or more dense/sparse data fields plus a batch scalar."""
    entries = (data, *others)
    if any(isinstance(entry, list) for entry in entries):
        if not all(isinstance(entry, list) for entry in entries):
            msg = "Cannot combine dense and sparse transport data."
            raise TypeError(msg)

        reference = data
        assert isinstance(reference, list)
        if scalar.shape[0] != len(reference):
            msg = f"Condition batch size {scalar.shape[0]} does not match sparse data length {len(reference)}."
            raise ValueError(msg)

        outputs = []
        for sample_index, reference_sample in enumerate(reference):
            samples = []
            for entry in entries:
                assert isinstance(entry, list)
                if len(entry) != len(reference):
                    msg = (
                        f"Sparse transport data lists must have the same length, got "
                        f"{len(reference)} and {len(entry)}."
                    )
                    raise ValueError(msg)
                samples.append(entry[sample_index])
            sample_scalar = _sample_scalar_for_data(reference_sample, scalar, sample_index)
            outputs.append(fn(*samples, sample_scalar))
        return outputs

    return fn(*entries, scalar)


def multiply_batch_scalar_data(data: Data, scalar: torch.Tensor) -> Data:
    """Multiply dense or sparse data by a per-batch/per-ensemble scalar."""
    return apply_batch_scalar_data(data, scalar, lambda sample, sample_scalar: sample * sample_scalar)


def broadcast_batch_scalar_data(data: Data, scalar: torch.Tensor) -> Data:
    """Broadcast a per-batch/per-ensemble scalar to dense or sparse data structure."""
    if isinstance(data, list):
        if scalar.shape[0] != len(data):
            msg = f"Condition batch size {scalar.shape[0]} does not match sparse data length {len(data)}."
            raise ValueError(msg)
        return [_sample_scalar_for_data(sample, scalar, sample_index) for sample_index, sample in enumerate(data)]
    return scalar


def add_scaled_data(target_data: Data, source_data: Data, scalar: torch.Tensor) -> Data:
    """Return ``target_data + source_data * scalar`` for dense or sparse data."""
    scaled_source = multiply_batch_scalar_data(source_data, scalar)
    return add_data(target_data, scaled_source)
