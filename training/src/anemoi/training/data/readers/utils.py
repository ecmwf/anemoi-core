# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes


def _as_dict(value: str | dict | DictConfig) -> str | dict:
    """Convert DictConfig payloads, including nested ones, to plain containers."""
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def _normalize_dataset_config(dataset_config: str | dict | DictConfig) -> dict:
    """Normalize dataset payload to the open_dataset dictionary contract."""
    dataset_config = _as_dict(dataset_config)
    if isinstance(dataset_config, str):
        return {"dataset": dataset_config}

    if "dataset" not in dataset_config:
        msg = "dataset_config must contain the 'dataset' key."
        raise ValueError(msg)

    if dataset_config["dataset"] is None:
        msg = "dataset_config.dataset cannot be None."
        raise ValueError(msg)

    invalid_inner_keys = {"start", "end"} & set(dataset_config)
    if invalid_inner_keys:
        invalid = ", ".join(sorted(invalid_inner_keys))
        msg = f"dataset_config cannot contain [{invalid}]. Use outer keys 'start' and 'end' instead."
        raise ValueError(msg)

    # Keep only explicitly set options to avoid passing None-valued kwargs
    # (e.g. select=None), which can trigger downstream subset selection issues.
    return {key: value for key, value in dataset_config.items() if value is not None}


def _normalize_reader_config(dataset_config: dict | DictConfig) -> dict:
    """Validate and normalize reader configuration.

    Parameters
    ----------
    dataset_config : dict or DictConfig
        Dataset configuration dictionary.

    Returns
    -------
    dict
        Normalized dataset configuration dictionary with the following contract:
        {
            "dataset_config": {
                "dataset": str,
                "window": str,  # optional, for tabular datasets, e.g. "(-6h,0]"
                "frequency": str,  # optional, for tabular datasets
                ... other open_dataset kwargs ...
            },
            "start": datetime | int | None,  # optional
            "end": datetime | int | None,  # optional
            "trajectory": {  # optional, for 5-D trajectory datasets
                "sampling": {"stride": int | None},  # optional
            }
        }
    """
    normalized = _as_dict(dataset_config)

    if "dataset" in normalized:
        msg = (
            "Invalid dataloader dataset schema: use 'dataset_config' (outer key) "
            "and 'dataset' inside it. The legacy outer 'dataset' key is no longer supported."
        )
        raise ValueError(msg)

    if normalized.get("dataset_config", None) is None:
        msg = "Missing required 'dataset_config' in dataset reader configuration."
        raise ValueError(msg)

    return normalized


def _to_local_window_shard_data(
    data: torch.Tensor,
    latitudes: torch.Tensor,
    longitudes: torch.Tensor,
    timedeltas: torch.Tensor,
    boundaries: list[slice],
    *,
    reader_group_rank: int,
    reader_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[slice], list[ShardSizes] | None]:
    """Project sparse windowed tensors to the local reader-rank shard.

    Parameters
    ----------
    data : torch.Tensor
        Full sparse payload of shape ``(N, V)``.
    latitudes : torch.Tensor
        Full latitudes of shape ``(N,)``.
    longitudes : torch.Tensor
        Full longitudes of shape ``(N,)``.
    timedeltas : torch.Tensor
        Full per-point timedeltas of shape ``(N,)``.
    boundaries : list[slice]
        One boundary slice per logical time window over the flattened ``N`` axis.
    reader_group_rank : int
        Rank inside the reader group.
    reader_group_size : int
        Size of the reader group.

    Returns
    -------
    tuple
        ``(data_local, latitudes_local, longitudes_local, timedeltas_local, boundaries_local, window_shard_sizes)``
        where ``window_shard_sizes`` stores the per-window balanced partition sizes, or is ``None``
        when a single reader reads everything (the payload is then not sharded).
    """
    if reader_group_size <= 1:
        return data, latitudes, longitudes, timedeltas, boundaries, None

    # Each window is split across the reader group; this rank keeps its part of every window.
    local_slices: list[slice] = []
    boundaries_local: list[slice] = []
    window_shard_sizes: list[ShardSizes] = []
    offset = 0
    for boundary in boundaries:
        shard_sizes = get_balanced_partition_sizes(boundary.stop - boundary.start, reader_group_size)
        start, end = get_partition_range(shard_sizes, reader_group_rank)
        local_slices.append(slice(boundary.start + start, boundary.start + end))
        boundaries_local.append(slice(offset, offset + end - start))
        window_shard_sizes.append(shard_sizes)
        offset += end - start

    def take(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[s] for s in local_slices]) if local_slices else tensor[:0]

    return take(data), take(latitudes), take(longitudes), take(timedeltas), boundaries_local, window_shard_sizes
