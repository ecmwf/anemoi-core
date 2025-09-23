# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import torch
from torch.utils.data import IterableDataset
from einops import rearrange

from anemoi.training.data.dataset.singledataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple datasets."""

    def __init__(
        self,
        datasets_config: dict,
        grid_indices_config: dict,
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "multi",
    ) -> None:
        """Initialize multi-dataset with synchronized datasets.

        Parameters
        ----------
        datasets_config : dict
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        grid_indices_config : dict
            Dictionary mapping dataset names to their grid_indices
            Format: {"dataset_a": grid_indices_a, "dataset_b": grid_indices_b, ...}
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample
        timestep : str, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        """
        self.label = label
        self.dataset_names = list(datasets_config.keys())

        # Create individual NativeGridDataset for each dataset with its own grid_indices
        self.datasets = {}
        for name, data_reader in datasets_config.items():
            if name not in grid_indices_config:
                msg = f"No grid_indices configuration found for dataset '{name}'"
                raise ValueError(msg)

            self.datasets[name] = NativeGridDataset(
                data_reader=data_reader,
                grid_indices=grid_indices_config[name],
                relative_date_indices=relative_date_indices,
                timestep=timestep,
                shuffle=shuffle,  # Will be overridden in __iter__
                label=f"{label}_{name}",
            )

        # Use the first dataset as the primary for shared properties
        self.primary_dataset = next(iter(self.datasets.values()))

        # Verify all datasets have the same number of valid indices
        primary_count = len(self.primary_dataset.valid_date_indices)
        for name, dataset in self.datasets.items():
            dataset_count = len(dataset.valid_date_indices)
            if dataset_count != primary_count:
                msg = (
                    f"Dataset '{name}' has {dataset_count} valid indices, "
                    f"but expected {primary_count} to match other datasets"
                )
                raise ValueError(msg)

        LOGGER.info(
            "MultiDataset initialized with %d datasets (%s), %d valid indices each",
            len(self.datasets),
            ", ".join(self.dataset_names),
            primary_count,
        )

    @cached_property
    def statistics(self) -> dict:
        """Return combined statistics from all datasets."""
        combined_stats = {}
        for name, dataset in self.datasets.items():
            combined_stats[name] = dataset.statistics
        return combined_stats

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return combined tendency statistics from all datasets."""
        combined_stats = {}
        for name, dataset in self.datasets.items():
            stats = dataset.statistics_tendencies
            if stats is not None:
                combined_stats[name] = stats
        return combined_stats if combined_stats else None

    @cached_property
    def metadata(self) -> dict:
        """Return combined metadata from all datasets."""
        combined_metadata = {}
        for name, dataset in self.datasets.items():
            combined_metadata[name] = dataset.metadata
        return combined_metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return combined supporting arrays from all datasets."""
        combined_arrays = {}
        for name, dataset in self.datasets.items():
            combined_arrays[name] = dataset.supporting_arrays
        return combined_arrays

    @cached_property
    def name_to_index(self) -> dict:
        """Return combined name_to_index mapping from all datasets."""
        combined_mapping = {}
        for name, dataset in self.datasets.items():
            combined_mapping[name] = dataset.name_to_index
        return combined_mapping

    @cached_property
    def resolution(self) -> dict:
        """Return combined resolution from all datasets."""
        combined_resolution = {}
        for name, dataset in self.datasets.items():
            combined_resolution[name] = dataset.resolution
        return combined_resolution

    @property
    def data(self) -> dict:
        """Return data from all datasets as dictionary."""
        return {name: dataset.data for name, dataset in self.datasets.items()}

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set communication group information for all datasets."""
        for dataset in self.datasets.values():
            dataset.set_comm_group_info(
                global_rank,
                model_comm_group_id,
                model_comm_group_rank,
                model_comm_num_groups,
                reader_group_rank,
                reader_group_size,
            )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize all datasets for this worker."""
        for dataset in self.datasets.values():
            dataset.per_worker_init(n_workers, worker_id)

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator that yields dictionaries of synchronized samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        # Get the shuffled indices from the primary dataset
        # All datasets will use the same shuffled indices for synchronization
        primary_dataset = self.primary_dataset

        if primary_dataset.shuffle:
            shuffled_chunk_indices = primary_dataset.rng.choice(
                primary_dataset.valid_date_indices,
                size=len(primary_dataset.valid_date_indices),
                replace=False,
            )[primary_dataset.chunk_index_range]
        else:
            shuffled_chunk_indices = primary_dataset.valid_date_indices[primary_dataset.chunk_index_range]

        LOGGER.debug(
            "MultiDataset worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            primary_dataset.worker_id,
            primary_dataset.worker_id,
            shuffled_chunk_indices[:10],
        )
        # todo: improve this...
        dataset_iterators = {}
        for name, dataset in self.datasets.items():
            dataset_iterators[name] = self._build_dataset_iterator(dataset, shuffled_chunk_indices)

        for _ in shuffled_chunk_indices:
            sample_dict = {}
            for name in self.dataset_names:
                sample_dict[name] = next(dataset_iterators[name])
            yield sample_dict

    def _build_dataset_iterator(self, dataset: NativeGridDataset, indices):  # type: ignore[no-untyped-def]
        """Create an iterator for a dataset using the provided indices."""
        for i in indices:
            start = i + dataset.relative_date_indices[0]
            end = i + dataset.relative_date_indices[-1] + 1
            timeincrement = dataset.relative_date_indices[1] - dataset.relative_date_indices[0]

            grid_shard_indices = dataset.grid_indices.get_shard_indices(dataset.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                x = dataset.data[start:end:timeincrement, :, :, grid_shard_indices]
            else:
                x = dataset.data[start:end:timeincrement, :, :, :]
                x = x[..., grid_shard_indices]

            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        dataset_info = "\n".join([f"  {name}: {dataset.data}" for name, dataset in self.datasets.items()])
        return f"""
            {super().__repr__()}
            Datasets:
{dataset_info}
            Relative dates: {self.primary_dataset.relative_date_indices}
        """
