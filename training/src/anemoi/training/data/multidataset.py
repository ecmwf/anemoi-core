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

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.training.data.dataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple datasets."""

    def __init__(
        self,
        data_readers: dict,
        grid_indices: dict,
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "multi",
        **kwargs,
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
        self.shuffle = shuffle
        self.dataset_names = list(data_readers.keys())

        # Create individual NativeGridDataset for each dataset with its own grid_indices
        self.datasets = {}
        for name, data_reader in data_readers.items():
            if name not in grid_indices:
                msg = f"No grid_indices configuration found for dataset '{name}'"
                raise ValueError(msg)

            self.datasets[name] = NativeGridDataset(
                data_reader=data_reader,
                grid_indices=grid_indices[name],
                relative_date_indices=relative_date_indices,
                timestep=timestep,
                shuffle=self.shuffle,  # Will be overridden in __iter__
                label=f"{label}_{name}",
                **kwargs,
            )

        # Use the first dataset as the primary for shared properties
        self.primary_dataset = next(iter(self.datasets.values()))

        LOGGER.info(
            "MultiDataset initialized with %d datasets (%s), %d valid indices each",
            len(self.datasets),
            ", ".join(self.dataset_names),
            len(self.valid_date_indices),
        )

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all datasets."""
        combined_attr = {}
        for name, dataset in self.datasets.items():
            combined_attr[name] = getattr(dataset, attr_name)
        return combined_attr

    def _apply_to_all_datasets(self, method_name: str, *args, **kwargs) -> None:
        """Call a method by name with given arguments on all datasets."""
        for dataset in self.datasets.values():
            getattr(dataset, method_name)(*args, **kwargs)

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all datasets."""
        return self._collect("statistics")

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None]:
        """Return combined tendency statistics from all datasets."""
        return self._collect("statistics_tendencies")

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all datasets."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all datasets."""
        return self._collect("supporting_arrays")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all datasets."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all datasets."""
        return self._collect("resolution")

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices from all datasets."""
        valid_date_indices = self._collect("valid_date_indices")
        reference_indices = None
        for name, indices in valid_date_indices.items():
            if reference_indices is None:
                reference_indices = indices
            assert all(
                indices == reference_indices,
            ), f"Dataset '{name}' has different valid_date_indices than other datasets"
        return reference_indices

    @property
    def data(self) -> dict:
        """Return data from all datasets as dictionary."""
        return self._collect("data")

    def set_comm_group_info(self, *args, **kwargs) -> None:
        """Set communication group information for all datasets."""
        self._apply_to_all_datasets("set_comm_group_info", *args, **kwargs)

    def set_ens_comm_group_info(self, *args, **kwargs) -> None:
        """Set ensemble communication group information for all datasets."""
        self._apply_to_all_datasets("set_ens_comm_group_info", *args, **kwargs)

    def per_worker_init(self, *args, **kwargs) -> None:
        """Initialize all datasets for this worker."""
        self._apply_to_all_datasets("per_worker_init", *args, **kwargs)

    def _get_sample(self, index: int) -> dict[str, torch.Tensor]:
        return {name: dataset._get_sample(index) for name, dataset in self.datasets.items()}

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

        if self.shuffle:
            shuffled_chunk_indices = primary_dataset.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[primary_dataset.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[primary_dataset.chunk_index_range]

        LOGGER.debug(
            "%s worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            self.__class__.__name__,
            primary_dataset.worker_id,
            primary_dataset.worker_id,
            shuffled_chunk_indices[:10],
        )
        # TODO: improve this...
        for i in shuffled_chunk_indices:
            yield self._get_sample(i)

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for name, dataset in self.datasets.items():
            subtree = dataset.tree(prefix=name)
            tree.add(subtree)
        return tree
