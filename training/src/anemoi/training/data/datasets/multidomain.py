# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from collections.abc import Generator
from collections.abc import Mapping

import numpy as np
import torch

from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.datasets import AnemoiDataset
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)

# it may be that multidomain is not a good name for this, but it is what it is for now


class MultiDomainSampler:
    """Sample domain and index pairs for a multi-domain worker."""

    def __init__(
        self,
        valid_date_indices: Mapping[str, np.ndarray],
        chunk_index_range: Mapping[str, np.ndarray],
        rng: np.random.Generator,
        shuffle: bool = True,
    ) -> None:
        self.valid_date_indices = valid_date_indices
        self.chunk_index_range = chunk_index_range
        self.shuffle = shuffle
        self.rng = rng

    def __len__(self) -> int:
        """Return the number of samples assigned to the worker."""
        return sum(len(indices) for indices in self.chunk_index_range.values())

    def __iter__(self) -> Generator[tuple[str, int], None, None]:
        """Yield domain and index pairs in worker sampling order."""
        domain_indices = {
            domain: (
                self.rng.choice(indices, size=len(indices), replace=False)[self.chunk_index_range[domain]]
                if self.shuffle
                else indices[self.chunk_index_range[domain]]
            )
            for domain, indices in self.valid_date_indices.items()
        }
        samples = [(domain, int(index)) for domain, indices in domain_indices.items() for index in indices]
        if self.shuffle:
            order = self.rng.choice(len(samples), size=len(samples), replace=False)
            samples = [samples[int(index)] for index in order]
        yield from samples


class MultiDomainDataset(AnemoiDataset):
    """Sample independent domains through one iterable dataset.

    Unlike :class:`MultiDataset`, which returns synchronized samples from every
    reader, each iteration yields one domain. Readers retain independent grids
    and date ranges. Mixing single-sequence native-grid readers with
    multi-sequence trajectory readers is currently unsupported.

    Selection policy: anchors, worker shards and sample counts are kept PER
    domain (dictionaries keyed by domain name); every sample reads ONE reader.
    """

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multidomain",
        epoch: int = 0,
        rollout: int = 1,
        check_variables_compatibility: Mapping[str, object] | None = None,
    ) -> None:
        """A dataset that combines multiple data_readers together.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Domain names mapped to their data readers.
        relative_date_indices : dict[str, TimeIndices]
            Domain names mapped to their relative date indices.
        shuffle : bool, optional
            Whether to shuffle samples, by default True.
        label : str, optional
            Dataset label, by default ``"multidomain"``.
        epoch : int, optional
            Epoch used for deterministic shuffling, by default 0.
        rollout : int, optional
            Rollout length represented by the relative date indices, by default 1.
        check_variables_compatibility : Mapping[str, object], optional
            Options forwarded to ``Variable.check_compatibility``. The options
            follow ``CheckVariablesCompatibilitySchema``.
        """
        super().__init__(
            data_readers=data_readers,
            shuffle=shuffle,
            label=label,
            epoch=epoch,
            rollout=rollout,
        )
        self._check_no_mixed_sequence_types()
        self._set_relative_date_indices(relative_date_indices)
        self._check_datasets_units(**dict(check_variables_compatibility or {}))
        LOGGER.info("valid date indices: %s", self.valid_date_indices)
        self.n_samples_per_worker = {}  # overwrite base to empty dict
        self.chunk_index_range = {}  # overwrite base to empty dict

    def _compute_anchors(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        # Independent anchors per domain, each with its own flat index.
        self.anchors = {
            name: data_reader.compute_anchors(relative_date_indices[name])
            for name, data_reader in self.data_readers.items()
        }
        self.valid_date_indices = {
            name: np.arange(len(anchors), dtype=np.int64) for name, anchors in self.anchors.items()
        }

    def _check_datasets_units(self, **options: object) -> None:
        """Check that all datasets have the same units.

        Raises
        ------
            ValueError: If the datasets have different units.
        """
        from anemoi.transform.variables import Variable

        domains_with_units = [
            domain for domain, metadata in self.metadata.items() if metadata.get("variables_metadata", {})
        ]

        if len(domains_with_units) == 0:
            LOGGER.warning("All datasets have empty metadata, skipping units check.")
            return
        if len(domains_with_units) == 1:
            LOGGER.warning("Only one dataset has variable metadata, skipping units check.")
            return

        # need to cross check all datasets, as some may have missing metadata for some variables
        for i, domain1 in enumerate(domains_with_units):
            for domain2 in domains_with_units[i + 1 :]:

                variable_domain1 = {
                    name: Variable.from_dict(name, data)
                    for name, data in self.metadata[domain1]["variables_metadata"].items()
                }
                variable_domain2 = {
                    name: Variable.from_dict(name, data)
                    for name, data in self.metadata[domain2]["variables_metadata"].items()
                }

                try:
                    Variable.check_compatibility(variable_domain1, variable_domain2, **options)
                except ValueError as e:
                    msg = f"Variable compatibility check failed for domain1 '{domain1}' and domain2 '{domain2}': {e}"
                    raise ValueError(msg) from e

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize a specific worker.

        Args:
            n_workers : int
                The total number of workers.
            worker_id : int
                The ID of the current worker (0-indexed).

        Returns
        -------
            None
        """
        self.worker_id = worker_id

        for dataset in self.dataset_names:
            self.n_samples_per_worker[dataset], self.chunk_index_range[dataset] = self._shard_indices(
                len(self.valid_date_indices[dataset]),
                n_workers,
                worker_id,
            )

        self._seed_worker()

    def get_sample(self, domain_name: str, index: int) -> dict[str, torch.Tensor]:
        """Get a sample from the specified domain and index.

        Args:
            domain_name (str): The name of the domain to sample from.
            index (int): The index of the sample to retrieve.

        Returns
        -------
            dict[str, torch.Tensor]: The sample retrieved from the specified domain and index.
        """
        sequence, position = (int(value) for value in self.anchors[domain_name][index])
        return {domain_name: self._read(domain_name, sequence, position)}

    def __iter__(self) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield samples from independently partitioned domains.

        Each domain is shuffled before its worker slice is selected. The slices
        are then combined and shuffled again, giving sampling proportional to
        each domain's available samples. All sample communication groups use the
        same seed and therefore process domains in the same order, avoiding
        mismatched collective operations. ``MultiDomainSampler`` owns this
        ordering because PyTorch does not support a DataLoader sampler for
        ``IterableDataset``.

        Returns
        -------
        Generator[dict[str, torch.Tensor], None, None]
            A generator yielding dictionaries containing tensor samples and their corresponding domain names
        """
        labeled_samples = list(
            MultiDomainSampler(
                self.valid_date_indices,
                self.chunk_index_range,
                self.rng,
                self.shuffle,
            ),
        )

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            labeled_samples[:10],
        )

        for domain_name, index in labeled_samples:
            yield self.get_sample(domain_name, index)
