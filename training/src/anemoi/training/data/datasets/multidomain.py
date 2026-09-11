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

from anemoi.training.data.batch_meta import META_KEY
from anemoi.training.data.batch_meta import PARTICIPANT_FIELD
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.datasets import AnemoiDataset
from anemoi.training.data.datasets.anemoidataset import normalize_participant_readers
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices

LOGGER = logging.getLogger(__name__)

# it may be that multidomain is not a good name for this, but it is what it is for now


class MultiDomainSampler:
    """Sample domain and index pairs for a multi-domain worker in domain-pure blocks.

    Each domain's worker slice is cut into consecutive blocks of ``batch_size``
    samples; blocks are then interleaved across domains. Because the DataLoader
    collates ``batch_size`` consecutive samples from ONE worker's iterator, every
    batch contains a single domain. Trailing samples that do not fill a block
    are dropped for the current epoch (they are reshuffled into blocks in later
    epochs when ``shuffle`` is on).
    """

    def __init__(
        self,
        valid_date_indices: Mapping[str, np.ndarray],
        chunk_index_range: Mapping[str, np.ndarray],
        rng: np.random.Generator,
        shuffle: bool = True,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        self.valid_date_indices = valid_date_indices
        self.chunk_index_range = chunk_index_range
        self.shuffle = shuffle
        self.rng = rng
        self.batch_size = batch_size

    def num_blocks(self, domain: str) -> int:
        """Return the number of full blocks this worker yields for ``domain``."""
        return len(self.chunk_index_range[domain]) // self.batch_size

    def num_dropped(self, domain: str) -> int:
        """Return the number of trailing samples of ``domain`` that do not fill a block."""
        return len(self.chunk_index_range[domain]) % self.batch_size

    def __len__(self) -> int:
        """Return the number of samples yielded to the worker (full blocks only)."""
        return sum(self.num_blocks(domain) for domain in self.chunk_index_range) * self.batch_size

    def _domain_blocks(self, domain: str) -> list[list[tuple[str, int]]]:
        """Return this worker's slice of ``domain`` cut into blocks of ``batch_size``."""
        indices = self.valid_date_indices[domain]
        if self.shuffle:
            indices = self.rng.choice(indices, size=len(indices), replace=False)
        indices = indices[self.chunk_index_range[domain]]
        n_full = self.num_blocks(domain) * self.batch_size
        return [
            [(domain, int(index)) for index in indices[start : start + self.batch_size]]
            for start in range(0, n_full, self.batch_size)
        ]

    def __iter__(self) -> Generator[tuple[str, int], None, None]:
        """Yield domain and index pairs block by block in worker sampling order."""
        blocks = [block for domain in self.valid_date_indices for block in self._domain_blocks(domain)]
        if self.shuffle:
            order = self.rng.permutation(len(blocks))
            blocks = [blocks[int(index)] for index in order]
        for block in blocks:
            yield from block


class MultiDomainDataset(AnemoiDataset):
    """Sample the participants of ONE dataset through one iterable dataset.

    Unlike :class:`MultiDataset`, which returns synchronized samples from every
    reader, each iteration yields one participant (domain) of the dataset.
    Participants retain independent grids and date ranges but share variables
    and frequency. Mixing single-sequence native-grid readers with
    multi-sequence trajectory readers is currently unsupported.

    Selection policy: anchors, worker shards and sample counts are kept PER
    participant (dictionaries keyed by participant name); the anchors of a
    participant are those of its participant row (intersection across datasets,
    here a single one). Every sample reads ONE reader. Relative date indices,
    shard sizes and the dataset-level properties (statistics, metadata, ...)
    are per dataset, i.e. keyed by :attr:`dataset_name`.
    """

    def __init__(
        self,
        data_readers: Mapping[str, Mapping[str, BaseAnemoiReader]],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multidomain",
        epoch: int = 0,
        rollout: int = 1,
        batch_size: int = 1,
        reference_participants: Mapping[str, str] | None = None,
        check_variables_compatibility: Mapping[str, object] | None = None,
    ) -> None:
        """A dataset that interchanges the participants of one dataset.

        Parameters
        ----------
        data_readers : Mapping[str, Mapping[str, BaseAnemoiReader]]
            ``{dataset_name: {participant: reader}}`` with exactly one dataset name.
        relative_date_indices : dict[str, TimeIndices]
            Relative date indices keyed by dataset name (shared by all participants).
        shuffle : bool, optional
            Whether to shuffle samples, by default True.
        label : str, optional
            Dataset label, by default ``"multidomain"``.
        epoch : int, optional
            Epoch used for deterministic shuffling, by default 0.
        rollout : int, optional
            Rollout length represented by the relative date indices, by default 1.
        batch_size : int, optional
            Per-GPU batch size the DataLoader collates, by default 1. Samples are
            yielded in participant-pure blocks of this size so every batch holds one participant.
        reference_participants : Mapping[str, str], optional
            ``{dataset_name: participant}`` selecting the participant whose statistics,
            metadata and variable indices represent the dataset (config: ``statistics_from``).
            Defaults to the first participant.
        check_variables_compatibility : Mapping[str, object], optional
            Options forwarded to ``Variable.check_compatibility``. The options
            follow ``CheckVariablesCompatibilitySchema``.
        """
        nested = normalize_participant_readers(data_readers)
        if len(nested) != 1:
            msg = (
                "MultiDomainDataset supports exactly one dataset with several participants, got datasets "
                f"{list(nested)}. Declare the domains as 'participants:' of a single dataset."
            )
            raise ValueError(msg)
        ((self.dataset_name, participants),) = nested.items()
        self.participants = list(participants)
        super().__init__(
            data_readers=nested,
            shuffle=shuffle,
            label=label,
            epoch=epoch,
            rollout=rollout,
            batch_size=batch_size,
            reference_participants=reference_participants,
        )
        self._check_no_mixed_sequence_types()
        self._set_relative_date_indices(relative_date_indices)
        self._check_datasets_units(**dict(check_variables_compatibility or {}))
        LOGGER.info("valid date indices: %s", self.valid_date_indices)
        self.n_samples_per_worker = {}  # overwrite base to empty dict
        self.chunk_index_range = {}  # overwrite base to empty dict

    def _compute_anchors(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        # Independent anchors per participant, each with its own flat index. A
        # participant's anchors are the valid anchors of its row of readers
        # (intersected across datasets; a single dataset here).
        self.anchors = {}
        for participant in self.participants:
            try:
                self.anchors[participant] = compute_valid_anchors(
                    self.participant_row(participant),
                    relative_date_indices,
                )
            except ValueError as e:
                msg = f"Participant '{participant}': {e}"
                raise ValueError(msg) from e
        self.valid_date_indices = {
            name: np.arange(len(anchors), dtype=np.int64) for name, anchors in self.anchors.items()
        }

    def _check_datasets_units(self, **options: object) -> None:
        """Check that all participants have the same units.

        Raises
        ------
            ValueError: If the participants have different units.
        """
        from anemoi.transform.variables import Variable

        metadata = self._collect_participants("metadata")[self.dataset_name]
        domains_with_units = [domain for domain, meta in metadata.items() if meta.get("variables_metadata", {})]

        if len(domains_with_units) == 0:
            LOGGER.warning("All participants have empty metadata, skipping units check.")
            return
        if len(domains_with_units) == 1:
            LOGGER.warning("Only one participant has variable metadata, skipping units check.")
            return

        # need to cross check all participants, as some may have missing metadata for some variables
        for i, domain1 in enumerate(domains_with_units):
            for domain2 in domains_with_units[i + 1 :]:

                variable_domain1 = {
                    name: Variable.from_dict(name, data)
                    for name, data in metadata[domain1]["variables_metadata"].items()
                }
                variable_domain2 = {
                    name: Variable.from_dict(name, data)
                    for name, data in metadata[domain2]["variables_metadata"].items()
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

        for participant in self.participants:
            self.n_samples_per_worker[participant], self.chunk_index_range[participant] = self._shard_indices(
                len(self.valid_date_indices[participant]),
                n_workers,
                worker_id,
            )

        self._seed_worker()

    def get_sample(self, domain_name: str, index: int) -> dict[str, torch.Tensor]:
        """Get a sample from the specified participant (domain) and index.

        Args:
            domain_name (str): The participant to sample from.
            index (int): The index of the sample to retrieve.

        Returns
        -------
            dict[str, torch.Tensor | dict]: ``{dataset_name: tensor, META_KEY: {"participant": domain_name}}``.
            The tensor is keyed by the DATASET name (as in ``MultiDataset``); the
            participant travels as batch metadata (see ``anemoi.training.data.batch_meta``).
        """
        sequence, position = (int(value) for value in self.anchors[domain_name][index])
        return {
            self.dataset_name: self._read(self.dataset_name, sequence, position, participant=domain_name),
            META_KEY: {PARTICIPANT_FIELD: domain_name},
        }

    def __iter__(self) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield samples from independently partitioned domains in domain-pure blocks.

        Each domain is shuffled before its worker slice is selected and cut into
        blocks of ``batch_size``. The blocks of all domains are then shuffled
        together, giving sampling proportional to each domain's available
        samples while keeping every DataLoader batch within a single domain.
        All sample communication groups use the same seed and therefore process
        domains in the same order, avoiding mismatched collective operations.
        ``MultiDomainSampler`` owns this ordering because PyTorch does not
        support a DataLoader sampler for ``IterableDataset``.

        Returns
        -------
        Generator[dict[str, torch.Tensor], None, None]
            A generator yielding ``{dataset_name: tensor, META_KEY: {"participant": name}}`` samples
        """
        sampler = MultiDomainSampler(
            self.valid_date_indices,
            self.chunk_index_range,
            self.rng,
            self.shuffle,
            batch_size=self.batch_size,
        )
        for domain_name in self.participants:
            if sampler.num_dropped(domain_name):
                LOGGER.info(
                    "Worker %d (%s): dropping %d trailing sample(s) of domain '%s' that do not fill a "
                    "batch of %d (%d full batches kept).",
                    self.worker_id,
                    self.label,
                    sampler.num_dropped(domain_name),
                    domain_name,
                    self.batch_size,
                    sampler.num_blocks(domain_name),
                )
        labeled_samples = list(sampler)

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
