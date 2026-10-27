# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from functools import cached_property

from rich.console import Console
from rich.tree import Tree
from torch.utils.data import Dataset

from anemoi.models.data import BaseSample
from anemoi.training.data.readers import DataReader
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices

LOGGER = logging.getLogger(__name__)


class DataReaderCollection:
    """Dataset-level metadata collected from ``self.data_readers``, shared by multi-reader datasets."""

    data_readers: dict[str, DataReader]

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all data readers."""
        return self._collect("statistics")

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all data readers."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all data readers."""
        return self._collect("supporting_arrays")

    @cached_property
    def variables(self) -> dict[str, list[str]]:
        """Return combined variables from all data readers."""
        return self._collect("variables")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all data readers."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all data readers."""
        return self._collect("resolution")

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return combined frequency from all data readers."""
        freqs = self._collect("frequency")
        freq_ref = None
        for name, freq in freqs.items():
            if freq_ref is None:
                freq_ref = freq
            assert freq == freq_ref, f"Data reader '{name}' has different frequency than other data readers"
        return freq_ref

    def set_reader_group_info(self, reader_group_rank: int, reader_group_size: int) -> None:
        """Set the reader group information, called by the DDP strategies.

        Each rank in a reader group reads its own part of the data.

        Parameters
        ----------
        reader_group_rank : int
            Reader group rank.
        reader_group_size : int
            Reader group size.
        """
        for reader in self.data_readers.values():
            reader.set_reader_group_info(reader_group_rank, reader_group_size)


class MultiDataset(DataReaderCollection, Dataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers.

    Each index refers to one valid anchor shared by all data readers. The
    dataloader's sampler decides which indices a rank loads and in which order.
    """

    def __init__(
        self,
        data_readers: dict[str, DataReader],
        relative_date_indices: dict[str, TimeIndices],
        fake_dataloading: bool = False,
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, DataReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader
        fake_dataloading : bool, optional
            Load one real sample and reuse it for subsequent accesses, by default False
        """
        self.data_readers = data_readers
        self.dataset_names = list(data_readers.keys())
        self.stride = self._sampling_stride(data_readers)

        self.fake_dataloading = fake_dataloading
        self._fake_sample: dict[str, BaseSample] | None = None
        if self.fake_dataloading:
            LOGGER.info("Using fake dataloading")

        self.set_relative_date_indices(relative_date_indices)

    @staticmethod
    def _sampling_stride(data_readers: dict[str, DataReader]) -> int | None:
        """Return the anchor stride the readers agree on; stride 1 when none is configured."""
        configured = {name: reader.sampling.get("stride") for name, reader in data_readers.items() if reader.sampling}
        strides = set(configured.values())
        if len(strides) > 1:
            msg = f"Data readers disagree on the sampling stride: {configured}."
            raise ValueError(msg)
        return strides.pop() if strides else 1

    def set_relative_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Set the time steps loaded for each sample, and the samples that provide them.

        Call this before the dataloader is created: the number of samples changes
        with the time steps, and the sampler reads it when it is constructed.
        """
        # Find the anchors (times every reader can sample) for the required time steps.
        # Each reader gets its own (sequence, position) per anchor, so readers need not share
        # a frequency or date range. Each dataset index refers to one anchor.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices, stride=self.stride)

        # Normalize the date indices to use slices where possible, which can improve downstream indexing performance.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.anchors)

    def get_sample(self, index: int) -> dict[str, BaseSample]:
        """Return per-dataset samples for ``index``.

        Each value is the reader's :class:`~anemoi.models.data.BaseSample`, so that
        the dataloader's collate function can build a :class:`anemoi.models.data.Batch`.
        Observation timedeltas are measured from the sample's reference time.
        """
        x: dict[str, BaseSample] = {}
        for name, data_reader in self.data_readers.items():
            sequence, position = (int(v) for v in self.anchors.rows[name][index])
            time_steps = offset_time_indices(position, self.relative_date_indices[name])
            sample = data_reader.get_sample(sequence, time_steps)
            x[name] = data_reader.measure_from_reference(sample, position, time_steps)

        return x

    def __getitem__(self, index: int) -> dict[str, BaseSample]:
        """Return ``{dataset_name: BaseSample}`` for one anchor.

        Parameters
        ----------
        index : int
            Index of the anchor, in the range [0, len(self)).

        Returns
        -------
        dict[str, BaseSample]
            One sample for each dataset.
        """
        if not self.fake_dataloading:
            return self.get_sample(index)
        # Each worker process reads one real sample and then returns it for every index.
        if self._fake_sample is None:
            self._fake_sample = self.get_sample(index)
        return self._fake_sample

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for name, reader in self.data_readers.items():
            subtree = reader.tree(prefix=name)
            tree.add(subtree)
        return tree
