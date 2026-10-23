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

import numpy as np
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import Dataset

from anemoi.models.data import BaseSample
from anemoi.training.data.readers import DataReader
from anemoi.training.data.usable_indices import Anchors
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices
from anemoi.utils.dates import frequency_to_string

LOGGER = logging.getLogger(__name__)


class MultiDataset(Dataset):
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
        self.relative_date_indices_are_native = True
        self.frequency_seconds: int | None = None
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


class MixedFrequencyDataset(MultiDataset):
    """Map-style dataset aligned on the highest-frequency reader's anchors."""

    def __init__(
        self,
        data_readers: dict[str, DataReader],
        relative_date_indices: list[int],
        frequency: datetime.timedelta,
        tolerance: datetime.timedelta | None = None,
        fake_dataloading: bool = False,
    ) -> None:
        self.data_readers = data_readers
        self.dataset_names = list(data_readers)
        self.relative_date_indices_are_native = False
        self.frequency_seconds = int(frequency.total_seconds())
        self.tolerance = tolerance
        self.stride = self._sampling_stride(data_readers)
        self.fake_dataloading = fake_dataloading
        self._fake_sample: dict[str, BaseSample] | None = None
        self.model_relative_date_indices = np.array([], dtype=np.int64)
        self.model_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self.data_relative_date_indices_by_dataset: dict[str, np.ndarray] = {}
        self._sample_positions: dict[str, np.ndarray] = {}
        self._reference_positions: dict[str, np.ndarray] = {}
        self._reference_offsets: dict[str, np.ndarray] = {}
        self._sequences: dict[str, int] = {}
        if self.fake_dataloading:
            LOGGER.info("Using fake dataloading")

        trajectory = [name for name, reader in data_readers.items() if reader.valid_anchors([0]).base_dates is not None]
        if trajectory:
            msg = (
                "Mixed-frequency alignment currently supports only datasets with a global time axis. "
                f"Trajectory datasets are unsupported: {trajectory}."
            )
            raise ValueError(msg)

        self.set_relative_date_indices(relative_date_indices)

    def set_relative_date_indices(self, relative_date_indices: list[int]) -> None:
        """Resolve shared model-grid indices to each reader's available timestamps."""
        self.model_relative_date_indices = np.array(sorted({int(value) for value in relative_date_indices}))
        if len(self.model_relative_date_indices) == 0:
            msg = "`relative_date_indices` cannot be empty."
            raise ValueError(msg)

        self.model_relative_date_indices_by_dataset = self._build_model_relative_indices_by_dataset()
        self.data_relative_date_indices_by_dataset = {
            name: self._to_native_relative_indices(name, indices)
            for name, indices in self.model_relative_date_indices_by_dataset.items()
        }
        self.relative_date_indices = {
            name: normalize_time_indices(indices.tolist())
            for name, indices in self.data_relative_date_indices_by_dataset.items()
        }

        self._anchor_name = min(self.dataset_names, key=lambda name: self.data_readers[name].frequency)
        anchor_offsets = {self._anchor_name: self.data_relative_date_indices_by_dataset[self._anchor_name]}
        anchor_table = compute_valid_anchors(
            {self._anchor_name: self.data_readers[self._anchor_name]},
            anchor_offsets,
            stride=self.stride,
        )
        anchor_times = anchor_table.times.astype("datetime64[s]").astype(np.int64)
        valid = np.ones(len(anchor_times), dtype=bool)
        sample_positions: dict[str, np.ndarray] = {}
        reference_positions: dict[str, np.ndarray] = {}
        reference_offsets: dict[str, np.ndarray] = {}
        used_fallback = {}

        for name, reader in self.data_readers.items():
            if name == self._anchor_name:
                used_fallback[name] = False
                continue
            available = reader.valid_anchors([0])
            order = np.argsort(available.times)
            available_times = available.times[order].astype("datetime64[s]").astype(np.int64)
            available_positions = available.positions[order]
            available_sequences = available.sequences[order]
            if len(available_times) == 0:
                msg = f"Dataset '{name}' has no available dates."
                raise ValueError(msg)
            if np.any(available_sequences != available_sequences[0]):
                msg = f"Dataset '{name}' does not expose a single global time sequence."
                raise ValueError(msg)
            self._sequences[name] = int(available_sequences[0])

            requested = anchor_times[:, None] + (
                self.model_relative_date_indices_by_dataset[name][None, :] * self.frequency_seconds
            )
            indices = np.searchsorted(available_times, requested, side="right") - 1
            clipped = np.clip(indices, 0, len(available_times) - 1)
            age = requested - available_times[clipped]
            tolerance_delta = self.tolerance if self.tolerance is not None else reader.frequency
            tolerance = int(tolerance_delta.total_seconds())
            valid &= np.all((indices >= 0) & (age <= tolerance), axis=1)
            sample_positions[name] = available_positions[clipped]
            used_fallback[name] = bool(np.any(age > 0))

            reference_index = np.searchsorted(available_times, anchor_times, side="right") - 1
            reference_clipped = np.clip(reference_index, 0, len(available_times) - 1)
            reference_age = anchor_times - available_times[reference_clipped]
            valid &= (reference_index >= 0) & (reference_age <= tolerance)
            reference_positions[name] = available_positions[reference_clipped]
            reference_offsets[name] = available_times[reference_clipped] - anchor_times

        if not np.any(valid):
            msg = "No valid anchors found after mixed-frequency alignment across all datasets."
            raise ValueError(msg)

        self.anchors = Anchors(
            times=anchor_table.times[valid],
            base_dates=None,
            rows={self._anchor_name: anchor_table.rows[self._anchor_name][valid]},
        )
        self._sample_positions = {name: positions[valid] for name, positions in sample_positions.items()}
        self._reference_positions = {name: positions[valid] for name, positions in reference_positions.items()}
        self._reference_offsets = {name: offsets[valid] for name, offsets in reference_offsets.items()}
        self._fake_sample = None

        for name, fallback in used_fallback.items():
            if fallback:
                LOGGER.warning(
                    "Dataset '%s' uses the most recent available timestamp within %s for some model times.",
                    name,
                    frequency_to_string(
                        self.tolerance if self.tolerance is not None else self.data_readers[name].frequency,
                    ),
                )

    def _build_model_relative_indices_by_dataset(self) -> dict[str, np.ndarray]:
        result = {}
        for name, reader in self.data_readers.items():
            native_seconds = int(reader.frequency.total_seconds())
            relative_seconds = self.model_relative_date_indices * self.frequency_seconds
            indices = self.model_relative_date_indices[relative_seconds % native_seconds == 0]
            if len(indices) == 0:
                msg = (
                    f"Dataset '{name}' has no native timestamps among model-relative indices "
                    f"{self.model_relative_date_indices.tolist()}."
                )
                raise ValueError(msg)
            result[name] = indices
        return result

    def _to_native_relative_indices(self, name: str, indices: np.ndarray) -> np.ndarray:
        native_seconds = int(self.data_readers[name].frequency.total_seconds())
        return (indices * self.frequency_seconds // native_seconds).astype(np.int64, copy=False)

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return the highest native frequency across readers."""
        return min(self._collect("frequency").values())

    def get_sample(self, index: int) -> dict[str, BaseSample]:
        """Return every reader's windows measured from the shared anchor time."""
        result = {}
        for name, reader in self.data_readers.items():
            if name == self._anchor_name:
                sequence, reference = (int(value) for value in self.anchors.rows[name][index])
                positions = reference + self.data_relative_date_indices_by_dataset[name]
                offset = 0
            else:
                sequence = self._sequences[name]
                positions = self._sample_positions[name][index]
                reference = int(self._reference_positions[name][index])
                offset = int(self._reference_offsets[name][index])
            sample = reader.get_sample(sequence, positions)
            sample = reader.measure_from_reference(sample, reference, positions)
            if reader.data_type == "tabular":
                sample = sample.with_time_offsets([offset] * len(np.atleast_1d(positions)))
            result[name] = sample
        return result
