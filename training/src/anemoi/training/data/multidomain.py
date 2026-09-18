# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.sampler import CrossDatasetSampler
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices


class MultiDomainDataset(MultiDataset):
    """Sample independent domains through one iterable dataset.

    Unlike :class:`MultiDataset`, which returns synchronized samples from every
    reader, each iteration yields one domain. Readers retain independent grids
    and date ranges. Mixing single-sequence native-grid readers with
    multi-sequence trajectory readers is currently unsupported.
    """

    check_dataset_units = True
    default_label = "multidomain"
    sampler_class = CrossDatasetSampler

    def _set_date_indices(self, relative_date_indices: dict[str, TimeIndices]) -> None:
        """Set independent anchors and relative date indices for each domain."""
        self.anchors = {
            name: data_reader.compute_anchors(relative_date_indices[name])
            for name, data_reader in self.data_readers.items()
        }
        for name, anchors in self.anchors.items():
            if len(anchors) == 0:
                msg = f"No valid anchors found for data reader '{name}': {self.data_readers[name]}"
                raise ValueError(msg)
        self.valid_date_indices = {
            name: np.arange(len(anchors), dtype=np.int64) for name, anchors in self.anchors.items()
        }
        # Normalize the date indices to use slices where possible, which can improve downstream indexing performance.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }
