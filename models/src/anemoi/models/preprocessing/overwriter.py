# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from typing import Optional

from torch import Tensor

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class ZeroOverwriter(BasePreprocessor):
    """Overwrite selected timesteps of specified input variables with zero.

    Config
    -------
    groups: list[dict]
        - vars: list[str]
            Variable names to overwrite with zero.
        - time_indices: list[int]
            Timesteps (relative to the input window) to set to zero, e.g., [0].

    Notes
    -----
    - Operates on inputs; indices are built for both training and inference inputs.
    - See `training/config/data/zarr_interpolator.yaml` for an example configuration.
    """

    def __init__(self, config=None, data_indices: Optional[IndexCollection] = None, statistics: Optional[dict] = None):
        super().__init__(config, data_indices, statistics)
        groups = [] if config is None else config.groups

        self.num_training_input_vars = len(self.data_indices.data.input.name_to_index)
        self.num_inference_input_vars = len(self.data_indices.model.input.name_to_index)
        self._train_time_to_idxs: dict[int, list[int]] = {}
        self._infer_time_to_idxs: dict[int, list[int]] = {}

        train_map = self.data_indices.data.input.name_to_index
        infer_map = self.data_indices.model.input.name_to_index

        # Accept config.groups as:
        # - list[{"vars": [...], "time_indices": [int]}]
        for entry in groups:
            if not isinstance(entry, MutableMapping):
                LOGGER.warning("ZeroOverwriter group must be a dict, got %r; skipping.", type(entry).__name__)
                continue

            vars_list = entry["vars"]
            time_indices = entry["time_indices"]

            if not isinstance(vars_list, MutableSequence) or not all(isinstance(v, str) for v in vars_list):
                LOGGER.warning("ZeroOverwriter group 'vars' must be a list[str], got %r; skipping.", vars_list)
                continue

            if not isinstance(time_indices, MutableSequence) or not all(isinstance(t, int) for t in time_indices):
                LOGGER.warning(
                    "ZeroOverwriter group 'time_indices' must be a list[int], got %r; skipping.", time_indices
                )
                continue

            train_idxs = [train_map[v] for v in vars_list if v in train_map]
            infer_idxs = [infer_map[v] for v in vars_list if v in infer_map]

            missing_train = [v for v in vars_list if v not in train_map]
            missing_infer = [v for v in vars_list if v not in infer_map]
            if missing_train or missing_infer:
                LOGGER.debug("ZeroOverwriter: variables not found (train=%s, infer=%s)", missing_train, missing_infer)

            self._train_time_to_idxs.setdefault(time_indices, [])
            self._infer_time_to_idxs.setdefault(time_indices, [])

            for idx in train_idxs:
                if idx not in self._train_time_to_idxs[time_indices]:
                    self._train_time_to_idxs[time_indices].append(idx)
            for idx in infer_idxs:
                if idx not in self._infer_time_to_idxs[time_indices]:
                    self._infer_time_to_idxs[time_indices].append(idx)

    def transform(self, x: Tensor, in_place: bool = True) -> Tensor:
        if not in_place:
            x = x.clone()

        if x.shape[-1] == self.num_training_input_vars:
            time_to_idxs = self._train_time_to_idxs
        elif x.shape[-1] == self.num_inference_input_vars:
            time_to_idxs = self._infer_time_to_idxs
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match training ({self.num_training_input_vars}) "
                f"or inference ({self.num_inference_input_vars})"
            )

        for t_idx_list, idxs in time_to_idxs.items():
            if not idxs:
                continue
            x[:, t_idx_list, ..., idxs] = 0

        return x
