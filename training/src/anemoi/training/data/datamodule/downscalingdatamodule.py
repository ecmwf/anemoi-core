# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from functools import cached_property

import numpy as np
from hydra.utils import instantiate

from anemoi.datasets import open_dataset
from anemoi.training.data.datamodule.singledatamodule import AnemoiDatasetsDataModule
from anemoi.training.data.dataset.downscalingdataset import DownscalingDataset
from anemoi.training.data.grid_indices import BaseGridIndices

LOGGER = logging.getLogger(__name__)


class DownscalingAnemoiDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Downscaling Datasets data module for PyTorch Lightning."""

    @cached_property
    def statistics(self) -> dict:

        statistics = list(self.ds_train.statistics)
        if not hasattr(self.config.hardware.paths, "residual_statistics") or not hasattr(
            self.config.hardware.files,
            "residual_statistics",
        ):
            LOGGER.warning("No residual_statistics path configured, using base statistics only")
            return tuple(statistics)

        residual_statistics = np.load(
            os.path.join(
                self.config.hardware.paths.residual_statistics,
                self.config.hardware.files.residual_statistics,
            ),
            allow_pickle=True,
        ).item()
        out_name_to_index = self.ds_train.name_to_index[2]
        use_residual_for = getattr(self.config.data, "residual_fields", None)
        if use_residual_for is None:
            # Backward compatible: use residual stats for ALL variables
            LOGGER.info("Not using residual statistics for any variable (no filter specified)")
            variables_to_use_residual = []  # reduced_name_to_index
        else:
            # Only use residual stats for specified variables
            variables_to_use_residual = [var for var in use_residual_for if var in out_name_to_index.keys()]
            LOGGER.info(
                f"Using residual statistics for {len(variables_to_use_residual)} variables: {variables_to_use_residual}",
            )

            # Log variables that were specified but not found
            missing_vars = [var for var in use_residual_for if var not in out_name_to_index.keys()]
            if missing_vars:
                LOGGER.warning(
                    f"Variables specified for residual normalization but not found in dataset: {missing_vars}",
                )
        LOGGER.info(f"Variables to be used for residuals with residual statistics: {variables_to_use_residual}")
        base_stats = statistics[2]  # Original high-res statistics
        mean_array = []
        stdev_array = []
        maximum_array = []
        minimum_array = []

        for field_name in list(out_name_to_index.keys()):
            if field_name in variables_to_use_residual:
                # Use residual statistics for this variable
                mean_array.append(residual_statistics["mean"][field_name])
                stdev_array.append(residual_statistics["stdev"][field_name])
                maximum_array.append(residual_statistics["maximum"][field_name])
                minimum_array.append(residual_statistics["minimum"][field_name])
            else:
                # Use base statistics for this variable
                idx = out_name_to_index[field_name]
                mean_array.append(base_stats["mean"][idx])
                stdev_array.append(base_stats["stdev"][idx])
                maximum_array.append(base_stats["maximum"][idx])
                minimum_array.append(base_stats["minimum"][idx])
        # Create the combined statistics dictionary
        combined_statistics = {
            "mean": np.array(mean_array),
            "stdev": np.array(stdev_array),
            "maximum": np.array(maximum_array),
            "minimum": np.array(minimum_array),
        }

        statistics[2] = combined_statistics
        # Carry the original (non-residual) high-res output statistics as a 4th
        # entry so the autoregressive `prev_hres` normalizer can normalize the
        # previous full CERRA state with its true (non-residual) statistics.
        # Existing consumers only index [0]/[1]/[2], so this is backward-safe.
        statistics.append(base_stats)
        return tuple(statistics)
        # return reduced_residual_statistics

    @staticmethod
    def _cfg_get(obj, key, default):
        """Read ``key`` from a config node that may be a dict, an OmegaConf/DotDict
        node or a pydantic schema object, returning ``default`` if absent."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @cached_property
    def _ar_num_previous_steps(self) -> int:
        """Number of previous high-res steps fed as autoregressive conditioning.

        Read from ``config.model.model.autoregressive``. Defaults to 0 (no AR),
        which reproduces the pure-downscaler single-slice behaviour exactly.
        """
        model_node = self._cfg_get(self._cfg_get(self.config, "model", None), "model", None)
        ar_node = self._cfg_get(model_node, "autoregressive", None)
        if ar_node is None or not bool(self._cfg_get(ar_node, "use_previous_step", False)):
            return 0
        return int(self._cfg_get(ar_node, "num_previous_steps", 1))

    @cached_property
    def _ar_relative_date_indices(self) -> np.ndarray:
        """Relative date offsets to load per sample.

        ``[0]`` for the pure downscaler (single slice). For autoregressive with
        ``p`` previous steps: ``[0, inc, 2*inc, ..., p*inc]`` where ``inc`` is the
        model timeincrement; the last offset is the current target, the offset
        just before it is the previous step used for temporal conditioning.
        """
        num_prev = self._ar_num_previous_steps
        if num_prev <= 0:
            return np.array([0], dtype=np.int64)
        inc = self.timeincrement
        return np.array([inc * k for k in range(num_prev + 1)], dtype=np.int64)

    def _get_dataset(
        self,
        data_reader,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
        overfit_on_index: int | None = None,
    ) -> DownscalingDataset:

        data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets
        data = DownscalingDataset(
            data_reader=data_reader,
            relative_date_indices=self._ar_relative_date_indices,
            shuffle=shuffle,
            label=label,
            lres_grid_indices=self.lres_grid_indices,
            hres_grid_indices=self.hres_grid_indices,
            overfit_on_index=overfit_on_index,
        )
        return data

    @cached_property
    def lres_grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        lres_grid_indices = instantiate(
            self.config.dataloader.lres_grid_indices,
            reader_group_size=reader_group_size,
        )
        lres_grid_indices.setup(self.graph_data)
        return lres_grid_indices

    @cached_property
    def hres_grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        hres_grid_indices = instantiate(
            self.config.dataloader.hres_grid_indices,
            reader_group_size=reader_group_size,
        )
        hres_grid_indices.setup(self.graph_data)
        return hres_grid_indices

    @cached_property
    def supporting_arrays(self) -> dict:
        return {
            k: v[1] for k, v in self.ds_train.supporting_arrays.items()
        }  # | {k: v[1] for k, v in self.grid_indices.supporting_arrays.items()}

    @cached_property
    def ds_valid(self) -> DownscalingDataset:

        return self._get_dataset(
            open_dataset(self.config.dataloader.validation),
            shuffle=False,
            label="validation",
        )
