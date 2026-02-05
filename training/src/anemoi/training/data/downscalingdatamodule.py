# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from hydra.utils import instantiate
from functools import cached_property
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

import numpy as np


LOGGER = logging.getLogger(__name__)


class DownscalingAnemoiDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Downscaling Datasets data module for PyTorch Lightning."""

    @cached_property
    def statistics(self) -> dict:
        """Return statistics, optionally using residual statistics for specified variables.

        Configuration:
        - system.input.residual_statistics: Path to residual statistics file
        - data.use_residual_normalization_for: List of variable names that should use residual statistics
          If not specified or empty, ALL variables will use residual statistics (backward compatible)
        """
        statistics = self.ds_train.statistics

        # Check if residual statistics are configured
        if not hasattr(self.config.system.input, "residual_statistics"):
            LOGGER.warning("No residual_statistics path configured, using base statistics only")
            return tuple(statistics)

        # Load residual statistics from file
        residual_statistics = np.load(
            os.path.join(self.config.system.input.residual_statistics),
            allow_pickle=True,
        ).item()

        # Get the name_to_index mapping for the high-res dataset (index 2)
        reduced_name_to_index = self.ds_train.name_to_index["out_hres"].keys()

        # Get list of variables that should use residual normalization
        # If not specified, default to ALL variables (backward compatible)
        use_residual_for = getattr(self.config.data, "use_residual_normalization_for", None)

        if use_residual_for is None:
            # Backward compatible: use residual stats for ALL variables
            LOGGER.info("Using residual statistics for ALL variables (no filter specified)")
            variables_to_use_residual = reduced_name_to_index
        else:
            # Only use residual stats for specified variables
            variables_to_use_residual = [var for var in use_residual_for if var in reduced_name_to_index]
            LOGGER.info(
                f"Using residual statistics for {len(variables_to_use_residual)} variables: {variables_to_use_residual}"
            )

            # Log variables that were specified but not found
            missing_vars = [var for var in use_residual_for if var not in reduced_name_to_index]
            if missing_vars:
                LOGGER.warning(
                    f"Variables specified for residual normalization but not found in dataset: {missing_vars}"
                )

        # Build statistics arrays, selectively using residual stats
        base_stats = statistics["out_hres"]  # Original high-res statistics
        field_names_ordered = list(reduced_name_to_index)

        mean_array = []
        stdev_array = []
        maximum_array = []
        minimum_array = []

        for field_name in field_names_ordered:
            if field_name in variables_to_use_residual:
                # Use residual statistics for this variable
                mean_array.append(residual_statistics["mean"][field_name])
                stdev_array.append(residual_statistics["stdev"][field_name])
                maximum_array.append(residual_statistics["maximum"][field_name])
                minimum_array.append(residual_statistics["minimum"][field_name])
            else:
                # Use base statistics for this variable
                idx = field_names_ordered.index(field_name)
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

        statistics["out_hres"] = combined_statistics
        return statistics
