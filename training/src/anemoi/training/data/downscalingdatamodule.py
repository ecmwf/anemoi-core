# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from collections.abc import Mapping
from functools import cached_property

import numpy as np

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)


class DownscalingAnemoiDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Downscaling Datasets data module for PyTorch Lightning.

    Handles residual statistics for diffusion downscaling by loading them from
    an external file and exposing them via `statistics_tendencies`.

    The model interface uses these to build separate normalizers:
    - `pre_processors[target]`: normalizes with state statistics (for direct prediction)
    - `pre_processors_tendencies[target]`: normalizes with residual statistics (for residual prediction)

    Configuration:
    - system.input.residual_statistics: Path to .npy file with residual statistics
      Format: dict with keys "mean", "stdev", "maximum", "minimum",
      each mapping variable names to scalar values.
    - model.model.residual_prediction: dict mapping target_dataset -> source_dataset,
      or True (defaults to {"out_hres": "in_lres"}), or False (no residual).
    """

    @cached_property
    def _residual_target_datasets(self) -> list[str]:
        """Determine which datasets are residual targets from config.

        Reads model.model.residual_prediction which must be:
        - dict: {target: source} — keys are the target datasets
        - False/empty: no residual prediction
        """
        raw = self.config.model.model.get("residual_prediction", False)
        if isinstance(raw, Mapping):
            return list(raw.keys())
        return []

    def _load_residual_statistics(self) -> dict | None:
        """Load residual statistics from the configured file path.

        Returns
        -------
        dict or None
            Dict keyed by variable name: {"mean": {"10u": val, ...}, "stdev": {...}, ...}
            or None if not configured.
        """
        if not hasattr(self.config.system.input, "residual_statistics"):
            return None

        path = self.config.system.input.residual_statistics
        if path is None:
            return None

        LOGGER.info("Loading residual statistics from %s", path)
        return np.load(os.path.join(path), allow_pickle=True).item()

    def _residual_stats_to_array(self, residual_stats: dict, dataset_name: str) -> dict:
        """Convert residual statistics from {stat: {var: val}} to {stat: np.array}.

        Reorders to match the target dataset's variable ordering. Variables not present in
        residual statistics are filled with neutral defaults (mean=0, stdev=1).

        Parameters
        ----------
        residual_stats : dict
            Raw residual statistics: {"mean": {"10u": val, ...}, ...}
        dataset_name : str
            Name of the target dataset to match variable ordering.

        Returns
        -------
        dict
            {"mean": np.array, "stdev": np.array, "maximum": np.array, "minimum": np.array}
        """
        field_names = list(self.ds_train.name_to_index[dataset_name].keys())

        mean_array = []
        stdev_array = []
        maximum_array = []
        minimum_array = []

        for field_name in field_names:
            if field_name in residual_stats.get("mean", {}):
                mean_array.append(residual_stats["mean"][field_name])
                stdev_array.append(residual_stats["stdev"][field_name])
                maximum_array.append(residual_stats["maximum"][field_name])
                minimum_array.append(residual_stats["minimum"][field_name])
            else:
                # Variable not in residual stats — use neutral defaults
                # (mean=0, stdev=1 means no normalization change for these channels)
                LOGGER.debug("Variable %s not in residual statistics, using neutral defaults", field_name)
                mean_array.append(0.0)
                stdev_array.append(1.0)
                maximum_array.append(1.0)
                minimum_array.append(-1.0)

        return {
            "mean": np.array(mean_array),
            "stdev": np.array(stdev_array),
            "maximum": np.array(maximum_array),
            "minimum": np.array(minimum_array),
        }

    @cached_property
    def statistics_tendencies(self) -> dict[str, dict | None] | None:
        """Return residual statistics as tendency statistics for residual target datasets.

        This hooks into the existing tendency processor infrastructure:
        the model interface builds `pre_processors_tendencies[target]` from these stats,
        which is then used by compute_residuals/add_interp_to_state to normalize
        residual channels separately from direct prediction channels.

        The returned format wraps residual stats in the lead_times structure expected
        by both tendency processors and tendency scalers:
        {dataset_name: {"lead_times": [lt], lt: {mean: ..., stdev: ...}} or None}

        Returns
        -------
        dict or None
            {dataset_name: stats_or_None, ...} for all datasets
        """
        residual_stats = self._load_residual_statistics()
        if residual_stats is None:
            LOGGER.info("No residual statistics configured, tendency processors will not be built")
            return None

        target_datasets = self._residual_target_datasets

        # Use lead_time from config to wrap stats in the expected format
        lead_time = self._lead_time_for_step(1)

        # Build the dict keyed by dataset name (matching statistics dict structure)
        stats = {}
        for dataset_name in self.ds_train.statistics.keys():
            if dataset_name in target_datasets:
                array_stats = self._residual_stats_to_array(residual_stats, dataset_name)
                stats[dataset_name] = {
                    "lead_times": [lead_time],
                    lead_time: array_stats,
                }
                LOGGER.info(
                    "Residual statistics loaded for %s (%d variables, lead_time=%s)",
                    dataset_name,
                    len(array_stats["mean"]),
                    lead_time,
                )
            else:
                stats[dataset_name] = None

        return stats
