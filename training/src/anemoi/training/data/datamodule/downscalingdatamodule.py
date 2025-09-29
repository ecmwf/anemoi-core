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
from anemoi.training.data.datamodule.singledatamodule import AnemoiDatasetsDataModule
from anemoi.training.data.dataset.downscalingdataset import DownscalingDataset
import numpy as np

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.datasets.data import open_dataset

LOGGER = logging.getLogger(__name__)


class DownscalingAnemoiDatasetsDataModule(AnemoiDatasetsDataModule):
    """Anemoi Downscaling Datasets data module for PyTorch Lightning."""

    @cached_property
    def statistics(self) -> dict:

        residual_statistics = np.load(
            os.path.join(
                self.config.hardware.paths.residual_statistics,
                self.config.hardware.files.residual_statistics,
            ),
            allow_pickle=True,
        ).item()

        reduced_name_to_index = self.ds_train.name_to_index[2].keys()
        reduced_residual_statistics = {
            "mean": np.array(
                [
                    residual_statistics["mean"][field_name]
                    for field_name in reduced_name_to_index
                ]
            ),
            "stdev": np.array(
                [
                    residual_statistics["stdev"][field_name]
                    for field_name in reduced_name_to_index
                ]
            ),
            "maximum": np.array(
                [
                    residual_statistics["maximum"][field_name]
                    for field_name in reduced_name_to_index
                ]
            ),
            "minimum": np.array(
                [
                    residual_statistics["minimum"][field_name]
                    for field_name in reduced_name_to_index
                ]
            ),
        }

        statistics = list(
            self.ds_train.statistics
        )  # to list as statistics is a tuple immutable
        statistics[2] = reduced_residual_statistics
        return tuple(statistics)
        # return reduced_residual_statistics

    def _get_dataset(
        self,
        data_reader,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> DownscalingDataset:

        data_reader = self.add_trajectory_ids(
            data_reader
        )  # NOTE: Functionality to be moved to anemoi datasets
        data = DownscalingDataset(
            data_reader=data_reader,
            relative_date_indices=np.array([0]),
            shuffle=shuffle,
            label=label,
            lres_grid_indices=self.lres_grid_indices,
            hres_grid_indices=self.hres_grid_indices,
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
