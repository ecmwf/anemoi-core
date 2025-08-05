# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from anemoi.training.data.refactor.dataset import NativeGridMultDataset
from anemoi.training.data.refactor.read_config import convert_data_config
from anemoi.training.data.refactor.read_config import convert_sample_config
from anemoi.training.data.refactor.sample_provider import sample_provider_factory
from anemoi.training.data.utils import get_dataloader_config
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func

LOGGER = logging.getLogger(__name__)


class AnemoiMultipleDatasetsDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration
        """
        super().__init__()
        data_dict = convert_data_config(config.data.data_handlers)
        sample_dict = convert_sample_config(config.model.sample)

        training_context = dict(sources=data_dict, **config.dataloader.sampler.training)
        validation_context = dict(sources=data_dict, **config.dataloader.sampler.validation)

        # Create Sampler provider
        self.training_samples = sample_provider_factory(**training_context, **sample_dict)
        self.validation_samples = sample_provider_factory(**validation_context, **sample_dict)

        dl_keys_to_ignore = ["sampler", "read_group_size", "grid_indices", "limit_batches"]
        self.train_dataloader_config = get_dataloader_config(
            config.dataloader,
            "training",
            keys_to_ignore=dl_keys_to_ignore,
        )
        self.val_dataloader_config = get_dataloader_config(
            config.dataloader,
            "validation",
            keys_to_ignore=dl_keys_to_ignore,
        )

        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.VALIDATION])
        # data_handlers[stage.TRAINING].check_no_overlap(data_handlers[stage.TEST])
        # data_handlers[stage.VALIDATION].check_no_overlap(data_handlers[stage.TEST])

    def train_dataloader(self) -> dict[str, DataLoader]:
        dataset = NativeGridMultDataset(self.training_samples)
        return DataLoader(dataset, worker_init_fn=worker_init_func, **self.train_dataloader_config)

    def val_dataloader(self) -> dict[str, DataLoader]:
        dataset = NativeGridMultDataset(self.validation_samples)
        return DataLoader(dataset, worker_init_fn=worker_init_func, **self.val_dataloader_config)
