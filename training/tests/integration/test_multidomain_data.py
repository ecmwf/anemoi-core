# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.sampler import CrossDatasetSampler
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


@skip_if_offline
@pytest.mark.slow
def test_multidomain_dataloader(
    multidomain_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = multidomain_config
    for url in urls:
        get_test_archive(url)

    cfg = convert_to_omegaconf(BaseSchema(**cfg))
    datamodule = AnemoiDatasetsDataModule(cfg, instantiate(cfg.task))
    assert isinstance(datamodule.ds_train, MultiDataset)
    assert isinstance(datamodule.ds_train.sampler, CrossDatasetSampler)

    sampled_domains = set()
    grid_sizes = {}
    for batch in datamodule.train_dataloader():
        assert len(batch) == 1
        domain = next(iter(batch))
        sampled_domains.add(domain)
        grid_sizes[domain] = batch[domain].shape[-2]
        assert grid_sizes[domain] == datamodule.ds_train.data_readers[domain].grid_size
        if sampled_domains == {"era5", "cerra"}:
            break

    assert sampled_domains == {"era5", "cerra"}
    assert len(set(grid_sizes.values())) == 2


def test_config_validation_multidomain(multidomain_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidomain_config
    cfg = convert_to_omegaconf(BaseSchema(**cfg))
    assert cfg.dataloader.sampler._target_ == "anemoi.training.data.sampler.CrossDatasetSampler"
