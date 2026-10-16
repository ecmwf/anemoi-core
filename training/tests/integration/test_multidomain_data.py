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
from torch.utils.data import ConcatDataset

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.data.loading import CrossDatasetLoadingPolicy
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.seeding import set_base_seed
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
    set_base_seed(0)
    assert isinstance(datamodule.ds_train, ConcatDataset)
    assert isinstance(datamodule.loading_policy, CrossDatasetLoadingPolicy)

    sampled_domains = set()
    grid_sizes = {}
    data_readers = {
        name: reader for domain in datamodule.ds_train.datasets for name, reader in domain.data_readers.items()
    }
    for batch in datamodule.train_dataloader():
        assert len(batch) == 1
        domain = next(iter(batch))
        sampled_domains.add(domain)
        grid_sizes[domain] = batch[domain].shape[-2]
        assert grid_sizes[domain] == data_readers[domain].grid_size
        if sampled_domains == {"sg_1", "sg_2"}:
            break

    assert sampled_domains == {"sg_1", "sg_2"}
    assert len(set(grid_sizes.values())) == 2


def test_config_validation_multidomain(multidomain_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidomain_config
    cfg = convert_to_omegaconf(BaseSchema(**cfg))
    assert cfg.dataloader.loading_policy._target_ == "anemoi.training.data.loading.CrossDatasetLoadingPolicy"
    assert cfg.model.model.hidden_nodes_name == {
        "sg_1": "sg_1_hidden",
        "sg_2": "sg_2_hidden",
    }


def test_config_validation_multidomain_ensemble(
    multidomain_ensemble_config: tuple[DictConfig, list[str]],
) -> None:
    cfg, _ = multidomain_ensemble_config
    cfg = convert_to_omegaconf(BaseSchema(**cfg))
    assert cfg.model.model._target_ == "anemoi.models.models.AnemoiEnsModelEncProcDec"
    assert cfg.training.method._target_ == "anemoi.training.train.methods.EnsembleTraining"
    assert cfg.training.training_loss.datasets.sg_1._target_ == "anemoi.training.losses.CRPS"
    assert cfg.training.training_loss.datasets.sg_2._target_ == "anemoi.training.losses.CRPS"
