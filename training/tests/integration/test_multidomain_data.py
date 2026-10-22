# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import DictConfig

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf


def test_config_validation_multidomain(multidomain_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidomain_config
    cfg = convert_to_omegaconf(BaseSchema(**cfg))
    assert cfg.dataloader.loading_policy._target_ == "anemoi.training.data.loading.CrossDatasetLoadingPolicy"
    assert cfg.model.model.hidden_nodes_name == {
        "sg_1": "sg_1_hidden",
        "sg_2": "sg_2_hidden",
    }
