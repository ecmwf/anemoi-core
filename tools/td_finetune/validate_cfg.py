# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys, glob, os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from anemoi.training.schemas.base_schema import BaseSchema
d = sys.argv[1]
for f in sorted(glob.glob(f"{d}/temporal_downscaler_imerg_ft_*.yaml")):
    name = os.path.basename(f)[:-5]
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    BaseSchema(**cfg)
    print("schema OK", name, "| steps", cfg.training.max_steps, "lr", cfg.training.optimization.lr,
          "| imerg ds", cfg.dataloader.training.datasets.data.dataset_config.dataset[2].dataset.split("/")[-1][:30],
          "| warm", cfg.system.input.warm_start.split("/")[-1], "| tl", cfg.training.transfer_learning)
