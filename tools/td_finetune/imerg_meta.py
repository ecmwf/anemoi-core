# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json, sys
from omegaconf import OmegaConf
from anemoi.datasets import open_dataset
from anemoi.training.data.data_reader import _normalize_dataset_config
cfg = OmegaConf.load("/scratch/momc/td_ag/configs/temporal_downscaler_imerg_ft_5k_lr5e-6.yaml")
OmegaConf.resolve(cfg)
dc = cfg.dataloader.validation.datasets.data.dataset_config
ds = open_dataset(_normalize_dataset_config(dc))
vm = ds.variables_metadata
print(type(ds).__name__, len(ds.variables), "vars; last 3:", ds.variables[-3:])
print("imerg:", json.dumps(vm["imerg"], default=str))
print("tp:", json.dumps(vm["tp"], default=str))
json.dump(vm["imerg"], open("/scratch/momc/td_ag/imerg_variable_metadata.json", "w"), default=str, indent=1)
