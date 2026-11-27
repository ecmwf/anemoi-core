# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU-only check on a login node: compose the config, build the trainer's model with the
weights-only warm start, and compare every tensor with the converted checkpoint."""
import sys, torch
from hydra import compose, initialize_config_dir
from anemoi.training.train.train import AnemoiTrainer

BASE = "/gpfs/scratch/ehpc01/ecm807005/td_mn5"
with initialize_config_dir(config_dir=f"{BASE}/configs", version_base=None):
    cfg = compose(config_name="td_no_w_mn5", overrides=[
        "system.hardware.accelerator=cpu", "system.hardware.num_nodes=1",
        "system.hardware.num_gpus_per_node=1", "system.hardware.num_gpus_per_model=1",
        "system.hardware.num_gpus_per_ensemble=1",
        "diagnostics.log.mlflow.enabled=false", "dataloader.num_workers.training=0",
        "dataloader.num_workers.validation=0", "dataloader.num_workers.test=0",
    ])
t = AnemoiTrainer(cfg)
model = t.model
print("model built:", type(model).__name__)
ck = torch.load(f"{BASE}/ckpt/no_w_weights_only.ckpt", map_location="cpu", weights_only=False)["state_dict"]
sd = model.state_dict()
missing = [k for k in ck if k not in sd]
extra = [k for k in sd if k not in ck]
bad = [k for k in ck if k in sd and (ck[k].shape != sd[k].shape or (ck[k].is_floating_point() and not torch.equal(ck[k], sd[k])))]
print(f"ckpt tensors {len(ck)}, model tensors {len(sd)}; ckpt-not-in-model {len(missing)}, model-not-in-ckpt {len(extra)}, differing {len(bad)}")
print("missing:", missing[:8]); print("extra:", extra[:8]); print("differing:", bad[:8])
