# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Turn an anemoi *inference* checkpoint (pickled AnemoiModelInterface) into a weights-only
Lightning-style *training* checkpoint. No optimizer/scheduler/loop state: use with
training.load_weights_only=True (fine-tuning / warm start). Also writes the graph used by the model."""
import collections, json, sys, zipfile
import torch
from omegaconf import OmegaConf

src, dst, graph_dst = sys.argv[1:4]
# optional: NAME=INDEX pairs for target-only variables added to the dataset after pretraining (e.g. imerg=100).
# The transfer-learning check (compare_variables) rejects a variable that is in the data but not in the
# checkpoint's data_indices; adding it to the checkpoint's name_to_index (appended at the end) satisfies it.
extra_vars = dict((a.split("=")[0], int(a.split("=")[1])) for a in sys.argv[4:] if "=" in a)
# NAME@file.json: variable metadata (units, period, ...) for such a variable, added to the checkpoint's metadata so the
# unit-compatibility check (which needs identical variable sets) passes.
extra_meta = dict((a.split("@")[0], json.load(open(a.split("@")[1]))) for a in sys.argv[4:] if "@" in a)
model = torch.load(src, weights_only=False, map_location="cpu")
with zipfile.ZipFile(src) as z:
    name = next(n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json"))
    metadata = json.loads(z.read(name))
sd = model.state_dict()
ckpt = {
    "state_dict": collections.OrderedDict(("model." + k, v) for k, v in sd.items()),
    "epoch": 0,
    "global_step": 0,
    "pytorch-lightning_version": "2.6.5",
    "loops": {},
    "callbacks": {},
    "optimizer_states": [],
    "lr_schedulers": [],
    "hyper_parameters": {
        # transfer_learning_loading() runs checkpoint migrations that read hyper_parameters["config"]
        "config": OmegaConf.create(metadata["config"]),
        "data_indices": model.data_indices,
        "metadata": metadata,
        "statistics": model.statistics,
        "statistics_tendencies": model.statistics_tendencies,
        "supporting_arrays": model.supporting_arrays,
    },
}
for name, vm in extra_meta.items():
    for ds_name in metadata["dataset"]:
        metadata["dataset"][ds_name]["variables_metadata"][name] = vm
    print("added variables_metadata for", name)
if extra_vars:
    import copy
    di = copy.deepcopy(model.data_indices)
    for ds_name, ic in di.items():
        for name, idx in extra_vars.items():
            assert name not in ic.name_to_index and idx == len(ic.name_to_index), (name, idx, len(ic.name_to_index))
            ic.name_to_index[name] = idx
    ckpt["hyper_parameters"]["data_indices"] = di
    print("added to data_indices.name_to_index:", extra_vars)
torch.save(ckpt, dst)
torch.save(model.graph_data, graph_dst)
print("state_dict tensors:", len(sd), "| data_indices datasets:", list(model.data_indices))
