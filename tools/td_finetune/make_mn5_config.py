# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json, copy, yaml
m = json.load(open("/scratch/momc/td_inference_tmp/mn5_prep/no_w_metadata.json"))
c = copy.deepcopy(m["config"])
BASE = "/gpfs/scratch/ehpc01/ecm807005/td_mn5"
DATA = "/gpfs/scratch/ehpc01/ai-ml/datasets/"
paths = set()
def fix(o):
    if isinstance(o, dict): return {k: fix(v) for k, v in o.items()}
    if isinstance(o, list): return [fix(v) for v in o]
    if isinstance(o, str):
        for pre in ("/e/data1/jureap-data/ecmwf/ai-ml/datasets/", "/e/data1/jureap-data/ai-ml/datasets/"):
            if o.startswith(pre):
                o = DATA + o[len(pre):]
        if o.startswith(DATA): paths.add(o)
        return o
    return o
c = fix(c)
# JSON turned the integer key of const_imputer.config into a string; the schema needs the number back
ci = c["data"]["datasets"]["data"]["processors"]["const_imputer"]["config"]
c["data"]["datasets"]["data"]["processors"]["const_imputer"]["config"] = {(int(k) if str(k).isdigit() else k): v for k, v in ci.items()}
s = c["system"]
s["hardware"].update(num_nodes=2)  # overridden from the job script
s["input"].update(graph=f"{BASE}/graph/graph_td_n320_from_ckpt.pt", warm_start=f"{BASE}/ckpt/no_w_weights_only.ckpt")
s["output"] = {
    "root": f"{BASE}/out",
    "logs": {"root": f"{BASE}/logs", "wandb": f"{BASE}/logs/wandb", "mlflow": f"{BASE}/logs/mlflow", "tensorboard": "tensorboard"},
    "checkpoints": {**{k: v for k, v in s["output"]["checkpoints"].items() if k != "root"}, "root": f"{BASE}/out/checkpoint"},
    "plots": f"{BASE}/out/plots",
    "profiler": f"{BASE}/out/profiler",
}
c["diagnostics"]["log"]["mlflow"]["save_dir"] = f"{BASE}/logs/mlflow"
c["graph"] = {"overwrite": False, "post_processors": c["graph"].get("post_processors", [])}  # load graph as-is
t = c["training"]
t.update(run_id=None, fork_run_id=None, load_weights_only=True, transfer_learning=False)
yaml.safe_dump(c, open("/scratch/momc/td_inference_tmp/mn5_prep/td_no_w_mn5.yaml", "w"), sort_keys=False, default_flow_style=False, width=140)
print("dataset paths:"); [print(" ", p) for p in sorted(paths)]
