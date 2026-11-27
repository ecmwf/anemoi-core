# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build the six IMERG fine-tune configs for MN5 as flat, self-contained yaml files.

Base  = the training config stored in the pretraining checkpoint (td_no_w_mn5.yaml: model, graph-free, MN5 paths)
Layer = what the JUPITER IMERG configs changed: data group (zarr_td_imerg), dataloader group (native_grid_td_imerg)
        and the top-level overrides of temporal_downscaler_imerg_finetune.yaml (loss, scalers, schedule ...).
The JUPITER Hydra groups graph/multi_scale_knn and model/graphtransformer_ens_td are not needed: the model comes
from the checkpoint's stored config and the graph is loaded from file.
"""
import sys
from omegaconf import OmegaConf

SRC = "/scratch/momc/td_inference_tmp/mn5_prep/imerg_src/jup_cfg"
OUT = sys.argv[1]
BASE = "/gpfs/scratch/ehpc01/ecm807005/td_mn5"
base = OmegaConf.load("/scratch/momc/td_inference_tmp/mn5_prep/td_no_w_mn5.yaml")
top = OmegaConf.load(f"{SRC}/temporal_downscaler_imerg_finetune.yaml")
data = OmegaConf.load(f"{SRC}/data/zarr_td_imerg.yaml")
dl = OmegaConf.load(f"{SRC}/dataloader/native_grid_td_imerg.yaml")

overrides = OmegaConf.create({k: v for k, v in OmegaConf.to_container(top, resolve=False).items()
                              if k not in ("defaults", "config_validation")})
cfg = base.copy()
cfg.data = data          # group replaced, as Hydra does
# IMERG has ~20-40 NaN points per field; the processors assert no NaNs after pre-processing (first batch), so impute
# them like `ro` (NaN -> 0 in normalised space = 0 precipitation). Not in the JUPITER configs, which never ran a step.
ci = cfg.data.datasets.data.processors.const_imputer.config
ci[0] = list(ci[0]) + ["imerg"]
cfg.dataloader = dl
cfg = OmegaConf.merge(cfg, overrides)
# MN5 paths (the JUPITER ones in the overrides are replaced)
cfg.system.input.dataset = "/gpfs/scratch/ehpc01/ai-ml/datasets/aifs-od-fc-oper-0001-mars-n320-2016-2025-6h-1h-v5.zarr"
cfg.system.input.auxiliary_dataset = "/gpfs/scratch/ehpc01/ai-ml/datasets/aifs-od-an-oper-0001-mars-n320-2016-2025-6h-v1-for-single-v2.zarr"
cfg.system.input.imerg_dataset = f"{BASE}/data/aifs-imerg-final-v07b-nasa-n320-1998-2025-1h-v1-observations.zarr"
cfg.system.input.graph = f"{BASE}/graph/graph_td_n320_latblend_phi50.pt"
cfg.system.input.warm_start = f"{BASE}/ckpt/no_w_weights_only_imerg.ckpt"
# 6 multiscale loss terms each smooth the full tensor: split the sparse projection to fit in GPU memory
# (exact same maths, a few extra kernel launches).
import os
if os.environ.get("NESTED_TP_LOSS") == "1":   # NOT schema-valid in anemoi-training 0.17 (nested CombinedLoss rejected)
    # The four tp-only terms select tp only AFTER the multiscale smoothing (the variable filter sits on the inner loss), so each
    # smooths all 88 variables -> out of memory on 64/96 GB GPUs. Put them in two CombinedLosses whose own predicted/target
    # variable filter selects tp (forecast) / tp vs imerg BEFORE the smoothing. Same maths, ~87x smaller tensors for those terms.
    tl = OmegaConf.to_container(cfg.training.training_loss.datasets.data, resolve=False)
    L, W = tl["losses"], tl["loss_weights"]
    def strip(child):
        pl = child["per_scale_loss"] if "per_scale_loss" in child else None
        inner = pl if pl is not None else child
        leaf = inner["loss_fn"] if "loss_fn" in inner else inner
        opts = {k: leaf.pop(k) for k in ("predicted_variables", "target_variables", "check_variables_compatibility") if k in leaf}
        return child, opts
    def combined(children, weights, ignore_nans):
        stripped = [strip(c) for c in children]
        opts = stripped[0][1]
        return {"_target_": "anemoi.training.losses.combined.CombinedLoss", "scalers": [], "ignore_nans": ignore_nans,
                "predicted_variables": opts["predicted_variables"], "target_variables": opts["target_variables"],
                "check_variables_compatibility": opts["check_variables_compatibility"],
                "losses": [c for c, _ in stripped], "loss_weights": weights}
    tp_fc = combined([L[2], L[4]], [W[2], W[4]], False)    # per-step + time-aggregate, forecast tp
    tp_im = combined([L[3], L[5]], [W[3], W[5]], True)     # per-step + time-aggregate, tp vs IMERG
    tl["losses"] = [L[0], L[1], tp_fc, tp_im]
    tl["loss_weights"] = [W[0], W[1], 1.0, 1.0]
    cfg.training.training_loss.datasets.data = OmegaConf.create(tl)
for lw in [l for l in cfg.training.training_loss.datasets.data.losses if "per_scale_loss" in l]:
    lw.sparse_projector_num_chunks = 8
cfg.graph = {"overwrite": False, "post_processors": []}      # load the grafted graph as-is
cfg.system.hardware.num_nodes = 8                            # overridden from the allocation by run_td.sh

NODES_FOR_LR = 2 * cfg.system.hardware.num_nodes  # effective lr = nodes*4 GPUs*lr/2 (ensemble.py) = 2*nodes*lr
runs = {  # name -> (steps, effective peak lr)
    "5k_lr2p5e-6": (5000, 2.5e-6), "5k_lr5e-6": (5000, 5e-6), "5k_lr1e-5": (5000, 1e-5),
    "10k_lr2p5e-6": (10000, 2.5e-6), "10k_lr5e-6": (10000, 5e-6), "10k_lr1e-5": (10000, 1e-5),
}
for name, (steps, peak) in runs.items():
    c = cfg.copy()
    c.training.max_steps = steps
    c.training.optimization.lr = peak / NODES_FOR_LR   # 8 nodes -> effective peak = `peak`
    c.diagnostics.log.mlflow.run_name = f"td_imerg_ft_{name}"
    path = f"{OUT}/temporal_downscaler_imerg_ft_{name}.yaml"
    header = (f"# IMERG fine-tune of the TD no_w checkpoint (MN5). {steps} steps, effective peak lr {peak:g} "
              f"(config lr {peak / NODES_FOR_LR:g} x {NODES_FOR_LR} for 8 nodes x 4 GPUs / 2 per ensemble), "
              f"warmup 300, cosine to 3e-7.\n# Generated by make_imerg_configs.py; MUST be run on 8 nodes or the effective lr changes.\n")
    open(path, "w").write(header + OmegaConf.to_yaml(c, resolve=False))
    print(name, steps, c.training.optimization.lr)
