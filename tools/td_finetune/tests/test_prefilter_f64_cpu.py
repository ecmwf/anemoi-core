# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Equivalence + memory test of the 'select variables before smoothing' patch (multiscale.py / variable_mapper.py).
Builds the real training loss of the IMERG config and compares each of its terms with the patch off (stock) and on."""
import sys, time, torch
from hydra import compose, initialize_config_dir
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.index_space import IndexSpace
import anemoi.training.losses.multiscale as ms

BASE = "/scratch/momc/td_ag"
with initialize_config_dir(config_dir=f"{BASE}/configs", version_base=None):
    cfg = compose(config_name="temporal_downscaler_imerg_ft_5k_lr5e-6", overrides=[
        "system.hardware.accelerator=cpu", "system.hardware.num_nodes=1", "system.hardware.num_gpus_per_node=1",
        "system.hardware.num_gpus_per_model=1", "system.hardware.num_gpus_per_ensemble=1",
        "system.input.warm_start=null", "training.load_weights_only=false", "training.transfer_learning=false",
        "diagnostics.log.mlflow.enabled=false", "dataloader.num_workers.training=0",
        "dataloader.num_workers.validation=0", "dataloader.num_workers.test=0",
        f"system.output.root={BASE}/out_test", f"system.output.checkpoints.root={BASE}/out_test/ckpt",
        f"system.output.plots={BASE}/out_test/plots"])
t0 = time.time()
model = AnemoiTrainer(cfg).model
loss = model.loss["data"].cpu().double()
n_conv = 0
for m in loss.modules():
    for prov in getattr(m, "smoothing_matrices", []) or []:
        if prov is not None and prov.projection_matrix.dtype == torch.float32:
            prov.projection_matrix = prov.projection_matrix.double()
            n_conv += 1
print("converted sparse projection matrices to float64:", n_conv)
print(f"built in {time.time()-t0:.0f}s; loss = {type(loss).__name__} with {len(loss.losses)} terms, weights {loss.loss_weights}", flush=True)
print("patch default:", ms.PREFILTER_VARIABLES)

G, NV_OUT, NV_DATA = 542080, 87, 101
torch.manual_seed(0)
dev = "cpu"
pred = (torch.randn(1, 6, 2, G, NV_OUT, device=dev, dtype=torch.float64) * 0.5).abs()
tgt = (torch.randn(1, 6, 1, G, NV_DATA, device=dev, dtype=torch.float64) * 0.5).abs()
kw = dict(grid_shard_slice=None, group=None, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
def run(child, flag):
    ms.PREFILTER_VARIABLES = flag
    p = pred.clone().requires_grad_(True)
    out = child(p, tgt, **kw)
    out.backward()
    return out.detach().item(), p.grad.detach()
def l2(a, b):
    return ((a - b).norm() / max(b.norm().item(), 1e-300)).item()
names = ["3 per-step tp/fc", "4 per-step tp/imerg", "5 time-agg tp/fc", "6 time-agg tp/imerg"]
for i, child in zip(range(2, 6), loss.losses[2:]):
    ls, gs = run(child, False)
    lp, gp = run(child, True)
    print(f"[float64] term {names[i-2]:22s} loss stock {ls:.15e} patched {lp:.15e} rel diff {abs(lp-ls)/abs(ls):.1e} | grad rel-L2 patched-vs-stock {l2(gp, gs):.1e}", flush=True)
