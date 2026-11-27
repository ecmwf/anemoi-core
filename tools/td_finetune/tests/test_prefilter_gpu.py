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
loss = model.loss["data"].cuda()
print(f"built in {time.time()-t0:.0f}s; loss = {type(loss).__name__} with {len(loss.losses)} terms, weights {loss.loss_weights}", flush=True)
print("patch default:", ms.PREFILTER_VARIABLES)

G, NV_OUT, NV_DATA = 542080, 87, 101
torch.manual_seed(0)
dev = "cuda"
pred = (torch.randn(1, 6, 2, G, NV_OUT, device=dev) * 0.5).abs()      # (bs, time, ens, grid, vars)
tgt = (torch.randn(1, 6, 1, G, NV_DATA, device=dev) * 0.5).abs()      # (bs, time, 1, grid, vars) in DATA_FULL layout
kw = dict(grid_shard_slice=None, group=None, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)

def run(child, flag):
    ms.PREFILTER_VARIABLES = flag
    p = pred.clone().requires_grad_(True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = child(p, tgt, **kw)
    fwd_peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    retained = (torch.cuda.memory_allocated() - base) / 2**30
    out.backward()
    return out.detach().double().item(), p.grad.detach().double(), fwd_peak, retained

names = ["1 per-step all-but-tp", "2 time-agg all-but-tp", "3 per-step tp/fc", "4 per-step tp/imerg", "5 time-agg tp/fc", "6 time-agg tp/imerg"]
def l2(a, b):
    return ((a - b).norm() / max(b.norm().item(), 1e-30)).item()
def tpcols(g):   # gradient is only non-zero in the selected prediction column(s)
    return g
rows = []
for i, child in enumerate(loss.losses):
    ls1, gs1, pk_s, rt_s = run(child, False)
    ls2, gs2, _, _ = run(child, False)
    lp1, gp1, pk_p, rt_p = run(child, True)
    lp2, gp2, _, _ = run(child, True)
    nz = (gs1.abs().sum(dim=tuple(d for d in range(gs1.dim()) if d != gs1.dim() - 1)) > 0).sum().item()
    print(f"term {names[i]:24s} loss: stock {ls1:.8e} | stock-vs-stock {abs(ls2-ls1)/abs(ls1):.1e} | patched-vs-stock {abs(lp1-ls1)/abs(ls1):.1e}   "
          f"grad rel-L2: stock-vs-stock {l2(gs2, gs1):.1e} | patched-vs-patched {l2(gp2, gp1):.1e} | patched-vs-stock {l2(gp1, gs1):.1e}   "
          f"(nonzero variable columns in grad: {nz})   retained GiB {rt_s:.1f} -> {rt_p:.1f}, fwd peak {pk_s:.1f} -> {pk_p:.1f}", flush=True)
