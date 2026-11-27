# Temporal downscaler: warm-start / IMERG fine-tuning on MN5 and ag

Tooling used to continue and fine-tune the AIFS temporal downscaler (TD, run `4753602a…`, `no_w`, 150k steps)
on MareNostrum 5 (x86, H100 64 GB) and ECMWF ag (aarch64, GH200 96 GB) while JUPITER was unavailable.
Branch `feat/td-imerg-finetune` = `training-0.17.1` + the loss patch below + this directory.

**Everything in here is site specific**: paths (`/scratch/momc/...`, `/gpfs/scratch/ehpc01/ecm807005/...`,
`/lus/h1aiws01/...`), accounts and QoS are hard-coded for one user. Treat the scripts as worked examples to edit.

## 1. The code change (in `training/`)

`MultiscaleLossWrapper` now selects the variables of a `LossVariableMapper` below it **before** smoothing
(`losses/multiscale.py`, `losses/variable_mapper.py`). A `tp`-only term built as
`MultiscaleLossWrapper(per_scale_loss=CRPS(predicted_variables=[tp], ...))` used to smooth all 87 model outputs at every
scale and only then keep `tp`; with the six-term IMERG loss that needed ~97 GiB retained for backward and ran out of memory on
64 GB (MN5) and 96 GB (GH200) GPUs. The patch gives identical results (bit-identical in float64, float32-rounding otherwise)
and frees ~24 GiB. `ANEMOI_NO_LOSS_PREFILTER=1` restores the old behaviour. Unit tests:
`training/tests/unit/losses/test_multiscale_prefilter.py`. Real-loss equivalence checks: `tests/test_prefilter_gpu.py`
(single precision, GPU) and `tests/test_prefilter_f64_cpu.py` (float64, CPU).

## 2. Installing the branch into a venv

anemoi-training is pure Python, so one wheel works on x86 and aarch64:

    cd training
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ANEMOI_TRAINING=0.17.1+tdimerg.<sha> uv build --wheel --out-dir <wheels> .
    uv pip install --python <venv>/bin/python --no-deps --reinstall \
        <wheels>/anemoi_training-0.17.1+tdimerg.<sha>-py3-none-any.whl \
        anemoi_datasets-0.5.42-py3-none-any.whl anemoi_utils-0.5.12-py3-none-any.whl

The AIFS v3 lock files pin anemoi-datasets 0.5.40 / utils 0.5.9; datasets >= 0.5.42 is needed to join the forecast
trajectory zarr with hourly IMERG (0.5.40: "Trajectories datasets have two time axes"). MN5 login nodes have no internet:
build/download the wheels elsewhere and install with `uv pip install --offline`. The MN5 venv used by the other aifs-dev
variants is shared, so a copy (`td_mn5/venv`, console-script shebangs rewritten) is used for the TD runs.

## 3. Workflow

1. `convert_inference_to_train_ckpt.py <inference.ckpt> <out.ckpt> <graph_out.pt> [imerg=100 imerg@meta.json]`
   - inference checkpoint -> weights-only training checkpoint (no optimiser state: warm start / fine-tune only).
   - `transfer_learning_loading` reads `hyper_parameters["config"]`, `data_indices`, `metadata`, `statistics`: all stored.
   - For the IMERG fine-tune the extra target-only variable has to be in the checkpoint's `data_indices.name_to_index`
     (`imerg=100`) and `variables_metadata` (`imerg@file.json`, taken from `imerg_meta.py`), otherwise the two
     compatibility checks (`compare_variables`, `check_variables_metadata_compatibility`) raise.
2. `make_mn5_config.py` rebuilds a flat, self-contained training config from the config stored in the checkpoint.
3. `graft_node_attrs.py <td_graph.pt> <latblend_node_attrs.pt> <out.pt>` adds `oper_lat_weight`/`imerg_lat_weight`
   node weights to the TD graph (edges untouched). `latblend_node_attrs.pt` (11 MB) is not committed.
4. `make_imerg_configs.py <out_dir>` layers the IMERG changes (`jupiter_configs/`, recovered from the original JUPITER
   set-up) onto that config and writes the six runs (5k/10k steps x effective lr 2.5e-6/5e-6/1e-5).
5. `validate_cfg.py <config_dir>` (schema), `check_load.py` (warm-start weights identical on CPU),
   `jobscripts/*/smoke_imerg.sh` (1 node, 30 steps), then `jobscripts/*/run_imerg.sh` (8 nodes).

## 4. Things the JUPITER configs did not account for (none had run a training step)

- **lr scaling**: effective lr = `num_nodes * 4 * lr / 2`. The JUPITER job script exported `SLURM_NNODES=4`; on 8 real nodes
  the config `lr` is `target / 16`. The run scripts refuse to start on a node count other than 8.
- **NaNs in IMERG** (~20-40 points per field) trip the pre-processor assertion on the first batch. `imerg` is added to the
  `const_imputer` `0:` list (NaN -> 0 in normalised space, i.e. "no rain"; ~0.006 % of points). This is the only change that
  alters the loss relative to the JUPITER design.
- **Loss memory** (section 1) and `sparse_projector_num_chunks: 8` on the multiscale wrappers.
- Nested `CombinedLoss` is rejected by the 0.17 config schema, so the memory problem could not be solved in the config.
- `--kill-on-bad-exit=1` on `srun`: one rank dying otherwise left the others hanging for ~30 minutes.

## 5. Contents

    convert_inference_to_train_ckpt.py   make_mn5_config.py   make_imerg_configs.py   graft_node_attrs.py
    imerg_meta.py   check_load.py   validate_cfg.py
    tests/                  GPU / float64 equivalence checks of the loss patch on the real loss
    jobscripts/mn5|ag/      env_td.sh, run_imerg.sh, smoke_imerg.sh (mn5 also run_td.sh, smoke_td.sh: no-IMERG warm start)
    jupiter_configs/        the original Hydra configs (base, data, dataloader, six runs), as recovered; not runnable as is
