# MUST FOLLOW — read this before doing anything

## Python environment

Always use this virtualenv (activate from the hpc-login home directory) before
running anything Python — imports, tests, training, etc.:

```
source anemoi-combine-code/bin/activate
```

## Base config location

The reference/base training config for this work lives outside this repo, at:

```
/home/ecm1715/prototype/aifs-dev/aifs/single-mse
```

Always treat `single_2.0_rollout_esuite_2024.yaml` in that directory as the
**base** config. The two IMERG variants next to it are experiments layered on
top of that same base:

- `single_2.0_rollout_esuite_2024.yaml` — **base**. OPER AN-only rollout
  training config (dataset `aifs-od-an-oper-...`, 2017-07-01 → 2024-07-30, 12
  epochs, rollout maximum 12, `num_channels: 1024`, lr `8e-7`).
- `single_2.0_rollout_esuite_2024_IMERG.yaml` — **IMERG replace**. Same base,
  but joins in `nasa-imerg-grib-n320-1998-2024-6h-v1.zarr` and *replaces* the
  OPER AN `tp` channel with IMERG `tp` (drops `tp` from the OPER AN dataset,
  selects `tp` from the imerg dataset). Uses an explicit `reorder` map to
  restore the original variable ordering from the fork checkpoint
  (`fork_run_id=fe0b7068`), since dropping/re-appending `tp` would otherwise
  shift every later index by one. IMERG `tp` has NaNs near the poles
  (regridding artifact), so `training_loss.datasets.data.ignore_nans: True` is
  set to mask them out.
- `single_2.0_rollout_esuite_2024_IMERG_combined.yaml` — **IMERG combined**.
  Keeps OPER AN `tp` as-is and *adds* IMERG `tp` as a new, separate `imerg`
  channel (renamed rather than selected, so it's appended, no reorder needed).
  Key differences from the base:
  - `data.datasets.data.target: [imerg]` — `imerg` is declared as a
    target-only variable (ground truth for the loss), not part of the
    forecast state, and is deliberately **not** imputed (so raw NaNs reach the
    loss where `ignore_nans` masks them).
  - Normalizer remaps `imerg` to `tp`'s stats and standardizes it like `tp`.
  - Validation is restricted (`limit_batches.validation: 20`) to a 2-day tail
    slice (2024-06-01 → 2024-07-30) because IMERG's zarr ends 2024-07-31, and
    validation must still run every epoch for `on_validation_end` to fire
    (that's the only hook that saves `last.ckpt` in this anemoi-training
    version).
  - Uses a `CombinedLoss` with three components:
    1. main MSE over all variables (weight `1.0`), including the untouched
       OPER AN `tp`,
    2. MSE of predicted `tp` vs. target `imerg` (weight `0.01`) — trains `tp`
       against the IMERG ground truth,
    3. monitoring-only MSE of predicted `tp` vs. target `tp`/OPER AN (weight
       `0.0` — logged, never backpropagated).

Both IMERG variants build on top of the same base defaults list (`data:
zarr_single2.0`, `dataloader: native_grid_single2.0_oper_2024`, `diagnostics:
evaluation`, `system: slurm_2.0`, `graph: encoder_decoder_only_single2.0`,
`model: transformer_single2.0`, `task: forecaster`, `training:
default_single2.0`) and only override what's needed for the IMERG join.

## What this means for work in this repo (anemoi-core)

When implementing or debugging features in `anemoi-core` (graphs / models /
training) that relate to this IMERG work, always cross-check behaviour
against these three configs first — they define the actual dataset joins,
target/prognostic variable handling, normalizer remaps, NaN-masking, and
combined-loss setup this work depends on. Don't assume defaults from this
repo's own example configs match; the base configs above are ground truth.

## Project context / roadmap

[PROJECT.MD](PROJECT.MD) is a copy of `/home/ecm1715/prototype/aifs-dev/IMERG.md`
— the analysis and roadmap for blending OPER AN and IMERG `tp` as training
targets (why the combined run lands "in the middle," and the proposed
latitude-weighted blending approach). Read it for the *why* behind the combined
config's loss weights and structure. Note it uses "ERA5" loosely to refer to
the OPER AN dataset, not the ERA5 reanalysis product.

If anything found during this work contradicts PROJECT.MD, update PROJECT.MD
to match reality.
