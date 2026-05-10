# Session Context

Update this file at the start or end of a work session so we both have the same snapshot of what matters right now.

- **Current goal**: Verification run using trained checkpoint; export plots and metrics; optional export of raw preds/targets.
- **Active branch**: e.g., `feature/high-res-dataset`.
- **Key files/dirs**:
  - `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify.yaml`
  - `training/docs/user-guide/examples/run_rrfs_verify.sh`
  - `training/src/anemoi/training/diagnostics/callbacks/evaluation.py`
  - `training/src/anemoi/training/diagnostics/callbacks/plot.py`
- **Recent decisions**:
  - Use `anemoi-training train` with the verify YAML to run validation-style rollouts from a checkpoint.
  - Verification outputs are plots + metrics (no raw arrays saved by default).
  - Plots are denormalized via post-processors and saved under `system.output.plots/plots/`.
  - Merge resolution: keep main MultiDataset datamodule, re-add `timeincrement()` and use it in `relative_date_indices()`.
  - If anemoi-models version mismatch occurs, install with `SETUPTOOLS_SCM_PRETEND_VERSION=0.9.5` to satisfy training dependency.
  - Key LAM lesson: when graph nodes are a masked subset of dataset cells (e.g., graph has `298122` nodes vs dataset `310249` cells), `FullGrid` can cause wrong spatial mapping because it slices `0:N` cells instead of using graph node indices.
  - Correct LAM setting: use `MaskedGrid` with `node_attribute_name: indices_connected_nodes` in training/verify YAMLs so dataloader cell selection follows graph indexing.
  - Symptom/diagnostic: if both target and prediction plots show the same wrong spatial pattern, suspect graph/data index mismatch (not model skill). Confirm by checking node counts and `grid_indices` target class.
  - Treat NOAA-EPIC/EAGLE as a useful workflow reference, not as the primary technical authority for this RRFS work. The most important knowledge sources remain:
    - direct Anemoi code behavior in this checkout,
    - local RRFS/Anemoi experiments and diagnostics,
    - verified NOAA/Ursa runtime behavior.
  - EAGLE is valuable mainly as an example of end-to-end organization: data creation, graph/grid assets, training, inference, pre-verification, wxvx verification, and plotting.
- **Open questions/risks**:
  - If raw predictions/targets are needed for external verification, add a custom callback to export NetCDF/Zarr.
- **Next actions**:
  - Run `training/docs/user-guide/examples/run_rrfs_verify.sh` with checkpoint + sample time range.
  - Inspect outputs under `system.output.plots/plots/`.
  - Decide whether to implement a raw pred/target export callback.
- **Pointers**: link to today's note, experiment folders, issues/PRs.
- **External reference projects**:
  - NOAA-EPIC/EAGLE:
    - Public repo: https://github.com/NOAA-EPIC/EAGLE
    - Local checkout: `C:\Users\Ting.Lei\Documents\GitHub\dr-eagle\EAGLE`
    - Current local state when reviewed: branch `main`, clean worktree, commit `30b658c` (`remove env (#145)`).
    - Public README describes EAGLE as an end-to-end ML weather pipeline built around Make targets and `uwtools` drivers:
      - environment setup,
      - Zarr data creation via `ufs2arco`,
      - Anemoi training,
      - Anemoi inference,
      - postprocessing for `wxvx`,
      - verification and plotting.
    - Useful ideas to borrow:
      - pipeline-level organization using Make targets and driver classes,
      - composed platform/experiment config,
      - explicit stages for data/training/inference/verification,
      - wxvx integration for formal verification,
      - nested global/LAM dataset and graph examples,
      - graph attributes including `cutout_mask`, `boundary_mask`, `area_weight`, and `lam_area_weight`.
    - Cautions before copying:
      - some driver commands use `shell=True` and `check=False`, which can hide failures if logs are not checked;
      - inference can choose the latest `inference-last.ckpt` by file modification time, which is convenient but risky for controlled experiments;
      - default training settings such as `max_steps: 20` are demonstration/smoke-test scale, not production settings;
      - hard-coded LAM assumptions such as `lam_index`, sampled HRRR grid shape, fixed dates, and mesh trimming must be independently verified;
      - EAGLE variable conventions differ from the RRFS/refc setup here.
    - Local files of interest:
      - `src/Makefile`: workflow targets and environment separation.
      - `src/config/base.yaml`: composed data/training/inference/vx config.
      - `src/config/ursa.yaml`: Ursa platform settings.
      - `src/eagle/training/training.py`: Anemoi config generation/provisioning.
      - `src/eagle/inference/inference.py`: checkpoint-based inference driver.
      - `src/eagle/data/grids_and_meshes.py`: nested global/CONUS grid and latent mesh preparation.
      - `src/eagle/wxvx/wxvx.py`: pre-wxvx and wxvx config/run driver.
- **Current RRFS/Anemoi technical knowledge**:
  - `refc` behavior:
    - Including `refc` as an input can encourage a persistence-like solution; removing `refc` from inputs makes initial loss larger but is scientifically cleaner if `refc` is diagnostic/output-only.
    - A checkpoint verified with a mismatched input-variable config may not fail loudly if parameter shapes still match enough through Anemoi metadata/checkpoint behavior; always inspect checkpoint metadata and model input/output variable lists.
    - No-refc-input checkpoints verified so far show `refc` can collapse toward near-constant/near-zero fields despite decreasing global loss, so conditional/storm-focused diagnostics are needed.
    - A direct +1h no-refc-input GraphTransformer experiment exists as `d-2GPU-1hr-refc_value-no_refc_input-202405.sh`; a matched GNN/interaction-network-style variant exists as `d-2GPU-1hr-refc_value-no_refc_input-gnn-202405.sh`.
  - Graph experiments:
    - Baseline RRFS graph was created from `graphs/docs/usage/yaml/rrfs-lam-graph.yaml` to `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/graphs/rrfs-3km-lam-graph-20km.pt`.
    - Baseline `LimitedAreaTriNodes resolution: 6` produced observed data->hidden cutoff radii `78.2 km` for `cutoff_factor: 0.6` and `195.5 km` for `cutoff_factor: 1.5`, implying hidden target-node reference spacing about `130 km`.
    - `finer_graph_v1` is recorded in `graphs/docs/usage/yaml/rrfs-lam-graph-finer_graph_v1.yaml`.
    - `finer_graph_v1` keeps the same graph structure but uses hidden `resolution: 10`, expected hidden spacing about `8 km`, and hidden-hidden `scale_resolutions: [8, 9, 10]`.
    - `finer_graph_v1` intentionally keeps `data->hidden cutoff_factor` and `hidden->data num_nearest_neighbours: 3` unchanged initially, so the first comparison isolates the finer hidden mesh effect.
    - The created `finer_graph_v1` graph is documented in `docs/graph-finer_graph_v1.md`; creation logs showed `cutoff_factor: 0.6 -> 4.9 km` and `cutoff_factor: 1.5 -> 12.2 km`, confirming hidden target-node spacing about `8.1 km`.
    - A direct +1h no-refc-input training variant using this graph is `d-2GPU-1hr-refc_value-no_refc_input-finer_graph_v1-202405.sh`; it uses config `anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-loss-no-refc-input-finer-graph-v1`.
    - Anemoi training rewrites graph paths ending in `.pt` to dataset-specific names, e.g. `rrfs-3km-lam-graph-finer_graph_v1.pt` -> `rrfs-3km-lam-graph-finer_graph_v1_data.pt` for dataset `data`. The finer-graph sbatch script creates this symlink before training.
  - A matching direct +1h verification path for finer_graph_v1 now exists:
      - config: `anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-loss-no-refc-input-finer-graph-v1`
      - wrapper: `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_no_refc_input_finer_graph_v1.sh`
      - sbatch: `dd-verify-1hr_leadtime-1month-1hr_refc_value_loss-no_refc_input-finer_graph_v1.sh`
    - This verification variant uses the finer_graph_v1 graph, keeps the no-refc-input role split and refc-value loss weights, and sets `training.multistep_output: 1` so only the direct +1h forecast is plotted/exported.
  - A matching direct +1h verification path for the diffusion finer-graph run now exists:
    - config: `anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-loss-no-refc-input-diffusion-finer-graph-v1`
    - wrapper: `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_no_refc_input_diffusion_finer_graph_v1.sh`
    - sbatch: `dd-verify-1hr_leadtime-1month-1hr_refc_value_loss-no_refc_input-diffusion-finer_graph_v1.sh`
    - verify root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/diffusion_finer_graph_v1_1h_refc_value_no_refc_input/`
    - this keeps the no-refc-input role split and finer_graph_v1 graph, but
      overrides to `/model: graphtransformer_diffusion` and `/training: diffusion`
      so the checkpoint architecture matches the diffusion training run.
  - Loss weighting:
    - `TargetValueRangeScaler` and `TargetTendencyRangeScaler` are local enhancements for range/tendency-dependent multiplicative weighting.
    - Range/tendency factors multiply the original configured variable weights; they should not replace them.
    - Dynamic range/tendency weights are sample/time/grid dependent, not just batch-global.
    - The loss-table diagnostic now distinguishes:
      - `scaled_loss`: configured Anemoi loss with scalers applied,
      - `raw_unscaled_loss`: same loss without configured scalers,
      - `loss_contribution_to_total`: variable-specific contribution normalized to the full training-loss scale.
    - For variable-specific rows, raw single-variable loss values are not directly comparable to the full training loss unless normalized by the number of output variables and lead/time weighting.
  - Verification:
    - Normal PlotSample output can be misleading for `refc` unless filenames and titles clearly identify included variables and reference fields.
    - For no-refc-input verification, plotting can still show `refc` as an input-reference-style panel, but titles must state it is a diagnostic reference field, not a model input.
    - The 6h verification config uses two rollout steps for a 3-output model: `rstep00 out00/out01/out02` = +1/+2/+3h, `rstep01 out00/out01/out02` = +4/+5/+6h.
    - Loss-table verification is expensive because it recomputes loss by sample/lead/variable; request much longer walltime or restrict variables, e.g. `variables: [refc]`.
    - Added a dedicated finer-graph 1h loss-table verification path:
      - YAML: `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-loss-no-refc-input-finer-graph-v1-loss-table.yaml`
      - wrapper: `training/docs/user-guide/examples/run_rrfs_verify_loss_table_1h_refc_value_no_refc_input_finer_graph_v1.sh`
      - sbatch: `dd-verify-1hr_leadtime-1month-1hr_refc_value_loss-no_refc_input-finer_graph_v1-loss_table.sh`
      - output dir: `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/loss_tables/1h_refc_value_no_refc_input_finer_graph_v1`
  - Restart runs:
    - Added a restart launcher for the finer-graph 1h GraphTransformer run:
      - `d-restart-2GPU-1hr-refc_value-no_refc_input-finer_graph_v1-202405.sh`
    - It resumes from:
      - `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/checkpoint/${RUN_ID}/last.ckpt`
    - Before use, set:
      - `RUN_ID=...`
      - `MAX_EPOCHS=...`
    - It also recreates the expected `_data.pt` symlink for the finer graph before launching training.
  - Training variants:
    - Added a 1h no-refc-input GNN + finer-graph training variant:
      - YAML: `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-loss-no-refc-input-gnn-finer-graph-v1.yaml`
      - sbatch: `d-2GPU-1hr-refc_value-no_refc_input-gnn-finer_graph_v1-202405.sh`
    - This combines:
      - model override `/model: gnn`
      - finer graph override `rrfs-3km-lam-graph-finer_graph_v1.pt`
      - dedicated output root `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/gnn_finer_graph_v1/`
    - The launcher also ensures the expected `_data.pt` symlink exists before training starts.
    - Added an initial standard-diffusion 1h no-refc-input finer-graph variant:
      - YAML: `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-loss-no-refc-input-diffusion-finer-graph-v1.yaml`
      - sbatch: `d-2GPU-1hr-refc_value-no_refc_input-diffusion-finer_graph_v1-202405.sh`
      - output root: `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/diffusion_finer_graph_v1/`
    - This uses Anemoi's built-in standard diffusion path:
      - model override `/model: graphtransformer_diffusion`
      - training override `/training: diffusion`
      - single-step rollout and `GraphDiffusionForecaster`
    - It keeps the current RRFS no-refc-input data role split and reuses the refc range-weighting scalers for the first experiment.
    - The first loss-table summaries showed a strong `sample_global_index % 4` pattern in `refc` loss. This is likely a validation dataloader ordering artifact: `MultiDataset` partitions validation samples into contiguous worker chunks and PyTorch returns worker-interleaved batches when `dataloader.num_workers.validation=4`.
    - Loss-table CSVs now include `sample_dataset_index`, `sample_worker_id`, `sample_worker_position`, `input_times`, and `target_times` so sample loss can be tied back to actual forecast times.
    - The no-refc-input loss-table wrapper scripts now force `dataloader.num_workers.validation=1`, making `sample_global_index` chronological for easier interpretation.
    - Loss-summary plots now use `target_times` directly when available: sample-sequence plots show sparse time ticks, top-sample heatmaps label samples with sample/dataset index plus first target time, and overview plots include the target-time range.
  - Ursa resources:
    - H100 nodes on `u1-h100` are x86_64 and compatible with the existing x86 conda env.
    - GH200 nodes on `u1-gh` are `aarch64`; they require a separate ARM/aarch64 conda/Anemoi/PyTorch environment.
    - Current RRFS training setup is data-parallel when `num_gpus_per_model=1`; each GPU has a full model replica. Reducing from 4 GPUs to 2 GPUs reduces global effective batch size unless `accum_grad_batches` is adjusted.
- **Known bug fixes (running record)**:
  - `anemoi-transform` CLI rename in newer versions:
    - Old command `make-regrid-matrix` is removed.
    - New command family is `make-regrid-file ...`.
    - For RRFS matrix creation use:
      - `anemoi-transform make-regrid-file mir-matrix --source-grid <src.grib2> --target-grid <target_grid.nc> --output <out.npz>`
  - `anemoi-transform make-regrid-file mir-matrix` argument bug (observed in env `anemoi-training-env-python3.12`):
    - Symptom: `AttributeError: 'Namespace' object has no attribute 'mir_kwargs'. Did you mean: 'mir_args'?`
    - Root cause: installed `make-regrid-file.py` reads `args.mir_kwargs` while parser exposes `--mir_args`.
    - Hotfix used:
      - `sed -i 's/args\.mir_kwargs/args.mir_args/g' /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/envs/anemoi-training-env-python3.12/lib/python3.12/site-packages/anemoi/transform/commands/make-regrid-file.py`
    - Note: no shell restart needed after in-place patch; next `anemoi-transform` call picks it up.
  - Custom training scaler integration (`TargetValueRangeScaler`) requires two coordinated changes:
    - Symptom: config validation fails with many `...refc_value_ranges...` schema errors plus `training_loss Extra inputs are not permitted`.
    - Root cause 1: adding a new scaler class under `training.scalers...` is not enough; the training pydantic schema must also include a matching scaler schema in `ScalerSchema`.
    - Root cause 2: loss overrides belong under `training.training_loss`, not as a top-level `training_loss:` block in the user YAML.
    - Local fix used:
      - add `TargetValueRangeScalerSchema` to `training/src/anemoi/training/schemas/training.py`
      - include it in the `ScalerSchema` union
      - move the YAML override to `training.training_loss.datasets...`
  - Passing `batch=` into updating scalers requires backward-compatible callback signatures:
    - Symptom: `TypeError: NaNMaskScaler.on_batch_start() got an unexpected keyword argument 'batch'`
    - Root cause: the new batch-aware scaler plumbing forwards `batch=` to all `BaseUpdatingScaler` callbacks, but older scalers such as `NaNMaskScaler` only accepted `(model, dataset_name)`.
    - Local fix used:
      - update `NaNMaskScaler.on_batch_start(...)` to accept `**_kwargs`
  - `TargetValueRangeScaler` batch callback bug:
    - Symptom: `UnboundLocalError: cannot access local variable 'model' where it is not associated with a value`
    - Root cause: `TargetValueRangeScaler.on_batch_start(...)` deleted `model` and then later accessed `model.n_step_input` / `model.n_step_output`.
    - Local fix used:
      - remove the stray `del model` from `training/src/anemoi/training/losses/scalers/value_range.py`
  - `TargetValueRangeScaler` step-count lookup bug:
    - Symptom: `AttributeError: 'AnemoiModelInterface' object has no attribute 'n_step_output'`
    - Root cause: the scaler tried to read `n_step_output` from `AnemoiModelInterface`, but that object only exposes `n_step_input`; the output-step count lives in `model.config.training.multistep_output` (or on the outer training task module).
    - Local fix used:
      - change `TargetValueRangeScaler` to read `n_step_output` from `model.config.training.multistep_output`
  - `TargetValueRangeScaler` enhancement:
    - Added `apply_to` option so a reference variable such as `refc` can define the thresholds while the resulting weights are applied to:
      - `"self"`: only that variable
      - `"all"`: all model output variables
      - `[var1, var2, ...]`: an explicit variable subset
    - Naming cleanup:
      - use `refc_range_weight_factors` for the scaler config entry
      - use `range_weight_factors:` for the per-bin multiplicative factors
  - `TargetTendencyRangeScaler` enhancement:
    - Added a second refc-focused scaler for persistence-like failures.
    - It computes raw `abs(target_refc - previous_refc)` for each output step and maps that tendency magnitude to multiplicative factors.
    - For multi-output forecasts, comparisons are consecutive: `X3-X2`, `X4-X3`, `X5-X4`.
    - Config entry added to the 123h-loss YAML as `refc_tendency_weight_factors`.
- **Ursa GPU inventory (from `sinfo -N -o "%P %N %G" | grep -i gpu`)**:
  - `u1-h100`: `gpu:h100:2` per node.
    - Observed nodes include `u20g01..u23g14` (and similarly listed `u2x`/`u3x` blocks), each with 2x H100.
  - `u1-gh`: `gpu:gh200:1` per node.
    - Observed nodes: `e01a01..e01a08`, each with 1x GH200.
  - `u1-mi300x`: `gpu:mi300x:8` per node.
    - Observed nodes: `e01g01..e01g03`, each with 8x MI300X.
  - `admin` partition mirrors the same hardware nodes/GRES entries.
- **New RRFS experiment variants**:
  - A single-input variant of the finer-graph 1h GraphTransformer run was created to test whether `multistep_input: 2` is overly smoothing for fast convective `refc` evolution.
  - Files:
    - `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-loss-no-refc-input-finer-graph-v1-single-input.yaml`
    - `d-2GPU-1hr-refc_value-no_refc_input-finer_graph_v1-single_input-202405.sh`
  - Output root:
    - `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/finer_graph_v1_single_input/`
  - A matching 1-hour verification path was created for the GNN + finer-graph run:
    - YAML:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-loss-no-refc-input-gnn-finer-graph-v1.yaml`
    - wrapper:
      `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_no_refc_input_gnn_finer_graph_v1.sh`
    - sbatch:
      `dd-verify-1hr_leadtime-1month-1hr_refc_value_loss-no_refc_input-gnn-finer_graph_v1.sh`
    - verify root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/gnn_finer_graph_v1_1h_refc_value_no_refc_input/`
  - A matching restart launcher was created for the GNN + finer-graph training run:
    - `d-restart-2GPU-1hr-refc_value-no_refc_input-gnn-finer_graph_v1-202405.sh`
    - It resumes from:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/gnn_finer_graph_v1/checkpoint/${RUN_ID}/last.ckpt`

When you say "use `docs/session-context.md` as our project context file," I'll read this and treat it as the source of truth for the session.

- Added a first-week/single-input/refc-input finer-graph variant that:
  - keeps only `925` and `500` hPa for non-hydrometeor pressure-level variables
  - removes hydrometeor variables `clmr/icmr/rwmr/snmr/grle` entirely
  from train/validation/test loading:
  - config:
    `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-finer-graph-v1-single-input-first-week-pl925500-reduced-vars.yaml`
  - script:
    `d-2GPU-1hr-refc_input-finer_graph_v1-single_input-first_week-pl925500-reduced_vars-202405.sh`
  - output root:
    `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/finer_graph_v1_single_input_first_week_refc_input_pl925500/`
  - key changes:
    `data.datasets.data.diagnostic: []`
    `model.num_channels: 512`
    `dataloader.training.datasets.data.end: "2024-05-09T08:00:00"`
    drop list removes `height/temp/ugrd/vgrd/sphum` at `850` and `200` hPa,
    removes all `clmr/icmr/rwmr/snmr/grle` levels (`925/850/500/200`), and
    keeps the inherited `boundary_mask` drop.
  - matching 1-hour verification path:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-input-finer-graph-v1-single-input-first-week-pl925500-reduced-vars.yaml`
    - wrapper:
      `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_input_finer_graph_v1_single_input_first_week_pl925500_reduced_vars.sh`
    - sbatch:
      `dd-verify-1hr_leadtime-1month-1hr_refc_input-finer_graph_v1-single_input-first_week-pl925500-reduced_vars.sh`
    - verify root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/finer_graph_v1_single_input_first_week_refc_input_pl925500_reduced_vars/`
  - matching reduced-variable GNN training path:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-gnn-finer-graph-v1-single-input-first-week-pl925500-reduced-vars.yaml`
    - script:
      `d-2GPU-1hr-refc_input-gnn-finer_graph_v1-single_input-first_week-pl925500-reduced_vars-202405.sh`
    - output root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/gnn_finer_graph_v1_single_input_first_week_refc_input_pl925500_reduced_vars/`
    - this keeps the same reduced variable set, `refc` input/prognostic role,
      `multistep_input: 1`, `max_epochs: 200`, `model.num_channels: 512`,
      and `finer_graph_v1`, but switches the model family to `/model: gnn`.
  - matching reduced-variable Transformer training path:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-transformer-finer-graph-v1-single-input-first-week-pl925500-reduced-vars.yaml`
    - script:
      `d-2GPU-1hr-refc_input-transformer-finer_graph_v1-single_input-first_week-pl925500-reduced_vars-202405.sh`
    - output root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/transformer_finer_graph_v1_single_input_first_week_refc_input_pl925500_reduced_vars/`
    - this keeps the same reduced variable set, `refc` input/prognostic role,
      `multistep_input: 1`, `max_epochs: 200`, `model.num_channels: 512`,
      and `finer_graph_v1`, but switches the model family to
      `/model: transformer_transformermapper` because the plain local
      `/model: transformer` path failed for this RRFS graph setup with
      `AssertionError('Edge attributes must be provided')`.
    - The local `transformer_transformermapper.yaml` required a repo-level fix
      because it used `encoder.window_size: -1` and `decoder.window_size: -1`,
      while the current schema only accepts nonnegative integers or `null`.
      Both were changed to `null`, which corresponds to the unrestricted /
      full-attention path. A previous experiment-level top-level
      `encoder:/decoder:` override was removed because Hydra/Pydantic treated
      those keys as invalid extras.
  - added a single-day Transformer reduced-variable variant for 2024-05-05:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-transformer-finer-graph-v1-single-input-day20240505-pl925500-reduced-vars.yaml`
    - script:
      `d-2GPU-1hr-refc_input-transformer-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars-202405.sh`
    - output root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/transformer_finer_graph_v1_single_input_day20240505_refc_input_pl925500_reduced_vars/`
    - this inherits the reduced-variable Transformer setup and only changes the
      training window to `2024-05-05T00:00:00` through `2024-05-05T23:00:00`.
    - matching 1-hour verification path:
      - config:
        `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-input-transformer-finer-graph-v1-single-input-day20240505-pl925500-reduced-vars.yaml`
      - wrapper:
        `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_input_transformer_finer_graph_v1_single_input_day20240505_pl925500_reduced_vars.sh`
      - sbatch:
        `dd-verify-1hr_leadtime-1month-1hr_refc_input-transformer-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars.sh`
      - verify root:
        `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/transformer_finer_graph_v1_single_input_day20240505_refc_input_pl925500_reduced_vars/`
  - added a single-day GraphTransformer reduced-variable variant for 2024-05-05:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-graphtransformer-finer-graph-v1-single-input-day20240505-pl925500-reduced-vars.yaml`
    - script:
      `d-2GPU-1hr-refc_input-graphtransformer-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars-202405.sh`
    - output root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/graphtransformer_finer_graph_v1_single_input_day20240505_refc_input_pl925500_reduced_vars/`
    - matching 1-hour verification path:
      - config:
        `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-input-graphtransformer-finer-graph-v1-single-input-day20240505-pl925500-reduced-vars.yaml`
      - wrapper:
        `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_input_graphtransformer_finer_graph_v1_single_input_day20240505_pl925500_reduced_vars.sh`
      - sbatch:
        `dd-verify-1hr_leadtime-1month-1hr_refc_input-graphtransformer-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars.sh`
      - verify root:
        `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/graphtransformer_finer_graph_v1_single_input_day20240505_refc_input_pl925500_reduced_vars/`

- 2026-05-09: diffusion 1h verify produced no plots because callbacks inferred forecast length from pl_module.rollout, but GraphDiffusionForecaster has no rollout attribute. Patched training/src/anemoi/training/diagnostics/callbacks/export_predictions.py and plot.py to infer rollout length from validation outputs[1] when rollout is absent.

- 2026-05-09: added diffusion diagnostics compatibility on feature branch by setting self.rollout = 1 in training/src/anemoi/training/train/tasks/diffusionforecaster.py so existing plot/export callbacks do not skip diffusion verification outputs.
- 2026-05-09: added a `base` single-input finer-graph GraphTransformer variant with `refc` in both input and output and all hydrometeor variables removed from both input and output:
  - training config:
    `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml`
  - training script:
    `d-2GPU-1hr-refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh`
  - experiment root:
    `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/`
  - matching 1-hour verification path:
    - config:
      `training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-base-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml`
    - wrapper:
      `training/docs/user-guide/examples/run_rrfs_verify_export_1h_refc_value_base_refc_input_no_hydrometeors_finer_graph_v1_single_input.sh`
    - sbatch:
      `dd-verify-1hr_leadtime-1month-1hr_refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input.sh`
    - verify root:
      `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/verify/`
  - retuned `base` refc weighting:
    - `general_variable.weights.refc: 20` (was 80)
    - `refc_range_weight_factors.range_weight_factors: [1, 2, 2, 4]` (was `[0.1, 10, 40, 80]`)
    - matching verification config updated so configured-loss reporting stays aligned with training
