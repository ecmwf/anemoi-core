# Golden metadata fixtures

One trimmed checkpoint metadata dict per format in use. Tests load them through
`tests/unit/conftest.py`: the `golden_metadata` fixture runs a test once per file and yields
`(format_name, dict)`; `load_golden(name)` / the `golden_loader` fixture load one file by name.
The format name is the file stem.

Format ids (F0 to F8) are those of the adapter plan (`.ecmwf-multi-agents/plans/metadata-adapters.md`).
"Real" means the content is a real metadata file, trimmed and scrubbed; "synthetic" means keys or values were
added, removed or rewritten by `build_fixtures.py` for a format no real file is available for. F9 (written by this
package, `version` "3.1") has no file: tests build it from `f8_richer_batch.json` plus `component_versions`.

Real checkpoint metadata was read from the zip entry (`anemoi.json` or `ai-models.json`) only; no weights or arrays
were read or copied.

| File | Format | `version` | Source | Date | Real? | KB |
|---|---|---|---|---|---|---|
| `f0_ai_models.json` | F0 | absent | inference `tests/unit/checkpoints/simple.yaml` | n/a (hand-made) | synthetic | 1 |
| `f0_no_variables_metadata.json` | F0 | absent | as above | n/a | synthetic | 1 |
| `f1_single_dataset.json` | F1 | `"1.0"` | `aifs-subs-crps-0.1.ckpt` (public) | 2026-02-28 | real | 69 |
| `f1_ensemble.json` | F1 | `"1.0"` | `aifs-ens-crps-2.0.ckpt` (public) | 2025-12-10 | real | 55 |
| `f1_null_diagnostic.json` | F1 | `"1.0"` | inference `tests/unit/checkpoints/ocean.json` | n/a | partly synthetic | 6 |
| `f2_early.json` | F2 early (f537d7f20) | `"2.0"` | inference `global-multi-steps-2in-2out/metadata.json` | 2026-02-27 | synthetic | 18 |
| `f2_dev_scalar_index.json` | F2 dev output | `"2.0"` | inference `tests/unit/checkpoints/multi-single.json` | 2025-12-17 | real | 35 |
| `f3_single_dataset.json` | F3 | `"2.0"` | inference `global-multi-steps-2in-2out/metadata.json` | 2026-02-27 | real | 18 |
| `f3_multi_dataset.json` | F3 | `"2.0"` | inference `era5-cerra/metadata.json` | 2026-03-31 | real | 28 |
| `f3_temporal_downscaler.json` | F3 | `"2.0"` | inference `temporal-downscaler/metadata.json` | 2026-02-11 | real | 18 |
| `f4_task_top_level.json` | F4 | `"2.0"` | anemoi-training 127c10618 local run, `test_metadata` (`real-f4/inference-last.ckpt.json`) | 2026-11-23 (run of a 2026-10-06 commit) | real | 39 |
| `f5_offsets.json` | F5 | `"2.0"` | inference `rmi-lam/metadata.json` | 2026-10-21 | real | 32 |
| `f6_main.json` | F6 | `"2.0"` | anemoi-training main (before richer-batch) local run, `tests/integration` `test_metadata` (`real-f6/inference-last.ckpt.json`) | 2026-11-23 | real | 39 |
| `f6_warm_start.json` | F6 | `"2.0"` | `f6_main` plus a lineage entry | 2026-11-23 | synthetic variant | 39 |
| `f8_richer_batch.json` | F8 | `"3.0"` | anemoi-training after richer-batch (#1435), `tests/integration` `test_metadata` (`real-f8/inference-last.ckpt.json`), plus a synthetic tabular `obs` dataset | 2026-11-26 | real (gridded) + synthetic obs | 41 |

"inference" files are from anemoi-inference `main` (`tests/integration/*/metadata.json` and
`tests/unit/checkpoints/*`).

## What each file shows

- `f1_single_dataset`: flat `dataset`, `data_request` only deep in the chain (several per `datasets[]` branch),
  `version "1.0"`, `parent_uuid`, `uuid`, `model`, `tracker` (mlflow), `training`, `supporting_arrays_paths`. Richest
  F1 chain. The checkpoint is an ensemble-trained one: `config.datamodule`, `ensemble_size_per_device` 4,
  `config.data.timestep` 24h, O96 grid, 131 variables.
- `f1_ensemble`: no `parent_uuid`, no `config.datamodule`, `config.training.noise_initial_conditions` 0.0,
  `ensemble_size_per_device` 1, `timestep` 6h, N320 grid (`config.data.resolution` "n320"), 134 variables, `dataset`
  keys sorted, and a different `datasets[]` chain layout. The metadata structure is otherwise the same as
  `f1_single_dataset`; it is kept for those layout and value differences.
- `f1_null_diagnostic`: the only file with `config.data.diagnostic: null`. The source is a unit-test fixture that
  holds `config`, `data_indices` and `dataset` only (flat, no `data_request`, no `specific`). The keys the F1 trainer
  wrote (`version`, `seed`, `run_id`, `provenance_training`, `timestamp`) were added with made-up values. It is
  therefore not representative of the full F1 layout (no `uuid`, `model`, `tracker`, `training`,
  `supporting_arrays_paths`); use the other F1 files for that. Replace it when a real F1 with null diagnostic turns up.
- `f2_dev_scalar_index`: `output_relative_date_indices` is the bare int `2`, not a list.
- `f4_task_top_level`: top-level `task: "forecaster"`, `metadata_inference.task` null, `config.task` (`_target_`,
  `multistep_input`, `multistep_output`, `timestep`, `rollout`, `validation_rollout`), `tracker` is `{"null": null}`,
  `shapes` and `base_seed` present. Written by anemoi-training at commit 127c10618 (2026-10-06, the parent of
  9ced7fa32, which moved `task` into `metadata_inference`).
- `f5_offsets`: `offset-forecaster` with `input_offsets`, `output_offsets`, `rollout_shift`, `advance_map`,
  `base_seed`; `shapes.grid` is null.
- `f6_main`: what training wrote before richer-batch (#1435). Top-level `version "2.0"`, `base_seed`, `checkpoint_lineage` `[]` and no
  `parent_uuid` (fresh run), `uuid`, `model`, `tracker` `{"null": null}`, `training`, `supporting_arrays_paths`;
  `config.task` (`Forecaster`, `multistep_input` 2, `multistep_output` 1, `timestep` "6H", `rollout`,
  `validation_rollout`); `metadata_inference.task` "forecaster". `metadata_inference.<ds>.shapes` is written
  (the model's `fill_metadata` of that time). Plain forecaster, so offsets are covered by `f5_offsets` only.

## Remaining synthetic files

- `f0_ai_models`, `f0_no_variables_metadata`: no real F0 (ai-models era, no `version`) was available. Derived from the
  hand-made `simple.yaml` (`accumulated: true`, `param` + `mars`, computed-forcing flags). `dataset.shape` is removed so
  the grid size needs the O96 table. `simple.yaml` has `data_request` directly under `dataset`; no real checkpoint
  does, so it is moved into `dataset.specific.attrs.data_request`, the position the real ones use (cut down to
  the chain's shape). What the oldest ai-models checkpoints really held is unknown. The second file also has no
  `variables_metadata`. The `param`-only `variables_metadata` layout could not be derived from any real file.
- `f2_early`: `f3_single_dataset` with `metadata_inference.task`, `shapes` and the input/output relative date
  indices removed, as f537d7f20 wrote it.
- `f6_warm_start`: `f6_main` with `checkpoint_lineage` `[{"loading_strategy": "warm_start", "parent_uuid": <id>}]`
  and a scalar `parent_uuid` (inserted before `uuid`) as `training/.../checkpoint/lineage.py`
  (`record_checkpoint_lineage`) writes them. The parent id is made up. Everything else is the real F6.
- `f8_richer_batch`: the real `test_metadata` run of anemoi-training after richer-batch (#1435, 2026-11-26), trimmed
  like `f6_main`, with one synthetic dataset added. The real part (writers `data/datamodule.py:282-311`,
  `readers/gridded.py`, `train/train.py:465-492`): `version "3.0"`, no `shapes` (the models' `fill_metadata` is
  empty), no `component_versions`, and for the gridded `data` dataset `data_type "gridded"`, `grid_size` 40320,
  `data_indices`, `variable_types` and `timesteps`; `uuid`, `timestamp` and `run_id` are the real ones. The synthetic
  part: `obs` is tabular (`grid_size` null) with six made-up variables, `dataset.obs` is `{}`, `supporting_arrays_paths.obs`
  is `{}` (as `save_metadata` writes it for a dataset without arrays), the same `timesteps` as `data`, no `shapes`;
  the `config` entries for `obs` are stubs. `data_indices.obs` is built by a helper that `build_fixtures.py` checks
  against the real F5, F6 and F8 indices. No real checkpoint with a tabular dataset was available.

## What was trimmed (all files)

- `provenance_training`: stub with the keys `gather_provenance_info` writes (`time`, `python`, `module_versions`
  limited to anemoi packages, numpy and torch without local build suffixes, `distribution_names`, `git_versions`
  with zeroed hashes).
- `model`: the `torchinfo` summary string is replaced by a marker; class name and parameter counts are kept.
- `config`: only `data`, `training`, `dataloader`, `task` and `datamodule` are kept. In `data` the normaliser and
  imputer blocks go; `training` is limited to `run_id`, `seed`, `multistep_input`, `multistep_output`, `precision`,
  `rollout`, `explicit_times`, `target_forcing`, `method`, `model_task`, `deterministic`, `ensemble_size_per_device`,
  `noise_initial_conditions`; `dataloader` to `batch_size`, the partitions, `dataset`, `validation_rollout`,
  `grid_indices`. Model, graph, diagnostics, hardware and system groups are removed.
- `dataset`: every top-level key is kept in full (variables, `variables_metadata`, `shape`, `frequency`,
  `arguments`, `sources`, `supporting_arrays`, dates). In the nested `specific` / `sources` chains statistics,
  recipes, history, origins, provenance, per-level variable lists and similar bulk are removed; the chain and every
  `data_request`, `constant_fields` and `frequency` in it are kept.
- Absolute paths and file URLs become `.../<last component>`. `build_fixtures.py` refuses to write a file that still
  contains a home, scratch or file-URL prefix, an e-mail address or a user or host marker (`FORBIDDEN` in the script).
- `data_indices`, `metadata_inference`, `supporting_arrays_paths` (written by `anemoi.utils.checkpoints.save_metadata`;
  the arrays are not stored), `uuid` and `tracker` are untouched apart from path scrubbing.
- Sizes: the aim was under 30 KB. The F1 files (69, 55 KB), `f4_task_top_level` (40 KB), `f2_dev_scalar_index`,
  `f5_offsets`, `f6_*` and `f8_richer_batch` (32 to 41 KB) are larger because real `variables_metadata` (100 to 135
  variables with full `mars` blocks) and the `data_request` chains are kept whole.

## Refresh and structure check

The sources are not in this repository. `<dir>` holds anemoi-inference `main` as `inf-main/` and the metadata JSON of
three real training runs: `real-f8/inference-last.ckpt.json` (training with richer-batch, format "3.0"),
`real-f6/inference-last.ckpt.json` (training just before richer-batch) and `real-f4/inference-last.ckpt.json`
(training at commit 127c10618). Each is the metadata JSON (`anemoi-metadata/anemoi.json`) of the inference
checkpoint that the `tests/integration` `test_metadata` run produces. Without the F6 file, `f6_main` and
`f6_warm_start` are skipped; without the F8 file, `f8_richer_batch` is; without the F4 file, `f4_task_top_level` is.
Also get the two public checkpoints
(`aifs-ens-crps-2.0.ckpt`, `aifs-subs-crps-0.1.ckpt`) into one directory (or pass `--ens` and `--subs`
explicitly), then from `metadata/`:

```
uv run --script tests/data/build_fixtures.py --sources <dir> --checkpoint-dir <ckpt-dir>           # rebuild
uv run --script tests/data/build_fixtures.py --check <dir> --checkpoint-dir <ckpt-dir>             # compare only
```

Without a checkpoint the matching fixture is skipped (and reported), so the other files can still be rebuilt.

`--check` compares each real fixture with its source: key paths with dataset names normalised, dicts of more than 25
keys collapsed, `config` followed to depth 2, other keys to depth 6. It fails when a key path or a value type from
the source is missing or different in the fixture, unless trimming removes it on purpose (provenance, config groups
and keys outside the keep lists, bulk keys in the nested dataset chains). It also fails when any `data_request`,
`constant_fields` or `frequency` anywhere in `dataset` is lost, and when a kept `config` key is missing or has another
type. Synthetic files (`f0_*`, `f2_early`, `f6_warm_start`) are not checked; `f6_main` and `f8_richer_batch` (whose
added `obs` dataset is ignored) are checked against their real runs, and `f1_null_diagnostic` against its source (which has fewer keys than the fixture). After changing the trimming rules, rebuild and run the check.
