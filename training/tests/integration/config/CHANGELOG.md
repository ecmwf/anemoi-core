# Integration test changelog

Changes to the integration tests and their configurations, with the reason and the PR.
Add a new entry at the top of the relevant section whenever a test or its reference changes.

## Accuracy test

Changes to the reference run (`reference_id` in `training/tests/integration/test_accuracy.py`)
or to the accuracy test configuration (`accuracy_testing/`).
A candidate reference can be checked first by dispatching the HPC integration workflow
with `accuracy_reference_id` set to its run ID, before committing it.

### 2026-10-09 — torch 2.11 ([#1455](https://github.com/ecmwf/anemoi-core/pull/1455))
- Reference run: `8b3bc3d3bd6947318cb825d777c9d750`
- Reason: torch releases are not bit-reproducible; no visible effect on the loss.

### 2026-09-17 — pyg-lib support ([#1407](https://github.com/ecmwf/anemoi-core/pull/1407))
- Reference run: `eddb276a2bbd40f6a856b465cc9a995d`
- Reason: TODO

### 2026-07-24 — seed overflow fix ([#1263](https://github.com/ecmwf/anemoi-core/pull/1263))
- Reference run: `527c02a5f2554f1a96faa0f3e3a9f1a3`
- Reason: the seed overflow fix ([#1228](https://github.com/ecmwf/anemoi-core/pull/1228)) changed the seeding.

### 2026-07-14 — initial loss convergence test ([#1201](https://github.com/ecmwf/anemoi-core/pull/1201))
- Reference run: `684814cd86ed45f383bed3f0a87a782b`
- Reason: initial reference.

## Benchmark tests

Changes to the benchmark tests (`test_benchmark.py`) and their configurations (`benchmark/`).

### 2026-10-12 — changelog started (PR: TODO)
- Not the initial change: the benchmark tests existed and changed before this date,
  but earlier changes are not recorded here. See the git history for those.

## Integration tests

Changes to the other integration tests (e.g. `test_training_cycle.py`) and their configurations.

### 2026-10-12 — changelog started (PR: TODO)
- Not the initial change: the integration tests existed and changed before this date,
  but earlier changes are not recorded here. See the git history for those.
