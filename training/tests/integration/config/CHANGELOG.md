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
- Reason: Torch releases are not guaranteed to be bit-reproducible; in this case they changed slightly. No visible effect on the loss.

### 2026-09-17 — pyg-lib support ([#1407](https://github.com/ecmwf/anemoi-core/pull/1407))

- Reference run: `eddb276a2bbd40f6a856b465cc9a995d`
- Reason: Use `pyg-lib` for graphs instead of `scikit-learn`. `pyg-lib` uses a
  different algorithm to build the graph. In the case of a tie (e.g. KNN=3 with 4
  equidistant neighbours), `pyg-lib` and `scikit-learn` can select two different
  but equally correct graphs, resulting in graphs which are slightly different
  (~0.01% different nodes). Seeded training runs with both libraries have shown no
  meaningful difference in the final losses.

### 2026-07-24 — seed overflow fix ([#1263](https://github.com/ecmwf/anemoi-core/pull/1263))

- Reference run: `527c02a5f2554f1a96faa0f3e3a9f1a3`
- Reason: The seed overflow fix ([#1228](https://github.com/ecmwf/anemoi-core/pull/1228)) changed the seeding.

### 2026-07-14 — initial loss convergence test ([#1201](https://github.com/ecmwf/anemoi-core/pull/1201))

- Reference run: `684814cd86ed45f383bed3f0a87a782b`
- Reason: initial reference.

## Benchmark tests

Changes to the benchmark tests (`test_benchmark.py`) and their configurations (`benchmark/`). Also documents performance changes from underlying model optimisations.

## 2026-10-14 - Improved loss and data reader performance ([#1469](https://github.com/ecmwf/anemoi-core/pull/1469))

- Commit: 3452dec6caa6629615b17b6ca862a4766c411958
- Benchmark changed: No
- Performance change: Throughput and memory improvement for temporal downscaler model
- Reason: LossVariableMapper expensive indexing replaced with index\_select. Improved dataloading when using Trajectory datasets. Performance increased from 3.86it/s to 4.8it/s

## 2026-07-22 - Merged multiscale loss optimisation ([#1212](https://github.com/ecmwf/anemoi-core/pull/1212))


- Commit: bc89ef3c58f2646ff3385fc6881946693e6ab732
- Benchmark changed: No
- Performance change: higher throughput ensemble, diffusion and temporal downscaler models
- Reason: A PR improving the multiscale loss performance was merged. the memory increase comes from batching all the sparse matmuls into a single call in the sparse\_projector. this can be configured by the user if memory is needed. the speedups come from batching the sparse matmuls in sparse projector, precomputing the projection matrices in CSR format and caching and compiling the ensemble CRPS loss
```
# edm diffusion tendency
15:19:19 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (2.84iter/s local vs 1.47iter/s reference)
15:19:19 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (8326.93MB local vs 6686.93MB reference)

# temporal downscaler
INFO     anemoi.training.diagnostics.benchmark_server:benchmark_server.py:325 PASS. Local value for avThroughputIterPerS has improved compared to the reference value (2.84iter/s local vs 1.47iter/s reference)
```

## 2026-07-22 - Ignore warmup iterations ([#1254](https://github.com/ecmwf/anemoi-core/pull/1254))

- Commit: cfd5ca2342d1e909bf5785c39d0356b22fbf9c86
- Benchmark changed: Yes
- Performance change: Throughput improved for all models
- Reason: There was a throughput increase for all models because the throughput calculation was changed in this PR to avoid counting the first 100 warmup iterations. this was done because the initial iterations can vary dramatically and be much slower as code is being compiled and blocks of GPU memory are being allocated. This change should make the throughput results more consistent. it will also allow us to see the benefits of compilation in the benchmark tests, since the overhead of compilation won't be included in the scores.

## 2026-07-20 - Multiscale loss added to ensemble benchmark ([#1172](https://github.com/ecmwf/anemoi-core/pull/1172))

- Commit: 84b5385dc4d6b5f38457415146ed582346f54e2f
- Benchmark changed: Yes
- Performance change: Throughput and memory usage regression for ensemble model
- Reason: There was a throughput and memory regression in the ensemble tests. This is due to adding the multiscale loss to the ensemble benchmark test configuration. This makes the benchmark test workload more representative of real workloads. The multiscale loss is very expensive, hence the large throughput and memory regression. this should be improved in a follow-up PR optimising the multi-scale loss.
```
17:04:03 INFO Reference benchmark results:
--------------------
avThroughputIterPerS: 1.21iter/s (date: 2026-07-10, commit: 96d9bca43debfe9f5e0a70fe63d39bb75190bbb3)
peakMemoryMB: 10989.28MB (date: 2026-07-10, commit: 96d9bca43debfe9f5e0a70fe63d39bb75190bbb3)
--------------------

17:04:03 INFO Local benchmark results:
--------------------
avThroughputIterPerS: 0.62iter/s (date: 2026-07-10, commit: 84b5385dc4d6b5f38457415146ed582346f54e2f)
peakMemoryMB: 16335.80MB (date: 2026-07-10, commit: 84b5385dc4d6b5f38457415146ed582346f54e2f)
--------------------
```

## 2026-06-12 - Triton GT stability fix merged ([#1172](https://github.com/ecmwf/anemoi-core/pull/1172))

- Commit: bbca17482d6176f53c16adb1d268ff1874840397
- Benchmark changed: No
- Performance change: Throughput and memory usage regression for all models
- Reason: The fix for the triton GT increased the memory usage by ~100-300MB because it has to store a copy of the output tensor of the forward in fp32. additionally it reduced throughput by ~0-10% depending on the model. The benchmarking test results were manually edited to allow this test to pass.
```
# LAM
00:05:19 INFO FAIL. Local value for avThroughputIterPerS has degraded compared to the reference value (1.30iter/s local vs 1.46iter/s reference)
00:05:19 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (10891.67MB local vs 10614.55MB reference)

# Graphtransformer
00:09:53 INFO PASS. Local value for avThroughputIterPerS is within 10% tolerance of the reference value (2.13iter/s local vs 2.16iter/s reference)
00:09:53 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (6184.58MB local vs 6052.24MB reference)

# Stretched
00:17:54 INFO PASS. Local value for avThroughputIterPerS is within 10% tolerance of the reference value (1.38iter/s local vs 1.47iter/s reference)
00:17:54 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (11364.48MB local vs 11076.22MB reference)

# Ensemble
00:23:48 INFO PASS. Local value for avThroughputIterPerS is within 10% tolerance of the reference value (1.19iter/s local vs 1.19iter/s reference)
00:23:48 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (10987.28MB local vs 10724.59MB reference)

# edm_Diffusiontend
00:30:11 INFO PASS. Local value for avThroughputIterPerS is within 10% tolerance of the reference value (1.66iter/s local vs 1.73iter/s reference)
00:30:11 INFO FAIL. Local value for peakMemoryMB has degraded compared to the reference value (6539.96MB local vs 6409.62MB reference)
```

## 2026-06-03 - Graph partition refactor merged ([#1102](https://github.com/ecmwf/anemoi-core/pull/1102))

- Commit: 2d86f740724e0abbe0e22bf9ee9c276827e5593a
- Benchmark changed: No
- Performance change: Throughput and memory usage improved for all models
- Reason: Graph partition refactor was merged. Memory and performance improved across all models. The graph partition refactor removed the expensive edge → CSR preprocessing step to the triton GT forward pass.
```
# LAM
15:29:21 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (1.45iter/s local vs 1.30iter/s reference)
15:29:21 INFO PASS. Local value for peakMemoryMB has improved compared to the reference value (10614.55MB local vs 11499.01MB reference)

# Graphtransformer
15:31:16 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (2.20iter/s local vs 1.98iter/s reference)
15:31:16 INFO PASS. Local value for peakMemoryMB has improved compared to the reference value (6054.24MB local vs 6465.58MB reference)

# Stretched
15:36:01 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (1.62iter/s local vs 1.43iter/s reference)
15:36:01 INFO PASS. Local value for peakMemoryMB has improved compared to the reference value (11078.22MB local vs 11990.19MB reference)

# Ensemble
15:39:03 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (1.20iter/s local vs 1.06iter/s reference)
15:39:03 INFO PASS. Local value for peakMemoryMB has improved compared to the reference value (10724.59MB local vs 12082.84MB reference)

# Diffusiontend
15:41:17 INFO PASS. Local value for avThroughputIterPerS has improved compared to the reference value (1.74iter/s local vs 1.63iter/s reference)
15:41:17 INFO PASS. Local value for peakMemoryMB has improved compared to the reference value (6409.62MB local vs 6825.10MB reference)
```

## 2026-03-10 - Benchmark test case made smaller ([#898](https://github.com/ecmwf/anemoi-core/commit/c6d42cac6e5bbb6a146f9827dbb348771e65a429))

- Commit: c6d42cac6e5bbb6a146f9827dbb348771e65a429
- Benchmark changed: Yes
- Performance change: Throughput and memory usage improved for all models
- Reason: Benchmark test cases made smaller, from 16 to 2 layers.

## 2026-01-21 - throughput drop, not understood

- Commit: f537d7f209092215a94b9e7c613acc781ca4966b
- Benchmark changed: No
- Performance change: Throughput drop across all models
- Reason: 10% performance drop observed across all models. (e.g. LAM 1.03 it/s → 0.94). The reason for this is NOT understood. The regression occurred after multi-datasets was merged. but then subsequent testing before and after multi-datasets couldn't link the regression to this PR. The performance regression could have been caused by a change in a dependency, or some change at the system level. Dataloader benchmark tests were subsequently added to detect regressions coming from dataloading, as a safety net.

## 2025-12-17 - Ensemble model performance regression ([#758](https://github.com/ecmwf/anemoi-core/issues/758))

- Commit: 88bf1e5b59e2bd88ccee7c46125a52dbefc3ff71
- Benchmark changed: Yes
- Performance change: Ensemble model throughput decreased and memory usage decreased
- Reason: QK\_norm was added to the ensemble benchmark test (e.g. model.processor.qk\_norm: True). This test configuration is now closer to the typical ensemble model configuration.


## 2025-11-21 - Triton GT and torch.compile() merged ([#631](https://github.com/ecmwf/anemoi-core/issues/631))

- Commit: b40b6c610af47d15e8bb087c7bee79e692f7f2d7
- Benchmark changed: No
- Performance change: Increased throughput and decreased memory usage across all models
- Reason: Throughput increased and memory usage decreased for all models. This is from merging torch compile (used in conditional layer norms and pyg.conv) and from adding the triton GT. Further speedups from the triton GT expected once the conversion step is removed.

## Integration tests

Changes to the other integration tests (e.g. `test_training_cycle.py`) and their configurations.

### 2026-10-12 — changelog started (PR: TODO)

- Not the initial change: the integration tests existed and changed before this date,
  but earlier changes are not recorded here. See the git history for those.
