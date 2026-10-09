# TODO list

branch: feat/richer-batch

### General

- [ ] Update unit tests
- [ ] Update integration
- [ ] Update docs
- [ ] Trajectory datasets


### Inference
- [ x ] Model interface `predict_step` with gridded datasets IN & OUT.
- [ ] Diffusion model interface `predict_step`.
- [ ] Add support for tabular datasets IN.
- [ ] Add integration test in anemoi-models

### Metadata
- [ ] Bump model metadata version
- [ ] add predic_step() signature to metadata_inference with dtypes and shapes?
- [ ] add data_type (tabular/gridded) to metadata_inferece

### Batch
- Does it make sense to have the `Batch` or can we have a `dict[str, SourceView]`?
- [ ] Add date to Sample/Sources/Templates, it should match `time_size`
- [ ] Clean and refactor all examples
- [ ] Write new explanation of Samples, Sources and Templates. Improve those connections and assumptions.

### Evaluation
- [ ] Update scalers. Use `TensorLayout` from the batch instead of the `TensorDim`.
- [ ] Validation metrics
- [ ] Callbacks
