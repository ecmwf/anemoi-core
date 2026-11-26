# TODO list

branch: feat/richer-batch

### General

- [ ] Update unit tests
- [ ] Update integration
- [ ] Update docs
- [ ] Trajectory datasets
- [ ] Ensemble model: decode through the shared `_decode_rows`, as the deterministic and transport models do.


### Inference
- [x] Model interface `predict_step` with gridded datasets IN & OUT.
- [x] Diffusion model interface `predict_step`: transport models take the interface's inputs, target templates and optional forcings, and sample onto the templates.
- [ ] Add support for tabular datasets IN (transport `predict_step` already takes them).
- [ ] Add integration test in anemoi-models
- [ ] Prediction across GPUs: split the target templates like the inputs (deterministic and transport). Until then prediction is correct on one GPU only.

### Transport
- [ ] Short training runs after the latest changes: EDM state and tendency on the grid, flow matching, EDM with observations.
- [ ] Training across 2 GPUs with observations, plus a gradient check against 1 GPU.
- [ ] Save, reload and run a transport checkpoint through the inference interface.
- [ ] Decide whether datasets that are only decoded should have their noisy target encoded.

### Metadata
- [ ] Bump model metadata version
- [ ] add predic_step() signature to metadata_inference with dtypes and shapes?
- [x] add data_type (tabular/gridded) to metadata_inferece

### Batch
- Does it make sense to have the `Batch` or can we have a `dict[str, SourceView]`?
- [ ] Add date to Sample/Sources/Templates, it should match `time_size`
- [ ] Clean and refactor all examples
- [ ] Write new explanation of Samples, Sources and Templates. Improve those connections and assumptions.

### Evaluation
- [ ] Update scalers. Use `TensorLayout` from the batch instead of the `TensorDim`.
- [ ] Validation metrics
- [ ] Callbacks
