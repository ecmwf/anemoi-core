# TODO list

branch: feat/richer-batch

### General
- [ ] Support model parallel (where should sharding logic live?)
- [ ] Updates tests
- [ ] Update docs
- [ ] Update schemas
- [ ] Update ensemble
- [ ] Update diffusion/transport

### Normalizer
- remap ???
- date_index ???
- implement `.clone()`


### Scalers
- Scalers over time dimension not implemented for `layout.time_in_grid = True` (TabularDatasets)


### Batch
- Does it make sense to have the `Batch` or can we have a `dict[str, SourceView]`?
- [ ] Add more information like `ds.metadata` (as variable_metadata ??), `ds.statistics`, ...
- [ ] Split current `Batch` into `data` + `spec`. The motivation is to pass the target spec to the `model.forward()`. Another alternative would be implement an `empty()` to return a batch without the data.
- [ ] We currently have `Batch.apply(func)` as a batch method, we would like something similar to use with 2 batches. Motivation: loss function -> `loss(y, y_pred)`.
- [ ] Introduce `SingletonSourceView` to avoid `torch.cat` operation with lists of one dataset.

### Evaluation
- [ ] Update scalers. Use `TensorLayout` from the batch instead of the `TensorDim`.
- [ ] Validation metrics
- [ ] Callbacks

### Naming
- Alternatives to `Batch`.
- Alternatives to `SourceView`. Alternatives to `flatten_data_2d()`, `flatten_coords_2d()`, ...
