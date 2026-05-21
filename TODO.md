# TODO list
branch: feat/richer-batch

### General
- [ ] Support model parallel (where should sharding logic live?)
- [ ] Work on processors (normalizer, imputer)
    - should we pass the full batch structure to the processor?
    - stateless ??
- [ ] Updates tests
- [ ] Update docs
- [ ] Update schemas
- [ ] Update ensemble
- [ ] Update diffusion/transport

### Batch
- [ ] Add more information like `ds.metadata` (as variable_metadata ??), `ds.statistics`, ...
- [ ] Split current `Batch` into `data` + `spec`. The motivation is to pass the target spec to the `model.forward()`. Another alternative would be implement an `empty()` to return a batch without the data.
- [ ] We currently have `apply()` as a batch method, would be something similar to use with 2 batches. Motivation: loss function -> `loss(y, y_pred)`.
- [ ] Introduce `SingletonSourceView` to avoid `torch.cat` operation with lists of one dataset.

### Evaluation
- [ ] Update scalers to work over variable dims
- [ ] Validation metrics
- [ ] Callbacks

### Naming
- Alternatives to `Batch`. 
- Alternatives to `SourceView`. Alternatives to `flatten_data_2d()`, `flatten_coords_2d()`, ...
