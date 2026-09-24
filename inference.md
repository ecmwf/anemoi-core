# The model interface for anemoi-inference

**Status:** Draft

 **Description**: Main interface between `anemoi-models` and `anemoi-inference` as implemented on `feat/richer-batch`


**Version:** 21st September, 2026

## What changed

Before:

```python
model.predict_step(input: dict[str, Tensor], ...) -> dict[str, Tensor]
```

One tensor per dataset in, one tensor per dataset out.


Now:

```python
prediction = model.predict_step(
    input: dict[str, Payload],     # one payload per input dataset
    target_forcings: dict[str, Payload],    # one payload per decoded dataset
    target_template: dict[str, Payload],    # one payload per decoded dataset
    ...
) -> dict[str, Payload]
```

NOTE: Names and structure (kwargs vs args) to be discussed.

A `Payload` is a plain `dict`. The model now works on a `Batch` of `SourceView`s
(`models/src/anemoi/models/data/`) that carries coordinates, variable names, per-node
time offsets, layout and sharding next to the data, so the interface has to carry the
same things.


Why it was necessary:

| Capability | What a tensor could not say |
|---|---|
| Moving grids (multi-domain, obs, ...) | where the points are |
| Tabular observations | that there is no time axis (time is folded into the grid) |
| Decoding onto a new geometry | what the *output* grid is |
| Variable agnosticity (multi-domain) | which inde is which variable |

## The input payload

model.predict_step(input, target_forcings=target_forcings, target_template=target_template)

Per dataset, in `input`:

| Key | Required | Type | Shape | Units | Notes |
|---|---|---|---|---|---|
| `data` | yes | `torch.Tensor` | ? | model dtype | No need to add a dummy batch dimension.  |
| `latitudes`   | yes | `torch.Tensor` | `(grid, )` | **degrees** | Omit for a fixed grid and the model falls back to its graph nodes. Required for moving grids and tabular. |
| `longitudes`  | yes | `torch.Tensor` | `(grid, )` | **degrees** | Must match `latitudes` in shape. |
| `variables`   | yes | `list[str]`  | `(variables, ) | 
| `layout`      | yes | `tuple[str, ...]` | — | — | Axis names in order. Defaults below. |
| `timedeltas`  | **tabular only** | array-like | `(grid, )` | **seconds** | Per-point time offset. |
| `boundaries`  | **tabular only** | `list[(int, int)]` | one per time slot | — | `(start, stop)` into the grid axis. A tabular dataset without them raises. |
| `shard_sizes` | no | — | — | — | Only for distributed runs. |

Per dataset, in `target_forcings`:

| Key | Required | Type | Shape | Units | Notes |
|---|---|---|---|---|---|
| `data`        | yes | `torch.Tensor` | see below | model dtype | This data represent input data to the model, forcing variables defined in the target grid and timestamp.  |
| `latitudes`   | yes | `torch.Tensor` | `(grid, )` | **degrees** | Omit for a fixed grid and the model falls back to its graph nodes. Required for moving grids and tabular. |
| `longitudes`  | yes | `torch.Tensor` | `(grid, )` | **degrees** | Must match `latitudes` in shape. |
| `variables`   | yes | `list[str]`  | `(variables, ) | 
| `layout`      | yes | `tuple[str, ...]` | — | — | Axis names in order. Defaults below. |
| `timedeltas`  | **tabular only** | array-like | `(grid, )` | **seconds** | Per-point time offset. |
| `boundaries`  | **tabular only** | `list[(int, int)]` | one per time slot | — | `(start, stop)` into the grid axis. A tabular dataset without them raises. |
| `shard_sizes` | no | — | — | — | Only for distributed runs. |

Per dataset, in `target_template`:

| Key | Required | Type | Shape | Units | Notes |
|---|---|---|---|---|---|
| `latitudes`   | yes | `torch.Tensor` | `(grid, )` | **degrees** | Required for moving grids and tabular. |
| `longitudes`  | yes | `torch.Tensor` | `(grid, )` | **degrees** | Must match `latitudes` in shape. |
| `variables`   | yes | `list[str]`  | `(variables, ) | 
| `layout`      | yes | `tuple[str, ...]` | — | — | Axis names in order. Defaults below. |
| `timedeltas`  | **tabular only** | array-like | `(grid, )` | **seconds** | Per-point time offset. |
| `boundaries`  | **tabular only** | `list[(int, int)]` | one per time slot | — | `(start, stop)` into the grid axis. A tabular dataset without them raises. |
| `shard_sizes` | no | — | — | — | Only for distributed runs. |

Default layouts:

- gridded: `("time", "ensemble", "grid", "variables")`
- tabular: `("grid", "variables")`

The ensemble and batch axis are **NOT mandatory** for gridded data.

## The target payload

`target_template` is mandatory. It is the decoder conditioning, and it defines the output.

- One payload for **every** name in `model.target_datasets`. A missing one raises a
  `ValueError` naming it. Extra names are ignored.
- `data` is **NOT SUPPORTED**. 
- The target's coordinates, `boundaries`, `timedeltas` and time length become the
  **output's** geometry and time extent. This is the mechanism for decoding onto a grid
  that differs from the input.

Same key table as the input payload; same units and layout rules.



## The target forcings

- `target_forcings` holds the **forcing variables at the output valid times** — not the input times.
- Its variables are the dataset's input forcings, in index order. You can read the names
  straight off the checkpoint:
  `metadata_inference[<dataset>]["variable_types"]["forcing"]`. Nothing cross-checks the
  names, so the ordering is yours to get right.
- A decoder with no forcings takes a zero-width variable axis — `(1, 1, grid, 0)` is legal.


## The returned payload (`target_template` + new_data)

Keys are `model.target_datasets` — the decoded datasets, **not** the input dataset names. 

| Key | Type | Units | Notes |
|---|---|---|---|
| `data` | `torch.Tensor` | model dtype | Batch axis already removed. |
| `variables` | `list[str]` | — | Variables | 
| `layout` | `tuple[str, ...]` | — | Axis names, no `batch`. |
| `latitudes`, `longitudes` | `torch.Tensor` | **degrees** | Converted back from radians. Present when the batch carried coordinates. |
| `timedeltas` | `torch.Tensor` | seconds | Only for tabular data. |
| `boundaries` | `list[(int, int)]` | — | Back as int pairs. |

Not returned: `statistics`, `grid_size`, `shard_sizes` (?).

## Examples

### Gridded, one dataset, 2 input steps → 1 output step

```python
result = model.predict_step(
    {
        "era5": {
            "data": x,                       # (2, 1, N, V_in)  time, ensemble, grid, vars
            "latitudes": lats,               # (N,) degrees
            "longitudes": lons,              # (N,) degrees
            "layout": ("time", "ensemble", "grid", "variables"),
        }
    },
    target={
        "era5": {
            "data": forcings,                # (1, 1, N, V_forcing) at the OUTPUT valid time
            "latitudes": lats,
            "longitudes": lons,
            "layout": ("time", "ensemble", "grid", "variables"),
        }
    },
)

out = result["era5"]
out["data"]       # (1, 1, N, V_out)
out["variables"]  # model output names, in tensor column order
out["layout"]     # ("time", "ensemble", "grid", "variables")
out["latitudes"]  # (N,) degrees
```

For a fixed grid you may drop `latitudes`/`longitudes` entirely and let the model use its
graph nodes.

### Tabular / observations

```python
model.predict_step(
    {
        "obs": {
            "data": y,                       # (M, V_in)  grid, vars — time folded into grid
            "latitudes": obs_lats,           # (M,) degrees
            "longitudes": obs_lons,          # (M,) degrees
            "timedeltas": offsets,           # (M,) seconds
            "boundaries": [(0, k), (k, M)],  # one (start, stop) per input time slot
            "layout": ("grid", "variables"),
        }
    },
    target={"obs": {...}},                   # same shape, at the output window
)
```

### Two datasets, one decoded

```python
model.predict_step(
    {"era5": {...}, "obs": {...}},   # every dataset the model encodes
    target_template={"era5": {...}},          # only model.target_datasets
)
# result.keys() == {"era5"}
```

## What to expect when it goes wrong

The interface validates at the boundary and then again during collation, so errors
surface in two places:

- `data.ndim` must equal `len(layout)`, and the variable axis must match the expected
  variable count — both asserted in `_prepare_data`.
- `latitudes` and `longitudes` must have the same shape.
- A tabular dataset without `boundaries` raises
  `ValueError("Tabular dataset ... needs boundaries!")`.
- A layout must name each tensor axis exactly once (`TensorLayout.normalized`). Passing a
  `"batch"` axis fails here, with a message that does not obviously point at the cause.
- Target and output coordinates must agree, asserted after the forward pass.


## Open questions for the team
- numpy or torch? `data`/
`latitudes`/`longitudes`/`timedeltas` 
- What about the statistics?
- The checkpoint carries `is_static_grid`, `is_tabular`, `grid_size`, `data_indices`,
  `variable_types`, `shapes` and `timesteps` under `metadata_inference[<dataset>]`, but
  nothing about how to build `boundaries` or `timedeltas` for a tabular dataset. What
  needs adding?


## Pointers

| What | Where |
|---|---|
| The interface | `models/src/anemoi/models/interface/__init__.py` |
| `Batch`, `SourceView`, `TensorLayout` | `models/src/anemoi/models/data/` |
| Model-side `predict_step` | `models/src/anemoi/models/models/base.py` |
| Output assembly / coordinate assert | `models/src/anemoi/models/models/encoder_processor_decoder.py` |
| Checkpoint metadata | `training/src/anemoi/training/data/datamodule.py` (`fill_metadata`) |
| Worked end-to-end test | `models/tests/models/test_moving_grid.py::test_inference_forcing_only_target_preserves_output_metadata` |
