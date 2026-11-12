# The model interface for anemoi-inference

**Status:** Draft

**Description:** The interface between `anemoi-models` and `anemoi-inference`, as implemented on `feat/richer-batch`.

**Version:** 10 November 2026

## What changed

Before:

```python
model.predict_step(x: dict[str, Tensor], ...) -> dict[str, Tensor]
```

One tensor per dataset in, one tensor per dataset out.

Now (`AnemoiModelInterface.predict_step`):

```python
prediction = model.predict_step(
    x: dict[str, dict],                                # one payload per input dataset
    target_template: Optional[dict[str, dict]] = None, # one payload per decoded dataset; required
    target_forcing: Optional[dict[str, dict]] = None,  # one payload per decoded dataset, if it takes forcings
    model_comm_group: Optional[ProcessGroup] = None,
    gather_out: bool = True,
    **kwargs,                                          # passed on to the model, e.g. sampler settings
) -> dict[str, dict]
```

A payload is a plain `dict` of tensors, arrays, names and tuples. Nothing from `anemoi.models` crosses the
boundary in either direction: `anemoi-inference` never imports `anemoi.models`. Inside, the interface builds a
`Batch` of `Source`s (`models/src/anemoi/models/data/`). These carry coordinates, variable names, per-point
time offsets, layout and sharding next to the data, and the payloads have to carry the same things.

Why it was necessary:

| Capability | What a tensor could not say |
|---|---|
| Moving grids (multi-domain, obs, ...) | where the points are |
| Tabular observations | that there is no time axis (time is folded into the grid) |
| Decoding onto a new geometry | what the *output* grid is |
| Variable agnosticism (multi-domain) | which index is which variable |

Pre- and post-processing (normalisation, imputation, remapping, output constraints) happen inside
`predict_step`, with the statistics stored in the checkpoint. Payloads are in **physical units** in and out;
the caller passes no statistics.

## The input payload (`x`)

Per input dataset (every dataset the model encodes):

| Key | Required | Type | Shape | Units | Notes |
|---|---|---|---|---|---|
| `data_type` | yes | `str` | — | — | `"gridded"` or `"tabular"`; picks the sample kind. Read it off `metadata_inference[<dataset>]["data_type"]`. |
| `data` | yes | `torch.Tensor` | follows `layout` | physical | The model's input variables, in model-input order. No batch axis. |
| `variables` | yes | `list[str]` | `(variables,)` | — | Names of the `data` columns. See "Variable order" below. |
| `layout` | yes | `tuple[str, ...]` | — | — | Axis names of `data`, in order; see "Layouts" below. |
| `latitudes` | yes | array-like | `(grid,)` | **degrees** | No fallback to the graph nodes, even for a fixed grid. |
| `longitudes` | yes | array-like | `(grid,)` | **degrees** | Must match `latitudes` in shape. |
| `timedeltas` | **tabular only** | array-like | `(grid,)` | **seconds** | Per-point time offset. |
| `boundaries` | **tabular only** | `list[(int, int)]` | one per time slot | — | `(start, stop)` into the grid axis. |
| `grid_size` | no | `int` | — | — | Gridded only. Defaults to the number of points (or the sum of `shard_sizes`); any other value raises. |
| `shard_sizes` | no | — | — | — | Only for distributed runs. |

### Layouts

- gridded: `("time", "ensemble", "grid", "variables")`. All four axes are **required**: a gridded payload
  without an `ensemble` axis fails inside the model (an `EinopsError`). Use an ensemble axis of size 1 for a
  single member.
- tabular: `("grid", "variables")`, or `("ensemble", "grid", "variables")`. There is no `time` axis: the time
  slots are stacked along the grid axis and described by `boundaries`.
- No layout may contain a `batch` axis: the interface adds it (gridded) or holds the batch as a list
  (tabular). A `batch` axis fails at collation with a dimension-count `ValueError`.
- The axis order is free, for inputs, forcings and templates alike. `create_batched_struct` permutes `data`
  to the canonical order above (`TensorLayout.AXES`: `time`, `ensemble`, `grid`, `variables`), so the model
  only ever sees that order. The output always comes back in the canonical order, whatever order the
  template named, and its `layout` says so.
- The layout names the axes; nothing checks their **lengths** against the checkpoint. In particular the
  `time` axis must hold exactly `n_step_input` steps, i.e.
  `len(metadata_inference[<dataset>]["timesteps"]["input_relative_date_indices"])` (what `anemoi-inference`
  reads as `multi_step_input`). A tensor with the right rank but mislabelled axes passes validation and fails deep
  in the model (see "What to expect when it goes wrong").

### Variable order

The model reads the variable axis **by position**. `variables` is only used to slice the checkpoint
statistics for the pre-processors, and every name must exist in the dataset (an unknown name raises
`KeyError`). Nothing checks that the order matches the model's: a permuted `data` with correspondingly
permuted `variables` runs and gives wrong results. Take the order from the checkpoint:

- input variables: `metadata_inference[<dataset>]["data_indices"]["input"]` (name → model-input index);
- output variables: `metadata_inference[<dataset>]["data_indices"]["output"]`.

## The target template (`target_template`)

`target_template` is required (`predict_step` asserts it is given). It defines the output.

- One payload per dataset to decode. Only the datasets given here are decoded: a decoded dataset left out is
  **silently** missing from the output, and an empty dict returns an empty dict.
- Same keys as an input payload, **without `data`**. Its `variables` name the outputs, in model-output order.
  Passing `data` makes it parse as a sample, which fails with a misleading variable-count error.
- The template's coordinates, `boundaries`, `timedeltas` and layout become the **output's** geometry. This is
  the mechanism for decoding onto a grid that differs from the input. Gridded outputs have the model's
  `n_step_output` time steps; tabular outputs have one time window per `boundaries` entry.
- The ensemble and batch sizes are taken from the input.

## The target forcings (`target_forcing`)

- `target_forcing` holds the **forcing variables at the output valid times**, not at the input times. It is
  normalised with the input processors and conditions the decoder (the `target_forcings` target feature).
- Same keys as an input payload. Its variables are the dataset's input forcings, in model-input order: take
  the names in `metadata_inference[<dataset>]["variable_types"]["forcing"]`, ordered by
  `data_indices["input"]`.
- Optional. A decoder that does not use `target_forcings` (e.g. `encoded_data` only) needs none, and `None`
  is fine. A decoder with no forcing variables takes a zero-width variable axis: `(1, 1, grid, 0)` is legal.

## The returned payload

One payload per decoded dataset (the keys of `target_template` that are decoder targets):

| Key | Type | Units | Notes |
|---|---|---|---|
| `data` | `torch.Tensor` | physical | Post-processed. No batch axis. |
| `variables` | `list[str]` | — | Output names, in tensor column order. |
| `layout` | `tuple[str, ...]` | — | Axis names, no `batch`, always in canonical order: `("time", "ensemble", "grid", "variables")` for gridded, `("grid", "variables")` or `("ensemble", "grid", "variables")` for tabular. |
| `latitudes`, `longitudes` | `torch.Tensor` | **degrees** | Converted back from radians. Always present. |
| `timedeltas` | `torch.Tensor` | seconds | Tabular only. |
| `boundaries` | `list[(int, int)]` | — | Tabular only. Back as int pairs. |

Not returned: `data_type`, `statistics`, `grid_size`, `shard_sizes`. Only a batch of one sample can be
unwrapped (asserted).

## The model-side `predict_step`

`AnemoiModelInterface.predict_step` builds the `Batch` and the templates and calls the model's
`predict_step` with keyword arguments:

```python
model.predict_step(
    x=Batch,
    target_template=dict[str, BaseTemplate],
    target_forcing=Batch,
    pre_processors=...,
    post_processors=...,
    model_comm_group=...,
    gather_out=...,
    statistics_tendencies=...,  # only when the checkpoint has tendency statistics
    spatial_pre_processors=...,  # only when a dataset has a spatial processor
    **kwargs,  # from the caller
)
```

| Model | `predict_step` | Extra keyword arguments |
|---|---|---|
| Deterministic and ensemble (`AnemoiModelEncProcDec` and subclasses) | `BaseGraphModel.predict_step` (`models/base.py`): pre-process, `forward`, post-process, gather | none; remaining `kwargs` go to the pre-processors and `forward` |
| Transport, state (`AnemoiTransportModelEncProcDec`) | samples with the transport objective, then post-processes | `schedule_params`, `sampler_params` (override the checkpoint's `inference_defaults`); remaining `kwargs` go to the sampler |
| Transport, tendency (`AnemoiTransportTendModelEncProcDec`) | as above; the sampled tendencies are added to the latest input state | needs `statistics_tendencies` (passed by the interface); output constraints and remappings apply to the rebuilt states, not to the tendencies |

All return a `Batch` on the template's nodes, post-processed and, with `gather_out=True`, gathered across
`model_comm_group`.

## Examples

### Gridded, one dataset, 2 input steps → 1 output step

```python
names_in = list(md["era5"]["data_indices"]["input"])  # model-input order
names_out = list(md["era5"]["data_indices"]["output"])  # model-output order
forcings = [n for n in names_in if n in md["era5"]["variable_types"]["forcing"]]
geometry = {
    "data_type": "gridded",
    "latitudes": lats,  # (N,) degrees
    "longitudes": lons,  # (N,) degrees
    "layout": ("time", "ensemble", "grid", "variables"),
}

result = model.predict_step(
    {"era5": {**geometry, "data": x, "variables": names_in}},  # x: (2, 1, N, V_in)
    target_template={"era5": {**geometry, "variables": names_out}},
    target_forcing={"era5": {**geometry, "data": f, "variables": forcings}},  # f: (1, 1, N, V_forcing), OUTPUT time
)

out = result["era5"]
out["data"]  # (1, 1, N, V_out)
out["variables"]  # == names_out
out["layout"]  # ("time", "ensemble", "grid", "variables")
out["latitudes"]  # (N,) degrees
```

### Tabular / observations

```python
model.predict_step(
    {
        "obs": {
            "data_type": "tabular",
            "data": y,  # (M, V_in)  grid, vars; time folded into grid
            "variables": obs_names_in,
            "latitudes": obs_lats,  # (M,) degrees
            "longitudes": obs_lons,  # (M,) degrees
            "timedeltas": offsets,  # (M,) seconds
            "boundaries": [(0, k), (k, M)],  # one (start, stop) per input time slot
            "layout": ("grid", "variables"),
        }
    },
    target_template={"obs": {...}},  # the output points and window(s), no data
    target_forcing={"obs": {...}},  # forcings at the output points and times
)
# out["obs"]["data"]: (M_out, V_out), layout ("grid", "variables"), with timedeltas and boundaries
```

### Two datasets, one decoded

```python
model.predict_step(
    {"era5": {...}, "obs": {...}},  # every dataset the model encodes
    target_template={"era5": {...}},  # the datasets to decode
    target_forcing={"era5": {...}},
)
# result.keys() == {"era5"}
```

### Transport sampling settings

```python
model.predict_step(
    x,
    target_template=templates,
    target_forcing=forcings,
    schedule_params={"num_steps": 20},
    sampler_params={"sampler": "heun"},
)
```

## What to expect when it goes wrong

The payloads are validated in `create_batched_struct` (`models/src/anemoi/models/data/sample/__init__.py`),
then again at collation:

| Mistake | Error |
|---|---|
| `target_template` not given | `AssertionError("target_template must be provided for prediction.")` |
| `data_type`, `latitudes` or `longitudes` missing | `TypeError` (missing keyword argument of `create_batched_struct`) |
| `latitudes` and `longitudes` shapes differ | `ValueError("latitudes ... and longitudes ... must describe the same points.")` |
| `data.ndim != len(layout)` | `ValueError("data of shape ... does not match the layout ...")` |
| variable axis ≠ `len(variables)` | `ValueError("data carries ... variables but ... names were given.")` |
| grid axis ≠ number of coordinates | `ValueError("data carries ... points but ... coordinates were given.")` |
| unknown variable name | `KeyError(<name>)` |
| tabular without `timedeltas` or `boundaries` | `ValueError("tabular samples require timedeltas and boundaries.")` |
| `boundaries` past the last point | `ValueError("boundaries ... extend past the ... points of the sample.")` |
| gridded layout without `time` | `ValueError("GriddedSample requires a layout with a time axis ...")` |
| tabular layout with `time` or `batch` | `ValueError("TabularSample requires a layout without time and batch axes ...")` |
| gridded layout with `batch` | `ValueError` at collation: the layout describes fewer axes than the data |
| gridded layout without `ensemble` | `EinopsError` inside the model |
| `time` axis length ≠ `n_step_input` (e.g. `(1, T, grid, vars)` labelled `("time", "ensemble", "grid", "variables")`) | `RuntimeError` inside the model, e.g. `mat1 and mat2 shapes cannot be multiplied`; not checked at the boundary |
| inference checkpoint saved without the graph, before the model kept its graph geometry | `RuntimeError` naming `restore_graph_geometry(graph)`; see "The model in the checkpoint" |
| `encoded_data` target feature on a dataset whose input and target nodes differ | `AssertionError("Decoder x_dst rows ... != target node count ...")` |

Not caught: a wrong variable order (see "Variable order"), a decoded dataset missing from
`target_template`, and axis lengths that do not match the checkpoint.

## Checkpoint metadata

Under `metadata_inference`: `dataset_names` and `task`, and per dataset:

| Key | Content |
|---|---|
| `data_type` | `"gridded"` or `"tabular"` (replaces the old `is_static_grid` / `is_tabular`) |
| `grid_size` | always present: the full grid size for gridded datasets, `None` for tabular |
| `data_indices` | `{"input": name → model-input index, "output": name → model-output index}` |
| `variable_types` | `forcing`, `target`, `prognostic`, `diagnostic` name lists |
| `timesteps` | `input_relative_date_indices`, `output_relative_date_indices`, `relative_date_indices_training`, `timestep`, and for forecasters `input_offsets`, `output_offsets`, `rollout_shift`, `advance_map` |

There is no `shapes` entry any more: the number of input and output steps is the length of
`timesteps["input_relative_date_indices"]` / `["output_relative_date_indices"]`, the number of variables the
length of `data_indices["input"]` / `["output"]`, and the grid size is `grid_size`. The model's own
`fill_metadata` writes nothing; every entry comes from the data module and the task.

## The model in the checkpoint

The inference checkpoint pickles the whole `AnemoiModelInterface`, without the static graph: its
`graph_data` is `None`, and the model itself does not keep the graph. It does not need it, and the caller
passes none: the graph providers hold the edges, and `model.model.graph_geometry` holds the node and edge
counts and the coordinates of the hidden nodes and of the gridded datasets' grids (what spatial
pre-processors project onto; the pre-processors themselves stay stateless).

Inference checkpoints written between the checkpoint trim (#1456, merged into `feat/richer-batch` on
22 October 2026) and the graph geometry have neither the graph nor the geometry. `predict_step` on them
raises a `RuntimeError`. Give the model the training graph once, after loading, and they run:

```python
model = torch.load(path, map_location="cpu", weights_only=False)
# graph: the training graph file, or rebuilt with anemoi-graphs from the checkpoint's metadata["config"]["graph"]
model.model.restore_graph_geometry(graph)
```

Older checkpoints, which still carry the graph, build the geometry from it on load.

## Open questions for the team

- `anemoi-inference`'s `RicherBatchRunner` (`runners/richer_batch.py`) builds gridded payloads and templates
  but passes no `target_forcing`. Decoders that use `target_forcings` therefore cannot run through it yet.
- The checkpoint says nothing about how to build `boundaries` or `timedeltas` for a tabular dataset. What
  needs adding?
- Should the interface check `variables` against `data_indices` (order, and that every template
  dataset is decoded) instead of trusting the caller?
- numpy or torch for `latitudes`/`longitudes`/`timedeltas`? Both work today (`torch.as_tensor`); `data` should
  be a `torch.Tensor`.

## Pointers

| What | Where |
|---|---|
| The interface (`predict_step`, `_get_batch`, `_get_template`, `unwrap_batch`) | `models/src/anemoi/models/interface/__init__.py` |
| Payload validation (`create_batched_struct`), sample kinds | `models/src/anemoi/models/data/sample/` |
| `Batch`, `Source`, templates, `TensorLayout` | `models/src/anemoi/models/data/` |
| Model-side `predict_step` (deterministic, ensemble) | `models/src/anemoi/models/models/base.py` |
| Transport `predict_step`, tendency `_sampled_to_physical` | `models/src/anemoi/models/models/transport_encoder_processor_decoder.py` |
| Output assembly / target node assert | `models/src/anemoi/models/models/encoder_processor_decoder.py` |
| Checkpoint metadata | `training/src/anemoi/training/data/datamodule.py` (`fill_metadata`) and the task's `fill_metadata` (`training/src/anemoi/training/tasks/`) |
| What the model keeps of the graph (`StaticGraphGeometry`) | `models/src/anemoi/models/layers/graph.py`; `BaseGraphModel.graph_geometry` / `restore_graph_geometry` in `models/base.py` |
| End-to-end tests of the boundary | `models/tests/interface/test_predict_step.py` |
| Inference-side caller | `anemoi-inference`: `src/anemoi/inference/runners/richer_batch.py` |
