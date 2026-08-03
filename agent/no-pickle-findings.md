# No-pickle: findings from `anemoi-inference`, `anemoi-utils` and `anemoi-training`

Notes gathered while writing `models/examples/rebuild_from_checkpoint.py`. They record
where the reconstruction inputs actually live, what the current inference path does, and
the gap that remains before inference is truly pickle-free.

## 1. The line we are replacing

`anemoi-inference/src/anemoi/inference/runner.py:314`:

```python
model = torch.load(self.checkpoint.path, map_location=self.device, weights_only=False).to(self.device)
```

This is the "Case 1" path from `no-pickle.md`: it **unpickles the whole `nn.Module`**.
`weights_only=False` is required precisely because the stored object is a live model
instance (and its graph/statistics), not just tensors. Replacing this with a
constructor + `load_state_dict` is the goal.

## 2. Where each reconstruction input lives

`AnemoiModelInterface.__init__` needs: `config`, `graph_data`, `statistics`,
`statistics_tendencies`, `data_indices`, `metadata`, `supporting_arrays`,
`n_step_input`, `n_step_output`.

| Input | Source today | Notes |
|-------|--------------|-------|
| `config` | `load_metadata(path)["config"]` (JSON) | Fully **resolved** config — all OmegaConf interpolations already substituted. Exactly what the native `instantiate` backend wants. Confirmed in `anemoi-inference/.../metadata.py:126` (`self._metadata.config`). |
| `data_indices` | metadata JSON key **and** checkpoint `hyper_parameters` | The JSON form (`metadata["data_indices"]`) is **serialised index lists**, not the `IndexCollection` objects the model expects (it accesses `data_indices.model.output.name_to_index`). The Lightning `hyper_parameters` holds the real `IndexCollection` objects. |
| `statistics` | checkpoint `hyper_parameters` / supporting arrays | Large arrays — not in the JSON metadata. `load_metadata(..., supporting_arrays=True)` returns supporting arrays separately. |
| `graph_data` | checkpoint `hyper_parameters` | A `HeteroData` object with tensors — **not** in the JSON metadata. In the current pickled model it lives inside the model object. |
| `metadata` | checkpoint `hyper_parameters` | Provenance dict. |
| `supporting_arrays` | `load_metadata(..., supporting_arrays=True)` / `hyper_parameters` | |
| `n_step_input` / `n_step_output` | derived from the `task` object | **Not stored directly.** `training/.../methods/base.py` sets them from `self.task.num_input_timesteps` / `num_output_timesteps`; the `task` object is kept in `hyper_parameters` via `save_hyperparameters()`. |

### How `hyper_parameters` gets populated

`training/src/anemoi/training/train/methods/base.py` calls `self.save_hyperparameters()`
with **no `ignore=` list**, so every `__init__` argument of the training module is stored
in `checkpoint["hyper_parameters"]` as a real Python object: `config`, `task`,
`graph_data`, `statistics`, `statistics_tendencies`, `data_indices`, `metadata`,
`supporting_arrays`. This is why the example script can source the non-config inputs from
there.

### Metadata JSON top-level keys (from `anemoi-inference` mock)

`anemoi-inference/src/anemoi/inference/testing/mock_checkpoint.py` shows the shape of the
metadata JSON: top-level keys `config`, `data_indices`, `dataset` (the latter holding
`variables`, `shape`, `variables_metadata`, `data_request`). `anemoi-inference` wraps the
whole dict in `anemoi.utils.config.DotDict` — the same container we chose for the native
instantiate path.

## 3. The caveat (important)

**The example script rebuilds the *model* without pickle, but the *auxiliary data* is
still unpickled.**

* What the no-pickle change fixes: the **model tree** (the `nn.Module` graph of encoders,
  processors, decoders, mappers, layer kernels, bounding, …) is recreated by calling the
  constructor with the resolved config, using the native `instantiate` backend. This needs
  neither the model class definitions to be import-pickle-compatible nor Hydra.

* What is still pickled in the example: `graph_data`, `statistics`, `data_indices` (and
  `task`) are read from `checkpoint["hyper_parameters"]`, which `torch.load(...,
  weights_only=False)` unpickles. These are **data** objects (a graph, arrays, index
  collections) — they do not carry the Hydra-construction problem and do not require the
  model classes — but they are still un-pickled, so the example is not yet a *fully*
  pickle-free load.

### To make inference fully pickle-free (follow-up work)

1. **Serialise the graph** separately (e.g. a `torch.save` of tensors / a graph artifact)
   so `graph_data` can be loaded with `weights_only=True` or rebuilt, not unpickled.
2. **Serialise statistics** as plain arrays (already largely available as supporting
   arrays).
3. **Rebuild `data_indices`** (`IndexCollection`) from the serialised
   `metadata["data_indices"]` lists instead of relying on the pickled objects — i.e. add a
   `from_serialised(...)` path so the JSON form is sufficient.
4. **Save a pure `state_dict`** (weights only) alongside the resolved config, so the load
   becomes: build from config (native backend) → `load_state_dict`.

`_AUX_INPUTS` in `models/examples/rebuild_from_checkpoint.py` is the single place to adapt
as these artifacts become available.

## 3a. What still gets unpickled — itemised, with proposed solutions

Loading a checkpoint with `torch.load(..., weights_only=False)` unpickles **everything**
in the file. After the model itself is rebuilt from config, the objects below are still
recreated by unpickling. Each is listed with what makes it a pickle, the risk it carries,
and a concrete way to remove the dependency.

Risk legend: 🔴 blocks a Hydra-free / class-free load, 🟡 a benign data object (no model
classes, no Hydra) but still arbitrary-code unpickling, 🟢 already trivially fixable.

### A. The model object itself — 🔴 (solved by this work)

* **What it is:** the full `AnemoiModelInterface` `nn.Module` tree.
* **Why it pickles badly:** every `__init__` calls Hydra `instantiate`; unpickling needs
  the exact class layout + import paths present at training time.
* **Solution (done):** build from the resolved `config` via the native `instantiate`
  backend, then `load_state_dict`. This is the core change.

### B. `graph_data` (`HeteroData`) — 🟡 → 🔴 in practice

* **What it is:** the graph (node/edge tensors, attributes) as a PyG `HeteroData`.
* **Why it pickles:** stored as a live object; carries `torch_geometric` (and possibly
  custom) classes, so unpickling pulls in PyG internals and is version-sensitive.
* **Proposed solutions (in order of preference):**
  1. **Save the graph as a tensor-only artifact.** Serialise the `HeteroData` to a flat
     dict of tensors (`{f"{store}.{key}": tensor}`) plus a small JSON describing node/edge
     types and shapes; reload with `torch.load(weights_only=True)` and reassemble. No
     arbitrary-code unpickling.
  2. **Reference the graph recipe, not the object.** The graph is itself built by
     `anemoi.graphs` from a config. Store that (already-resolved) graph config in the
     metadata and rebuild the graph at load time — fully pickle-free, at the cost of graph
     build time.
  3. **Interim:** keep loading it from a companion training checkpoint, but isolate it
     behind a `load_graph(checkpoint)` helper so the format can change without touching
     callers.
* **Recommendation:** (1) for inference speed and robustness; (2) is attractive if graph
  build is cheap and we want a single source of truth.

### C. `data_indices` (`IndexCollection`) — 🔴

* **What it is:** per-dataset index collections; the model accesses
  `data_indices.model.output.name_to_index` etc.
* **Why it pickles:** the rich `IndexCollection` objects are stored directly. The JSON
  metadata only has the **serialised index lists** (`metadata["data_indices"]`), which the
  model cannot consume as-is.
* **Proposed solution:** add a constructor/classmethod
  `IndexCollection.from_metadata(metadata["data_indices"], name_to_index=...)` that rebuilds
  the object graph from the serialised lists + the variable list in `metadata["dataset"]`.
  Then inference needs only the JSON, no pickle. (`anemoi-inference` already reconstructs an
  equivalent view in `metadata.py`; consolidate that logic into a shared rebuilder so model
  and inference agree.)

### D. `statistics` / `statistics_tendencies` — 🟡

* **What it is:** dicts of mean/std/min/max arrays per dataset (and per lead time for
  tendencies).
* **Why it pickles:** carried as Python dicts of NumPy arrays inside `hyper_parameters`.
* **Proposed solution:** persist as **supporting arrays** (already a first-class checkpoint
  feature: `load_metadata(..., supporting_arrays=True)` returns them) plus a JSON manifest
  mapping `dataset → statistic → array name`. Reassemble the dict from arrays at load time.
  NumPy arrays via the supporting-arrays mechanism avoid arbitrary-object unpickling.

### E. `metadata` (provenance dict) — 🟢

* **What it is:** provenance / bookkeeping dict passed to the interface.
* **Why it pickles:** read from `hyper_parameters`.
* **Proposed solution:** it is already JSON in the checkpoint metadata — pass
  `load_metadata(path)` (or the relevant sub-dict) directly instead of taking it from the
  pickled `hyper_parameters`. No new work beyond wiring.

### F. `n_step_input` / `n_step_output` — 🟢

* **What it is:** two integers.
* **Why it "pickles":** not stored directly; today derived from the unpickled `task`
  object (`task.num_input_timesteps` / `num_output_timesteps`), which drags the whole
  training `task` into the load.
* **Proposed solution:** write the two integers into the metadata JSON at save time (and/or
  store them directly in `hyper_parameters`). Then no `task` object is needed at inference.

### G. `task` object — 🔴 (avoid entirely)

* **What it is:** the training task module, present in `hyper_parameters` because
  `save_hyperparameters()` captures it.
* **Why it pickles badly:** it is a `anemoi.training` object — unpickling it imports the
  full training stack into inference, defeating the purpose.
* **Proposed solution:** never load it at inference. Removing the need for it falls out of
  fix **F** (store `n_step_*` directly). Separately, on the training side consider
  `save_hyperparameters(ignore=["task", "graph_data", ...])` and persisting those via the
  dedicated artifacts above, so they are never pickled into `hyper_parameters` in the first
  place.

### Summary table

| Artifact | Status | Proposed solution |
|----------|--------|-------------------|
| Model `nn.Module` | 🔴 → ✅ done | Build from config via native `instantiate` + `load_state_dict` |
| `graph_data` | 🟡/🔴 | Tensor-only artifact (`weights_only=True`) or rebuild from graph config |
| `data_indices` | 🔴 | `IndexCollection.from_metadata(...)` rebuilt from JSON |
| `statistics(_tendencies)` | 🟡 | Supporting arrays + JSON manifest |
| `metadata` | 🟢 | Use the JSON metadata directly |
| `n_step_input/output` | 🟢 | Store the integers in metadata JSON |
| `task` | 🔴 | Do not load; `ignore=` it at save time |

### End-state load (target)

```python
md = load_metadata(path, supporting_arrays=True)          # JSON + arrays, no pickle
config       = DotDict(md["config"])
data_indices = IndexCollection.from_metadata(md["data_indices"], md["dataset"])
statistics   = stats_from_supporting_arrays(arrays, md)
graph_data   = load_graph_artifact(path)                  # tensor-only / rebuilt from config
n_in, n_out  = md["n_step_input"], md["n_step_output"]

with instantiation_backend("native"):
    model = AnemoiModelInterface(config=config, graph_data=graph_data,
                                 statistics=statistics, data_indices=data_indices,
                                 metadata=md, n_step_input=n_in, n_step_output=n_out)

weights = torch.load(weights_path, weights_only=True)     # tensors only
model.load_state_dict(strip_state_dict_prefix(weights))
```

No `weights_only=False` anywhere — nothing is unpickled.

## 7. Step: make graph & data-index tensors torch-managed (state_dict)

Goal of this step: the graph and data-index **tensors** should be written to / loaded from
the checkpoint `state_dict` alongside the weights (torch-managed: registered as buffers),
so inference does not depend on a separately-pickled graph / `IndexCollection` object.

### 7.1 Inventory — what was already torch-managed vs not

Audited before changing anything (file:line):

**Graph tensors**

| Tensor | Where | Before | In state_dict? |
|--------|-------|--------|----------------|
| node coords `latlons_*` | `layers/graph.py` `NamedNodesAttributes.register_coordinates` | `register_buffer(persistent=True)` | ✅ yes |
| trainable node/edge tensors | `layers/graph.py` `TrainableTensor` | `nn.Parameter` | ✅ yes |
| `edge_attr` | `layers/graph_provider.py` `StaticGraphProvider` | `register_buffer(persistent=False)` | ❌ no |
| `edge_index_base` | same | `persistent=False` | ❌ no |
| `edge_inc` | same | `persistent=False` | ❌ no |
| `perm` | same | `persistent=False`, **init-only** (not used at forward) | ❌ (correctly) |

So the graph topology + edge attributes were the only graph tensors missing from the
checkpoint. They were rebuilt from `graph_data` at every `__init__`, which is why they were
not persisted.

**Data-index tensors**

* `preprocessing/normalizer.py` already registers the forward-used index tensors as
  persistent buffers: `_input_idx`, `_output_idx`, `_model_output_idx` (lines 104–127). ✅
* `preprocessing/imputer.py` used `self.data_indices.data.input.full` (a tensor on the
  plain `IndexCollection` object) directly at forward (lines 214/216). ❌
* `IndexCollection` itself (`data_indices/collection.py`, `tensor.py`) is a **plain object**,
  not an `nn.Module`. Its index tensors are `torch.int` tensors built in
  `BaseTensorIndex._build_idx_from_list` (`tensor.py:59-62`).

### 7.2 Changes made

1. **`layers/graph_provider.py` — `StaticGraphProvider`:**
   * `edge_attr`, `edge_index_base`, `edge_inc` → `persistent=True` (now in `state_dict`).
   * `perm` left `persistent=False` (construction-only).
   * Added a tolerant `_load_from_state_dict` that drops these three keys from
     `missing_keys` when absent, so **checkpoints written before this change still load
     under `strict=True`** (the buffers are always rebuilt from `graph_data` at `__init__`).
2. **`preprocessing/imputer.py` — `BaseImputer`:**
   * Registered `_data_input_full_idx` as a persistent buffer (= `data_indices.data.input.full`).
   * Forward now uses the buffer instead of `self.data_indices.data.input.full`.
   * Same tolerant `_load_from_state_dict` for the new key.

Tests added/run:
* `models/tests/layers/test_graph_provider_state_dict.py` — edge tensors present in
  `state_dict`, round-trip restores the graph, legacy checkpoint (keys deleted) still loads
  under `strict=True`. (3 passed.)
* Existing imputer + normalizer suites: 60 passed. Broader layers/models/preprocessing:
  361 passed (one unrelated pre-existing order-dependent flake in a pointwise-MLP init test,
  passes in isolation).

### 7.3 Tradeoffs / notes

* **Checkpoint size grows** by the size of the edge index + edge attributes for every static
  graph provider (encoder/processor/decoder). For large graphs this can be substantial. This
  is the deliberate trade for not pickling the graph separately. If undesirable for some
  configs, the persistence could later be gated by a flag.
* **DDP safety:** `edge_index_base` is the *full, unsharded* topology (sharding happens at
  forward in `_get_edges_impl`), identical across ranks — safe to persist.
* No migration was required thanks to the tolerant load; a migration could still be added to
  formally record the new keys if desired (the migration system lives in
  `models/src/anemoi/models/migrations/`).

### 7.4 Remaining work to fully drop the pickled `data_indices` / graph at inference

The tensors now travel in the checkpoint, but a few **non-tensor** dependencies on the
`IndexCollection` object remain at forward time and still need it (or a precomputed
substitute):

* `name_to_index` **dicts** are used at forward in `imputer.py` (413/416, and `len(...)` at
  223), `remapper.py`, and `postprocessor.py`. These are mappings, not tensors — they cannot
  be buffers. Proposed fix: precompute the needed integer indices into Python lists / int
  buffers at `__init__` (as the normalizer already does for masks), or rebuild the small
  `name_to_index` dict from the JSON metadata (`metadata["dataset"]` already carries the
  variable list and order).
* `models/transport_encoder_processor_decoder.py` uses `self.data_indices[...].data.output.full`
  (775) and other index fields; audit whether any are at forward and register as needed.
* **Constructing the model without `graph_data`:** `StaticGraphProvider.__init__` still
  asserts a valid graph (it builds the buffers from it). To construct purely from the
  checkpoint, add a path that builds empty/placeholder edge buffers of the right dtype and
  lets `load_state_dict` fill them — then `graph_data` is no longer needed at construction.

Once these are done, an inference load needs only: resolved config (native `instantiate`) +
`state_dict` (graph + indices + weights) — no pickled objects.

## 8. Step: JSON "reconstruction bundle" stored in the checkpoint metadata

This step removes the dependency on the pickled `IndexCollection` (and records the
non-tensor facts needed to construct a model) by writing a small JSON bundle into the
checkpoint's Anemoi metadata via `anemoi.utils.checkpoints`.

### 8.1 Key realisation

`IndexCollection.__init__(data_config, name_to_index)` **derives everything** (forcing /
diagnostic / prognostic / model+data index spaces / position maps). So an `IndexCollection`
is losslessly described by just `{config, name_to_index}` — both JSON-safe. No pickling.

### 8.2 What was added

* **`data_indices/collection.py`**: `IndexCollection.to_serialised()` /
  `IndexCollection.from_serialised(dict)` — minimal `{config, name_to_index}` round-trip.
* **`anemoi/models/checkpoint.py`** (new module):
  * `serialise_data_indices` / `deserialise_data_indices` — per-dataset map.
  * `serialise_graph_structure(graph_data)` — JSON summary of the graph *shape* (node
    counts, edge counts, attribute dims); tensors stay in the `state_dict`.
  * `build_reconstruction_metadata(*, data_indices, n_step_input, n_step_output, graph_data=None)`
    — assembles the bundle (`version`, `data_indices`, `n_step_input`, `n_step_output`,
    `graph`).
  * `build_reconstruction_metadata_from_interface(model_interface)` — producer-side
    convenience (the interface already holds all inputs); call at checkpoint-save time.
  * `add_reconstruction_metadata(path, bundle)` — merges the bundle under the
    `"reconstruction"` key into the checkpoint's existing Anemoi metadata using
    `anemoi.utils.checkpoints` (`load_metadata` + `replace_metadata`, or `save_metadata`
    when none exists). No pickling.
  * `load_reconstruction_metadata(path)` / `rebuild_data_indices(bundle)` — consumer side.

Note: `anemoi.utils`’ `replace_metadata` requires a top-level `"version"` key; the helper
sets/preserves one.

### 8.3 Tests

`models/tests/test_checkpoint_reconstruction.py` (6 passed):
* `IndexCollection` JSON round-trip (`==`, plus `data.input.full` / `model.output.full`
  tensor equality);
* `to_serialised` survives a `json.dumps`/`loads` round-trip;
* bundle build + `rebuild_data_indices`;
* graph structural summary (counts + attr dims, JSON-safe);
* **write into a real `torch.save` checkpoint via `anemoi.utils` and read back**;
* adding the bundle preserves existing metadata (`config`).

### 8.4 Example wired up

`models/examples/rebuild_from_checkpoint.py` now rebuilds `data_indices` and the timestep
counts **from the bundle** (`load_reconstruction_metadata` → `rebuild_data_indices`),
falling back to `hyper_parameters` only for older checkpoints. `graph_data` and
`statistics` are the last two inputs still taken from `hyper_parameters`.

### 8.5 Status toward fully pickle-free

| Input | Source now | Pickle-free? |
|-------|-----------|--------------|
| model `nn.Module` | config + native `instantiate` + `state_dict` | ✅ |
| graph **tensors** | `state_dict` (persistent buffers) | ✅ |
| graph **shape** | reconstruction bundle (`graph`) | ✅ (recorded) |
| `data_indices` | reconstruction bundle | ✅ |
| `n_step_input/output` | reconstruction bundle | ✅ |
| `metadata` | JSON metadata | ✅ |
| `statistics` | `hyper_parameters` (pickled) | ❌ remaining |
| graph object at `__init__` | still required by provider `__init__` | ❌ remaining |

Remaining to close the loop: (a) persist `statistics` as supporting arrays (or rely on the
already-persistent normalizer buffers and make construction tolerant of `statistics=None`),
and (b) a construct-without-`graph_data` path that builds placeholder buffers from the
bundle's `graph` summary and lets `load_state_dict` fill them.

## 9. Step: producer hook + placeholder construction (closing the loop)

### 9.1 (a) Producer hook — training writes the bundle

`anemoi.training.utils.checkpoint.save_inference_checkpoint` is where the inference
checkpoint + metadata is written (`torch.save(model)` + `save_metadata`). Here `model` is
the `AnemoiModelInterface` (it carries `data_indices`, `n_step_input/output`, `graph_data`).
Added `_add_reconstruction_metadata(metadata, model)` which calls
`build_reconstruction_metadata_from_interface(model)` and injects it under the
`"reconstruction"` key — best-effort (never blocks saving). So every new inference
checkpoint ships the bundle.

### 9.2 (b) Placeholder construction — build without graph / statistics

The graph tensors and the statistics-derived tensors are **all persistent buffers** in the
`state_dict`. So to construct without the pickled graph / statistics objects we only need
correctly-**shaped** placeholders; `load_state_dict` then overwrites the values.

* `build_placeholder_graph(graph_summary)` — a zero-filled `HeteroData` with the right node
  counts, node-`x` dims (these size the encoders via `NamedNodesAttributes.attr_ndims =
  2*x_dim + trainable`), edge counts and edge-attribute dims.
* `build_placeholder_statistics(data_indices)` — per-dataset neutral stats (mean=0, stdev=1,
  min=0, max=1; plus `min`/`max`/`stdev_tend` aliases) sized to `len(name_to_index)`, so the
  normalizer/bounding build valid scale/offset buffers without dividing by zero.
* `build_model_inputs(checkpoint_path)` — high-level: reads JSON metadata only and returns
  every `AnemoiModelInterface` constructor input (config, rebuilt `data_indices`, placeholder
  `graph_data`/`statistics`, `n_step_*`, `metadata`). No `torch.load`, no unpickling.

Validated by component round-trips in `models/tests/test_placeholder_reconstruction.py`
(4 passed): for `StaticGraphProvider`, `NamedNodesAttributes` and `InputNormalizer`, a module
built from a placeholder has identical buffer shapes to the real one and, after
`load_state_dict`, identical values (edge index/attr, lat/lon coords, `_norm_mul`/`_norm_add`).

### 9.3 Example: fully pickle-free inputs

`models/examples/rebuild_from_checkpoint.py` now builds **all** constructor inputs via
`build_model_inputs` (metadata only) and constructs under the native backend; weights load
from the `state_dict` (with a `weights_only=True` attempt, falling back to `False` only to
open a Lightning checkpoint, using just its `state_dict` tensors).

### 9.4 Updated status

| Input | Source now | Pickle-free? |
|-------|-----------|--------------|
| model `nn.Module` | config + native `instantiate` + `state_dict` | ✅ |
| graph tensors | `state_dict` (persistent buffers) | ✅ |
| graph shape (to construct) | bundle `graph` → placeholder | ✅ |
| `data_indices` | bundle → `IndexCollection.from_serialised` | ✅ |
| `statistics` (to construct) | placeholder sized from `data_indices`; values from `state_dict` | ✅ |
| `n_step_input/output` | bundle | ✅ |
| `metadata` | JSON metadata | ✅ |

All constructor inputs are now pickle-free. The only place a pickle is still opened is to
read the `state_dict` out of a **Lightning** checkpoint (which co-stores `hyper_parameters`);
a checkpoint that stores a bare `state_dict` (or the weights as supporting arrays) loads with
`weights_only=True` and needs no unpickling at all — the natural next packaging step.

### 9.5 Caveats

* `serialise_graph_structure` records node/edge attributes with `ndim >= 2`; coordinate `x`
  and edge attributes are captured. If a model reads *values* (not shapes) off the stored
  `_graph_data` at forward (e.g. transport's edge-count access uses only `.shape`), the
  placeholder zeros are fine — but a path that needs real graph *values* outside the provider
  buffers would need those persisted too. None on the standard enc-proc-dec path.
* `statistics_tendencies` is not yet placeholdered (separate, lead-time-keyed structure); add
  analogously if a tendency model must be constructed without it.

## 4. State-dict key prefix

A Lightning checkpoint stores the interface as `self.model` on the training module, so its
weight keys are doubly prefixed (`model.model.<...>`, `model.pre_processors.<...>`). A
standalone `AnemoiModelInterface` expects one fewer `model.` level. The example strips a
single leading `model.` prefix before `load_state_dict`.

## 5. Status of the core change (recap)

* `anemoi.models.utils.instantiate` provides the backend switch; default `"hydra"` keeps
  training byte-for-byte unchanged, `"native"` is Hydra-free. (See `no-pickle-plan.md`.)
* Only the model-construction files in the `models` package were switched to the shim.
* Native↔Hydra parity is covered by tests; the model-build path is exercised by the
  existing model tests.

## 6. Open question for review

The example pulls auxiliary inputs from a **Lightning** checkpoint's `hyper_parameters`.
The intended end-state inference checkpoint (zip with metadata JSON + weights) does not
carry those Python objects — it would need items 1–4 in §3. Decision needed: do we (a)
extend the inference checkpoint format to embed serialisable graph/statistics + a pure
state_dict, or (b) keep sourcing them from a companion training checkpoint during the
transition? The example supports (b) today and is structured to move to (a).
