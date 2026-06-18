# Pickle-free model reconstruction in anemoi-core

This document describes the work done to let an Anemoi model be **rebuilt from a checkpoint
without unpickling a Python object**. It starts with the goal and the problem it solves,
then explains the main concepts of the changes, the end-to-end flow, and what remains.

Companion documents (working notes accumulated along the way):
`no-pickle.md` (original brief), `no-pickle-plan.md` (design), `no-pickle-findings.md`
(detailed inventory + per-step notes), `no-pickle-env-findings.md` (dev setup).

---

## 1. The goal

### The problem

An Anemoi checkpoint stores the **pickled model object** next to the weights. Inference
loads it with, in effect:

```python
model = torch.load("inference.ckpt", weights_only=False)   # "Case 1"
```

Unpickling a whole `torch.nn.Module` is fragile and heavy:

- it requires the **exact class definitions and import paths** present at training time;
- it pulls the **training stack** (Hydra, OmegaConf, `anemoi.training`) into the inference
  process — the model's `__init__` methods build their children with
  `hydra.utils.instantiate`;
- `weights_only=False` means **arbitrary code runs at load time**.

### What we want instead

The boring, robust PyTorch idiom — "Case 2":

```python
model = AnemoiModelInterface(config=..., ...)   # plain constructor
model.load_state_dict(torch.load("weights.pt")) # just tensors
```

i.e. **construct the model from a (resolved) config and load tensors into it**, with no
unpickling and no Hydra at inference time.

### Why Case 2 did not work before

Two obstacles:

1. **Construction needed Hydra.** Every module built its children with
   `hydra.utils.instantiate`, so you could not rebuild the tree without importing Hydra.
2. **Construction needed pickled objects.** The constructor also needs the graph
   (`HeteroData`), per-dataset statistics, data indices (`IndexCollection`) and a couple of
   integers. These travelled as pickled Python objects, and several of the tensors they
   carry were not in the `state_dict`.

The work below removes both obstacles.

---

## 2. Main concepts of the changes

There are **four** concepts. Together they make a model reconstructable from
`config + state_dict + a small JSON bundle`, all pickle-free.

### Concept 1 — A switchable `instantiate` (remove the Hydra dependency)

We replaced the direct dependency on `hydra.utils.instantiate` with our own
`anemoi.models.utils.instantiate`, a drop-in with two **backends**:

- **`hydra`** (default): forwards to `hydra.utils.instantiate`. Training is byte-for-byte
  unchanged.
- **`native`**: a small pure-Python implementation that recreates objects from a resolved
  config dict (`_target_` / `_args_` / `_partial_` / `_recursive_` / `_convert_`,
  call-kwargs override config) **without importing Hydra**.

The active backend is the "global switch" that distinguishes training from inference:

```python
from anemoi.models.utils import instantiation_backend
with instantiation_backend("native"):
    model = AnemoiModelInterface(config=cfg, ...)   # no Hydra import happens
```

Selection priority: context manager → explicit setter → `ANEMOI_INSTANTIATE_BACKEND` env
var → default (`hydra`). Only the model-construction files in the `models` package were
switched to the shim (the inference-critical path); training/graphs keep their direct Hydra
imports.

The config container at inference time is `anemoi.utils.config.DotDict` (attribute access +
`.get()`, nested) — no OmegaConf/Hydra required.

### Concept 2 — Make graph & data-index *tensors* torch-managed

For a model to be filled from a `state_dict`, the tensors it needs must actually be **in**
the `state_dict`. We audited what was registered as a buffer and what was not, and closed
the gaps:

- **Graph edge tensors** (`edge_attr`, `edge_index_base`, `edge_inc` in
  `StaticGraphProvider`) were `persistent=False` — excluded from the `state_dict`. They are
  now `persistent=True`. (Node coordinates `latlons_*` were already persistent buffers.)
- **Imputer index tensor** (`data.input.full`) was read off the pickled `IndexCollection`
  at forward; it is now a registered buffer `_data_input_full_idx`. (The normalizer already
  registered its index tensors.)

Because these buffers are still rebuilt at `__init__`, older checkpoints lack the new keys.
A small tolerant `_load_from_state_dict` drops exactly these keys from `missing_keys`, so
**older checkpoints still load under `strict=True`** — no migration required.

### Concept 3 — A JSON "reconstruction bundle" in the checkpoint metadata

The remaining inputs are **non-tensor** facts. We serialise them to JSON and store them
inside the checkpoint's Anemoi metadata using `anemoi.utils.checkpoints` (no pickling).

Key insight: an `IndexCollection` is **fully determined by `(data_config, name_to_index)`**
— everything else is derived in `__init__`. So `data_indices` serialises losslessly to a
tiny dict. The bundle (`anemoi.models.checkpoint`) contains:

```jsonc
{
  "version": 1,
  "data_indices": { "<dataset>": { "config": {...}, "name_to_index": {...} } },
  "n_step_input": 2,
  "n_step_output": 1,
  "graph": { "nodes": {...counts/dims...}, "edges": {...counts/dims...} }
}
```

It is written under the `"reconstruction"` key via `add_reconstruction_metadata` (which uses
`anemoi.utils`' `load_metadata`/`replace_metadata`/`save_metadata`, preserving the existing
`config`).

### Concept 4 — Placeholder construction (build without the graph / statistics)

The graph tensors and statistics-derived tensors are all **persistent buffers**. So to
construct a model without the pickled graph / statistics objects we only need correctly
**shaped** placeholders; `load_state_dict` then overwrites the *values*:

- `build_placeholder_graph(graph_summary)` — a zero-filled `HeteroData` with the right node
  counts, node-`x` dims (these size the encoders), edge counts and edge-attribute dims.
- `build_placeholder_statistics(data_indices)` — neutral stats (mean 0, stdev 1, …) sized to
  `len(name_to_index)`, so the normalizer builds valid scale/offset buffers without dividing
  by zero.

`build_model_inputs(checkpoint_path)` ties it together: reads the JSON metadata only and
returns **every** constructor input (config, rebuilt `data_indices`, placeholder
`graph_data`/`statistics`, `n_step_*`, metadata) — no `torch.load`, no unpickling.

---

## 3. End-to-end flow

### Producer (training, at checkpoint save)

`anemoi.training.utils.checkpoint.save_inference_checkpoint` now injects the bundle into the
metadata before writing it (`_add_reconstruction_metadata`, best-effort — never blocks a
save). The `model` it receives is the `AnemoiModelInterface`, which already holds
`data_indices`, `n_step_*` and `graph_data`, so the bundle is built straight from it.

### Consumer (inference, pickle-free)

```python
from anemoi.models.checkpoint import build_model_inputs
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.utils import instantiation_backend
from anemoi.utils.config import DotDict

inputs = build_model_inputs(checkpoint_path)        # JSON metadata only — no unpickling
config = DotDict(inputs.pop("config"))

with instantiation_backend("native"):               # no Hydra import
    model = AnemoiModelInterface(config=config, **inputs)

model.load_state_dict(state_dict, strict=False)      # fills graph/coords/stats buffers
```

A runnable version is `models/examples/rebuild_from_checkpoint.py`.

---

## 4. Where things live

| File | Change |
|------|--------|
| `models/.../utils/instantiate.py` (new) | switchable `instantiate` (hydra/native) + backend selection + `get_object`/`get_class` + `InstantiationError` |
| `models/.../utils/__init__.py` | re-export the instantiate API |
| `models/.../interface/`, `models/.../models/*`, `models/.../layers/bounding.py`, `layers/utils.py` | import `instantiate` from the shim instead of Hydra |
| `models/.../layers/graph_provider.py` | edge tensors → persistent buffers + tolerant load |
| `models/.../preprocessing/imputer.py` | forward index tensor → registered buffer + tolerant load |
| `models/.../data_indices/collection.py` | `IndexCollection.to_serialised()` / `from_serialised()` |
| `models/.../checkpoint.py` (new) | bundle build/serialise, metadata I/O via `anemoi.utils`, placeholder builders, `build_model_inputs` |
| `training/.../utils/checkpoint.py` | `save_inference_checkpoint` injects the bundle |
| `models/examples/rebuild_from_checkpoint.py` (new) | end-to-end pickle-free rebuild example |

---

## 5. Testing

- `models/tests/utils/test_instantiate.py` — native backend semantics + Hydra parity.
- `models/tests/layers/test_graph_provider_state_dict.py` — edge tensors in `state_dict`,
  round-trip, legacy checkpoint still loads under `strict=True`.
- `models/tests/test_checkpoint_reconstruction.py` — `IndexCollection` JSON round-trip,
  bundle build, and **writing/reading the bundle in a real checkpoint via `anemoi.utils`**.
- `models/tests/test_placeholder_reconstruction.py` — `StaticGraphProvider`,
  `NamedNodesAttributes` and `InputNormalizer` built from placeholders match the real module
  in shape and, after `load_state_dict`, in value.
- Existing preprocessing/layers/models suites pass unchanged.

---

## 6. Status and what remains

**All constructor inputs are now pickle-free**: the model is built from `config` via the
native backend; `data_indices`, `n_step_*` and the graph shape come from the JSON bundle;
graph/statistics tensors come from the `state_dict`.

The one pickle still opened is reading the `state_dict` out of a **Lightning** checkpoint
(which co-stores `hyper_parameters`). The natural final packaging step is to also write a
**bare `state_dict`** (or weights as supporting arrays) so the load is `weights_only=True`
end-to-end.

Minor follow-ups (see `no-pickle-findings.md` §9.5):

- `statistics_tendencies` is not yet placeholdered (separate, lead-time-keyed structure).
- A handful of `name_to_index` *dict* lookups remain at forward in some preprocessors; these
  are non-tensor and still read off the (rebuilt) `data_indices` object — fine, since it is
  now reconstructed from the bundle rather than unpickled.
- Checkpoint size grows by the persisted edge tensors — the deliberate cost of not pickling
  the graph separately; could be gated by a flag if needed.
