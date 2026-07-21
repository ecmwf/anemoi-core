# Parametrisation refactor — decisions & rationale

This records the design conversation behind replacing Hydra `instantiate` with an abstract
**`Parametrisation`** threaded through model and graph constructors. It is the follow-up to
`agent/no-pickle-plan.md` (which introduced a backend-switchable `instantiate` shim); that
shim is now **removed**.

> **Update (round 2).** Following further direction in `refactor.md`, the ABC and its
> concrete impl moved to **`anemoi.utils.parametrisation`** (from `anemoi.graphs`); the
> concrete impl is renamed `DictParametrisation` (was `JSONParametrisation`); the free
> `build()` function is **gone** — all construction is `params.create_module(...)`, including
> in graphs; the interface constructor keyword is `params=` (was `config=`); and training
> gets a `TrainingParametrisation(DictParametrisation)` subclass built via `.from_config`.
> The sections below are updated to the final state; superseded choices are noted inline.

## Goal (from `refactor.md`)

- Get rid of `instantiate`. Remove Hydra *for now* (to be reintroduced later behind
  `create_module`).
- Pass an abstract config object to **all** model/graph constructors:
  `def __init__(self, params: Parametrisation, ...)`.
- Let end-users build models/graphs directly in Python, injecting sub-modules:
  `model = MyModel(params, some_layer=MyLayer(params, ...))`.
- The object is an ABC (different subclass for training vs inference) and must be
  JSON-serialisable. In training it is built from the dataset; in inference from the JSON of
  a serialised training parametrisation.
- Rename the type to **`Parametrisation`** and the variables to **`params`**.

## The sub-module injection pattern

Every module that used to call `instantiate` on a child now takes that child as a
constructor argument (default `None`) and dispatches:

```python
class MyClass(nn.Module):
    def __init__(self, params, my_sub_module=None):
        match my_sub_module:
            case None:      self.my_sub_module = MyDefaultSubModule(params)   # default class
            case str():     self.my_sub_module = params.create_module(my_sub_module)
            case _:         self.my_sub_module = my_sub_module                # prebuilt instance
```

In `anemoi.models` this is factored into `BaseGraphModel._build_submodule(value, *,
spec_key, default=None, **runtime)`:

- `None` → `params.create_module(params.get(spec_key, default), **runtime)` — builds the
  class named in the parametrisation, injecting the model-computed runtime dims
  (`in_channels_src`, `edge_dim`, …);
- `str` → `params.create_module(value, **runtime)`;
- instance → used as-is.

`create_module` is the single choke point where construction happens; Hydra can be
reattached there later without touching call sites.

## Decisions taken (and the options rejected)

### 1. Where the ABC lives → **`anemoi.utils.parametrisation`**
Final home is `anemoi.utils`, the lowest layer shared by graphs, models *and* training.
(Round 1 first placed it in `anemoi.graphs` to avoid a cross-repo release; round 2 moved it
to `anemoi.utils` as directed, since a clone was available to iterate on.) Rejected:
- **One ABC per package (models + graphs each define their own):** avoids coupling but
  duplicates the `get`/`create_module` surface and the construction engine.

### 2. Key access convention → **flat, dotted semantic keys on a JSON dict**
`params.get("model.encoder.num_heads", default)`; missing key with no default raises
`ParametrisationError`. Rejected:
- **Scoped child parametrisations** (parent hands each sub-module a `params.child("encoder")`
  view): more machinery than needed right now.
- **Passing sub-config *trees* around** (the old Hydra style): explicitly what we are moving
  away from.
Note: structural params (num_heads, num_layers, …) are still read from the parametrisation
today. Turning them into explicit constructor args on the mapper/processor classes ("purely
flat") is a deferred follow-up — those classes are treated as leaves for now (see scope).

### 3. How aggressively to remove `instantiate` → **rip it out now, across the construction path**
The dual-backend `instantiate` shim is deleted. Rejected:
- **Keep the shim / migrate gradually:** leaves two ways to build objects and keeps the Hydra
  import reachable.

### 4. Sub-module injection shape → **class/str/instance dispatch, model injects runtime dims**
Encoder/processor/decoder need dims (`edge_dim`, `in_channels_*`) that only exist *inside*
`_build_networks` (they depend on the per-dataset graph provider). So:
- the default is the concrete class, built by the model with those dims injected;
- a user override may be a dotted-path **string** or a **prebuilt instance**.
Rejected:
- **Require users to pass fully-built encoder instances:** impossible in general, because
  `edge_dim` isn't known until the model is being built.

### 5. Concrete implementations → **`DictParametrisation` + `TrainingParametrisation`**
`anemoi.utils.parametrisation.DictParametrisation` is the dict-backed, JSON-serialisable
base (`from_json` / `from_file` / `to_json`), used for inference and tests.
`anemoi.training.parametrisation.TrainingParametrisation(DictParametrisation)` is the
training-side subclass, built via `TrainingParametrisation.from_config(omegaconf_cfg)`
(resolves interpolations; dataset-derived values can be layered on as `overrides`). The
training→model boundary in `train/methods/base.py` uses it.

### 6. No free `build()` function → **construction only via `create_module`**
The module-level `build()` is removed; the engine is private (`_construct`) and reached only
through `Parametrisation.create_module`. Everything — models *and* graphs — passes a
`Parametrisation` and calls `create_module`. In graphs this means `GraphCreator` holds a
`DictParametrisation` and threads it through `update_graph`/`register_attributes`/post-
processors (an optional `parametrisation` arg, defaulted to a stateless instance so builders
used directly in tests still work). Leaf helpers with no parametrisation in scope
(`load_layer_kernels`) use a module-level stateless `DictParametrisation`.

### 7. Interface constructor keyword → **`params=`**
`AnemoiModelInterface(params=...)` (was `config=...`); the stored attribute is `model.params`.

## Architecture

- `anemoi/utils/parametrisation.py` (in the **anemoi-utils** repo)
  - `Parametrisation` (ABC): `get`, `to_dict`, `create_module`, `resolve`.
  - `DictParametrisation`: dict-backed concrete impl (+ JSON I/O).
  - `_construct` (private): Hydra-free engine (dotted `_target_`, `_partial_`, `_recursive_`,
    `_args_`, call-time-kwargs-win), reached only via `create_module`. **No public `build`.**
  - `get_object` / `get_class` / `ParametrisationError`.
- `anemoi/training/parametrisation.py`: `TrainingParametrisation(DictParametrisation)`.
- Graphs construction path (`create.py`, node/edge builders, `post_process.py`) builds via a
  `Parametrisation.create_module`; `GraphCreator` threads its instance through the builder
  methods (post-processors still also receive the raw graph config).
- Models: `BaseGraphModel` and all subclasses take `params: Parametrisation` and build
  children via `_build_submodule` / `params.create_module`. The interface passes the
  `Parametrisation` object directly to the model class.

## Scope

**In scope (converted):** `anemoi.graphs` construction path; `anemoi.models` model tree
(base, encoder_processor_decoder, ens, hierarchical, hierarchical_autoencoder, autoencoder,
transport ×2), interface, `layers/bounding`, `layers/utils` (layer kernels),
`utils/compile`. `models/utils/instantiate.py` deleted.

**Left as leaves for now** (per `refactor.md`: "we will leave other Module e.g. FFT2D"):
mapper/processor/attention/FFT layers keep their explicit-kwarg constructors; the model
reads their structural params from the parametrisation and passes them in.

**Training (round 3 — now converted):** every `hydra.utils.instantiate` / `get_class` in
`anemoi.training` now goes through a `Parametrisation` (`create_module` / `get_class`):

* `train/methods/base.py` holds `self.parametrisation = TrainingParametrisation.from_config(config)`
  and uses it for the model, `output_mask`, optimiser and scheduler;
* stand-alone modules (`train.py`, `utils/hydra.py`, `checkpoint/pipeline.py`,
  `diagnostics/logger.py`, `diagnostics/callbacks/*`, `losses/loss.py`,
  `losses/scalers/scalers.py`) build through a module-level stateless `DictParametrisation`
  (same pattern as the graph builders / `load_layer_kernels`);
* `hydra.errors.InstantiationException` handling becomes `ParametrisationError` (two tests
  updated to match).

**Still Hydra (intentionally):** the *launcher / config-loading* framework only —
`@hydra.main`, `hydra.compose`/`initialize`, and the search-path plugin. That is config
loading, not object construction, and is out of scope for the "no instantiate" goal.

## Follow-ups

- Reintroduce Hydra *behind* `create_module` (string specs → Hydra targets).
- Optionally split `Parametrisation` into training/inference subclasses.
- Optionally push structural params down as explicit constructor args on the mapper/processor
  classes (the "purely flat semantic keys" end state).
- Serialise the `Parametrisation` JSON into the checkpoint next to a pure `state_dict`
  (the original no-pickle goal).
