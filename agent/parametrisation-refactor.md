# Parametrisation refactor — decisions & rationale

This records the design conversation behind replacing Hydra `instantiate` with an abstract
**`Parametrisation`** threaded through model and graph constructors. It is the follow-up to
`agent/no-pickle-plan.md` (which introduced a backend-switchable `instantiate` shim); that
shim is now **removed**.

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

### 1. Where the ABC lives → **`anemoi.graphs.parametrisation`**
`anemoi.models` already depends on `anemoi.graphs`, so the ABC must live at the shared
lower layer. Rejected:
- **`anemoi.utils` (external repo):** cleanest long-term home but needs a separate package
  release to iterate — too much coordination for now.
- **One ABC per package (models + graphs each define their own):** avoids coupling but
  duplicates the `get`/`create_module` surface and the build engine.

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

### 5. Concrete implementation → **`JSONParametrisation`**
A dict-backed, JSON-serialisable `Parametrisation` with `from_json` / `from_file` /
`to_json`. Serves both training (built from the resolved OmegaConf config via
`OmegaConf.to_container(cfg, resolve=True)`) and inference (rebuilt from checkpoint JSON).
A dedicated training-vs-inference subclass split can come later; one impl suffices now.

## Architecture

- `anemoi/graphs/parametrisation.py`
  - `Parametrisation` (ABC): `get`, `to_dict`, `create_module`, `resolve`.
  - `JSONParametrisation`: dict-backed concrete impl (+ JSON I/O).
  - `build(spec, *args, **kwargs)`: Hydra-free engine (dotted `_target_`, `_partial_`,
    `_recursive_`, `_args_`, call-time-kwargs-win). `create_module` delegates to it.
  - `get_object` / `get_class` / `ParametrisationError`.
- Graphs construction path (`create.py`, node/edge builders, `post_process.py`) uses the
  module-level `build` (behaviour-preserving; post-processors still receive the raw graph
  config), so Hydra is gone from graph creation.
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

**Out of scope:** `anemoi.training` keeps Hydra as its launcher/config backbone. The
training→model boundary wraps the resolved OmegaConf config into a `JSONParametrisation`
(`train/methods/base.py`). Training's own `instantiate` calls (optimiser, scheduler, losses,
scalers, callbacks, output_mask) are untouched.

## Follow-ups

- Reintroduce Hydra *behind* `create_module` (string specs → Hydra targets).
- Optionally split `Parametrisation` into training/inference subclasses.
- Optionally push structural params down as explicit constructor args on the mapper/processor
  classes (the "purely flat semantic keys" end state).
- Serialise the `Parametrisation` JSON into the checkpoint next to a pure `state_dict`
  (the original no-pickle goal).
