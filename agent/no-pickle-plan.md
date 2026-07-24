# Removing the Hydra dependency from model reconstruction (a.k.a. "no more pickle")

## 1. The problem

Today an Anemoi checkpoint stores the **pickled model object** next to the weights.
Inference does, in effect:

```python
model = torch.load("inference-last.ckpt")   # Case 1 — unpickles the whole nn.Module
```

Unpickling a full `nn.Module` is fragile: it ties the checkpoint to the exact module
layout, import paths and class definitions present at training time, and it drags the
training stack (Hydra, OmegaConf, the whole `anemoi.training` import graph) into the
inference process.

What we want instead is the boring, robust PyTorch idiom:

```python
model = MyModelClass(...)            # Case 2 — plain constructor
weights = torch.load("weights.pt")   # a state_dict, just tensors
model.load_state_dict(weights)
```

### Why Case 2 does not work today

The model is a tree of `torch.nn.Module`s whose `__init__` methods build their children
with **Hydra**:

```python
from hydra.utils import instantiate

class AnemoiModelEncProcDec(...):
    def __init__(self, model_config, ...):
        self.encoder = instantiate(model_config.model.encoder, _recursive_=False, ...)
        self.processor = instantiate(model_config.model.processor, _recursive_=False, ...)
        ...
```

So to *construct* the model you need:

1. a resolved config tree, and
2. Hydra's `instantiate` machinery (and therefore Hydra + OmegaConf) importable at
   inference time.

The config part is fine — we can serialise the resolved config. The hard dependency on
`hydra.utils.instantiate` scattered across ~20 call sites is the blocker.

## 2. The idea (confirmed workable)

Exactly as proposed in `no-pickle.md`:

* **Step 1 — indirection.** Replace every
  `from hydra.utils import instantiate`
  with
  `from anemoi.models.utils import instantiate`
  where our `instantiate` *forwards to Hydra by default*. This is behaviour-preserving:
  training is unaffected.

* **Step 2 — native backend.** Provide a second, pure-Python implementation of
  `instantiate` that recreates objects from a **plain resolved config dict** (loaded from
  JSON/YAML) without importing Hydra. A global switch selects the backend.

* The bulk of the new code lives in **one file**:
  `models/src/anemoi/models/utils/instantiate.py`. Everywhere else we only touch imports.

### Why this is sound

I audited every `instantiate(...)` call reachable while building a model
(`AnemoiModelInterface.__init__` → model module `__init__`s → layers). The only Hydra
features used are:

| Feature        | Where                                                        |
|----------------|--------------------------------------------------------------|
| `_target_`     | everywhere — dotted import path of the callable              |
| `_convert_`    | top-level model instantiate (`"none"` default)               |
| `_recursive_`  | almost every call passes `_recursive_=False`                 |
| `_partial_`    | `layers/utils.py::load_layer_kernels` (returns a factory)    |
| call-time kwargs | every call (e.g. `in_channels_src=...`) — override config  |
| `_args_` (positional) | **not used anywhere** — confirmed by grep             |

Recursion is mostly driven *manually* by the modules themselves (they call `instantiate`
on their own sub-configs, which is why `_recursive_=False` is everywhere). That makes a
native implementation small: it must resolve a dotted `_target_`, honour
`_partial_`/`_recursive_`/`_convert_`, merge config params with call-time kwargs, and
recurse into nested `_target_` nodes when `_recursive_` is true. No need to reimplement
Hydra's struct-config/interpolation engine — interpolations are already resolved in the
serialised config.

### Config container at inference time

Modules navigate the config by **attribute access and `.get()`**
(`model_config.model.encoder`, `cfg.get("trainable_size", 0)`). A plain `dict` does not
support attribute access. We already depend on `anemoi.utils.config.DotDict`, which:

* subclasses `dict` (so `.get(...)` works),
* exposes attribute access, and
* wraps nested dicts as `DotDict` recursively.

So the inference entry point wraps the resolved config in `DotDict` — **no OmegaConf, no
Hydra**. (OmegaConf itself is not the problem; we just don't want to *require* it for
inference. The native backend handles OmegaConf objects if present, but never imports
it.)

## 3. Design

### 3.1 New module: `anemoi/models/utils/instantiate.py`

Public surface (re-exported from `anemoi.models.utils`):

```python
instantiate(config, *args, **kwargs)          # backend-dispatching drop-in
get_class(path) / get_object(path)            # dotted-path resolvers (replace hydra.utils)
InstantiationError                            # replaces hydra.errors.InstantiationException

set_instantiation_backend("hydra" | "native") # global switch
instantiation_backend("native")               # context manager
current_backend()                             # introspection
```

#### Backend selection (the "global variable")

```
priority:  context manager  >  explicit set_instantiation_backend()  >  env var  >  default
```

* Default backend is `"hydra"` → **training and all current behaviour unchanged**.
* `ANEMOI_INSTANTIATE_BACKEND=native` env var lets inference flip it process-wide.
* If `import hydra` fails while the backend is `"hydra"`, we raise a clear error telling
  the user to switch to the native backend (so a Hydra-free install degrades loudly, not
  mysteriously).

#### `instantiate` dispatch

```python
def instantiate(config, *args, **kwargs):
    if _backend() == "native":
        return _native_instantiate(config, *args, **kwargs)
    from hydra.utils import instantiate as _hydra
    return _hydra(config, *args, **kwargs)
```

#### `_native_instantiate` semantics (matching the audited subset of Hydra)

* `config is None` → `None`.
* `config` is a list/tuple → element-wise instantiate when recursive, else returned as a
  converted list.
* `config` is a mapping **with** `_target_`:
  1. resolve target via `get_object(_target_)`;
  2. read node-level `_partial_` / `_recursive_` / `_convert_`, each overridable by the
     matching call-time keyword;
  3. positional args = `config["_args_"]` (if any) + `*args`;
  4. params = all non-`_`-prefixed config keys, then **call-time kwargs win**;
  5. if `recursive`, instantiate any param whose value is itself a `_target_` node (or a
     list of them);
  6. apply `_convert_` to leftover container values;
  7. return `functools.partial(target, *pos, **params)` if `partial`, else
     `target(*pos, **params)`.
* `config` is a mapping **without** `_target_` → treated as a config node: returned as a
  (converted) dict, recursing into nested `_target_` nodes when recursive. This mirrors
  Hydra and is what makes `_recursive_=False` pass sub-configs through untouched.
* Errors are wrapped in `InstantiationError` with the offending `_target_` for
  debuggability (parity with `load_layer_kernels`, which catches the exception).

`_convert_` handling: `"none"` leaves containers as-is; `"all"`/`"partial"`/`"object"`
coerce OmegaConf/`DotDict` containers to plain `dict`/`list`. Since the inference config
is already plain `DotDict`, this is effectively a no-op there.

### 3.2 Files that change (imports only)

Replace `from hydra.utils import instantiate` → `from anemoi.models.utils import
instantiate` in the **model-construction path** (the inference-critical set):

* `interface/__init__.py`
* `models/base.py`, `models/encoder_processor_decoder.py`,
  `models/ens_encoder_processor_decoder.py`,
  `models/transport_encoder_processor_decoder.py`,
  `models/hierarchical.py`, `models/hierarchical_autoencoder.py`
* `layers/bounding.py`, `layers/utils.py`

Also in `layers/utils.py`: replace `from hydra.errors import InstantiationException`
with our `InstantiationError` (aliased so the existing `except` keeps working).

These are the only `instantiate` calls reachable from `AnemoiModelInterface.__init__`.

### 3.3 Files intentionally left on direct Hydra imports (for now)

* `anemoi.training` (`train.py`, callbacks, losses, scalers, checkpoint/pipeline, …) —
  training-time only; not exercised during inference reconstruction.
* `anemoi.graphs` (graph/edge/node builders) — graph-build time, not model build.
* `utils/compile.py` (`hydra.utils.get_class`) — not on the construction path (a native
  `get_class` is provided so it *can* migrate later).
* tests under `*/tests/`.

They can adopt `anemoi.models.utils.instantiate` later for consistency; doing so now adds
churn without serving the inference goal. (Decision flagged for review — easy to extend
the sweep if you'd rather flip everything in step 1.)

### 3.4 Inference reconstruction (consumer side, e.g. anemoi-inference)

```python
from anemoi.models.utils import instantiation_backend
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict

cfg = DotDict(load_resolved_config(path))      # plain JSON/YAML -> DotDict

with instantiation_backend("native"):          # no hydra import happens
    model = AnemoiModelInterface(config=cfg, graph_data=graph, statistics=..., ...)

model.load_state_dict(torch.load("weights.pt"))
```

The serialised "resolved config" is the OmegaConf-resolved training config dumped to
JSON/YAML (a single object, all interpolations resolved). Producing/saving that artifact
is a small, separate change on the training/checkpoint side and is out of scope for this
first PR.

## 4. Implementation steps

1. **`utils/instantiate.py`** — native engine + dispatch + backend switch + dotted-path
   resolvers + `InstantiationError`.
2. **`utils/__init__.py`** — re-export the public surface.
3. **Swap imports** in the model-construction files (3.2).
4. **`layers/utils.py`** — swap the `InstantiationException` import/usage.
5. **Tests** — `tests/utils/test_instantiate.py`:
   * native `_target_`, nested `_target_`, `_partial_`, `_recursive_=False`,
     call-kwargs-override-config, `DotDict` input;
   * backend switch / context manager;
   * parity spot-check against Hydra for a representative config when Hydra is installed.
6. **Smoke test** — build a small `AnemoiModelInterface` under
   `instantiation_backend("native")` and confirm it matches the Hydra-built one
   parameter-for-parameter (state_dict keys/shapes).

## 5. Risks & mitigations

* **Behaviour drift from Hydra.** Mitigated by defaulting to the Hydra backend and adding
  a parity test; native path only runs when explicitly selected.
* **A construction-path `instantiate` was missed.** The native backend raises a clear
  `InstantiationError`; the smoke test builds a real model end-to-end to flush these out.
* **`_convert_` subtleties.** Only `"none"` is used on the live path and the inference
  config is already plain containers, so conversion is essentially a no-op; other modes
  implemented conservatively.
* **Hidden Hydra import elsewhere on the path** (e.g. `get_class`). Audited; `compile.py`
  is off-path. A native `get_class` is provided for a clean later migration.

## 6. Out of scope (follow-ups)

* Serialising the resolved config into the checkpoint and saving a pure `state_dict`.
* Migrating `anemoi.training` / `anemoi.graphs` call sites to the shim.
* Removing Hydra from inference install requirements (possible once the above lands).
