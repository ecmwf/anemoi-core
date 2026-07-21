---
name: parametrisation
description: How to construct anemoi models/graphs the Hydra-free way, using the Parametrisation ABC and create_module. Use when adding or editing a model, a model sub-module, a graph builder, or any code that used to call hydra.utils.instantiate in anemoi.models or anemoi.graphs.
---

# Parametrisation pattern (no Hydra, no `instantiate`, no `build`)

Object construction in `anemoi.models` and `anemoi.graphs` goes through
`anemoi.utils.parametrisation`, **not** Hydra. Do not import `hydra` or
`anemoi.models.utils.instantiate` (deleted). There is **no free `build()` function** —
construction only happens via `parametrisation.create_module(...)`.

See `agent/parametrisation-refactor.md` for the full rationale and rejected options.

## Core API (`anemoi.utils.parametrisation`)

- `Parametrisation` (ABC): `get(key, default=MISSING)` (dotted keys, raises
  `ParametrisationError` if missing and no default), `to_dict()`, and `create_module(spec, ...)`
  — the **single construction entry point**. `create_module` dispatches on `spec`:
  - a **class** → instantiated `spec(*args, **kwargs)` (spec-build directives `_recursive_`/`_partial_` are dropped);
  - a **dotted-path string** / `_target_` **mapping** / **list** → built via `_build_spec`;
  - `None` → `None`; an already-built **instance** → returned unchanged.
- `_build_spec` is the overridable spec-builder: Hydra-free (`_construct`) in
  `DictParametrisation`; `hydra.utils.instantiate` in `HydraParametrisation`.
- `DictParametrisation(dict)`: JSON-serialisable impl (`from_json`/`from_file`/`to_json`).
  Used by `anemoi.graphs`, `anemoi.models`, tests, examples and inference (Hydra-free).
- `HydraParametrisation` + `TrainingParametrisation` live in **`anemoi.training.parametrisation`**
  (Hydra is a training concern). `TrainingParametrisation.from_config(omegaconf_cfg)` is built
  by the trainer and handed to the model. `anemoi.graphs`/`anemoi.models` must NOT import them.
- No free `build()` function; no `_build_submodule` (deleted); no `resolve` (deleted).

## Writing a module that builds children

Every constructor takes `params: Parametrisation` first. A child sub-module is a keyword arg
whose **default is its class (in code)** — not `None`, not a config `_target_` lookup:

```python
from anemoi.utils.parametrisation import Parametrisation

class MyModel(BaseGraphModel):
    def __init__(self, params: Parametrisation, *, encoder=GraphTransformerForwardMapper, **kwargs):
        object.__setattr__(self, "_encoder", encoder)   # stash before nn.Module.__init__
        super().__init__(params, **kwargs)

    def _build_networks(self):                          # no args; read self.params
        self.encoder = self.params.create_module(
            self._encoder,                              # class default / str / instance
            _recursive_=False,                          # keep layer_kernels as config (spec path)
            **self._submodule_settings("encoder"),      # structural settings from params (num_heads, …)
            in_channels_src=..., edge_dim=...,          # runtime dims the model computes
        )
```

`self.params.create_module(value, **kwargs)` handles class/str/instance/spec/None directly.
When a module builds *many* children from one config spec (e.g. hierarchical down/up
processors), call `self.params.create_module(self.params.get("model.processor"),
_recursive_=False, ...)` with the spec mapping.

## Rules

- Construct only via `params.create_module(...)`. Never `import hydra` or
  `hydra.utils.instantiate` outside `HydraParametrisation._build_spec`.
- Sub-module **class defaults live in the signature (code)**, not in config `_target_`.
  Structural **values** (num_heads, num_layers, layer_kernels, …) still come from `params`.
- Read scalars/specs with `self.params.get("dotted.key", default)`.
- `layer_kernels` is a `LayerKernels` (a `Parametrisation`); leaf layers keep
  `layer_kernels.Linear(...)` / `["LayerNorm"]` access. Its type hint is `Parametrisation`.
- Leaf layers (mappers, processors, attention, FFT) keep explicit-kwarg constructors — don't
  thread `params` into them unless asked.
- Tests/examples build models with `DictParametrisation({...})`, not OmegaConf/DotDict.
- Interface: keyword is `params=`; stored attribute is `model.params` (not `model.config`).
- Graph builders build attributes via a `parametrisation` (default: a stateless
  `DictParametrisation`); `GraphCreator` threads its own instance through.
