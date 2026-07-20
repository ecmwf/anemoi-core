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
  `ParametrisationError` if missing and no default), `to_dict()`, `create_module(spec, ...)`
  (dotted-path string or `_target_` mapping; `_partial_`/`_recursive_`/`_args_`; call-time
  kwargs win), `resolve(value, default_factory, ...)`.
- `DictParametrisation(dict)`: concrete, JSON-serialisable impl (`from_json` / `from_file` /
  `to_json`). Use in tests, examples and inference.
- `TrainingParametrisation` (in `anemoi.training.parametrisation`): the training-side
  subclass, `TrainingParametrisation.from_config(omegaconf_cfg)` — used at the training→model
  boundary.
- When a leaf helper or a graph builder method needs to build something but has no
  parametrisation in scope, it takes one as an (optional) argument, or falls back to a
  module-level stateless `DictParametrisation()`.

## Writing a module that builds children

Every constructor takes `params: Parametrisation` first. A child sub-module is a keyword arg
defaulting to `None`, dispatched by type:

```python
from anemoi.utils.parametrisation import Parametrisation

class MyModel(BaseGraphModel):
    def __init__(self, params: Parametrisation, *, encoder=None, **kwargs):
        self._encoder = encoder
        super().__init__(params, **kwargs)

    def _build_networks(self):                      # no args; read self.params
        self.encoder = self._build_submodule(
            self._encoder,
            spec_key="model.encoder",               # None -> build the configured class
            _recursive_=False,                      # keep layer_kernels as config for the child
            in_channels_src=..., edge_dim=...,      # runtime dims the model computes
        )
```

`_build_submodule(value, *, spec_key, default=None, **runtime)` (on `BaseGraphModel`):
`None` → `create_module(get(spec_key, default), **runtime)`; `str` → `create_module(value,
**runtime)`; instance → used as-is.

When a module builds *many* children from one spec (e.g. hierarchical down/up processors),
call `self.params.create_module(self.params.get("model.processor"), _recursive_=False, ...)`
directly instead of `_build_submodule`.

## Rules

- Read scalars/specs with `self.params.get("dotted.key", default)`. Never navigate config
  trees by attribute access, and never pass sub-config dicts around as constructor kwargs.
- Read structural params (num_heads, num_layers, …) from `params`; inject model-computed
  dims (edge_dim, in_channels_*) as explicit kwargs.
- Leaf layers (mappers, processors, attention, FFT) keep their explicit-kwarg constructors —
  do not thread `params` into them unless asked.
- Tests/examples build models with `DictParametrisation({...})`, not OmegaConf/DotDict.
- The stored attribute on the interface is `model.params` (not `model.config`); the
  `AnemoiModelInterface` constructor keyword is `params=`.
- Graph builders (`update_graph`/`register_attributes`) and post-processors take an optional
  `parametrisation` argument; `GraphCreator` threads its own instance through.
