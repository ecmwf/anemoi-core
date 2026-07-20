---
name: parametrisation
description: How to construct anemoi models/graphs the Hydra-free way, using the Parametrisation ABC and create_module. Use when adding or editing a model, a model sub-module, a graph builder, or any code that used to call hydra.utils.instantiate in anemoi.models or anemoi.graphs.
---

# Parametrisation pattern (no Hydra, no `instantiate`)

Object construction in `anemoi.models` and `anemoi.graphs` goes through
`anemoi.graphs.parametrisation`, **not** Hydra. Do not import `hydra` or
`anemoi.models.utils.instantiate` (deleted) on the construction path.

See `agent/parametrisation-refactor.md` for the full rationale and rejected options.

## Core API (`anemoi.graphs.parametrisation`)

- `Parametrisation` (ABC): `get(key, default=MISSING)` (dotted keys, raises
  `ParametrisationError` if missing and no default), `to_dict()`, `create_module(spec, ...)`,
  `resolve(value, default_factory, ...)`.
- `JSONParametrisation(dict)`: concrete, JSON-serialisable impl. `from_json` / `from_file` /
  `to_json`. Use this in tests, examples, and the training→model boundary
  (`JSONParametrisation(OmegaConf.to_container(cfg, resolve=True))`).
- `build(spec, *args, **kwargs)`: the Hydra-free construction engine (`_target_`,
  `_partial_`, `_recursive_`, `_args_`; call-time kwargs win). `create_module` delegates to it.

## Writing a module that builds children

Every constructor takes `params: Parametrisation` first. A child sub-module is a keyword arg
defaulting to `None`, dispatched by type:

```python
from anemoi.graphs.parametrisation import Parametrisation

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
- Tests/examples build models with `JSONParametrisation({...})`, not OmegaConf/DotDict.
- The stored attribute on the interface is `model.params` (not `model.config`).
