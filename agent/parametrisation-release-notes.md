# Parametrisation refactor — release notes / changelog entries

`CHANGELOG.md` files are auto-generated (release-please) from commits/PRs, so the
human-written change notes for this refactor live here. Copy the relevant entries into the
release PR description when the branch is merged. See `agent/parametrisation-refactor.md`
for the design rationale and the migration table.

## anemoi-utils

### Features

* **`anemoi.utils.parametrisation`** — Hydra-free construction mechanism shared by
  `anemoi.graphs`, `anemoi.models` and `anemoi.training`:
  * `Parametrisation` (ABC): `get(key, default)`, `to_dict()`, `from_dict(mapping)` (factory)
    and `create_module(spec, ...)` — the single construction entry point. `create_module`
    dispatches on the argument: a class is instantiated, a dotted-path string / `_target_`
    mapping / list is built via `_build_spec`, `None` returns `None`, an already-built instance
    is returned unchanged. Serialisation is dict-only (`from_dict` / `to_dict`); wrap with
    `json.loads` / `json.dumps` for a string.
  * `DictParametrisationBase` (common base, never instantiated) and the Hydra-free leaf
    `DictParametrisation`; `_build_spec` is Hydra-free.
  * `ParametrisationError`, and dotted-path helpers `get_object` / `get_class`.

## models

### ⚠ BREAKING CHANGES

* **Hydra-free, Parametrisation-based construction.** Models and layers are no longer built
  with Hydra `instantiate`; objects come from a `Parametrisation` via
  `params.create_module(...)`. The `anemoi.models.utils.instantiate` shim is removed.
* **`AnemoiModelInterface`** now takes `params:` (a `Parametrisation`) instead of `config:`
  (OmegaConf `DictConfig`), stored as `model.params` (was `model.config`). Inference builds it
  from the checkpoint's config dict with `Parametrisation.from_dict(...)`; training builds the
  same way from the resolved OmegaConf config.
* **Sub-module class selection.** `encoder`, `processor`, `decoder`, `residual`,
  `noise_injector` are constructor arguments (default `None`). The class is resolved in
  priority order: an explicit constructor override (class / dotted-path string / built
  `nn.Module`) → the `model.<sub>._target_` named in the parameters → the code default class
  (e.g. `GraphTransformerForwardMapper`). This keeps `model=gnn` / `transformer` selectable via
  config while providing a code fallback. Structural settings (num_heads, num_layers,
  layer_kernels, …) are read from the parametrisation. See `BaseGraphModel._create_submodule`.
* **`load_layer_kernels` returns a `LayerKernels`** (a `Parametrisation`) instead of a
  `DotDict`; attribute/item access (`layer_kernels.Linear(...)`, `["LayerNorm"]`) is unchanged.
  A kernel that cannot be built raises `ParametrisationError` (was `InstantiationException`).

## graphs

### Features

* **Python graph API** — `GraphBuilder(nodes=[...], edges=[...], post_processors=[...])` built
  from node/edge/attribute objects. Config-driven `GraphCreator(config)` still works (it parses
  the config into those objects).

### ⚠ BREAKING CHANGES

* **Hydra-free construction** — builders and attributes are built through a `Parametrisation`
  (`create_module`); `GraphCreator` normalises its config into a `Parametrisation`, so the
  builders no longer receive attribute config trees (they take pre-built attribute objects).

## training

### ⚠ BREAKING CHANGES

* **Object construction via Parametrisation** — the model, optimiser, scheduler, losses,
  scalers, callbacks, loggers, output mask and checkpoint-pipeline stages are built through a
  `Parametrisation` (`create_module`) instead of direct `hydra.utils.instantiate` / `get_class`.
  Behaviour is preserved: `TrainingParametrisation` (a `HydraParametrisation`) delegates
  `create_module` to `hydra.utils.instantiate` and reads the OmegaConf config in `get`. The
  model is built with `params=TrainingParametrisation.from_config(config)`.
* The Hydra **launcher / config-loading** (`@hydra.main`, `compose`/`initialize`, the
  search-path plugin) is unchanged.
