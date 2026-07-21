# Parametrisation refactor — release notes / changelog entries

`CHANGELOG.md` files are auto-generated (release-please) from commits/PRs, so the
human-written change notes for this refactor live here. Copy the relevant entries into the
release PR description when the branch is merged. See `agent/parametrisation-refactor.md`
for the design rationale and the migration table.

## anemoi-utils

### Features

* **`anemoi.utils.parametrisation`** — Hydra-free construction mechanism shared by
  `anemoi.graphs`, `anemoi.models` and `anemoi.training`:
  * `Parametrisation` (ABC): `get(key, default)`, `to_dict()`, and `create_module(spec, ...)` —
    the single construction entry point. `create_module` dispatches on the argument: a class
    is instantiated, a dotted-path string / `_target_` mapping / list is built via
    `_build_spec`, `None` returns `None`, an already-built instance is returned unchanged.
  * `DictParametrisation`: JSON-serialisable, dict-backed impl (`from_json`/`from_file`/`to_json`);
    `_build_spec` is Hydra-free.
  * `ParametrisationError`, and dotted-path helpers `get_object` / `get_class`.

## models

### ⚠ BREAKING CHANGES

* **Hydra-free, Parametrisation-based construction.** Models and layers are no longer built
  with Hydra `instantiate`; objects come from a `Parametrisation` via
  `params.create_module(...)`. The `anemoi.models.utils.instantiate` shim is removed.
* **`AnemoiModelInterface`** now takes `params:` (a `Parametrisation`) instead of `config:`
  (OmegaConf `DictConfig`), stored as `model.params` (was `model.config`). Inference builds it
  from checkpoint JSON with `DictParametrisation(...)`; training uses `TrainingParametrisation`.
* **Sub-modules default to classes in code.** `encoder`, `processor`, `decoder`, `residual`,
  `noise_injector` are constructor arguments defaulting to their classes (e.g.
  `encoder=GraphTransformerForwardMapper`). A config `model.<sub>._target_` no longer *selects*
  the class — override with a class, dotted-path string, or built `nn.Module`. Structural
  settings (num_heads, num_layers, layer_kernels, …) are still read from the parametrisation.
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
