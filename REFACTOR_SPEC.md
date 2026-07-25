# Anemoi "Great Refactoring": Dependency-Injection Builder — Full Specification

This document captures, unambiguously, the goal, principles, architecture, and per-package
plan for replacing Hydra `instantiate` with a dict-based dependency-injection **Builder**
across `anemoi-core` (`graphs`, `models`, `training`). It supersedes and consolidates the
original notes in `refactor.md` and `graph.md`. It is written so the work can be re-run
from scratch with no ambiguity.

---

## 1. Goal

Stop constructing polymorphic sub-objects *inside* constructors via
`hydra.utils.instantiate(...)`. Instead:

- Every object receives its polymorphic members as **fully-built objects passed to its
  constructor** (`has-a` / object injection).
- A separate **Builder** reads the configuration and builds the whole object tree
  bottom-up, wiring runtime-computed values explicitly.
- Hydra is **removed from construction**; backward compatibility is preserved by a single
  config→plain-`dict` conversion at each boundary.

Motivation: transparency (docstrings, "go to definition"), static typing, IDE
introspection, testability, and decoupling every class from Hydra/config.

---

## 2. Core principles (MUST hold — these are the non-negotiable rules)

**P1 — No `instantiate`/`build` inside constructors.** A class constructor never calls
`hydra.utils.instantiate` nor `anemoi.utils.builder.build`. Replacing `instantiate(cfg)`
with `build(cfg)` *inside* a constructor is explicitly WRONG — it is the "just substitute
Hydra for a similar function" anti-pattern. `build`/`build_all`/`as_dict` appear **only**
in Builder code.

**P2 (relaxed) — Constructors may receive parametrisation, but must not use it to build
polymorphic sub-objects.**
A constructor may receive `int`/`float`/`str`/`bool`, lists/tuples, **already-built
objects**, AND **parametrisation** (config subtrees / settings such as a `..._params`
`DotDict`, a `layer_kernels` choice, thresholds, option dicts). The hard constraint is on
*what the constructor does with it*: it MUST NOT use passed config/parametrisation to call
a **factory that builds a polymorphic sub-object** — i.e. no `instantiate`/`build` over a
`_target_` inside a constructor.

Practical boundary:
- **Must be Builder-built and injected** — the interchangeable model *components* that a
  `config.model.<x>._target_` selects: encoder, processor, decoder, mappers, residual,
  boundings, noise_injector/noise_embedder, level processors, downscale/upscale, graph
  providers, node attributes.
- **May be resolved in-constructor from parametrisation** — leaf configuration that does
  not build a polymorphic component: parsing a `..._params` subtree into a settings
  dataclass (`Settings.from_config(...)`), reading scalars/flags/thresholds, and selecting
  `layer_kernels` (which plain `nn` layer classes to use). These are parametrisation, not
  architecture.

Rationale: the earlier "no settings object at all" rule made value-object-heavy variants
(e.g. transport) needlessly painful; the real target is that *architecture is injected*,
not that every scalar is threaded by hand.

**P3 — The Builder is the ONLY code that reads configuration.** All knowledge of config
keys, `_target_`, interpolations, defaults, and runtime wiring lives in Builders. The
Builder extracts scalars from config and passes them as explicit primitives; it builds
child objects and passes them in; it builds "factory" registries (e.g. layer kernels) and
passes the built registry.

**P4 — Object containment is mirrored by the Builder.** The Builder builds children first,
computes any runtime values needed (e.g. `edge_dim` from a graph provider,
`in_channels`/`hidden_dim` from data indices + node attributes), then constructs the
parent with those children + values injected. Deeper polymorphic objects are injected too,
recursively, at every depth (there is no depth at which config is allowed to leak into a
constructor).

**P5 — Hydra is removed from construction; backward compatibility via conversion.**
Existing Hydra/YAML configs keep working. At each boundary where a Hydra `DictConfig`
arrives, convert it once to a plain `dict` (resolving interpolations) with
`anemoi.utils.builder.as_dict`, then operate on plain dicts + built objects. The
`_target_` convention is preserved in the dict form. `nn.Module` attribute names are
preserved so existing checkpoints (`state_dict` keys) keep loading.

**P6 — Scope: all three packages** — `graphs`, `models`, `training` — plus shared code in
`anemoi-utils`.

---

## 3. Shared machinery — `anemoi.utils.builder`

Module: `anemoi-utils/src/anemoi/utils/builder.py` (the live checkout is
`/lus/.../git/anemoi-utils`, editable-installed; NOT the copy nested in anemoi-core).

- `locate(path) -> obj` — resolve a dotted import path (class/function). Replaces
  `hydra.utils.get_class`.
- `build(spec, /, *args, **injected) -> obj` — construct ONE object from a
  `{"_target_": ..., **params}` spec: resolve the target, merge spec params with the
  injected runtime kwargs (injected override), call it. Honours `_partial_` (returns a
  `functools.partial`), `_recursive_`, `_convert_`. Recursion-capable but, per P1, it is
  only ever called from Builders.
- `build_all(specs, /, **injected)` — map `build` over a `dict`/`list` of homogeneous
  specs, applying the same injected kwargs to every element.
- `as_dict(config)` — the Hydra boundary shim: OmegaConf `DictConfig`/`ListConfig` →
  plain `dict`/`list` (interpolations resolved); `DotDict`/`dict` pass through unchanged.
- `to_dict(obj)` / `target_path(obj)` — the **inverse of `build`** (serialisation): turn a
  built object into a JSON-able `{"_target_": <path>, ...}` spec (recursively), so
  `build(to_dict(obj))` reconstructs an equivalent object. An object serialises via, in
  order: a `__anemoi_spec__` mapping recorded at build time; a `to_dict()` method; or
  introspection (`_target_` + each `__init__` parameter read from the same-named attribute).
- `Builder` (config-carrying convenience) and `BuilderError`.

Tests: `anemoi-utils/tests/test_builder.py`.

---

## 4. The two layers

**Layer 1 — classes become containers.** Each class that used `instantiate` internally is
changed so its polymorphic members are constructor parameters; the constructor only stores
them (`self.encoder = encoder`) and computes cheap derived bookkeeping (indices/shapes)
needed at runtime. No config, no building.

**Layer 2 — Builders read config and build bottom-up.** Per-package Builder classes mirror
the object hierarchy and encode the wiring: build leaves, compute runtime values, build
parents with children injected. Dispatch to the right Builder is by the object's
`_target_`, resolved to a class (so re-export aliases resolve correctly).

---

## 5. Per-package plan

### 5.1 Graphs  (DONE — reference implementation)
- `nodes/builders/base.py`, `edges/builders/base.py`: node/edge attributes are built
  objects injected via an `attributes` constructor param (dict `{name: obj}` or list,
  normalised by `utils.normalise_attributes`); `register_attributes` uses them (no
  `instantiate`).
- `create.py`: object-API **`GraphBuilder`** (matches the `graph.md` Python API:
  `GraphBuilder(nodes=[...], edges=[...], post_processors=[...])`) is the base;
  config-driven **`GraphCreator(GraphBuilder)`** parses config via `as_dict`/`build`/
  `build_all` and delegates.
- `processors/post_process.py`: edge-attribute recompute uses `build` (Builder-side).
- Result: no `instantiate` in `graphs/src`; full test suite green.

### 5.2 Models  (IN PROGRESS)
- New **`models/models/builder.py`** with a `ModelBuilder` hierarchy (registry keyed by
  model CLASS via `locate`). It builds: `node_attributes` (`NamedNodesAttributes`), graph
  providers (`create_graph_provider`), encoder/processor/decoder (and variant extras),
  residual, boundings, **and the layer-kernel registry**, then constructs the model with
  everything injected. Dim computations (`input_dim`/`input_dim_latent`/`target_dim`/
  `output_dim`) live on the Builder and are overridable per variant (e.g. auto-encoder's
  forcing-based `target_dim`; ensemble's fcstep/conditional input dim).
- `models/models/base.py` `BaseGraphModel` is a container: `__init__` takes injected
  `node_attributes`/`residual`/`boundings` + scalars (`hidden_nodes_name`, `num_channels`,
  `latent_skip`) + runtime data (`data_indices`, `statistics`, `graph_data`, `n_step_*`).
  It keeps `_calculate_shapes_and_indices` (needs the injected `node_attributes`). No
  `_build_networks`/`_build_residual`.
- Variant classes (`encoder_processor_decoder`, `autoencoder`, `ens_*`, `hierarchical`,
  `hierarchical_autoencoder`, `transport_*`): each `__init__` receives its built nets and
  stores them (e.g. hierarchical receives downscale/upscale/level-processors + providers
  and derives `hidden_dims` from `num_channels`). No `_build_networks`.
- `interface/__init__.py` builds the model via `build_model(...)` (the ModelBuilder entry).
  Pre/post processors should likewise be built by a builder and injected.
- Architecture components (encoder/processor/decoder/mappers/residual/boundings/noise_*/
  level processors/downscale/upscale) are Builder-built and injected. Under the relaxed P2,
  `layer_kernels` is *parametrisation* (which plain `nn` layer classes to use): a mapper may
  receive the `layer_kernels` config/registry and resolve it via `load_layer_kernels`, OR
  receive an already-built registry. `load_layer_kernels` is now Hydra-free (`build`) and
  idempotent (a built registry passes through), so both work.
- `layers/utils.py::load_layer_kernels`, `layers/bounding.py::build_boundings`,
  `utils/compile.py` are factory/lookup helpers and may use `build`/`build_all`/`locate`.
- Checkpoints: keep every `self.<name>` attribute name identical → `state_dict` keys
  unchanged.

### 5.3 Training  (DONE)
`AnemoiTrainer` + the LightningModule assembly + the factory functions
(`get_loss_function`, `create_scalers`, `get_callbacks`, `get_*_logger`, the checkpoint
pipeline) are the training builder — they read config and build/inject. All Hydra
*construction* removed: `instantiate`/`get_class` → `build`/`locate`;
`instantiate_with_runtime_kwargs` is now a thin `build` wrapper. Hydra is kept only as the
config *loader* (`@hydra.main`, `compose`/`initialize`). Sites converted: `train/train.py`
(task, training method, strategy), `train/methods/base.py` (output_mask, optimizer,
scheduler), `losses/loss.py`, `losses/scalers/scalers.py`, `checkpoint/pipeline.py`,
`diagnostics/callbacks/*`, `diagnostics/logger.py`, `layers`/`utils`. `grep "instantiate("
training/src` → none (only method names/strings). The relaxed P2 applies: the LightningModule
receives `config` (parametrisation) and assembles via the factories; the model architecture
under it is already injected by the ModelBuilder.

Run with the repo `.venv` (uv), not the py312 venv (its `pytorch_lightning` lacks
`WeightAveraging`). Pre-existing suite failures are unrelated env gaps: `pytest-asyncio`
missing (async checkpoint tests), network/`obstore` (checkpoint sources), `wandb`/`azure`/
DCT deps not installed.

---

## 5.4 Serialisation (round-trip through the builder)

A model/graph can be serialised to a JSON-able dict and rebuilt into the same thing:

- **Model**: `build_model` records the architecture recipe on the model
  (`model.__anemoi_spec__ = as_dict(model_config)`). `to_dict(model)` returns that recipe;
  `json.dumps` it; `build_model(spec, data_indices=…, statistics=…, graph_data=…, …)`
  rebuilds an identical architecture (same `state_dict` keys, same parameter count). The
  *recipe* (architecture) is serialised, not the weights or the runtime context (graph /
  indices), which are supplied again at rebuild. `build_model` accepts OmegaConf, `DotDict`
  or plain/JSON `dict`.
- **Graph**: the recipe is the graph config (already basic types) — it round-trips through
  JSON and `GraphCreator(config).create()` yields an identical graph.
- **Generic objects**: `to_dict`/`build` round-trip any object whose `__init__` params are
  stored as same-named attributes, or that defines `as_dict()` / `__anemoi_spec__`.
  `to_dict` serialises `torch.dtype` (→ short string), numpy arrays/scalars (→ lists/
  scalars) and `os.PathLike` (→ path string) so common *transformed* params round-trip by
  introspection. A class that transforms a param in a way introspection can't recover
  should define `as_dict()` — the recommended body is `return introspect(self)`, or a
  hand-built spec. The graph node/edge builders and node/edge attribute base classes carry
  such an `as_dict()`. Example: `examples/serialise_roundtrip.py`; e.g.
  `build(to_dict(EdgeLength(norm="unit-std")))` reconstructs the attribute.

## 6. Verification

- `anemoi-utils`: `python -c "import tests.test_builder as t; [f() for n,f in vars(t).items() if n.startswith('test_')]"` (the package pytest hangs in conftest on azurite).
- Graphs: `pytest graphs/tests`.
- Models: `pytest models/tests`; plus an end-to-end `build_model` construction test that
  builds a real model (e.g. GNN) and asserts networks are injected; confirm an existing
  checkpoint's `state_dict` keys still load.
- Training: `pytest training/tests/unit`; run the integration training cycle.
- Gate: `grep -rn "from hydra.utils import instantiate" graphs/src models/src training/src`
  returns nothing; and no constructor receives a `DictConfig`/`DotDict`/settings object.

---

## 7. Environment notes (this HPC checkout)

- Two `anemoi-utils` checkouts exist; the editable-installed one is
  `/lus/.../git/anemoi-utils`. The copy nested in anemoi-core is a stray — do not edit it.
- `graphs`/`models`/`training` are `pip install -e` editable, so `pytest`/imports use the
  checkout directly.
- A pre-existing deadlock in the uncommitted anemoi-utils `settings.py` WIP blocked
  `import anemoi.utils.caching` (settings init re-entered under a non-reentrant lock via
  schema-plugin discovery → anemoi-registry → `remote/s3.py` reading `SETTINGS` at import).
  Fixed by making `anemoi.utils.remote.s3` import `SETTINGS` lazily.
- No GPU on the login node; heavy CUDA work goes through SLURM (`srun`).

---

## 8. Status snapshot (update as work proceeds)

- anemoi-utils `builder`: DONE (build/build_all/as_dict/locate + tests).
- Graphs: DONE, test suite green (313). Node/edge builder subclass constructors now forward
  `attributes=`, so the `graph.md` constructor-kwarg API works end-to-end
  (`KNNEdges(source_name=…, target_name=…, num_nearest_neighbours=3, attributes=[EdgeLength(...)])`)
  and a full `GraphBuilder` object serialises + round-trips via `to_dict`/`build`
  (`graphs/tests/test_graph_builder_api.py`).
- Models: DONE. `ModelBuilder` + container base + interface; ALL 6 variants migrated
  (EncProcDec [end-to-end validated], AutoEncoder, Ens, Hierarchical, Transport [+Tend],
  HierarchicalAutoEncoder). `grep hydra models/src` → none. Models suite 521 pass.
  `load_layer_kernels` is Hydra-free + idempotent; `examples/create_model.py` runs both the
  config path and a pure no-settings object path. (Under relaxed P2, `layer_kernels` config
  reaching a mapper is acceptable parametrisation; optionally inject the built registry too.)
- Training: TODO.
