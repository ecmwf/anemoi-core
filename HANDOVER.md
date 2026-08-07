# HANDOVER

## Task (one line)
Replace Hydra `instantiate`-based construction with a dict-based dependency-injection
`Builder` across anemoi-core (`graphs`, `models`, `training`). Branch:
`feat/great-refactoring-option-2`. Spec: `refactor.md` + `graph.md` (repo root). Full plan:
`/home/mab/.claude/plans/parallel-spinning-scone.md`.

## Status
- **Layer 0 (anemoi.utils.builder): DONE, tested.**
- **Graphs: DONE, tested** (full suite 313 pass).
- **Models: IN PROGRESS.** DI `ModelBuilder` (`models/src/anemoi/models/models/builder.py`)
  built. Base is now a container; `interface` uses `build_model`. Migrated + models suite
  521 pass: AnemoiModelEncProcDec (validated END-TO-END, `test_model_builder.py`),
  AutoEncoder, Ens, Hierarchical. REMAINING (construction broken until migrated, same
  pattern): `transport_encoder_processor_decoder`, `hierarchical_autoencoder`.
- **Training: NOT STARTED.**
- Fixed an unrelated blocker (settings import deadlock) — see Environment gotchas.
- Packages are now `pip install -e` editable — plain `pytest`/imports hit the checkout
  (the `PYTHONPATH=graphs/src` prefix is no longer required).

### Correct models pattern (do NOT `instantiate`→`build` inside constructors)
`build`/`build_all` live ONLY in the `ModelBuilder`. Model `__init__`s receive built
sub-objects and store them (`self.encoder = encoder`). To migrate the two remaining
variants: give each an injected `__init__` (mirror `hierarchical.py`/`ens_*`), delete its
`_build_networks`, and add a `ModelBuilder` subclass (set `model_cls`, implement/override
`build_networks` and any dim overrides — `input_dim`/`output_dim`/`target_dim`). The
registry is keyed by model CLASS via `locate`, so config `_target_` aliases resolve.
Transport also needs its settings objects (NoiseConditioningSettings/EdmSettings/...) —
build them in the builder and inject, or pass `transport_params`.

## First actions on a new session
```bash
cd /lus/h2resw01/hpcperm/mab/git/anemoi-core
git branch --show-current            # expect feat/great-refactoring-option-2
git status --short
```
Then read `refactor.md`, `graph.md`, and the plan file above.

## Environment gotchas (CRITICAL — read before running anything)
1. **Two `anemoi-utils` checkouts.** Python imports the editable-installed one at
   `/lus/h2resw01/hpcperm/mab/git/anemoi-utils` (branch `main`). The `anemoi-utils/` dir
   nested inside anemoi-core is a STRAY clone NOT on sys.path — do not edit it. All
   anemoi-utils edits (builder.py, s3.py) were made in the installed checkout.
2. **Core packages are installed NON-editable** into site-packages
   (`.../venv/py312/.../site-packages/anemoi/{graphs,models,training}`). So plain
   `pytest graphs/tests` runs the STALE installed copy, NOT the checkout. **Always prefix
   `PYTHONPATH=graphs/src` (add `models/src`, `training/src` as needed)** so the checkout
   wins via namespace-package merge. Consider `pip install -e graphs models training`.
3. **Settings deadlock (FIXED this session, uncommitted).** `import anemoi.utils.caching`
   used to hang: `SETTINGS.__getattr__` → `_get_settings_class` (holds a non-reentrant
   `threading.Lock`) → `_discover_schema_plugins` imports `anemoi.registry` → `s3.py`
   read `SETTINGS` at module top → re-entered under the lock → deadlock. Fix: made
   `anemoi.utils.remote.s3` import `SETTINGS` lazily (inside the function, not module top).
   This unblocked all graphs/models/training imports. If a fresh checkout reintroduces it,
   re-apply the same lazy-import fix.
4. **First `torch`/`torch_geometric` import is slow (cold ~2min).** Use generous timeouts.
5. **pytest `-q` buffers output**; a `timeout`-kill loses it → looks like a hang. Prefer
   running to completion, or write progress to a file with explicit `flush=True`.
6. `pytest anemoi-utils/tests` hangs at collection (conftest imports azurite/obstore). To
   test one module there, bypass conftest:
   `python -c "import tests.test_builder as t;[f() for n,f in vars(t).items() if n.startswith('test_')]"`.

## Verified vs unverified
- VERIFIED: `anemoi.utils.builder` — 18/18 unit tests pass.
- VERIFIED: graphs full suite — **313 passed** via
  `PYTHONPATH=graphs/src python -m pytest graphs/tests`.
- VERIFIED: `grep -rn "instantiate" graphs/src` → none left.
- VERIFIED: settings/caching import no longer deadlocks (~2s).
- UNVERIFIED: models, training (untouched).

## Suggested next steps (priority order)
1. **Models refactor** (task #3). Models tree is currently at a CLEAN, GREEN baseline
   (all original Hydra `instantiate`; 520 pass). This is a big-bang change (shared
   `BaseGraphModel.__init__` in `models/src/anemoi/models/models/base.py` feeds all 6
   variants: encoder_processor_decoder, hierarchical, hierarchical_autoencoder, ens,
   transport, autoencoder), so base + all variants + interface + a new `ModelBuilder` must
   land together. Needs checkpoint-state_dict validation → do as a focused pass.

   CRITICAL — this is object INJECTION, not an `instantiate`→`build` swap (see the mistake
   made & reverted this session). Do NOT put `build(...)` inside `_build_networks`/`__init__`.
   Instead: model constructors RECEIVE built sub-objects as params and just store them
   (`self.encoder = encoder`). A separate **`ModelBuilder`** (mirror `graphs/create.py`
   `GraphBuilder`) reads config and builds bottom-up:
   (a) build `node_attributes` (NamedNodesAttributes — direct construction; move
       `_build_named_node_attributes_graph` + `broadcast_config_keys` to the builder),
   (b) compute dims (input_dim/input_dim_latent/target_dim/output_dim — note `target_dim`
       is polymorphic: autoencoder overrides `_calculate_target_dim`),
   (c) build graph providers via `create_graph_provider`,
   (d) `build(model_config.model.encoder, _recursive_=False, in_channels_src=dims..., edge_dim=provider.edge_dim, ...)`
       for encoder/processor/decoder — INSIDE the builder,
   (e) build `residual` (per-dataset) + `boundings` (`build_boundings` factory is fine to
       keep — it's builder-side; it may use `build`),
   (f) construct the model passing everything in.
   Model `__init__`s gain params (encoder, processor, decoder, node_attributes, graph
   providers, residual, boundings, +variant extras noise_injector/noise_embedder/
   downscale/upscale/level_processors) stored under the SAME `self.<name>` attrs →
   state_dict keys unchanged → checkpoints OK. Base keeps `_calculate_shapes_and_indices`
   (needs injected node_attributes + data_indices) since dims are used at runtime too
   (e.g. `encoder_processor_decoder.py:333` metadata, transport dims). Remove
   `_build_networks`/`_build_residual` from the model; drop `model_config` from model ctors
   (pass the scalars hidden_nodes_name/num_channels/latent_skip instead).
   `interface/__init__.py`: inject built `model` + pre/post processors (a ModelBuilder
   builds them). `layers/utils.py::load_layer_kernels` + `layers/bounding.py` +
   `utils/compile.py` may switch `instantiate`/`get_class`→`build`/`locate` (factory/lookup
   code — allowed). `_recursive_=False`/`_convert_` handling then goes away.
   Validate: `models/tests/models/test_base_graph_model.py` (constructs
   `DummyGraphModel(BaseGraphModel)` directly — WILL need updating to pass injected
   node_attributes/residual/boundings) + add an `AnemoiModelEncProcDec` ModelBuilder test.
   Confirm an existing checkpoint's `state_dict` keys still load.
2. **Training refactor** (task #4): `train.py`, `methods/base.py` (output_mask/optimizer/
   scheduler), `losses/loss.py`, `losses/scalers/scalers.py`, `checkpoint/pipeline.py`,
   `diagnostics/callbacks` + `logger.py`. Replace `instantiate`/`get_class` with
   `build`/`build_all`/`resolve_target`(=`locate`); delete
   `utils/hydra.py::instantiate_with_runtime_kwargs` (use `build(spec, **runtime)`).
   Run `training/tests/integration/test_training_cycle.py::test_training_cycle_global`.
3. **Graphs follow-up** (task #5, low priority): node/edge builder SUBCLASS constructors
   (~20) don't forward `attributes=` to the base, so the `graph.md` constructor-kwarg API
   (e.g. `CutOffEdges(..., attributes=[...])`) doesn't work yet. Base + config path +
   object-API `GraphBuilder` base ARE done. Add forwarding + a GraphBuilder-vs-YAML
   equivalence test.

## Design decisions locked in (do not relitigate)
- Shared engine is `anemoi.utils.builder`: `build(spec, **injected)` (recursive-capable,
  honours `_target_`/`_partial_`/`_recursive_`/`_convert_`), `build_all(specs, **injected)`
  (same injected kwargs to every element of a dict/list), `as_dict(cfg)` (OmegaConf→plain
  dict; the Hydra-compat boundary), `locate(path)`, `Builder`, `BuilderError`.
- User decisions: do all three packages; **replace Hydra entirely** but keep backward compat
  via `as_dict` (DictConfig→dict) at each boundary. Preserve `_target_` config shape and
  `nn.Module` attribute names (so YAML configs + checkpoints keep working).
- Requirement is NOT a like-for-like `instantiate` swap: polymorphic members must be passed
  as fully-built objects to constructors (Layer 1); a Builder reads config and builds the
  tree bottom-up wiring runtime values (Layer 2).
- Graphs: `create.py` has object-API `GraphBuilder` (base) + config-driven
  `GraphCreator(GraphBuilder)`. Node/edge attributes are built objects injected via
  `attributes` (dict `{name: obj}` or list, normalised by `utils.normalise_attributes`).
  Attributes registered separately in the config path (no subclass ctor changes needed).
  post_process edge-attr recompute uses `build` (handles spec-or-object).

## Key file map
- `/lus/.../git/anemoi-utils/src/anemoi/utils/builder.py` — the DI engine (+ tests
  `/lus/.../git/anemoi-utils/tests/test_builder.py`).
- `/lus/.../git/anemoi-utils/src/anemoi/utils/remote/s3.py` — lazy-SETTINGS deadlock fix.
- `graphs/src/anemoi/graphs/create.py` — GraphBuilder + GraphCreator.
- `graphs/src/anemoi/graphs/{nodes,edges}/builders/base.py` — attribute injection.
- `graphs/src/anemoi/graphs/utils.py` — `normalise_attributes`, `camel_to_snake`.
- `graphs/src/anemoi/graphs/processors/post_process.py` — `build` recompute.
- MODELS (next): `models/src/anemoi/models/models/base.py`,
  `.../models/encoder_processor_decoder.py`, `.../interface/__init__.py`,
  `.../layers/bounding.py`, `.../layers/utils.py`. Validate:
  `models/tests/models/test_base_graph_model.py`.

## Quick regression commands
```bash
cd /lus/h2resw01/hpcperm/mab/git/anemoi-core
# builder unit tests (bypass hanging anemoi-utils conftest):
( cd /lus/h2resw01/hpcperm/mab/git/anemoi-utils && \
  python -c "import tests.test_builder as t;[f() for n,f in vars(t).items() if n.startswith('test_')];print('builder OK')" )
# graphs suite (must use PYTHONPATH — site-packages copy is stale):
PYTHONPATH=graphs/src python -m pytest graphs/tests -o addopts="" -p no:cacheprovider -q
# confirm no hydra instantiate left in graphs:
grep -rn "from hydra.utils import instantiate" graphs/src || echo "graphs clean"
```
