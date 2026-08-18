# Refactoring findings — Anemoi DI Builder

Concrete technical findings gathered while implementing the DI Builder (see
`REFACTOR_SPEC.md` for the requirements/principles). These are the non-obvious facts that
shape a correct implementation.

---

## 1. Model object-containment tree — what is polymorphic vs fixed

A model (`AnemoiModelInterface` → inner `BaseGraphModel` subclass) contains:

| Member | Polymorphic (config `_target_`)? | Built by | Injected as |
|---|---|---|---|
| `model` (inner) | yes (variant) | ModelBuilder | into interface |
| `pre/post_processors` | yes (list) | (interface today; should be a builder) | into interface |
| `node_attributes` | no (`NamedNodesAttributes`) | ModelBuilder | into model |
| `*_graph_provider` | no (`create_graph_provider`) | ModelBuilder | into model |
| `encoder`/`processor`/`decoder` | **yes** | ModelBuilder | into model |
| `residual` | **yes** | ModelBuilder | into model |
| `boundings` | **yes** (list per dataset) | ModelBuilder (`build_boundings`) | into model |
| `noise_injector` (ens) | **yes** | ModelBuilder | into model |
| `noise_embedder` + settings (transport) | **yes**/value-objects | ModelBuilder | into model |
| `down/up_level_processor`, `downscale`, `upscale` (hierarchical) | **yes** | ModelBuilder | into model |

Key point: even the *non-polymorphic* members (`node_attributes`, graph providers) must be
built by the Builder and injected, because the Builder needs them to compute the runtime
args of the polymorphic ones (see §2). So the model becomes a pure container.

## 2. Why construction must be bottom-up (the dim wiring)

The encoder/processor/decoder constructors need runtime values that only exist mid-build:

- `edge_dim = graph_provider.edge_dim` — the provider is derived from the graph's edges +
  `sub_graph_edge_attributes`.
- `in_channels_src = input_dim[ds] = n_step_input * len(data_indices[ds].model.input) +
  node_attributes.attr_ndims[ds]` — needs the built `node_attributes`.
- `in_channels_dst = input_dim_latent = node_attributes.attr_ndims[hidden]`.
- decoder `in_channels_dst = target_dim[ds]`, `out_channels_dst = output_dim[ds]`.

So the Builder order is: **node_attributes → dims → graph_providers → encoder/processor/
decoder → residual/boundings → model**. `target_dim` and `input_dim` are POLYMORPHIC across
variants (auto-encoder overrides `target_dim` to a forcing-based formula; ensemble adds
`+1` for fcstep and, if `condition_on_residual`, `+ num_input_channels_prognostic`). Hence
the ModelBuilder exposes `input_dim`/`input_dim_latent`/`target_dim`/`output_dim` as
overridable methods, and the ModelBuilder subclasses mirror the model class hierarchy.

The model ALSO computes these dims in `_calculate_shapes_and_indices` (for runtime
metadata, e.g. `fill_metadata` uses `self.input_dim`); that is fine — it recomputes the
same values from the injected `node_attributes`, so Builder and model agree.

## 3. `layer_kernels` — the deep P2 leak

Only **3** sites turn a `layer_kernels` config into a built registry:
`layers/mapper.py`, `layers/processor.py`, `layers/ensemble.py`, each doing
`self.layer_factory = load_layer_kernels(layer_kernels)`. Everything below them
(`block.py`, `attention.py`, `conv.py`) already receives the **built** registry (a
`DotDict` of `functools.partial` factories) and uses it directly.

Consequence for P2: `build(encoder_cfg, ...)` inside the Builder passes the raw
`layer_kernels` **config** into the mapper constructor — so the config-driven full-model
build still hands a settings object to a constructor.

DONE so far: `load_layer_kernels` is now idempotent (a built registry — a `DotDict` whose
values are callables that are not `{_target_: ...}` mappings — passes through unchanged) and
Hydra-free (`build(..., _partial_=True)`, raising `BuilderError`). This is the enabler:
a mapper/processor/ensemble can now be constructed with an already-built registry, so a
model can be built with NO settings — proven runnable in `examples/create_model.py`
(`build_from_objects`).

REMAINING for full P2 on the config path: the ModelBuilder should call `load_layer_kernels`
itself and inject the built registry into encoder/processor/decoder/noise_injector, so the
config-driven build also passes an object (not config) to the mappers. (Optionally then
drop the now-idempotent `load_layer_kernels` call from mapper/processor/ensemble.)

## 4. Transport model — settings objects in `__init__`

`AnemoiTransportModelEncProcDec.__init__` reads `transport_params =
model_config.model.model.transport` and builds many value objects
(`NoiseConditioningSettings.from_config`, `EdmSettings`, `StochasticInterpolantSettings`,
`TransportSourceBuilder`, `get_transport_model_objective(...)`), then
`instantiate(transport_params.noise_embedder)` and a small MLP; it also overrides
`_calculate_input_dim`/`_calculate_output_dim`. Per P2 these settings objects + the noise
embedder must be built in the Builder and injected; the constructor should receive built
objects + primitives only. This is why transport is the most involved variant.

## 5. `hierarchical_autoencoder` quirks

It extends `AnemoiModelAutoEncoder` but has a **custom `__init__`** that duplicates base
setup, and notably builds `NamedNodesAttributes(model_config.model.trainable_parameters,
self._graph_data)` from the FULL graph and does NOT call `broadcast_config_keys` (unlike
base). When migrating, verify equivalence with the ModelBuilder's `build_node_attributes`
(reduced graph + broadcast). Its `_build_networks` is the hierarchical one; its decoder
uses the auto-encoder `target_dim`. Builder = HierarchicalBuilder + auto-encoder
`target_dim` override.

## 6. Checkpoint safety

`state_dict` keys are derived from `nn.Module` attribute names, not constructor signatures.
The refactor preserves every `self.encoder`/`self.processor`/`self.decoder`/
`self.encoder_graph_provider`/`self.node_attributes`/`self.residual`/`self.boundings`/…
name, so existing checkpoints load unchanged. Migration scripts manipulate checkpoint
dicts directly (no `instantiate`) and are unaffected.

## 7. `_recursive_=False` / `_convert_`

In the original code `instantiate(encoder_cfg, _recursive_=False, ...)` used
`_recursive_=False` specifically to STOP Hydra from building the nested `layer_kernels`
config early (the mapper builds it itself). `anemoi.utils.builder.build` supports the same
`_recursive_` / `_convert_` keys, so the verbatim move keeps behaviour. Once layer kernels
are injected as built objects (§3), `_recursive_` is no longer semantically needed for that
purpose.

## 8. Config shape + dispatch

- The model config subtree passed around is `config.model`; the inner model spec is
  `config.model.model` with `config.model.model._target_`. Encoder/processor/decoder specs
  are `config.model.encoder|processor|decoder`; residual is `config.model.residual`;
  boundings `config.model.bounding`; trainable params `config.model.trainable_parameters`.
- Configs use the re-export alias `anemoi.models.models.AnemoiModelEncProcDec` (not the
  full module path). Therefore the ModelBuilder registry is keyed by the resolved **class**
  (`locate(target)`), so any alias that resolves to the class selects the right builder.

## 9. What is validated vs not (as of writing)

- Validated end-to-end: `build_model` builds a real `AnemoiModelEncProcDec` (GNN) with all
  networks injected — `models/tests/models/test_base_graph_model.py` +
  `test_model_builder.py`, models suite 521 pass.
- `examples/create_model.py` runs BOTH the config-driven `build_model` path AND a pure
  `build_from_objects` path that constructs the model from built objects + primitives with
  NO settings object (enabled by the idempotent `load_layer_kernels`).
- Partial P2 on the CONFIG path: the config-driven build still hands `layer_kernels` config
  to mappers (§3, "REMAINING"). The pure-object path is fully P2-clean.
- Not migrated: transport, hierarchical_autoencoder (construction raises a clear
  "no ModelBuilder registered" error until done).
- Not started: training package.

## 10. Environment findings

- Edit the editable-installed anemoi-utils at `/lus/.../git/anemoi-utils` (the copy nested
  in anemoi-core is a stray, not on `sys.path`).
- `graphs`/`models`/`training` are now `pip install -e` editable → plain `pytest`/imports
  hit the checkout.
- A pre-existing deadlock in the uncommitted anemoi-utils `settings.py` WIP blocked
  `import anemoi.utils.caching` (re-entrant settings init via schema-plugin discovery →
  anemoi-registry → `remote/s3.py` reading `SETTINGS` at import, under a non-reentrant
  lock). Fixed by importing `SETTINGS` lazily inside `remote/s3.py`.
- anemoi-utils' own pytest hangs at collection (conftest imports azurite/obstore). Validate
  a single module by importing it and calling its `test_*` functions directly.
- No GPU on the login node; CUDA work goes via SLURM (`srun`).
