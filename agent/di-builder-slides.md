---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 24px; }
  section.lead h1 { font-size: 52px; }
  pre, code { font-size: 0.82em; }
  h1 { color: #0b6; }
  table { font-size: 0.8em; }
  .small { font-size: 0.7em; color: #666; }
---

<!-- _class: lead -->

# Dependency-Injection in Anemoi
## A Hydra-free **Builder** across graphs · models · training

Replacing `hydra.utils.instantiate` with explicit object injection

<span class="small">refactor.md · graph.md · REFACTOR_SPEC.md · REFACTOR_FINDINGS.md</span>

---

# The problem

Today, objects build their own polymorphic sub-objects **inside** the constructor, via Hydra:

```python
class AnemoiModelEncProcDec(BaseGraphModel):
    def _build_networks(self, model_config):       # config passed in ...
        self.encoder[ds] = instantiate(            # ... and used to build the encoder, in the class
            model_config.model.encoder,
            _recursive_=False,
            in_channels_src=self.input_dim[ds],
            edge_dim=self.encoder_graph_provider[ds].edge_dim,
        )
```

- Architecture is hidden behind config → no "go to definition", no static typing
- Every class is coupled to Hydra / OmegaConf
- Hard to build a model in plain Python or unit-test in isolation

---

# The goal

**Object injection ("has-a").** Every object receives its polymorphic members as
**already-built objects** passed to its constructor:

```python
class AnemoiModelEncProcDec(BaseGraphModel):
    def __init__(self, *, encoder, processor, decoder, ...):
        self.encoder = encoder        # just store what you are given
        self.processor = processor
        self.decoder = decoder
```

A separate **Builder** reads config and builds the whole tree bottom-up.

> Hydra is removed from construction. Backward compatibility via a single
> `DictConfig → dict` conversion at the boundary. `_target_` shape and `nn.Module`
> attribute names are preserved, so YAML configs *and* checkpoints keep working.

---

# The rules (principles)

| # | Rule |
|---|------|
| **P1** | No `instantiate`/`build` **inside** a constructor. Swapping `instantiate`→`build` in `__init__` is the anti-pattern, not the fix. |
| **P2** *(relaxed)* | A constructor **may** receive parametrisation (settings, `..._params`, `layer_kernels` choice, scalars) — but **must not use it to call a factory** that builds a polymorphic sub-object. |
| **P3** | The **Builder is the only code that reads configuration**. |
| **P4** | The Builder mirrors object containment: build children → compute runtime values → construct parent with children injected. |
| **P5** | Hydra removed from construction; backward-compat via `as_dict`. |
| **P6** | Scope: `graphs`, `models`, `training` (+ shared `anemoi-utils`). |

---

# The relaxed P2 — the key distinction

**Must be Builder-built and injected** — the interchangeable *architecture components*
a `config.model.<x>._target_` selects:

> encoder · processor · decoder · mappers · residual · boundings · noise_injector /
> noise_embedder · level processors · downscale / upscale · graph providers · node attributes

**May be resolved in-constructor from parametrisation** — leaf configuration that does
*not* build a polymorphic component:

> settings dataclasses (`Settings.from_config(...)`) · scalars / flags / thresholds ·
> `layer_kernels` (which plain `nn` layer classes to use)

<span class="small">Rationale: "no settings at all" made value-object-heavy variants (transport) needlessly painful. The real target is *architecture is injected*, not that every scalar is threaded by hand.</span>

---

# The shared engine — `anemoi.utils.builder`

```python
locate(path)                      # dotted import path -> object (replaces get_class)
as_dict(config)                   # DictConfig/ListConfig -> plain dict/list  (the Hydra boundary)
build(spec, /, **injected)        # {"_target_": ..., **params} -> one object
build_all(specs, /, **injected)   # map build over a dict/list, same injected kwargs to each
to_dict(obj)  /  introspect(obj)  # the INVERSE of build: object -> JSON-able spec
```

```python
>>> build({"_target_": "torch.nn.Linear", "in_features": 4}, out_features=2)
Linear(in_features=4, out_features=2, bias=True)
>>> to_dict(_)
{'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 2, 'bias': True, ...}
```

- Honours `_target_`, `_partial_`, `_recursive_`, `_convert_`
- **Only ever called from Builders** (P1)

---

# Two layers

**Layer 1 — classes become containers**
Polymorphic members are constructor parameters; the constructor stores them and computes
cheap runtime bookkeeping. No config, no building.

**Layer 2 — Builders read config and build bottom-up**
Per-package Builder classes mirror the object hierarchy and encode the wiring. Dispatch is
by the object's `_target_`, resolved to a **class** (so re-export aliases resolve).

```text
config ──as_dict──▶ dict ──Builder──▶ builds leaves ─▶ computes runtime values
                                     ─▶ builds parents(children=...) ─▶ object tree
```

---

# Graphs — before / after

**Before** (`GraphCreator.update_graph`):
```python
graph = instantiate(nodes_cfg.node_builder, name=name).update_graph(
    graph, attrs_config=nodes_cfg.get("attributes", {}))     # builds attrs from config inside
```

**After** — attributes are built objects injected into the builder:
```python
node_builder = build(nodes_cfg["node_builder"], name=name)   # in GraphCreator (a Builder)
attributes   = build_all(nodes_cfg.get("attributes", {}))
graph = node_builder.update_graph(graph, attributes)         # receives built attrs
```

`GraphCreator` (config-driven) subclasses `GraphBuilder` (pure object API). ✅ 313 tests pass.

---

# Graphs — pure Python object API

The `graph.md` vision, now real — attributes are constructor kwargs:

```python
from anemoi.graphs.create import GraphBuilder
from anemoi.graphs.nodes import ReducedGaussianGridNodes
from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.edges.attributes import EdgeLength

builder = GraphBuilder(
    nodes=[ReducedGaussianGridNodes(grid="o48", name="data",
                                    attributes=[SphericalAreaWeights(norm="unit-max")])],
    edges=[KNNEdges(source_name="data", target_name="data", num_nearest_neighbours=3,
                    attributes=[EdgeLength(norm="unit-std")])],
)
graph = builder.create()
```

Config path and object path share the same `GraphBuilder` base — and it **round-trips**
through `to_dict`/`build` (see serialisation).

---

# Models — the `ModelBuilder`

```python
def build_model(model_config, *, data_indices, statistics, graph_data,
                n_step_input, n_step_output) -> BaseGraphModel:
    model_cls   = locate(model_config.model.model["_target_"])   # alias-safe
    builder_cls = ModelBuilder.registry[model_cls]               # dispatch by class
    return builder_cls(model_config, ...).build()
```

The builder builds **everything** and injects it:
`node_attributes → graph_providers → encoder/processor/decoder → residual → boundings → model`.

The model classes become pure containers. State-dict keys unchanged → **checkpoints load**.

---

# Models — the encoder, before / after

**Before** — inside the model:
```python
self.encoder[ds] = instantiate(model_config.model.encoder, _recursive_=False,
    in_channels_src=self.input_dim[ds], edge_dim=self.encoder_graph_provider[ds].edge_dim, ...)
```

**After** — inside `AnemoiModelEncProcDecBuilder.build_networks()` (a Builder):
```python
provider     = self.create_graph_provider(encoder_cfg, ds, hidden)
encoder[ds]  = build(encoder_cfg, _recursive_=False,
                     in_channels_src=self.input_dim(ds),
                     in_channels_dst=self.input_dim_latent(),
                     hidden_dim=self.num_channels, edge_dim=provider.edge_dim)
# ... then: AnemoiModelEncProcDec(encoder=encoder, processor=..., decoder=..., ...)
```

---

# Why construction is bottom-up (the dim wiring)

The encoder/decoder need runtime values that only exist mid-build:

```text
node_attributes ─▶ attr_ndims ─┐
                                ├─▶ input_dim / latent / target / output
data_indices ──────────────────┘
graph edges ─▶ graph_provider ─▶ edge_dim
```

`target_dim` / `input_dim` are **polymorphic** across variants
(auto-encoder overrides `target_dim`; ensemble adds fcstep + conditional prognostic).

→ the `ModelBuilder` exposes them as overridable methods; builder subclasses mirror the
model class hierarchy.

---

# Example 1 — create a model from config

```python
from anemoi.models.models.builder import build_model

model = build_model(
    model_config,                       # the usual Hydra/YAML config
    data_indices=data_indices,
    statistics=statistics,
    graph_data=graph,
    n_step_input=1, n_step_output=1,
)
# -> AnemoiModelEncProcDec with encoder/processor/decoder/residual/boundings injected
```

The `ModelBuilder` is the **only** code that touched the config.

<span class="small">examples/create_model.py :: build_from_config()</span>

---

# Example 2 — create a model with **no settings**

```python
encoder = nn.ModuleDict({"data": GNNForwardMapper(
    in_channels_src=input_dim, in_channels_dst=latent_dim, hidden_dim=16,
    edge_dim=enc_gp["data"].edge_dim, num_chunks=1, mlp_extra_layers=0,
    layer_kernels=load_layer_kernels(None))})          # a built registry (an object)

model = AnemoiModelEncProcDec(
    encoder=encoder, processor=processor, decoder=decoder,
    encoder_graph_provider=enc_gp, processor_graph_provider=proc_gp,
    decoder_graph_provider=dec_gp, node_attributes=node_attributes,
    residual=residual, boundings=boundings,
    data_indices=data_indices, statistics=stats, graph_data=graph,
    n_step_input=1, n_step_output=1,
    hidden_nodes_name="hidden", num_channels=16, latent_skip=True)
```

Only ints/strings/bools + built objects — **no `DictConfig` reaches any constructor.**
<span class="small">examples/create_model.py :: build_from_objects() — runs.</span>

---

# Transport — the relaxed rule in action

```python
class AnemoiTransportModelEncProcDec(AnemoiModelEncProcDec):
    def __init__(self, *, noise_embedder, transport_params, **base_kwargs):
        # parametrisation -> settings value objects  (allowed: not a polymorphic factory)
        self.noise_conditioning = NoiseConditioningSettings.from_config(transport_params)
        self.edm                = EdmSettings.from_config(transport_params)
        ...
        super().__init__(**base_kwargs)
        self.noise_embedder = noise_embedder            # polymorphic -> INJECTED
```

The builder builds the polymorphic `noise_embedder` and passes `transport_params` as data:
```python
networks["noise_embedder"]   = build(self.transport_params.noise_embedder)
networks["transport_params"] = self.transport_params
```

---

# Backward compatibility

- **Configs**: unchanged. `as_dict(config)` converts `DictConfig → dict` once at the boundary;
  the `_target_` convention is preserved.
- **Checkpoints**: every `self.encoder` / `self.processor` / `self.node_attributes` / …
  attribute name is unchanged → `state_dict` keys unchanged → **existing checkpoints load**.
- **`load_layer_kernels`** is now Hydra-free and *idempotent*: an already-built registry
  passes through, so a Builder can construct kernels once and inject them.

---

# Serialisation — the inverse of `build`

`to_dict(obj)` turns a built object back into a JSON-able `{"_target_": ...}` spec, so
**`build(to_dict(obj))` reconstructs it**.

An object serialises via, in order:

1. a recorded `__anemoi_spec__` (e.g. a model's architecture recipe),
2. an **`as_dict()`** method (custom control), or
3. introspection — `_target_` + each `__init__` param read from the same-named attribute.

`torch.dtype` (→ `"float32"`), numpy arrays/scalars (→ lists) and `os.PathLike` (→ str) are
handled, so *transformed* constructor args round-trip too.

---

# Serialise a model → JSON → rebuild

```python
model = build_model(config, data_indices=…, graph_data=graph, …)
spec  = to_dict(model)                       # JSON-able architecture recipe
text  = json.dumps(spec)                     # dump / store anywhere

model2 = build_model(json.loads(text),       # rebuild from the recipe
                     data_indices=…, graph_data=graph, …)
assert set(model.state_dict()) == set(model2.state_dict())   # identical architecture
```

The **recipe** (architecture) is serialised; weights + runtime context (graph, indices) are
supplied again at rebuild. `build_model` accepts OmegaConf, `DotDict` or plain JSON `dict`.

---

# Serialise a graph and its objects

```python
spec = to_dict(graph_builder)                # {_target_: GraphBuilder, nodes:[…], edges:[…]}
same = build(json.loads(json.dumps(spec)))   # -> identical GraphBuilder (attributes preserved)

>>> to_dict(EdgeLength(norm="unit-std"))
{'_target_': 'anemoi.graphs.edges.attributes.EdgeLength', 'norm': 'unit-std', 'dtype': 'float32'}
```

Graph builder + attribute classes carry an `as_dict()`.
<span class="small">examples/serialise_roundtrip.py · graphs/tests/test_graph_builder_api.py — green.</span>

---

# Status

| Package | State | Tests |
|---|---|---|
| `anemoi.utils.builder` | ✅ build · build_all · as_dict · locate · **to_dict / introspect** | 27 |
| **graphs** | ✅ `GraphBuilder`/`GraphCreator`, object API + serialisation | 313 (+3) |
| **models** | ✅ `ModelBuilder`, all 6 variants, no Hydra, serialisation | 521 |
| **training** | ✅ trainer + factories build via `build`/`locate` | 950 |

`grep "instantiate(" {graphs,models,training}/src` → nothing (Hydra kept only as the config
*loader*). Examples: `create_model.py` (config + no-settings), `train_model.py` (GPU/A100),
`serialise_roundtrip.py` (round-trip).

---

# Training — same recipe

The training **builder** = `AnemoiTrainer` + the LightningModule assembly + factory functions
(`get_loss_function` · `create_scalers` · `get_callbacks` · `get_*_logger` · checkpoint pipeline).

```python
# builder-side (reads config): instantiate(...) -> build(...) ; get_class(...) -> locate(...)
optimizer = build(optimization_config.optimizer, params=params, lr=self.effective_lr)
model     = build(training_method_cfg, task=task, graph_data=graph, ...)  # -> LightningModule
```

Relaxed P2: the LightningModule receives `config` (parametrisation) and assembles via the
factories; the **model architecture underneath is already injected** by the `ModelBuilder`.

---

<!-- _class: lead -->

# Done — all four packages

utils ✅ · graphs ✅ · models ✅ · training ✅ · **serialisation ✅**

Hydra removed from construction everywhere; kept only as the config loader.
Configs and checkpoints unchanged. A model/graph can be built, **trained**, and
**serialised to JSON and rebuilt** — with no settings.

<span class="small">REFACTOR_SPEC.md · REFACTOR_FINDINGS.md · examples/create_model.py · examples/train_model.py</span>
