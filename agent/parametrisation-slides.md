---
marp: true
paginate: true
style: |
  section { font-size: 24px; }
  h1 { font-size: 34px; }
  h2 { font-size: 29px; }
  h3 { font-size: 25px; }
  pre, code { font-size: 0.84em; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  table { font-size: 0.9em; }
---

<!--
Slide deck (Marp). Render: agent/render-slides.sh agent/parametrisation-slides.md
-->

# Hydra-free construction with `Parametrisation`

**The Great Refactoring — anemoi-core + anemoi-utils**

A single, small abstraction for **parameter-based instantiation** of models,
graphs and training objects — replacing hard-wired Hydra `instantiate`.

---

## The problem

- Models, layers, graphs and training objects were all built by **Hydra
  `instantiate`** on config trees (`_target_`, `_recursive_`, …).
- Consequences: **Hydra + the whole training stack pulled into inference**;
  models only reconstructable by **unpickling**; **no way to build a model in
  Python** without a YAML config.

---

## `Parametrisation`: one abstract type, three services

**`Parametrisation` is an abstract base class** (`anemoi.utils.parametrisation`). Concrete
subclasses (`DictParametrisation`, `HydraParametrisation`) plug in *how* objects are built,
but every constructor only ever sees the abstract type. It provides **three services**:

- **Carry config** — `get("model.num_heads", default)`: dotted-key access to parameters
- **Act as a factory** — `create_module(spec, **runtime)`: build (or return) objects
- **Serialise** — `to_dict()` / `from_dict()`: round-trip through a plain dict (which the checkpoint stores as JSON)

Because it is abstract, the *policy* (Hydra vs Hydra-free) is swappable without touching a
single call site.

---

## The idea: parameter-based instantiation

The abstraction is threaded through *every* constructor as `params`. Objects are created
*from parameters*, via a single entry point:

```python
def __init__(self, params: Parametrisation, *, encoder=None, ...):
    self.encoder = self._create_submodule("encoder", encoder, GraphTransformerForwardMapper, ...)
```

**`create_module(spec, **runtime)`** dispatches on what it is given:

- a **class** → instantiated with runtime args
- a **dotted string / `_target_` dict** → built from parameters
- an **already-built instance** → used as-is
- **`None`** → `None`

A sub-module's class is resolved in priority order: an **explicit constructor override**, then
the **`model.<sub>._target_`** named in the parameters, then the **code default class**.
Runtime dimensions are always injected.

---

## What it unlocks (1/3) — Hydra backward compatibility

Two implementations behind one interface — call sites never change:

- **`DictParametrisation`** — Hydra-free, JSON-backed (`Parametrisation.from_dict(...)`).
- **`HydraParametrisation`** — delegates `create_module` to `hydra.utils.instantiate`
  (imported lazily).

The **training launcher and all existing YAML configs work unchanged.**

---

## What it unlocks (2/3) — Pickle-free checkpoints

A checkpoint carries its parametrisation as **JSON**. Inference rebuilds the model from
`Parametrisation.from_dict(metadata)` + weights — **no unpickling, no Hydra, no training stack.**

```python
model = AnemoiModelInterface(params=Parametrisation.from_dict(meta["config"]), **inputs)
model.load_state_dict(weights)
```

---

## What it unlocks (3/3) — Object injection: build a model *without a config*

Any sub-module can be passed as a **built object**, a **class**, or a **dotted string** — and mixed:

```python
params = Parametrisation.from_dict({...})                       # or from a checkpoint's dict
model  = AnemoiModelEncProcDec(params, **data)                  # all code defaults
model  = AnemoiModelEncProcDec(params, processor=MyProcessor(...))   # inject an instance
model  = AnemoiModelEncProcDec(params, residual="pkg.SkipConnection") # inject by name
```

Graphs get the same treatment: `GraphBuilder(nodes=[...], edges=[...])` from real objects.

---

## Breaking changes to be aware of

- **`AnemoiModelInterface(params=...)`** replaces `config=...`; stored as `model.params`.
- **Sub-module class selection** — resolved as: explicit constructor override → the
  `model.<sub>._target_` in the parameters → the code default class. Structural settings
  (`num_heads`, `num_layers`, `layer_kernels`, …) are read from the parametrisation.
- **`load_layer_kernels`** returns a `LayerKernels` (a `Parametrisation`), not a `DotDict`
  (attribute/item access unchanged). Build failures raise **`ParametrisationError`**.
- The `anemoi.models.utils.instantiate` shim and stray `DotDict` usages are **removed**.

**Migration in one line:** build objects through `params.create_module(...)`, and construct
a parametrisation with `Parametrisation.from_dict(...)` — never name a concrete subclass at
the call site.
