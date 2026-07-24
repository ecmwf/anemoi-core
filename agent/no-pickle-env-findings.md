# No-pickle: environment / dev-setup findings

Notes about running and testing the no-pickle work in this checkout.

## Python environment

* `python` / `python3` in this shell already resolve to the target venv
  `/etc/ecmwf/nfs/dh1_home_a/mab/venv/py312` (`sys.prefix` confirms it). No
  `source .../activate` is needed — and `source` would not persist across separate
  tool invocations anyway (each shell call starts fresh).

## Installed packages are older than the working tree

* The **installed** `anemoi.models` in the venv
  (`.../py312/lib/python3.12/site-packages/anemoi/models`) **predates** the
  `anemoi.models.utils` subpackage, so `import anemoi.models` does **not** see the
  no-pickle changes.
* The working tree with the changes is at
  `/lus/h2resw01/hpcperm/mab/git/anemoi-core/{models,graphs,training}/src`.

### How to run against the working tree

Two options:

1. **PYTHONPATH prefix** (non-invasive, used so far):

   ```bash
   PYTHONPATH=models/src:graphs/src:training/src python -m pytest models/tests/utils/test_instantiate.py -q
   ```

   Note: the repo packages are a namespace package (`anemoi.*`); the first matching
   `anemoi/models` on the path wins, so prefixing `models/src` shadows the installed copy.
   `graphs/src` and `training/src` are needed too because `anemoi.models` imports
   `anemoi.graphs.*`, and that import resolves the installed (old) `anemoi.graphs` unless
   the working-tree one is also on the path.

2. **Editable install** (permanent, no PYTHONPATH):

   ```bash
   pip install -e models/ -e graphs/ -e training/
   ```

   After this `import anemoi.models` resolves to the working tree.

## Test commands that work here

```bash
PYTHONPATH=models/src:graphs/src:training/src python -m pytest \
    models/tests/utils/test_instantiate.py \
    models/tests/layers/test_bounding.py \
    models/tests/models -q
```

* `models/tests/layers/test_layer_utils.py::TestLayerUtils::test_kernel_forward_pass` can
  fail with `hypothesis.errors.DeadlineExceeded` (warm-up timing, ~480 ms > 200 ms
  deadline) — it is a pre-existing flake unrelated to the changes; it passes when run in
  isolation.
