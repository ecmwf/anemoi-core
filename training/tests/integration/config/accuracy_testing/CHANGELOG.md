# Accuracy test changelog

Changes to the reference run (`reference_id` in `training/tests/integration/test_accuracy.py`)
or to the accuracy test configuration, with the reason and the PR.
Add a new entry at the top whenever either changes.

## 2026-10-09 — torch 2.11 ([#1455](https://github.com/ecmwf/anemoi-core/pull/1455))
- Reference run: `8b3bc3d3bd6947318cb825d777c9d750`
- Reason: Torch releases are not bit-reproducible; no visible effect on the loss.

## 2026-09-17 — pyg-lib support ([#1407](https://github.com/ecmwf/anemoi-core/pull/1407))
- Reference run: `eddb276a2bbd40f6a856b465cc9a995d`
- Reason: Use pyg-lib for graphs.

## 2026-07-24 — seed overflow fix ([#1263](https://github.com/ecmwf/anemoi-core/pull/1263))
- Reference run: `527c02a5f2554f1a96faa0f3e3a9f1a3`
- Reason: The seed overflow fix ([#1228](https://github.com/ecmwf/anemoi-core/pull/1228)) changed the seeding.

## 2026-07-14 — initial loss convergence test ([#1201](https://github.com/ecmwf/anemoi-core/pull/1201))
- Reference run: `684814cd86ed45f383bed3f0a87a782b`
- Reason: Initial reference.
