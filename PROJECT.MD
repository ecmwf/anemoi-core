Source: `/home/ecm1715/prototype/aifs-dev/IMERG.md` (copied here for context — edit the
source file, not this copy, if it changes).

Note: this source doc uses "ERA5" loosely to mean the OPER AN dataset
(`aifs-od-an-oper-...zarr`), not the ERA5 reanalysis product itself. See
[MUST_FOLLOW.MD](MUST_FOLLOW.MD) for the corrected terminology.

---

# IMERG × ERA5 precipitation supervision — analysis & roadmap

Notes on blending two precipitation ground-truths (ECMWF oper/an ERA5 `tp` and NASA
IMERG) as training targets for AIFS 2.0, why the current "combined" run lands in the
middle, and how to push it toward "best of both worlds."

Reference config: [`aifs/single-mse/single_2.0_rollout_esuite_2024_IMERG_combined.yaml`](aifs/single-mse/single_2.0_rollout_esuite_2024_IMERG_combined.yaml)
Scaler definitions: [`aifs/single-mse/training/scalers/global_single2.0.yaml`](aifs/single-mse/training/scalers/global_single2.0.yaml)

---

## 1. Background

- **Full swap experiment** (train `tp` against IMERG instead of ERA5): improvement in
  the **tropics**, degradation in the **extratropics**.
- **Combined experiment** (train against both): a middle ground — slightly *less*
  tropical improvement and slightly *less* extratropical degradation than the swap.
- **Goal:** recover the tropical gain *and* the extratropical skill simultaneously
  (region-dependent trust in each source).

### Why each source is trustworthy where it is

| Region | Preferred source | Reason |
|---|---|---|
| Tropics / ocean | **IMERG** | PMW+IR retrievals calibrated with GPM CO / gauges (Final run); convective precip; sparse conventional obs for reanalysis |
| Extratropics / data-rich land | **ERA5 / oper-an** | Large-scale synoptic precip well constrained by dense assimilated observations |
| High latitudes (> ~55–60°) | **ERA5** | IMERG PMW retrieval fails over snow/ice; morphing-only, low skill |
| Complex terrain / coasts | caution | Both degrade; IMERG has known coast/orographic artefacts |

---

## 2. How the current "combined" config actually weights `tp`

The combined loss (`anemoi.training.losses.combined.CombinedLoss`) has three terms:

```yaml
losses:
  - MSELoss  # all variables vs their own targets (incl. tp vs ERA5 tp)
  - MSELoss  # predicted tp vs IMERG   (predicted_variables=[tp], target_variables=[imerg])
  - MSELoss  # predicted tp vs ERA5 tp (monitoring only)
loss_weights: [1.0, 0.01, 0.0]
```

Both tp-bearing terms also pass through the `general_variable` scaler, where
`tp: 0.025`. **Crucial subtlety:** term 1 is an MSE *averaged over all ~100 output
variables*, so `tp`'s share there is diluted by ~1/N_vars; term 2 is `tp`-only, so it
is **not** diluted. The **effective** per-`tp` gradient budget is therefore:

```
ERA5-tp  (term 1):  loss_weight 1.0  × general_variable(tp) 0.025 / N_vars(~100) ≈ 2.5e-4
IMERG-tp (term 2):  loss_weight 0.01 × general_variable(tp) 0.025 / 1            ≈ 2.5e-4
----------------------------------------------------------------------------------------
ERA5 : IMERG  ≈  1 : 1   (roughly balanced)
```

**Takeaway:** the `0.01` loss_weight on the IMERG term was evidently chosen to *match*
the ~1/N dilution of `tp` inside the all-variable term, bringing IMERG-`tp` to parity
with ERA5-`tp`. The blend is ~50/50 on `tp`, **not** ERA5-dominant.

### The mechanism behind the "middle ground"

Because the two `tp` supervisions are ~equally weighted, **MSE against two disagreeing
targets regresses to their mean.** Wherever both sources are active and disagree
(tropics especially), MSE drives `tp` toward the average of the two → **damped
extremes**. A balanced blend landing between the two single-source results is exactly
what mean-collapse predicts: "less improvement *and* less degradation," smeared toward
the middle.

---

## 3. Proposed next step — latitude-weighted source blending

Weight ERA5-`tp` more in the extratropics and less in the tropics; IMERG-`tp` the
inverse. Sound and observationally motivated. Guardrails:

### 3.1 Isolate `tp` — don't scale term 1 by latitude

Term 1 covers **all** variables. A latitude scaler there would reweight `z, t, u, v …`
by latitude too. Restructure into three `tp`-aware terms:

- main MSE over all variables **except `tp`**,
- a dedicated **ERA5-`tp`** term with an *extratropics* latitude weight,
- a dedicated **IMERG-`tp`** term with a *tropics* latitude weight.

Now the latitude weight touches only `tp`.

### 3.2 Partition of unity

Enforce `w_era5(lat) + w_imerg(lat) = const` at every latitude, scaled by a common `tp`
magnitude. This changes *which source* supervises `tp` without changing *how hard* `tp`
is trained per band — otherwise the two effects are confounded and the ablation is
uninterpretable.

### 3.3 Smooth taper, not a hard cut

A step at ±23.5/±30° invites a visible precip **seam** at the boundary. Use a smooth
taper in `sin(lat)` or `|lat|`:

```
# logistic crossover at latitude φ0 with width Δ (degrees)
w_imerg(φ) = 1 / (1 + exp((|φ| − φ0) / Δ))
w_era5(φ)  = 1 − w_imerg(φ)
# hard floor IMERG poleward of ~55–60° (snow/ice PMW failure)
w_imerg(φ) = 0  for |φ| > 58
```

Start point to sweep: `φ0 ≈ 30–35°`, `Δ ≈ 5–8°`.

### 3.4 Preserve the effective `tp` weight when you split terms

Good news: the base blend is already ~1:1, so a latitude tilt starts from balance rather
than a 100× hole. **But** mind the dilution trap: `tp` currently rides *inside* the
all-variable term, where it is divided by ~N_vars. If you pull `tp` out into its own
dedicated per-source terms (§3.1), it *loses* that 1/N dilution — so set the new
dedicated `tp` loss_weights to ~`0.025 / N_vars` (order `1e-3`–`1e-2`), **not** `0.025`,
or you will suddenly train `tp` ~100× harder and destabilise every other variable.

### 3.5 Clean in-repo implementation path

`node_weights` already uses `GraphNodeAttributeScaler` on the `area_weight` node
attribute. Reuse the pattern:

1. Precompute two node attributes on the graph — `tropics_weight`, `extratropics_weight`
   — as the smooth latitude taper above (functions of node latitude).
2. Reference each as a scaler on the respective `tp` sub-loss:
   - IMERG-`tp` term → `scalers: [..., tropics_weight]`
   - ERA5-`tp`  term → `scalers: [..., extratropics_weight]`

No new scaler *class* is needed — only a node attribute plus two
`GraphNodeAttributeScaler` entries. Spatial (per-node) weighting is exactly what
`GraphNodeAttributeScaler` does.

---

## 4. Confounders to pre-empt

- **Climatology mismatch → seams.** IMERG is only *remapped to `tp`'s normalizer stats*;
  its own mean / intensity / drizzle-frequency distribution is not matched to ERA5.
  Blending two climatologies across a band can bake a discontinuity into output `tp`.
  **Mitigation:** quantile-map IMERG to ERA5 (or a common) climatology *before* using it
  as a target, so the latitude blend changes spatial *skill*, not the *climatology*.
- **Circular evaluation.** "Tropical improvement" measured against IMERG is partly
  self-fulfilling if you trained on IMERG there. Verify `tp` per latitude band against an
  **independent third reference** (gauges / GPCP / held-out product) and inspect maps at
  the crossover for banding.
- **MSE mean-collapse for convective extremes.** Even with a clean blend, MSE damps
  tropical extremes. If extremes matter, use an intensity/frequency-aware or
  distributional/quantile term for `tp` (or reuse the CRPS from the ens variant).
- **Accumulation / units.** IMERG zarr is 6 h aggregated and ends 2024-07-31; confirm the
  accumulation window and units line up with ERA5 `tp` (they share the normalizer, which
  assumes the same physical scale).

---

## 5. Experiment backlog

| # | Experiment | What it isolates |
|---|---|---|
| 1 | Restructure into 3 `tp`-aware terms + latitude node-attribute scalers (§3) | The core latitude blend |
| 2 | Crossover sweep: `φ0`, `Δ` (2–3 cheap runs) | Tropics↔extratropics tradeoff frontier |
| 3 | Skill / inverse-error weighting: derive `w(lat)` from offline ERA5-vs-IMERG error variance | Removes hand-tuned profile; OI-style optimal blend |
| 4 | Add land/sea axis: `w(lat, land-sea)` | IMERG-Final gauge-corrected over land, satellite-only over ocean |
| 5 | Bias-correct-then-blend (quantile-map IMERG first) | Whether seams/curvature come from climatology vs skill |
| 6 | Distributional/quantile or CRPS `tp` loss | Recovering convective extremes lost to MSE |

### Instrumentation (do this regardless)

- Keep the existing weight-0.0 `tp`-vs-ERA5 monitor; add a symmetric `tp`-vs-IMERG
  monitor.
- Add per-latitude-band loss logging so you can watch where each source dominates during
  training and catch a seam early.

---

## 6. Summary

- The "middle ground" is expected: the two `tp` supervisions are ~1:1 (the `0.01` IMERG
  weight matches the ~1/N dilution of `tp` inside the all-variable term), and MSE against
  two comparably-weighted disagreeing targets collapses toward their mean.
- Latitude-weighted source blending is the right next step, but must (a) touch only `tp`,
  (b) be a smooth partition of unity, (c) preserve the effective `tp` weight when `tp` is
  split out of the all-variable term (mind the ~1/N dilution), and (d) ideally follow a
  climatology bias-correction to avoid seams.
- Implement via graph node-attribute scalers (`GraphNodeAttributeScaler`) — no new code,
  just two latitude-taper node attributes.
- Validate against an independent reference, per latitude band, and watch for banding at
  the crossover.

---

## 7. Implementation progress (2026-08-21)

### 7.1 Land/sea loss weighting — `ThresholdVariableWeights` (new node-attribute class)

Implements experiment #4's land/sea axis for the **non-combined** IMERG target. Goal:
weight IMERG `tp` supervision differently over land vs ocean (e.g. `1.0` land, `0.2`
sea), since IMERG-Final is gauge-corrected over land but satellite-only over ocean.

Unlike the latitude tapers (§3.5), the land/sea case **did** need a new class. The stock
`GraphNodeAttributeScaler` / boolean node attributes cannot produce a "land = a, sea = b"
two-value remap out of the box:

- the boolean-complement path (`~mask`) asserts a bool dtype (`BooleanNot`), and
- `AnemoiDatasetVariableWeights` uses the raw `lsm` values directly.

So `ThresholdVariableWeights(AnemoiDatasetVariableWeights)` was added in `anemoi-graphs`
([`graphs/src/anemoi/graphs/nodes/attributes/area_weights.py`](graphs/src/anemoi/graphs/nodes/attributes/area_weights.py)):
it reads a dataset variable (e.g. `lsm`) and returns `above_weight` where the value is
≥ `threshold`, else `below_weight`. Exported in
[`attributes/__init__.py`](graphs/src/anemoi/graphs/nodes/attributes/__init__.py) and
registered in
[`schemas/node_attributes_schemas.py`](graphs/src/anemoi/graphs/schemas/node_attributes_schemas.py).
Verified importing and remapping correctly in the `anemoi-combine-code` env on `ag-login1`.

> Corrects §3.5: for the **land/sea** axis a new class *is* required (a plain two-value
> remap of a dataset variable is not expressible with the stock scaler/boolean
> attributes). The **latitude** tapers still need only node attributes + stock scalers.

Wiring (still to do): graph node attribute (`ThresholdVariableWeights`, `variable: lsm`,
`above_weight: 1.0`, `below_weight: 0.2`) → `GraphNodeAttributeScaler` in
`training/scalers/global_single2.0.yaml` → listed under `scalers:` on a **dedicated `tp`
loss term only** (per §3.1, so it does not reweight every variable).

### 7.2 Prototype runner variant — `aifs-single-imerg`

New variant `aifs/aifs-single-imerg/` in the prototype (`aifs-dev`), copied from
`single-mse`: all subconfig folders (`data`, `dataloader`, `diagnostics`, `graph`,
`model`, `system`, `training`), `config.yaml`, the `uvm` symlink, `.platforms/`, and the
IMERG top-level configs (`single_2.0_rollout_esuite_2024_IMERG.yaml`,
`single_2.0_rollout_esuite_2024_IMERG_combined.yaml`, `single_2.0_rollout_esuite_imerg.yaml`,
plus `single_2.0.yaml` phase-1 base and `single_2.0_rollout_esuite_2024.yaml` OPER-AN
reference).

The runner builds the env with **uv** (`uvm sync`), not `pip install -e`; deps come from
`aifs/aifs-single-imerg/.platforms/<platform>/pyproject.toml`.

Deployment plan for the modified `anemoi-graphs` (the new weighting):

1. Test config plumbing on atos-hoppers (AG) with the copied pinned env.
2. Push the `anemoi-core` branch to remote origin.
3. Point `.platforms/atos-hoppers/pyproject.toml` `[tool.uv.sources]` at that branch
   (git source + `subdirectory` / `rev`) for `anemoi-graphs` (and `anemoi-models` /
   `anemoi-training`, to keep the monorepo mutually consistent), so `uvm sync` picks up
   `ThresholdVariableWeights`.
