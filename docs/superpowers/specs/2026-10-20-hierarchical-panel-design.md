# Design: hierarchical reconciliation before conformal PID on a leaves × forecasters panel

## Context

Phase 1 (`2026-09-29-hierarchical-conformal-pid-design.md`) asked whether reconciling the COVIDhub ensemble's quantiles across the US → 56-leaf hierarchy before conformal control improves the **US** series' temporal coverage. The pre-registered answer was null (D19), and post-hoc diagnostics show why:

- the endpoint has little headroom on one 112-week series;
- quantile-wise reconciliation changes the interval's *scale* (by assuming comonotone leaves), not its location;
- the reconciled US median is barely better than the base;
- PI over-covers on this conservative base however it starts.

Phase 2 moves the question to where reconciliation theory predicts gains and where there is statistical power. The **leaves** are the noisy bottom level that borrows strength from the top, and they are also the paper's own state-level setting. Running **several forecasters** tests whether the effect generalizes and gives miscalibrated base models to work on.

## Research questions

- **H1 (primary, D20):** for `COVIDhub-4_week_ensemble`, does `mint_shrink` reduce the mean leaf coverage deviation compared with `none`, under PI at lr 0.1?
- **H2 (key secondary, D21):** does the same contrast hold across the 8 eligible forecasters, at the leaf-mean and US levels?
- **Mechanism (D26):** `mint_median` keeps the base spread and moves only the location. Its contrast against `none` isolates the location effect from the widening.

## Decisions

D19–D27 in `docs/protocol-decisions.md`. They were registered on 2026-10-20, before any leaf-level outcome was computed. The eligible forecaster list was derived from data availability only.

## Architecture (additions to Phase 1)

```
src/hcp/
  evaluate.py     + evaluate_batch (vectorized over leading axes), coverage_deviation_floor
  reconcile.py    + "mint_median" in RECONCILERS
  data.py         read_forecasts (one read) + build_bundle(frame, forecaster); load_hierarchy unchanged
  eligibility.py  EligibilityRule, eligibility(bundles, rule) -> table (D21)
  panel.py        PanelConfig, run_panel(bundles, cfg) -> PanelResult (numeric only)
  report.py       + render_panel_figures (forest per forecaster, per-leaf strip)
  run.py          RunConfig.design = "us" | "panel"; panel writer; shared manifest builder
scripts/analysis/us_hier_v2.py   v1 diagnostics (D19) -> docs/analysis/
configs/panel_leaves.toml        run_id "panel-leaves-v1", the 8 forecasters
results/panel-leaves-v1/         intervals, metrics, level_metrics, contrasts, summary, eligibility (parquet),
                                 manifest.json, figures/*.pdf
```

`design = "us"` (the default) keeps Phase 1 byte-for-byte reproducible. The panel's US-node metrics on the ensemble must equal `results/us-hier-v2/metrics.parquet` (D27).

## Deferred

- Probabilistic reconciliation from the 23 quantiles with an empirical copula.
- An aligned scorecaster and a causal lr scale (D15).
- Conformalize-then-reconcile, or hierarchical conformal prediction.
- The lasso-QR scorecaster.
- Obsidian binding.
