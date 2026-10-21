# Plan — Hierarchical reconciliation (state → US) before Conformal PID

## Context

This fork of *Conformal PID Control for Time Series Prediction* (Angelopoulos et al., 2023) conformalizes the **CDC COVIDhub-4_week_ensemble** q0.1/q0.9 forecasts of 4-week-ahead **statewide** COVID-19 deaths. Lasso-QR in the notebook is only the scorecaster, and the notebook explicitly drops `geo_value == "us"`.

**Research question:** does reconciling base forecasts across the hierarchy (56 leaves → US) before conformal control improve the **temporal coverage** of the US series?

**What we found in exploration:**

- **Data is complete.** `data/deaths.csv` (gitignored) is byte-identical to the paper's LFS input: SHA-256 `cb4dccfd…ac06`.
  - It contains us + 51 states/DC + 5 territories, 78 forecasters, the 23-quantile grid, and a constant 26-day lag between forecast date and target.
  - 132 weeks (2020-07-25 → 2023-03-04) have all 57 nodes complete.
- **Leaves must include the territories.** With them, US − Σleaves has a median of 1 death/week; with the 51 states/DC only, it is about 45. Rare spikes of ±600 are truth revisions.
- **Base US medians are incoherent**: (US − Σleaves)/US averages 4.5%, ranging from −23% to +27%.
- **The base US 80% interval already covers 82.6% of weeks.** Marginal coverage is therefore a weak endpoint, and the primary endpoint has to be temporal.
- **`core/methods.py` crashes on NumPy 2** because it uses `np.infty` (about 10 sites, plus `core/quantile.py:8`). There are no pytest tests. `tests/` is the paper's experiment harness.
- **`net-sci-epi` provides the architecture to mirror:**
  - a flat `src/` package
  - TOML configs loaded into frozen dataclasses
  - a module-level `dict` as the registry
  - numeric modules that do no I/O; only `run.py` writes
  - `results/<run_id>/` created with `exist_ok=False`, holding `manifest.json`, parquet and npz files
  - `SeedSequence` for seeds
  - an argparse CLI per stage
  - pytest markers `slow` / `integration`
  - `docs/protocol-decisions.md`

  Its `src/headd_l0/reconcile.py` (summing matrix, bottom-up, MinT(Shrink) with the Schäfer–Strimmer λ) is ported almost verbatim.

## Decisions (user-approved)

| # | Decision |
|---|---|
| D1 | Data source: `data/deaths.csv`, forecaster `COVIDhub-4_week_ensemble`, `ahead = 4`. The SHA-256 is verified at load. |
| D2 | Order: **reconcile, then conformalize**. Reconcile the base quantiles q0.1/q0.5/q0.9 quantile-wise into a coherent US forecast, then compute cqr-asymmetric scores, then run the controller. |
| D3 | Code layout: new package `src/hcp/`. It **imports `core.methods`**. `core/` gets only the `np.infty → np.inf` fix, and only after characterization tests are in place. |
| D4 | Controllers: **primary is PI** (`core.methods.quantile_integrator_log`, `Csat = 2`, `KI = 1000`, `T_burnin = 5`, α = 0.2, split 0.1 per side). **Secondary is PID-Theta** (`quantile_integrator_log_scorecaster`, core's univariate Theta). A `raw` control uses lr = 0 (base forecaster). The lasso-QR scorecaster is deferred. |

## Decisions I'm proposing (pre-registered in `docs/protocol-decisions.md`)

| # | Decision |
|---|---|
| D5 | Hierarchy nodes: `us` + 56 leaves (51 states/DC + `as`, `gu`, `mp`, `pr`, `vi`). S is 57×56. |
| D6 | Evaluation grid and missing data. The grid is the weeks where the US forecast and truth are complete. A missing leaf quantile is filled causally with LOCF, and the count of imputed cells goes in the manifest. Truth is never imputed. The last 3–4 weeks have no actuals and are dropped. |
| D7 | No look-ahead. The MinT covariance for target t uses only median errors of targets s ≤ `forecast_date(t)` (in practice s ≤ t − 28 days), on an expanding window. The first `mint_warmup = 20` weeks are excluded from evaluation **for every arm**, so all arms share one window. |
| D8 | Quantile crossing after MinT is handled by sorting the three levels per node and week. The number of crossings is recorded. |
| D9 | Truth for evaluation is the **reported US actual**, which is what the paper would score. The gap between truth and the coherent sum is logged as a diagnostic. |
| D10 | **Primary endpoint:** mean \|rolling coverage(w = 10) − 0.8\| on the US series. Primary contrast: `mint_shrink` vs `none` under PI at lr = 0.1, the paper's COVID lr. |
| D11 | **Secondary endpoints:** `bottom_up` vs `none`; longest miscoverage run (the paper's metric); marginal coverage; mean and median width; 80% interval score. Holm-corrected. |
| D12 | Statistics: paired moving-block bootstrap over weeks with block = 8 (sensitivity 4 and 12), B = 2000, a single `root_seed` fed to `SeedSequence`. Sensitivity also covers the lr grid {1, 0.5, 0.1, 0.05}. |

## Architecture

```
core/                     upstream; only change is np.infty → np.inf
src/hcp/
  __init__.py
  data.py        load_hierarchy(path, cfg) → HierarchyBundle(frozen): dates, forecast_dates,
                 nodes, y[57,T], q[57,T,3]; sha256 check, LOCF + imputation mask, read-only arrays
  reconcile.py   summing_matrix, shrinkage_covariance, RECONCILERS={"none","bottom_up","mint_shrink"},
                 online_mint(bundle, warmup) → coherent q (causal expanding covariance)
                 [port of net-sci-epi/src/headd_l0/reconcile.py, generalized to n leaves]
  conformal.py   CONTROLLERS={"raw","pi","pid_theta"}; conformalize(q_us, y_us, controller, lr, cfg)
                 → (lo, hi); mirrors tests/base_test.py cqr-asymmetric: scores [q.1−y, y−q.9],
                 α/2 per side, sets [q.1−q_lo, q.9+q_hi]
  evaluate.py    rolling_coverage, coverage_deviation, longest_miss_run, width, interval_score
  stats.py       block_bootstrap_diff(metric, a, b, block, B, rng) → estimate, CI
  report.py      figures (OO matplotlib): rolling coverage per arm, US intervals over time, contrast CIs
  run.py         argparse CLI: `uv run python -m hcp.run configs/us_hier.toml`
configs/us_hier.toml     run_id, root_seed, data_path, expected_sha256, alpha, lr, lr_grid,
                         reconcilers, controllers, mint_warmup, rolling_window, block, n_boot
results/<run_id>/        manifest.json, intervals.parquet (long: week, reconciler, controller, lr,
                         lo, hi, y), metrics.parquet, contrasts.parquet, figures/*.pdf
tests/unit/ tests/integration/ tests/e2e/   (pytest; paper harness in tests/*.py untouched)
docs/protocol-decisions.md, docs/development.md, docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md
AGENTS.md                 conventions, mirroring net-sci-epi's
```

The arms are {none, bottom_up, mint_shrink} × {raw, pi, pid_theta} at lr = 0.1, plus a sensitivity pass over the lr grid.

## Implementation steps (TDD; each step is red → green → commit)

0. **Housekeeping.**
   - Ask before committing the existing uncommitted uv migration and ruff reformat as their own commits (`build: migrate to uv`, `style: ruff format`).
   - Branch `feat/hierarchical-reconciliation` off `develop`.
   - `pyproject.toml`:
     - dev group: `pytest`, `ruff`
     - runtime: `pyarrow`
     - hatchling `packages = ["src/hcp", "core"]`
     - `[tool.pytest.ini_options]`: `testpaths = ["tests/unit","tests/integration","tests/e2e"]` (so `tests/base_test.py` isn't collected), `pythonpath = ["src","."]`, markers `integration` and `slow`
2. **Protect `core/`** (skill `working-with-legacy-code`).
   - `tests/unit/test_core_characterization.py` monkeypatches `np.infty = np.inf`, runs `quantile` and `quantile_integrator_log` on seeded synthetic scores, and pins the outputs to `tests/fixtures/core_golden.npz`.
   - Then replace `np.infty` with `np.inf` in `core/methods.py` and `core/quantile.py`, and drop the monkeypatch. The tests must stay green.
3. **`data.py` + tests.**
   - Synthetic CSV fixture in `tests/conftest.py`, written in the real column schema.
   - Tests cover node ordering, SHA mismatch raising `ValueError`, LOCF causality, and truth never imputed.
   - **Integration** (`@integration`, real file): for the 7 states that have pickles, `y` and q0.1/q0.9 must equal `tests/datasets/covid-ts-proc/statewide/*_proc_4wkdeaths.pkl` on the overlapping weeks. This is the parity check with the paper.
4. **`reconcile.py` + tests.**
   - Port the net-sci-epi functions.
   - Property tests: output is coherent (S·leaves = US), bottom-up equals the leaf sum, and MinT is a projection (P² = P).
   - `online_mint` is prefix-invariant: truncating the future doesn't change the past.
   - `ValueError` on an unknown method.
5. **`conformal.py` + tests.**
   - `raw` gives [q.1, q.9].
   - With PI on a constant series, coverage converges to about 0.8.
   - The asymmetric wiring matches `tests/base_test.py` on the tx pickle, in both a unit test and an integration test.
   - `pid_theta`: verify core's `.cache/scorecaster/{config_name}` behavior and pass a unique `config_name` per (run_id, arm, side) so no stale cache is reused.
6. **`evaluate.py`, `stats.py` + tests.** Hand-computed metrics on toy arrays. Bootstrap tests check determinism for a fixed seed and CI coverage on synthetic data (`@slow`).
7. **`report.py`, `run.py` + e2e.** `tests/e2e/test_run.py` runs `main()` on the synthetic hierarchy CSV and checks for all artifacts, the manifest fields (config, git SHA and dirty flag, input SHA-256, seed, numpy/pandas versions, imputation and crossing counts), and that a second run with the same `run_id` raises `FileExistsError`.
8. **Docs:** the spec file, `docs/protocol-decisions.md` (D1–D12), `docs/development.md` (commands), `AGENTS.md`, and a README section.

Skills per phase:
- Step 2: `working-with-legacy-code`
- Steps 3–7: `superpowers:test-driven-development` and `property-based-testing`
- Figures: `dataviz`
- Execution: `superpowers:subagent-driven-development`
- Wrap-up: `analyze-results`

## Verification

- `uv run ruff check . && uv run ruff format --check . && uv run pytest` must pass with clean output.
- `uv run pytest -o addopts='' -m integration` exercises parity with the paper's pickles and `tests/base_test.py`.
- `uv run python -m hcp.run configs/us_hier.toml` writes `results/us-hier-v1/` with all artifacts.
- Sanity check: the `none × raw` arm gives US marginal coverage of about 0.826 on the full grid, matching the exploration figure.
- The rolling-coverage figure for `none × pi` should look like the paper's COVID coverage plots.

## Out of scope / follow-ups

- Porting the lasso-QR scorecaster as the "hierarchy via scorecaster" arm.
- Conformalize-then-reconcile.
- Leaf-level coverage effects.
- Binding the repo to Obsidian project memory (`obsidian-project-bootstrap`); this needs confirmation.
