# Protocol decisions

## 2026-09-29

| # | Decision |
|---|---|
| D1 | Data source: `data/deaths.csv`, forecaster `COVIDhub-4_week_ensemble`, `ahead = 4`. The SHA-256 is verified at load. |
| D2 | Order: **reconcile, then conformalize**. Reconcile the base quantiles q0.1/q0.5/q0.9 quantile-wise into a coherent US forecast, then compute cqr-asymmetric scores, then run the controller. |
| D3 | Code layout: new package `src/hcp/`. It **imports `core.methods`**. `core/` gets only the `np.infty → np.inf` fix, and only after characterization tests are in place. |
| D4 | Controllers: **primary is PI** (`core.methods.quantile_integrator_log`, `Csat = 2`, `KI = 1000`, `T_burnin = 5`, α = 0.2, split 0.1 per side). **Secondary is PID-Theta** (`quantile_integrator_log_scorecaster`, core's univariate Theta). A `raw` control uses lr = 0 (base forecaster). The lasso-QR scorecaster is deferred. |
| D5 | Hierarchy nodes: `us` + 56 leaves (51 states/DC + `as`, `gu`, `mp`, `pr`, `vi`). S is 57×56. |
| D6 | Evaluation grid and missing data. The grid is the weeks where the US forecast and truth are complete. A missing leaf quantile is filled causally with LOCF, and the count of imputed cells goes in the manifest. Truth is never imputed. The last 3–4 weeks have no actuals and are dropped. |
| D6a | Leading weeks in which any leaf has never been forecast are trimmed, since LOCF has nothing to carry. |
| D7 | No look-ahead. The MinT covariance for target t uses only median errors of targets s ≤ `forecast_date(t)` (in practice s ≤ t − 28 days), on an expanding window. The first `mint_warmup = 20` weeks are excluded from evaluation **for every arm**, so all arms share one window. |
| D7a | Every arm, controller and metric starts at `evaluation_start(bundle, mint_warmup)`, and controllers are re-initialized there. |
| D8 | Quantile crossing after MinT is handled by sorting the three levels per node and week. The number of crossings is recorded. |
| D9 | Truth for evaluation is the **reported US actual**, which is what the paper would score. The gap between truth and the coherent sum is logged as a diagnostic. |
| D10 | **Primary endpoint:** mean \|rolling coverage(w = 10) − 0.8\| on the US series. Primary contrast: `mint_shrink` vs `none` under PI at lr = 0.1, the paper's COVID lr. |
| D11 | **Secondary endpoints:** `bottom_up` vs `none`; longest miscoverage run (the paper's metric); marginal coverage; mean and median width; 80% interval score. Holm-corrected. |
| D12 | Statistics: paired moving-block bootstrap over weeks with block = 8 (sensitivity 4 and 12), B = 2000, a single `root_seed` fed to `SeedSequence`. Sensitivity also covers the lr grid {1, 0.5, 0.1, 0.05}. |
| D13 | `pid_theta` runs in a temporary working directory, so core's `.cache/scorecaster` is never reused across runs. |
| D14 | The upstream Theta scorecaster in `core/methods.py:250` assigned the whole `ThetaModel.forecast(ahead)` Series into a scalar slot and crashed under current numpy/pandas for every `ahead`. It now takes the ahead-step value, `model.forecast(ahead).iloc[-1]`. This is the only change to `core/` besides `np.infty` → `np.inf`, and both are pinned by `tests/unit/test_core_characterization.py`. |

## 2026-09-29 (continued)

| # | Decision |
|---|---|
| D15 | **Upstream properties kept, documented.** (1) core's proportional learning rate `lr_t = lr * range(scores[t-T_burnin:t])` sets `q[t+1]` using scores up to `t-1` while only scores up to `t-ahead+1` are known — ~2 weeks of look-ahead in the step size for PI/PID at `ahead = 4`; inherited from the paper, identical across arms so paired contrasts stay fair, but PI/PID are not strictly causal in the lr scale. (2) the Theta scorecaster trains on `scores[:t_pred]` and stores `forecast(ahead)` at index `t + ahead`, so the scorecast used for target τ is a forecast of `s[τ−4]` from data to `τ−8` (2·ahead stale); PID-Theta results are labelled "paper-core scorecaster (2·ahead stale)" and are not evidence about an aligned PID. (3) evaluation and MinT errors use final revised truth (as in the paper), not as-of vintages. |
| D16 | **Start-up caveat on the primary endpoint.** Controllers are re-initialised at `evaluation_start` (2021-01-16, winter-wave peak; 112 weeks); the rolling-coverage endpoint is dominated by the start-up transient (PI offsets jump then decay slowly), so the primary contrast reflects transient plus steady state. PI over-coverage (0.90–0.96 vs 0.80) is a property of the controller on a conservative base with 4-week feedback delay, not an adapter bug (wiring pinned by `test_pi_matches_paper_harness_wiring`). |
| D17 | **Reporting clarifications.** Contrasts are computed only at `lr` (`lr_grid` results are descriptive metrics); contrasts at non-primary block lengths are Holm-adjusted inside the secondary family; with `block < rolling_window` the rolling/run metrics straddle block joins in resamples; MinT may produce negative lower quantiles on small leaves (US row unaffected); crossings are counted with relative tolerance 1e-9 and reported separately for the US row (amends D8's count). |

New decisions are appended with a date; old ones are never deleted. This file overrides plans.
