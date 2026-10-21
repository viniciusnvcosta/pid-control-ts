---
type: results-report
date: 2026-10-20
experiment_line: us-hierarchical-reconciliation
round: 1
purpose: phase-1-closeout
status: final
source_artifacts:
  - results/us-hier-v2/manifest.json
  - results/us-hier-v2/metrics.parquet
  - results/us-hier-v2/contrasts.parquet
  - results/us-hier-v2/intervals.parquet
  - docs/analysis/us-hier-v2-diagnostics.json
---

# Phase 1 results: reconciliation before conformal PI on the US series (`us-hier-v2`)

**Summary.** The pre-registered primary contrast is null. Reconciling the 57-node hierarchy
with MinT(Shrink) before PI control does not reduce the US coverage deviation (estimate
+0.0097, 95% CI [0.000, 0.029], p = 0.18; positive means worse). On the two arms with
contrasts, the base interval and PI at lr 0.1, the reconciled intervals are wider. Post-hoc diagnostics (exploratory, Section 5) suggest three reasons. First,
the endpoint has little headroom on one 112-week series. Second, quantile-wise reconciliation
mostly changes the interval's scale, not its location. Third, PI over-covers on this base
whether it starts cold or warm. Phase 1 is closed; Phase 2 proceeds as pre-registered
(D20–D27).

Sources: `R/` stands for `results/us-hier-v2/` and `J` for
`docs/analysis/us-hier-v2-diagnostics.json`. Every number below comes from one of these
files. Section 5 numbers come from `J`, which is regenerated with
`uv run python -m scripts.analysis.us_hier_v2 --run-dir results/us-hier-v2`.

## 1. Question and pre-registered design

Does reconciling the COVIDhub-4_week_ensemble quantile forecasts across the US → 56-leaf
hierarchy, before conformal control, improve the temporal coverage of the **US** 4-week-ahead
death forecasts? The design was fixed before any outcome was computed
(`docs/protocol-decisions.md`, D1–D12):

- **Data (D1, D5, D6).** `data/deaths.csv` (SHA-256 `cb4dccfd…ac06`, verified at load),
  `ahead = 4`. The nodes are `us` plus 56 leaves (51 states and DC, and 5 territories). Missing
  leaf quantiles are filled by causal LOCF (10 imputed cells, `R/manifest.json`). Truth is
  never imputed.
- **Arms (D2, D4).** Reconcile q0.1/q0.5/q0.9 quantile-wise (`none`, `bottom_up`,
  `mint_shrink`), then conformalize the US interval with cqr-asymmetric scores. The
  controllers are `raw` (the base interval), PI (`core.methods.quantile_integrator_log`,
  `Csat = 2`, `KI = 1000`, `T_burnin = 5`, α = 0.2 split 0.1 per side) and PID-Theta.
- **Window (D7, D7a).** MinT uses a causal, expanding covariance of median errors. Every arm is
  scored from the same week, `evaluation_start = 2021-01-16`, for 112 weeks, and controllers
  are re-initialised there.
- **Endpoint (D10, D11).** Primary: coverage deviation, the mean |rolling coverage (w = 10) −
  0.8| on the US series. Primary contrast: `mint_shrink − none` under PI at lr 0.1. The
  secondaries are `bottom_up − none`, longest miss run, marginal coverage, mean and median
  width, and the 80% interval score, all Holm-adjusted.
- **Statistics (D12).** A paired moving-block bootstrap over weeks with block 8 (sensitivity
  4 and 12) and B = 2000, seeded by `SeedSequence(20260929)`. CIs are percentile intervals;
  p-values are two-sided shift p-values with the +1 correction, so the smallest is
  1/2001 ≈ 0.0005.

`us-hier-v2` reruns `us-hier-v1` at commit `a3a91e5` with only `run_id` changed, and every
metric is identical (D19). The manifest records `git_dirty = false`.

## 2. Primary result

| Contrast | Controller | Metric | Estimate | 95% CI | p |
|---|---|---|---:|---|---:|
| `mint_shrink − none` | PI, lr 0.1 | coverage deviation | +0.0097 | [0.0000, 0.0291] | 0.184 |

Source: `R/contrasts.parquet`, row `primary == True` (block 8; the lower bound is
−1.4·10⁻¹⁷, i.e. zero). Under PI the base scores 0.1505 and MinT 0.1602 (`R/metrics.parquet`),
so MinT is 6.4% worse in relative terms, and the interval excludes any improvement. The
hypothesis that reconciliation reduces US coverage deviation is not supported.

The estimate rests on one week. In `R/intervals.parquet` the PI hit sequences of the two arms
differ only at target week 2021-12-25, which MinT covers and the base misses (107 vs 106 of
112 weeks covered). Both arms over-cover there, so the extra hit moves each of the 10 rolling
windows that contain the week 0.1 further from 0.8. That gives 10 × 0.1 / 103 windows =
0.0097, exactly the estimate. This decomposition was done after the result was known and is
descriptive.

## 3. Secondary results (block 8, Holm-adjusted)

Holm families are (block, controller). The PI family at block 8 holds the 11 non-primary
contrasts, and the primary itself is unadjusted. The raw family holds 12. Source:
`R/contrasts.parquet`.

| Controller | Treatment − none | Metric | Estimate | 95% CI | p | p (Holm) |
|---|---|---|---:|---|---:|---:|
| PI, lr 0.1 | mint_shrink | coverage_deviation (primary) | +0.0097 | [0.0000, 0.0291] | 0.184 | 0.184 |
| PI, lr 0.1 | mint_shrink | longest_miss_run | 0 | [0, 0] | 1.000 | 1.000 |
| PI, lr 0.1 | mint_shrink | marginal_coverage | +0.0089 | [0.0000, 0.0268] | 0.404 | 1.000 |
| PI, lr 0.1 | mint_shrink | mean_width | +1471 | [1200, 1710] | 0.0005 | 0.005 |
| PI, lr 0.1 | mint_shrink | median_width | +1071 | [863, 1932] | 0.001 | 0.013 |
| PI, lr 0.1 | mint_shrink | interval_score | +1093 | [782, 1398] | 0.0005 | 0.005 |
| PI, lr 0.1 | bottom_up | coverage_deviation | +0.0184 | [−0.0214, 0.0563] | 0.366 | 1.000 |
| PI, lr 0.1 | bottom_up | longest_miss_run | −1 | [−2, 0] | 0.049 | 0.343 |
| PI, lr 0.1 | bottom_up | marginal_coverage | +0.0179 | [−0.0179, 0.0625] | 0.415 | 1.000 |
| PI, lr 0.1 | bottom_up | mean_width | +710 | [167, 1196] | 0.008 | 0.068 |
| PI, lr 0.1 | bottom_up | median_width | −137 | [−383, 1028] | 0.801 | 1.000 |
| PI, lr 0.1 | bottom_up | interval_score | +282 | [−334, 739] | 0.274 | 1.000 |
| raw | mint_shrink | coverage_deviation | +0.0029 | [−0.0408, 0.0553] | 0.901 | 1.000 |
| raw | mint_shrink | longest_miss_run | −2 | [−3, 0] | 0.011 | 0.066 |
| raw | mint_shrink | marginal_coverage | +0.0536 | [0.0179, 0.1074] | 0.028 | 0.085 |
| raw | mint_shrink | mean_width | +1155 | [881, 1388] | 0.0005 | 0.006 |
| raw | mint_shrink | median_width | +736 | [629, 1278] | 0.002 | 0.014 |
| raw | mint_shrink | interval_score | +639 | [204, 954] | 0.001 | 0.013 |
| raw | bottom_up | coverage_deviation | +0.0078 | [−0.0379, 0.0573] | 0.733 | 1.000 |
| raw | bottom_up | longest_miss_run | −2 | [−3, 0] | 0.011 | 0.066 |
| raw | bottom_up | marginal_coverage | +0.0625 | [0.0179, 0.1163] | 0.016 | 0.066 |
| raw | bottom_up | mean_width | +1967 | [1515, 2385] | 0.0005 | 0.006 |
| raw | bottom_up | median_width | +1586 | [1157, 2134] | 0.001 | 0.013 |
| raw | bottom_up | interval_score | +1302 | [677, 1869] | 0.0005 | 0.006 |

Widths and interval scores are in weekly deaths. Coverage deviation, longest miss run, width
and interval score are better when lower; marginal coverage is better when closer to 0.8.

The only contrasts that survive Holm are the width and interval-score increases: both
reconcilers widen the US interval and score worse on the interval score. No coverage contrast
survives. Longest miss run falls by 2 weeks under `raw` (adjusted p = 0.066), which goes with
the higher marginal coverage of the wider intervals rather than with better temporal
calibration. At blocks 4 and 12 the coverage-deviation estimates are the same (they do not
depend on the resamples) and every adjusted p-value is 1.000. The primary contrast's
unadjusted p is 0.180 at block 4 and 0.206 at block 12.

## 4. Descriptive metrics (lr grid)

Metrics over the 112-week window, from `R/metrics.parquet`. Contrasts exist only at lr 0.1
(D17), so the grid is descriptive. † PID-Theta uses the paper-core scorecaster, which is
2·ahead stale (D15). Its rows are not evidence about an aligned PID.

**Coverage deviation** (lower is better)

| Controller | lr | none | bottom_up | mint_shrink |
|---|---:|---:|---:|---:|
| raw | — | 0.151 | 0.159 | 0.154 |
| pi | 1 | 0.213 | 0.165 | 0.202 |
| pi | 0.5 | 0.144 | 0.182 | 0.198 |
| pi | 0.1 | 0.150 | 0.169 | 0.160 |
| pi | 0.05 | 0.160 | 0.159 | 0.150 |
| pid_theta† | 1 | 0.186 | 0.221 | 0.186 |
| pid_theta† | 0.5 | 0.174 | 0.165 | 0.171 |
| pid_theta† | 0.1 | 0.199 | 0.256 | 0.195 |
| pid_theta† | 0.05 | 0.259 | 0.299 | 0.271 |

**Marginal coverage** (target 0.8)

| Controller | lr | none | bottom_up | mint_shrink |
|---|---:|---:|---:|---:|
| raw | — | 0.884 | 0.946 | 0.938 |
| pi | 1 | 0.857 | 0.911 | 0.902 |
| pi | 0.5 | 0.929 | 0.911 | 0.920 |
| pi | 0.1 | 0.946 | 0.964 | 0.955 |
| pi | 0.05 | 0.929 | 0.955 | 0.946 |
| pid_theta† | 1 | 0.946 | 0.893 | 0.929 |
| pid_theta† | 0.5 | 0.902 | 0.902 | 0.902 |
| pid_theta† | 0.1 | 0.759 | 0.768 | 0.768 |
| pid_theta† | 0.05 | 0.670 | 0.670 | 0.679 |

**Mean width** (weekly deaths) and **interval score**

| Controller | lr | width none | width bottom_up | width mint_shrink | IS none | IS bottom_up | IS mint_shrink |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | — | 4839 | 6806 | 5994 | 6024 | 7325 | 6663 |
| pi | 1 | 11043 | 15245 | 14282 | 12836 | 16287 | 15289 |
| pi | 0.5 | 10554 | 10244 | 9609 | 10777 | 10485 | 10213 |
| pi | 0.1 | 6978 | 7688 | 8448 | 7573 | 7856 | 8666 |
| pi | 0.05 | 6163 | 7410 | 7380 | 6885 | 7673 | 7779 |
| pid_theta† | 1 | 16509 | 18127 | 16155 | 17117 | 20400 | 18010 |
| pid_theta† | 0.5 | 11032 | 13079 | 12238 | 13249 | 14299 | 13427 |
| pid_theta† | 0.1 | 6630 | 6036 | 6975 | 11868 | 13973 | 12357 |
| pid_theta† | 0.05 | 4600 | 4754 | 5210 | 12364 | 14346 | 13022 |

No reconciler is consistently better across the grid. The ranking of the three reconcilers on
coverage deviation changes with lr. Across the grid, PI's coverage deviation ranges from 0.144
to 0.213 and is never more than 0.008 below the base interval's 0.1515 (`none`, lr 0.5). PI
over-covers at every lr (marginal coverage 0.857–0.964), and its interval score is worse than the uncontrolled base on every reconciler. PID-Theta
under-covers at lr ≤ 0.1, which is consistent with the stale scorecast documented in D15.

## 5. Post-hoc diagnostics (exploratory)

> These analyses were chosen **after** the primary result was known (D19). They are not
> pre-registered, carry no inferential claims, and should be read as hypotheses for Phase 2,
> not as findings. All numbers come from `J` (n_weeks = 112, the window of Section 1).

### 5.1 Headroom: the i.i.d. noise floor

Suppose an interval covers each week independently with probability exactly 0.8. On this
endpoint (112 weeks, w = 10, so 103 rolling windows) it still scores a coverage deviation of
**0.0969** on average. The 2.5–97.5% range is [0.063, 0.138] (10,000 simulations seeded from
`root_seed`; `J.floor`). D19 quotes 0.0966; the 0.0003 difference is consistent with Monte
Carlo error at this number of simulations. The base interval scores 0.1515, about 0.055 above
the floor mean. That gap is roughly what an interval with calibrated, independent misses could
remove, and it is about the half-width of the raw-arm contrast CIs at block 8 (0.048 for
`mint_shrink`, 0.048
for `bottom_up`; Section 3). Even a reconciler that removed all the excess deviation would
be hard to detect on one series.

[Figure 1](figures/us-hier-v2-floor.pdf) shows the simulated distribution. The vertical lines
mark the observed coverage deviation of each run arm, `raw` solid and PI at lr 0.1 dashed. The
run has no `mint_median` arm, so the figure has three reconcilers. Every arm lies above the
97.5th percentile of the floor. Two things can put an arm there: marginal over-coverage (all
arms cover 0.88–0.96) and misses that cluster in time. This diagnostic does not separate the
two. The floor is a reference, not a lower bound: an interval whose hits follow a regular
pattern could score below it, but a forecaster with calibrated, independent misses scores
about 0.097.

### 5.2 Scale, not location

The base interval (`raw`) after each reconciler (`J.location_scale`):

| Reconciler | Mean width | vs none | Median MAE | vs none | Coverage dev. | Marginal cov. | Longest miss run | Interval score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 4839 | — | 1319.9 | — | 0.151 | 0.884 | 4 | 6024 |
| bottom_up | 6806 | +40.6% | 1287.6 | −2.4% | 0.159 | 0.946 | 2 | 7325 |
| mint_shrink | 5994 | +23.9% | 1299.9 | −1.5% | 0.154 | 0.938 | 2 | 6663 |
| mint_median | 4839 | 0.0% | 1299.9 | −1.5% | 0.183 | 0.848 | 8 | 6252 |

Quantile-wise reconciliation sets the US q0.1 and q0.9 to (weighted) sums of leaf quantiles.
A sum of quantiles is the quantile of the sum only when the leaves are comonotone, so for
imperfectly dependent leaves the summed interval is too wide. That matches the +41% (bottom-up)
and +24% (MinT) widening. The reconciled medians improve the US median MAE only slightly,
by 2.4% for bottom-up and 1.5% for MinT.

`mint_median` (D26) keeps the base spread and moves every level by the MinT median shift.
Its width therefore equals the base width, and its median equals MinT's. Taken alone, the
coherent location makes the base interval worse: coverage deviation rises from 0.151 to 0.183,
the longest miss run from 4 to 8 weeks, and the interval score from 6024 to 6252 (+3.8%). On
this series the higher coverage of the reconciled intervals comes from their width, not from a
better centre.

### 5.3 Controller start-up: warm vs cold PI

PI at lr 0.1 started at `evaluation_start` (cold, as in the run), compared with PI run from the
first complete week (25 weeks earlier) and scored on the same 112 weeks (warm). Source:
`J.warm_start`.

| Reconciler | Start | Coverage dev. | Marginal cov. | Mean width | Interval score | Longest miss run |
|---|---|---:|---:|---:|---:|---:|
| none | cold | 0.150 | 0.946 | 6978 | 7573 | 2 |
| none | warm | 0.151 | 0.911 | 6544 | 7173 | 3 |
| bottom_up | cold | 0.169 | 0.964 | 7688 | 7856 | 1 |
| bottom_up | warm | 0.153 | 0.911 | 7068 | 7460 | 3 |

The cold rows match the run's PI metrics at lr 0.1 (`R/metrics.parquet`), which checks that
the script uses the same controller wiring. For `none`, warm-starting reduces over-coverage
(0.946 → 0.911), width (−6.2%) and interval score (−5.3%), but coverage deviation stays at
0.151. For `bottom_up`, warm-starting lowers coverage deviation from 0.169 to 0.153 and
marginal coverage from 0.964 to 0.911. The warm PI interval for `none` is still 35% wider than
the base interval (6544 vs 4839). D16's
start-up transient therefore explains part of PI's over-coverage, but not all of it.
[Figure 2](figures/us-hier-v2-warm-start.pdf) shows the two offsets over the window for `none`.
The cold offsets are zero for the first weeks, while coverage feedback is still in flight.
They then jump during the 2021 winter wave and take more than a year to decay.

The update rule explains the slow decay. In `core/methods.py` each side updates
`q[t+1] = q[t] − lr_t · g`, with `g = α` after a hit and `g = −(1 − α)` after a miss, where α
= 0.1 per side. A miss therefore raises the offset 9× as much as a hit lowers it. The update
at week t also uses the coverage of week t − ahead + 1, so feedback is several weeks late. On a
base that already covers 88% of weeks, a few misses during a surge build up offsets that later
hits remove only slowly. This is a property of the controller and of this base, not a wiring
error (D16; pinned by `test_pi_matches_paper_harness_wiring`).

## 6. Interpretation and threats to validity

**Supported.** On the US series, under the pre-registered design, quantile-wise reconciliation
(MinT(Shrink) or bottom-up) does not improve temporal coverage under PI at lr 0.1 or for the base
interval. It makes the interval wider and worsens the interval score (Holm-adjusted p ≤ 0.014
for every width and interval-score contrast except `bottom_up` under PI).

**Not supported, and not refuted.** The study does not show that reconciliation harms
coverage. The primary CI includes zero and rests on one week (Section 2). It also does not
show that reconciliation is useless elsewhere in the hierarchy, because the leaves were not
evaluated.

**Tentative (Section 5, post-hoc).** The null is consistent with low headroom, a change in
scale rather than location, and a controller that over-covers on a conservative base. These
explanations were formed after seeing the outcome and need a design in which they are tested
prospectively.

**Threats to validity.**

- *A single series.* One forecaster, one node and 112 weekly outcomes. The block bootstrap
  resamples weeks of that one realization, so the CIs say nothing about other seasons,
  forecasters or nodes.
- *Lr look-ahead (D15.1).* core's proportional step `lr_t` uses score ranges up to t − 1, while
  only scores up to t − ahead + 1 are known. PI therefore has about 2 weeks of look-ahead in
  its step size. This is inherited from the paper and is the same in every arm, so the paired
  contrasts stay fair, but PI is not strictly causal.
- *Stale scorecaster (D15.2).* PID-Theta's scorecast is 2·ahead stale. Its rows are reported
  only descriptively.
- *Final revised truth (D15.3).* MinT errors and evaluation use final revised truth, not
  as-of vintages, as the paper does. Truth revisions show up as a US − Σleaves gap of up to 602
  deaths (`R/manifest.json`, `truth_incoherence`).
- *Start-up transient (D16).* The window starts at the 2021 winter peak, and controllers are
  re-initialised there, so the primary endpoint mixes the transient with the steady state.
  Section 5.3 bounds the effect for `none` and `bottom_up` but was not pre-registered.
- *Reporting choices (D17).* Contrasts exist only at lr 0.1. With block < w, the rolling
  metrics in resamples straddle block joins. MinT re-sorted 261 crossed (node, week) cells,
  none of them on the US row (`R/manifest.json`).

## 7. Implications for Phase 2

Phase 2 is specified in
[`docs/superpowers/specs/2026-10-20-hierarchical-panel-design.md`](../superpowers/specs/2026-10-20-hierarchical-panel-design.md)
and registered as D20–D27 before any leaf-level outcome was computed. This report does not
change that design. The diagnostics bear on it as follows:

1. **Power (D20).** One US series has about 0.055 of reducible deviation, close to the CI
   half-width. The leaf-mean endpoint averages up to 51 scored series under a single shared
   week-index draw. This is expected to narrow the interval compared with one series, although
   dependence across leaves limits the gain. When the Phase 2 primary is
   interpreted, the floor for that forecaster's window should be reported alongside it,
   since the same headroom argument applies to each leaf.
2. **Mechanism (D26).** On the US row the coherent location alone hurt coverage (Section 5.2).
   `mint_median − none` at the leaves tests whether that holds where reconciliation borrows
   strength from the top.
3. **Scale (D23).** Reconciliation acts mainly through width, so the scale-free width and
   interval-score secondaries carry much of the information at the leaves.
4. **Controller (D22, D7a).** The panel keeps the cold start, so PI there has the same start-up
   transient. On this series, Section 5.3 suggests the transient affects width and marginal
   coverage more than coverage deviation. The additional forecasters (D21) may include
   bases that under-cover, on which PI has miscoverage to correct; their calibration was not
   examined here.
5. **Regression anchor (D27).** The panel's US-node metrics for the ensemble must equal
   `R/metrics.parquet`. This report's Section 4 tables are the same numbers.

**Decision.** Phase 1 is closed with a null primary result. Proceed with Phase 2 as
pre-registered. The deferred items in the Phase 2 spec stay deferred: probabilistic
reconciliation with an empirical copula, an aligned scorecaster, and conformalize-then-
reconcile. The comonotone widening in Section 5.2 is the strongest argument for the first of
these.

## Artifact and reproducibility index

| Artifact | Content |
|---|---|
| `results/us-hier-v2/manifest.json` | config, git SHA `a3a91e5`, input SHA-256, seeds, versions, run diagnostics |
| `results/us-hier-v2/metrics.parquet` | Sections 4, 5.1 (observed lines), 5.3 (cold cross-check) |
| `results/us-hier-v2/contrasts.parquet` | Sections 2, 3 |
| `results/us-hier-v2/intervals.parquet` | Section 2 (single-week decomposition) |
| `docs/analysis/us-hier-v2-diagnostics.json` | Section 5 (`floor`, `location_scale`, `warm_start`) |
| `docs/analysis/figures/us-hier-v2-floor.pdf` | Figure 1 |
| `docs/analysis/figures/us-hier-v2-warm-start.pdf` | Figure 2 |
| `scripts/analysis/us_hier_v2.py` | regenerates the JSON and both figures (byte-identical on rerun) |
