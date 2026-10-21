---
type: results-report
date: 2026-10-21
experiment_line: probabilistic-model-benchmarks
round: 1
purpose: exploratory-benchmark-closeout
status: final
source_artifacts:
  - results/model-benchmarks-v1/manifest.json
  - results/model-benchmarks-v1/base_forecasts.parquet
  - results/model-benchmarks-v1/metrics.parquet
  - results/model-benchmarks-v1/level_metrics.parquet
  - results/model-benchmarks-v1/hierarchy_contrasts.parquet
  - results/model-benchmarks-v1/model_contrasts.parquet
  - results/model-benchmarks-v1/intervals.parquet
---

# Probabilistic model benchmark results (`model-benchmarks-v1`)

**Summary.** On the common 63-week evaluation window, none of the generated models beats
the submitted COVIDhub ensemble. Without reconciliation or PI, COVIDhub has the best mean
leaf coverage deviation (0.1336) and scaled interval score (2.4619). DeepAR is the closest
generated model on scaled interval score (3.9008), followed by ARIMA (4.2956) and SARIMA
(4.4298); all three score differences from COVIDhub are Holm-significant. PI makes coverage
deviations similar across models, but does so by expanding mean leaf widths roughly three- to
six-fold and leaves every generated model with a significantly worse scaled interval score.

Hierarchical reconciliation does not change that ranking or provide a general calibration
gain. At the leaf mean under PI, the MinT(Shrink) coverage-deviation changes are small and
unsupported for all four models. In the raw arm, MinT significantly worsens COVIDhub leaf
coverage deviation by 0.0220; the generated-model effects are unsupported after Holm
adjustment. At the US node, MinT often widens intervals and significantly worsens scaled
interval score for COVIDhub and SARIMA. The durable result is therefore model quality plus a
coverage-width trade-off, not a hierarchy-driven improvement.

Sources below use `R/` for `results/model-benchmarks-v1/`. Negative contrasts improve
coverage deviation, width, interval score, and longest miss run. All p-values described as
significant are Holm-adjusted within the D30 families. This benchmark is exploratory and has
no unadjusted primary endpoint.

## 1. Common-window diagnostics

These diagnostics are reported before outcomes because every comparison depends on the same
eligible weeks and nodes.

- The common window is 2021-12-25 through 2023-03-04: 63 weekly targets, which clears the
  registered minimum of 60. Every model is scored on exactly these dates.
- All 57 hierarchy nodes are present, with 51 state/DC endpoint leaves. No node is unscored.
- Generated-model bundles require no forecast imputation. COVIDhub uses 10 causally LOCF-
  imputed leaf cells, 0.13% of leaf cells on the model grid.
- Each generated model has 86 eligible rolling origins. The manifest's `fit_count` counts
  origins, not the 57 independent node fits inside each classical-model origin.
- Forecast generation took 160 seconds for ARIMA, 3,613 seconds for SARIMA, and 384 seconds
  for DeepAR on CPU. Captured fit warnings were ARIMA: 66 `ConvergenceWarning` and 13
  `EstimationWarning`; SARIMA: 164 and 362 respectively; DeepAR: 258 `FutureWarning` and
  136,704 `UserWarning`. No fit failed, and every retained predictive sample was finite.
- The retained process stderr contains Matplotlib cache and Lightning CPU/training status
  messages, but no warning, error, exception, traceback, failed-fit, NaN, or Inf match.
- No quantile crossings occur for generated models. COVIDhub MinT(Shrink) creates 269
  non-root crossings, handled by the registered sorting rule; no US crossing occurs.

The official run used reviewed implementation commit `e13beb3`. The manifest correctly
records `git_dirty = true`: the worktree contained a pre-existing, unrelated user edit to the
Phase 2 design specification. The benchmark implementation and registered config were
committed, and the immutable artifact directory was absent before atomic publication.

## 2. Model quality without hierarchical reconciliation

The raw `none` arm isolates the submitted/generated probabilistic forecasts from both
hierarchical reconciliation and PI.

| Model | Level | Coverage deviation | Marginal coverage | Scaled width | Scaled interval score |
|---|---|---:|---:|---:|---:|
| COVIDhub | leaf mean | 0.1336 | 0.8540 | 1.5121 | 2.4619 |
| DeepAR | leaf mean | 0.1633 | 0.9287 | 3.2527 | 3.9008 |
| ARIMA | leaf mean | 0.1796 | 0.8926 | 3.0860 | 4.2956 |
| SARIMA | leaf mean | 0.1822 | 0.8637 | 3.0200 | 4.4298 |
| COVIDhub | US | 0.1611 | 0.9524 | 0.8899 | 0.9264 |
| SARIMA | US | 0.1556 | 0.9048 | 1.4835 | 1.7796 |
| ARIMA | US | 0.2130 | 0.8095 | 1.4936 | 2.0338 |
| DeepAR | US | 0.2000 | 1.0000 | 3.3278 | 3.3278 |

At the leaf mean, the generated-model minus COVIDhub coverage-deviation estimates are +0.0297
for DeepAR, +0.0460 for ARIMA and +0.0487 for SARIMA; all are Holm-significant. Their scaled
interval-score penalties are +1.4389, +1.8337 and +1.9679 respectively, also all significant.
Thus the generated models' greater marginal coverage comes from intervals about twice as wide
on the scale-free measure, without better temporal calibration.

At the US node, coverage-deviation differences are not significant, but all three generated
models have significantly worse scaled interval scores than COVIDhub: SARIMA +0.8532, ARIMA
+1.1075 and DeepAR +2.4014. DeepAR's perfect marginal coverage is over-coverage, not evidence
of superior calibration: its coverage deviation is the maximum 0.2000 and its scaled width is
3.74 times COVIDhub's.

[Figure 1](figures/model-benchmarks-v1-coverage-score-tradeoff.pdf) shows this separation
between temporal coverage deviation and interval score. Its purpose is to prevent a high
marginal-coverage number from being read as model superiority when it is purchased with much
wider intervals.

## 3. Hierarchy effects within each model

For the registered MinT(Shrink) versus `none` comparison at block 8, no leaf-mean PI
coverage-deviation effect is supported:

| Model | Raw estimate [95% CI] | Raw Holm p | PI estimate [95% CI] | PI Holm p |
|---|---:|---:|---:|---:|
| ARIMA | -0.0008 [-0.0056, 0.0048] | 1.000 | -0.0041 [-0.0080, 0.0016] | 1.000 |
| SARIMA | +0.0105 [-0.0010, 0.0232] | 1.000 | -0.0007 [-0.0069, 0.0066] | 1.000 |
| DeepAR | +0.0006 [-0.0013, 0.0032] | 1.000 | -0.0001 [-0.0037, 0.0040] | 1.000 |
| COVIDhub | +0.0220 [0.0112, 0.0305] | 0.0075 | -0.0023 [-0.0068, 0.0012] | 1.000 |

The only supported leaf coverage result is harm to raw COVIDhub. MinT reduces its marginal
coverage from 0.8540 to 0.7647 and narrows scaled width from 1.5121 to 1.3726. The lower width
does not improve scaled interval score (contrast +0.0198, Holm p = 0.468) and makes temporal
coverage deviation worse.

For generated raw leaf forecasts, MinT effects are smaller and model-specific. ARIMA narrows
scaled width by 0.1260 (Holm p = 0.0075) with unsupported changes in coverage deviation and
score. SARIMA narrows width by 0.4199 and increases longest miss run by 0.6667 (both Holm p =
0.0075), while its coverage and interval-score contrasts remain unsupported. DeepAR changes
neither endpoint materially. Bottom-up leaves equal base leaves by construction.

At the US node, reconciliation more often imposes a scale cost. MinT raises scaled interval
score for raw COVIDhub by 0.1805 (Holm p = 0.0075) and raw SARIMA by 0.5373 (p = 0.0180);
the corresponding PI penalties are +0.2020 and +0.5103 (both p = 0.0075). ARIMA's positive
score contrasts and DeepAR's small negative contrasts are unsupported. Location-only
`mint_median` is also model-dependent: under PI it improves ARIMA and DeepAR US scaled score,
but worsens COVIDhub's; it does not establish a common reconciliation benefit.

[Figure 2](figures/model-benchmarks-v1-hierarchy-effect.pdf) presents the within-model
hierarchy contrasts. The key reading is the lack of a consistent direction across level,
model and controller—not an average benefit hidden by one weak model.

## 4. Model comparisons after MinT(Shrink)

MinT does not rescue the generated models. In the raw leaf arm, generated minus COVIDhub
scaled interval-score penalties remain +1.4268 for DeepAR, +1.7609 for ARIMA and +1.8771 for
SARIMA (all Holm p = 0.0030). SARIMA also has significantly worse coverage deviation
(+0.0372, p = 0.0210); ARIMA's +0.0232 has p = 0.0320 despite a percentile interval that
touches both signs, while DeepAR's +0.0084 is unsupported. The p-value/percentile-interval
mismatch is possible because these are distinct bootstrap summaries and should not be
silently resolved in favor of either one.

At the US node after raw MinT, every generated model again has a significant score penalty:
SARIMA +1.2101, ARIMA +1.6168 and DeepAR +2.1813 (all Holm p = 0.0030). Coverage-deviation
differences remain unsupported. Hierarchical reconciliation changes absolute scores, but not
the conclusion that COVIDhub is the stronger probabilistic benchmark.

[Figure 3](figures/model-benchmarks-v1-forest-leaf-mean-pi.pdf) and
[Figure 4](figures/model-benchmarks-v1-forest-us-pi.pdf) show the PI model contrasts at the
leaf mean and US node. They emphasize that similar coverage deviations coexist with reliably
worse interval scores for the generated models.

## 5. PI coverage-width trade-off

PI is not a registered direct contrast with raw, so this section is descriptive. With no
reconciliation, scaled mean leaf width increases from 1.5121 to 9.7954 for COVIDhub, 3.0860
to 10.9609 for ARIMA, 3.0200 to 11.0377 for SARIMA and 3.2527 to 10.8102 for DeepAR. Mean
leaf coverage deviation converges to a narrow 0.1793–0.1825 range, but scaled interval score
rises to 10.79–12.06.

Consequently, no generated model differs significantly from COVIDhub in PI leaf coverage
deviation, with or without MinT, while every generated model has a significantly worse PI
scaled interval score. The same score conclusion holds at the US node. PI therefore masks
raw calibration differences by over-expanding intervals; it does not make the probabilistic
models equivalent in forecast quality.

## 6. Interpretation and limitations

- This is one exploratory disease/horizon/window benchmark. The fixed ARIMA, SARIMA and
  10-epoch Darts LSTM settings are protocol controls, not exhaustive tuning exercises.
- DeepAR here means the registered Darts global LSTM with Gaussian likelihood, not GluonTS.
- Model training, MinT error estimation and evaluation use final revised truth rather than
  as-of vintages. PI inherits the paper core's learning-rate look-ahead described in D15.
- Classical fit warnings are numerous, especially for annual SARIMA. They were captured and
  disclosed; no fit or finite-output invariant failed. Results remain evidence about this
  fixed fitting procedure, not every possible ARIMA/SARIMA implementation.
- The common-window rule improves comparability at the cost of discarding model-specific
  usable weeks. With 63 weeks, interval estimates remain sensitive to serial dependence;
  blocks 4 and 12 are available in the artifacts as registered sensitivity analyses.
- MinT's 269 repaired COVIDhub leaf crossings show that quantile-wise reconciliation is not a
  joint probabilistic coherence method. Median coherence plus sorted marginal quantiles does
  not establish distributional coherence.

## 7. Decision and reproducibility

The benchmark closes without support for replacing COVIDhub with any of the three generated
models and without support for a general hierarchical-panel calibration advantage. Future
work should prioritize joint probabilistic hierarchical models, stronger model selection or
tuning, and as-of truth. Repeating the same fixed arm grid is not warranted by these results.

- Registered config: `configs/model_benchmarks.toml`.
- Immutable artifacts: `results/model-benchmarks-v1/`.
- Manifest implementation SHA: `e13beb37962d9f4a7dc5b3ceef8c9deac022268c`;
  input SHA-256: `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`.
- Run command: `uv run python -m hcp.run configs/model_benchmarks.toml`.
- Protocol: `docs/protocol-decisions.md`, D30–D31.
