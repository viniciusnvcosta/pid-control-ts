# Design: probabilistic ARIMA, SARIMA, and DeepAR hierarchy benchmarks

## Context and intent

Phases 1 and 2 tested whether reconciling submitted COVIDhub quantile forecasts before
Conformal PID improves coverage at the US and leaf levels. Phase 2 closed as a negative
result for the registered forecaster panel (D29). This benchmark asks a related mechanism
question: does reconciliation help when the base quantiles come from three controlled,
reproducible time-series models rather than heterogeneous external submissions?

The benchmark must compare ARIMA, SARIMA, and DeepAR both without reconciliation and
with the existing hierarchy methods. It must also retain `COVIDhub-4_week_ensemble` as an
external reference. The forecasting protocol follows the intent of
`core/model_scores.py`: Darts models, expanding historical forecasts, periodic retraining,
and a configurable forecast horizon. It corrects that helper's conflation of
`fit_every` with `forecast_horizon`, retains the actual fourth forecast step, and extends
point forecasts to the q0.1/q0.5/q0.9 quantiles required by the current pipeline.

This is an exploratory benchmark, not a new confirmatory primary hypothesis. Every
inferential comparison is multiplicity-adjusted, and the protocol is recorded in
`docs/protocol-decisions.md` before the official result is generated or inspected.

## Scope

### In scope

- Generate causal four-week-ahead q0.1/q0.5/q0.9 forecasts for all 57 hierarchy nodes
  with ARIMA, SARIMA, and DeepAR.
- Compare each generated model under `none`, `bottom_up`, `mint_shrink`, and
  `mint_median` reconciliation.
- Apply the existing `raw` and `pi` controllers and the existing Phase 2 metric suite.
- Compare hierarchy versus no hierarchy within each model.
- Compare each generated model with `COVIDhub-4_week_ensemble` on one shared window and
  shared set of scored nodes.
- Produce an atomic, non-overwriting run with forecasts, intervals, metrics, contrasts,
  diagnostics, figures, and a manifest.
- Review and fix the implementation before the official run, then write a research
  report under `docs/analysis/`.

### Out of scope

- Changing `core/model_scores.py` or any other upstream `core/` behavior.
- Hyperparameter search or choosing model settings after inspecting benchmark outcomes.
- GPU-specific tuning, distributed training, exogenous variables, ensembles, or
  alternative likelihoods.
- Replacing the Phase 1 or Phase 2 results or treating the benchmark as an independent
  replication across datasets.
- Conformalize-then-reconcile or other hierarchical conformal algorithms.

## Fixed protocol

### Data and hierarchy

The input is `data/deaths.csv`, protected by the existing SHA-256 check. The hierarchy is
the existing `us` root plus 56 leaves. The target and forecast-date grid is taken from
`COVIDhub-4_week_ensemble` at `ahead = 4`, preserving the 26-day forecast-to-target lag.

Actuals are canonicalized by `(geo_value, target_end_date)` across the input rows. All
finite reported values for a key must agree exactly; disagreement is a hard error. A
model forecast for target `t` may train only on target values whose `target_end_date` is
no later than `forecast_dates[t]`. This uses final revised truth with the same availability
assumption as D15; the manifest and report must repeat that limitation.

Missing observations inside an available training prefix are linearly interpolated using
only that prefix, matching the interpolation intent in `core/model_scores.py`. Leading
missing values are dropped. Truth used for scoring is never imputed. The returned
diagnostics record, per node and origin, the training start, training end, number of raw
observations, and number of interpolated observations.

All models forecast every hierarchy node independently except DeepAR, which is one global
model trained jointly on the 57 node histories. This intentionally creates independently
generated, generally incoherent root and leaf forecasts for reconciliation to act on.

### Temporal protocol

Forecast generation uses an expanding window and refits at every weekly origin
(`refit_every = 1`). Each fit predicts four weekly steps; only step four is retained for
the target week. No model may consume the target itself or any later actual.

The minimum training history is 60 observed or prefix-interpolated weeks for every node.
This accommodates the annual SARIMA component and defines the first origin at which all
three generated models are eligible. Forecasts before eligibility are NaN and are not
silently backfilled.

The benchmark evaluation begins at the latest week satisfying all of these conditions:

1. every benchmark model and the COVIDhub reference has finite q0.1/q0.5/q0.9 forecasts
   for every hierarchy node;
2. each benchmark bundle has at least `mint_warmup = 20` complete median-error weeks
   known by the current forecast date;
3. the US root and every scored endpoint leaf have finite truth for the full remaining
   evaluation window; and
4. at least `min_eval_weeks = 60` common evaluation weeks remain.

Failure to obtain such a window aborts the run. Every model, reconciler, controller, and
model-to-reference comparison uses exactly this common window. Territories participate
in fitting and reconciliation but remain excluded from `leaf_mean`, as in D25.

### Forecast models

All models operate on the untransformed weekly death series. Negative revision values in
the source data make a log transform invalid. Forecast samples and quantiles are not
clipped at zero.

The module-level `FORECASTERS` registry maps these configuration keys to constructors:

- `arima`: Darts `ARIMA(p=3, d=0, q=0)`. This is the exact order used by the existing
  `model_scores.py` key `"ar"`.
- `sarima`: Darts `ARIMA(p=3, d=0, q=0, seasonal_order=(1, 0, 0, 52))`.
- `deepar`: Darts `RNNModel` with `model="LSTM"`, `input_chunk_length=12`,
  `training_length=16`, `hidden_dim=25`, `n_rnn_layers=1`, `dropout=0`, a Gaussian
  likelihood, batch size 32, 10 epochs, and Adam's Darts-default learning rate. This is
  the DeepAR-style autoregressive recurrent model used by this study; the report must use
  that exact description rather than implying an Amazon library implementation.

DeepAR is fit globally across all nodes. Each node is divided by
`max(mean(abs(training_values)), 1)` at each origin, and predictions are transformed back
before quantiles are taken. ARIMA and SARIMA remain on their native scale.

Each model produces `n_samples = 500` predictive paths at each origin. The base forecast
is the sample quantiles `(0.1, 0.5, 0.9)` at horizon four. `SeedSequence(root_seed)`
spawns deterministic children by model, origin, and node where applicable; configuration
order, never dictionary or process completion order, defines seed assignment.

DeepAR runs on CPU with one device, zero data-loader workers, deterministic PyTorch
algorithms, checkpointing disabled, and progress output disabled. The dependency is added
with `uv add 'darts[torch]>=0.47.0'`; direct dependency edits are forbidden.

### Reference forecast

`COVIDhub-4_week_ensemble` is rebuilt on the same canonical truth and shared date grid as
the generated models. Its submitted q0.1/q0.5/q0.9 forecasts retain the existing causal
LOCF rule for missing forecast cells. Canonical truth may therefore make this benchmark's
MinT error history differ from Phase 2; the benchmark must not claim byte or metric parity
with `panel-leaves-v1`.

## Architecture

```text
src/hcp/
  benchmark_data.py      canonical actual panel + COVIDhub reference bundle
  forecast.py            numeric rolling forecast generation; FORECASTERS registry
  benchmark.py           common-window panel evaluation and paired comparisons
  benchmark_report.py    benchmark-specific figures, no file I/O
  config.py              frozen benchmark settings and validation
  run.py                 design="benchmark" orchestration and sole artifact writer
configs/
  model_benchmarks.toml  pre-registered model-benchmarks-v1 run
results/model-benchmarks-v1/
  base_forecasts.parquet
  intervals.parquet
  metrics.parquet
  level_metrics.parquet
  hierarchy_contrasts.parquet
  model_contrasts.parquet
  manifest.json
  figures/*.pdf
docs/analysis/
  model-benchmarks-v1.md
  figures/model-benchmarks-v1-*.pdf
```

Numeric modules perform no I/O. `run.py` reads the input, calls numeric stages, writes to
a sibling staging directory, writes the manifest last, and publishes with the existing
atomic no-replace operation. A failed run must leave no final result directory.

Public functions and dataclasses use type hints and Google docstrings. New Python files
begin with two `# ABOUTME:` lines. Modules should stay within the repository's 150–300
line target and split before roughly 400 lines.

## Component interfaces

### `benchmark_data.py`

`ActualPanel` is a frozen dataclass containing the canonical weekly dates, ordered nodes,
truth matrix, and per-cell availability. `canonical_actuals(frame) -> ActualPanel`
validates cross-row agreement. `reference_bundle(frame, actuals, forecaster) ->
HierarchyBundle` aligns submitted quantiles to canonical truth and the registered grid.

### `forecast.py`

`ForecastConfig` is frozen and contains the model list, horizon, minimum training weeks,
refit interval, sample count, quantile levels, root seed, and fixed model hyperparameters.
`ForecastResult` contains a mapping from model name to `HierarchyBundle`, a long base
forecast table, and diagnostics.

`generate_model_bundles(actuals, dates, forecast_dates, cfg) -> ForecastResult` is the
public numeric entry point. Small model-specific helpers may wrap Darts, but registry
values share one callable protocol and return horizon-four predictive samples. Unknown
model keys, non-weekly grids, insufficient history, non-finite returned samples, and
quantile crossings after sampling are hard errors. Equal adjacent quantiles are valid;
the implementation must not sort or otherwise repair invalid model output.

### `benchmark.py`

`BenchmarkConfig` combines the existing controller, reconciliation, metric, bootstrap,
and common-window settings. `run_benchmark(bundles, cfg) -> BenchmarkResult` requires the
three generated bundles plus the COVIDhub reference and selects one explicit common
window before any arm is evaluated.

`BenchmarkResult` contains base forecasts, intervals, node metrics, level metrics,
hierarchy contrasts, model contrasts, and diagnostics. Existing reconciliation,
conformalization, node metrics, block draws, Holm correction, and result schemas are
reused rather than reimplemented.

### Configuration and orchestration

`RunConfig.design` accepts `"benchmark"` in addition to `"us"` and `"panel"`.
Benchmark-only fields are rejected for the other designs where ambiguity would affect a
run. `benchmark_config(cfg)` maps the validated run configuration to the two numeric
configs. Existing Phase 1 and Phase 2 configs and outputs remain unchanged.

`configs/model_benchmarks.toml` fixes the run ID, seed, model keys, all model
hyperparameters, sample count, common-window rule, and the existing reconciliation,
controller, bootstrap, and evaluation settings.

## Comparisons and multiplicity

### Hierarchy effects within models

For each of ARIMA, SARIMA, DeepAR, and the COVIDhub reference, compare each of
`bottom_up`, `mint_shrink`, and `mint_median` with `none`. Use the existing paired moving-
block bootstrap with blocks `(8, 4, 12)` and `B = 2000`. One week-index draw is shared by
all nodes and arms within a `(model, block)` comparison.

Report every Phase 2 metric at `leaf_mean` and `us`. All p-values are Holm-adjusted within
`(block, model, level, controller)`; there is no unadjusted primary. The report must state
that `bottom_up - none` is identically zero at `leaf_mean`.

### Generated models versus COVIDhub

Compare each generated model with COVIDhub for `coverage_deviation` and
`scaled_interval_score`, at both levels, for controllers `raw` and `pi`, and for
reconcilers `none` and `mint_shrink`. The estimate is `generated_model - COVIDhub`, so a
negative value favors the generated model for both selected metrics. A single paired
week-index draw is shared by every model in a `(block, level, controller, reconciler)`
family. Holm correction covers the six members formed by three models times two metrics.

Direct metric tables for all arms remain the main model-quality comparison. Inferential
model contrasts are explicitly exploratory because the three algorithms are not sampled
from a population of forecasters.

## Artifacts and manifest

`base_forecasts.parquet` contains `model, node, week, forecast_date, q_lo, q_med, q_hi`.
The interval and metric tables retain the panel schema with `model` replacing
`forecaster`. Contrast tables are separate because their treatment dimensions differ.

The manifest includes:

- the complete frozen configuration and input SHA-256;
- Git SHA and dirty state;
- Python, NumPy, pandas, statsmodels, Darts, PyTorch, and Lightning versions;
- device and deterministic-algorithm status;
- the full seed-spawn description;
- model hyperparameters and Darts class names;
- common evaluation start/end, weeks, and scored nodes;
- per-model fit count, warning count by category, elapsed time, and forecast crossings;
- canonical-truth disagreement and interpolation diagnostics;
- the D15 final-revised-truth limitation; and
- `analysis_note = "exploratory benchmark; no unadjusted primary"`.

Warnings from model fitting are captured into diagnostics. A fit exception, a missing
model output, an invalid common window, or a non-finite evaluated forecast aborts the run;
the pipeline never substitutes another model or silently drops a model.

## Figures and analysis

The run produces, at minimum:

1. a model-by-reconciler forest plot for PI leaf-mean coverage deviation;
2. the corresponding US forest plot;
3. a coverage-deviation versus scaled-interval-score trade-off plot for all models under
   `none` and `mint_shrink`; and
4. a per-model hierarchy-effect plot comparing `mint_shrink - none` under raw and PI.

After the reviewed implementation is frozen, the official run writes
`results/model-benchmarks-v1/`. The analysis report copies publication figures into
`docs/analysis/figures/`, reports the common-window diagnostics before outcomes, separates
hierarchy effects from model-quality comparisons, discusses coverage-width trade-offs,
and appends the benchmark outcome as a new dated protocol decision.

## Testing and acceptance criteria

Development follows strict red-green-refactor TDD.

- Characterization tests pin the relevant current behavior of `core/model_scores.py`
  without modifying it.
- Unit tests prove canonical actual agreement checks, prefix-only interpolation, exact
  horizon-four alignment, registry validation, seed stability, quantile shapes/order,
  common-window selection, comparison signs, Holm families, and hard failures.
- Lightweight fake model adapters exercise forecast orchestration without training neural
  networks.
- Darts integration tests fit each real model on small deterministic synthetic weekly
  series and assert forecast shape, reproducibility, and no use of post-origin values.
- A real-data integration test runs a reduced origin/model/sample configuration and checks
  hierarchy shape, dates, and finite outputs without inspecting the registered outcome.
- An end-to-end test runs the benchmark design with fake adapters and verifies atomic
  artifacts and manifest fields.
- Existing Phase 1 and Phase 2 regression tests remain green and their registered configs
  parse identically.
- Before the official run: full pytest, Ruff check, Ruff format check, whole-branch code
  review, all Important findings fixed and re-reviewed, clean worktree, and a protocol
  decision recording the frozen benchmark design.
- The official run is launched exactly once for `model-benchmarks-v1`; stderr is retained
  and checked for warnings/errors. Results are never overwritten.

## Known limitations

- The benchmark uses one disease, one hierarchy, and final revised truth.
- DeepAR is a DeepAR-style Darts RNN configuration, not Amazon GluonTS DeepAR.
- Gaussian and ARIMA simulation distributions can generate negative death forecasts;
  preserving them avoids model-specific clipping.
- Weekly refitting, annual SARIMA, and global DeepAR are computationally expensive. The
  registered run favors protocol fidelity over speed and must record elapsed time.
- The models have fixed, literature-style settings rather than tuned settings, so results
  compare complete fixed pipelines rather than each model's best attainable performance.
