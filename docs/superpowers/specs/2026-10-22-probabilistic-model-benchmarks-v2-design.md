# Design: cumulative probabilistic model benchmarks v2

Date: 2026-10-22. Status: approved by the user on 2026-10-22; implementation plan pending review.
Study identity: `model-benchmarks-v2`.

## 1. Intent, evidence status, and scope

This exploratory study compares eight forecasting pipelines under four principal
processing arms: original forecasts, normalized PI, location reconciliation, and
location reconciliation followed by normalized PI. It asks whether either intervention
or their combination improves probabilistic forecast quality, and whether seven generated
models improve on the submitted COVIDhub reference under the same processing arm.

The study is cumulative in its model set, not an unchanged replication of v1. It retains
COVIDhub, ARIMA, SARIMA, and the original DeepAR-style Darts LSTM, and adds Prophet,
Transformer, local L1 quantile regression, and a stronger DeepAR-style variant. The
original DeepAR and the stronger variant have separate registry keys and diagnostics.
`model-benchmarks-v1`, its configuration, protocol, specification, and immutable results
remain unchanged.

V1 outcomes informed the new controller, endpoints, and focus on `mint_median`. V2 is
therefore explicitly exploratory, including after prospective registration of this
protocol. The two Holm families control their respective declared collections; they do
not provide a single experiment-wide error guarantee across both collections or all
sensitivity analyses. There is no unadjusted primary hypothesis.

The selected approach is **cumulative models with focused inferential arms**. A full
inferential reconciler/controller grid was rejected for its cost and multiplicity;
outcome-based model screening was rejected because it would introduce selection before
the hierarchy comparison. All eight models remain eligible for every principal arm.

The study evaluates fixed forecasting procedures and declared tuning budgets, not the
best attainable performance of each family. In particular, the stronger DeepAR selects
its hidden width, whereas the Transformer architecture is fixed.

Out of scope: changing `core/`, as-of truth reconstruction, repairing D15's learning-rate
look-ahead, negative-binomial models, truncation at zero, new joint hierarchical
likelihoods, GPU training, pretrained models, outcome-driven model dropping, and
Informer/Autoformer/PatchTST/TFT implementation.

Technical completions made while writing this spec are proposals for the written-spec
review: the 20-dimensional embedding, exact neural four-step validation objective and
seed layout, explicit Prophet defaults, and the empty-conformal-set scoring/reporting
policy in sections 5-6. They implement the agreed direction but were not individually
approved in the conversational questions. Written-spec approval includes these details.

## 2. Data, chronology, and common evaluation window

Retain the v1 input `data/deaths.csv`, expected SHA-256
`cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`.
Canonicalize finite actuals across ahead-four rows by `(geo_value, target_end_date)`;
any finite disagreement aborts. Use the submitted `COVIDhub-4_week_ensemble` target and
forecast-date grid, preserving its 26-day forecast-to-target lag and weekly targets.

All 57 nodes participate in generation and reconciliation: reported US plus 56 leaves.
The inferential leaf set is the fixed 51 states/DC, excluding `as`, `gu`, `mp`, `pr`,
and `vi`. V1 already excluded these territories from `leaf_mean`; the review's claim
that territories dominated that endpoint is not adopted. The reported US actual remains
the US scoring target, with its discrepancy from the sum of leaves recorded.

Every outer fit uses an expanding prefix with at least 60 weekly observations after
prefix-only interpolation. At an origin for target T, the last usable weekly observation
is T minus 28 days and must also be no later than `forecast_date(T)`. Predict four steps
and retain step four. Refit every model at every eligible weekly origin, including its
declared hyperparameter selection. No stale-fit schedule or cross-origin forecast cache
is part of this protocol.

Drop leading missing values; interpolate internal training gaps only within the prefix
available to that fit. Repeat prefix preparation independently inside validation splits.
Neither interpolation nor a fitted scaler may use a later validation or forecast target.
Scoring truth and validation truth are never imputed. Retain interpolation counts and
training boundaries by model, node, and origin.

Build one common evaluation window, before calculating arm metrics, using the existing
v1 availability rule: all eight models have finite base quantiles for all nodes; every
bundle has at least 20 complete median-error weeks known by the current forecast date;
US and all 51 endpoint truths are complete over the remaining evaluation window. Use the
earliest common start satisfying these conditions, with at least 60 weeks remaining.
Abort if this window does not exist. Never relax a requirement or omit a model to rescue
the registered run. Territories with incomplete evaluation truth remain in reconciliation
but have no per-node score; their exclusion is diagnostic, not a change to endpoint leaves.

All arms and contrasts use this window and the same endpoint nodes. Controllers are
reinitialized at its start, as in D7a. The forecast grid before evaluation supplies MinT
warm-up errors. No model-specific evaluation windows or cross-v1 paired significance
tests are included.

Model fitting, validation, and MinT updates respect forecast-date availability under the
final-revised-truth assumption. This is not as-of-vintage causality. Both PI variants
retain the inherited learning-rate look-ahead in D15, so the full pipeline must not be
described as strictly causal.

## 3. Forecast models and native output contract

Generated-model configuration order is `arima`, `sarima`, `deepar`, `prophet`,
`transformer`, `qr_l1`, `deepar_enhanced`; the reference key is `covidhub`.
All outputs are in original death-count units, including negative values. No forecast,
interval, training observation, or evaluation truth is clipped at zero.

### 3.1 Models retained for continuity

- `arima`: independent node-wise Darts ARIMA(3,0,0).
- `sarima`: independent node-wise Darts ARIMA(3,0,0) with seasonal order (1,0,0,52).
- `deepar`: global Darts `RNNModel`, LSTM, input length 12, training length 16,
  hidden width 25, one recurrent layer, dropout 0, batch 32, ten epochs, Adam learning
  rate 0.001, Gaussian likelihood. Divide each node by
  `max(mean(abs(training_values)), 1)` at each origin and restore original units.
- `covidhub`: submitted q0.1/q0.5/q0.9, aligned to canonical truth with the existing
  causal LOCF rule for missing forecast cells; no synthetic sample representation.

Retain these model definitions and the first three generated-model seed assignments.
Regression tests establish continuity of their adapter behavior. A different common
window, dependency version, or processing arm does not imply byte/metric parity with v1.
Neither ARIMA nor SARIMA orders are retuned following their v1 warnings.

### 3.2 Prophet

`prophet` is a separate local Darts Prophet fit per node and weekly origin. Use linear
growth with changepoints; disable yearly, weekly, and daily seasonalities explicitly.
No holidays, external regressors, or logistic floor/cap are used. Fix 25 changepoints,
changepoint range 0.8, changepoint prior scale 0.05, and `mcmc_samples=0`. Generate 500
seeded posterior-predictive paths using Prophet's MAP-fit uncertainty procedure, rather
than describing them as full Bayesian posterior samples. Fit/predict seeds come from
the declared model/origin/node stream. Preserve weekly timestamps.

Install the runtime dependency only during authorized implementation using `uv add
prophet`; commit the resulting lockfile and record Prophet/CmdStan backend versions.
The v2 adapter must not alter the original paper harness's use of Prophet.

### 3.3 Local direct L1 quantile regression

`qr_l1` fits separate scikit-learn `QuantileRegressor` models for levels 0.1, 0.5,
and 0.9 at every node. Use native L1, `solver="highs"`, and an unpenalized intercept.
For a feature endpoint u, X contains `y[u], ..., y[u-11]` and the response is
`y[u+4]`. A training example is eligible only if its response was available at the fit's
cutoff. These are twelve latest observations, with no seasonal features or node pooling.

Divide features and responses by `max(mean(abs(training_prefix)), 1)`, computed separately
inside each fit. This QR scale is independent of the PI and evaluation scales. After
prediction, restore original units.

Search alpha in ascending order `[0.001, 0.01, 0.1, 1.0]`, selecting one alpha jointly
for the three quantiles, per node and outer origin. Reserve the last twelve consecutive
weekly targets with observed truth in that node's available outer prefix. Each of these
validation targets is predicted directly from its own four-step-earlier features.
Each candidate is fitted once, using only training responses available at the first
validation forecast date; feature construction may subsequently use observations known
at each validation origin, but candidate coefficients and scaling stay fixed throughout
the holdout. Apply the actual forecast-date cutoff, not just row adjacency, when purging
training responses. A validation feature prefix is prepared without future observations.

Select the alpha minimizing the mean pinball loss over all twelve targets and all three
levels, before quantile repair. An exact numerical tie selects the larger alpha. Refit
the selected alpha on all eligible examples in the outer prefix, recomputing its scale.
Require at least twelve eligible training examples in each inner fit; unavailable splits
make the outer origin ineligible before common-window selection. A failed solver at an
eligible origin aborts; no alpha candidate is silently discarded.

Sort the three final direct forecasts and record repair diagnostics. Return native
horizon-four quantiles, not 500 repeated pseudo-samples.

### 3.4 Fixed global Transformer

Use Darts `TransformerModel` with Gaussian likelihood across the 57 separately scaled
univariate series. Scaling is `max(mean(abs(training_prefix)), 1)`, independently per
fit/node, with original-unit samples restored before quantiles. Global parameter sharing
does not itself create a joint cross-node predictive distribution.

| Parameter | Value |
|---|---|
| Input / output chunk lengths | 12 / 4 |
| Model dimension / attention heads | 32 / 4 |
| Encoder / decoder layers | 2 / 2 |
| Feed-forward dimension | 128 |
| Dropout / activation | 0.1 / ReLU |
| Normalization / positional encoding | Darts standard LayerNorm / sinusoidal |
| Batch / optimizer / learning rate | 32 / Adam / 0.001 |
| Likelihood / inference dropout | Gaussian / MC dropout disabled |

No external covariates or architecture search. Use the neural holdout, stopping, and
refit protocol below. Dropout regularizes training; predictive randomness comes from
the Gaussian likelihood. Generate 500 four-step predictive paths.

### 3.5 Stronger global DeepAR-style variant

`deepar_enhanced` uses three LSTM layers, hidden-width candidates `[40, 80, 120]`,
Gaussian likelihood, batch 32, Adam learning rate 0.001, and dropout 0. It remains a
DeepAR-style adaptation rather than an exact reproduction of the paper. Retain input
length 12 and training length 16 to fit the short available initial histories; do not
silently adopt the paper's 52-week context and lose the registered common window.

Add a learned 20-dimensional node-identity embedding, standardized age in weeks since
the first observed node value, and numeric week-of-year. Calendar and age are known
through the forecast horizon. Covariate normalization uses only the training portion;
zero-variance components use unit scale. Node vocabulary is the fixed hierarchy order,
with no target-encoded identifiers.

Use `nu_i = 1 + mean(abs(training_values_i))` per fit. Divide autoregressive values by
nu_i and rescale Gaussian location and standard deviation consistently to original
units. Absolute values are an explicit adaptation for negative revisions. The paper's
negative-binomial parameter scaling is not applicable to this Gaussian variant.

For training, select a node with probability proportional to nu_i, then a valid window
uniformly within that node, with replacement. An epoch draws as many examples as the
total number of available training windows. This defines window sampling, not a
substitution of loss weighting for sampling. The training sampler uses its own seeded
generator. Validation averages predictive NLL equally across nodes and target weeks,
without sampling weights. Root and leaves share parameters despite overlapping targets;
they are not treated as independent replicates for statistical inference.

Darts' stock LSTM is insufficient to claim learned embeddings and weighted sampling
merely by attaching static covariates. Use a focused Darts custom recurrent module and
dataset/sampler integration under `src/hcp/`, importing the library's training and
likelihood machinery. Tests must demonstrate that embeddings affect the forward pass
and that the requested sampler is actually consumed. No upstream `core/` edits.

### 3.6 Neural validation, stopping, and final refit

For both new neural models, reserve the last twelve consecutive fully observed weekly
targets of the available outer history across all 57 nodes. Fit on the preceding
prefix; fitted scales, age/calendar standardization, and sampling weights exclude those
targets. Validation context may include earlier validation observations when predicting
a later window, only if they precede that window's forecast origin. Every scored
validation target must be genuinely out of sample with a four-step gap.

Use the nine horizon-four origins whose four-step outputs lie entirely inside the
twelve-week holdout. Aggregate **fourth-step** Gaussian NLL in original death-count
units, equally over nodes and origins, at each training epoch. Darts' generic one-step
training/validation loss must
not silently replace this selection objective. For the autoregressive model, obtain the
fourth-step density as the Monte Carlo mixture of conditional Gaussian densities along
500 seeded simulated histories, evaluated with log-sum-exp. The Transformer provides its
fourth-step Gaussian parameters directly. Validation uses a dedicated, reset seed stream
at every epoch to reduce changes due solely to Monte Carlo draws.

Train for at most 100 epochs; stop after ten consecutive epochs without a strictly lower
finite validation NLL (`min_delta=0`). Keep best weights in memory, with checkpoint files
disabled. An equal NLL retains the earlier epoch. For enhanced DeepAR, select the width
with lowest best NLL; exact ties select the smaller width. The Transformer selects only
its stopping epoch, not architecture. Record every candidate's best epoch and score.

Reinitialize the selected architecture with its final-fit seed and fit the full outer
prefix for exactly the selected best-epoch count, recomputing all preprocessing on that
prefix. No validation-target loss controls this final refit. A missing/invalid inner
split makes the outer origin ineligible; a nonfinite loss or fit failure at an eligible
origin aborts rather than removing a candidate.

### 3.7 Frozen native prediction object

Keep a single module-level `FORECASTERS` registry owned by `forecast.py`. Each adapter
returns a frozen tagged prediction object containing node order, `ahead=4`, native-unit
output, and numeric diagnostics. Exactly one payload is populated:

- `kind="samples"`: finite read-only `[node, 4, 500]` predictive paths;
- `kind="quantiles"`: finite read-only `[node, 3]` horizon-four values with explicit
  levels `(0.1, 0.5, 0.9)`.

A common bundle builder validates the tag, dimensions, levels, nodes, and horizon.
For samples it uses NumPy's linear empirical quantiles at the fourth step; for native
quantiles it preserves the adapter values after the declared QR repair. No synthesized
intermediate-horizon QR forecasts or pseudo-samples. The reference bundle retains its
native submitted quantiles. Frozen dataclasses also defensively protect their arrays.

## 4. Controllers, reconciliation, and distinct scales

The principal arms per model are:

| Arm | Reconciler | Controller |
|---|---|---|
| A | `none` | `raw` |
| B | `none` | `pi_norm` |
| C | `mint_median` | `raw` |
| D | `mint_median` | `pi_norm` |

Order is **base quantiles -> reconciliation -> asymmetric CQR scores -> controller**.
Never apply PI and then reconcile its interval. `mint_median` makes the medians coherent
and shifts every quantile by the same node-specific amount, preserving base spreads.
Final PI intervals have no guaranteed hierarchical coherence; this study does not claim
joint distributional coherence even for the reconciled marginal quantiles.

Register `pi_norm` in `CONTROLLERS`, preserving `raw`, `pi`, and all legacy behavior.
Compute one fixed scale per node at the first common evaluated forecast origin:

`pi_scale_i = max(mean(abs(available_pre_evaluation_training_prefix_i)), 1)`.

Use that origin's prepared prefix and its actual forecast-date cutoff. Reuse the same
scale for every model, reconciler, interval side, and evaluation week. Pass CQR scores
divided by this scale to the existing `core.methods.quantile_integrator_log`; multiply
the returned offsets by the scale before adjusting interval endpoints. Normalized
settings: `KI=1`, `Csat=2`, `lr=0.1`, `T_burnin=5`, `ahead=4`, miscoverage 0.2 split
0.1 per side. Preserve proportional learning rate and the D15 disclosure. No parameter
search is performed for PI. `KI=1` is prospective, not an asserted correction validated
by v1.

Generate all four reconcilers `none`, `bottom_up`, `mint_shrink`, `mint_median` with
`raw`, `pi_norm`, and original `pi` at lr 0.1, giving twelve cells per model. Original PI
uses `KI=1000`, `Csat=2`, and the other v1 controller parameters. Cells outside A-D are
descriptive only. Do not repeat the v1 learning-rate grid. `bottom_up - none` is
structurally zero at the leaf level and is not a new inferential target.

Keep three scale concepts distinct in config, code, and manifest:

1. Model scalers: recomputed per outer/inner fit, using its own available training prefix.
2. PI scale: fixed per node before the common evaluation window, without evaluation truth.
3. Scoring scale: `max(mean(abs(y_i)), 1)` on the common evaluation window, as in D23;
   fixed across arms and bootstrap draws, used only to score outputs.

## 5. Crossing, diagnostics, and failures

Use ascending sorting as the common **base/reconciliation quantile** repair policy,
with D17 relative tolerance 1e-9 for counting crossings. QR repairs its independently
fitted native quantiles; `mint_shrink` retains the existing sort after projection.
Sample quantiles and `mint_median` must already be ordered; unexpected violations abort.
Record counts, maximum and mean absolute adjustments, node/origin identifiers, stage,
and root-specific counts. Store pre-repair values for every repaired cell. Report
post-repair coherence residuals; sorting after a projection can destroy coherence.
V1's 269 COVIDhub crossings were already counted and repaired under D8/D17; v2 does not
reinterpret that registered result as an unrecorded bug.

The repair rule does not sort final PI lower/upper bounds. `lo > hi` represents an empty
conformal set, with coverage false; it is distinct from crossing base quantile levels.
Preserve these limits and record empty-set counts/rates by node, model, arm, and week.
An inspection during spec self-review found 3815 such cells in the v1 intervals table
across all its configurations, not 3815 distinct node/weeks. V1 artifacts and reported
metrics are not rewritten. Empty sets are not a fit failure or grounds for dropping a
model. The v2 score/width reporting definitions below prevent treating a negative signed
span as physical interval width. Never silently swap PI bounds to create a nonempty set.

Capture warnings by category, message, source location, model, origin, and candidate,
with deduplicated messages and occurrence counts. Existing category totals remain.
The synthetic v1 investigation reproduced NVML initialization, CPU `pin_memory`, empty
Lightning validation-loader, and pytree deprecation messages. Its count reconstruction
was 135672 + 946 + 86 = 136704 UserWarnings across 86 fits, and 258 FutureWarnings.
These are reconstructed categories/messages, not preserved original message logs.
They do not demonstrate successful convergence or prove all historical warnings benign.

CPU training uses one device, one compute thread, zero loader workers, deterministic
PyTorch algorithms, no disk checkpoints, and no training-progress output. Disable
pinned memory explicitly. Preserve warning evidence rather than globally filtering
UserWarnings. Any environment change to avoid GPU discovery must be included in
provenance and shown on synthetic fixtures to preserve seeded CPU predictions.

Hard failures include input disagreement/hash mismatch, unknown registry key, invalid
config, nonfinite fit/output/metric, invalid forecast shape, missing eligible model,
solver failure, inability to obtain the common window, dirty/unverifiable Git state,
and destination-exists publication races. No replacement models, discarded candidates,
outcome-based retries, or numerical retuning of the registered realization.

## 6. Endpoints, contrasts, and bootstrap

Principal metrics, computed per node before leaf aggregation:

- `scaled_interval_score`: mean central-80% interval score divided by the scoring scale;
  for crossed final bounds retain its equivalent two-quantile loss definition below.
- `marginal_coverage_error`: `abs(mean(lo <= y <= hi) - 0.8)`.

`leaf_mean` is the unweighted mean of these per-node metrics over the 51 endpoint
leaves. In particular, take the absolute error **before** averaging leaves; errors
above and below target must not cancel. Recompute coverage error from resampled weekly
hits inside each bootstrap replicate, not from an already aggregated scalar. Recompute
interval score using the paired resampled outcomes/bounds and keep the scoring scale
fixed. Lower is better for both metrics.

Report US metrics, per-node metrics, leaf medians, rolling coverage deviation (window
10), marginal coverage, widths, scaled widths, and longest miss runs descriptively.
Preserve legacy metric meanings; add the new metric rather than renaming the old one.

For arbitrary final bounds L and U, retain the existing numeric score
`(U-L) + 10*(L-y)*I(y<L) + 10*(y-U)*I(y>U)`. This equals
`10*(pinball_0.1(y-L) + pinball_0.9(y-U))`, including when L > U. It is the usual
80% interval score for ordered bounds and a quantile-pair score for an empty set; label
this extension explicitly rather than claiming the empty set is an ordinary interval.
The reported principal metric continues to use this unchanged numeric formula, with
no sorting of final PI bounds. Add mean/set width `mean(max(U-L,0))` and its scaled
version for physical-width figures; keep legacy `mean_width`, `median_width`, and their
legacy scaled values as signed-bound-span compatibility diagnostics, labeled as such
where empties occur. Empty sets count as misses in every coverage/run metric. Publish
empty-set rates beside principal endpoint plots and contrast tables so lower scores
cannot be read as evidence that every forecast yields a valid nonempty interval.

For each model and principal metric, let A-D denote that arm's leaf-mean metric. The
six intervention contrasts are:

| Key | Formula | Question |
|---|---|---|
| `pi_effect` | B - A | PI alone |
| `reconciliation_effect` | C - A | Reconciliation alone |
| `combined_effect` | D - A | Combined procedure versus base |
| `reconciliation_given_pi` | D - B | Reconciliation added to PI |
| `pi_given_reconciliation` | D - C | PI added to reconciliation |
| `interaction` | (D - C) - (B - A) | Difference between PI effects |

A negative ordinary contrast favors its treatment. A negative interaction means the PI
effect is more favorable with reconciliation on that metric's scale; it does not mean
the combined procedure improves the baseline or prove a mechanistic interaction.

**Intervention family:** eight models x two metrics x six contrasts = 96 p-values,
jointly Holm-adjusted per block length. Derived/redundant contrasts stay in the declared
family; no after-the-fact pruning to reduce the adjustment.

**Model-to-reference family:** seven generated models x four matched arms x two metrics
= 56 p-values, jointly Holm-adjusted per block. Estimate is generated model minus
COVIDhub under the same arm. No pairwise all-model search, enhanced-versus-original
DeepAR significance family, or US inferential family is added implicitly.

Use the existing paired noncircular moving-block bootstrap, block 8 for the main
exploratory analysis and blocks 4/12 for sensitivity, B=2000, percentile 95% intervals,
and the existing two-sided centered-bootstrap p-value with +1 correction. Within each
block length, use one shared week-index draw matrix for every node, model, arm, metric,
and both families. Bootstrap fixed completed intervals; do not retrain models or
controllers inside resamples. This is temporal uncertainty conditional on fitted
pipelines, not uncertainty over diseases, hierarchies, forecaster populations, or tuning.

With B=2000, the minimum raw p-value is 1/2001 and its worst-rank Holm adjustment in a
96-member family is about 0.048. Disclose this coarse tail resolution and borderline
decisions; do not increase B after inspecting outcomes. Percentile intervals are
pointwise, not simultaneous or Holm-adjusted, and may disagree with p-value thresholds.
Sensitivity blocks cannot replace an unsupported main-block conclusion. Descriptive
arms and metrics carry no significance labels.

## 7. Seed protocol and reproducibility

Fix `root_seed=20261021`, preserving v1 model stream continuity. Use
`SeedSequence(root_seed)` children `(0,)` for forecasting and `(1,)` for evaluation.
Forecast children follow the seven-model configuration order above, then the complete
reference target-grid origin order, including ineligible origins. Classical local fits
spawn by fixed node order. QR is deterministic; still reserve its model/origin slots.

Retain the original three adapters' stream consumption. New neural origin children
spawn distinct candidate-training, validation-path, final-refit, and prediction streams
in documented fixed order. Enhanced candidates follow width order; Transformer has one
candidate. Derive explicit integer seeds and sampler generators from these children;
no randomness from dictionary order, completion order, wall time, or an unseeded global
NumPy generator. Reset validation-path generators reproducibly each epoch. Evaluation
spawns by configured block order `(8,4,12)` and shares each block's draw matrix across
all contrasts. Record the complete stream map in the manifest.

The official v2 run requires a verifiable clean Git worktree including untracked
nonignored files, before expensive generation and again before publication. Git command
failure is an error rather than an empty clean result. Record initial/final Git SHA and
require equality. Retain exact package versions, lockfile/config digests, thread/device
settings, and deterministic-algorithm status. Seeds and deterministic CPU settings do
not promise bitwise identity across arbitrary dependency versions or machines.

## 8. Components and configuration boundaries

Keep the existing flat `src/hcp/` package. Numeric modules perform no file I/O. Public
functions have type hints and Google docstrings; dataclasses are frozen and arrays
protected; new Python files start with two ABOUTME lines. Target 150-300 lines per module
and split focused helpers before approximately 400 lines.

| Module | Responsibility |
|---|---|
| `benchmark_data.py` | Canonical actuals, reference bundle, training prefix boundaries |
| `forecast_types.py` | Prepared histories and frozen native prediction contract |
| `forecast.py` | Registry ownership, weekly generation, shared bundle construction |
| `forecast_classical.py` | ARIMA, SARIMA, Prophet adapters |
| `forecast_quantile.py` | Direct QR features, alpha selection, native quantile repair |
| `forecast_neural.py` | Legacy DeepAR and fixed Transformer adapters |
| `forecast_deepar.py` | Enhanced recurrent module, embeddings, weighted dataset integration |
| `forecast_validation.py` | Causal inner splits, neural selection, in-memory stopping/refit |
| `conformal.py` | Existing core adapters plus explicit normalized-controller boundary |
| `benchmark.py` | Common window, arm orchestration, result assembly |
| `benchmark_stats.py` | Paired factorial/model contrasts and the two Holm families |
| `benchmark_report.py` | In-memory figures, with raw and normalized PI visible |
| `config.py` | Frozen versioned benchmark configuration and validation |
| `run.py` | File reads/writes, Git gates, manifest, atomic publication |

These are responsibility boundaries, not permission for unrelated refactoring. Split
an enhanced-model helper further if necessary to keep modules reviewable. Preserve
existing imports through small compatibility re-exports from `forecast_models.py`.
Legacy characterization and harness contracts remain unchanged.

Add `configs/model_benchmarks_v2.toml`, with `design="benchmark"` and an explicit
`protocol_version=2` in the benchmark settings. Missing version means v1 with its current
fixed constraints; v2 gets separate validation rather than weakening those constraints.
Reject incompatible tags, model/arm lists, grids, and inference settings before fitting.
No Hydra or plugin selection at runtime. Dependencies change only with `uv add` during
implementation; no dependency is installed by writing this specification.

## 9. Artifacts, reporting, and publication

Only `run.py` writes under `results/model-benchmarks-v2/`. Build in a sibling staging
directory, write the manifest last, and publish with the existing atomic no-replace
primitive. A destination created by another process during generation/publication must
never be overwritten. No final directory remains after a failed run. Runtime logging
is retained separately from scientific artifacts, and no cache becomes an alternate
official publisher.

Write the six legacy benchmark tables (base forecasts, intervals, node metrics, level
metrics, hierarchy contrasts, model contrasts), plus `forecast_diagnostics.parquet`,
`tuning.parquet`, `quantile_repairs.parquet`, and `warnings.parquet`. The hierarchy table
now identifies the six contrast keys and A-D operands; both contrast tables include
family ID/size, block, raw and Holm p-values, pointwise interval, model, and metric.
Descriptive metrics carry explicit role labels; no bootstrap inference rows are created
for descriptive-only cells. Base forecasts identify sample-derived versus native
quantiles. PI scales are indexed by node in manifest diagnostics.

The manifest contains frozen config and hashes, registered protocol identifier, model
classes and settings, seed stream map, device/versions, clean Git evidence, common dates
and endpoint nodes, candidate/actual fit counts, selected alphas/widths/epochs, warnings,
interpolation, stage-specific repairs, coherence residuals, negative forecasts and
truth counts, timings, PI/model scale definitions, scoring scales, and both Holm family
definitions. Include D15 limitations and a statement that v1 outcomes informed v2.

Figures must show all four principal arms, with raw and normalized PI distinguishable:
model-quality endpoint plots; factorial contrast forests with pointwise-interval labels;
coverage-width/score plots; and original-versus-normalized PI diagnostics by node scale.
Put US and other descriptive reconcilers in clearly labeled separate panels. Do not
present only original PI figures or interpret overlapping pointwise intervals as a test.

After the official run, analysis scripts read results and write only under
`docs/analysis/`; the report `docs/analysis/model-benchmarks-v2.md` starts with window,
missingness, warnings, and tuning diagnostics before outcomes. Retain null/negative
results and failed runs. Changed numerical choices after inspection require a distinct
prospective protocol/run identity, not overwriting or retuning this realization.

## 10. Verification and authorization gates

Implementation follows TDD after written-spec and implementation-plan approval. Required
coverage includes:

- Legacy v1 configs, adapters, metrics, and seed streams remain compatible; `core/`
  characterization/harness tests stay green.
- Tagged sample/native-quantile contracts, defensive immutability, registry validation,
  exact fourth-step extraction, QR horizon alignment, sorting and repair accounting.
- Prefix perturbation tests establish no post-origin information in preprocessing,
  features, alpha selection, width/epoch selection, embedding/covariates, or PI scale.
- Synthetic inner-split fixtures prove purged QR response availability and nine genuine
  four-step neural validation origins; one-step validation loss cannot pass as the
  declared objective. Monte Carlo mixture NLL agrees with simple analytic fixtures.
- Weighted sampler frequencies, seed determinism, node-order preservation, embedding
  participation, Gaussian unit restoration, selected-epoch full refits, and candidate
  failure behavior are tested independently of forecast quality.
- Normalized PI is equivalent to core on scaled scores with rescaled offsets; `raw`
  identity and legacy PI behavior remain intact. D15 behavior is characterized, not
  silently claimed fixed.
- Four-arm formulas and interaction signs, per-node absolute coverage aggregation,
  shared paired draws, fixed scoring scales, 96/56 family memberships, sensitivity
  separation, empty-set coverage/width semantics, equality of the score to two pinball
  losses on ordered and crossed bounds, and zero-effect fixtures.
- Real-adapter integration on small synthetic weekly series, including Prophet and both
  new neural adapters; reduced real-data availability checks outside the official v2
  outcome. E2E fake-adapter tests verify artifacts, manifest, dirty-Git rejection,
  failures, and destination-appears publication races.

Before launch: full unit/integration/e2e suite, Ruff check and format check, whole-branch
review, finding re-grade, Critical/Important fixes and re-review, and fresh verification
with a clean implementation worktree. No official v2 pipeline is executed or inspected
by a reviewer as a runtime probe. Use synthetic or explicitly reduced development
fixtures instead.

`docs/protocol-decisions.md` overrides this spec. D31 closes the v1 study and discourages
mechanical arm-grid expansion. Before v2 launch, append D32 (or the next unused decision
if the ledger has advanced), explicitly recording this new exploratory study, v1-informed
choices, eight-model set, four inferential arms, two families, and unchanged D15 limits.
Do not amend D30/D31. The registration commit must precede generation or inspection of
the official result. Launch the registered identity once only after every gate passes.

This document does not authorize implementation or launch. Next: user reviews this
written spec; after approval, invoke `writing-plans`, produce the formal implementation
plan, and obtain the execution-method selection. Implementation then uses an isolated
worktree from freshly verified `develop`, local Conventional Commits, no push, and no
`--no-verify`.

## 11. References and deliberate exclusions

- Repository: D15, D23, D25, D30-D31 in `docs/protocol-decisions.md`; v1 specification
  `2026-10-21-probabilistic-model-benchmarks-design.md` and v1 outcome report.
- [DeepAR](https://arxiv.org/abs/1704.04110): global autoregressive likelihood modeling,
  scale treatment, weighted sampling, embeddings, and calendar/age features. V2's Gaussian
  variant, short context, absolute-value scale, and temporal validation are declared
  adaptations, not an exact reproduction of the count-data experiments.
- [Informer](https://arxiv.org/abs/2012.07436): efficient long-sequence attention; the
  current twelve-step input does not motivate adding this efficiency mechanism.
- [Autoformer](https://arxiv.org/abs/2106.13008): long-term decomposition/autocorrelation;
  adopting it would introduce another architecture and seasonal assumptions to specify.
- [PatchTST](https://arxiv.org/abs/2211.14730): temporal patches and channel-independent
  shared weights; longer-context patch design is a separate future question.
- [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012): multi-horizon quantiles and mixed
  covariates, a relevant alternative if those become the attention model's main purpose.

These are different architectures, not hyperparameters borrowed for the fixed Darts
Transformer. Their exclusion is a scope decision, not evidence that they perform worse.
Theta is also excluded: its native point-forecast contract would require a separately
specified uncertainty model. No result from the fixed Transformer is generalized to
these excluded families or to state-of-the-art performance.
