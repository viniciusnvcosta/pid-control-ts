# Probabilistic Model Benchmarks v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved eight-model exploratory benchmark with four principal arms, normalized PI, native quantiles, causal tuning, and immutable publication.

**Architecture:** Extend the existing flat `hcp` pipeline with a tagged immutable forecast object and focused model/validation modules. Preserve the v1 APIs and configuration path; v2 explicitly selects its endpoints, controllers, and joint Holm families. `run.py` remains the only scientific-artifact writer.

**Tech Stack:** Python >=3.11.13, NumPy, pandas, scikit-learn QuantileRegressor, Darts/PyTorch/Lightning, Prophet, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-10-22-probabilistic-model-benchmarks-v2-design.md` (approved in chat after commit `8af47cf`). Read both documents; `docs/protocol-decisions.md` overrides this plan.

## Global Constraints

- Local Conventional Commits, no push, no `--no-verify`; no changes to `core/` or immutable v1 artifacts/config/spec/protocol entries.
- Numeric modules perform no file I/O. Only `run.py` writes under `results/<run_id>/`; staging plus atomic no-replace publication.
- Frozen dataclasses with protected arrays; public type hints and Google docstrings; two ABOUTME lines per Python file; target 150-300 lines, split before approximately 400.
- Dependencies change only through `uv add`; add Prophet during Task 5, not during planning.
- Generated order: `arima`, `sarima`, `deepar`, `prophet`, `transformer`, `qr_l1`, `deepar_enhanced`; reference `covidhub` / `COVIDhub-4_week_ensemble`.
- `root_seed=20261021`; forecasting spawn key `(0,)`, evaluation `(1,)`; preserve first three adapter streams and reserve ineligible origin positions.
- All 57 nodes; 51 state/DC endpoint leaves; no territories in leaf aggregates; four-step forecasts, weekly refits, minimum 60 training weeks, MinT warm-up 20, minimum 60 common evaluated weeks.
- Quantiles `(0.1,0.5,0.9)`, 500 native predictive paths, no clipping at zero or pseudo-samples for QR.
- QR: 12 latest observations, direct horizon four, intercept, HiGHS, alpha `[0.001,0.01,0.1,1.0]`, joint three-quantile holdout selection, exact ties choose larger alpha, 12 holdout targets.
- New neural models: holdout 12 weeks / nine valid fourth-step targets, max 100 epochs, patience 10, `min_delta=0`, selected-epoch full-prefix refit. Enhanced widths `[40,80,120]`, three layers, embedding 20; Transformer settings exactly spec section 3.4.
- Normalized PI: fixed pre-evaluation node scale, `KI=1`, `Csat=2`, `lr=0.1`, `T_burnin=5`, `ahead=4`, alpha 0.2 split per side. Original PI stays descriptive with `KI=1000`.
- A/B/C/D = none/raw, none/pi_norm, mint_median/raw, mint_median/pi_norm; twelve total descriptive/principal cells per model at lr 0.1.
- Endpoints: mean per-node scaled interval score and absolute marginal coverage error; 96 intervention and 56 reference tests per block, separate joint Holm families; blocks `(8,4,12)`, B=2000; shared draws across models/nodes/arms/families.
- Preserve empty conformal sets, coverage false, original signed bounds and quantile-pair score; add nonnegative physical set width and empty-set diagnostics. Never sort final PI bounds.
- Preserve D15: final revised truth and inherited PI learning-rate look-ahead. No claim that full pipeline is strictly causal or v2 confirmatory.
- Do not generate or inspect the official v2 realization before registration, complete review/fix wave, and fresh verification. Planning/implementation review uses synthetic or reduced development fixtures.

## Review Focus

1. Leading/internal gaps near inner validation boundaries must not be interpolated using holdout/future truth (Tasks 2, 6).
2. Native-quantile objects backed by caller-owned arrays must remain unchanged after caller mutation (Task 1).
3. Empty final PI sets must remain misses with a valid quantile-pair score and physical width zero (Task 3).
4. Git failure, staged/untracked changes, or a competing destination at publication must fail safely without altering another process's artifacts (Task 13).
5. Candidate failure, tied NLLs, or Monte Carlo RNG advancement must not silently change the selection objective, candidate set, or subsequent seeds (Tasks 6-9).

## Execution protocol and dependencies

The user selected **subagents + TDD**. This document is the plan-review artifact; approval of the spec and selection of method do not skip review of the newly written plan. After plan approval, execute all implementation tasks without routine reconfirmation.

- Use `using-git-worktrees` to create/verify an isolated branch from freshly verified `develop`. Check clean status, branch ancestry, and that this approved spec and plan are available; if absent from develop, bring only their documentation commits into the implementation branch. Do not reset or silently replace develop with main; report inability to refresh remotes separately from local evidence.
- Create this plan's workspace using `subagent-driven-development/scripts/sdd-workspace`; maintain `progress.md`, task briefs, reports, review packages, red/green evidence, and rulings there. Resume from completed ledger entries after compaction.
- Before Task 1, record a preflight table covering each task's internal consistency and every shared-file/interface dependency. The controller self-reviews the written plan; this is not a delegated plan-review task.
- Dispatch one fresh implementer per task with `fork_turns="none"`, the task brief, spec, global constraints, exact predecessor interfaces, and report path. Implementers never spawn helpers/reviewers. Do not run implementers concurrently in the shared checkout.
- Select model/effort explicitly using the skill's role/complexity guidance and the live model allowlist: standard for integration; stronger for custom neural/causality work; final whole-branch reviewer on the most capable available model. Preserve both settings on dispatch.
- Each code task: failing behavioral test -> run and record intended RED -> minimal implementation -> focused GREEN -> refactor with GREEN -> local commit -> independent task reviewer verdicts for both spec compliance and code quality. Add each next behavioral case through its own red/green cycle.
- Review packages cover the recorded task BASE through actual HEAD, not just HEAD~1. Fix rounds resume the implementer (fresh stronger worker after round three), get scoped re-review, and follow the skill's five-round cap/ruling ledger. Scientific Critical/Important findings must actually be resolved before registration/launch even if execution bookkeeping parks one.
- Task dependency spine: 1 -> 2; 1 -> 3; 2 -> 4/5/6; 6 -> 7 -> 8/9; 3+4+5+8+9 -> 10 -> 11 -> 12 -> 13 -> 14. Execute the numbered order; dependencies explain what each brief must carry.
- Commands below use `uv run pytest -o addopts=''` so project defaults do not silently exclude integration tests. On read-only uv-cache failure use `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache`. Test fixtures never use the registered output identity. Network/cache failures are environmental evidence, not passing tests.
- Administrative verification/review/documentation steps are not forced into artificial RED tests. Do not add tests that merely mirror implementation constants without checking behavior.

## File map and shared interfaces

Existing numeric entry points remain: `generate_model_bundles(actuals, dates, forecast_dates, cfg) -> ForecastResult`, `run_benchmark(bundles, cfg) -> BenchmarkResult`, and `conformalize(y, q_lo, q_hi, controller, cfg) -> tuple[np.ndarray,np.ndarray]`.

New modules and types are defined by their owning tasks below. Common model adapter signature is `(histories: tuple[PreparedHistory,...], settings: ForecastSettings | V2ModelSettings, seed: np.random.SeedSequence) -> NativePrediction`. Legacy `*_samples` functions keep their ndarray return; registry wrappers adapt those arrays for v2. Do not make existing fake-adapter/unit tests accept an accidental incompatible legacy API.

Use `DiagnosticsTables` (Task 1) to carry four in-memory DataFrames: `forecast_diagnostics`, `tuning`, `quantile_repairs`, `warnings`. Preserve `ForecastResult.diagnostics` and `BenchmarkResult.diagnostics` mappings for existing callers; add defaulted typed table fields rather than removing fields.

### Task 1: Immutable forecast contract and compatibility boundary

**Files:** Create `src/hcp/forecast_types.py`, `tests/unit/test_forecast_types.py`; modify `src/hcp/forecast_models.py`, `src/hcp/forecast.py`, `tests/unit/test_forecast.py`.

**Interfaces:** Produce frozen `NativePrediction(nodes, ahead, kind, samples=None, quantiles=None, levels=(0.1,0.5,0.9), diagnostics=None)` with defensive read-only copies; `prediction_quantiles(prediction: NativePrediction) -> np.ndarray`; frozen `DiagnosticsTables` with the four named tables; frozen `V2ModelSettings` containing the spec's fixed model/selection settings. Move/re-export `PreparedHistory` and `ForecastSettings` without changing legacy fields/defaults. Extend `PreparedHistory` only with defaulted optional `raw_values`, `observed`, and `forecast_dates` arrays; v2 populates them from canonical raw history before outer interpolation. Legacy three-argument construction remains valid. Explicitly map every field in `V2ModelSettings` to the approved spec's model tables during this task, so later adapters share one settings vocabulary.

- [ ] Write failing shape/tag/horizon/aliasing tests and legacy import tests. Representative behavioral test:
  ```python
  def test_native_quantiles_own_their_storage():
      source = np.array([[1., 2., 3.]])
      result = NativePrediction(nodes=("us",), ahead=4, kind="quantiles", quantiles=source)
      source[0, 0] = 99
      assert result.quantiles[0, 0] == 1
      assert not result.quantiles.flags.writeable
  ```
  Also reject both payloads, wrong quantile levels, wrong node count, nonfinite payloads; sample extraction must retain step four, not step one.
- [ ] RED: `uv run pytest -o addopts='' tests/unit/test_forecast_types.py -q`; confirm intended missing-contract failure.
- [ ] Implement contract, sample extraction, defaulted diagnostic tables, and compatibility re-exports. Keep legacy adapters and registry behavior usable while v2 wrappers are added later. Protect all optional metadata arrays and preserve missing values in raw_values (finite validation applies to model payloads, not deliberately missing raw observations).
- [ ] GREEN: run new tests plus `tests/unit/test_forecast.py` and `tests/unit/test_core_harness_contract.py`.
- [ ] Commit `feat(forecast): add immutable native prediction contract`; independent task review.

### Task 2: Purged inner splits and explicit seed streams

**Files:** Create `src/hcp/forecast_validation.py`, `tests/unit/test_forecast_validation.py`; modify `src/hcp/forecast.py`, `tests/unit/test_forecast.py` only for shared prefix preparation.

**Interfaces:** Produce frozen `TemporalSplit(training_cutoff, validation_targets, validation_cutoffs)`; `qr_split(dates: np.ndarray, observed: np.ndarray, forecast_dates: np.ndarray) -> TemporalSplit | None`; `neural_split(dates: np.ndarray, observed: np.ndarray, forecast_dates: np.ndarray) -> TemporalSplit | None`; `prepare_prefix(history: PreparedHistory, cutoff: np.datetime64) -> PreparedHistory`; `neural_seed_streams(seed: np.random.SeedSequence, n_candidates: int) -> dict[str, tuple[np.random.SeedSequence,...]]`. Stream keys are `training`, `validation`, `refit`, `prediction`; training/validation have candidate count entries, refit/prediction one each, spawned in that order.

- [ ] Write RED tests `test_qr_responses_are_purged_by_forecast_date`, `test_neural_holdout_scores_nine_fourth_steps`, `test_gap_interpolation_cannot_cross_split`, `test_ineligible_origins_reserve_seed_slots`. With 60 weekly inputs require twelve QR validation targets, at least twelve train pairs, and nine neural target positions 51..59 for zero-based dates; shifting unseen targets must leave prefixes and seed keys unchanged.
- [ ] RED: run `tests/unit/test_forecast_validation.py` and the new seed test in `test_forecast.py`.
- [ ] Implement raw-prefix slicing from `history.raw_values` before interpolation, observed-truth eligibility, leading-gap handling, actual date cutoffs, and deterministic stream creation. Populate this metadata in outer preparation from canonical actuals. For inner targets preceding the submitted grid, construct dates using the same fixed 26-day forecast lag and enforce the 28-day last-observation boundary. Reject missing raw metadata on v2 validation paths rather than treating interpolated outer values as raw truth. Return `None` only for pre-fit structural ineligibility, never catch solver/model failures as ineligibility.
- [ ] GREEN: same tests plus `tests/unit/test_benchmark_data.py`; verify irregular calendars and incomplete holdouts fail explicitly.
- [ ] Commit `feat(forecast): add causal validation splits and seed streams`; review.

### Task 3: Normalized PI and empty-set metric semantics

**Files:** Modify `src/hcp/conformal.py`, `src/hcp/evaluate.py`, `src/hcp/panel_metrics.py`; create `tests/unit/test_conformal_normalized.py`, `tests/unit/test_benchmark_metrics_v2.py`.

**Interfaces:** Extend `conformalize` with optional keyword-only `scale: float | None = None`, preserving existing calls; `pi_norm` requires a finite positive explicit scale. Produce `fixed_pi_scales(histories: tuple[PreparedHistory,...]) -> np.ndarray` in `conformal.py`; `v2_node_metrics(y, lo, hi, scale, *, alpha, window) -> dict[str,np.ndarray]` in `panel_metrics.py` adding `marginal_coverage_error`, `mean_set_width`, `scaled_mean_set_width`, `empty_set_rate` without changing legacy metrics.

- [ ] Write failing scale-equivariance and no-look-ahead tests. Assert normalized offsets equal `scale * core_PI(scores/scale, KI=1)["q"]`, raw remains identity, and omission of scale fails only for pi_norm.
- [ ] Add crossed-bound fixture `y=1,L=2,U=0`: coverage false, set width zero, signed width -2, unscaled score 18. For two leaves with coverages 0.6/1.0 assert mean absolute coverage error 0.2 rather than zero.
- [ ] RED: `uv run pytest -o addopts='' tests/unit/test_conformal_normalized.py tests/unit/test_benchmark_metrics_v2.py -q`.
- [ ] Implement adapter over imported core; never normalize using evaluation truth. Keep numeric score identical to existing two-pinball formula and preserve inverted final bounds.
- [ ] GREEN: new tests plus `test_conformal.py`, `test_evaluate.py`, core characterization and harness contracts.
- [ ] Commit `feat(conformal): add normalized PI and empty-set diagnostics`; review.

### Task 4: Local QR with joint causal alpha selection

**Files:** Create `src/hcp/forecast_quantile.py`, `tests/unit/test_forecast_quantile.py`; add `tests/integration/test_forecast_quantile.py`.

**Interfaces:** `direct_examples(history: PreparedHistory, cutoff: np.datetime64) -> tuple[np.ndarray,np.ndarray]`; `qr_prediction(histories, settings, seed) -> NativePrediction`. Consume Tasks 1-2 types/splits. Diagnostics contain per-node chosen alpha, candidate losses, training scale/bounds, final crossing cells and pre/post values.

- [ ] RED tests: `test_features_end_four_weeks_before_response` asserts `[y[u],...,y[u-11]] -> y[u+4]`; `test_alpha_is_shared_across_quantiles_and_ties_choose_larger`; `test_holdout_scale_excludes_validation_values`; `test_solver_failure_does_not_drop_candidate`; `test_crossing_repair_preserves_negative_forecasts`.
- [ ] Run `uv run pytest -o addopts='' tests/unit/test_forecast_quantile.py -q` and record intentional failures.
- [ ] Implement native HiGHS QuantileRegressor fits, four alphas x three quantiles, mean unrepaired pinball selection, full-prefix refit, original-unit restoration and sorted final output. Protect horizon-four semantics and expose no fake sample dimension.
- [ ] GREEN: focused unit file and small synthetic real-solver integration file; repeat seeded predictions and perturb post-cutoff truth.
- [ ] Commit `feat(forecast): add causally selected local L1 quantiles`; review.

### Task 5: Classical adapter split and Prophet

**Files:** Create `src/hcp/forecast_classical.py`, `tests/unit/test_forecast_prophet.py`, `tests/integration/test_forecast_prophet.py`; modify `forecast_models.py`, `pyproject.toml`, `uv.lock`.

**Interfaces:** Move/re-export unchanged `arima_samples`, `sarima_samples`; produce `prophet_prediction(histories, settings, seed) -> NativePrediction`. Add native wrappers for classical samples without changing their RNG consumption.

- [ ] RED: unit fake-Prophet test checks explicit disabled seasonalities, linear growth, 25 changepoints, range 0.8, prior 0.05, mcmc 0, weekly timestamps and per-node child seeds. Assert 500 four-step draws and no clipping; fit failure propagates with node/origin context.
- [ ] Run focused unit file before implementing the adapter.
- [ ] Add dependency via `uv add prophet`; implement lazy optional imports so legacy imports do not require Prophet before it is used. Preserve original ARIMA/SARIMA numerical behavior. When relocating a dependency, update test monkeypatch targets to the actual consuming module rather than leaving a compatibility alias that no longer intercepts calls.
- [ ] GREEN: focused unit/integration tests plus existing `tests/integration/test_forecast_models.py`; real Prophet on small synthetic weekly data must produce finite reproducible original-unit outputs.
- [ ] Commit `feat(forecast): add seeded weekly Prophet adapter`; review.

### Task 6: Neural selection objective and stopping/refit lifecycle

**Files:** Extend `src/hcp/forecast_validation.py`; create `src/hcp/forecast_neural_validation.py` if needed for size; create `tests/unit/test_forecast_neural_validation.py`.

**Interfaces:** Frozen `SelectionResult(candidate: int, best_epoch: int, best_nll: float, tuning: pd.DataFrame)`; Protocol `NeuralCandidate` with `train_epoch() -> None`, `fourth_step_nll(split: TemporalSplit, seed: np.random.SeedSequence) -> np.ndarray`, `fit_full(epochs: int, seed: np.random.SeedSequence) -> None`; `select_neural(candidates: tuple[NeuralCandidate,...], split: TemporalSplit, seed: np.random.SeedSequence, *, max_epochs: int=100, patience: int=10) -> SelectionResult`; `gaussian_mixture_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray` for path axis last. Numeric development fixtures can explicitly reduce epochs/paths, but registered config validation fixes 100/10/500. Candidates retain the full raw outer history as well as their inner split; `fit_full` rebuilds preprocessing/model state from the full outer prefix and never reuses inner-fit scales or weights.

- [ ] RED tests: known constant-Gaussian mixture matches analytic NLL; finite log-sum-exp under extreme densities; nine fourth-step validations exclude target leakage; losses `[3,2,2,...]` select epoch two and stop after ten unimproved epochs; tied candidates choose smaller width; any nonfinite candidate aborts; resetting validation stream gives identical path draws at each epoch.
- [ ] Run focused unit test file; use deterministic fake candidates to avoid 100-epoch test training.
- [ ] Implement equal-node/equal-origin original-unit NLL, 500-path mixture for autoregression, strictly lower comparison/min_delta zero, max 100, in-memory best-state ownership, and selected-epoch refit lifecycle. Do not replace objective with generic one-step Lightning val_loss.
- [ ] GREEN: all lifecycle/split tests, including one intentionally misleading one-step-loss candidate that must lose under fourth-step NLL.
- [ ] Commit `feat(forecast): implement fourth-step neural model selection`; review.

### Task 7: Enhanced DeepAR module, embeddings, and weighted windows

**Files:** Create `src/hcp/forecast_deepar.py`, `src/hcp/forecast_deepar_data.py`, `tests/unit/test_forecast_deepar.py`.

**Interfaces:** `EnhancedRNNModule` subclasses Darts `CustomRNNModule` with Gaussian parameters; `enhanced_covariates(histories, training_cutoff, prediction_dates) -> tuple[np.ndarray,...]`; `weighted_window_sampler(window_counts: np.ndarray, scales: np.ndarray, seed: np.random.SeedSequence) -> torch.utils.data.Sampler`; factory `make_enhanced_candidate(histories, settings, width: int, seed) -> NeuralCandidate`.

- [ ] RED: changing node ID with equal autoregressive inputs changes Gaussian parameters through a learned 20-dimensional embedding; 3 LSTM layers and widths 40/80/120 have valid shapes; age origin is the first observed date, calendar extends through prediction, and covariate stats exclude holdout.
- [ ] Add sampler test with unequal window counts: empirically/node-exact probability mass follows nu, not nu times number of windows; use deterministic inverse-CDF boundaries or a fixed-seed large draw with justified tolerance. Assert replacement, total-window epoch length, explicit generator and consumption by the actual training DataLoader.
- [ ] Run focused tests RED, implement custom Darts module/dataset hookup, nu=`1+mean(abs(prefix))`, Gaussian scale restoration, no disk checkpoints, one-thread CPU/zero workers/pin_memory false.
- [ ] GREEN: shape/gradient/embedding/sampler/unit restoration tests; keep Gaussian library machinery imported rather than reimplementing a trainer.
- [ ] Commit `feat(forecast): add embedded weighted DeepAR components`; review.

### Task 8: Enhanced DeepAR adapter and real synthetic integration

**Files:** Extend `forecast_deepar.py`; create `tests/integration/test_forecast_deepar_enhanced.py`, `tests/unit/test_forecast_deepar_selection.py`.

**Interfaces:** `enhanced_deepar_prediction(histories, settings, seed) -> NativePrediction`, consuming Tasks 6-7 and returning `[node,4,500]` paths plus all candidate and final-fit diagnostics.

- [ ] RED: fake candidate lifecycle asserts width grid order, chosen epoch refit on full outer history, distinct train/validation/refit/predict streams, and complete candidate warning records even when one raises.
- [ ] Run unit RED; implement adapter using 12 input/16 training length, shared 57-node vocabulary and fixed sampler/covariate contracts.
- [ ] GREEN: unit file and synthetic real-Darts integration with development-only reduced epochs/samples. A separate fake-backend contract test checks production 100/10/500 settings; development reductions must never change registered defaults. Repeat a seed, perturb a future observation, verify scales and outputs unchanged before its availability.
- [ ] Commit `feat(forecast): integrate selected enhanced DeepAR forecasts`; review.

### Task 9: Fixed Transformer adapter and legacy neural compatibility

**Files:** Create `src/hcp/forecast_neural.py`, `tests/unit/test_forecast_transformer.py`, `tests/integration/test_forecast_transformer.py`; modify compatibility exports in `forecast_models.py`.

**Interfaces:** Re-export legacy `deepar_samples` with unchanged API/seed use; `transformer_prediction(histories, settings, seed) -> NativePrediction`; Transformer candidate implements Task 6 protocol with direct fourth-step Gaussian NLL.

- [ ] RED: fake backend verifies 12/4 chunks, d_model32, heads4, encoder2/decoder2, FF128, dropout0.1, ReLU, standard LayerNorm/sinusoidal positions, Gaussian likelihood, Adam0.001/batch32 and MC dropout off at prediction. Assert per-fit max(mean(abs(prefix)),1) scaling and selected-epoch full refit.
- [ ] Run focused unit RED; implement candidate and adapter using shared selection utilities. Keep dropout stochastic only during training and preserve legacy DeepAR scaling and ten-epoch behavior. Existing tests patch `forecast_models.RNNModel`; either keep that consuming boundary or update the patch to `forecast_neural.RNNModel` and demonstrate that the fake still intercepts the actual adapter call.
- [ ] GREEN: focused unit/real synthetic integration and original forecast-model integration tests; validate repeated seed, fourth-step extraction and post-origin perturbation invariance.
- [ ] Commit `feat(forecast): add fixed global probabilistic Transformer`; review.

### Task 10: Versioned config, registry wiring, and forecast diagnostics

**Files:** Modify `src/hcp/config.py`, `src/hcp/forecast.py`; create `src/hcp/config_benchmark_v2.py`, `configs/model_benchmarks_v2.toml`, `tests/unit/test_config_benchmark_v2.py`, `tests/unit/test_forecast_v2.py`.

**Interfaces:** Frozen `BenchmarkV2Settings` in `config_benchmark_v2.py`, separate from legacy `BenchmarkSettings`; missing protocol_version selects unchanged v1. Extend `benchmark_configs(cfg)` to emit version-aware numeric configs; `ForecastConfig` gains default protocol_version1 and optional v2 settings; retain `generate_model_bundles` signature. Populate the one `FORECASTERS` registry with native-wrapper adapters for v2 and legacy-compatible dispatch for v1.

- [ ] RED: load old TOML and assert its serialized config/behavior unchanged; v2 selects exactly seven generated keys and spec settings; invalid versions/arms/seeds/unknown keys fail before fit. A recording adapter proves first three model/origin seed states match v1 while ineligible slots stay reserved.
- [ ] Add warning tests: category/message/source/candidate dedup retains counts, exception context, and no global UserWarning filter. Native QR rows identify quantile origin and other rows sample-derived origin; all diagnostic tables have schemas even when empty.
- [ ] Run focused RED; implement version dispatch, config file and full registry integration, splitting modules as specified. Add no official result directory or pipeline execution.
- [ ] GREEN: new tests plus existing forecast/config/run unit tests and v1 config parsing regression.
- [ ] Commit `feat(config): wire versioned cumulative benchmark forecasts`; review.

### Task 11: Common window, twelve cells, and factorial inference

**Files:** Modify `src/hcp/benchmark.py`, `src/hcp/panel.py` only through backward-compatible optional boundaries; create `src/hcp/benchmark_stats.py`, `tests/unit/test_benchmark_v2.py`, `tests/unit/test_benchmark_stats_v2.py`.

**Interfaces:** `BenchmarkConfig` gains protocol_version1 default; `BenchmarkResult` gains defaulted `DiagnosticsTables`; `factorial_contrasts(metric_values: Mapping[str,np.ndarray]) -> dict[str,np.ndarray]` accepts A-D arrays; `v2_contrast_tables(arms: Mapping[str,ForecasterArms], draws: Mapping[int,np.ndarray], *, alpha: float, window: int) -> tuple[pd.DataFrame,pd.DataFrame]`. Preserve legacy `run_benchmark` path unchanged.

- [ ] RED: synthetic eight-bundle fixture establishes all 57 finite forecasts, 51 scored endpoint leaves, common start with twenty known errors and >=60 weeks; 59 weeks aborts. PI scales derived once at first eval cutoff are identical across models/arms and insensitive to later truth.
- [ ] Contrast test uses A=10,B=8,C=9,D=6: expect -2,-1,-4,-2,-3,-1 for the six spec keys. Assert exactly 96/56 rows per block, family IDs/sizes, no US/descriptive inference rows, model-reference sign, and shared index matrices across both families.
- [ ] Recompute per-node absolute coverage error on each resample; keep scoring scale fixed and controller/model fits outside resampling. Add empty-set and all-zero contrast fixtures.
- [ ] Run focused RED; implement twelve-cell orchestration, fixed PI scales, new metrics, factorial/reference tables using imported `draw_blocks`, `summarize_diff`, `holm`. Draw once per `(8,4,12)` with B2000; no per-family independent draws.
- [ ] GREEN: new files plus existing benchmark/panel/stats unit suites; verify v1 6-member model families unchanged.
- [ ] Commit `feat(benchmark): add paired factorial arms and joint Holm families`; review.

### Task 12: Four-arm figures and readable diagnostic reporting

**Files:** Modify `src/hcp/benchmark_report.py`; create `src/hcp/benchmark_report_v2.py`, `tests/unit/test_benchmark_report_v2.py`.

**Interfaces:** `v2_figures(result: BenchmarkResult, *, block: int=8) -> dict[str,matplotlib.figure.Figure]`; existing `benchmark_figures` dispatches by result/config version without I/O. Names: `model-quality`, `factorial-effects`, `coverage-score-tradeoff`, `pi-scale-diagnostics`, `us-descriptive`.

- [ ] RED figure-contract tests inspect data/artists/labels: all A-D present, raw distinguishable from normalized PI, empty rates visible, physical widths nonnegative, signed spans never labeled ordinary width when empty; US descriptive; intervals labeled pointwise, significance based on Holm values.
- [ ] Run focused RED; implement in-memory plots, with original PI and additional reconcilers in descriptive panels; no plotting-only outcome selection or clipping CI evidence.
- [ ] GREEN: new and existing benchmark-report tests; close figures in fixtures and verify no files created by numeric report functions.
- [ ] Commit `feat(report): show four-arm benchmark and empty-set diagnostics`; review.

### Task 13: Provenance, atomic writer, and end-to-end v2 CLI

**Files:** Modify `src/hcp/run.py`; create `tests/unit/test_run_benchmark_v2.py`, `tests/e2e/test_run_benchmark_v2_cli.py`.

**Interfaces:** `require_clean_git(repo: Path) -> str` in `run.py` raises on Git error, staged/unstaged/untracked nonignored changes; `write_benchmark_artifacts` keeps its public signature and version-dispatches to the ten-table v2 writer. Store initial SHA and require unchanged clean state before publication.

- [ ] RED: fixture Git repos cover clean status, untracked file, staged edit, command error and SHA changed between fit/publish. Test the existing no-replace primitive with a destination appearing immediately before rename; another process's sentinel survives and no owned final output appears on failure.
- [ ] E2E fake native/sample adapters produce all eight models/twelve cells, ten parquet files, five figure PDFs and manifest. Assert manifest last, tables' roles/96+56 families, PI scale map, seeds, selected candidates, repair records, exact dependency/device/config/lockfile hashes and D15/exploratory disclosure. Empty warning/repair tables still write valid schemas.
- [ ] Run focused RED; implement I/O solely in run.py. Clean up only this attempt's staging path on failure; never delete an existing final destination, a pre-existing staging directory, or another process's files. Git ignored results/staging must not dirty the worktree check.
- [ ] GREEN: unit/e2e v2 tests plus all existing CLI/atomic-writer regression tests, using temporary output identities only.
- [ ] Commit `feat(run): publish provenance-checked v2 benchmark artifacts`; review.

### Task 14: Full verification, whole-branch review, and preregistration readiness

**Files:** Add `tests/integration/test_benchmark_v2_development.py`, `docs/analysis/model-benchmarks-v2-protocol-readiness.md`; update `README.md` and append a new entry to `docs/protocol-decisions.md` only after review/fixes. Do not create an outcome report before an outcome exists.

**Interfaces:** This task consumes the CLI/config/public numeric contracts; it adds no alternate orchestration or artifact writer. Its deliverable is a reviewed, verified, registered implementation ready for a separately authorized official run.

- [ ] TDD a reduced real-data availability/integration fixture that uses a distinct temporary run ID, reduced declared development settings, all adapter types, causality assertions, and manifest/shape checks without official-window scientific contrasts. Verify its intended RED if it exposes an integration gap, then fix via a delegated implementer and re-review. Do not invent a failing test for documentation or duplicate existing passing integration coverage.
- [ ] Run `uv run pytest -o addopts='' tests/unit tests/integration tests/e2e -q`, `uv run ruff check .`, `uv run ruff format --check .`. Include all markers rather than the project's default exclusions; record pass/skip counts, versions and any environment failures separately. Missing source data/dependencies cannot be reported as successful real integration.
- [ ] Dispatch whole-branch reviewer with full BASE..HEAD package, approved spec/plan, ledger, test evidence, and deferred/parked findings. Reviewer must not launch/inspect official v2 or reproduce it in memory as a runtime probe.
- [ ] Delegate one consolidated Critical/Important fix wave, require covering RED/GREEN evidence where applicable, scoped re-review, and fresh relevant/full checks after changes. No unresolved scientific Critical/Important finding can cross the launch gate.
- [ ] Write readiness report recording verification, reviewer verdicts, remaining limitations, numerical protocol hash and no-official-run evidence; update reproduction instructions without claiming new results.
- [ ] Append D32, or the next unused ledger number, with date, immutable identity/config/hash, all eight models, A-D, 96/56 families, seed/root, B/blocks, exploratory provenance, empty-set policy and D15 limits. Preserve all old decisions. Commit `docs(protocol): register cumulative model benchmark v2`; run fresh Git/status/diff checks and required regression checks after any code change.
- [ ] Report readiness and exact prospective command `uv run python -m hcp.run configs/model_benchmarks_v2.toml`. Do not execute it as part of this implementation plan: the approved spec requires the official-run gate to be completed first, and the present request authorizes plan construction. Official execution/analysis is a subsequent action after readiness review, never a reviewer probe or test fixture.

## Self-review and handoff

The controller checks spec sections 1-11 against Tasks 1-14: model contracts/compatibility (1,5,9,10), causal validation/QR/neural tuning (2,4,6-9), controllers/empty sets (3,11), window/inference/scales (11), diagnostics/reporting/publication (10,12,13), seeds (2,8-11), and review/registration (14). All five Review Focus conditions have named behavioral tests in their owning tasks. Self-review must check every named producer/consumer and resolve any actual inconsistency before committing this plan.

Subagents + TDD is already selected; do not ask the user to choose it again. After this written plan is reviewed and approved, begin the worktree/preflight/ledger setup and execute the numbered tasks using that method. Plan approval does not authorize a merge, push, or premature official run.
