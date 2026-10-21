# Probabilistic Model Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and pre-register a causal four-week probabilistic ARIMA, SARIMA, and DeepAR benchmark that compares each model with and without hierarchical reconciliation and against COVIDhub on a shared evaluation window.

**Architecture:** Add a canonical-truth input boundary, focused Darts model adapters, a numeric rolling-forecast orchestrator, and a benchmark evaluation stage that reuses the existing reconciliation, conformal, metric, and bootstrap code. Extend the frozen TOML configuration and `run.py` with an atomic `design = "benchmark"` path; extract the existing Phase 1 numeric stage first so `run.py` stays below the repository's size limit.

**Tech Stack:** Python 3.11, NumPy, pandas, Darts 0.47 with its torch extra, statsmodels, PyTorch Lightning, matplotlib, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-10-21-probabilistic-model-benchmarks-design.md`

## Global Constraints

- `docs/protocol-decisions.md` overrides this plan; append decisions, never rewrite history.
- Do not change `core/`; characterize the relevant `core/model_scores.py` behavior instead.
- Forecasts are q0.1/q0.5/q0.9 at the true fourth weekly horizon, using only actuals whose target date is no later than the forecast date.
- Use the untransformed death counts; do not clip negative samples or quantiles.
- Fixed models: ARIMA(3,0,0), SARIMA(3,0,0)(1,0,0,52), and global Darts LSTM `RNNModel` with Gaussian likelihood, input 12, training 16, hidden dimension 25, one layer, dropout 0, batch 32, 10 epochs.
- Fixed forecasting protocol: minimum 60 training weeks, weekly refit, 500 samples, CPU-only deterministic DeepAR, configuration-order `SeedSequence` spawning.
- Fixed evaluation protocol: common window, at least 60 weeks, reconcilers `none`, `bottom_up`, `mint_shrink`, `mint_median`, controllers `raw`, `pi`, blocks `(8, 4, 12)`, B=2000.
- The benchmark has no unadjusted primary; all hierarchy and model-comparison p-values are Holm-adjusted in the families specified by the spec.
- Numeric modules do no I/O. Only `run.py` writes result artifacts, through a sibling staging directory and atomic no-replace publication.
- Add dependencies only with `uv add`; never edit dependency declarations or the lockfile by hand.
- Preserve Phase 1 and Phase 2 behavior and manifest configuration shape when `benchmark` is absent.
- Public functions have type hints and Google docstrings; frozen dataclasses; two `# ABOUTME:` lines per new Python file; target 150–300 lines and split before roughly 400.
- Use `uv run`; commit locally with Conventional Commits; never push and never use `--no-verify`.
- Do not launch `configs/model_benchmarks.toml` or inspect registered benchmark outcomes until the whole-branch review and fix wave are clean.

## Review Focus

- A model that returns one NaN/Inf sample or raises at one origin must abort the run without silently dropping that node/model; Task 5 adds `test_nonfinite_or_failed_adapter_aborts_generation`.
- Canonical actual rows that disagree, including finite-vs-finite disagreement amid duplicate NaNs, must fail before forecasting; Task 2 adds `test_canonical_actuals_rejects_any_finite_disagreement`.
- Common-window selection at exactly 60 remaining weeks must succeed, while 59 must fail without shifting models independently; Task 6 adds `test_common_window_accepts_60_and_rejects_59_weeks`.
- Deterministic DeepAR must not silently select GPU/MPS or create checkpoints/logs when those devices are available; Task 4 adds `test_deepar_uses_cpu_deterministic_trainer_without_writes`.
- A stale staging directory or a renderer failure must never publish a partial final run; Task 9 adds `test_benchmark_writer_refuses_stale_staging_and_never_publishes_partial_output`.

---

### Task 1: Extract the existing Phase 1 numeric stage without changing behavior

**Files:**

- Create: `src/hcp/experiment.py`
- Modify: `src/hcp/run.py:24-208`
- Modify: `tests/unit/test_run.py`
- Test: `tests/unit/test_panel.py:124-158`

**Interfaces:**

- Consumes: existing `RunConfig`, `HierarchyBundle`, conformal/evaluate/stats functions.
- Produces: `hcp.experiment.ExperimentResult`, `PRIMARY`, and `run_experiment(cfg: RunConfig, bundle: HierarchyBundle) -> ExperimentResult`; `hcp.run` re-exports the same names for compatibility.

- [ ] **Step 1: Record the green refactor baseline**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_run.py tests/unit/test_panel.py::test_us_node_metrics_match_run_experiment`

Expected: PASS with the current tests; save the output as the pre-refactor characterization evidence.

- [ ] **Step 2: Move the numeric Phase 1 code into `hcp.experiment`**

Move `PRIMARY`, `COMPARISONS`, `ExperimentResult`, `_arm`, `_contrasts`, and `run_experiment` unchanged. Import `controller_config` from `hcp.config`; keep `run.py` imports that re-export `ExperimentResult`, `PRIMARY`, and `run_experiment`.

- [ ] **Step 3: Run the characterization tests after the move**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_run.py tests/unit/test_panel.py::test_us_node_metrics_match_run_experiment`

Expected: PASS with the same test count and no warnings.

- [ ] **Step 4: Verify the extracted module and reduced orchestrator**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run ruff check src/hcp/experiment.py src/hcp/run.py tests/unit/test_run.py tests/unit/test_panel.py && UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run ruff format --check src/hcp/experiment.py src/hcp/run.py tests/unit/test_run.py tests/unit/test_panel.py`

Expected: both commands exit 0; `wc -l src/hcp/run.py` is below 300 before benchmark additions.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/experiment.py src/hcp/run.py tests/unit/test_run.py tests/unit/test_panel.py
git commit -m "refactor(run): extract phase one experiment stage"
```

### Task 2: Canonicalize truth and build the aligned COVIDhub reference

**Files:**

- Create: `src/hcp/benchmark_data.py`
- Create: `tests/unit/test_benchmark_data.py`
- Modify: `tests/helpers.py`

**Interfaces:**

- Consumes: raw `read_forecasts()` frame, `ROOT`, `HierarchyBundle`, `build_bundle()`.
- Produces: frozen `ActualPanel(dates: np.ndarray, nodes: tuple[str, ...], y: np.ndarray, observed: np.ndarray)`, `canonical_actuals(frame: pd.DataFrame) -> ActualPanel`, and `reference_bundle(frame: pd.DataFrame, actuals: ActualPanel, forecaster: str) -> HierarchyBundle`.

- [ ] **Step 1: Write failing canonical-truth tests**

Add tests that independently assert:

- identical duplicate actuals collapse to root-first/sorted-node weekly arrays;
- `observed` is false only where every duplicate is NaN;
- returned arrays are read-only;
- `test_canonical_actuals_rejects_any_finite_disagreement` raises `ValueError` naming the `(node, week)` when two finite values differ, even when a third duplicate is NaN;
- a non-weekly date grid raises `ValueError`.

- [ ] **Step 2: Run the canonical tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_data.py -k canonical`

Expected: FAIL because `hcp.benchmark_data` does not exist.

- [ ] **Step 3: Implement `ActualPanel` and `canonical_actuals`**

Use exact equality among finite duplicates, preserve NaN when no finite actual exists, require `ROOT`, order root then sorted leaves, validate seven-day spacing, and mark every returned NumPy array read-only.

- [ ] **Step 4: Run canonical tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_data.py -k canonical`

Expected: PASS.

- [ ] **Step 5: Write failing reference-alignment tests**

Add `test_reference_bundle_uses_submitted_quantiles_and_canonical_truth` with a literal gap fixture. Assert submitted q values/LOCF equal `build_bundle`, `y` comes from `ActualPanel` rather than missing forecaster rows, dates/forecast dates align, and the imputation mask remains forecast-only. Add mismatched-node and unavailable-date error cases.

- [ ] **Step 6: Run reference tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_data.py -k reference`

Expected: FAIL because `reference_bundle` is missing.

- [ ] **Step 7: Implement `reference_bundle` and verify GREEN**

Build submitted quantiles with `build_bundle`, reindex canonical `y` to its nodes/dates, reject missing alignments, and return a new frozen `HierarchyBundle` with read-only arrays.

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_data.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcp/benchmark_data.py tests/unit/test_benchmark_data.py tests/helpers.py
git commit -m "feat(benchmark): add canonical truth input stage"
```

### Task 3: Add probabilistic ARIMA and SARIMA adapters and protect the upstream reference

**Files:**

- Create: `src/hcp/forecast_models.py`
- Create: `tests/unit/test_core_model_scores_characterization.py`
- Create: `tests/integration/test_forecast_models.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**

- Consumes: Darts `TimeSeries` and `ARIMA`, NumPy `SeedSequence`.
- Produces: frozen `PreparedHistory(node: str, dates: np.ndarray, values: np.ndarray)`;
  frozen `ForecastSettings(ahead=4, n_samples=500, arima_order=(3,0,0),
  sarima_seasonal_order=(1,0,0,52), deepar_model="LSTM",
  deepar_input_chunk_length=12, deepar_training_length=16, deepar_hidden_dim=25,
  deepar_n_rnn_layers=1, deepar_dropout=0.0, deepar_batch_size=32,
  deepar_n_epochs=10)`; `arima_samples(...) -> np.ndarray`; and
  `sarima_samples(...) -> np.ndarray`. Both adapters return
  `[n_nodes, settings.ahead, settings.n_samples]`.

- [ ] **Step 1: Characterize `core.model_scores.generate_forecasts` without changing it**

Add a narrow test using controlled fake Darts boundaries and a temporary cache path. Assert that the existing `"ar"` branch constructs order `(3, 0, 0)`, requests `historical_forecasts(..., forecast_horizon=fit_every, retrain=fit_every)`, returns the last-point array aligned at the tail, and reuses a readable cache when `overwrite=False`.

- [ ] **Step 2: Run the characterization test**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_core_model_scores_characterization.py`

Expected: PASS against unchanged upstream code; this is characterization evidence, not a behavior RED.

- [ ] **Step 3: Write failing real-adapter integration tests**

For a literal 64-week synthetic `PreparedHistory`, add tests that each adapter returns finite `[1, 4, 7]` samples, repeats exactly for the same seed, differs for a different seed, and changes the retained fourth-step distribution when the final training values change. Assert SARIMA accepts the fixed annual seasonal order with 60+ observations.

- [ ] **Step 4: Run adapter tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py -k 'arima or sarima'`

Expected: FAIL because `hcp.forecast_models` does not exist.

- [ ] **Step 5: Add the Darts torch dependency**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv add 'darts[torch]>=0.47.0'`

Expected: `pyproject.toml` expresses the torch extra and `uv.lock` resolves torch, PyTorch Lightning, TensorBoardX, Hugging Face Hub, and safetensors without removing current runtime dependencies.

- [ ] **Step 6: Implement the classical adapters**

Define `ForecastSettings` with every fixed hyperparameter from the spec. `arima_samples` constructs `ARIMA(p=3,d=0,q=0)` per node; `sarima_samples` adds `seasonal_order=(1,0,0,52)`. Fit each history, predict exactly `ahead`, request `n_samples`, derive node child seeds in input order, and reject wrong shapes or non-finite samples.

- [ ] **Step 7: Run adapter tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_core_model_scores_characterization.py && UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py -k 'arima or sarima'`

Expected: all selected tests PASS without modifying `core/`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/hcp/forecast_models.py tests/unit/test_core_model_scores_characterization.py tests/integration/test_forecast_models.py
git commit -m "feat(forecast): add probabilistic arima adapters"
```

### Task 4: Add the deterministic global DeepAR-style adapter

**Files:**

- Modify: `src/hcp/forecast_models.py`
- Modify: `tests/integration/test_forecast_models.py`

**Interfaces:**

- Consumes: Task 3 `PreparedHistory`, `ForecastSettings`, Darts `RNNModel`, `GaussianLikelihood`.
- Produces: `deepar_samples(histories: tuple[PreparedHistory, ...], settings: ForecastSettings, seed: np.random.SeedSequence) -> np.ndarray` with `[n_nodes, ahead, n_samples]` on original node scales.

- [ ] **Step 1: Write failing DeepAR behavior tests**

Add a two-series synthetic test asserting finite `[2, 4, 7]` output, exact same-seed repeatability, configuration-order preservation, and scale inversion. Add `test_deepar_uses_cpu_deterministic_trainer_without_writes`, changing CWD to `tmp_path` and asserting no checkpoint, Lightning log, TensorBoard, or model directory appears.

- [ ] **Step 2: Run and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py -k deepar`

Expected: FAIL because `deepar_samples` is missing.

- [ ] **Step 3: Implement `deepar_samples`**

Scale each series by `max(mean(abs(values)), 1)`, fit one global `RNNModel` with the spec's exact LSTM/Gaussian settings, CPU device, deterministic trainer, zero workers, disabled logger/progress/checkpoints/model summary, and the integer generated from its seed. Predict four steps with `num_samples`, restore node scales, preserve input order, and validate shape/finiteness.

- [ ] **Step 4: Run and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py -k deepar`

Expected: PASS and leave `tmp_path` empty of trainer artifacts.

- [ ] **Step 5: Run all adapter tests**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_core_model_scores_characterization.py && UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcp/forecast_models.py tests/integration/test_forecast_models.py
git commit -m "feat(forecast): add deterministic deepar adapter"
```

### Task 5: Generate causal model bundles on the registered grid

**Files:**

- Create: `src/hcp/forecast.py`
- Create: `tests/unit/test_forecast.py`

**Interfaces:**

- Consumes: Task 2 `ActualPanel`; Task 3/4 sample adapters and `ForecastSettings`; `HierarchyBundle`.
- Produces: module-level `FORECASTERS`; frozen
  `ForecastConfig(models, quantiles, min_train_weeks, refit_every, root_seed,
  seed_spawn_key, settings)`; frozen
  `ForecastResult(bundles: dict[str, HierarchyBundle], base_forecasts: pd.DataFrame,
  diagnostics: dict)`; and
  `generate_model_bundles(actuals: ActualPanel, dates: np.ndarray,
  forecast_dates: np.ndarray, cfg: ForecastConfig) -> ForecastResult`.

- [ ] **Step 1: Write failing orchestration tests with deterministic adapters**

Use adapter functions that return literal samples rather than mocks. Cover:

- forecast targets use only actual dates `<= forecast_date`;
- changing any later actual leaves earlier quantiles byte-identical;
- an interior training gap is interpolated within the prefix and counted, while a leading gap is dropped;
- horizon four, not horizon one, supplies q0.1/q0.5/q0.9;
- configuration order controls model and seed order even if `FORECASTERS` insertion order changes;
- forecasts before 60 training weeks remain NaN, but zero eligible origins raises;
- unknown model, non-weekly grids, node/date mismatches, invalid sample shape, and invalid quantile order raise;
- `test_nonfinite_or_failed_adapter_aborts_generation` raises rather than returning a partial model set;
- output arrays are read-only and `base_forecasts` has the exact spec columns.

- [ ] **Step 2: Run and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_forecast.py`

Expected: FAIL because `hcp.forecast` does not exist.

- [ ] **Step 3: Implement forecasting dataclasses, registry, and prefix preparation**

Set `FORECASTERS = {"arima": arima_samples, "sarima": sarima_samples, "deepar": deepar_samples}`. `ForecastConfig` fixes model tuple, quantiles, minimum history, refit interval, root seed, seed spawn key `(0,)`, and `ForecastSettings`; validate exact weekly grids, `ahead=4`, `refit_every=1`, ordered quantiles, and positive samples/history.

- [ ] **Step 4: Implement `generate_model_bundles`**

Iterate config models then target origins; prepare all node prefixes; spawn seeds by model/origin; capture warning categories/counts and elapsed fit time; quantify samples without repair; construct read-only bundles using canonical truth and zero forecast-imputation masks; return long forecasts and fit/interpolation diagnostics.

- [ ] **Step 5: Run and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_forecast.py`

Expected: PASS.

- [ ] **Step 6: Run forecast unit plus real-adapter integration tests**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_forecast.py && UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_forecast_models.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcp/forecast.py tests/unit/test_forecast.py
git commit -m "feat(forecast): generate causal hierarchy bundles"
```

### Task 6: Evaluate all models on one common hierarchy window

**Files:**

- Create: `src/hcp/benchmark.py`
- Create: `tests/unit/test_benchmark.py`
- Modify: `src/hcp/panel.py:70-128`
- Modify: `tests/unit/test_panel.py`

**Interfaces:**

- Consumes: model-to-`HierarchyBundle` mapping, existing `PanelConfig`, `forecaster_arms`, `forecaster_contrasts`, panel metric helpers, bootstrap/stat functions.
- Produces: optional keyword `evaluation_index` on `forecaster_arms`; frozen
  `BenchmarkConfig(panel, models, reference_model, min_eval_weeks,
  seed_spawn_key)`; frozen `BenchmarkResult(base_forecasts, intervals, metrics,
  level_metrics, hierarchy_contrasts, model_contrasts, diagnostics)`;
  `common_evaluation_index(bundles: Mapping[str, HierarchyBundle], cfg:
  BenchmarkConfig) -> int`; and `run_benchmark(bundles:
  Mapping[str, HierarchyBundle], cfg: BenchmarkConfig) -> BenchmarkResult`.

- [ ] **Step 1: Write a failing explicit-start panel test**

Add `test_forecaster_arms_respects_valid_explicit_evaluation_index` and invalid-index cases. Assert the explicit start is used exactly, still requires 20 known errors, and leaves the existing no-argument Phase 2 path unchanged.

- [ ] **Step 2: Run and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_panel.py -k evaluation_index`

Expected: FAIL because `forecaster_arms` has no such keyword.

- [ ] **Step 3: Add the explicit common-start hook and verify GREEN**

Refactor the existing body minimally: compute `t0` as today when omitted; when supplied, validate bounds and `len(available_errors(bundle, t0)) >= mint_warmup`. Do not change existing results.

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_panel.py`

Expected: PASS, including the Phase 1/Phase 2 metric anchor.

- [ ] **Step 4: Write failing common-window and benchmark-result tests**

Build literal aligned synthetic bundles and assert:

- the latest per-bundle MinT-ready start is selected once for all models;
- `test_common_window_accepts_60_and_rejects_59_weeks` exercises the exact boundary;
- mismatched model keys, dates, nodes, forecast dates, root truth, or finite forecasts raise;
- intervals/metrics use column `model`, never `forecaster`;
- hierarchy contrasts contain all 3-vs-`none` comparisons, no primary exemption, and Holm families `(block, model, level, controller)`;
- `bottom_up-none` leaf rows are literal zero;
- model contrasts contain exactly three generated models × two metrics for each `(block, level, controller, reconciler)` family, use estimate `generated-COVIDhub`, share paired draws, and Holm-adjust six members;
- repeated root seed is frame-identical and a changed root seed changes bootstrap output.

- [ ] **Step 5: Run benchmark tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark.py`

Expected: FAIL because `hcp.benchmark` does not exist.

- [ ] **Step 6: Implement the common-window evaluator**

Define `BenchmarkConfig(panel, models, reference_model, min_eval_weeks, seed_spawn_key=(1,))`. Reuse `forecaster_arms`, frame builders, `forecaster_contrasts`, `node_metrics`, `level_values`, `draw_blocks`, `summarize_diff`, and `holm`. Reconstruct `base_forecasts` from aligned bundles. Apply no unadjusted primary.

- [ ] **Step 7: Run and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark.py tests/unit/test_panel.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcp/benchmark.py src/hcp/panel.py tests/unit/test_benchmark.py tests/unit/test_panel.py
git commit -m "feat(benchmark): compare models on a common hierarchy window"
```

### Task 7: Validate and pre-register the benchmark configuration

**Files:**

- Modify: `src/hcp/config.py:19-126`
- Modify: `tests/unit/test_run.py`
- Create: `configs/model_benchmarks.toml`

**Interfaces:**

- Consumes: Task 5 `ForecastConfig`, Task 6 `BenchmarkConfig`, existing `controller_config` and `panel_config`.
- Produces: frozen `BenchmarkSettings`, optional `RunConfig.benchmark`,
  `benchmark_configs(cfg: RunConfig) -> tuple[ForecastConfig, BenchmarkConfig]`,
  and `config_manifest(cfg: RunConfig) -> dict`.

- [ ] **Step 1: Write failing benchmark-config tests**

Assert the registered TOML loads these exact values: run ID `model-benchmarks-v1`, root seed `20261021`, models `("arima","sarima","deepar")`, reference key `covidhub`, reference forecaster `COVIDhub-4_week_ensemble`, quantiles `(0.1,0.5,0.9)`, samples 500, history 60, refit 1, fixed model hyperparameters, Phase 2 arms/controllers/lr/blocks/B, and min evaluation 60. Assert seed namespaces `(0,)` and `(1,)` are disjoint.

Add invalid tests for benchmark settings on non-benchmark designs, missing settings on benchmark, unknown/duplicate models, wrong horizon/refit/quantiles/model constants, `pid_theta`, missing `none`, and reference-key collision. Assert existing config manifest mappings do not gain `benchmark: null`.

- [ ] **Step 2: Run and verify RED**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_run.py -k benchmark`

Expected: FAIL because benchmark configuration is unsupported.

- [ ] **Step 3: Implement frozen benchmark configuration and mapper**

Parse a nested `[benchmark]` TOML table into `BenchmarkSettings`; reject unknown keys through dataclass construction. Extend `design` validation to `"benchmark"` without applying panel forecaster-list rules. Implement `config_manifest` so it omits only absent benchmark settings, preserving old manifest shape.

- [ ] **Step 4: Add the registered TOML**

Write every fixed value from the spec explicitly; include comments that it must not be run before review/fix clearance and that it is exploratory with no unadjusted primary.

- [ ] **Step 5: Run config tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_run.py`

Expected: PASS, including existing US/panel config tests.

- [ ] **Step 6: Commit**

```bash
git add src/hcp/config.py tests/unit/test_run.py configs/model_benchmarks.toml
git commit -m "feat(config): register probabilistic model benchmark"
```

### Task 8: Render benchmark figures as in-memory objects

**Files:**

- Create: `src/hcp/benchmark_report.py`
- Create: `tests/unit/test_benchmark_report.py`

**Interfaces:**

- Consumes: Task 6 `BenchmarkResult` frames.
- Produces: `benchmark_figures(result: BenchmarkResult, *, block: int, lr: float) -> dict[str, matplotlib.figure.Figure]`; it performs no file I/O.

- [ ] **Step 1: Write failing figure tests**

With literal result frames, assert the mapping has exactly `forest-leaf-mean-pi`, `forest-us-pi`, `coverage-score-tradeoff`, and `hierarchy-effect`; each figure has axes and the expected model/reconciler labels; estimates whose intervals exclude them render without negative-error exceptions; and calling the function creates no files under `tmp_path`.

- [ ] **Step 2: Run and verify RED**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_report.py`

Expected: FAIL because `hcp.benchmark_report` does not exist.

- [ ] **Step 3: Implement the four OO-matplotlib figures**

Follow existing report colors and clipped asymmetric-error handling. Use consistent model order from result diagnostics; label COVIDhub as the external reference and DeepAR as `DeepAR-style (Darts LSTM)`.

- [ ] **Step 4: Run and verify GREEN**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/unit/test_benchmark_report.py`

Expected: PASS and no files created by the report module.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/benchmark_report.py tests/unit/test_benchmark_report.py
git commit -m "feat(report): add model benchmark figures"
```

### Task 9: Add atomic benchmark CLI orchestration and artifacts

**Files:**

- Modify: `src/hcp/run.py:211-end`
- Modify: `tests/unit/test_run.py`
- Create: `tests/e2e/test_run_benchmark_cli.py`

**Interfaces:**

- Consumes: Tasks 2, 5, 6, 7, and 8 public interfaces.
- Produces: `write_benchmark_artifacts(cfg: RunConfig, result: BenchmarkResult, forecast_diagnostics: dict, input_sha256: str) -> Path` and the `main()` branch for `design="benchmark"`.

- [ ] **Step 1: Write failing manifest and writer tests**

Assert the writer creates the six parquet files, four PDFs, and manifest from the spec; manifest config preserves old-design shape and includes Darts/PyTorch/Lightning versions, device/determinism, seed tree, hyperparameters, common window/nodes, fit/warning/timing/interpolation data, D15 limitation, and exact `analysis_note`.

Add `test_benchmark_writer_refuses_stale_staging_and_never_publishes_partial_output`: pre-create the sibling partial directory and expect `FileExistsError`; separately make figure generation raise and assert the final directory never appears.

- [ ] **Step 2: Run writer tests and verify RED**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/e2e/test_run_benchmark_cli.py -k writer`

Expected: FAIL because `write_benchmark_artifacts` is missing.

- [ ] **Step 3: Implement the atomic benchmark writer**

Write all tables into `.<run_id>.partial`, ask `benchmark_figures` for in-memory figures and save them from `run.py`, close figures, write manifest last, recheck the destination, then call `_rename_noreplace`. Do not clean or overwrite a stale partial directory automatically.

- [ ] **Step 4: Run writer tests and verify GREEN**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/e2e/test_run_benchmark_cli.py -k writer`

Expected: PASS.

- [ ] **Step 5: Write the failing benchmark CLI test**

Use real CSV parsing/canonical/reference logic and deterministic in-memory forecast adapters through monkeypatching only the expensive registry values. Run `main()` with a small benchmark TOML, assert every artifact/schema/manifest field, exact model order, one common window, and second-run `FileExistsError`. Also assert US and panel e2e tests remain unchanged.

- [ ] **Step 6: Run CLI test and verify RED**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/e2e/test_run_benchmark_cli.py -k cli`

Expected: FAIL because `main()` has no benchmark branch.

- [ ] **Step 7: Implement the benchmark CLI branch**

Read and hash once; canonicalize truth; build the aligned reference; map configs; generate model bundles; add reference under its configured key; run the benchmark; write atomically. Keep the early final-output refusal before any model fit.

- [ ] **Step 8: Run all CLI tests and enforce module size**

Run: `MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q tests/e2e/test_run_cli.py tests/e2e/test_run_panel_cli.py tests/e2e/test_run_benchmark_cli.py && wc -l src/hcp/run.py`

Expected: PASS; `run.py` remains below roughly 400 lines.

- [ ] **Step 9: Commit**

```bash
git add src/hcp/run.py tests/unit/test_run.py tests/e2e/test_run_benchmark_cli.py
git commit -m "feat(run): orchestrate atomic model benchmarks"
```

### Task 10: Add real-data integration, pre-registration, and operator documentation

**Files:**

- Create: `tests/integration/test_benchmark_real.py`
- Modify: `docs/protocol-decisions.md`
- Modify: `docs/development.md`
- Modify: `README.md`

**Interfaces:**

- Consumes: registered config and complete benchmark pipeline.
- Produces: a reduced non-outcome integration anchor, pre-launch protocol decision D30, and documented review/launch commands.

- [ ] **Step 1: Add the reduced real-data integration anchor**

On `data/deaths.csv`, restrict to one late forecast origin, all 57 nodes, `models=("arima",)`, `n_samples=5`, one block, and B=5. Assert canonical/reference/generated node/date equality, true horizon-four cutoff, finite q shape, hierarchy coherence for reconciled output, and no file under `results/`. Do not compute or print the registered benchmark metrics.

- [ ] **Step 2: Run the real-data integration anchor**

Run: `UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q -m integration tests/integration/test_benchmark_real.py`

Expected: PASS without writing results or showing registered outcome values. A failure is
a defect in Tasks 2–9 and must be diagnosed and fixed under the owning task's contract
before documentation proceeds.

- [ ] **Step 3: Append pre-launch decision D30**

Record the exploratory status, data/hash, canonical-truth rule, exact model/hyperparameter/probabilistic settings, causal horizon/refit rule, common window, hierarchy/model contrasts, Holm families, seeds, artifacts, D15 limitation, and the prohibition on result inspection before review clearance. State that DeepAR is Darts LSTM + Gaussian likelihood, not GluonTS.

- [ ] **Step 4: Document development and launch commands**

Add focused-test, full-test, registered-run, stderr-review, and analysis paths. Note expected high CPU/runtime, CPU-only DeepAR, non-overwrite behavior, and that stale `.model-benchmarks-v1.partial` requires operator inspection rather than automatic deletion.

- [ ] **Step 5: Run full pre-review verification**

Run:

```bash
MPLBACKEND=Agg UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run pytest -o addopts='' -q
UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run ruff check src core scripts tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run ruff format --check src core scripts tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
git diff --check
wc -l src/hcp/benchmark_data.py src/hcp/forecast_models.py src/hcp/forecast.py src/hcp/benchmark.py src/hcp/benchmark_report.py src/hcp/run.py
```

Expected: all tests pass; Ruff and diff checks exit 0; each module stays below roughly 400 lines; `results/model-benchmarks-v1` does not exist.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_benchmark_real.py docs/protocol-decisions.md docs/development.md README.md
git commit -m "docs(protocol): preregister model benchmark (D30)"
```

## Post-implementation review, fix wave, and official run

These operations occur only after Tasks 1–10 are complete. Because inline execution was
selected, follow `superpowers:executing-plans`: generate the whole-branch review package,
dispatch one fresh most-capable reviewer, re-grade every finding, fix all Critical and
Important findings in one RED→GREEN pass, run the full suite, and ledger deferred Minors
and rulings. Do not launch while any Critical/Important finding remains.

After the review/fix gate is clean:

1. Confirm a clean worktree and that `results/model-benchmarks-v1/` and its sibling partial
   directory do not exist.
2. Run exactly:

   ```bash
   UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run python -m hcp.run configs/model_benchmarks.toml 2> /tmp/model-benchmarks-v1.stderr
   ```

3. Check the retained stderr for warnings, errors, exceptions, tracebacks, failed fits,
   NaN, and Inf; reconcile every match against the manifest warning diagnostics.
4. Analyze `results/model-benchmarks-v1/` without modifying it. Write
   `docs/analysis/model-benchmarks-v1.md`, copy selected figures under
   `docs/analysis/figures/`, and report hierarchy effects separately from model-quality
   comparisons and coverage-width trade-offs.
5. Append dated outcome decision D31 and commit analysis and protocol documentation in
   separate Conventional Commits.
6. Run fresh full tests, Ruff checks, format checks, `git diff --check`, and verify the
   manifest Git SHA/dirty fields describe the reviewed implementation commit.
7. Use `superpowers:finishing-a-development-branch` with merge target `develop`; never
   push without explicit permission.
