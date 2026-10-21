# Development

## Setup
```bash
uv sync
# data/deaths.csv (219 MB, gitignored) = the paper's statewide LFS file, from the README's Google Drive link.
sha256sum data/deaths.csv   # cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06
```

## Checks
```bash
uv run ruff check src scripts tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
uv run ruff format --check src scripts tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
uv run pytest                        # unit + e2e (synthetic data)
uv run pytest -m integration         # real data parity and sanity (needs data/deaths.csv)
uv run pytest -o addopts='' -q       # everything

# Probabilistic benchmark focus
uv run pytest -o addopts='' -q tests/unit/test_benchmark_data.py tests/unit/test_forecast.py tests/unit/test_benchmark.py tests/unit/test_benchmark_report.py
uv run pytest -o addopts='' -q -m integration tests/integration/test_benchmark_real.py tests/integration/test_forecast_models.py
```

## Run
```bash
uv run python -m hcp.run configs/us_hier.toml       # -> results/us-hier-v1/
uv run python -m hcp.run configs/panel_leaves.toml  # -> results/panel-leaves-v1/
```
Artifacts: `manifest.json` (config, git SHA/dirty, input SHA-256, seed, versions, diagnostics),
`intervals.parquet` (long: week, reconciler, controller, lr, y, q_lo, q_med, q_hi, lo, hi),
`metrics.parquet`, `contrasts.parquet` (primary flag, Holm-adjusted p), `figures/*.pdf`.
A run never overwrites: copy the TOML and change `run_id`.

The panel design (`configs/panel_leaves.toml`) additionally writes `level_metrics.parquet`
(per-level metrics, `leaf_mean`/`us`), `summary.parquet` (cross-forecaster summary, D24), and
`eligibility.parquet` (the D21 eligibility table for every forecaster in the input).

### Probabilistic model benchmark

Run only after the D30 whole-branch review gate is clean. The registered run is CPU-heavy:
ARIMA and SARIMA refit every node at every eligible weekly origin, and DeepAR runs on CPU
with deterministic training, one device, and no data-loader workers. Expect high CPU use
and a long runtime.

```bash
UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run python -m hcp.run configs/model_benchmarks.toml 2> /tmp/model-benchmarks-v1.stderr
rg -n -i 'warning|error|exception|traceback|failed|nan|inf' /tmp/model-benchmarks-v1.stderr
```

The immutable run is written to `results/model-benchmarks-v1/`; analysis belongs in
`docs/analysis/model-benchmarks-v1.md` with selected publication figures under
`docs/analysis/figures/`. Reconcile every stderr match with the manifest's warning
diagnostics before analysis. Runs never overwrite. If either the final directory or
`results/.model-benchmarks-v1.partial` exists, inspect it and choose a recovery action;
the runner deliberately never deletes stale staging output.

## Post-hoc analysis
```bash
uv run python -m scripts.analysis.us_hier_v2 --run-dir results/us-hier-v2
```
Exploratory, non-confirmatory diagnostics for a finished US-design run; writes JSON and
figures under `--out` (default `docs/analysis/`).

## Paper harness
`tests/*.py`, `tests/configs/*.yaml`, `run_tests.sh`, `make_plots.sh` are the upstream experiment
harness, kept unchanged and excluded from pytest and ruff.
