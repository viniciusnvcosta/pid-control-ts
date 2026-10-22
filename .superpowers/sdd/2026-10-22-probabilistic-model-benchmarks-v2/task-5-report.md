# Task 5 implementation evidence

Status: DONE. Base: `36a98f6`. Scope: classical relocation/native wrappers and weekly Darts Prophet only; registry wiring remains Task 10.

## Changes and interfaces

- Added `src/hcp/forecast_classical.py`: unchanged `arima_samples` / `sarima_samples`, native `arima_prediction` / `sarima_prediction`, and `prophet_prediction(histories, V2ModelSettings, SeedSequence) -> NativePrediction`.
- `forecast_models.py` re-exports the legacy sample functions and types; DeepAR remains there unchanged. Direct comparison against `git show 36a98f6:src/hcp/forecast_models.py` confirmed the seed helper/classical function block and the DeepAR function are byte-identical after relocation. No old ARIMA monkeypatch consumers existed to retarget (`rg` checked tests).
- Prophet explicitly receives linear growth, all three seasonalities disabled, 25 changepoints, range 0.8, prior scale 0.05, and MCMC 0. It retains weekly dates and original signed units, and returns finite immutable `[node,4,n_samples]` paths (500 by default).
- Each fixed-order node consumes one origin child, derives its uint32 integer once, and uses that integer for Darts constructor, actual underlying Prophet MAP fit, and prediction. Numeric diagnostics expose `node_seeds` in node order. NumPy legacy global state is restored on success and failure.
- Backend failures retain node and origin-prefix end context and chain the original exception. Shape/finiteness failures use the same contextual boundary.
- Added fake-backend unit contracts and real synthetic integration, named `test_forecast_prophet_integration.py` per controller ruling to avoid duplicate basenames with mandatory `-o addopts=''`.

## TDD and validation

All test commands ran in the task worktree with:

```sh
MPLCONFIGDIR=/tmp/pid-control-ts-mpl UV_PROJECT_ENVIRONMENT=/home/vinvs/projects/research/pid-control-ts/.venv UV_CACHE_DIR=/tmp/pid-control-ts-uv-cache uv run --no-sync
```

1. RED before creating the adapter: `pytest -o addopts='' -q tests/unit/test_forecast_prophet.py` => `6 failed in 0.09s`, all failing inside test calls with `ModuleNotFoundError: No module named 'hcp.forecast_classical'`. Added real integration before implementation and reran both files => `7 failed in 0.74s`, same expected missing feature.
2. Initial GREEN: focused unit, real integration, existing `tests/integration/test_forecast_models.py` => `12 passed, 16 warnings in 12.76s`. Warnings were statsmodels convergence/nonstationary fixture warnings.
3. Additional behavioral RED for explicit MAP seeding: `pytest -o addopts='' -q tests/integration/test_forecast_prophet_integration.py::test_real_prophet_passes_child_seed_to_map_fit tests/unit/test_forecast_prophet.py::test_prophet_refuses_backend_without_seedable_fit_boundary` => `2 failed in 0.86s`: actual real backend fit observed `None` versus required `2567351042`; incompatible backend produced an unhelpful missing-fit exception. These failures proved NumPy seeding alone was insufficient.
4. GREEN after instance-local fit binding: `pytest -o addopts='' -q tests/unit/test_forecast_prophet.py tests/integration/test_forecast_prophet_integration.py tests/integration/test_forecast_models.py` => `14 passed, 8 warnings in 12.36s`. Added narrowly scoped filtering of the known classical fixture nonstationarity warnings, then formatted and linted changed Python files.
5. Final complete suite, once after final code/test edits: `pytest -o addopts='' -q` => **`345 passed, 1 skipped in 30.13s`**, exit 0, no warnings. Includes unit, integration and e2e, versus baseline 336 passed/1 skipped.
6. `ruff check` on all four changed Python files => `All checks passed!`; `ruff format` formatted three, left one unchanged. `git diff --check` => no output/exit 0.
7. Fresh interpreter import-hook blocked `prophet` and all submodules: legacy `arima_samples`, `sarima_samples`, `deepar_samples` imports all succeeded. No eager Prophet dependency was introduced by the new module.

Real integration verifies finite 500-path outputs, original negative count scale, stochastic paths, exact repeatability with the same root, sensitivity to seed and recent history, and an actual Prophet.fit call receiving the literal first node child seed. Unit tests cover two fixed-order child seeds, weekly dates, explicit parameters, no clipping, immutable output, state restoration after fit/predict failures, and unsupported backend refusal.

## Dependencies and environment

Used only `uv add prophet` to update `pyproject.toml` and this worktree's `uv.lock`. Initial sandboxed attempt failed fetching PyPI with `Temporary failure in name resolution`; authorized network escalation succeeded. Resolved 108 packages; additions only: `prophet==1.5.0`, `cmdstanpy==1.3.0`, `stanio==0.5.1`; existing versions were not updated. Shared environment editable package was reinstalled to this task worktree.

Verified installed `darts==0.47.0`, `numpy==2.3.5`; Prophet backend `CMDSTANPY`, bundled CmdStan `2.37.0` (`CmdStanPyBackend.CMDSTAN_VERSION`). Direct Prophet import logs that optional Plotly is unavailable; forecasting integrations and final suite succeed without it. One initial shell editing attempt used unavailable bare `python`, performed no edits, and was rerun successfully via `uv run --no-sync python`.

## Self-review and limitations

- Controller approved the private Darts `_model_builder` boundary after installed source inspection proved no public fit-seed passthrough. Darts 0.47 `_fit` invokes underlying `fit` with no seed; CmdStanPy 1.3 otherwise uses entropy-seeded `default_rng()` independently of NumPy global state. The small factory wraps only this model instance's backend fit with `functools.partial(..., seed=node_seed)`. Missing/incompatible boundaries fail contextually; no global or upstream monkeypatch is used in production.
- This private Darts boundary is version-coupled and must remain covered by the real integration when dependencies change. NumPy state containment is sequential and does not claim thread-safe concurrent local fitting; current forecasting is sequential.
- Forecast API has no issued-origin timestamp argument, so failure context reports the exact training-prefix end rather than inventing an issuance date.
- No core, v1, result artifacts, registry, or official realization changed/generated. Production path-count/config enforcement remains Task 10; reduced seven-path fixtures remain supported.
