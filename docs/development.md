# Development

## Setup
```bash
uv sync
# data/deaths.csv (219 MB, gitignored) = the paper's statewide LFS file, from the README's Google Drive link.
sha256sum data/deaths.csv   # cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06
```

## Checks
```bash
uv run ruff check src tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
uv run ruff format --check src tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
uv run pytest                        # unit + e2e (synthetic data)
uv run pytest -m integration         # real data parity and sanity (needs data/deaths.csv)
uv run pytest -o addopts='' -q       # everything
```

## Run
```bash
uv run python -m hcp.run configs/us_hier.toml   # -> results/us-hier-v1/
```
Artifacts: `manifest.json` (config, git SHA/dirty, input SHA-256, seed, versions, diagnostics),
`intervals.parquet` (long: week, reconciler, controller, lr, y, q_lo, q_med, q_hi, lo, hi),
`metrics.parquet`, `contrasts.parquet` (primary flag, Holm-adjusted p), `figures/*.pdf`.
A run never overwrites: copy the TOML and change `run_id`.

## Paper harness
`tests/*.py`, `tests/configs/*.yaml`, `run_tests.sh`, `make_plots.sh` are the upstream experiment
harness, kept unchanged and excluded from pytest and ruff.
