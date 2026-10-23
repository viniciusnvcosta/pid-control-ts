# Task 1 Report: Freeze and Validate the CUDA Execution Contract

## Status

Implemented and committed locally as `feat(benchmark): register deterministic CUDA execution`.

The public constructor shapes and field validations for `ExecutionIdentity`,
`OriginForecastBatch`, `ProgressSnapshot`, and `NeuralExecution` followed the
parent's written ruling. The execution contract is frozen to `cuda:0`,
`NVIDIA GeForce RTX 4080`, `:4096:8`, TF32 disabled, and `32-true`; direct CPU
fixtures use `NeuralExecution.cpu()` while production v2 settings reject them.

## Implementation

- Added immutable execution, identity, origin-batch, and progress contracts in
  `src/hcp/benchmark_batch.py`. Identity hashing normalizes nested mappings,
  sequences, and NumPy scalars to compact sorted-key JSON before SHA-256.
- Added defensive array ownership, diagnostic table snapshots, nested mapping
  freezes, and contract validation for batch and progress values.
- Added the registered execution values to the v2 TOML, converted execution
  mappings to `NeuralExecution`, rejected deviations from the registered CUDA
  defaults, and included the execution values in config manifest serialization.
- Added `DiagnosticsTables.copy()` to provide independent owned table copies.
- Added configuration and contract tests for the registered values, invalid
  execution overrides, CPU test fixture construction, defensive copies, and
  insertion-order-independent identity digests.

## Test Evidence

RED command:

```text
UV_CACHE_DIR=/tmp/model-benchmarks-v2-sdd-uv-cache
UV_PROJECT_ENVIRONMENT=/home/vinvs/projects/research/pid-control-ts/.venv
UV_NO_SYNC=1 uv run pytest -o addopts='' tests/unit/test_config_benchmark_v2.py tests/unit/test_benchmark_batch.py -q
```

Before implementation, the run reported **16 failed, 19 passed**. The config
failure showed that `BenchmarkV2Settings` had no `execution` attribute, and the
new batch API was absent. Several missing-API cases initially appeared as
`ImportError` failures; the test helper was then changed to report the missing
contract as an assertion failure.

Focused GREEN command (after implementation, and again after formatting):

```text
UV_CACHE_DIR=/tmp/model-benchmarks-v2-sdd-uv-cache
UV_PROJECT_ENVIRONMENT=/home/vinvs/projects/research/pid-control-ts/.venv
UV_NO_SYNC=1 uv run pytest -o addopts='' tests/unit/test_config_benchmark_v2.py tests/unit/test_benchmark_batch.py -q
```

Result: **37 passed**.

Full suite, run once before commit:

```text
UV_CACHE_DIR=/tmp/model-benchmarks-v2-sdd-uv-cache
UV_PROJECT_ENVIRONMENT=/home/vinvs/projects/research/pid-control-ts/.venv
UV_NO_SYNC=1 uv run pytest -o addopts='' -q
```

Result: **464 passed, 1 skipped, 6 warnings in 45.03s**. Warnings included
synthetic/test framework warnings and a CUDA NVML initialization warning from
the existing transformer integration test.

Additional checks after formatting: Ruff check passed for all changed Python
files; `git diff --check` passed. Ruff format was applied to changed Python files.

## Files

- `src/hcp/benchmark_batch.py` (new)
- `src/hcp/config_benchmark_v2.py`
- `src/hcp/forecast_types.py`
- `src/hcp/config.py`
- `configs/model_benchmarks_v2.toml`
- `tests/unit/test_config_benchmark_v2.py`
- `tests/unit/test_benchmark_batch.py` (new)

## Self-review

- Confirmed the TOML values flow through `load_config` as a frozen
  `NeuralExecution` and survive `config_manifest` serialization.
- Confirmed mappings and NumPy arrays are copied/frozen, and diagnostic tables
  are independently copied at batch construction.
- Confirmed malformed device, TF32, precision, GPU name, workspace, quantile
  shape, timing, node, and progress values are rejected.
- No runtime file I/O was introduced in `benchmark_batch.py`.
- No changes were made to model definitions, forecast formulas, or `core/`.

## Concerns

The full suite emitted a CUDA NVML warning in an existing integration test;
this task did not exercise the physical GPU. The initial RED output also
included missing-module `ImportError` failures before the test helper was
corrected; all focused tests passed in the final GREEN run.
