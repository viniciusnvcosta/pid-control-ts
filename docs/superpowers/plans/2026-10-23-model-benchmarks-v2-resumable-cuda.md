# Model Benchmarks v2 Resumable CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the registered v2 benchmark resumable at each model/origin boundary, quiet and observable, and deterministic on the local RTX 4080 while retaining atomic final publication.

**Architecture:** Split forecast generation into a pure plan, one-origin execution, and deterministic assembly. `run.py` remains the only writer: it stores immutable batch directories in a hidden fixed work directory, reloads verified batches on resume, records structured progress, evaluates only after all batches exist, and atomically renames the completed work directory. A frozen execution contract is threaded explicitly into neural adapters; classical models remain on CPU.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, PyTorch 2.14 CUDA, PyTorch Lightning, Darts, pytest, Ruff, Git no-replace publication.

**Spec:** `docs/superpowers/specs/2026-10-23-model-benchmarks-v2-resumable-cuda-design.md`

## Global Constraints

- Only `src/hcp/run.py` writes files; numeric modules perform no I/O.
- Production v2 uses `cuda:0` on NVIDIA GeForce RTX 4080 for `deepar`, `transformer`, and `deepar_enhanced`; no runtime CPU fallback.
- CUDA uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms with `warn_only=False`, TF32 disabled, float32 precision, zero loader workers, and unpinned host memory.
- Model definitions, tuning budgets, causal splits, seed addresses, model order, inference arms, and families remain unchanged.
- A completed batch is immutable and atomically published; a mismatched or damaged batch is a hard failure.
- `results/model-benchmarks-v2` remains absent until the hidden work directory is complete and atomically renamed without replacement.
- Preserve scientific warnings in diagnostics while suppressing framework banners, tips, summaries, and progress bars.
- No `core/` changes; use `TQDM_DISABLE=1` at the run boundary.
- New public functions use type hints and Google docstrings; new source files start with two `# ABOUTME:` lines.

## File Structure

- Create `src/hcp/benchmark_batch.py`: pure frozen batch, execution-identity, and progress contracts; schema and digest validation only, no file I/O.
- Modify `src/hcp/config_benchmark_v2.py`: registered frozen execution settings and strict TOML validation.
- Modify `src/hcp/forecast_types.py`: explicit neural execution value threaded to adapters and immutable origin-batch payload.
- Modify `src/hcp/forecast.py`: pure forecast plan, one-origin execution, deterministic assembly, and compatibility wrapper.
- Modify `src/hcp/forecast_neural.py`: device-aware legacy DeepAR and Transformer training/prediction.
- Modify `src/hcp/forecast_deepar.py`: device-aware enhanced DeepAR training.
- Modify `src/hcp/forecast_deepar_rollout.py`: device-aware seeded rollout and CPU NumPy boundary.
- Modify `src/hcp/run.py`: CUDA launch gate, all batch/progress/log writes, resume orchestration, final assembly, and atomic publication.
- Modify `configs/model_benchmarks_v2.toml`: explicit registered CUDA execution block.
- Modify focused unit/integration/e2e tests named in each task.
- Modify readiness/protocol documentation only after verification records exact commands and limitations.

## Review Focus

- A valid-looking batch copied from another origin or seed must fail identity validation; Task 6 tests swapped manifests and spawn keys.
- A crash between payload writes and `batch.json` must leave no reusable batch; Task 6 tests temporary and incomplete directories.
- A driver/runtime/GPU UUID change on resume must fail before loading any forecast; Task 7 parameterizes every device identity field.
- Resume ordering must not shift seeds when earlier origins are ineligible or already complete; Task 5 compares seed addresses and outputs.
- Final publication racing another process must preserve the other process and all resumable evidence; Task 8 exercises the no-replace race.

---

### Task 1: Freeze and validate the CUDA execution contract

**Files:**
- Create: `src/hcp/benchmark_batch.py`
- Modify: `src/hcp/config_benchmark_v2.py`
- Modify: `src/hcp/forecast_types.py`
- Modify: `src/hcp/config.py`
- Modify: `configs/model_benchmarks_v2.toml`
- Test: `tests/unit/test_config_benchmark_v2.py`
- Test: `tests/unit/test_benchmark_batch.py`

**Interfaces:**
- Produces: `NeuralExecution(device: str, required_gpu_name: str, cublas_workspace_config: str, allow_tf32: bool, precision: str)`; `ExecutionIdentity`; `OriginForecastBatch`; `ProgressSnapshot`; `execution_identity_digest(identity: ExecutionIdentity) -> str`.
- Consumes: existing `V2ModelSettings`, `DiagnosticsTables`, and config serialization.

- [ ] **Step 1: Write failing configuration and contract tests**

Add tests asserting the registered TOML loads `cuda:0`, exact RTX 4080 name, `:4096:8`, `allow_tf32=False`, and `precision="32-true"`; unknown devices, TF32, mixed precision, or altered registered defaults raise `ValueError`. Add defensive-copy tests for batch arrays/tables and stable identity-digest tests independent of mapping insertion order.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_config_benchmark_v2.py tests/unit/test_benchmark_batch.py -q`

Expected: FAIL because the execution and batch contracts do not exist.

- [ ] **Step 3: Implement the frozen contracts and TOML mapping**

Add the exact dataclasses above. `BenchmarkV2Settings.__post_init__` converts an execution mapping into `NeuralExecution` and requires the registered production value. Keep direct test fixtures able to construct `NeuralExecution.cpu()` explicitly. Normalize identity JSON with sorted keys and compact separators before SHA-256.

- [ ] **Step 4: Run focused and existing config tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_config.py tests/unit/test_config_benchmark_v2.py tests/unit/test_benchmark_batch.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/benchmark_batch.py src/hcp/config_benchmark_v2.py src/hcp/forecast_types.py src/hcp/config.py configs/model_benchmarks_v2.toml tests/unit/test_config_benchmark_v2.py tests/unit/test_benchmark_batch.py
git commit -m "feat(benchmark): register deterministic CUDA execution"
```

### Task 2: Add the CUDA launch gate and provenance probe

**Files:**
- Modify: `src/hcp/benchmark_batch.py`
- Modify: `src/hcp/run.py`
- Test: `tests/unit/test_benchmark_batch.py`
- Test: `tests/unit/test_run_benchmark_v2.py`

**Interfaces:**
- Consumes: `NeuralExecution` from Task 1.
- Produces: `DeviceProvenance`; pure `device_identity(execution: NeuralExecution, torch_module: object, nvidia_identity: Mapping[str, str]) -> DeviceProvenance`; run-owned `_query_nvidia_identity(device_index: int) -> Mapping[str, str]` and `_configure_v2_cuda(execution: NeuralExecution) -> DeviceProvenance`.

- [ ] **Step 1: Write failing device-gate tests**

Use a small fake Torch CUDA boundary to assert GPU UUID/name/capability, driver, Torch/CUDA versions, TF32 flags, deterministic mode, and cuBLAS setting are captured. Assert unavailable CUDA, wrong GPU name, more than the selected visible device, and inability to enable deterministic algorithms fail before forecast calls.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py -q`

Expected: FAIL because device probing/configuration is absent.

- [ ] **Step 3: Implement the launch gate**

Keep environment mutation and Torch global configuration in `run.py`; keep provenance normalization in `benchmark_batch.py`. Set `CUBLAS_WORKSPACE_CONFIG` before the first CUDA operation, disable both matmul and cuDNN TF32, set highest float32 matmul precision, enable strict deterministic algorithms, select device zero, and return exact provenance including UUID.

- [ ] **Step 4: Run focused tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py -q`

Expected: PASS without requiring physical CUDA.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/benchmark_batch.py src/hcp/run.py tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py
git commit -m "feat(run): enforce registered CUDA launch gate"
```

### Task 3: Thread device execution through all neural adapters

**Files:**
- Modify: `src/hcp/forecast.py`
- Modify: `src/hcp/forecast_neural.py`
- Modify: `src/hcp/forecast_deepar.py`
- Modify: `src/hcp/forecast_deepar_rollout.py`
- Modify: `tests/unit/test_forecast.py`
- Modify: `tests/unit/test_forecast_v2.py`
- Modify: `tests/unit/test_forecast_deepar.py`
- Modify: `tests/unit/test_forecast_transformer.py`
- Modify: `tests/unit/test_forecast_deepar_rollout.py`
- Modify: `tests/integration/test_forecast_models.py`
- Modify: `tests/integration/test_forecast_deepar_enhanced.py`
- Modify: `tests/integration/test_forecast_transformer_integration.py`

**Interfaces:**
- Consumes: `NeuralExecution` from Task 1.
- Produces: four-argument `ForecastAdapter(histories, settings, seed, execution)`; device-aware `deepar_samples`, `transformer_prediction`, `enhanced_deepar_prediction`, and `recurrent_rollout`.

- [ ] **Step 1: Write failing adapter-boundary tests**

Assert the three neural adapters receive `cuda:0`; Lightning gets `accelerator="gpu"`, `devices=[0]`, deterministic strict mode, and `precision="32-true"`; loader workers remain zero and pinning false. Assert classical adapters ignore the execution device. Extend rollout tests so tensors follow a fake module device and NumPy conversion happens only after `.cpu()`. Add reduced real-CUDA integration tests that run each adapter twice with the same seed and require exact samples, tuning rows, warning counts, and seed evidence; these skip only when CUDA is unavailable.

- [ ] **Step 2: Run unit tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast.py tests/unit/test_forecast_v2.py tests/unit/test_forecast_deepar.py tests/unit/test_forecast_transformer.py tests/unit/test_forecast_deepar_rollout.py tests/integration/test_forecast_models.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py -q`

Expected: FAIL on the missing fourth adapter argument and CPU-hardcoded trainers/tensors, including the real-CUDA assertions when run with GPU access.

- [ ] **Step 3: Implement explicit device plumbing**

Update the adapter protocol and registry wrappers. Use device-aware `fork_rng`, CPU and CUDA seed calls derived from the existing seed integer, model-device tensor construction, and `.detach().cpu().numpy()` at output boundaries. Preserve CPU execution for explicit test fixtures and keep every existing seed spawn address unchanged.

- [ ] **Step 4: Run unit and reduced real-backend tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast.py tests/unit/test_forecast_v2.py tests/unit/test_forecast_deepar.py tests/unit/test_forecast_transformer.py tests/unit/test_forecast_deepar_rollout.py tests/integration/test_forecast_models.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py -q`

Expected: PASS on CPU fixtures; CUDA-specific real tests may skip only when CUDA is unavailable.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/forecast.py src/hcp/forecast_neural.py src/hcp/forecast_deepar.py src/hcp/forecast_deepar_rollout.py tests/unit/test_forecast.py tests/unit/test_forecast_v2.py tests/unit/test_forecast_deepar.py tests/unit/test_forecast_transformer.py tests/unit/test_forecast_deepar_rollout.py tests/integration/test_forecast_models.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py
git commit -m "feat(forecast): run neural adapters on explicit devices"
```

### Task 4: Train each neural candidate with one Lightning lifecycle

**Files:**
- Modify: `src/hcp/forecast_neural_validation.py`
- Modify: `src/hcp/forecast_neural.py`
- Modify: `src/hcp/forecast_deepar.py`
- Modify: `src/hcp/forecast_warnings.py`
- Modify: `tests/unit/test_forecast_neural_validation.py`
- Modify: `tests/unit/test_forecast_deepar_selection.py`
- Modify: `tests/unit/test_forecast_transformer.py`
- Modify: `tests/integration/test_forecast_deepar_enhanced.py`
- Modify: `tests/integration/test_forecast_transformer_integration.py`

**Interfaces:**
- Consumes: device-aware candidates from Task 3 and the existing fourth-step NLL objective.
- Produces: `CandidateTrainingResult`; `NeuralCandidate.fit_selection(split, validation_seed, max_epochs, patience) -> CandidateTrainingResult`; unchanged external `select_neural(...) -> SelectionResult`.

- [ ] **Step 1: Write failing lifecycle and equivalence tests**

Assert one Trainer/`fit` lifecycle per candidate selection and one for the selected full refit, rather than one per epoch. Pin epoch-by-epoch NLL `[3, 2, 2, ...]`, earliest-best retention, ten-unimproved stopping, width tie-breaking, Adam/sampler continuity, warning phases/counts, and exact same-seed outputs against the current reduced CPU characterization. Add a bounded-log assertion proving trainer startup text does not scale with epoch count.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast_neural_validation.py tests/unit/test_forecast_deepar_selection.py tests/unit/test_forecast_transformer.py -q`

Expected: FAIL because real candidates currently construct or invoke a Trainer once per epoch.

- [ ] **Step 3: Implement one-fit selection callbacks**

Move epoch validation, strict improvement, best-state copying, and patience stopping into a Lightning callback used by each real candidate's single selection fit. Full refit uses one independent Trainer for exactly `best_epoch` epochs. Keep `select_neural` responsible for candidate comparison and independent refit streams; keep warning capture separated into training, validation, retention, and refit phases. Do not change NLL calculation, sampler order, optimizer, or seed addresses.

- [ ] **Step 4: Run focused unit and real-backend tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast_neural_validation.py tests/unit/test_forecast_deepar_selection.py tests/unit/test_forecast_transformer.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py -q`

Expected: PASS with exact reduced characterization and bounded framework output.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/forecast_neural_validation.py src/hcp/forecast_neural.py src/hcp/forecast_deepar.py src/hcp/forecast_warnings.py tests/unit/test_forecast_neural_validation.py tests/unit/test_forecast_deepar_selection.py tests/unit/test_forecast_transformer.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py
git commit -m "perf(forecast): reuse neural training lifecycles"
```

### Task 5: Split forecasting into plan, origin execution, and assembly

**Files:**
- Modify: `src/hcp/forecast.py`
- Modify: `src/hcp/forecast_types.py`
- Modify: `tests/unit/test_forecast.py`
- Modify: `tests/unit/test_forecast_v2.py`

**Interfaces:**
- Consumes: `OriginForecastBatch`, `NeuralExecution`, existing preparation and adapter registry.
- Produces: `ForecastPlan`; `plan_model_forecasts(...) -> ForecastPlan`; `generate_origin_batch(plan: ForecastPlan, model: str, target_index: int, cfg: ForecastConfig) -> OriginForecastBatch`; `assemble_model_bundles(plan: ForecastPlan, batches: tuple[OriginForecastBatch, ...], cfg: ForecastConfig) -> ForecastResult`; unchanged `generate_model_bundles(...) -> ForecastResult` compatibility wrapper.

- [ ] **Step 1: Write failing decomposition/equivalence tests**

With fake adapters, assert plan keys are model-major/origin-minor, ineligible origins retain their seed positions, each eligible key calls exactly once, arbitrary input batch ordering assembles into registered ordering, duplicate/missing/wrong-origin batches fail, and split execution is scientifically equal to the compatibility wrapper. Include completed earlier origins to prove later seeds do not shift.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast.py tests/unit/test_forecast_v2.py -q`

Expected: FAIL because the decomposition API is absent.

- [ ] **Step 3: Implement the pure decomposition**

Move one iteration of the existing nested loop into `generate_origin_batch`; centralize warning/diagnostic normalization there. `assemble_model_bundles` recreates the exact `q` arrays, long tables, model elapsed summaries, diagnostic tables, and source-kind labels. Implement the legacy wrapper solely by calling the three new interfaces.

- [ ] **Step 4: Run forecast tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_forecast.py tests/unit/test_forecast_v2.py tests/unit/test_forecast_diagnostics.py -q`

Expected: PASS with unchanged legacy and v2 values.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/forecast.py src/hcp/forecast_types.py tests/unit/test_forecast.py tests/unit/test_forecast_v2.py
git commit -m "refactor(forecast): expose resumable origin batches"
```

### Task 6: Add atomic batch storage and validation owned by `run.py`

**Files:**
- Modify: `src/hcp/benchmark_batch.py`
- Modify: `src/hcp/run.py`
- Modify: `tests/unit/test_benchmark_batch.py`
- Modify: `tests/unit/test_run_benchmark_v2.py`

**Interfaces:**
- Consumes: `ExecutionIdentity`, `OriginForecastBatch`, existing `_rename_noreplace`.
- Produces: run-owned `_initialize_batch_workdir`, `_write_origin_batch`, `_read_origin_batch`, `_write_progress`, `_append_runtime_event`, and `_read_runtime_events`; pure payload schemas/digests in `benchmark_batch.py`.

- [ ] **Step 1: Write failing atomicity and corruption tests**

Assert `run-state.json` is immutable; every payload hash and row count is verified; swapped model/origin manifests and seed keys fail; a crash before `batch.json`, malformed Parquet, truncated JSON, extra payload, duplicate final batch, schema mismatch, and execution digest mismatch all fail without adapter fallback. Assert a fully written batch round-trips defensive immutable values and temporary siblings never count as complete.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py -q`

Expected: FAIL because batch persistence is absent.

- [ ] **Step 3: Implement atomic batch and progress writes**

Only functions in `run.py` call filesystem write APIs. Build each batch in a UUID temporary sibling, write four diagnostic Parquets plus forecasts, write `batch.json` last, then no-replace rename. Replace `progress.json` through a temporary file and `os.replace`; append one compact JSON object per runtime line and tolerate only a truncated final line when reading logs.

- [ ] **Step 4: Run atomic writer tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/benchmark_batch.py src/hcp/run.py tests/unit/test_benchmark_batch.py tests/unit/test_run_benchmark_v2.py
git commit -m "feat(run): persist immutable forecast batches"
```

### Task 7: Orchestrate resume, concise logging, and quiet dependencies

**Files:**
- Modify: `src/hcp/run.py`
- Modify: `tests/unit/test_run_benchmark_v2.py`
- Modify: `tests/e2e/test_run_benchmark_v2_cli.py`

**Interfaces:**
- Consumes: Tasks 2, 5, and 6 interfaces.
- Produces: `_generate_or_resume_v2(...) -> tuple[ForecastResult, ResumeSummary]`; stable human console events and structured `runtime.jsonl` records.

- [ ] **Step 1: Write failing resume and logging tests**

Simulate interruption after two batches, rerun, and assert only missing adapters execute while final forecasts/diagnostics equal an uninterrupted run. Parameterize Git/config/lock/data/GPU UUID/name/capability/driver/runtime mismatches and assert failure before batch load or fit. Capture output and assert one model-start/batch/resume/evaluation event, ETA only after a completion, no Lightning banner/tip, no tqdm bar, and preserved warning-table rows.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_v2_cli.py -q`

Expected: FAIL because CLI resume orchestration is absent.

- [ ] **Step 3: Implement the resumable loop and scoped log suppression**

Initialize or validate identity, iterate the pure plan, read valid completed batches or generate/write missing ones, update progress after each batch, then assemble. Scope Lightning/Darts logger levels around neural calls without changing Python warning filters. Set and restore `TQDM_DISABLE=1` around evaluation. Errors emit one structured event and preserve the work directory.

- [ ] **Step 4: Run focused CLI tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_v2_cli.py -q`

Expected: PASS with bounded captured output.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/run.py tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_v2_cli.py
git commit -m "feat(run): resume v2 forecasts with concise progress"
```

### Task 8: Finalize the work directory and publish atomically

**Files:**
- Modify: `src/hcp/run.py`
- Modify: `tests/unit/test_run_benchmark_v2.py`
- Modify: `tests/e2e/test_run_benchmark_v2_cli.py`

**Interfaces:**
- Consumes: `ForecastResult`, `ResumeSummary`, execution identity, verified batch work directory.
- Produces: `_finalize_benchmark_v2_artifacts(...) -> Path`; final manifest batch/resume/device/artifact provenance.

- [ ] **Step 1: Write failing finalization tests**

Assert ten Parquets, five PDFs, batches, logs, progress, and final manifest coexist only after publication; manifest hashes every canonical artifact and records completed/resumed counts and exact CUDA provenance. Assert Git change, damaged batch after evaluation, render failure, existing destination, and destination race preserve the hidden work directory and never overwrite a final directory. Assert manifest is the last run-level artifact written.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_v2_cli.py -q`

Expected: FAIL against the old UUID writer/cleanup semantics.

- [ ] **Step 3: Implement final assembly and no-replace publication**

Write canonical outputs beside the verified batch evidence, hash them, write `manifest.json` last, recheck all batch hashes and clean unchanged Git SHA, then rename the fixed hidden work directory to the final destination with `_rename_noreplace`. Never delete resumable state on an exception.

- [ ] **Step 4: Run all CLI and publication tests GREEN**

Run: `uv run pytest -o addopts='' tests/unit/test_run.py tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_cli.py tests/e2e/test_run_benchmark_cli.py tests/e2e/test_run_benchmark_v2_cli.py tests/e2e/test_run_panel_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcp/run.py tests/unit/test_run_benchmark_v2.py tests/e2e/test_run_benchmark_v2_cli.py
git commit -m "feat(run): publish resumed benchmark atomically"
```

### Task 9: Verify CUDA repeatability, document operations, and prepare relaunch

**Files:**
- Modify: `docs/analysis/model-benchmarks-v2-protocol-readiness.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed implementation and registered RTX 4080.
- Produces: CUDA repeatability evidence and exact launch/progress/resume instructions.

- [ ] **Step 1: Run the CUDA repeatability gates outside the sandbox**

Run: `UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda UV_NO_SYNC=1 uv run pytest -o addopts='' tests/integration/test_forecast_models.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py -q`

Expected: PASS twice on the RTX 4080 with the exact assertions introduced before Task 3 implementation.

- [ ] **Step 2: Update readiness and reproduction documentation**

Record the aborted pre-amendment attempts as unpublished, the amendment and implementation SHAs, GPU/driver/Torch/CUDA identity, verification commands, batch directory contract, `progress.json` monitoring command, resume command, and prohibition on commits during execution. Do not add outcome interpretation or claim that the official run has completed.

- [ ] **Step 3: Run complete verification**

Run:

```bash
uv run pytest -o addopts='' tests/unit tests/integration tests/e2e -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Expected: all tests pass with only reviewed dependency warnings; Ruff and diff checks pass.

- [ ] **Step 4: Commit**

```bash
git add docs/analysis/model-benchmarks-v2-protocol-readiness.md README.md
git commit -m "docs(benchmark): register resumable CUDA launch evidence"
```

## Final review and launch gate

- [ ] Request a whole-branch review against both v2 specs, resolve all Critical/Important findings with RED/GREEN evidence, and rerun scoped tests after each fix.
- [ ] Run the full verification commands from Task 9 after the last review change.
- [ ] Confirm the worktree is clean, `HEAD` will remain fixed, `results/model-benchmarks-v2` is absent, and any hidden work directory either matches the exact execution identity or is preserved outside the registered path.
- [ ] Run `uv run python -m hcp.run configs/model_benchmarks_v2.toml` in an interactive terminal with CUDA access.
- [ ] Monitor `results/.model-benchmarks-v2.partial/progress.json` and `runtime.jsonl`; do not commit, merge, rebase, pull, or checkout until publication finishes.
- [ ] Validate the final folder, manifest, canonical artifact counts/hashes, batch hashes, Git SHA, and absence of hidden staging.
- [ ] Push the reviewed feature branch and create the authorized PR from `feat/model-benchmarks-v2` to `develop`, including verification and official-run status without scientific interpretation.
