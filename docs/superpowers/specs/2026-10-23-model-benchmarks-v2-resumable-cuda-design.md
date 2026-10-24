# Model Benchmarks v2 Resumable CUDA Execution Design

**Date:** 2026-10-23  
**Status:** Proposed prospective amendment; implementation and relaunch require approval  
**Amends:** `2026-10-22-probabilistic-model-benchmarks-v2-design.md`, sections 7 and 9

## Intent

The first full v2 attempt completed most computation but correctly refused publication
because Git `HEAD` changed during the run. A second attempt was stopped before changing
the execution design. Neither attempt published `results/model-benchmarks-v2`, so this
amendment is prospective: it changes the registered execution environment and recovery
mechanism before the first successful official result.

The amended runner must expose useful progress, preserve completed forecast work across
process interruption, reduce repetitive framework output, and use the available NVIDIA
GPU for neural models without weakening causal validation, seed separation, failure
handling, or atomic final publication.

## Goals

- Persist every completed `(model, eligible outer origin)` as an independently validated
  batch below `results/.model-benchmarks-v2.partial/`.
- Resume only batches whose complete execution identity and content hashes match the new
  process exactly.
- Keep `results/model-benchmarks-v2/` absent until all forecasts, evaluation tables,
  figures, and the final manifest are complete.
- Run ARIMA, SARIMA, Prophet, and L1 quantile regression on CPU; run legacy DeepAR,
  Transformer, and enhanced DeepAR on one CUDA device.
- Provide concise model/origin progress, elapsed time, and ETA while keeping scientific
  warning evidence.
- Preserve the registered model definitions, tuning budgets, causal splits, seed tree,
  common evaluation window, arms, contrasts, and inferential families.

## Non-goals and anti-goals

This amendment does not parallelize outer origins, retune models, change epoch budgets,
alter batch sizes, change likelihoods, add mixed precision, enable TF32, retry failed
fits, or accept a CPU fallback after launch. It does not publish partial batches as
scientific results. A resume must never combine batches produced by different code,
configuration, data, lockfile, device contract, or seed layout.

## Considered approaches

### Complete-model batches

Writing one batch after each model requires the smallest forecast refactor. It still
loses every origin of a long neural model if the process stops before that model
finishes. This does not address the observed failure cost adequately.

### Model-origin batches with sequential execution — selected

The runner writes one atomic batch after each eligible origin and resumes at the next
missing batch. It follows the current deterministic model-major/origin-minor order and
does not add scheduling nondeterminism. The extra small files and assembly logic are an
acceptable cost for fine-grained recovery.

### Concurrent CPU and GPU queues

Running classical CPU work while neural CUDA work proceeds could reduce wall time
further. It adds process coordination, resource contention, interleaved logs, and more
complex failure and seed tests. It is deferred until one resumable sequential official
run establishes a measured baseline.

## Execution identity and launch gates

`run.py` creates the fixed sibling work directory
`results/.model-benchmarks-v2.partial/` with an immutable `run-state.json` before the
first fit. The state records:

- protocol and run IDs;
- initial clean Git SHA;
- SHA-256 digests of the registered TOML, `uv.lock`, and input data;
- root seed, model order, eligible origin keys, quantiles, and batch schema version;
- Python and package versions;
- execution mode, CUDA device index, GPU UUID, GPU name and compute capability, NVIDIA
  driver, Torch CUDA runtime, deterministic-algorithm flag, TF32 flags, and cuBLAS
  workspace setting.

On a fresh launch the final directory and work directory must both be absent. On resume,
the final directory must be absent and every field in `run-state.json` must equal the
current identity. Every recorded CUDA identity field, including GPU UUID, driver, and
runtime versions, must match exactly on resume because any of them can alter kernel
selection or numerics. A mismatch is a hard failure and the runner leaves the work
directory untouched for operator inspection. It never deletes, overwrites, or silently
ignores a mismatched batch.

The registered neural device is `cuda:0`. Launch fails before forecasting if CUDA is
unavailable, the selected device differs from the recorded RTX 4080 execution contract,
or deterministic CUDA initialization cannot be established. There is no automatic CPU
fallback.

Before the official relaunch, reduced synthetic tests and a local execution probe must
show that every registered neural adapter runs twice with the same seed and produces
identical outputs on the selected GPU. Cross-device equality with the previous CPU
implementation is not required because no v2 result was published, but CPU fixtures
remain supported for automated tests through an explicit test-only execution setting.

## CUDA determinism

The CLI configures CUDA before model construction:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `torch.use_deterministic_algorithms(True, warn_only=False)`;
- Python/NumPy/Torch CPU and CUDA seeds derived from the existing `SeedSequence` tree;
- TF32 disabled for matmul and cuDNN;
- one visible device and one Lightning device;
- full float32 precision, without automatic or mixed precision;
- zero data-loader workers and unpinned host memory unless a later measured amendment
  explicitly changes them.

The model seed addresses and sampling order remain unchanged. Device-aware RNG scopes
include `cuda:0`; prediction and validation tensors are created on the model device and
copied to CPU before conversion to NumPy. The manifest records the realized device
contract. Any nondeterministic-operation error aborts the run and preserves completed
batches.

## Batch contract

The numeric forecast layer remains free of file I/O. It exposes a model/origin iterator
that yields one immutable result containing quantiles, optional sample-derived metadata,
interpolation records, diagnostic tables, warning counts, and elapsed timing. `run.py`
is the sole batch sink and the only module that writes files.

Each completed batch is built in a uniquely named temporary sibling and atomically
renamed without replacement to:

```text
results/.model-benchmarks-v2.partial/
  run-state.json
  progress.json
  runtime.jsonl
  batches/<model>/<origin>.batch/
    forecasts.parquet
    forecast_diagnostics.parquet
    tuning.parquet
    quantile_repairs.parquet
    warnings.parquet
    batch.json
```

`batch.json` is written last inside the temporary batch directory. It records model,
origin, model/origin seed spawn keys, row counts, elapsed time, execution identity hash,
and SHA-256 for every payload. An existing valid batch is loaded and yielded without
calling its adapter. A missing, partial, malformed, duplicate, schema-incompatible, or
hash-mismatched batch is a hard failure; the runner never repairs it by guessing which
files are trustworthy.

Ineligible origins have no batch. Their ordered keys are fixed in `run-state.json`, so
resume cannot alter the common grid. Assembly reconstructs each model bundle and all
diagnostics in the same registered ordering as an uninterrupted in-memory run. Tests
must establish semantic equality between uninterrupted and interrupted/resumed output.
Here semantic equality means equal scientific values, schemas, row ordering, seed
evidence, warning counts, selected candidates, and deterministic diagnostics.
Wall-clock timings, ETA estimates, runtime events, completed-versus-resumed counters,
and PDF container metadata describe execution history and are intentionally excluded.
Figures must represent equal plotted data and labels; their raw PDF bytes need not
match.

## Progress and logging

After every batch, `run.py` atomically replaces `progress.json`. It contains the current
model and origin, completed and total batch counts, per-model counts, elapsed seconds,
and an ETA computed only from completed batches. This file is advisory; batch manifests
are authoritative on resume.

The CLI emits one concise line at model start, batch completion, resume skip, evaluation
start, and final publication. `runtime.jsonl` retains the same structured events plus
errors. A truncated final JSONL line after interruption is ignored when reading logs;
it never determines resume eligibility.

Lightning device banners, trainer tips, model summaries, checkpoint messages, and
per-epoch progress are suppressed with scoped logger configuration around neural fits.
`TQDM_DISABLE=1` suppresses the unconditional progress bars inherited from `core` during
evaluation. Python warnings are still captured by the existing phase-aware collectors,
deduplicated with counts, and written to batch and final warning tables. No global
warning filter removes scientific evidence.

## Final assembly and publication

After all registered batches validate, the runner assembles the eight bundles, runs the
existing benchmark evaluation, and writes the ten canonical Parquet tables and five PDF
figures into the same work directory. It writes `manifest.json` last. The manifest adds
the batch schema, completed/resumed counts, execution identity hash, device details, and
the hashes of every canonical artifact.

Immediately before publication, `run.py` rechecks clean Git state and exact equality to
the initial SHA, revalidates every batch hash, and confirms the final destination is
absent. It then atomically renames the complete work directory to
`results/model-benchmarks-v2` with the existing no-replace primitive. Thus the final
result includes the runtime log and batch evidence that produced it, but only the
canonical tables, figures, and manifest are scientific outputs. A failure preserves the
hidden work directory for exact resume; it never exposes a final directory.

## Failure semantics

- Adapter, solver, CUDA determinism, nonfinite output, schema, hash, Git, or publication
  failures abort immediately.
- A failed current batch is never renamed into place.
- Previously completed valid batches remain immutable and resumable.
- Changing code or committing during execution invalidates the run identity. The work
  directory remains evidence and cannot be resumed under the new SHA.
- `--restart` or automatic deletion is not provided. Operators must preserve or move a
  rejected work directory explicitly before starting a different registered run.

## Test strategy

Implementation follows red-green-refactor with synthetic adapters and reduced neural
budgets. Tests must cover:

1. deterministic model-major/origin-minor events and unchanged seed addresses;
2. atomic batch creation and refusal to overwrite;
3. interruption before and after batch publication;
4. resumed versus uninterrupted semantic equality for forecasts, diagnostics, and final
   canonical artifacts;
5. rejection of every execution-identity and payload-hash mismatch;
6. absence of the final directory until complete publication and destination-race
   preservation;
7. concise progress/log output with warning evidence retained;
8. CUDA device-aware seeds, deterministic settings, tensor placement, manifest fields,
   and hard failure without the registered GPU;
9. two same-seed CUDA runs of each neural adapter on a reduced real backend;
10. all existing unit, integration, end-to-end, core characterization, Ruff, and format
    checks.

## Implementation boundaries

The main changes belong in a small run-owned batch/progress module, the forecast
iterator, neural device plumbing, frozen config dataclasses, and `run.py` orchestration.
No forecasting or evaluation formula changes. No `core/` source change is necessary;
the runner supplies `TQDM_DISABLE=1` while invoking existing code. Public functions keep
Google docstrings and type hints, dataclasses remain frozen, and new files use two
`# ABOUTME:` lines.
