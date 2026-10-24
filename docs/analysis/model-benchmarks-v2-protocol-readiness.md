# Model benchmarks v2 protocol readiness

**Status: the official run published successfully from clean Git SHA `06eb8017e6f302825debecf0acbc65abc4b7960d` on 2026-10-24.** The final whole-branch review and the scoped UUID-launch-fix review both passed. Fresh clean-commit physical-CUDA verification passed with 620 tests and one legacy-artifact skip before launch. This is a protocol and verification record, not an outcome report; it does not authorize scientific interpretation.

## Registered identity

The immutable study and run identity is `model-benchmarks-v2`, configured by [`configs/model_benchmarks_v2.toml`](../../configs/model_benchmarks_v2.toml). The numerical protocol hash is SHA-256 over the exact bytes of that registered TOML: `c9978b8d990f699f7e5e96170a118e9abeacba941ee939a9876edd4679e0f8bf`. This is the historical pre-amendment config SHA-256; the amended CUDA TOML SHA-256 is `18c326dfb661097c6aa701b6eda5f6f0ac149addbf4d87e32cd3862c95997602`. The lockfile SHA-256 at registration is `551bf39a924beed6236e2e8a778f7124b2e9e63f6a60ad5c51d0a9ee25453355`; the input `data/deaths.csv` SHA-256 is `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`.

The seven generated models are ARIMA, SARIMA, Darts DeepAR-style LSTM, Prophet, Darts Transformer, local native-L1 quantile regression, and enhanced DeepAR-style LSTM. The reference is `covidhub` / `COVIDhub-4_week_ensemble`. All 57 hierarchy nodes participate; the inferential leaf endpoint is the 51 states/DC. The design uses untransformed deaths, weekly expanding fits with at least 60 training weeks, four-step forecasts retaining step four, weekly refits, and one common evaluation window with at least 60 weeks.

Principal arms are A=`none/raw`, B=`none/pi_norm`, C=`mint_median/raw`, D=`mint_median/pi_norm`. The twelve cells from four reconcilers crossed with `raw`, original `pi`, and `pi_norm` are produced per model; cells outside A-D are descriptive. Normalized PI uses a per-node scale fixed from the prepared prefix at the first common evaluation origin and the existing core PI with `KI=1`, `Csat=2`, `lr=0.1`, `T_burnin=5`, `ahead=4`, and alpha 0.2 split 0.1 per side. Original PI retains `KI=1000`.

The two principal leaf-mean endpoints are per-node scaled interval score and absolute marginal coverage error, averaged unweighted over endpoint leaves after computing the per-node errors. The intervention family contains eight models × two metrics × six A-D contrasts = 96 tests, Holm-adjusted jointly per block. The model-reference family contains seven generated models × four matched arms × two metrics = 56 tests, Holm-adjusted jointly per block. The paired noncircular moving-block bootstrap uses B=2000, block 8 as the main exploratory analysis, and blocks 4 and 12 for sensitivity. Root seed is 20261021; forecast and evaluation `SeedSequence` spawn keys are `(0,)` and `(1,)`; each block's week-index draws are shared across nodes, models, arms, metrics, and both families.

Empty PI sets (`lo > hi`) are retained without sorting final bounds and count as misses. The registered score remains the signed quantile-pair formula, including for empty sets; reports also provide nonnegative physical set width and empty-set rates. D15 limitations remain: the pipeline uses final revised truth rather than as-of vintages, and inherited PI learning-rate-scale look-ahead remains, so the full pipeline is not strictly causal. V1 outcomes informed v2, which remains explicitly exploratory; its two Holm families do not provide a study-wide error guarantee across all analyses.

## Review and verification

The whole-branch review of the implementation found two Important and three Minor findings. Consolidated fixes were committed in `bd67f17` (`fix(benchmark): close v2 whole-branch review findings`). The scoped rereview resolved all five findings and reported no new Critical or Important breakage. The review did not run, inspect, or reproduce the official v2 pipeline.

After the fix commit, the complete unit, integration, and end-to-end suite passed: **447 passed, 1 skipped, 6 warnings**. This includes the reduced real-data integration that runs all seven generated adapters on a one-origin US/California slice and checks causal cutoffs, finite output shapes, and no publication. The synthetic v2 CLI test separately checks all eight model identities, artifact shapes, manifest contracts, provenance, and publication ordering.

Repository Ruff gates passed after the consolidated fix: `ruff check .` reported all checks passed; `ruff format --check .` reported 106 files already formatted. The fresh final suite and Ruff commands used the existing project environment offline. An initial offline uv invocation attempted to build the local project and could not resolve uncached `hatchling`; rerunning with `--no-sync` reused the installed environment without dependency or network changes. Versions: Python 3.11.13, pytest 9.1.1, Ruff 0.16.9, Darts 0.47.0, PyTorch 2.14.1, Prophet 1.5.0.

## Remaining limits and launch boundary

The reduced development integration checks adapter availability and interfaces, not forecast quality or the registered common-window result. Bootstrap intervals are pointwise and conditional on completed fitted pipelines. The study is exploratory, and D15's truth-vintage and PI look-ahead limitations remain. These do not change after protocol registration.

## Official execution and publication audit

The official command ran on the registered physical `cuda:0` device and exited with code 0 after 11,715.53 seconds. It generated all 602 registered model-origin batches: 86 origins for each of ARIMA, SARIMA, DeepAR, Prophet, Transformer, local L1 quantile regression, and enhanced DeepAR. The runner emitted external `publication-complete` evidence with execution-identity digest `6c01c5381b453e14d6104f006acf92a39c295ae8b506ea3022234150cc744951`, atomically promoted the work directory, and left no `.model-benchmarks-v2.partial` directory.

The published bundle is the local, gitignored `results/model-benchmarks-v2/` directory. A read-only post-publication audit recalculated all 3,630 SHA-256 entries in `manifest.json`; no file was missing or different. The manifest, `run-state.json`, config, input, lockfile, initial/final Git SHA, exact CUDA identity, and canonical identity digest agree. All 602 batch directories contain the registered six-file payload, and all 34,314 generated batch forecast rows have 57 nodes with finite ordered quantiles.

The ten canonical Parquet tables opened successfully. Their row counts are 62,472 base forecasts, 344,736 evaluated intervals, 5,472 node metrics, 192 level metrics, 168 model contrasts, 288 hierarchy contrasts, 19,952 tuning records, 794 quantile-repair records, 2,360 warning records, and 10,086 forecast-diagnostic records. The evaluation tables cover eight models, 57 nodes, 63 common weeks, and all twelve reconciler-controller cells with finite scored metrics and contrasts. The 20,349 unavailable rows in the longer 137-week base-forecast frame use complete three-quantile null triplets; none enters the finite common evaluation window. The 1,542 final rows with `lo > hi` are registered empty PI sets and remain visible in `empty_set_rate`.

The warning table represents 30,945 occurrences after aggregation. Darts-backed models account for 30,340: 344 for DeepAR, 1,032 for enhanced DeepAR, and 28,964 for Transformer. The remaining 605 are ARIMA/SARIMA estimation or convergence warnings. These counts are retained diagnostic evidence; the successful hard gates establish structural and numerical validity, not forecast quality.

## Prospective CUDA amendment and unpublished attempts

The first pre-amendment full attempt completed most computation but refused publication because Git `HEAD` changed during execution. A second attempt was stopped before changing the execution design. Both attempts remain unpublished; neither produced `results/model-benchmarks-v2/`. This record makes no outcome interpretation.

The prospective amendment design is `5a6b13b81ec450f3f5f4d97aa0866c18ec149d97`; registration/contracts are `e5a7d91fdb4b84d9f7ff4bda900032968e830450`. Task 9 verified implementation through atomic resumed publication at `2aea38b4cdb11afc92a222dab9107fafa34a5711`. These anchors are not a future launch SHA: launch must record the final reviewed clean `HEAD`, including documentation and review fixes.

D33 in [`protocol-decisions.md`](../protocol-decisions.md) prospectively registers the amended exact config hash and execution/provenance clauses while retaining D32 as history. Scientific settings remain fixed; D33 does not authorize execution or interpretation.

On 2026-10-24 the prescribed repeatability suite ran twice outside the sandbox on physical `cuda:0`: NVIDIA GeForce RTX 4080, UUID `GPU-f3dad84a-4909-bd89-f509-0c2f79debd6f`, 16376 MiB, driver `610.88`, Torch `2.14.1+cu130`, CUDA runtime `13.0`, cuDNN `92400`; CUDA available. Both invocations passed **15 tests**, with **77 reviewed dependency warnings** each, in **23.03 s** and **21.99 s**. CPU characterization cases coexist with actual CUDA cases. Warnings concern CPU fixtures, absent validation loaders, pytree deprecation, low worker count, nested tensors, and empty prediction loaders.

```bash
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda UV_NO_SYNC=1 uv run pytest -o addopts='' tests/integration/test_forecast_models.py tests/integration/test_forecast_deepar_enhanced.py tests/integration/test_forecast_transformer_integration.py -q
```

Full Task 9 verification used the following commands. The exact initial `uv run pytest` command failed because the default uv cache is read-only; with a writable cache, build dependency resolution failed fetching `hatchling` from PyPI due to DNS. Therefore `--no-sync` reused the installed environment without changing dependencies.

```bash
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda uv run --no-sync pytest -o addopts='' tests/unit tests/integration tests/e2e -q
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda uv run --no-sync ruff check .
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda uv run --no-sync ruff format --check .
git diff --check
```

Outside the sandbox, the full suite reported **601 passed, 1 skipped, 78 warnings, 1 teardown error (50.68 s)**. The unit DeepAR warning fixture expects `Can't initialize NVML`, whereas the physical GPU supplies `GPU available but not used`; its exact warning-set assertion fails. Ruff lint passed. Format verification failed on existing `src/hcp/forecast_assembly.py:104` (one file would be reformatted; 113 already formatted). These were the historical pre-review failures. The authorized Task 9 review fix normalized only the two exact reviewed device-warning alternatives while preserving the count of 32 and every other warning assertion, and applied Ruff formatting to that single call. The focused unit module first reproduced the physical failure (12 passed, 1 teardown error in 2.81 s), then passed outside the sandbox (12 passed in 2.95 s) and inside it (12 passed in 11.61 s). After both fixes, the full suite above ran outside the sandbox on the physical RTX 4080 with `UV_NO_SYNC=1`: **601 passed, 1 skipped, 78 reviewed warnings in 50.37 s**. Ruff lint, Ruff format (114 files already formatted), and `git diff --check` passed. No model numerical logic changed.

## Interactive launch, monitoring, and exact resume

After whole-branch review, fresh full verification, and explicit launch authorization, use an interactive terminal with physical CUDA access. For a fresh execution both registered directories must be absent. Preserve any incompatible work directory outside the registered path; never edit identity or batches to force reuse.

```bash
cd /home/vinvs/projects/research/pid-control-ts/.worktrees/model-benchmarks-v2
git status --short
git rev-parse HEAD
test ! -e results/model-benchmarks-v2
test ! -e results/.model-benchmarks-v2.partial
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda UV_NO_SYNC=1 uv run python -m hcp.run configs/model_benchmarks_v2.toml
```

From a second terminal in the same worktree:

```bash
watch -n 5 cat results/.model-benchmarks-v2.partial/progress.json
tail -f results/.model-benchmarks-v2.partial/runtime.jsonl
```

The fixed work directory contains immutable `run-state.json`, advisory `progress.json`, `runtime.jsonl`, and atomic `batches/<model>/<origin>.batch/` directories. Each batch contains `forecasts.parquet`, `forecast_diagnostics.parquet`, `tuning.parquet`, `quantile_repairs.parquet`, `warnings.parquet`, and `batch.json` with schema, identity, and payload hashes. Generation follows the deterministic model-major/origin-minor plan. Valid existing batches are skipped; absent batches are generated. Invalid completed batches fail closed; temporary writes are never accepted as completed batches.

Resume uses the **same command**, without a resume flag, from the same worktree and exact clean Git SHA:

```bash
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda UV_NO_SYNC=1 uv run python -m hcp.run configs/model_benchmarks_v2.toml
```

Resume requires absent final output and matching full execution identity: Git SHA, config/input/lock hashes, seed and origin plan, packages, device, GPU UUID, driver, runtime, and deterministic precision settings. The CUDA contract uses `cuda:0`, FP32 `32-true`, no TF32, and cuBLAS workspace `:4096:8`; no CPU fallback is accepted. A mismatch preserves evidence and stops. Keep `HEAD` fixed through publication and across resume: **do not commit, merge, rebase, pull, or checkout during execution**. Do not change dependencies, inputs, config, or device environment. Progress and ETA are advisory; validated batches govern recovery.

Final publication atomically renames the hidden work directory to immutable `results/model-benchmarks-v2/` after revalidating evidence and Git identity. An existing result is never overwritten. Task 9 itself did not launch the benchmark; the separately authorized official execution later published the audited bundle described above.

## Canonical diagnostics and final-review fixes

Generation tables and evaluation-stage evidence are combined before canonical publication. `tuning.parquet` retains one row per QR node/origin/alpha with a numeric `score`, string `candidate`, and Boolean `selected`; exactly one alpha per node/origin is selected. Neural tuning uses one row per width, including its best fourth-step NLL, earliest best epoch, actual epochs run, inner-fit scales and enhanced covariate transforms. Exact ties retain the registered alpha/width rules. The canonical manifest's selected-candidate summary uses these same records.

`forecast_diagnostics.parquet` stores structured events by model, node (null for global records), forecast date, phase, name and value. Nested adapter evidence is canonical JSON in `value`: neural child-stream addresses, Prophet node seeds, QR adjustment summaries, and full-refit width/epochs/scales/covariates survive Arrow persistence and exact resume. The `origin_seed` event includes the complete NumPy seed state; `final_fit` events use phase `full_refit`, separately from inner-fit tuning rows. Arrays in manifest summaries remain numeric lists, and absent fields in heterogeneous schemas become JSON null.

`quantile_repairs.parquet` retains base and reconciliation stages separately. Reconciliation records include method, node, forecast/target dates, pre/post quantiles, crossing flag and absolute adjustment summaries. Per-origin events retain pre/post root-minus-leaf-sum coherence residuals at all three quantiles, adjusted/crossing cell counts, root crossings and mean/max adjustments for each reconciler. Sorting after projection can break quantile coherence; these events disclose that loss. Legacy reconciliation formulas and v1 output behavior remain unchanged.

Both evaluation and the final writer reject nonfinite scored metric or contrast columns, including descriptive arms, with cell context. The existing incoherent 57-node synthetic fixture triggers the core PI tangent saturation under `bottom_up/pi_norm`; its regression requires failure. This is a synthetic characterization, not an official outcome. No finite gate, controller setting, score, seed, or `core/` implementation was changed to accommodate it.

The final manifest includes `execution_identity_digest`, using the same compact canonical identity hash as each immutable batch, and describes independent `(block, family)` Holm groups with 96 and 56 members. `publication-ready` is a prepublication runtime event included in the artifact hashes before the manifest is written last. Only a successful atomic rename emits external `publication-complete` JSON; a failure can leave a ready event in the preserved partial directory, never a completion event. Published files are not modified after hashing.

The consolidated fix evidence is `.superpowers/sdd/2026-10-23-model-benchmarks-v2-resumable-cuda/final-fix-report.md`. The independent scoped rereview resolved all six P2 and five P3 findings with no new Critical or P2 finding. It independently passed 167 focused tests and the reduced real-data integration, and accepted the retained clean-commit physical-CUDA result of 614 passed with one legacy-artifact skip. The official benchmark was not used to test these fixes.
