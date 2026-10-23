# Model benchmarks v2 protocol readiness

**Status: registered and implementation gates passed; awaiting separate launch authorization.** This is a protocol and verification record, not an outcome report. No official v2 realization or scientific contrast has been generated or inspected.

## Registered identity

The immutable study and run identity is `model-benchmarks-v2`, configured by [`configs/model_benchmarks_v2.toml`](../../configs/model_benchmarks_v2.toml). The numerical protocol hash is SHA-256 over the exact bytes of that registered TOML: `c9978b8d990f699f7e5e96170a118e9abeacba941ee939a9876edd4679e0f8bf`. This is also the config SHA-256. The lockfile SHA-256 at registration is `551bf39a924beed6236e2e8a778f7124b2e9e63f6a60ad5c51d0a9ee25453355`; the input `data/deaths.csv` SHA-256 is `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`.

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

Registration, review, tests, and lint checks do not authorize execution. `results/model-benchmarks-v2/` is absent. No official run, result, or in-memory reproduction was created or inspected. The prospective command is:

```bash
uv run python -m hcp.run configs/model_benchmarks_v2.toml
```

Run it only after separate explicit launch authorization. The runner publishes once to the registered identity and refuses to overwrite an existing result.
