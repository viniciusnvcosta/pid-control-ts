# Hierarchical Panel (leaves × forecasters) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 1 US-only study with a reproducible post-hoc diagnostics report (D19), then extend `src/hcp/` to the pre-registered Phase 2 panel. The panel reconciles, conformalizes and scores every scored node of 8 eligible COVIDhub forecasters, with the leaf-mean coverage deviation as the primary endpoint (D20–D27).

**Architecture:**
- `design = "us"` (the default) keeps Phase 1 byte-for-byte reproducible.
- `design = "panel"` reads the CSV once, builds one `HierarchyBundle` per forecaster, applies the D21 eligibility rule, and runs `panel.run_panel`.
- `run_panel` reuses `reconcile_bundle`, `conformalize`, a vectorized `evaluate_batch`, and `stats` per forecaster. Each forecaster draws one set of bootstrap week indices that every node shares.
- `run.py` stays the only writer under `results/`.

**Tech Stack:** Python 3.11, uv, numpy 2, pandas 3, matplotlib (OO API), pyarrow, pytest, ruff. No new dependencies.

**Design spec:** `docs/superpowers/specs/2026-10-20-hierarchical-panel-design.md`. Decisions: `docs/protocol-decisions.md` D19–D27, which override this plan.

## Global Constraints

- Every new `.py` file starts with two `# ABOUTME: ` lines. Type hints on every function; Google docstrings on public functions; frozen dataclasses for configs and results.
- Numeric modules (`reconcile`, `conformal`, `evaluate`, `stats`, `eligibility`, `panel`) do no I/O. `data` only reads. Only `run.py` writes, under `results/<run_id>/`, never overwriting. `scripts/analysis/*` writes only under its `--out` directory (default `docs/analysis/`).
- `core/` is not modified.
- `design = "us"` must reproduce Phase 1 exactly: every existing test keeps passing unchanged, and `load_hierarchy`, `run_experiment` and `write_artifacts` keep their behaviour.
- No look-ahead: anything used for target t must be known by `forecast_dates[t]`. Truth is never imputed.
- Randomness only via `np.random.SeedSequence(root_seed)`.
- Hierarchy: row 0 is `us`, then the leaves in alphabetical order; quantile levels `(0.1, 0.5, 0.9)` on the last axis; territories are `as`, `gu`, `mp`, `pr`, `vi` (D25).
- Panel arms (D22): reconcilers `none`, `bottom_up`, `mint_shrink`, `mint_median`; controllers `raw`, `pi`; `pid_theta` is rejected for `design = "panel"`. Comparisons: `mint_shrink`, `bottom_up`, `mint_median` each against `none`.
- Scale (D23): `s = max(mean |y|, 1)` per node over the forecaster's evaluation window; `scaled_mean_width = mean_width / s`, `scaled_interval_score = interval_score / s`. `s` is fixed by the full window, not recomputed per resample.
- Seeds (D24): `SeedSequence(root_seed).spawn(n_forecasters + 1)`. Forecaster k (configuration order) spawns one child per block length; the last child draws the cross-forecaster summary. Holm within (block, forecaster, level, controller); the primary is unadjusted.
- Primary (D20): forecaster `COVIDhub-4_week_ensemble`, level `leaf_mean`, controller `pi`, lr 0.1, `mint_shrink − none`, metric `coverage_deviation`, first block (8).
- Lint: `uv run ruff check src scripts tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py` and `uv run ruff format --check` on the same paths.
- Tests: `uv run pytest` (unit and e2e) must be green with pristine output. `uv run pytest -o addopts='' -m integration` needs `data/deaths.csv` (SHA-256 `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`).
- Modules stay at 150–300 lines; split above about 400.
- Conventional Commits, local only: never push, never `--no-verify`. Dependencies only via `uv add` (none are expected).

## Cost routing

| Task | Implementer | Reviewer | Why |
|---|---|---|---|
| 1 evaluate_batch + floor | haiku | sonnet | Complete code given |
| 2 mint_median | haiku | sonnet | Complete code given |
| 3 v1 diagnostics + report | opus | sonnet | Research analysis and writing judgment |
| 4 data split + panel fixtures | sonnet | sonnet | Refactor that must keep behaviour |
| 5 eligibility | sonnet | sonnet | Rule logic, real-data pin |
| 6 panel | sonnet | sonnet | Multi-module integration |
| 7 report figures | sonnet | sonnet | Figure judgment (dataviz) |
| 8 run dispatch + e2e | sonnet | sonnet | Wiring and manifest |
| 9 config, anchor, docs | sonnet | sonnet | Real-data regression anchor |
| Final review | opus | — | Whole branch |

## File map

```
src/hcp/evaluate.py          + evaluate_batch, coverage_deviation_floor           (Task 1)
src/hcp/reconcile.py         + mint_median, COVARIANCE_METHODS                    (Task 2)
scripts/__init__.py, scripts/analysis/__init__.py, scripts/analysis/us_hier_v2.py (Task 3)
docs/analysis/us-hier-v2.md, docs/analysis/us-hier-v2-diagnostics.json,
docs/analysis/figures/us-hier-v2-*.pdf                                            (Task 3)
src/hcp/data.py              read_forecasts, build_bundle, build_bundles, TERRITORIES (Task 4)
tests/helpers.py, tests/conftest.py   + panel_rows, panel_csv, panel_frame        (Task 4)
src/hcp/eligibility.py       EligibilityRule, eligibility, require_eligible       (Task 5)
src/hcp/panel.py             PanelConfig, ForecasterArms, PanelResult, run_panel  (Task 6)
src/hcp/report.py            + panel_forest_figure, leaf_contrast_figure, render_panel_figures (Task 7)
src/hcp/run.py               RunConfig panel fields, _manifest, write_panel_artifacts, dispatch (Task 8)
configs/panel_leaves.toml, tests/integration/test_panel_real.py, docs           (Task 9)
```

---

### Task 1: Vectorized evaluation and the endpoint noise floor

**Files:**
- Modify: `src/hcp/evaluate.py`
- Test: `tests/unit/test_evaluate.py`

**Interfaces:**
- Consumes: the existing `coverage_indicator`, `interval_score` and `METRIC_NAMES` in `src/hcp/evaluate.py`.
- Produces:
  - `evaluate_batch(y, lo, hi, *, alpha: float, window: int) -> dict[str, np.ndarray]`. Same keys and order as `METRIC_NAMES`, computed over the last axis; it returns arrays shaped like the leading axes.
  - `coverage_deviation_floor(n_weeks: int, *, alpha: float, window: int, n_sim: int, rng: np.random.Generator) -> np.ndarray`, of shape `[n_sim]`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/test_evaluate.py`, and extend its import block with `coverage_deviation_floor, evaluate_batch`:

```python
def _random_arms(shape: tuple[int, ...], seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=shape)
    lo = y - rng.normal(0.5, 1.0, size=shape)
    hi = y + rng.normal(0.5, 1.0, size=shape)
    return y, lo, hi


def test_evaluate_batch_matches_evaluate_row_by_row() -> None:
    y, lo, hi = _random_arms((7, 40))
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=10)
    assert list(batch) == list(METRIC_NAMES)
    for row in range(7):
        single = evaluate(y[row], lo[row], hi[row], alpha=0.2, window=10)
        for name in METRIC_NAMES:
            np.testing.assert_allclose(batch[name][row], single[name], rtol=1e-12, atol=1e-12)


def test_evaluate_batch_keeps_leading_axes() -> None:
    y, lo, hi = _random_arms((3, 4, 30), seed=1)
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=5)
    assert all(value.shape == (3, 4) for value in batch.values())


def test_evaluate_batch_longest_miss_run_hand_values() -> None:
    hits = np.array([[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 0, 0]])
    y = np.zeros(hits.shape)
    lo = np.where(hits == 1, -1.0, 1.0)
    hi = np.ones(hits.shape)
    batch = evaluate_batch(y, lo, hi, alpha=0.2, window=2)
    np.testing.assert_array_equal(batch["longest_miss_run"], [2.0, 5.0, 0.0, 3.0])


@pytest.mark.parametrize("window", [0, 6])
def test_evaluate_batch_window_out_of_range_raises(window: int) -> None:
    y, lo, hi = _random_arms((2, 5))
    with pytest.raises(ValueError, match="window"):
        evaluate_batch(y, lo, hi, alpha=0.2, window=window)


def test_coverage_deviation_floor_matches_binomial_expectation() -> None:
    floor = coverage_deviation_floor(
        112, alpha=0.2, window=10, n_sim=4000, rng=np.random.default_rng(0)
    )
    assert floor.shape == (4000,)
    # E|Bin(10, 0.8)/10 - 0.8| = 0.0966368 exactly.
    assert abs(floor.mean() - 0.0966368) < 0.003


def test_coverage_deviation_floor_is_reproducible() -> None:
    first = coverage_deviation_floor(50, alpha=0.2, window=5, n_sim=10, rng=np.random.default_rng(3))
    again = coverage_deviation_floor(50, alpha=0.2, window=5, n_sim=10, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(first, again)
```

- [ ] **Step 2: Run them and watch them fail.** `uv run pytest tests/unit/test_evaluate.py -q` should report `ImportError: cannot import name 'coverage_deviation_floor'`.

- [ ] **Step 3: Implement.** Append to `src/hcp/evaluate.py`. Leave the existing functions untouched, and update the second `ABOUTME` line to mention batching:

```python
def _coverage_deviation(hit: np.ndarray, alpha: float, window: int) -> np.ndarray:
    """Mean ``|rolling coverage - (1 - alpha)|`` over the last axis of ``hit``."""
    n = hit.shape[-1]
    if not 1 <= window <= n:
        raise ValueError(f"window must be in [1, {n}], got {window}")
    totals = np.cumsum(hit, axis=-1, dtype=float)
    padded = np.concatenate([np.zeros((*hit.shape[:-1], 1)), totals], axis=-1)
    rolling = (padded[..., window:] - padded[..., :-window]) / window
    return np.mean(np.abs(rolling - (1 - alpha)), axis=-1)


def evaluate_batch(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, window: int
) -> dict[str, np.ndarray]:
    """``evaluate`` over the last (week) axis of ``[..., n]`` arrays.

    Args:
        y: Truth ``[..., n]``.
        lo: Lower bounds, broadcastable to ``y``.
        hi: Upper bounds, broadcastable to ``y``.
        alpha: Miscoverage level; the target coverage is ``1 - alpha``.
        window: Rolling-coverage window in weeks.

    Returns:
        Every metric in ``METRIC_NAMES`` order, each shaped like the leading axes.

    Raises:
        ValueError: If ``window`` is not in ``[1, n]``.
    """
    y, lo, hi = np.broadcast_arrays(y, lo, hi)
    hit = coverage_indicator(y, lo, hi)
    positions = np.arange(hit.shape[-1])
    last_hit = np.maximum.accumulate(np.where(hit, positions, -1), axis=-1)
    width = hi - lo
    return {
        "coverage_deviation": _coverage_deviation(hit, alpha, window),
        "longest_miss_run": np.where(hit, 0, positions - last_hit).max(axis=-1).astype(float),
        "marginal_coverage": hit.mean(axis=-1),
        "mean_width": width.mean(axis=-1),
        "median_width": np.median(width, axis=-1),
        "interval_score": interval_score(y, lo, hi, alpha).mean(axis=-1),
    }


def coverage_deviation_floor(
    n_weeks: int, *, alpha: float, window: int, n_sim: int, rng: np.random.Generator
) -> np.ndarray:
    """Coverage deviation of ``n_sim`` series covered i.i.d. with probability ``1 - alpha``.

    This is what a calibrated interval with independent misses still scores on the
    primary endpoint (D19): its noise floor.

    Returns:
        Simulated coverage deviations ``[n_sim]``.
    """
    hits = rng.random((n_sim, n_weeks)) < 1 - alpha
    return _coverage_deviation(hits, alpha, window)
```

- [ ] **Step 4: Run the tests and watch them pass.** `uv run pytest tests/unit/test_evaluate.py -q` should report everything passed with no warnings. Then run the lint commands from Global Constraints.
- [ ] **Step 5: Commit.** `git add src/hcp/evaluate.py tests/unit/test_evaluate.py && git commit -m "feat(evaluate): add batched metrics and the coverage-deviation noise floor"`

---

### Task 2: `mint_median` reconciler (location-only MinT, D26)

**Files:**
- Modify: `src/hcp/reconcile.py`
- Test: `tests/unit/test_reconcile.py`

**Interfaces:**
- Consumes: `_mint_shrink`, `RECONCILERS`, `reconcile`, `reconcile_bundle` and `summing_matrix` in `src/hcp/reconcile.py`.
- Produces:
  - `RECONCILERS["mint_median"]`;
  - `COVARIANCE_METHODS: frozenset[str] = frozenset({"mint_shrink", "mint_median"})`;
  - `reconcile_bundle(bundle, "mint_median", warmup)`, which returns a `Reconciled` whose NaN pattern and shrinkage match `mint_shrink`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/test_reconcile.py`. Update `test_registry_names` (line 34) so the expected set includes `"mint_median"`; read it first and add only that name.

```python
def _spd(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


@pytest.mark.parametrize("seed", range(5))
def test_mint_median_medians_are_coherent_and_spreads_kept(seed: int) -> None:
    rng = np.random.default_rng(seed)
    S = summing_matrix(4)
    median = rng.uniform(10, 100, size=5)
    base = np.stack([median - rng.uniform(1, 5, 5), median, median + rng.uniform(1, 5, 5)], axis=1)
    W = _spd(5, rng)
    out = reconcile(base, S, "mint_median", covariance=W)
    np.testing.assert_allclose(S @ out[1:, 1], out[:, 1])
    np.testing.assert_allclose(out - out[:, [1]], base - base[:, [1]])
    np.testing.assert_allclose(out[:, 1], reconcile(base, S, "mint_shrink", covariance=W)[:, 1])


def test_mint_median_without_covariance_raises() -> None:
    with pytest.raises(ValueError, match="covariance"):
        reconcile(np.ones((4, 3)), summing_matrix(3), "mint_median")


def test_mint_median_bundle_matches_mint_definition_window(hier_bundle) -> None:
    median = reconcile_bundle(hier_bundle, "mint_median", WARMUP)
    shrink = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    np.testing.assert_array_equal(np.isnan(median.q), np.isnan(shrink.q))
    np.testing.assert_array_equal(median.shrinkage, shrink.shrinkage)
    assert median.crossings == 0
    defined = np.isfinite(median.q).all(axis=(0, 2))
    np.testing.assert_allclose(
        median.q[:, defined, 2] - median.q[:, defined, 0],
        hier_bundle.q[:, defined, 2] - hier_bundle.q[:, defined, 0],
    )


def test_mint_median_is_prefix_invariant(hier_bundle) -> None:
    full = reconcile_bundle(hier_bundle, "mint_median", WARMUP)
    part = reconcile_bundle(_truncate(hier_bundle, 15), "mint_median", WARMUP)
    np.testing.assert_array_equal(part.q, full.q[:, :15])
```

- [ ] **Step 2: Run them and watch them fail.** `uv run pytest tests/unit/test_reconcile.py -q` should fail with `ValueError: unknown reconciliation method: mint_median`, plus the registry test.

- [ ] **Step 3: Implement.** In `src/hcp/reconcile.py`:

  1. After `_mint_shrink`, add:

     ```python
     def _mint_median(
         base: np.ndarray, S: np.ndarray, covariance: np.ndarray | None
     ) -> np.ndarray:
         shift = _mint_shrink(base, S, covariance)[:, 1] - base[:, 1]
         return base + shift[:, None]
     ```

  2. Add `"mint_median": _mint_median,` to `RECONCILERS`. Below the dict, add:

     ```python
     COVARIANCE_METHODS = frozenset({"mint_shrink", "mint_median"})
     ```

  3. In `reconcile_bundle`, replace `if method == "mint_shrink":` with `if method in COVARIANCE_METHODS:`.

  4. Append to the module docstring: `` ``mint_median`` (D26) moves every level of a node by that node's MinT median shift, so medians are coherent and the base spread is kept. `` In `reconcile`'s `Raises:`, change `` ``mint_shrink`` without a covariance `` to `a MinT method without a covariance`.

- [ ] **Step 4: Run the tests and watch them pass.** `uv run pytest -q` should be all green, and lint clean.
- [ ] **Step 5: Commit.** `git add src/hcp/reconcile.py tests/unit/test_reconcile.py && git commit -m "feat(reconcile): add location-only mint_median reconciler (D26)"`

---

### Task 3: Phase 1 post-hoc diagnostics script and report (D19)

Use the `analyze-results` workflow (`results-analysis`, then `results-report`) for the report's standards. Everything is written under `docs/analysis/`; never write into `results/`.

**Files:**
- Create: `scripts/__init__.py` (the two ABOUTME lines only), `scripts/analysis/__init__.py` (the same), and `scripts/analysis/us_hier_v2.py`
- Create (generated): `docs/analysis/us-hier-v2-diagnostics.json`, `docs/analysis/figures/us-hier-v2-floor.pdf`, `docs/analysis/figures/us-hier-v2-warm-start.pdf`
- Create (written): `docs/analysis/us-hier-v2.md`
- Test: `tests/e2e/test_analysis_us_hier.py`

**Interfaces:**
- Consumes (Tasks 1 and 2):
  - `hcp.evaluate.coverage_deviation_floor` and `evaluate`;
  - `hcp.reconcile.reconcile_bundle(bundle, "mint_median", warmup)` and `evaluation_start`;
  - `hcp.conformal.conformalize` and `ControllerConfig`;
  - `hcp.data.load_hierarchy`;
  - `hcp.run.RunConfig`, rebuilt from `manifest["config"]` with `RunConfig(**manifest["config"])`.
- Produces:
  - `diagnose(bundle: HierarchyBundle, cfg: RunConfig, *, n_sim: int, seed: int) -> dict`;
  - `main(argv: list[str] | None = None) -> int`, with CLI `uv run python -m scripts.analysis.us_hier_v2 --run-dir results/us-hier-v2 [--data PATH] [--out docs/analysis] [--n-sim 10000]`.

The `diagnose` output, with metric dicts from `evaluate` on the evaluation window:

```
{"run_id": str, "n_weeks": int, "evaluation_start": "YYYY-MM-DD",
 "floor": {"n_sim": int, "mean": float, "p2.5": float, "p97.5": float},
 "location_scale": {reconciler: {**evaluate(raw interval), "median_mae": float}}
                   for reconciler in ("none", "bottom_up", "mint_shrink", "mint_median"),
 "warm_start": {reconciler: {"cold": evaluate(...), "warm": evaluate(...)}}
               for reconciler in ("none", "bottom_up"), PI at cfg.lr}
```

- **Cold** means `conformalize` on the window only, as `run_experiment` does.
- **Warm** means `conformalize` on every week from 0, then sliced to `[t0:]`. Weeks before `t0` are finite for `none` and `bottom_up`.
- `seed` is `cfg.root_seed`.

`main` reads `<run-dir>/manifest.json` and builds `RunConfig(**manifest["config"])`. `--data` overrides `data_path`. It loads the bundle with the manifest's `input_sha256`, and writes the JSON (indent 2) and two figures named after the run_id:

1. **floor:** a histogram of the simulated floor, with vertical lines at the observed `coverage_deviation` of the `raw` arm and of `pi` at `cfg.lr`, both read from `<run-dir>/metrics.parquet` for every reconciler.
2. **warm-start:** the PI offsets `q_lo − lo` and `hi − q_hi` over the window for `none`, cold vs warm.

Follow the `dataviz` skill, with matplotlib's OO API (`Figure` and `FigureCanvasAgg`, as in `src/hcp/report.py`).

- [ ] **Step 1: Write the failing e2e test** `tests/e2e/test_analysis_us_hier.py`. With two ABOUTME lines, it:
  - builds the synthetic US run with `hcp.run.main` and the same config values as `tests/e2e/test_run_cli.py::_config`, using `"controllers": ["raw", "pi"]`;
  - calls `scripts.analysis.us_hier_v2.main(["--run-dir", str(run_dir), "--out", str(tmp_path / "analysis"), "--n-sim", "200"])` and asserts it returns `0`;
  - loads the JSON and asserts the top-level keys above, the four `location_scale` reconcilers, and `0 < floor["mean"] < 1`;
  - asserts that `location_scale["mint_median"]["mean_width"] == pytest.approx(location_scale["none"]["mean_width"])`;
  - asserts that both PDFs exist and are non-empty.
- [ ] **Step 2: Run it and watch it fail** with `ModuleNotFoundError: scripts`.
- [ ] **Step 3: Implement** `scripts/analysis/us_hier_v2.py`, with two ABOUTME lines, type hints and docstrings. Keep it under 200 lines. Add `scripts` to the lint paths in `docs/development.md`.
- [ ] **Step 4: Run the test and watch it pass**, then lint clean.
- [ ] **Step 5: Run on real data.** `uv run python -m scripts.analysis.us_hier_v2 --run-dir results/us-hier-v2`. Check the expected values from D19 (tolerance 1e-3, floor mean ±0.003):
  - raw deviation: none 0.151, bottom_up 0.159, mint_shrink 0.154, mint_median 0.183;
  - raw mean width: 4839, 6806 and 5994 for none, bottom_up and mint_shrink;
  - median MAE: none ≈ 1320, mint_shrink ≈ 1300;
  - warm PI on none: deviation 0.151, mean width ≈ 6544;
  - floor mean ≈ 0.0966.

  If any differ, stop and report `DONE_WITH_CONCERNS` with the numbers.
- [ ] **Step 6: Write `docs/analysis/us-hier-v2.md`** following `results-report` standards. Sections:
  1. Question and pre-registered design (D1–D12, short).
  2. Primary result, with its CI and p-value from `contrasts.parquet` (block 8).
  3. Secondary results: a Holm-adjusted table for PI and raw.
  4. Descriptive metrics table (lr grid).
  5. Post-hoc diagnostics, clearly labelled exploratory: floor, location/scale, warm start, with the figures.
  6. Interpretation and threats to validity (D15–D17 caveats, single series, final revised truth).
  7. Implications for Phase 2 (link the spec).

  Plain academic prose, no inflated wording. Every number traceable to a parquet or JSON file.
- [ ] **Step 7: Commit** in two commits:
  - `git add scripts tests/e2e/test_analysis_us_hier.py docs/development.md && git commit -m "feat(analysis): add post-hoc diagnostics script for US-design runs"`
  - `git add docs/analysis && git commit -m "docs(analysis): report us-hier-v2 results and post-hoc diagnostics (D19)"`

---

### Task 4: Multi-forecaster data loading and panel fixtures

**Files:**
- Modify: `src/hcp/data.py`, `tests/helpers.py`, `tests/conftest.py`
- Test: `tests/unit/test_data.py`

**Interfaces:**
- Produces:
  - `TERRITORIES: tuple[str, ...] = ("as", "gu", "mp", "pr", "vi")`;
  - `read_forecasts(path: Path, expected_sha256: str | None = None) -> pd.DataFrame`, which returns only `ahead == AHEAD` rows;
  - `build_bundle(frame: pd.DataFrame, forecaster: str = FORECASTER) -> HierarchyBundle`;
  - `build_bundles(frame: pd.DataFrame) -> dict[str, HierarchyBundle | None]`, keyed by sorted forecaster name, with `None` where `build_bundle` raises `ValueError`;
  - `load_hierarchy(path, expected_sha256=None)`, unchanged behaviour, now `build_bundle(read_forecasts(path, expected_sha256))`.
- Test helpers:
  - `ALT_FORECASTER = "synth-alt"`, `DUP_FORECASTER = "synth-dup"`, `SHORT_FORECASTER = "synth-short"`, `SHORT_WEEKS = 8`;
  - `panel_rows(seed: int = 0) -> list[dict]`;
  - fixtures `panel_csv(tmp_path) -> Path` and `panel_frame(panel_csv) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/test_data.py`, importing `read_forecasts, build_bundle, build_bundles, TERRITORIES, FORECASTER, load_hierarchy` and the helper constants:

```python
def _assert_same_bundle(a, b) -> None:
    assert a.nodes == b.nodes
    for name in ("dates", "forecast_dates", "y", "q", "imputed"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_load_hierarchy_equals_build_bundle_of_read_forecasts(hier_csv: Path) -> None:
    _assert_same_bundle(load_hierarchy(hier_csv), build_bundle(read_forecasts(hier_csv)))


def test_read_forecasts_keeps_only_ahead_4(hier_csv: Path) -> None:
    assert set(read_forecasts(hier_csv)["ahead"]) == {4}


def test_read_forecasts_checks_sha256(hier_csv: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        read_forecasts(hier_csv, "0" * 64)


def test_build_bundle_selects_forecaster(panel_frame) -> None:
    base = build_bundle(panel_frame)
    alt = build_bundle(panel_frame, ALT_FORECASTER)
    np.testing.assert_array_equal(alt.dates, base.dates)
    np.testing.assert_allclose(alt.q, 1.1 * base.q)
    np.testing.assert_array_equal(alt.y, base.y)


def test_build_bundle_unknown_forecaster_raises(panel_frame) -> None:
    with pytest.raises(ValueError, match="no rows for forecaster 'nope'"):
        build_bundle(panel_frame, "nope")


def test_build_bundles_maps_every_forecaster_and_none_on_failure(panel_frame) -> None:
    bundles = build_bundles(panel_frame)
    assert list(bundles) == sorted(panel_frame["forecaster"].unique())
    assert bundles[SHORT_FORECASTER] is None or len(bundles[SHORT_FORECASTER].dates) <= SHORT_WEEKS
    _assert_same_bundle(bundles[DUP_FORECASTER], bundles[FORECASTER])


def test_territories_constant() -> None:
    assert TERRITORIES == ("as", "gu", "mp", "pr", "vi")
```

- [ ] **Step 2: Add the helpers and fixtures** so that collection works.

  In `tests/helpers.py`, below `synthetic_rows`, add the constants and:

  ```python
  def panel_rows(seed: int = 0) -> list[dict]:
      """``synthetic_rows`` plus three panel forecasters.

      ``ALT_FORECASTER`` has every quantile x 1.1, ``DUP_FORECASTER`` copies
      ``FORECASTER`` exactly, and ``SHORT_FORECASTER`` has only the last
      ``SHORT_WEEKS`` target weeks.
      """
      rows = synthetic_rows(seed)
      cutoff = START + pd.Timedelta(weeks=N_WEEKS - SHORT_WEEKS)
      extra = []
      for row in rows:
          if row["forecaster"] != FORECASTER or row["ahead"] != 4:
              continue
          extra.append({**row, "forecaster": ALT_FORECASTER, **{c: row[c] * 1.1 for c in QUANTILE_COLUMNS}})
          extra.append({**row, "forecaster": DUP_FORECASTER})
          if pd.Timestamp(row["target_end_date"]) >= cutoff:
              extra.append({**row, "forecaster": SHORT_FORECASTER})
      return rows + extra
  ```

  Update the helpers' second ABOUTME line to mention the panel forecasters.

  In `tests/conftest.py`, add `panel_csv` (same pattern as `hier_csv`, file `panel.csv`, `panel_rows()`) and `panel_frame` (`read_forecasts(panel_csv)`).

- [ ] **Step 3: Run the tests and watch them fail.** `uv run pytest tests/unit/test_data.py -q` should report `ImportError: cannot import name 'read_forecasts'`.

- [ ] **Step 4: Implement.** In `src/hcp/data.py`:
  - add `TERRITORIES` after `ROOT`;
  - move the SHA-256 check and `pd.read_csv` into `read_forecasts`, which returns `frame[frame["ahead"] == AHEAD]`;
  - move the rest of the body into `build_bundle`. It first filters `frame[(frame["ahead"] == AHEAD) & (frame["forecaster"] == forecaster)]`, and raises `ValueError(f"no rows for forecaster {forecaster!r}")` if that is empty;
  - make `load_hierarchy` a one-line composition that keeps its docstring;
  - add:

    ```python
    def build_bundles(frame: pd.DataFrame) -> dict[str, HierarchyBundle | None]:
        """One bundle per forecaster in ``frame``, sorted by name; ``None`` where none can be built."""
        bundles: dict[str, HierarchyBundle | None] = {}
        for name in sorted(frame["forecaster"].unique()):
            try:
                bundles[name] = build_bundle(frame, name)
            except ValueError:
                bundles[name] = None
        return bundles
    ```

  Update the module docstring and ABOUTME to say "for any forecaster". Google docstrings with `Raises:` on the new public functions.

- [ ] **Step 5: Run the tests and watch them pass.** `uv run pytest -q` should be all green, including every pre-existing data test and the e2e; lint clean.
- [ ] **Step 6: Commit.** `git commit -am`, after adding the new files: `"refactor(data): read forecasts once and build a bundle per forecaster"`

---

### Task 5: Forecaster eligibility (D21)

**Files:**
- Create: `src/hcp/eligibility.py`
- Test: `tests/unit/test_eligibility.py`, `tests/integration/test_panel_real.py` (create it with the eligibility test only; Task 9 adds the anchor)

**Interfaces:**
- Consumes: `HierarchyBundle` and `build_bundles` (Task 4), and `hcp.reconcile.evaluation_start`.
- Produces:

```python
@dataclass(frozen=True)
class EligibilityRule:
    nodes: tuple[str, ...]
    mint_warmup: int
    min_eval_weeks: int = 60
    max_leaf_imputed: float = 0.05
    duplicate_share: float = 0.95

def evaluation_weeks(bundle: HierarchyBundle, mint_warmup: int) -> int  # len(dates) - evaluation_start; 0 if it raises
def duplicate_share(a: HierarchyBundle, b: HierarchyBundle) -> float   # share of shared (node, week) cells with all 3 quantiles np.isclose; 0.0 without shared weeks
def eligibility(bundles: Mapping[str, HierarchyBundle | None], rule: EligibilityRule) -> pd.DataFrame
def require_eligible(table: pd.DataFrame, forecasters: Sequence[str]) -> None  # ValueError("ineligible forecasters: [...]") listing unknown or ineligible names
```

`eligibility` returns one row per input key, in input order, with columns `forecaster`, `n_eval_weeks` (int), `leaf_imputed_share` (float, NaN for `None`), `eligible` (bool) and `reason` (str; `""` when eligible).

The checks run in this order; the first failure sets `reason`:

1. `None` gives `"no complete week"`.
2. `bundle.nodes != rule.nodes` gives `"nodes differ"`.
3. `n_eval_weeks < min_eval_weeks` gives `f"{n} evaluation weeks < {min}"`.
4. `leaf_imputed_share > max_leaf_imputed` gives `f"leaf imputation {share:.3f} > {max}"`, where `share = bundle.imputed[1:].mean()`.
5. Duplicates. Visit the forecasters that passed checks 1–4 in order of decreasing `n_eval_weeks`, ties by name. A forecaster whose `duplicate_share` against an already kept one is greater than `rule.duplicate_share` gets `f"duplicate of {kept}"`; otherwise it is kept.

- [ ] **Step 1: Write the failing unit tests** in `tests/unit/test_eligibility.py`, using `panel_frame` and `build_bundles`. Rule: `EligibilityRule(nodes=("us", "aa", "bb", "cc"), mint_warmup=5, min_eval_weeks=10)`. Assert:
  - `FORECASTER` and `ALT_FORECASTER` are eligible with reason `""`;
  - `DUP_FORECASTER` has reason `f"duplicate of {FORECASTER}"`, because of the tie on eval weeks and because `"COVIDhub-4_week_ensemble" < "synth-dup"`;
  - `SHORT_FORECASTER` is ineligible, with reason `"no complete week"` or one starting with `"0 evaluation weeks"`; assert it is ineligible and that the reason is one of those;
  - `nodes=("us", "aa", "bb", "cc", "dd")` makes every row `"nodes differ"` (or `"no complete week"` for `None`);
  - `max_leaf_imputed=0.01` makes `FORECASTER` ineligible, with a reason starting `"leaf imputation 0.0"` (the synthetic gap gives 1/78 ≈ 0.013);
  - `duplicate_share(b, b) == 1.0`, and `duplicate_share(base, alt) == 0.0`;
  - `require_eligible(table, [FORECASTER, ALT_FORECASTER])` passes, while `[DUP_FORECASTER]` and `["unknown"]` raise `ValueError` matching `"ineligible forecasters"`;
  - the table columns and the input order.
- [ ] **Step 2: Write the failing integration test** `tests/integration/test_panel_real.py`, marked as in `test_run_real.py`:

```python
D21_ELIGIBLE = [
    "CEID-Walk", "COVIDhub-4_week_ensemble", "COVIDhub-baseline", "COVIDhub-ensemble",
    "COVIDhub-trained_ensemble", "JHUAPL-Bucky", "RobertWalraven-ESG", "UMass-MechBayes",
]

def test_eligibility_reproduces_d21() -> None:
    bundles = build_bundles(read_forecasts(REAL_DATA, REAL_SHA256))
    rule = EligibilityRule(nodes=bundles[FORECASTER].nodes, mint_warmup=20)
    table = eligibility(bundles, rule).set_index("forecaster")
    assert sorted(table.index[table["eligible"]]) == D21_ELIGIBLE
    assert table.loc["COVIDhub_CDC-ensemble", "reason"] == "duplicate of COVIDhub-ensemble"
    assert table.loc["JHU_IDD-CovidSP", "reason"].startswith("leaf imputation")
```

- [ ] **Step 3: Run them and watch them fail** (ImportError).
- [ ] **Step 4: Implement** `src/hcp/eligibility.py`: two ABOUTME lines, a module docstring citing D21, no I/O, about 120 lines. For `duplicate_share`, take the intersection of `dates`, index each `q` with `np.isin`, and compute `np.isclose(qa, qb).all(axis=-1).mean()`.
- [ ] **Step 5: Run the tests and watch them pass.** `uv run pytest -q`, then `uv run pytest -o addopts='' -m integration tests/integration/test_panel_real.py -q` (this takes minutes; it reads 219 MB once). Lint clean.
- [ ] **Step 6: Commit.** `feat(eligibility): add the D21 forecaster eligibility rule`

---

### Task 6: Panel stage

**Files:**
- Create: `src/hcp/panel.py`
- Test: `tests/unit/test_panel.py`

**Interfaces:**
- Consumes:
  - `evaluate_batch`, `METRIC_NAMES` (Task 1);
  - `RECONCILERS` incl. `mint_median`, `reconcile_bundle`, `evaluation_start`;
  - `conformalize`, `ControllerConfig`;
  - `draw_blocks`, `summarize_diff`, `holm`;
  - `TERRITORIES` and `HierarchyBundle` (Task 4);
  - `RAW_LR` from `hcp.report` (as `run.py` does).
- Produces (later tasks rely on these exact names):

```python
LEVELS = ("leaf_mean", "us")
PANEL_METRICS = ("coverage_deviation", "longest_miss_run", "marginal_coverage",
                 "scaled_mean_width", "scaled_interval_score")
SUMMARY_METRICS = ("coverage_deviation", "scaled_interval_score")
COMPARISONS = (("mint_shrink", "none"), ("bottom_up", "none"), ("mint_median", "none"))
PRIMARY = {"level": "leaf_mean", "controller": "pi", "treatment": "mint_shrink",
           "control": "none", "metric": "coverage_deviation"}

@dataclass(frozen=True)
class PanelConfig:
    controller: ControllerConfig      # controller.lr is the contrast lr
    lr_grid: tuple[float, ...]
    reconcilers: tuple[str, ...]
    controllers: tuple[str, ...]
    mint_warmup: int
    rolling_window: int
    blocks: tuple[int, ...]
    n_boot: int
    root_seed: int
    primary_forecaster: str

@dataclass(frozen=True)
class ForecasterArms:
    weeks: np.ndarray                 # [n]
    nodes: tuple[str, ...]            # scored nodes, "us" first (D25)
    y: np.ndarray                     # [n_scored, n]
    q: dict[str, np.ndarray]          # reconciler -> [n_scored, n, 3]
    intervals: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]]  # (reconciler, controller, lr) -> (lo, hi) [n_scored, n]
    endpoint: np.ndarray              # bool [n_scored]: scored leaves not in TERRITORIES
    diagnostics: dict

@dataclass(frozen=True)
class PanelResult:
    intervals: pd.DataFrame     # forecaster, node, week, reconciler, controller, lr, y, q_lo, q_med, q_hi, lo, hi
    metrics: pd.DataFrame       # forecaster, node, reconciler, controller, lr, *METRIC_NAMES, scaled_mean_width, scaled_interval_score
    level_metrics: pd.DataFrame # forecaster, level, reconciler, controller, lr, *PANEL_METRICS
    contrasts: pd.DataFrame     # block, forecaster, level, controller, lr, treatment, control, metric, estimate, low, high, p_value, primary, p_holm
    summary: pd.DataFrame       # level, controller, lr, treatment, control, metric, n_forecasters, n_improved, mean_estimate, low, high
    diagnostics: dict           # forecaster -> ForecasterArms.diagnostics

def node_scale(y: np.ndarray) -> np.ndarray          # max(mean |y|, 1) over the last axis (D23)
def forecaster_arms(bundle: HierarchyBundle, cfg: PanelConfig) -> ForecasterArms
def forecaster_contrasts(arms: ForecasterArms, cfg: PanelConfig, seed: np.random.SeedSequence) -> pd.DataFrame
def summarize_forecasters(contrasts: pd.DataFrame, *, block: int, n_boot: int, rng: np.random.Generator) -> pd.DataFrame
def run_panel(bundles: Mapping[str, HierarchyBundle], cfg: PanelConfig) -> PanelResult
```

Reference implementation. Transcribe it, then fill in the frame builders, which are plain pandas:

```python
def node_scale(y: np.ndarray) -> np.ndarray:
    """Per-node scale ``max(mean |y|, 1)`` over the week axis (D23)."""
    return np.maximum(np.abs(y).mean(axis=-1), 1.0)


def _node_metrics(y, lo, hi, scale, *, alpha: float, window: int) -> dict[str, np.ndarray]:
    values = evaluate_batch(y, lo, hi, alpha=alpha, window=window)
    values["scaled_mean_width"] = values["mean_width"] / scale
    values["scaled_interval_score"] = values["interval_score"] / scale
    return values


def _levels(values: dict[str, np.ndarray], endpoint: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """``values[metric]`` has nodes on axis 0; returns level -> metric -> value(s)."""
    return {
        "leaf_mean": {m: values[m][endpoint].mean(axis=0) for m in PANEL_METRICS},
        "us": {m: values[m][0] for m in PANEL_METRICS},
    }


def forecaster_arms(bundle: HierarchyBundle, cfg: PanelConfig) -> ForecasterArms:
    """Reconcile and conformalize every scored node on the forecaster's evaluation window."""
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    window_y = bundle.y[:, t0:]
    scored = np.isfinite(window_y).all(axis=1)
    nodes = tuple(name for name, keep in zip(bundle.nodes, scored, strict=True) if keep)
    y = window_y[scored]
    endpoint = np.array([i > 0 and name not in TERRITORIES for i, name in enumerate(nodes)])
    q, intervals, crossings, root_crossings, shrinkage = {}, {}, {}, {}, {}
    for reconciler in cfg.reconcilers:
        rec = reconcile_bundle(bundle, reconciler, cfg.mint_warmup)
        crossings[reconciler] = rec.crossings
        root_crossings[reconciler] = rec.root_crossings
        if np.isfinite(rec.shrinkage[t0:]).all():
            shrinkage[reconciler] = float(np.mean(rec.shrinkage[t0:]))
        q[reconciler] = rec.q[scored, t0:, :]
        for controller in cfg.controllers:
            for lr in (RAW_LR,) if controller == "raw" else cfg.lr_grid:
                settings = replace(cfg.controller, lr=lr)
                bounds = [
                    conformalize(y[i], q[reconciler][i, :, 0], q[reconciler][i, :, 2], controller, settings)
                    for i in range(len(nodes))
                ]
                intervals[(reconciler, controller, lr)] = (
                    np.array([lo for lo, _ in bounds]),
                    np.array([hi for _, hi in bounds]),
                )
    diagnostics = {
        "n_weeks": int(y.shape[1]),
        "evaluation_start": str(pd.Timestamp(bundle.dates[t0]).date()),
        "imputed_cells": int(bundle.imputed.sum()),
        "leaf_imputed_share": float(bundle.imputed[1:].mean()),
        "unscored_nodes": [n for n, keep in zip(bundle.nodes, scored, strict=True) if not keep],
        "crossings": crossings,
        "root_crossings": root_crossings,
        "mean_shrinkage": shrinkage,
    }
    return ForecasterArms(bundle.dates[t0:], nodes, y, q, intervals, endpoint, diagnostics)


def forecaster_contrasts(arms: ForecasterArms, cfg: PanelConfig, seed: np.random.SeedSequence) -> pd.DataFrame:
    """Paired block-bootstrap contrasts at ``cfg.controller.lr``; one week-index draw per block is shared by every node."""
    alpha, window = cfg.controller.alpha, cfg.rolling_window
    scale = node_scale(arms.y)
    rows = []
    for block, child in zip(cfg.blocks, seed.spawn(len(cfg.blocks)), strict=True):
        ix = draw_blocks(arms.y.shape[1], block, cfg.n_boot, np.random.default_rng(child))
        for controller in cfg.controllers:
            lr = RAW_LR if controller == "raw" else cfg.controller.lr
            full, boot = {}, {}
            for reconciler in cfg.reconcilers:
                lo, hi = arms.intervals[(reconciler, controller, lr)]
                full[reconciler] = _levels(_node_metrics(arms.y, lo, hi, scale, alpha=alpha, window=window), arms.endpoint)
                per_node = [
                    _node_metrics(arms.y[i][ix], lo[i][ix], hi[i][ix], scale[i], alpha=alpha, window=window)
                    for i in range(len(arms.nodes))
                ]
                stacked = {m: np.stack([values[m] for values in per_node]) for m in PANEL_METRICS}
                boot[reconciler] = _levels(stacked, arms.endpoint)
            for treatment, control in COMPARISONS:
                if treatment not in cfg.reconcilers:
                    continue
                for level in LEVELS:
                    for metric in PANEL_METRICS:
                        contrast = summarize_diff(
                            float(full[treatment][level][metric] - full[control][level][metric]),
                            boot[treatment][level][metric] - boot[control][level][metric],
                        )
                        rows.append({"block": block, "level": level, "controller": controller, "lr": lr,
                                     "treatment": treatment, "control": control, "metric": metric,
                                     **asdict(contrast)})
    return pd.DataFrame(rows)


def summarize_forecasters(contrasts: pd.DataFrame, *, block: int, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Descriptive cross-forecaster summary (D24); one forecaster resample draw is shared by every row."""
    rows = contrasts[(contrasts["block"] == block) & contrasts["metric"].isin(SUMMARY_METRICS)]
    forecasters = list(pd.unique(rows["forecaster"]))
    draws = rng.integers(0, len(forecasters), size=(n_boot, len(forecasters)))
    keys = ["level", "controller", "lr", "treatment", "control", "metric"]
    out = []
    for key, group in rows.groupby(keys, sort=False):
        estimates = group.set_index("forecaster").loc[forecasters, "estimate"].to_numpy()
        low, high = np.quantile(estimates[draws].mean(axis=1), [0.025, 0.975])
        out.append({**dict(zip(keys, key, strict=True)), "n_forecasters": len(forecasters),
                    "n_improved": int((estimates < 0).sum()), "mean_estimate": float(estimates.mean()),
                    "low": float(low), "high": float(high)})
    return pd.DataFrame(out)
```

`run_panel` does the following:
1. Raises `ValueError` when `cfg.primary_forecaster not in bundles`.
2. Spawns `seeds = SeedSequence(cfg.root_seed).spawn(len(bundles) + 1)`.
3. For each forecaster (mapping order) with `seeds[k]`, builds the arms, the interval, metric and level-metric frames, and its contrasts (adding a `forecaster` column).
4. Concatenates.
5. Flags `primary`: `forecaster == primary_forecaster`, `block == cfg.blocks[0]`, `lr == cfg.controller.lr`, and every `PRIMARY` key/value.
6. Sets `p_holm = p_value`, then, on the non-primary rows, applies `groupby(["block", "forecaster", "level", "controller"])["p_value"].transform(lambda p: holm(p.to_numpy()))`, as `run._contrasts` does.
7. Builds the summary with `summarize_forecasters(contrasts, block=cfg.blocks[0], n_boot=cfg.n_boot, rng=np.random.default_rng(seeds[-1]))`.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_panel.py`. Use `panel_frame` and `build_bundle` for `FORECASTER` and `ALT_FORECASTER`, with `SMALL = PanelConfig(controller=ControllerConfig(lr=0.1, T_burnin=2), lr_grid=(0.1,), reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"), controllers=("raw", "pi"), mint_warmup=5, rolling_window=5, blocks=(4,), n_boot=10, root_seed=1, primary_forecaster=FORECASTER)`. The cases:
  1. `forecaster_arms`:
     - `nodes == ("us", "aa", "bb", "cc")` and `endpoint == [False, True, True, True]`;
     - with nodes renamed to `("us", "aa", "bb", "pr")` via `dataclasses.replace`, the endpoint is `[False, True, True, False]`;
     - a NaN injected into leaf `cc`'s truth inside the window (use `replace` with a copied `y`) drops `cc` from `nodes`, and `diagnostics["unscored_nodes"] == ["cc"]`.
  2. `("none", "raw", 0.0)` intervals equal the bundle's base q0.1/q0.9 on the window.
  3. **US anchor (synthetic):** with `RunConfig` set to `SMALL`'s values (`tests/unit/test_run.py::SMALL`, controllers `raw` and `pi`, v1 reconcilers), the US-node rows of `run_panel({FORECASTER: bundle}, cfg).metrics` equal `run_experiment(...).metrics` (`np.testing.assert_allclose`, rtol 1e-12) for every `METRIC_NAMES` column and every (reconciler, controller, lr).
  4. `mint_median` raw widths equal `none` raw widths at every node.
  5. Scaled metrics equal `mean_width / max(mean |y|, 1)` (the same for interval score) per node.
  6. The `leaf_mean` contrast `estimate` equals the mean over endpoint nodes of the per-node metric difference from `metrics`, for `coverage_deviation`, `pi`, `mint_shrink − none`.
  7. There is exactly one `primary` row, with `p_holm == p_value`, on `FORECASTER` / `leaf_mean`.
  8. With two forecasters, the contrasts are deterministic for the same `root_seed` and differ for another seed.
  9. `summarize_forecasters` on a hand-made contrasts frame with three forecasters, estimates `[-1.0, 2.0, -3.0]` for one key: `n_forecasters == 3`, `n_improved == 2`, `mean_estimate == pytest.approx(-2/3)`, and `low <= mean <= high`.
  10. A missing primary forecaster raises `ValueError`.
- [ ] **Step 2: Run them and watch them fail** (ImportError).
- [ ] **Step 3: Implement** `src/hcp/panel.py`: two ABOUTME lines, a module docstring citing D20–D26, Google docstrings on public functions, and no I/O. Target ≤ 320 lines; if it goes over, report `DONE_WITH_CONCERNS` rather than splitting.
- [ ] **Step 4: Run the tests and watch them pass.** `uv run pytest -q` should be green with pristine output (core's tqdm bars are on stderr and captured by pytest); lint clean.
- [ ] **Step 5: Commit.** `feat(panel): add the leaves x forecasters panel stage (D20-D26)`

---

### Task 7: Panel figures

Follow the `dataviz` skill.

**Files:**
- Modify: `src/hcp/report.py`
- Test: `tests/unit/test_report.py`

**Interfaces:**
- Consumes: `TERRITORIES` from `hcp.data`. Do not import `hcp.panel`, which imports `hcp.report`. Also consumes the `PanelResult.metrics` and `.contrasts` column schemas (Task 6).
- Produces:
  - `RECONCILER_COLORS["mint_median"] = "#059669"`;
  - `panel_forest_figure(contrasts: pd.DataFrame, *, level: str, controller: str, lr: float, block: int, primary_forecaster: str) -> Figure`;
  - `leaf_contrast_figure(metrics: pd.DataFrame, *, forecaster: str, controller: str, lr: float, treatment: str = "mint_shrink", control: str = "none", metric: str = "coverage_deviation") -> Figure`;
  - `render_panel_figures(metrics: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path, *, lr: float, block: int, primary_forecaster: str) -> list[Path]`.

Figure requirements:
- **`panel_forest_figure`**:
  - uses only the `coverage_deviation` rows at the given block, level, controller and lr;
  - one row group per forecaster in first-appearance order, with one point and 95% CI per treatment, colored by `RECONCILER_COLORS` with a legend;
  - a dashed zero line;
  - the primary forecaster's tick label in bold;
  - x label `Δ coverage deviation vs none (95% block-bootstrap CI, block={block})`.
- **`leaf_contrast_figure`**:
  - per-node `metric[treatment] − metric[control]` for the forecaster, controller and lr;
  - excludes `us` and `TERRITORIES`;
  - sorted ascending as a horizontal dot plot with node labels and a dashed zero line.
- **`render_panel_figures`** writes, for each controller in `contrasts["controller"].unique()` (lr is `RAW_LR` for `raw`, else `lr`):
  - `panel_forest_leaf_mean_{controller}.pdf` and `panel_forest_us_{controller}.pdf`;
  - `leaf_contrasts_{controller}.pdf` for `primary_forecaster`.

- [ ] **Step 1: Write the failing tests.** Hand-made `metrics` and `contrasts` frames in the Task 6 schema: two forecasters, nodes `us`, `aa`, `bb`, `pr`, controllers `raw` (lr 0.0) and `pi` (lr 0.1), and the three comparisons. Assert:
  - `render_panel_figures` writes exactly the six expected names, all non-empty;
  - the y tick labels of `leaf_contrast_figure` are `{"aa", "bb"}`, with no `us` and no `pr`;
  - the forest figure's bold tick label is the primary forecaster's.
- [ ] **Step 2: Run them and watch them fail** (ImportError).
- [ ] **Step 3: Implement** using the existing `_figure` helper and the OO API. Update the ABOUTME lines. `render_figures` stays unchanged.
- [ ] **Step 4: Run the tests and watch them pass.** `uv run pytest -q` should be green with no matplotlib warnings; lint clean.
- [ ] **Step 5: Commit.** `feat(report): add per-forecaster forest and per-leaf contrast figures`

---

### Task 8: Run dispatch, panel artifacts and e2e

**Files:**
- Modify: `src/hcp/run.py`
- Test: `tests/unit/test_run.py`, `tests/e2e/test_run_panel_cli.py`

**Interfaces:**
- Consumes:
  - `read_forecasts`, `build_bundles`, `FORECASTER` (Task 4);
  - `EligibilityRule`, `eligibility`, `require_eligible` (Task 5);
  - `PanelConfig`, `PanelResult`, `run_panel`, `PRIMARY as PANEL_PRIMARY` (Task 6);
  - `render_panel_figures` (Task 7).
- Produces:
  - `RunConfig` fields, with defaults: `design: str = "us"`, `forecasters: tuple[str, ...] = (FORECASTER,)`, `min_eval_weeks: int = 60`, `max_leaf_imputed: float = 0.05`, `duplicate_share: float = 0.95`;
  - `panel_config(cfg: RunConfig) -> PanelConfig`;
  - `write_panel_artifacts(cfg, result: PanelResult, eligibility_table: pd.DataFrame, input_sha256: str) -> Path`.

Requirements:
1. **`RunConfig.__post_init__`**:
   - add `"forecasters"` to the tuple conversion;
   - raise `ValueError` when `design` is not in `("us", "panel")`;
   - for `"us"`, raise when `forecasters != (FORECASTER,)`;
   - for `"panel"`, raise when `"pid_theta"` is in `controllers` (message mentions D22), when `FORECASTER` is not in `forecasters`, or when `forecasters` has duplicates.
2. **Refactor** the manifest dict in `write_artifacts` into `_manifest(cfg, input_sha256: str, *, primary: dict, seed: dict, diagnostics: dict) -> dict`. `write_artifacts` passes `primary=PRIMARY` and `seed={"root_seed": cfg.root_seed, "block_children": len(cfg.blocks)}`, so the US manifest keeps its keys and values. The keys also gain the new config fields through `asdict`, which is expected.
3. **`panel_config`** maps every field. `controller` is `_controller_config(cfg, cfg.lr)`; `primary_forecaster` is `FORECASTER`.
4. **`write_panel_artifacts`**:
   - `mkdir(exist_ok=False)`;
   - writes `intervals`, `metrics`, `level_metrics`, `contrasts`, `summary` and `eligibility` `.parquet` files;
   - calls `render_panel_figures(result.metrics, result.contrasts, out / "figures", lr=cfg.lr, block=cfg.blocks[0], primary_forecaster=FORECASTER)`;
   - writes the manifest from `_manifest(..., primary={"forecaster": FORECASTER, **PANEL_PRIMARY}, seed={"root_seed": cfg.root_seed, "forecaster_children": len(cfg.forecasters) + 1, "block_children": len(cfg.blocks)}, diagnostics=result.diagnostics)`, plus `"forecasters": list(cfg.forecasters)` and `"eligibility": eligibility_table.to_dict(orient="records")`.
5. **`main`**:
   - after the exists check, branches on `cfg.design`;
   - the panel branch reads the frame with `read_forecasts`, builds the candidates with `build_bundles`, and raises `ValueError` if `candidates[FORECASTER]` is `None`;
   - it then builds the rule `EligibilityRule(nodes=candidates[FORECASTER].nodes, mint_warmup=cfg.mint_warmup, min_eval_weeks=..., max_leaf_imputed=..., duplicate_share=...)`, computes `table = eligibility(candidates, rule)`, and calls `require_eligible(table, cfg.forecasters)`;
   - finally it runs `run_panel({name: candidates[name] for name in cfg.forecasters}, panel_config(cfg))` and `write_panel_artifacts`;
   - the US branch is unchanged.
6. Update the module docstring for the two designs. `run.py` should stay ≤ ~410 lines; report a concern if it goes over.

- [ ] **Step 1: Write the failing unit tests** in `tests/unit/test_run.py`:
  - extend `test_invalid_config_raises` with these parametrized overrides: `{"design": "both"}`; `{"forecasters": ("x",)}` (us design); `{"design": "panel", "controllers": ("raw", "pi", "pid_theta"), "forecasters": (FORECASTER,)}`; `{"design": "panel", "controllers": ("raw", "pi"), "forecasters": ("x",)}`; and `{"design": "panel", "controllers": ("raw", "pi"), "forecasters": (FORECASTER, FORECASTER)}`;
  - add `test_panel_config_maps_fields`;
  - add `test_us_design_manifest_keys_unchanged`, which compares the `_manifest` keys with the keys of the existing e2e expectations (read `tests/e2e/test_run_cli.py` first).
- [ ] **Step 2: Write the failing e2e test** `tests/e2e/test_run_panel_cli.py`, with two ABOUTME lines. Config on `panel_csv`:
  - `design = "panel"`, `forecasters = [FORECASTER, ALT_FORECASTER]`;
  - `controllers = ["raw", "pi"]`, `reconcilers = ["none", "bottom_up", "mint_shrink", "mint_median"]`;
  - `lr = 0.1`, `lr_grid = [0.1, 0.5]`, `T_burnin = 2`;
  - `mint_warmup = 5`, `rolling_window = 5`, `blocks = [4]`, `n_boot = 20`;
  - `min_eval_weeks = 10`, plus `expected_sha256`.

  Assert:
  - every artifact exists: 6 parquets, `manifest.json`, and the 6 figures named in Task 7;
  - the manifest has `design == "panel"`, the `forecasters` list, an eligibility record for `synth-dup` whose reason starts `"duplicate of"`, and `primary["forecaster"] == FORECASTER`;
  - `contrasts["primary"].sum() == 1`;
  - `summary["n_forecasters"]` is all 2;
  - a config with `forecasters = [FORECASTER, DUP_FORECASTER]` raises `ValueError` matching `"ineligible"`;
  - a repeated run_id raises `FileExistsError`.
- [ ] **Step 3: Run them and watch them fail.**
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run the tests and watch them pass.** `uv run pytest -q` should be all green, including the untouched `tests/e2e/test_run_cli.py`; lint clean.
- [ ] **Step 6: Commit.** `feat(run): dispatch the panel design with eligibility and panel artifacts`

---

### Task 9: Panel config, real-data anchor and docs

**Files:**
- Create: `configs/panel_leaves.toml`
- Modify: `tests/integration/test_panel_real.py`, `tests/unit/test_run.py`, `docs/development.md`, `README.md` (fork section), `AGENTS.md` (Layout list only)

**Interfaces:**
- Consumes: `load_config`, `panel_config`, `RunConfig` (Task 8); `run_panel`, `PanelConfig` (Task 6); `read_forecasts`, `build_bundle` (Task 4).

- [ ] **Step 1: Create `configs/panel_leaves.toml` exactly:**

```toml
# Pre-registered Phase 2 panel run (docs/protocol-decisions.md D20-D27).
# Run from the project root: `uv run python -m hcp.run configs/panel_leaves.toml`.
run_id = "panel-leaves-v1"
root_seed = 20261020
data_path = "data/deaths.csv"
expected_sha256 = "cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06"
output_dir = "results"
design = "panel"
forecasters = ["COVIDhub-4_week_ensemble", "CEID-Walk", "COVIDhub-baseline", "COVIDhub-ensemble", "COVIDhub-trained_ensemble", "JHUAPL-Bucky", "RobertWalraven-ESG", "UMass-MechBayes"]
alpha = 0.2
lr = 0.1
lr_grid = [1.0, 0.5, 0.1, 0.05]
Csat = 2.0
KI = 1000.0
T_burnin = 5
ahead = 4
reconcilers = ["none", "bottom_up", "mint_shrink", "mint_median"]
controllers = ["raw", "pi"]
mint_warmup = 20
rolling_window = 10
blocks = [8, 4, 12]
n_boot = 2000
min_eval_weeks = 60
max_leaf_imputed = 0.05
duplicate_share = 0.95
```

- [ ] **Step 2: Write the tests.**
  - Unit `test_panel_leaves_config_is_the_preregistered_run`: `load_config(ROOT_DIR / "configs/panel_leaves.toml")` gives `design == "panel"`, `sorted(forecasters) == D21_ELIGIBLE` (move `D21_ELIGIBLE` to `tests/helpers.py` and import it in both test files), and `forecasters[0] == FORECASTER`.
  - Integration `test_panel_us_node_reproduces_v2_metrics` (D27):
    - skip unless `ROOT_DIR / "results/us-hier-v2/metrics.parquet"` exists;
    - build the bundle with `build_bundle(read_forecasts(REAL_DATA, REAL_SHA256))`;
    - use `PanelConfig(controller=ControllerConfig(), lr_grid=(1.0, 0.5, 0.1, 0.05), reconcilers=("none", "bottom_up", "mint_shrink"), controllers=("raw", "pi"), mint_warmup=20, rolling_window=10, blocks=(8,), n_boot=10, root_seed=1, primary_forecaster=FORECASTER)`;
    - merge the `node == "us"` rows of `metrics` with v2's `metrics.parquet` on (reconciler, controller, lr);
    - assert there are 15 rows (3 reconcilers × (1 raw + 4 pi)) and that every `METRIC_NAMES` column is `assert_allclose(rtol=1e-10)`.
- [ ] **Step 3: Run them.** Unit: `uv run pytest -q`. Integration: `uv run pytest -o addopts='' -m integration -q`. Both green.
- [ ] **Step 4: Docs.**
  - `docs/development.md`: add the panel run command and its artifacts (`level_metrics`, `summary`, `eligibility` parquet), the analysis-script command from Task 3, and `scripts` in the lint paths.
  - `README.md` fork section: one short paragraph stating the Phase 1 null (link `docs/analysis/us-hier-v2.md`), the Phase 2 panel question, and the run command.
  - `AGENTS.md` Layout: change the `src/hcp/` bullet to name `eligibility` and `panel` in the stage list (`data → reconcile → conformal → evaluate → stats → eligibility/panel → report, orchestrated by run.py`), and add a bullet: `` `scripts/analysis/` — post-hoc analysis scripts; read results, write only under `docs/analysis/`. ``
- [ ] **Step 5: Commit.** Two commits:
  - `feat(config): pre-register the panel-leaves-v1 run`, covering the config and tests;
  - `docs: document the panel design, analysis script and run commands`.

---

## After Task 9 (controller)

1. Run the full suite, unit and integration, plus lint.
2. Final whole-branch review on opus, then fix.
3. `uv run python -m hcp.run configs/panel_leaves.toml` → `results/panel-leaves-v1/`.
4. Analyze it with `analyze-results` → `docs/analysis/panel-leaves-v1.md`. Record the outcome as D28, and update memory.
5. `superpowers:finishing-a-development-branch`.
