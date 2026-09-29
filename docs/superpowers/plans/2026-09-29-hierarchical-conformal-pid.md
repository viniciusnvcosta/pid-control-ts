# Hierarchical Reconciliation + Conformal PID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/hcp/`, a pipeline that reconciles COVIDhub 4-week-ahead death quantile forecasts across the US → 56-leaf hierarchy, conformalizes the US interval with the paper's P/PI/PID controllers, and measures whether reconciliation improves the US series' temporal coverage.

**Architecture:**
- A flat package with one module per stage, orchestrated by `run.py`: data → reconcile → conformal → evaluate → stats → report.
- It mirrors `~/projects/net-sci-epi`:
  - a TOML config loaded into a frozen dataclass
  - a module-level `dict` as the registry
  - numeric modules that never touch the filesystem
  - `results/<run_id>/` created with `exist_ok=False`, holding `manifest.json`
  - `np.random.SeedSequence` for all randomness
- The conformal controllers come from the upstream `core.methods` and are never reimplemented.

**Tech Stack:** Python ≥ 3.11, uv, numpy 2, pandas 3, statsmodels (Theta, via `core`), matplotlib (object-oriented API), pyarrow, pytest, ruff.

**Design spec:** `docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md` (decisions D1–D12).

## Global Constraints

- Every new `.py` file starts with two `# ABOUTME: ` lines.
- Type hints on every function. Google docstrings on public functions. Frozen dataclasses for configs and results.
- Numeric modules (`data`, `reconcile`, `conformal`, `evaluate`, `stats`) never write files. `report.py` writes figures only when `run.py` calls it.
- Dependencies are added only with `uv add`. Never edit `uv.lock` by hand.
- Lint only the new code: `uv run ruff check src tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py`. The upstream `core/` and `tests/*.py` harness are not linted.
- Test output must be clean. `uv run pytest` runs unit and e2e tests; `-m integration` needs `data/deaths.csv` (gitignored, SHA-256 `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`).
- Use Conventional Commits. Never pass `--no-verify`. Never push.
- The only changes allowed in `core/` are `np.infty` → `np.inf` (Task 1) and, by user decision D14, `scorecasts[t + ahead] = model.forecast(ahead).iloc[-1]` at `core/methods.py:250` (Task 4), each behind characterization tests.
- Hierarchy: row 0 is `us`, then the leaves in alphabetical order (51 states/DC plus `as`, `gu`, `mp`, `pr`, `vi`). Quantile levels are `(0.1, 0.5, 0.9)` on the last axis.

## Cost routing for subagent-driven development

| Task | Implementer | Spec reviewer | Quality reviewer | Why |
|---|---|---|---|---|
| 0 Housekeeping | controller (inline) | — | — | Needs user confirmation for commits |
| 1 Tooling + core characterization | **haiku** | haiku | haiku | Fully scripted, mechanical |
| 2 Data stage | **sonnet** | haiku | sonnet | pandas reshaping, LOCF edge cases |
| 3 Reconcile stage | **sonnet** | haiku | sonnet | Linear algebra plus causality |
| 4 Conformal adapter | **sonnet** | haiku | sonnet | Wraps legacy code, cache side effect |
| 5 Evaluate + stats | **haiku** | haiku | haiku | Small pure functions, code given |
| 6 Report | **sonnet** | haiku | sonnet | Figure layout judgment |
| 7 Run orchestration + e2e | **sonnet** | haiku | sonnet | Wiring, manifest, bootstrap loop |
| 8 Docs | **haiku** | haiku | — | Text given verbatim |
| Final whole-branch review | controller (opus) | — | — | One review pass over the whole branch |

Every step below ships complete code, so implementers only transcribe, run and report. If a test fails in a way this plan doesn't predict, the implementer stops and reports `BLOCKED` with the output instead of improvising.

## File map

```
core/methods.py, core/quantile.py      modify: np.infty → np.inf only (Task 1)
pyproject.toml                         modify: build system, dev deps, pytest/ruff config (Task 1)
src/hcp/__init__.py                    package marker (Task 2)
src/hcp/data.py                        load_hierarchy → HierarchyBundle (Task 2)
src/hcp/reconcile.py                   summing matrix, none / bottom_up / mint_shrink, causal online MinT (Task 3)
src/hcp/conformal.py                   CONTROLLERS raw / pi / pid_theta over core.methods (Task 4)
src/hcp/evaluate.py                    temporal-coverage metrics (Task 5)
src/hcp/stats.py                       moving-block bootstrap, Holm (Task 5)
src/hcp/report.py                      PDF figures (Task 6)
src/hcp/run.py                         RunConfig, run_experiment, write_artifacts, CLI (Task 7)
configs/us_hier.toml                   pre-registered run config (Task 7)
tests/helpers.py                       synthetic CSV builder, TOML writer, real-data constants (Task 2)
tests/conftest.py                      hier_csv / hier_bundle fixtures (Task 2)
tests/fixtures/core_golden.npz         pinned upstream outputs (Task 1)
tests/unit/test_*.py                   one per module (Tasks 1–7)
tests/integration/test_*.py            real-data parity and sanity (Tasks 2, 7)
tests/e2e/test_run_cli.py              CLI end to end on synthetic data (Task 7)
AGENTS.md, docs/protocol-decisions.md, docs/development.md, README.md   (Task 8)
```

---

### Task 0: Housekeeping (controller, inline)

- [ ] **Step 1:** Show the user `git status`. Ask whether to commit the currently staged uv migration and ruff reformat. Suggested messages: `build: migrate dependency management to uv` (pyproject.toml, uv.lock, requirements.txt removal, .gitignore) and `style: apply ruff format to upstream code` (core/, tests/*.py, the notebook). `.claude/logs/…` is staged but matches `.claude/**` in `.gitignore`, so unstage it with `git restore --staged .claude`.
- [ ] **Step 2:** `git switch -c feat/hierarchical-reconciliation`
- [ ] **Step 3:** Copy the approved design to `docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md` and commit it along with this plan: `docs: add hierarchical conformal PID spec and plan`.

---

### Task 1: Tooling, characterization of `core`, NumPy 2 fix [haiku]

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/unit/test_core_characterization.py`, `tests/fixtures/core_golden.npz`
- Modify: `core/methods.py` (every `np.infty`), `core/quantile.py:8`

**Interfaces:**
- Produces: working `uv run pytest`, and the `core.methods` functions `quantile`, `quantile_integrator_log`, `quantile_integrator_log_scorecaster` and `aci` running under NumPy 2 with unchanged numerics.

- [ ] **Step 1: Add dependencies**

```bash
uv add pyarrow
uv add --dev pytest ruff
```

- [ ] **Step 2: Append the build and tool configuration to `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/hcp", "core"]

[tool.pytest.ini_options]
testpaths = ["tests/unit", "tests/integration", "tests/e2e"]
pythonpath = ["src", "."]
addopts = "--import-mode=importlib -m 'not slow and not integration'"
markers = [
    "slow: expensive statistical checks",
    "integration: needs data/deaths.csv (gitignored, see docs/development.md)",
]

[tool.ruff.lint]
extend-select = ["PLC0415"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["PLC0415"]
```

`testpaths` stops pytest from collecting the paper's harness (`tests/base_test.py` matches `*_test.py`).

- [ ] **Step 3: Record the golden outputs with the NumPy-1 alias shimmed in**

```bash
mkdir -p tests/fixtures tests/unit tests/integration tests/e2e
uv run python - <<'EOF'
import numpy as np
np.infty = np.inf  # alias removed in NumPy 2; unmodified core still uses it
from core.methods import aci, quantile, quantile_integrator_log

rng = np.random.default_rng(0)
scores = rng.normal(size=200)
saturating = 100.0 + rng.normal(size=60)
np.savez_compressed(
    "tests/fixtures/core_golden.npz",
    scores=scores,
    saturating=saturating,
    quantile=quantile(scores, 0.1, 0.1, 4, T_burnin=5)["q"],
    integrator=quantile_integrator_log(scores, 0.1, 0.1, 2.0, 1000.0, 4, 5)["q"],
    saturated=quantile_integrator_log(saturating, 0.1, 0.0, 0.1, 1.0, 4, 5)["q"],
    aci=aci(scores, 0.1, 0.01, window_length=100000, T_burnin=5, ahead=4)["q"],
)
EOF
```

Expected: `tests/fixtures/core_golden.npz` exists, and `saturated` contains `inf` (this exercises the `mytan` branches).

- [ ] **Step 4: Write the characterization test, keeping the shim for now**

`tests/unit/test_core_characterization.py`:

```python
# ABOUTME: Characterization tests pinning the upstream conformal controllers in core/methods.py.
# ABOUTME: Golden outputs were recorded before the NumPy 2 np.infty -> np.inf fix; they must never change.
from pathlib import Path

import numpy as np
import pytest

from core.methods import aci, quantile, quantile_integrator_log

GOLDEN = np.load(Path(__file__).resolve().parents[1] / "fixtures" / "core_golden.npz")


@pytest.fixture(autouse=True)
def _numpy_infty_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "infty", np.inf, raising=False)


def test_quantile_tracker_matches_golden() -> None:
    q = quantile(GOLDEN["scores"], 0.1, 0.1, 4, T_burnin=5)["q"]
    np.testing.assert_array_equal(q, GOLDEN["quantile"])


def test_pi_controller_matches_golden() -> None:
    q = quantile_integrator_log(GOLDEN["scores"], 0.1, 0.1, 2.0, 1000.0, 4, 5)["q"]
    np.testing.assert_array_equal(q, GOLDEN["integrator"])


def test_saturated_integrator_matches_golden() -> None:
    q = quantile_integrator_log(GOLDEN["saturating"], 0.1, 0.0, 0.1, 1.0, 4, 5)["q"]
    assert np.isinf(GOLDEN["saturated"]).any()
    np.testing.assert_array_equal(q, GOLDEN["saturated"])


def test_aci_matches_golden() -> None:
    q = aci(GOLDEN["scores"], 0.1, 0.01, window_length=100000, T_burnin=5, ahead=4)["q"]
    np.testing.assert_array_equal(q, GOLDEN["aci"])
```

- [ ] **Step 5: Run it.** `uv run pytest tests/unit/test_core_characterization.py -q`. Expected: `4 passed`.

- [ ] **Step 6: Apply the fix, then delete the shim fixture**

```bash
sed -i 's/np\.infty/np.inf/g' core/methods.py core/quantile.py
grep -rn "np.infty" core/ && echo "LEFTOVER" || echo "clean"
```

Expected: `clean`. Then delete the `_numpy_infty_alias` fixture from the test file. Once the fixture is gone the test no longer uses `pytest`, so also delete the `import pytest` line; otherwise ruff F401 fails.

- [ ] **Step 7: Run again.** `uv run pytest -q` should report `4 passed`. `uv run ruff check tests/unit` should pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock core/methods.py core/quantile.py tests/unit/test_core_characterization.py tests/fixtures/core_golden.npz
git commit -m "fix(core): replace np.infty for NumPy 2 behind characterization tests"
```

---

### Task 2: Data stage [sonnet]

**Files:**
- Create: `src/hcp/__init__.py`, `src/hcp/data.py`, `tests/helpers.py`, `tests/conftest.py`, `tests/unit/test_data.py`, `tests/integration/test_data_parity.py`

**Interfaces:**
- Produces:
  - `hcp.data.HierarchyBundle(dates, forecast_dates, nodes, y, q, imputed)`, frozen with read-only arrays: `dates` / `forecast_dates` are `datetime64[T]`, `nodes` is `tuple[str, ...]`, `y` is `[n,T]`, `q` is `[n,T,3]`, `imputed` is a `[n,T]` bool array.
  - `load_hierarchy(path: Path, expected_sha256: str | None = None) -> HierarchyBundle`
  - `file_sha256(path: Path) -> str`
  - Constants `ROOT`, `FORECASTER`, `AHEAD`, `QUANTILE_LEVELS`, `QUANTILE_COLUMNS`.
  - Fixtures `hier_csv` (Path) and `hier_bundle`.
  - `tests.helpers`: `START`, `N_WEEKS`, `LEAVES`, `LATE_LEAF`, `LATE_WEEKS`, `GAP_LEAF`, `GAP_WEEK`, `TRUTHLESS_US_WEEKS`, `REAL_DATA`, `REAL_SHA256`, `PICKLES`, `synthetic_rows(seed)`, `write_config(path, values)`.
- Synthetic bundle facts that later tasks rely on:
  - 26 weeks, nodes `("us","aa","bb","cc")`.
  - `cc` is imputed at week index 8 and its truth there is NaN.
  - With `warmup=5`, `evaluation_start == 8`.

- [ ] **Step 1: Create `src/hcp/__init__.py`**

```python
# ABOUTME: Hierarchical reconciliation + conformal PID pipeline for US COVID-19 death forecasts.
# ABOUTME: Stages data -> reconcile -> conformal -> evaluate -> stats -> report, orchestrated by run.
"""Does state -> US reconciliation improve the temporal coverage of conformal PID?"""

__all__: list[str] = []
```

- [ ] **Step 2: Create `tests/helpers.py`**

```python
# ABOUTME: Test helpers: a synthetic forecast CSV in the real deaths.csv schema, a TOML writer, real-data paths.
# ABOUTME: The synthetic hierarchy is us + aa/bb/cc with a late leaf, a missing leaf week and truthless US weeks.
from pathlib import Path

import numpy as np
import pandas as pd

from hcp.data import FORECASTER, QUANTILE_COLUMNS, QUANTILE_LEVELS

ROOT_DIR = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT_DIR / "data" / "deaths.csv"
REAL_SHA256 = "cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06"
PICKLES = ROOT_DIR / "tests" / "datasets" / "covid-ts-proc" / "statewide"
PICKLE_STATES = ("ak", "ca", "fl", "ga", "ks", "ny", "tx")

START = pd.Timestamp("2021-01-02")
N_WEEKS = 30
LEAVES = ("aa", "bb", "cc")
LATE_LEAF, LATE_WEEKS = "bb", 2  # no bb rows in weeks 0-1, so both weeks are trimmed
GAP_LEAF, GAP_WEEK = "cc", 10  # no cc row in week 10, so LOCF fills it
TRUTHLESS_US_WEEKS = 2  # last two weeks lack US truth, so they leave the grid


def synthetic_rows(seed: int = 0) -> list[dict]:
    """Rows in the deaths.csv schema, plus decoy forecaster and ahead=1 rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(N_WEEKS):
        target = START + pd.Timedelta(weeks=week)
        leaf_truth = {
            leaf: float(rng.poisson(50 * (i + 1))) for i, leaf in enumerate(LEAVES)
        }
        truth = {"us": sum(leaf_truth.values()), **leaf_truth}
        for geo, value in truth.items():
            if geo == LATE_LEAF and week < LATE_WEEKS:
                continue
            if geo == GAP_LEAF and week == GAP_WEEK:
                continue
            median = value * float(rng.uniform(0.8, 1.2))
            truthless = geo == "us" and week >= N_WEEKS - TRUTHLESS_US_WEEKS
            row = {
                "ahead": 4,
                "geo_value": geo,
                "forecaster": FORECASTER,
                "forecast_date": (target - pd.Timedelta(days=26)).date(),
                "target_end_date": target.date(),
                "forecast_0.25": median,
                "actual": np.nan if truthless else value,
            }
            for level, column in zip(QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True):
                row[column] = median * (1 + 1.5 * (level - 0.5))
            rows.append(row)
            rows.append(
                {**row, "forecaster": "COVIDhub-baseline", **dict.fromkeys(QUANTILE_COLUMNS, -1.0)}
            )
            rows.append({**row, "ahead": 1, **dict.fromkeys(QUANTILE_COLUMNS, -2.0)})
    return rows


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return '"' + str(value) + '"'


def write_config(path: Path, values: dict) -> Path:
    """Write a flat TOML file from ``values``."""
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n")
    return path
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
# ABOUTME: Shared pytest fixtures: the synthetic hierarchy CSV and its loaded bundle.
# ABOUTME: See tests/helpers.py for the exact gaps the synthetic data contains.
from pathlib import Path

import pandas as pd
import pytest

from hcp.data import HierarchyBundle, load_hierarchy
from tests.helpers import synthetic_rows


@pytest.fixture
def hier_csv(tmp_path: Path) -> Path:
    path = tmp_path / "deaths.csv"
    pd.DataFrame(synthetic_rows()).to_csv(path)  # index column mimics the real leading ""
    return path


@pytest.fixture
def hier_bundle(hier_csv: Path) -> HierarchyBundle:
    return load_hierarchy(hier_csv)
```

- [ ] **Step 4: Write the failing unit tests in `tests/unit/test_data.py`**

```python
# ABOUTME: Unit tests for the data stage on the synthetic hierarchy CSV.
# ABOUTME: Covers node order, filtering, SHA-256 check, week grid, trimming, LOCF and truth handling.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcp.data import file_sha256, load_hierarchy
from tests.helpers import (
    GAP_LEAF,
    GAP_WEEK,
    LATE_WEEKS,
    N_WEEKS,
    START,
    TRUTHLESS_US_WEEKS,
    synthetic_rows,
)

GAP_INDEX = GAP_WEEK - LATE_WEEKS


def test_nodes_are_root_then_sorted_leaves(hier_bundle) -> None:
    assert hier_bundle.nodes == ("us", "aa", "bb", "cc")


def test_grid_drops_truthless_us_weeks_and_trims_late_leaf(hier_bundle) -> None:
    n_weeks = N_WEEKS - TRUTHLESS_US_WEEKS - LATE_WEEKS
    assert hier_bundle.y.shape == (4, n_weeks)
    assert hier_bundle.q.shape == (4, n_weeks, 3)
    assert pd.Timestamp(hier_bundle.dates[0]) == START + pd.Timedelta(weeks=LATE_WEEKS)


def test_keeps_only_ensemble_at_ahead_4(hier_bundle) -> None:
    assert (hier_bundle.q > 0).all()  # decoy rows carry -1 and -2


def test_quantile_levels_are_on_last_axis(hier_bundle) -> None:
    q = hier_bundle.q
    np.testing.assert_allclose(q[..., 0], 0.4 * q[..., 1])
    np.testing.assert_allclose(q[..., 2], 1.6 * q[..., 1])


def test_forecast_dates_are_26_days_before_target(hier_bundle) -> None:
    lag = (hier_bundle.dates - hier_bundle.forecast_dates).astype("timedelta64[D]")
    assert (lag.astype(int) == 26).all()


def test_missing_leaf_week_is_carried_forward_and_flagged(hier_bundle) -> None:
    row = hier_bundle.nodes.index(GAP_LEAF)
    np.testing.assert_array_equal(
        hier_bundle.q[row, GAP_INDEX], hier_bundle.q[row, GAP_INDEX - 1]
    )
    assert hier_bundle.imputed[row, GAP_INDEX]
    assert hier_bundle.imputed.sum() == 1


def test_truth_is_never_imputed(hier_bundle) -> None:
    row = hier_bundle.nodes.index(GAP_LEAF)
    assert np.isnan(hier_bundle.y[row, GAP_INDEX])
    assert np.isfinite(np.delete(hier_bundle.y, GAP_INDEX, axis=1)).all()


def test_us_truth_is_sum_of_leaves(hier_bundle) -> None:
    y = np.delete(hier_bundle.y, GAP_INDEX, axis=1)
    np.testing.assert_allclose(y[0], y[1:].sum(axis=0))


def test_arrays_are_read_only(hier_bundle) -> None:
    for name in ("dates", "forecast_dates", "y", "q", "imputed"):
        assert not getattr(hier_bundle, name).flags.writeable, name


def test_sha256_mismatch_raises(hier_csv: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        load_hierarchy(hier_csv, expected_sha256="0" * 64)


def test_sha256_match_loads(hier_csv: Path) -> None:
    bundle = load_hierarchy(hier_csv, expected_sha256=file_sha256(hier_csv))
    assert bundle.nodes[0] == "us"


def test_missing_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "leaves.csv"
    rows = [row for row in synthetic_rows() if row["geo_value"] != "us"]
    pd.DataFrame(rows).to_csv(path)
    with pytest.raises(ValueError, match="root"):
        load_hierarchy(path)


def test_duplicate_rows_raise(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    rows = synthetic_rows()
    pd.DataFrame(rows + rows[:1]).to_csv(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_hierarchy(path)
```

- [ ] **Step 5: Run and confirm the failure.** `uv run pytest tests/unit/test_data.py -q` fails with `ModuleNotFoundError: No module named 'hcp.data'`, raised from `tests/helpers.py` during conftest import.

- [ ] **Step 6: Implement `src/hcp/data.py`**

```python
# ABOUTME: Loads COVIDhub 4-week-ahead death forecasts into a US -> 56-leaf hierarchy bundle.
# ABOUTME: Verifies the input SHA-256, carries missing leaf quantiles forward causally, never imputes truth.
"""Data stage (decisions D1, D5, D6).

Rows are the root ``us`` followed by the leaves in alphabetical order. The week grid is
the set of target weeks where the US forecast and truth are complete. Leading weeks in
which some leaf has never been forecast are trimmed, and later leaf gaps are filled by
last observation carried forward (LOCF) and flagged in ``imputed``.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "us"
FORECASTER = "COVIDhub-4_week_ensemble"
AHEAD = 4
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
QUANTILE_COLUMNS = tuple(f"forecast_{level}" for level in QUANTILE_LEVELS)
_COLUMNS = [
    "ahead",
    "geo_value",
    "forecaster",
    "forecast_date",
    "target_end_date",
    *QUANTILE_COLUMNS,
    "actual",
]


@dataclass(frozen=True)
class HierarchyBundle:
    """Aligned truth and base quantiles for the root and its leaves.

    Attributes:
        dates: Target end dates ``[T]``.
        forecast_dates: Earliest forecast date among the nodes for each target ``[T]``.
        nodes: ``ROOT`` then the leaves, sorted.
        y: Reported truth ``[n_nodes, T]``; NaN where not reported.
        q: Base quantiles ``[n_nodes, T, 3]`` at ``QUANTILE_LEVELS`` after LOCF.
        imputed: ``[n_nodes, T]``; True where ``q`` was carried forward.
    """

    dates: np.ndarray
    forecast_dates: np.ndarray
    nodes: tuple[str, ...]
    y: np.ndarray
    q: np.ndarray
    imputed: np.ndarray


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hierarchy(path: Path, expected_sha256: str | None = None) -> HierarchyBundle:
    """Load the ensemble's 4-week-ahead forecasts for ``us`` and every leaf.

    Args:
        path: The deaths.csv file (paper's LFS input schema).
        expected_sha256: If given, the file's SHA-256 must match.

    Raises:
        ValueError: On a SHA-256 mismatch, duplicate rows, a missing root, or no usable week.
    """
    path = Path(path)
    if expected_sha256 is not None:
        digest = file_sha256(path)
        if digest != expected_sha256:
            raise ValueError(f"{path} has sha256 {digest}, expected {expected_sha256}")
    frame = pd.read_csv(
        path, usecols=_COLUMNS, parse_dates=["forecast_date", "target_end_date"]
    )
    frame = frame[(frame["ahead"] == AHEAD) & (frame["forecaster"] == FORECASTER)]
    if frame.duplicated(["geo_value", "target_end_date"]).any():
        raise ValueError("duplicate (geo_value, target_end_date) rows")
    if ROOT not in set(frame["geo_value"]):
        raise ValueError(f"root series {ROOT!r} is missing")
    nodes = (ROOT, *sorted(set(frame["geo_value"]) - {ROOT}))
    root = frame[frame["geo_value"] == ROOT].dropna(subset=[*QUANTILE_COLUMNS, "actual"])
    dates = pd.DatetimeIndex(sorted(root["target_end_date"]))

    def wide(column: str) -> np.ndarray:
        table = frame.pivot(index="target_end_date", columns="geo_value", values=column)
        return table.reindex(index=dates, columns=list(nodes)).to_numpy(dtype=float).T

    raw_q = np.stack([wide(column) for column in QUANTILE_COLUMNS], axis=-1)
    filled = np.stack(
        [pd.DataFrame(raw_q[:, :, k].T).ffill().to_numpy().T for k in range(raw_q.shape[-1])],
        axis=-1,
    )
    complete = np.isfinite(filled).all(axis=(0, 2))
    if not complete.any():
        raise ValueError("no week has forecasts for every node")
    start = int(np.argmax(complete))
    imputed = np.isnan(raw_q).any(axis=-1) & np.isfinite(filled).all(axis=-1)
    forecast_dates = frame.groupby("target_end_date")["forecast_date"].min().reindex(dates)
    arrays = {
        "dates": dates.to_numpy()[start:],
        "forecast_dates": forecast_dates.to_numpy()[start:],
        "y": wide("actual")[:, start:],
        "q": filled[:, start:],
        "imputed": imputed[:, start:],
    }
    for array in arrays.values():
        array.setflags(write=False)
    return HierarchyBundle(nodes=nodes, **arrays)
```

- [ ] **Step 7: Run the unit tests.** `uv run pytest tests/unit/test_data.py -q` should report `13 passed`.

- [ ] **Step 8: Write the real-data parity test in `tests/integration/test_data_parity.py`**

```python
# ABOUTME: Integration tests: the real deaths.csv hierarchy and exact parity with the paper's state pickles.
# ABOUTME: Needs data/deaths.csv (gitignored); run with `uv run pytest -m integration`.
import numpy as np
import pandas as pd
import pytest

from hcp.data import load_hierarchy
from tests.helpers import PICKLE_STATES, PICKLES, REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]


@pytest.fixture(scope="module")
def real_bundle():
    return load_hierarchy(REAL_DATA, REAL_SHA256)


def test_real_hierarchy_is_us_plus_56_leaves(real_bundle) -> None:
    assert real_bundle.nodes[0] == "us"
    assert len(real_bundle.nodes) == 57
    assert {"dc", "as", "gu", "mp", "pr", "vi"} <= set(real_bundle.nodes)


@pytest.mark.parametrize("state", PICKLE_STATES)
def test_matches_paper_pickle(real_bundle, state: str) -> None:
    paper = pd.read_pickle(PICKLES / f"{state}_proc_4wkdeaths.pkl").pivot(
        index="timestamp", columns="variable", values="target"
    )
    row = real_bundle.nodes.index(state)
    ours = pd.DataFrame(
        {
            "our_y": real_bundle.y[row],
            "our_lo": real_bundle.q[row, :, 0],
            "our_hi": real_bundle.q[row, :, 2],
            "imputed": real_bundle.imputed[row],
        },
        index=pd.DatetimeIndex(real_bundle.dates),
    )
    joined = ours.join(paper, how="inner")
    joined = joined[~joined["imputed"] & joined["our_y"].notna()]
    assert len(joined) > 100
    forecasts = np.stack(joined["forecasts"].to_numpy())
    np.testing.assert_array_equal(joined["our_y"], joined["y"].astype(float))
    np.testing.assert_array_equal(joined["our_lo"], forecasts[:, 0])
    np.testing.assert_array_equal(joined["our_hi"], forecasts[:, 1])
```

- [ ] **Step 9: Run it.** `uv run pytest -m integration tests/integration/test_data_parity.py -q` should report `8 passed`. It takes about 30 s to parse 219 MB.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check src tests/unit tests/integration tests/helpers.py tests/conftest.py && uv run ruff format src tests/unit tests/integration tests/helpers.py tests/conftest.py
git add src/hcp tests/helpers.py tests/conftest.py tests/unit/test_data.py tests/integration/test_data_parity.py
git commit -m "feat(data): load US and 56-leaf COVIDhub hierarchy with causal LOCF"
```

---

### Task 3: Reconcile stage [sonnet]

**Files:**
- Create: `src/hcp/reconcile.py`, `tests/unit/test_reconcile.py`

**Interfaces:**
- Consumes: `HierarchyBundle` and the `hier_bundle` fixture (Task 2).
- Produces:
  - `summing_matrix(n_leaves: int) -> np.ndarray`
  - `shrinkage_covariance(errors: np.ndarray) -> tuple[np.ndarray, float]`
  - `RECONCILERS: dict[str, Callable]` with keys `"none"`, `"bottom_up"`, `"mint_shrink"`
  - `reconcile(base, S, method, *, covariance=None) -> np.ndarray`
  - `available_errors(bundle, t) -> np.ndarray [n_known, n_nodes]`
  - `evaluation_start(bundle, warmup: int) -> int`
  - `Reconciled(q, crossings, shrinkage)`
  - `reconcile_bundle(bundle, method: str, warmup: int) -> Reconciled`

- [ ] **Step 1: Write the failing tests in `tests/unit/test_reconcile.py`**

```python
# ABOUTME: Unit tests for reconciliation: summing matrix, bottom-up, MinT(Shrink) and the causal online bundle pass.
# ABOUTME: Checks coherence, projection, prefix invariance and that unknown truth never leaks in.
from dataclasses import replace

import numpy as np
import pytest

from hcp.reconcile import (
    RECONCILERS,
    available_errors,
    evaluation_start,
    reconcile,
    reconcile_bundle,
    shrinkage_covariance,
    summing_matrix,
)

WARMUP = 5
T0 = 8  # synthetic bundle: 5 complete error rows are known first at t = 8


def _truncate(bundle, k: int):
    return replace(
        bundle,
        dates=bundle.dates[:k],
        forecast_dates=bundle.forecast_dates[:k],
        y=bundle.y[:, :k],
        q=bundle.q[:, :k],
        imputed=bundle.imputed[:, :k],
    )


def test_registry_names() -> None:
    assert set(RECONCILERS) == {"none", "bottom_up", "mint_shrink"}


def test_summing_matrix_is_ones_over_identity() -> None:
    S = summing_matrix(3)
    assert S.shape == (4, 3)
    np.testing.assert_array_equal(S[0], np.ones(3))
    np.testing.assert_array_equal(S[1:], np.eye(3))


def test_none_returns_a_copy_of_base() -> None:
    base = np.arange(12.0).reshape(4, 3)
    out = reconcile(base, summing_matrix(3), "none")
    np.testing.assert_array_equal(out, base)
    assert out is not base


@pytest.mark.parametrize("seed", range(5))
def test_bottom_up_sums_leaves(seed: int) -> None:
    base = np.random.default_rng(seed).normal(size=(4, 3))
    out = reconcile(base, summing_matrix(3), "bottom_up")
    np.testing.assert_allclose(out[0], base[1:].sum(axis=0))
    np.testing.assert_allclose(out[1:], base[1:])


@pytest.mark.parametrize("seed", range(5))
def test_mint_is_coherent_and_fixes_coherent_input(seed: int) -> None:
    rng = np.random.default_rng(seed)
    S = summing_matrix(3)
    covariance, lam = shrinkage_covariance(rng.normal(size=(30, 4)))
    out = reconcile(rng.normal(size=(4, 3)), S, "mint_shrink", covariance=covariance)
    np.testing.assert_allclose(out[0], out[1:].sum(axis=0))
    coherent = S @ rng.normal(size=(3, 3))
    np.testing.assert_allclose(
        reconcile(coherent, S, "mint_shrink", covariance=covariance), coherent
    )
    assert 0.0 <= lam <= 1.0


def test_shrinkage_covariance_is_symmetric_positive_definite() -> None:
    covariance, _ = shrinkage_covariance(np.random.default_rng(0).normal(size=(20, 6)))
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown reconciliation method"):
        reconcile(np.zeros((4, 3)), summing_matrix(3), "nope")


def test_mint_without_covariance_raises() -> None:
    with pytest.raises(ValueError, match="covariance"):
        reconcile(np.zeros((4, 3)), summing_matrix(3), "mint_shrink")


def test_available_errors_uses_targets_known_by_forecast_date(hier_bundle) -> None:
    assert available_errors(hier_bundle, T0).shape == (WARMUP, 4)


def test_evaluation_start_is_first_week_with_warmup_errors(hier_bundle) -> None:
    assert evaluation_start(hier_bundle, WARMUP) == T0


def test_evaluation_start_raises_without_enough_history(hier_bundle) -> None:
    with pytest.raises(ValueError, match="complete error weeks"):
        evaluation_start(hier_bundle, 1000)


def test_warmup_below_two_raises(hier_bundle) -> None:
    with pytest.raises(ValueError, match="warmup"):
        reconcile_bundle(hier_bundle, "mint_shrink", 1)


def test_none_bundle_equals_base(hier_bundle) -> None:
    rec = reconcile_bundle(hier_bundle, "none", WARMUP)
    np.testing.assert_array_equal(rec.q, hier_bundle.q)
    assert rec.crossings == 0


@pytest.mark.parametrize("method", ["bottom_up", "mint_shrink"])
def test_bundle_is_coherent_and_sorted_where_defined(hier_bundle, method: str) -> None:
    rec = reconcile_bundle(hier_bundle, method, WARMUP)
    defined = np.isfinite(rec.q).all(axis=(0, 2))
    assert (np.diff(rec.q[:, defined], axis=-1) >= 0).all()
    if rec.crossings == 0:
        np.testing.assert_allclose(rec.q[0, defined], rec.q[1:, defined].sum(axis=0))


def test_mint_is_undefined_before_evaluation_start(hier_bundle) -> None:
    rec = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    assert np.isnan(rec.q[:, :T0]).all()
    assert np.isfinite(rec.q[:, T0:]).all()
    assert np.isnan(rec.shrinkage[:T0]).all()
    assert np.isfinite(rec.shrinkage[T0:]).all()


def test_mint_is_prefix_invariant(hier_bundle) -> None:
    full = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    part = reconcile_bundle(_truncate(hier_bundle, 15), "mint_shrink", WARMUP)
    np.testing.assert_array_equal(part.q, full.q[:, :15])


def test_mint_ignores_truth_unknown_at_forecast_date(hier_bundle) -> None:
    t = 12
    y = hier_bundle.y.copy()
    y[:, t - 3 :] = 1e9  # targets less than 4 weeks before t are unknown at its forecast date
    leaked = reconcile_bundle(replace(hier_bundle, y=y), "mint_shrink", WARMUP)
    clean = reconcile_bundle(hier_bundle, "mint_shrink", WARMUP)
    np.testing.assert_array_equal(leaked.q[:, t], clean.q[:, t])
```

- [ ] **Step 2: Run and confirm the failure.** `uv run pytest tests/unit/test_reconcile.py -q` fails with `ModuleNotFoundError: No module named 'hcp.reconcile'`.

- [ ] **Step 3: Implement `src/hcp/reconcile.py`**

```python
# ABOUTME: Summing matrix, bottom-up and MinT(Shrink) reconciliation of the base quantile forecasts.
# ABOUTME: MinT uses a causal expanding covariance of median errors known by each forecast date (D7).
"""Reconciliation stage (decisions D2, D5, D7, D8).

``summing_matrix``, ``shrinkage_covariance``, bottom-up and MinT(Shrink) are ported from
net-sci-epi ``src/headd_l0/reconcile.py`` and generalized to any number of leaves. MinT
follows Wickramasuriya, Athanasopoulos & Hyndman (2019), shrinking the error covariance
towards its diagonal with the Schafer & Strimmer (2005) lambda. Each quantile level is
reconciled with the same projection; levels that cross afterwards are re-sorted.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hcp.data import HierarchyBundle


@dataclass(frozen=True)
class Reconciled:
    """Reconciled quantiles for every node.

    Attributes:
        q: ``[n_nodes, T, 3]``; NaN in weeks before the MinT warm-up.
        crossings: Number of (node, week) cells whose levels had to be re-sorted.
        shrinkage: MinT lambda per week ``[T]``; NaN where no covariance was estimated.
    """

    q: np.ndarray
    crossings: int
    shrinkage: np.ndarray


def summing_matrix(n_leaves: int) -> np.ndarray:
    """Return S ``[n_leaves + 1, n_leaves]``: a row of ones, then the identity."""
    return np.vstack([np.ones((1, n_leaves)), np.eye(n_leaves)])


def shrinkage_covariance(errors: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the error covariance towards its diagonal (MinT(Shrink)).

    Args:
        errors: Training errors ``[n_train, n_series]``.

    Returns:
        ``(W, lambda)`` with ``W = lambda diag(W1) + (1 - lambda) W1`` and a floor of
        ``1e-8 * max(trace(W) / n_series, 1)`` on the diagonal. A series with zero
        variance contributes zero correlation; lambda is 1 when every correlation is 0.
    """
    n, p = errors.shape
    w1 = errors.T @ errors / n
    sd = np.sqrt(np.diag(w1))
    scale = np.where(sd > 0, sd, 1.0)
    xs = np.where(sd > 0, errors / scale, 0.0)
    correlation = xs.T @ xs / n
    variance = (xs.T**2 @ xs**2 - (xs.T @ xs) ** 2 / n) / (n * (n - 1))
    off = ~np.eye(p, dtype=bool)
    denominator = float((correlation[off] ** 2).sum())
    lam = (
        1.0
        if denominator == 0
        else min(max(variance[off].sum() / denominator, 0.0), 1.0)
    )
    covariance = lam * np.diag(np.diag(w1)) + (1 - lam) * w1
    floor = 1e-8 * max(np.trace(covariance) / p, 1.0)
    np.fill_diagonal(covariance, np.maximum(np.diag(covariance), floor))
    return covariance, lam


def _none(base: np.ndarray, _S: np.ndarray, _covariance: np.ndarray | None) -> np.ndarray:
    return base.copy()


def _bottom_up(
    base: np.ndarray, S: np.ndarray, _covariance: np.ndarray | None
) -> np.ndarray:
    return S @ base[1:]


def _mint_shrink(
    base: np.ndarray, S: np.ndarray, covariance: np.ndarray | None
) -> np.ndarray:
    if covariance is None:
        raise ValueError("mint_shrink needs the training error covariance")
    weighted = np.linalg.solve(covariance, S)  # W^-1 S
    gain = np.linalg.solve(S.T @ weighted, weighted.T)  # (S' W^-1 S)^-1 S' W^-1
    return S @ (gain @ base)


RECONCILERS: dict[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]
] = {
    "none": _none,
    "bottom_up": _bottom_up,
    "mint_shrink": _mint_shrink,
}


def reconcile(
    base: np.ndarray,
    S: np.ndarray,
    method: str,
    *,
    covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Project base forecasts ``[n_nodes, ...]`` (root first) onto the coherent subspace.

    Raises:
        ValueError: On an unknown method, or ``mint_shrink`` without a covariance.
    """
    if method not in RECONCILERS:
        raise ValueError(f"unknown reconciliation method: {method}")
    return RECONCILERS[method](base, S, covariance)


def available_errors(bundle: HierarchyBundle, t: int) -> np.ndarray:
    """Median errors ``[n_known, n_nodes]`` of targets reported by ``forecast_dates[t]``.

    Only weeks where every node has a finite error are returned.
    """
    known = bundle.dates <= bundle.forecast_dates[t]
    errors = (bundle.y - bundle.q[:, :, 1]).T[known]
    return errors[np.isfinite(errors).all(axis=1)]


def evaluation_start(bundle: HierarchyBundle, warmup: int) -> int:
    """First week index whose forecast date already knows ``warmup`` complete error weeks.

    Raises:
        ValueError: If no week reaches the warm-up.
    """
    for t in range(len(bundle.dates)):
        if len(available_errors(bundle, t)) >= warmup:
            return t
    raise ValueError(f"fewer than {warmup} complete error weeks before the last forecast")


def reconcile_bundle(bundle: HierarchyBundle, method: str, warmup: int) -> Reconciled:
    """Reconcile every week; MinT re-estimates its covariance causally each week.

    Args:
        bundle: Data stage output.
        method: A key of ``RECONCILERS``.
        warmup: Minimum number of known complete error weeks before MinT is defined.

    Raises:
        ValueError: On an unknown method or ``warmup < 2``.
    """
    if method not in RECONCILERS:
        raise ValueError(f"unknown reconciliation method: {method}")
    if warmup < 2:
        raise ValueError("warmup must be at least 2 weeks")
    S = summing_matrix(len(bundle.nodes) - 1)
    out = np.full(bundle.q.shape, np.nan)
    shrinkage = np.full(len(bundle.dates), np.nan)
    for t in range(len(bundle.dates)):
        covariance = None
        if method == "mint_shrink":
            errors = available_errors(bundle, t)
            if len(errors) < warmup:
                continue
            covariance, shrinkage[t] = shrinkage_covariance(errors)
        out[:, t, :] = reconcile(bundle.q[:, t, :], S, method, covariance=covariance)
    defined = np.isfinite(out).all(axis=-1)
    crossed = (np.diff(out, axis=-1) < 0).any(axis=-1) & defined
    return Reconciled(
        q=np.sort(out, axis=-1), crossings=int(crossed.sum()), shrinkage=shrinkage
    )
```

- [ ] **Step 4: Run the tests.** `uv run pytest tests/unit/test_reconcile.py -q` should report all passed (26 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests/unit && uv run ruff format src tests/unit
git add src/hcp/reconcile.py tests/unit/test_reconcile.py
git commit -m "feat(reconcile): add bottom-up and causal online MinT(Shrink) reconciliation"
```

---

### Task 4: Conformal adapter [sonnet]

**Files:**
- Create: `src/hcp/conformal.py`, `tests/unit/test_conformal.py`

**Interfaces:**
- Consumes: `core.methods.quantile_integrator_log` and `quantile_integrator_log_scorecaster` (Task 1). `tests.helpers.PICKLES` and `PICKLE_STATES` (Task 2).
- Produces:
  - `ControllerConfig(alpha=0.2, lr=0.1, Csat=2.0, KI=1000.0, T_burnin=5, ahead=4)`, frozen
  - `CONTROLLERS: dict[str, Callable[[np.ndarray, ControllerConfig, bool], np.ndarray]]` with keys `"raw"`, `"pi"`, `"pid_theta"`
  - `cqr_scores(y, q_lo, q_hi) -> np.ndarray [T, 2]`
  - `conformalize(y, q_lo, q_hi, controller: str, cfg: ControllerConfig) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing tests in `tests/unit/test_conformal.py`**

```python
# ABOUTME: Unit tests for the conformal adapter over core.methods (raw, PI, PID with Theta scorecaster).
# ABOUTME: Pins parity with the paper harness wiring (tests/base_test.py) and the paper's COVID score pickles.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.methods import quantile_integrator_log
from hcp.conformal import CONTROLLERS, ControllerConfig, conformalize, cqr_scores
from tests.helpers import PICKLE_STATES, PICKLES


def _narrow_series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, ...]:
    y = np.random.default_rng(seed).normal(size=n)
    return y, np.full(n, -0.3), np.full(n, 0.3)


def test_registry_names() -> None:
    assert set(CONTROLLERS) == {"raw", "pi", "pid_theta"}


def test_cqr_scores_follow_paper_definition() -> None:
    y = np.array([1.0, 5.0])
    scores = cqr_scores(y, np.array([2.0, 1.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(scores, [[1.0, -2.0], [-4.0, 1.0]])


@pytest.mark.parametrize("state", PICKLE_STATES)
def test_cqr_scores_match_paper_pickles(state: str) -> None:
    frame = pd.read_pickle(PICKLES / f"{state}_proc_4wkdeaths.pkl").pivot(
        index="timestamp", columns="variable", values="target"
    )
    y = frame["y"].astype(float).to_numpy()
    keep = np.isfinite(y)
    forecasts = np.stack(frame["forecasts"].to_numpy())[keep]
    paper = np.stack(frame["scores"].to_numpy())[keep].astype(float)
    np.testing.assert_allclose(cqr_scores(y[keep], forecasts[:, 0], forecasts[:, 1]), paper)


def test_raw_returns_base_interval() -> None:
    y, lo, hi = _narrow_series()
    out_lo, out_hi = conformalize(y, lo, hi, "raw", ControllerConfig())
    np.testing.assert_array_equal(out_lo, lo)
    np.testing.assert_array_equal(out_hi, hi)


def test_pi_matches_paper_harness_wiring() -> None:
    y, lo, hi = _narrow_series()
    cfg = ControllerConfig(lr=0.1)
    scores = cqr_scores(y, lo, hi)
    kwargs = {
        "Csat": cfg.Csat,
        "KI": cfg.KI,
        "T_burnin": cfg.T_burnin,
        "data": None,
        "seasonal_period": None,
        "config_name": "parity",
        "ahead": cfg.ahead,
    }
    q_lo = quantile_integrator_log(scores[:, 0], cfg.alpha / 2, cfg.lr, upper=False, **kwargs)
    q_hi = quantile_integrator_log(scores[:, 1], cfg.alpha / 2, cfg.lr, upper=True, **kwargs)
    out_lo, out_hi = conformalize(y, lo, hi, "pi", cfg)
    np.testing.assert_array_equal(out_lo, lo - q_lo["q"])
    np.testing.assert_array_equal(out_hi, hi + q_hi["q"])


def test_pi_restores_coverage_of_a_too_narrow_interval() -> None:
    y, lo, hi = _narrow_series(n=400)
    out_lo, out_hi = conformalize(y, lo, hi, "pi", ControllerConfig(lr=0.1))
    raw = ((lo <= y) & (y <= hi)).mean()
    adjusted = ((out_lo <= y) & (y <= out_hi))[100:].mean()
    assert raw < 0.3
    assert 0.7 <= adjusted <= 0.9


def test_non_finite_truth_raises() -> None:
    y, lo, hi = _narrow_series()
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        conformalize(y, lo, hi, "pi", ControllerConfig())


def test_unknown_controller_raises() -> None:
    y, lo, hi = _narrow_series()
    with pytest.raises(ValueError, match="unknown controller"):
        conformalize(y, lo, hi, "nope", ControllerConfig())


def test_pid_theta_is_deterministic_and_leaves_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    y, lo, hi = _narrow_series(n=60)
    cfg = ControllerConfig(lr=0.1)
    first = conformalize(y, lo, hi, "pid_theta", cfg)
    second = conformalize(y, lo, hi, "pid_theta", cfg)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not (tmp_path / ".cache").exists()
    assert not np.allclose(first[1], conformalize(y, lo, hi, "pi", cfg)[1])
```

- [ ] **Step 2: Run and confirm the failure.** `uv run pytest tests/unit/test_conformal.py -q` fails with `ModuleNotFoundError: No module named 'hcp.conformal'`.

- [ ] **Step 3: Implement `src/hcp/conformal.py`**

```python
# ABOUTME: Conformal controllers for the US interval: raw base forecaster, PI, and PID with Theta scorecaster.
# ABOUTME: Thin adapter over core.methods using the paper's cqr-asymmetric scores, alpha/2 per side (D4).
"""Conformal stage (decision D4).

Mirrors ``tests/base_test.py`` for ``score_function_name = "cqr-asymmetric"``: scores are
``[q_lo - y, y - q_hi]``, each side runs its controller at ``alpha / 2``, and the set is
``[q_lo - offset_lo, q_hi + offset_hi]``. The controllers come from ``core.methods``
unchanged. ``pid_theta`` runs inside a throw-away working directory, because core caches
Theta scorecasts under ``./.cache/scorecaster/`` and would otherwise reuse stale ones.
"""

import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.methods import quantile_integrator_log, quantile_integrator_log_scorecaster


@dataclass(frozen=True)
class ControllerConfig:
    """Controller hyperparameters; defaults are the paper's COVID configs."""

    alpha: float = 0.2
    lr: float = 0.1
    Csat: float = 2.0
    KI: float = 1000.0
    T_burnin: int = 5
    ahead: int = 4


def _raw(scores: np.ndarray, _cfg: ControllerConfig, _upper: bool) -> np.ndarray:
    return np.zeros(len(scores))


def _pi(scores: np.ndarray, cfg: ControllerConfig, _upper: bool) -> np.ndarray:
    return quantile_integrator_log(
        scores, cfg.alpha / 2, cfg.lr, cfg.Csat, cfg.KI, cfg.ahead, cfg.T_burnin
    )["q"]


def _pid_theta(scores: np.ndarray, cfg: ControllerConfig, upper: bool) -> np.ndarray:
    no_scorecasts = pd.DataFrame(index=np.arange(len(scores)))
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        result = quantile_integrator_log_scorecaster(
            scores,
            cfg.alpha / 2,
            cfg.lr,
            no_scorecasts,
            cfg.T_burnin,
            cfg.Csat,
            cfg.KI,
            upper,
            cfg.ahead,
            config_name="pid_theta",
        )
    return result["q"]


CONTROLLERS: dict[str, Callable[[np.ndarray, ControllerConfig, bool], np.ndarray]] = {
    "raw": _raw,
    "pi": _pi,
    "pid_theta": _pid_theta,
}


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """Asymmetric CQR scores ``[T, 2]``: ``[q_lo - y, y - q_hi]``."""
    return np.stack([q_lo - y, y - q_hi], axis=1)


def conformalize(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    controller: str,
    cfg: ControllerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conformalized interval ``(lo, hi)`` for one series.

    Raises:
        ValueError: On an unknown controller or a non-finite input.
    """
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if not (np.isfinite(y).all() and np.isfinite(q_lo).all() and np.isfinite(q_hi).all()):
        raise ValueError("truth and base quantiles must be finite")
    scores = cqr_scores(y, q_lo, q_hi)
    offset_lo = CONTROLLERS[controller](scores[:, 0], cfg, False)
    offset_hi = CONTROLLERS[controller](scores[:, 1], cfg, True)
    return q_lo - offset_lo, q_hi + offset_hi
```

- [ ] **Step 4: Run the tests.** `uv run pytest tests/unit/test_conformal.py -q` should report 15 passed. If statsmodels emits warnings from the Theta fits, report `DONE_WITH_CONCERNS` and list them verbatim. Don't silence them; the controller decides.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests/unit && uv run ruff format src tests/unit
git add src/hcp/conformal.py tests/unit/test_conformal.py
git commit -m "feat(conformal): adapt core P/PI/PID controllers to cqr-asymmetric US intervals"
```

---

### Task 5: Evaluate and stats [haiku]

**Files:**
- Create: `src/hcp/evaluate.py`, `src/hcp/stats.py`, `tests/unit/test_evaluate.py`, `tests/unit/test_stats.py`

**Interfaces:**
- Produces:
  - `coverage_indicator(y, lo, hi) -> bool array`
  - `rolling_coverage(hit, window) -> np.ndarray`
  - `longest_miss_run(hit) -> int`
  - `interval_score(y, lo, hi, alpha) -> np.ndarray`
  - `METRIC_NAMES: tuple[str, ...]`
  - `evaluate(y, lo, hi, *, alpha, window) -> dict[str, float]`
  - `Contrast(estimate, low, high, p_value)`
  - `draw_blocks(n, block, n_boot, rng) -> np.ndarray [n_boot, n]`
  - `summarize_diff(estimate, diffs, level=0.95) -> Contrast`
  - `holm(p_values) -> np.ndarray`

- [ ] **Step 1: Write the failing tests in `tests/unit/test_evaluate.py`**

```python
# ABOUTME: Unit tests for temporal-coverage metrics against hand-computed values.
# ABOUTME: Covers inclusive coverage, rolling windows, longest miss run, interval score and the metric dict.
import numpy as np
import pytest

from hcp.evaluate import (
    METRIC_NAMES,
    coverage_indicator,
    evaluate,
    interval_score,
    longest_miss_run,
    rolling_coverage,
)

Y = np.array([5.0, 0.0, 12.0, 5.0])
LO = np.array([0.0, 1.0, 0.0, 0.0])
HI = np.array([10.0, 10.0, 10.0, 10.0])


def test_coverage_is_inclusive() -> None:
    hit = coverage_indicator(np.array([1.0, 2, 3, 4]), np.array([1.0, 0, 4, 0]), np.array([2.0, 1, 5, 4]))
    np.testing.assert_array_equal(hit, [True, False, False, True])


def test_rolling_coverage_valid_windows() -> None:
    hit = np.array([True, False, True, True])
    np.testing.assert_allclose(rolling_coverage(hit, 2), [0.5, 0.5, 1.0])


@pytest.mark.parametrize("window", [0, 5])
def test_rolling_window_out_of_range_raises(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_coverage(np.ones(4, dtype=bool), window)


def test_longest_miss_run() -> None:
    hit = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert longest_miss_run(hit) == 3
    assert longest_miss_run(np.ones(5, dtype=bool)) == 0


def test_interval_score_hand_values() -> None:
    np.testing.assert_allclose(interval_score(Y[:3], LO[:3], HI[:3], 0.2), [10.0, 19.0, 30.0])


def test_evaluate_hand_values() -> None:
    metrics = evaluate(Y, LO, HI, alpha=0.2, window=2)
    assert tuple(metrics) == METRIC_NAMES
    assert metrics["coverage_deviation"] == pytest.approx((0.3 + 0.8 + 0.3) / 3)
    assert metrics["longest_miss_run"] == 2.0
    assert metrics["marginal_coverage"] == 0.5
    assert metrics["mean_width"] == 9.75
    assert metrics["median_width"] == 10.0
    assert metrics["interval_score"] == pytest.approx(17.25)
```

- [ ] **Step 2: Write the failing tests in `tests/unit/test_stats.py`**

```python
# ABOUTME: Unit tests for the moving-block bootstrap, contrast summaries and Holm adjustment.
# ABOUTME: Checks block contiguity, determinism, interval/p-value arithmetic and a known Holm example.
import numpy as np
import pytest

from hcp.stats import draw_blocks, holm, summarize_diff


def test_draw_blocks_shape_range_and_contiguity() -> None:
    idx = draw_blocks(10, 3, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() <= 9
    blocks = idx[:, :9].reshape(50, 3, 3)
    assert (np.diff(blocks, axis=-1) == 1).all()


def test_draw_blocks_is_deterministic_for_a_seed() -> None:
    a = draw_blocks(20, 4, 30, np.random.default_rng(7))
    b = draw_blocks(20, 4, 30, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("block", [0, 11])
def test_draw_blocks_invalid_block_raises(block: int) -> None:
    with pytest.raises(ValueError, match="block"):
        draw_blocks(10, block, 5, np.random.default_rng(0))


def test_summarize_diff_symmetric_null() -> None:
    contrast = summarize_diff(0.501, np.linspace(-1.0, 1.0, 1001))
    assert contrast.estimate == 0.501
    assert contrast.low == pytest.approx(-0.95, abs=1e-9)
    assert contrast.high == pytest.approx(0.95, abs=1e-9)
    assert contrast.p_value == pytest.approx(0.5, abs=0.01)


def test_summarize_diff_detects_a_shift() -> None:
    diffs = 2.0 + np.random.default_rng(0).normal(scale=0.1, size=500)
    contrast = summarize_diff(2.0, diffs)
    assert contrast.low > 0
    assert contrast.p_value == pytest.approx(1 / 501)


def test_holm_known_example() -> None:
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
```

- [ ] **Step 3: Run and confirm the failure.** `uv run pytest tests/unit/test_evaluate.py tests/unit/test_stats.py -q` fails with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/hcp/evaluate.py`**

```python
# ABOUTME: Temporal-coverage metrics for one conformalized series: rolling deviation, miss runs, width, score.
# ABOUTME: The primary endpoint is coverage_deviation, the mean |rolling coverage - (1 - alpha)| (D10).
"""Evaluation stage (decisions D10, D11)."""

import numpy as np

METRIC_NAMES = (
    "coverage_deviation",
    "longest_miss_run",
    "marginal_coverage",
    "mean_width",
    "median_width",
    "interval_score",
)


def coverage_indicator(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where ``lo <= y <= hi``."""
    return (lo <= y) & (y <= hi)


def rolling_coverage(hit: np.ndarray, window: int) -> np.ndarray:
    """Coverage over each full trailing window, length ``len(hit) - window + 1``.

    Raises:
        ValueError: If ``window`` is not in ``[1, len(hit)]``.
    """
    if not 1 <= window <= len(hit):
        raise ValueError(f"window must be in [1, {len(hit)}], got {window}")
    return np.convolve(hit.astype(float), np.ones(window) / window, mode="valid")


def longest_miss_run(hit: np.ndarray) -> int:
    """Length of the longest run of consecutive misses."""
    longest = current = 0
    for covered in hit:
        current = 0 if covered else current + 1
        longest = max(longest, current)
    return longest


def interval_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float
) -> np.ndarray:
    """Gneiting-Raftery interval score of the central ``1 - alpha`` interval."""
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return (hi - lo) + 2 / alpha * below + 2 / alpha * above


def evaluate(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, window: int
) -> dict[str, float]:
    """All metrics in ``METRIC_NAMES`` order."""
    hit = coverage_indicator(y, lo, hi)
    width = hi - lo
    return {
        "coverage_deviation": float(
            np.mean(np.abs(rolling_coverage(hit, window) - (1 - alpha)))
        ),
        "longest_miss_run": float(longest_miss_run(hit)),
        "marginal_coverage": float(hit.mean()),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score(y, lo, hi, alpha))),
    }
```

- [ ] **Step 5: Implement `src/hcp/stats.py`**

```python
# ABOUTME: Paired moving-block bootstrap over weeks, percentile intervals, shift p-values and Holm adjustment.
# ABOUTME: Resample indices are drawn once per block length and shared by every arm (D12).
"""Statistics stage (decision D12)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contrast:
    """Treatment minus control for one metric."""

    estimate: float
    low: float
    high: float
    p_value: float


def draw_blocks(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices ``[n_boot, n]`` built from contiguous blocks.

    Raises:
        ValueError: If ``block`` is not in ``[1, n]``.
    """
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def summarize_diff(estimate: float, diffs: np.ndarray, level: float = 0.95) -> Contrast:
    """Percentile interval and two-sided shift p-value for a bootstrapped difference.

    The p-value centers the bootstrap differences at zero and counts how often
    ``|centered| >= |estimate|``, with the +1 correction.
    """
    tail = (1 - level) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    centered = diffs - diffs.mean()
    p_value = (np.sum(np.abs(centered) >= abs(estimate)) + 1) / (len(diffs) + 1)
    return Contrast(float(estimate), float(low), float(high), float(p_value))


def holm(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
```

- [ ] **Step 6: Run the tests.** `uv run pytest tests/unit/test_evaluate.py tests/unit/test_stats.py -q` should report all passed (14 tests).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src tests/unit && uv run ruff format src tests/unit
git add src/hcp/evaluate.py src/hcp/stats.py tests/unit/test_evaluate.py tests/unit/test_stats.py
git commit -m "feat(evaluate): add temporal-coverage metrics and block-bootstrap contrasts"
```

---

### Task 6: Report [sonnet]

**Files:**
- Create: `src/hcp/report.py`, `tests/unit/test_report.py`

**Interfaces:**
- Consumes: `coverage_indicator` and `rolling_coverage` (Task 5).
- Produces: `render_figures(intervals: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path, *, alpha: float, window: int, lr: float, block: int) -> list[Path]`.
  - The `intervals` columns are `week, reconciler, controller, lr, y, q_lo, q_med, q_hi, lo, hi`.
  - The `contrasts` columns are `block, controller, lr, treatment, control, metric, estimate, low, high, p_value, p_holm, primary`.
  - It writes `rolling_coverage_<controller>.pdf` and `intervals_<controller>.pdf` for each controller present, using lr 0.0 for `raw` and `lr` otherwise. It also writes `contrasts_coverage_deviation.pdf`.

The implementer should load the `dataviz` skill before Step 3. The palette in Step 3 is the default; keep it unless dataviz's validator rejects it.

- [ ] **Step 1: Write the failing test in `tests/unit/test_report.py`**

```python
# ABOUTME: Unit tests for the report stage: every expected PDF is written and non-empty.
# ABOUTME: Uses a small synthetic intervals/contrasts table in the run stage's column schema.
from pathlib import Path

import numpy as np
import pandas as pd

from hcp.report import render_figures


def _intervals() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2021-01-02", periods=30, freq="7D")
    frames = []
    for reconciler in ("none", "bottom_up", "mint_shrink"):
        for controller, lr in (("raw", 0.0), ("pi", 0.1)):
            y = rng.normal(size=30)
            frames.append(
                pd.DataFrame(
                    {
                        "week": weeks, "reconciler": reconciler, "controller": controller,
                        "lr": lr, "y": y, "q_lo": y - 1, "q_med": y, "q_hi": y + 1,
                        "lo": y - rng.uniform(0, 2, 30), "hi": y + rng.uniform(0, 2, 30),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _contrasts() -> pd.DataFrame:
    rows = [
        {
            "block": 8, "controller": controller, "lr": lr, "treatment": treatment,
            "control": "none", "metric": "coverage_deviation", "estimate": -0.01,
            "low": -0.05, "high": 0.03, "p_value": 0.4, "p_holm": 0.8, "primary": False,
        }
        for controller, lr in (("raw", 0.0), ("pi", 0.1))
        for treatment in ("bottom_up", "mint_shrink")
    ]
    return pd.DataFrame(rows)


def test_render_figures_writes_every_pdf(tmp_path: Path) -> None:
    paths = render_figures(
        _intervals(), _contrasts(), tmp_path / "figures", alpha=0.2, window=5, lr=0.1, block=8
    )
    names = sorted(path.name for path in paths)
    assert names == [
        "contrasts_coverage_deviation.pdf",
        "intervals_pi.pdf",
        "intervals_raw.pdf",
        "rolling_coverage_pi.pdf",
        "rolling_coverage_raw.pdf",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
```

- [ ] **Step 2: Run and confirm the failure.** `uv run pytest tests/unit/test_report.py -q` fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/hcp/report.py`**

```python
# ABOUTME: PDF figures for a run: rolling US coverage and intervals per controller, and a contrast forest plot.
# ABOUTME: Uses the object-oriented matplotlib API (Figure + Agg canvas) so no global pyplot state leaks.
"""Report stage. Called only by ``run.write_artifacts``."""

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hcp.evaluate import coverage_indicator, rolling_coverage

RAW_LR = 0.0
RECONCILER_COLORS = {"none": "#6b7280", "bottom_up": "#2563eb", "mint_shrink": "#d97706"}


def _arms(intervals: pd.DataFrame, controller: str, lr: float) -> dict[str, pd.DataFrame]:
    selected = intervals[(intervals["controller"] == controller) & (intervals["lr"] == lr)]
    return {name: group for name, group in selected.groupby("reconciler", sort=False)}


def _figure(width: float = 8.0, height: float = 3.5) -> Figure:
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def rolling_coverage_figure(
    intervals: pd.DataFrame, *, controller: str, lr: float, alpha: float, window: int
) -> Figure:
    """Rolling coverage of the US interval for each reconciler."""
    figure = _figure()
    axes = figure.add_subplot()
    for reconciler, group in _arms(intervals, controller, lr).items():
        hit = coverage_indicator(
            group["y"].to_numpy(), group["lo"].to_numpy(), group["hi"].to_numpy()
        )
        axes.plot(
            group["week"].to_numpy()[window - 1 :],
            rolling_coverage(hit, window),
            label=reconciler,
            color=RECONCILER_COLORS.get(reconciler),
            linewidth=1.5,
        )
    axes.axhline(1 - alpha, color="black", linestyle="--", linewidth=1)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel(f"Rolling coverage (w={window})")
    axes.set_title(f"US 4-week-ahead deaths · {controller}, lr={lr}")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def intervals_figure(intervals: pd.DataFrame, *, controller: str, lr: float) -> Figure:
    """US truth with each reconciler's conformalized interval as a band."""
    figure = _figure()
    axes = figure.add_subplot()
    truth = None
    for reconciler, group in _arms(intervals, controller, lr).items():
        weeks = group["week"].to_numpy()
        axes.fill_between(
            weeks,
            group["lo"].to_numpy(),
            group["hi"].to_numpy(),
            color=RECONCILER_COLORS.get(reconciler),
            alpha=0.25,
            label=reconciler,
            linewidth=0,
        )
        truth = (weeks, group["y"].to_numpy())
    if truth is not None:
        axes.plot(*truth, color="black", marker="o", markersize=2, linewidth=0.8, label="truth")
    axes.set_ylabel("Weekly deaths")
    axes.set_title(f"US 80% intervals · {controller}, lr={lr}")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.tight_layout()
    return figure


def contrasts_figure(contrasts: pd.DataFrame, *, block: int) -> Figure:
    """Forest plot of coverage-deviation contrasts (treatment - control) at ``block``."""
    rows = contrasts[
        (contrasts["block"] == block) & (contrasts["metric"] == "coverage_deviation")
    ].reset_index(drop=True)
    figure = _figure(6.0, 0.5 + 0.4 * max(len(rows), 1))
    axes = figure.add_subplot()
    positions = range(len(rows))
    axes.errorbar(
        rows["estimate"],
        list(positions),
        xerr=[rows["estimate"] - rows["low"], rows["high"] - rows["estimate"]],
        fmt="o",
        color="black",
        capsize=3,
    )
    axes.set_yticks(
        list(positions),
        [f"{r.controller}: {r.treatment} − {r.control}" for r in rows.itertuples()],
    )
    axes.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    axes.set_xlabel(f"Δ coverage deviation (95% block-bootstrap CI, block={block})")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def render_figures(
    intervals: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    alpha: float,
    window: int,
    lr: float,
    block: int,
) -> list[Path]:
    """Write every figure as PDF into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Figure] = {}
    for controller in intervals["controller"].unique():
        arm_lr = RAW_LR if controller == "raw" else lr
        figures[f"rolling_coverage_{controller}.pdf"] = rolling_coverage_figure(
            intervals, controller=controller, lr=arm_lr, alpha=alpha, window=window
        )
        figures[f"intervals_{controller}.pdf"] = intervals_figure(
            intervals, controller=controller, lr=arm_lr
        )
    figures["contrasts_coverage_deviation.pdf"] = contrasts_figure(contrasts, block=block)
    paths = []
    for name, figure in figures.items():
        path = out_dir / name
        figure.savefig(path)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run the test.** `uv run pytest tests/unit/test_report.py -q` should report `1 passed`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests/unit && uv run ruff format src tests/unit
git add src/hcp/report.py tests/unit/test_report.py
git commit -m "feat(report): add rolling coverage, interval and contrast figures"
```

---

### Task 7: Run orchestration, config, e2e [sonnet]

**Files:**
- Create: `src/hcp/run.py`, `configs/us_hier.toml`, `tests/unit/test_run.py`, `tests/e2e/test_run_cli.py`, `tests/integration/test_run_real.py`

**Interfaces:**
- Consumes everything above.
- Produces:
  - `RunConfig`
  - `load_config(path) -> RunConfig`
  - `ExperimentResult(intervals, metrics, contrasts, diagnostics)`
  - `run_experiment(cfg, bundle) -> ExperimentResult`
  - `write_artifacts(cfg, result, input_sha256) -> Path`
  - `main(argv: list[str] | None = None) -> int`
  - The CLI `uv run python -m hcp.run configs/us_hier.toml`.

- [ ] **Step 1: Write the failing unit tests in `tests/unit/test_run.py`**

```python
# ABOUTME: Unit tests for run configuration validation and the in-memory experiment on synthetic data.
# ABOUTME: Checks config errors, arm grid, raw/none parity with base quantiles, and the single primary contrast.
from pathlib import Path

import numpy as np
import pytest

from hcp.run import RunConfig, load_config, run_experiment
from tests.helpers import write_config

BASE = {
    "run_id": "unit",
    "root_seed": 1,
    "data_path": "unused.csv",
    "output_dir": "results",
}
SMALL = {
    "controllers": ("raw", "pi"),
    "lr_grid": (0.1,),
    "T_burnin": 2,
    "mint_warmup": 5,
    "rolling_window": 5,
    "blocks": (4,),
    "n_boot": 10,
}


def test_load_config_converts_lists_and_paths(tmp_path: Path) -> None:
    path = write_config(tmp_path / "c.toml", {**BASE, "blocks": [8, 4], "lr_grid": [0.1, 0.5]})
    cfg = load_config(path)
    assert cfg.blocks == (8, 4)
    assert cfg.lr_grid == (0.1, 0.5)
    assert isinstance(cfg.data_path, Path)


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = write_config(tmp_path / "c.toml", {**BASE, "surprise": 1})
    with pytest.raises(ValueError, match="invalid run configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "a/b"},
        {"root_seed": -1},
        {"root_seed": True},
        {"reconcilers": ("bottom_up", "mint_shrink")},
        {"reconcilers": ("none", "nope")},
        {"controllers": ("nope",)},
        {"lr": 0.3},
        {"blocks": ()},
        {"n_boot": 0},
    ],
)
def test_invalid_config_raises(override: dict) -> None:
    with pytest.raises(ValueError):
        RunConfig(**{**BASE, **override})


def test_run_experiment_arm_grid_and_parity(hier_bundle) -> None:
    cfg = RunConfig(**BASE, **SMALL)
    result = run_experiment(cfg, hier_bundle)
    n_weeks = 26 - 8
    assert len(result.intervals) == 3 * (1 + 1) * n_weeks
    assert np.isfinite(result.intervals[["lo", "hi"]].to_numpy()).all()
    raw = result.intervals.query("reconciler == 'none' and controller == 'raw'")
    np.testing.assert_array_equal(raw["lo"], hier_bundle.q[0, 8:, 0])
    np.testing.assert_array_equal(raw["hi"], hier_bundle.q[0, 8:, 2])
    assert len(result.metrics) == 6
    assert result.contrasts["primary"].sum() == 1
    assert result.diagnostics["n_weeks"] == n_weeks
    assert result.diagnostics["imputed_cells"] == 1
    assert set(result.diagnostics["crossings"]) == {"none", "bottom_up", "mint_shrink"}
```

- [ ] **Step 2: Write the failing e2e test in `tests/e2e/test_run_cli.py`**

```python
# ABOUTME: End-to-end test of the CLI on the synthetic hierarchy: TOML in, results/<run_id>/ artifacts out.
# ABOUTME: Verifies every artifact, manifest fields, the arm grid and the no-overwrite rule.
import json
from pathlib import Path

import pandas as pd
import pytest

from hcp.data import file_sha256
from hcp.run import main
from tests.helpers import write_config


def _config(tmp_path: Path, csv: Path) -> Path:
    return write_config(
        tmp_path / "e2e.toml",
        {
            "run_id": "e2e",
            "root_seed": 7,
            "data_path": str(csv),
            "output_dir": str(tmp_path / "results"),
            "expected_sha256": file_sha256(csv),
            "lr": 0.1,
            "lr_grid": [0.1, 0.5],
            "T_burnin": 2,
            "mint_warmup": 5,
            "rolling_window": 5,
            "blocks": [4],
            "n_boot": 20,
        },
    )


def test_cli_writes_every_artifact(tmp_path: Path, hier_csv: Path) -> None:
    assert main([str(_config(tmp_path, hier_csv))]) == 0
    out = tmp_path / "results" / "e2e"
    for name in ("manifest.json", "intervals.parquet", "metrics.parquet", "contrasts.parquet"):
        assert (out / name).is_file(), name
    figures = {path.name for path in (out / "figures").glob("*.pdf")}
    assert {"rolling_coverage_pi.pdf", "rolling_coverage_pid_theta.pdf", "contrasts_coverage_deviation.pdf"} <= figures
    manifest = json.loads((out / "manifest.json").read_text())
    expected_keys = {"config", "git_sha", "git_dirty", "timestamp_utc", "input_sha256", "seed", "versions", "diagnostics"}
    assert expected_keys <= manifest.keys()
    assert manifest["input_sha256"] == file_sha256(hier_csv)
    assert manifest["diagnostics"]["imputed_cells"] == 1
    intervals = pd.read_parquet(out / "intervals.parquet")
    assert len(intervals) == 3 * (1 + 2 + 2) * (26 - 8)
    contrasts = pd.read_parquet(out / "contrasts.parquet")
    primary = contrasts[contrasts["primary"]]
    assert len(primary) == 1
    assert primary["metric"].item() == "coverage_deviation"


def test_cli_refuses_to_overwrite_a_run(tmp_path: Path, hier_csv: Path) -> None:
    config = _config(tmp_path, hier_csv)
    assert main([str(config)]) == 0
    with pytest.raises(FileExistsError):
        main([str(config)])
```

- [ ] **Step 3: Write the real-data sanity test in `tests/integration/test_run_real.py`**

```python
# ABOUTME: Integration sanity check of the full experiment on the real deaths.csv (raw and PI arms only).
# ABOUTME: The unreconciled base US interval must cover roughly its nominal 80% on the evaluation window.
from pathlib import Path

import pytest

from hcp.data import load_hierarchy
from hcp.run import RunConfig, run_experiment
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]


def test_real_run_sanity() -> None:
    cfg = RunConfig(
        run_id="it", root_seed=1, data_path=REAL_DATA, output_dir=Path("unused"),
        controllers=("raw", "pi"), lr_grid=(0.1,), blocks=(8,), n_boot=50,
    )
    result = run_experiment(cfg, load_hierarchy(REAL_DATA, REAL_SHA256))
    metrics = result.metrics.set_index(["reconciler", "controller"])
    assert 0.75 <= metrics.loc[("none", "raw"), "marginal_coverage"] <= 0.9
    assert result.diagnostics["n_weeks"] > 100
    assert result.contrasts["primary"].sum() == 1
```

- [ ] **Step 4: Run and confirm the failure.** `uv run pytest tests/unit/test_run.py tests/e2e -q` fails with `ModuleNotFoundError: No module named 'hcp.run'`.

- [ ] **Step 5: Implement `src/hcp/run.py`**

```python
# ABOUTME: CLI and orchestration: TOML config -> data -> reconcile -> conformal -> evaluate -> stats -> report.
# ABOUTME: Only this module writes files; each run goes to results/<run_id>/ with a manifest and never overwrites.
"""Run stage.

Usage: ``uv run python -m hcp.run configs/us_hier.toml``.

Every arm (reconciler x controller x lr) starts at the common evaluation week given by
the MinT warm-up (D7), so all arms are scored on the same weeks. Contrasts use one set
of moving-block resamples per block length, shared by every arm (paired, D12).
"""

import argparse
import json
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

from hcp.conformal import CONTROLLERS, ControllerConfig, conformalize
from hcp.data import HierarchyBundle, file_sha256, load_hierarchy
from hcp.evaluate import METRIC_NAMES, evaluate
from hcp.reconcile import RECONCILERS, evaluation_start, reconcile_bundle
from hcp.report import RAW_LR, render_figures
from hcp.stats import draw_blocks, holm, summarize_diff

PRIMARY = {
    "treatment": "mint_shrink",
    "control": "none",
    "controller": "pi",
    "metric": "coverage_deviation",
}
COMPARISONS = (("mint_shrink", "none"), ("bottom_up", "none"))


@dataclass(frozen=True)
class RunConfig:
    """One pre-registered experiment; see docs/protocol-decisions.md."""

    run_id: str
    root_seed: int
    data_path: Path
    output_dir: Path
    expected_sha256: str | None = None
    alpha: float = 0.2
    lr: float = 0.1
    lr_grid: tuple[float, ...] = (1.0, 0.5, 0.1, 0.05)
    Csat: float = 2.0
    KI: float = 1000.0
    T_burnin: int = 5
    ahead: int = 4
    reconcilers: tuple[str, ...] = ("none", "bottom_up", "mint_shrink")
    controllers: tuple[str, ...] = ("raw", "pi", "pid_theta")
    mint_warmup: int = 20
    rolling_window: int = 10
    blocks: tuple[int, ...] = (8, 4, 12)
    n_boot: int = 2000

    def __post_init__(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a bare directory name")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")
        for name in ("lr_grid", "reconcilers", "controllers", "blocks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("data_path", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if "none" not in self.reconcilers:
            raise ValueError("reconcilers must include the 'none' control")
        if unknown := set(self.reconcilers) - set(RECONCILERS):
            raise ValueError(f"unknown reconcilers: {sorted(unknown)}")
        if unknown := set(self.controllers) - set(CONTROLLERS):
            raise ValueError(f"unknown controllers: {sorted(unknown)}")
        if self.lr not in self.lr_grid:
            raise ValueError("lr must be one of lr_grid")
        if not self.blocks:
            raise ValueError("blocks must name at least one block length")
        if self.n_boot < 1:
            raise ValueError("n_boot must be positive")


@dataclass(frozen=True)
class ExperimentResult:
    """In-memory outputs of one run."""

    intervals: pd.DataFrame
    metrics: pd.DataFrame
    contrasts: pd.DataFrame
    diagnostics: dict


def load_config(path: Path) -> RunConfig:
    """Read a TOML file into a validated ``RunConfig``.

    Raises:
        ValueError: On unknown or missing keys, or invalid values.
    """
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    try:
        return RunConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid run configuration: {exc}") from exc


def _controller_config(cfg: RunConfig, lr: float) -> ControllerConfig:
    return ControllerConfig(
        alpha=cfg.alpha, lr=lr, Csat=cfg.Csat, KI=cfg.KI, T_burnin=cfg.T_burnin, ahead=cfg.ahead
    )


def _arm(
    intervals: pd.DataFrame, reconciler: str, controller: str, lr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = intervals[
        (intervals["reconciler"] == reconciler)
        & (intervals["controller"] == controller)
        & (intervals["lr"] == lr)
    ]
    return rows["y"].to_numpy(), rows["lo"].to_numpy(), rows["hi"].to_numpy()


def _contrasts(cfg: RunConfig, intervals: pd.DataFrame) -> pd.DataFrame:
    n = intervals["week"].nunique()
    children = np.random.SeedSequence(cfg.root_seed).spawn(len(cfg.blocks))
    rows = []
    for block, child in zip(cfg.blocks, children, strict=True):
        resamples = draw_blocks(n, block, cfg.n_boot, np.random.default_rng(child))
        for controller in cfg.controllers:
            lr = RAW_LR if controller == "raw" else cfg.lr
            arms = {r: _arm(intervals, r, controller, lr) for r in cfg.reconcilers}
            full = {
                r: evaluate(*arm, alpha=cfg.alpha, window=cfg.rolling_window)
                for r, arm in arms.items()
            }
            boot = {
                r: pd.DataFrame(
                    [
                        evaluate(y[ix], lo[ix], hi[ix], alpha=cfg.alpha, window=cfg.rolling_window)
                        for ix in resamples
                    ]
                )
                for r, (y, lo, hi) in arms.items()
            }
            for treatment, control in COMPARISONS:
                if treatment not in arms:
                    continue
                for metric in METRIC_NAMES:
                    contrast = summarize_diff(
                        full[treatment][metric] - full[control][metric],
                        (boot[treatment][metric] - boot[control][metric]).to_numpy(),
                    )
                    rows.append(
                        {
                            "block": block, "controller": controller, "lr": lr,
                            "treatment": treatment, "control": control, "metric": metric,
                            **asdict(contrast),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["primary"] = (frame["block"] == cfg.blocks[0]) & np.logical_and.reduce(
        [frame[key] == value for key, value in PRIMARY.items()]
    )
    frame["p_holm"] = frame["p_value"]
    secondary = ~frame["primary"]
    frame.loc[secondary, "p_holm"] = (
        frame[secondary]
        .groupby(["block", "controller"])["p_value"]
        .transform(lambda p: holm(p.to_numpy()))
    )
    return frame


def run_experiment(cfg: RunConfig, bundle: HierarchyBundle) -> ExperimentResult:
    """Run every arm in memory and compute metrics and contrasts."""
    t0 = evaluation_start(bundle, cfg.mint_warmup)
    weeks = bundle.dates[t0:]
    y = bundle.y[0, t0:]
    gap = bundle.y[0] - bundle.y[1:].sum(axis=0)
    diagnostics: dict = {
        "n_weeks": int(len(weeks)),
        "evaluation_start": str(pd.Timestamp(weeks[0]).date()),
        "imputed_cells": int(bundle.imputed.sum()),
        "truth_incoherence": {
            "median": float(np.nanmedian(gap)),
            "max_abs": float(np.nanmax(np.abs(gap))),
        },
        "crossings": {},
    }
    frames = []
    for reconciler in cfg.reconcilers:
        reconciled = reconcile_bundle(bundle, reconciler, cfg.mint_warmup)
        diagnostics["crossings"][reconciler] = reconciled.crossings
        if reconciler == "mint_shrink":
            diagnostics["mean_shrinkage"] = float(np.mean(reconciled.shrinkage[t0:]))
        q_us = reconciled.q[0, t0:, :]
        for controller in cfg.controllers:
            for lr in (RAW_LR,) if controller == "raw" else cfg.lr_grid:
                lo, hi = conformalize(
                    y, q_us[:, 0], q_us[:, 2], controller, _controller_config(cfg, lr)
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "week": weeks, "reconciler": reconciler, "controller": controller,
                            "lr": lr, "y": y, "q_lo": q_us[:, 0], "q_med": q_us[:, 1],
                            "q_hi": q_us[:, 2], "lo": lo, "hi": hi,
                        }
                    )
                )
    intervals = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(
        [
            {
                "reconciler": reconciler, "controller": controller, "lr": lr,
                **evaluate(
                    group["y"].to_numpy(), group["lo"].to_numpy(), group["hi"].to_numpy(),
                    alpha=cfg.alpha, window=cfg.rolling_window,
                ),
            }
            for (reconciler, controller, lr), group in intervals.groupby(
                ["reconciler", "controller", "lr"], sort=False
            )
        ]
    )
    return ExperimentResult(intervals, metrics, _contrasts(cfg, intervals), diagnostics)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), *args],
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip()


def write_artifacts(cfg: RunConfig, result: ExperimentResult, input_sha256: str) -> Path:
    """Write tables, figures and the manifest to ``output_dir / run_id``.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    out = cfg.output_dir / cfg.run_id
    out.mkdir(parents=True, exist_ok=False)
    result.intervals.to_parquet(out / "intervals.parquet", index=False)
    result.metrics.to_parquet(out / "metrics.parquet", index=False)
    result.contrasts.to_parquet(out / "contrasts.parquet", index=False)
    render_figures(
        result.intervals, result.contrasts, out / "figures",
        alpha=cfg.alpha, window=cfg.rolling_window, lr=cfg.lr, block=cfg.blocks[0],
    )
    manifest = {
        "config": asdict(cfg),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input_sha256": input_sha256,
        "seed": {"root_seed": cfg.root_seed, "block_children": len(cfg.blocks)},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "primary": PRIMARY,
        "diagnostics": result.diagnostics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = argparse.ArgumentParser(
        description="Hierarchical reconciliation + conformal PID on US COVID-19 deaths"
    )
    parser.add_argument("config", type=Path)
    cfg = load_config(parser.parse_args(argv).config)
    if (cfg.output_dir / cfg.run_id).exists():
        raise FileExistsError(f"{cfg.output_dir / cfg.run_id} exists; choose a new run_id")
    bundle = load_hierarchy(cfg.data_path, cfg.expected_sha256)
    result = run_experiment(cfg, bundle)
    out = write_artifacts(cfg, result, file_sha256(cfg.data_path))
    print(f"Artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Create `configs/us_hier.toml`**

```toml
# Pre-registered primary run (docs/protocol-decisions.md D1-D12).
# Paths are relative to the project root: run `uv run python -m hcp.run configs/us_hier.toml` from there.
run_id = "us-hier-v1"
root_seed = 20260929
data_path = "data/deaths.csv"
expected_sha256 = "cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06"
output_dir = "results"
alpha = 0.2
lr = 0.1
lr_grid = [1.0, 0.5, 0.1, 0.05]
Csat = 2.0
KI = 1000.0
T_burnin = 5
ahead = 4
reconcilers = ["none", "bottom_up", "mint_shrink"]
controllers = ["raw", "pi", "pid_theta"]
mint_warmup = 20
rolling_window = 10
blocks = [8, 4, 12]
n_boot = 2000
```

- [ ] **Step 7: Run the tests.**
  - `uv run pytest -q` should pass everything, including 12 tests in `test_run.py` and 2 in e2e.
  - `uv run pytest -m integration -q` should pass. `test_run_real` takes a few minutes because of the real-data parse and bootstrap.

- [ ] **Step 8: Lint (scoped) and commit**

```bash
uv run ruff check src tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py && uv run ruff format src tests/unit tests/integration tests/e2e tests/helpers.py tests/conftest.py
git add src/hcp/run.py configs/us_hier.toml tests/unit/test_run.py tests/e2e/test_run_cli.py tests/integration/test_run_real.py
git commit -m "feat(run): orchestrate reconciliation x controller arms with manifest and CLI"
```


---

### Task 8: Docs [haiku]

**Files:**
- Create: `AGENTS.md`, `docs/protocol-decisions.md`, `docs/development.md`
- Modify: `README.md` (append one section)

- [ ] **Step 1: `docs/protocol-decisions.md`.** Copy the D1–D12 tables from `docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md` verbatim under the heading `# Protocol decisions` and a `## 2026-09-29` subheading. Then add these three entries, which the implementation made concrete:
  - **D6a:** Leading weeks in which any leaf has never been forecast are trimmed, since LOCF has nothing to carry.
  - **D7a:** Every arm, controller and metric starts at `evaluation_start(bundle, mint_warmup)`, and controllers are re-initialized there.
  - **D13:** `pid_theta` runs in a temporary working directory, so core's `.cache/scorecaster` is never reused across runs.

  End the file with this rule: "New decisions are appended with a date; old ones are never deleted. This file overrides plans."

- [ ] **Step 2: `docs/development.md`**

````markdown
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
````

- [ ] **Step 3: `AGENTS.md`**

```markdown
# AGENTS.md

Hierarchical reconciliation (state → US) before Conformal PID (Angelopoulos et al., 2023).
Design: docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md. Decisions: docs/protocol-decisions.md (overrides plans).

## Layout
- `core/` — upstream conformal methods. Import, never reimplement. Changes only behind tests/unit/test_core_characterization.py.
- `src/hcp/` — flat package, one module per stage: data → reconcile → conformal → evaluate → stats → report, orchestrated by run.py. Modules 150–300 lines; split above ~400.
- `configs/*.toml` — one file per experiment, loaded into a frozen dataclass. No Hydra.
- `results/<run_id>/` — only run.py writes; never overwritten.

## Conventions
- Registries are module-level dicts (`RECONCILERS`, `CONTROLLERS`); configs name components by key.
- Numeric modules do no I/O. Randomness only via `np.random.SeedSequence(root_seed)`.
- No look-ahead: anything used for target t must be known by `forecast_dates[t]`.
- Type hints and Google docstrings on public functions; frozen dataclasses; two `# ABOUTME:` lines per file.
- Dependencies only via `uv add`. Conventional Commits; commit locally, never push, never `--no-verify`.

## Definition of done
Tests green and clean (unit, integration, e2e); component referenced from a config by name; outputs under `results/<run_id>/` with the manifest.
```

- [ ] **Step 4: Append to `README.md`**

```markdown
## Fork: hierarchical reconciliation (state → US)

This fork asks whether reconciling the COVIDhub ensemble's 4-week-ahead death quantiles across
the US → 56-leaf hierarchy (bottom-up, MinT(Shrink)) before conformal P/PI/PID control improves
the temporal coverage of the US series. See `docs/development.md` to run it and
`docs/protocol-decisions.md` for the pre-registered protocol.
```

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md docs/protocol-decisions.md docs/development.md README.md
git commit -m "docs: add agent conventions, protocol decisions and development guide"
```

---

### Final: whole-branch review and first real run (controller)

- [ ] Run `uv run pytest -o addopts='' -q` and the scoped ruff checks. Both must be clean.
- [ ] Run one opus code review over `git diff develop...HEAD` using the `code-review` plugin.
- [ ] Run `uv run python -m hcp.run configs/us_hier.toml`. Check that `none × raw` marginal coverage is about 0.83, and that `diagnostics.crossings` and `imputed_cells` are plausible.
- [ ] Hand off to `/analyze-results` to interpret the results. That happens only after the user has seen the manifest.
