# ABOUTME: Runs every v2 forecast adapter on a reduced real-data development slice.
# ABOUTME: Checks cutoffs and real diagnostic persistence with temporary development artifacts.
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcp.benchmark_data import ActualPanel, canonical_actuals, reference_bundle
from hcp.data import FORECASTER, file_sha256, read_forecasts
from hcp.forecast import (
    ForecastConfig,
    assemble_model_bundles,
    generate_origin_batch,
    plan_model_forecasts,
)
from hcp.forecast_types import V2ModelSettings
from tests.helpers import REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings(
        "ignore:Maximum Likelihood optimization failed to converge"
    ),
    pytest.mark.filterwarnings("ignore:Non-stationary starting autoregressive"),
]

MODELS = (
    "arima",
    "sarima",
    "deepar",
    "prophet",
    "transformer",
    "qr_l1",
    "deepar_enhanced",
)


def test_all_v2_adapters_integrate_with_reduced_real_data_and_preserve_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all adapters with one real origin and development-only budgets."""
    assert file_sha256(REAL_DATA) == REAL_SHA256
    frame = read_forecasts(REAL_DATA, REAL_SHA256)
    complete = canonical_actuals(frame)
    reference = reference_bundle(frame, complete, FORECASTER)

    # Keep the US and one endpoint leaf so each fitting path sees actual data while
    # the development check remains much smaller than the registered 57-node run.
    node_indices = [complete.nodes.index(node) for node in ("us", "ca")]
    actuals = ActualPanel(
        complete.dates,
        tuple(complete.nodes[index] for index in node_indices),
        complete.y[node_indices],
        complete.observed[node_indices],
    )
    target = reference.dates[-1:]
    cutoff = reference.forecast_dates[-1:]
    output_dir = tmp_path / "results"
    development_run_id = f"task14-development-{tmp_path.name}"

    settings = replace(
        V2ModelSettings(),
        n_samples=8,
        deepar_n_epochs=1,
        neural_max_epochs=1,
        neural_patience=1,
    )
    config = ForecastConfig(
        models=MODELS,
        min_train_weeks=60,
        root_seed=20261021,
        settings=settings,
        protocol_version=2,
        v2_settings=settings,
    )
    plan = plan_model_forecasts(actuals, target, cutoff, config)
    batches = tuple(
        generate_origin_batch(plan, model, index, config)
        for model, index in plan.eligible_keys
    )
    generated = assemble_model_bundles(plan, batches, config)

    assert set(generated.bundles) == set(MODELS)
    assert generated.diagnostics["models"].keys() == set(MODELS)
    for model, bundle in generated.bundles.items():
        assert bundle.nodes == actuals.nodes
        np.testing.assert_array_equal(bundle.dates, target)
        np.testing.assert_array_equal(bundle.forecast_dates, cutoff)
        assert bundle.q.shape == (2, 1, 3)
        assert np.isfinite(bundle.q).all(), model
    assert all(target[0] > cutoff[0] for _ in MODELS)
    assert not (output_dir / development_run_id).exists()
    assert not output_dir.exists()

    tuning = generated.tables.tuning
    transformer = tuning[tuning["model"] == "transformer"]
    assert transformer["candidate"].tolist() == ["32"]
    assert transformer["selected"].tolist() == [True]
    assert np.isfinite(transformer["score"]).all()
    events = generated.tables.forecast_diagnostics
    assert set(events.loc[events["name"] == "origin_seed", "model"]) == set(MODELS)
    assert {"streams", "final_fit", "node_seeds"}.issubset(set(events["name"]))

    # Persist the real adapter tables through the same exact-resume and final
    # writer boundaries as the CLI. Expensive adapter fits above run only once.
    from matplotlib import pyplot as plt

    from hcp import run as runtime
    from tests.unit.test_run_benchmark_v2 import (
        _cfg,
        _result,
        _storage_fixture,
    )

    cfg = _cfg(tmp_path)
    identity = replace(
        _storage_fixture()[0],
        git_sha="a" * 40,
        model_order=MODELS,
        root_seed=config.root_seed,
        run_id=cfg.run_id,
    )
    work = tmp_path / "real-adapter-batches"
    runtime._initialize_batch_workdir(work, identity)
    for batch in batches:
        runtime._write_origin_batch(work, identity, batch)

    monkeypatch.setattr(
        runtime,
        "generate_origin_batch",
        lambda *args: pytest.fail("resumed real adapter must not fit again"),
    )
    resumed, summary = runtime._generate_or_resume_v2(plan, config, identity, work)
    assert summary.resumed_batches == 7
    # Canonical JSON compares nested numeric lists after Arrow restores them as
    # arrays; actual candidate/scale/stream evidence must remain unchanged.
    from hcp.forecast_diagnostics import diagnostic_records

    for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings"):
        before = diagnostic_records(getattr(generated.tables, name))
        after = diagnostic_records(getattr(resumed.tables, name))
        assert before == after
    for model in MODELS:
        np.testing.assert_array_equal(
            resumed.bundles[model].q, generated.bundles[model].q
        )
    summary = replace(summary, workdir=cfg.output_dir / f".{cfg.run_id}.partial")
    summary.workdir.parent.mkdir(parents=True, exist_ok=True)
    work.rename(summary.workdir)
    monkeypatch.setattr(runtime, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(
        runtime, "benchmark_figures", lambda *args, **kwargs: {"slice": plt.figure()}
    )
    published = runtime._finalize_benchmark_v2_artifacts(
        cfg, _result(), resumed, summary, "datahash", Path.cwd()
    )
    manifest = json.loads((published / "manifest.json").read_text())
    selected = manifest["diagnostic_summary"]["selected_candidates"]
    assert {row["model"] for row in selected} == {
        "qr_l1",
        "transformer",
        "deepar_enhanced",
    }
    assert len([row for row in selected if row["model"] == "qr_l1"]) == 2
    for model in ("transformer", "deepar_enhanced"):
        (selected_row,) = [row for row in selected if row["model"] == model]
        assert selected_row["best_epoch"] == selected_row["epochs_run"] == 1
        assert len(selected_row["scales"]) == 2
    final_events = pd.read_parquet(published / "forecast_diagnostics.parquet")
    assert diagnostic_records(final_events) == diagnostic_records(
        resumed.tables.forecast_diagnostics
    )
