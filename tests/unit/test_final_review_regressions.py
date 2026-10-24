# ABOUTME: Regressions for real adapter persistence and fail-closed v2 publication.
# ABOUTME: Exercises canonical evidence across generation, resume, and evaluation seams.
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcp import run as runtime
from hcp.benchmark import run_benchmark
from hcp.forecast import (
    ForecastConfig,
    assemble_model_bundles,
    generate_origin_batch,
    plan_model_forecasts,
)
from hcp.forecast_types import V2ModelSettings
from tests.unit.test_benchmark_v2 import MODELS, _actuals, _bundles, _config
from tests.unit.test_run_benchmark_v2 import (
    _finalization_fixture,
    _result,
    _storage_fixture,
)


@pytest.mark.parametrize("repaired", [False, True])
def test_real_qr_batch_roundtrip_and_assembly(tmp_path, monkeypatch, repaired):
    if repaired:
        from hcp import forecast_quantile

        real_fit = forecast_quantile._fit_predict

        def crossed_refit(*args, **kwargs):
            output = real_fit(*args, **kwargs)
            if len(output) == 1:
                output[0] = [output[0].max() + 1, output[0, 1], output[0].min() - 1]
            return output

        monkeypatch.setattr(forecast_quantile, "_fit_predict", crossed_refit)
    from hcp.benchmark_data import ActualPanel

    settings = V2ModelSettings()
    cfg = ForecastConfig(
        models=MODELS, protocol_version=2, settings=settings, v2_settings=settings
    )
    dates = pd.date_range("2020-01-04", periods=75, freq="7D").to_numpy()
    values = np.vstack([5 + np.sin(np.arange(75) / 4), 2 + np.cos(np.arange(75) / 3)])
    actuals = ActualPanel(dates, ("us", "aa"), values, np.isfinite(values))
    plan = plan_model_forecasts(
        actuals, dates[70:71], dates[70:71] - np.timedelta64(26, "D"), cfg
    )
    plan = replace(plan, eligible_keys=(("qr_l1", 0),))
    batch = generate_origin_batch(plan, "qr_l1", 0, cfg)
    identity = replace(_storage_fixture()[0], model_order=MODELS)
    work = tmp_path / "work"
    runtime._initialize_batch_workdir(work, identity)
    path = runtime._write_origin_batch(work, identity, batch)
    restored = runtime._read_origin_batch(
        path,
        identity,
        model=batch.model,
        target_index=0,
        origin=batch.origin,
        target=batch.target,
        nodes=batch.nodes,
        seed_spawn_key=batch.seed_spawn_key,
        source_kind=batch.source_kind,
    )
    generated = assemble_model_bundles(plan, (restored,), cfg)
    np.testing.assert_array_equal(restored.quantiles, batch.quantiles)
    pd.testing.assert_frame_equal(restored.diagnostics.tuning, batch.diagnostics.tuning)
    assert len(generated.tables.tuning) == 8
    assert generated.tables.tuning["selected"].sum() == 2
    assert set(generated.tables.tuning["candidate"]) == {"0.001", "0.01", "0.1", "1.0"}
    assert not generated.tables.forecast_diagnostics.empty
    assert np.isfinite(generated.tables.tuning["score"]).all()
    if repaired:
        assert len(generated.tables.quantile_repairs) == 2
        assert generated.tables.quantile_repairs["count"].tolist() == [1, 1]
        assert generated.tables.quantile_repairs["max_absolute_adjustment"].min() > 0


def test_nonfinite_real_evaluation_fails_with_cell_context():
    bundles = _bundles()
    with pytest.raises(
        ValueError,
        match=r"nonfinite.*model=arima.*reconciler=bottom_up.*controller=pi_norm.*node=us.*metric=mean_width",
    ):
        run_benchmark(bundles, _config(), actuals=_actuals(bundles))


def test_final_publication_composes_evidence_and_identity(tmp_path, monkeypatch):
    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)
    result = _result()
    out = runtime._finalize_benchmark_v2_artifacts(
        cfg, result, generated, summary, "datahash", Path.cwd()
    )
    assert (
        len(pd.read_parquet(out / "warnings.parquet"))
        == len(generated.tables.warnings)
        > 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for batch_manifest in out.glob("batches/*/*.batch/batch.json"):
        metadata = json.loads(batch_manifest.read_text())
        assert metadata["execution_digest"] == manifest["execution_identity_digest"]
    from hcp.benchmark_batch import execution_identity_digest

    assert manifest["execution_identity_digest"] == execution_identity_digest(
        summary.identity
    )
    assert manifest["families"]["hierarchy"]["definition"] == "(block, family)"
    assert manifest["families"]["model"]["definition"] == "(block, family)"
    events = runtime._read_runtime_events(out)
    assert events[-1]["event"] == "publication-ready"
    assert (
        manifest["publication"]["completion_evidence"]
        == "atomic no-replace rename returned successfully; reported externally"
    )


def test_nonfinite_metrics_cannot_publish_and_partial_survives(tmp_path, monkeypatch):
    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)
    bad = replace(
        _result(),
        metrics=pd.DataFrame(
            {"model": ["qr_l1"], "node": ["us"], "interval_score": [np.nan]}
        ),
    )
    with pytest.raises(ValueError, match="nonfinite.*interval_score"):
        runtime._finalize_benchmark_v2_artifacts(
            cfg, bad, generated, summary, "datahash", Path.cwd()
        )
    assert summary.workdir.is_dir()
    assert not (cfg.output_dir / cfg.run_id).exists()
    assert not any(
        event["event"] == "publication-ready"
        for event in runtime._read_runtime_events(summary.workdir)
    )


def test_projection_repairs_retain_prepost_and_coherence_without_v1_change():
    from hcp.benchmark_reconciliation import reconciliation_tables
    from hcp.data import HierarchyBundle
    from hcp.reconcile import reconcile_bundle

    dates = pd.date_range("2020-01-04", periods=40, freq="7D").to_numpy()
    q = np.repeat(
        np.array([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 50.0, 100.0]])[
            :, None, :
        ],
        40,
        axis=1,
    )
    y = q[:, :, 1] + np.sin(np.arange(40)[None, :] + np.arange(3)[:, None] * 2)
    bundle = HierarchyBundle(
        dates,
        dates - np.timedelta64(26, "D"),
        ("us", "aa", "bb"),
        y,
        q,
        np.zeros_like(y, bool),
    )
    legacy = reconcile_bundle(bundle, "mint_shrink", 5)
    captured = reconcile_bundle(bundle, "mint_shrink", 5, collect_evidence=True)
    np.testing.assert_array_equal(legacy.q, captured.q)
    assert legacy.pre_repair is None
    tables = reconciliation_tables("qr_l1", bundle, {"mint_shrink": captured})
    repairs = tables.quantile_repairs
    assert len(repairs) > 0
    assert set(repairs["stage"]) == {"reconciliation"}
    assert repairs["max_absolute_adjustment"].max() > 0
    events = tables.forecast_diagnostics
    residual = events[events["name"] == "post_repair_coherence_residual"]["value"].map(
        json.loads
    )
    assert any(np.max(np.abs(v)) > 1e-6 for v in residual)
    assert all(len(v) == 3 for v in residual)


def test_canonical_provenance_and_neural_selection_survive_arrow(tmp_path):
    from hcp.forecast_diagnostics import _diagnostic_frame, adapter_diagnostic_rows
    from hcp.forecast_types import DiagnosticsTables, NativePrediction

    output = NativePrediction(
        ("us",),
        4,
        "quantiles",
        quantiles=np.array([[0.0, 1.0, 2.0]]),
        diagnostics={
            "streams": {"training": [(0, 4, 3, 0)], "refit": [(0, 4, 3, 2)]},
            "node_seeds": {"us": 123},
            "tuning": [
                {
                    "candidate": 0,
                    "best_nll": 2.0,
                    "best_epoch": 2,
                    "epochs_run": 4,
                    "scales": np.array([10.0]),
                },
                {
                    "candidate": 1,
                    "best_nll": 1.0,
                    "best_epoch": 3,
                    "epochs_run": 5,
                    "scales": np.array([10.0]),
                },
            ],
            "final_fit": {"width": 80, "epochs": 3, "scales": np.array([12.0])},
        },
    )
    rows = adapter_diagnostic_rows(
        output.diagnostics, "deepar_enhanced", pd.Timestamp("2020-01-01")
    )
    for name, records in rows.items():
        frame = _diagnostic_frame(
            name, records, getattr(DiagnosticsTables(), name).columns
        )
        frame.to_parquet(tmp_path / f"{name}.parquet")
    tuning = pd.read_parquet(tmp_path / "tuning.parquet")
    assert tuning["selected"].tolist() == [False, True]
    events = pd.read_parquet(tmp_path / "forecast_diagnostics.parquet").set_index(
        "name"
    )
    assert json.loads(events.loc["streams", "value"])["refit"] == [[0, 4, 3, 2]]
    assert json.loads(events.loc["final_fit", "value"])["scales"] == [12.0]
    assert events.loc["final_fit", "phase"] == "full_refit"


def test_v2_evaluation_preserves_native_and_submitted_source_kind(monkeypatch):
    from hcp import panel

    monkeypatch.setattr(
        panel, "conformalize", lambda y, lo, hi, *args, **kwargs: (lo.copy(), hi.copy())
    )
    bundles = _bundles()
    result = run_benchmark(bundles, _config(), actuals=_actuals(bundles))
    kinds = result.base_forecasts.groupby("model")["source_kind"].unique()
    assert kinds["qr_l1"].tolist() == ["native_quantiles"]
    assert kinds["covidhub"].tolist() == ["submitted_quantiles"]
    assert kinds["arima"].tolist() == ["sample_derived"]


def test_lifecycle_protocol_declares_callback_and_fit_hooks():
    import inspect

    from hcp import forecast_neural_validation as validation

    protocol = getattr(validation, "NeuralLifecycle", None)
    assert protocol is not None
    assert all(
        name in protocol.__dict__
        for name in ("_finish_epoch", "_resume_training", "_fit_epochs")
    )
    assert (
        inspect.signature(validation.fit_candidate_selection)
        .parameters["candidate"]
        .annotation
        is protocol
    )
    assert (
        inspect.signature(validation.NeuralEpochCallback.__init__)
        .parameters["candidate"]
        .annotation
        is protocol
    )


@pytest.mark.parametrize("resume", [False, True])
def test_real_generation_evaluation_canonical_publication(
    tmp_path, monkeypatch, resume
):
    import warnings

    from matplotlib import pyplot as plt

    from hcp.benchmark_data import ActualPanel
    from hcp.forecast import FORECASTERS
    from hcp.forecast_types import NativePrediction
    from tests.unit.test_benchmark_v2 import NODES
    from tests.unit.test_run_benchmark_v2 import _cfg

    settings = V2ModelSettings()
    fcfg = ForecastConfig(
        models=MODELS, protocol_version=2, settings=settings, v2_settings=settings
    )
    dates = pd.date_range("2018-01-06", periods=100, freq="7D").to_numpy()
    y = np.ones((57, 100))
    y[0] = 56
    actuals = ActualPanel(dates, NODES, y, np.isfinite(y))
    plan = plan_model_forecasts(
        actuals, dates[80:88], dates[80:88] - np.timedelta64(26, "D"), fcfg
    )

    # All models retain the real generation/batch machinery. QR uses real HiGHS;
    # other architectures are narrow finite fixtures, avoiding expensive fits.
    def adapter(histories, settings, seed, execution):
        warnings.warn("seam evidence", UserWarning)
        median = np.array([h.values[-1] for h in histories])
        samples = np.broadcast_to(
            median[:, None, None], (57, 4, settings.n_samples)
        ).copy()
        return NativePrediction(
            NODES,
            4,
            "samples",
            samples=samples,
            diagnostics={
                "streams": {"training": [seed.spawn_key]},
                "tuning": [
                    {"candidate": 0, "best_nll": 1.0, "best_epoch": 1, "epochs_run": 1}
                ],
                "final_fit": {"epochs": 1},
                "quantile_repairs": [
                    {
                        "node": "us",
                        "stage": "base",
                        "before": [57.0, 56.0, 55.0],
                        "after": [55.0, 56.0, 57.0],
                        "count": 1,
                    }
                ],
            },
        )

    for model in MODELS:
        if model != "qr_l1":
            monkeypatch.setitem(FORECASTERS, model, adapter)
    cfg = _cfg(tmp_path)
    identity = replace(
        _storage_fixture()[0],
        git_sha="a" * 40,
        run_id=cfg.run_id,
        root_seed=fcfg.root_seed,
        model_order=MODELS,
    )
    work = cfg.output_dir / f".{cfg.run_id}.partial"
    generated, summary = runtime._generate_or_resume_v2(plan, fcfg, identity, work)
    if resume:
        monkeypatch.setattr(
            runtime,
            "generate_origin_batch",
            lambda *args: pytest.fail("completed batch refit during resume"),
        )
        generated, summary = runtime._generate_or_resume_v2(plan, fcfg, identity, work)
        assert summary.resumed_batches == len(plan.eligible_keys)
    bcfg = _config(weeks=2)
    bcfg = replace(
        bcfg, panel=replace(bcfg.panel, mint_warmup=2, rolling_window=2, blocks=(1,))
    )
    reference = replace(generated.bundles["qr_l1"])
    evaluated = run_benchmark(
        {**generated.bundles, "covidhub": reference}, bcfg, actuals=actuals
    )
    assert not evaluated.tables.forecast_diagnostics.empty
    monkeypatch.setattr(runtime, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(
        runtime, "benchmark_figures", lambda *args, **kwargs: {"seam": plt.figure()}
    )
    out = runtime._finalize_benchmark_v2_artifacts(
        cfg, evaluated, generated, summary, "datahash", Path.cwd()
    )
    for name in ("warnings", "quantile_repairs", "tuning", "forecast_diagnostics"):
        frame = pd.read_parquet(out / f"{name}.parquet")
        assert not frame.empty, name
        assert len(frame) == len(getattr(evaluated.tables, name)) + len(
            getattr(generated.tables, name)
        )
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["diagnostic_summary"]["selected_candidates"]) == len(
        generated.tables.tuning.query("selected == True")
    )
    for key, family in [("hierarchy", "intervention"), ("model", "model_reference")]:
        frame = (
            evaluated.hierarchy_contrasts
            if key == "hierarchy"
            else evaluated.model_contrasts
        )
        assert (
            frame.groupby(["block", "family"])
            .size()
            .eq(manifest["families"][key]["size"])
            .all()
        )


def test_failed_atomic_publication_emits_ready_without_completion(
    tmp_path, monkeypatch, capsys
):
    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)

    def fail(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(runtime, "_rename_noreplace", fail)
    with pytest.raises(OSError, match="injected"):
        runtime._finalize_benchmark_v2_artifacts(
            cfg, _result(), generated, summary, "datahash", Path.cwd()
        )
    assert (
        runtime._read_runtime_events(summary.workdir)[-1]["event"]
        == "publication-ready"
    )
    assert "publication-complete" not in capsys.readouterr().out
    manifest = json.loads((summary.workdir / "manifest.json").read_text())
    assert manifest["publication"]["state"] == "ready"
    assert (
        runtime.file_sha256(summary.workdir / "runtime.jsonl")
        == manifest["artifacts"]["runtime.jsonl"]
    )


def test_manifest_repairs_are_structured_numeric_records(tmp_path, monkeypatch):
    from hcp.forecast_types import DiagnosticsTables

    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)
    repairs = pd.DataFrame(
        [
            {
                "model": "qr_l1",
                "node": "us",
                "stage": "base",
                "before": np.array([3.0, 1.0, 2.0]),
                "after": np.array([1.0, 2.0, 3.0]),
                "count": 1,
            }
        ]
    )
    result = replace(_result(), tables=DiagnosticsTables(quantile_repairs=repairs))
    out = runtime._finalize_benchmark_v2_artifacts(
        cfg, result, generated, summary, "datahash", Path.cwd()
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["diagnostic_summary"]["repairs"][0]["before"] == [3.0, 1.0, 2.0]
