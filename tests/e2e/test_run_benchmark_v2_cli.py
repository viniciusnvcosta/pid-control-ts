# ABOUTME: Exercises the v2 CLI publication path with synthetic adapter output.
# ABOUTME: Keeps all data and result identities in a temporary directory.
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import hcp.run as run_module
from hcp.benchmark import BenchmarkResult
from hcp.config import RunConfig
from hcp.forecast_types import DiagnosticsTables

MODELS = (
    "arima",
    "sarima",
    "deepar",
    "prophet",
    "transformer",
    "qr_l1",
    "deepar_enhanced",
    "covidhub",
)


def test_v2_cli_writes_all_models_cells_tables_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    file_hashes = {
        "deaths.csv": "cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06",
        "model_benchmarks_v2.toml": "c" * 64,
        "uv.lock": "d" * 64,
    }
    cfg = RunConfig(
        run_id="model-benchmarks-v2",
        root_seed=20261021,
        data_path=Path("data/deaths.csv"),
        output_dir=tmp_path / "results",
        expected_sha256="cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06",
        design="benchmark",
        controllers=("raw", "pi", "pi_norm"),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        lr=0.1,
        lr_grid=(0.1,),
        benchmark={"protocol_version": 2},
    )
    monkeypatch.setattr(run_module, "load_config", lambda path: cfg)
    git_checks = []

    def checked_git(repo):
        git_checks.append("a" * 40)
        return git_checks[-1]

    monkeypatch.setattr(run_module, "require_clean_git", checked_git)
    monkeypatch.setattr(
        run_module,
        "file_sha256",
        lambda path: file_hashes[Path(path).name],
    )
    monkeypatch.setattr(run_module, "read_forecasts", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        run_module,
        "canonical_actuals",
        lambda frame: SimpleNamespace(observed=np.ones((1, 1), dtype=bool)),
    )
    monkeypatch.setattr(
        run_module,
        "reference_bundle",
        lambda *args: SimpleNamespace(
            dates=np.array(["2025-01-04"], dtype="datetime64[D]"),
            forecast_dates=np.array(["2024-12-09"], dtype="datetime64[D]"),
        ),
    )
    monkeypatch.setattr(
        run_module, "benchmark_configs", lambda cfg: (object(), object())
    )
    monkeypatch.setattr(
        run_module,
        "generate_model_bundles",
        lambda *args: SimpleNamespace(
            bundles={model: object() for model in MODELS[:-1]},
            diagnostics={"seed": {"root_seed": cfg.root_seed, "spawn_key": [0]}},
        ),
    )
    # This case isolates the existing final-writer boundary; resume is covered below.
    monkeypatch.setattr(run_module, "plan_model_forecasts", lambda *args: object())
    monkeypatch.setattr(run_module, "_v2_execution_identity", lambda *args: object())

    def generated_v2(*args, **kwargs):
        workdir = args[3]
        workdir.mkdir(parents=True)
        return run_module.generate_model_bundles(), object()

    monkeypatch.setattr(run_module, "_generate_or_resume_v2", generated_v2)
    cells = [
        {
            "model": model,
            "reconciler": reconciler,
            "controller": controller,
            "role": (
                "principal"
                if (reconciler, controller)
                in {
                    ("none", "raw"),
                    ("none", "pi_norm"),
                    ("mint_median", "raw"),
                    ("mint_median", "pi_norm"),
                }
                else "descriptive"
            ),
        }
        for model in MODELS
        for reconciler in cfg.reconcilers
        for controller in cfg.controllers
    ]
    frame = pd.DataFrame(cells)
    hierarchy = pd.DataFrame(
        [
            {"model": model, "family_size": 96, "p_holm": 1.0}
            for model in MODELS
            for _level in range(2)
            for _controller in range(3)
            for _metric in range(2)
        ]
    )
    model_contrasts = pd.DataFrame(
        [
            {"model": model, "family_size": 56, "p_holm": 1.0}
            for model in MODELS
            for _ in range(7)
        ]
    )
    diagnostics_tables = DiagnosticsTables(
        tuning=pd.DataFrame(
            [
                {
                    "model": "qr_l1",
                    "node": "CA",
                    "forecast_date": "2024-01-01",
                    "candidate": 0.1,
                    "selected": True,
                    "score": 1.0,
                }
            ]
        ),
        quantile_repairs=pd.DataFrame(
            [
                {
                    "model": "qr_l1",
                    "node": "CA",
                    "forecast_date": "2024-01-01",
                    "stage": "native",
                    "before": [2.0, 1.0, 3.0],
                    "after": [1.5, 1.5, 3.0],
                    "count": 1,
                }
            ]
        ),
    )
    fake_result = BenchmarkResult(
        frame,
        frame,
        frame,
        frame,
        hierarchy,
        model_contrasts,
        {
            "protocol_version": 2,
            "model_order": list(MODELS),
            "n_weeks": 60,
            "pi_scales": {"US": 10.0},
        },
        diagnostics_tables,
    )
    monkeypatch.setattr(
        run_module, "run_benchmark", lambda *args, **kwargs: fake_result
    )
    monkeypatch.setattr(
        run_module,
        "benchmark_figures",
        lambda *args, **kwargs: {f"figure-{index}": plt.figure() for index in range(5)},
    )
    events = []
    original_write_text = Path.write_text

    def observe_manifest_write(path, *args, **kwargs):
        if path.name == "manifest.json":
            assert len(list(path.parent.glob("*.parquet"))) == 10
            assert len(list((path.parent / "figures").glob("*.pdf"))) == 5
            events.append("manifest")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", observe_manifest_write)
    original_publish = run_module._rename_noreplace

    def observe_publish(staging, destination):
        assert events == ["manifest"]
        assert (staging / "manifest.json").is_file()
        assert len(list(staging.glob("*.parquet"))) == 10
        assert len(list((staging / "figures").glob("*.pdf"))) == 5
        events.append("publish")
        original_publish(staging, destination)

    monkeypatch.setattr(run_module, "_rename_noreplace", observe_publish)

    assert run_module.main([str(tmp_path / "config.toml")]) == 0

    out = cfg.output_dir / cfg.run_id
    assert len(pd.read_parquet(out / "metrics.parquet")) == 96
    assert len(pd.read_parquet(out / "hierarchy_contrasts.parquet")) == 96
    assert len(pd.read_parquet(out / "model_contrasts.parquet")) == 56
    assert len(list(out.glob("*.parquet"))) == 10
    assert len(list((out / "figures").glob("*.pdf"))) == 5
    metrics = pd.read_parquet(out / "metrics.parquet")
    roles = metrics.groupby(["model", "role"]).size().unstack(fill_value=0)
    assert set(roles.index) == set(MODELS)
    assert (roles["principal"] == 4).all()
    assert (roles["descriptive"] == 8).all()
    principal_pairs = {
        ("none", "raw"),
        ("none", "pi_norm"),
        ("mint_median", "raw"),
        ("mint_median", "pi_norm"),
    }
    for model in MODELS:
        model_rows = metrics[metrics["model"] == model]
        assert (
            set(
                model_rows.loc[
                    model_rows["role"] == "principal", ["reconciler", "controller"]
                ].itertuples(index=False, name=None)
            )
            == principal_pairs
        )
    assert events == ["manifest", "publish"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["initial_git_sha"] == "a" * 40
    assert manifest["final_git_sha"] == manifest["initial_git_sha"]
    assert manifest["git_sha"] == manifest["initial_git_sha"]
    assert len(git_checks) == 3
    assert manifest["diagnostics"]["benchmark"]["model_order"] == list(MODELS)
    assert manifest["seed"]["streams"]["forecast"]["spawn_key"] == [0]
    assert manifest["seed"]["streams"]["evaluation"]["spawn_key"] == [1]
    assert manifest["families"]["hierarchy"]["size"] == 96
    assert manifest["families"]["model"]["size"] == 56
    assert manifest["execution"] == {
        "device": "cpu",
        "devices": 1,
        "data_loader_workers": 0,
        "deterministic_algorithms": True,
    }
    assert manifest["provenance"] == {
        "config_sha256": "c" * 64,
        "lockfile_sha256": "d" * 64,
    }
    for key, distribution in {
        "darts": "darts",
        "torch": "torch",
        "lightning": "pytorch-lightning",
        "prophet": "prophet",
        "scikit_learn": "scikit-learn",
        "matplotlib": "matplotlib",
        "pyarrow": "pyarrow",
        "scipy": "scipy",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
    }.items():
        assert manifest["versions"][key] == importlib.metadata.version(distribution)
    assert manifest["models"]["scales"] == {"US": 10.0}
    assert manifest["diagnostic_summary"]["selected_candidates"] == [
        {
            "model": "qr_l1",
            "node": "CA",
            "forecast_date": "2024-01-01",
            "candidate": 0.1,
            "selected": True,
            "score": 1.0,
        }
    ]
    assert len(manifest["diagnostic_summary"]["repairs"]) == 1
    assert pd.read_parquet(out / "warnings.parquet").empty
    assert pd.read_parquet(out / "quantile_repairs.parquet").shape[0] == 1
    assert "D15" in manifest["limitations"]


def test_v2_evaluation_scopes_tqdm_and_preserves_failed_workdir(
    tmp_path, monkeypatch, capsys
):
    import os

    import pytest

    from core import methods

    evaluate = getattr(run_module, "_evaluate_v2", None)
    assert evaluate is not None, "quiet evaluation boundary is absent"
    original_tqdm = methods.tqdm
    monkeypatch.setenv("TQDM_DISABLE", "previous")

    def benchmark(*args, **kwargs):
        assert os.environ["TQDM_DISABLE"] == "1"
        list(methods.tqdm(range(3)))
        raise RuntimeError("evaluation interrupted")

    monkeypatch.setattr(run_module, "run_benchmark", benchmark)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        evaluate({}, object(), actuals=object(), workdir=tmp_path)
    assert os.environ["TQDM_DISABLE"] == "previous"
    assert methods.tqdm is original_tqdm
    events = run_module._read_runtime_events(tmp_path)
    assert [event["event"] for event in events] == ["evaluation_start", "error"]
    captured = capsys.readouterr()
    assert "evaluation_start" in captured.out
    assert "it/s" not in captured.err and "100%" not in captured.err


def test_cli_resumes_after_two_batches_before_evaluating(tmp_path, monkeypatch, capsys):
    from dataclasses import replace

    import pytest

    from hcp.benchmark_batch import DeviceProvenance
    from hcp.benchmark_data import ActualPanel
    from hcp.config import load_config
    from hcp.forecast import FORECASTERS, ForecastConfig

    cfg = replace(
        load_config(Path("configs/model_benchmarks_v2.toml")),
        output_dir=tmp_path / "results",
    )
    real_hash = run_module.file_sha256
    monkeypatch.setattr(
        run_module,
        "file_sha256",
        lambda path: (
            cfg.expected_sha256 if Path(path) == cfg.data_path else real_hash(path)
        ),
    )
    forecast_cfg = ForecastConfig(models=("deepar",), root_seed=cfg.root_seed)
    dates = pd.date_range("2020-01-04", periods=70, freq="7D").to_numpy()
    actuals = ActualPanel(dates, ("us", "aa"), np.ones((2, 70)), np.ones((2, 70), bool))
    monkeypatch.setattr(run_module, "load_config", lambda path: cfg)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(run_module, "read_forecasts", lambda path: pd.DataFrame())
    monkeypatch.setattr(run_module, "canonical_actuals", lambda frame: actuals)
    monkeypatch.setattr(
        run_module,
        "reference_bundle",
        lambda *args: SimpleNamespace(
            dates=dates[64:68], forecast_dates=dates[64:68] - np.timedelta64(26, "D")
        ),
    )
    monkeypatch.setattr(
        run_module, "benchmark_configs", lambda cfg: (forecast_cfg, object())
    )
    monkeypatch.setattr(
        run_module,
        "_configure_v2_cuda",
        lambda execution: DeviceProvenance(
            "cuda:0",
            0,
            "GPU-test",
            "NVIDIA GeForce RTX 4080",
            (8, 9),
            "570",
            "2.8",
            "12.8",
            True,
            False,
            False,
            "highest",
            ":4096:8",
        ),
    )
    calls = []
    interrupted = True

    def adapter(histories, settings, seed, execution):
        if interrupted and len(calls) == 2:
            raise RuntimeError("interrupted")
        calls.append(seed.spawn_key)
        return np.random.default_rng(seed).normal(size=(2, 4, settings.n_samples))

    monkeypatch.setitem(FORECASTERS, "deepar", adapter)
    evaluated = []

    def evaluate(bundles, cfg, *, actuals):
        evaluated.append(bundles["deepar"].q.copy())
        return object()

    monkeypatch.setattr(run_module, "run_benchmark", evaluate)
    # Task 8 owns final publication; this test ends at that interface.
    monkeypatch.setattr(
        run_module, "_write_benchmark_v2_artifacts", lambda *args: tmp_path / "final"
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run_module.main(["configs/model_benchmarks_v2.toml"])
    assert not evaluated
    assert not (cfg.output_dir / cfg.run_id).exists()
    workdir = cfg.output_dir / f".{cfg.run_id}.partial"
    assert len(list(workdir.glob("batches/*/*.batch"))) == 2
    interrupted = False
    calls.clear()
    assert run_module.main(["configs/model_benchmarks_v2.toml"]) == 0
    assert calls == [(0, 0, 2), (0, 0, 3)]
    assert len(evaluated) == 1
    events = run_module._read_runtime_events(workdir)
    assert [event["event"] for event in events].count("evaluation_start") == 1
    assert [event["event"] for event in events].count("resume_skip") == 2
    assert len(capsys.readouterr().out.splitlines()) == 11
