# ABOUTME: Exercises the v2 CLI publication path with synthetic adapter output.
# ABOUTME: Keeps all data and result identities in a temporary directory.
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
    input_path = tmp_path / "synthetic.csv"
    input_path.write_text("synthetic")
    cfg = RunConfig(
        run_id="synthetic-v2",
        root_seed=20261021,
        data_path=input_path,
        output_dir=tmp_path / "results",
        design="benchmark",
        controllers=("raw", "pi", "pi_norm"),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        benchmark={"protocol_version": 2},
    )
    monkeypatch.setattr(run_module, "load_config", lambda path: cfg)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(run_module, "file_sha256", lambda path: "b" * 64)
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
    cells = [
        {
            "model": model,
            "reconciler": reconciler,
            "controller": controller,
            "role": "descriptive",
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
        DiagnosticsTables(),
    )
    monkeypatch.setattr(
        run_module, "run_benchmark", lambda *args, **kwargs: fake_result
    )
    monkeypatch.setattr(
        run_module,
        "benchmark_figures",
        lambda *args, **kwargs: {f"figure-{index}": plt.figure() for index in range(5)},
    )

    assert run_module.main([str(tmp_path / "config.toml")]) == 0

    out = cfg.output_dir / cfg.run_id
    assert len(pd.read_parquet(out / "metrics.parquet")) == 96
    assert len(pd.read_parquet(out / "hierarchy_contrasts.parquet")) == 96
    assert len(pd.read_parquet(out / "model_contrasts.parquet")) == 56
    assert len(list(out.glob("*.parquet"))) == 10
    assert len(list((out / "figures").glob("*.pdf"))) == 5
    manifest = __import__("json").loads((out / "manifest.json").read_text())
    assert manifest["diagnostics"]["benchmark"]["model_order"] == list(MODELS)
    assert manifest["seed"]["streams"]["forecast"]["spawn_key"] == [0]
    assert manifest["seed"]["streams"]["evaluation"]["spawn_key"] == [1]
    assert manifest["families"]["hierarchy"]["size"] == 96
    assert manifest["families"]["model"]["size"] == 56
    assert "D15" in manifest["limitations"]
