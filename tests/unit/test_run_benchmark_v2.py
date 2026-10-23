# ABOUTME: Tests v2 Git provenance and no-replace artifact publication contracts.
# ABOUTME: Uses temporary Git repositories and synthetic artifacts only.
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

import hcp.run as run_module
from hcp.benchmark import BenchmarkResult
from hcp.config import RunConfig
from hcp.forecast_types import DiagnosticsTables
from hcp.run import require_clean_git, write_benchmark_artifacts


def _repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked").write_text("base")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "base"],
        check=True,
        capture_output=True,
    )
    return path


def test_require_clean_git_returns_sha_and_rejects_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    sha = require_clean_git(repo)
    assert len(sha) == 40

    (repo / "untracked").write_text("new")
    with pytest.raises(RuntimeError, match="not clean"):
        require_clean_git(repo)
    (repo / "untracked").unlink()
    (repo / "tracked").write_text("staged")
    subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
    with pytest.raises(RuntimeError, match="not clean"):
        require_clean_git(repo)


def test_require_clean_git_fails_on_git_command_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git status --porcelain"):
        require_clean_git(tmp_path)


def _cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="v2-test",
        root_seed=20261021,
        data_path=tmp_path / "input.csv",
        output_dir=tmp_path / "results",
        design="benchmark",
        controllers=("raw", "pi", "pi_norm"),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        benchmark={"protocol_version": 2},
    )


def _result() -> BenchmarkResult:
    frame = pd.DataFrame({"model": ["arima"], "value": [1.0]})
    return BenchmarkResult(
        frame,
        frame,
        frame,
        frame,
        frame,
        frame,
        {
            "protocol_version": 2,
            "model_order": ["arima", "covidhub"],
            "pi_scales": {"US": 1.0},
        },
        DiagnosticsTables(),
    )


def test_v2_writer_publishes_ten_tables_five_figures_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(
        run_module,
        "benchmark_figures",
        lambda *args, **kwargs: {f"fig-{n}": plt.figure() for n in range(5)},
    )
    result = _result()
    out = write_benchmark_artifacts(
        cfg, result, {"seed": {"spawn_key": [0]}}, "datahash"
    )

    assert len(list(out.glob("*.parquet"))) == 10
    assert len(list((out / "figures").glob("*.pdf"))) == 5
    manifest = __import__("json").loads((out / "manifest.json").read_text())
    assert manifest["initial_git_sha"] == "a" * 40
    assert manifest["final_git_sha"] == manifest["initial_git_sha"]
    assert manifest["git_sha"] == manifest["initial_git_sha"]
    assert manifest["provenance"]["config_sha256"]
    assert manifest["provenance"]["lockfile_sha256"]
    assert manifest["families"]["hierarchy"]["size"] == 96
    assert manifest["families"]["model"]["size"] == 56
    assert "D15" in manifest["limitations"]
    assert "v1 outcomes informed v2" in manifest["analysis_note"]
    assert list(pd.read_parquet(out / "warnings.parquet").columns) == [
        "category",
        "message",
        "source",
        "model",
        "node",
        "origin",
        "candidate",
        "phase",
        "count",
    ]
    assert list(pd.read_parquet(out / "quantile_repairs.parquet").columns) == [
        "model",
        "node",
        "forecast_date",
        "stage",
        "before",
        "after",
        "count",
    ]
    assert not list(cfg.output_dir.glob(".*.partial"))


def test_v2_writer_removes_only_owned_staging_when_destination_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(run_module, "benchmark_figures", lambda *args, **kwargs: {})
    out = cfg.output_dir / cfg.run_id
    sentinel = out / "other-process.txt"
    original = run_module._rename_noreplace

    def race(staging: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        sentinel.write_text("preserve")
        original(staging, destination)

    monkeypatch.setattr(run_module, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        write_benchmark_artifacts(
            cfg, _result(), {"seed": {"spawn_key": [0]}}, "datahash"
        )
    assert sentinel.read_text() == "preserve"
    assert not list(cfg.output_dir.glob(".*.partial"))


def test_v2_writer_rejects_sha_change_and_cleans_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    shas = iter(("a" * 40, "a" * 40, "b" * 40))
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: next(shas))
    monkeypatch.setattr(run_module, "benchmark_figures", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="HEAD changed"):
        write_benchmark_artifacts(
            cfg, _result(), {"seed": {"spawn_key": [0]}}, "datahash"
        )
    assert not (cfg.output_dir / cfg.run_id).exists()
    assert not list(cfg.output_dir.glob(".*.partial"))
