# ABOUTME: CLI and orchestration: TOML config -> data -> reconcile -> conformal -> evaluate -> stats -> report.
# ABOUTME: Only this module writes files; each run goes to results/<run_id>/ with a manifest and never overwrites.
"""Run stage.

Usage: ``uv run python -m hcp.run configs/us_hier.toml``.

Two designs share this orchestrator. ``design = "us"`` (Phase 1) runs every arm
(reconciler x controller x lr) on the single US series, starting at the common
evaluation week given by the MinT warm-up (D7). ``design = "panel"`` (Phase 2) runs
the leaves x forecasters panel (D20-D26): it reads every forecaster, checks D21
eligibility, and runs the panel reconcilers (``none``, ``bottom_up``, ``mint_shrink``,
``mint_median``) and controllers (``raw``, ``pi``; ``pid_theta`` is rejected, D22) on
each eligible forecaster's own evaluation window. Contrasts use one set of
moving-block resamples per block length, shared by every arm (paired, D12).
"""

import argparse
import ctypes
import errno
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from matplotlib import pyplot as plt

# RunConfig, load_config and panel_config are re-exported as part of the run API.
from hcp.benchmark import BenchmarkResult, run_benchmark
from hcp.benchmark_batch import DeviceProvenance, NeuralExecution, device_identity
from hcp.benchmark_data import canonical_actuals, reference_bundle
from hcp.benchmark_report import benchmark_figures
from hcp.config import (
    RunConfig,
    benchmark_configs,
    config_manifest,
    load_config,
    panel_config,
)
from hcp.data import (
    FORECASTER,
    build_bundles,
    file_sha256,
    load_hierarchy,
    read_forecasts,
)
from hcp.eligibility import EligibilityRule, eligibility, require_eligible
from hcp.experiment import PRIMARY, ExperimentResult, run_experiment
from hcp.forecast import generate_model_bundles
from hcp.panel import PRIMARY as PANEL_PRIMARY
from hcp.panel import PanelResult, run_panel
from hcp.report import render_figures, render_panel_figures


def _query_nvidia_identity(device_index: int) -> Mapping[str, str]:
    """Query driver versions by UUID, independent of CUDA visibility ordering.

    Args:
        device_index: Registered visible device index.

    Raises:
        RuntimeError: If the driver query fails or returns ambiguous identities.
    """
    if device_index != 0:
        raise RuntimeError("only CUDA device zero is registered")
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if completed.returncode:
            raise RuntimeError("NVIDIA identity query failed")
        identities = {}
        for line in completed.stdout.splitlines():
            gpu_uuid, driver = (item.strip() for item in line.split(","))
            if not gpu_uuid or not driver or gpu_uuid in identities:
                raise ValueError("missing or duplicate NVIDIA identity")
            identities[gpu_uuid] = driver
        if not identities:
            raise ValueError("empty NVIDIA identity")
        return identities
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RuntimeError("NVIDIA identity query failed") from exc


def _configure_v2_cuda(execution: NeuralExecution) -> DeviceProvenance:
    """Establish strict registered CUDA settings before forecasting.

    Args:
        execution: Registered neural device and precision settings.

    Raises:
        RuntimeError: If CUDA hardware or deterministic initialization fails.
    """
    if (
        execution.device != "cuda:0"
        or execution.cublas_workspace_config != ":4096:8"
        or execution.required_gpu_name != "NVIDIA GeForce RTX 4080"
    ):
        raise RuntimeError("registered CUDA execution settings are required")
    import torch  # noqa: PLC0415 - inspect runtime state only at the launch boundary.

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA runtime is already initialized")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = execution.cublas_workspace_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA requires exactly one visible device")
    if torch.cuda.get_device_properties(0).name != execution.required_gpu_name:
        raise RuntimeError("selected GPU differs from registered GPU")
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.cuda.set_device(0)
        return device_identity(execution, torch, _query_nvidia_identity(0))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"deterministic CUDA initialization failed: {exc}") from exc


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def require_clean_git(repo: Path) -> str:
    """Return HEAD's SHA only when Git inspection succeeds and the tree is clean.

    Args:
        repo: Repository directory to inspect.

    Raises:
        RuntimeError: If Git commands fail or the worktree has changes.
    """

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"git {' '.join(args)} failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Git worktree is not clean:\n{status}")
    return git("rev-parse", "HEAD")


def _manifest(
    cfg: RunConfig,
    input_sha256: str,
    *,
    primary: dict,
    seed: dict,
    diagnostics: dict,
) -> dict:
    """Build the manifest dict shared by both designs; callers add their own keys."""
    return {
        "config": config_manifest(cfg),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input_sha256": input_sha256,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "primary": primary,
        "diagnostics": diagnostics,
    }


def write_artifacts(
    cfg: RunConfig, result: ExperimentResult, input_sha256: str
) -> Path:
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
        result.intervals,
        result.contrasts,
        out / "figures",
        alpha=cfg.alpha,
        window=cfg.rolling_window,
        lr=cfg.lr,
        block=cfg.blocks[0],
    )
    manifest = _manifest(
        cfg,
        input_sha256,
        primary=PRIMARY,
        seed={"root_seed": cfg.root_seed, "block_children": len(cfg.blocks)},
        diagnostics=result.diagnostics,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def _rename_noreplace(staging: Path, out: Path) -> None:
    """Atomically publish a directory, refusing replacement or unsupported platforms."""
    system = platform.system()
    if system == "Windows":
        staging.rename(out)  # Windows rename already refuses an existing destination.
        return
    rename = (
        getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if system == "Linux"
        else None
    )
    if rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unavailable", str(out))
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    # AT_FDCWD = -100; RENAME_NOREPLACE = 1 forbids replacing even an empty directory.
    if rename(-100, os.fsencode(staging), -100, os.fsencode(out), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(out))


def write_panel_artifacts(
    cfg: RunConfig,
    result: PanelResult,
    eligibility_table: pd.DataFrame,
    input_sha256: str,
) -> Path:
    """Stage panel artifacts, then publish them together at ``output_dir / run_id``.

    Raises:
        FileExistsError: If the run directory already exists.
        OSError: If atomic no-replace publication is unavailable or fails.
    """
    out = cfg.output_dir / cfg.run_id
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run_id")
    staging = out.with_name(f".{out.name}.partial")
    staging.mkdir(parents=True, exist_ok=False)
    result.intervals.to_parquet(staging / "intervals.parquet", index=False)
    result.metrics.to_parquet(staging / "metrics.parquet", index=False)
    result.level_metrics.to_parquet(staging / "level_metrics.parquet", index=False)
    result.contrasts.to_parquet(staging / "contrasts.parquet", index=False)
    result.summary.to_parquet(staging / "summary.parquet", index=False)
    eligibility_table.to_parquet(staging / "eligibility.parquet", index=False)
    render_panel_figures(
        result.metrics,
        result.contrasts,
        staging / "figures",
        lr=cfg.lr,
        block=cfg.blocks[0],
        primary_forecaster=FORECASTER,
    )
    manifest = _manifest(
        cfg,
        input_sha256,
        primary={"forecaster": FORECASTER, **PANEL_PRIMARY},
        seed={
            "root_seed": cfg.root_seed,
            "forecaster_children": len(cfg.forecasters) + 1,
            "block_children": len(cfg.blocks),
        },
        diagnostics=result.diagnostics,
    )
    manifest["forecasters"] = list(cfg.forecasters)
    manifest["summary_note"] = "descriptive only (D24)"
    manifest["eligibility"] = eligibility_table.to_dict(orient="records")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run_id")
    _rename_noreplace(staging, out)
    return out


def write_benchmark_artifacts(
    cfg: RunConfig,
    result: BenchmarkResult,
    forecast_diagnostics: dict,
    input_sha256: str,
) -> Path:
    """Stage and atomically publish a complete model-benchmark result.

    Args:
        cfg: Validated benchmark run configuration.
        result: Completed common-window benchmark result.
        forecast_diagnostics: Fit, warning, timing, interpolation, and seed records.
        input_sha256: SHA-256 of the source forecast file.

    Returns:
        Published result directory.

    Raises:
        FileExistsError: If the final or sibling staging directory already exists.
        OSError: If atomic no-replace publication is unavailable or fails.
    """
    if cfg.design != "benchmark" or cfg.benchmark is None:
        raise ValueError("write_benchmark_artifacts requires design 'benchmark'")
    if getattr(cfg.benchmark, "protocol_version", 1) == 2:
        repo = Path(__file__).resolve().parents[2]
        return _write_benchmark_v2_artifacts(
            cfg,
            result,
            forecast_diagnostics,
            input_sha256,
            repo,
            require_clean_git(repo),
        )
    out = cfg.output_dir / cfg.run_id
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run_id")
    staging = out.with_name(f".{out.name}.partial")
    staging.mkdir(parents=True, exist_ok=False)
    tables = {
        "base_forecasts": result.base_forecasts,
        "intervals": result.intervals,
        "metrics": result.metrics,
        "level_metrics": result.level_metrics,
        "hierarchy_contrasts": result.hierarchy_contrasts,
        "model_contrasts": result.model_contrasts,
    }
    for name, frame in tables.items():
        frame.to_parquet(staging / f"{name}.parquet", index=False)

    figures_dir = staging / "figures"
    figures_dir.mkdir()
    figures = benchmark_figures(result, block=cfg.blocks[0], lr=cfg.lr)
    try:
        for name, figure in figures.items():
            figure.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    finally:
        for figure in figures.values():
            plt.close(figure)

    settings = cfg.benchmark
    manifest = _manifest(
        cfg,
        input_sha256,
        primary={},
        seed={
            "root_seed": cfg.root_seed,
            "forecast": forecast_diagnostics["seed"],
            "evaluation": {
                "root_seed": cfg.root_seed,
                "spawn_key": list(settings.evaluation_seed_spawn_key),
                "model_order": [*settings.models, settings.reference_model],
                "block_order": list(cfg.blocks),
            },
        },
        diagnostics={
            "benchmark": result.diagnostics,
            "forecast": forecast_diagnostics,
        },
    )
    manifest["versions"].update(
        {
            "darts": importlib.metadata.version("darts"),
            "torch": importlib.metadata.version("torch"),
            "lightning": importlib.metadata.version("pytorch-lightning"),
        }
    )
    manifest["execution"] = {
        "device": "cpu",
        "devices": 1,
        "data_loader_workers": 0,
        "deterministic_algorithms": True,
    }
    manifest["models"] = {
        "classes": {
            "arima": "darts.models.ARIMA",
            "sarima": "darts.models.ARIMA",
            "deepar": "darts.models.RNNModel",
        },
        "hyperparameters": config_manifest(cfg)["benchmark"],
    }
    manifest["limitations"] = {
        "D15": (
            "Model training, MinT reconciliation errors, and evaluation use final "
            "revised truth rather than as-of vintages. PI retains core's inherited "
            "look-ahead in the learning-rate scale at ahead=4."
        )
    }
    manifest["analysis_note"] = "exploratory benchmark; no unadjusted primary"
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run_id")
    _rename_noreplace(staging, out)
    return out


def _write_benchmark_v2_artifacts(
    cfg: RunConfig,
    result: BenchmarkResult,
    forecast_diagnostics: dict,
    input_sha256: str,
    repo: Path,
    initial_sha: str,
) -> Path:
    """Write the v2 evidence bundle into owned staging and publish atomically."""
    out = cfg.output_dir / cfg.run_id
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new run_id")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        tables = {
            "base_forecasts": result.base_forecasts,
            "intervals": result.intervals,
            "metrics": result.metrics,
            "level_metrics": result.level_metrics,
            "hierarchy_contrasts": result.hierarchy_contrasts,
            "model_contrasts": result.model_contrasts,
            "forecast_diagnostics": result.tables.forecast_diagnostics,
            "tuning": result.tables.tuning,
            "quantile_repairs": result.tables.quantile_repairs,
            "warnings": result.tables.warnings,
        }
        for name, frame in tables.items():
            frame.to_parquet(staging / f"{name}.parquet", index=False)

        figures_dir = staging / "figures"
        figures_dir.mkdir()
        figures = benchmark_figures(result, block=cfg.blocks[0], lr=cfg.lr)
        try:
            for name, figure in figures.items():
                figure.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
        finally:
            for figure in figures.values():
                plt.close(figure)

        current_sha = require_clean_git(repo)
        if current_sha != initial_sha:
            raise RuntimeError(
                f"Git HEAD changed during benchmark: {initial_sha} -> {current_sha}"
            )
        final_sha = current_sha
        manifest = _manifest(
            cfg,
            input_sha256,
            primary={},
            seed={
                "root_seed": cfg.root_seed,
                "forecast": forecast_diagnostics["seed"],
                "evaluation": {
                    "root_seed": cfg.root_seed,
                    "spawn_key": list(cfg.benchmark.evaluation_seed_spawn_key),
                    "model_order": [
                        *cfg.benchmark.models,
                        cfg.benchmark.reference_model,
                    ],
                    "block_order": list(cfg.blocks),
                },
            },
            diagnostics={
                "benchmark": result.diagnostics,
                "forecast": forecast_diagnostics,
            },
        )
        manifest["git_sha"] = initial_sha
        manifest["initial_git_sha"] = initial_sha
        manifest["final_git_sha"] = final_sha
        manifest["git_dirty"] = False
        manifest["protocol_id"] = "probabilistic-model-benchmarks-v2"
        manifest["provenance"] = {
            "config_sha256": file_sha256(repo / "configs/model_benchmarks_v2.toml"),
            "lockfile_sha256": file_sha256(repo / "uv.lock"),
        }
        manifest["seed"]["streams"] = {
            "forecast": {"spawn_key": [0], "order": list(cfg.benchmark.models)},
            "evaluation": {
                "spawn_key": [1],
                "order": [*cfg.benchmark.models, cfg.benchmark.reference_model],
                "blocks": list(cfg.blocks),
            },
        }
        manifest["models"] = {
            "classes": {
                "arima": "statsmodels.tsa.arima.model.ARIMA",
                "sarima": "statsmodels.tsa.statespace.sarimax.SARIMAX",
                "deepar": "darts.models.RNNModel",
                "prophet": "prophet.forecaster.Prophet",
                "transformer": "darts.models.TransformerModel",
                "qr_l1": "sklearn.linear_model.QuantileRegressor",
                "deepar_enhanced": "darts.models.RNNModel with hcp.forecast_deepar.EnhancedRNNModule",
                "covidhub": "external forecaster COVIDhub-4_week_ensemble",
            },
            "settings": config_manifest(cfg)["benchmark"],
            "scales": result.diagnostics.get("pi_scales", {}),
        }
        manifest["diagnostic_summary"] = {
            "selected_candidates": result.tables.tuning.loc[
                result.tables.tuning.get("selected", pd.Series(dtype=bool)).fillna(
                    False
                )
            ].to_dict(orient="records"),
            "repairs": result.tables.quantile_repairs.to_dict(orient="records"),
            "warnings": result.tables.warnings.to_dict(orient="records"),
            "forecast_events": result.tables.forecast_diagnostics.to_dict(
                orient="records"
            ),
        }
        manifest["families"] = {
            "hierarchy": {
                "size": 96,
                "definition": "(block, model, level, controller, metric)",
            },
            "model": {
                "size": 56,
                "definition": "(block, level, controller, reconciler, metric)",
            },
        }
        manifest["versions"].update(
            {
                package: importlib.metadata.version(distribution)
                for package, distribution in {
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
                }.items()
            }
        )
        manifest["execution"] = {
            "device": "cpu",
            "devices": 1,
            "data_loader_workers": 0,
            "deterministic_algorithms": True,
        }
        manifest["limitations"] = {
            "D15": (
                "Model training, MinT reconciliation errors, and evaluation use final revised "
                "truth rather than as-of vintages. PI retains core's inherited learning-rate "
                "look-ahead."
            )
        }
        manifest["analysis_note"] = (
            "Exploratory benchmark; v1 outcomes informed v2; no unadjusted primary."
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        current_sha = require_clean_git(repo)
        if current_sha != initial_sha or current_sha != final_sha:
            raise RuntimeError(
                f"Git HEAD changed before publication: {initial_sha} -> {current_sha}"
            )
        _rename_noreplace(staging, out)
        return out
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = argparse.ArgumentParser(
        description="Hierarchical reconciliation + conformal PID on US COVID-19 deaths"
    )
    parser.add_argument("config", type=Path)
    cfg = load_config(parser.parse_args(argv).config)
    if (cfg.output_dir / cfg.run_id).exists():
        raise FileExistsError(
            f"{cfg.output_dir / cfg.run_id} exists; choose a new run_id"
        )
    if cfg.design == "benchmark":
        out_path = cfg.output_dir / cfg.run_id
        staging = out_path.with_name(f".{out_path.name}.partial")
        if staging.exists():
            raise FileExistsError(f"{staging} exists; remove it or choose a new run_id")
        repo = Path(__file__).resolve().parents[2]
        initial_git_sha = (
            require_clean_git(repo)
            if getattr(cfg.benchmark, "protocol_version", 1) == 2
            else None
        )
        digest = file_sha256(cfg.data_path)
        if cfg.expected_sha256 is not None and digest != cfg.expected_sha256:
            raise ValueError(
                f"{cfg.data_path} has sha256 {digest}, expected {cfg.expected_sha256}"
            )
        frame = read_forecasts(cfg.data_path)
        actuals = canonical_actuals(frame)
        assert cfg.benchmark is not None
        reference = reference_bundle(frame, actuals, cfg.benchmark.reference_forecaster)
        forecast_cfg, evaluation_cfg = benchmark_configs(cfg)
        generated = generate_model_bundles(
            actuals,
            reference.dates,
            reference.forecast_dates,
            forecast_cfg,
        )
        bundles = {
            **generated.bundles,
            cfg.benchmark.reference_model: reference,
        }
        result = run_benchmark(bundles, evaluation_cfg, actuals=actuals)
        forecast_diagnostics = {
            **generated.diagnostics,
            "canonical_truth": {
                "finite_disagreements": 0,
                "observed_cells": int(actuals.observed.sum()),
                "missing_cells": int((~actuals.observed).sum()),
            },
        }
        if initial_git_sha is None:
            out = write_benchmark_artifacts(cfg, result, forecast_diagnostics, digest)
        else:
            out = _write_benchmark_v2_artifacts(
                cfg, result, forecast_diagnostics, digest, repo, initial_git_sha
            )
    elif cfg.design == "panel":
        frame = read_forecasts(cfg.data_path, cfg.expected_sha256)
        candidates = build_bundles(frame)
        if candidates.get(FORECASTER) is None:
            raise ValueError(f"forecaster {FORECASTER!r} has no usable bundle")
        rule = EligibilityRule(
            nodes=candidates[FORECASTER].nodes,
            mint_warmup=cfg.mint_warmup,
            min_eval_weeks=cfg.min_eval_weeks,
            max_leaf_imputed=cfg.max_leaf_imputed,
            duplicate_share=cfg.duplicate_share,
        )
        table = eligibility(candidates, rule)
        require_eligible(table, cfg.forecasters)
        bundles = {name: candidates[name] for name in cfg.forecasters}
        result = run_panel(bundles, panel_config(cfg))
        out = write_panel_artifacts(cfg, result, table, file_sha256(cfg.data_path))
    else:
        bundle = load_hierarchy(cfg.data_path, cfg.expected_sha256)
        result = run_experiment(cfg, bundle)
        out = write_artifacts(cfg, result, file_sha256(cfg.data_path))
    print(f"Artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
