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
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

# RunConfig, load_config and panel_config are re-exported as part of the run API.
from hcp.config import RunConfig, config_manifest, load_config, panel_config
from hcp.data import (
    FORECASTER,
    build_bundles,
    file_sha256,
    load_hierarchy,
    read_forecasts,
)
from hcp.eligibility import EligibilityRule, eligibility, require_eligible
from hcp.experiment import PRIMARY, ExperimentResult, run_experiment
from hcp.panel import PRIMARY as PANEL_PRIMARY
from hcp.panel import PanelResult, run_panel
from hcp.report import render_figures, render_panel_figures


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


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
    if cfg.design == "panel":
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
