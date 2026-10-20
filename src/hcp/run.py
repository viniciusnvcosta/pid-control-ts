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
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

# RunConfig, load_config and panel_config are re-exported as part of the run API.
from hcp.config import RunConfig, controller_config, load_config, panel_config
from hcp.conformal import conformalize
from hcp.data import (
    FORECASTER,
    HierarchyBundle,
    build_bundles,
    file_sha256,
    load_hierarchy,
    read_forecasts,
)
from hcp.eligibility import EligibilityRule, eligibility, require_eligible
from hcp.evaluate import METRIC_NAMES, evaluate
from hcp.panel import PRIMARY as PANEL_PRIMARY
from hcp.panel import PanelResult, run_panel
from hcp.reconcile import evaluation_start, reconcile_bundle
from hcp.report import RAW_LR, render_figures, render_panel_figures
from hcp.stats import draw_blocks, holm, summarize_diff

PRIMARY = {
    "treatment": "mint_shrink",
    "control": "none",
    "controller": "pi",
    "metric": "coverage_deviation",
}
COMPARISONS = (("mint_shrink", "none"), ("bottom_up", "none"))


@dataclass(frozen=True)
class ExperimentResult:
    """In-memory outputs of one run."""

    intervals: pd.DataFrame
    metrics: pd.DataFrame
    contrasts: pd.DataFrame
    diagnostics: dict


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
                        evaluate(
                            y[ix],
                            lo[ix],
                            hi[ix],
                            alpha=cfg.alpha,
                            window=cfg.rolling_window,
                        )
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
                            "block": block,
                            "controller": controller,
                            "lr": lr,
                            "treatment": treatment,
                            "control": control,
                            "metric": metric,
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
    gap = bundle.y[0, t0:] - bundle.y[1:, t0:].sum(axis=0)
    diagnostics: dict = {
        "n_weeks": len(weeks),
        "evaluation_start": str(pd.Timestamp(weeks[0]).date()),
        "imputed_cells": int(bundle.imputed.sum()),
        "truth_incoherence": {
            "median": float(np.nanmedian(gap)),
            "max_abs": float(np.nanmax(np.abs(gap))),
        },
        "crossings": {},
        "root_crossings": {},
    }
    frames = []
    for reconciler in cfg.reconcilers:
        reconciled = reconcile_bundle(bundle, reconciler, cfg.mint_warmup)
        diagnostics["crossings"][reconciler] = reconciled.crossings
        diagnostics["root_crossings"][reconciler] = reconciled.root_crossings
        if reconciler == "mint_shrink":
            diagnostics["mean_shrinkage"] = float(np.mean(reconciled.shrinkage[t0:]))
        q_us = reconciled.q[0, t0:, :]
        for controller in cfg.controllers:
            for lr in (RAW_LR,) if controller == "raw" else cfg.lr_grid:
                lo, hi = conformalize(
                    y, q_us[:, 0], q_us[:, 2], controller, controller_config(cfg, lr)
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "week": weeks,
                            "reconciler": reconciler,
                            "controller": controller,
                            "lr": lr,
                            "y": y,
                            "q_lo": q_us[:, 0],
                            "q_med": q_us[:, 1],
                            "q_hi": q_us[:, 2],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                )
    intervals = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(
        [
            {
                "reconciler": reconciler,
                "controller": controller,
                "lr": lr,
                **evaluate(
                    group["y"].to_numpy(),
                    group["lo"].to_numpy(),
                    group["hi"].to_numpy(),
                    alpha=cfg.alpha,
                    window=cfg.rolling_window,
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
        "config": asdict(cfg),
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


def write_panel_artifacts(
    cfg: RunConfig,
    result: PanelResult,
    eligibility_table: pd.DataFrame,
    input_sha256: str,
) -> Path:
    """Write panel tables, figures, eligibility and the manifest to ``output_dir / run_id``.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    out = cfg.output_dir / cfg.run_id
    out.mkdir(parents=True, exist_ok=False)
    result.intervals.to_parquet(out / "intervals.parquet", index=False)
    result.metrics.to_parquet(out / "metrics.parquet", index=False)
    result.level_metrics.to_parquet(out / "level_metrics.parquet", index=False)
    result.contrasts.to_parquet(out / "contrasts.parquet", index=False)
    result.summary.to_parquet(out / "summary.parquet", index=False)
    eligibility_table.to_parquet(out / "eligibility.parquet", index=False)
    render_panel_figures(
        result.metrics,
        result.contrasts,
        out / "figures",
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
