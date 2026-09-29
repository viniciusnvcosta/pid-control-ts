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
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or self.root_seed < 0
        ):
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
        alpha=cfg.alpha,
        lr=lr,
        Csat=cfg.Csat,
        KI=cfg.KI,
        T_burnin=cfg.T_burnin,
        ahead=cfg.ahead,
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
                    y, q_us[:, 0], q_us[:, 2], controller, _controller_config(cfg, lr)
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
        raise FileExistsError(
            f"{cfg.output_dir / cfg.run_id} exists; choose a new run_id"
        )
    bundle = load_hierarchy(cfg.data_path, cfg.expected_sha256)
    result = run_experiment(cfg, bundle)
    out = write_artifacts(cfg, result, file_sha256(cfg.data_path))
    print(f"Artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
