# ABOUTME: Read-only comparison of the immutable v1 and v2 benchmark bundles.
# ABOUTME: Audits v2 and exports descriptive tables and figures under docs/analysis.
"""Reproduce the evidence accompanying the Portuguese v2 comparison report."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from hcp.stats import holm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/model-benchmarks-v2-evidence"
ARMS = {
    "A": ("none", "raw"),
    "B": ("none", "pi_norm"),
    "C": ("mint_median", "raw"),
    "D": ("mint_median", "pi_norm"),
}
METRICS = ("scaled_interval_score", "marginal_coverage_error")


def main() -> None:
    """Audit existing artifacts and export comparison evidence without fitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    args = parser.parse_args()
    v2 = ROOT / "results/model-benchmarks-v2"
    manifest = json.loads((v2 / "manifest.json").read_text())
    failures = [
        name
        for name, digest in manifest["artifacts"].items()
        if hashlib.sha256((v2 / name).read_bytes()).hexdigest() != digest
    ]
    assert not failures, failures
    tables = {p.stem: pd.read_parquet(p) for p in sorted(v2.glob("*.parquet"))}
    level = tables["level_metrics"]
    intervals = tables["intervals"].copy()
    assert len(intervals) == 8 * 57 * 63 * 12
    assert (
        intervals.groupby(["model", "node", "reconciler", "controller"])
        .week.nunique()
        .eq(63)
        .all()
    )
    intervals["covered"] = (intervals.lo <= intervals.y) & (intervals.y <= intervals.hi)
    intervals["empty"] = intervals.lo > intervals.hi
    intervals["physical_width"] = np.maximum(intervals.hi - intervals.lo, 0)
    intervals["score"] = (
        intervals.hi
        - intervals.lo
        + 10 * (intervals.lo - intervals.y) * (intervals.y < intervals.lo)
        + 10 * (intervals.y - intervals.hi) * (intervals.y > intervals.hi)
    )
    intervals["abs_y"] = intervals.y.abs()
    keys = ["model", "node", "reconciler", "controller"]
    independent = intervals.groupby(keys).agg(
        marginal_coverage=("covered", "mean"),
        empty_set_rate=("empty", "mean"),
        width=("physical_width", "mean"),
        score=("score", "mean"),
        scale=("abs_y", "mean"),
    )
    independent["marginal_coverage_error"] = abs(independent.marginal_coverage - 0.8)
    independent["scaled_interval_score"] = independent.score / independent.scale.clip(
        lower=1
    )
    independent["scaled_mean_set_width"] = independent.width / independent.scale.clip(
        lower=1
    )
    checked = [*METRICS, "marginal_coverage", "empty_set_rate", "scaled_mean_set_width"]
    stored = tables["metrics"].set_index(keys).sort_index()
    independent = independent.sort_index()
    assert stored.index.equals(independent.index)
    np.testing.assert_allclose(stored[checked], independent[checked], atol=1e-12)
    leaves = independent.reset_index()
    leaves = leaves[~leaves.node.isin(["us", "as", "gu", "mp", "pr", "vi"])]
    leaf_mean = leaves.groupby(["model", "reconciler", "controller"])[checked].mean()
    principal = []
    for arm, (rec, ctrl) in ARMS.items():
        part = level[(level.reconciler == rec) & (level.controller == ctrl)].copy()
        part["arm"] = arm
        principal.append(part)
    principal = pd.concat(principal, ignore_index=True)
    for row in principal[principal.level == "leaf_mean"].itertuples():
        actual = leaf_mean.loc[(row.model, row.reconciler, row.controller)]
        np.testing.assert_allclose(
            actual[checked], [getattr(row, m) for m in checked], atol=1e-12
        )
    pivot = principal[principal.level == "leaf_mean"].set_index(["model", "arm"])
    h, r = tables["hierarchy_contrasts"], tables["model_contrasts"]
    for row in h.itertuples():
        a, b, c, d = [pivot.loc[(row.model, arm), row.metric] for arm in "ABCD"]
        effects = {
            "pi_effect": b - a,
            "reconciliation_effect": c - a,
            "combined_effect": d - a,
            "reconciliation_given_pi": d - b,
            "pi_given_reconciliation": d - c,
            "interaction": (d - c) - (b - a),
        }
        np.testing.assert_allclose(row.estimate, effects[row.contrast], atol=1e-12)
    for row in r.itertuples():
        expected = (
            pivot.loc[(row.model, row.arm), row.metric]
            - pivot.loc[("covidhub", row.arm), row.metric]
        )
        np.testing.assert_allclose(row.estimate, expected, atol=1e-12)
    for table, size in [(h, 96), (r, 56)]:
        for _, group in table.groupby("block"):
            assert len(group) == size
            np.testing.assert_allclose(group.p_holm, holm(group.p_value.to_numpy()))
    old = pd.read_parquet(args.v1 / "level_metrics.parquet")
    comparison = level.merge(
        old,
        on=["model", "level", "reconciler", "controller", "lr"],
        suffixes=("_v2", "_v1"),
    )
    for metric in [
        "scaled_interval_score",
        "coverage_deviation",
        "marginal_coverage",
        "scaled_mean_width",
    ]:
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_v2"] - comparison[f"{metric}_v1"]
        )
    old_base = pd.read_parquet(args.v1 / "base_forecasts.parquet")
    shared = tables["base_forecasts"].merge(
        old_base, on=["model", "node", "week", "forecast_date"], suffixes=("_v2", "_v1")
    )
    base_deltas = {}
    for model, group in shared.groupby("model"):
        base_deltas[model] = max(
            float((group[f"{q}_v2"] - group[f"{q}_v1"]).abs().max())
            for q in ["q_lo", "q_med", "q_hi"]
        )
    OUT.mkdir(parents=True, exist_ok=True)
    principal.to_csv(OUT / "principal-arms.csv", index=False)
    comparison.to_csv(OUT / "v1-v2-matched-cells.csv", index=False)
    h.to_csv(OUT / "intervention-all-blocks.csv", index=False)
    r.to_csv(OUT / "reference-all-blocks.csv", index=False)
    warnings = tables["warnings"].groupby("model")["count"].sum()
    summary = {
        "v1_source": str(args.v1.resolve()),
        "v1_manifest_sha256": hashlib.sha256(
            (args.v1 / "manifest.json").read_bytes()
        ).hexdigest(),
        "v2_manifest_sha256": hashlib.sha256(
            (v2 / "manifest.json").read_bytes()
        ).hexdigest(),
        "v2_git_sha": manifest["git_sha"],
        "v2_artifacts_checked": len(manifest["artifacts"]),
        "row_counts": {k: len(v) for k, v in tables.items()},
        "empty_rows": int(intervals["empty"].sum()),
        "empty_rows_by_controller": intervals.groupby("controller")["empty"]
        .sum()
        .to_dict(),
        "warning_occurrences": warnings.to_dict(),
        "v1_v2_max_abs_base_quantile_delta": base_deltas,
        "independent_metric_recomputation": "passed, all nodes and cells",
        "contrast_estimates_and_holm_recomputation": "passed, all blocks",
        "bootstrap_resampling_rerun": False,
    }
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    models = [
        "covidhub",
        "transformer",
        "deepar_enhanced",
        "deepar",
        "qr_l1",
        "arima",
        "sarima",
        "prophet",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for arm in "ABCD":
        frame = pivot.xs(arm, level="arm").reindex(models)
        for ax, metric in zip(axes, METRICS, strict=True):
            ax.plot(np.arange(8), frame[metric], "o-", label=arm, linewidth=0.8)
            ax.set_xticks(np.arange(8), models, rotation=35, ha="right")
            ax.set_ylabel(metric)
            ax.legend(title="Braço")
    fig.suptitle(
        "V2 · média de 51 estados/DC · menor é melhor\nA: none/raw; B: none/pi_norm; C: mint_median/raw; D: mint_median/pi_norm"
    )
    fig.savefig(OUT / "principal-arms.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    none = level[(level.level == "leaf_mean") & (level.reconciler == "none")]
    for ctrl, label in [
        ("raw", "Raw"),
        ("pi", "PI original"),
        ("pi_norm", "PI normalizado"),
    ]:
        frame = none[none.controller == ctrl].set_index("model").reindex(models)
        for ax, metric in zip(
            axes, ["scaled_interval_score", "scaled_mean_set_width"], strict=True
        ):
            ax.plot(np.arange(8), frame[metric], "o-", label=label, linewidth=0.8)
            ax.set_xticks(np.arange(8), models, rotation=35, ha="right")
            ax.set_ylabel(metric)
            ax.legend()
    fig.suptitle("V2 · custo do PI sem reconciliação · comparação descritiva")
    fig.savefig(OUT / "pi-cost.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
