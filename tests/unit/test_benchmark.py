# ABOUTME: Unit tests for common-window hierarchy and model benchmark evaluation.
# ABOUTME: Covers alignment, exact window boundaries, Holm families, paired draws, signs, and seed stability.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.benchmark import (
    BenchmarkConfig,
    common_evaluation_index,
    run_benchmark,
)
from hcp.conformal import ControllerConfig
from hcp.data import HierarchyBundle
from hcp.panel import PanelConfig
from hcp.reconcile import evaluation_start

MODELS = ("arima", "sarima", "deepar")
REFERENCE = "covidhub"


def _panel_config(seed: int = 41) -> PanelConfig:
    return PanelConfig(
        controller=ControllerConfig(lr=0.1, T_burnin=2),
        lr_grid=(0.1,),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        controllers=("raw", "pi"),
        mint_warmup=20,
        rolling_window=5,
        blocks=(4,),
        n_boot=30,
        root_seed=seed,
        primary_forecaster=REFERENCE,
    )


def _config(*, weeks: int = 60, seed: int = 41) -> BenchmarkConfig:
    return BenchmarkConfig(
        panel=_panel_config(seed),
        models=MODELS,
        reference_model=REFERENCE,
        min_eval_weeks=weeks,
    )


def _bundle(n_weeks: int, model: str) -> HierarchyBundle:
    dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D").to_numpy()
    index = np.arange(n_weeks, dtype=float)
    aa = 30.0 + 0.2 * index + 3.0 * np.sin(index / 5.0)
    bb = 45.0 + 0.1 * index + 2.0 * np.cos(index / 7.0)
    y = np.vstack([aa + bb, aa, bb])
    bias = {"arima": 2.0, "deepar": 2.0, "sarima": 5.0, REFERENCE: -1.0}[model]
    node_bias = np.array([1.5, 1.0, -0.5])[:, None]
    median = y + bias * node_bias * (1.0 + 0.3 * np.sin(index / 4.0))
    width = (8.0 + abs(bias)) * (1.0 + 0.1 * np.cos(index / 6.0))
    q = np.stack([median - width, median, median + width], axis=-1)
    return HierarchyBundle(
        dates=dates,
        forecast_dates=dates - np.timedelta64(26, "D"),
        nodes=("us", "aa", "bb"),
        y=y,
        q=q,
        imputed=np.zeros((3, n_weeks), dtype=bool),
    )


def _bundles(n_weeks: int = 90) -> dict[str, HierarchyBundle]:
    return {name: _bundle(n_weeks, name) for name in (*MODELS, REFERENCE)}


def test_common_window_uses_latest_mint_ready_start() -> None:
    bundles = _bundles()
    q = bundles["sarima"].q.copy()
    q[:, :2] = np.nan
    bundles["sarima"] = replace(bundles["sarima"], q=q)

    start = common_evaluation_index(bundles, _config())

    expected = max(evaluation_start(bundle, 20) for bundle in bundles.values())
    assert start == expected


def test_common_window_accepts_60_and_rejects_59_weeks() -> None:
    accepted = _bundles(83)
    rejected = _bundles(82)

    start = common_evaluation_index(accepted, _config())

    assert len(accepted[REFERENCE].dates) - start == 60
    with pytest.raises(ValueError, match="60 common evaluation weeks"):
        common_evaluation_index(rejected, _config())


@pytest.mark.parametrize(
    "mismatch",
    ["keys", "dates", "nodes", "forecast_dates", "root_truth", "nonfinite"],
)
def test_common_window_rejects_misaligned_or_nonfinite_bundles(mismatch: str) -> None:
    bundles = _bundles()
    if mismatch == "keys":
        bundles.pop("arima")
    elif mismatch == "dates":
        bundles["arima"] = replace(
            bundles["arima"], dates=bundles["arima"].dates + np.timedelta64(7, "D")
        )
    elif mismatch == "nodes":
        bundles["arima"] = replace(bundles["arima"], nodes=("us", "aa", "cc"))
    elif mismatch == "forecast_dates":
        bundles["arima"] = replace(
            bundles["arima"],
            forecast_dates=bundles["arima"].forecast_dates + np.timedelta64(7, "D"),
        )
    elif mismatch == "root_truth":
        y = bundles["arima"].y.copy()
        y[0, -1] += 1.0
        bundles["arima"] = replace(bundles["arima"], y=y)
    else:
        q = bundles["arima"].q.copy()
        q[0, -1, 0] = np.nan
        bundles["arima"] = replace(bundles["arima"], q=q)

    with pytest.raises(ValueError, match="model keys|dates|nodes|truth|finite"):
        common_evaluation_index(bundles, _config())


def test_result_schema_hierarchy_families_and_bottom_up_identity() -> None:
    result = run_benchmark(_bundles(), _config())

    for frame in (result.intervals, result.metrics, result.level_metrics):
        assert "model" in frame
        assert "forecaster" not in frame
    contrasts = result.hierarchy_contrasts
    assert not contrasts["primary"].any()
    assert set(contrasts["treatment"]) == {
        "bottom_up",
        "mint_shrink",
        "mint_median",
    }
    family_sizes = contrasts.groupby(["block", "model", "level", "controller"]).size()
    assert set(family_sizes) == {15}
    zeros = contrasts[
        (contrasts["level"] == "leaf_mean") & (contrasts["treatment"] == "bottom_up")
    ]
    np.testing.assert_array_equal(zeros["estimate"], 0.0)


def test_model_contrasts_have_six_member_paired_holm_families_and_sign() -> None:
    result = run_benchmark(_bundles(), _config())
    contrasts = result.model_contrasts

    sizes = contrasts.groupby(["block", "level", "controller", "reconciler"]).size()
    assert set(sizes) == {6}
    assert set(contrasts["metric"]) == {
        "coverage_deviation",
        "scaled_interval_score",
    }
    assert set(contrasts["model"]) == set(MODELS)
    assert set(contrasts["reference_model"]) == {REFERENCE}
    direct = result.level_metrics.set_index(
        ["model", "level", "controller", "lr", "reconciler"]
    )
    row = contrasts.iloc[0]
    key = (row["model"], row["level"], row["controller"], row["lr"], row["reconciler"])
    ref_key = (REFERENCE, row["level"], row["controller"], row["lr"], row["reconciler"])
    assert row["estimate"] == pytest.approx(
        direct.loc[key, row["metric"]] - direct.loc[ref_key, row["metric"]]
    )
    paired = contrasts[contrasts["model"].isin(["arima", "deepar"])]
    pivot = paired.pivot(
        index=["block", "level", "controller", "reconciler", "metric"],
        columns="model",
        values=["estimate", "low", "high", "p_value"],
    )
    np.testing.assert_array_equal(
        pivot.xs("arima", level=1, axis=1), pivot.xs("deepar", level=1, axis=1)
    )
    assert np.all(contrasts["p_holm"] >= contrasts["p_value"])


def test_benchmark_bootstrap_is_seed_stable_and_seed_sensitive() -> None:
    bundles = _bundles()
    same_a = run_benchmark(bundles, _config()).model_contrasts
    same_b = run_benchmark(bundles, _config()).model_contrasts
    other = run_benchmark(bundles, _config(seed=42)).model_contrasts

    pd.testing.assert_frame_equal(same_a, same_b)
    assert not same_a[["low", "high", "p_value"]].equals(
        other[["low", "high", "p_value"]]
    )
