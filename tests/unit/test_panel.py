# ABOUTME: Unit tests for the leaves x forecasters panel stage: arms, node/level metrics,
# ABOUTME: paired block-bootstrap contrasts, Holm adjustment, and the cross-forecaster summary.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.conformal import ControllerConfig
from hcp.data import FORECASTER, build_bundle
from hcp.evaluate import METRIC_NAMES
from hcp.panel import (
    PanelConfig,
    forecaster_arms,
    forecaster_contrasts,
    node_scale,
    run_panel,
    summarize_forecasters,
)
from hcp.report import RAW_LR
from hcp.run import RunConfig, run_experiment
from tests.helpers import ALT_FORECASTER

SMALL = PanelConfig(
    controller=ControllerConfig(lr=0.1, T_burnin=2),
    lr_grid=(0.1,),
    reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
    controllers=("raw", "pi"),
    mint_warmup=5,
    rolling_window=5,
    blocks=(4,),
    n_boot=10,
    root_seed=1,
    primary_forecaster=FORECASTER,
)

# Deviation from the brief's literal SMALL for the forecaster_arms cases only: under
# mint_warmup=5, t0 == 8, which collides with the synthetic panel's own GAP_WEEK == 10
# truth gap for leaf "cc" (trimmed index 10 - 2 == 8, since "bb"'s absent weeks 0-1
# shift the grid by 2). "cc" would then already be unscored before any NaN is injected,
# contradicting the brief's literal oracle values. mint_warmup=6 moves t0 to 9, past the
# gap, with every other SMALL field unchanged; the brief's three expected node/endpoint
# values then match exactly (verified empirically).
ARMS_SMALL = replace(SMALL, mint_warmup=6)

# Mirrors tests/unit/test_run.py::BASE and ::SMALL, for the US-node regression anchor
# (test case 3): same alpha/Csat/KI/ahead defaults, same T_burnin/mint_warmup/window/
# blocks/n_boot/root_seed, so the panel's "none"/"bottom_up"/"mint_shrink" US-node
# metrics must equal run_experiment's metrics exactly.
RUN_BASE = {
    "run_id": "unit",
    "root_seed": 1,
    "data_path": "unused.csv",
    "output_dir": "results",
}
RUN_SMALL = {
    "controllers": ("raw", "pi"),
    "lr_grid": (0.1,),
    "T_burnin": 2,
    "mint_warmup": 5,
    "rolling_window": 5,
    "blocks": (4,),
    "n_boot": 10,
}


def test_forecaster_arms_nodes_and_endpoint(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    arms = forecaster_arms(bundle, ARMS_SMALL)
    assert arms.nodes == ("us", "aa", "bb", "cc")
    np.testing.assert_array_equal(arms.endpoint, [False, True, True, True])


def test_forecaster_arms_endpoint_excludes_territories(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    bundle = replace(bundle, nodes=("us", "aa", "bb", "pr"))
    arms = forecaster_arms(bundle, ARMS_SMALL)
    np.testing.assert_array_equal(arms.endpoint, [False, True, True, False])


def test_forecaster_arms_drops_unscored_node_with_nan_truth(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    y = bundle.y.copy()
    y[bundle.nodes.index("cc"), -1] = np.nan
    bundle = replace(bundle, y=y)
    arms = forecaster_arms(bundle, ARMS_SMALL)
    assert arms.nodes == ("us", "aa", "bb")
    assert arms.diagnostics["unscored_nodes"] == ["cc"]


def test_raw_none_intervals_equal_base_quantiles(panel_frame) -> None:
    from hcp.reconcile import evaluation_start

    bundle = build_bundle(panel_frame, FORECASTER)
    arms = forecaster_arms(bundle, SMALL)
    t0 = evaluation_start(bundle, SMALL.mint_warmup)
    scored = np.isfinite(bundle.y[:, t0:]).all(axis=1)
    expected_lo = bundle.q[scored, t0:, 0]
    expected_hi = bundle.q[scored, t0:, 2]
    lo, hi = arms.intervals[("none", "raw", RAW_LR)]
    np.testing.assert_array_equal(lo, expected_lo)
    np.testing.assert_array_equal(hi, expected_hi)


def test_us_node_metrics_match_run_experiment(hier_bundle) -> None:
    run_cfg = RunConfig(**RUN_BASE, **RUN_SMALL)
    run_result = run_experiment(run_cfg, hier_bundle)

    panel_cfg = PanelConfig(
        controller=ControllerConfig(lr=0.1, T_burnin=2),
        lr_grid=(0.1,),
        reconcilers=("none", "bottom_up", "mint_shrink"),
        controllers=("raw", "pi"),
        mint_warmup=5,
        rolling_window=5,
        blocks=(4,),
        n_boot=10,
        root_seed=1,
        primary_forecaster=FORECASTER,
    )
    panel_result = run_panel({FORECASTER: hier_bundle}, panel_cfg)

    us_metrics = (
        panel_result.metrics[panel_result.metrics["node"] == "us"]
        .sort_values(["reconciler", "controller", "lr"])
        .reset_index(drop=True)
    )
    run_metrics = run_result.metrics.sort_values(
        ["reconciler", "controller", "lr"]
    ).reset_index(drop=True)

    assert len(us_metrics) == len(run_metrics)
    np.testing.assert_array_equal(us_metrics["reconciler"], run_metrics["reconciler"])
    np.testing.assert_array_equal(us_metrics["controller"], run_metrics["controller"])
    np.testing.assert_array_equal(us_metrics["lr"], run_metrics["lr"])
    for column in METRIC_NAMES:
        np.testing.assert_allclose(
            us_metrics[column].to_numpy(), run_metrics[column].to_numpy(), rtol=1e-12
        )


def test_mint_median_raw_widths_equal_none_raw_widths(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    arms = forecaster_arms(bundle, SMALL)
    lo_m, hi_m = arms.intervals[("mint_median", "raw", RAW_LR)]
    lo_n, hi_n = arms.intervals[("none", "raw", RAW_LR)]
    np.testing.assert_allclose(hi_m - lo_m, hi_n - lo_n)


def test_scaled_metrics_equal_width_and_score_over_scale(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    arms = forecaster_arms(bundle, SMALL)
    scale = node_scale(arms.y)
    scale_by_node = dict(zip(arms.nodes, scale, strict=True))

    result = run_panel({FORECASTER: bundle}, SMALL)
    sub = result.metrics[result.metrics["forecaster"] == FORECASTER]
    for _, row in sub.iterrows():
        s = scale_by_node[row["node"]]
        assert row["scaled_mean_width"] == pytest.approx(row["mean_width"] / s)
        assert row["scaled_interval_score"] == pytest.approx(row["interval_score"] / s)


def test_leaf_mean_contrast_estimate_matches_metrics_diff(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    result = run_panel({FORECASTER: bundle}, SMALL)
    sub = result.metrics[result.metrics["forecaster"] == FORECASTER]
    lr = SMALL.controller.lr
    treat = sub[
        (sub["reconciler"] == "mint_shrink")
        & (sub["controller"] == "pi")
        & (sub["lr"] == lr)
    ]
    ctrl = sub[
        (sub["reconciler"] == "none") & (sub["controller"] == "pi") & (sub["lr"] == lr)
    ]
    treat = treat.set_index("node")["coverage_deviation"]
    ctrl = ctrl.set_index("node")["coverage_deviation"]
    endpoint_nodes = [n for n in treat.index if n != "us"]
    expected = (treat.loc[endpoint_nodes] - ctrl.loc[endpoint_nodes]).mean()

    row = result.contrasts[
        (result.contrasts["forecaster"] == FORECASTER)
        & (result.contrasts["level"] == "leaf_mean")
        & (result.contrasts["controller"] == "pi")
        & (result.contrasts["treatment"] == "mint_shrink")
        & (result.contrasts["control"] == "none")
        & (result.contrasts["metric"] == "coverage_deviation")
        & (result.contrasts["block"] == SMALL.blocks[0])
    ]
    assert len(row) == 1
    np.testing.assert_allclose(row["estimate"].iloc[0], expected, rtol=1e-9)


def test_exactly_one_primary_row(panel_frame) -> None:
    bundle = build_bundle(panel_frame, FORECASTER)
    result = run_panel({FORECASTER: bundle}, SMALL)
    primary_rows = result.contrasts[result.contrasts["primary"]]
    assert len(primary_rows) == 1
    row = primary_rows.iloc[0]
    assert row["forecaster"] == FORECASTER
    assert row["level"] == "leaf_mean"
    assert row["p_holm"] == row["p_value"]


def test_contrasts_are_deterministic_for_the_same_seed_and_differ_for_another(
    panel_frame,
) -> None:
    bundles = {
        FORECASTER: build_bundle(panel_frame, FORECASTER),
        ALT_FORECASTER: build_bundle(panel_frame, ALT_FORECASTER),
    }
    same_a = run_panel(bundles, SMALL).contrasts
    same_b = run_panel(bundles, SMALL).contrasts
    other = run_panel(bundles, replace(SMALL, root_seed=2)).contrasts

    pd.testing.assert_frame_equal(same_a, same_b)
    assert not same_a["p_value"].equals(other["p_value"])


def test_summarize_forecasters_descriptive_stats() -> None:
    contrasts = pd.DataFrame(
        [
            {
                "block": 8,
                "forecaster": name,
                "level": "leaf_mean",
                "controller": "pi",
                "lr": 0.1,
                "treatment": "mint_shrink",
                "control": "none",
                "metric": "coverage_deviation",
                "estimate": estimate,
            }
            for name, estimate in zip(
                ["f1", "f2", "f3"], [-1.0, 2.0, -3.0], strict=True
            )
        ]
    )
    summary = summarize_forecasters(
        contrasts, block=8, n_boot=1000, rng=np.random.default_rng(0)
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_forecasters"] == 3
    assert row["n_improved"] == 2
    assert row["mean_estimate"] == pytest.approx(-2 / 3)
    assert row["low"] <= row["mean_estimate"] <= row["high"]


def test_run_panel_raises_on_missing_primary_forecaster(panel_frame) -> None:
    bundle = build_bundle(panel_frame, ALT_FORECASTER)
    cfg = replace(SMALL, primary_forecaster=FORECASTER)
    with pytest.raises(ValueError, match="primary forecaster"):
        run_panel({ALT_FORECASTER: bundle}, cfg)


def test_forecaster_contrasts_is_a_public_helper(panel_frame) -> None:
    # Exercises forecaster_contrasts directly (not just through run_panel), confirming
    # its signature and that it does not itself add a "forecaster" column.
    bundle = build_bundle(panel_frame, FORECASTER)
    arms = forecaster_arms(bundle, SMALL)
    seed = np.random.SeedSequence(SMALL.root_seed)
    contrasts = forecaster_contrasts(arms, SMALL, seed)
    assert "forecaster" not in contrasts.columns
    assert set(contrasts["block"]) == set(SMALL.blocks)
