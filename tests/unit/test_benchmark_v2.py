# ABOUTME: Synthetic tests for the v2 common window and fixed-scale twelve-cell arms.
# ABOUTME: Exercises the benchmark without fitting any forecasting models.
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hcp.benchmark import BenchmarkConfig, common_evaluation_index, run_benchmark
from hcp.conformal import ControllerConfig
from hcp.data import HierarchyBundle
from hcp.panel import PanelConfig

MODELS = (
    "arima",
    "sarima",
    "deepar",
    "prophet",
    "transformer",
    "qr_l1",
    "deepar_enhanced",
)
REFERENCE = "covidhub"
NODES = ("us", *(f"s{i:02}" for i in range(1, 52)), "as", "gu", "mp", "pr", "vi")


def _config(weeks: int = 60) -> BenchmarkConfig:
    panel = PanelConfig(
        controller=ControllerConfig(lr=0.1, T_burnin=5),
        lr_grid=(0.1,),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        controllers=("raw", "pi_norm", "pi"),
        mint_warmup=20,
        rolling_window=10,
        blocks=(8, 4, 12),
        n_boot=3,
        root_seed=20261021,
        primary_forecaster=REFERENCE,
    )
    return BenchmarkConfig(panel, MODELS, REFERENCE, weeks, protocol_version=2)


def _bundle(n_weeks: int, model: str) -> HierarchyBundle:
    dates = pd.date_range("2020-01-04", periods=n_weeks, freq="7D").to_numpy()
    t = np.arange(n_weeks, dtype=float)
    base = 4 + np.arange(len(NODES), dtype=float)[:, None] + 0.03 * t
    y = base + np.sin(t[None, :] / 7)
    bias = MODELS.index(model) + 1 if model in MODELS else -1
    node = np.arange(len(NODES), dtype=float)[:, None]
    residual = np.sin(t[None, :] / 3 + node / 5)
    median = y + bias * (1 + 0.1 * residual) + residual
    q = np.stack([median - 2, median, median + 2], axis=-1)
    return HierarchyBundle(
        dates, dates - np.timedelta64(26, "D"), NODES, y, q, np.zeros_like(y, bool)
    )


def _bundles(n: int = 90) -> dict[str, HierarchyBundle]:
    return {m: _bundle(n, m) for m in (*MODELS, REFERENCE)}


def test_v2_window_requires_all_models_and_retains_51_endpoint_leaves() -> None:
    bundles = _bundles(90)
    q = bundles["prophet"].q.copy()
    q[:, :2] = np.nan
    bundles["prophet"] = replace(bundles["prophet"], q=q)
    start = common_evaluation_index(bundles, _config())
    assert start >= 20
    endpoint = [
        node for node in NODES[1:] if node not in {"as", "gu", "mp", "pr", "vi"}
    ]
    assert len(endpoint) == 51


def test_v2_window_accepts_60_weeks_and_rejects_59() -> None:
    accepted = _bundles(83)
    start = common_evaluation_index(accepted, _config(60))
    assert len(accepted[REFERENCE].dates) - start >= 60
    with pytest.raises(ValueError, match="60 common evaluation weeks"):
        common_evaluation_index(_bundles(82), _config(60))


def test_v2_fixed_pi_scales_ignore_evaluation_truth(monkeypatch) -> None:
    import hcp.panel as panel_module

    captured = []

    def fake_conformalize(y, lo, hi, controller, cfg, *, scale=None):
        if controller == "pi_norm":
            captured.append(scale)
        return lo.copy(), hi.copy()

    monkeypatch.setattr(panel_module, "conformalize", fake_conformalize)
    bundles = _bundles(90)
    result = run_benchmark(bundles, _config())
    cells = result.metrics[
        ["model", "reconciler", "controller", "lr"]
    ].drop_duplicates()
    assert cells.groupby("model").size().eq(12).all()
    first = tuple(captured)
    captured.clear()
    for name, bundle in list(bundles.items()):
        y = bundle.y.copy()
        y[:, 25:] += 10000
        bundles[name] = replace(bundle, y=y)
    run_benchmark(bundles, _config())
    assert first == tuple(captured)
    n_nodes = len(NODES)
    per_model = 4 * n_nodes
    assert len(first) == len(MODELS + (REFERENCE,)) * per_model
    for offset in range(0, len(first), per_model):
        np.testing.assert_array_equal(first[offset : offset + n_nodes], first[:n_nodes])
