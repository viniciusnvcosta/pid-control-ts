# ABOUTME: Unit tests for the moving-block bootstrap, contrast summaries and Holm adjustment.
# ABOUTME: Checks block contiguity, determinism, interval/p-value arithmetic and a known Holm example.
import numpy as np
import pytest

from hcp.stats import draw_blocks, holm, summarize_diff


def test_draw_blocks_shape_range_and_contiguity() -> None:
    idx = draw_blocks(10, 3, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() <= 9
    blocks = idx[:, :9].reshape(50, 3, 3)
    assert (np.diff(blocks, axis=-1) == 1).all()


def test_draw_blocks_is_deterministic_for_a_seed() -> None:
    a = draw_blocks(20, 4, 30, np.random.default_rng(7))
    b = draw_blocks(20, 4, 30, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("block", [0, 11])
def test_draw_blocks_invalid_block_raises(block: int) -> None:
    with pytest.raises(ValueError, match="block"):
        draw_blocks(10, block, 5, np.random.default_rng(0))


def test_summarize_diff_symmetric_null() -> None:
    contrast = summarize_diff(0.501, np.linspace(-1.0, 1.0, 1001))
    assert contrast.estimate == 0.501
    assert contrast.low == pytest.approx(-0.95, abs=1e-9)
    assert contrast.high == pytest.approx(0.95, abs=1e-9)
    assert contrast.p_value == pytest.approx(0.5, abs=0.01)


def test_summarize_diff_detects_a_shift() -> None:
    diffs = 2.0 + np.random.default_rng(0).normal(scale=0.1, size=500)
    contrast = summarize_diff(2.0, diffs)
    assert contrast.low > 0
    assert contrast.p_value == pytest.approx(1 / 501)


def test_holm_known_example() -> None:
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
