# ABOUTME: Integration tests: the real deaths.csv hierarchy and exact parity with the paper's state pickles.
# ABOUTME: Needs data/deaths.csv (gitignored); run with `uv run pytest -m integration`.
import numpy as np
import pandas as pd
import pytest

from hcp.data import load_hierarchy
from tests.helpers import PICKLE_STATES, PICKLES, REAL_DATA, REAL_SHA256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA.exists(), reason="data/deaths.csv not downloaded"),
]


@pytest.fixture(scope="module")
def real_bundle():
    return load_hierarchy(REAL_DATA, REAL_SHA256)


def test_real_hierarchy_is_us_plus_56_leaves(real_bundle) -> None:
    assert real_bundle.nodes[0] == "us"
    assert len(real_bundle.nodes) == 57
    assert {"dc", "as", "gu", "mp", "pr", "vi"} <= set(real_bundle.nodes)


@pytest.mark.parametrize("state", PICKLE_STATES)
def test_matches_paper_pickle(real_bundle, state: str) -> None:
    paper = pd.read_pickle(PICKLES / f"{state}_proc_4wkdeaths.pkl").pivot(
        index="timestamp", columns="variable", values="target"
    )
    row = real_bundle.nodes.index(state)
    ours = pd.DataFrame(
        {
            "our_y": real_bundle.y[row],
            "our_lo": real_bundle.q[row, :, 0],
            "our_hi": real_bundle.q[row, :, 2],
            "imputed": real_bundle.imputed[row],
        },
        index=pd.DatetimeIndex(real_bundle.dates),
    )
    joined = ours.join(paper, how="inner")
    joined = joined[~joined["imputed"] & joined["our_y"].notna()]
    assert len(joined) > 100
    forecasts = np.stack(joined["forecasts"].to_numpy())
    np.testing.assert_array_equal(joined["our_y"], joined["y"].astype(float))
    np.testing.assert_array_equal(joined["our_lo"], forecasts[:, 0])
    np.testing.assert_array_equal(joined["our_hi"], forecasts[:, 1])
