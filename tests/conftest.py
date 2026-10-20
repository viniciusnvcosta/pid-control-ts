# ABOUTME: Shared pytest fixtures: the synthetic hierarchy CSV and its loaded bundle.
# ABOUTME: See tests/helpers.py for the exact gaps the synthetic data contains.
from pathlib import Path

import pandas as pd
import pytest

from hcp.data import HierarchyBundle, load_hierarchy, read_forecasts
from tests.helpers import panel_rows, synthetic_rows


@pytest.fixture
def hier_csv(tmp_path: Path) -> Path:
    path = tmp_path / "deaths.csv"
    pd.DataFrame(synthetic_rows()).to_csv(
        path
    )  # index column mimics the real leading ""
    return path


@pytest.fixture
def hier_bundle(hier_csv: Path) -> HierarchyBundle:
    return load_hierarchy(hier_csv)


@pytest.fixture
def panel_csv(tmp_path: Path) -> Path:
    path = tmp_path / "panel.csv"
    pd.DataFrame(panel_rows()).to_csv(path)  # index column mimics the real leading ""
    return path


@pytest.fixture
def panel_frame(panel_csv: Path) -> pd.DataFrame:
    return read_forecasts(panel_csv)
