# ABOUTME: Experiment configuration: RunConfig, TOML loading, and the controller/panel config mappers.
# ABOUTME: Read-only: this module only parses TOML into a validated RunConfig and never writes files.
"""Run configuration.

Validates ``RunConfig`` (see docs/protocol-decisions.md) and maps it onto the
``ControllerConfig`` and ``PanelConfig`` consumed by ``hcp.conformal`` and ``hcp.panel``.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from hcp.conformal import CONTROLLERS, ControllerConfig
from hcp.data import FORECASTER
from hcp.panel import PanelConfig
from hcp.reconcile import RECONCILERS


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
    design: str = "us"
    forecasters: tuple[str, ...] = (FORECASTER,)
    min_eval_weeks: int = 60
    max_leaf_imputed: float = 0.05
    duplicate_share: float = 0.95

    def __post_init__(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a bare directory name")
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or self.root_seed < 0
        ):
            raise ValueError("root_seed must be a non-negative integer")
        for name in ("lr_grid", "reconcilers", "controllers", "blocks", "forecasters"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("data_path", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if self.design not in ("us", "panel"):
            raise ValueError(f"design must be 'us' or 'panel', got {self.design!r}")
        if self.design == "us":
            if self.forecasters != (FORECASTER,):
                raise ValueError("design 'us' requires forecasters == (FORECASTER,)")
        else:
            if "pid_theta" in self.controllers:
                raise ValueError("pid_theta is rejected for design 'panel' (D22)")
            if FORECASTER not in self.forecasters:
                raise ValueError(
                    f"forecasters must include the primary forecaster {FORECASTER!r}"
                )
            if len(set(self.forecasters)) != len(self.forecasters):
                raise ValueError("forecasters must not contain duplicates")
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


def controller_config(cfg: RunConfig, lr: float) -> ControllerConfig:
    return ControllerConfig(
        alpha=cfg.alpha,
        lr=lr,
        Csat=cfg.Csat,
        KI=cfg.KI,
        T_burnin=cfg.T_burnin,
        ahead=cfg.ahead,
    )


def panel_config(cfg: RunConfig) -> PanelConfig:
    """Map a panel ``RunConfig`` onto the ``PanelConfig`` consumed by ``run_panel``."""
    return PanelConfig(
        controller=controller_config(cfg, cfg.lr),
        lr_grid=cfg.lr_grid,
        reconcilers=cfg.reconcilers,
        controllers=cfg.controllers,
        mint_warmup=cfg.mint_warmup,
        rolling_window=cfg.rolling_window,
        blocks=cfg.blocks,
        n_boot=cfg.n_boot,
        root_seed=cfg.root_seed,
        primary_forecaster=FORECASTER,
    )
