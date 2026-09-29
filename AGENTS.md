# AGENTS.md

Hierarchical reconciliation (state → US) before Conformal PID (Angelopoulos et al., 2023).
Design: docs/superpowers/specs/2026-09-29-hierarchical-conformal-pid-design.md. Decisions: docs/protocol-decisions.md (overrides plans).

## Layout
- `core/` — upstream conformal methods. Import, never reimplement. Changes only behind tests/unit/test_core_characterization.py.
- `src/hcp/` — flat package, one module per stage: data → reconcile → conformal → evaluate → stats → report, orchestrated by run.py. Modules 150–300 lines; split above ~400.
- `configs/*.toml` — one file per experiment, loaded into a frozen dataclass. No Hydra.
- `results/<run_id>/` — only run.py writes; never overwritten.

## Conventions
- Registries are module-level dicts (`RECONCILERS`, `CONTROLLERS`); configs name components by key.
- Numeric modules do no I/O. Randomness only via `np.random.SeedSequence(root_seed)`.
- No look-ahead: anything used for target t must be known by `forecast_dates[t]`.
- Type hints and Google docstrings on public functions; frozen dataclasses; two `# ABOUTME:` lines per file.
- Dependencies only via `uv add`. Conventional Commits; commit locally, never push, never `--no-verify`.

## Definition of done
Tests green and clean (unit, integration, e2e); component referenced from a config by name; outputs under `results/<run_id>/` with the manifest.
