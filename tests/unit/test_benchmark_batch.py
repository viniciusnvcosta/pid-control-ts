# ABOUTME: Tests immutable contracts for resumable benchmark batch execution.
# ABOUTME: Covers defensive payload ownership and canonical execution identity hashes.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from hcp.forecast_types import DiagnosticsTables


def _contract_api():
    """Return the new public contracts with an assertion-style missing API failure."""
    try:
        from hcp import benchmark_batch
    except (ModuleNotFoundError, ImportError):
        pytest.fail("benchmark batch contracts are not implemented")
    return benchmark_batch


def _identity(package_versions, runtime_provenance):
    ExecutionIdentity = _contract_api().ExecutionIdentity
    return ExecutionIdentity(
        protocol_id="benchmark-v2",
        run_id="model-benchmarks-v2",
        git_sha="abc123",
        config_sha256="a" * 64,
        lockfile_sha256="b" * 64,
        input_sha256="c" * 64,
        root_seed=20261021,
        model_order=("arima", "deepar"),
        eligible_origin_keys=("2020-01-01",),
        quantiles=(0.1, 0.5, 0.9),
        batch_schema_version=1,
        python_version="3.11.13",
        package_versions=package_versions,
        runtime_provenance=runtime_provenance,
    )


def test_execution_identity_digest_ignores_mapping_insertion_order() -> None:
    execution_identity_digest = _contract_api().execution_identity_digest
    first = _identity(
        {"torch": "2.0", "numpy": "2.0"}, {"cuda": {"name": "RTX", "index": 0}}
    )
    second = _identity(
        {"numpy": "2.0", "torch": "2.0"}, {"cuda": {"index": 0, "name": "RTX"}}
    )

    assert execution_identity_digest(first) == execution_identity_digest(second)


def test_execution_identity_owns_and_freezes_nested_mappings() -> None:
    versions = {"torch": "2.0"}
    provenance = {"cuda": {"index": np.int64(0)}}
    identity = _identity(versions, provenance)
    versions["torch"] = "changed"
    provenance["cuda"]["index"] = 1

    assert identity.package_versions["torch"] == "2.0"
    assert identity.runtime_provenance["cuda"]["index"] == 0
    with pytest.raises(TypeError):
        identity.package_versions["numpy"] = "2.0"
    with pytest.raises(TypeError):
        identity.runtime_provenance["cuda"]["index"] = 2


def test_origin_batch_defensively_owns_quantiles_and_diagnostic_tables() -> None:
    OriginForecastBatch = _contract_api().OriginForecastBatch
    values = np.array([[1.0, 2.0, 3.0]])
    source_table = pd.DataFrame({"model": ["deepar"], "value": [1.0]})
    tables = DiagnosticsTables(forecast_diagnostics=source_table)
    batch = OriginForecastBatch(
        model="deepar",
        target_index=0,
        target=np.datetime64("2020-01-05"),
        origin=np.datetime64("2020-01-05"),
        seed_spawn_key=(0, 2, 0),
        nodes=("US",),
        quantiles=values,
        source_kind="samples",
        diagnostics=tables,
        warning_counts={"ConvergenceWarning": 1},
        interpolation=({"node": "US", "count": 0},),
        elapsed_seconds=1.5,
    )
    values[0, 0] = -1.0
    source_table.loc[0, "value"] = -1.0

    assert batch.quantiles[0, 0] == 1.0
    assert batch.diagnostics.forecast_diagnostics.loc[0, "value"] == 1.0
    with pytest.raises(ValueError):
        batch.quantiles[0, 0] = 4.0
    with pytest.raises(TypeError):
        batch.warning_counts["other"] = 2
    with pytest.raises(TypeError):
        batch.interpolation[0]["count"] = 4
    with pytest.raises(FrozenInstanceError):
        batch.target_index = 1


@pytest.mark.parametrize(
    "changes",
    [
        {"target_index": -1},
        {"elapsed_seconds": float("inf")},
        {"nodes": ()},
        {"nodes": ("US", "US")},
        {"quantiles": np.array([[1.0, 2.0]])},
        {"quantiles": np.array([[1.0, np.nan, 3.0]])},
    ],
)
def test_origin_batch_rejects_invalid_payload(changes) -> None:
    OriginForecastBatch = _contract_api().OriginForecastBatch
    fields = {
        "model": "deepar",
        "target_index": 0,
        "target": np.datetime64("2020-01-05"),
        "origin": np.datetime64("2020-01-05"),
        "seed_spawn_key": (0, 2, 0),
        "nodes": ("US",),
        "quantiles": np.array([[1.0, 2.0, 3.0]]),
        "source_kind": None,
        "diagnostics": DiagnosticsTables(),
        "warning_counts": {},
        "interpolation": (),
        "elapsed_seconds": 1.5,
    }
    fields.update(changes)
    with pytest.raises(ValueError):
        OriginForecastBatch(**fields)


def test_progress_snapshot_validates_and_freezes_counts() -> None:
    ProgressSnapshot = _contract_api().ProgressSnapshot
    counts = {"deepar": 1}
    snapshot = ProgressSnapshot("deepar", "2020-01-05", 1, 2, counts, 3.0, 3.0)
    counts["deepar"] = 9

    assert snapshot.per_model["deepar"] == 1
    with pytest.raises(TypeError):
        snapshot.per_model["deepar"] = 2
    with pytest.raises(ValueError):
        ProgressSnapshot(None, None, 3, 2, {}, 0.0, None)
    with pytest.raises(ValueError):
        ProgressSnapshot(None, None, 0, 2, {}, 0.0, -1.0)


def test_neural_execution_cpu_fixture_contract() -> None:
    NeuralExecution = _contract_api().NeuralExecution
    assert NeuralExecution.cpu() == NeuralExecution(
        device="cpu",
        required_gpu_name="",
        cublas_workspace_config="",
        allow_tf32=False,
        precision="32-true",
    )


def test_forecast_types_has_module_docstring():
    import hcp.forecast_types as module
    assert module.__doc__ == "In-memory forecast contracts with defensive array ownership."


def test_device_identity_is_pure_and_rejects_missing_driver():
    from types import SimpleNamespace
    api = _contract_api()
    probe = getattr(api, "device_identity", None)
    assert probe is not None, "pure device identity normalization is absent"
    execution = api.NeuralExecution("cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true")
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(name=execution.required_gpu_name, uuid="GPU-test"),
        get_device_capability=lambda index: (8, 9)))
    with pytest.raises(ValueError, match="driver"):
        probe(execution, torch, {})
