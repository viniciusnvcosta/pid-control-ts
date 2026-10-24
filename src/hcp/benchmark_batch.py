# ABOUTME: Immutable in-memory contracts for resumable model-origin batches.
# ABOUTME: Provides canonical execution identity hashing without performing file I/O.
"""Frozen contracts shared by the resumable benchmark runner and forecast layer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np

from hcp.forecast_types import DiagnosticsTables


def _freeze_json(value: Any) -> Any:
    """Copy JSON-shaped values into immutable containers and normalize scalars."""
    if isinstance(value, np.generic):
        return _freeze_json(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("identity mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        f"identity values must be JSON-compatible, got {type(value).__name__}"
    )


def _json_value(value: Any) -> Any:
    """Convert immutable contract values to canonical JSON-compatible values."""
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return value


@dataclass(frozen=True)
class NeuralExecution:
    """Registered device and precision settings for neural benchmark models."""

    device: str
    required_gpu_name: str
    cublas_workspace_config: str
    allow_tf32: bool
    precision: str

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda:0"):
            raise ValueError("device must be 'cpu' or the registered 'cuda:0'")
        if self.allow_tf32 is not False:
            raise ValueError("TF32 must be disabled")
        if self.precision != "32-true":
            raise ValueError("neural precision must be '32-true'")
        if self.device == "cuda:0" and (
            not self.required_gpu_name or not self.cublas_workspace_config
        ):
            raise ValueError("CUDA execution requires a GPU name and cuBLAS workspace")
        if self.device == "cpu" and (
            self.required_gpu_name or self.cublas_workspace_config
        ):
            raise ValueError("CPU fixture execution must not declare CUDA settings")

    @classmethod
    def cpu(cls) -> NeuralExecution:
        """Return the explicit CPU execution fixture for automated tests."""
        return cls("cpu", "", "", False, "32-true")


@dataclass(frozen=True)
class ExecutionIdentity:
    """Frozen provenance identifying every input that can affect a benchmark run."""

    protocol_id: str
    run_id: str
    git_sha: str
    config_sha256: str
    lockfile_sha256: str
    input_sha256: str
    root_seed: int
    model_order: tuple[str, ...]
    eligible_origin_keys: tuple[str, ...]
    quantiles: tuple[float, ...]
    batch_schema_version: int
    python_version: str
    package_versions: Mapping[str, str]
    runtime_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.package_versions, Mapping) or not isinstance(
            self.runtime_provenance, Mapping
        ):
            raise TypeError("execution provenance fields must be mappings")
        object.__setattr__(self, "model_order", tuple(self.model_order))
        object.__setattr__(
            self, "eligible_origin_keys", tuple(self.eligible_origin_keys)
        )
        object.__setattr__(
            self, "quantiles", tuple(float(value) for value in self.quantiles)
        )
        object.__setattr__(
            self, "package_versions", _freeze_json(self.package_versions)
        )
        object.__setattr__(
            self, "runtime_provenance", _freeze_json(self.runtime_provenance)
        )


def execution_identity_digest(identity: ExecutionIdentity) -> str:
    """Return SHA-256 of the identity's canonical compact JSON representation."""
    payload = {field.name: getattr(identity, field.name) for field in fields(identity)}
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OriginForecastBatch:
    """One model's immutable forecasts and diagnostics for a single origin."""

    model: str
    target_index: int
    target: np.datetime64
    origin: np.datetime64
    seed_spawn_key: tuple[int, ...]
    nodes: tuple[str, ...]
    quantiles: np.ndarray
    source_kind: str | None
    diagnostics: DiagnosticsTables
    warning_counts: Mapping[str, int]
    interpolation: tuple[Mapping[str, object], ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if (
            isinstance(self.target_index, bool)
            or not isinstance(self.target_index, int)
            or self.target_index < 0
        ):
            raise ValueError("target_index must be non-negative")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative")
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(node, str) or not node for node in nodes):
            raise ValueError("nodes must be non-empty strings")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must be unique")
        quantiles = np.array(self.quantiles, dtype=float, copy=True)
        if quantiles.shape != (len(nodes), 3):
            raise ValueError("quantiles must have shape (len(nodes), 3)")
        valid_rows = np.isfinite(quantiles).all(axis=1) | np.isnan(quantiles).all(
            axis=1
        )
        if not valid_rows.all():
            raise ValueError("each quantile row must be finite or entirely NaN")
        quantiles.setflags(write=False)
        object.__setattr__(self, "seed_spawn_key", tuple(self.seed_spawn_key))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "diagnostics", self.diagnostics.copy())
        object.__setattr__(self, "warning_counts", _freeze_json(self.warning_counts))
        object.__setattr__(
            self,
            "interpolation",
            tuple(_freeze_json(record) for record in self.interpolation),
        )


@dataclass(frozen=True)
class ProgressSnapshot:
    """Advisory progress counters and timing for one benchmark run."""

    current_model: str | None
    current_origin: str | None
    completed: int
    total: int
    per_model: Mapping[str, int]
    elapsed_seconds: float
    eta_seconds: float | None

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.completed, self.total)
        ):
            raise ValueError("progress counts must be integers")
        if not (0 <= self.completed <= self.total):
            raise ValueError("progress counts require 0 <= completed <= total")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("progress counts must be non-negative integers")
        if self.eta_seconds is not None and (
            isinstance(self.eta_seconds, bool)
            or not math.isfinite(self.eta_seconds)
            or self.eta_seconds < 0
        ):
            raise ValueError("eta_seconds must be finite and non-negative")
        counts = dict(self.per_model)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError("per-model counts must be non-negative integers")
        object.__setattr__(self, "per_model", MappingProxyType(counts))


@dataclass(frozen=True)
class DeviceProvenance:
    """Exact configured device provenance used to authorize CUDA execution."""

    device: str
    device_index: int
    gpu_uuid: str
    gpu_name: str
    compute_capability: tuple[int, int]
    driver_version: str
    torch_version: str
    cuda_version: str
    deterministic_algorithms: bool
    matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    float32_matmul_precision: str
    cublas_workspace_config: str


def device_identity(
    execution: NeuralExecution,
    torch_module: object,
    nvidia_identity: Mapping[str, str],
) -> DeviceProvenance:
    """Normalize configured device properties without environment or process I/O.

    Args:
        execution: Registered neural settings.
        torch_module: Configured Torch boundary.
        nvidia_identity: NVIDIA driver versions keyed by GPU UUID.

    Raises:
        ValueError: If selected GPU provenance or deterministic settings differ.
    """
    if execution.device != "cuda:0":
        raise ValueError("device identity requires registered CUDA execution")
    properties = torch_module.cuda.get_device_properties(0)
    gpu_uuid = str(properties.uuid)
    driver = nvidia_identity.get(gpu_uuid, "")
    if not driver:
        raise ValueError("NVIDIA driver identity missing for selected GPU UUID")
    if properties.name != execution.required_gpu_name:
        raise ValueError("selected GPU differs from registered GPU")
    result = DeviceProvenance(
        execution.device,
        0,
        gpu_uuid,
        properties.name,
        tuple(torch_module.cuda.get_device_capability(0)),
        driver,
        str(torch_module.__version__),
        str(torch_module.version.cuda),
        torch_module.are_deterministic_algorithms_enabled(),
        torch_module.backends.cuda.matmul.allow_tf32,
        torch_module.backends.cudnn.allow_tf32,
        torch_module.get_float32_matmul_precision(),
        execution.cublas_workspace_config,
    )
    if (
        not result.deterministic_algorithms
        or torch_module.is_deterministic_algorithms_warn_only_enabled()
        or result.matmul_allow_tf32
        or result.cudnn_allow_tf32
        or result.float32_matmul_precision != "highest"
    ):
        raise ValueError("deterministic CUDA settings could not be established")
    return result


BATCH_SCHEMA_VERSION = 1
DIAGNOSTIC_PAYLOADS = ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings")
BATCH_PAYLOADS = ("forecasts",) + DIAGNOSTIC_PAYLOADS


def contract_json(value: object) -> dict:
    """Return a detached JSON representation of a frozen contract dataclass."""
    return _json_value(
        {field.name: getattr(value, field.name) for field in fields(value)}
    )


def payload_digest(payload: bytes) -> str:
    """Return the exact persisted payload's SHA-256 without performing I/O."""
    return hashlib.sha256(payload).hexdigest()


def batch_metadata(batch: OriginForecastBatch, identity: ExecutionIdentity) -> dict:
    """Build scalar batch metadata bound to the immutable execution identity."""
    if identity.batch_schema_version != BATCH_SCHEMA_VERSION:
        raise ValueError("unsupported batch schema version")
    if batch.model not in identity.model_order:
        raise ValueError("batch model differs from execution identity")
    if batch.seed_spawn_key[-2:] != (
        identity.model_order.index(batch.model),
        batch.target_index,
    ):
        raise ValueError("batch seed address differs from model/target identity")
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "execution_digest": execution_identity_digest(identity),
        "model": batch.model,
        "target_index": batch.target_index,
        "target": str(batch.target),
        "origin": str(batch.origin),
        "seed_spawn_key": list(batch.seed_spawn_key),
        "nodes": list(batch.nodes),
        "source_kind": batch.source_kind,
        "warning_counts": _json_value(batch.warning_counts),
        "interpolation": _json_value(batch.interpolation),
        "elapsed_seconds": batch.elapsed_seconds,
    }


def validate_batch_metadata(metadata: dict, expected: dict) -> None:
    """Reject missing, extra or mismatched manifest identity fields."""
    if set(metadata) != set(expected) | {"payloads"}:
        raise ValueError("batch manifest schema differs")
    if any(
        json.dumps(metadata[name], sort_keys=True, allow_nan=False)
        != json.dumps(value, sort_keys=True, allow_nan=False)
        for name, value in expected.items()
    ):
        raise ValueError("batch metadata differs from expected identity")
    payloads = metadata["payloads"]
    if not isinstance(payloads, dict) or set(payloads) != {
        f"{name}.parquet" for name in BATCH_PAYLOADS
    }:
        raise ValueError("batch payload schema differs")
    for record in payloads.values():
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "rows", "columns", "dtypes"}
            or type(record["rows"]) is not int
            or record["rows"] < 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or not isinstance(record["columns"], list)
            or not isinstance(record["dtypes"], list)
        ):
            raise ValueError("invalid batch payload descriptor")
