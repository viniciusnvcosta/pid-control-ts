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
