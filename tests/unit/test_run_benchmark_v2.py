# ABOUTME: Tests v2 Git provenance and no-replace artifact publication contracts.
# ABOUTME: Uses temporary Git repositories and synthetic artifacts only.
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

import hcp.run as run_module
from hcp.benchmark import BenchmarkResult
from hcp.config import RunConfig
from hcp.forecast_types import DiagnosticsTables
from hcp.run import require_clean_git, write_benchmark_artifacts


def _repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked").write_text("base")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "base"],
        check=True,
        capture_output=True,
    )
    return path


def test_require_clean_git_returns_sha_and_rejects_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    sha = require_clean_git(repo)
    assert len(sha) == 40

    (repo / "untracked").write_text("new")
    with pytest.raises(RuntimeError, match="not clean"):
        require_clean_git(repo)
    (repo / "untracked").unlink()
    (repo / "tracked").write_text("staged")
    subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
    with pytest.raises(RuntimeError, match="not clean"):
        require_clean_git(repo)


def test_require_clean_git_fails_on_git_command_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git status --porcelain"):
        require_clean_git(tmp_path)


def _cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="model-benchmarks-v2",
        root_seed=20261021,
        data_path=Path("data/deaths.csv"),
        output_dir=tmp_path / "results",
        expected_sha256="cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06",
        design="benchmark",
        controllers=("raw", "pi", "pi_norm"),
        reconcilers=("none", "bottom_up", "mint_shrink", "mint_median"),
        lr=0.1,
        lr_grid=(0.1,),
        benchmark={"protocol_version": 2},
    )


def _result() -> BenchmarkResult:
    frame = pd.DataFrame({"model": ["arima"], "value": [1.0]})
    return BenchmarkResult(
        frame,
        frame,
        frame,
        frame,
        frame,
        frame,
        {
            "protocol_version": 2,
            "model_order": ["arima", "covidhub"],
            "pi_scales": {"US": 1.0},
        },
        DiagnosticsTables(),
    )


def test_v2_writer_publishes_ten_tables_five_figures_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(
        run_module,
        "benchmark_figures",
        lambda *args, **kwargs: {f"fig-{n}": plt.figure() for n in range(5)},
    )
    result = _result()
    out = write_benchmark_artifacts(
        cfg, result, {"seed": {"spawn_key": [0]}}, "datahash"
    )

    assert len(list(out.glob("*.parquet"))) == 10
    assert len(list((out / "figures").glob("*.pdf"))) == 5
    manifest = __import__("json").loads((out / "manifest.json").read_text())
    assert manifest["initial_git_sha"] == "a" * 40
    assert manifest["final_git_sha"] == manifest["initial_git_sha"]
    assert manifest["git_sha"] == manifest["initial_git_sha"]
    assert manifest["provenance"]["config_sha256"]
    assert manifest["provenance"]["lockfile_sha256"]
    assert manifest["families"]["hierarchy"]["size"] == 96
    assert manifest["families"]["model"]["size"] == 56
    assert "D15" in manifest["limitations"]
    assert "v1 outcomes informed v2" in manifest["analysis_note"]
    assert list(pd.read_parquet(out / "warnings.parquet").columns) == [
        "category",
        "message",
        "source",
        "model",
        "node",
        "origin",
        "candidate",
        "phase",
        "count",
    ]
    assert list(pd.read_parquet(out / "quantile_repairs.parquet").columns) == [
        "model",
        "node",
        "forecast_date",
        "stage",
        "before",
        "after",
        "count",
    ]
    assert not list(cfg.output_dir.glob(".*.partial"))


def test_v2_writer_removes_only_owned_staging_when_destination_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(run_module, "benchmark_figures", lambda *args, **kwargs: {})
    out = cfg.output_dir / cfg.run_id
    sentinel = out / "other-process.txt"
    original = run_module._rename_noreplace

    def race(staging: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        sentinel.write_text("preserve")
        original(staging, destination)

    monkeypatch.setattr(run_module, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        write_benchmark_artifacts(
            cfg, _result(), {"seed": {"spawn_key": [0]}}, "datahash"
        )
    assert sentinel.read_text() == "preserve"
    assert not list(cfg.output_dir.glob(".*.partial"))


def test_v2_writer_rejects_sha_change_and_cleans_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    shas = iter(("a" * 40, "a" * 40, "b" * 40))
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: next(shas))
    monkeypatch.setattr(run_module, "benchmark_figures", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="HEAD changed"):
        write_benchmark_artifacts(
            cfg, _result(), {"seed": {"spawn_key": [0]}}, "datahash"
        )
    assert not (cfg.output_dir / cfg.run_id).exists()
    assert not list(cfg.output_dir.glob(".*.partial"))


def _cuda_torch(
    monkeypatch,
    *,
    available=True,
    count=1,
    name="NVIDIA GeForce RTX 4080",
    deterministic=True,
):
    import os
    import sys
    from types import SimpleNamespace

    def operation(value=None):
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return value

    state = SimpleNamespace(enabled=False)

    def enable(enabled, *, warn_only):
        assert warn_only is False
        if not deterministic:
            raise RuntimeError("deterministic initialization failed")
        state.enabled = enabled

    fake = SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_initialized=lambda: False,
            is_available=lambda: operation(available),
            device_count=lambda: operation(count),
            set_device=lambda index: operation(index),
            get_device_properties=lambda index: SimpleNamespace(
                name=name, uuid="GPU-test"
            ),
            get_device_capability=lambda index: (8, 9),
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        set_float32_matmul_precision=lambda value: setattr(state, "precision", value),
        get_float32_matmul_precision=lambda: state.precision,
        use_deterministic_algorithms=enable,
        are_deterministic_algorithms_enabled=lambda: state.enabled,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(
        run_module,
        "_query_nvidia_identity",
        lambda index: {"GPU-test": "570.1"},
        raising=False,
    )
    return fake


def _execution():
    from hcp.benchmark_batch import NeuralExecution

    return NeuralExecution(
        "cuda:0", "NVIDIA GeForce RTX 4080", ":4096:8", False, "32-true"
    )


def test_cuda_gate_captures_exact_runtime(monkeypatch):
    fake = _cuda_torch(monkeypatch)
    gate = getattr(run_module, "_configure_v2_cuda", None)
    assert gate is not None, "CUDA launch gate is absent"
    provenance = gate(_execution())
    assert provenance.device == "cuda:0"
    assert provenance.device_index == 0
    assert provenance.gpu_uuid == "GPU-test"
    assert provenance.gpu_name == "NVIDIA GeForce RTX 4080"
    assert provenance.compute_capability == (8, 9)
    assert provenance.driver_version == "570.1"
    assert provenance.torch_version == "2.8.0"
    assert provenance.cuda_version == "12.8"
    assert provenance.deterministic_algorithms is True
    assert provenance.matmul_allow_tf32 is False
    assert provenance.cudnn_allow_tf32 is False
    assert provenance.float32_matmul_precision == "highest"
    assert provenance.cublas_workspace_config == ":4096:8"
    assert fake.backends.cuda.matmul.allow_tf32 is False


def test_cuda_gate_normalizes_torch_uuid_and_preserves_nvidia_identity(monkeypatch):
    from types import SimpleNamespace

    fake = _cuda_torch(monkeypatch)
    fake.cuda.get_device_properties = lambda index: SimpleNamespace(
        name="NVIDIA GeForce RTX 4080",
        uuid="f3dad84a-4909-bd89-f509-0c2f79debd6f",
    )
    monkeypatch.setattr(
        run_module,
        "_query_nvidia_identity",
        lambda index: {
            "GPU-other": "wrong-driver",
            "GPU-f3dad84a-4909-bd89-f509-0c2f79debd6f": "580.95.05",
        },
    )

    provenance = run_module._configure_v2_cuda(_execution())

    assert provenance.gpu_uuid == "GPU-f3dad84a-4909-bd89-f509-0c2f79debd6f"
    assert provenance.driver_version == "580.95.05"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"available": False}, "CUDA"),
        ({"count": 2}, "one visible"),
        ({"name": "other GPU"}, "GPU"),
        ({"deterministic": False}, "deterministic"),
    ],
)
def test_cuda_gate_fails_closed(monkeypatch, kwargs, match):
    _cuda_torch(monkeypatch, **kwargs)
    gate = getattr(run_module, "_configure_v2_cuda", None)
    assert gate is not None, "CUDA launch gate is absent"
    with pytest.raises(RuntimeError, match=match):
        gate(_execution())


def test_nvidia_query_matches_visible_uuid_not_physical_index(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def query(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=0, stdout="GPU-other, 570.1\nGPU-test, 570.1\n"
        )

    monkeypatch.setattr(run_module.subprocess, "run", query)
    probe = getattr(run_module, "_query_nvidia_identity", None)
    assert probe is not None, "NVIDIA identity probe is absent"
    result = probe(0)
    assert result == {"GPU-other": "570.1", "GPU-test": "570.1"}
    assert "--query-gpu=uuid,driver_version" in calls[0]


@pytest.mark.parametrize(
    "output",
    ["", "GPU-test, 570.1\nGPU-test, 570.2\n", "GPU-test, \n"],
)
def test_nvidia_query_rejects_missing_or_ambiguous_identity(monkeypatch, output):
    from types import SimpleNamespace

    monkeypatch.setattr(
        run_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    with pytest.raises(RuntimeError, match="identity query failed"):
        run_module._query_nvidia_identity(0)


@pytest.mark.parametrize("identities", [{}, {"GPU-other": "570.1"}])
def test_cuda_gate_rejects_missing_or_nonmatching_uuid(monkeypatch, identities):
    _cuda_torch(monkeypatch)
    monkeypatch.setattr(run_module, "_query_nvidia_identity", lambda index: identities)
    with pytest.raises(RuntimeError, match="driver identity missing"):
        run_module._configure_v2_cuda(_execution())


def test_cuda_gate_rejects_matching_unregistered_gpu(monkeypatch):
    from dataclasses import replace

    _cuda_torch(monkeypatch, name="NVIDIA GeForce RTX 4090")
    execution = replace(_execution(), required_gpu_name="NVIDIA GeForce RTX 4090")
    with pytest.raises(RuntimeError, match="registered.*GPU|registered CUDA"):
        run_module._configure_v2_cuda(execution)


def test_cuda_gate_rejects_initialized_runtime_without_mutation(monkeypatch):
    import os

    fake = _cuda_torch(monkeypatch)
    fake.cuda.is_initialized = lambda: True
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "existing-setting")
    with pytest.raises(RuntimeError, match="already initialized"):
        run_module._configure_v2_cuda(_execution())
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "existing-setting"
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True


def _storage_fixture():
    import numpy as np

    from hcp.benchmark_batch import ExecutionIdentity, OriginForecastBatch

    identity = ExecutionIdentity(
        "benchmark-v2",
        "fixture",
        "abc",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        42,
        ("arima", "deepar"),
        ("2020-01-05",),
        (0.1, 0.5, 0.9),
        1,
        "3.11",
        {"numpy": "fixture"},
        {"device": "cpu"},
    )
    batch = OriginForecastBatch(
        "arima",
        3,
        np.datetime64("2020-02-02"),
        np.datetime64("2020-01-05"),
        (0, 0, 3),
        ("US",),
        np.array([[1.0, 2.0, 3.0]]),
        "sample_derived",
        DiagnosticsTables(),
        {"warning": 2},
        ({"node": "US", "count": 1},),
        0.5,
    )
    return identity, batch


def _read_batch(path, identity, batch, **changes):
    expected = {
        name: getattr(batch, name)
        for name in (
            "model",
            "target_index",
            "target",
            "origin",
            "nodes",
            "seed_spawn_key",
            "source_kind",
        )
    }
    expected.update(changes)
    return run_module._read_origin_batch(path, identity, **expected)


def test_batch_storage_round_trip_and_immutable_state(tmp_path):
    from dataclasses import replace

    import numpy as np

    identity, batch = _storage_fixture()
    work = tmp_path / ".fixture.partial"
    run_module._initialize_batch_workdir(work, identity)
    state = (work / "run-state.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_module._initialize_batch_workdir(work, identity)
    run_module._initialize_batch_workdir(work, identity, resume=True)
    with pytest.raises(ValueError):
        run_module._initialize_batch_workdir(
            work, replace(identity, root_seed=43), resume=True
        )
    assert (work / "run-state.json").read_bytes() == state
    path = run_module._write_origin_batch(work, identity, batch)
    restored = _read_batch(path, identity, batch)
    np.testing.assert_array_equal(restored.quantiles, batch.quantiles)
    assert restored.warning_counts == batch.warning_counts
    assert restored.interpolation == batch.interpolation
    assert restored.elapsed_seconds == 0.5
    for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings"):
        pd.testing.assert_frame_equal(
            getattr(restored.diagnostics, name), getattr(batch.diagnostics, name)
        )
    with pytest.raises(ValueError):
        restored.quantiles[0, 0] = 0
    with pytest.raises(TypeError):
        restored.warning_counts["new"] = 1
    with pytest.raises(FileExistsError):
        run_module._write_origin_batch(work, identity, batch)
    np.testing.assert_array_equal(
        _read_batch(path, identity, batch).quantiles, batch.quantiles
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "deepar"),
        ("target_index", 4),
        ("origin", "2020-01-12"),
        ("target", "2020-02-09"),
        ("seed_spawn_key", (0, 1, 3)),
        ("nodes", ("OTHER",)),
        ("source_kind", "native_quantiles"),
    ],
)
def test_batch_reader_rejects_swapped_expected_identity(tmp_path, field, value):
    identity, batch = _storage_fixture()
    run_module._initialize_batch_workdir(tmp_path / "work", identity)
    path = run_module._write_origin_batch(tmp_path / "work", identity, batch)
    with pytest.raises(ValueError):
        _read_batch(path, identity, batch, **{field: value})


@pytest.mark.parametrize(
    "corruption",
    [
        "hash",
        "rows",
        "model",
        "origin",
        "seed_spawn_key",
        "schema_version",
        "execution_digest",
        "missing_manifest",
        "json",
        "extra",
        "parquet",
    ],
)
def test_batch_reader_fails_closed_on_corruption(tmp_path, corruption):
    import hashlib
    import json

    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    manifest = path / "batch.json"
    payload = json.loads(manifest.read_text())
    if corruption == "missing_manifest":
        manifest.unlink()
    elif corruption == "json":
        manifest.write_text("{")
    elif corruption == "extra":
        (path / "unexpected").write_text("extra")
    elif corruption in ("hash", "parquet"):
        forecast = path / "forecasts.parquet"
        forecast.write_bytes(b"invalid parquet")
        if corruption == "parquet":
            payload["payloads"]["forecasts.parquet"]["sha256"] = hashlib.sha256(
                forecast.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(payload))
    else:
        if corruption == "rows":
            payload["payloads"]["forecasts.parquet"]["rows"] += 1
        else:
            payload[corruption] = {
                "model": "deepar",
                "origin": "2020-01-12",
                "seed_spawn_key": [0, 1, 3],
                "schema_version": 999,
                "execution_digest": "wrong",
            }[corruption]
        manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        _read_batch(path, identity, batch)


def test_batch_payload_hash_and_row_count_cover_every_table(tmp_path):
    import json

    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    manifest = json.loads((path / "batch.json").read_text())
    for name in manifest["payloads"]:
        saved = (path / name).read_bytes()
        (path / name).write_bytes(saved + b"corrupt")
        with pytest.raises(ValueError):
            _read_batch(path, identity, batch)
        (path / name).write_bytes(saved)
        manifest["payloads"][name]["rows"] += 1
        (path / "batch.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            _read_batch(path, identity, batch)
        manifest["payloads"][name]["rows"] -= 1
        (path / "batch.json").write_text(json.dumps(manifest))


def test_batch_crash_before_manifest_and_temporary_sibling_are_not_reusable(
    tmp_path, monkeypatch
):
    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    original = Path.write_text

    def crash(path, *args, **kwargs):
        if path.name == "batch.json":
            assert len(list(path.parent.glob("*.parquet"))) == 5
            raise OSError("simulated crash")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(OSError, match="simulated crash"):
        run_module._write_origin_batch(work, identity, batch)
    assert not list(work.glob("batches/*/*.batch"))
    temporary = next(work.glob("batches/*/.*"))
    with pytest.raises(ValueError):
        _read_batch(temporary, identity, batch)
    monkeypatch.setattr(Path, "write_text", original)
    complete = run_module._write_origin_batch(work, identity, batch)
    assert _read_batch(complete, identity, batch).model == "arima"


def test_progress_atomic_replacement_and_runtime_truncated_final_line(
    tmp_path, monkeypatch
):
    import json

    from hcp.benchmark_batch import ProgressSnapshot

    snapshot = ProgressSnapshot("arima", "2020-01-05", 1, 2, {"arima": 1}, 3.0, 3.0)
    run_module._write_progress(tmp_path, snapshot)
    old = (tmp_path / "progress.json").read_bytes()
    original = run_module.os.replace
    monkeypatch.setattr(
        run_module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("crash"))
    )
    with pytest.raises(OSError):
        run_module._write_progress(tmp_path, snapshot)
    assert (tmp_path / "progress.json").read_bytes() == old
    monkeypatch.setattr(run_module.os, "replace", original)
    run_module._write_progress(tmp_path, snapshot)
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 1
    run_module._append_runtime_event(tmp_path, {"event": "completed", "count": 1})
    with (tmp_path / "runtime.jsonl").open("ab") as stream:
        stream.write(b'{"event": "crash\xe2')
    assert run_module._read_runtime_events(tmp_path) == (
        {"event": "completed", "count": 1},
    )
    (tmp_path / "runtime.jsonl").write_bytes(b"{broken}\n")
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)
    (tmp_path / "runtime.jsonl").write_bytes(b"{broken}\n{}")
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)
    (tmp_path / "runtime.jsonl").write_bytes(b"[]\n")
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)


@pytest.mark.parametrize(
    "line", [b"{broken}", b'{"event":invalid}', b'{"event":NaN}', b'{"event":"\xff"}']
)
def test_runtime_reader_rejects_malformed_unterminated_final_line(tmp_path, line):
    (tmp_path / "runtime.jsonl").write_bytes(line)
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)


def test_batch_read_checks_execution_schema_and_complete_temporary_name(tmp_path):
    from dataclasses import replace

    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    for other in (
        replace(identity, root_seed=1),
        replace(identity, batch_schema_version=2),
    ):
        with pytest.raises(ValueError):
            _read_batch(path, other, batch)
    temporary = path.with_name(".completed.tmp")
    path.rename(temporary)
    with pytest.raises(ValueError):
        _read_batch(temporary, identity, batch)


def test_batch_writer_no_replace_race_preserves_destination(tmp_path, monkeypatch):
    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    original = run_module._rename_noreplace

    def race(source, destination):
        assert (source / "batch.json").exists()
        destination.mkdir()
        (destination / "sentinel").write_text("original")
        original(source, destination)

    monkeypatch.setattr(run_module, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        run_module._write_origin_batch(work, identity, batch)
    assert next(work.glob("batches/*/*.batch/sentinel")).read_text() == "original"


def test_workdir_admission_rejects_existing_final_and_incomplete_state(tmp_path):
    identity, _ = _storage_fixture()
    final = tmp_path / "final"
    final.mkdir()
    work = tmp_path / "work"
    with pytest.raises(FileExistsError):
        run_module._initialize_batch_workdir(work, identity, final_dir=final)
    assert not work.exists()
    work.mkdir()
    (work / "run-state.json").write_text("{")
    with pytest.raises(ValueError):
        run_module._initialize_batch_workdir(work, identity, resume=True)


def test_batch_round_trip_preserves_nonempty_diagnostics_and_owned_values(tmp_path):
    from dataclasses import replace

    identity, batch = _storage_fixture()
    table = pd.DataFrame(
        {
            "model": ["arima"],
            "origin": pd.to_datetime(["2020-01-05"]),
            "node": ["US"],
            "value": [1.25],
            "count": [2],
        }
    )
    names = ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings")
    batch = replace(
        batch, diagnostics=DiagnosticsTables(**{name: table for name in names})
    )
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    restored = _read_batch(path, identity, batch)
    for name in names:
        pd.testing.assert_frame_equal(getattr(restored.diagnostics, name), table)
    restored.diagnostics.forecast_diagnostics.loc[0, "value"] = 99
    assert (
        _read_batch(path, identity, batch).diagnostics.forecast_diagnostics.loc[
            0, "value"
        ]
        == 1.25
    )


@pytest.mark.parametrize("schema", ["columns", "dtypes"])
def test_batch_rejects_payload_schema_corruption(tmp_path, schema):
    import json

    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    manifest = path / "batch.json"
    payload = json.loads(manifest.read_text())
    payload["payloads"]["tuning.parquet"][schema] = ["incorrect"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        _read_batch(path, identity, batch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("target_index", 3.0),
        ("seed_spawn_key", [0.0, 0.0, 3.0]),
    ],
)
def test_batch_rejects_coercible_identity_types(tmp_path, field, value):
    import json

    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    manifest = path / "batch.json"
    payload = json.loads(manifest.read_text())
    payload[field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        _read_batch(path, identity, batch)


@pytest.mark.parametrize(
    "event",
    [
        {"event": "ação 😀", "elapsed": 1.25, "estimate": -1.2e-125},
        {
            "event": "nested",
            "data": [True, False, None, {"quote": '\\"', "number": 1.5e125}],
        },
    ],
)
def test_runtime_reader_accepts_every_writer_byte_truncation(tmp_path, event):
    import json

    run_module._append_runtime_event(tmp_path, {"event": "previous"})
    prefix = (tmp_path / "runtime.jsonl").read_bytes()
    run_module._append_runtime_event(tmp_path, event)
    encoded = (tmp_path / "runtime.jsonl").read_bytes()[len(prefix) :]
    for length in range(len(encoded) + 1):
        (tmp_path / "runtime.jsonl").write_bytes(prefix + encoded[:length])
        restored = run_module._read_runtime_events(tmp_path)
        expected = ({"event": "previous"},)
        try:
            decoded = json.loads(encoded[:length])
        except ValueError:
            pass
        else:
            expected += (decoded,)
        assert restored == expected, (length, encoded[:length])


@pytest.mark.parametrize(
    "fragment", [b'{"event":true}\xe2', b'{"event":true} \xf0\x9f']
)
def test_runtime_reader_rejects_complete_object_before_partial_utf8(tmp_path, fragment):
    (tmp_path / "runtime.jsonl").write_bytes(fragment)
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)


@pytest.mark.parametrize("value", ["1e999", "-1e999"])
@pytest.mark.parametrize("template", ['{"event":%s}\n', '{"nested":[{"value":%s}]}\n'])
def test_runtime_reader_rejects_overflowing_numbers(tmp_path, value, template):
    (tmp_path / "runtime.jsonl").write_text(template % value)
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)


@pytest.mark.parametrize(
    "component", ["workdir", "batches", "model", "state", "manifest"]
)
def test_batch_reader_rejects_symlinked_ancestors_and_metadata(tmp_path, component):
    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    path = run_module._write_origin_batch(work, identity, batch)
    selected = {
        "workdir": work,
        "batches": work / "batches",
        "model": path.parent,
        "state": work / "run-state.json",
        "manifest": path / "batch.json",
    }[component]
    external = tmp_path / "external"
    selected.rename(external)
    selected.symlink_to(external, target_is_directory=external.is_dir())
    with pytest.raises(ValueError):
        _read_batch(path, identity, batch)


@pytest.mark.parametrize("component", ["workdir", "batches", "model", "state"])
def test_batch_writer_rejects_symlinked_ancestors_without_external_writes(
    tmp_path, component
):
    identity, batch = _storage_fixture()
    work = tmp_path / "work"
    run_module._initialize_batch_workdir(work, identity)
    external = tmp_path / "external"
    if component == "workdir":
        work.rename(external)
        work.symlink_to(external, target_is_directory=True)
    elif component == "state":
        (work / "run-state.json").rename(external)
        (work / "run-state.json").symlink_to(external)
    else:
        external.mkdir()
        selected = work / "batches"
        if component == "model":
            selected.mkdir()
            selected /= "arima"
        selected.symlink_to(external, target_is_directory=True)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        run_module._write_origin_batch(work, identity, batch)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("operation", ["initialize", "progress", "append"])
def test_other_storage_writers_reject_redirected_workdir(tmp_path, operation):
    from hcp.benchmark_batch import ProgressSnapshot

    external = tmp_path / "external"
    external.mkdir()
    work = tmp_path / "link" / "work"
    work.parent.symlink_to(external, target_is_directory=True)
    if operation != "initialize":
        (external / "work").mkdir()
    if operation == "initialize":
        action = lambda: run_module._initialize_batch_workdir(
            work, _storage_fixture()[0]
        )
    elif operation == "progress":
        action = lambda: run_module._write_progress(
            work, ProgressSnapshot(None, None, 0, 0, {}, 0.0, None)
        )
    else:
        action = lambda: run_module._append_runtime_event(work, {"event": "test"})
    before = sorted(str(p) for p in external.rglob("*"))
    with pytest.raises(ValueError):
        action()
    assert sorted(str(p) for p in external.rglob("*")) == before


@pytest.mark.parametrize("name", ["progress.json", "runtime.jsonl"])
def test_advisory_writers_reject_symlinked_destination(tmp_path, name):
    from hcp.benchmark_batch import ProgressSnapshot

    work = tmp_path / "work"
    work.mkdir()
    external = tmp_path / "external"
    external.write_text("preserved")
    (work / name).symlink_to(external)
    with pytest.raises(ValueError):
        if name == "progress.json":
            run_module._write_progress(
                work, ProgressSnapshot(None, None, 0, 0, {}, 0.0, None)
            )
        else:
            run_module._append_runtime_event(work, {"event": "test"})
    assert external.read_text() == "preserved"


@pytest.mark.parametrize(
    "fragment",
    [
        b'{"event":"\\u0x',
        b'{"event":1.e',
        b'{"event":01',
        b'{"event":1e++',
        b'{"event":true} trailing',
        b'{"event":[1,]}',
        b'{"event":"bad\\x',
        b'{"event":true,"nested":1e999',
    ],
)
def test_runtime_reader_rejects_invalid_incomplete_object_prefixes(tmp_path, fragment):
    (tmp_path / "runtime.jsonl").write_bytes(fragment)
    with pytest.raises(ValueError):
        run_module._read_runtime_events(tmp_path)


def _resume_fixture(monkeypatch):
    import warnings

    import numpy as np

    from hcp.benchmark_batch import ExecutionIdentity
    from hcp.benchmark_data import ActualPanel
    from hcp.forecast import FORECASTERS, ForecastConfig, plan_model_forecasts

    calls = []

    def adapter(histories, settings, seed, execution):
        calls.append(seed.spawn_key)
        warnings.warn("scientific evidence", UserWarning)
        return np.random.default_rng(seed).normal(size=(2, 4, settings.n_samples))

    monkeypatch.setitem(FORECASTERS, "deepar", adapter)
    cfg = ForecastConfig(models=("deepar",), root_seed=17)
    dates = pd.date_range("2020-01-04", periods=70, freq="7D").to_numpy()
    actuals = ActualPanel(dates, ("us", "aa"), np.ones((2, 70)), np.ones((2, 70), bool))
    plan = plan_model_forecasts(
        actuals, dates[64:68], dates[64:68] - np.timedelta64(26, "D"), cfg
    )
    identity = ExecutionIdentity(
        "benchmark-v2",
        "fixture",
        "abc",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        17,
        cfg.models,
        tuple(str(plan.origins[i]) for _, i in plan.eligible_keys),
        cfg.quantiles,
        1,
        "3.11",
        {"numpy": "fixture"},
        {
            "gpu_uuid": "GPU-test",
            "gpu_name": "RTX4080",
            "compute_capability": [8, 9],
            "driver_version": "570",
            "cuda_version": "12.8",
            "device": "cuda:0",
            "device_index": 0,
            "torch_version": "2.8",
            "deterministic_algorithms": True,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
    )
    return cfg, plan, identity, calls


def test_resume_executes_only_missing_batches_and_preserves_science(
    tmp_path, monkeypatch, capsys, caplog
):
    import logging

    from hcp.forecast import FORECASTERS

    cfg, plan, identity, calls = _resume_fixture(monkeypatch)
    from dataclasses import replace

    original_generate = run_module.generate_origin_batch

    def populated(*args):
        batch = original_generate(*args)
        tables = {
            name: (
                getattr(batch.diagnostics, name)
                if not getattr(batch.diagnostics, name).empty
                else pd.DataFrame(
                    {"model": [batch.model], "selected": [True], "value": [1.0]}
                )
            )
            for name in (
                "forecast_diagnostics",
                "tuning",
                "quantile_repairs",
                "warnings",
            )
        }
        return replace(batch, diagnostics=DiagnosticsTables(**tables))

    monkeypatch.setattr(run_module, "generate_origin_batch", populated)
    execute = getattr(run_module, "_generate_or_resume_v2", None)
    assert execute is not None, "resumable orchestration is absent"
    workdir = tmp_path / ".fixture.partial"
    adapter = FORECASTERS["deepar"]
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    monkeypatch.setattr(logger, "level", logging.INFO)

    def interrupted(*args):
        logger.info("GPU available banner and tip")
        if len(calls) == 2:
            raise RuntimeError("interrupt")
        return adapter(*args)

    monkeypatch.setitem(FORECASTERS, "deepar", interrupted)
    with pytest.raises(RuntimeError, match="interrupt"):
        execute(plan, cfg, identity, workdir)
    assert len(list(workdir.glob("batches/*/*.batch"))) == 2
    assert logger.level == logging.INFO
    monkeypatch.setitem(FORECASTERS, "deepar", adapter)
    calls.clear()
    resumed, summary = execute(plan, cfg, identity, workdir)
    assert calls == [(0, 0, 2), (0, 0, 3)]
    assert (
        summary.resumed_batches,
        summary.generated_batches,
        summary.total_batches,
    ) == (2, 2, 4)
    complete, _ = execute(plan, cfg, identity, tmp_path / "fresh")
    pd.testing.assert_frame_equal(resumed.base_forecasts, complete.base_forecasts)
    for name in ("forecast_diagnostics", "tuning", "quantile_repairs", "warnings"):
        assert not getattr(resumed.tables, name).empty
        pd.testing.assert_frame_equal(
            getattr(resumed.tables, name), getattr(complete.tables, name)
        )
    assert len(resumed.tables.warnings) == 4
    assert resumed.diagnostics["models"]["deepar"]["warning_counts"] == {
        "UserWarning": 4
    }
    events = run_module._read_runtime_events(workdir)
    assert [event["event"] for event in events].count("error") == 1
    assert [event["event"] for event in events].count("resume_skip") == 2
    assert [event["event"] for event in events].count("batch_complete") == 4
    assert events[0]["event"] == "model_start" and events[0]["eta_seconds"] is None
    assert (
        __import__("json").loads((workdir / "progress.json").read_text())["completed"]
        == 4
    )
    assert "GPU available" not in capsys.readouterr().out
    assert "GPU available" not in caplog.text


@pytest.mark.parametrize(
    "field",
    [
        "git_sha",
        "config_sha256",
        "lockfile_sha256",
        "input_sha256",
        "gpu_uuid",
        "gpu_name",
        "compute_capability",
        "driver_version",
        "cuda_version",
        "device",
        "device_index",
        "torch_version",
        "deterministic_algorithms",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "float32_matmul_precision",
        "cublas_workspace_config",
    ],
)
def test_resume_identity_mismatch_precedes_batch_read_and_fit(
    tmp_path, monkeypatch, field
):
    from dataclasses import replace

    cfg, plan, identity, calls = _resume_fixture(monkeypatch)
    execute = getattr(run_module, "_generate_or_resume_v2", None)
    assert execute is not None, "resumable orchestration is absent"
    run_module._initialize_batch_workdir(tmp_path / "work", identity)
    if field in {"git_sha", "config_sha256", "lockfile_sha256", "input_sha256"}:
        changed = replace(identity, **{field: "changed"})
    else:
        changed = replace(
            identity,
            runtime_provenance={**identity.runtime_provenance, field: "changed"},
        )

    def forbidden(*args, **kwargs):
        pytest.fail("batch read before identity validation")

    monkeypatch.setattr(run_module, "_read_origin_batch", forbidden)
    before = (tmp_path / "work" / "run-state.json").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        execute(plan, cfg, changed, tmp_path / "work")
    assert not calls
    assert (tmp_path / "work" / "run-state.json").read_bytes() == before
    assert not (tmp_path / "work" / "runtime.jsonl").exists()


def test_resume_drops_only_verified_truncated_log_tail(tmp_path, monkeypatch):
    cfg, plan, identity, _ = _resume_fixture(monkeypatch)
    workdir = tmp_path / "work"
    run_module._generate_or_resume_v2(plan, cfg, identity, workdir)
    original = (workdir / "runtime.jsonl").read_bytes()
    with (workdir / "runtime.jsonl").open("ab") as stream:
        stream.write(b'{"event":"interru')
    run_module._generate_or_resume_v2(plan, cfg, identity, workdir)
    events = run_module._read_runtime_events(workdir)
    assert len(events) == 10
    assert (workdir / "runtime.jsonl").read_bytes().startswith(original)


def test_resume_refuses_corrupted_batch_without_refit(tmp_path, monkeypatch):
    cfg, plan, identity, calls = _resume_fixture(monkeypatch)
    workdir = tmp_path / "work"
    run_module._generate_or_resume_v2(plan, cfg, identity, workdir)
    batch = next(workdir.glob("batches/*/*.batch/forecasts.parquet"))
    batch.write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="invalid origin batch"):
        run_module._generate_or_resume_v2(plan, cfg, identity, workdir)
    assert not calls
    assert batch.read_bytes() == b"corrupt"


def _finalization_fixture(tmp_path, monkeypatch):
    from dataclasses import replace

    forecast_cfg, plan, identity, _ = _resume_fixture(monkeypatch)
    cfg = _cfg(tmp_path)
    identity = replace(identity, git_sha="a" * 40, run_id=cfg.run_id)
    workdir = cfg.output_dir / f".{cfg.run_id}.partial"
    generated, summary = run_module._generate_or_resume_v2(
        plan, forecast_cfg, identity, workdir
    )
    monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "a" * 40)
    monkeypatch.setattr(
        run_module,
        "benchmark_figures",
        lambda *a, **kw: {f"fig-{n}": plt.figure() for n in range(5)},
    )
    return cfg, generated, summary


def test_finalize_publishes_verified_evidence_and_hashes_manifest_last(
    tmp_path, monkeypatch
):
    import json

    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)
    finalize = getattr(run_module, "_finalize_benchmark_v2_artifacts", None)
    assert finalize is not None, "resumable finalizer is absent"
    original = Path.write_text
    manifests = []

    def observe(path, *args, **kwargs):
        assert not manifests, "run artifact written after manifest"
        if path.name == "manifest.json":
            assert len(list(path.parent.glob("*.parquet"))) == 10
            assert len(list((path.parent / "figures").glob("*.pdf"))) == 5
            assert not (cfg.output_dir / cfg.run_id).exists()
            manifests.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", observe)
    out = finalize(cfg, _result(), generated, summary, "datahash", Path.cwd())
    assert out == cfg.output_dir / cfg.run_id
    assert not summary.workdir.exists()
    manifest = json.loads((out / "manifest.json").read_text())
    paths = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert set(manifest["artifacts"]) == paths - {"manifest.json"}
    for name, digest in manifest["artifacts"].items():
        assert digest == run_module.file_sha256(out / name)
    assert (
        manifest["execution"]
        == run_module.contract_json(summary.identity)["runtime_provenance"]
    )
    assert manifest["resume"] == {
        "completed_batches": 4,
        "resumed_batches": 0,
        "generated_batches": 4,
        "total_batches": 4,
    }
    assert (
        "run-state.json" in paths
        and "runtime.jsonl" in paths
        and "progress.json" in paths
    )
    assert len(list(out.glob("batches/*/*.batch"))) == 4


@pytest.mark.parametrize(
    "fault",
    [
        "git",
        "dirty",
        "batch",
        "render",
        "existing",
        "race",
        "late_batch",
        "late_git",
        "publish_error",
    ],
)
def test_finalize_preserves_workdir_and_destination_on_failure(
    tmp_path, monkeypatch, fault
):
    cfg, generated, summary = _finalization_fixture(tmp_path, monkeypatch)
    finalize = getattr(run_module, "_finalize_benchmark_v2_artifacts", None)
    assert finalize is not None, "resumable finalizer is absent"
    out = cfg.output_dir / cfg.run_id
    sentinel = out / "sentinel"
    if fault == "git":
        monkeypatch.setattr(run_module, "require_clean_git", lambda repo: "b" * 40)
    elif fault == "dirty":

        def dirty(repo):
            raise RuntimeError("Git working tree is not clean")

        monkeypatch.setattr(run_module, "require_clean_git", dirty)
    elif fault == "late_git":
        shas = iter(("a" * 40, "b" * 40))
        monkeypatch.setattr(run_module, "require_clean_git", lambda repo: next(shas))
    elif fault == "publish_error":

        def unavailable(source, destination):
            raise OSError("atomic publication unavailable")

        monkeypatch.setattr(run_module, "_rename_noreplace", unavailable)
    elif fault == "batch":
        next(summary.workdir.glob("batches/*/*.batch/forecasts.parquet")).write_bytes(
            b"bad"
        )
    elif fault == "render":

        def render(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(run_module, "benchmark_figures", render)
    elif fault == "existing":
        out.mkdir()
        sentinel.write_text("preserve")
    elif fault == "race":
        original = run_module._rename_noreplace

        def race(source, destination):
            destination.mkdir()
            sentinel.write_text("preserve")
            original(source, destination)

        monkeypatch.setattr(run_module, "_rename_noreplace", race)
    elif fault == "late_batch":
        original_write = Path.write_text

        def damage(path, *args, **kwargs):
            value = original_write(path, *args, **kwargs)
            if path.name == "manifest.json":
                next(
                    summary.workdir.glob("batches/*/*.batch/forecasts.parquet")
                ).write_bytes(b"bad")
            return value

        monkeypatch.setattr(Path, "write_text", damage)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        finalize(cfg, _result(), generated, summary, "datahash", Path.cwd())
    assert summary.workdir.is_dir()
    assert (summary.workdir / "run-state.json").is_file()
    assert len(list(summary.workdir.glob("batches/*/*.batch"))) == 4
    if fault in ("existing", "race"):
        assert sentinel.read_text() == "preserve"
    else:
        assert not out.exists()


def test_runtime_console_bounds_and_sanitizes_error_but_preserves_evidence(
    tmp_path, capsys
):
    message = "multiline\n\x1b[31m" + "x" * 2000
    run_module._runtime_event(tmp_path, "error", message=message)
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert "\x1b" not in output
    assert len(output) <= 300
    assert run_module._read_runtime_events(tmp_path)[0]["message"] == message
