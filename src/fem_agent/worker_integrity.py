"""Verify and recover persisted FEM worker responses from committed manifests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Mapping

from .artifacts import (
    ArtifactStore,
    RunDirectory,
    atomic_write_json,
    read_json_file,
    safe_child,
)
from .diagnostics import has_errors
from .schemas import (
    ArtifactRecord,
    Diagnostic,
    ResultSummary,
    RunManifest,
    RunStatus,
)
from .state import RevisionRecord
from .worker_protocol import (
    WORKER_SCHEMA_VERSION,
    WorkerRequest,
    WorkerResponse,
    _array,
    _mapping,
    _sha256,
    _timestamp,
)


class WorkerResponseIntegrityError(RuntimeError):
    """A persisted response for the current attempt failed validation."""


def _recover_response_from_manifest(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    response_path: Path,
) -> WorkerResponse:
    """Rebuild a missing response only from a fully verified commit manifest."""

    try:
        response = _verified_worker_response_from_manifest(
            store,
            record,
            run,
            allow_manifest_registration=True,
        )
        atomic_write_json(response_path, response.to_dict())
        return response
    except WorkerResponseIntegrityError:
        raise
    except Exception as error:
        raise WorkerResponseIntegrityError(
            "The committed worker manifest failed integrity validation."
        ) from error


def load_verified_worker_response(
    store: ArtifactStore,
    record: RevisionRecord,
    run_id: str,
) -> WorkerResponse:
    """Load one persisted response after rebuilding its committed truth."""

    try:
        run = store.run_directory(record.session_id, run_id)
        response_path = safe_child(
            run.path,
            "logs",
            "worker-response.json",
        )
        persisted = WorkerResponse.from_dict(read_json_file(response_path))
        _require_response_identity(persisted, record, run)
        verified = _verified_worker_response_from_manifest(
            store,
            record,
            run,
            allow_manifest_registration=False,
        )
        if persisted != verified:
            raise ValueError(
                "persisted worker response does not match its committed manifest"
            )
        return persisted
    except WorkerResponseIntegrityError:
        raise
    except Exception as error:
        raise WorkerResponseIntegrityError(
            "The persisted worker response failed integrity validation."
        ) from error


def _verified_worker_response_from_manifest(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    *,
    allow_manifest_registration: bool,
) -> WorkerResponse:
    manifest_path = safe_child(run.path, "manifest.json")
    manifest = RunManifest.from_dict(read_json_file(manifest_path))
    _require_manifest_identity(store, record, run, manifest)
    verified_paths = _verify_manifest_artifacts(
        store,
        record.session_id,
        run,
        manifest,
    )
    result_summary = _manifest_result_summary(
        run,
        manifest,
        verified_paths,
    )
    manifest_artifact = _verified_manifest_artifact(
        store,
        record.session_id,
        run,
        manifest,
        allow_registration=allow_manifest_registration,
    )
    response = WorkerResponse(
        session_id=manifest.session_id,
        revision=manifest.revision,
        revision_hash=manifest.revision_hash,
        run_id=manifest.run_id,
        status=manifest.status,
        result_summary=result_summary,
        artifacts=(*manifest.artifacts, manifest_artifact),
        diagnostics=manifest.diagnostics,
        elapsed_seconds=manifest.durations_seconds["total"],
    )
    _require_response_identity(response, record, run)
    return response


def _require_manifest_identity(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    manifest: RunManifest,
) -> None:
    if (
        manifest.session_id != record.session_id
        or manifest.revision != record.revision
        or manifest.revision_hash != record.revision_hash
        or manifest.run_id != run.run_id
        or manifest.source_sha256 != record.spec.source_sha256
    ):
        raise ValueError("worker manifest identity does not match its run")
    if manifest.status not in {
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }:
        raise ValueError("worker manifest does not contain a terminal status")
    if manifest.status == RunStatus.SUCCEEDED and has_errors(
        manifest.diagnostics
    ):
        raise ValueError("successful worker manifest contains error diagnostics")
    if (
        manifest.status in {RunStatus.FAILED, RunStatus.CANCELLED}
        and not has_errors(manifest.diagnostics)
    ):
        raise ValueError("failed worker manifest contains no error diagnostic")
    if (
        manifest.unit_context != record.spec.unit_context
        or manifest.analysis_step != record.spec.analysis_step
    ):
        raise ValueError("worker manifest analysis specification mismatch")
    expected_parameters = {
        "requested_queries": [
            item.to_dict() for item in record.spec.requested_queries
        ],
        "export_formats": [
            item.value for item in record.spec.export_formats
        ],
        "resource_limits": record.spec.resource_limits.to_dict(),
    }
    actual_parameters = dict(manifest.tool_parameters)
    solution_model_sha256 = actual_parameters.pop(
        "solution_model_sha256",
        None,
    )
    if solution_model_sha256 is not None:
        _sha256(solution_model_sha256, "solution_model_sha256")
    if actual_parameters != expected_parameters:
        raise ValueError("worker manifest tool parameters mismatch")
    has_solution = any(
        artifact.kind == "solution"
        for artifact in manifest.artifacts
    )
    if solution_model_sha256 is not None and not has_solution:
        raise ValueError(
            "worker manifest has a model fingerprint without a solution"
        )
    if any(
        diagnostic not in manifest.diagnostics
        for diagnostic in manifest.validation_diagnostics
    ):
        raise ValueError("worker manifest validation diagnostics mismatch")
    if set(manifest.timestamps) != {"started_at", "completed_at"}:
        raise ValueError("worker manifest timestamps are incomplete")
    started = _timestamp(manifest.timestamps["started_at"], "started_at")
    completed = _timestamp(manifest.timestamps["completed_at"], "completed_at")
    if completed < started:
        raise ValueError("worker manifest completion precedes its start")
    if "total" not in manifest.durations_seconds:
        raise ValueError("worker manifest is missing its total duration")

    source = store.get_artifact(
        record.session_id,
        record.spec.source_artifact_id,
    )
    if (
        source.kind != "input"
        or source.sha256 != record.spec.source_sha256
    ):
        raise ValueError("worker manifest source artifact mismatch")
    store.resolve_artifact(
        record.session_id,
        source.artifact_id,
        verify=True,
    )


def _verify_manifest_artifacts(
    store: ArtifactStore,
    session_id: str,
    run: RunDirectory,
    manifest: RunManifest,
) -> dict[str, Path]:
    records = manifest.artifacts
    identifiers = [item.artifact_id for item in records]
    display_paths = [item.display_path for item in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("worker manifest contains duplicate artifact identifiers")
    if len(set(display_paths)) != len(display_paths):
        raise ValueError("worker manifest contains duplicate artifact paths")

    verified: dict[str, Path] = {}
    export_kinds = set(manifest.tool_parameters["export_formats"])
    allowed_kinds = {
        "diagnostics",
        "result_summary",
        "solution",
        *export_kinds,
    }
    for expected in records:
        parts = PurePosixPath(expected.display_path).parts
        if len(parts) < 3 or parts[:2] != ("runs", run.run_id):
            raise ValueError("worker manifest artifact escapes its run")
        if expected.kind == "manifest":
            raise ValueError("worker manifest cannot list itself as an artifact")
        if expected.kind not in allowed_kinds:
            raise ValueError("worker manifest contains an unexpected artifact kind")
        if expected.kind in export_kinds and (
            len(parts) != 4 or parts[2] != "exports"
        ):
            raise ValueError("worker export artifact is stored at an invalid path")
        persisted = store.get_artifact(session_id, expected.artifact_id)
        if persisted != expected:
            raise ValueError("worker manifest artifact metadata mismatch")
        resolved = store.resolve_artifact(
            session_id,
            expected.artifact_id,
            verify=True,
        )
        if not resolved.is_relative_to(run.path):
            raise ValueError("worker manifest artifact resolves outside its run")
        verified[expected.artifact_id] = resolved

    diagnostics = [item for item in records if item.kind == "diagnostics"]
    expected_diagnostics_path = (
        PurePosixPath("runs") / run.run_id / "diagnostics.json"
    ).as_posix()
    if (
        len(diagnostics) != 1
        or diagnostics[0].display_path != expected_diagnostics_path
    ):
        raise ValueError("worker manifest diagnostics artifact mismatch")
    diagnostics_payload = read_json_file(
        verified[diagnostics[0].artifact_id]
    )
    if set(diagnostics_payload) != {"schema_version", "diagnostics"}:
        raise ValueError("worker diagnostics artifact has invalid fields")
    if diagnostics_payload["schema_version"] != WORKER_SCHEMA_VERSION:
        raise ValueError("worker diagnostics artifact has an unsupported version")
    persisted_diagnostics = tuple(
        Diagnostic.from_dict(_mapping(item, "diagnostic"))
        for item in _array(
            diagnostics_payload["diagnostics"],
            "diagnostics",
        )
    )
    if persisted_diagnostics != manifest.diagnostics:
        raise ValueError("worker diagnostics artifact does not match its manifest")

    solutions = [item for item in records if item.kind == "solution"]
    expected_solution_path = (
        PurePosixPath("runs") / run.run_id / "solution.npy"
    ).as_posix()
    if len(solutions) > 1:
        raise ValueError("worker manifest contains duplicate solution artifacts")
    if solutions and solutions[0].display_path != expected_solution_path:
        raise ValueError("worker solution is stored at an invalid path")
    solution_path = safe_child(run.path, "solution.npy")
    if solution_path.exists() != bool(solutions):
        raise ValueError("worker manifest solution artifact mismatch")

    listed_display_paths = set(display_paths)
    exports = safe_child(run.path, "exports")
    for candidate in exports.iterdir():
        candidate_display_path = (
            PurePosixPath("runs")
            / run.run_id
            / "exports"
            / candidate.name
        ).as_posix()
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate_display_path not in listed_display_paths
        ):
            raise ValueError("worker export files do not match their manifest")
    return verified


def _manifest_result_summary(
    run: RunDirectory,
    manifest: RunManifest,
    verified_paths: Mapping[str, Path],
) -> ResultSummary | None:
    result_records = tuple(
        item for item in manifest.artifacts if item.kind == "result_summary"
    )
    result_path = safe_child(run.path, "result-summary.json")
    if manifest.result_summary_artifact_id is None:
        if result_records or result_path.exists():
            raise ValueError("worker manifest result-summary identity mismatch")
        return None
    if (
        len(result_records) != 1
        or result_records[0].artifact_id
        != manifest.result_summary_artifact_id
    ):
        raise ValueError("worker manifest result-summary artifact mismatch")
    expected_display_path = (
        PurePosixPath("runs") / run.run_id / "result-summary.json"
    ).as_posix()
    record = result_records[0]
    if record.display_path != expected_display_path:
        raise ValueError("worker result summary is stored at an invalid path")
    summary = ResultSummary.from_dict(
        read_json_file(verified_paths[record.artifact_id])
    )
    if summary.run_id != run.run_id or summary.step != manifest.analysis_step:
        raise ValueError("worker result summary identity mismatch")
    if any(
        scalar.run_id != run.run_id or scalar.step != manifest.analysis_step
        for scalar in summary.scalars
    ):
        raise ValueError("worker scalar result identity mismatch")
    return summary


def _verified_manifest_artifact(
    store: ArtifactStore,
    session_id: str,
    run: RunDirectory,
    manifest: RunManifest,
    *,
    allow_registration: bool,
) -> ArtifactRecord:
    expected_display_path = (
        PurePosixPath("runs") / run.run_id / "manifest.json"
    ).as_posix()
    run_records = tuple(
        item
        for item in store.list_artifacts(session_id)
        if _artifact_belongs_to_run(item, run.run_id)
    )
    manifest_records = tuple(
        item
        for item in run_records
        if item.display_path == expected_display_path
    )
    if len(manifest_records) > 1:
        raise ValueError("worker manifest has duplicate artifact metadata")
    listed_ids = {item.artifact_id for item in manifest.artifacts}
    allowed_ids = listed_ids | {
        item.artifact_id for item in manifest_records
    }
    if any(item.artifact_id not in allowed_ids for item in run_records):
        raise ValueError("worker run contains artifacts absent from its manifest")

    manifest_path = safe_child(run.path, "manifest.json")
    if manifest_records:
        manifest_artifact = manifest_records[0]
        if manifest_artifact.kind != "manifest":
            raise ValueError("worker manifest artifact has the wrong kind")
        resolved = store.resolve_artifact(
            session_id,
            manifest_artifact.artifact_id,
            verify=True,
        )
        if resolved != manifest_path:
            raise ValueError("worker manifest artifact path mismatch")
    elif allow_registration:
        manifest_artifact = store.register_run_artifact(
            session_id,
            run.run_id,
            manifest_path,
            kind="manifest",
        )
    else:
        raise ValueError("worker manifest artifact metadata is missing")
    reloaded = RunManifest.from_dict(read_json_file(manifest_path))
    if reloaded != manifest:
        raise ValueError("worker manifest changed during recovery")
    store.resolve_artifact(
        session_id,
        manifest_artifact.artifact_id,
        verify=True,
    )
    return manifest_artifact


def _artifact_belongs_to_run(
    artifact: ArtifactRecord,
    run_id: str,
) -> bool:
    parts = PurePosixPath(artifact.display_path).parts
    return len(parts) >= 3 and parts[:2] == ("runs", run_id)


def _require_response_identity(
    response: WorkerResponse,
    record: RevisionRecord,
    run: RunDirectory,
) -> None:
    if (
        response.session_id != record.session_id
        or response.revision != record.revision
        or response.revision_hash != record.revision_hash
        or response.run_id != run.run_id
    ):
        raise ValueError("persisted worker response identity does not match its run")
    if response.status == RunStatus.RUNNING:
        raise ValueError("persisted worker response is not terminal")
    if response.status == RunStatus.SUCCEEDED and has_errors(response.diagnostics):
        raise ValueError("successful worker response contains error diagnostics")
    if (
        response.status in {RunStatus.FAILED, RunStatus.CANCELLED}
        and not has_errors(response.diagnostics)
    ):
        raise ValueError("failed worker response contains no error diagnostic")


def _require_request_identity(
    request: WorkerRequest,
    record: RevisionRecord,
    run: RunDirectory,
    idempotency_key: str,
) -> None:
    if (
        request.session_id != record.session_id
        or request.revision != record.revision
        or request.revision_hash != record.revision_hash
        or request.run_id != run.run_id
        or request.idempotency_key != idempotency_key
    ):
        raise ValueError("persisted worker request identity does not match its run")


__all__ = ["WorkerResponseIntegrityError", "load_verified_worker_response"]
