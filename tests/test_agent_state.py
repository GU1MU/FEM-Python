from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fem_agent.schemas import (
    ExportFormat,
    ImportAnalysisSpec,
    ResultQuery,
    ResultQueryKind,
    UnitContext,
)
from fem_agent.state import (
    IdempotencyConflictError,
    RevisionCorruptionError,
    RevisionStore,
    StaleRevisionError,
    hash_revision_spec,
)


def _spec(
    session_id: str,
    *,
    revision: int = 1,
    unit_context: UnitContext | None = None,
    analysis_step: str | None = "Step-1",
    queries: tuple[ResultQuery, ...] | None = None,
) -> ImportAnalysisSpec:
    return ImportAnalysisSpec(
        session_id=session_id,
        revision=revision,
        source_artifact_id="art_input",
        source_sha256=hashlib.sha256(b"immutable input").hexdigest(),
        unit_context=unit_context
        or UnitContext(
            length="mm",
            force="N",
            stress="MPa",
            density="tonne/mm^3",
            acceleration="mm/s^2",
        ),
        analysis_step=analysis_step,
        requested_queries=queries
        if queries is not None
        else (
            ResultQuery(kind=ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE),
        ),
        export_formats=(ExportFormat.CSV, ExportFormat.VTK),
        assumptions=("用户确认单位制",),
    )


def test_initial_revision_round_trips_with_deterministic_hash(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = RevisionStore(workspace)
    spec = _spec("ses_state")

    record = store.initialize(spec, idempotency_key="initial_import")
    reopened = RevisionStore(workspace).get("ses_state", 1)

    assert reopened == record
    assert record.revision_hash == hash_revision_spec(spec)
    assert len(record.revision_hash) == 64
    assert record.spec.assumptions == ("用户确认单位制",)
    assert store.require_current(
        "ses_state",
        expected_revision=1,
        expected_hash=record.revision_hash,
    ) == record


def test_mutation_is_append_only_and_invalidates_stale_expectation(
    tmp_path: Path,
) -> None:
    store = RevisionStore(tmp_path / "workspace")
    first = store.initialize(
        _spec("ses_mutation"),
        idempotency_key="initial_import",
    )
    new_units = UnitContext(
        length="m",
        force="N",
        stress="Pa",
        density="kg/m^3",
        acceleration="m/s^2",
    )

    second = store.mutate(
        first.session_id,
        expected_revision=first.revision,
        idempotency_key="set_units",
        operation="set_unit_context",
        changes={"unit_context": new_units},
    )

    assert second.revision == 2
    assert second.expected_revision == 1
    assert second.spec.unit_context == new_units
    assert store.get(first.session_id, 1) == first
    assert store.latest(first.session_id) == second
    with pytest.raises(StaleRevisionError):
        store.require_current(first.session_id, expected_revision=1)


def test_same_mutation_idempotency_key_returns_original_revision(
    tmp_path: Path,
) -> None:
    store = RevisionStore(tmp_path / "workspace")
    first = store.initialize(
        _spec("ses_idempotent"),
        idempotency_key="initial_import",
    )
    changes = {"analysis_step": "Only-Step"}

    second = store.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="select_step",
        changes=changes,
    )
    retry = store.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="select_step",
        changes=changes,
    )

    assert retry == second
    assert len(store.list_records(first.session_id)) == 2


def test_reusing_idempotency_key_for_different_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    store = RevisionStore(tmp_path / "workspace")
    first = store.initialize(
        _spec("ses_conflict"),
        idempotency_key="initial_import",
    )
    store.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="same_key",
        changes={"analysis_step": "Step-A"},
    )

    with pytest.raises(IdempotencyConflictError):
        store.mutate(
            first.session_id,
            expected_revision=1,
            idempotency_key="same_key",
            changes={"analysis_step": "Step-B"},
        )


def test_new_mutation_against_stale_revision_fails(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "workspace")
    first = store.initialize(
        _spec("ses_stale"),
        idempotency_key="initial_import",
    )
    store.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="first_change",
        changes={"analysis_step": "Step-A"},
    )

    with pytest.raises(StaleRevisionError):
        store.mutate(
            first.session_id,
            expected_revision=1,
            idempotency_key="late_change",
            changes={"analysis_step": "Step-B"},
        )
    with pytest.raises(StaleRevisionError):
        store.mutate(
            first.session_id,
            expected_revision=9,
            idempotency_key="future_change",
            changes={"analysis_step": "Step-C"},
        )


def test_revision_tampering_is_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = RevisionStore(workspace)
    store.initialize(
        _spec("ses_tamper"),
        idempotency_key="initial_import",
    )
    path = (
        workspace
        / "sessions"
        / "ses_tamper"
        / "revisions"
        / "00000001.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["spec"]["analysis_step"] = "Changed"
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RevisionCorruptionError):
        store.get("ses_tamper", 1)


def test_revision_sequence_gap_is_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = RevisionStore(workspace)
    store.initialize(
        _spec("ses_gap"),
        idempotency_key="initial_import",
    )
    revisions = workspace / "sessions" / "ses_gap" / "revisions"
    (revisions / "00000003.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RevisionCorruptionError):
        store.list_records("ses_gap")


def test_revision_files_are_published_without_temporary_residue(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = RevisionStore(workspace)
    store.initialize(
        _spec("ses_atomic"),
        idempotency_key="initial_import",
    )
    revisions = workspace / "sessions" / "ses_atomic" / "revisions"

    assert (revisions / "00000001.json").is_file()
    assert not list(revisions.glob("*.tmp"))
