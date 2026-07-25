from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fem_agent.confirmation import (
    ConfirmationCorruptionError,
    ConfirmationNotReadyError,
    ConfirmationRequiredError,
    ConfirmationStore,
)
from fem_agent.schemas import (
    ExportFormat,
    ImportAnalysisSpec,
    ResultQuery,
    ResultQueryKind,
    UnitContext,
)
from fem_agent.state import RevisionStore, StaleRevisionError


def _spec(
    session_id: str,
    *,
    revision: int = 1,
    ready: bool = True,
) -> ImportAnalysisSpec:
    return ImportAnalysisSpec(
        session_id=session_id,
        revision=revision,
        source_artifact_id="art_input",
        source_sha256=hashlib.sha256(b"input").hexdigest(),
        unit_context=UnitContext(
            length="mm",
            force="N",
            stress="MPa",
            density="tonne/mm^3",
            acceleration="mm/s^2",
        )
        if ready
        else None,
        analysis_step="Step-1" if ready else None,
        requested_queries=(
            ResultQuery(kind=ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE),
        )
        if ready
        else (),
        export_formats=(ExportFormat.CSV,),
    )


def test_confirmation_is_persisted_and_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    current = revisions.initialize(
        _spec("ses_confirm"),
        idempotency_key="initial_import",
    )
    confirmations = ConfirmationStore(workspace, revisions)

    first = confirmations.confirm(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )
    retry = confirmations.confirm(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )
    reopened = ConfirmationStore(workspace).require_confirmed(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )

    assert retry == first
    assert reopened == first
    assert confirmations.current(current.session_id) == first
    assert confirmations.is_confirmed(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )


def test_confirmation_does_not_require_precomputed_result_queries(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    spec = replace(
        _spec("ses_confirm_without_queries"),
        requested_queries=(),
    )
    current = revisions.initialize(
        spec,
        idempotency_key="initial_import",
    )

    confirmation = ConfirmationStore(workspace, revisions).confirm(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )

    assert confirmation.revision == current.revision


def test_new_revision_naturally_invalidates_old_confirmation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    first = revisions.initialize(
        _spec("ses_invalidate"),
        idempotency_key="initial_import",
    )
    confirmations = ConfirmationStore(workspace, revisions)
    historical = confirmations.confirm(
        first.session_id,
        revision=1,
        revision_hash=first.revision_hash,
    )

    second = revisions.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="change_outputs",
        changes={"export_formats": (ExportFormat.VTK,)},
    )

    assert confirmations.get(first.session_id, 1) == historical
    assert confirmations.current(first.session_id) is None
    assert not confirmations.is_confirmed(first.session_id)
    with pytest.raises(ConfirmationRequiredError):
        confirmations.require_confirmed(
            first.session_id,
            revision=second.revision,
            revision_hash=second.revision_hash,
        )


def test_stale_revision_or_hash_cannot_be_confirmed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    first = revisions.initialize(
        _spec("ses_stale_confirm"),
        idempotency_key="initial_import",
    )
    second = revisions.mutate(
        first.session_id,
        expected_revision=1,
        idempotency_key="change_step",
        changes={"analysis_step": "Only-Step"},
    )
    confirmations = ConfirmationStore(workspace, revisions)

    with pytest.raises(StaleRevisionError):
        confirmations.confirm(
            first.session_id,
            revision=first.revision,
            revision_hash=first.revision_hash,
        )
    with pytest.raises(StaleRevisionError):
        confirmations.confirm(
            second.session_id,
            revision=second.revision,
            revision_hash="0" * 64,
        )


def test_incomplete_specification_cannot_be_confirmed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    current = revisions.initialize(
        _spec("ses_not_ready", ready=False),
        idempotency_key="initial_import",
    )
    confirmations = ConfirmationStore(workspace, revisions)

    with pytest.raises(ConfirmationNotReadyError):
        confirmations.confirm(
            current.session_id,
            revision=current.revision,
            revision_hash=current.revision_hash,
        )


def test_unconfirmed_current_revision_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    current = revisions.initialize(
        _spec("ses_required"),
        idempotency_key="initial_import",
    )
    confirmations = ConfirmationStore(workspace, revisions)

    with pytest.raises(ConfirmationRequiredError):
        confirmations.require_confirmed(
            current.session_id,
            revision=current.revision,
            revision_hash=current.revision_hash,
        )


def test_confirmation_tampering_is_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    revisions = RevisionStore(workspace)
    current = revisions.initialize(
        _spec("ses_confirmation_tamper"),
        idempotency_key="initial_import",
    )
    confirmations = ConfirmationStore(workspace, revisions)
    confirmations.confirm(
        current.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )
    path = (
        workspace
        / "sessions"
        / current.session_id
        / "confirmations"
        / "00000001.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["revision_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfirmationCorruptionError):
        confirmations.get(current.session_id, 1)
