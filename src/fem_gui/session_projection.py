"""Decide whether an ordered Session transition needs scene projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.application import ChangeKind, SessionDelta, SessionSnapshot


_METADATA_CHANGES = frozenset(
    {ChangeKind.SAVED_STATE, ChangeKind.VALIDATIONS, ChangeKind.RUNS}
)
_PROJECT_SAVE_TERMINAL_REASONS = frozenset(
    {"project_save task failed", "project_save task cancelled"}
)


@dataclass(frozen=True, slots=True)
class SessionProjectionPlan:
    """The newest revision and the UI scope that can safely project it."""

    session_revision: int
    scope: Literal["metadata", "scene"]
    changed: frozenset[ChangeKind]

    @property
    def saved_state_transition(self) -> bool:
        return self.scope == "metadata" and self.changed == frozenset(
            {ChangeKind.SAVED_STATE}
        )

    @property
    def project_save_terminal_transition(self) -> bool:
        return self.scope == "metadata" and self.changed == frozenset(
            {ChangeKind.SESSION}
        )


def plan_session_projection(
    previous: SessionSnapshot,
    current: SessionSnapshot,
    *,
    requested_revision: int,
    applied_revision: int,
    delta: SessionDelta | None,
) -> SessionProjectionPlan | None:
    """Keep metadata incremental only when no intervening transition is lost.

    Unknown deltas, skipped revisions, and coalesced newer snapshots require
    the complete scene path. Revision and payload identities keep this check
    independent of Qt and avoid inspecting or copying model arrays.
    """

    if (
        requested_revision <= applied_revision
        or current.session_revision < requested_revision
        or (delta is not None and not delta.accepted)
    ):
        return None

    changed = frozenset() if delta is None else delta.changed
    ordered = (
        delta is not None
        and previous.session_revision == applied_revision
        and current.session_revision == requested_revision == applied_revision + 1
        and previous.session_id == current.session_id
    )
    metadata_change = (
        bool(changed) and changed <= _METADATA_CHANGES
    ) or (
        changed == frozenset({ChangeKind.SESSION})
        and delta is not None
        and delta.reason in _PROJECT_SAVE_TERMINAL_REASONS
    )
    scene_is_unchanged = (
        previous.is_open == current.is_open
        and previous.source_kind == current.source_kind
        and previous.project_revision == current.project_revision
        and previous.mesh_input_revision == current.mesh_input_revision
        and previous.model_revision == current.model_revision
        and previous.active_part_id == current.active_part_id
        and previous.artifact is current.artifact
        and previous.displayed_result_run_id == current.displayed_result_run_id
        and previous.displayed_result is current.displayed_result
    )
    metadata_only = (
        ordered
        and metadata_change
        and not delta.invalidated
        and scene_is_unchanged
    )
    return SessionProjectionPlan(
        session_revision=current.session_revision,
        scope="metadata" if metadata_only else "scene",
        changed=changed,
    )
