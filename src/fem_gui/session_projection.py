"""Plan incremental Session, model and result projections from identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.application import ChangeKind, SessionDelta, SessionSnapshot
from fem.application.results import ResultCatalog, ResultProvider, ResultSourceKey, ScalarFieldSelection


_METADATA_CHANGES = frozenset(
    {ChangeKind.SAVED_STATE, ChangeKind.VALIDATIONS, ChangeKind.RUNS}
)
_PROJECT_SAVE_TERMINAL_REASONS = frozenset(
    {"project_save task failed", "project_save task cancelled"}
)
_DEFINITION_CHANGES = frozenset(
    {
        ChangeKind.PROJECT_INPUTS,
        ChangeKind.NAMED_REGIONS,
        ChangeKind.DEFINITIONS,
        ChangeKind.MODEL,
        ChangeKind.VALIDATIONS,
        ChangeKind.RUNS,
        ChangeKind.RESULTS,
        ChangeKind.DISPLAYED_RESULT,
    }
)


def authoring_facts_key(snapshot: SessionSnapshot) -> tuple[object, ...]:
    """Identify Session-owned capability inputs without scanning model data.

    Project/model/mesh revisions cover region, definition and Part edits.
    Payload identities also distinguish independently detached snapshots;
    selecting an active Part has its own dependency outside those revisions.
    Saved paths, validations and runs affect only the current lifecycle.
    """

    return (
        snapshot.session_id,
        snapshot.is_open,
        snapshot.source_kind,
        snapshot.project_revision,
        snapshot.mesh_input_revision,
        snapshot.model_revision,
        snapshot.active_part_id,
        id(snapshot.model),
        id(snapshot.geometry_recipe),
        id(snapshot.mesh_settings),
    )


@dataclass(frozen=True, slots=True)
class SessionProjectionPlan:
    """The newest revision and the UI scope that can safely project it."""

    session_revision: int
    scope: Literal["metadata", "definitions", "scene"]
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


@dataclass(frozen=True, slots=True)
class ModelProjectionPlan:
    """The concrete model action allowed by the installed artifact identities."""

    scope: Literal["definitions", "scene"]
    action: Literal["clear", "keep", "rebind", "install"]
    invalidate_selection_topology: bool


def plan_model_projection(
    previous: SessionSnapshot,
    current: SessionSnapshot,
    *,
    scope: Literal["metadata", "definitions", "scene"],
    geometry_artifact_id: str | None,
    viewport_artifact_id: str | None,
) -> ModelProjectionPlan:
    """Require an actually installed old mesh before retaining its topology."""

    previous_id = None if previous.artifact is None else previous.artifact.artifact_id
    current_id = None if current.artifact is None else current.artifact.artifact_id
    definitions = (
        scope == "definitions"
        and previous_id is not None
        and geometry_artifact_id == viewport_artifact_id == previous_id
    )
    effective_scope = "definitions" if definitions else "scene"
    if current_id is None:
        action = "clear"
    elif definitions:
        action = "rebind"
    elif (
        previous_id != current_id
        or geometry_artifact_id != current_id
        or viewport_artifact_id != current_id
    ):
        action = "install"
    else:
        action = "keep"
    return ModelProjectionPlan(
        effective_scope, action,
        not definitions and (current_id is None or previous_id != current_id),
    )


@dataclass(frozen=True, slots=True)
class ResultProjectionConsumers:
    """Only identities observed from the installed result consumers."""

    provider: ResultProvider | None
    selection: ScalarFieldSelection | None
    viewport_identity: tuple[ResultSourceKey, int, ScalarFieldSelection, str | None] | None
    catalog: ResultCatalog | None
    tree_has_selection: bool
    has_inspection: bool
    inspection_provider: ResultProvider | None


def result_consumers_are_current(
    provider: ResultProvider, consumers: ResultProjectionConsumers,
) -> bool:
    """Require the same provider, field and generation in every consumer."""

    if consumers.provider is not provider or type(consumers.selection) is not ScalarFieldSelection:
        return False
    selection = consumers.selection
    try:
        availability = provider.field_status(selection.field_key)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        selection.component in availability.descriptor.columns
        and consumers.viewport_identity == (
            provider.source, provider.snapshot.generation, selection, provider.source.run_id,
        )
        and consumers.catalog is provider.catalog()
        and consumers.tree_has_selection
        and (not consumers.has_inspection or consumers.inspection_provider is provider)
    )


def plan_session_projection(
    previous: SessionSnapshot,
    current: SessionSnapshot,
    *,
    requested_revision: int,
    applied_revision: int,
    delta: SessionDelta | None,
    preserve_mesh_topology: bool = False,
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
    definition_only = (
        preserve_mesh_topology
        and ordered
        and bool(changed & {ChangeKind.DEFINITIONS, ChangeKind.NAMED_REGIONS})
        and changed <= _DEFINITION_CHANGES
        and previous.artifact is not None
        and current.artifact is not None
        and current.model_current
        and previous.source_kind == current.source_kind != "result"
        and previous.active_part_id == current.active_part_id
        and previous.part_revisions == current.part_revisions
        and previous.geometry_recipe is current.geometry_recipe
        and previous.mesh_settings is current.mesh_settings
        and (
            previous.mesh_input_revision == current.mesh_input_revision
            or previous.model.mesh is current.model.mesh
        )
    )
    return SessionProjectionPlan(
        session_revision=current.session_revision,
        scope=(
            "metadata" if metadata_only
            else "definitions" if definition_only
            else "scene"
        ),
        changed=changed,
    )
