"""Atomic native-model branches for geometry iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Callable, Iterable

from fem.application import ModelSession, RevisionConflictError, SessionSnapshot

from .workspace import DocumentLineage, FEMWorkspace, WorkspaceDocument


_REPORT_ITEM_LIMIT = 6
_REPORT_TEXT_BYTES = 256
_REPORT_ITEM_TEXT_BYTES = 128
_REPORT_MAX_BYTES = 24_000


def _text(
    value: object,
    label: str,
    *,
    maximum: int = _REPORT_TEXT_BYTES,
) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    clean = value.strip()
    if len(clean.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds its bound")
    return clean


def _items(values: Iterable[object], label: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _text(
                    str(value),
                    label,
                    maximum=_REPORT_ITEM_TEXT_BYTES,
                )
                for value in values
            }
        )
    )


class GeometryEditPolicy(str, Enum):
    IN_PLACE = "in_place"
    BRANCH = "branch"


class GeometryEditUnavailableError(RuntimeError):
    """The selected workspace document cannot accept a geometry edit."""


def geometry_edit_policy(document: WorkspaceDocument) -> GeometryEditPolicy:
    """Choose the deterministic geometry-edit mode for one document."""

    if type(document) is not WorkspaceDocument:
        raise TypeError("document must be WorkspaceDocument")
    if document.kind != "model":
        raise GeometryEditUnavailableError("geometry editing requires a model document")
    snapshot = document.session.projection_snapshot()
    if not snapshot.is_open:
        raise GeometryEditUnavailableError(
            "geometry editing requires an open model document"
        )
    if snapshot.source_kind != "native":
        raise GeometryEditUnavailableError(
            "geometry editing requires a native model document"
        )
    if not snapshot.parts or not any(
        part.geometry_recipe is not None for part in snapshot.parts
    ):
        raise GeometryEditUnavailableError(
            "geometry editing requires existing native geometry"
        )
    has_downstream_state = bool(
        snapshot.artifact is not None
        or snapshot.named_regions
        or snapshot.materials
        or snapshot.sections
        or snapshot.assignments
        or snapshot.steps
        or snapshot.validations
        or snapshot.runs
        or snapshot.result_generations
        or snapshot.displayed_result is not None
    )
    return (
        GeometryEditPolicy.BRANCH
        if has_downstream_state
        else GeometryEditPolicy.IN_PLACE
    )


@dataclass(frozen=True, slots=True)
class MigrationItems:
    preserved: tuple[str, ...] = ()
    rewritten: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    omitted_item_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        omitted = 0
        for field_name in ("preserved", "rewritten", "dropped"):
            values = _items(getattr(self, field_name), field_name)
            normalized[field_name] = values
            omitted += max(0, len(values) - _REPORT_ITEM_LIMIT)
            object.__setattr__(
                self,
                field_name,
                values[:_REPORT_ITEM_LIMIT],
            )
        if (
            set(normalized["preserved"]) & set(normalized["rewritten"])
            or set(normalized["preserved"]) & set(normalized["dropped"])
            or set(normalized["rewritten"]) & set(normalized["dropped"])
        ):
            raise ValueError("migration item states must be disjoint")
        object.__setattr__(self, "omitted_item_count", omitted)

    @property
    def truncated(self) -> bool:
        return self.omitted_item_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "preserved": list(self.preserved),
            "rewritten": list(self.rewritten),
            "dropped": list(self.dropped),
            "truncated": self.truncated,
            "omitted_item_count": self.omitted_item_count,
        }


@dataclass(frozen=True, slots=True)
class MigrationReport:
    source_document_id: int
    source_session_id: str
    source_session_revision: int
    source_project_revision: int
    source_model_name: str
    source_run_id: str | None
    target_document_id: int
    target_session_id: str
    target_session_revision: int
    target_project_revision: int
    target_model_name: str
    part_id: str
    named_regions: MigrationItems
    mesh_settings: MigrationItems
    local_mesh_controls: MigrationItems
    materials: MigrationItems
    sections: MigrationItems
    assignments: MigrationItems
    analysis_steps: MigrationItems
    requires_remesh: bool = True
    validations_reset: bool = True
    runs_migrated: bool = False
    results_migrated: bool = False
    reason: str = "geometry_edit"

    def __post_init__(self) -> None:
        for field_name in ("source_document_id", "target_document_id"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "source_session_revision",
            "source_project_revision",
            "target_session_revision",
            "target_project_revision",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "source_session_id",
            "source_model_name",
            "target_session_id",
            "target_model_name",
            "part_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if self.source_run_id is not None:
            object.__setattr__(
                self,
                "source_run_id",
                _text(self.source_run_id, "source_run_id"),
            )
        for field_name in (
            "named_regions",
            "mesh_settings",
            "local_mesh_controls",
            "materials",
            "sections",
            "assignments",
            "analysis_steps",
        ):
            if type(getattr(self, field_name)) is not MigrationItems:
                raise TypeError(f"{field_name} must be MigrationItems")
        for field_name in (
            "requires_remesh",
            "validations_reset",
            "runs_migrated",
            "results_migrated",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")
        if not self.requires_remesh or not self.validations_reset:
            raise ValueError("geometry branches must remesh and reset validations")
        if self.runs_migrated or self.results_migrated:
            raise ValueError("geometry branches cannot migrate runs or results")
        if self.reason != "geometry_edit":
            raise ValueError("unsupported migration reason")
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > _REPORT_MAX_BYTES:
            raise ValueError("migration report exceeds its provider-safe budget")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": {
                "document_id": self.source_document_id,
                "session_id": self.source_session_id,
                "session_revision": self.source_session_revision,
                "project_revision": self.source_project_revision,
                "model_name": self.source_model_name,
                "run_id": self.source_run_id,
            },
            "target": {
                "document_id": self.target_document_id,
                "session_id": self.target_session_id,
                "session_revision": self.target_session_revision,
                "project_revision": self.target_project_revision,
                "model_name": self.target_model_name,
            },
            "part_id": self.part_id,
            "reason": self.reason,
            "named_regions": self.named_regions.to_dict(),
            "mesh_settings": self.mesh_settings.to_dict(),
            "local_mesh_controls": self.local_mesh_controls.to_dict(),
            "materials": self.materials.to_dict(),
            "sections": self.sections.to_dict(),
            "assignments": self.assignments.to_dict(),
            "analysis_steps": self.analysis_steps.to_dict(),
            "requires_remesh": self.requires_remesh,
            "validations": "reset",
            "runs": "not_migrated",
            "results": "not_migrated",
            "source_state": {
                "runs": "retained",
                "results": "retained",
            },
            "target_state": {
                "mesh": "not_migrated",
                "validations": "reset",
                "runs": "not_migrated",
                "results": "not_migrated",
            },
        }


@dataclass(frozen=True, slots=True)
class ModelIterationResult:
    document: WorkspaceDocument
    report: MigrationReport


class ModelIterationService:
    """Create and activate one geometry-edited child model atomically."""

    def __init__(self, workspace: FEMWorkspace) -> None:
        if type(workspace) is not FEMWorkspace:
            raise TypeError("workspace must be FEMWorkspace")
        self._workspace = workspace

    def branch_geometry_edit(
        self,
        source_document_id: int,
        part_id: str,
        geometry_recipe: Any,
        *,
        target_model_name: str | None = None,
        source_run_id: str | None = None,
        expected_source_session_revision: int | None = None,
        activate_child: Callable[[WorkspaceDocument], bool] | None = None,
    ) -> ModelIterationResult:
        source = self._workspace.document(source_document_id)
        geometry_edit_policy(source)
        source_snapshot = source.session.projection_snapshot()
        source_revision = source_snapshot.session_revision
        if (
            expected_source_session_revision is not None
            and expected_source_session_revision != source_revision
        ):
            raise RevisionConflictError(
                expected_source_session_revision,
                source_revision,
            )
        normalized_run_id = None
        if source_run_id is not None:
            normalized_run_id = _text(source_run_id, "source_run_id")
            if normalized_run_id not in {run.run_id for run in source_snapshot.runs}:
                raise ValueError("source_run_id does not belong to the source session")
        model_name = (
            self._next_model_name(source_snapshot.model_name or source.display_name)
            if target_model_name is None
            else _text(target_model_name, "target_model_name")
        )
        branch_snapshot = source.session.project_snapshot_for_branch(
            model_name,
            expected_session_revision=source_revision,
        )
        child_session = ModelSession()
        child_session.replace_from_snapshot(branch_snapshot)
        before_edit = child_session.projection_snapshot()
        child_session.replace_part_geometry(part_id, geometry_recipe)
        after_edit = child_session.projection_snapshot()
        if source.session.session_revision != source_revision:
            raise RevisionConflictError(
                source_revision,
                source.session.session_revision,
            )

        lineage = DocumentLineage(
            source_document_id=source.document_id,
            source_session_id=source_snapshot.session_id,
            source_session_revision=source_revision,
            source_project_revision=source_snapshot.project_revision,
            source_run_id=normalized_run_id,
        )
        workspace_state = self._workspace_state()
        try:
            child = self._workspace.add_model(
                child_session,
                after_edit,
                display_name=model_name,
                source_path=None,
                lineage=lineage,
            )
            if child.session is not child_session:
                raise RuntimeError("workspace did not insert the model branch")
            self._workspace.activate(child)
            report = _migration_report(
                source,
                source_snapshot,
                child,
                before_edit,
                after_edit,
                part_id,
                normalized_run_id,
            )
            if activate_child is not None and not activate_child(child):
                raise RuntimeError("geometry iteration GUI activation failed")
            return ModelIterationResult(child, report)
        except Exception:
            self._rollback_workspace(workspace_state)
            raise

    def _next_model_name(self, source_model_name: str) -> str:
        base = _text(source_model_name, "source_model_name")
        preferred = f"{base}-迭代"
        existing = {
            str(document.projection.model_name or document.display_name).casefold()
            for document in self._workspace.models.values()
        }
        if preferred.casefold() not in existing:
            return preferred
        number = 1
        while f"{preferred}({number})".casefold() in existing:
            number += 1
        return f"{preferred}({number})"

    def _workspace_state(self) -> tuple[set[int], int | None, str | None, int, int]:
        return (
            {document.document_id for document in self._workspace.documents()},
            self._workspace.active_document_id,
            self._workspace.active_kind,
            self._workspace._next_document_id,
            self._workspace._next_model_number,
        )

    def _rollback_workspace(
        self,
        state: tuple[set[int], int | None, str | None, int, int],
    ) -> None:
        document_ids, active_id, active_kind, next_document_id, next_model_number = (
            state
        )
        for registry, path_index in (
            (self._workspace.models, self._workspace.model_paths),
            (self._workspace.results, self._workspace.result_paths),
        ):
            for document_id in tuple(registry):
                if document_id in document_ids:
                    continue
                removed = registry.pop(document_id)
                if removed.source_path is not None:
                    from .workspace import canonical_path

                    path_index.pop(canonical_path(removed.source_path), None)
        self._workspace.active_document_id = active_id
        self._workspace.active_kind = active_kind
        self._workspace._next_document_id = next_document_id
        self._workspace._next_model_number = next_model_number


def _migration_report(
    source: WorkspaceDocument,
    source_snapshot: SessionSnapshot,
    target: WorkspaceDocument,
    before: SessionSnapshot,
    after: SessionSnapshot,
    part_id: str,
    source_run_id: str | None,
) -> MigrationReport:
    return MigrationReport(
        source_document_id=source.document_id,
        source_session_id=source_snapshot.session_id,
        source_session_revision=source_snapshot.session_revision,
        source_project_revision=source_snapshot.project_revision,
        source_model_name=str(source_snapshot.model_name),
        source_run_id=source_run_id,
        target_document_id=target.document_id,
        target_session_id=after.session_id,
        target_session_revision=after.session_revision,
        target_project_revision=after.project_revision,
        target_model_name=str(after.model_name),
        part_id=part_id,
        named_regions=_named_regions(before, after),
        mesh_settings=_mesh_settings(before, after, part_id),
        local_mesh_controls=_local_controls(before, after, part_id),
        materials=_named_values(before.materials, after.materials, "name"),
        sections=_named_values(before.sections, after.sections, "name"),
        assignments=_named_values(
            before.assignments,
            after.assignments,
            "region_name",
            secondary="section_name",
        ),
        analysis_steps=_named_values(before.steps, after.steps, "name"),
    )


def _named_regions(before: SessionSnapshot, after: SessionSnapshot) -> MigrationItems:
    old = dict(before.named_regions)
    new = dict(after.named_regions)
    shared = set(old) & set(new)
    return MigrationItems(
        preserved=(name for name in shared if old[name] == new[name]),
        rewritten=(name for name in shared if old[name] != new[name]),
        dropped=set(old) - set(new),
    )


def _part(snapshot: SessionSnapshot, part_id: str) -> Any:
    return next((part for part in snapshot.parts if part.id == part_id), None)


def _mesh_settings(
    before: SessionSnapshot,
    after: SessionSnapshot,
    part_id: str,
) -> MigrationItems:
    old = getattr(_part(before, part_id), "mesh_settings", None)
    new = getattr(_part(after, part_id), "mesh_settings", None)
    if old is None:
        return MigrationItems()
    if new is None:
        return MigrationItems(dropped=("global",))
    old_global = _without_local_controls(old)
    new_global = _without_local_controls(new)
    if old_global == new_global:
        return MigrationItems(preserved=("global",))
    return MigrationItems(rewritten=("global",))


def _without_local_controls(settings: Any) -> Any:
    from dataclasses import replace

    return replace(settings, local_controls=())


def _local_controls(
    before: SessionSnapshot,
    after: SessionSnapshot,
    part_id: str,
) -> MigrationItems:
    old_settings = getattr(_part(before, part_id), "mesh_settings", None)
    new_settings = getattr(_part(after, part_id), "mesh_settings", None)
    old = () if old_settings is None else tuple(old_settings.local_controls)
    new = () if new_settings is None else tuple(new_settings.local_controls)
    retained = set(old) & set(new)
    return MigrationItems(
        preserved=(_control_id(control) for control in old if control in retained),
        dropped=(_control_id(control) for control in old if control not in retained),
    )


def _control_id(control: Any) -> str:
    return str(control.target.logical_id)


def _named_values(
    before: Iterable[Any],
    after: Iterable[Any],
    primary: str,
    *,
    secondary: str | None = None,
) -> MigrationItems:
    def identity(value: Any) -> str:
        first = str(getattr(value, primary))
        return first if secondary is None else f"{first}:{getattr(value, secondary)}"

    old = {identity(value): value for value in before}
    new = {identity(value): value for value in after}
    shared = set(old) & set(new)
    return MigrationItems(
        preserved=(name for name in shared if old[name] == new[name]),
        rewritten=(name for name in shared if old[name] != new[name]),
        dropped=set(old) - set(new),
    )


__all__ = [
    "GeometryEditPolicy",
    "GeometryEditUnavailableError",
    "MigrationItems",
    "MigrationReport",
    "ModelIterationResult",
    "ModelIterationService",
    "geometry_edit_policy",
]
