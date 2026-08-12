"""Minimal document ownership for the FEM GUI workspace.

The first workspace phase deliberately keeps the visible application single
document.  ``FEMWorkspace`` owns document identity and the active context so
later phases can add incremental trees without changing the Session API.
The objects in this module are intentionally shallow: Session projections and
all model/result payloads remain owned by ``ModelSession`` and are never
deep-copied here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fem.application import ModelSession, SessionSnapshot

from .task_controller import BackgroundTaskController


def canonical_path(path: str | os.PathLike[str] | Path) -> str:
    """Return the one key used by model/result path indexes.

    ``resolve(strict=False)`` intentionally does not touch the filesystem so
    callers can index a path before a save or while a detached load is in
    flight.  ``normcase`` handles case-insensitive platforms without adding a
    second index.
    """

    resolved = Path(path).expanduser().resolve(strict=False)
    return os.path.normcase(str(resolved))


@dataclass(slots=True)
class DocumentPresentationState:
    """Small, GUI-only state retained per document.

    Phase 1 does not install this state into the viewport yet.  Keeping the
    fields here gives later phases a stable owner without putting presentation
    data on ``ModelSession``.
    """

    module_name: str | None = None
    step_name: str | None = None
    result_selection: object | None = None
    display_state: object | None = None
    camera_state: object | None = None
    selection_mode: str | None = None
    result_scale_mode: str = "auto"
    result_scale_value: float = 1.0
    contour_options: dict[str, object] = field(default_factory=dict)
    overlay_undeformed: bool = False


@dataclass(slots=True)
class DocumentPresentationCache:
    """Identity-keyed display cache owned by one document.

    Values are references to immutable or Session-owned adapters.  The cache
    intentionally does not clone any value when it is installed or replaced.
    """

    artifact_id: str | None = None
    result_source: object | None = None
    result_generation: int | None = None
    model_geometry: object | None = None
    result_model_view: object | None = None
    inspection_service: object | None = None

    def matches_artifact(self, artifact_id: object | None) -> bool:
        """Return whether the cached model adapters belong to ``artifact_id``."""

        return (
            artifact_id is not None
            and self.artifact_id is not None
            and str(self.artifact_id) == str(artifact_id)
            and self.model_geometry is not None
        )

    def matches_result(
        self,
        source: object | None,
        generation: int | None,
    ) -> bool:
        """Return whether the cached result adapters match one identity."""

        return (
            source is not None
            and self.result_source == source
            and self.result_generation is not None
            and generation is not None
            and int(self.result_generation) == int(generation)
            and self.result_model_view is not None
        )

    def invalidate_model(self) -> None:
        """Drop only model-derived adapters, preserving result identity fields."""

        self.artifact_id = None
        self.model_geometry = None
        self.inspection_service = None

    def invalidate_result(self) -> None:
        """Drop only result-derived adapters."""

        self.result_source = None
        self.result_generation = None
        self.result_model_view = None


@dataclass(slots=True)
class WorkspaceDocument:
    """The one mutable ownership record for a model or result document."""

    document_id: int
    session: ModelSession
    projection: SessionSnapshot
    display_name: str
    source_path: Path | None
    task_controller: BackgroundTaskController
    kind: str = "model"
    presentation_state: DocumentPresentationState = field(
        default_factory=DocumentPresentationState
    )
    presentation_cache: DocumentPresentationCache = field(
        default_factory=DocumentPresentationCache
    )

    @property
    def revision(self) -> int:
        """Current projection revision (a cheap identity check for callers)."""

        return int(self.projection.session_revision)

    @property
    def id(self) -> int:
        """Short alias useful when writing Qt item roles."""

        return self.document_id

    @property
    def path(self) -> Path | None:
        return self.source_path

    @property
    def is_result(self) -> bool:
        return (
            self.kind == "result"
            or getattr(
                self.projection,
                "source_kind",
                None,
            )
            == "result"
        )


ControllerFactory = Callable[[object | None], BackgroundTaskController]


class FEMWorkspace:
    """Own document contexts while preserving the single-document GUI path."""

    def __init__(
        self,
        parent: object | None = None,
        *,
        controller_factory: ControllerFactory | None = None,
        open_controller: BackgroundTaskController | None = None,
    ) -> None:
        self._parent = parent
        self._controller_factory = controller_factory
        self.models: dict[int, WorkspaceDocument] = {}
        self.results: dict[int, WorkspaceDocument] = {}
        self.model_paths: dict[str, int] = {}
        self.result_paths: dict[str, int] = {}
        self.active_kind: str | None = None
        self.active_document_id: int | None = None
        self._next_document_id = 1
        self._next_model_number = 1
        self._next_result_number = 1
        self._next_job_number = 1
        self._job_names: set[str] = set()
        # Detached file decoding and its controller belong to a later phase;
        # retaining an injection point avoids creating a second controller in
        # the Phase 1 single-document startup path.
        self.open_controller = open_controller

    @property
    def next_document_id(self) -> int:
        """The next never-used document identity (without reserving it)."""

        return self._next_document_id

    @property
    def next_model_number(self) -> int:
        """The next default display number for a newly created model."""

        return self._next_model_number

    def next_job_name(self) -> str:
        """Return the next workspace-global default job name."""

        while f"作业-{self._next_job_number}".casefold() in self._job_names:
            self._next_job_number += 1
        return f"作业-{self._next_job_number}"

    def job_name_exists(self, name: str) -> bool:
        """Return whether a job display name was already used in this workspace."""

        normalized = str(name).strip().casefold()
        if not normalized:
            return False
        return normalized in self._job_names

    def remember_job_name(self, name: str) -> None:
        """Reserve one job name for the lifetime of the workspace."""

        clean = str(name).strip()
        if not clean:
            raise ValueError("job name must not be empty")
        self._job_names.add(clean.casefold())
        prefix = "作业-"
        if clean.startswith(prefix) and clean[len(prefix) :].isdigit():
            self._next_job_number = max(
                self._next_job_number,
                int(clean[len(prefix) :]) + 1,
            )

    @property
    def document_count(self) -> int:
        return len(self.models) + len(self.results)

    def documents(self) -> tuple[WorkspaceDocument, ...]:
        """Return all live contexts without copying Session payloads."""

        return tuple((*self.models.values(), *self.results.values()))

    def busy_documents(self) -> tuple[WorkspaceDocument, ...]:
        """Return contexts with an active cooperative background task."""

        return tuple(
            context for context in self.documents() if context.task_controller.busy
        )

    def any_busy(self) -> bool:
        """Return whether any model/result context owns a running task."""

        return any(context.task_controller.busy for context in self.documents())

    def add_model(
        self,
        session: ModelSession | None = None,
        projection: SessionSnapshot | None = None,
        *,
        display_name: str | None = None,
        source_path: str | os.PathLike[str] | Path | None = None,
        task_controller: BackgroundTaskController | None = None,
    ) -> WorkspaceDocument:
        """Add or return a model context without copying Session data."""

        return self._add(
            "model",
            session=session,
            projection=projection,
            display_name=display_name,
            source_path=source_path,
            task_controller=task_controller,
        )

    def add_result(
        self,
        session: ModelSession | None = None,
        projection: SessionSnapshot | None = None,
        *,
        display_name: str | None = None,
        source_path: str | os.PathLike[str] | Path | None = None,
        task_controller: BackgroundTaskController | None = None,
    ) -> WorkspaceDocument:
        """Add or return an external result context."""

        return self._add(
            "result",
            session=session,
            projection=projection,
            display_name=display_name,
            source_path=source_path,
            task_controller=task_controller,
        )

    def ensure_open_controller(self) -> BackgroundTaskController:
        """Return the one workspace-owned detached file-open controller."""

        if self.open_controller is None:
            self.open_controller = self._new_controller()
        return self.open_controller

    def activate(self, document_id: int | WorkspaceDocument) -> WorkspaceDocument:
        """Make a document active using an O(1) integer lookup."""

        context = (
            document_id
            if isinstance(document_id, WorkspaceDocument)
            else self.document(document_id)
        )
        self.active_kind = self._kind_for(context.document_id)
        self.active_document_id = context.document_id
        return context

    def remove(
        self,
        document_id: int | WorkspaceDocument,
    ) -> WorkspaceDocument:
        """Remove one context and its path index entry.

        Removing the active context selects the first remaining document in
        insertion order.  With no documents left the active state is empty.
        IDs are never reused.
        """

        context = (
            document_id
            if isinstance(document_id, WorkspaceDocument)
            else self.document(document_id)
        )
        kind = self._kind_for(context.document_id)
        registry = self.models if kind == "model" else self.results
        registry.pop(context.document_id)
        path_index = self.model_paths if kind == "model" else self.result_paths
        if context.source_path is not None:
            key = canonical_path(context.source_path)
            if path_index.get(key) == context.document_id:
                path_index.pop(key, None)

        if self.active_document_id == context.document_id:
            replacement = next(iter(self.models.values()), None)
            if replacement is None:
                replacement = next(iter(self.results.values()), None)
            if replacement is None:
                self.active_kind = None
                self.active_document_id = None
            else:
                self.activate(replacement)
        return context

    def active_document(self) -> WorkspaceDocument | None:
        """Return the active context, or ``None`` for an empty workspace."""

        if self.active_document_id is None:
            return None
        try:
            return self.document(self.active_document_id)
        except KeyError:
            # Keep the public state self-healing if a caller edited a registry
            # directly while inspecting a test fixture.
            self.active_kind = None
            self.active_document_id = None
            return None

    def document(self, document_id: int) -> WorkspaceDocument:
        """Look up a context by its process-local integer identity."""

        if isinstance(document_id, bool):
            raise TypeError("document_id must be an integer")
        try:
            normalized = int(document_id)
        except (TypeError, ValueError) as error:
            raise TypeError("document_id must be an integer") from error
        context = self.models.get(normalized)
        if context is None:
            context = self.results.get(normalized)
        if context is None:
            raise KeyError(normalized)
        return context

    def update_projection(
        self,
        context: int | WorkspaceDocument,
        projection: SessionSnapshot,
    ) -> WorkspaceDocument:
        """Install a trusted projection by identity, without a deepcopy."""

        target = (
            context
            if isinstance(context, WorkspaceDocument)
            else self.document(context)
        )
        path_index = self.model_paths if target.kind == "model" else self.result_paths
        old_path = target.source_path
        old_model_name = str(getattr(target.projection, "model_name", "") or "").strip()
        if old_path is not None:
            old_key = canonical_path(old_path)
            if path_index.get(old_key) == target.document_id:
                path_index.pop(old_key, None)

        target.projection = projection
        new_model_name = str(getattr(projection, "model_name", "") or "").strip()
        if (
            target.kind == "model"
            and new_model_name
            and new_model_name != old_model_name
        ):
            target.display_name = self._unique_display_name(
                "model",
                new_model_name,
                exclude_document_id=target.document_id,
            )
        self._remember_projection_job_names(projection)
        path = (
            getattr(projection, "path", None)
            or getattr(projection, "source_path", None)
            or getattr(projection, "project_path", None)
        )
        target.source_path = None if path is None else Path(path)
        if target.source_path is not None:
            path_index[canonical_path(target.source_path)] = target.document_id
        return target

    def _add(
        self,
        kind: str,
        *,
        session: ModelSession | None,
        projection: SessionSnapshot | None,
        display_name: str | None,
        source_path: str | os.PathLike[str] | Path | None,
        task_controller: BackgroundTaskController | None,
    ) -> WorkspaceDocument:
        if kind not in {"model", "result"}:
            raise ValueError("workspace document kind must be model or result")
        if session is None:
            session = ModelSession()
        if projection is None:
            projection = session.projection_snapshot()

        raw_path = source_path
        if raw_path is None:
            raw_path = getattr(projection, "path", None)
        path = None if raw_path is None else Path(raw_path)
        path_index = self.model_paths if kind == "model" else self.result_paths
        registry = self.models if kind == "model" else self.results
        if path is not None:
            key = canonical_path(path)
            existing_id = path_index.get(key)
            if existing_id is not None:
                existing = registry[existing_id]
                self.activate(existing)
                return existing

        document_id = self._next_document_id
        self._next_document_id += 1
        name = str(display_name or "").strip()
        if not name:
            name = str(getattr(projection, "model_name", "") or "").strip()
        if not name and path is not None:
            name = path.stem
        if not name:
            if kind == "result":
                name = f"Result-{self._next_result_number}"
                self._next_result_number += 1
            else:
                name = f"模型-{self._next_model_number}"
                self._next_model_number += 1
        elif (
            kind == "model"
            and path is None
            and name == f"模型-{self._next_model_number}"
        ):
            self._next_model_number += 1
        name = self._unique_display_name(kind, name)
        controller = task_controller or self._new_controller()
        context = WorkspaceDocument(
            document_id=document_id,
            session=session,
            projection=projection,
            display_name=name,
            source_path=path,
            task_controller=controller,
            kind=kind,
        )
        registry[document_id] = context
        self._remember_projection_job_names(projection)
        if path is not None:
            path_index[canonical_path(path)] = document_id
        if self.active_document_id is None:
            self.activate(context)
        return context

    def _unique_display_name(
        self,
        kind: str,
        preferred: str,
        *,
        exclude_document_id: int | None = None,
    ) -> str:
        registry = self.models if kind == "model" else self.results
        existing = {
            context.display_name.casefold()
            for context in registry.values()
            if context.document_id != exclude_document_id
        }
        if preferred.casefold() not in existing:
            return preferred
        number = 1
        while f"{preferred}({number})".casefold() in existing:
            number += 1
        return f"{preferred}({number})"

    def _remember_projection_job_names(self, projection: SessionSnapshot) -> None:
        for run in projection.runs:
            name = str(getattr(run, "name", "")).strip()
            if name:
                self.remember_job_name(name)

    def _kind_for(self, document_id: int) -> str:
        if document_id in self.models:
            return "model"
        if document_id in self.results:
            return "result"
        raise KeyError(document_id)

    def _new_controller(self) -> BackgroundTaskController:
        factory = self._controller_factory
        if factory is None:
            return BackgroundTaskController(self._parent)
        return factory(self._parent)


__all__ = [
    "DocumentPresentationCache",
    "DocumentPresentationState",
    "FEMWorkspace",
    "WorkspaceDocument",
    "canonical_path",
]
