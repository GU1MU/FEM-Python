"""Own committed native previews and replay against their originating document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fem import geometry as geometry_runtime
from fem.application import (
    NativePart, SessionSnapshot, StrictBodyBooleanPreview, StrictPlanarBooleanPreview,
    prepare_strict_body_recipe_preview, prepare_strict_part_recipe_preview,
    prepare_strict_planar_recipe_preview,
)
from fem.geometry import MultiBodyGeometry, geometry_dimension
from fem.geometry.gmsh_coordinator import PROCESS_GMSH_COORDINATOR

from .document_tasks import DocumentTaskRouter
from .geometry_preview import (
    GeometryPreview, build_geometry_preview, build_strict_body_boolean_previews,
    build_strict_part_boolean_preview, build_strict_planar_boolean_preview,
    namespace_part_geometry_preview,
)
from .geometry_recipe_queries import contains_proven_strict_boolean
from .native_geometry_state import (
    NativeGeometryCache, NativeGeometryReplayRequest, NativePartPreview,
    NativePartPreviewKey, NativePreviewState,
)
from .part_geometry_preview import build_multi_part_geometry_preview, localize_part_geometry_preview
from .task_controller import BackgroundTaskState, TaskApplyOutcome, TaskCompletion
from .workers import TaskContext
from .workspace import WorkspaceDocument


@dataclass(frozen=True, slots=True)
class NativeGeometryDisplay:
    preview: GeometryPreview
    ghosts: GeometryPreview | None


@dataclass(frozen=True, slots=True)
class NativeGeometryPresentationPort:
    can_present: Callable[[WorkspaceDocument], bool]
    install: Callable[[WorkspaceDocument, GeometryPreview], None]
    set_status: Callable[[str, int], None]
    defer: Callable[[Callable[[], None]], None]
    shutting_down: Callable[[], bool]


def prepare_persisted_geometry_preview(
    key: NativePartPreviewKey, context: TaskContext,
) -> GeometryPreview:
    """Replay authenticated lineage using the existing CAD coordinator/builders."""

    with PROCESS_GMSH_COORDINATOR.acquire(
        "Rebuild Boolean OCC Preview", cancelled=lambda: context.is_cancelled,
    ):
        context.checkpoint()
        preview = _build_persisted_geometry_preview(key, context)
        context.checkpoint()
        return preview


def _build_persisted_geometry_preview(key: NativePartPreviewKey, context: TaskContext) -> GeometryPreview:
    recipe = key.recipe
    if isinstance(recipe, MultiBodyGeometry):
        previews: list[StrictBodyBooleanPreview] = []
        for body in recipe.bodies:
            if not contains_proven_strict_boolean(body.recipe):
                continue
            context.report(f"Rebuilding {body.name} [{body.id}] OCC preview...")
            with geometry_runtime.model(f"{recipe.name}-{body.id}-persisted-preview", dimension=3) as cad:
                previews.append(prepare_strict_body_recipe_preview(cad, body.id, body.recipe))
            context.checkpoint()
        local = build_strict_body_boolean_previews(recipe, tuple(previews))
        return namespace_part_geometry_preview(key.part_id, local)
    if contains_proven_strict_boolean(recipe, kind="part"):
        context.report("Rebuilding part boolean OCC preview...")
        with geometry_runtime.model(f"{getattr(recipe, 'name', 'part-boolean')}-persisted-preview", dimension=3) as cad:
            payload = prepare_strict_part_recipe_preview(cad, key.part_id, recipe)
        context.checkpoint()
        if type(payload) is not StrictBodyBooleanPreview:
            raise TypeError("persisted Part Boolean preview returned invalid data")
        return build_strict_part_boolean_preview(payload)
    context.report("Rebuilding 2D boolean OCC preview...")
    with geometry_runtime.model(f"{getattr(recipe, 'name', 'planar')}-persisted-preview", dimension=geometry_dimension(recipe)) as cad:
        payload = prepare_strict_planar_recipe_preview(cad, recipe)
    context.checkpoint()
    if type(payload) is not StrictPlanarBooleanPreview:
        raise TypeError("persisted planar Boolean preview returned invalid data")
    return namespace_part_geometry_preview(key.part_id, build_strict_planar_boolean_preview(recipe, payload))


class NativeGeometryPresentation:
    """Retain one current preview per Part, independently of the visible scene."""

    def __init__(self, tasks: DocumentTaskRouter, presentation: NativeGeometryPresentationPort) -> None:
        self.tasks = tasks
        self.presentation = presentation

    @staticmethod
    def _key(snapshot: SessionSnapshot, part: NativePart) -> NativePartPreviewKey:
        return NativePartPreviewKey(snapshot.session_id, part.id, snapshot.part_revision(part.id), part.geometry_recipe)

    def synchronize(self, document: WorkspaceDocument, snapshot: SessionSnapshot) -> NativeGeometryCache:
        cache = document.presentation_cache.native_geometry
        if cache.session is not document.session or snapshot.source_kind != "native":
            cache.clear()
            cache.session = document.session
        current = {part.id: self._key(snapshot, part) for part in snapshot.parts}
        for part_id, entry in tuple(cache.parts.items()):
            if current.get(part_id) != entry.key:
                del cache.parts[part_id]
        return cache

    def seed_exact(self, document: WorkspaceDocument, snapshot: SessionSnapshot, preview: GeometryPreview) -> None:
        if type(preview) is not GeometryPreview:
            raise TypeError("geometry_preview must be a GeometryPreview")
        active = snapshot.active_part
        if snapshot.source_kind != "native" or snapshot.artifact is not None or active is None:
            return
        cache = self.synchronize(document, snapshot)
        cache.parts[active.id] = NativePartPreview(self._key(snapshot, active), preview, NativePreviewState.EXACT)

    def preview(
        self, document: WorkspaceDocument, *, show_ghosts: bool = False,
        retry_failed: bool = False,
    ) -> NativeGeometryDisplay | None:
        snapshot = document.projection
        cache = self.synchronize(document, snapshot)
        active = snapshot.active_part
        if snapshot.source_kind != "native" or not snapshot.parts:
            return None
        ghosts = self._ghosts(cache, snapshot, show_ghosts)
        if active is None or active.suppressed or active.geometry_recipe is None:
            return NativeGeometryDisplay(GeometryPreview((), (), (), topological_dimension=2), ghosts)
        entry = cache.parts.get(active.id)
        if entry is None:
            strict = contains_proven_strict_boolean(active.geometry_recipe)
            entry = NativePartPreview(
                self._key(snapshot, active), build_multi_part_geometry_preview((active,)),
                NativePreviewState.COARSE if strict else NativePreviewState.EXACT,
            )
            cache.parts[active.id] = entry
        if entry.state is NativePreviewState.COARSE or (
            retry_failed and entry.state in {NativePreviewState.FAILED, NativePreviewState.CANCELLED}
        ):
            self._request_replay(document, entry)
        return NativeGeometryDisplay(entry.preview, ghosts)

    @staticmethod
    def _ghosts(cache: NativeGeometryCache, snapshot: SessionSnapshot, show_ghosts: bool) -> GeometryPreview | None:
        suppressed = tuple(part for part in snapshot.parts if part.suppressed and part.geometry_recipe is not None)
        ghost_key = tuple(NativeGeometryPresentation._key(snapshot, part) for part in suppressed) if show_ghosts else ()
        if cache.ghost_key != ghost_key:
            cache.ghost_key = ghost_key
            cache.ghost_preview = (
                build_multi_part_geometry_preview(suppressed, include_suppressed=True)
                if show_ghosts and suppressed else None
            )
        return cache.ghost_preview

    def source_preview(self, document: WorkspaceDocument, part_id: str, recipe: object) -> GeometryPreview:
        snapshot = document.session.projection_snapshot()
        cache = self.synchronize(document, snapshot)
        entry = cache.parts.get(part_id)
        if entry is not None and entry.key.recipe == recipe:
            return localize_part_geometry_preview(part_id, entry.preview)
        return build_geometry_preview(recipe)

    def cancel_pending(self, document: WorkspaceDocument, *, closing: bool = False) -> None:
        cache = document.presentation_cache.native_geometry
        cache.closing = cache.closing or closing
        for entry in cache.parts.values():
            if entry.request is not None:
                entry.request = None
                entry.state = NativePreviewState.CANCELLED

    def dispose_document(self, document: WorkspaceDocument) -> None:
        document.presentation_cache.native_geometry.clear()

    def _current_entry(self, request: NativeGeometryReplayRequest) -> NativePartPreview | None:
        document = request.document
        if (
            self.presentation.shutting_down() or not self.tasks.is_live(document)
            or document.session is not request.session
        ):
            return None
        snapshot = request.session.projection_snapshot()
        if not snapshot.is_open:
            self.dispose_document(document)
            return None
        cache = self.synchronize(document, snapshot)
        entry = cache.parts.get(request.key.part_id)
        if cache.closing or entry is None or entry.key != request.key or entry.request is not request:
            return None
        return entry

    def _request_replay(self, document: WorkspaceDocument, entry: NativePartPreview) -> None:
        if (
            self.presentation.shutting_down() or not self.tasks.is_live(document)
            or document.presentation_cache.native_geometry.closing or entry.request is not None
        ):
            return
        request = NativeGeometryReplayRequest(document, document.session, document.task_controller, entry.key)
        entry.request = request
        entry.state = NativePreviewState.PENDING
        self._schedule(request)

    def _schedule(self, request: NativeGeometryReplayRequest) -> None:
        if request.scheduled:
            return
        request.scheduled = True

        def launch() -> None:
            request.scheduled = False
            entry = self._current_entry(request)
            if entry is None:
                return
            if request.controller.busy:
                self._schedule(request)
                return
            self._launch(request)

        self.tasks.after_idle(request.controller, lambda: self.presentation.defer(launch))

    def _launch(self, request: NativeGeometryReplayRequest) -> None:
        def apply(payload: object) -> TaskApplyOutcome:
            entry = self._current_entry(request)
            if entry is None:
                return TaskApplyOutcome.stale("Native geometry preview belongs to a changed or closed Part")
            if type(payload) is not GeometryPreview:
                raise TypeError("persisted Boolean preview task returned invalid data")
            # Acceptance is document-owned; renderer failures cannot lose this value.
            entry.preview = payload
            entry.state = NativePreviewState.EXACT
            entry.request = None
            return TaskApplyOutcome.accepted(payload)

        def present(payload: object) -> None:
            document = request.document
            if document.projection.active_part_id == request.key.part_id and self.presentation.can_present(document):
                self.presentation.install(document, payload)
                self.presentation.set_status("Boolean preview rebuilt", 4000)

        def terminal(record: TaskCompletion) -> None:
            entry = self._current_entry(request)
            if entry is None:
                return
            entry.request = None
            entry.state = (
                NativePreviewState.CANCELLED if record.state is BackgroundTaskState.CANCELLED
                else NativePreviewState.FAILED
            )
            if record.state is BackgroundTaskState.FAILED and self.presentation.can_present(request.document):
                self.presentation.set_status(f"Failed to rebuild boolean OCC preview: {record.message}", 6000)

        try:
            started = self.tasks.start(
                lambda context: prepare_persisted_geometry_preview(request.key, context), present,
                "Boolean OCC Preview", context=request.document, controller=request.controller,
                task_name="Rebuild Boolean OCC Preview", apply_result=apply,
                on_failure=lambda _message: None, on_inactive_failure=lambda _message: None,
                on_terminal=terminal,
            )
        except Exception as error:
            started = False
            if self.presentation.can_present(request.document):
                self.presentation.set_status(f"Failed to start boolean OCC preview: {error}", 6000)
        if not started:
            entry = self._current_entry(request)
            if entry is not None:
                entry.request = None
                entry.state = NativePreviewState.FAILED
