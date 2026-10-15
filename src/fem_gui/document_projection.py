"""Project accepted Session state into document-owned snapshots and caches."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fem.application import ModelSession, SessionDelta, SessionSnapshot
from fem.application.results import ResultSourceKey

from .geometry_preview import GeometryPreview
from .inspection_service import InspectionService
from .native_geometry_presentation import NativeGeometryPresentation
from .session_projection import SessionProjectionPlan, plan_session_projection
from .visualization.model_adapter import ArchiveModelView, ModelGeometry
from .workspace import FEMWorkspace, WorkspaceDocument


@dataclass(frozen=True, slots=True)
class DocumentProjectionUpdate:
    """One document update and the display payloads belonging to its revision."""

    document: WorkspaceDocument
    session: ModelSession
    previous: SessionSnapshot
    snapshot: SessionSnapshot
    plan: SessionProjectionPlan
    model_geometry: ModelGeometry | None
    result_model_view: ArchiveModelView | None
    timings: dict[str, float] | None
    source_label: str | None


class DocumentSessionProjection:
    """Own document projection decisions independently of visible scene progress.

    The workspace retains snapshots and shallow caches. A visible revision is
    supplied by the window; absent that value, only the document's progress is
    relevant. Accepted document state survives a later renderer failure.
    """

    def __init__(
        self, workspace: FEMWorkspace, native_geometry: NativeGeometryPresentation,
    ) -> None:
        self.workspace = workspace
        self.native_geometry = native_geometry

    def project(
        self,
        document: WorkspaceDocument,
        delta: object,
        *,
        visible_revision: int | None = None,
        model_geometry: ModelGeometry | None = None,
        result_model_view: ArchiveModelView | None = None,
        geometry_preview: GeometryPreview | None = None,
        timings: dict[str, float] | None = None,
        source_label: str | None = None,
        preserve_mesh_topology: bool = False,
    ) -> DocumentProjectionUpdate | None:
        """Keep the latest shallow snapshot, rejecting stale display payloads."""

        if not bool(getattr(delta, "accepted", True)):
            return None
        requested_revision = int(getattr(delta, "session_revision"))
        previous = document.projection
        visible = visible_revision is not None
        applied_revision = previous.session_revision if not visible else visible_revision
        if requested_revision <= applied_revision:
            return None
        session = document.session
        snapshot = (
            session.projection_snapshot(previous, delta.changed)
            if isinstance(delta, SessionDelta) and session.session_revision == requested_revision
            else session.projection_snapshot()
        )
        plan = plan_session_projection(
            previous, snapshot,
            requested_revision=requested_revision,
            applied_revision=applied_revision,
            delta=delta if isinstance(delta, SessionDelta) else None,
            preserve_mesh_topology=visible and preserve_mesh_topology,
        )
        if plan is None:
            return None
        if visible and geometry_preview is not None and type(geometry_preview) is not GeometryPreview:
            raise TypeError("geometry_preview must be a GeometryPreview")
        if snapshot.session_revision != requested_revision:
            model_geometry = None
            result_model_view = None
            geometry_preview = None
            timings = None
            source_label = None

        if visible:
            if geometry_preview is not None:
                self.native_geometry.seed_exact(document, snapshot, geometry_preview)
            model_geometry = self._retain_model_geometry(document, snapshot, model_geometry)
        self.workspace.update_projection(document, snapshot)
        self.native_geometry.synchronize(document, snapshot)
        if not visible:
            if geometry_preview is not None:
                self.native_geometry.seed_exact(document, snapshot, geometry_preview)
            if plan.scope != "metadata":
                model_geometry = self._retain_model_geometry(document, snapshot, model_geometry)
                identity = session.current_result_identity()
                if identity is None or not document.presentation_cache.matches_result(*identity):
                    document.presentation_cache.invalidate_result()

        return DocumentProjectionUpdate(
            document, session, previous, snapshot, plan,
            model_geometry, result_model_view, timings, source_label,
        )

    @staticmethod
    def _retain_model_geometry(
        document: WorkspaceDocument,
        snapshot: SessionSnapshot,
        geometry: ModelGeometry | None,
    ) -> ModelGeometry | None:
        artifact = snapshot.artifact
        artifact_id = None if artifact is None else artifact.artifact_id
        cache = document.presentation_cache
        if cache.artifact_id != artifact_id:
            cache.invalidate_model()
        if artifact is not None and geometry is not None:
            if geometry.artifact_id != artifact_id:
                geometry = replace(geometry, artifact_id=artifact_id)
            cache.artifact_id = artifact_id
            cache.model_geometry = geometry
        return geometry

    @staticmethod
    def record_model_display(
        update: DocumentProjectionUpdate,
        geometry: ModelGeometry,
        inspection_service: InspectionService | None,
    ) -> None:
        """Retain successfully installed model consumers in the owning cache."""

        cache = update.document.presentation_cache
        cache.artifact_id = update.snapshot.artifact.artifact_id
        cache.model_geometry = geometry
        cache.inspection_service = inspection_service

    @staticmethod
    def record_result_display(
        update: DocumentProjectionUpdate,
        identity: tuple[ResultSourceKey, int] | None,
        model_view: ArchiveModelView | None = None,
        inspection_service: InspectionService | None = None,
    ) -> None:
        """Record completed result projection without owning rendering policy."""

        cache = update.document.presentation_cache
        if identity is None:
            cache.invalidate_result()
            return
        cache.result_source, cache.result_generation = identity
        cache.result_model_view = model_view
        cache.inspection_service = inspection_service
