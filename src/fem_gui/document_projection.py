"""Project accepted Session state into document-owned snapshots and caches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Literal

from fem.application import (
    BeamFrameReport, ModelSession, RegionRef, SessionDelta, SessionSnapshot,
    resolve_effective_beam_frames,
)
from fem.application.results import ResultProvider, ResultSourceKey

from .geometry_preview import GeometryPreview
from .inspection_service import InspectionService
from .native_geometry_presentation import NativeGeometryPresentation
from .session_projection import SessionProjectionPlan, plan_session_projection
from .visualization.model_adapter import ArchiveModelView, ModelGeometry, build_model_geometry
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


@dataclass(frozen=True, slots=True)
class PreparedModelProjection:
    """One model binding prepared without changing Qt or viewport consumers."""

    model: object
    geometry: ModelGeometry
    inspection_service: InspectionService
    effective_frame_query: Callable[[RegionRef | int], BeamFrameReport] | None
    timings: dict[str, float]


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
                cache = document.presentation_cache
                if identity is None or cache.result_source != identity[0]:
                    cache.invalidate_result()
                elif not cache.matches_result_identity(*identity):
                    # Field generations change rendering, not the archive's
                    # source-owned model facade and mesh topology.
                    cache.result_generation = identity[1]
                    cache.result_rendering.clear()

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
    def prepare_model(
        update: DocumentProjectionUpdate,
        action: Literal["rebind", "install"],
        installed_geometry: ModelGeometry | None,
    ) -> PreparedModelProjection:
        """Resolve only owned adapters and prepare shared model consumers."""

        snapshot = update.snapshot
        artifact = snapshot.artifact
        if artifact is None:
            raise RuntimeError("model projection requires a model artifact")
        cache = update.document.presentation_cache
        geometry = (
            installed_geometry if action == "rebind"
            else update.model_geometry or (
                cache.model_geometry if cache.matches_artifact(artifact.artifact_id) else None
            )
        )
        if snapshot.source_kind == "result":
            identity = update.session.current_result_identity()
            # The archive facade depends on source topology; loading another
            # field generation does not replace that model adapter.
            model = update.result_model_view or (
                cache.result_model_view
                if identity is not None and cache.result_source == identity[0]
                else None
            )
            if geometry is None or model is None:
                raise RuntimeError("result-only projection requires worker display payload")
            frame_query = None
        else:
            model = artifact.model
            geometry = geometry or build_model_geometry(model)

            def frame_query(target: RegionRef | int) -> BeamFrameReport:
                return resolve_effective_beam_frames(model, target)

        if geometry.artifact_id != artifact.artifact_id:
            geometry = replace(geometry, artifact_id=artifact.artifact_id)
        timings = dict(update.timings or {})
        inspection = (
            cache.inspection_service
            if action == "install" and cache.matches_artifact(artifact.artifact_id)
            else None
        )
        if inspection is None:
            started = perf_counter()
            inspection = InspectionService(
                model, definitions=snapshot, effective_frame_query=frame_query,
            )
            timings["InspectionService initialization"] = perf_counter() - started
        return PreparedModelProjection(model, geometry, inspection, frame_query, timings)

    @staticmethod
    def result_provider(
        update: DocumentProjectionUpdate, projected_provider: ResultProvider | None,
    ) -> ResultProvider | None:
        """Resolve the exact accepted provider after model consumers changed."""

        identity = update.session.current_result_identity()
        if identity is None:
            return None
        if (
            type(projected_provider) is ResultProvider
            and (projected_provider.source, projected_provider.snapshot.generation) == identity
        ):
            return projected_provider
        provider = update.session.current_result_provider()
        if provider is None:
            raise RuntimeError("current result identity has no provider")
        return provider

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
        cache.result_model_view = model_view if update.snapshot.source_kind == "result" else None
        cache.inspection_service = inspection_service
