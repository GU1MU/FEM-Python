"""Detached launch context for sketching on one resolved solid Face."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fem.application import NativePart, SessionSnapshot
from fem.geometry import (
    FaceSketchBooleanGeometry,
    MultiBodyGeometry,
    ResolvedFaceWorkplane,
    SketchExternalCoincidence,
    SketchExternalReference,
    SketchGeometry,
    SketchReferencePoint,
)

from .sketch_editor import SketchDraftController, SketchDraftSnapshot


@dataclass(frozen=True, slots=True)
class FaceSketchBodySnapshot:
    """The target Body identity and recipe captured when editing starts."""

    logical_id: str
    name: str
    recipe: object


@dataclass(frozen=True, slots=True)
class FaceSketchLaunchSnapshot:
    """Immutable compare-and-swap inputs for a face-supported sketch draft."""

    session_id: str
    session_revision: int
    part: NativePart
    part_revision: int
    body: FaceSketchBodySnapshot
    workplane: ResolvedFaceWorkplane
    sketch: SketchDraftSnapshot

    @property
    def part_id(self) -> str:
        return self.part.id

    @property
    def body_id(self) -> str:
        return self.body.logical_id

    @property
    def sketch_revision(self) -> int:
        return self.sketch.revision


@dataclass(frozen=True, slots=True)
class FaceSketchBooleanFeatureRequest:
    """Detached hand-off for the Phase-6 atomic Session commit."""

    launch: FaceSketchLaunchSnapshot
    geometry: FaceSketchBooleanGeometry
    sketch_revision: int
    preview_generation: int


class FaceSupportedSketchController:
    """Own a detached draft plus all authoritative state observed at launch."""

    def __init__(
        self,
        session: SessionSnapshot,
        part_id: str,
        workplane: ResolvedFaceWorkplane,
        *,
        root: SketchGeometry | None = None,
        name: str = "面草图-1",
        reference_points: tuple[SketchReferencePoint, ...] = (),
        external_references: tuple[SketchExternalReference, ...] = (),
        external_coincidences: tuple[SketchExternalCoincidence, ...] = (),
        unresolved_reference_ids: tuple[str, ...] = (),
    ) -> None:
        if type(session) is not SessionSnapshot:
            raise TypeError("session must be a SessionSnapshot")
        if type(workplane) is not ResolvedFaceWorkplane:
            raise TypeError("workplane must be a ResolvedFaceWorkplane")
        if session.source_kind != "native":
            raise ValueError("面支撑草图只能从自主模型启动")
        part = session.part(part_id)
        if part.suppressed or part.geometry_recipe is None:
            raise ValueError("目标 Part 不可编辑")
        body = _body_snapshot(part, workplane.target_body_id)
        draft = (
            SketchDraftController(root=root)
            if root is not None
            else SketchDraftController(name=name, plane=workplane.plane)
        )
        if external_references or external_coincidences or unresolved_reference_ids:
            draft = SketchDraftController(
                snapshot=replace(
                    draft.snapshot(),
                    external_references=tuple(external_references),
                    external_coincidences=tuple(external_coincidences),
                    unresolved_reference_ids=tuple(unresolved_reference_ids),
                )
            )
        draft.refresh_external_references(tuple(reference_points))
        if draft.plane != workplane.plane:
            raise ValueError("草图工作平面与所选支撑面不一致")
        sketch_snapshot = draft.snapshot()
        self._draft = draft
        self._launch = FaceSketchLaunchSnapshot(
            session.session_id,
            session.session_revision,
            part,
            session.part_revision(part.id),
            body,
            workplane,
            sketch_snapshot,
        )

    @property
    def draft(self) -> SketchDraftController:
        return self._draft

    @property
    def launch_snapshot(self) -> FaceSketchLaunchSnapshot:
        return self._launch

    @property
    def workplane(self) -> ResolvedFaceWorkplane:
        return self._launch.workplane

    def sketch_snapshot(self) -> SketchDraftSnapshot:
        """Return the current detached sketch revision without touching Session."""

        return self._draft.snapshot()

    def refresh_reference_points(
        self,
        reference_points: tuple[SketchReferencePoint, ...],
    ) -> SketchDraftSnapshot:
        """Rebuild associations by exact source identity for the current Face."""

        self._draft.refresh_external_references(tuple(reference_points))
        return self._draft.snapshot()

    def launch_is_current(self, session: SessionSnapshot) -> bool:
        """Check the launch Session and Part revisions for later atomic commit."""

        if type(session) is not SessionSnapshot:
            raise TypeError("session must be a SessionSnapshot")
        launch = self._launch
        if (
            session.session_id != launch.session_id
            or session.session_revision != launch.session_revision
        ):
            return False
        try:
            return session.part_revision(launch.part_id) == launch.part_revision
        except KeyError:
            return False


def _body_snapshot(
    part: NativePart,
    target_body_id: str,
) -> FaceSketchBodySnapshot:
    recipe = part.geometry_recipe
    if isinstance(recipe, MultiBodyGeometry):
        if not target_body_id.startswith("body:"):
            raise ValueError("工作面目标 Body ID 无效")
        body_id = target_body_id.removeprefix("body:")
        try:
            body = recipe.body(body_id)
        except (KeyError, ValueError) as error:
            raise ValueError("工作面目标 Body 已失效") from error
        return FaceSketchBodySnapshot(
            target_body_id,
            body.name,
            body.recipe,
        )
    if target_body_id != "body:domain":
        raise ValueError("工作面目标 Body 与 Part 几何不一致")
    return FaceSketchBodySnapshot(
        target_body_id,
        part.body_name,
        recipe,
    )


__all__ = [
    "FaceSketchBooleanFeatureRequest",
    "FaceSketchBodySnapshot",
    "FaceSketchLaunchSnapshot",
    "FaceSupportedSketchController",
]
