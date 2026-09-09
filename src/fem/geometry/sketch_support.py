"""Stable planar-face workplane preparation for associated sketches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal, Mapping

from .recipes import (
    FaceSketchBooleanDirection,
    FaceSketchWorkplaneStrategy,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchPlane,
)
from .references import LogicalEntityRef


_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_FRAME_TOLERANCE = 1.0e-9


@dataclass(frozen=True, slots=True)
class SketchReferencePoint:
    """One resolved external point in both global and sketch coordinates."""

    reference: SketchExternalReference
    position: tuple[float, float, float]
    u: float
    v: float

    def __post_init__(self) -> None:
        if type(self.reference) is not SketchExternalReference:
            raise TypeError("reference must be a SketchExternalReference")
        position = tuple(float(value) for value in self.position)
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("Invalid reference point 3D coordinates")
        if not math.isfinite(float(self.u)) or not math.isfinite(float(self.v)):
            raise ValueError("Invalid reference point U/V coordinates")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "u", float(self.u))
        object.__setattr__(self, "v", float(self.v))

    @property
    def derived_type(self) -> SketchExternalReferenceType:
        return self.reference.derived_type


SketchSnapKind = Literal[
    "sketch_point",
    "intersection",
    "topology_vertex",
    "circle_center",
    "arc_center",
    "line_midpoint",
    "face_center",
    "grid",
]


@dataclass(frozen=True, slots=True)
class SketchSnapCandidate:
    """A screen-space snap choice used by both planar sketch workflows."""

    kind: SketchSnapKind
    screen_x: float
    screen_y: float
    u: float
    v: float
    reference_point: SketchReferencePoint | None = None
    sketch_point_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SNAP_PRIORITIES:
            raise ValueError("Invalid sketch snap candidate type")
        values = (self.screen_x, self.screen_y, self.u, self.v)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Invalid sketch snap candidate coordinates")
        if self.reference_point is not None:
            expected = self.reference_point.derived_type.value
            if self.kind != expected:
                raise ValueError("The external reference and snap candidate types do not match")
        if self.kind == "sketch_point" and not self.sketch_point_id:
            raise ValueError("The sketch point snap candidate is missing a point ID")


_SNAP_PRIORITIES: dict[str, int] = {
    "sketch_point": 0,
    "topology_vertex": 1,
    "intersection": 2,
    "circle_center": 2,
    "arc_center": 2,
    "line_midpoint": 3,
    "face_center": 4,
    "grid": 5,
}


class FaceWorkplaneResolutionError(ValueError):
    """A stable Chinese diagnostic for a workplane that cannot be resolved."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code.strip():
            raise ValueError("Workplane diagnostic code cannot be empty")
        if type(message) is not str or not message.strip():
            raise ValueError("Workplane diagnostic message cannot be empty")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class ResolvedFaceWorkplane:
    """Current exact OCC entities and the deterministic right-handed frame."""

    support_face_id: str
    target_body_id: str
    surface: Any
    volume: Any
    plane: SketchPlane
    outward_normal: tuple[float, float, float]
    strategy: FaceSketchWorkplaneStrategy
    area: float

    def __post_init__(self) -> None:
        face = LogicalEntityRef(self.support_face_id)
        body = LogicalEntityRef(self.target_body_id)
        if face.kind != "face" or body.kind != "body":
            raise ValueError("Invalid workplane or target Body logical ID type")
        if getattr(self.surface, "dimension", None) != 2:
            raise ValueError("The workplane OCC entity must be 2D")
        if getattr(self.volume, "dimension", None) != 3:
            raise ValueError("The target Body OCC entity must be 3D")
        if type(self.plane) is not SketchPlane:
            raise TypeError("The workplane must use SketchPlane")
        if type(self.strategy) is not FaceSketchWorkplaneStrategy:
            raise TypeError("Invalid workplane coordinate strategy")
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError("Workplane area must be finite and positive")

    @property
    def origin(self) -> tuple[float, float, float]:
        return self.plane.origin

    @property
    def u_axis(self) -> tuple[float, float, float]:
        return self.plane.x_direction

    @property
    def v_axis(self) -> tuple[float, float, float]:
        return self.plane.y_direction

    def direction_vector(
        self,
        direction: FaceSketchBooleanDirection,
    ) -> tuple[float, float, float]:
        """Return the signed extrusion direction in global coordinates."""

        if type(direction) is not FaceSketchBooleanDirection:
            raise TypeError("Invalid Extrude direction")
        return direction.vector(self.outward_normal)


def resolve_face_workplane(
    cad: Any,
    logical_entities: Mapping[str, tuple[Any, ...]],
    support_face: str | LogicalEntityRef | None,
    strategy: FaceSketchWorkplaneStrategy | None = None,
) -> ResolvedFaceWorkplane:
    """Resolve one stable logical plane face to its unique owning solid frame."""

    if support_face is None or (
        type(support_face) is str and not support_face.strip()
    ):
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-required",
            "No workplane selected",
        )
    reference = (
        support_face
        if type(support_face) is LogicalEntityRef
        else LogicalEntityRef(support_face)
    )
    if reference.kind != "face":
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-required",
            "No workplane selected",
        )
    entity_mapping = getattr(logical_entities, "logical_entities", logical_entities)
    try:
        surfaces = tuple(entity_mapping.get(reference.logical_id, ()))
    except AttributeError as error:
        raise TypeError("logical_entities must be a mapping of logical entities") from error
    if len(surfaces) != 1 or getattr(surfaces[0], "dimension", None) != 2:
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-stale",
            "The selected workplane is stale and cannot be uniquely restored from the original logical face ID",
        )
    surface = surfaces[0]
    try:
        geometry_type = cad.geometry_type(surface)
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-stale",
            "The selected workplane is stale and cannot be uniquely restored from the original logical face ID",
        ) from error
    if geometry_type.casefold() != "plane":
        raise FaceWorkplaneResolutionError(
            "face-workplane.not-plane",
            "The selected workplane is not an analytic plane",
        )

    try:
        adjacent_volumes = tuple(cad.adjacent(surface, dimension=3))
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.body-ambiguous",
            "The workplane cannot uniquely identify the target Body",
        ) from error
    if len(adjacent_volumes) != 1:
        raise FaceWorkplaneResolutionError(
            "face-workplane.body-ambiguous",
            "The workplane cannot uniquely identify the target Body",
        )
    volume = adjacent_volumes[0]
    body_id = _resolve_target_body_id(entity_mapping, volume)

    try:
        outward = tuple(cad.outward_surface_normal(surface, volume))
        origin = tuple(cad.center_of_mass(surface))
        area = float(cad.area(surface))
        normalized_strategy, u_axis = _resolve_u_axis(outward, strategy)
        v_axis = _normalized_cross(outward, u_axis)
        plane = SketchPlane(origin, u_axis, v_axis)
    except FaceWorkplaneResolutionError:
        raise
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.frame-unresolved",
            "The workplane coordinate system cannot be restored using the saved strategy",
        ) from error
    if not math.isfinite(area) or area <= 0.0:
        raise FaceWorkplaneResolutionError(
            "face-workplane.frame-unresolved",
            "The workplane coordinate system cannot be restored using the saved strategy",
        )
    return ResolvedFaceWorkplane(
        reference.logical_id,
        body_id,
        surface,
        volume,
        plane,
        outward,
        normalized_strategy,
        area,
    )


def _resolve_target_body_id(
    logical_entities: Mapping[str, tuple[Any, ...]],
    volume: Any,
) -> str:
    matches: list[str] = []
    for logical_id, entities in logical_entities.items():
        try:
            reference = LogicalEntityRef(logical_id)
        except (TypeError, ValueError):
            continue
        if reference.kind == "body" and volume in tuple(entities):
            matches.append(logical_id)
    specific = sorted(value for value in matches if value != "body:domain")
    if len(specific) == 1:
        return specific[0]
    if not specific and matches.count("body:domain") == 1:
        domain = tuple(logical_entities["body:domain"])
        if len(domain) == 1:
            return "body:domain"
    raise FaceWorkplaneResolutionError(
        "face-workplane.body-ambiguous",
        "The workplane cannot uniquely identify the target Body",
    )


def _resolve_u_axis(
    outward: tuple[float, ...],
    strategy: FaceSketchWorkplaneStrategy | None,
) -> tuple[FaceSketchWorkplaneStrategy, tuple[float, float, float]]:
    if len(outward) != 3:
        raise ValueError("The outward normal must contain three components")
    magnitude = math.sqrt(sum(value * value for value in outward))
    if not math.isfinite(magnitude) or magnitude <= _FRAME_TOLERANCE:
        raise ValueError("Invalid outward normal")
    normal = tuple(value / magnitude for value in outward)
    candidates = (
        tuple(_AXES)
        if strategy is None
        else (strategy.seed_axis,)
    )
    for axis_name in candidates:
        seed = _AXES[axis_name]
        dot = sum(left * right for left, right in zip(seed, normal, strict=True))
        projected = tuple(
            component - dot * normal_component
            for component, normal_component in zip(seed, normal, strict=True)
        )
        projected_magnitude = math.sqrt(sum(value * value for value in projected))
        if projected_magnitude <= _FRAME_TOLERANCE:
            continue
        sign = 1 if strategy is None else strategy.sign
        u_axis = tuple(sign * value / projected_magnitude for value in projected)
        return FaceSketchWorkplaneStrategy(axis_name, sign), u_axis
    raise FaceWorkplaneResolutionError(
        "face-workplane.frame-unresolved",
        "The workplane coordinate system cannot be restored using the saved strategy",
    )


def _normalized_cross(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    values = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= _FRAME_TOLERANCE:
        raise ValueError("The workplane V axis cannot be restored")
    return tuple(value / magnitude for value in values)


def provide_face_reference_points(
    cad: Any,
    logical_entities: Mapping[str, tuple[Any, ...]],
    workplane: ResolvedFaceWorkplane,
) -> tuple[SketchReferencePoint, ...]:
    """Return the five supported reference-point classes for one Face."""

    if type(workplane) is not ResolvedFaceWorkplane:
        raise TypeError("workplane must be a ResolvedFaceWorkplane")
    entity_mapping = getattr(logical_entities, "logical_entities", logical_entities)
    edges = tuple(cad.boundary((workplane.surface,), combined=True, recursive=False))
    vertices: set[Any] = set()
    points: list[SketchReferencePoint] = []
    for edge in edges:
        edge_vertices = tuple(cad.boundary((edge,), combined=True, recursive=False))
        vertices.update(edge_vertices)
        source_id = _logical_source_id(entity_mapping, edge, "edge")
        if source_id is None:
            continue
        geometry_type = str(cad.geometry_type(edge)).casefold()
        if geometry_type == "line":
            points.append(
                _reference_point(
                    workplane.plane,
                    source_id,
                    SketchExternalReferenceType.LINE_MIDPOINT,
                    cad.center_of_mass(edge),
                )
            )
        elif geometry_type in {"circle", "ellipse"}:
            derived_type = (
                SketchExternalReferenceType.CIRCLE_CENTER
                if len(edge_vertices) <= 1
                else SketchExternalReferenceType.ARC_CENTER
            )
            try:
                center = cad.circle_center(edge)
            except ValueError:
                continue
            points.append(
                _reference_point(
                    workplane.plane,
                    source_id,
                    derived_type,
                    center,
                )
            )
    for vertex in sorted(vertices, key=_entity_sort_key):
        source_id = _logical_source_id(entity_mapping, vertex, "point")
        if source_id is None:
            continue
        points.append(
            _reference_point(
                workplane.plane,
                source_id,
                SketchExternalReferenceType.TOPOLOGY_VERTEX,
                cad.center_of_mass(vertex),
            )
        )
    points.append(
        _reference_point(
            workplane.plane,
            workplane.support_face_id,
            SketchExternalReferenceType.FACE_CENTER,
            workplane.origin,
        )
    )
    return tuple(
        sorted(
            points,
            key=lambda point: (
                _SNAP_PRIORITIES[point.derived_type.value],
                point.reference.source.logical_id,
            ),
        )
    )


def resolve_external_reference_points(
    references: Iterable[SketchExternalReference],
    available: Iterable[SketchReferencePoint],
) -> tuple[SketchReferencePoint | None, ...]:
    """Resolve by exact logical source and derivation, never by proximity."""

    lookup = {
        (point.reference.source.logical_id, point.derived_type): point
        for point in available
    }
    result: list[SketchReferencePoint | None] = []
    for reference in references:
        if type(reference) is not SketchExternalReference:
            raise TypeError("references must contain SketchExternalReference values")
        result.append(lookup.get((reference.source.logical_id, reference.derived_type)))
    return tuple(result)


def select_sketch_snap_candidate(
    candidates: Iterable[SketchSnapCandidate],
    cursor: tuple[float, float],
    *,
    pixel_threshold: float = 9.0,
) -> SketchSnapCandidate | None:
    """Choose the nearest in-threshold candidate with the fixed tie priority."""

    cursor_x, cursor_y = (float(value) for value in cursor)
    threshold = float(pixel_threshold)
    if not all(math.isfinite(value) for value in (cursor_x, cursor_y, threshold)):
        raise ValueError("Invalid snap pixel parameters")
    if threshold < 0.0:
        raise ValueError("The snap pixel threshold cannot be negative")
    ranked: list[tuple[float, int, str, SketchSnapCandidate]] = []
    for candidate in candidates:
        if type(candidate) is not SketchSnapCandidate:
            raise TypeError("candidates must contain SketchSnapCandidate values")
        distance = math.hypot(
            candidate.screen_x - cursor_x,
            candidate.screen_y - cursor_y,
        )
        if distance <= threshold:
            stable_id = (
                candidate.sketch_point_id
                or (
                    candidate.reference_point.reference.id
                    if candidate.reference_point is not None
                    else ""
                )
            )
            ranked.append(
                (distance, _SNAP_PRIORITIES[candidate.kind], stable_id, candidate)
            )
    return min(ranked, default=None, key=lambda item: item[:3])[3] if ranked else None


def _reference_point(
    plane: SketchPlane,
    source_id: str,
    derived_type: SketchExternalReferenceType,
    position: Iterable[float],
) -> SketchReferencePoint:
    coordinates = tuple(float(value) for value in position)
    source = LogicalEntityRef(source_id)
    reference = SketchExternalReference(
        f"reference:{derived_type.value}:{source.logical_id}",
        source,
        derived_type,
    )
    u, v = plane.to_local(coordinates)
    return SketchReferencePoint(reference, coordinates, u, v)


def _logical_source_id(
    logical_entities: Mapping[str, tuple[Any, ...]],
    entity: Any,
    expected_kind: str,
) -> str | None:
    candidates: list[str] = []
    for logical_id, entities in logical_entities.items():
        try:
            reference = LogicalEntityRef(logical_id)
        except (TypeError, ValueError):
            continue
        if reference.kind == expected_kind and entity in tuple(entities):
            candidates.append(reference.logical_id)
    if not candidates:
        return None
    specific = tuple(value for value in candidates if not value.endswith(":domain"))
    return min(specific or tuple(candidates))


def _entity_sort_key(entity: Any) -> tuple[int, int]:
    return int(getattr(entity, "dimension", -1)), int(getattr(entity, "tag", -1))


__all__ = [
    "FaceWorkplaneResolutionError",
    "ResolvedFaceWorkplane",
    "SketchReferencePoint",
    "SketchSnapCandidate",
    "provide_face_reference_points",
    "resolve_external_reference_points",
    "resolve_face_workplane",
    "select_sketch_snap_candidate",
]
