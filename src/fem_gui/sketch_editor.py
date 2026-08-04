"""Qt-free draft controller for Phase 1 planar sketch authoring.

The controller owns only detached GUI state.  It deliberately has no Session,
viewport, or OCC dependency so incomplete gestures can be edited and undone
without changing a native project.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from fem.geometry.recipe_analysis import (
    SketchDiagnostic,
    SketchProfile,
    SketchProfileAnalysis,
    analyze_sketch_profiles,
    legacy_sketch_to_strict,
)
from fem.geometry.recipes import (
    SketchArc,
    SketchCircle,
    SketchCurve,
    SketchExternalCoincidence,
    SketchExternalReference,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem.geometry.sketch_support import SketchReferencePoint


@dataclass(frozen=True, slots=True)
class SketchDraftPoint:
    """Detached point data exposed by a draft snapshot."""

    id: str
    u: float
    v: float

    def to_strict(self) -> SketchPoint:
        return SketchPoint(self.id, self.u, self.v)


@dataclass(frozen=True, slots=True)
class SketchDraftSnapshot:
    """Immutable, detached state suitable for rendering or restoring."""

    name: str
    plane: SketchPlane
    points: tuple[SketchPoint, ...]
    curves: tuple[SketchCurve, ...]
    selected_ids: tuple[str, ...] = ()
    selected_kind: str | None = None
    pending_command: str | None = None
    revision: int = 0
    external_references: tuple[SketchExternalReference, ...] = ()
    external_coincidences: tuple[SketchExternalCoincidence, ...] = ()
    unresolved_reference_ids: tuple[str, ...] = ()

    @property
    def selected_entity_ids(self) -> tuple[str, ...]:
        return self.selected_ids

    @property
    def dirty(self) -> bool:
        return self.revision > 0


@dataclass(frozen=True, slots=True)
class _SketchGeometryState:
    """Undoable draft data, excluding transient editor interaction state."""

    name: str
    plane: SketchPlane
    points: tuple[SketchPoint, ...]
    curves: tuple[SketchCurve, ...]
    revision: int
    external_references: tuple[SketchExternalReference, ...]
    external_coincidences: tuple[SketchExternalCoincidence, ...]
    unresolved_reference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SketchInteractionState:
    """Transient selection and gesture state that never enters undo history."""

    selected_ids: tuple[str, ...]
    selected_kind: str | None
    pending_command: str | None


SketchDraftCurve = SketchCurve
SketchDraftDiagnostic = SketchDiagnostic

_TRIM_TOLERANCE = 1.0e-8


class _DiagnosticTuple(tuple[SketchDiagnostic, ...]):
    """Tuple that supports both property-style and Wire-editor call-style access."""

    def __new__(cls, values: tuple[SketchDiagnostic, ...] = ()):
        return super().__new__(cls, values)

    def __call__(self) -> tuple[SketchDiagnostic, ...]:
        return tuple(self)


class SketchDraftValidationError(ValueError):
    """Raised when a draft cannot be converted into a strict sketch."""

    def __init__(self, diagnostics: tuple[SketchDiagnostic, ...]):
        self.diagnostics = tuple(diagnostics)
        message = "; ".join(item.message for item in self.diagnostics)
        super().__init__(message or "草图无法完成")


class SketchDraftController:
    """Own and validate one detached Phase 1 sketch draft."""

    def __init__(
        self,
        name: str | SketchDraftSnapshot = "Sketch",
        *,
        snapshot: SketchDraftSnapshot | None = None,
        root: SketchGeometry | None = None,
        plane: SketchPlane | None = None,
        history_limit: int = 256,
    ) -> None:
        if isinstance(name, SketchDraftSnapshot):
            if snapshot is not None:
                raise ValueError("provide a draft snapshot only once")
            snapshot = name
            name = snapshot.name
        if snapshot is not None and root is not None:
            raise ValueError("provide either snapshot or root, not both")
        if root is not None:
            if type(root) is not SketchGeometry:
                raise TypeError("root must be a SketchGeometry")
            restored = root if root.is_strict else legacy_sketch_to_strict(root)
            snapshot = SketchDraftSnapshot(
                restored.name,
                restored.plane,
                restored.points,
                restored.curves,
            )
            name = snapshot.name
            plane = snapshot.plane
        if snapshot is not None and type(snapshot) is not SketchDraftSnapshot:
            raise TypeError("snapshot must be a SketchDraftSnapshot")
        if isinstance(history_limit, bool) or not isinstance(history_limit, int):
            raise TypeError("history_limit must be an integer")
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self._plane = SketchPlane.xy() if plane is None else plane
        if type(self._plane) is not SketchPlane:
            raise TypeError("plane must be a SketchPlane")
        initial = snapshot or SketchDraftSnapshot(
            str(name).strip(),
            self._plane,
            (),
            (),
        )
        self._plane = initial.plane
        self._name = initial.name
        self._points: dict[str, SketchPoint] = {point.id: point for point in initial.points}
        self._curves: dict[str, SketchCurve] = {curve.id: curve for curve in initial.curves}
        self._selected_ids: tuple[str, ...] = tuple(initial.selected_ids)
        self._selected_kind: str | None = initial.selected_kind
        self._pending_command: str | None = initial.pending_command
        self._revision = initial.revision
        if len({item.id for item in initial.external_references}) != len(
            initial.external_references
        ):
            raise ValueError("外部参考 ID 不能重复")
        if len({item.point_id for item in initial.external_coincidences}) != len(
            initial.external_coincidences
        ):
            raise ValueError("每个草图点最多只能绑定一个外部参考")
        self._external_references: dict[str, SketchExternalReference] = {
            reference.id: reference for reference in initial.external_references
        }
        self._external_coincidences: dict[str, SketchExternalCoincidence] = {
            coincidence.point_id: coincidence
            for coincidence in initial.external_coincidences
        }
        self._unresolved_reference_ids = set(initial.unresolved_reference_ids)
        self._validate_associations()
        self._undo: list[_SketchGeometryState] = []
        self._redo: list[_SketchGeometryState] = []
        self._history_limit = history_limit
        self._known_profile_ids: set[str] = set()

    @classmethod
    def from_geometry(cls, root: SketchGeometry) -> "SketchDraftController":
        return cls(root=root)

    @staticmethod
    def snapshot_from_geometry(root: SketchGeometry) -> SketchDraftSnapshot:
        if type(root) is not SketchGeometry:
            raise TypeError("root must be a SketchGeometry")
        strict = root if root.is_strict else legacy_sketch_to_strict(root)
        return SketchDraftSnapshot(strict.name, strict.plane, strict.points, strict.curves)

    @classmethod
    def from_sketch_geometry(cls, sketch: SketchGeometry) -> "SketchDraftController":
        """Restore a detached controller from a committed or legacy sketch."""

        if not isinstance(sketch, SketchGeometry):
            raise TypeError("sketch must be a SketchGeometry")
        strict = sketch if sketch.is_strict else legacy_sketch_to_strict(sketch)
        controller = cls(strict.name, plane=strict.plane)
        controller._points = {point.id: point for point in strict.points}
        controller._curves = {curve.id: curve for curve in strict.curves}
        controller._revision = 0
        return controller

    def restore_from_geometry(self, root: SketchGeometry) -> None:
        """Replace the detached draft with a committed sketch root."""

        if type(root) is not SketchGeometry:
            raise TypeError("root must be a SketchGeometry")
        strict = root if root.is_strict else legacy_sketch_to_strict(root)
        self._name = strict.name
        self._plane = strict.plane
        self._points = {point.id: point for point in strict.points}
        self._curves = {curve.id: curve for curve in strict.curves}
        self._selected_ids = ()
        self._selected_kind = None
        self._pending_command = None
        self._revision = 0
        self._external_references.clear()
        self._external_coincidences.clear()
        self._unresolved_reference_ids.clear()
        self._undo.clear()
        self._redo.clear()
        self._known_profile_ids.clear()

    restore_from_committed = restore_from_geometry

    @property
    def name(self) -> str:
        return self._name

    @property
    def plane(self) -> SketchPlane:
        return self._plane

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self._selected_ids

    @property
    def selected_entity_ids(self) -> tuple[str, ...]:
        return self._selected_ids

    @property
    def selection(self) -> tuple[str, str] | None:
        if not self._selected_ids:
            return None
        return (
            self._selected_kind or self._kind_for_id(self._selected_ids[0]),
            self._selected_ids[0],
        )

    @property
    def pending_command(self) -> str | None:
        return self._pending_command

    @property
    def is_dirty(self) -> bool:
        return self._revision > 0

    @property
    def dirty(self) -> bool:
        return self.is_dirty

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def snapshot(self) -> SketchDraftSnapshot:
        """Return a detached immutable snapshot of current draft state."""

        return SketchDraftSnapshot(
            self._name,
            self._plane,
            tuple(self._points.values()),
            tuple(self._curves.values()),
            self._selected_ids,
            self._selected_kind,
            self._pending_command,
            self._revision,
            tuple(self._external_references.values()),
            tuple(self._external_coincidences.values()),
            tuple(sorted(self._unresolved_reference_ids)),
        )

    @property
    def state(self) -> SketchDraftSnapshot:
        return self.snapshot()

    @property
    def current_snapshot(self) -> SketchDraftSnapshot:
        return self.snapshot()

    def set_sketch_name(self, name: str) -> None:
        normalized = str(name).strip()

        def apply() -> None:
            self._name = normalized

        self._mutate(apply)

    set_name = set_sketch_name

    def add_point(
        self,
        *args: object,
        point_id: str | None = None,
        name: str | None = None,
        u: float | None = None,
        v: float | None = None,
        x: float | None = None,
        y: float | None = None,
        external_reference: SketchReferencePoint | None = None,
    ) -> SketchPoint:
        if point_id is not None and name is not None:
            raise ValueError("point_id and name are aliases; provide only one")
        normalized_id = point_id or name
        if len(args) == 3:
            if normalized_id is not None or not isinstance(args[0], str):
                raise TypeError("three positional point arguments are (id, u, v)")
            normalized_id = args[0]
            u, v = args[1], args[2]
        elif len(args) == 2:
            if u is not None or v is not None or x is not None or y is not None:
                raise TypeError("point coordinates were provided twice")
            u, v = args
        elif len(args) == 1:
            if isinstance(args[0], str) and normalized_id is None:
                normalized_id = args[0]
            else:
                raise TypeError("a single positional point argument must be an ID")
        elif args:
            raise TypeError("point requires two coordinates")
        if u is None:
            u = x
        if v is None:
            v = y
        if u is None or v is None:
            raise TypeError("point requires u/v coordinates")
        normalized_id = normalized_id or self._next_id("P", self._points)
        point = SketchPoint(normalized_id, u, v)
        if external_reference is not None and type(external_reference) is not SketchReferencePoint:
            raise TypeError("external_reference must be a SketchReferencePoint")
        if external_reference is not None:
            point = SketchPoint(point.id, external_reference.u, external_reference.v)

        def apply() -> None:
            self._assert_new_id(point.id)
            self._points[point.id] = point
            if external_reference is not None:
                self._bind_external_reference(point.id, external_reference)

        self._mutate(apply)
        return point

    place_point = add_point

    def move_point(
        self,
        point_id: str,
        u: float | None = None,
        v: float | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
    ) -> SketchPoint:
        point = self._require_point(point_id)
        if point_id in self._external_coincidences:
            raise ValueError("关联点不能直接移动，请先执行“解除关联”")
        replacement = SketchPoint(
            point.id,
            point.u if u is None and x is None else (u if u is not None else x),
            point.v if v is None and y is None else (v if v is not None else y),
        )

        def apply() -> None:
            self._points[point.id] = replacement

        self._mutate(apply)
        return replacement

    update_point = move_point
    set_point_coordinates = move_point

    def move_points(
        self,
        coordinates: dict[str, tuple[float, float]],
    ) -> tuple[SketchPoint, ...]:
        """Move several free points as one atomic, undoable edit."""

        if not isinstance(coordinates, dict):
            raise TypeError("coordinates must be a point ID mapping")
        replacements: list[SketchPoint] = []
        for point_id, coordinate in coordinates.items():
            point = self._require_point(point_id)
            if point_id in self._external_coincidences:
                raise ValueError("关联点不能直接移动，请先执行“解除关联”")
            u, v = _coordinate_pair(coordinate, "coordinates")
            replacements.append(SketchPoint(point.id, u, v))

        def apply() -> None:
            self._points.update({point.id: point for point in replacements})

        self._mutate(apply)
        return tuple(replacements)

    batch_move_points = move_points

    def external_reference_for_point(
        self,
        point_id: str,
    ) -> SketchExternalReference | None:
        self._require_point(point_id)
        coincidence = self._external_coincidences.get(point_id)
        return (
            None
            if coincidence is None
            else self._external_references[coincidence.reference_id]
        )

    def association_status(self, point_id: str) -> str:
        reference = self.external_reference_for_point(point_id)
        if reference is None:
            return "自由"
        if reference.id in self._unresolved_reference_ids:
            return "未解析"
        return "已关联"

    def associate_point(
        self,
        point_id: str,
        reference_point: SketchReferencePoint,
    ) -> None:
        self._require_point(point_id)
        if type(reference_point) is not SketchReferencePoint:
            raise TypeError("reference_point must be a SketchReferencePoint")

        def apply() -> None:
            self._points[point_id] = SketchPoint(
                point_id,
                reference_point.u,
                reference_point.v,
            )
            self._bind_external_reference(point_id, reference_point)

        self._mutate(apply)

    def release_point_association(self, point_id: str) -> None:
        self._require_point(point_id)
        if point_id not in self._external_coincidences:
            return

        def apply() -> None:
            self._remove_point_association(point_id)

        self._mutate(apply)

    def refresh_external_references(
        self,
        available: tuple[SketchReferencePoint, ...],
    ) -> None:
        """Re-resolve exact sources and retain coordinates for missing sources."""

        lookup = {
            (point.reference.source.logical_id, point.derived_type): point
            for point in available
        }
        unresolved: set[str] = set()
        for point_id, coincidence in self._external_coincidences.items():
            reference = self._external_references[coincidence.reference_id]
            resolved = lookup.get((reference.source.logical_id, reference.derived_type))
            if resolved is None:
                unresolved.add(reference.id)
                continue
            self._points[point_id] = SketchPoint(point_id, resolved.u, resolved.v)
        self._unresolved_reference_ids = unresolved

    def delete_point(self, point_id: str) -> None:
        self._require_point(point_id)

        def apply() -> None:
            del self._points[point_id]
            dependent = tuple(
                curve_id
                for curve_id, curve in self._curves.items()
                if point_id in _curve_point_ids(curve)
            )
            for curve_id in dependent:
                del self._curves[curve_id]
            self._remove_point_association(point_id)
            self._clear_selection_id(point_id)

        self._mutate(apply)

    def dependent_curve_ids(self, point_id: str) -> tuple[str, ...]:
        """Return curves that will be removed when the point is deleted."""

        self._require_point(point_id)
        return tuple(
            curve_id
            for curve_id, curve in self._curves.items()
            if point_id in _curve_point_ids(curve)
        )

    def point_usage(self, point_id: str) -> tuple[str, ...]:
        """Return the stable geometric roles of one draft point."""

        self._require_point(point_id)
        endpoint = False
        circle_center = False
        arc_center = False
        for curve in self._curves.values():
            if isinstance(curve, SketchLine) and point_id in {
                curve.start_point_id,
                curve.end_point_id,
            }:
                endpoint = True
            elif isinstance(curve, SketchCircle) and point_id == curve.center_point_id:
                circle_center = True
            elif isinstance(curve, SketchArc):
                if point_id in {curve.start_point_id, curve.end_point_id}:
                    endpoint = True
                if point_id == curve.center_point_id:
                    arc_center = True
        return tuple(
            label
            for active, label in (
                (endpoint, "端点"),
                (circle_center, "圆心"),
                (arc_center, "弧心"),
            )
            if active
        )

    def add_line(
        self,
        *args: object,
        start_point_id: str | None = None,
        end_point_id: str | None = None,
        curve_id: str | None = None,
        name: str | None = None,
    ) -> SketchLine:
        if curve_id is not None and name is not None:
            raise ValueError("curve_id and name are aliases; provide only one")
        curve_id = curve_id or name
        if len(args) == 3:
            if any(value is not None for value in (start_point_id, end_point_id, curve_id)):
                raise TypeError("line arguments were provided twice")
            curve_id, start_point_id, end_point_id = args
        elif len(args) == 2:
            if start_point_id is not None or end_point_id is not None:
                raise TypeError("line endpoints were provided twice")
            start_point_id, end_point_id = args
        elif args:
            raise TypeError("line requires start and end point IDs")
        if not isinstance(start_point_id, str) or not isinstance(end_point_id, str):
            raise TypeError("line endpoints must be point IDs")
        self._require_point(start_point_id)
        self._require_point(end_point_id)
        line = SketchLine(
            curve_id or self._next_id("L", self._curves),
            start_point_id,
            end_point_id,
        )

        def apply() -> None:
            self._assert_new_id(line.id)
            self._curves[line.id] = line

        self._mutate(apply)
        return line

    def add_line_to_point(
        self,
        start_point_id: str,
        end: object,
        *,
        curve_id: str | None = None,
        point_id: str | None = None,
        external_reference: SketchReferencePoint | None = None,
    ) -> SketchLine:
        """Draw one segment, creating its end point in the same transaction."""

        self._require_point(start_point_id)
        if external_reference is not None and type(external_reference) is not SketchReferencePoint:
            raise TypeError("external_reference must be a SketchReferencePoint")
        if isinstance(end, str):
            if point_id is not None or external_reference is not None:
                raise ValueError("existing end points cannot be replaced or associated")
            end_point = self._require_point(end)
            create_point = False
        else:
            u, v = _coordinate_pair(end, "end")
            end_point = SketchPoint(
                point_id or self._next_id("P", self._points),
                external_reference.u if external_reference is not None else u,
                external_reference.v if external_reference is not None else v,
            )
            create_point = True
        line = SketchLine(
            curve_id or self._next_id("L", self._curves),
            start_point_id,
            end_point.id,
        )

        def apply() -> None:
            if create_point:
                self._assert_new_id(end_point.id)
                self._points[end_point.id] = end_point
            self._assert_new_id(line.id)
            self._curves[line.id] = line
            if external_reference is not None:
                self._bind_external_reference(end_point.id, external_reference)

        self._mutate(apply)
        return line

    def add_rectangle(
        self,
        *args: object,
        first: tuple[float, float] | None = None,
        second: tuple[float, float] | None = None,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        point_ids: tuple[str, str, str, str] | None = None,
        curve_ids: tuple[str, str, str, str] | None = None,
        external_references: tuple[
            SketchReferencePoint | None,
            SketchReferencePoint | None,
            SketchReferencePoint | None,
            SketchReferencePoint | None,
        ] | None = None,
    ) -> tuple[SketchPoint, ...]:
        if len(args) == 2:
            if first is not None or second is not None:
                raise TypeError("rectangle corners were provided twice")
            first, second = args
        elif len(args) == 4:
            if any(value is not None for value in (first, second, x, y, width, height)):
                raise TypeError("rectangle coordinates were provided twice")
            first = (args[0], args[1])  # type: ignore[assignment]
            second = (args[2], args[3])  # type: ignore[assignment]
        elif args:
            raise TypeError("rectangle requires two corners or four coordinates")
        if first is None or second is None:
            if None in {x, y, width, height}:
                raise TypeError("rectangle requires x, y, width, and height")
            first = (x, y)  # type: ignore[assignment]
            second = (x + width, y + height)  # type: ignore[operator]
        x1, y1 = _coordinate_pair(first, "first")
        x2, y2 = _coordinate_pair(second, "second")
        if math.isclose(x1, x2) or math.isclose(y1, y2):
            raise ValueError("矩形宽度和高度必须大于零")
        left, right = sorted((x1, x2))
        bottom, top = sorted((y1, y2))
        ids = point_ids or tuple(self._next_id("P", self._points, offset=index) for index in range(4))
        if len(ids) != 4:
            raise ValueError("rectangle requires four point IDs")
        curve_names = curve_ids or tuple(
            self._next_id("L", self._curves, offset=index) for index in range(4)
        )
        if len(curve_names) != 4:
            raise ValueError("rectangle requires four curve IDs")
        points = (
            SketchPoint(ids[0], left, bottom),
            SketchPoint(ids[1], right, bottom),
            SketchPoint(ids[2], right, top),
            SketchPoint(ids[3], left, top),
        )
        references = external_references or (None, None, None, None)
        if len(references) != 4 or any(
            item is not None and type(item) is not SketchReferencePoint
            for item in references
        ):
            raise TypeError("rectangle external references must contain four values")
        lines = (
            SketchLine(curve_names[0], ids[0], ids[1]),
            SketchLine(curve_names[1], ids[1], ids[2]),
            SketchLine(curve_names[2], ids[2], ids[3]),
            SketchLine(curve_names[3], ids[3], ids[0]),
        )

        def apply() -> None:
            for item in (*points, *lines):
                self._assert_new_id(item.id)
            self._points.update({item.id: item for item in points})
            self._curves.update({item.id: item for item in lines})
            for point, reference_point in zip(points, references, strict=True):
                if reference_point is not None:
                    self._points[point.id] = SketchPoint(
                        point.id,
                        reference_point.u,
                        reference_point.v,
                    )
                    self._bind_external_reference(point.id, reference_point)

        self._mutate(apply)
        return points

    def add_circle(
        self,
        center: object,
        radius: object | None = None,
        third: object | None = None,
        *,
        point_id: str | None = None,
        curve_id: str | None = None,
        external_reference: SketchReferencePoint | None = None,
    ) -> SketchCircle:
        if isinstance(center, (int, float)) and radius is not None and third is not None:
            center_pair = (center, radius)
            radius_value = third
        else:
            center_pair = _coordinate_pair(center, "center")
            radius_value = radius
        if radius_value is None:
            raise TypeError("circle radius is required")
        x, y = _coordinate_pair(center_pair, "center")
        center_name = point_id or self._next_id("P", self._points)
        circle_name = curve_id or self._next_id("C", self._curves)
        point = SketchPoint(center_name, x, y)
        if external_reference is not None and type(external_reference) is not SketchReferencePoint:
            raise TypeError("external_reference must be a SketchReferencePoint")
        if external_reference is not None:
            point = SketchPoint(center_name, external_reference.u, external_reference.v)
        circle = SketchCircle(circle_name, center_name, radius_value)

        def apply() -> None:
            self._assert_new_id(point.id)
            self._assert_new_id(circle.id)
            self._points[point.id] = point
            self._curves[circle.id] = circle
            if external_reference is not None:
                self._bind_external_reference(point.id, external_reference)

        self._mutate(apply)
        return circle

    def add_arc(
        self,
        start: object,
        through: object,
        end: object,
        *,
        start_point_id: str | None = None,
        end_point_id: str | None = None,
        center_point_id: str | None = None,
        curve_id: str | None = None,
        start_external_reference: SketchReferencePoint | None = None,
        center_external_reference: SketchReferencePoint | None = None,
        end_external_reference: SketchReferencePoint | None = None,
    ) -> SketchArc:
        new_points: list[SketchPoint] = []
        start_id = self._coerce_endpoint(start, start_point_id, "P", new_points)
        end_id = self._coerce_endpoint(end, end_point_id, "P", new_points)
        through_pair = _coordinate_pair(through, "through")
        start_point = self._require_point_or_new(start_id, new_points)
        end_point = self._require_point_or_new(end_id, new_points)
        center_x, center_y = _circumcenter(
            (start_point.u, start_point.v),
            through_pair,
            (end_point.u, end_point.v),
        )
        center_id = center_point_id or self._next_id(
            "P",
            {**self._points, **{point.id: point for point in new_points}},
        )
        center_point = SketchPoint(center_id, center_x, center_y)
        cross = (
            (through_pair[0] - center_x) * (end_point.v - center_y)
            - (through_pair[1] - center_y) * (end_point.u - center_x)
        )
        orientation = "ccw" if cross >= 0.0 else "cw"
        arc = SketchArc(
            curve_id or self._next_id("A", self._curves),
            start_id,
            center_id,
            end_id,
            orientation,
        )
        for reference_point in (
            start_external_reference,
            center_external_reference,
            end_external_reference,
        ):
            if reference_point is not None and type(reference_point) is not SketchReferencePoint:
                raise TypeError("arc external references must be SketchReferencePoint values")

        def apply() -> None:
            for point in (*new_points, center_point):
                self._assert_new_id(point.id)
            self._assert_new_id(arc.id)
            self._points.update({point.id: point for point in (*new_points, center_point)})
            self._curves[arc.id] = arc
            for point_id, reference_point in (
                (start_id, start_external_reference),
                (center_id, center_external_reference),
                (end_id, end_external_reference),
            ):
                if reference_point is not None:
                    if point_id != center_id:
                        self._points[point_id] = SketchPoint(
                            point_id,
                            reference_point.u,
                            reference_point.v,
                        )
                    self._bind_external_reference(point_id, reference_point)

        self._mutate(apply)
        return arc

    def update_curve_parameters(self, curve_id: str, **parameters: object) -> SketchCurve:
        curve = self._require_curve(curve_id)
        allowed = set(parameters)
        if isinstance(curve, SketchLine):
            unknown = allowed - {"start_point_id", "end_point_id"}
            if unknown:
                raise ValueError(f"line parameters are unsupported: {sorted(unknown)}")
            replacement: SketchCurve = SketchLine(
                curve.id,
                parameters.get("start_point_id", curve.start_point_id),
                parameters.get("end_point_id", curve.end_point_id),
            )
        elif isinstance(curve, SketchArc):
            unknown = allowed - {
                "start_point_id",
                "center_point_id",
                "end_point_id",
                "orientation",
            }
            if unknown:
                raise ValueError(f"arc parameters are unsupported: {sorted(unknown)}")
            replacement = SketchArc(
                curve.id,
                parameters.get("start_point_id", curve.start_point_id),
                parameters.get("center_point_id", curve.center_point_id),
                parameters.get("end_point_id", curve.end_point_id),
                parameters.get("orientation", curve.orientation),
            )
        else:
            unknown = allowed - {"center_point_id", "radius"}
            if unknown:
                raise ValueError(f"circle parameters are unsupported: {sorted(unknown)}")
            replacement = SketchCircle(
                curve.id,
                parameters.get("center_point_id", curve.center_point_id),
                parameters.get("radius", curve.radius),
            )
        for point_id in _curve_point_ids(replacement):
            self._require_point(point_id)

        def apply() -> None:
            self._curves[curve_id] = replacement

        self._mutate(apply)
        return replacement

    update_curve = update_curve_parameters

    def delete_curve(self, curve_id: str) -> None:
        curve = self._require_curve(curve_id)
        candidate_point_ids = _curve_point_ids(curve)

        def apply() -> None:
            del self._curves[curve_id]
            used_point_ids = {
                point_id
                for remaining in self._curves.values()
                for point_id in _curve_point_ids(remaining)
            }
            for point_id in candidate_point_ids:
                if point_id not in used_point_ids:
                    self._points.pop(point_id, None)
                    self._remove_point_association(point_id)
            self._clear_selection_id(curve_id)

        self._mutate(apply)

    def delete(self, entity_id: str) -> None:
        """Delete a point or curve, matching the generic editor action."""

        if entity_id in self._points:
            self.delete_point(entity_id)
            return
        if entity_id in self._curves:
            self.delete_curve(entity_id)
            return
        raise KeyError(entity_id)

    def delete_many(self, entity_ids: tuple[str, ...] | list[str]) -> None:
        """Delete several points or curves as one atomic undoable edit."""

        ids = tuple(dict.fromkeys(entity_ids))
        if not ids:
            return
        for entity_id in ids:
            self._assert_entity(entity_id)

        point_ids = {entity_id for entity_id in ids if entity_id in self._points}
        explicit_curve_ids = {
            entity_id for entity_id in ids if entity_id in self._curves
        }
        curve_ids = set(explicit_curve_ids)
        curve_ids.update(
            curve_id
            for curve_id, curve in self._curves.items()
            if point_ids.intersection(_curve_point_ids(curve))
        )

        def apply() -> None:
            candidate_point_ids = {
                point_id
                for curve_id in explicit_curve_ids
                for point_id in _curve_point_ids(self._curves[curve_id])
            }
            for curve_id in curve_ids:
                del self._curves[curve_id]
            used_point_ids = {
                point_id
                for curve in self._curves.values()
                for point_id in _curve_point_ids(curve)
            }
            removable = point_ids | {
                point_id
                for point_id in candidate_point_ids
                if point_id not in used_point_ids
            }
            for point_id in removable:
                self._points.pop(point_id, None)
                self._remove_point_association(point_id)
            self._selected_ids = tuple(
                entity_id
                for entity_id in self._selected_ids
                if entity_id in self._points or entity_id in self._curves
            )
            if not self._selected_ids:
                self._selected_kind = None

        self._mutate(apply)

    delete_entities = delete_many

    def trim_curve(
        self,
        curve_id: str,
        point: object | None = None,
    ) -> tuple[SketchCurve, ...]:
        """Remove the clicked curve interval as one undoable operation.

        A line is split at its explicit intersections and the interval nearest
        ``point`` is removed.  With one intersection, the clicked end is
        removed.  If no unique split is available, or the target is a circle
        or arc, the whole target curve is deleted.
        """

        target = self._require_curve(curve_id)
        if not isinstance(target, SketchLine):
            self.delete_curve(curve_id)
            return ()
        start = self._require_point(target.start_point_id)
        end = self._require_point(target.end_point_id)
        start_xy = (start.u, start.v)
        end_xy = (end.u, end.v)
        intersections: list[tuple[float, tuple[float, float]]] = []
        for other in self._curves.values():
            if other.id == curve_id:
                continue
            try:
                intersections.extend(
                    _line_curve_intersections(
                        start_xy,
                        end_xy,
                        other,
                        self._points,
                    )
                )
            except ValueError:
                # Collinear overlaps and tangencies do not define a unique
                # trim boundary.  Other usable intersections remain valid.
                continue
        intersections = _unique_trim_intersections(intersections)
        if not intersections:
            self.delete_curve(curve_id)
            return ()
        parameters = [0.0, *(item[0] for item in intersections), 1.0]
        if point is None:
            if len(intersections) >= 2:
                lower_index = 1
            else:
                lower_index = 0
        else:
            target_point = _coordinate_pair(point, "trim point")
            target_parameter = _line_projection_parameter(
                start_xy,
                end_xy,
                target_point,
            )
            if target_parameter is None:
                raise ValueError("无法确定修剪位置")
            lower_index = min(
                range(len(parameters) - 1),
                key=lambda index: (
                    0.0
                    if parameters[index] <= target_parameter <= parameters[index + 1]
                    else 1.0,
                    abs(
                        target_parameter
                        - 0.5 * (parameters[index] + parameters[index + 1])
                    ),
                ),
            )
        lower = parameters[lower_index]
        upper = parameters[lower_index + 1]
        if upper - lower <= _TRIM_TOLERANCE:
            raise ValueError("trim interval is smaller than the geometry tolerance")

        def coordinate_at(parameter: float) -> tuple[float, float]:
            return (
                start_xy[0] + parameter * (end_xy[0] - start_xy[0]),
                start_xy[1] + parameter * (end_xy[1] - start_xy[1]),
            )

        point_ids: dict[tuple[float, float], str] = {}
        new_points: list[SketchPoint] = []
        occupied_points = dict(self._points)

        def point_id_at(parameter: float) -> str:
            if parameter <= _TRIM_TOLERANCE:
                return target.start_point_id
            if parameter >= 1.0 - _TRIM_TOLERANCE:
                return target.end_point_id
            coordinate = coordinate_at(parameter)
            key = (round(coordinate[0], 12), round(coordinate[1], 12))
            if key in point_ids:
                return point_ids[key]
            existing_id = _find_point_at(occupied_points, coordinate)
            if existing_id is None:
                existing_id = self._next_id("P", occupied_points)
                new_point = SketchPoint(existing_id, *coordinate)
                occupied_points[existing_id] = new_point
                new_points.append(new_point)
            point_ids[key] = existing_id
            return existing_id

        left_start_id = target.start_point_id
        left_end_id = point_id_at(lower)
        right_start_id = point_id_at(upper)
        right_end_id = target.end_point_id
        replacement_endpoints: list[tuple[str, str]] = []
        if lower > _TRIM_TOLERANCE:
            replacement_endpoints.append((left_start_id, left_end_id))
        if upper < 1.0 - _TRIM_TOLERANCE:
            replacement_endpoints.append((right_start_id, right_end_id))
        replacements = [
            SketchLine(
                target.id if index == 0 else self._next_id("L", self._curves),
                start_id,
                end_id,
            )
            for index, (start_id, end_id) in enumerate(replacement_endpoints)
        ]

        def apply() -> None:
            for item in new_points:
                self._assert_new_id(item.id)
                self._points[item.id] = item
            del self._curves[target.id]
            for item in replacements:
                self._assert_new_id(item.id)
                self._curves[item.id] = item
            used_point_ids = {
                point_id
                for remaining in self._curves.values()
                for point_id in _curve_point_ids(remaining)
            }
            for point_id in _curve_point_ids(target):
                if point_id not in used_point_ids:
                    self._points.pop(point_id, None)
                    self._remove_point_association(point_id)

        self._mutate(apply)
        return tuple(replacements)

    trim = trim_curve
    trim_segment = trim_curve

    def select(self, entity_id: str, *, kind: str | None = None) -> tuple[str, ...]:
        normalized_kind = kind or self._kind_for_id(entity_id)
        if normalized_kind == "point":
            self._require_point(entity_id)
        elif normalized_kind == "edge":
            self._require_curve(entity_id)
        elif normalized_kind != "profile" or entity_id not in self._known_profile_ids:
            raise KeyError(entity_id)

        self._selected_ids = (entity_id,)
        self._selected_kind = normalized_kind
        return self._selected_ids

    select_entity = select

    def select_point(self, point_id: str) -> tuple[str, ...]:
        return self.select(point_id, kind="point")

    def select_curve(self, curve_id: str) -> tuple[str, ...]:
        return self.select(curve_id, kind="edge")

    def select_profile(self, profile_id: str) -> tuple[str, ...]:
        return self.select(profile_id, kind="profile")

    def select_many(self, entity_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        ids = tuple(dict.fromkeys(entity_ids))
        if not ids:
            return self.clear_selection()
        kinds = {self._kind_for_id(item) for item in ids}
        if len(kinds) != 1:
            raise ValueError("同一种实体类型才能多选")
        for item in ids:
            self._assert_selectable(item)
        self._selected_ids = ids
        self._selected_kind = next(iter(kinds))
        return ids

    def clear_selection(self) -> tuple[str, ...]:
        self._selected_ids = ()
        self._selected_kind = None
        return ()

    def begin_pending_command(self, command: str) -> None:
        normalized = str(command).strip()
        if not normalized:
            raise ValueError("pending command cannot be empty")

        self._pending_command = normalized

    begin_command = begin_pending_command

    def cancel_pending_command(self) -> None:
        if self._pending_command is None:
            return

        self._pending_command = None

    cancel_command = cancel_pending_command

    def complete_pending_command(self) -> str | None:
        command = self._pending_command
        self.cancel_pending_command()
        return command

    complete_command = complete_pending_command

    def undo(self) -> SketchDraftSnapshot:
        if not self._undo:
            return self.snapshot()
        current = self._geometry_state()
        previous = self._undo.pop()
        self._redo.append(current)
        self._restore_geometry_state(previous)
        self._prune_selection()
        return self.snapshot()

    def redo(self) -> SketchDraftSnapshot:
        if not self._redo:
            return self.snapshot()
        current = self._geometry_state()
        following = self._redo.pop()
        self._undo.append(current)
        self._restore_geometry_state(following)
        self._prune_selection()
        return self.snapshot()

    def restore_snapshot(self, snapshot: SketchDraftSnapshot) -> None:
        if type(snapshot) is not SketchDraftSnapshot:
            raise TypeError("snapshot must be a SketchDraftSnapshot")

        def apply() -> None:
            self._restore_state(snapshot)

        self._mutate(apply)

    restore = restore_snapshot

    def derive_profiles(self) -> SketchProfileAnalysis:
        try:
            sketch = self._to_strict_unchecked()
        except (TypeError, ValueError) as error:
            analysis = SketchProfileAnalysis(
                (),
                (
                    SketchDiagnostic(
                        "sketch.incomplete-draft",
                        str(error),
                        blocking=False,
                        severity="warning",
                    ),
                ),
            )
        else:
            analysis = analyze_sketch_profiles(sketch)
        self._known_profile_ids = {profile.id for profile in analysis.profiles}
        return analysis

    @property
    def profiles(self) -> tuple[SketchProfile, ...]:
        return self.derive_profiles().profiles

    @property
    def editing_diagnostics(self) -> tuple[SketchDiagnostic, ...]:
        diagnostics: list[SketchDiagnostic] = []
        if not self._name:
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.blank-name",
                    "草图名称不能为空",
                    blocking=False,
                    severity="warning",
                )
            )
        analysis = self.derive_profiles()
        for diagnostic in analysis.diagnostics:
            diagnostics.append(
                SketchDiagnostic(
                    diagnostic.code,
                    diagnostic.message,
                    diagnostic.entity_ids,
                    blocking=False,
                    severity="warning",
                )
            )
        return _DiagnosticTuple(_unique_diagnostics(diagnostics))

    @property
    def finish_diagnostics(self) -> tuple[SketchDiagnostic, ...]:
        diagnostics: list[SketchDiagnostic] = []
        if not self._name:
            diagnostics.append(SketchDiagnostic("sketch.blank-name", "草图名称不能为空"))
        try:
            sketch = self._to_strict_unchecked()
        except (TypeError, ValueError) as error:
            diagnostics.append(SketchDiagnostic("sketch.invalid-domain", str(error)))
            return _DiagnosticTuple(_unique_diagnostics(diagnostics))
        analysis = analyze_sketch_profiles(sketch)
        diagnostics.extend(analysis.diagnostics)
        if not analysis.profiles:
            diagnostics.append(
                SketchDiagnostic("sketch.no-profile", "草图至少需要一个闭合轮廓")
            )
        return _DiagnosticTuple(_unique_diagnostics(diagnostics))

    @property
    def can_finish(self) -> bool:
        return not any(item.blocking for item in self.finish_diagnostics)

    def to_sketch_geometry(self) -> SketchGeometry:
        diagnostics = self.finish_diagnostics
        if any(item.blocking for item in diagnostics):
            raise SketchDraftValidationError(diagnostics)
        return self._to_strict_unchecked()

    build_sketch_geometry = to_sketch_geometry
    finish = to_sketch_geometry
    to_geometry = to_sketch_geometry
    build_geometry = to_sketch_geometry
    serialize_complete_domain = to_sketch_geometry

    def _to_strict_unchecked(self) -> SketchGeometry:
        return SketchGeometry(
            self._name,
            self._plane,
            tuple(self._points.values()),
            tuple(self._curves.values()),
        )

    def _mutate(self, operation):
        before = self._geometry_state()
        interaction = self._interaction_state()
        try:
            result = operation()
        except BaseException:
            self._restore_geometry_state(before)
            self._restore_interaction_state(interaction)
            raise
        after = self._geometry_state()
        if after == before:
            return result
        self._known_profile_ids.clear()
        self._undo.append(before)
        if len(self._undo) > self._history_limit:
            del self._undo[0]
        self._redo.clear()
        self._revision += 1
        return result

    def _geometry_state(self) -> _SketchGeometryState:
        return _SketchGeometryState(
            self._name,
            self._plane,
            tuple(self._points.values()),
            tuple(self._curves.values()),
            self._revision,
            tuple(self._external_references.values()),
            tuple(self._external_coincidences.values()),
            tuple(sorted(self._unresolved_reference_ids)),
        )

    def _interaction_state(self) -> _SketchInteractionState:
        return _SketchInteractionState(
            self._selected_ids,
            self._selected_kind,
            self._pending_command,
        )

    def _restore_geometry_state(self, state: _SketchGeometryState) -> None:
        self._name = state.name
        self._plane = state.plane
        self._points = {point.id: point for point in state.points}
        self._curves = {curve.id: curve for curve in state.curves}
        self._revision = state.revision
        self._external_references = {
            reference.id: reference for reference in state.external_references
        }
        self._external_coincidences = {
            coincidence.point_id: coincidence
            for coincidence in state.external_coincidences
        }
        self._unresolved_reference_ids = set(state.unresolved_reference_ids)
        self._known_profile_ids.clear()
        self._validate_associations()

    def _restore_interaction_state(self, state: _SketchInteractionState) -> None:
        self._selected_ids = state.selected_ids
        self._selected_kind = state.selected_kind
        self._pending_command = state.pending_command

    def _restore_state(self, snapshot: SketchDraftSnapshot) -> None:
        if len({item.id for item in snapshot.external_references}) != len(
            snapshot.external_references
        ):
            raise ValueError("外部参考 ID 不能重复")
        if len({item.point_id for item in snapshot.external_coincidences}) != len(
            snapshot.external_coincidences
        ):
            raise ValueError("每个草图点最多只能绑定一个外部参考")
        self._name = snapshot.name
        self._plane = snapshot.plane
        self._points = {point.id: point for point in snapshot.points}
        self._curves = {curve.id: curve for curve in snapshot.curves}
        self._selected_ids = tuple(snapshot.selected_ids)
        self._selected_kind = snapshot.selected_kind
        self._pending_command = snapshot.pending_command
        self._revision = snapshot.revision
        self._external_references = {
            reference.id: reference for reference in snapshot.external_references
        }
        self._external_coincidences = {
            coincidence.point_id: coincidence
            for coincidence in snapshot.external_coincidences
        }
        self._unresolved_reference_ids = set(snapshot.unresolved_reference_ids)
        self._known_profile_ids.clear()
        self._validate_associations()

    def _bind_external_reference(
        self,
        point_id: str,
        reference_point: SketchReferencePoint,
    ) -> None:
        self._remove_point_association(point_id)
        reference = reference_point.reference
        self._external_references[reference.id] = reference
        self._external_coincidences[point_id] = SketchExternalCoincidence(
            point_id,
            reference.id,
        )
        self._unresolved_reference_ids.discard(reference.id)

    def _remove_point_association(self, point_id: str) -> None:
        coincidence = self._external_coincidences.pop(point_id, None)
        if coincidence is None:
            return
        reference_id = coincidence.reference_id
        if not any(
            item.reference_id == reference_id
            for item in self._external_coincidences.values()
        ):
            self._external_references.pop(reference_id, None)
            self._unresolved_reference_ids.discard(reference_id)

    def _validate_associations(self) -> None:
        for point_id, coincidence in self._external_coincidences.items():
            if point_id not in self._points:
                raise ValueError("外部重合关系引用了不存在的草图点")
            if coincidence.reference_id not in self._external_references:
                raise ValueError("外部重合关系引用了不存在的外部参考")
        if not self._unresolved_reference_ids.issubset(self._external_references):
            raise ValueError("未解析状态引用了不存在的外部参考")

    def _assert_new_id(self, entity_id: str) -> None:
        folded = entity_id.casefold()
        if any(existing.casefold() == folded for existing in (*self._points, *self._curves)):
            raise ValueError(f"草图实体 ID 已被占用：{entity_id}")

    def _next_id(
        self,
        prefix: str,
        values: dict[str, object],
        *,
        offset: int = 0,
    ) -> str:
        used = {key.casefold() for key in (*self._points, *self._curves, *values)}
        number = 1
        skipped = 0
        while True:
            candidate = f"{prefix}{number}"
            if candidate.casefold() not in used:
                if skipped < offset:
                    skipped += 1
                else:
                    return candidate
            number += 1

    def _require_point(self, point_id: str) -> SketchPoint:
        try:
            return self._points[point_id]
        except KeyError as error:
            raise KeyError(f"unknown sketch point: {point_id}") from error

    def _require_point_or_new(
        self,
        point_id: str,
        new_points: list[SketchPoint],
    ) -> SketchPoint:
        try:
            return self._points[point_id]
        except KeyError:
            for point in new_points:
                if point.id == point_id:
                    return point
            raise

    def _require_curve(self, curve_id: str) -> SketchCurve:
        try:
            return self._curves[curve_id]
        except KeyError as error:
            raise KeyError(f"unknown sketch curve: {curve_id}") from error

    def _assert_entity(self, entity_id: str) -> None:
        if entity_id not in self._points and entity_id not in self._curves:
            raise KeyError(entity_id)

    def _assert_selectable(self, entity_id: str) -> None:
        if entity_id in self._points or entity_id in self._curves:
            return
        if entity_id not in self._known_profile_ids:
            raise KeyError(entity_id)

    def _kind_for_id(self, entity_id: str) -> str:
        if entity_id in self._points:
            return "point"
        if entity_id in self._curves:
            return "edge"
        if entity_id in self._known_profile_ids:
            return "profile"
        raise KeyError(entity_id)

    def _clear_selection_id(self, entity_id: str) -> None:
        self._selected_ids = tuple(item for item in self._selected_ids if item != entity_id)
        if not self._selected_ids:
            self._selected_kind = None

    def _prune_selection(self) -> None:
        if self._selected_kind == "point":
            valid_ids = self._points
        elif self._selected_kind == "edge":
            valid_ids = self._curves
        else:
            valid_ids = {profile.id for profile in self.derive_profiles().profiles}
        self._selected_ids = tuple(
            entity_id for entity_id in self._selected_ids if entity_id in valid_ids
        )
        if not self._selected_ids:
            self._selected_kind = None

    def _coerce_endpoint(
        self,
        value: object,
        explicit_id: str | None,
        prefix: str,
        new_points: list[SketchPoint],
    ) -> str:
        if isinstance(value, str):
            self._require_point(value)
            return value
        coordinates = _coordinate_pair(value, "endpoint")
        point = SketchPoint(
            explicit_id or self._next_id(
                prefix,
                {**self._points, **{item.id: item for item in new_points}},
            ),
            *coordinates,
        )
        new_points.append(point)
        return point.id


def _curve_point_ids(curve: SketchCurve) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    return (curve.center_point_id,)


def _line_curve_intersections(
    start: tuple[float, float],
    end: tuple[float, float],
    curve: SketchCurve,
    points: dict[str, SketchPoint],
) -> list[tuple[float, tuple[float, float]]]:
    if isinstance(curve, SketchLine):
        other_start = points[curve.start_point_id]
        other_end = points[curve.end_point_id]
        other_start_xy = (other_start.u, other_start.v)
        other_end_xy = (other_end.u, other_end.v)
        line = (end[0] - start[0], end[1] - start[1])
        other_line = (
            other_end_xy[0] - other_start_xy[0],
            other_end_xy[1] - other_start_xy[1],
        )
        denominator = line[0] * other_line[1] - line[1] * other_line[0]
        offset = (
            other_start_xy[0] - start[0],
            other_start_xy[1] - start[1],
        )
        if abs(denominator) <= _TRIM_TOLERANCE:
            if abs(offset[0] * line[1] - offset[1] * line[0]) <= _TRIM_TOLERANCE:
                raise ValueError("trim cannot resolve overlapping line intersections")
            return []
        target_parameter = (
            offset[0] * other_line[1] - offset[1] * other_line[0]
        ) / denominator
        other_parameter = (offset[0] * line[1] - offset[1] * line[0]) / denominator
        if not (
            _TRIM_TOLERANCE < target_parameter < 1.0 - _TRIM_TOLERANCE
            and -_TRIM_TOLERANCE <= other_parameter <= 1.0 + _TRIM_TOLERANCE
        ):
            return []
        return [
            (
                target_parameter,
                (
                    start[0] + target_parameter * line[0],
                    start[1] + target_parameter * line[1],
                ),
            )
        ]
    if isinstance(curve, (SketchCircle, SketchArc)):
        center = points[curve.center_point_id]
        center_xy = (center.u, center.v)
        if isinstance(curve, SketchCircle):
            radius = curve.radius
        else:
            arc_start = points[curve.start_point_id]
            radius = math.hypot(arc_start.u - center.u, arc_start.v - center.v)
        direction = (end[0] - start[0], end[1] - start[1])
        relative = (start[0] - center_xy[0], start[1] - center_xy[1])
        quadratic_a = direction[0] ** 2 + direction[1] ** 2
        quadratic_b = 2.0 * (
            relative[0] * direction[0] + relative[1] * direction[1]
        )
        quadratic_c = relative[0] ** 2 + relative[1] ** 2 - radius**2
        discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
        if discriminant < -_TRIM_TOLERANCE:
            return []
        if abs(discriminant) <= _TRIM_TOLERANCE:
            raise ValueError("trim cannot resolve a tangent intersection")
        root = math.sqrt(discriminant)
        candidates: list[tuple[float, tuple[float, float]]] = []
        for target_parameter in (
            (-quadratic_b - root) / (2.0 * quadratic_a),
            (-quadratic_b + root) / (2.0 * quadratic_a),
        ):
            if not _TRIM_TOLERANCE < target_parameter < 1.0 - _TRIM_TOLERANCE:
                continue
            coordinate = (
                start[0] + target_parameter * direction[0],
                start[1] + target_parameter * direction[1],
            )
            if isinstance(curve, SketchArc) and not _point_on_arc(
                coordinate,
                curve,
                points,
            ):
                continue
            candidates.append((target_parameter, coordinate))
        return candidates
    raise TypeError(f"unsupported trim curve: {type(curve).__name__}")


def _point_on_arc(
    point: tuple[float, float],
    arc: SketchArc,
    points: dict[str, SketchPoint],
) -> bool:
    center = points[arc.center_point_id]
    start = points[arc.start_point_id]
    end = points[arc.end_point_id]
    radius = math.hypot(start.u - center.u, start.v - center.v)
    if abs(math.hypot(point[0] - center.u, point[1] - center.v) - radius) > 1.0e-7:
        return False
    start_angle = math.atan2(start.v - center.v, start.u - center.u)
    point_angle = math.atan2(point[1] - center.v, point[0] - center.u)
    end_angle = math.atan2(end.v - center.v, end.u - center.u)
    if arc.orientation == "ccw":
        travelled = (point_angle - start_angle) % (2.0 * math.pi)
        total = (end_angle - start_angle) % (2.0 * math.pi)
    else:
        travelled = (start_angle - point_angle) % (2.0 * math.pi)
        total = (start_angle - end_angle) % (2.0 * math.pi)
    return travelled <= total + 1.0e-7


def _line_parameter(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float | None:
    parameter = _line_projection_parameter(start, end, point)
    if parameter is None:
        return None
    direction = (end[0] - start[0], end[1] - start[1])
    projected = (
        start[0] + parameter * direction[0],
        start[1] + parameter * direction[1],
    )
    if math.hypot(point[0] - projected[0], point[1] - projected[1]) > _TRIM_TOLERANCE:
        return None
    if not -_TRIM_TOLERANCE <= parameter <= 1.0 + _TRIM_TOLERANCE:
        return None
    return min(1.0, max(0.0, parameter))


def _line_projection_parameter(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float | None:
    direction = (end[0] - start[0], end[1] - start[1])
    length_squared = direction[0] ** 2 + direction[1] ** 2
    if length_squared <= _TRIM_TOLERANCE**2:
        return None
    parameter = (
        (point[0] - start[0]) * direction[0]
        + (point[1] - start[1]) * direction[1]
    ) / length_squared
    return min(1.0, max(0.0, parameter))


def _unique_trim_intersections(
    intersections: list[tuple[float, tuple[float, float]]],
) -> list[tuple[float, tuple[float, float]]]:
    result: list[tuple[float, tuple[float, float]]] = []
    for item in sorted(intersections, key=lambda value: value[0]):
        if result and abs(item[0] - result[-1][0]) <= _TRIM_TOLERANCE:
            continue
        result.append(item)
    return result


def _find_point_at(
    points: dict[str, SketchPoint],
    coordinate: tuple[float, float],
) -> str | None:
    for point in points.values():
        if math.hypot(point.u - coordinate[0], point.v - coordinate[1]) <= _TRIM_TOLERANCE:
            return point.id
    return None


def _coordinate_pair(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a two-component coordinate")
    try:
        coordinates = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be a two-component coordinate") from error
    if len(coordinates) != 2:
        raise ValueError(f"{label} must contain exactly two components")
    normalized = tuple(float(item) for item in coordinates)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"{label} must contain finite coordinates")
    return normalized  # type: ignore[return-value]


def _circumcenter(
    first: tuple[float, float],
    through: tuple[float, float],
    last: tuple[float, float],
) -> tuple[float, float]:
    denominator = 2.0 * (
        first[0] * (through[1] - last[1])
        + through[0] * (last[1] - first[1])
        + last[0] * (first[1] - through[1])
    )
    if abs(denominator) <= 1.0e-12:
        raise ValueError("三点圆弧的三个点不能共线")
    first_square = first[0] * first[0] + first[1] * first[1]
    through_square = through[0] * through[0] + through[1] * through[1]
    last_square = last[0] * last[0] + last[1] * last[1]
    return (
        (
            first_square * (through[1] - last[1])
            + through_square * (last[1] - first[1])
            + last_square * (first[1] - through[1])
        )
        / denominator,
        (
            first_square * (last[0] - through[0])
            + through_square * (first[0] - last[0])
            + last_square * (through[0] - first[0])
        )
        / denominator,
    )


def _unique_diagnostics(
    diagnostics: list[SketchDiagnostic],
) -> tuple[SketchDiagnostic, ...]:
    result: list[SketchDiagnostic] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for diagnostic in diagnostics:
        key = diagnostic.code, diagnostic.entity_ids
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return tuple(result)


__all__ = [
    "SketchDraftController",
    "SketchDraftCurve",
    "SketchDraftDiagnostic",
    "SketchDraftPoint",
    "SketchDraftSnapshot",
    "SketchDraftValidationError",
]
