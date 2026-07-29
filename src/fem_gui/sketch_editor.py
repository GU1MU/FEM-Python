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
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)


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

    @property
    def selected_entity_ids(self) -> tuple[str, ...]:
        return self.selected_ids

    @property
    def dirty(self) -> bool:
        return self.revision > 0


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
        self._undo: list[SketchDraftSnapshot] = []
        self._redo: list[SketchDraftSnapshot] = []
        self._history_limit = history_limit

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
        self._undo.clear()
        self._redo.clear()

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

        def apply() -> None:
            self._assert_new_id(point.id)
            self._points[point.id] = point

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
            self._clear_selection_id(point_id)

        self._mutate(apply)

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
        circle = SketchCircle(circle_name, center_name, radius_value)

        def apply() -> None:
            self._assert_new_id(point.id)
            self._assert_new_id(circle.id)
            self._points[point.id] = point
            self._curves[circle.id] = circle

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

        def apply() -> None:
            for point in (*new_points, center_point):
                self._assert_new_id(point.id)
            self._assert_new_id(arc.id)
            self._points.update({point.id: point for point in (*new_points, center_point)})
            self._curves[arc.id] = arc

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

        self._mutate(apply)
        return tuple(replacements)

    trim = trim_curve
    trim_segment = trim_curve

    def select(self, entity_id: str, *, kind: str | None = None) -> tuple[str, ...]:
        if entity_id not in self._points and entity_id not in self._curves:
            analysis = self.derive_profiles()
            if entity_id not in {profile.id for profile in analysis.profiles}:
                raise KeyError(entity_id)
        normalized_kind = kind or self._kind_for_id(entity_id)

        def apply() -> None:
            self._selected_ids = (entity_id,)
            self._selected_kind = normalized_kind

        self._mutate(apply)
        return self._selected_ids

    select_entity = select

    def select_point(self, point_id: str) -> tuple[str, ...]:
        return self.select(point_id, kind="point")

    def select_curve(self, curve_id: str) -> tuple[str, ...]:
        return self.select(curve_id, kind="edge")

    def select_profile(self, profile_id: str) -> tuple[str, ...]:
        return self.select(profile_id, kind="profile")

    def select_many(self, entity_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        ids = tuple(entity_ids)
        if not ids:
            return self.clear_selection()
        kinds = {self._kind_for_id(item) for item in ids}
        if len(kinds) != 1:
            raise ValueError("同一种实体类型才能多选")
        for item in ids:
            self.select(item)
        # ``select`` intentionally records each single selection.  Collapse
        # the gesture into one final state for callers that use multi-select.
        self._selected_ids = ids
        self._selected_kind = next(iter(kinds))
        return ids

    def clear_selection(self) -> tuple[str, ...]:
        def apply() -> None:
            self._selected_ids = ()
            self._selected_kind = None

        self._mutate(apply)
        return ()

    def begin_pending_command(self, command: str) -> None:
        normalized = str(command).strip()
        if not normalized:
            raise ValueError("pending command cannot be empty")

        def apply() -> None:
            self._pending_command = normalized

        self._mutate(apply)

    begin_command = begin_pending_command

    def cancel_pending_command(self) -> None:
        if self._pending_command is None:
            return

        def apply() -> None:
            self._pending_command = None

        self._mutate(apply)

    cancel_command = cancel_pending_command

    def complete_pending_command(self) -> str | None:
        command = self._pending_command
        self.cancel_pending_command()
        return command

    complete_command = complete_pending_command

    def undo(self) -> SketchDraftSnapshot:
        if not self._undo:
            return self.snapshot()
        current = self.snapshot()
        previous = self._undo.pop()
        self._redo.append(current)
        self._restore_state(previous)
        return self.snapshot()

    def redo(self) -> SketchDraftSnapshot:
        if not self._redo:
            return self.snapshot()
        current = self.snapshot()
        following = self._redo.pop()
        self._undo.append(current)
        self._restore_state(following)
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
            return SketchProfileAnalysis(
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
        return analyze_sketch_profiles(sketch)

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
        before = self.snapshot()
        try:
            result = operation()
        except BaseException:
            self._restore_state(before)
            raise
        after = self.snapshot()
        if after == before:
            return result
        self._undo.append(before)
        if len(self._undo) > self._history_limit:
            del self._undo[0]
        self._redo.clear()
        self._revision += 1
        return result

    def _restore_state(self, snapshot: SketchDraftSnapshot) -> None:
        self._name = snapshot.name
        self._plane = snapshot.plane
        self._points = {point.id: point for point in snapshot.points}
        self._curves = {curve.id: curve for curve in snapshot.curves}
        self._selected_ids = tuple(snapshot.selected_ids)
        self._selected_kind = snapshot.selected_kind
        self._pending_command = snapshot.pending_command
        self._revision = snapshot.revision

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

    def _kind_for_id(self, entity_id: str) -> str:
        if entity_id in self._points:
            return "point"
        if entity_id in self._curves:
            return "edge"
        return "face"

    def _clear_selection_id(self, entity_id: str) -> None:
        self._selected_ids = tuple(item for item in self._selected_ids if item != entity_id)
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
