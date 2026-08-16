from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from typing import Any, Sequence

import numpy as np
import pytest

from fem import geometry
from fem.core import Mesh2D
from fem.geometry._gmsh import backend as _gmsh_backend
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


class _FakeOcc:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.calls: list[tuple[Any, ...]] = []
        self.synchronize_calls = 0
        self.fail_next: set[str] = set()
        self.loop_tags: list[int] = []
        self.wire_tags: list[int] = []
        self.distance_override: Sequence[float] | None = None

    def _data(self) -> dict[str, Any]:
        return self._model._current_data()

    def _allocate(self, dimension: int) -> int:
        data = self._data()
        tag = data["next_tags"].get(dimension, 1)
        data["next_tags"][dimension] = tag + 1
        data["entities"].add((dimension, tag))
        return tag

    def _curve(
        self,
        call: tuple[Any, ...],
        start_tag: int,
        end_tag: int,
        point_tags: Sequence[int],
    ) -> int:
        self.calls.append(call)
        tag = self._allocate(1)
        data = self._data()
        data["curve_endpoints"][tag] = (start_tag, end_tag)
        data["curve_points"][tag] = tuple(point_tags)
        point_boxes = [data["boxes"][(0, point_tag)] for point_tag in point_tags]
        data["boxes"][(1, tag)] = _union_boxes(point_boxes)
        coordinates = [_box_center(box) for box in point_boxes]
        data["masses"][(1, tag)] = sum(
            math.dist(first, second)
            for first, second in zip(coordinates, coordinates[1:])
        )
        data["centers"][(1, tag)] = tuple(
            sum(point[axis] for point in coordinates) / len(coordinates)
            for axis in range(3)
        )
        data["adjacencies"][(1, tag)] = ([], [start_tag, end_tag])
        for point_tag in {start_tag, end_tag}:
            upward, downward = data["adjacencies"].setdefault(
                (0, point_tag), ([], [])
            )
            upward.append(tag)
        return tag

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.calls.append(("synchronize", self._model.current))

    def getEntities(self, dimension: int = -1) -> list[tuple[int, int]]:
        return sorted(
            pair
            for pair in self._data()["entities"]
            if dimension == -1 or pair[0] == dimension
        )

    def getBoundingBox(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, ...]:
        self.calls.append(("getBoundingBox", dimension, tag))
        return self._data()["boxes"][(dimension, tag)]

    def getMass(self, dimension: int, tag: int) -> float:
        self.calls.append(("getMass", dimension, tag))
        return self._data()["masses"][(dimension, tag)]

    def getCenterOfMass(self, dimension: int, tag: int) -> tuple[float, ...]:
        self.calls.append(("getCenterOfMass", dimension, tag))
        return self._data()["centers"][(dimension, tag)]

    def getDistance(
        self,
        left_dimension: int,
        left_tag: int,
        right_dimension: int,
        right_tag: int,
    ) -> tuple[float, float, float, float, float, float, float]:
        self.calls.append(
            (
                "getDistance",
                left_dimension,
                left_tag,
                right_dimension,
                right_tag,
            )
        )
        if self.distance_override is not None:
            return tuple(self.distance_override)  # type: ignore[return-value]
        if left_dimension != 1 or right_dimension != 1:
            return (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        data = self._data()
        left_points = tuple(
            _box_center(data["boxes"][(0, point_tag)])
            for point_tag in data["curve_endpoints"][left_tag]
        )
        right_points = tuple(
            _box_center(data["boxes"][(0, point_tag)])
            for point_tag in data["curve_endpoints"][right_tag]
        )
        distance = _segment_distance_2d(
            left_points[0],
            left_points[1],
            right_points[0],
            right_points[1],
        )
        return (distance, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def addPoint(
        self,
        x: float,
        y: float,
        z: float,
        meshSize: float = 0.0,
        tag: int = -1,
    ) -> int:
        self.calls.append(("addPoint", x, y, z, meshSize, tag))
        allocated = self._allocate(0)
        data = self._data()
        data["boxes"][(0, allocated)] = (x, y, z, x, y, z)
        data["centers"][(0, allocated)] = (x, y, z)
        data["adjacencies"][(0, allocated)] = ([], [])
        return allocated

    def addLine(self, start_tag: int, end_tag: int, tag: int = -1) -> int:
        return self._curve(
            ("addLine", start_tag, end_tag, tag),
            start_tag,
            end_tag,
            (start_tag, end_tag),
        )

    def addCircleArc(
        self,
        start_tag: int,
        center_tag: int,
        end_tag: int,
        tag: int = -1,
        center: bool = True,
    ) -> int:
        curve_tag = self._curve(
            (
                "addCircleArc",
                start_tag,
                center_tag,
                end_tag,
                tag,
                center,
            ),
            start_tag,
            end_tag,
            (start_tag, center_tag, end_tag),
        )
        data = self._data()
        start = _box_center(data["boxes"][(0, start_tag)])
        center_point = _box_center(data["boxes"][(0, center_tag)])
        end = _box_center(data["boxes"][(0, end_tag)])
        first = (start[0] - center_point[0], start[1] - center_point[1])
        second = (end[0] - center_point[0], end[1] - center_point[1])
        angle = math.atan2(
            first[0] * second[1] - first[1] * second[0],
            first[0] * second[0] + first[1] * second[1],
        )
        if angle <= 0.0:
            angle += 2.0 * math.pi
        data["masses"][(1, curve_tag)] = math.hypot(*first) * angle
        return curve_tag

    def addEllipseArc(
        self,
        start_tag: int,
        center_tag: int,
        major_tag: int,
        end_tag: int,
        tag: int = -1,
    ) -> int:
        return self._curve(
            (
                "addEllipseArc",
                start_tag,
                center_tag,
                major_tag,
                end_tag,
                tag,
            ),
            start_tag,
            end_tag,
            (start_tag, center_tag, major_tag, end_tag),
        )

    def addSpline(
        self,
        point_tags: Sequence[int],
        tag: int = -1,
        tangents: Sequence[float] = (),
    ) -> int:
        materialized = tuple(point_tags)
        return self._curve(
            ("addSpline", materialized, tag, tuple(tangents)),
            materialized[0],
            materialized[-1],
            materialized,
        )

    def addBSpline(
        self,
        point_tags: Sequence[int],
        tag: int = -1,
        degree: int = 3,
        weights: Sequence[float] = (),
        knots: Sequence[float] = (),
        multiplicities: Sequence[int] = (),
    ) -> int:
        materialized = tuple(point_tags)
        return self._curve(
            (
                "addBSpline",
                materialized,
                tag,
                degree,
                tuple(weights),
                tuple(knots),
                tuple(multiplicities),
            ),
            materialized[0],
            materialized[-1],
            materialized,
        )

    def addCurveLoop(
        self,
        curve_tags: Sequence[int],
        tag: int = -1,
    ) -> int:
        materialized = tuple(curve_tags)
        self.calls.append(("addCurveLoop", materialized, tag))
        if self.loop_tags:
            allocated = self.loop_tags.pop(0)
        else:
            data = self._data()
            allocated = data["next_loop_tag"]
            data["next_loop_tag"] += 1
        if allocated > 0:
            self._data()["loops"][allocated] = materialized
        return allocated

    def addWire(
        self,
        curve_tags: Sequence[int],
        tag: int = -1,
        checkClosed: bool = False,
    ) -> int:
        materialized = tuple(curve_tags)
        self.calls.append(("addWire", materialized, tag, checkClosed))
        if self.wire_tags:
            allocated = self.wire_tags.pop(0)
        else:
            data = self._data()
            allocated = data["next_wire_tag"]
            data["next_wire_tag"] += 1
        if allocated > 0:
            self._data()["wires"][allocated] = (materialized, checkClosed)
        return allocated

    def addPlaneSurface(
        self,
        wire_tags: Sequence[int],
        tag: int = -1,
    ) -> int:
        materialized = tuple(wire_tags)
        self.calls.append(("addPlaneSurface", materialized, tag))
        allocated = self._allocate(2)
        data = self._data()
        curves = tuple(
            curve_tag
            for wire_tag in materialized
            for curve_tag in data["loops"][wire_tag]
        )
        data["surface_curves"][allocated] = curves
        data["boxes"][(2, allocated)] = _union_boxes(
            [data["boxes"][(1, curve_tag)] for curve_tag in curves]
        )
        data["masses"][(2, allocated)] = 1.0
        data["centers"][(2, allocated)] = _box_center(
            data["boxes"][(2, allocated)]
        )
        data["adjacencies"][(2, allocated)] = ([], list(curves))
        for curve_tag in curves:
            upward, downward = data["adjacencies"].setdefault(
                (1, curve_tag), ([], [])
            )
            upward.append(allocated)
        return allocated

    def translate(
        self,
        entities: Sequence[tuple[int, int]],
        dx: float,
        dy: float,
        dz: float,
    ) -> None:
        self.calls.append(("translate", tuple(entities), dx, dy, dz))
        self._maybe_fail("translate")

    def rotate(
        self,
        entities: Sequence[tuple[int, int]],
        x: float,
        y: float,
        z: float,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        angle: float,
    ) -> None:
        self.calls.append(
            (
                "rotate",
                tuple(entities),
                x,
                y,
                z,
                axis_x,
                axis_y,
                axis_z,
                angle,
            )
        )
        self._maybe_fail("rotate")

    def fuse(
        self,
        objects: Sequence[tuple[int, int]],
        tools: Sequence[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "fuse", objects, tools, tag, removeObject, removeTool
        )

    def cut(
        self,
        objects: Sequence[tuple[int, int]],
        tools: Sequence[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "cut", objects, tools, tag, removeObject, removeTool
        )

    def fragment(
        self,
        objects: Sequence[tuple[int, int]],
        tools: Sequence[tuple[int, int]],
        tag: int = -1,
        removeObject: bool = True,
        removeTool: bool = True,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        return self._boolean(
            "fragment", objects, tools, tag, removeObject, removeTool
        )

    def _boolean(
        self,
        name: str,
        objects: Sequence[tuple[int, int]],
        tools: Sequence[tuple[int, int]],
        tag: int,
        remove_objects: bool,
        remove_tools: bool,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        object_pairs = tuple(objects)
        tool_pairs = tuple(tools)
        self.calls.append(
            (
                name,
                object_pairs,
                tool_pairs,
                tag,
                remove_objects,
                remove_tools,
            )
        )
        self._maybe_fail(name)
        dimension = object_pairs[0][0]
        output = (dimension, self._allocate(dimension))
        data = self._data()
        source_boxes = [
            data["boxes"][pair]
            for pair in (*object_pairs, *tool_pairs)
            if pair in data["boxes"]
        ]
        if source_boxes:
            data["boxes"][output] = _union_boxes(source_boxes)
        if remove_objects:
            data["entities"].difference_update(object_pairs)
        if remove_tools:
            data["entities"].difference_update(tool_pairs)
        data["entities"].add(output)
        return [output], [[output] for _ in (*object_pairs, *tool_pairs)]

    def _maybe_fail(self, operation: str) -> None:
        if operation in self.fail_next:
            self.fail_next.remove(operation)
            raise RuntimeError(f"fake {operation} failure")


class _FakeModel:
    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.current = ""
        self.calls: list[tuple[Any, ...]] = []
        self.boundary_overrides: dict[
            tuple[int, int], list[tuple[int, int]]
        ] = {}
        self.occ = _FakeOcc(self)

    @staticmethod
    def _new_data() -> dict[str, Any]:
        return {
            "attributes": {},
            "entities": set(),
            "next_tags": {},
            "next_loop_tag": 1,
            "next_wire_tag": 1,
            "boxes": {},
            "masses": {},
            "centers": {},
            "curve_endpoints": {},
            "curve_points": {},
            "loops": {},
            "wires": {},
            "surface_curves": {},
            "adjacencies": {},
        }

    def _current_data(self) -> dict[str, Any]:
        return self.models[self.current]

    def list(self) -> list[str]:
        return list(self.models)

    def getCurrent(self) -> str:
        return self.current

    def setCurrent(self, name: str) -> None:
        if name not in self.models:
            raise RuntimeError(f"unknown model {name!r}")
        self.current = name

    def add(self, name: str) -> None:
        if name in self.models:
            raise RuntimeError(f"duplicate model {name!r}")
        self.models[name] = self._new_data()
        self.current = name

    def remove(self) -> None:
        del self.models[self.current]
        self.current = next(iter(self.models), "")

    def getAttribute(self, name: str) -> list[str]:
        return list(self._current_data()["attributes"].get(name, ()))

    def setAttribute(self, name: str, values: Sequence[str]) -> None:
        self._current_data()["attributes"][name] = [
            str(item) for item in values
        ]

    def getEntities(self, dimension: int = -1) -> list[tuple[int, int]]:
        return sorted(
            pair
            for pair in self._current_data()["entities"]
            if dimension == -1 or pair[0] == dimension
        )

    def getBoundary(
        self,
        entities: Sequence[tuple[int, int]],
        combined: bool = True,
        oriented: bool = True,
        recursive: bool = False,
    ) -> list[tuple[int, int]]:
        materialized = tuple(entities)
        self.calls.append(
            (
                "getBoundary",
                materialized,
                combined,
                oriented,
                recursive,
                self.current,
            )
        )
        data = self._current_data()
        boundary: list[tuple[int, int]] = []
        for dimension, signed_tag in materialized:
            tag = abs(signed_tag)
            override = self.boundary_overrides.get((dimension, tag))
            if override is not None:
                boundary.extend(override)
            elif dimension == 1:
                start, end = data["curve_endpoints"][tag]
                if oriented and signed_tag < 0:
                    start, end = end, start
                boundary.extend(((0, start), (0, end)))
            elif dimension == 2:
                boundary.extend(
                    (1, curve_tag)
                    for curve_tag in data["surface_curves"].get(tag, ())
                )
        return boundary

    def getBoundingBox(self, dimension: int, tag: int) -> tuple[float, ...]:
        self.calls.append(("getBoundingBox", dimension, tag, self.current))
        return self._current_data()["boxes"][(dimension, tag)]

    def getAdjacencies(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[list[int], list[int]]:
        self.calls.append(("getAdjacencies", dimension, tag, self.current))
        upward, downward = self._current_data()["adjacencies"].get(
            (dimension, tag), ([], [])
        )
        return list(upward), list(downward)

    def getParametrizationBounds(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[list[float], list[float]]:
        self.calls.append(("getParametrizationBounds", dimension, tag, self.current))
        if dimension != 1:
            raise RuntimeError("fake parametrization requires a curve")
        return [0.0], [1.0]

    def getValue(
        self,
        dimension: int,
        tag: int,
        parameters: Sequence[float],
    ) -> list[float]:
        self.calls.append(("getValue", dimension, tag, tuple(parameters), self.current))
        if dimension != 1:
            raise RuntimeError("fake evaluation requires a curve")
        data = self._current_data()
        coordinates = tuple(
            _box_center(data["boxes"][(0, point_tag)])
            for point_tag in data["curve_points"][tag]
        )
        result: list[float] = []
        for parameter in parameters:
            position = min(1.0, max(0.0, float(parameter))) * (
                len(coordinates) - 1
            )
            segment = min(int(math.floor(position)), len(coordinates) - 2)
            fraction = position - segment
            start = coordinates[segment]
            end = coordinates[segment + 1]
            result.extend(
                start[axis] + fraction * (end[axis] - start[axis])
                for axis in range(3)
            )
        return result


class _FakeGmsh:
    def __init__(self) -> None:
        self.initialized = True
        self.model = _FakeModel()

    def isInitialized(self) -> bool:
        return self.initialized

    def initialize(self) -> None:
        self.initialized = True

    def finalize(self) -> None:
        self.initialized = False


def _union_boxes(boxes: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(
        min(box[axis] for box in boxes) if axis < 3 else max(
            box[axis] for box in boxes
        )
        for axis in range(6)
    )


def _box_center(box: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        0.5 * (box[axis] + box[axis + 3]) for axis in range(3)
    )  # type: ignore[return-value]


def _segment_distance_2d(
    left_start: Sequence[float],
    left_end: Sequence[float],
    right_start: Sequence[float],
    right_end: Sequence[float],
) -> float:
    if _segments_intersect_2d(left_start, left_end, right_start, right_end):
        return 0.0
    return min(
        _point_segment_distance_2d(point, start, end)
        for point, start, end in (
            (left_start, right_start, right_end),
            (left_end, right_start, right_end),
            (right_start, left_start, left_end),
            (right_end, left_start, left_end),
        )
    )


def _segments_intersect_2d(
    left_start: Sequence[float],
    left_end: Sequence[float],
    right_start: Sequence[float],
    right_end: Sequence[float],
) -> bool:
    orientations = (
        _orientation_2d(left_start, left_end, right_start),
        _orientation_2d(left_start, left_end, right_end),
        _orientation_2d(right_start, right_end, left_start),
        _orientation_2d(right_start, right_end, left_end),
    )
    if orientations[0] * orientations[1] < 0.0 and (
        orientations[2] * orientations[3] < 0.0
    ):
        return True
    tolerance = 1.0e-12
    pairs = (
        (orientations[0], right_start, left_start, left_end),
        (orientations[1], right_end, left_start, left_end),
        (orientations[2], left_start, right_start, right_end),
        (orientations[3], left_end, right_start, right_end),
    )
    return any(
        abs(orientation) <= tolerance and _point_on_segment_2d(point, start, end)
        for orientation, point, start, end in pairs
    )


def _orientation_2d(
    start: Sequence[float],
    end: Sequence[float],
    point: Sequence[float],
) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])


def _point_on_segment_2d(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> bool:
    tolerance = 1.0e-12
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _point_segment_distance_2d(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    parameter = (
        (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ) / length_squared
    parameter = min(1.0, max(0.0, parameter))
    closest = (start[0] + parameter * delta_x, start[1] + parameter * delta_y)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


@pytest.fixture
def fake_gmsh(monkeypatch: pytest.MonkeyPatch) -> _FakeGmsh:
    backend = _FakeGmsh()
    monkeypatch.setattr(_gmsh_backend, "load_gmsh", lambda: backend)
    return backend


def _square(
    cad: geometry.GeometryModel,
    x: float,
    y: float,
    size: float,
    *,
    reverse: bool = False,
) -> tuple[
    tuple[geometry.EntityRef, ...],
    tuple[geometry.EntityRef, ...],
    geometry.CurveLoopRef,
]:
    points = (
        cad.point(x, y),
        cad.point(x + size, y),
        cad.point(x + size, y + size),
        cad.point(x, y + size),
    )
    curves = tuple(
        cad.line(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    if reverse:
        oriented = tuple(
            cad.orient(curve, reversed=True) for curve in reversed(curves)
        )
    else:
        oriented = tuple(cad.orient(curve) for curve in curves)
    return points, curves, cad.curve_loop(oriented)


def _count_calls(backend: _FakeGmsh, name: str) -> int:
    return sum(call[0] == name for call in backend.model.occ.calls)


@pytest.mark.parametrize(
    ("dimension", "z"),
    ((1, 0.4), (2, 0.0), (3, 0.4)),
)
def test_fake_lower_dimensional_points_and_lines_forward_in_all_models(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    z: float,
) -> None:
    with geometry.model(f"lower-{dimension}", dimension=dimension) as cad:
        start = cad.point(0.25, 0.5, z)
        end = cad.point(1.25, 1.5, z)
        curve = cad.line(start, end)

        assert (start.dimension, curve.dimension) == (0, 1)
        assert (
            "addPoint",
            0.25,
            0.5,
            z,
            0.0,
            -1,
        ) in fake_gmsh.model.occ.calls
        assert (
            "addLine",
            start.tag,
            end.tag,
            -1,
        ) in fake_gmsh.model.occ.calls


def test_fake_2d_point_rejects_nonplanar_z_before_native_mutation(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("planar-point", dimension=2) as cad:
        before = _count_calls(fake_gmsh, "addPoint")
        with pytest.raises(ValueError, match="global XY plane"):
            cad.point(0.0, 0.0, 0.25)
        assert _count_calls(fake_gmsh, "addPoint") == before


def test_fake_arc_and_spline_primitives_forward_native_arguments(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("primitive-forwarding", dimension=2) as cad:
        center = cad.point(0.0, 0.0)
        start = cad.point(1.0, 0.0)
        end = cad.point(0.0, 1.0)
        middle = cad.point(0.6, 0.7)

        circle = cad.circular_arc(start, center, end)
        ellipse = cad.elliptical_arc(start, center, start, end)
        spline = cad.spline((start, middle, end, start))
        bspline = cad.bspline((start, middle, end))

        assert {circle.dimension, ellipse.dimension, spline.dimension, bspline.dimension} == {
            1
        }
        calls = fake_gmsh.model.occ.calls
        assert (
            "addCircleArc",
            start.tag,
            center.tag,
            end.tag,
            -1,
            True,
        ) in calls
        assert (
            "addEllipseArc",
            start.tag,
            center.tag,
            start.tag,
            end.tag,
            -1,
        ) in calls
        assert (
            "addSpline",
            (start.tag, middle.tag, end.tag, start.tag),
            -1,
            (),
        ) in calls
        assert (
            "addBSpline",
            (start.tag, middle.tag, end.tag),
            -1,
            3,
            (),
            (),
            (),
        ) in calls


def test_fake_degenerate_curves_fail_before_curve_creation(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("primitive-validation", dimension=2) as cad:
        origin = cad.point(0.0, 0.0)
        same_origin = cad.point(0.0, 0.0)
        unit_x = cad.point(1.0, 0.0)
        unit_y = cad.point(0.0, 1.0)
        far_y = cad.point(0.0, 2.0)
        same_unit_x = cad.point(1.0, 0.0)

        with pytest.raises(ValueError, match="distinct coordinates"):
            cad.line(origin, same_origin)
        with pytest.raises(ValueError, match="equidistant"):
            cad.circular_arc(unit_x, origin, far_y)
        with pytest.raises(ValueError, match="radius must be positive"):
            cad.circular_arc(same_origin, origin, unit_y)
        with pytest.raises(ValueError, match="at least two"):
            cad.spline((origin,))
        with pytest.raises(ValueError, match="duplicate-free"):
            cad.spline((origin, unit_x, origin, unit_y))
        with pytest.raises(ValueError, match="distinct coordinates"):
            cad.bspline((origin, unit_x, same_unit_x))

        curve_names = {
            "addLine",
            "addCircleArc",
            "addEllipseArc",
            "addSpline",
            "addBSpline",
        }
        assert not any(call[0] in curve_names for call in fake_gmsh.model.occ.calls)


@pytest.mark.parametrize(
    ("start", "major", "end", "message"),
    (
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            "elliptical_arc center and major_axis_point must be distinct",
        ),
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            "elliptical_arc endpoints must differ from center",
        ),
        (
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "elliptical_arc start and end must be distinct",
        ),
    ),
)
def test_fake_elliptical_arc_caller_prechecks_preserve_validation_order(
    fake_gmsh: _FakeGmsh,
    start: tuple[float, float, float],
    major: tuple[float, float, float],
    end: tuple[float, float, float],
    message: str,
) -> None:
    with geometry.model("ellipse-caller-precheck", dimension=2) as cad:
        center_point = cad.point(0.0, 0.0, 0.0)
        start_point = cad.point(*start)
        major_point = cad.point(*major)
        end_point = cad.point(*end)
        before = _count_calls(fake_gmsh, "addEllipseArc")

        with pytest.raises(ValueError) as captured:
            cad.elliptical_arc(
                start_point,
                center_point,
                major_point,
                end_point,
            )

        assert str(captured.value) == message
        assert _count_calls(fake_gmsh, "addEllipseArc") == before


@pytest.mark.parametrize(
    ("dimension", "start", "major", "end", "message"),
    (
        (2, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "major axis"),
        (
            3,
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 2.0),
            (-1.0, 0.0, -1.0),
            "major axis",
        ),
        (
            2,
            (math.sqrt(3.0), 0.5, 0.0),
            (2.0, 0.0, 0.0),
            (math.sqrt(3.0), -0.5, 0.0),
            "unique major and minor radii",
        ),
        (
            2,
            (3.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            "positive finite radii",
        ),
    ),
)
def test_fake_elliptical_arc_degeneracies_fail_before_occ_mutation(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    start: tuple[float, float, float],
    major: tuple[float, float, float],
    end: tuple[float, float, float],
    message: str,
) -> None:
    with geometry.model(f"ellipse-degenerate-{dimension}", dimension=dimension) as cad:
        center_point = cad.point(0.0, 0.0, 0.0)
        start_point = cad.point(*start)
        major_point = cad.point(*major)
        end_point = cad.point(*end)
        before = _count_calls(fake_gmsh, "addEllipseArc")

        with pytest.raises(ValueError, match=message):
            cad.elliptical_arc(
                start_point,
                center_point,
                major_point,
                end_point,
            )

        assert _count_calls(fake_gmsh, "addEllipseArc") == before


def test_fake_foreign_and_raw_invalidated_profile_references_fail_closed(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("profile-owner", dimension=2) as outer:
        points, curves, loop = _square(outer, 0.0, 0.0, 1.0)
        with geometry.model("profile-foreign", dimension=2) as inner:
            before_surface = _count_calls(fake_gmsh, "addPlaneSurface")
            with pytest.raises(geometry.EntityOwnershipError):
                inner.orient(curves[0])
            with pytest.raises(geometry.EntityOwnershipError):
                inner.plane_surface(loop)
            assert _count_calls(fake_gmsh, "addPlaneSurface") == before_surface

        outer.raw_occ
        with pytest.raises(geometry.StaleEntityError):
            outer.line(points[0], points[1])
        with pytest.raises(geometry.StaleEntityError):
            outer.plane_surface(loop)


def test_fake_missing_native_curve_invalidates_its_dependent_loop(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("profile-liveness", dimension=2) as cad:
        _, curves, loop = _square(cad, 0.0, 0.0, 1.0)
        fake_gmsh.model._current_data()["entities"].remove((1, curves[0].tag))

        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            cad.length(curves[0])
        with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
            cad.plane_surface(loop)


def test_fake_reversed_loop_uses_signed_traversal_but_positive_occ_tags(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("reversed-loop", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
        )
        curves = tuple(
            cad.line(points[index], points[(index + 1) % 4])
            for index in range(4)
        )
        oriented = tuple(
            cad.orient(curve, reversed=True) for curve in reversed(curves)
        )
        loop = cad.curve_loop(oriented)

        add_call = next(
            call
            for call in reversed(fake_gmsh.model.occ.calls)
            if call[0] == "addCurveLoop"
        )
        assert add_call[1] == tuple(curve.tag for curve in reversed(curves))
        assert all(tag > 0 for tag in add_call[1])
        boundary_inputs = [
            call[1][0][1]
            for call in fake_gmsh.model.calls
            if call[0] == "getBoundary" and call[2:5] == (False, True, False)
        ]
        assert boundary_inputs[-4:] == [-curve.tag for curve in reversed(curves)]
        assert loop.curves == oriented
        with pytest.raises(FrozenInstanceError):
            loop.tag = loop.tag + 1  # type: ignore[misc]
        with pytest.raises(TypeError):
            loop.curves[0] = loop.curves[0]  # type: ignore[index]


def test_fake_wire_builds_open_and_closed_frozen_slotted_references(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-shape", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(2.0, 0.5),
        )
        curves = (
            cad.line(points[0], points[1]),
            cad.line(points[1], points[2]),
        )
        oriented = tuple(cad.orient(curve) for curve in curves)

        path = cad.wire(oriented, closed=False)
        _, square_curves, _ = _square(cad, 3.0, 0.0, 1.0)
        section_curves = tuple(cad.orient(curve) for curve in square_curves)
        section = cad.wire(section_curves, closed=True)

        assert isinstance(path, geometry.WireRef)
        assert (path.tag, path.curves, path.closed) == (1, oriented, False)
        assert (section.tag, section.curves, section.closed) == (
            2,
            section_curves,
            True,
        )
        assert not hasattr(path, "__dict__")
        assert (
            "addWire",
            tuple(curve.tag for curve in curves),
            -1,
            False,
        ) in fake_gmsh.model.occ.calls
        assert (
            "addWire",
            tuple(curve.tag for curve in square_curves),
            -1,
            True,
        ) in fake_gmsh.model.occ.calls
        with pytest.raises(FrozenInstanceError):
            path.tag = path.tag + 1  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            path.closed = True  # type: ignore[misc]
        with pytest.raises(TypeError):
            path.curves[0] = path.curves[0]  # type: ignore[index]


def test_fake_wire_preserves_signed_reversal_and_preflights_invalid_chains(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-validation", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(2.0, 0.0),
        )
        first = cad.line(points[0], points[1])
        backward_second = cad.line(points[2], points[1])
        oriented = (
            cad.orient(first),
            cad.orient(backward_second, reversed=True),
        )

        path = cad.wire(oriented, closed=False)

        assert path.curves == oriented
        assert (
            "addWire",
            (first.tag, -backward_second.tag),
            -1,
            False,
        ) in fake_gmsh.model.occ.calls
        before = _count_calls(fake_gmsh, "addWire")
        with pytest.raises(ValueError, match="continuous"):
            cad.wire(
                (cad.orient(first), cad.orient(backward_second)),
                closed=False,
            )
        with pytest.raises(ValueError, match="close|closed"):
            cad.wire((cad.orient(first),), closed=True)
        with pytest.raises(ValueError, match="duplicate-free"):
            cad.wire(
                (cad.orient(first), cad.orient(first, reversed=True)),
                closed=False,
            )
        with pytest.raises(ValueError, match="at least one"):
            cad.wire((), closed=False)
        with pytest.raises(TypeError, match="OrientedCurveRef"):
            cad.wire((first,), closed=False)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="boolean"):
            cad.wire((cad.orient(first),), closed=1)  # type: ignore[arg-type]

        _, closed_curves, _ = _square(cad, 3.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="open|distinct"):
            cad.wire(
                tuple(cad.orient(curve) for curve in closed_curves),
                closed=False,
            )
        assert _count_calls(fake_gmsh, "addWire") == before


def test_fake_wire_native_failure_is_contextual_and_fail_closed(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with geometry.model("wire-native-failure", dimension=3) as cad:
        first_start = cad.point(0.0, 0.0, 0.0)
        first_end = cad.point(1.0, 0.0, 0.0)
        first_curve = cad.line(first_start, first_end)
        existing = cad.wire((cad.orient(first_curve),), closed=False)
        second_start = cad.point(0.0, 1.0, 0.0)
        second_end = cad.point(1.0, 1.0, 0.0)
        second_curve = cad.line(second_start, second_end)

        def fail_add_wire(*args: Any, **kwargs: Any) -> int:
            raise RuntimeError("injected addWire failure")

        monkeypatch.setattr(fake_gmsh.model.occ, "addWire", fail_add_wire)
        with pytest.raises(geometry.GeometryError, match="native OCC wire") as caught:
            cad.wire((cad.orient(second_curve),), closed=False)

        assert isinstance(caught.value.__cause__, RuntimeError)
        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad._normalize_wires((existing,), operation="test")
        with pytest.raises(geometry.StaleEntityError, match="stale"):
            cad.length(first_curve)


def test_fake_wire_rejects_spatial_2d_chain_and_accepts_it_in_3d(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-spatial-2d", dimension=2) as cad:
        start = cad.point(0.0, 0.0)
        end = cad.point(1.0, 0.0)
        curve = cad.line(start, end)
        fake_gmsh.model._current_data()["boxes"][(1, curve.tag)] = (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.25,
        )
        before = _count_calls(fake_gmsh, "addWire")

        with pytest.raises(ValueError, match="global XY plane"):
            cad.wire((cad.orient(curve),), closed=False)

        assert _count_calls(fake_gmsh, "addWire") == before

    with geometry.model("wire-spatial-3d", dimension=3) as cad:
        points = (
            cad.point(0.0, 0.0, 0.0),
            cad.point(1.0, 0.0, 0.5),
            cad.point(1.5, 0.5, 1.0),
        )
        curves = (
            cad.line(points[0], points[1]),
            cad.line(points[1], points[2]),
        )

        path = cad.wire(tuple(cad.orient(curve) for curve in curves), closed=False)

        assert path.closed is False
        assert path.curves == tuple(cad.orient(curve) for curve in curves)


def test_fake_wire_rejects_foreign_and_missing_native_members_before_occ(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-owner", dimension=3) as outer:
        start = outer.point(0.0, 0.0, 0.0)
        end = outer.point(1.0, 0.0, 0.0)
        curve = outer.line(start, end)
        oriented = outer.orient(curve)
        wire = outer.wire((oriented,), closed=False)

        with geometry.model("wire-foreign", dimension=3) as inner:
            before = _count_calls(fake_gmsh, "addWire")
            with pytest.raises(geometry.EntityOwnershipError):
                inner.wire((oriented,), closed=False)
            assert _count_calls(fake_gmsh, "addWire") == before

        fake_gmsh.model._current_data()["entities"].remove((1, curve.tag))
        before = _count_calls(fake_gmsh, "addWire")
        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            outer.wire((oriented,), closed=False)

        assert _count_calls(fake_gmsh, "addWire") == before
        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            outer._normalize_wires((wire,), operation="test")


def test_fake_raw_access_invalidates_every_typed_wire_identity(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-raw-invalidation", dimension=3) as cad:
        first_start = cad.point(0.0, 0.0, 0.0)
        first_end = cad.point(1.0, 0.0, 0.0)
        second_start = cad.point(0.0, 1.0, 0.0)
        second_end = cad.point(1.0, 1.0, 0.0)
        first_curve = cad.line(first_start, first_end)
        second_curve = cad.line(second_start, second_end)
        first = cad.wire((cad.orient(first_curve),), closed=False)
        second = cad.wire((cad.orient(second_curve),), closed=False)
        assert cad._normalize_wires((first, second), operation="test") == (
            first,
            second,
        )

        cad.raw_occ

        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad._normalize_wires((first,), operation="test")
        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad._normalize_wires((second,), operation="test")


def test_fake_cleanup_failure_still_invalidates_entity_loop_and_wire_identity(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cad = geometry.model("cleanup-reference-invalidation", dimension=2)
    cad.__enter__()
    _, curves, loop = _square(cad, 0.0, 0.0, 1.0)
    wire = cad.wire(
        tuple(cad.orient(curve) for curve in curves),
        closed=True,
    )
    original_remove = fake_gmsh.model.remove

    def fail_remove() -> None:
        raise RuntimeError("injected model removal failure")

    monkeypatch.setattr(fake_gmsh.model, "remove", fail_remove)
    with pytest.raises(geometry.GeometryError, match="remove facade model"):
        cad.__exit__(None, None, None)

    with pytest.raises(geometry.StaleEntityError, match="stale entity"):
        cad._normalize_entities((curves[0],), operation="test")
    with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
        cad._normalize_curve_loops((loop,), operation="test")
    with pytest.raises(geometry.StaleEntityError, match="stale wire"):
        cad._normalize_wires((wire,), operation="test")

    monkeypatch.setattr(fake_gmsh.model, "remove", original_remove)
    cad.__exit__(None, None, None)


def test_fake_member_mutation_invalidates_only_dependent_wire(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-targeted-invalidation", dimension=3) as cad:
        points = tuple(cad.point(float(index), 0.0, 0.0) for index in range(6))
        first_curves = (
            cad.line(points[0], points[1]),
            cad.line(points[1], points[2]),
        )
        second_curves = (
            cad.line(points[3], points[4]),
            cad.line(points[4], points[5]),
        )
        unrelated_start = cad.point(0.0, 2.0, 0.0)
        unrelated_end = cad.point(1.0, 2.0, 0.0)
        unrelated = cad.line(unrelated_start, unrelated_end)
        first = cad.wire(
            tuple(cad.orient(curve) for curve in first_curves),
            closed=False,
        )
        second = cad.wire(
            tuple(cad.orient(curve) for curve in second_curves),
            closed=False,
        )

        cad.translate((unrelated,), 0.0, 0.0, 0.5)
        assert cad._normalize_wires((first, second), operation="test") == (
            first,
            second,
        )

        cad.translate((first_curves[0],), 0.0, 0.0, 0.5)

        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad._normalize_wires((first,), operation="test")
        assert cad._normalize_wires((second,), operation="test") == (second,)


def test_fake_wire_and_curve_loop_can_share_native_tag_without_collision(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-loop-tag-collision", dimension=2) as cad:
        _, curves, loop = _square(cad, 0.0, 0.0, 1.0)
        wire = cad.wire(
            tuple(cad.orient(curve) for curve in curves),
            closed=True,
        )

        assert wire.tag == loop.tag == 1
        assert loop._loop_token is not wire._wire_token
        assert cad._normalize_curve_loops((loop,), operation="test") == (loop,)
        assert cad._normalize_wires((wire,), operation="test") == (wire,)
        assert cad.plane_surface(loop).dimension == 2

        cad.translate((curves[0],), 0.25, 0.0, 0.0)

        with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
            cad._normalize_curve_loops((loop,), operation="test")
        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad._normalize_wires((wire,), operation="test")


def test_fake_wire_rejects_malformed_and_duplicate_native_identity(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("wire-nonpositive", dimension=3) as cad:
        fake_gmsh.model.occ.wire_tags[:] = [0]
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(1.0, 0.0, 0.0)
        curve = cad.line(start, end)
        with pytest.raises(geometry.GeometryError, match="invalid wire tag"):
            cad.wire((cad.orient(curve),), closed=False)

    with geometry.model("wire-duplicate", dimension=3) as cad:
        fake_gmsh.model.occ.wire_tags[:] = [23, 23]
        points = (
            cad.point(0.0, 0.0, 0.0),
            cad.point(1.0, 0.0, 0.0),
            cad.point(0.0, 1.0, 0.0),
            cad.point(1.0, 1.0, 0.0),
        )
        first_curve = cad.line(points[0], points[1])
        second_curve = cad.line(points[2], points[3])
        first = cad.wire((cad.orient(first_curve),), closed=False)

        with pytest.raises(geometry.GeometryError, match="duplicate wire tag"):
            cad.wire((cad.orient(second_curve),), closed=False)

        with pytest.raises(geometry.StaleEntityError, match="stale wire"):
            cad._normalize_wires((first,), operation="test")


def test_fake_curve_loop_validates_continuity_closure_and_duplicates_first(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("loop-validation", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
        )
        curves = tuple(
            cad.line(points[index], points[(index + 1) % 4])
            for index in range(4)
        )
        before = _count_calls(fake_gmsh, "addCurveLoop")

        with pytest.raises(ValueError, match="continuous.*close"):
            cad.curve_loop(tuple(cad.orient(curve) for curve in curves[:3]))
        with pytest.raises(ValueError, match="continuous.*close"):
            cad.curve_loop(
                (
                    cad.orient(curves[0]),
                    cad.orient(curves[2]),
                    cad.orient(curves[1]),
                    cad.orient(curves[3]),
                )
            )
        with pytest.raises(ValueError, match="duplicate-free"):
            cad.curve_loop(
                tuple(cad.orient(curve) for curve in (*curves, curves[0]))
            )
        with pytest.raises(TypeError, match="OrientedCurveRef"):
            cad.curve_loop(curves)  # type: ignore[arg-type]

        assert _count_calls(fake_gmsh, "addCurveLoop") == before


@pytest.mark.parametrize("builder_name", ("spline", "bspline"))
def test_fake_periodic_point_curve_forms_one_curve_loop_and_surface(
    fake_gmsh: _FakeGmsh,
    builder_name: str,
) -> None:
    with geometry.model(f"periodic-{builder_name}", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
        )
        builder = getattr(cad, builder_name)
        curve = builder((*points, points[0]))
        fake_gmsh.model.boundary_overrides[(1, curve.tag)] = [
            (0, points[0].tag),
            (0, points[0].tag),
        ]

        loop = cad.curve_loop((cad.orient(curve),))
        surface = cad.plane_surface(loop)

        assert loop.curves == (cad.orient(curve),)
        assert surface.dimension == 2
        assert (
            "addCurveLoop",
            (curve.tag,),
            -1,
        ) in fake_gmsh.model.occ.calls


def test_fake_curve_loop_rejects_malformed_and_duplicate_native_identity(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("malformed-loop", dimension=2) as cad:
        _, curves, _ = _square(cad, 0.0, 0.0, 1.0)
        fake_gmsh.model.boundary_overrides[(1, curves[0].tag)] = [
            (0, curves[0].tag)
        ]
        with pytest.raises(geometry.GeometryError, match="two ordered endpoints"):
            cad.curve_loop(tuple(cad.orient(curve) for curve in curves))
    fake_gmsh.model.boundary_overrides.clear()

    with geometry.model("nonpositive-loop", dimension=2) as cad:
        fake_gmsh.model.occ.loop_tags[:] = [0]
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
        )
        curves = tuple(
            cad.line(points[index], points[(index + 1) % 4])
            for index in range(4)
        )
        with pytest.raises(geometry.GeometryError, match="invalid loop tag"):
            cad.curve_loop(tuple(cad.orient(curve) for curve in curves))

    with geometry.model("duplicate-loop", dimension=2) as cad:
        fake_gmsh.model.occ.loop_tags[:] = [23, 23]
        _, _, first = _square(cad, 0.0, 0.0, 1.0)
        with pytest.raises(geometry.GeometryError, match="duplicate loop tag"):
            _square(cad, 2.0, 0.0, 1.0)
        with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
            cad.plane_surface(first)


def test_fake_plane_surface_forwards_holes_and_rejects_ambiguous_loops(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("surface-loops", dimension=2) as cad:
        outer_points, outer_curves, outer = _square(cad, 0.0, 0.0, 2.0)
        _, _, hole = _square(cad, 0.5, 0.5, 0.5, reverse=True)
        surface = cad.plane_surface(outer, holes=(hole,))

        assert surface.dimension == 2
        assert (
            "addPlaneSurface",
            (outer.tag, hole.tag),
            -1,
        ) in fake_gmsh.model.occ.calls
        before = _count_calls(fake_gmsh, "addPlaneSurface")
        with pytest.raises(ValueError, match="duplicate-free"):
            cad.plane_surface(outer, holes=(outer,))

        lower_left = cad.point(0.0, -1.0)
        lower_right = cad.point(2.0, -1.0)
        shared_curves = (
            outer_curves[0],
            cad.line(outer_points[1], lower_right),
            cad.line(lower_right, lower_left),
            cad.line(lower_left, outer_points[0]),
        )
        shared = cad.curve_loop(
            (
                cad.orient(shared_curves[0]),
                cad.orient(shared_curves[1]),
                cad.orient(shared_curves[2]),
                cad.orient(shared_curves[3]),
            )
        )
        with pytest.raises(ValueError, match="share member curves"):
            cad.plane_surface(outer, holes=(shared,))
        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


def test_fake_plane_surface_rejects_distinct_curves_sharing_a_vertex(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("surface-shared-vertex", dimension=2) as cad:
        outer_points, _, outer = _square(cad, 0.0, 0.0, 2.0)
        inner_points = (
            outer_points[0],
            cad.point(0.6, 0.2),
            cad.point(0.2, 0.6),
        )
        inner_curves = tuple(
            cad.line(inner_points[index], inner_points[(index + 1) % 3])
            for index in range(3)
        )
        inner = cad.curve_loop(tuple(cad.orient(curve) for curve in inner_curves))
        before = _count_calls(fake_gmsh, "addPlaneSurface")

        with pytest.raises(ValueError, match="boundary points"):
            cad.plane_surface(outer, holes=(inner,))

        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


@pytest.mark.parametrize(
    ("x", "y"),
    ((1.75, 0.5), (0.5, 0.0)),
)
def test_fake_plane_surface_rejects_crossing_or_tangent_distinct_boundaries(
    fake_gmsh: _FakeGmsh,
    x: float,
    y: float,
) -> None:
    with geometry.model("surface-boundary-contact", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 2.0)
        _, _, inner = _square(cad, x, y, 0.5)
        before = _count_calls(fake_gmsh, "addPlaneSurface")

        with pytest.raises(ValueError, match="touch or intersect"):
            cad.plane_surface(outer, holes=(inner,))

        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


def test_fake_plane_surface_rejects_outside_and_nested_holes_before_occ(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("surface-hole-containment", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 5.0)
        _, _, outside = _square(cad, 6.0, 1.0, 0.5)
        before = _count_calls(fake_gmsh, "addPlaneSurface")
        with pytest.raises(ValueError, match="strictly inside"):
            cad.plane_surface(outer, holes=(outside,))
        assert _count_calls(fake_gmsh, "addPlaneSurface") == before

        _, _, enclosing_hole = _square(cad, 1.0, 1.0, 2.5)
        _, _, nested_hole = _square(cad, 1.5, 1.5, 0.5)
        with pytest.raises(ValueError, match="disjoint enclosed regions"):
            cad.plane_surface(outer, holes=(enclosing_hole, nested_hole))
        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


def test_fake_plane_surface_accepts_two_disjoint_contained_holes(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("surface-disjoint-holes", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 5.0)
        _, _, first_hole = _square(cad, 0.75, 0.75, 0.5)
        _, _, second_hole = _square(cad, 3.5, 3.5, 0.5)

        surface = cad.plane_surface(outer, holes=(first_hole, second_hole))

        assert surface.dimension == 2


def test_fake_plane_surface_rejects_a_self_intersecting_periodic_spline(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("surface-self-intersection", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
            cad.point(1.0, 0.0),
        )
        curve = cad.spline((*points, points[0]))
        loop = cad.curve_loop((cad.orient(curve),))
        before = _count_calls(fake_gmsh, "addPlaneSurface")

        with pytest.raises(ValueError, match="self-intersect"):
            cad.plane_surface(loop)

        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


@pytest.mark.parametrize(
    "distance_result",
    ((-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (float("nan"),) * 7, (0.5,)),
)
def test_fake_plane_surface_fails_closed_on_invalid_distance_results(
    fake_gmsh: _FakeGmsh,
    distance_result: tuple[float, ...],
) -> None:
    with geometry.model("surface-invalid-distance", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 2.0)
        fake_gmsh.model.occ.distance_override = distance_result
        before = _count_calls(fake_gmsh, "addPlaneSurface")

        with pytest.raises(
            geometry.GeometryError,
            match="curve-(?:distance|loop boundary)",
        ):
            cad.plane_surface(outer)

        assert _count_calls(fake_gmsh, "addPlaneSurface") == before


def test_curve_loop_distance_wraps_occ_handle_activation_failure(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_gmsh
    with geometry.model("surface-distance-activation", dimension=2) as cad:
        first = cad.line(cad.point(0.0, 0.0), cad.point(1.0, 0.0))
        second = cad.line(cad.point(0.0, 1.0), cad.point(1.0, 1.0))

        def fail_activate(session: Any, operation: str) -> Any:
            del session, operation
            raise RuntimeError("fake distance activation failure")

        monkeypatch.setattr(type(cad._session), "activate", fail_activate)
        with pytest.raises(
            geometry.GeometryError,
            match="curve-loop boundary separation",
        ) as captured:
            cad._occ_curve_distance(
                first,
                second,
                operation="plane_surface",
            )

        assert isinstance(captured.value.__cause__, RuntimeError)


def test_fake_transform_and_boolean_failures_preserve_loops_until_success(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("profile-invalidation", dimension=2) as cad:
        _, curves, loop = _square(cad, 0.0, 0.0, 1.0)
        fake_gmsh.model.occ.fail_next.add("translate")
        with pytest.raises(RuntimeError, match="translate"):
            cad.translate((curves[0],), 0.25, 0.0, 0.0)
        cad.plane_surface(loop)

        cad.translate((curves[0],), 0.25, 0.0, 0.0)
        with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
            cad.plane_surface(loop)

        _, boolean_curves, boolean_loop = _square(cad, 2.0, 0.0, 1.0)
        tool_start = cad.point(3.5, 0.0)
        tool_end = cad.point(4.0, 0.0)
        tool = cad.line(tool_start, tool_end)
        fake_gmsh.model.occ.fail_next.add("cut")
        with pytest.raises(RuntimeError, match="cut"):
            cad.cut((boolean_curves[0],), (tool,))
        cad.plane_surface(boolean_loop)

        cad.cut((boolean_curves[0],), (tool,))
        with pytest.raises(geometry.StaleEntityError, match="stale curve loop"):
            cad.plane_surface(boolean_loop)


def test_fake_geometry_queries_forward_validate_and_sort_adjacencies(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("profile-queries", dimension=2) as cad:
        points, curves, loop = _square(cad, 0.0, 0.0, 1.0)
        surface = cad.plane_surface(loop)
        data = fake_gmsh.model._current_data()
        data["boxes"][(1, curves[0].tag)] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        data["masses"][(1, curves[0].tag)] = 1.25
        data["masses"][(2, surface.tag)] = 0.75
        data["centers"][(1, curves[0].tag)] = (0.4, 0.0, 0.0)
        data["adjacencies"][(1, curves[0].tag)] = (
            [surface.tag, surface.tag],
            [points[1].tag, points[0].tag, points[1].tag],
        )

        assert cad.bounding_box(curves[0]) == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert cad.length(curves[0]) == pytest.approx(1.25)
        assert cad.area(surface) == pytest.approx(0.75)
        assert cad.center_of_mass(curves[0]) == pytest.approx((0.4, 0.0, 0.0))
        assert cad.center_of_mass(points[0]) == pytest.approx((0.0, 0.0, 0.0))
        assert cad.adjacent(curves[0], dimension=2) == (surface,)
        assert cad.adjacent(curves[0], dimension=0) == tuple(
            sorted((points[0], points[1]), key=lambda item: item.tag)
        )

        data["boxes"][(1, curves[0].tag)] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(geometry.GeometryError, match="inverted bounding box"):
            cad.bounding_box(curves[0])
        data["masses"][(1, curves[0].tag)] = -1.0
        with pytest.raises(ValueError, match=">= 0"):
            cad.length(curves[0])
        data["centers"][(1, curves[0].tag)] = (0.0, 0.0)
        with pytest.raises(geometry.GeometryError, match="invalid center of mass"):
            cad.center_of_mass(curves[0])
        data["adjacencies"][(1, curves[0].tag)] = ([surface.tag + 1000], [])
        with pytest.raises(geometry.GeometryError, match="adjacent returned missing"):
            cad.adjacent(curves[0], dimension=2)
        with pytest.raises(ValueError, match="differ.*one"):
            cad.adjacent(curves[0], dimension=1)


def test_fake_mesher_binding_seals_mutation_but_preserves_geometry_queries(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("bound-profile", dimension=2) as cad:
        _, _, loop = _square(cad, 0.0, 0.0, 1.0)
        surface = cad.plane_surface(loop)
        fake_gmsh.model._current_data()["masses"][(2, surface.tag)] = 1.0

        gmsh_meshing.Mesher(cad)

        assert cad.area(surface) == pytest.approx(1.0)
        assert cad.bounding_box(surface) == pytest.approx(
            (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        )
        with pytest.raises(geometry.GeometryStateError):
            cad.point(2.0, 2.0)
        with pytest.raises(geometry.GeometryStateError):
            cad.raw_occ


@pytest.mark.parametrize(("dimension", "z"), ((2, 0.0), (3, 0.75)))
def test_real_lower_dimensional_construction_in_2d_and_3d(
    real_gmsh: Any,
    dimension: int,
    z: float,
) -> None:
    with geometry.model(f"real-lower-{dimension}", dimension=dimension) as cad:
        start = cad.point(0.0, 0.0, z)
        end = cad.point(2.0, 0.0, z)
        curve = cad.line(start, end)

        assert curve.dimension == 1
        assert cad.length(curve) == pytest.approx(2.0, rel=1.0e-10)
        assert cad.bounding_box(curve) == pytest.approx(
            (0.0, 0.0, z, 2.0, 0.0, z), abs=2.0e-6
        )
        assert cad.center_of_mass(curve) == pytest.approx(
            (1.0, 0.0, z), abs=1.0e-10
        )
        assert set(cad.adjacent(curve, dimension=0)) == {start, end}


def test_real_arc_spline_and_bspline_properties(real_gmsh: Any) -> None:
    with geometry.model("real-profile-curves", dimension=2) as cad:
        center = cad.point(0.0, 0.0)
        start = cad.point(1.0, 0.0)
        end = cad.point(0.0, 1.0)
        circle = cad.circular_arc(start, center, end)
        ellipse_start = cad.point(2.0, 0.0)
        ellipse = cad.elliptical_arc(
            ellipse_start, center, ellipse_start, end
        )
        spline_points = (
            cad.point(0.0, 2.0),
            cad.point(0.5, 2.6),
            cad.point(1.0, 2.0),
        )
        bspline_points = (
            cad.point(0.0, 3.0),
            cad.point(0.5, 3.7),
            cad.point(1.0, 3.0),
        )
        spline = cad.spline(spline_points)
        bspline = cad.bspline(bspline_points)

        assert cad.length(circle) == pytest.approx(math.pi / 2.0, rel=1.0e-9)
        assert math.pi / 2.0 < cad.length(ellipse) < math.pi
        assert cad.length(spline) > 1.0
        assert cad.length(bspline) > 1.0
        assert cad.bounding_box(circle) == pytest.approx(
            (0.0, 0.0, 0.0, 1.0, 1.0, 0.0), abs=2.0e-6
        )
        assert np.all(np.isfinite(cad.center_of_mass(circle)))
        assert set(cad.adjacent(circle, dimension=0)) == {start, end}


def test_real_elliptical_arc_preflight_rejects_axis_degeneracy_and_allows_tilt(
    real_gmsh: Any,
) -> None:
    with geometry.model("real-ellipse-axis-degenerate", dimension=2) as cad:
        center = cad.point(0.0, 0.0)
        start = cad.point(1.0, 0.0)
        major = cad.point(2.0, 0.0)
        end = cad.point(-1.0, 0.0)
        before = cad.entities(1)

        with pytest.raises(ValueError, match="major axis"):
            cad.elliptical_arc(start, center, major, end)

        assert cad.entities(1) == before

    with geometry.model("real-ellipse-tilted", dimension=3) as cad:
        center = cad.point(0.0, 0.0, 0.0)
        start = cad.point(1.0, 0.0, 1.0)
        major = cad.point(2.0, 0.0, 2.0)
        end = cad.point(0.0, 1.0, 0.0)
        curve = cad.elliptical_arc(start, center, major, end)

        assert cad.length(curve) > 0.0


@pytest.mark.parametrize("builder_name", ("spline", "bspline"))
def test_real_periodic_point_curve_forms_one_curve_planar_surface(
    real_gmsh: Any,
    builder_name: str,
) -> None:
    with geometry.model(f"real-periodic-{builder_name}", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
        )
        builder = getattr(cad, builder_name)
        curve = builder((*points, points[0]))
        loop = cad.curve_loop((cad.orient(curve),))
        surface = cad.plane_surface(loop)

        assert cad.length(curve) > 2.5
        assert cad.area(surface) > 0.5
        assert np.all(np.isfinite(cad.center_of_mass(surface)))
        assert cad.adjacent(surface, dimension=1) == (curve,)


def test_real_reversed_typed_hole_subtracts_area(real_gmsh: Any) -> None:
    with geometry.model("real-reversed-hole", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 2.0)
        _, _, hole = _square(cad, 0.75, 0.75, 0.5, reverse=True)
        surface = cad.plane_surface(outer, holes=(hole,))

        assert cad.area(surface) == pytest.approx(3.75, rel=1.0e-10)
        assert cad.bounding_box(surface) == pytest.approx(
            (0.0, 0.0, 0.0, 2.0, 2.0, 0.0), abs=2.0e-6
        )
        assert cad.center_of_mass(surface) == pytest.approx(
            (1.0, 1.0, 0.0), abs=1.0e-10
        )
        assert len(cad.adjacent(surface, dimension=1)) == 8


def test_real_plane_surface_preflight_rejects_boundary_overlap_and_bad_holes(
    real_gmsh: Any,
) -> None:
    with geometry.model("real-crossing-loop", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 2.0)
        _, _, crossing = _square(cad, 1.75, 0.5, 0.5)
        before = cad.entities(2)

        with pytest.raises(ValueError, match="touch or intersect"):
            cad.plane_surface(outer, holes=(crossing,))

        assert cad.entities(2) == before

    with geometry.model("real-invalid-hole-containment", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 5.0)
        _, _, outside = _square(cad, 6.0, 1.0, 0.5)
        with pytest.raises(ValueError, match="strictly inside"):
            cad.plane_surface(outer, holes=(outside,))

        _, _, enclosing_hole = _square(cad, 1.0, 1.0, 2.5)
        _, _, nested_hole = _square(cad, 1.5, 1.5, 0.5)
        with pytest.raises(ValueError, match="disjoint enclosed regions"):
            cad.plane_surface(outer, holes=(enclosing_hole, nested_hole))

        assert cad.entities(2) == ()


def test_real_plane_surface_accepts_multiple_disjoint_contained_holes(
    real_gmsh: Any,
) -> None:
    with geometry.model("real-disjoint-holes", dimension=2) as cad:
        _, _, outer = _square(cad, 0.0, 0.0, 5.0)
        _, _, first_hole = _square(cad, 0.75, 0.75, 0.5)
        _, _, second_hole = _square(cad, 3.5, 3.5, 0.5)

        surface = cad.plane_surface(outer, holes=(first_hole, second_hole))

        assert cad.area(surface) == pytest.approx(24.5, rel=1.0e-12)


def test_real_plane_surface_rejects_a_self_intersecting_periodic_spline(
    real_gmsh: Any,
) -> None:
    with geometry.model("real-self-intersecting-loop", dimension=2) as cad:
        points = (
            cad.point(0.0, 0.0),
            cad.point(1.0, 1.0),
            cad.point(0.0, 1.0),
            cad.point(1.0, 0.0),
        )
        curve = cad.spline((*points, points[0]))
        loop = cad.curve_loop((cad.orient(curve),))

        with pytest.raises(ValueError, match="self-intersect"):
            cad.plane_surface(loop)

        assert cad.entities(2) == ()


def _irregular_surface(
    cad: geometry.GeometryModel,
) -> geometry.EntityRef:
    lower_left = cad.point(0.0, 0.0)
    lower_right = cad.point(2.0, 0.0)
    arc_start = cad.point(2.0, 1.0)
    arc_center = cad.point(1.5, 1.0)
    arc_end = cad.point(1.5, 1.5)
    upper_left = cad.point(0.0, 1.5)
    outer_curves = (
        cad.line(lower_left, lower_right),
        cad.line(lower_right, arc_start),
        cad.circular_arc(arc_start, arc_center, arc_end),
        cad.line(arc_end, upper_left),
        cad.line(upper_left, lower_left),
    )
    outer = cad.curve_loop(tuple(cad.orient(curve) for curve in outer_curves))
    _, _, hole = _square(cad, 0.5, 0.5, 0.3, reverse=True)
    return cad.plane_surface(outer, holes=(hole,))


def _assert_positive_jacobians(gmsh: Any, dimension: int) -> None:
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dimension)
    checked = 0
    for element_type, tags in zip(element_types, element_tags, strict=True):
        if len(tags) == 0:
            continue
        local_coordinates, weights = gmsh.model.mesh.getIntegrationPoints(
            element_type, "Gauss2"
        )
        _, determinants, _ = gmsh.model.mesh.getJacobians(
            element_type, local_coordinates
        )
        values = np.asarray(determinants, dtype=float)
        assert values.size == len(tags) * len(weights)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        checked += len(tags)
    assert checked > 0


@pytest.mark.parametrize(("order", "expected_type"), ((1, "Tri3"), (2, "Tri6")))
def test_real_irregular_curved_profile_with_hole_generates_supported_fem_mesh(
    real_gmsh: Any,
    order: int,
    expected_type: str,
) -> None:
    with geometry.model(f"real-irregular-{order}", dimension=2) as cad:
        surface = _irregular_surface(cad)
        expected_area = 2.75 + math.pi / 16.0 - 0.09
        assert cad.area(surface) == pytest.approx(expected_area, rel=1.0e-10)

        mesher = gmsh_meshing.Mesher(cad)
        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(size=0.2, order=order, recombine=False)
        )
        mesh = gmsh_io.read(native_mesh)
        _assert_positive_jacobians(real_gmsh, 2)

    assert isinstance(mesh, Mesh2D)
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert {element.type for element in mesh.elements} == {expected_type}
