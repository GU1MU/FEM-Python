from __future__ import annotations

from collections import defaultdict
import math

import pytest

from fem import geometry
from fem.application import ModelSession
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    analyze_sketch_profiles,
)
from tests.helpers.fixtures.planar_construction_phase0 import (
    EXPECTED_PLATE_PROFILE_ROLES,
    H_SLOT_AREA,
    H_SLOT_BOUNDARY_LINE_COUNT,
)


Point2D = tuple[float, float]
Segment2D = tuple[Point2D, Point2D]


def _point_coordinates(gmsh, point_tag: int) -> Point2D:
    bounds = gmsh.model.occ.getBoundingBox(0, abs(point_tag))
    return tuple(
        round((float(bounds[axis]) + float(bounds[axis + 3])) * 0.5, 12)
        for axis in range(2)
    )


def _curve_endpoints(gmsh, signed_curve_tag: int) -> tuple[Point2D, ...]:
    raw = gmsh.model.getBoundary(
        [(1, signed_curve_tag)],
        combined=False,
        oriented=True,
        recursive=False,
    )
    return tuple(_point_coordinates(gmsh, point_tag) for _dim, point_tag in raw)


def _surface_curves(gmsh, surface_tag: int) -> tuple[int, ...]:
    gmsh.model.occ.synchronize()
    return tuple(
        int(signed_tag)
        for dimension, signed_tag in gmsh.model.getBoundary(
            [(2, surface_tag)],
            combined=False,
            oriented=True,
            recursive=False,
        )
        if int(dimension) == 1
    )


def _line_segments(gmsh, surface_tag: int) -> tuple[Segment2D, ...]:
    segments = []
    for signed_tag in _surface_curves(gmsh, surface_tag):
        endpoints = _curve_endpoints(gmsh, signed_tag)
        if len(endpoints) != 2:
            continue
        chord = math.dist(endpoints[0], endpoints[1])
        length = float(gmsh.model.occ.getMass(1, abs(signed_tag)))
        if not math.isclose(chord, length, rel_tol=1.0e-10, abs_tol=1.0e-10):
            continue
        segments.append((endpoints[0], endpoints[1]))
    return tuple(segments)


def _canonical_line_loop(segments: tuple[Segment2D, ...]) -> tuple[Point2D, ...]:
    neighbors: dict[Point2D, list[Point2D]] = defaultdict(list)
    for start, end in segments:
        neighbors[start].append(end)
        neighbors[end].append(start)
    assert neighbors and all(len(items) == 2 for items in neighbors.values())

    start = min(neighbors)
    result = [start]
    previous = None
    current = start
    while True:
        candidates = sorted(
            candidate for candidate in neighbors[current] if candidate != previous
        )
        following = candidates[0]
        if following == start:
            break
        result.append(following)
        previous, current = current, following
    area = 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(
            result,
            (*result[1:], result[0]),
            strict=True,
        )
    )
    if area < 0.0:
        result.reverse()
    first = result.index(min(result))
    return tuple((*result[first:], *result[:first]))


def _loop_count(gmsh, surface_tag: int) -> int:
    curves = _surface_curves(gmsh, surface_tag)
    endpoints = {
        signed_tag: _curve_endpoints(gmsh, signed_tag) for signed_tag in curves
    }
    point_to_curves: dict[Point2D, set[int]] = defaultdict(set)
    full_curves = 0
    for signed_tag, points in endpoints.items():
        if not points:
            full_curves += 1
        for point in points:
            point_to_curves[point].add(signed_tag)

    unseen = {tag for tag, points in endpoints.items() if points}
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            current = pending.pop()
            for point in endpoints[current]:
                adjacent = point_to_curves[point] & unseen
                unseen.difference_update(adjacent)
                pending.extend(adjacent)
    return components + full_curves


def _h_surface(cad):
    left = cad.rectangle(110.0, 20.0, 10.0, 60.0)
    cross = cad.rectangle(110.0, 45.0, 80.0, 10.0)
    right = cad.rectangle(180.0, 20.0, 10.0, 60.0)
    return cad.fuse((left,), (cross, right)).of_dimension(2)[0]


def _line_loop_sketch(
    name: str,
    loops: tuple[tuple[Point2D, ...], ...],
) -> SketchGeometry:
    points = []
    lines = []
    for loop_index, loop in enumerate(loops, start=1):
        point_ids = []
        for point_index, (x, y) in enumerate(loop, start=1):
            point_id = f"P{loop_index}_{point_index}"
            point_ids.append(point_id)
            points.append(SketchPoint(point_id, x, y))
        for line_index, point_id in enumerate(point_ids, start=1):
            lines.append(
                SketchLine(
                    f"L{loop_index}_{line_index}",
                    point_id,
                    point_ids[line_index % len(point_ids)],
                )
            )
    return SketchGeometry(name, SketchPlane.xy(), tuple(points), tuple(lines))


def _h_loop(real_gmsh, model_name: str) -> tuple[Point2D, ...]:
    with geometry.model(model_name, dimension=2) as cad:
        surface = _h_surface(cad)
        assert len(cad.entities(2)) == 1
        assert cad.area(surface) == pytest.approx(H_SLOT_AREA)
        assert len(cad.boundary((surface,))) == H_SLOT_BOUNDARY_LINE_COUNT
        segments = _line_segments(real_gmsh, surface.tag)
        assert len(segments) == H_SLOT_BOUNDARY_LINE_COUNT
        assert _loop_count(real_gmsh, surface.tag) == 1
        return _canonical_line_loop(segments)


@pytest.mark.gmsh
def test_phase0_h_boundary_materializes_and_recompiles_equivalently(
    real_gmsh,
) -> None:
    session = ModelSession()
    initial_snapshot = session.snapshot()
    first_loop = _h_loop(real_gmsh, "phase0-h-boundary-first")
    second_loop = _h_loop(real_gmsh, "phase0-h-boundary-second")
    assert first_loop == second_loop

    h_sketch = _line_loop_sketch("H region", (first_loop,))
    h_analysis = analyze_sketch_profiles(h_sketch)
    assert h_analysis.valid
    assert [profile.role for profile in h_analysis.profiles] == ["outer"]
    with geometry.model("phase0-h-recompile", dimension=2) as cad:
        compiled_h = compile_recipe(cad, h_sketch)
        assert len(compiled_h.domain) == 1
        assert cad.area(compiled_h.domain[0]) == pytest.approx(H_SLOT_AREA)
        assert len(cad.boundary(compiled_h.domain)) == H_SLOT_BOUNDARY_LINE_COUNT

    plate_loop = ((0.0, 0.0), (300.0, 0.0), (300.0, 100.0), (0.0, 100.0))
    plate_sketch = _line_loop_sketch("Plate with H hole", (plate_loop, first_loop))
    analysis = analyze_sketch_profiles(plate_sketch)
    assert analysis.valid
    assert tuple(profile.role for profile in analysis.profiles) == (
        EXPECTED_PLATE_PROFILE_ROLES
    )

    with geometry.model("phase0-h-cut-native", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 300.0, 100.0)
        h_surface = _h_surface(cad)
        native_cut = cad.cut((plate,), (h_surface,)).of_dimension(2)
        assert len(native_cut) == 1
        native_area = cad.area(native_cut[0])
        native_boundary_count = len(cad.boundary(native_cut))
        assert _loop_count(real_gmsh, native_cut[0].tag) == 2

    with geometry.model("phase0-h-cut-recompile", dimension=2) as cad:
        compiled = compile_recipe(cad, plate_sketch)
        assert len(compiled.domain) == 1
        assert cad.area(compiled.domain[0]) == pytest.approx(native_area)
        assert len(cad.boundary(compiled.domain)) == native_boundary_count == 16
        assert _loop_count(real_gmsh, compiled.domain[0].tag) == 2

    assert session.snapshot() == initial_snapshot


def _curve_parameters(gmsh, curve_tag: int):
    minimum, maximum = gmsh.model.getParametrizationBounds(1, curve_tag)
    start = float(minimum[0])
    end = float(maximum[0])
    values = tuple(
        tuple(float(value) for value in gmsh.model.getValue(1, curve_tag, [parameter]))
        for parameter in (start, (start + end) * 0.5, end)
    )
    return start, end, values


def _circle_center_from_samples(samples) -> Point2D:
    (x1, y1), (x2, y2), (x3, y3) = ((sample[0], sample[1]) for sample in samples)
    denominator = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    assert abs(denominator) > 1.0e-12
    first = x1 * x1 + y1 * y1
    second = x2 * x2 + y2 * y2
    third = x3 * x3 + y3 * y3
    return (
        (first * (y2 - y3) + second * (y3 - y1) + third * (y1 - y2)) / denominator,
        (first * (x3 - x2) + second * (x1 - x3) + third * (x2 - x1)) / denominator,
    )


@pytest.mark.gmsh
def test_phase0_circle_boolean_recovers_full_circle_arc_and_line_parameters(
    real_gmsh,
) -> None:
    with geometry.model("phase0-contained-circle", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 4.0, 3.0)
        disk = cad.disk(2.0, 1.5, 0.5)
        surface = cad.cut((plate,), (disk,)).of_dimension(2)[0]
        curves = _surface_curves(real_gmsh, surface.tag)
        types = [real_gmsh.model.getType(1, abs(tag)) for tag in curves]
        assert types.count("Line") == 4
        assert sum(kind in {"Circle", "Ellipse"} for kind in types) == 1
        full_circle = next(
            abs(tag)
            for tag in curves
            if real_gmsh.model.getType(1, abs(tag)) in {"Circle", "Ellipse"}
        )
        endpoints = _curve_endpoints(real_gmsh, full_circle)
        assert len(endpoints) == 2
        assert endpoints[0] == endpoints[1]
        start, end, samples = _curve_parameters(real_gmsh, full_circle)
        assert end - start == pytest.approx(2.0 * math.pi)
        assert samples[0] == pytest.approx(samples[2])
        assert cad.circle_center(cad.entity(1, full_circle)) == pytest.approx(
            (2.0, 1.5, 0.0)
        )
        assert cad.length(cad.entity(1, full_circle)) == pytest.approx(math.pi)

    with geometry.model("phase0-circle-arc", dimension=2) as cad:
        rectangle = cad.rectangle(0.0, 0.0, 3.0, 2.0)
        disk = cad.disk(3.0, 1.0, 0.6)
        surface = cad.fuse((rectangle,), (disk,)).of_dimension(2)[0]
        curves = _surface_curves(real_gmsh, surface.tag)
        assert len(_line_segments(real_gmsh, surface.tag)) == 5
        arcs = []
        for signed_tag in curves:
            curve_tag = abs(signed_tag)
            endpoints = _curve_endpoints(real_gmsh, curve_tag)
            if len(endpoints) != 2:
                continue
            if float(real_gmsh.model.occ.getMass(1, curve_tag)) > (
                math.dist(*endpoints) + 1.0e-10
            ):
                arcs.append(curve_tag)
        assert arcs
        assert sum(
            float(real_gmsh.model.occ.getMass(1, tag)) for tag in arcs
        ) == pytest.approx(math.pi * 0.6)
        for arc in arcs:
            endpoints = _curve_endpoints(real_gmsh, arc)
            assert endpoints[0] != endpoints[1]
            start, end, samples = _curve_parameters(real_gmsh, arc)
            assert 0.0 < end - start < 2.0 * math.pi
            assert samples[0][:2] == pytest.approx(endpoints[0])
            assert samples[2][:2] == pytest.approx(endpoints[1])
            center = _circle_center_from_samples(samples)
            assert center == pytest.approx((3.0, 1.0))
            assert all(
                math.dist(center, sample[:2]) == pytest.approx(0.6)
                for sample in samples
            )
