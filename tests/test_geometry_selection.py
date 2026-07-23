from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal, get_type_hints

import pytest

from fem import geometry, selection
from fem.selection import (
    curves,
    edges,
    elements,
    faces,
    nodes,
    points,
    surfaces,
    volumes,
)
from tests.helpers.gmsh_fake import _FakeGmsh, _install_backend


@pytest.fixture
def fake_gmsh(monkeypatch: pytest.MonkeyPatch) -> _FakeGmsh:
    backend = _FakeGmsh()
    _install_backend(monkeypatch, backend)
    return backend


class _SinglePassEntities:
    def __init__(self, values: tuple[geometry.EntityRef, ...]) -> None:
        self.values = values
        self.iterations = 0

    def __iter__(self) -> Iterator[geometry.EntityRef]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("candidate iterable was consumed more than once")
        yield from self.values


def _representative_entities(
    cad: geometry.GeometryModel,
) -> dict[int, geometry.EntityRef]:
    first = cad.point(0.0, 0.0, 0.0)
    second = cad.point(1.0, 0.0, 0.0)
    return {
        0: first,
        1: cad.line(first, second),
        2: cad.rectangle(2.0, 0.0, 2.0, 3.0),
        3: cad.box(5.0, 0.0, 0.0, 2.0, 3.0, 4.0),
    }


def _select_points(
    cad: geometry.GeometryModel,
    candidates: Any,
) -> tuple[geometry.EntityRef, ...]:
    return points.in_box(cad, candidates, xmin=-100.0)


def _select_curves(
    cad: geometry.GeometryModel,
    candidates: Any,
) -> tuple[geometry.EntityRef, ...]:
    return curves.in_box(cad, candidates, xmin=-100.0)


def _select_surfaces(
    cad: geometry.GeometryModel,
    candidates: Any,
) -> tuple[geometry.EntityRef, ...]:
    return surfaces.in_box(cad, candidates, xmin=-100.0)


def _select_volumes(
    cad: geometry.GeometryModel,
    candidates: Any,
) -> tuple[geometry.EntityRef, ...]:
    return volumes.in_box(cad, candidates, xmin=-100.0)


_CANDIDATE_SELECTORS: tuple[
    tuple[
        int,
        Callable[
            [geometry.GeometryModel, Any],
            tuple[geometry.EntityRef, ...],
        ],
    ],
    ...,
] = (
    (0, _select_points),
    (1, _select_curves),
    (2, _select_surfaces),
    (3, _select_volumes),
)


def _shared_surface_topology(
    cad: geometry.GeometryModel,
    backend: _FakeGmsh,
) -> dict[str, geometry.EntityRef]:
    first = cad.point(0.0, 0.0)
    shared_start = cad.point(1.0, 0.0)
    shared_end = cad.point(1.0, 1.0)
    last = cad.point(2.0, 1.0)
    left_curve = cad.line(first, shared_start)
    shared_curve = cad.line(shared_start, shared_end)
    right_curve = cad.line(shared_end, last)
    left_surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
    right_surface = cad.rectangle(1.0, 0.0, 1.0, 1.0)

    data = backend.model._current_data()
    configured = {
        (2, left_surface.tag): [
            (1, left_curve.tag),
            (1, shared_curve.tag),
        ],
        (2, right_surface.tag): [
            (1, shared_curve.tag),
            (1, right_curve.tag),
        ],
    }
    data["boundaries"].update(configured)
    data["boundary_priority"].update(configured)
    return {
        "first": first,
        "shared_start": shared_start,
        "shared_end": shared_end,
        "last": last,
        "left_curve": left_curve,
        "shared_curve": shared_curve,
        "right_curve": right_curve,
        "left_surface": left_surface,
        "right_surface": right_surface,
    }


def test_selection_exports_cad_modules_without_reusing_mesh_names() -> None:
    expected = [
        "curves",
        "edges",
        "elements",
        "faces",
        "nodes",
        "points",
        "surfaces",
        "volumes",
    ]
    modules = (curves, edges, elements, faces, nodes, points, surfaces, volumes)

    assert selection.__all__ == expected
    assert tuple(getattr(selection, name) for name in expected) == modules
    assert faces is not surfaces
    assert faces.all is not surfaces.all
    assert not hasattr(selection, "lines")
    assert not hasattr(selection, "by_center")
    assert "by_coord" not in volumes.__all__


@pytest.mark.parametrize(
    "module",
    (points, curves, surfaces, volumes),
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_public_adjacency_modes_use_literal_annotations(module: Any) -> None:
    assert get_type_hints(module.adjacent_to)["mode"] == Literal["any", "all"]


def test_all_and_omitted_candidates_use_live_dimension_entities(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("selection-all", dimension=3) as cad:
        isolated = cad.point(99.0, 0.0, 0.0)
        start = cad.point(0.0, 0.0, 0.0)
        end = cad.point(1.25, 0.0, 0.0)
        curve = cad.line(start, end)
        surface = cad.rectangle(0.0, 4.0, 7.0, 1.0)
        volume = cad.box(10.0, 0.0, 0.0, 2.0, 3.0, 4.0)

        assert points.all(cad) == cad.entities(0)
        assert curves.all(cad) == cad.entities(1)
        assert surfaces.all(cad) == cad.entities(2)
        assert volumes.all(cad) == cad.entities(3)
        assert points.by_x(cad, 99.0) == (isolated,)
        assert curves.by_length(cad, value=1.25) == (curve,)
        assert surfaces.by_area(cad, value=7.0) == (surface,)
        assert volumes.by_volume(cad, value=24.0) == (volume,)
        assert all(
            isinstance(result, tuple)
            for result in (
                points.all(cad),
                curves.all(cad),
                surfaces.all(cad),
                volumes.all(cad),
            )
        )


def test_explicit_candidates_empty_candidates_and_generator_order_are_stable(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("selection-candidates", dimension=2) as cad:
        first = cad.line(cad.point(0.0, 0.0), cad.point(0.0, 1.0))
        second = cad.line(cad.point(0.0, 2.0), cad.point(0.0, 3.0))
        candidates = _SinglePassEntities((second, first, second, first))

        assert curves.by_x(cad, 0.0, candidates) == (second, first)
        assert candidates.iterations == 1
        assert curves.by_x(cad, 0.0, ()) == ()

        first_surface = cad.rectangle(2.0, 0.0, 2.0, 2.0)
        second_surface = cad.rectangle(5.0, 0.0, 2.0, 2.0)
        assert surfaces.by_area(
            cad,
            (second_surface,),
            value=4.0,
        ) == (second_surface,)
        assert surfaces.by_area(cad, (), value=4.0) == ()
        assert first_surface in surfaces.by_area(cad, value=4.0)


@pytest.mark.parametrize(
    ("dimension", "selector"),
    _CANDIDATE_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_candidate_validation_is_shared_across_cad_dimensions(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    selector: Callable[
        [geometry.GeometryModel, Any],
        tuple[geometry.EntityRef, ...],
    ],
) -> None:
    del fake_gmsh
    with geometry.model(f"candidate-validation-{dimension}", dimension=3) as cad:
        representatives = _representative_entities(cad)
        later_representatives = _representative_entities(cad)
        target = representatives[dimension]
        later = later_representatives[dimension]
        wrong = representatives[(dimension + 1) % 4]

        assert selector(cad, (later, target, later, target)) == (later, target)
        with pytest.raises(TypeError, match="EntityRef"):
            selector(cad, (object(),))
        with pytest.raises(TypeError, match="iterable"):
            selector(cad, 42)
        with pytest.raises(ValueError, match="dimension"):
            selector(cad, (wrong,))
        with pytest.raises(ValueError, match="dimension"):
            selector(cad, (target, wrong))
        with pytest.raises(TypeError, match="EntityRef"):
            selector(cad, (target, target, object()))
        with pytest.raises(ValueError, match="dimension"):
            selector(cad, (target, target, wrong))


def test_coordinate_selection_matches_complete_entities_and_axis_wrappers(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("coordinate-selection", dimension=3) as cad:
        vertical = cad.line(cad.point(0.0, 0.0), cad.point(0.0, 1.0))
        crossing = cad.line(cad.point(-1.0, 0.5), cad.point(1.0, 0.5))
        axis_line = cad.line(
            cad.point(0.0, 2.0, 0.0),
            cad.point(0.0, 2.0, 1.0),
        )

        assert curves.by_x(
            cad,
            0.0,
            (axis_line, crossing, vertical),
        ) == (axis_line, vertical)
        assert curves.by_coord(
            cad,
            (vertical, axis_line),
            x=0.0,
            y=2.0,
        ) == (axis_line,)
        assert curves.by_y(cad, 0.5, (crossing, vertical)) == (crossing,)
        assert curves.by_z(cad, 0.0, (axis_line, vertical)) == (vertical,)

        exact = cad.point(2.0, 3.0, 4.0)
        near = cad.point(2.0 + 5.0e-9, 3.0, 4.0)
        assert points.by_coord(
            cad,
            (near, exact),
            x=2.0,
            y=3.0,
            tolerance=1.0e-8,
        ) == (near, exact)
        assert points.by_z(cad, 4.0, (exact, near)) == (exact, near)

        base = cad.rectangle(4.0, 0.0, 1.0, 1.0, z=0.0)
        raised = cad.rectangle(4.0, 0.0, 1.0, 1.0, z=1.0)
        assert surfaces.by_z(cad, 1.0, (base, raised)) == (raised,)


def test_center_selection_uses_all_supplied_coordinates_and_tolerance(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("center-selection", dimension=3) as cad:
        first_disk = cad.disk(1.0, 1.0, 0.25)
        other_y_disk = cad.disk(1.0, 3.0, 0.25)
        near_disk = cad.disk(1.0 + 5.0e-9, 1.0, 0.25)
        first_curve = cad.boundary((first_disk,))[0]
        other_y_curve = cad.boundary((other_y_disk,))[0]
        near_curve = cad.boundary((near_disk,))[0]

        assert curves.by_center(
            cad,
            (other_y_curve, near_curve, first_curve),
            x=1.0,
            y=1.0,
        ) == (near_curve, first_curve)
        assert curves.by_center(
            cad,
            (near_curve, first_curve),
            x=1.0,
            y=1.0,
            tolerance=0.0,
        ) == (first_curve,)

        surface = cad.rectangle(4.0, 6.0, 2.0, 4.0, z=1.0)
        volume = cad.box(10.0, 20.0, 30.0, 2.0, 4.0, 6.0)
        assert surfaces.by_center(cad, (surface,), x=5.0, y=8.0, z=1.0) == (
            surface,
        )
        assert volumes.by_center(cad, (volume,), x=11.0, y=22.0, z=33.0) == (
            volume,
        )


def test_box_selection_compensates_occ_padding_and_requires_containment(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("box-selection", dimension=2) as cad:
        inside = cad.line(cad.point(0.0, 0.0), cad.point(1.0, 1.0))
        intersecting = cad.line(
            cad.point(-0.25, 0.5),
            cad.point(0.25, 0.5),
        )
        data = fake_gmsh.model._current_data()
        data["boxes"][(1, inside.tag)] = (
            -1.0e-7,
            -1.0e-7,
            -1.0e-7,
            1.0 + 1.0e-7,
            1.0 + 1.0e-7,
            1.0e-7,
        )

        assert curves.in_box(
            cad,
            (intersecting, inside),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=0.0,
            tolerance=0.0,
        ) == (inside,)
        assert curves.in_box(
            cad,
            (intersecting,),
            xmax=0.5,
            ymin=0.0,
            ymax=1.0,
        ) == (intersecting,)

        point = cad.point(1.0, 1.0)
        assert points.in_box(cad, (point,), xmin=1.0, ymax=1.0) == (point,)


def test_nearest_point_preserves_first_tie_and_optional_z_semantics(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("nearest-point", dimension=3) as cad:
        high = cad.point(0.0, 0.0, 10.0)
        low = cad.point(0.0, 0.0, 0.0)
        left = cad.point(-1.0, 0.0, 0.0)
        right = cad.point(1.0, 0.0, 0.0)

        assert points.nearest(cad, 0.0, 0.0, entities=(high, low)) == high
        assert points.nearest(cad, 0.0, 0.0, 0.0, entities=(high, low)) == low
        assert (
            points.nearest(
                cad,
                0.0,
                0.0,
                entities=(right, left, right, left),
            )
            == right
        )
        assert points.nearest(cad, 0.0, 0.0, entities=()) is None


def test_nearest_point_comparison_does_not_overflow_for_large_coordinates(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("nearest-point-large", dimension=3) as cad:
        farther = cad.point(2.0e200, 0.0, 0.0)
        nearer = cad.point(1.0e200, 0.0, 0.0)

        assert points.nearest(
            cad,
            0.0,
            0.0,
            entities=(farther, nearer),
        ) == nearer


def test_measure_selection_and_volume_query_are_dimension_specific(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("measure-selection", dimension=3) as cad:
        first_curve = cad.line(cad.point(0.0, 0.0), cad.point(2.0, 0.0))
        second_curve = cad.line(cad.point(0.0, 1.0), cad.point(2.0, 1.0))
        near_curve = cad.line(cad.point(0.0, 2.0), cad.point(2.005, 2.0))
        first_surface = cad.rectangle(0.0, 4.0, 2.0, 3.0)
        second_surface = cad.rectangle(4.0, 4.0, 3.0, 2.0)
        first_volume = cad.box(0.0, 10.0, 0.0, 2.0, 3.0, 4.0)
        second_volume = cad.box(5.0, 10.0, 0.0, 4.0, 3.0, 2.0)

        assert curves.by_length(
            cad,
            (second_curve, near_curve, first_curve),
            value=2.0,
            tolerance=0.01,
        ) == (second_curve, near_curve, first_curve)
        assert surfaces.by_area(
            cad,
            (second_surface, first_surface),
            value=6.0,
        ) == (second_surface, first_surface)
        assert volumes.by_volume(
            cad,
            (second_volume, first_volume),
            value=24.0,
        ) == (second_volume, first_volume)
        assert cad.volume(first_volume) == pytest.approx(24.0)

        with pytest.raises(ValueError, match="dimension-three"):
            cad.volume(first_surface)


@pytest.mark.parametrize(
    "bad_result",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(True, id="boolean"),
        pytest.param("invalid", id="nonnumeric"),
    ],
)
def test_geometry_model_volume_rejects_invalid_backend_results(
    fake_gmsh: _FakeGmsh,
    bad_result: Any,
) -> None:
    with geometry.model("invalid-volume-result", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        fake_gmsh.model._current_data()["masses"][(3, volume.tag)] = bad_result

        with pytest.raises(ValueError, match="volume"):
            cad.volume(volume)


def test_adjacency_any_all_and_every_target_dimension(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("selection-adjacency", dimension=3) as cad:
        topology = _shared_surface_topology(cad, fake_gmsh)
        candidates = (
            topology["right_curve"],
            topology["shared_curve"],
            topology["left_curve"],
        )
        anchors = (
            topology["left_surface"],
            topology["right_surface"],
        )

        assert curves.adjacent_to(
            cad,
            (anchor for anchor in anchors),
            candidates,
            mode="any",
        ) == candidates
        assert curves.adjacent_to(
            cad,
            anchors,
            candidates,
            mode="all",
        ) == (topology["shared_curve"],)
        assert points.adjacent_to(
            cad,
            (topology["left_curve"], topology["shared_curve"]),
            mode="all",
        ) == (topology["shared_start"],)

        volume = cad.box(10.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        volume_surfaces = cad.boundary((volume,), combined=False)
        assert surfaces.adjacent_to(cad, (volume,)) == volume_surfaces
        assert volumes.adjacent_to(cad, (volume_surfaces[0],)) == (volume,)


def test_adjacency_deduplicates_inputs_and_uses_one_explicit_live_snapshot(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("selection-adjacency-snapshot", dimension=2) as cad:
        topology = _shared_surface_topology(cad, fake_gmsh)
        isolated = cad.line(cad.point(5.0, 0.0), cad.point(6.0, 0.0))
        candidates = (
            topology["right_curve"],
            isolated,
            topology["shared_curve"],
            topology["right_curve"],
            isolated,
            topology["left_curve"],
        )
        anchors = (
            topology["left_surface"],
            topology["right_surface"],
            topology["left_surface"],
            topology["right_surface"],
        )

        fake_gmsh.model.calls.clear()
        assert curves.adjacent_to(cad, anchors, candidates) == (
            topology["right_curve"],
            topology["shared_curve"],
            topology["left_curve"],
        )
        assert [
            call[1]
            for call in fake_gmsh.model.calls
            if call[0] == "getEntities"
        ] == [1]
        assert [
            call
            for call in fake_gmsh.model.calls
            if call[0] == "getBoundingBox"
        ] == []
        assert [
            (call[1], call[2])
            for call in fake_gmsh.model.calls
            if call[0] == "getAdjacencies"
        ] == [
            (2, topology["left_surface"].tag),
            (2, topology["right_surface"].tag),
        ]


def test_adjacency_omitted_and_empty_candidates_do_not_take_extra_snapshots(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("selection-adjacency-default-candidates", dimension=2) as cad:
        topology = _shared_surface_topology(cad, fake_gmsh)

        fake_gmsh.model.calls.clear()
        assert curves.adjacent_to(cad, (topology["left_surface"],)) == (
            topology["left_curve"],
            topology["shared_curve"],
        )
        assert [
            call[1]
            for call in fake_gmsh.model.calls
            if call[0] == "getEntities"
        ] == [1]
        assert [
            call
            for call in fake_gmsh.model.calls
            if call[0] == "getBoundingBox"
        ] == []

        fake_gmsh.model.calls.clear()
        assert curves.adjacent_to(cad, (topology["left_surface"],), ()) == ()
        assert [
            call[1]
            for call in fake_gmsh.model.calls
            if call[0] == "getEntities"
        ] == [0]
        assert [
            call
            for call in fake_gmsh.model.calls
            if call[0] == "getBoundingBox"
        ] == []
        assert [
            (call[1], call[2])
            for call in fake_gmsh.model.calls
            if call[0] == "getAdjacencies"
        ] == [(2, topology["left_surface"].tag)]


def test_adjacency_rejects_invalid_anchors_dimensions_and_modes(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("adjacency-validation", dimension=2) as cad:
        topology = _shared_surface_topology(cad, fake_gmsh)
        candidate = topology["shared_curve"]

        with pytest.raises(ValueError, match="at least one anchor"):
            curves.adjacent_to(cad, (), (candidate,))
        with pytest.raises(TypeError, match="iterable"):
            curves.adjacent_to(cad, None, (candidate,))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="EntityRef"):
            curves.adjacent_to(cad, (object(),), (candidate,))
        with pytest.raises(ValueError, match="common dimension"):
            curves.adjacent_to(
                cad,
                (topology["left_surface"], topology["left_curve"]),
                (candidate,),
            )
        with pytest.raises(ValueError, match="differ"):
            curves.adjacent_to(cad, (topology["left_curve"],), (candidate,))
        with pytest.raises(ValueError, match="differ"):
            surfaces.adjacent_to(cad, (topology["first"],))
        with pytest.raises(ValueError, match="mode"):
            curves.adjacent_to(
                cad,
                (topology["left_surface"],),
                (candidate,),
                mode="some",
            )


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda cad: points.by_coord(cad, ()),
            id="coordinate-required",
        ),
        pytest.param(
            lambda cad: curves.by_center(cad, ()),
            id="center-required",
        ),
        pytest.param(
            lambda cad: points.by_x(cad, float("nan"), ()),
            id="coordinate-nan",
        ),
        pytest.param(
            lambda cad: surfaces.by_z(cad, True, ()),
            id="coordinate-boolean",
        ),
        pytest.param(
            lambda cad: curves.by_x(cad, 0.0, (), tolerance=-1.0),
            id="coordinate-negative-tolerance",
        ),
        pytest.param(
            lambda cad: curves.by_center(
                cad,
                (),
                x=0.0,
                tolerance=float("inf"),
            ),
            id="center-infinite-tolerance",
        ),
        pytest.param(
            lambda cad: points.in_box(cad, ()),
            id="box-bound-required",
        ),
        pytest.param(
            lambda cad: curves.in_box(cad, (), xmin=1.0, xmax=0.0),
            id="box-inverted-x",
        ),
        pytest.param(
            lambda cad: surfaces.in_box(cad, (), ymin=2.0, ymax=1.0),
            id="box-inverted-y",
        ),
        pytest.param(
            lambda cad: volumes.in_box(cad, (), zmin=1.0, zmax=0.0),
            id="box-inverted-z",
        ),
        pytest.param(
            lambda cad: points.in_box(cad, (), xmin=float("nan")),
            id="box-nan-bound",
        ),
        pytest.param(
            lambda cad: curves.in_box(cad, (), xmin=0.0, tolerance=-1.0),
            id="box-negative-tolerance",
        ),
        pytest.param(
            lambda cad: curves.by_length(cad, (), value=-1.0),
            id="negative-length",
        ),
        pytest.param(
            lambda cad: surfaces.by_area(cad, (), value=float("inf")),
            id="infinite-area",
        ),
        pytest.param(
            lambda cad: volumes.by_volume(cad, (), value=True),
            id="boolean-volume",
        ),
        pytest.param(
            lambda cad: volumes.by_volume(
                cad,
                (),
                value=1.0,
                tolerance=-1.0,
            ),
            id="negative-measure-tolerance",
        ),
        pytest.param(
            lambda cad: points.nearest(cad, float("nan"), 0.0, entities=()),
            id="nearest-nan-x",
        ),
        pytest.param(
            lambda cad: points.nearest(cad, 0.0, 0.0, True, entities=()),
            id="nearest-boolean-z",
        ),
    ],
)
def test_numeric_validation_rejects_nonfinite_and_negative_values(
    fake_gmsh: _FakeGmsh,
    operation: Callable[[geometry.GeometryModel], object],
) -> None:
    del fake_gmsh
    with geometry.model("selection-numeric-validation", dimension=3) as cad:
        with pytest.raises(ValueError):
            operation(cad)


def test_selection_requires_a_geometry_model() -> None:
    with pytest.raises(TypeError, match="GeometryModel"):
        points.all(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GeometryModel"):
        volumes.by_volume(object(), value=1.0)  # type: ignore[arg-type]


def test_selection_propagates_foreign_stale_and_closed_model_errors(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("selection-owner-outer", dimension=2) as outer:
        foreign_curve = outer.line(
            outer.point(0.0, 0.0),
            outer.point(1.0, 0.0),
        )
        foreign_surface = outer.rectangle(0.0, 0.0, 1.0, 1.0)
        with geometry.model("selection-owner-inner", dimension=2) as inner:
            local_curve = inner.line(
                inner.point(0.0, 0.0),
                inner.point(1.0, 0.0),
            )
            local_surface = inner.rectangle(0.0, 0.0, 1.0, 1.0)

            with pytest.raises(geometry.EntityOwnershipError):
                curves.by_length(inner, (foreign_curve,), value=1.0)
            with pytest.raises(geometry.EntityOwnershipError):
                curves.adjacent_to(
                    inner,
                    (local_surface,),
                    (local_curve, local_curve, foreign_curve),
                )
            with pytest.raises(geometry.EntityOwnershipError):
                curves.adjacent_to(
                    inner,
                    (local_surface, local_surface, foreign_surface),
                    (local_curve,),
                )

    with geometry.model("selection-owner-3d", dimension=3) as outer:
        foreign_volume = outer.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        with geometry.model("selection-owner-2d", dimension=2) as inner:
            local_surface = inner.rectangle(0.0, 0.0, 1.0, 1.0)

            with pytest.raises(geometry.EntityOwnershipError):
                volumes.adjacent_to(
                    inner,
                    (local_surface,),
                    (foreign_volume,),
                )

    with geometry.model("selection-stale", dimension=2) as cad:
        curve = cad.line(cad.point(0.0, 0.0), cad.point(1.0, 0.0))
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        cad.raw_occ
        current_curve = cad.entity(1, curve.tag)
        current_surface = cad.entity(2, surface.tag)

        with pytest.raises(geometry.StaleEntityError):
            curves.by_length(cad, (curve,), value=1.0)
        with pytest.raises(geometry.StaleEntityError):
            curves.adjacent_to(
                cad,
                (current_surface,),
                (current_curve, current_curve, curve),
            )
        with pytest.raises(geometry.StaleEntityError):
            curves.adjacent_to(
                cad,
                (current_surface, current_surface, surface),
                (current_curve,),
            )

    closed = geometry.GeometryModel("selection-closed", dimension=2)
    with closed:
        closed_curve = closed.line(
            closed.point(0.0, 0.0),
            closed.point(1.0, 0.0),
        )
        closed_surface = closed.rectangle(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        curves.by_length(closed, (closed_curve,), value=1.0)
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        curves.all(closed)
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        curves.by_length(closed, (), value=1.0)
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        points.nearest(closed, 0.0, 0.0, entities=())
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        curves.adjacent_to(closed, (closed_surface,), (closed_curve,))
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        curves.adjacent_to(closed, (closed_surface,), ())

    assert fake_gmsh.model.current == ""


def test_geometry_model_volume_propagates_owner_stale_and_closed_errors(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("volume-owner-outer", dimension=3) as outer:
        foreign = outer.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        with geometry.model("volume-owner-inner", dimension=3) as inner:
            with pytest.raises(geometry.EntityOwnershipError):
                inner.volume(foreign)

    with geometry.model("volume-stale", dimension=3) as cad:
        stale = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        cad.raw_occ
        with pytest.raises(geometry.StaleEntityError):
            cad.volume(stale)

    closed = geometry.GeometryModel("volume-closed", dimension=3)
    with closed:
        volume = closed.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        closed.volume(volume)

    assert fake_gmsh.model.current == ""
