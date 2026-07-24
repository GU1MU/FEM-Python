from __future__ import annotations

from collections.abc import Callable, Iterator
import inspect
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


_V12_MODULE_EXPORTS = (
    (
        points,
        [
            "adjacent_to",
            "all",
            "by_coord",
            "by_x",
            "by_y",
            "by_z",
            "in_box",
            "intersects_box",
            "nearest",
            "nearest_to",
            "within_distance",
        ],
    ),
    (
        curves,
        [
            "adjacent_to",
            "all",
            "by_center",
            "by_coord",
            "by_length",
            "by_length_range",
            "by_x",
            "by_y",
            "by_z",
            "in_box",
            "intersects_box",
            "nearest_to",
            "within_distance",
        ],
    ),
    (
        surfaces,
        [
            "adjacent_to",
            "all",
            "by_area",
            "by_area_range",
            "by_center",
            "by_coord",
            "by_x",
            "by_y",
            "by_z",
            "in_box",
            "intersects_box",
            "nearest_to",
            "within_distance",
        ],
    ),
    (
        volumes,
        [
            "adjacent_to",
            "all",
            "by_center",
            "by_volume",
            "by_volume_range",
            "in_box",
            "intersects_box",
            "nearest_to",
            "within_distance",
        ],
    ),
)


_V12_INTERSECTION_SELECTORS = (
    (0, points),
    (1, curves),
    (2, surfaces),
    (3, volumes),
)


_V12_RANGE_SELECTORS = (
    (1, curves, "by_length_range"),
    (2, surfaces, "by_area_range"),
    (3, volumes, "by_volume_range"),
)


def _same_dimension_entities(
    cad: geometry.GeometryModel,
    dimension: int,
    count: int,
) -> tuple[geometry.EntityRef, ...]:
    return tuple(
        _representative_entities(cad)[dimension] for _ in range(count)
    )


def _install_distance_batch_spy(
    monkeypatch: pytest.MonkeyPatch,
    distance_by_entity: dict[geometry.EntityRef, float],
) -> list[tuple[geometry.EntityRef, tuple[geometry.EntityRef, ...]]]:
    calls: list[tuple[geometry.EntityRef, tuple[geometry.EntityRef, ...]]] = []

    def distances_to(
        cad: geometry.GeometryModel,
        anchor: geometry.EntityRef,
        entities: Any,
    ) -> tuple[float, ...]:
        del cad
        candidates = tuple(entities)
        calls.append((anchor, candidates))
        return tuple(distance_by_entity[candidate] for candidate in candidates)

    def distance(
        cad: geometry.GeometryModel,
        left: geometry.EntityRef,
        right: geometry.EntityRef,
    ) -> float:
        del cad, left, right
        raise AssertionError("distance selectors must not call cad.distance()")

    monkeypatch.setattr(geometry.GeometryModel, "distances_to", distances_to)
    monkeypatch.setattr(geometry.GeometryModel, "distance", distance)
    return calls


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
    ("module", "expected"),
    _V12_MODULE_EXPORTS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_v12_cad_modules_export_the_complete_public_contract(
    module: Any,
    expected: list[str],
) -> None:
    assert module.__all__ == expected
    assert tuple(name for name in expected if hasattr(module, name)) == tuple(expected)


@pytest.mark.parametrize(
    "module",
    (points, curves, surfaces, volumes),
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_intersects_box_public_signature_and_docstring(module: Any) -> None:
    function = module.intersects_box
    signature = inspect.signature(function)
    parameters = signature.parameters

    assert tuple(parameters) == (
        "cad",
        "entities",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "tolerance",
    )
    assert parameters["cad"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["cad"].default is inspect.Parameter.empty
    assert parameters["entities"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["entities"].default is None
    for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "tolerance"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        assert parameters[name].default is None
    assert parameters["tolerance"].default == 1.0e-8

    hints = get_type_hints(function)
    assert hints["cad"] is geometry.GeometryModel
    assert hints["return"] == tuple[geometry.EntityRef, ...]
    documentation = (function.__doc__ or "").lower()
    assert "bounding box" in documentation
    assert "not an exact" in documentation
    assert "does not enter" in documentation


@pytest.mark.parametrize(
    ("module", "name"),
    (
        (curves, "by_length_range"),
        (surfaces, "by_area_range"),
        (volumes, "by_volume_range"),
    ),
    ids=("length", "area", "volume"),
)
def test_measure_range_public_signatures(module: Any, name: str) -> None:
    function = getattr(module, name)
    signature = inspect.signature(function)
    parameters = signature.parameters

    assert tuple(parameters) == (
        "cad",
        "entities",
        "minimum",
        "maximum",
        "tolerance",
    )
    assert parameters["cad"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["cad"].default is inspect.Parameter.empty
    assert parameters["entities"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["entities"].default is None
    for parameter in ("minimum", "maximum", "tolerance"):
        assert parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["minimum"].default is None
    assert parameters["maximum"].default is None
    assert parameters["tolerance"].default == 1.0e-8
    assert get_type_hints(function)["return"] == tuple[geometry.EntityRef, ...]


@pytest.mark.parametrize(
    "module",
    (points, curves, surfaces, volumes),
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_entity_distance_selector_public_signatures_and_docstrings(
    module: Any,
) -> None:
    nearest_signature = inspect.signature(module.nearest_to)
    nearest_parameters = nearest_signature.parameters
    assert tuple(nearest_parameters) == ("cad", "anchor", "entities")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in nearest_parameters.values()
    )
    assert nearest_parameters["cad"].default is inspect.Parameter.empty
    assert nearest_parameters["anchor"].default is inspect.Parameter.empty
    assert nearest_parameters["entities"].default is None
    nearest_hints = get_type_hints(module.nearest_to)
    assert nearest_hints["anchor"] is geometry.EntityRef
    assert nearest_hints["return"] == geometry.EntityRef | None

    within_signature = inspect.signature(module.within_distance)
    within_parameters = within_signature.parameters
    assert tuple(within_parameters) == (
        "cad",
        "anchor",
        "entities",
        "max_distance",
        "tolerance",
    )
    assert within_parameters["cad"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert (
        within_parameters["anchor"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert (
        within_parameters["entities"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert within_parameters["entities"].default is None
    assert within_parameters["max_distance"].kind is inspect.Parameter.KEYWORD_ONLY
    assert within_parameters["max_distance"].default is inspect.Parameter.empty
    assert within_parameters["tolerance"].kind is inspect.Parameter.KEYWORD_ONLY
    assert within_parameters["tolerance"].default == 1.0e-8
    within_hints = get_type_hints(module.within_distance)
    assert within_hints["anchor"] is geometry.EntityRef
    assert within_hints["return"] == tuple[geometry.EntityRef, ...]

    for function in (module.nearest_to, module.within_distance):
        documentation = (function.__doc__ or "").lower()
        assert "minimum euclidean distance" in documentation
        assert "zero" in documentation
        assert "boundary" in documentation
        assert "hausdorff" in documentation


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


@pytest.mark.parametrize(
    ("dimension", "module"),
    _V12_INTERSECTION_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_intersects_box_supports_every_dimension_and_stable_candidates(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    module: Any,
) -> None:
    with geometry.model(f"intersects-box-dimension-{dimension}", dimension=3) as cad:
        outside, later, first = _same_dimension_entities(cad, dimension, 3)
        data = fake_gmsh.model._current_data()
        data["boxes"][(dimension, outside.tag)] = (
            2.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.0,
        )
        data["boxes"][(dimension, later.tag)] = (
            0.75,
            0.75,
            0.75,
            1.25,
            1.25,
            1.25,
        )
        data["boxes"][(dimension, first.tag)] = (
            0.25,
            0.25,
            0.25,
            0.5,
            0.5,
            0.5,
        )
        candidates = _SinglePassEntities(
            (outside, later, first, later, first)
        )

        assert module.intersects_box(
            cad,
            candidates,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            tolerance=0.0,
        ) == (later, first)
        assert candidates.iterations == 1
        assert module.intersects_box(cad, (), xmin=0.0) == ()


def test_intersects_box_distinguishes_overlap_contact_enclosure_and_separation(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("intersects-box-relations", dimension=3) as cad:
        contained, partial, touching, enclosing, separated = (
            _same_dimension_entities(cad, 1, 5)
        )
        boxes = {
            contained: (0.2, 0.2, 0.0, 0.8, 0.8, 0.0),
            partial: (-0.5, 0.2, 0.0, 0.5, 0.8, 0.0),
            touching: (1.0, 0.2, 0.0, 2.0, 0.8, 0.0),
            enclosing: (-1.0, -1.0, 0.0, 2.0, 2.0, 0.0),
            separated: (1.01, 0.2, 0.0, 2.0, 0.8, 0.0),
        }
        data = fake_gmsh.model._current_data()
        for entity, bounds in boxes.items():
            data["boxes"][(1, entity.tag)] = bounds
        candidates = (separated, enclosing, touching, partial, contained)

        assert curves.intersects_box(
            cad,
            candidates,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            tolerance=0.0,
        ) == (enclosing, touching, partial, contained)
        assert curves.in_box(
            cad,
            candidates,
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            tolerance=0.0,
        ) == (contained,)


@pytest.mark.parametrize(
    "query",
    (
        {"xmin": 0.8},
        {"xmax": 0.2},
        {"ymin": 0.8},
        {"ymax": 0.2},
        {"zmin": 0.8},
        {"zmax": 0.2},
    ),
    ids=("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
)
def test_intersects_box_accepts_each_one_sided_axis_bound(
    fake_gmsh: _FakeGmsh,
    query: dict[str, float],
) -> None:
    with geometry.model("intersects-box-one-sided", dimension=3) as cad:
        curve = _representative_entities(cad)[1]
        fake_gmsh.model._current_data()["boxes"][(1, curve.tag)] = (
            0.2,
            0.2,
            0.2,
            0.8,
            0.8,
            0.8,
        )

        assert curves.intersects_box(
            cad,
            (curve,),
            tolerance=0.0,
            **query,
        ) == (curve,)


def test_intersects_box_uses_effective_occ_padding_plus_user_tolerance(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("intersects-box-tolerance", dimension=3) as cad:
        within, outside = _same_dimension_entities(cad, 1, 2)
        native_padding = cad.effective_bounding_box_tolerance(0.0)
        user_tolerance = 1.0e-8
        data = fake_gmsh.model._current_data()
        data["boxes"][(1, within.tag)] = (
            1.0 + native_padding + 0.5 * user_tolerance,
            0.0,
            0.0,
            2.0,
            1.0,
            0.0,
        )
        data["boxes"][(1, outside.tag)] = (
            1.0 + native_padding + 1.5 * user_tolerance,
            0.0,
            0.0,
            2.0,
            1.0,
            0.0,
        )

        assert curves.intersects_box(
            cad,
            (outside, within),
            xmax=1.0,
            tolerance=user_tolerance,
        ) == (within,)
        assert curves.intersects_box(
            cad,
            (within,),
            xmax=1.0,
            tolerance=0.0,
        ) == ()


def test_intersects_box_deliberately_allows_an_aabb_false_positive(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("intersects-box-aabb-false-positive", dimension=2) as cad:
        diagonal = cad.line(cad.point(0.0, 0.0), cad.point(1.0, 1.0))

        # The actual diagonal has y=x and misses x<=0.2, y>=0.8, while its
        # axis-aligned bounding box overlaps both independent query bounds.
        assert curves.intersects_box(
            cad,
            (diagonal,),
            xmax=0.2,
            ymin=0.8,
            tolerance=0.0,
        ) == (diagonal,)
        assert curves.in_box(
            cad,
            (diagonal,),
            xmax=0.2,
            ymin=0.8,
            tolerance=0.0,
        ) == ()


@pytest.mark.parametrize(
    "operation",
    (
        lambda cad: points.intersects_box(cad, ()),
        lambda cad: curves.intersects_box(cad, (), xmin=1.0, xmax=0.0),
        lambda cad: surfaces.intersects_box(cad, (), ymin=1.0, ymax=0.0),
        lambda cad: volumes.intersects_box(cad, (), zmin=1.0, zmax=0.0),
        lambda cad: points.intersects_box(cad, (), xmin=True),
        lambda cad: curves.intersects_box(cad, (), xmax=float("nan")),
        lambda cad: surfaces.intersects_box(cad, (), ymin=float("inf")),
        lambda cad: volumes.intersects_box(cad, (), zmax="invalid"),
        lambda cad: points.intersects_box(
            cad,
            (),
            xmin=0.0,
            tolerance=True,
        ),
        lambda cad: curves.intersects_box(
            cad,
            (),
            xmin=0.0,
            tolerance=-1.0,
        ),
        lambda cad: surfaces.intersects_box(
            cad,
            (),
            xmin=0.0,
            tolerance=float("inf"),
        ),
    ),
)
def test_intersects_box_rejects_invalid_bounds_and_tolerances(
    fake_gmsh: _FakeGmsh,
    operation: Callable[[geometry.GeometryModel], object],
) -> None:
    del fake_gmsh
    with geometry.model("intersects-box-invalid", dimension=3) as cad:
        with pytest.raises(ValueError):
            operation(cad)


def test_intersects_box_propagates_malformed_native_bounding_boxes(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("intersects-box-malformed", dimension=2) as cad:
        curve = cad.line(cad.point(0.0, 0.0), cad.point(1.0, 0.0))
        fake_gmsh.model._current_data()["boxes"][(1, curve.tag)] = (
            0.0,
            0.0,
            0.0,
            1.0,
            float("nan"),
            0.0,
        )

        with pytest.raises(geometry.GeometryError, match="bounding box"):
            curves.intersects_box(cad, (curve,), xmin=0.0)


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
    ("dimension", "module", "name"),
    _V12_RANGE_SELECTORS,
    ids=("length", "area", "volume"),
)
def test_measure_ranges_support_one_sided_closed_ranges_and_candidate_order(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    module: Any,
    name: str,
) -> None:
    with geometry.model(f"measure-range-{name}", dimension=3) as cad:
        lower, middle, upper = _same_dimension_entities(cad, dimension, 3)
        data = fake_gmsh.model._current_data()
        for entity, measure in (
            (lower, 0.9),
            (middle, 1.0),
            (upper, 1.1),
        ):
            data["masses"][(dimension, entity.tag)] = measure
        selector = getattr(module, name)
        candidates = _SinglePassEntities((upper, middle, lower, upper, middle))

        assert selector(
            cad,
            candidates,
            minimum=0.95,
            maximum=1.05,
            tolerance=0.0,
        ) == (middle,)
        assert candidates.iterations == 1
        assert selector(
            cad,
            (upper, middle, lower),
            minimum=1.0,
            tolerance=0.0,
        ) == (upper, middle)
        assert selector(
            cad,
            (upper, middle, lower),
            maximum=1.0,
            tolerance=0.0,
        ) == (middle, lower)
        assert selector(
            cad,
            (upper, middle, lower),
            minimum=1.0,
            maximum=1.0,
            tolerance=0.0,
        ) == (middle,)
        assert selector(cad, (), minimum=0.0) == ()


@pytest.mark.parametrize(
    ("dimension", "module", "name"),
    _V12_RANGE_SELECTORS,
    ids=("length", "area", "volume"),
)
def test_measure_ranges_expand_both_closed_endpoints_by_absolute_tolerance(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    module: Any,
    name: str,
) -> None:
    with geometry.model(f"measure-range-tolerance-{name}", dimension=3) as cad:
        below, exact, above, missed = _same_dimension_entities(cad, dimension, 4)
        tolerance = 1.0e-8
        data = fake_gmsh.model._current_data()
        for entity, measure in (
            (below, 1.0 - tolerance),
            (exact, 1.0),
            (above, 1.0 + tolerance),
            (missed, 1.0 + 2.0 * tolerance),
        ):
            data["masses"][(dimension, entity.tag)] = measure

        assert getattr(module, name)(
            cad,
            (missed, above, exact, below),
            minimum=1.0,
            maximum=1.0,
            tolerance=tolerance,
        ) == (above, exact, below)


@pytest.mark.parametrize(
    "operation",
    (
        lambda cad: curves.by_length_range(cad, ()),
        lambda cad: curves.by_length_range(cad, (), minimum=-1.0),
        lambda cad: surfaces.by_area_range(cad, (), maximum=-1.0),
        lambda cad: volumes.by_volume_range(
            cad,
            (),
            minimum=2.0,
            maximum=1.0,
            tolerance=100.0,
        ),
        lambda cad: curves.by_length_range(cad, (), minimum=True),
        lambda cad: surfaces.by_area_range(cad, (), maximum="invalid"),
        lambda cad: volumes.by_volume_range(cad, (), minimum=float("nan")),
        lambda cad: curves.by_length_range(cad, (), maximum=float("inf")),
        lambda cad: surfaces.by_area_range(
            cad,
            (),
            minimum=0.0,
            tolerance=True,
        ),
        lambda cad: volumes.by_volume_range(
            cad,
            (),
            maximum=1.0,
            tolerance=-1.0,
        ),
        lambda cad: curves.by_length_range(
            cad,
            (),
            minimum=0.0,
            tolerance=float("inf"),
        ),
    ),
)
def test_measure_ranges_reject_invalid_limits_and_tolerances(
    fake_gmsh: _FakeGmsh,
    operation: Callable[[geometry.GeometryModel], object],
) -> None:
    del fake_gmsh
    with geometry.model("measure-range-invalid", dimension=3) as cad:
        with pytest.raises(ValueError):
            operation(cad)


@pytest.mark.parametrize(
    ("dimension", "module", "name"),
    _V12_RANGE_SELECTORS,
    ids=("length", "area", "volume"),
)
def test_measure_ranges_propagate_invalid_native_measures(
    fake_gmsh: _FakeGmsh,
    dimension: int,
    module: Any,
    name: str,
) -> None:
    with geometry.model(f"measure-range-native-{name}", dimension=3) as cad:
        entity = _representative_entities(cad)[dimension]
        fake_gmsh.model._current_data()["masses"][(dimension, entity.tag)] = True

        with pytest.raises(ValueError):
            getattr(module, name)(cad, (entity,), minimum=0.0)


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


@pytest.mark.parametrize(
    ("dimension", "module"),
    _V12_INTERSECTION_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_entity_distance_selectors_batch_cross_dimension_candidates_once(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    module: Any,
) -> None:
    del fake_gmsh
    with geometry.model(f"distance-selector-{dimension}", dimension=3) as cad:
        farther, nearer = _same_dimension_entities(cad, dimension, 2)
        anchor_dimension = (dimension + 1) % 4
        anchor = _representative_entities(cad)[anchor_dimension]
        calls = _install_distance_batch_spy(
            monkeypatch,
            {farther: 1.5, nearer: 0.25},
        )
        candidates = _SinglePassEntities((farther, nearer, farther, nearer))

        assert module.nearest_to(cad, anchor, candidates) == nearer
        assert candidates.iterations == 1
        assert module.within_distance(
            cad,
            anchor,
            (farther, nearer, farther, nearer),
            max_distance=0.25,
            tolerance=0.0,
        ) == (nearer,)
        assert calls == [
            (anchor, (farther, nearer)),
            (anchor, (farther, nearer)),
        ]


@pytest.mark.parametrize(
    ("dimension", "module"),
    _V12_INTERSECTION_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_entity_distance_selectors_use_default_candidates_in_model_order(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    module: Any,
) -> None:
    del fake_gmsh
    with geometry.model(f"distance-selector-default-{dimension}", dimension=3) as cad:
        _same_dimension_entities(cad, dimension, 2)
        anchor = _representative_entities(cad)[(dimension + 1) % 4]
        expected = cad.entities(dimension)
        assert expected
        calls = _install_distance_batch_spy(
            monkeypatch,
            {entity: float(index) for index, entity in enumerate(expected)},
        )

        assert module.nearest_to(cad, anchor) == expected[0]
        assert module.within_distance(
            cad,
            anchor,
            max_distance=0.0,
            tolerance=0.0,
        ) == (expected[0],)
        assert calls == [(anchor, expected), (anchor, expected)]


@pytest.mark.parametrize(
    ("dimension", "module"),
    _V12_INTERSECTION_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_entity_distance_selectors_include_anchor_and_retain_first_tie(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    module: Any,
) -> None:
    del fake_gmsh
    with geometry.model(f"distance-selector-anchor-{dimension}", dimension=3) as cad:
        anchor, first, second = _same_dimension_entities(cad, dimension, 3)
        calls = _install_distance_batch_spy(
            monkeypatch,
            {anchor: 0.0, first: 0.5, second: 0.5},
        )

        assert module.nearest_to(cad, anchor, (first, second)) == first
        assert module.nearest_to(cad, anchor, (first, anchor, second)) == anchor
        assert module.within_distance(
            cad,
            anchor,
            (first, anchor, second),
            max_distance=0.0,
            tolerance=0.0,
        ) == (anchor,)
        assert calls == [
            (anchor, (first, second)),
            (anchor, (first, anchor, second)),
            (anchor, (first, anchor, second)),
        ]


def test_within_distance_includes_endpoint_and_absolute_tolerance(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_gmsh
    with geometry.model("distance-selector-threshold", dimension=3) as cad:
        endpoint, tolerated, missed = _same_dimension_entities(cad, 2, 3)
        anchor = _representative_entities(cad)[1]
        tolerance = 1.0e-8
        calls = _install_distance_batch_spy(
            monkeypatch,
            {
                endpoint: 0.5,
                tolerated: 0.5 + tolerance,
                missed: 0.5 + 2.0 * tolerance,
            },
        )

        assert surfaces.within_distance(
            cad,
            anchor,
            (missed, tolerated, endpoint),
            max_distance=0.5,
            tolerance=tolerance,
        ) == (tolerated, endpoint)
        assert calls == [(anchor, (missed, tolerated, endpoint))]


@pytest.mark.parametrize(
    ("dimension", "module"),
    _V12_INTERSECTION_SELECTORS,
    ids=("points", "curves", "surfaces", "volumes"),
)
def test_entity_distance_selectors_batch_even_explicit_empty_candidates(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    module: Any,
) -> None:
    del fake_gmsh
    with geometry.model(f"distance-selector-empty-{dimension}", dimension=3) as cad:
        anchor = _representative_entities(cad)[(dimension + 1) % 4]
        calls = _install_distance_batch_spy(monkeypatch, {})

        assert module.nearest_to(cad, anchor, ()) is None
        assert (
            module.within_distance(
                cad,
                anchor,
                (),
                max_distance=0.0,
            )
            == ()
        )
        assert calls == [(anchor, ()), (anchor, ())]


@pytest.mark.parametrize(
    "operation",
    (
        lambda cad, anchor: points.within_distance(
            cad,
            anchor,
            (),
            max_distance=-1.0,
        ),
        lambda cad, anchor: curves.within_distance(
            cad,
            anchor,
            (),
            max_distance=True,
        ),
        lambda cad, anchor: surfaces.within_distance(
            cad,
            anchor,
            (),
            max_distance=float("nan"),
        ),
        lambda cad, anchor: volumes.within_distance(
            cad,
            anchor,
            (),
            max_distance=float("inf"),
        ),
        lambda cad, anchor: points.within_distance(
            cad,
            anchor,
            (),
            max_distance="invalid",
        ),
        lambda cad, anchor: curves.within_distance(
            cad,
            anchor,
            (),
            max_distance=0.0,
            tolerance=-1.0,
        ),
        lambda cad, anchor: surfaces.within_distance(
            cad,
            anchor,
            (),
            max_distance=0.0,
            tolerance=True,
        ),
        lambda cad, anchor: volumes.within_distance(
            cad,
            anchor,
            (),
            max_distance=0.0,
            tolerance=float("inf"),
        ),
    ),
)
def test_within_distance_rejects_invalid_thresholds_before_batch_query(
    fake_gmsh: _FakeGmsh,
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[[geometry.GeometryModel, geometry.EntityRef], object],
) -> None:
    del fake_gmsh
    with geometry.model("distance-selector-invalid-threshold", dimension=3) as cad:
        anchor = cad.point(0.0, 0.0, 0.0)
        calls = _install_distance_batch_spy(monkeypatch, {})

        with pytest.raises(ValueError):
            operation(cad, anchor)
        assert calls == []


def test_entity_distance_selectors_reject_anchor_and_candidate_types_and_dimensions(
    fake_gmsh: _FakeGmsh,
) -> None:
    del fake_gmsh
    with geometry.model("distance-selector-types", dimension=2) as cad:
        point = cad.point(0.0, 0.0)
        curve = cad.line(point, cad.point(1.0, 0.0))

        with pytest.raises(TypeError, match="anchor.*EntityRef"):
            points.nearest_to(cad, object(), ())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="iterable"):
            points.nearest_to(cad, point, 42)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="EntityRef"):
            points.nearest_to(cad, point, (object(),))
        with pytest.raises(ValueError, match="dimension"):
            points.nearest_to(cad, point, (curve,))

    with geometry.model("distance-selector-high-dimension", dimension=3) as outer:
        volume = outer.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        with geometry.model("distance-selector-lower-dimension", dimension=2) as inner:
            point = inner.point(0.0, 0.0)
            oversized_local = inner._wrap_entity((3, 999))
            with pytest.raises(geometry.EntityOwnershipError):
                points.nearest_to(inner, volume, (point,))
            with pytest.raises(ValueError, match="dimension"):
                points.nearest_to(inner, oversized_local, (point,))


def test_entity_distance_selectors_propagate_foreign_and_stale_references(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("distance-selector-owner-outer", dimension=2) as outer:
        foreign_anchor = outer.rectangle(0.0, 0.0, 1.0, 1.0)
        foreign_candidate = outer.point(0.0, 0.0)
        with geometry.model("distance-selector-owner-inner", dimension=2) as inner:
            local_anchor = inner.rectangle(0.0, 0.0, 1.0, 1.0)
            local_candidate = inner.point(0.0, 0.0)

            with pytest.raises(geometry.EntityOwnershipError):
                points.nearest_to(
                    inner,
                    foreign_anchor,
                    (local_candidate,),
                )
            with pytest.raises(geometry.EntityOwnershipError):
                points.within_distance(
                    inner,
                    local_anchor,
                    (local_candidate, foreign_candidate),
                    max_distance=1.0,
                )

    with geometry.model("distance-selector-stale", dimension=2) as cad:
        stale_anchor = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        stale_candidate = cad.point(0.0, 0.0)
        other_candidate = cad.point(1.0, 0.0)
        cad.raw_occ
        current_anchor = cad.entity(2, stale_anchor.tag)
        current_candidate = cad.entity(0, stale_candidate.tag)
        current_other = cad.entity(0, other_candidate.tag)

        with pytest.raises(geometry.StaleEntityError):
            points.nearest_to(cad, stale_anchor, (current_candidate,))
        with pytest.raises(geometry.StaleEntityError):
            points.within_distance(
                cad,
                current_anchor,
                (current_other, stale_candidate),
                max_distance=1.0,
            )

    assert fake_gmsh.model.current == ""


def test_entity_distance_selectors_validate_native_liveness_for_empty_and_nonempty(
    fake_gmsh: _FakeGmsh,
) -> None:
    with geometry.model("distance-selector-native-missing", dimension=2) as cad:
        anchor = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        candidate = cad.point(0.0, 0.0)
        data = fake_gmsh.model._current_data()
        data["entities"].remove((2, anchor.tag))

        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            points.nearest_to(cad, anchor, ())

        current_anchor = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        data["entities"].remove((0, candidate.tag))
        with pytest.raises(geometry.StaleEntityError, match="no longer exists"):
            points.within_distance(
                cad,
                current_anchor,
                (candidate,),
                max_distance=1.0,
            )


def test_entity_distance_selectors_reject_closed_models_even_when_empty(
    fake_gmsh: _FakeGmsh,
) -> None:
    closed = geometry.GeometryModel("distance-selector-closed", dimension=2)
    with closed:
        anchor = closed.rectangle(0.0, 0.0, 1.0, 1.0)
        oversized_anchor = closed._wrap_entity((3, 999))

    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        points.nearest_to(closed, anchor, ())
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        points.nearest_to(closed, oversized_anchor, ())
    with pytest.raises(geometry.GeometryStateError, match="CLOSED"):
        points.within_distance(
            closed,
            anchor,
            (),
            max_distance=0.0,
        )
    assert fake_gmsh.model.current == ""


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
