from __future__ import annotations

import math

import pytest

from fem.application.planar_construction import (
    PlanarConstructionCompileError,
    compile_planar_construction,
)
from fem.geometry import PlanarConstructionIR, SketchArc, SketchCircle, SketchLine
from tests.fixtures.planar_construction_phase0 import EXPECTED_H_CONSTRUCTION


def _compile(nodes: list[dict[str, object]], result: str = "result"):
    return compile_planar_construction(
        PlanarConstructionIR.from_dict(
            {
                "schema_version": 1,
                "name": "phase2 fixture",
                "plane": "XY",
                "nodes": nodes,
                "result_node_id": result,
            }
        )
    )


def _rectangle(node_id: str, x: float, y: float, width: float, height: float):
    return {
        "id": node_id,
        "kind": "rectangle",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _circle(node_id: str, x: float, y: float, radius: float):
    return {
        "id": node_id,
        "kind": "circle",
        "center_x": x,
        "center_y": y,
        "radius": radius,
    }


def _shape_nodes(shape: str) -> list[dict[str, object]]:
    rectangles = {
        "h": (
            _rectangle("a", 0, 0, 10, 60),
            _rectangle("b", 0, 25, 80, 10),
            _rectangle("c", 70, 0, 10, 60),
        ),
        "t": (
            _rectangle("a", 0, 50, 60, 10),
            _rectangle("b", 25, 0, 10, 60),
            _rectangle("c", 25, 20, 10, 10),
        ),
        "e": (
            _rectangle("a", 0, 0, 10, 60),
            _rectangle("b", 0, 50, 50, 10),
            _rectangle("c", 0, 25, 40, 10),
            _rectangle("d", 0, 0, 50, 10),
        ),
        "cross": (
            _rectangle("a", 20, 0, 10, 50),
            _rectangle("b", 0, 20, 50, 10),
            _rectangle("c", 20, 20, 10, 10),
        ),
    }[shape]
    return [
        *rectangles,
        {
            "id": "result",
            "kind": "union",
            "operands": [item["id"] for item in rectangles],
        },
    ]


@pytest.mark.gmsh
def test_h_plate_fixture_is_exact_deterministic_and_tag_free(real_gmsh) -> None:
    before = tuple(real_gmsh.model.list())
    ir = PlanarConstructionIR.from_dict(EXPECTED_H_CONSTRUCTION)
    first = compile_planar_construction(ir)
    second = compile_planar_construction(ir)

    assert first.recipe == second.recipe
    assert first.curve_lineage == second.curve_lineage
    assert first.proof == second.proof
    assert first.proof.area == pytest.approx(30_000 - 1_800 - 4 * math.pi)
    assert first.proof.material_profile_count == 1
    assert first.proof.hole_count == 5
    assert first.proof.curve_type_counts == (("circle", 4), ("line", 16))
    h_profile = next(
        profile
        for profile in first.profile_analysis.profiles
        if profile.is_hole and len(profile.curve_ids) == 12
    )
    assert h_profile.signed_area == pytest.approx(1_800)
    assert all(
        "tag" not in repr(value).casefold()
        for value in (first.proof, first.curve_lineage, first.preview)
    )
    assert first.preview.faces and first.preview.edges
    assert tuple(real_gmsh.model.list()) == before


@pytest.mark.gmsh
@pytest.mark.parametrize("shape", ("h", "t", "e", "cross"))
def test_three_or_more_rectangles_form_one_exact_region(shape: str) -> None:
    result = _compile(_shape_nodes(shape))
    assert result.profile_analysis.valid
    assert result.proof.component_count == 1
    assert result.proof.material_profile_count == 1
    assert result.proof.hole_count == 0
    assert all(isinstance(curve, SketchLine) for curve in result.recipe.curves)


@pytest.mark.gmsh
@pytest.mark.parametrize("shape", ("h", "t", "e", "cross"))
def test_plate_difference_preserves_each_composite_slot(shape: str) -> None:
    nodes = _shape_nodes(shape)
    nodes[-1] = {**nodes[-1], "id": "slot"}
    nodes.extend(
        (
            _rectangle("plate", -10, -10, 100, 80),
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["slot"],
            },
        )
    )
    result = _compile(nodes)
    assert result.profile_analysis.valid
    assert result.proof.material_profile_count == 1
    assert result.proof.hole_count == 1


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("nodes", "area", "types"),
    (
        (
            [
                _rectangle("body", -4, -2, 8, 4),
                _circle("left", -4, 0, 2),
                _circle("right", 4, 0, 2),
                {
                    "id": "result",
                    "kind": "union",
                    "operands": ["body", "left", "right"],
                },
            ],
            32 + 4 * math.pi,
            {SketchLine, SketchArc},
        ),
        (
            [
                _circle("head", 0, 3, 3),
                _rectangle("stem", -1, -4, 2, 7),
                {"id": "result", "kind": "union", "operands": ["head", "stem"]},
            ],
            None,
            {SketchLine, SketchArc},
        ),
        (
            [
                _circle("outer", 0, 0, 5),
                _circle("inner", 0, 0, 2),
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "outer",
                    "subtract": ["inner"],
                },
            ],
            21 * math.pi,
            {SketchCircle},
        ),
    ),
)
def test_curved_boolean_matrix(nodes, area, types) -> None:
    result = _compile(nodes)
    assert result.profile_analysis.valid
    if area is not None:
        assert result.proof.area == pytest.approx(area)
    assert {type(curve) for curve in result.recipe.curves} == types


@pytest.mark.gmsh
def test_polygon_and_positive_intersection_compile_exactly() -> None:
    result = _compile(
        [
            {
                "id": "triangle",
                "kind": "polygon",
                "vertices": [[0, 0], [4, 0], [0, 4]],
            },
            _rectangle("window", 0, 0, 2, 2),
            {
                "id": "result",
                "kind": "intersection",
                "operands": ["triangle", "window"],
            },
        ]
    )
    assert result.profile_analysis.valid
    assert result.proof.area == pytest.approx(4)
    assert result.proof.curve_type_counts == (("line", 4),)


@pytest.mark.gmsh
@pytest.mark.parametrize("cap", ("butt", "square", "round"))
def test_path_stroke_supports_all_caps(cap: str) -> None:
    result = _compile(
        [
            {
                "id": "result",
                "kind": "path_stroke",
                "points": [[0, 0], [10, 0]],
                "width": 2,
                "cap": cap,
                "join": "miter",
            }
        ]
    )
    expected = {"butt": 20, "square": 24, "round": 20 + math.pi}[cap]
    assert result.proof.area == pytest.approx(expected)


@pytest.mark.gmsh
@pytest.mark.parametrize("join", ("miter", "bevel", "round"))
def test_path_stroke_supports_all_joins(join: str) -> None:
    result = _compile(
        [
            {
                "id": "result",
                "kind": "path_stroke",
                "points": [[0, 0], [10, 0], [10, 10], [4, 10], [4, 5]],
                "width": 2,
                "cap": "butt",
                "join": join,
            }
        ]
    )
    assert result.profile_analysis.valid
    assert result.proof.component_count == 1


@pytest.mark.gmsh
def test_path_stroke_accepts_collinear_intermediate_points() -> None:
    result = _compile(
        [
            {
                "id": "result",
                "kind": "path_stroke",
                "points": [[0, 0], [4, 0], [10, 0]],
                "width": 2,
                "cap": "butt",
                "join": "miter",
            }
        ]
    )
    assert result.proof.area == pytest.approx(20)


@pytest.mark.gmsh
def test_transform_and_pattern_nodes_feed_boolean_exactly() -> None:
    nodes = [
        _rectangle("plate", -20, -20, 40, 40),
        _circle("move_seed", 2, 0, 0.5),
        _circle("rect_seed", -10, -10, 0.5),
        _circle("circle_seed", 10, 0, 0.5),
        {"id": "moved", "kind": "translate", "source": "move_seed", "dx": 2, "dy": 0},
        {
            "id": "turned",
            "kind": "rotate",
            "source": "moved",
            "center_x": 0,
            "center_y": 0,
            "angle_degrees": 90,
        },
        {
            "id": "mirrored",
            "kind": "mirror",
            "source": "turned",
            "line_point_x": 0,
            "line_point_y": 0,
            "line_direction_x": 1,
            "line_direction_y": 0,
        },
        {
            "id": "linear",
            "kind": "linear_pattern",
            "seed": "mirrored",
            "count": 2,
            "step_x": 5,
            "step_y": 0,
        },
        {
            "id": "rectangular",
            "kind": "rectangular_pattern",
            "seed": "rect_seed",
            "count_x": 2,
            "count_y": 2,
            "spacing_x": 3,
            "spacing_y": 3,
        },
        {
            "id": "circular",
            "kind": "circular_pattern",
            "seed": "circle_seed",
            "count": 4,
            "center_x": 7,
            "center_y": 0,
            "total_angle_degrees": 360,
        },
        {
            "id": "result",
            "kind": "difference",
            "base": "plate",
            "subtract": ["linear", "rectangular", "circular"],
        },
    ]
    result = _compile(nodes)
    assert result.profile_analysis.valid
    assert result.proof.material_profile_count == 1
    assert result.proof.hole_count == 10
    assert result.proof.area == pytest.approx(1600 - 2.5 * math.pi)


@pytest.mark.gmsh
def test_multiple_components_and_material_island_are_preserved() -> None:
    result = _compile(
        [
            _circle("outer", 0, 0, 6),
            _circle("hole", 0, 0, 4),
            {
                "id": "annulus",
                "kind": "difference",
                "base": "outer",
                "subtract": ["hole"],
            },
            _circle("island", 0, 0, 2),
            {"id": "result", "kind": "union", "operands": ["annulus", "island"]},
        ]
    )
    assert result.proof.component_count == 2
    assert result.proof.material_profile_count == 2
    assert result.proof.hole_count == 1
    assert sorted(
        profile.nesting_depth for profile in result.profile_analysis.profiles
    ) == [0, 1, 2]


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "nodes",
    (
        [
            _rectangle("base", 0, 0, 2, 2),
            _rectangle("tool", -1, -1, 4, 4),
            {
                "id": "result",
                "kind": "difference",
                "base": "base",
                "subtract": ["tool"],
            },
        ],
        [
            _rectangle("left", 0, 0, 1, 1),
            _rectangle("right", 1 - 1.0e-10, 0, 1, 1),
            {"id": "result", "kind": "intersection", "operands": ["left", "right"]},
        ],
    ),
)
def test_empty_and_near_degenerate_booleans_fail_closed(nodes) -> None:
    with pytest.raises(PlanarConstructionCompileError) as caught:
        _compile(nodes)
    assert caught.value.diagnostic.code in {
        "planar-ir.boolean-empty",
        "planar-ir.degenerate-result",
    }
    assert caught.value.diagnostic.model_unchanged is True


@pytest.mark.gmsh
def test_point_tangent_union_fails_closed() -> None:
    with pytest.raises(PlanarConstructionCompileError) as caught:
        _compile(
            [
                _circle("left", 0, 0, 1),
                _circle("right", 2, 0, 1),
                {"id": "result", "kind": "union", "operands": ["left", "right"]},
            ]
        )
    assert caught.value.diagnostic.code == "planar-ir.degenerate-result"
