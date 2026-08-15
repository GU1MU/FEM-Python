from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from fem.geometry.construction_ir import (
    MAX_BOOLEAN_OPERANDS,
    MAX_CANONICAL_PAYLOAD_BYTES,
    MAX_DAG_DEPTH,
    MAX_NODES,
    MAX_PATH_POINTS,
    MAX_PATTERN_INSTANCES,
    MAX_POLYGON_VERTICES,
    PlanarConstructionIR,
    PlanarIRValidationError,
)


def _construction(
    nodes: list[dict[str, object]],
    result_node_id: str | None = None,
    *,
    name: str = "construction",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": name,
        "plane": "XY",
        "nodes": nodes,
        "result_node_id": result_node_id or str(nodes[-1]["id"]),
    }


def _rectangle(node_id: str, x: float = 0.0) -> dict[str, object]:
    return {
        "id": node_id,
        "kind": "rectangle",
        "x": x,
        "y": 0.0,
        "width": 10.0,
        "height": 5.0,
    }


def _diagnostic(payload: dict[str, object]) -> object:
    with pytest.raises(PlanarIRValidationError) as caught:
        PlanarConstructionIR.from_dict(payload)
    diagnostic = caught.value.diagnostic
    assert diagnostic.retryable is True
    assert diagnostic.model_unchanged is True
    assert len(diagnostic.message) <= 240
    return diagnostic


def _generic_shape_fixtures() -> list[dict[str, object]]:
    return [
        _construction(
            [
                _rectangle("a", 0),
                _rectangle("b", 4),
                _rectangle("c", 8),
                {"id": "result", "kind": "union", "operands": ["a", "b", "c"]},
            ],
            name="fixture-1",
        ),
        _construction(
            [
                _rectangle("stem"),
                _rectangle("top", -3),
                {"id": "result", "kind": "union", "operands": ["stem", "top"]},
            ],
            name="fixture-2",
        ),
        _construction(
            [
                _rectangle("spine"),
                _rectangle("bar1", 2),
                _rectangle("bar2", 4),
                _rectangle("bar3", 6),
                {
                    "id": "result",
                    "kind": "union",
                    "operands": ["spine", "bar1", "bar2", "bar3"],
                },
            ],
            name="fixture-3",
        ),
        _construction(
            [
                _rectangle("horizontal"),
                _rectangle("vertical", 2),
                {
                    "id": "result",
                    "kind": "intersection",
                    "operands": ["horizontal", "vertical"],
                },
            ],
            name="fixture-4",
        ),
        _construction(
            [
                {
                    "id": "round",
                    "kind": "circle",
                    "center_x": 0,
                    "center_y": 0,
                    "radius": 3,
                },
                _rectangle("stem", 0),
                {"id": "result", "kind": "union", "operands": ["round", "stem"]},
            ],
            name="fixture-5",
        ),
    ]


def test_generic_nodes_express_composite_shape_matrix() -> None:
    for payload in _generic_shape_fixtures():
        ir = PlanarConstructionIR.from_dict(payload)
        assert ir.result_node_id == "result"
        assert len(ir.digest()) == 64


def test_all_v1_node_types_parse_to_immutable_dtos_and_summary() -> None:
    payload = _construction(
        [
            _rectangle("rect"),
            {
                "id": "circle",
                "kind": "circle",
                "center_x": 2,
                "center_y": 2,
                "radius": 1,
            },
            {"id": "poly", "kind": "polygon", "vertices": [[0, 0], [2, 0], [0, 2]]},
            {
                "id": "stroke",
                "kind": "path_stroke",
                "points": [[0, 0], [1, 0], [1, 1]],
                "width": 0.5,
                "cap": "round",
                "join": "bevel",
            },
            {"id": "moved", "kind": "translate", "source": "rect", "dx": 1, "dy": 2},
            {
                "id": "turned",
                "kind": "rotate",
                "source": "circle",
                "center_x": 0,
                "center_y": 0,
                "angle_degrees": 450,
            },
            {
                "id": "reflected",
                "kind": "mirror",
                "source": "poly",
                "line_point_x": 0,
                "line_point_y": 0,
                "line_direction_x": 1,
                "line_direction_y": 0,
            },
            {
                "id": "linear",
                "kind": "linear_pattern",
                "seed": "stroke",
                "count": 2,
                "step_x": 3,
                "step_y": 0,
            },
            {
                "id": "grid",
                "kind": "rectangular_pattern",
                "seed": "moved",
                "count_x": 2,
                "count_y": 2,
                "spacing_x": 20,
                "spacing_y": 10,
            },
            {
                "id": "radial",
                "kind": "circular_pattern",
                "seed": "turned",
                "count": 3,
                "center_x": 0,
                "center_y": 0,
                "total_angle_degrees": 360,
            },
            {
                "id": "overlap",
                "kind": "intersection",
                "operands": ["reflected", "linear"],
            },
            {"id": "cuts", "kind": "union", "operands": ["grid", "radial", "overlap"]},
            {
                "id": "result",
                "kind": "difference",
                "base": "rect",
                "subtract": ["cuts"],
            },
        ]
    )

    ir = PlanarConstructionIR.from_dict(payload)
    summary = ir.provider_safe_summary()

    assert summary.to_dict() == {
        "schema_version": 1,
        "node_count": 13,
        "primitive_count": 4,
        "boolean_count": 3,
        "transform_count": 3,
        "pattern_count": 3,
        "expanded_pattern_instances": 9,
        "dag_depth": 5,
        "canonical_payload_bytes": len(ir.canonical_json().encode("utf-8")),
        "canonical_digest_short": ir.digest()[:12],
    }
    with pytest.raises(FrozenInstanceError):
        ir.name = "changed"  # type: ignore[misc]


def test_declaration_and_commutative_operand_order_are_canonical() -> None:
    nodes = [
        _rectangle("left", 0),
        _rectangle("right", 5),
        {"id": "result", "kind": "union", "operands": ["left", "right"]},
    ]
    reordered = [deepcopy(nodes[2]), deepcopy(nodes[1]), deepcopy(nodes[0])]
    reordered[0]["operands"] = ["right", "left"]

    first = PlanarConstructionIR.from_dict(_construction(nodes))
    second = PlanarConstructionIR.from_dict(_construction(reordered, "result"))

    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()


def test_polygon_ring_and_rotation_angle_are_canonical() -> None:
    first = _construction(
        [
            {
                "id": "p",
                "kind": "polygon",
                "vertices": [[0, 0], [2, 0], [2, 1], [0, 1]],
            },
            {
                "id": "result",
                "kind": "rotate",
                "source": "p",
                "center_x": 0,
                "center_y": 0,
                "angle_degrees": 450,
            },
        ]
    )
    second = deepcopy(first)
    second["nodes"][0]["vertices"] = [[2, 1], [2, 0], [0, 0], [0, 1]]  # type: ignore[index]
    second["nodes"][1]["angle_degrees"] = 90  # type: ignore[index]

    assert (
        PlanarConstructionIR.from_dict(first).digest()
        == PlanarConstructionIR.from_dict(second).digest()
    )


def test_difference_base_and_dimension_change_digest() -> None:
    payload = _construction(
        [
            _rectangle("plate"),
            _rectangle("cut", 2),
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["cut"],
            },
        ]
    )
    base_changed = deepcopy(payload)
    base_changed["nodes"][2]["base"] = "cut"  # type: ignore[index]
    base_changed["nodes"][2]["subtract"] = ["plate"]  # type: ignore[index]
    size_changed = deepcopy(payload)
    size_changed["nodes"][0]["width"] = 11  # type: ignore[index]

    digest = PlanarConstructionIR.from_dict(payload).digest()
    assert PlanarConstructionIR.from_dict(base_changed).digest() != digest
    assert PlanarConstructionIR.from_dict(size_changed).digest() != digest


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_have_stable_diagnostic(value: float) -> None:
    payload = _construction([_rectangle("result")])
    payload["nodes"][0]["x"] = value  # type: ignore[index]
    diagnostic = _diagnostic(payload)
    assert diagnostic.code == "planar-ir.schema-invalid"  # type: ignore[attr-defined]
    assert diagnostic.node_id == "result"  # type: ignore[attr-defined]
    assert diagnostic.allowed_fields == ("x",)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("mutate", "code", "node_id"),
    [
        (
            lambda payload: payload["nodes"].append(deepcopy(payload["nodes"][0])),
            "planar-ir.duplicate-node-id",
            "result",
        ),
        (
            lambda payload: payload["nodes"].append(
                {"id": "u", "kind": "union", "operands": ["missing"]}
            )
            or payload.update(result_node_id="u"),
            "planar-ir.reference-missing",
            "u",
        ),
        (
            lambda payload: payload["nodes"][0].update(extra=True),
            "planar-ir.schema-invalid",
            "result",
        ),
        (
            lambda payload: payload["nodes"].append(_rectangle("unused", 20)),
            "planar-ir.unreachable-node",
            "unused",
        ),
    ],
)
def test_structural_failures_have_stable_diagnostics(
    mutate: object, code: str, node_id: str
) -> None:
    payload = _construction([_rectangle("result")])
    mutate(payload)  # type: ignore[operator]

    diagnostic = _diagnostic(payload)

    assert diagnostic.code == code  # type: ignore[attr-defined]
    assert diagnostic.node_id == node_id  # type: ignore[attr-defined]


def test_cycle_reports_shortest_stable_path() -> None:
    payload = _construction(
        [
            {"id": "a", "kind": "translate", "source": "b", "dx": 0, "dy": 0},
            {"id": "b", "kind": "translate", "source": "a", "dx": 0, "dy": 0},
        ],
        "a",
    )
    diagnostic = _diagnostic(payload)
    assert diagnostic.code == "planar-ir.cycle-detected"  # type: ignore[attr-defined]
    assert diagnostic.message == "Cycle detected: a -> b -> a."  # type: ignore[attr-defined]


def test_nested_patterns_cannot_bypass_global_expansion_budget() -> None:
    payload = _construction(
        [
            _rectangle("seed"),
            {
                "id": "inner",
                "kind": "linear_pattern",
                "seed": "seed",
                "count": 16,
                "step_x": 1,
                "step_y": 0,
            },
            {
                "id": "result",
                "kind": "linear_pattern",
                "seed": "inner",
                "count": 16,
                "step_x": 0,
                "step_y": 1,
            },
        ]
    )
    diagnostic = _diagnostic(payload)
    assert diagnostic.code == "planar-ir.budget-exceeded"  # type: ignore[attr-defined]
    assert diagnostic.node_id == "result"  # type: ignore[attr-defined]


def test_boolean_wrapper_cannot_hide_nested_pattern_expansion() -> None:
    payload = _construction(
        [
            _rectangle("seed"),
            {
                "id": "inner",
                "kind": "linear_pattern",
                "seed": "seed",
                "count": 16,
                "step_x": 1,
                "step_y": 0,
            },
            {"id": "wrapped", "kind": "union", "operands": ["inner"]},
            {
                "id": "result",
                "kind": "linear_pattern",
                "seed": "wrapped",
                "count": 16,
                "step_x": 0,
                "step_y": 1,
            },
        ]
    )

    diagnostic = _diagnostic(payload)

    assert diagnostic.code == "planar-ir.budget-exceeded"  # type: ignore[attr-defined]
    assert diagnostic.node_id == "result"  # type: ignore[attr-defined]


def test_depth_and_collection_budgets_are_frozen() -> None:
    assert (
        MAX_NODES,
        MAX_BOOLEAN_OPERANDS,
        MAX_POLYGON_VERTICES,
        MAX_PATH_POINTS,
        MAX_PATTERN_INSTANCES,
        MAX_DAG_DEPTH,
        MAX_CANONICAL_PAYLOAD_BYTES,
    ) == (64, 16, 128, 64, 256, 16, 32_768)

    nodes = [_rectangle("n0")]
    for index in range(MAX_DAG_DEPTH):
        nodes.append(
            {
                "id": f"n{index + 1}",
                "kind": "translate",
                "source": f"n{index}",
                "dx": 1,
                "dy": 0,
            }
        )
    diagnostic = _diagnostic(_construction(nodes))
    assert diagnostic.code == "planar-ir.budget-exceeded"  # type: ignore[attr-defined]
    assert "depth 17" in diagnostic.message  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "node",
    [
        {
            "id": "result",
            "kind": "polygon",
            "vertices": [[0, 0], [2, 2], [0, 2], [2, 0]],
        },
        {
            "id": "result",
            "kind": "path_stroke",
            "points": [[0, 0], [2, 2], [0, 2], [2, 0]],
            "width": 1,
            "cap": "butt",
            "join": "miter",
        },
    ],
)
def test_self_intersecting_polygon_and_path_are_rejected(
    node: dict[str, object],
) -> None:
    diagnostic = _diagnostic(_construction([node]))
    assert diagnostic.code in {  # type: ignore[attr-defined]
        "planar-ir.invalid-primitive",
        "planar-ir.invalid-path-stroke",
    }


def test_path_adjacent_reversal_overlap_is_rejected() -> None:
    payload = _construction(
        [
            {
                "id": "result",
                "kind": "path_stroke",
                "points": [[0, 0], [2, 0], [1, 0]],
                "width": 1,
                "cap": "butt",
                "join": "miter",
            }
        ]
    )

    diagnostic = _diagnostic(payload)

    assert diagnostic.code == "planar-ir.invalid-path-stroke"  # type: ignore[attr-defined]
    assert diagnostic.node_id == "result"  # type: ignore[attr-defined]
    assert diagnostic.message == (  # type: ignore[attr-defined]
        "Path segments 0 and 1 overlap at a reversal."
    )


def test_canonical_payload_is_provider_safe_json_within_budget() -> None:
    ir = PlanarConstructionIR.from_dict(_generic_shape_fixtures()[0])
    canonical = ir.canonical_json()
    decoded = json.loads(canonical)

    assert decoded["schema_version"] == 1
    assert len(canonical.encode("utf-8")) <= MAX_CANONICAL_PAYLOAD_BYTES
    assert "NaN" not in canonical
    assert "Infinity" not in canonical


def test_schema_version_and_unknown_top_level_fields_are_rejected() -> None:
    payload = _construction([_rectangle("result")])
    payload["schema_version"] = 2
    assert _diagnostic(payload).code == "planar-ir.schema-invalid"  # type: ignore[attr-defined]

    payload = _construction([_rectangle("result")])
    payload["future"] = True
    diagnostic = _diagnostic(payload)
    assert diagnostic.code == "planar-ir.schema-invalid"  # type: ignore[attr-defined]
    assert "future" in diagnostic.message  # type: ignore[attr-defined]
