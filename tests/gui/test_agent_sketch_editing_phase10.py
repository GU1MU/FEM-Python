from __future__ import annotations

import math

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, UnitContext, derive_feature_history
from fem.core.model import MaterialDefinition
from fem.geometry import (
    BooleanGeometry,
    ExtrudedGeometry,
    SketchArc,
    SketchAngleDimension,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchConcentricConstraint,
    SketchDistanceDimension,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPerpendicularConstraint,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchTangentConstraint,
    SketchVerticalConstraint,
    analyze_sketch_profiles,
    resolve_extrusion_source_faces,
)
from fem_agent.authoring import AuthoringContractError, ProposalState
from fem_agent.authoring_runtime import _PREPARE_GEOMETRY_EDIT
from fem_agent.geometry_authoring import (
    GeometryDraft,
    add_planar_constraint,
    add_planar_line,
    apply_planar_edit_batch,
    delete_planar_constraints,
    delete_planar_curves,
    geometry_recipe_from_payload,
    geometry_recipe_to_payload,
    geometry_draft,
    planar_geometry_catalog,
    replace_planar_constraint,
    update_planar_arc,
    update_planar_circle,
    update_planar_line,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.main_window import FEMMainWindow


def _square(*constraints: object) -> SketchGeometry:
    return SketchGeometry(
        "Phase 10 sketch",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 1.0, 0.0),
            SketchPoint("P3", 1.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
        constraints,
    )


def _controller(session: ModelSession):
    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, lambda: None))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    return controller, bridge


def test_all_constraint_payloads_round_trip_with_exact_planar_fields() -> None:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 2.0, 2.0),
        SketchPoint("P4", 0.0, 2.0),
        SketchPoint("PC", 1.0, 1.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
        SketchCircle("C1", "PC", 0.5),
        SketchCircle("C2", "PC", 0.75),
    )
    constraints = (
        SketchCoincidentConstraint("K1", "P1", "P2", enabled=False),
        SketchPointOnCurveConstraint("K2", "P3", "L1", enabled=False),
        SketchHorizontalConstraint("K3", "L1", enabled=False),
        SketchVerticalConstraint("K4", "L2", enabled=False),
        SketchParallelConstraint("K5", "L1", "L3", enabled=False),
        SketchPerpendicularConstraint("K6", "L1", "L2", enabled=False),
        SketchEqualLengthConstraint("K7", "L1", "L2", enabled=False),
        SketchTangentConstraint("K8", "L1", "C1", -1, enabled=False),
        SketchEqualRadiusConstraint("K9", "C1", "C2", enabled=False),
        SketchConcentricConstraint("K10", "C1", "C2", enabled=False),
        SketchFixedConstraint("K11", "P1", 0.0, 0.0, enabled=False),
        SketchDistanceDimension("K12", "P1", "P2", 2.0, False, enabled=False),
        SketchRadiusDimension("K13", "C1", 0.5, False, enabled=False),
        SketchAngleDimension("K14", "L1", "L2", math.pi / 2.0, False, enabled=False),
    )
    sketch = SketchGeometry(
        "all constraints", SketchPlane.xy(), points, curves, constraints
    )

    payload = geometry_recipe_to_payload(sketch)

    assert set(payload) == {
        "schema_version",
        "kind",
        "name",
        "plane",
        "points",
        "curves",
        "constraints",
    }
    assert [item["kind"] for item in payload["constraints"]] == [
        "coincident",
        "point_on_curve",
        "horizontal",
        "vertical",
        "parallel",
        "perpendicular",
        "equal_length",
        "tangent",
        "equal_radius",
        "concentric",
        "fixed",
        "distance",
        "radius",
        "angle",
    ]
    assert geometry_recipe_from_payload(payload) == sketch


def test_planar_constraint_bound_is_reachable_nested_and_strict() -> None:
    constraints = tuple(
        SketchHorizontalConstraint(f"K{index}", "L1", enabled=False)
        for index in range(1, 129)
    )
    bounded = _square(*constraints)

    payload = geometry_recipe_to_payload(bounded)
    nested = geometry_recipe_to_payload(
        ExtrudedGeometry(bounded, 2.0, ("face:domain",))
    )

    assert len(payload["constraints"]) == 128
    assert len(nested["base"]["constraints"]) == 128
    assert geometry_recipe_from_payload(payload) == bounded
    assert geometry_draft(bounded).recipe == bounded

    oversized = _square(
        *constraints,
        SketchVerticalConstraint("K129", "L2", enabled=False),
    )
    with pytest.raises(ValueError, match="constraints exceed"):
        geometry_recipe_to_payload(oversized)
    with pytest.raises(ValueError, match="128-entity bound"):
        GeometryDraft(oversized, {}, geometry_draft(_square()).preview, {})


def test_planar_payload_requires_canonical_constraints_and_exact_constraint_fields() -> (
    None
):
    empty_payload = geometry_recipe_to_payload(_square())
    assert empty_payload["constraints"] == []
    assert geometry_recipe_from_payload(empty_payload) == _square()

    missing_constraints = dict(empty_payload)
    missing_constraints.pop("constraints")
    with pytest.raises(ValueError, match="fields do not match"):
        geometry_recipe_from_payload(missing_constraints)

    constrained_payload = geometry_recipe_to_payload(
        _square(SketchFixedConstraint("K1", "P1", 0.0, 0.0))
    )
    for field in ("enabled", "point_id"):
        malformed = dict(constrained_payload)
        constraint = dict(malformed["constraints"][0])
        constraint.pop(field)
        malformed["constraints"] = [constraint]
        with pytest.raises(ValueError, match="constraint fields"):
            geometry_recipe_from_payload(malformed)

    extra = dict(constrained_payload)
    constraint = dict(extra["constraints"][0])
    constraint["unexpected"] = True
    extra["constraints"] = [constraint]
    with pytest.raises(ValueError, match="constraint fields"):
        geometry_recipe_from_payload(extra)


def test_catalog_exposes_constraint_rows_and_solver_diagnostics() -> None:
    sketch = _square(SketchAngleDimension("K1", "L1", "L2", math.pi / 2.0))

    catalog = planar_geometry_catalog(sketch)

    assert catalog["constraint_summary"]["capability"] == {
        "read": True,
        "create": True,
        "edit": True,
    }
    assert catalog["constraints"][0]["angle_degrees"] == pytest.approx(90.0)
    assert set(catalog["solve"]) == {
        "status",
        "remaining_dof",
        "max_residual",
        "redundant_constraint_ids",
        "conflicting_constraint_ids",
    }


def test_circle_radius_edit_rejects_solver_restoration_by_driving_constraint() -> None:
    sketch = SketchGeometry(
        "radius authority",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 10.0, 0.0),
            SketchPoint("P3", 10.0, 10.0),
            SketchPoint("P4", 0.0, 10.0),
            SketchPoint("PC", 5.0, 5.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchCircle("C1", "PC", 1.0),
        ),
        (SketchRadiusDimension("K1", "C1", 1.0),),
    )

    with pytest.raises(
        AuthoringContractError,
        match="driving or equality constraint.*replace or remove",
    ):
        update_planar_circle(sketch, circle_id="C1", radius=2.0)

    assert sketch.curve("C1").radius == 1.0


def test_batch_may_open_temporarily_but_direct_delete_must_remain_exact() -> None:
    sketch = _square()
    with pytest.raises(AuthoringContractError, match="sketch.open-loop"):
        delete_planar_curves(sketch, curve_ids=["L4"])

    draft = apply_planar_edit_batch(
        sketch,
        edits=(
            {"operation": "delete_curves", "curve_ids": ["L4"]},
            {
                "operation": "add_line",
                "start": {"point_id": "P4"},
                "end": {"point_id": "P1"},
            },
        ),
    )

    assert draft.proof.exact is True
    assert [curve.id for curve in draft.recipe.curves] == ["L1", "L2", "L3", "L4"]


def test_coordinate_point_ref_rejects_ambiguous_existing_points() -> None:
    square = _square()
    ambiguous = SketchGeometry(
        square.name,
        square.plane,
        (*square.points, SketchPoint("P5", 0.0, 0.0)),
        square.curves,
    )

    with pytest.raises(AuthoringContractError, match="multiple points; use point_id"):
        add_planar_line(
            ambiguous,
            start={"x": 0.0, "y": 0.0},
            end={"point_id": "P2"},
        )


def test_batch_builds_inner_d_hole_with_three_reused_coordinate_points() -> None:
    sketch = SketchGeometry(
        "outer rectangle",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 10.0, 0.0),
            SketchPoint("P3", 10.0, 10.0),
            SketchPoint("P4", 0.0, 10.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )

    draft = apply_planar_edit_batch(
        sketch,
        edits=(
            {
                "operation": "add_arc",
                "start": {"x": 2.0, "y": 2.0},
                "center": {"x": 3.0, "y": 2.0},
                "end": {"x": 4.0, "y": 2.0},
                "orientation": "ccw",
            },
            {
                "operation": "add_line",
                "start": {"x": 4.0, "y": 2.0},
                "end": {"x": 2.0, "y": 2.0},
            },
        ),
    )

    assert draft.proof.exact is True
    assert len(draft.recipe.points) == 7
    assert any(
        isinstance(curve, SketchArc) and curve.id == "A1"
        for curve in draft.recipe.curves
    )
    catalog = planar_geometry_catalog(draft.recipe)
    assert catalog["point_count"] == 7
    assert catalog["curve_count"] == 6
    assert [
        profile.role for profile in analyze_sketch_profiles(draft.recipe).profiles
    ].count("hole") == 1


@pytest.mark.parametrize(
    ("edit", "expected_code"),
    (
        (
            {
                "operation": "add_line",
                "start": {"x": 0.2, "y": 0.2},
                "end": {"x": 0.8, "y": 0.8},
            },
            "sketch.open-loop",
        ),
        (
            {
                "operation": "add_polygon",
                "vertices": [
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.8, "y": 0.8},
                    {"x": 0.2, "y": 0.8},
                    {"x": 0.8, "y": 0.2},
                ],
            },
            "sketch.crossing",
        ),
    ),
    ids=("open-contour", "self-intersecting-contour"),
)
def test_invalid_freeform_profile_returns_actionable_diagnostics_atomically(
    edit: dict[str, object],
    expected_code: str,
) -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "invalid freeform", UnitContext("mm", "N", "MPa"), _square()
    )
    controller, bridge = _controller(session)
    before = session.snapshot()

    result = controller.dispatch(
        "prepare_geometry_edit",
        {"part_id": "P1", "edit": edit},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            f"phase10-invalid-{expected_code.replace('.', '-')}",
        ),
    )

    assert not result.ok
    assert result.data is not None
    assert result.data["kind"] == "planar_edit_validation"
    assert result.data["status"] == "rejected"
    assert result.data["retry_guidance"]["action"] == (
        "revise_and_retry_same_geometry_edit"
    )
    codes = {item["code"] for item in result.data["diagnostics"]}
    assert expected_code in codes
    assert any(
        item["affected_logical_ids"]
        for item in result.data["diagnostics"]
        if item["code"] == expected_code
    )
    assert bridge._records == {}
    assert controller.stage.value == "mesh_ready"
    assert session.snapshot() == before


def test_freeform_profile_policy_guides_and_verifies_one_nonconvex_cutout() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "freeform cutout", UnitContext("mm", "N", "MPa"), _square()
    )
    controller, bridge = _controller(session)

    context = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-freeform-read-before",
        ),
    )
    assert context.ok
    assert context.data["profile_summary"]["hole_count"] == 0
    policy = context.data["freeform_profile_policy"]
    assert policy["two_dimensional_cut_representation"] == "closed_inner_profile"
    assert policy["part_boolean_required"] is False
    assert policy["preferred_operation_for_arbitrary_silhouette"] == "add_polygon"
    assert policy["preferred_operation_for_constant_width_slot"] == "add_path_slot"
    assert policy["constant_width_slot_operation_priority"] == [
        "add_path_slot",
        "planar_boolean(tool.kind=path_stroke)",
    ]
    assert policy["planar_boolean_path_stroke_role"] == "lower_level_equivalent"
    assert policy["primary_operation_for_closed_boundary_slot"] == "add_polygon"
    assert policy["representation_priority"] == [
        "add_path_slot_when_one_open_non_branching_centerline_"
        "and_constant_width_fully_define_the_slot",
        "add_polygon_for_other_single_closed_slot_boundaries",
        "ordered_lines_and_arcs_when_exact_curves_are_required",
    ]
    assert policy["rectangle_decomposition_for_one_connected_slot"] == "avoid"

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_polygon",
                "vertices": [
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.8, "y": 0.4},
                    {"x": 0.4, "y": 0.4},
                    {"x": 0.4, "y": 0.8},
                    {"x": 0.2, "y": 0.8},
                ],
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-freeform-prepare",
        ),
    )
    assert prepared.ok, prepared.summary
    receipt = bridge.accept_from_gui_control(str(prepared.data["proposal_id"]))
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    verified = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-freeform-read-after",
        ),
    )

    assert verified.ok
    summary = verified.data["profile_summary"]
    assert summary["topology_exact"] is True
    assert summary["profile_count"] == 2
    assert summary["hole_count"] == 1
    hole = next(item for item in summary["profiles"] if item["role"] == "hole")
    assert hole["curve_count"] == 6
    assert hole["bounding_box"] == [0.2, 0.2, 0.8, 0.8]


def test_path_slot_edit_builds_one_connected_s_cutout_and_enforces_topology() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "S path slot", UnitContext("mm", "N", "MPa"), _square()
    )
    controller, bridge = _controller(session)
    arguments = {
        "part_id": "P1",
        "edit": {
            "operation": "add_path_slot",
            "points": [
                {"x": 0.75, "y": 0.80},
                {"x": 0.30, "y": 0.80},
                {"x": 0.30, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.75, "y": 0.20},
                {"x": 0.30, "y": 0.20},
            ],
            "width": 0.08,
            "cap": "square",
            "join": "miter",
        },
    }

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        arguments,
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-path-slot-prepare",
        ),
    )

    assert prepared.ok, prepared.summary
    receipt = bridge.accept_from_gui_control(str(prepared.data["proposal_id"]))
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    recipe = session.snapshot().parts[0].geometry_recipe
    assert isinstance(recipe, BooleanGeometry)
    assert recipe.operation == "cut"
    assert isinstance(recipe.object_geometry, SketchGeometry)
    tool_analysis = analyze_sketch_profiles(recipe.tool_geometry)
    assert tool_analysis.valid
    profiles = [profile for profile in tool_analysis.profiles if profile.is_material]
    assert len(profiles) == 1
    assert profiles[0].bounding_box == pytest.approx((0.26, 0.16, 0.79, 0.84))

    outside = {
        **arguments,
        "edit": {
            **arguments["edit"],
            "points": [{"x": 1.2, "y": 0.2}, {"x": 1.2, "y": 0.8}],
        },
    }
    rejected = controller.dispatch(
        "prepare_geometry_edit",
        outside,
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-path-slot-outside",
        ),
    )
    assert not rejected.ok
    assert rejected.data["status"] == "rejected"
    assert rejected.data["diagnostics"][0]["code"] == ("sketch.path-slot-postcondition")


def test_path_slot_adds_s_beside_existing_h_cutout_and_preserves_four_holes() -> None:
    outer = SketchGeometry(
        "H plate",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 300.0, 0.0),
            SketchPoint("P3", 300.0, 100.0),
            SketchPoint("P4", 0.0, 100.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )
    base = apply_planar_edit_batch(
        outer,
        edits=(
            {
                "operation": "add_polygon",
                "vertices": [
                    {"x": 110, "y": 10},
                    {"x": 130, "y": 10},
                    {"x": 130, "y": 40},
                    {"x": 170, "y": 40},
                    {"x": 170, "y": 10},
                    {"x": 190, "y": 10},
                    {"x": 190, "y": 90},
                    {"x": 170, "y": 90},
                    {"x": 170, "y": 60},
                    {"x": 130, "y": 60},
                    {"x": 130, "y": 90},
                    {"x": 110, "y": 90},
                ],
            },
            *(
                {
                    "operation": "add_circle",
                    "center_x": x,
                    "center_y": y,
                    "radius": 1.5,
                }
                for x, y in ((5, 5), (295, 5), (5, 95), (295, 95))
            ),
        ),
    ).recipe
    session = ModelSession()
    session.create_native_project_with_first_part(
        "H plate plus S", UnitContext("mm", "N", "MPa"), base
    )
    controller, bridge = _controller(session)

    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_path_slot",
                "points": [
                    {"x": 75, "y": 82},
                    {"x": 40, "y": 82},
                    {"x": 40, "y": 50},
                    {"x": 75, "y": 50},
                    {"x": 75, "y": 18},
                    {"x": 40, "y": 18},
                ],
                "width": 6,
                "cap": "square",
                "join": "miter",
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-h-plus-s-prepare",
        ),
    )

    assert prepared.ok, prepared.summary
    receipt = bridge.accept_from_gui_control(str(prepared.data["proposal_id"]))
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert isinstance(recipe, BooleanGeometry)
    assert recipe.operation == "cut"
    original_analysis = analyze_sketch_profiles(recipe.object_geometry)
    assert original_analysis.valid
    original_holes = [
        profile for profile in original_analysis.profiles if profile.is_hole
    ]
    assert len(original_holes) == 5
    assert any(
        profile.bounding_box == pytest.approx((110.0, 10.0, 190.0, 90.0))
        for profile in original_holes
    )
    tool_analysis = analyze_sketch_profiles(recipe.tool_geometry)
    tool_profiles = [
        profile for profile in tool_analysis.profiles if profile.is_material
    ]
    assert len(tool_profiles) == 1
    assert any(
        profile.bounding_box == pytest.approx((37.0, 15.0, 78.0, 85.0))
        for profile in tool_profiles
    )


def test_planar_fuse_and_extrusion_append_native_features_in_order() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "feature chain",
        UnitContext("mm", "N", "MPa"),
        _square(),
    )
    controller, bridge = _controller(session)
    fused = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "planar_boolean",
                "boolean_operation": "fuse",
                "tool": {
                    "kind": "rectangle",
                    "x": 0.8,
                    "y": 0.2,
                    "width": 0.5,
                    "height": 0.6,
                },
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-planar-fuse",
        ),
    )
    assert fused.ok, fused.summary
    fuse_receipt = bridge.accept_from_gui_control(str(fused.data["proposal_id"]))
    assert fuse_receipt.state is ProposalState.SUCCEEDED
    bridge.bind_snapshot(session.snapshot())
    controller.observe_binding(bridge.context)
    controller.record_proposal_state(
        "geometry", fuse_receipt.state, fuse_receipt.message
    )
    planar_recipe = session.snapshot().parts[0].geometry_recipe
    assert [item.kind for item in derive_feature_history(planar_recipe)] == [
        "sketch",
        "fuse",
    ]

    source_face_ids = resolve_extrusion_source_faces(planar_recipe).face_ids
    extruded = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "extrude_profiles",
                "source_face_ids": list(source_face_ids),
                "height": 2.0,
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-feature-extrusion",
        ),
    )
    assert extruded.ok, extruded.summary
    assert (
        bridge.accept_from_gui_control(str(extruded.data["proposal_id"])).state
        is ProposalState.SUCCEEDED
    )
    solid_recipe = session.snapshot().parts[0].geometry_recipe
    assert type(solid_recipe) is ExtrudedGeometry
    assert [item.kind for item in derive_feature_history(solid_recipe)] == [
        "sketch",
        "fuse",
        "extrude",
    ]


def test_multi_turn_catalog_ids_drive_line_then_arc_edits() -> None:
    original = apply_planar_edit_batch(
        SketchGeometry(
            "outer rectangle",
            SketchPlane.xy(),
            (
                SketchPoint("P1", 0.0, 0.0),
                SketchPoint("P2", 10.0, 0.0),
                SketchPoint("P3", 10.0, 10.0),
                SketchPoint("P4", 0.0, 10.0),
            ),
            (
                SketchLine("L1", "P1", "P2"),
                SketchLine("L2", "P2", "P3"),
                SketchLine("L3", "P3", "P4"),
                SketchLine("L4", "P4", "P1"),
            ),
        ),
        edits=(
            {
                "operation": "add_arc",
                "start": {"x": 2.0, "y": 2.0},
                "center": {"x": 3.0, "y": 2.0},
                "end": {"x": 4.0, "y": 2.0},
                "orientation": "ccw",
            },
            {
                "operation": "add_line",
                "start": {"x": 4.0, "y": 2.0},
                "end": {"x": 2.0, "y": 2.0},
            },
        ),
    ).recipe
    first = planar_geometry_catalog(original)
    line = next(item for item in first["curves"] if item["id"] == "L1")

    line_edit = update_planar_line(
        original,
        line_id=line["id"],
        start={"point_id": line["end_point_id"]},
        end={"point_id": line["start_point_id"]},
    )
    second = planar_geometry_catalog(line_edit.recipe)
    edited_line = next(item for item in second["curves"] if item["id"] == "L1")
    arc = next(item for item in second["curves"] if item["kind"] == "arc")

    arc_edit = update_planar_arc(
        line_edit.recipe,
        arc_id=arc["id"],
        orientation="cw",
    )
    third = planar_geometry_catalog(arc_edit.recipe)

    assert line_edit.proof.exact and arc_edit.proof.exact
    assert edited_line["start_point_id"] == line["end_point_id"]
    assert edited_line["end_point_id"] == line["start_point_id"]
    assert (
        next(item for item in third["curves"] if item["id"] == arc["id"])["orientation"]
        == "cw"
    )


def test_agent_constraint_specs_cover_all_fourteen_kinds() -> None:
    square = _square()
    sketch = SketchGeometry(
        square.name,
        square.plane,
        (
            *square.points,
            SketchPoint("P5", 0.25, 0.25),
            SketchPoint("P6", 0.75, 0.75),
        ),
        (
            *square.curves,
            SketchCircle("C1", "P5", 0.1),
            SketchCircle("C2", "P6", 0.1),
        ),
    )
    specs = (
        {"kind": "coincident", "first_point_id": "P1", "second_point_id": "P2"},
        {"kind": "point_on_curve", "point_id": "P3", "curve_id": "L1"},
        {"kind": "horizontal", "line_id": "L1"},
        {"kind": "vertical", "line_id": "L2"},
        {"kind": "parallel", "first_line_id": "L1", "second_line_id": "L3"},
        {"kind": "perpendicular", "first_line_id": "L1", "second_line_id": "L2"},
        {"kind": "equal_length", "first_line_id": "L1", "second_line_id": "L2"},
        {
            "kind": "tangent",
            "first_curve_id": "L1",
            "second_curve_id": "C1",
            "branch_hint": 0,
        },
        {"kind": "equal_radius", "first_curve_id": "C1", "second_curve_id": "C2"},
        {"kind": "concentric", "first_curve_id": "C1", "second_curve_id": "C2"},
        {"kind": "fixed", "point_id": "P1"},
        {
            "kind": "distance",
            "first_point_id": "P1",
            "second_point_id": "P2",
            "value": 1.0,
            "driving": False,
        },
        {"kind": "radius", "curve_id": "C1", "value": 0.1, "driving": False},
        {
            "kind": "angle",
            "first_line_id": "L1",
            "second_line_id": "L2",
            "angle_degrees": 90.0,
            "driving": False,
        },
    )

    for spec in specs:
        sketch = add_planar_constraint(
            sketch, constraint={**spec, "enabled": False}
        ).recipe

    assert [item.id for item in sketch.constraints] == [
        f"K{index}" for index in range(1, 15)
    ]


def test_enabled_constraint_lifecycle_solves_replaces_and_deletes_exact_id() -> None:
    anchored = add_planar_constraint(
        _square(), constraint={"kind": "fixed", "point_id": "P1"}
    ).recipe
    horizontal = add_planar_constraint(
        anchored, constraint={"kind": "horizontal", "line_id": "L1"}
    ).recipe
    dimensioned = add_planar_constraint(
        horizontal,
        constraint={
            "kind": "distance",
            "first_point_id": "P1",
            "second_point_id": "P2",
            "value": 2.0,
        },
    ).recipe

    assert math.dist(
        (dimensioned.point("P1").u, dimensioned.point("P1").v),
        (dimensioned.point("P2").u, dimensioned.point("P2").v),
    ) == pytest.approx(2.0)
    replaced = replace_planar_constraint(
        dimensioned,
        constraint_id="K3",
        constraint={
            "kind": "distance",
            "first_point_id": "P1",
            "second_point_id": "P2",
            "value": 3.0,
        },
    ).recipe
    assert replaced.constraints[-1].id == "K3"
    assert replaced.constraints[-1].value == 3.0
    assert math.dist(
        (replaced.point("P1").u, replaced.point("P1").v),
        (replaced.point("P2").u, replaced.point("P2").v),
    ) == pytest.approx(3.0)

    deleted = delete_planar_constraints(replaced, constraint_ids=["K3"]).recipe
    assert [constraint.id for constraint in deleted.constraints] == ["K1", "K2"]


def test_conflicting_constraint_tool_result_is_atomic_and_registers_no_proposal() -> (
    None
):
    source = _square(
        SketchFixedConstraint("F1", "P1", 0.0, 0.0),
        SketchFixedConstraint("F2", "P2", 1.0, 0.0),
        SketchHorizontalConstraint("H1", "L1"),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "conflict", UnitContext("mm", "N", "MPa"), source
    )
    controller, bridge = _controller(session)
    before = session.snapshot()

    result = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_constraint",
                "constraint": {"kind": "vertical", "line_id": "L1"},
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-conflict",
        ),
    )

    assert not result.ok
    assert result.data is not None
    assert result.data["error"]["code"] == "geometry-edit.execution-rejected"
    assert result.data["error"]["failed_operation"] == "add_constraint"
    assert "conflict" in result.data["error"]["reason"]
    assert bridge._records == {}
    assert session.snapshot() == before


def test_gui_proposal_acceptance_retains_constraint_payload_round_trip() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "accepted constraint", UnitContext("mm", "N", "MPa"), _square()
    )
    controller, bridge = _controller(session)
    before_revision = session.session_revision
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_constraint",
                "constraint": {"kind": "horizontal", "line_id": "L1"},
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "phase10-accept",
        ),
    )

    assert prepared.ok, prepared.summary
    assert session.session_revision == before_revision
    receipt = bridge.accept_from_gui_control(str(prepared.data["proposal_id"]))
    accepted = session.snapshot().parts[0].geometry_recipe

    assert receipt.state is ProposalState.SUCCEEDED
    assert session.session_revision == before_revision + 1
    assert isinstance(accepted, SketchGeometry)
    assert accepted.constraints == (SketchHorizontalConstraint("K1", "L1"),)


def test_new_constraint_edit_uses_phase7_branch_migration_semantics() -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    window.session.create_native_project_with_first_part(
        "branch constraint", UnitContext("mm", "N", "MPa"), _square()
    )
    window.session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (),
        (),
        (),
    )
    window._rebuild_full_projection()
    source = window.workspace.active_document()
    assert source is not None
    window._bind_agent_document(source)
    source_before = source.session.snapshot()

    prepared = window.agent_authoring_controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_constraint",
                "constraint": {"kind": "horizontal", "line_id": "L1"},
            },
        },
        ToolExecutionContext(
            window.session.session_id,
            window.session.session_revision,
            "phase10-branch",
        ),
    )

    assert prepared.ok, prepared.summary
    assert prepared.data["geometry_edit_mode"] == "branch"
    proposal_id = str(prepared.data["proposal_id"])
    proposal = window.agent_authoring_bridge._records[proposal_id].proposal
    assert proposal.expected_changes["creates_iteration_model"] is True
    assert proposal.invalidation_impact["results"] is False
    receipt = window.agent_authoring_bridge.accept_from_gui_control(proposal_id)

    child = window.workspace.active_document()
    assert receipt.state is ProposalState.SUCCEEDED
    assert child is not None and child.document_id != source.document_id
    assert source.session.snapshot() == source_before
    child_snapshot = child.session.snapshot()
    child_recipe = child_snapshot.parts[0].geometry_recipe
    assert isinstance(child_recipe, SketchGeometry)
    assert child_recipe.constraints == (SketchHorizontalConstraint("K1", "L1"),)
    assert [material.name for material in child_snapshot.materials] == ["Steel"]
    report = window.agent_authoring_bridge.port.latest_geometry_iteration_report()
    assert report["mode"] == "branch"
    assert report["runs"] == "not_migrated"
    assert report["results"] == "not_migrated"
    window.close()


def test_prepare_geometry_edit_schema_advertises_all_phase10_operations() -> None:
    edit_schema = _PREPARE_GEOMETRY_EDIT.parameters["properties"]["edit"]
    assert "add_path_slot as the preferred geometry-edit entry" in (
        _PREPARE_GEOMETRY_EDIT.description
    )
    assert "planar_boolean(tool.kind=path_stroke)" in (
        _PREPARE_GEOMETRY_EDIT.description
    )
    spatial_relation = _PREPARE_GEOMETRY_EDIT.parameters["properties"][
        "spatial_relation"
    ]
    assert spatial_relation["properties"]["relation"]["enum"] == [
        "above",
        "below",
        "left_of",
        "right_of",
    ]
    assert spatial_relation["required"] == [
        "reference_feature_id",
        "relation",
        "clearance",
    ]
    operations = {
        branch["properties"]["operation"]["const"] for branch in edit_schema["oneOf"]
    }

    add_path_slot = next(
        branch
        for branch in edit_schema["oneOf"]
        if branch["properties"]["operation"]["const"] == "add_path_slot"
    )
    path_points_description = add_path_slot["properties"]["points"]["description"]
    assert 'straight [{"x":10,"y":10},{"x":40,"y":10}]' in (
        path_points_description
    )
    assert 'multiple bends [{"x":10,"y":10}' in path_points_description
    assert "Points describe the centerline itself" in path_points_description
    assert "first and last points must differ" in path_points_description

    planar_boolean = next(
        branch
        for branch in edit_schema["oneOf"]
        if branch["properties"]["operation"]["const"] == "planar_boolean"
    )
    assert "Low-level cut/fuse profile" in (
        planar_boolean["properties"]["tool"]["description"]
    )
    assert "prefer edit.operation=add_path_slot" in (
        planar_boolean["properties"]["tool"]["description"]
    )
    path_stroke = next(
        tool
        for tool in planar_boolean["properties"]["tool"]["oneOf"]
        if tool["properties"]["kind"]["const"] == "path_stroke"
    )
    boolean_points_description = path_stroke["properties"]["points"]["description"]
    assert boolean_points_description.split("Centerline point examples:", 1)[1] == (
        path_points_description.split("Centerline point examples:", 1)[1]
    )

    assert {
        "add_line",
        "add_arc",
        "add_path_slot",
        "update_line",
        "update_arc",
        "delete_curves",
        "add_constraint",
        "replace_constraint",
        "delete_constraints",
        "batch",
    } <= operations
    batch = next(
        branch
        for branch in edit_schema["oneOf"]
        if branch["properties"]["operation"]["const"] == "batch"
    )
    batch_operations = {
        branch["properties"]["operation"]["const"]
        for branch in batch["properties"]["edits"]["items"]["oneOf"]
    }
    assert (
        operations
        - {
            "translate",
            "rotate",
            "planar_boolean",
            "part_boolean",
            "body_boolean",
            "batch",
            "add_path_slot",
            "replace_planar_boolean_feature",
        }
        <= batch_operations
    )
    expected_kinds = {
        "coincident",
        "point_on_curve",
        "horizontal",
        "vertical",
        "parallel",
        "perpendicular",
        "equal_length",
        "tangent",
        "equal_radius",
        "concentric",
        "fixed",
        "distance",
        "radius",
        "angle",
    }
    for operation in ("add_constraint", "replace_constraint"):
        branch = next(
            item
            for item in edit_schema["oneOf"]
            if item["properties"]["operation"]["const"] == operation
        )
        kinds = {
            item["properties"]["kind"]["const"]
            for item in branch["properties"]["constraint"]["oneOf"]
        }
        assert kinds == expected_kinds

        batch_branch = next(
            item
            for item in batch["properties"]["edits"]["items"]["oneOf"]
            if item["properties"]["operation"]["const"] == operation
        )
        batch_kinds = {
            item["properties"]["kind"]["const"]
            for item in batch_branch["properties"]["constraint"]["oneOf"]
        }
        assert batch_kinds == expected_kinds
