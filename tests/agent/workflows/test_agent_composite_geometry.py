from __future__ import annotations

import pytest

from fem.application import ModelSession
from fem.geometry import (
    ExtrudedGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
)
from fem_agent.authoring import (
    AuthoringAuthorizationError,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalState,
)
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.geometry_authoring import geometry_recipe_to_payload
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


pytestmark = pytest.mark.local_session


def _controller(session: ModelSession):
    holder: dict[str, object] = {}

    def refresh() -> None:
        bridge.bind_snapshot(session.snapshot())
        controller = holder.get("controller")
        if controller is not None:
            controller.observe_binding(bridge.context)  # type: ignore[arg-type]

    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, refresh))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holder["controller"] = controller
    return bridge, controller


def _ring_geometry(
    height: float = 100.0,
    *,
    provisional: bool = False,
) -> dict[str, object]:
    geometry: dict[str, object] = {
        "kind": "extruded_profiles",
        "profiles": [
            {
                "kind": "circle",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 50.0,
                "role": "material",
            },
            {
                "kind": "circle",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 25.0,
                "role": "hole",
            },
        ],
        "height": height,
    }
    if provisional:
        geometry["provisional"] = True
    return geometry


def test_runtime_schema_retires_blank_composite_variants() -> None:
    _bridge, controller = _controller(ModelSession())
    definition = next(
        item
        for item in controller.definitions
        if item.name == "prepare_geometry_proposal"
    )
    variants = definition.parameters["properties"]["geometry"]["oneOf"]
    assert {
        item["properties"]["kind"].get("const") for item in variants
    } == {"wire", "box", "cylinder"}
    assert "prepare_planar_construction_proposal" in {
        item.name for item in controller.definitions
    }


def test_duplicate_polygon_vertex_returns_profile_input_diagnostic() -> None:
    session = ModelSession()
    _bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "带槽平板",
            "geometry": {
                "kind": "extruded_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": -5.0,
                        "y": -3.0,
                        "width": 10.0,
                        "height": 6.0,
                    },
                    {
                        "kind": "polygon",
                        "vertices": [
                            {"x": -2.0, "y": 0.0},
                            {"x": 0.0, "y": 1.0},
                            {"x": 2.0, "y": 0.0},
                            {"x": -2.0, "y": 0.0},
                        ],
                    },
                ],
                "height": 1.0,
            },
        },
        ToolExecutionContext("phase4-invalid-profile", 0, "invalid-profile"),
    )

    assert not result.ok
    diagnostic = result.data["diagnostic"]
    assert diagnostic["code"] == "profile-transform.invalid-profile"
    assert diagnostic["required_fields"] == ["profiles"]


@pytest.mark.usefixtures("real_gmsh")
def test_blank_ring_is_one_atomic_final_3d_proposal() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "中空圆柱", "geometry": _ring_geometry()},
        ToolExecutionContext("phase4", before.session_revision, "ring"),
    )

    assert result.ok, result.summary
    assert session.snapshot() == before
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert proposal.display_summary["dimension"] == 3
    assert "holes=1" in proposal.display_summary["summary"]
    assert "100" in proposal.display_summary["summary"]
    assert "provisional" not in proposal.display_summary["summary"]
    assert "50" in proposal.display_summary["summary"]
    assert "25" in proposal.display_summary["summary"]
    assert proposal.display_summary["expected_entity_count"] == 1
    assert proposal.display_summary["expected_new_objects"] == ["部件-中空圆柱"]

    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    accepted = session.snapshot()
    assert accepted.source_kind == "native"
    assert len(accepted.parts) == 1
    recipe = accepted.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    assert recipe.height == 100.0
    assert describe_recipe_topology(recipe).exact
    assert accepted.parts[0].mesh_settings is None


@pytest.mark.usefixtures("real_gmsh")
def test_blank_composite_reject_keeps_session_empty() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "带孔平板", "geometry": {
            "kind": "extruded_profiles",
            "profiles": [
                {"kind": "rectangle", "x": -5.0, "y": -3.0, "width": 10.0, "height": 6.0},
                {"kind": "circle", "center_x": 0.0, "center_y": 0.0, "radius": 1.0},
            ],
            "height": 2.0,
        }},
        ToolExecutionContext("phase4", before.session_revision, "reject"),
    )
    assert result.ok, result.summary
    proposal_id = result.data["proposal_id"]
    assert bridge._records[proposal_id].state is ProposalState.PENDING_CONFIRMATION
    receipt = bridge.reject_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.REJECTED
    assert bridge._records[proposal_id].state is ProposalState.REJECTED
    assert session.snapshot() == before


@pytest.mark.usefixtures("real_gmsh")
def test_center_hole_plate_accepts_without_analysis_side_effects() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "center-hole-plate",
            "geometry": {
                "kind": "extruded_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": -5.0,
                        "y": -3.0,
                        "width": 10.0,
                        "height": 6.0,
                    },
                    {
                        "kind": "circle",
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius": 1.0,
                    },
                ],
                "height": 2.0,
            },
        },
        ToolExecutionContext("phase4", 0, "center-hole"),
    )
    assert result.ok, result.summary
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert all(
        value in proposal.display_summary["summary"]
        for value in ("10", "6", "1", "2")
    )
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    snapshot = session.snapshot()
    recipe = snapshot.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert len(topology.entities_of("body", selectable_only=True)) == 1
    assert {"face:bottom", "face:top"} <= set(topology.signature.logical_ids)
    assert any(entity.semantic_role == "sweep.boundary.hole" for entity in topology.entities)
    assert snapshot.parts[0].mesh_settings is None
    assert snapshot.materials == ()
    assert snapshot.sections == ()
    assert snapshot.assignments == ()
    assert snapshot.steps == ()
    assert snapshot.artifact is None


@pytest.mark.usefixtures("real_gmsh")
def test_contour_order_does_not_choose_material_or_hole() -> None:
    normal = _ring_geometry()["profiles"]
    assert isinstance(normal, list)
    variants = (normal, list(reversed(normal)))
    topologies = []
    for profiles in variants:
        session = ModelSession()
        bridge, controller = _controller(session)
        result = controller.dispatch(
            "prepare_geometry_proposal",
            {
                "part_function": "order-independent",
                "geometry": {
                    "kind": "extruded_profiles",
                    "profiles": profiles,
                    "height": 12.0,
                },
            },
            ToolExecutionContext("phase4", 0, "order"),
        )
        assert result.ok, result.summary
        receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
        assert receipt.state is ProposalState.SUCCEEDED
        recipe = session.snapshot().parts[0].geometry_recipe
        topologies.append(describe_recipe_topology(recipe))
        assert topologies[-1].exact
        assert len(topologies[-1].entities_of("body", selectable_only=True)) == 1
        assert len(topologies[-1].entities_of("face")) == 4
    assert topologies[0].signature == topologies[1].signature


def test_multiple_disjoint_material_profiles_fail_closed() -> None:
    session = ModelSession()
    _bridge, controller = _controller(session)
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "two-plates",
            "geometry": {
                "kind": "extruded_profiles",
                "profiles": [
                    {"kind": "rectangle", "x": 0.0, "y": 0.0, "width": 2.0, "height": 2.0},
                    {"kind": "rectangle", "x": 5.0, "y": 0.0, "width": 2.0, "height": 2.0},
                ],
                "height": 1.0,
            },
        },
        ToolExecutionContext("phase4", before.session_revision, "two-plates"),
    )
    assert not result.ok
    assert session.snapshot() == before


@pytest.mark.usefixtures("real_gmsh")
def test_explicit_provisional_summary_contains_all_dimensions() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    geometry = _ring_geometry(18.0, provisional=True)
    geometry["profiles"] = [
        {
            "kind": "rectangle",
            "x": -21.0,
            "y": -9.0,
            "width": 42.0,
            "height": 18.0,
            "role": "material",
        },
        {
            "kind": "circle",
            "center_x": 0.0,
            "center_y": 0.0,
            "radius": 7.0,
            "role": "hole",
        },
    ]
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "provisional-plate", "geometry": geometry},
        ToolExecutionContext("phase4", 0, "provisional"),
    )
    assert result.ok, result.summary
    summary = bridge._records[result.data["proposal_id"]].proposal.display_summary["summary"]
    assert "42" in summary and "18" in summary and "7" in summary
    assert "provisional" in summary
    bridge.reject_from_gui_control(result.data["proposal_id"])
    assert session.snapshot().source_kind is None


@pytest.mark.usefixtures("real_gmsh")
def test_stale_accept_keeps_blank_session() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "stale-ring", "geometry": _ring_geometry()},
        ToolExecutionContext("phase4", 0, "stale"),
    )
    assert result.ok, result.summary
    proposal_id = result.data["proposal_id"]
    assert bridge.stale_pending_proposals_from_gui("binding changed") == (proposal_id,)
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal_id)
    assert session.snapshot().source_kind is None


def test_failed_composite_preflight_is_not_registered(monkeypatch) -> None:
    import fem_gui.agent_authoring as agent_authoring

    def fail(_recipe):
        raise AuthoringContractError("profile-transform.preflight-failed: test")

    monkeypatch.setattr(agent_authoring, "_preflight_composite_geometry", fail)
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "preflight-fail", "geometry": _ring_geometry()},
        ToolExecutionContext("phase4", 0, "preflight-fail"),
    )
    assert not result.ok
    assert bridge._records == {}
    assert session.snapshot().source_kind is None


@pytest.mark.usefixtures("real_gmsh")
def test_geometry_accept_enters_mesh_stage_and_exposes_mesh_tools() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "mesh-stage", "geometry": _ring_geometry()},
        ToolExecutionContext("phase4", 0, "mesh-stage"),
    )
    assert result.ok, result.summary
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state)
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    requirements = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "mesh-stage-requirements",
            "requirements": {
                "mesh_cell_shape": "tetrahedron",
                "mesh_order": 1,
                "mesh_global_size": 0.8,
            },
        },
        ToolExecutionContext("phase4", session.snapshot().session_revision, "mesh-req"),
    )
    assert requirements.ok, requirements.summary
    assert "prepare_mesh_proposal" in {item.name for item in controller.definitions}


def test_native_wire_recipe_passes_path_safety_but_path_values_do_not() -> None:
    wire = WireGeometry(
        "native-path",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 0.0, 0.0, 1.0),
        ),
        (WireMember("AB", "A", "B"),),
    )
    wire_shape = geometry_recipe_to_payload(wire)
    wire_shape.pop("schema_version")
    native_wire = {
        "query": {
            "path": wire_shape,
        }
    }
    operation = ModelOperation(OperationKind.REQUEST_RESULT_QUERY, native_wire)
    assert operation.parameters == native_wire
    for invalid_path in (
        "relative.txt",
        {
            "kind": "wire",
            "points": wire_shape["points"],
            "members": wire_shape["members"],
        },
        {
            "kind": "wire",
            "name": "native-path",
            "points": wire_shape["points"],
            "members": wire_shape["members"],
            "extra": True,
        },
        {"kind": "filesystem"},
    ):
        with pytest.raises(AuthoringContractError, match="paths"):
            ModelOperation(
                OperationKind.REQUEST_RESULT_QUERY,
                {"query": {"path": invalid_path}},
            )
