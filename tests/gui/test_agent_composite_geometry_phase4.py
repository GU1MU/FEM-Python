from __future__ import annotations

import pytest

from fem import geometry as geometry_runtime
from fem.application import ModelSession
from fem.application.preprocessing import generate_fem_model
from fem.application.native_regions import RecipeRegionSelector
from fem.application.recipe_compiler import compile_recipe
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    PathSweptGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
)
from fem.io.project import decode_project, encode_project
from fem.mesh.settings import MeshSettings
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


def _path_geometry(*, with_hole: bool = False, frame_strategy: str = "transport") -> dict[str, object]:
    profiles: list[dict[str, object]] = [
        {
            "kind": "rectangle",
            "x": -0.5,
            "y": -0.5,
            "width": 1.0,
            "height": 1.0,
        }
    ]
    if with_hole:
        profiles.append(
            {
                "kind": "circle",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 0.15,
            }
        )
    return {
        "kind": "path_swept_profile",
        "profiles": profiles,
        "path": {
            "points": [
                {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
                {"name": "B", "x": 0.0, "y": 0.0, "z": 3.0},
                {"name": "C", "x": 2.0, "y": 0.0, "z": 4.0},
            ],
            "members": [
                {"name": "AB", "start": "A", "end": "B"},
                {"name": "BC", "start": "B", "end": "C"},
            ],
        },
        "frame_strategy": frame_strategy,
    }


def _s_slot_plate_geometry() -> dict[str, object]:
    return {
        "kind": "extruded_path_slot_plate",
        "plate": {
            "x": -400.0,
            "y": -200.0,
            "width": 800.0,
            "height": 400.0,
        },
        "slot_path": [
            {"x": -300.0, "y": 0.0},
            {"x": -180.0, "y": 100.0},
            {"x": 0.0, "y": 0.0},
            {"x": 180.0, "y": -100.0},
            {"x": 300.0, "y": 0.0},
        ],
        "slot_width": 20.0,
        "height": 10.0,
        "provisional": True,
    }


def _h_slot_plate_2d_geometry() -> dict[str, object]:
    return {
        "kind": "planar_profiles",
        "profiles": [
            {
                "kind": "rectangle",
                "x": 0.0,
                "y": 0.0,
                "width": 500.0,
                "height": 300.0,
                "role": "material",
                "operation": "material",
            },
            {
                "kind": "polygon",
                "vertices": [
                    {"x": 225.0, "y": 110.0},
                    {"x": 235.0, "y": 110.0},
                    {"x": 235.0, "y": 138.0},
                    {"x": 265.0, "y": 138.0},
                    {"x": 265.0, "y": 110.0},
                    {"x": 275.0, "y": 110.0},
                    {"x": 275.0, "y": 190.0},
                    {"x": 265.0, "y": 190.0},
                    {"x": 265.0, "y": 162.0},
                    {"x": 235.0, "y": 162.0},
                    {"x": 235.0, "y": 190.0},
                    {"x": 225.0, "y": 190.0},
                ],
                "role": "hole",
                "operation": "cut",
            },
            *(
                {
                    "kind": "circle",
                    "center_x": x,
                    "center_y": y,
                    "radius": 10.0,
                    "role": "hole",
                    "operation": "cut",
                }
                for x, y in (
                    (40.0, 40.0),
                    (460.0, 40.0),
                    (40.0, 260.0),
                    (460.0, 260.0),
                )
            ),
        ],
    }


def test_phase6_runtime_schema_retires_blank_composite_variants() -> None:
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


def test_phase4_planar_h_slot_and_four_holes_are_one_exact_part() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "宽平板，中央H形槽，四周开孔",
            "geometry": _h_slot_plate_2d_geometry(),
        },
        ToolExecutionContext("phase4-h-slot", 0, "h-slot"),
    )

    assert result.ok, result.summary
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    analysis = geometry_runtime.analyze_sketch_profiles(recipe)
    assert analysis.valid
    assert sum(profile.role == "outer" for profile in analysis.profiles) == 1
    assert sum(profile.role == "hole" for profile in analysis.profiles) == 5


def test_phase4_blank_s_path_slot_plate_is_one_final_solid_proposal() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "带 S 形贯穿槽的平板",
            "geometry": _s_slot_plate_geometry(),
        },
        ToolExecutionContext("phase4-s-slot", 0, "s-slot"),
    )

    assert result.ok, result.summary
    proposal = bridge._records[result.data["proposal_id"]].proposal
    summary = proposal.display_summary["summary"]
    assert proposal.display_summary["dimension"] == 3
    assert proposal.display_summary["expected_entity_count"] == 1
    assert all(
        value in summary
        for value in ("路径槽平板", "槽宽=20", "拉伸高=10", "holes=1")
    )
    assert len(proposal.display_summary["preview"]["points"]) <= 128

    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert len(topology.entities_of("body", selectable_only=True)) == 1
    assert any(
        entity.semantic_role == "sweep.boundary.hole"
        for entity in topology.entities_of("face")
    )


def test_phase4_five_circular_holes_fit_extruded_preview_budget() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    profiles = [
        {
            "kind": "rectangle",
            "x": -400.0,
            "y": -200.0,
            "width": 800.0,
            "height": 400.0,
        },
        *(
            {
                "kind": "circle",
                "center_x": x,
                "center_y": y,
                "radius": 10.0,
            }
            for x, y in (
                (-300.0, 0.0),
                (-150.0, -60.0),
                (0.0, 60.0),
                (150.0, -60.0),
                (300.0, 0.0),
            )
        ),
    ]
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "五孔平板",
            "geometry": {
                "kind": "extruded_profiles",
                "profiles": profiles,
                "height": 10.0,
            },
        },
        ToolExecutionContext("phase4-five-holes", 0, "five-holes"),
    )

    assert result.ok, result.summary
    proposal = bridge._records[result.data["proposal_id"]].proposal
    preview = proposal.display_summary["preview"]
    assert preview["dimension"] == 3
    assert len(preview["points"]) <= 128
    assert "holes=5" in proposal.display_summary["summary"]


def test_phase4_duplicate_polygon_vertex_returns_profile_input_diagnostic() -> None:
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


def test_phase4_blank_ring_is_one_atomic_final_3d_proposal_and_persistent() -> None:
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
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe


def test_phase4_blank_composite_reject_keeps_session_empty() -> None:
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


def test_phase4_center_hole_plate_accepts_without_analysis_side_effects() -> None:
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


def test_phase4_contour_order_does_not_choose_material_or_hole() -> None:
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


def test_phase4_multiple_disjoint_material_profiles_fail_closed() -> None:
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


def test_phase4_explicit_provisional_summary_contains_all_dimensions() -> None:
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


def test_phase4_stale_accept_keeps_blank_session() -> None:
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


def test_phase4_failed_composite_preflight_is_not_registered(monkeypatch) -> None:
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


def test_phase4_geometry_accept_enters_mesh_stage_and_exposes_mesh_tools() -> None:
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


def test_phase4_blank_path_sweep_accepts_one_body_and_reopens() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "路径杆", "geometry": _path_geometry()},
        ToolExecutionContext("phase4", before.session_revision, "path"),
    )
    assert result.ok, result.summary
    assert session.snapshot() == before
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert proposal.display_summary["expected_entity_count"] == 1
    assert "frame=transport" in proposal.display_summary["summary"]
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry
    assert recipe.frame_strategy == "transport"
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe


def test_phase4_composite_path_enters_supported_tet_mesh_chain() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "可网格路径杆", "geometry": _path_geometry()},
        ToolExecutionContext("phase4", 0, "mesh"),
    )
    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    mesh = generate_fem_model(recipe, MeshSettings(0.8, cell_shape="tetrahedron"))
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}


def test_phase4_ring_replace_height_preserves_hole_lineage_and_meshes() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "replace-ring", "geometry": _ring_geometry()},
        ToolExecutionContext("phase4", 0, "replace-ring"),
    )
    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    original = reopened.parts[0].geometry_recipe
    assert type(original) is ExtrudedGeometry
    before_ids = describe_recipe_topology(original).signature.logical_ids
    edited = ExtrudedGeometry(original.base, 80.0, original.source_face_ids)
    session.replace_part_geometry("P1", edited)
    replaced = session.snapshot().parts[0].geometry_recipe
    topology = describe_recipe_topology(replaced)
    assert replaced.height == 80.0
    assert topology.exact
    assert topology.signature.logical_ids == before_ids
    assert {"face:bottom", "face:top"}.issubset(
        set(topology.signature.logical_ids)
    )
    assert any(entity.semantic_role == "sweep.boundary.hole" for entity in topology.entities)
    mesh = generate_fem_model(replaced, MeshSettings(15.0, cell_shape="tetrahedron"))
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}


@pytest.mark.parametrize("frame_strategy", ("fixed", "transport"))
def test_phase4_path_sweep_hole_fixed_and_transport_save_and_tet(
    frame_strategy: str,
) -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    geometry = _path_geometry(with_hole=True, frame_strategy=frame_strategy)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {"part_function": "hole-path", "geometry": geometry},
        ToolExecutionContext("phase4", 0, f"hole-path-{frame_strategy}"),
    )
    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry
    assert recipe.frame_strategy == frame_strategy
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert len(topology.entities_of("body", selectable_only=True)) == 1
    assert any(
        entity.semantic_role == "sweep.boundary.hole"
        for entity in topology.entities_of("face")
    )
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe
    mesh = generate_fem_model(recipe, MeshSettings(0.6, cell_shape="tetrahedron"))
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}


@pytest.mark.parametrize("frame_strategy", ("fixed", "transport"))
def test_phase4_path_compiler_side_bindings_are_complete_and_disjoint(
    frame_strategy: str,
) -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "compiler-lineage",
            "geometry": _path_geometry(
                with_hole=True,
                frame_strategy=frame_strategy,
            ),
        },
        ToolExecutionContext("phase4", 0, f"compiler-{frame_strategy}"),
    )
    assert result.ok, result.summary
    bridge.accept_from_gui_control(result.data["proposal_id"])
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry

    with geometry_runtime.model(
        f"phase4-path-lineage-{frame_strategy}",
        dimension=3,
    ) as cad:
        compiled = compile_recipe(cad, recipe)
        side_ids = tuple(
            entity.logical_id
            for entity in compiled.catalog.entities_of("face")
            if entity.logical_id.startswith("face:side/")
        )
        side_groups = {
            logical_id: set(compiled.resolve(LogicalEntityRef(logical_id)))
            for logical_id in side_ids
        }
        assert side_groups
        assert all(side_groups.values())
        groups = tuple(side_groups.values())
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        )
        start = set(compiled.logical_entities["face:start"])
        end = set(compiled.logical_entities["face:end"])
        all_side_faces = set(compiled.boundary).difference(start | end)
        assert set().union(*groups) == all_side_faces
        hole_ids = tuple(
            entity.logical_id
            for entity in compiled.catalog.entities_of("face")
            if entity.logical_id.startswith("face:side/")
            and "hole" in entity.semantic_role
        )
        outer_ids = tuple(
            entity.logical_id
            for entity in compiled.catalog.entities_of("face")
            if entity.logical_id.startswith("face:side/")
            and "outer" in entity.semantic_role
        )
        assert hole_ids and outer_ids
        assert set(compiled.region_bindings[RecipeRegionSelector.HOLE]) == set().union(
            *(side_groups[logical_id] for logical_id in hole_ids)
        )
        assert set(compiled.region_bindings[RecipeRegionSelector.OUTER]) == set().union(
            *(side_groups[logical_id] for logical_id in outer_ids)
        )


def test_phase4_native_wire_recipe_passes_path_safety_but_path_values_do_not() -> None:
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
