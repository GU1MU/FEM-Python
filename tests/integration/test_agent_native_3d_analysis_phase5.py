from __future__ import annotations

import numpy as np
import pytest

from fem import geometry as geometry_runtime
from fem.application import (
    ModelSession,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
    run_static_preflight,
)
from fem.application.preprocessing import generate_fem_model
from fem.application.results import build_solve_result_bundle
from fem.application.solid_boolean import prepare_solid_body_boolean
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    OutputRequest,
    SectionAssignment,
    SurfaceLoad,
)
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    MultiBodyGeometry,
    PathSweptGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    SketchCircle,
    SketchRectangle,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
    namespace_part_logical_id,
)
from fem.io.project import dumps_project, loads_project
from fem.materials.linear_elastic import material
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentPreflightState,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
)


def _path_sweep() -> PathSweptGeometry:
    return PathSweptGeometry(
        RectangleGeometry("Sweep profile", 0.4, 0.3),
        WireGeometry(
            "Ordered path",
            (
                WirePoint("A", 0.0, 0.0, 0.0),
                WirePoint("B", 0.0, 0.0, 1.0),
                WirePoint("C", 0.5, 0.0, 1.5),
            ),
            (
                WireMember("AB", "A", "B"),
                WireMember("BC", "B", "C"),
            ),
        ),
        ("face:domain",),
        "transport",
    )


def _cut_solid() -> BooleanGeometry:
    return BooleanGeometry(
        "Boolean cut",
        "cut",
        BoxGeometry("Target", 2.0, 2.0, 1.0),
        MovedGeometry(CylinderGeometry("Tool", 0.35, 1.0), 1.0, 1.0, 0.0),
    )


def _dispatch(
    controller: object,
    session: ModelSession,
    name: str,
    arguments: dict[str, object],
    suffix: str,
):
    return controller.dispatch(
        name,
        arguments,
        ToolExecutionContext(
            session.snapshot().session_id,
            session.session_revision,
            suffix,
        ),
    )


def _record_mesh_requirements(
    controller: object,
    session: ModelSession,
    *,
    cell_shape: str,
    order: int,
    global_size: float,
) -> None:
    recorded = _dispatch(
        controller,
        session,
        "set_authoring_requirements",
        {
            "turn_id": f"turn-mesh-{cell_shape}",
            "requirements": {
                "mesh_cell_shape": cell_shape,
                "mesh_order": order,
                "mesh_global_size": global_size,
            },
        },
        f"mesh-requirements-{cell_shape}",
    )
    assert recorded.ok, recorded.to_json()
    assert recorded.data["missing_requirements"] == []


def _phase5_production_controller(
    session: ModelSession,
    *,
    start_mesh_task=None,
) -> tuple[object, AgentAuthoringBridge]:
    state: dict[str, object] = {}

    def rebind() -> None:
        bridge = state["bridge"]
        controller = state["controller"]
        stale_ids = bridge.bind_snapshot(session.snapshot())
        controller.observe_binding(
            bridge.context,
            proposal_staled=bool(stale_ids),
        )

    def apply_definition_delta(_delta) -> None:
        rebind()

    def run_mesh(request) -> bool:
        if start_mesh_task is not None:
            return bool(start_mesh_task(request))
        candidate = generate_fem_model(request.task)
        assert state["port"].accept_mesh_result(
            request.proposal_id,
            candidate,
        ).accepted
        rebind()
        return True

    def run_preflight(request) -> bool:
        task = session.prepare_validation(request.step_name)
        report = run_static_preflight(
            task.model,
            task.step_name,
            token=task.token,
        )
        assert report.passed, report.diagnostics
        assert session.accept_validation(task.token, report).accepted
        rebind()
        state["port"].complete_preflight(
            request.request_id,
            AgentPreflightState.PASSED,
            "passed",
        )
        return True

    def run_solve(request) -> bool:
        task = session.prepare_solve(request.step_name, request.job_name)
        assert session.begin_run(task.token).accepted
        result = static_linear.solve(
            task.model,
            task.step_name,
            name=request.job_name,
        )
        assert session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        ).accepted
        rebind()
        state["port"].complete_solve(
            request.proposal_id,
            ProposalState.SUCCEEDED,
            "succeeded",
        )
        return True

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_mesh_task=run_mesh,
        apply_definition_delta=apply_definition_delta,
        start_solve_task=run_solve,
        start_preflight_task=run_preflight,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    state.update(port=port, bridge=bridge, controller=controller)
    return controller, bridge


def _apply_agent_definition(
    controller: object,
    session: ModelSession,
    action: str,
    parameters: dict[str, object],
    suffix: str,
) -> None:
    outcome = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {"action": action, "parameters": parameters},
        suffix,
    )
    assert outcome.ok, outcome.to_json()


def test_phase5_mesh_intent_schema_is_strict_and_hex_is_capability_gated() -> None:
    automatic = MeshIntent("tetrahedron", 2, auto_level=4)
    payload = automatic.to_dict()

    assert MeshIntent.from_dict(payload) == automatic
    assert automatic.to_auto_mesh_spec().cell_shape == "tet"
    assert MeshIntent("hexahedron", 1, global_size=0.25).to_mesh_settings(
        BoxGeometry("Structured", 1.0, 1.0, 1.0)
    ).cell_shape == "hexahedron"
    with pytest.raises(ValueError, match="mesh.hex.unsupported-shape"):
        MeshIntent("hexahedron", 1, global_size=0.25).to_mesh_settings(
            _path_sweep()
        )
    with pytest.raises(ValueError, match="fields do not match"):
        MeshIntent.from_dict({**payload, "fallback": "tetrahedron"})
    with pytest.raises(ValueError, match="schema 1.1"):
        MeshIntent(
            "tetrahedron",
            1,
            global_size=0.25,
            schema_version="1.1",
        )


def test_phase5_runtime_exposes_solid_mesh_requirements_for_derived_geometry() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent solid requirements",
        UnitContext("mm", "N", "MPa"),
        _path_sweep(),
        part_name="Swept member",
    )
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    requirement_tool = next(
        definition
        for definition in controller.definitions
        if definition.name == "set_authoring_requirements"
    )
    properties = requirement_tool.parameters["properties"]["requirements"][
        "properties"
    ]

    assert authoring_context_from_snapshot(session.snapshot()).parts[0].dimension == 3
    assert properties["mesh_cell_shape"]["enum"] == [
        "tetrahedron",
        "hexahedron",
    ]
    assert properties["mesh_order"]["enum"] == [1, 2]


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "recipe",
    (
        ExtrudedGeometry(RectangleGeometry("Block", 1.0, 0.8), 0.6),
        ExtrudedGeometry(
            PlateWithHoleGeometry("Perforated", 2.0, 1.5, 1.0, 0.75, 0.25),
            0.5,
        ),
        _path_sweep(),
        _cut_solid(),
    ),
    ids=("extruded-block", "extruded-hole", "path-sweep", "boolean-cut"),
)
def test_phase5_feature_results_generate_pure_tet4(real_gmsh, recipe) -> None:
    del real_gmsh

    model = generate_fem_model(
        recipe,
        MeshIntent("tetrahedron", 1, global_size=0.35).to_mesh_settings(recipe),
    )

    assert model.mesh.nodes
    assert model.mesh.elements
    assert {element.type for element in model.mesh.elements} == {"Tet4"}


@pytest.mark.gmsh
@pytest.mark.parametrize(("order", "expected"), ((1, "Tet4"), (2, "Tet10")))
def test_phase5_tet_intent_round_trip_generates_requested_order(
    real_gmsh,
    order: int,
    expected: str,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Ordered Tet", 1.0, 0.8), 0.6)
    intent = MeshIntent("tetrahedron", order, global_size=0.35)

    restored = MeshIntent.from_dict(intent.to_dict())
    model = generate_fem_model(recipe, restored.to_mesh_settings(recipe))

    assert restored == intent
    assert intent.to_dict()["schema_version"] == "1.2"
    assert intent.to_auto_mesh_spec() is None
    assert {element.type for element in model.mesh.elements} == {expected}


@pytest.mark.gmsh
def test_phase5_gui_bridge_commits_tet_intent_and_model_atomically(real_gmsh) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Agent block", 1.0, 0.8), 0.6)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Tet model",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Extruded part",
    )
    before = session.snapshot()
    proposal = create_mesh_proposal(
        proposal_id="proposal-phase5-tet",
        agent_session_id="agent-phase5",
        turn_id="turn-phase5",
        source_tool_call_ids=("call-phase5",),
        context=authoring_context_from_snapshot(before),
        draft_revision=5,
        part_id="P1",
        mesh_intent=MeshIntent("tetrahedron", 1, global_size=0.35),
    )
    requests = []
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: None,
            lambda request: requests.append(request) is None,
        )
    )
    bridge.bind_snapshot(before)
    bridge.register_proposal(proposal)

    running = bridge.accept_from_gui_control(proposal.proposal_id)
    assert running.state.value == "running"
    assert session.snapshot() == before
    task = requests[0].task
    candidate = generate_fem_model(task)
    delta = bridge.port.accept_mesh_result(proposal.proposal_id, candidate)

    assert delta.accepted
    assert bridge.state(proposal.proposal_id).value == "succeeded"
    assert session.snapshot().parts[0].mesh_settings == MeshSettings(
        0.35,
        cell_shape="tetrahedron",
        strict_cell_shape=True,
    )
    assert {element.type for element in session.snapshot().model.mesh.elements} == {
        "Tet4"
    }


@pytest.mark.gmsh
def test_phase5_agent_controller_bridge_completes_real_3d_loop_and_reopens(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Agent loop", 1.0, 1.0), 1.0)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent 3D loop",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Continuum part",
    )
    controller, bridge = _phase5_production_controller(session)
    _record_mesh_requirements(
        controller,
        session,
        cell_shape="tetrahedron",
        order=1,
        global_size=0.35,
    )

    proposed_mesh = _dispatch(
        controller,
        session,
        "prepare_mesh_proposal",
        {},
        "agent-loop-mesh",
    )
    assert proposed_mesh.ok, proposed_mesh.to_json()
    assert not session.snapshot().mesh_current
    mesh_receipt = bridge.accept_from_gui_control(
        proposed_mesh.data["proposal_id"],
    )
    controller.record_proposal_state(
        "mesh",
        mesh_receipt.state,
        mesh_receipt.message,
    )

    assert mesh_receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert {element.type for element in session.snapshot().model.mesh.elements} == {
        "Tet4"
    }
    definition_tool = next(
        item
        for item in controller.definitions
        if item.name == "apply_model_definition"
    )
    section_schema = next(
        item
        for item in definition_tool.parameters["oneOf"]
        if item["properties"]["action"]["const"] == "create_section"
    )
    assert any(
        variant["properties"].get("section_type", {}).get("const") == "solid"
        for variant in section_schema["properties"]["parameters"]["oneOf"]
    )

    topology = _dispatch(
        controller,
        session,
        "read_model_topology_context",
        {},
        "agent-loop-topology",
    )
    assert topology.ok, topology.to_json()

    def region_parameters(
        name: str,
        logical_id: str,
        mesh_kind: str,
    ) -> dict[str, object]:
        entry = next(
            item
            for item in topology.data["entries"]
            if item["logical_id"] == logical_id
            and item["mesh_kind"] == mesh_kind
        )
        assert entry["matched_count"] > 0
        return {
            "name": name,
            "part_id": entry["part_id"],
            "logical_ids": [entry["logical_id"]],
            "mesh_kind": entry["mesh_kind"],
            "expected_count": entry["matched_count"],
        }

    actions = (
        (
            "create_named_region",
            region_parameters("域-实体", "body:domain", "element"),
            "agent-loop-body",
        ),
        (
            "create_named_region",
            region_parameters("面-固定端", "face:side/left", "face"),
            "agent-loop-fixed-face",
        ),
        (
            "create_named_region",
            region_parameters("面-加载端", "face:side/right", "face"),
            "agent-loop-loaded-face",
        ),
        (
            "create_material",
            {
                "name": "材料-线弹性",
                "properties": {"E": 1000.0, "nu": 0.0},
            },
            "agent-loop-material",
        ),
        (
            "create_section",
            {
                "name": "截面-实体",
                "material": "材料-线弹性",
                "section_type": "solid",
                "properties": {},
            },
            "agent-loop-section",
        ),
        (
            "assign_section",
            {
                "section_name": "截面-实体",
                "region_name": "域-实体",
            },
            "agent-loop-assignment",
        ),
        (
            "create_static_step",
            {"name": "分析步-静力"},
            "agent-loop-step",
        ),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": "分析步-静力",
                "target_scope": "面-固定端",
                "target_kind": "surface",
                "first_component": 1,
                "last_component": 3,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            "agent-loop-boundary",
        ),
        (
            "create_load",
            {
                "name": "载荷-拉伸",
                "step_name": "分析步-静力",
                "target_scope": "面-加载端",
                "entity_type": "surface",
                "load_type": "surface_traction",
                "component": None,
                "vector": [10.0, 0.0, 0.0],
                "magnitude": None,
                "direction": "global_xyz",
                "unit": "MPa",
                "distribution": "uniform",
                "confirmed": True,
            },
            "agent-loop-load",
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-节点",
                "step_name": "分析步-静力",
                "target": "node",
                "variables": ["U", "RF"],
                "units": ["mm", "N"],
                "confirmed": True,
            },
            "agent-loop-output",
        ),
    )
    for action, parameters, suffix in actions:
        _apply_agent_definition(
            controller,
            session,
            action,
            parameters,
            suffix,
        )

    preflight = _dispatch(
        controller,
        session,
        "run_native_preflight",
        {},
        "agent-loop-preflight",
    )
    assert preflight.ok, preflight.to_json()
    assert preflight.data["passed"] is True
    proposed_solve = _dispatch(
        controller,
        session,
        "prepare_solve_proposal",
        {},
        "agent-loop-solve",
    )
    assert proposed_solve.ok, proposed_solve.to_json()
    solve_receipt = bridge.accept_from_gui_control(
        proposed_solve.data["proposal_id"],
    )
    controller.record_proposal_state(
        "solve",
        solve_receipt.state,
        solve_receipt.message,
    )

    assert solve_receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.RESULTS_READY
    result_record = session.current_result()
    assert result_record is not None
    result = result_record.result
    loaded_ids = sorted(
        {
            node_id
            for face in result.model.surfaces["面-加载端"].faces
            for node_id in face.node_ids
        }
    )
    assert np.array(
        [result.nodal_displacement(node_id, 1) for node_id in loaded_ids],
    ) == pytest.approx(0.01, abs=1.0e-10)
    assert float(result.reactions[0::3].sum()) == pytest.approx(-10.0, abs=1.0e-9)

    save_request = _dispatch(
        controller,
        session,
        "request_project_save",
        {},
        "agent-loop-save",
    )
    assert save_request.ok, save_request.to_json()
    assert save_request.data["proposal_id"]
    reopened_snapshot = loads_project(
        dumps_project(session.prepare_project_save()),
    ).snapshot
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(reopened_snapshot).accepted
    assert reopened.snapshot().sections[0].section_type == "solid"
    assert reopened.snapshot().named_regions == session.snapshot().named_regions
    reopened_controller, reopened_bridge = _phase5_production_controller(reopened)
    _record_mesh_requirements(
        reopened_controller,
        reopened,
        cell_shape="tetrahedron",
        order=1,
        global_size=0.35,
    )
    reopened_mesh_proposal = _dispatch(
        reopened_controller,
        reopened,
        "prepare_mesh_proposal",
        {},
        "agent-loop-reopened-mesh",
    )
    assert reopened_mesh_proposal.ok, reopened_mesh_proposal.to_json()
    reopened_mesh_receipt = reopened_bridge.accept_from_gui_control(
        reopened_mesh_proposal.data["proposal_id"],
    )
    reopened_controller.record_proposal_state(
        "mesh",
        reopened_mesh_receipt.state,
        reopened_mesh_receipt.message,
    )
    assert reopened_mesh_receipt.state is ProposalState.SUCCEEDED
    reopened_preflight = _dispatch(
        reopened_controller,
        reopened,
        "run_native_preflight",
        {},
        "agent-loop-reopened-preflight",
    )
    assert reopened_preflight.ok, reopened_preflight.to_json()
    reopened_proposal = _dispatch(
        reopened_controller,
        reopened,
        "prepare_solve_proposal",
        {},
        "agent-loop-reopened-solve",
    )
    assert reopened_proposal.ok, reopened_proposal.to_json()
    reopened_receipt = reopened_bridge.accept_from_gui_control(
        reopened_proposal.data["proposal_id"],
    )
    reopened_controller.record_proposal_state(
        "solve",
        reopened_receipt.state,
        reopened_receipt.message,
    )
    reopened_result = reopened.current_result()

    assert reopened_receipt.state is ProposalState.SUCCEEDED
    assert reopened_result is not None
    assert reopened_result.result.U == pytest.approx(result.U, abs=1.0e-12)


@pytest.mark.gmsh
def test_phase5_unsupported_hex_is_diagnostic_and_session_atomic(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = _path_sweep()
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Unsupported Hex",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Swept member",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.35, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    old_mesh_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        old_mesh_task.token,
        generate_fem_model(old_mesh_task),
    ).accepted
    before = session.snapshot()
    started = []
    controller, bridge = _phase5_production_controller(
        session,
        start_mesh_task=lambda request: started.append(request) or True,
    )
    _record_mesh_requirements(
        controller,
        session,
        cell_shape="hexahedron",
        order=1,
        global_size=0.3,
    )

    failed = _dispatch(
        controller,
        session,
        "prepare_mesh_proposal",
        {},
        "unsupported-hex-proposal",
    )

    assert not failed.ok
    assert "mesh.hex.unsupported-shape" in failed.diagnostics[0].message
    assert failed.data["diagnostic_code"] == "mesh.hex.unsupported-shape"
    assert "proposal_id" not in failed.data
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert bridge._records == {}
    assert started == []
    assert session.snapshot() == before

    # The GUI acceptance boundary repeats the same check for a tampered or
    # otherwise externally retained proposal.
    proposal = create_mesh_proposal(
        proposal_id="proposal-phase5-tampered-hex",
        agent_session_id="agent-phase5",
        turn_id="turn-phase5",
        source_tool_call_ids=("call-phase5",),
        context=authoring_context_from_snapshot(before),
        draft_revision=5,
        part_id="P1",
        mesh_intent=MeshIntent("hexahedron", 1, global_size=0.3),
    )
    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert "mesh.hex.unsupported-shape" in receipt.message
    assert started == []
    assert session.snapshot() == before


@pytest.mark.gmsh
def test_phase5_logical_face_region_survives_remesh_and_reopen(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Stable faces", 1.0, 0.8), 0.6)
    face_reference = LogicalEntityRef(
        namespace_part_logical_id("P1", "face:side/right")
    )
    regions = (
        NamedRegion("LoadedFace", (face_reference,)),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Stable face model",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Extruded part",
    )
    session.replace_named_regions(regions)
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.4, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    first_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        first_task.token,
        generate_fem_model(first_task),
    ).accepted
    first_count = len(session.snapshot().model.surfaces["LoadedFace"].faces)

    reopened = loads_project(dumps_project(session.prepare_project_save())).snapshot
    restored = ModelSession()
    restored.replace_from_snapshot(reopened)
    restored.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.25, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    second_task = restored.prepare_mesh_generation()
    assert restored.accept_generated_model(
        second_task.token,
        generate_fem_model(second_task),
    ).accepted
    model = restored.snapshot().model
    loaded = model.surfaces["LoadedFace"]
    nodes = {int(node.id): node for node in model.mesh.nodes}

    assert first_count > 0
    assert loaded.faces
    assert len(loaded.faces) > first_count
    assert all(
        float(nodes[node_id].x) == pytest.approx(1.0, abs=1.0e-8)
        for face in loaded.faces
        for node_id in face.node_ids
    )
    assert restored.snapshot().named_regions["LoadedFace"].references == (
        face_reference,
    )


def _analysis_model(load_type: str, *, order: int = 1):
    recipe = ExtrudedGeometry(RectangleGeometry("Oracle block", 1.0, 1.0), 1.0)
    model = generate_fem_model(
        recipe,
        MeshSettings(
            0.35,
            order=order,
            cell_shape="tetrahedron",
            strict_cell_shape=True,
        ),
        named_regions=(
            NamedRegion("Body", (LogicalEntityRef("body:domain"),)),
            NamedRegion("Fixed", (LogicalEntityRef("face:side/left"),)),
            NamedRegion("Loaded", (LogicalEntityRef("face:side/right"),)),
        ),
    )
    steel = material("Steel", 1000.0, 0.0)
    model.materials[steel.name] = steel
    model.sections.append(SectionAssignment("Body", steel.name, "solid"))
    surface_load = (
        SurfaceLoad("Loaded", (10.0, 0.0, 0.0), load_type="traction")
        if load_type == "traction"
        else SurfaceLoad("Loaded", magnitude=10.0, load_type="pressure")
    )
    model.steps.append(
        AnalysisStep(
            "Static",
            boundaries=(
                DisplacementConstraint(
                    "Fixed",
                    1,
                    3,
                    0.0,
                    target_kind="surface",
                ),
            ),
            surface_loads=(surface_load,),
            outputs=(
                OutputRequest("field", "node", ("U", "RF"), name="Nodal"),
                OutputRequest("field", "element", ("S",), name="Stress"),
            ),
            metadata={"nlgeom": False},
        )
    )
    return model


@pytest.mark.gmsh
def test_phase5_tet10_runs_preflight_and_real_solver(real_gmsh) -> None:
    del real_gmsh
    model = _analysis_model("traction", order=2)

    report = run_static_preflight(model, "Static")
    result = static_linear.solve(model, "Static")

    assert report.passed, report.diagnostics
    assert {element.type for element in model.mesh.elements} == {"Tet10"}
    assert np.all(np.isfinite(result.U))
    assert float(result.reactions[0::3].sum()) == pytest.approx(-10.0, abs=1.0e-9)


@pytest.mark.gmsh
def test_phase5_preflight_diagnoses_uncovered_solid_and_rigid_body_dofs(
    real_gmsh,
) -> None:
    del real_gmsh
    uncovered = _analysis_model("traction")
    uncovered.sections.clear()
    uncovered_report = run_static_preflight(uncovered, "Static")

    underconstrained = _analysis_model("traction")
    underconstrained.steps[0].boundaries = (
        DisplacementConstraint(
            "Fixed",
            1,
            1,
            0.0,
            target_kind="surface",
        ),
    )
    rigid_body_report = run_static_preflight(underconstrained, "Static")

    assert not uncovered_report.passed
    assert "definition.section.missing" in {
        diagnostic.code for diagnostic in uncovered_report.diagnostics
    }
    assert not rigid_body_report.passed
    assert "static.stiffness.singular" in {
        diagnostic.code for diagnostic in rigid_body_report.diagnostics
    }


def _strict_rectangle(name: str, width: float, height: float):
    sketch = planar_sketch_geometry(
        name,
        contours=(SketchRectangle("material", 0.0, 0.0, width, height),),
    ).recipe
    source = next(
        entity.logical_id
        for entity in geometry_runtime.describe_recipe_topology(sketch).entities
        if entity.kind == "face" and entity.semantic_role == "sketch.profile"
    )
    return sketch, source


def _body_boolean_result_face(prepared, source: str, source_logical_id: str) -> str:
    target = next(
        mapping.target_logical_id
        for mapping in prepared.proof.topology_mappings
        if mapping.source == source
        and mapping.source_logical_id == source_logical_id
        and mapping.target_logical_id.startswith("face:")
    )
    semantic_name = target.split(":", 1)[1]
    return target if semantic_name.startswith("B1/") else f"face:B1/{semantic_name}"


def _end_to_end_scenario(scenario: str):
    if scenario == "perforated-bracket":
        sketch = planar_sketch_geometry(
            "Bracket sketch",
            contours=(
                SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),
                SketchCircle("cut", 0.6, 0.5, 0.15),
            ),
        ).recipe
        source = next(
            entity.logical_id
            for entity in geometry_runtime.describe_recipe_topology(sketch).entities
            if entity.kind == "face" and entity.semantic_role == "sketch.profile"
        )
        target = ExtrudedGeometry(sketch, 0.5, (source,))
        source_geometry = MultiBodyGeometry(
            "Bracket bodies",
            (
                SolidBody("B1", "Bracket", target),
                SolidBody(
                    "B2",
                    "Cutting cylinder",
                    MovedGeometry(
                        CylinderGeometry("Tool", 0.1, 0.5),
                        1.5,
                        0.5,
                        0.0,
                    ),
                ),
            ),
        )
        with geometry_runtime.model("bracket-cut", dimension=3) as cad:
            prepared = prepare_solid_body_boolean(
                cad,
                source_geometry,
                "B1",
                "B2",
                "cut",
            )
        recipe = prepared.geometry
        body_id = next(
            entity.logical_id
            for entity in geometry_runtime.describe_recipe_topology(recipe).entities
            if entity.kind == "body" and entity.selectable
        )
        return (
            recipe,
            body_id,
            _body_boolean_result_face(prepared, "target", "face:bottom"),
            _body_boolean_result_face(prepared, "target", "face:top"),
            (0.0, 0.0, 1.0),
        )
    if scenario == "swept-member":
        sketch, source = _strict_rectangle("Sweep section", 0.4, 0.3)
        recipe = PathSweptGeometry(
            sketch,
            _path_sweep().path,
            (source,),
            "transport",
        )
        return recipe, "body:domain", "face:start", "face:end", (0.0, 0.0, 1.0)
    first_sketch, first_source = _strict_rectangle("First extrusion", 1.0, 1.0)
    second_sketch, second_source = _strict_rectangle("Second extrusion", 1.0, 1.0)
    source_geometry = MultiBodyGeometry(
        "Composite bodies",
        (
            SolidBody(
                "B1",
                "First",
                ExtrudedGeometry(first_sketch, 1.0, (first_source,)),
            ),
            SolidBody(
                "B2",
                "Second",
                MovedGeometry(
                    ExtrudedGeometry(second_sketch, 1.0, (second_source,)),
                    0.8,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    with geometry_runtime.model("composite-fuse", dimension=3) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source_geometry,
            "B1",
            "B2",
            "fuse",
        )
    recipe = prepared.geometry
    topology = geometry_runtime.describe_recipe_topology(recipe)
    body_id = next(
        entity.logical_id
        for entity in topology.entities
        if entity.kind == "body" and entity.selectable
    )
    return (
        recipe,
        body_id,
        _body_boolean_result_face(prepared, "target", "face:side/L4"),
        _body_boolean_result_face(prepared, "tool", "face:side/L2"),
        (1.0, 0.0, 0.0),
    )


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "scenario",
    ("perforated-bracket", "swept-member", "composite-fuse"),
)
def test_phase5_three_planned_scenarios_mesh_define_solve_and_reopen(
    real_gmsh,
    scenario: str,
) -> None:
    del real_gmsh
    recipe, body_id, fixed_id, loaded_id, traction = _end_to_end_scenario(scenario)
    session = ModelSession()
    session.create_native_project_with_first_part(
        f"Phase5 {scenario}",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Continuum part",
    )
    regions = tuple(
        NamedRegion(
            name,
            (
                LogicalEntityRef(
                    namespace_part_logical_id("P1", logical_id)
                ),
            ),
        )
        for name, logical_id in (
            ("Body", body_id),
            ("Fixed", fixed_id),
            ("Loaded", loaded_id),
        )
    )
    session.replace_named_regions(regions)
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.3, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    mesh_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        mesh_task.token,
        generate_fem_model(mesh_task),
    ).accepted
    if scenario == "perforated-bracket":
        catalog_bridge = AgentAuthoringBridge(
            SessionGeometryAuthoringPort(session, lambda: None)
        )
        catalog_bridge.bind_snapshot(session.snapshot())
        catalog_controller = create_session_authoring_workflow_controller(
            session,
            catalog_bridge,
            AgentResultQueryBridge(SessionResultQueryPort(session)),
        )
        catalog_result = catalog_controller.dispatch(
            "read_model_topology_context",
            {},
            ToolExecutionContext(
                "phase5-catalog",
                session.session_revision,
                "phase5-body-face-catalog",
            ),
        )
        assert catalog_result.ok, catalog_result.diagnostics
        entries = catalog_result.data["entries"]
        assert any(
            entry["logical_id"] == body_id
            and entry["mesh_kind"] == "element"
            for entry in entries
        )
        assert all(
            any(
                entry["logical_id"] == logical_id
                and entry["mesh_kind"] == "face"
                for entry in entries
            )
            for logical_id in (fixed_id, loaded_id)
        )
    snapshot = session.snapshot()
    steel = material("Steel", 1000.0, 0.25)
    step = AnalysisStep(
        "Static",
        boundaries=(
            DisplacementConstraint(
                "Fixed",
                1,
                3,
                0.0,
                target_kind="surface",
            ),
        ),
        surface_loads=(
            SurfaceLoad("Loaded", traction, load_type="traction"),
        ),
        outputs=(
            OutputRequest("field", "node", ("U", "RF"), name="Nodal"),
            OutputRequest("field", "element", ("S",), name="Stress"),
        ),
        metadata={"nlgeom": False},
    )
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            tuple(snapshot.named_regions.values()),
            (steel,),
            (SectionDefinition("Solid", steel.name, "solid"),),
            (RegionAssignment("Solid", "Body"),),
            (step,),
        )
    )
    compiled = session.snapshot().model
    report = run_static_preflight(compiled, "Static")
    result = static_linear.solve(compiled, "Static")

    assert report.passed, report.diagnostics
    assert np.all(np.isfinite(result.U))
    assert np.linalg.norm(result.U) > 0.0
    assert {element.type for element in compiled.mesh.elements} == {"Tet4"}

    reopened = loads_project(dumps_project(session.prepare_project_save())).snapshot
    restored = ModelSession()
    restored.replace_from_snapshot(reopened)
    reopened_task = restored.prepare_mesh_generation()
    assert restored.accept_generated_model(
        reopened_task.token,
        generate_fem_model(reopened_task),
    ).accepted
    restored_model = restored.snapshot().model
    restored_report = run_static_preflight(restored_model, "Static")
    restored_result = static_linear.solve(restored_model, "Static")

    assert restored_report.passed, restored_report.diagnostics
    assert restored.snapshot().geometry_recipe == recipe
    assert restored.snapshot().named_regions == session.snapshot().named_regions
    assert restored_result.U == pytest.approx(result.U, abs=1.0e-12)


@pytest.mark.gmsh
@pytest.mark.parametrize(
    ("load_type", "expected_displacement", "expected_reaction"),
    (
        ("traction", 0.01, -10.0),
        ("pressure", -0.01, 10.0),
    ),
)
def test_phase5_3d_tension_compression_oracle_and_reaction_balance(
    real_gmsh,
    load_type: str,
    expected_displacement: float,
    expected_reaction: float,
) -> None:
    del real_gmsh
    model = _analysis_model(load_type)

    report = run_static_preflight(model, "Static")
    result = static_linear.solve(model, "Static")
    loaded_ids = model.node_sets["Loaded"].node_ids
    loaded_displacements = np.array(
        [result.nodal_displacement(node_id, 1) for node_id in loaded_ids],
    )

    assert report.passed, report.diagnostics
    assert loaded_displacements == pytest.approx(expected_displacement, abs=1.0e-10)
    assert float(result.reactions[0::3].sum()) == pytest.approx(
        expected_reaction,
        abs=1.0e-9,
    )
    assert float(result.reactions[1::3].sum()) == pytest.approx(0.0, abs=1.0e-9)
    assert float(result.reactions[2::3].sum()) == pytest.approx(0.0, abs=1.0e-9)
