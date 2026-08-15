from __future__ import annotations

from copy import deepcopy
import json
from time import perf_counter

import pytest

from fem.application import ModelSession
from fem.application.planar_construction import compile_planar_construction
from fem.geometry import PlanarConstructionIR
from fem.io.project import decode_project, encode_project
import fem_agent.authoring_runtime as authoring_runtime
from fem_agent.authoring import ProposalState
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.routing import geometry_route_hint
from fem_agent.tools.registry import ToolExecutionContext, tool_schema_hash
import fem_gui.agent_authoring as agent_authoring_module
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.fixtures.planar_construction_phase0 import EXPECTED_H_CONSTRUCTION


_LEGACY_COMPOSITE_KINDS = {
    "planar_profiles",
    "extruded_profiles",
    "extruded_path_slot_plate",
    "path_swept_profile",
}
_VALIDATION_BUDGET_SECONDS = 0.5
_COMPILE_PREVIEW_BUDGET_SECONDS = 5.0
_PREVIEW_POINT_BUDGET = 4096
_PROVIDER_PLANAR_SCHEMA_BUDGET_BYTES = 32_768


class _ControllerDynamicTools:
    def __init__(self, controller) -> None:
        self.controller = controller
        self._snapshot = controller.set_published_tool_names(
            tuple(item.name for item in controller.definitions)
        )

    @property
    def definitions(self):
        return tuple(self.controller.definitions)

    @property
    def provider_snapshot(self):
        return self._snapshot

    def refresh_turn_snapshot(self, published_tool_names=()):
        names = tuple(published_tool_names) or tuple(
            item.name for item in self.controller.definitions
        )
        self._snapshot = self.controller.set_published_tool_names(names)
        return self._snapshot

    def dispatch(self, name, arguments, context):
        return self.controller.dispatch(name, arguments, context)


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


def _schema_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _rectangle_arguments() -> dict[str, object]:
    return {
        "part_function": "矩形板",
        "construction": {
            "schema_version": 1,
            "name": "矩形板",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 20,
                    "height": 10,
                }
            ],
            "result_node_id": "plate",
        },
        "output": "planar",
    }


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            tool_calls=(ToolCall(call_id, name, arguments),),
        ),
        finish_reason="tool_calls",
    )


def _max_nodes_ir() -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {
            "id": f"leaf-{index:02d}",
            "kind": "rectangle",
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        }
        for index in range(56)
    ]
    for group in range(4):
        nodes.append(
            {
                "id": f"group-{group}",
                "kind": "union",
                "operands": [
                    f"leaf-{index:02d}" for index in range(group * 14, (group + 1) * 14)
                ],
            }
        )
    nodes.extend(
        [
            {
                "id": "half-0",
                "kind": "union",
                "operands": ["group-0", "group-1"],
            },
            {
                "id": "half-1",
                "kind": "union",
                "operands": ["group-2", "group-3"],
            },
            {
                "id": "all",
                "kind": "union",
                "operands": ["half-0", "half-1"],
            },
            {
                "id": "result",
                "kind": "translate",
                "source": "all",
                "dx": 0.0,
                "dy": 0.0,
            },
        ]
    )
    assert len(nodes) == 64
    return {
        "schema_version": 1,
        "name": "validation-budget",
        "plane": "XY",
        "nodes": nodes,
        "result_node_id": "result",
    }


def test_phase6_provider_schema_has_one_planar_path_and_is_measured() -> None:
    _bridge, controller = _controller(ModelSession())
    definitions = {item.name: item for item in controller.definitions}
    general = definitions["prepare_geometry_proposal"]
    planar = definitions["prepare_planar_construction_proposal"]
    public_kinds = {
        branch["properties"]["kind"]["const"]
        for branch in general.parameters["properties"]["geometry"]["oneOf"]
    }

    assert public_kinds == {"wire", "box", "cylinder"}
    assert public_kinds.isdisjoint(_LEGACY_COMPOSITE_KINDS)
    serialized_public = json.dumps(general.parameters, ensure_ascii=False)
    assert not any(kind in serialized_public for kind in _LEGACY_COMPOSITE_KINDS)

    legacy_bytes = _schema_bytes(authoring_runtime._LEGACY_PREPARE_GEOMETRY.parameters)
    general_bytes = _schema_bytes(general.parameters)
    planar_bytes = _schema_bytes(planar.parameters)
    assert general_bytes < legacy_bytes
    assert planar_bytes <= _PROVIDER_PLANAR_SCHEMA_BUDGET_BYTES
    assert (
        general_bytes + planar_bytes
        < legacy_bytes + _PROVIDER_PLANAR_SCHEMA_BUDGET_BYTES
    )


@pytest.mark.parametrize(
    ("user_text", "operation", "dimension", "missing"),
    (
        ("创建一个 20×10 的矩形板", "planar_construction", 2, ()),
        ("新建一个半径 5 mm 的圆盘", "planar_construction", 2, ()),
        (
            "创建带组合槽的厚板并拉伸20mm",
            "planar_construction_extrusion",
            3,
            (),
        ),
        (
            "build a plate with a slot and extrude by 20 mm",
            "planar_construction_extrusion",
            3,
            (),
        ),
    ),
)
def test_phase6_new_planar_requests_route_only_to_ir(
    user_text: str,
    operation: str,
    dimension: int,
    missing: tuple[str, ...],
) -> None:
    hint = geometry_route_hint(user_text)

    assert hint is not None and hint.is_construction
    assert hint.requested_operation == operation
    assert hint.target_part_dimension == dimension
    assert hint.required_probe_tool == "read_authoring_context"
    assert hint.required_prepare_tool == "prepare_planar_construction_proposal"
    assert hint.missing_fields == missing


def test_phase6_existing_profile_transform_route_remains_dedicated() -> None:
    hint = geometry_route_hint("将当前二维轮廓拉伸 20 mm")

    assert hint is not None and hint.is_transform
    assert hint.required_prepare_tool == "prepare_profile_extrusion"


def test_phase6_follow_up_planar_cut_routes_through_geometry_edit_tools() -> None:
    hint = geometry_route_hint("当然，切除出S形状的槽即可")

    assert hint is not None and hint.is_edit
    assert hint.requested_operation == "planar_geometry_edit"
    assert hint.target_part_dimension == 2
    assert hint.required_probe_tool == "read_geometry_edit_context"
    assert hint.required_prepare_tool == "prepare_geometry_edit"
    assert geometry_route_hint("如果在H左边加上S字母，能做到吗？") is None
    direct = geometry_route_hint("请在H字母的左边加入一个字母S")
    assert direct is not None and direct.is_edit
    direction_fix = geometry_route_hint("还是不对，注意调整这个槽的开口方向")
    assert direction_fix is not None and direction_fix.is_edit
    regenerate = geometry_route_hint("重新生成这个槽轮廓")
    assert regenerate is not None and regenerate.is_edit


def test_phase6_legacy_decoder_is_deprecated_and_project_stores_only_recipe() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "兼容矩形板",
            "geometry": {
                "kind": "planar_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 10.0,
                        "height": 5.0,
                    }
                ],
            },
        },
        ToolExecutionContext("phase6-legacy", 0, "legacy-planar"),
    )

    assert result.ok
    assert result.data["authoring_path"] == "legacy_planar_profiles"
    assert result.data["compatibility_status"] == "deprecated"
    assert result.data["replacement_tool"] == ("prepare_planar_construction_proposal")
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED

    encoded = encode_project(session.prepare_project_save())
    serialized = json.dumps(encoded, ensure_ascii=False, sort_keys=True)
    assert "planar_construction" not in serialized.casefold()
    assert "construction_ir" not in serialized.casefold()
    reopened = decode_project(encoded).snapshot
    assert (
        reopened.parts[0].geometry_recipe == session.snapshot().parts[0].geometry_recipe
    )

    ir_session = ModelSession()
    ir_bridge, ir_controller = _controller(ir_session)
    ir_result = ir_controller.dispatch(
        "prepare_planar_construction_proposal",
        _rectangle_arguments(),
        ToolExecutionContext("phase6-ir-project", 0, "ir-project"),
    )
    assert ir_result.ok
    assert (
        ir_bridge.accept_from_gui_control(ir_result.data["proposal_id"]).state
        is ProposalState.SUCCEEDED
    )
    ir_encoded = encode_project(ir_session.prepare_project_save())
    ir_serialized = json.dumps(ir_encoded, ensure_ascii=False, sort_keys=True)
    assert "planar_construction" not in ir_serialized.casefold()
    assert "construction_ir" not in ir_serialized.casefold()
    assert (
        decode_project(ir_encoded).snapshot.parts[0].geometry_recipe
        == ir_session.snapshot().parts[0].geometry_recipe
    )


def test_phase6_provider_round_audit_proves_schema_route_and_one_call(tmp_path) -> None:
    session = ModelSession()
    _bridge, controller = _controller(session)
    dynamic = _ControllerDynamicTools(controller)
    provider = FakeProvider(
        [
            _tool(
                "prepare-ir",
                "prepare_planar_construction_proposal",
                _rectangle_arguments(),
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase6-routing-audit",
        provider,
        dynamic_tools=dynamic,
    )

    events = engine.send_message("创建一个 20×10 的矩形板")

    assert len(provider.requests) == 1
    assert [
        event.data["tool"]
        for event in events
        if event.event is EngineEventType.TOOL_STARTED
    ] == ["prepare_planar_construction_proposal"]
    request_definitions = {item.name: item for item in provider.requests[0].tools}
    general = request_definitions["prepare_geometry_proposal"]
    assert {
        branch["properties"]["kind"]["const"]
        for branch in general.parameters["properties"]["geometry"]["oneOf"]
    } == {"wire", "box", "cylinder"}

    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    entry = audit["entries"][-1]
    assert entry["route_hint"]["required_prepare_tool"] == (
        "prepare_planar_construction_proposal"
    )
    assert entry["tool_call_flags"]["called_tool_names"] == [
        "prepare_planar_construction_proposal"
    ]
    assert entry["published_tool_names"] == sorted(request_definitions)
    assert entry["schema_hashes"]["prepare_planar_construction_proposal"] == (
        tool_schema_hash(request_definitions["prepare_planar_construction_proposal"])
    )


def test_phase6_validation_budget_and_overbudget_never_enter_occ(monkeypatch) -> None:
    start = perf_counter()
    ir = PlanarConstructionIR.from_dict(_max_nodes_ir())
    canonical = ir.canonical_json()
    digest = ir.digest()
    validation_seconds = perf_counter() - start

    assert len(ir.nodes) == 64
    assert canonical and len(digest) == 64
    assert validation_seconds < _VALIDATION_BUDGET_SECONDS

    compile_start = perf_counter()
    compiled = compile_planar_construction(ir)
    compile_seconds = perf_counter() - compile_start
    assert compile_seconds < _COMPILE_PREVIEW_BUDGET_SECONDS
    assert compiled.proof.equivalent is True
    assert compiled.proof.material_profile_count == 1
    assert 0 < len(compiled.preview.points) <= _PREVIEW_POINT_BUDGET

    entered_occ = False

    def forbidden_compile(_construction):
        nonlocal entered_occ
        entered_occ = True
        raise AssertionError("overbudget IR entered OCC")

    monkeypatch.setattr(
        agent_authoring_module,
        "compile_planar_construction",
        forbidden_compile,
    )
    _bridge, controller = _controller(ModelSession())
    overbudget = deepcopy(_rectangle_arguments())
    overbudget["construction"]["nodes"] = [  # type: ignore[index]
        {
            "id": f"node-{index}",
            "kind": "rectangle",
            "x": index * 2.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        }
        for index in range(65)
    ]
    overbudget["construction"]["result_node_id"] = "node-64"  # type: ignore[index]
    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        overbudget,
        ToolExecutionContext("phase6-overbudget", 0, "overbudget"),
    )

    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.budget-exceeded"
    assert entered_occ is False


def test_phase6_boolean_compile_and_preview_stay_within_local_budget() -> None:
    construction = deepcopy(EXPECTED_H_CONSTRUCTION)
    start = perf_counter()
    compiled = compile_planar_construction(PlanarConstructionIR.from_dict(construction))
    compile_seconds = perf_counter() - start

    assert compile_seconds < _COMPILE_PREVIEW_BUDGET_SECONDS
    assert compiled.proof.equivalent is True
    assert compiled.proof.hole_count == 5
    assert 0 < len(compiled.preview.points) <= _PREVIEW_POINT_BUDGET
    assert compiled.preview.faces
