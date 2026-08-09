import math

import pytest

from fem.io import inp as abaqus
from fem.solvers import static_linear
from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import SessionPhase
from fem_agent.tools.registry import ToolExecutionContext
from tests.helpers.abaqus_builders import (
    write_hex20_block_inp,
    write_perforated_plate_style_inp,
)


pytestmark = pytest.mark.integration


def _tool_response(*calls):
    return ProviderResponse(
        AssistantMessage("assistant", tool_calls=tuple(calls)),
        finish_reason="tool_calls",
    )


def _text_response(text):
    return ProviderResponse(
        AssistantMessage("assistant", content=text),
        finish_reason="stop",
    )


@pytest.mark.integration
def test_full_fake_provider_confirmation_worker_and_export_matches_direct_fem(
    tmp_path,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "e2e_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "e2e_units",
                    "set_unit_context",
                    {
                        "length": "mm",
                        "force": "N",
                        "stress": "MPa",
                        "density": "tonne/mm^3",
                        "acceleration": "mm/s^2",
                    },
                ),
                ToolCall(
                    "e2e_results",
                    "set_result_requests",
                    {
                        "queries": [
                            {
                                "kind": "max_displacement_magnitude",
                                "edge": "Surf-right",
                            },
                            {
                                "kind": "reaction_sum",
                                "component": 1,
                                "node_set": "Set-left",
                            },
                        ],
                        "export_formats": ["csv", "vtk"],
                    },
                ),
            ),
            _tool_response(
                ToolCall("e2e_summary", "get_analysis_summary", {})
            ),
            _text_response("摘要已准备好，请输入 /confirm。"),
        ]
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        provider,
        session_id="ses_e2e",
    )
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)

    engine.send_message("使用这些单位，求最大位移与左侧合反力并导出。")
    assert engine.get_snapshot().phase == SessionPhase.AWAITING_CONFIRMATION

    run_events = engine.confirm_revision()

    completed = next(
        event
        for event in run_events
        if event.event == EngineEventType.RUN_COMPLETED
    )
    assert completed.data["status"] == "succeeded"
    summary = completed.data["result_summary"]
    maximum = next(
        scalar
        for scalar in summary["scalars"]
        if scalar["query_kind"] == "max_displacement_magnitude"
    )

    direct_model = abaqus.read(source)
    direct_result = static_linear.solve(
        direct_model,
        step=direct_model.steps[-1],
    )
    edge_node_ids = tuple(
        dict.fromkeys(
            node_id
            for edge in direct_model.edges["Surf-right"].edges
            for node_id in edge.node_ids
        )
    )
    direct_maximum = max(
        math.sqrt(
            sum(
                direct_result.nodal_displacement(node_id, component) ** 2
                for component in (1, 2)
            )
        )
        for node_id in edge_node_ids
    )
    assert maximum["value"] == pytest.approx(direct_maximum)
    assert maximum["region"] == "Surf-right"

    snapshot = engine.get_snapshot()
    assert snapshot.phase == SessionPhase.SOLVED
    assert snapshot.active_run_id == completed.data["run_id"]
    export_kinds = {
        item["kind"] for item in completed.data["artifacts"]
    }
    assert {"csv", "vtk", "manifest", "diagnostics", "result_summary"} <= export_kinds

    provider.queue(
        _tool_response(
            ToolCall(
                "e2e_changed_units",
                "set_unit_context",
                {
                    "length": "m",
                    "force": "N",
                    "stress": "Pa",
                    "density": "kg/m^3",
                    "acceleration": "m/s^2",
                },
            )
        ),
        _text_response("单位上下文已变更，需要重新确认。"),
    )
    engine.send_message("把单位上下文改成 SI。")
    changed = engine.get_snapshot()

    assert changed.phase == SessionPhase.AWAITING_CONFIRMATION
    assert changed.active_run_id is None
    review_events = engine.confirm_revision()
    assert any(
        event.event == EngineEventType.ANALYSIS_SUMMARY
        for event in review_events
    )
    assert not any(
        event.event == EngineEventType.RUN_COMPLETED
        for event in review_events
    )
    assert engine.get_snapshot().active_run_id is None
    reopened = AgentSessionEngine(
        workspace,
        FakeProvider(),
        session_id=engine.session_id,
    )
    assert reopened.get_snapshot().active_run_id is None


def test_confirmation_is_invalidated_by_a_new_specification_revision(tmp_path):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "stale_confirmation.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(workspace, FakeProvider(), session_id="ses_stale")
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(engine.session_id, first.revision, "units_once"),
    )
    after_units = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": ["csv"],
        },
        ToolExecutionContext(
            engine.session_id,
            after_units.revision,
            "results_once",
        ),
    )
    ready = engine.revisions.require_current(engine.session_id)
    engine.confirmations.confirm(
        engine.session_id,
        revision=ready.revision,
        revision_hash=ready.revision_hash,
    )
    assert engine.confirmations.is_confirmed(
        engine.session_id,
        revision=ready.revision,
        revision_hash=ready.revision_hash,
    )

    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "m",
            "force": "N",
            "stress": "Pa",
            "density": "kg/m^3",
            "acceleration": "m/s^2",
        },
        ToolExecutionContext(
            engine.session_id,
            ready.revision,
            "changed_units",
        ),
    )

    current = engine.revisions.require_current(engine.session_id)
    assert current.revision == ready.revision + 1
    assert not engine.confirmations.is_confirmed(
        engine.session_id,
        revision=current.revision,
        revision_hash=current.revision_hash,
    )


@pytest.mark.integration
def test_hex20_agent_pipeline_matches_direct_fem_result(tmp_path):
    source = write_hex20_block_inp(tmp_path, "agent_hex20.inp")
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "hex20_units",
                    "set_unit_context",
                    {
                        "length": "mm",
                        "force": "N",
                        "stress": "MPa",
                        "density": "tonne/mm^3",
                        "acceleration": "mm/s^2",
                    },
                ),
                ToolCall(
                    "hex20_requests",
                    "set_result_requests",
                    {
                        "queries": [
                            {"kind": "max_displacement_magnitude"},
                            {
                                "kind": "reaction_sum",
                                "component": 1,
                                "node_set": "Set-fixed",
                            },
                        ],
                        "export_formats": ["csv"],
                    },
                ),
            ),
            _tool_response(
                ToolCall("hex20_summary", "get_analysis_summary", {})
            ),
            _text_response("Hex20 摘要已准备好，请输入 /confirm。"),
        ]
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        provider,
        session_id="ses_hex20_e2e",
    )
    artifact = ArtifactStore(workspace).copy_input(
        engine.session_id,
        source,
    )
    engine.attach_artifact(artifact.artifact_id)

    engine.send_message("检查 Hex20，查询最大位移和固定端合反力。")
    completed = next(
        event
        for event in engine.confirm_revision()
        if event.event == EngineEventType.RUN_COMPLETED
    )

    assert completed.data["status"] == "succeeded"
    summary = completed.data["result_summary"]
    agent_maximum = next(
        scalar["value"]
        for scalar in summary["scalars"]
        if scalar["query_kind"] == "max_displacement_magnitude"
    )
    direct_model = abaqus.read(source)
    direct_result = static_linear.solve(
        direct_model,
        step=direct_model.steps[-1],
    )
    direct_maximum = max(
        math.sqrt(
            sum(
                direct_result.nodal_displacement(
                    node_id,
                    component,
                )
                ** 2
                for component in (1, 2, 3)
            )
        )
        for node_id in direct_model.mesh.node_ids
    )
    assert agent_maximum == pytest.approx(direct_maximum)

    provider.queue(
        _tool_response(
            ToolCall(
                "hex20_surface_query",
                "query_results",
                {
                    "queries": [
                        {
                            "kind": "max_displacement_magnitude",
                            "surface": "Surf-loaded",
                        }
                    ]
                },
            )
        ),
        _text_response("已分析加载表面的最大位移。"),
    )
    queried = engine.send_message("加载表面 Surf-loaded 的最大位移是多少？")
    tool_result = next(
        event.data["result"]
        for event in queried
        if event.event == EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "query_results"
    )
    surface_scalar = tool_result["data"]["result_summary"]["scalars"][0]
    assert tool_result["ok"] is True
    assert surface_scalar["region"] == "Surf-loaded"
    assert surface_scalar["value"] == pytest.approx(direct_maximum)
    assert engine.get_snapshot().phase == SessionPhase.SOLVED
    assert len(
        list(
            (
                workspace
                / "sessions"
                / engine.session_id
                / "runs"
            ).iterdir()
        )
    ) == 1
