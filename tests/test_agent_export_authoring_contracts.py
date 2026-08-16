"""Contract tests for the Phase 1 Agent export DTOs, ports, and schemas."""

from __future__ import annotations

import pytest

from fem_agent.export_authoring import (
    EXPORT_AUTHORING_SCHEMA_VERSION,
    EXPORT_CSV_TOOL_NAME,
    NO_WORKSPACE_DIAGNOSTIC_CODE,
    NO_WORKSPACE_DIAGNOSTIC_MESSAGE,
    RESULT_DISPLAY_CONTEXT_TOOL_NAME,
    AgentExportBridge,
    ExportAuthoringError,
    ExportCsvRequest,
    ExportCsvResponse,
    ExportDiagnostic,
    ExportFileReceipt,
    FakeAgentExportPort,
    ResultDisplayContext,
    ResultDisplayContextResponse,
    ResultDisplayField,
    export_result_csv_tool_schema,
    result_display_context_tool_schema,
)
from fem_agent.result_authoring import AcceptedResultSource


_DIGEST = "a" * 64


def _receipt() -> ExportFileReceipt:
    return ExportFileReceipt(
        workspace_relative_path="agent_exports/位移_run-1.csv",
        filename="位移_run-1.csv",
        sha256=_DIGEST,
        size_bytes=128,
        kind="csv",
    )


def _source() -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=2,
        step_name="Static-1",
        run_id="run-1",
    )


def _request(**overrides: object) -> ExportCsvRequest:
    payload: dict[str, object] = {
        "expected_source": _source(),
        "expected_materialization_generation": 3,
        "field_ref": "U@nodes:c0",
        "component": "Magnitude",
    }
    payload.update(overrides)
    return ExportCsvRequest(**payload)  # type: ignore[arg-type]


def _field() -> ResultDisplayField:
    return ResultDisplayField(
        field_ref="U@nodes:c0",
        display_name="位移",
        components=("Magnitude", "U1", "U2"),
        unit="mm",
    )


def _context(**overrides: object) -> ResultDisplayContext:
    payload: dict[str, object] = {
        "fields": (_field(),),
        "display_settings": {"edges": True, "decimals": 3},
        "contour_settings": {"manual": False, "levels": 12},
        "selected_field_ref": "U@nodes:c0",
        "selected_component": "Magnitude",
        "deformation_scale": 1.0,
    }
    payload.update(overrides)
    return ResultDisplayContext(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_export_csv_schema_is_closed_and_pins_identity_block() -> None:
    schema = export_result_csv_tool_schema()
    assert schema["name"] == EXPORT_CSV_TOOL_NAME
    input_schema = schema["input_schema"]
    assert input_schema["additionalProperties"] is False
    assert set(input_schema["required"]) == {
        "schema_version",
        "expected_source",
        "expected_materialization_generation",
        "field_ref",
        "component",
    }
    assert set(input_schema["properties"]) == set(
        input_schema["required"]
    ) | {"name"}
    assert input_schema["properties"]["schema_version"] == {
        "type": "string",
        "const": EXPORT_AUTHORING_SCHEMA_VERSION,
    }
    source_schema = input_schema["properties"]["expected_source"]
    assert source_schema["additionalProperties"] is False
    assert set(source_schema["required"]) == {
        "result_id",
        "session_id",
        "artifact_id",
        "model_revision",
        "step_name",
        "run_id",
    }
    assert "no_workspace" not in input_schema
    assert "do not retry" in schema["description"]


def test_display_context_schema_is_closed_and_argument_free() -> None:
    schema = result_display_context_tool_schema()
    assert schema["name"] == RESULT_DISPLAY_CONTEXT_TOOL_NAME
    input_schema = schema["input_schema"]
    assert input_schema == {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }


# ---------------------------------------------------------------------------
# DTO bounds
# ---------------------------------------------------------------------------


def test_receipt_bounds_and_round_trip() -> None:
    receipt = _receipt()
    assert receipt.to_dict() == {
        "workspace_relative_path": "agent_exports/位移_run-1.csv",
        "filename": "位移_run-1.csv",
        "sha256": _DIGEST,
        "size_bytes": 128,
        "kind": "csv",
    }
    assert ExportFileReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ExportAuthoringError):
        ExportFileReceipt(
            workspace_relative_path="agent_exports\\a.csv",
            filename="a.csv",
            sha256=_DIGEST,
            size_bytes=1,
            kind="csv",
        )
    with pytest.raises(ExportAuthoringError):
        ExportFileReceipt(
            workspace_relative_path="agent_exports/a.csv",
            filename="a.csv",
            sha256="A" * 64,
            size_bytes=1,
            kind="csv",
        )
    with pytest.raises(ExportAuthoringError):
        ExportFileReceipt(
            workspace_relative_path="agent_exports/a.csv",
            filename="a.csv",
            sha256=_DIGEST,
            size_bytes=1,
            kind="pdf",
        )
    with pytest.raises(ExportAuthoringError):
        ExportFileReceipt.from_dict({**receipt.to_dict(), "extra": 1})


def test_response_requires_exactly_one_receipt_or_diagnostics() -> None:
    assert ExportCsvResponse.success(_receipt()).ok is True
    failure = ExportCsvResponse.failure(
        "export.busy",
        "busy",
        retryable=False,
        clarification_required=False,
    )
    assert failure.ok is False
    assert failure.to_dict()["diagnostics"][0]["code"] == "export.busy"
    with pytest.raises(ExportAuthoringError):
        ExportCsvResponse()
    with pytest.raises(ExportAuthoringError):
        ExportCsvResponse(receipt=_receipt(), diagnostics=(
            ExportDiagnostic(
                code="x.y",
                message="m",
                retryable=False,
                clarification_required=False,
            ),
        ))


def test_no_workspace_diagnostic_uses_fixed_message() -> None:
    response = ExportCsvResponse.no_workspace()
    assert response.ok is False
    payload = response.to_dict()
    assert payload["export_receipt"] is None
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == NO_WORKSPACE_DIAGNOSTIC_CODE
    assert diagnostic["message"] == NO_WORKSPACE_DIAGNOSTIC_MESSAGE
    assert diagnostic == {
        "code": "export.no_workspace",
        "message": (
            "尚未选择工作区，请先执行 /workspace 选择目录，"
            "导出文件将保存到该目录下的 agent_exports 中"
        ),
        "retryable": False,
        "clarification_required": True,
        "phase": "export",
    }


def test_request_from_dict_is_closed_and_name_is_optional() -> None:
    request = _request()
    payload = request.to_dict()
    assert payload["schema_version"] == EXPORT_AUTHORING_SCHEMA_VERSION
    assert payload["name"] is None
    assert ExportCsvRequest.from_dict(payload) == request
    named = ExportCsvRequest.from_dict({**payload, "name": "我的导出"})
    assert named.name == "我的导出"
    with pytest.raises(ExportAuthoringError):
        ExportCsvRequest.from_dict({**payload, "extra": 1})
    missing = dict(payload)
    del missing["field_ref"]
    with pytest.raises(ExportAuthoringError):
        ExportCsvRequest.from_dict(missing)
    with pytest.raises(ExportAuthoringError):
        ExportCsvRequest.from_dict({**payload, "schema_version": "9.9"})
    with pytest.raises(ExportAuthoringError):
        _request(field_ref=" x ")


def test_display_context_bounds() -> None:
    context = _context()
    payload = context.to_dict()
    assert payload["fields"][0]["display_name"] == "位移"
    assert payload["deformation_scale"] == 1.0
    with pytest.raises(ExportAuthoringError):
        _context(display_settings={"unknown_key": True})
    with pytest.raises(ExportAuthoringError):
        _context(contour_settings={"unknown_key": True})
    with pytest.raises(ExportAuthoringError):
        _context(selected_component="U1", selected_field_ref=None)
    with pytest.raises(ExportAuthoringError):
        _context(selected_component="Von Mises")
    with pytest.raises(ExportAuthoringError):
        _context(selected_field_ref="S@elements:c0")
    with pytest.raises(ExportAuthoringError):
        _context(deformation_scale=-1.0)
    with pytest.raises(ExportAuthoringError):
        _context(fields=(_field(), _field()))
    assert _context(selected_field_ref=None, selected_component=None).fields
    with pytest.raises(ExportAuthoringError):
        ResultDisplayField(
            field_ref="U@nodes:c0",
            display_name="位移",
            components=("Magnitude", "Magnitude"),
            unit="mm",
        )
    unitless = ResultDisplayField(
        field_ref="RF@nodes:c0",
        display_name="反力",
        components=("RF1",),
        unit="",
    )
    assert unitless.unit == ""


def test_display_context_response_bounds() -> None:
    response = ResultDisplayContextResponse.success(_context())
    assert response.ok is True
    assert response.to_dict()["display_context"]["selected_component"] == (
        "Magnitude"
    )
    with pytest.raises(ExportAuthoringError):
        ResultDisplayContextResponse()
    failure = ResultDisplayContextResponse.failure(
        "export.context.unavailable",
        "不可用",
        retryable=False,
        clarification_required=True,
    )
    assert failure.ok is False


# ---------------------------------------------------------------------------
# Fake port + bridge
# ---------------------------------------------------------------------------


def test_fake_port_records_calls_and_honors_registration() -> None:
    port = FakeAgentExportPort()
    request = _request()
    unconfigured = port.export_accepted_result_csv(request)
    assert unconfigured.ok is False
    assert unconfigured.diagnostics[0].code == "export.not_configured"
    assert port.export_calls == [request]
    assert port.read_result_display_context().ok is False
    assert port.display_context_calls == 1

    success = ExportCsvResponse.success(_receipt())
    port.register_export(success)
    port.register_display_context(
        ResultDisplayContextResponse.success(_context())
    )
    assert port.export_accepted_result_csv(request) is success
    assert port.read_result_display_context().ok is True
    assert len(port.export_calls) == 2
    with pytest.raises(TypeError):
        port.export_accepted_result_csv({"not": "a request"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        port.register_export(object())  # type: ignore[arg-type]


def test_bridge_normalizes_requests_and_enforces_types() -> None:
    port = FakeAgentExportPort()
    bridge = AgentExportBridge(port)
    request = _request()
    assert bridge.export_csv(request.to_dict()).ok is False
    assert port.export_calls == [request]
    with pytest.raises(TypeError):
        AgentExportBridge(object())

    class _WrongReturnPort:
        def export_accepted_result_csv(self, request):
            return object()

        def export_viewport_image(self, request):
            return object()

        def read_result_display_context(self):
            return object()

    wrong = AgentExportBridge(_WrongReturnPort())
    with pytest.raises(TypeError):
        wrong.export_csv(request)
    with pytest.raises(TypeError):
        wrong.display_context()
