"""Phase 1 GUI 侧测试：导出台账、CSV 导出 handler 与聊天回执卡片。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QLabel, QToolButton, QWidget

from fem_agent.export_authoring import (
    EXPORT_AUTHORING_SCHEMA_VERSION,
    EXPORT_CSV_TOOL_NAME,
    NO_WORKSPACE_DIAGNOSTIC_CODE,
    NO_WORKSPACE_DIAGNOSTIC_MESSAGE,
    ExportCsvRequest,
    ResultDisplayContext,
    ResultDisplayField,
)
from fem_agent.result_authoring import AcceptedResultSource, AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionExportPort,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_workspace import (
    ExportLedgerRecord,
    UserWorkspace,
    append_export_ledger_record,
    export_ledger_path,
    read_export_ledger,
)
from fem_gui.widgets.agent_chat import ModelViewportOverlayHost
from tests.gui.test_agent_result_query_phase_a7 import _solved_session


_FIELD_REF = "U@nodes:c0"
_DIGEST = "b" * 64


def _display_context() -> ResultDisplayContext:
    return ResultDisplayContext(
        fields=(
            ResultDisplayField(
                field_ref=_FIELD_REF,
                display_name="位移",
                components=("Magnitude", "U1"),
                unit="mm",
            ),
        ),
        display_settings={"edges": True},
        contour_settings={"manual": False},
        selected_field_ref=_FIELD_REF,
        selected_component="Magnitude",
        deformation_scale=1.0,
    )


def _source() -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=1,
        step_name="Static-1",
        run_id="run-1",
    )


def _request(**overrides: object) -> ExportCsvRequest:
    payload: dict[str, object] = {
        "expected_source": _source(),
        "expected_materialization_generation": 1,
        "field_ref": _FIELD_REF,
        "component": "Magnitude",
    }
    payload.update(overrides)
    return ExportCsvRequest(**payload)  # type: ignore[arg-type]


class _FakeExportFacade:
    """导出门面的 Fake：落盘动作写入真实临时目录以便断言。"""

    def __init__(
        self,
        workspace: UserWorkspace | None,
        agent_data_root: Path,
        context: ResultDisplayContext | Exception,
    ) -> None:
        self.workspace = workspace
        self._agent_data_root = agent_data_root
        self._context = context
        self.export_calls: list[tuple[Path, object]] = []
        self.export_error: Exception | None = None

    def current_workspace(self) -> UserWorkspace | None:
        return self.workspace

    def agent_data_root(self) -> Path:
        return self._agent_data_root

    def document_id(self) -> str:
        return "document-1"

    def result_display_context(self) -> ResultDisplayContext:
        if isinstance(self._context, Exception):
            raise self._context
        return self._context

    def resolve_result_selection(self, field_ref: str, component: str):
        return ScalarFieldSelection(
            FieldMaterializationKey(
                FieldRequest(
                    ResultFieldId(ResultVariable.U, FieldPosition.NODE)
                ),
                recovery_contract=1,
            ),
            component,
        )

    def export_result_csv_to(self, path, spec) -> tuple[int, str]:
        target = Path(path)
        self.export_calls.append((target, spec))
        if self.export_error is not None:
            raise self.export_error
        data = "节点,U1\n1,0.5\n".encode("utf-8")
        target.write_bytes(data)
        return len(data), hashlib.sha256(data).hexdigest()


def _facade(tmp_path: Path, *, workspace: UserWorkspace | None = None):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    if workspace is None:
        workspace = UserWorkspace(workspace_id="ws-1", root=root)
    return _FakeExportFacade(workspace, tmp_path / "agent-data", _display_context())


# ---------------------------------------------------------------------------
# 导出台账
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> ExportLedgerRecord:
    payload: dict[str, object] = {
        "display_path": "agent_exports/a.csv",
        "kind": "csv",
        "sha256": _DIGEST,
        "size_bytes": 12,
        "document_id": "document-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "materialization_generation": 1,
        "overrides_summary": "none",
        "exported_at": "2026-08-16T00:00:00+00:00",
        "tool": EXPORT_CSV_TOOL_NAME,
    }
    payload.update(overrides)
    return ExportLedgerRecord(**payload)  # type: ignore[arg-type]


def test_ledger_append_is_atomic_and_readable(tmp_path: Path) -> None:
    data_root = tmp_path / "agent-data"
    path = append_export_ledger_record(data_root, "ws-1", _record())
    assert path == export_ledger_path(data_root, "ws-1")
    assert path.is_file()
    ledger_directory = path.parent
    assert not [
        item
        for item in ledger_directory.iterdir()
        if item.name.startswith(".") or item.suffix == ".tmp"
    ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["workspace_id"] == "ws-1"
    assert payload["records"] == [_record().to_dict()]
    assert read_export_ledger(data_root, "ws-1") == (_record(),)
    assert read_export_ledger(data_root, "missing-ws") == ()


def test_ledger_accumulates_across_sessions(tmp_path: Path) -> None:
    data_root = tmp_path / "agent-data"
    first = _record(session_id="session-1")
    second = _record(
        session_id="session-2",
        display_path="agent_exports/a(1).csv",
    )
    append_export_ledger_record(data_root, "ws-1", first)
    append_export_ledger_record(data_root, "ws-1", second)
    assert read_export_ledger(data_root, "ws-1") == (first, second)


def test_ledger_isolated_by_workspace_id(tmp_path: Path) -> None:
    data_root = tmp_path / "agent-data"
    record_a = _record(session_id="session-a")
    record_b = _record(session_id="session-b")
    append_export_ledger_record(data_root, "ws-1", record_a)
    append_export_ledger_record(data_root, "ws-2", record_b)
    assert read_export_ledger(data_root, "ws-1") == (record_a,)
    assert read_export_ledger(data_root, "ws-2") == (record_b,)
    assert (
        export_ledger_path(data_root, "ws-1")
        != export_ledger_path(data_root, "ws-2")
    )


# ---------------------------------------------------------------------------
# SessionExportPort handler
# ---------------------------------------------------------------------------


def test_export_without_workspace_returns_fixed_diagnostic(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    facade.workspace = None
    port = SessionExportPort(facade)
    response = port.export_accepted_result_csv(_request())
    assert response.ok is False
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == NO_WORKSPACE_DIAGNOSTIC_CODE
    assert diagnostic.message == NO_WORKSPACE_DIAGNOSTIC_MESSAGE
    assert diagnostic.retryable is False
    assert facade.export_calls == []


def test_export_lands_in_agent_exports_with_receipt_and_ledger(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    response = port.export_accepted_result_csv(_request())
    assert response.ok is True
    receipt = response.receipt
    assert receipt is not None
    assert receipt.kind == "csv"
    assert receipt.filename == "位移_run-1.csv"
    assert receipt.workspace_relative_path == "agent_exports/位移_run-1.csv"
    landed = facade.workspace.root / receipt.workspace_relative_path
    assert landed.is_file()
    assert receipt.sha256 == hashlib.sha256(landed.read_bytes()).hexdigest()
    assert receipt.size_bytes == landed.stat().st_size
    records = read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )
    assert len(records) == 1
    assert records[0].display_path == receipt.workspace_relative_path
    assert records[0].tool == EXPORT_CSV_TOOL_NAME
    assert records[0].document_id == "document-1"
    assert records[0].overrides_summary == "none"


def test_export_conflict_increments_and_never_overwrites(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    first = port.export_accepted_result_csv(_request())
    second = port.export_accepted_result_csv(_request(name="位移_run-1"))
    third = port.export_accepted_result_csv(_request(name="位移_run-1"))
    assert first.receipt.filename == "位移_run-1.csv"
    assert second.receipt.filename == "位移_run-1(1).csv"
    assert third.receipt.filename == "位移_run-1(2).csv"
    assert len(facade.export_calls) == 3
    assert len({path for path, _spec in facade.export_calls}) == 3


def test_export_rejects_busy_and_stale_identity(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    facade.export_error = RuntimeError("a background task is already running")
    busy = port.export_accepted_result_csv(_request())
    assert busy.ok is False
    assert busy.diagnostics[0].code == "export.busy"
    assert busy.diagnostics[0].clarification_required is False
    facade.export_error = ValueError(
        "materialization generation does not match"
    )
    stale = port.export_accepted_result_csv(_request())
    assert stale.ok is False
    assert stale.diagnostics[0].code == "export.result.stale"
    facade.export_error = OSError("workspace vanished")
    unavailable = port.export_accepted_result_csv(_request())
    assert unavailable.ok is False
    assert unavailable.diagnostics[0].code == "export.workspace.unavailable"
    records = read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )
    assert records == ()


def test_export_rejects_unknown_field_and_component(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    unknown_field = port.export_accepted_result_csv(
        _request(field_ref="S@elements:c0")
    )
    assert unknown_field.ok is False
    assert unknown_field.diagnostics[0].code == "export.field.unknown"
    assert _FIELD_REF in unknown_field.diagnostics[0].message
    unknown_component = port.export_accepted_result_csv(
        _request(component="Von Mises")
    )
    assert unknown_component.ok is False
    assert unknown_component.diagnostics[0].code == "export.component.unknown"
    assert facade.export_calls == []


def test_display_context_port_reports_unavailability(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    ok = port.read_result_display_context()
    assert ok.ok is True
    assert ok.context is not None
    assert ok.context.fields[0].field_ref == _FIELD_REF
    facade._context = RuntimeError("no READY result")
    failed = port.read_result_display_context()
    assert failed.ok is False
    assert failed.diagnostics[0].code == "export.context.unavailable"


# ---------------------------------------------------------------------------
# 控制器接线：能力门控与 dispatch
# ---------------------------------------------------------------------------


def _controller_with_export(tmp_path: Path):
    session = _solved_session()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(exist_ok=True)
    workspace = UserWorkspace(workspace_id="ws-ctl", root=workspace_root)
    facade = _FakeExportFacade(
        None,
        tmp_path / "agent-data",
        _display_context(),
    )
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
        export_facade=facade,
    )
    return session, bridge, controller, facade, workspace


def test_controller_gates_and_dispatches_export_tools(tmp_path: Path) -> None:
    session, bridge, controller, facade, workspace = _controller_with_export(
        tmp_path
    )
    context = ToolExecutionContext("export-test", 0, "phase1")

    bridge.bind_snapshot(session.snapshot(), workspace_selected=False)
    controller.observe_binding(bridge.context)
    hidden = {item.name for item in controller.definitions}
    assert EXPORT_CSV_TOOL_NAME not in hidden
    assert "read_result_display_context" in hidden

    facade.workspace = workspace
    bridge.bind_snapshot(session.snapshot(), workspace_selected=True)
    controller.observe_binding(bridge.context)
    visible = {item.name for item in controller.definitions}
    assert EXPORT_CSV_TOOL_NAME in visible

    read_result = controller.dispatch(
        "read_result_display_context",
        {},
        context,
    )
    assert read_result.ok is True
    assert read_result.data["display_context"]["fields"][0][
        "field_ref"
    ] == _FIELD_REF

    arguments = _request().to_dict()
    arguments["schema_version"] = EXPORT_AUTHORING_SCHEMA_VERSION
    export_result = controller.dispatch(
        EXPORT_CSV_TOOL_NAME,
        arguments,
        context,
    )
    assert export_result.ok is True
    receipt = export_result.data["export_receipt"]
    landed = facade.workspace.root / receipt["workspace_relative_path"]
    assert landed.is_file()
    assert read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )

    # 工作区消失后工具同步从可用集合中下线，dispatch 拒绝执行。
    facade.workspace = None
    denied = controller.dispatch(EXPORT_CSV_TOOL_NAME, arguments, context)
    assert denied.ok is False
    assert denied.data is None
    assert denied.diagnostics


# ---------------------------------------------------------------------------
# 聊天回执卡片
# ---------------------------------------------------------------------------


class _ViewportProbe(QWidget):
    nativeSurfaceUpdated = Signal()


class _FakeToolResult:
    def __init__(self, data: dict[str, object]) -> None:
        self.ok = True
        self.data = data


def _open_drawer(application: QApplication):
    host = ModelViewportOverlayHost(_ViewportProbe())
    host.resize(720, 520)
    host.show()
    host.set_drawer_open(True, animated=False)
    application.processEvents()
    application.processEvents()
    return host, host.agent_chat_drawer


def test_export_receipt_card_projects_and_opens(
    tmp_path: Path,
    monkeypatch,
    gui_application,
) -> None:
    host, drawer = _open_drawer(gui_application)
    drawer.workspace_commands.user_workspace = UserWorkspace(
        workspace_id="ws-card",
        root=tmp_path,
    )
    exports = tmp_path / "agent_exports"
    exports.mkdir()
    receipt = {
        "workspace_relative_path": "agent_exports/a.csv",
        "filename": "a.csv",
        "sha256": "a" * 64,
        "size_bytes": 3,
        "kind": "csv",
    }
    drawer._show_export_receipt(receipt)

    card = drawer.findChild(QWidget, "agentChatExportReceipt")
    assert card is not None
    texts = [label.text() for label in drawer.findChildren(QLabel)]
    assert "a.csv" in texts
    assert "agent_exports/a.csv" in texts

    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: (opened.append(url), True)[1],
    )
    button = drawer.findChild(QToolButton, "agentChatExportOpenButton")
    assert button is not None
    assert button.text() == "打开"
    # 文件缺失：降级为打开所在目录。
    button.click()
    assert opened == [QUrl.fromLocalFile(str(exports))]
    # 文件存在：直接打开文件。
    file_path = exports / "a.csv"
    file_path.write_text("x", encoding="utf-8")
    opened.clear()
    button.click()
    assert opened == [QUrl.fromLocalFile(str(file_path))]


def test_runtime_forwards_export_receipts_to_drawer(
    tmp_path: Path,
    gui_application,
) -> None:
    host, drawer = _open_drawer(gui_application)
    drawer.workspace_commands.user_workspace = UserWorkspace(
        workspace_id="ws-card",
        root=tmp_path,
    )
    receipt = {
        "workspace_relative_path": "agent_exports/b.csv",
        "filename": "b.csv",
        "sha256": "c" * 64,
        "size_bytes": 4,
        "kind": "csv",
    }
    drawer.agent_runtime._notify_export_receipt_owner_thread(
        _FakeToolResult({"export_receipt": receipt})
    )
    assert drawer.findChild(QWidget, "agentChatExportReceipt") is not None
    # 失败或无回执的结果不投影任何卡片。
    drawer.agent_runtime._notify_export_receipt_owner_thread(
        _FakeToolResult({"other": 1})
    )
    cards = [
        item
        for item in drawer.findChildren(QWidget)
        if item.objectName() == "agentChatExportReceipt"
    ]
    assert len(cards) == 1
