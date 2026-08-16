"""Phase 2 GUI 侧测试：视口导出 handler、事务式管线与回执卡片。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from fem_agent.export_authoring import (
    EXPORT_VIEWPORT_IMAGE_TOOL_NAME,
    NO_WORKSPACE_DIAGNOSTIC_CODE,
    ExportViewportImageRequest,
    ViewportImageOptions,
)
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import SessionExportPort
from fem_gui.agent_workspace import read_export_ledger
from fem_gui.main_window import FEMMainWindow
from tests.gui.test_agent_export_phase1 import (
    _FakeExportFacade,
    _FakeToolResult,
    _ViewportProbe,
    _controller_with_export,
    _display_context,
    _facade,
    _open_drawer,
)


def _viewport_request(
    *,
    format: str = "png",
    quality: int = 1,
    transparent: bool = False,
    display: dict[str, object] | None = None,
    contour: dict[str, object] | None = None,
) -> ExportViewportImageRequest:
    return ExportViewportImageRequest(
        image=ViewportImageOptions(
            format=format,
            quality=quality,
            transparent_background=transparent,
        ),
        display_overrides=display or {},
        contour_overrides=contour or {},
    )


# ---------------------------------------------------------------------------
# SessionExportPort handler
# ---------------------------------------------------------------------------


def test_viewport_export_without_workspace_returns_fixed_diagnostic(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path, workspace=None)
    facade.workspace = None
    port = SessionExportPort(facade)
    response = port.export_viewport_image(_viewport_request())
    assert response.ok is False
    assert response.diagnostics[0].code == NO_WORKSPACE_DIAGNOSTIC_CODE


def test_viewport_export_rejects_unavailable_viewport(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    facade.capture_available = False
    port = SessionExportPort(facade)
    response = port.export_viewport_image(_viewport_request())
    assert response.ok is False
    assert response.diagnostics[0].code == "export.viewport.unavailable"
    assert facade.viewport_calls == []


def test_viewport_export_rejects_contour_overrides_without_result(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = False
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(contour={"colormap": "jet"})
    )
    assert response.ok is False
    assert response.diagnostics[0].code == "export.contour.unavailable"
    # display 组不依赖结果显示，应当放行。
    display_only = port.export_viewport_image(
        _viewport_request(display={"legend": False})
    )
    assert display_only.ok is True
    assert len(facade.viewport_calls) == 1


def test_viewport_export_lands_png_with_receipt_and_ledger(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(
            quality=2,
            display={"legend": False, "decimals": 2},
            contour={"levels": 16},
        )
    )
    assert response.ok is True
    receipt = response.receipt
    assert receipt.kind == "png"
    assert receipt.filename.startswith("viewport_")
    assert receipt.filename.endswith(".png")
    landed = facade.workspace.root / receipt.workspace_relative_path
    assert landed.is_file()
    assert landed.parent.name == "agent_exports"
    target, options, overrides = facade.viewport_calls[-1]
    assert target == landed
    assert options == {"quality": 2, "transparent_background": False}
    assert overrides == {
        "display": {"legend": False, "decimals": 2},
        "contour": {"levels": 16},
    }
    records = read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )
    assert len(records) == 1
    assert records[0].kind == "png"
    assert records[0].tool == EXPORT_VIEWPORT_IMAGE_TOOL_NAME
    assert "image:pngq2" in records[0].overrides_summary
    assert "display:decimals,legend" in records[0].overrides_summary
    assert "contour:levels" in records[0].overrides_summary


def test_viewport_export_jpeg_uses_jpg_extension_and_ledger_kind(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(format="jpeg", transparent=True)
    )
    assert response.ok is True
    receipt = response.receipt
    assert receipt.kind == "jpeg"
    assert receipt.filename.endswith(".jpg")
    # jpeg 不支持透明背景；端口层只透传，事务层负责忽略。
    _, options, _ = facade.viewport_calls[-1]
    assert options["transparent_background"] is True
    records = read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )
    assert records[0].kind == "jpeg"


def test_viewport_export_increments_conflicting_names(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    first = port.export_viewport_image(_viewport_request())
    second = port.export_viewport_image(_viewport_request())
    assert first.ok is True and second.ok is True
    first_path = facade.workspace.root / first.receipt.workspace_relative_path
    second_path = facade.workspace.root / second.receipt.workspace_relative_path
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert "(1)" in second_path.name or "(2)" in second_path.name


def test_viewport_export_maps_facade_errors(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    port = SessionExportPort(facade)
    facade.viewport_error = RuntimeError(
        "a background task is already running"
    )
    busy = port.export_viewport_image(_viewport_request())
    assert busy.ok is False
    assert busy.diagnostics[0].code == "export.busy"
    assert busy.diagnostics[0].clarification_required is False
    facade.viewport_error = RuntimeError("the viewport cannot capture")
    unavailable = port.export_viewport_image(_viewport_request())
    assert unavailable.diagnostics[0].code == "export.viewport.unavailable"
    facade.viewport_error = OSError("workspace vanished")
    io_failure = port.export_viewport_image(_viewport_request())
    assert io_failure.diagnostics[0].code == "export.workspace.unavailable"
    assert (
        read_export_ledger(
            facade.agent_data_root(),
            facade.workspace.workspace_id,
        )
        == ()
    )


# ---------------------------------------------------------------------------
# 控制器接线：能力门控与 dispatch
# ---------------------------------------------------------------------------


def test_controller_gates_and_dispatches_viewport_export(
    tmp_path: Path,
) -> None:
    session, bridge, controller, facade, workspace = _controller_with_export(
        tmp_path
    )
    context = ToolExecutionContext("export-test", 0, "phase2")

    bridge.bind_snapshot(session.snapshot(), workspace_selected=False)
    controller.observe_binding(bridge.context)
    hidden = {item.name for item in controller.definitions}
    assert EXPORT_VIEWPORT_IMAGE_TOOL_NAME not in hidden

    facade.workspace = workspace
    facade.capture_available = False
    bridge.bind_snapshot(
        session.snapshot(),
        workspace_selected=True,
        viewport_capturable=False,
    )
    controller.observe_binding(bridge.context)
    uncapturable = {item.name for item in controller.definitions}
    assert EXPORT_VIEWPORT_IMAGE_TOOL_NAME not in uncapturable

    facade.capture_available = True
    bridge.bind_snapshot(
        session.snapshot(),
        workspace_selected=True,
        viewport_capturable=True,
    )
    controller.observe_binding(bridge.context)
    visible = {item.name for item in controller.definitions}
    # 视口导出在任何 ready 阶段发布，不要求结果存在。
    assert EXPORT_VIEWPORT_IMAGE_TOOL_NAME in visible

    result = controller.dispatch(
        EXPORT_VIEWPORT_IMAGE_TOOL_NAME,
        {"image": {"format": "jpeg", "quality": 2}},
        context,
    )
    assert result.ok is True
    receipt = result.data["export_receipt"]
    assert receipt["kind"] == "jpeg"
    landed = facade.workspace.root / receipt["workspace_relative_path"]
    assert landed.is_file()
    assert read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )

    # result 组在无已接受结果显示时由端口层诊断拒绝。
    rejected = controller.dispatch(
        EXPORT_VIEWPORT_IMAGE_TOOL_NAME,
        {"result": {"field_ref": "U@nodes:c0", "component": "U1"}},
        context,
    )
    assert rejected.ok is False

    facade.workspace = None
    denied = controller.dispatch(
        EXPORT_VIEWPORT_IMAGE_TOOL_NAME,
        {"image": {}},
        context,
    )
    assert denied.ok is False
    assert denied.data is None


# ---------------------------------------------------------------------------
# 事务式视口管线
# ---------------------------------------------------------------------------


def _contour_defaults() -> dict[str, object]:
    return {
        "manual": False,
        "minimum": 0.0,
        "maximum": 1.0,
        "levels": 12,
        "colormap": "rainbow",
        "style": "segmented",
        "legend": True,
        "edges": True,
        "edge_mode": "geometry",
        "show_coordinate_system": False,
        "averaging_threshold": 25.0,
    }


class _FakeViewport:
    """可断言事务快照/还原的最小视口替身。"""

    def __init__(self, contour_enabled: bool = True) -> None:
        self._contour = _contour_defaults()
        self._show_edges = True
        self._render_suppressed = False
        self._display = SimpleNamespace(contour_enabled=contour_enabled)
        self.can_capture = True
        self.screenshot_calls: list[tuple[str, int, bool, dict]] = []
        self.update_layer_calls = 0
        self.coordinate_calls = 0
        self.render_calls = 0
        self.screenshot_error: Exception | None = None

    def save_screenshot(
        self,
        path: str,
        scale: int = 1,
        transparent_background: bool = False,
    ) -> None:
        # 记录捕获瞬间的有效渲染配置，验证覆盖组确实被应用。
        self.screenshot_calls.append(
            (
                path,
                scale,
                transparent_background,
                dict(self._contour),
            )
        )
        if self.screenshot_error is not None:
            raise self.screenshot_error
        Path(path).write_bytes(b"\x89PNG fake")

    def _update_result_layer(self) -> None:
        self.update_layer_calls += 1

    def _refresh_coordinate_system_axes(self) -> None:
        self.coordinate_calls += 1

    def _render(self) -> None:
        self.render_calls += 1


class _WindowHarness:
    def __init__(
        self,
        viewport: _FakeViewport,
        *,
        busy: bool = False,
        result_provider: object = None,
    ) -> None:
        self.busy = busy
        self.viewport = viewport
        self._result_provider = result_provider

    def _current_result_provider(self) -> object:
        return self._result_provider


def test_transaction_applies_overrides_and_restores_key_by_key(
    tmp_path: Path,
) -> None:
    viewport = _FakeViewport()
    before_contour = dict(viewport._contour)
    harness = _WindowHarness(viewport, result_provider=object())
    target = tmp_path / "viewport.png"

    size, digest = FEMMainWindow.export_viewport_image_to(
        harness,
        target,
        {"quality": 2, "transparent_background": True},
        {
            "display": {"legend": False, "edge_mode": "none"},
            "contour": {"colormap": "jet", "levels": 24},
        },
    )

    assert size == target.stat().st_size
    assert len(digest) == 64
    path, scale, transparent, captured = viewport.screenshot_calls[0]
    assert path == str(target)
    assert scale == 2
    assert transparent is True
    # 捕获瞬间：显式覆盖键生效，edge_mode=none 推导出 edges=False。
    assert captured["colormap"] == "jet"
    assert captured["levels"] == 24
    assert captured["legend"] is False
    assert captured["edge_mode"] == "none"
    assert captured["edges"] is False
    # 未覆盖键沿用 live 状态。
    assert captured["manual"] is False
    # 事务期间重建了结果层。
    assert viewport.update_layer_calls >= 1
    # 导出后：_contour 逐键还原、_show_edges 与渲染抑制还原、重渲染一次。
    assert viewport._contour == before_contour
    assert list(viewport._contour) == list(before_contour)
    assert viewport._show_edges is True
    assert viewport._render_suppressed is False
    assert viewport.render_calls == 1


def test_transaction_restores_state_when_capture_fails(
    tmp_path: Path,
) -> None:
    viewport = _FakeViewport()
    before_contour = dict(viewport._contour)
    before_show_edges = viewport._show_edges
    viewport.screenshot_error = RuntimeError("capture failed mid-flight")
    harness = _WindowHarness(viewport, result_provider=object())

    with pytest.raises(RuntimeError, match="capture failed mid-flight"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            tmp_path / "viewport.png",
            {"quality": 1},
            {
                "display": {"legend": False},
                "contour": {"manual": True, "minimum": -5.0},
            },
        )

    # 中途失败后视口状态必须逐键还原。
    assert viewport._contour == before_contour
    assert viewport._show_edges == before_show_edges
    assert viewport._render_suppressed is False
    assert viewport.render_calls == 1


def test_transaction_validates_arguments(tmp_path: Path) -> None:
    viewport = _FakeViewport()
    harness = _WindowHarness(viewport, result_provider=object())
    target = tmp_path / "viewport.png"

    busy = _WindowHarness(viewport, busy=True)
    with pytest.raises(RuntimeError, match="background task"):
        FEMMainWindow.export_viewport_image_to(busy, target, None, None)
    with pytest.raises(ValueError, match="extension"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            tmp_path / "viewport.bmp",
            None,
            None,
        )
    with pytest.raises(ValueError, match="quality"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            {"quality": 3},
            None,
        )
    with pytest.raises(ValueError, match="only quality"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            {"width": 1024},
            None,
        )
    with pytest.raises(ValueError, match="only display, contour and result"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"camera": {"position": [0, 0, 0]}},
        )
    with pytest.raises(ValueError, match="unsupported keys"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"display": {"unknown_key": True}},
        )
    viewport.can_capture = False
    with pytest.raises(RuntimeError, match="cannot capture"):
        FEMMainWindow.export_viewport_image_to(harness, target, None, None)
    viewport.can_capture = True
    no_result = _WindowHarness(viewport, result_provider=None)
    with pytest.raises(RuntimeError, match="contour overrides require"):
        FEMMainWindow.export_viewport_image_to(
            no_result,
            target,
            None,
            {"contour": {"levels": 8}},
        )
    # display 组不依赖结果显示。
    FEMMainWindow.export_viewport_image_to(
        no_result,
        target,
        None,
        {"display": {"legend": False}},
    )
    assert viewport.screenshot_calls
    assert viewport.screenshot_calls[-1][3]["legend"] is False
    # jpeg 忽略透明背景。
    FEMMainWindow.export_viewport_image_to(
        harness,
        tmp_path / "viewport.jpg",
        {"transparent_background": True},
        None,
    )
    assert viewport.screenshot_calls[-1][2] is False


def test_transaction_keeps_screen_size_and_camera(tmp_path: Path) -> None:
    """屏幕视口尺寸与相机在导出后不变（离屏捕获管线自动保留）。"""

    _application = QApplication.instance() or QApplication([])
    assert _application is not None
    from fem_gui.widgets.viewport import FEMViewport

    viewport = FEMViewport()
    camera_state = object()
    camera_calls: list[tuple] = []
    camera = SimpleNamespace(
        copy=lambda: camera_state,
        DeepCopy=lambda state: camera_calls.append(("copy", state)),
        Modified=lambda: camera_calls.append(("modified",)),
    )
    plotter = SimpleNamespace(
        window_size=(720, 480),
        screenshot=lambda *args, **kwargs: None,
        camera=camera,
        renderer=SimpleNamespace(
            GetGradientBackground=lambda: False,
            GetBackground=lambda: (1.0, 1.0, 1.0),
            GetBackground2=lambda: (1.0, 1.0, 1.0),
            GetBackgroundAlpha=lambda: 1.0,
            GradientBackgroundOff=lambda: None,
            SetBackgroundAlpha=lambda _alpha: None,
            SetBackground=lambda *_color: None,
            SetBackground2=lambda *_color: None,
            SetGradientBackground=lambda _gradient: None,
        ),
        render=lambda: None,
        window_size_context=None,
    )
    viewport._plotter = plotter
    offscreen_calls: list[tuple] = []

    def _offscreen(path, size, *, transparent_background):
        offscreen_calls.append((path, size, transparent_background))
        Path(path).write_bytes(b"\x89PNG fake")

    viewport._save_offscreen_screenshot = _offscreen
    before_contour = dict(viewport._contour)
    harness = SimpleNamespace(
        busy=False,
        viewport=viewport,
        _current_result_provider=lambda: None,
    )
    target = tmp_path / "viewport.png"

    FEMMainWindow.export_viewport_image_to(
        harness,
        target,
        {"quality": 2},
        {"display": {"legend": False}},
    )

    assert target.is_file()
    assert offscreen_calls == [(str(target), (1440, 960), False)]
    assert tuple(plotter.window_size) == (720, 480)
    assert plotter.camera is camera
    assert camera_calls == [("copy", camera_state), ("modified",)]
    assert viewport._contour == before_contour
    assert viewport._render_suppressed is False

    viewport._plotter = None
    viewport.close()


# ---------------------------------------------------------------------------
# 聊天回执卡片
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ("png", "jpeg"))
def test_viewport_receipt_card_projects_png_and_jpeg(
    tmp_path: Path,
    gui_application,
    kind: str,
) -> None:
    host, drawer = _open_drawer(gui_application)
    from fem_gui.agent_workspace import UserWorkspace

    drawer.workspace_commands.user_workspace = UserWorkspace(
        workspace_id="ws-card",
        root=tmp_path,
    )
    extension = ".jpg" if kind == "jpeg" else ".png"
    receipt = {
        "workspace_relative_path": f"agent_exports/viewport_1{extension}",
        "filename": f"viewport_1{extension}",
        "sha256": "d" * 64,
        "size_bytes": 128,
        "kind": kind,
    }
    drawer._show_export_receipt(receipt)

    card = drawer.findChild(QWidget, "agentChatExportReceipt")
    assert card is not None
    assert card.property("exportPath") == receipt["workspace_relative_path"]


def test_runtime_forwards_viewport_receipts_to_drawer(
    tmp_path: Path,
    gui_application,
) -> None:
    host, drawer = _open_drawer(gui_application)
    from fem_gui.agent_workspace import UserWorkspace

    drawer.workspace_commands.user_workspace = UserWorkspace(
        workspace_id="ws-card",
        root=tmp_path,
    )
    receipt = {
        "workspace_relative_path": "agent_exports/viewport_2.png",
        "filename": "viewport_2.png",
        "sha256": "e" * 64,
        "size_bytes": 256,
        "kind": "png",
    }
    drawer.agent_runtime._notify_export_receipt_owner_thread(
        _FakeToolResult({"export_receipt": receipt})
    )
    assert drawer.findChild(QWidget, "agentChatExportReceipt") is not None
