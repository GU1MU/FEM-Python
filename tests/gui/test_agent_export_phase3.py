"""Phase 3 GUI 侧测试：result 参数组的端口诊断与事务式视口管线。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem_agent.export_authoring import (
    ExportViewportImageRequest,
    ViewportImageOptions,
)
from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.agent_authoring import SessionExportPort
from fem_gui.agent_workspace import read_export_ledger
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.scene import DisplayState
from tests.gui.test_agent_export_phase1 import (
    _FakeExportFacade,
    _display_context,
    _facade,
)
from tests.gui.test_agent_export_phase2 import _contour_defaults


_FIELD_REF = "U@nodes:c0"


def _scalar_selection() -> ScalarFieldSelection:
    return ScalarFieldSelection(
        FieldMaterializationKey(
            FieldRequest(
                ResultFieldId(ResultVariable.U, FieldPosition.NODE)
            ),
            recovery_contract=1,
        ),
        "Magnitude",
    )


def _viewport_request(
    *,
    display: dict[str, object] | None = None,
    contour: dict[str, object] | None = None,
    result: dict[str, object] | None = None,
) -> ExportViewportImageRequest:
    return ExportViewportImageRequest(
        image=ViewportImageOptions(),
        display_overrides=display or {},
        contour_overrides=contour or {},
        result_overrides=result or {},
    )


# ---------------------------------------------------------------------------
# SessionExportPort handler：result 组诊断
# ---------------------------------------------------------------------------


def test_port_rejects_result_group_without_displayed_result(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = False
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(result={"shape_mode": "undeformed"})
    )
    assert response.ok is False
    assert response.diagnostics[0].code == "export.result.unavailable"
    assert facade.viewport_calls == []
    # display 组不依赖结果显示，应当放行。
    display_only = port.export_viewport_image(
        _viewport_request(display={"legend": False})
    )
    assert display_only.ok is True
    assert len(facade.viewport_calls) == 1


def test_port_rejects_unknown_field_ref_with_ready_list(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(
            result={"field_ref": "S@elements:c0", "component": "S11"}
        )
    )
    assert response.ok is False
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "export.field.unknown"
    assert _FIELD_REF in diagnostic.message
    assert facade.viewport_calls == []


def test_port_rejects_component_outside_field(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(
            result={"field_ref": _FIELD_REF, "component": "Von Mises"}
        )
    )
    assert response.ok is False
    assert response.diagnostics[0].code == "export.component.unknown"
    assert facade.viewport_calls == []


def test_port_requires_field_ref_and_component_together(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(result={"field_ref": _FIELD_REF})
    )
    assert response.ok is False
    assert response.diagnostics[0].code == "export.result.incomplete"
    assert facade.viewport_calls == []


def test_port_forwards_result_group_and_ledger_summary(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(
            contour={"levels": 16},
            result={
                "field_ref": _FIELD_REF,
                "component": "Magnitude",
                "shape_mode": "deformed",
            },
        )
    )
    assert response.ok is True
    target, options, overrides = facade.viewport_calls[-1]
    assert overrides["result"] == {
        "field_ref": _FIELD_REF,
        "component": "Magnitude",
        "shape_mode": "deformed",
    }
    assert overrides["contour"] == {"levels": 16}
    records = read_export_ledger(
        facade.agent_data_root(),
        facade.workspace.workspace_id,
    )
    assert len(records) == 1
    assert "result:component,field_ref,shape_mode" in (
        records[0].overrides_summary
    )
    assert "contour:levels" in records[0].overrides_summary


def test_port_maps_not_ready_field_keyerror_to_ready_list(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    facade.viewport_error = KeyError(
        "only a READY catalog field can be rendered"
    )
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(
            result={"field_ref": _FIELD_REF, "component": "Magnitude"}
        )
    )
    assert response.ok is False
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "export.field.not_ready"
    # 诊断附上当前 READY 场清单，且不落盘、不登记台账。
    assert _FIELD_REF in diagnostic.message
    assert (
        read_export_ledger(
            facade.agent_data_root(),
            facade.workspace.workspace_id,
        )
        == ()
    )


def test_port_maps_result_precondition_error(tmp_path: Path) -> None:
    facade = _facade(tmp_path)
    facade.result_displayed = True
    facade.viewport_error = RuntimeError(
        "result overrides require a displayed accepted result"
    )
    port = SessionExportPort(facade)
    response = port.export_viewport_image(
        _viewport_request(result={"shape_mode": "undeformed"})
    )
    assert response.ok is False
    assert response.diagnostics[0].code == "export.result.unavailable"


# ---------------------------------------------------------------------------
# 事务式视口管线：result 组
# ---------------------------------------------------------------------------


class _ResultFakeViewport:
    """带结果层状态的最小视口替身，可断言事务快照/逐键还原。"""

    def __init__(self, contour_enabled: bool = True) -> None:
        self._contour = _contour_defaults()
        self._show_edges = True
        self._render_suppressed = False
        self._display = DisplayState("deformed", contour_enabled)
        self._result_render_payload = object()
        self._overlay_undeformed = False
        self._scalar_reuse_pending = False
        self._geometry_reuse_pending = False
        self._scalar_reuse_display = None
        self.can_capture = True
        # 捕获瞬间的完整有效状态，验证覆盖组确实被应用。
        self.screenshot_calls: list[tuple] = []
        self.update_layer_calls = 0
        self.coordinate_calls = 0
        self.render_calls = 0
        self.overlay_refresh_calls = 0
        self.installed_payloads: list[object] = []
        self.screenshot_error: Exception | None = None
        self.update_layer_error: Exception | None = None

    def set_result_render_payload(self, payload: object) -> None:
        self.installed_payloads.append(payload)
        self._result_render_payload = payload

    def save_screenshot(
        self,
        path: str,
        scale: int = 1,
        transparent_background: bool = False,
    ) -> None:
        self.screenshot_calls.append(
            (
                path,
                scale,
                transparent_background,
                dict(self._contour),
                self._display,
                self._result_render_payload,
                self._overlay_undeformed,
            )
        )
        if self.screenshot_error is not None:
            raise self.screenshot_error
        Path(path).write_bytes(b"\x89PNG fake")

    def _update_result_layer(self) -> None:
        self.update_layer_calls += 1
        if self.update_layer_error is not None:
            error = self.update_layer_error
            # 只抛一次：finally 里的还原重建不应再次失败。
            self.update_layer_error = None
            raise error

    def _refresh_undeformed_overlay(self) -> None:
        self.overlay_refresh_calls += 1

    def _refresh_coordinate_system_axes(self) -> None:
        self.coordinate_calls += 1

    def _render(self) -> None:
        self.render_calls += 1


class _ResultWindowHarness:
    """窗口替身：result 组依赖的方法全部可断言。"""

    def __init__(self, viewport: _ResultFakeViewport) -> None:
        self.busy = False
        self.viewport = viewport
        self._provider: object = object()
        self._contour_options = {"averaging_threshold": 25.0}
        self._display = DisplayState("deformed", True)
        self._scale_mode = "auto"
        self._scale_value = 1.0
        self._overlay_undeformed = False
        self.result_selection = _scalar_selection()
        self.resolved: tuple[str, str] | None = None
        self.averaging_calls: list[tuple[object, float]] = []
        self.payload_build_calls: list[tuple] = []
        self.build_payload_error: Exception | None = None
        self.built_payload = object()

    def _current_result_provider(self) -> object:
        return self._provider

    def _agent_resolve_result_selection(self, field_ref, component):
        self.resolved = (field_ref, component)
        return self.result_selection

    def _result_averaging_visual_selection(self, provider, selection):
        self.averaging_calls.append(
            (selection, float(self._contour_options["averaging_threshold"]))
        )
        return ("visual", selection)

    def _build_result_render_payload(
        self,
        provider,
        selection,
        *,
        shape_mode=None,
        scale_mode=None,
        scale_value=None,
    ):
        self.payload_build_calls.append(
            (selection, shape_mode, scale_mode, scale_value)
        )
        if self.build_payload_error is not None:
            raise self.build_payload_error
        return self.built_payload

    # 复用主窗口的真实 result 组构建逻辑（含 averaging 阈值前置生效）。
    _export_viewport_result_payload = (
        FEMMainWindow._export_viewport_result_payload
    )


def test_result_transaction_applies_overrides_and_restores_key_by_key(
    tmp_path: Path,
) -> None:
    viewport = _ResultFakeViewport()
    initial_payload = viewport._result_render_payload
    before_contour = dict(viewport._contour)
    before_display = viewport._display
    harness = _ResultWindowHarness(viewport)
    target = tmp_path / "viewport.png"

    size, digest = FEMMainWindow.export_viewport_image_to(
        harness,
        target,
        {"quality": 1},
        {
            "contour": {"colormap": "jet"},
            "result": {
                "field_ref": _FIELD_REF,
                "component": "Magnitude",
                "shape_mode": "undeformed",
                "overlay_undeformed": True,
            },
        },
    )

    assert size == target.stat().st_size
    assert len(digest) == 64
    (
        path,
        scale,
        transparent,
        captured_contour,
        captured_display,
        captured_payload,
        captured_overlay,
    ) = viewport.screenshot_calls[0]
    assert path == str(target)
    assert scale == 1
    assert transparent is False
    # 捕获瞬间：contour 覆盖与 result 组的脱体 payload/形状/叠加生效。
    assert captured_contour["colormap"] == "jet"
    assert captured_display == DisplayState("undeformed", True)
    assert captured_payload is harness.built_payload
    assert captured_overlay is True
    # 临时 payload 通过 set_result_render_payload 安装，finally 还原
    # 时再经同一接口装回原 payload。
    assert viewport.installed_payloads == [
        harness.built_payload,
        initial_payload,
    ]
    # field_ref/component 走既有解析路径。
    assert harness.resolved == (_FIELD_REF, "Magnitude")
    # 导出后：_contour/_display/_result_render_payload/_overlay_undeformed
    # 逐键还原，渲染抑制解除并重渲染一次。
    assert viewport._contour == before_contour
    assert viewport._display is before_display
    assert viewport._result_render_payload is initial_payload
    assert viewport._overlay_undeformed is False
    assert viewport._show_edges is True
    assert viewport._render_suppressed is False
    assert viewport.render_calls == 1
    assert viewport.update_layer_calls >= 1
    # 还原时未变形叠加被刷新回关闭状态。
    assert viewport.overlay_refresh_calls == 2


def test_result_transaction_restores_state_when_capture_fails(
    tmp_path: Path,
) -> None:
    viewport = _ResultFakeViewport()
    initial_payload = viewport._result_render_payload
    before_contour = dict(viewport._contour)
    before_display = viewport._display
    viewport.screenshot_error = RuntimeError("capture failed mid-flight")
    harness = _ResultWindowHarness(viewport)

    with pytest.raises(RuntimeError, match="capture failed mid-flight"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            tmp_path / "viewport.png",
            None,
            {
                "result": {
                    "field_ref": _FIELD_REF,
                    "component": "U1",
                    "shape_mode": "deformed",
                    "scale_mode": "custom",
                    "scale_value": 3.0,
                },
            },
        )

    # 中途失败后视口状态必须逐键还原。
    assert viewport._contour == before_contour
    assert viewport._display is before_display
    assert viewport._result_render_payload is initial_payload
    assert viewport._overlay_undeformed is False
    assert viewport._show_edges is True
    assert viewport._render_suppressed is False
    assert viewport.render_calls == 1
    assert not (tmp_path / "viewport.png").exists()


def test_result_transaction_restores_state_when_layer_rebuild_fails(
    tmp_path: Path,
) -> None:
    viewport = _ResultFakeViewport()
    initial_payload = viewport._result_render_payload
    before_contour = dict(viewport._contour)
    viewport.update_layer_error = RuntimeError("layer rebuild failed")
    harness = _ResultWindowHarness(viewport)
    target = tmp_path / "viewport.png"

    with pytest.raises(RuntimeError, match="layer rebuild failed"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"shape_mode": "undeformed"}},
        )

    # _update_result_layer 抛异常后，finally 仍逐键还原视口状态。
    assert viewport._contour == before_contour
    assert viewport._result_render_payload is initial_payload
    assert viewport._render_suppressed is False
    assert viewport.render_calls == 1
    # 失败发生在捕获之前，不落盘。
    assert viewport.screenshot_calls == []
    assert not target.exists()


def test_result_threshold_override_changes_field_request_before_payload(
    tmp_path: Path,
) -> None:
    viewport = _ResultFakeViewport()
    harness = _ResultWindowHarness(viewport)

    FEMMainWindow.export_viewport_image_to(
        harness,
        tmp_path / "viewport.png",
        None,
        {
            "contour": {"averaging_threshold": 60.0},
            "result": {
                "field_ref": _FIELD_REF,
                "component": "Magnitude",
                "shape_mode": "deformed",
                "scale_mode": "custom",
                "scale_value": 2.0,
            },
        },
    )

    # 坑 1：阈值覆盖在构建 payload 之前经
    # _result_averaging_visual_selection 改变场请求本身。
    assert harness.averaging_calls == [
        (harness.result_selection, 60.0),
    ]
    # payload 用的是转换后的场请求，并携带有效形状/变形比例。
    assert harness.payload_build_calls == [
        ((("visual", harness.result_selection)), "deformed", "custom", 2.0),
    ]
    # live 阈值在构建完成后立即还原，不被覆盖污染。
    assert harness._contour_options["averaging_threshold"] == 25.0

    # 未覆盖阈值时沿用 live 值。
    FEMMainWindow.export_viewport_image_to(
        harness,
        tmp_path / "viewport2.png",
        None,
        {"result": {"shape_mode": "undeformed"}},
    )
    assert harness.averaging_calls[-1] == (harness.result_selection, 25.0)
    assert harness.payload_build_calls[-1] == (
        ("visual", harness.result_selection),
        "undeformed",
        "auto",
        1.0,
    )


def test_result_non_ready_field_rejected_without_capture_or_mutation(
    tmp_path: Path,
) -> None:
    viewport = _ResultFakeViewport()
    initial_payload = viewport._result_render_payload
    before_contour = dict(viewport._contour)
    before_display = viewport._display
    harness = _ResultWindowHarness(viewport)
    harness.build_payload_error = KeyError(
        "only a READY catalog field can be rendered"
    )

    # 坑 2：非 READY 场直接拒绝，绝不降级；事务尚未开始。
    with pytest.raises(KeyError):
        FEMMainWindow.export_viewport_image_to(
            harness,
            tmp_path / "viewport.png",
            None,
            {
                "result": {
                    "field_ref": _FIELD_REF,
                    "component": "Magnitude",
                },
            },
        )

    assert viewport.screenshot_calls == []
    assert viewport.installed_payloads == []
    assert viewport.update_layer_calls == 0
    assert viewport.render_calls == 0
    assert viewport._contour == before_contour
    assert viewport._display is before_display
    assert viewport._result_render_payload is initial_payload
    assert not (tmp_path / "viewport.png").exists()


def test_result_group_requires_displayed_result(tmp_path: Path) -> None:
    viewport = _ResultFakeViewport()
    harness = _ResultWindowHarness(viewport)
    harness._provider = None
    with pytest.raises(RuntimeError, match="result overrides require"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            tmp_path / "viewport.png",
            None,
            {"result": {"shape_mode": "undeformed"}},
        )
    assert viewport.screenshot_calls == []


def test_result_group_validates_values(tmp_path: Path) -> None:
    viewport = _ResultFakeViewport()
    harness = _ResultWindowHarness(viewport)
    target = tmp_path / "viewport.png"

    with pytest.raises(ValueError, match="unsupported keys"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"colormap": "jet"}},
        )
    with pytest.raises(ValueError, match="together"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"field_ref": _FIELD_REF}},
        )
    with pytest.raises(ValueError, match="shape_mode"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"shape_mode": "exploded"}},
        )
    with pytest.raises(ValueError, match="scale_mode"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"scale_mode": "manual"}},
        )
    with pytest.raises(ValueError, match="scale_value"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"scale_value": -1.0}},
        )
    with pytest.raises(ValueError, match="overlay_undeformed"):
        FEMMainWindow.export_viewport_image_to(
            harness,
            target,
            None,
            {"result": {"overlay_undeformed": "yes"}},
        )
    # 所有校验都在事务之前，视口未被触碰。
    assert viewport.screenshot_calls == []
    assert viewport.render_calls == 0


def test_result_transaction_keeps_screen_size_and_camera(
    tmp_path: Path,
) -> None:
    """result 组导出后屏幕视口尺寸与相机不变（离屏管线自动保留）。"""

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
    before_display = viewport._display

    class _Harness:
        busy = False
        _contour_options = {"averaging_threshold": 25.0}
        _display = DisplayState("deformed", True)
        _scale_mode = "auto"
        _scale_value = 1.0
        _overlay_undeformed = False
        result_selection = _scalar_selection()

        def _current_result_provider(self):
            return object()

        def _agent_resolve_result_selection(self, field_ref, component):
            return self.result_selection

        def _result_averaging_visual_selection(self, provider, selection):
            return selection

        def _build_result_render_payload(
            self,
            provider,
            selection,
            *,
            shape_mode=None,
            scale_mode=None,
            scale_value=None,
        ):
            # 复用当前 payload（零开销路径），只改形状与叠加。
            return viewport._result_render_payload

        _export_viewport_result_payload = (
            FEMMainWindow._export_viewport_result_payload
        )

    harness = _Harness()
    harness.viewport = viewport
    target = tmp_path / "viewport.png"

    FEMMainWindow.export_viewport_image_to(
        harness,
        target,
        {"quality": 2},
        {
            "result": {
                "shape_mode": "undeformed",
                "overlay_undeformed": True,
            },
        },
    )

    assert target.is_file()
    assert offscreen_calls == [(str(target), (1440, 960), False)]
    assert tuple(plotter.window_size) == (720, 480)
    assert plotter.camera is camera
    assert camera_calls == [("copy", camera_state), ("modified",)]
    # result 组状态逐键还原。
    assert viewport._display is before_display
    assert viewport._overlay_undeformed is False
    assert viewport._render_suppressed is False

    viewport._plotter = None
    viewport.close()
