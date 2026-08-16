"""Contract tests for the export_viewport_image DTOs and schemas."""

from __future__ import annotations

import pytest

from fem_agent.export_authoring import (
    CONTOUR_SETTING_KEYS,
    DEFAULT_VIEWPORT_IMAGE_FORMAT,
    DEFAULT_VIEWPORT_IMAGE_QUALITY,
    DISPLAY_SETTING_KEYS,
    EXPORT_AUTHORING_SCHEMA_VERSION,
    EXPORT_VIEWPORT_IMAGE_TOOL_NAME,
    NO_WORKSPACE_DIAGNOSTIC_CODE,
    NO_WORKSPACE_DIAGNOSTIC_MESSAGE,
    RESULT_OVERRIDE_KEYS,
    AgentExportBridge,
    ExportAuthoringError,
    ExportFileReceipt,
    ExportViewportImageRequest,
    ExportViewportImageResponse,
    FakeAgentExportPort,
    ViewportImageOptions,
    export_viewport_image_tool_schema,
)


_DIGEST = "b" * 64


def _image_receipt(kind: str = "png", filename: str = "viewport_1.png"):
    return ExportFileReceipt(
        workspace_relative_path=f"agent_exports/{filename}",
        filename=filename,
        sha256=_DIGEST,
        size_bytes=4096,
        kind=kind,
    )


def _viewport_request(**overrides: object) -> ExportViewportImageRequest:
    payload: dict[str, object] = {
        "image": ViewportImageOptions(),
        "display_overrides": {},
        "contour_overrides": {},
    }
    payload.update(overrides)
    return ExportViewportImageRequest(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_viewport_image_schema_is_closed_with_result_group() -> None:
    schema = export_viewport_image_tool_schema()
    assert schema["name"] == EXPORT_VIEWPORT_IMAGE_TOOL_NAME
    input_schema = schema["input_schema"]
    assert input_schema["additionalProperties"] is False
    assert input_schema["required"] == []
    assert set(input_schema["properties"]) == {
        "schema_version",
        "image",
        "display",
        "contour",
        "result",
    }
    image_schema = input_schema["properties"]["image"]
    assert image_schema["additionalProperties"] is False
    assert set(image_schema["properties"]) == {
        "format",
        "quality",
        "transparent_background",
    }
    assert "width" not in image_schema["properties"]
    assert "height" not in image_schema["properties"]
    display_schema = input_schema["properties"]["display"]
    assert display_schema["additionalProperties"] is False
    assert set(display_schema["properties"]) == DISPLAY_SETTING_KEYS
    contour_schema = input_schema["properties"]["contour"]
    assert contour_schema["additionalProperties"] is False
    assert set(contour_schema["properties"]) == CONTOUR_SETTING_KEYS
    result_schema = input_schema["properties"]["result"]
    assert result_schema["additionalProperties"] is False
    assert set(result_schema["properties"]) == RESULT_OVERRIDE_KEYS
    assert result_schema["required"] == []
    assert result_schema["properties"]["shape_mode"]["enum"] == [
        "deformed",
        "undeformed",
    ]
    assert result_schema["properties"]["scale_mode"]["enum"] == [
        "auto",
        "real",
        "custom",
    ]
    assert "do not retry" in schema["description"]


# ---------------------------------------------------------------------------
# DTO bounds
# ---------------------------------------------------------------------------


def test_image_options_defaults_and_bounds() -> None:
    options = ViewportImageOptions()
    assert options.format == DEFAULT_VIEWPORT_IMAGE_FORMAT == "png"
    assert options.quality == DEFAULT_VIEWPORT_IMAGE_QUALITY == 1
    assert options.transparent_background is False
    assert options.to_dict() == {
        "format": "png",
        "quality": 1,
        "transparent_background": False,
    }
    assert ViewportImageOptions.from_dict(options.to_dict()) == options
    assert ViewportImageOptions.from_dict({}) == options
    assert ViewportImageOptions.from_dict({"format": "jpeg", "quality": 4}) == (
        ViewportImageOptions(format="jpeg", quality=4)
    )
    with pytest.raises(ExportAuthoringError):
        ViewportImageOptions(format="bmp")
    with pytest.raises(ExportAuthoringError):
        ViewportImageOptions(quality=3)
    with pytest.raises(ExportAuthoringError):
        ViewportImageOptions.from_dict({"width": 1024})
    with pytest.raises(TypeError):
        ViewportImageOptions(transparent_background="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ViewportImageOptions.from_dict({"quality": "2"})


def test_viewport_request_from_dict_is_closed_and_parses_result_group() -> None:
    request = _viewport_request(
        display_overrides={"legend": False, "decimals": 2},
        contour_overrides={"colormap": "jet", "levels": 16},
    )
    payload = request.to_dict()
    assert payload["schema_version"] == EXPORT_AUTHORING_SCHEMA_VERSION
    assert payload["display"] == {"legend": False, "decimals": 2}
    assert payload["contour"] == {"colormap": "jet", "levels": 16}
    assert payload["result"] == {}
    assert request.has_overrides is True
    assert ExportViewportImageRequest.from_dict(payload) == request
    # 省略所有参数组 = 沿用当前视口状态。
    bare = ExportViewportImageRequest.from_dict({})
    assert bare.image == ViewportImageOptions()
    assert bare.display_overrides == {}
    assert bare.contour_overrides == {}
    assert bare.result_overrides == {}
    assert bare.has_overrides is False
    # Phase 3 的 result 组被封闭解析接受并往返。
    with_result = ExportViewportImageRequest.from_dict(
        {**payload, "result": {"field_ref": "U@nodes:c0"}}
    )
    assert with_result.result_overrides == {"field_ref": "U@nodes:c0"}
    assert with_result.has_overrides is True
    with pytest.raises(ExportAuthoringError):
        ExportViewportImageRequest.from_dict({**payload, "extra": 1})
    with pytest.raises(ExportAuthoringError):
        ExportViewportImageRequest.from_dict(
            {**payload, "schema_version": "9.9"}
        )
    # 覆盖组键集与对话框设置严格对齐，未知键被拒绝。
    with pytest.raises(ExportAuthoringError):
        _viewport_request(display_overrides={"unknown_key": True})
    with pytest.raises(ExportAuthoringError):
        _viewport_request(contour_overrides={"field_ref": "U@nodes:c0"})
    with pytest.raises(TypeError):
        _viewport_request(contour_overrides={"levels": [12]})  # type: ignore[dict-item]


def test_viewport_response_bounds_and_no_workspace_diagnostic() -> None:
    success = ExportViewportImageResponse.success(_image_receipt())
    assert success.ok is True
    assert success.to_dict()["export_receipt"]["kind"] == "png"
    jpeg = ExportViewportImageResponse.success(
        _image_receipt(kind="jpeg", filename="viewport_1.jpg")
    )
    assert jpeg.to_dict()["export_receipt"]["kind"] == "jpeg"
    with pytest.raises(ExportAuthoringError):
        ExportViewportImageResponse()

    response = ExportViewportImageResponse.no_workspace()
    assert response.ok is False
    payload = response.to_dict()
    assert payload["export_receipt"] is None
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == NO_WORKSPACE_DIAGNOSTIC_CODE
    assert diagnostic["message"] == NO_WORKSPACE_DIAGNOSTIC_MESSAGE
    assert diagnostic["retryable"] is False
    assert diagnostic["clarification_required"] is True


# ---------------------------------------------------------------------------
# Fake port + bridge
# ---------------------------------------------------------------------------


def test_fake_viewport_port_records_calls_and_honors_registration() -> None:
    port = FakeAgentExportPort()
    request = _viewport_request()
    unconfigured = port.export_viewport_image(request)
    assert unconfigured.ok is False
    assert unconfigured.diagnostics[0].code == "export.not_configured"
    assert port.viewport_image_calls == [request]

    success = ExportViewportImageResponse.success(_image_receipt())
    port.register_viewport_image(success)
    assert port.export_viewport_image(request) is success
    assert len(port.viewport_image_calls) == 2
    with pytest.raises(TypeError):
        port.export_viewport_image({"not": "a request"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        port.register_viewport_image(object())  # type: ignore[arg-type]


def test_bridge_normalizes_viewport_requests_and_enforces_types() -> None:
    port = FakeAgentExportPort()
    bridge = AgentExportBridge(port)
    response = bridge.viewport_image(
        {"image": {"format": "jpeg", "quality": 2}}
    )
    assert response.ok is False
    normalized = port.viewport_image_calls[0]
    assert normalized.image.format == "jpeg"
    assert normalized.image.quality == 2
    # result 组在归一化阶段被接受并保留在 DTO 中。
    response = bridge.viewport_image(
        {"result": {"field_ref": "U@nodes:c0", "component": "U1"}}
    )
    assert response.ok is False
    assert port.viewport_image_calls[-1].result_overrides == {
        "field_ref": "U@nodes:c0",
        "component": "U1",
    }

    class _WrongReturnPort:
        def export_accepted_result_csv(self, request):
            return object()

        def export_viewport_image(self, request):
            return object()

        def read_result_display_context(self):
            return object()

    wrong = AgentExportBridge(_WrongReturnPort())
    with pytest.raises(TypeError):
        wrong.viewport_image(_viewport_request())
