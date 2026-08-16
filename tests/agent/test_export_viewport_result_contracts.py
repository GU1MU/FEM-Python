"""Phase 3 contract tests: export_viewport_image 的 result 参数组。"""

from __future__ import annotations

import pytest

from fem_agent.export_authoring import (
    COMPONENT_MAX_LENGTH,
    FIELD_REF_MAX_LENGTH,
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


_DIGEST = "c" * 64


def _viewport_request(**overrides: object) -> ExportViewportImageRequest:
    payload: dict[str, object] = {
        "image": ViewportImageOptions(),
        "display_overrides": {},
        "contour_overrides": {},
        "result_overrides": {},
    }
    payload.update(overrides)
    return ExportViewportImageRequest(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_result_group_schema_is_closed_and_bounded() -> None:
    schema = export_viewport_image_tool_schema()
    result_schema = schema["input_schema"]["properties"]["result"]
    assert result_schema["additionalProperties"] is False
    assert set(result_schema["properties"]) == RESULT_OVERRIDE_KEYS
    assert result_schema["required"] == []
    properties = result_schema["properties"]
    assert properties["field_ref"]["maxLength"] == FIELD_REF_MAX_LENGTH
    assert properties["component"]["maxLength"] == COMPONENT_MAX_LENGTH
    assert properties["scale_value"]["minimum"] == 0.0
    assert properties["overlay_undeformed"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# DTO 边界
# ---------------------------------------------------------------------------


def test_result_overrides_round_trip_through_dict() -> None:
    result_overrides = {
        "field_ref": "U@nodes:c0",
        "component": "Magnitude",
        "shape_mode": "deformed",
        "scale_mode": "custom",
        "scale_value": 2.5,
        "overlay_undeformed": True,
    }
    request = _viewport_request(result_overrides=result_overrides)
    assert request.result_overrides == result_overrides
    assert request.has_overrides is True
    payload = request.to_dict()
    assert payload["result"] == result_overrides
    assert ExportViewportImageRequest.from_dict(payload) == request
    # 各键均可省略；省略 = 沿用当前视口状态。
    partial = _viewport_request(result_overrides={"shape_mode": "undeformed"})
    assert partial.result_overrides == {"shape_mode": "undeformed"}


def test_result_overrides_reject_unknown_and_misplaced_keys() -> None:
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"unknown_key": True})
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"colormap": "jet"})
    with pytest.raises(TypeError):
        _viewport_request(result_overrides={"overlay_undeformed": "yes"})
    with pytest.raises(TypeError):
        _viewport_request(result_overrides={"scale_value": "2.5"})


def test_result_overrides_enforce_enum_and_numeric_bounds() -> None:
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"shape_mode": "exploded"})
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"scale_mode": "manual"})
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"scale_value": -1.0})
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"scale_value": float("inf")})


def test_result_overrides_enforce_field_ref_and_component_bounds() -> None:
    # 边界内放行。
    boundary_ref = "f" * FIELD_REF_MAX_LENGTH
    boundary_component = "c" * COMPONENT_MAX_LENGTH
    request = _viewport_request(
        result_overrides={
            "field_ref": boundary_ref,
            "component": boundary_component,
        }
    )
    assert request.result_overrides["field_ref"] == boundary_ref
    # 越界拒绝。
    with pytest.raises(ExportAuthoringError):
        _viewport_request(
            result_overrides={"field_ref": "f" * (FIELD_REF_MAX_LENGTH + 1)}
        )
    with pytest.raises(ExportAuthoringError):
        _viewport_request(
            result_overrides={
                "component": "c" * (COMPONENT_MAX_LENGTH + 1),
            }
        )
    with pytest.raises(ExportAuthoringError):
        _viewport_request(result_overrides={"field_ref": "  "})
    with pytest.raises(TypeError):
        _viewport_request(result_overrides={"field_ref": 12})


# ---------------------------------------------------------------------------
# Fake 端口与桥接
# ---------------------------------------------------------------------------


def test_fake_port_and_bridge_forward_result_group() -> None:
    port = FakeAgentExportPort()
    request = _viewport_request(
        result_overrides={"field_ref": "U@nodes:c0", "component": "U1"}
    )
    unconfigured = port.export_viewport_image(request)
    assert unconfigured.ok is False
    assert port.viewport_image_calls == [request]

    receipt = ExportFileReceipt(
        workspace_relative_path="agent_exports/viewport_1.png",
        filename="viewport_1.png",
        sha256=_DIGEST,
        size_bytes=128,
        kind="png",
    )
    port.register_viewport_image(ExportViewportImageResponse.success(receipt))
    success = port.export_viewport_image(request)
    assert success.ok is True

    bridge = AgentExportBridge(port)
    bridged = bridge.viewport_image(
        {"result": {"field_ref": "U@nodes:c0", "component": "U1"}}
    )
    assert bridged.ok is True
    assert port.viewport_image_calls[-1].result_overrides == {
        "field_ref": "U@nodes:c0",
        "component": "U1",
    }
