from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fem.application import ModelSession
from fem_agent.routing import geometry_route_hint
from fem_agent.tools.registry import ToolExecutionContext
import fem_gui.agent_authoring as agent_authoring_module


from tests.helpers.agent_planar_construction import (
    make_planar_authoring_controller as _controller,
    build_rectangle_arguments as _rectangle_arguments,
)


def test_planar_provider_schema_fits_context_budget() -> None:
    _bridge, controller = _controller(ModelSession())
    definition = next(
        item for item in controller.definitions
        if item.name == "prepare_planar_construction_proposal"
    )
    assert len(json.dumps(definition.parameters).encode("utf-8")) <= 32_768


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
def test_planar_requests_select_construction_tool(
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


def test_profile_transform_selects_extrusion_tool() -> None:
    hint = geometry_route_hint("将当前二维轮廓拉伸 20 mm")

    assert hint is not None and hint.is_transform
    assert hint.required_prepare_tool == "prepare_profile_extrusion"


def test_follow_up_planar_cut_routes_through_geometry_edit_tools() -> None:
    hint = geometry_route_hint("当然，切除出S形状的槽即可")

    assert hint is not None and hint.is_edit
    assert hint.requested_operation == "planar_geometry_edit"
    assert hint.target_part_dimension == 2
    assert hint.required_probe_tool == "read_geometry_edit_context"
    assert hint.required_prepare_tool == "prepare_geometry_edit"


def test_oversized_construction_is_rejected_before_cad(monkeypatch) -> None:
    def forbidden_compile(_construction):
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
        ToolExecutionContext("planar-construction-overbudget", 0, "overbudget"),
    )

    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.budget-exceeded"
