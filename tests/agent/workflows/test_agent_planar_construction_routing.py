from __future__ import annotations

from copy import deepcopy
import json

from fem.application import ModelSession
from fem_agent.tools.registry import ToolExecutionContext
import fem_gui.agent_handlers.geometry_creation as agent_authoring_module


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
