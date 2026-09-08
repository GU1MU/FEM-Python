from __future__ import annotations

from fem.application import ModelSession, UnitContext
from fem.geometry import BoxGeometry, MovedGeometry, MultiBodyGeometry, SolidBody


def make_boolean_part_session(target, tool) -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Boolean",
        UnitContext("mm", "N", "MPa"),
        target,
        part_name="Target Part",
    )
    session.add_native_part(tool, name="Tool Part")
    return session


def build_part_boolean_call(operation: str, *, result_name: str = "Boolean Result"):
    return {
        "part_id": "P1",
        "edit": {
            "operation": "part_boolean",
            "boolean_operation": operation,
            "tool_part_id": "P2",
            "result_name": result_name,
            "tool_handling": "consume_tool_part",
        },
    }


def build_body_boolean_call(operation: str, *, result_name: str = "Body Result"):
    return {
        "part_id": "P1",
        "edit": {
            "operation": "body_boolean",
            "boolean_operation": operation,
            "target_body_id": "B1",
            "tool_body_id": "B2",
            "result_name": result_name,
            "tool_handling": "consume_tool_body",
        },
    }


def make_multi_body_session() -> ModelSession:
    geometry = MultiBodyGeometry(
        "Canonical same-Part Bodies",
        (
            SolidBody("B1", "Target", BoxGeometry("Target", 2.0, 1.0, 1.0)),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
            ),
            SolidBody(
                "B3",
                "Unaffected",
                MovedGeometry(
                    BoxGeometry("Unaffected", 1.0, 1.0, 1.0),
                    5.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent Body Boolean",
        UnitContext("mm", "N", "MPa"),
        geometry,
        part_name="MultiBody Part",
    )
    return session
