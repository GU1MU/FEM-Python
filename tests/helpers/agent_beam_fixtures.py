from __future__ import annotations

from fem.application import ModelSession
from tests.helpers.agent_line_fixtures import apply_definition_action


BEAM_STEP_NAME = "分析步-梁"


def apply_line_scopes_and_material(controller: object, session: ModelSession) -> None:
    for action, parameters, suffix in (
        (
            "create_named_region",
            {
                "name": "点-固定端",
                "part_id": "P1",
                "logical_ids": ["point:Root"],
                "mesh_kind": "node",
            },
            "root-region",
        ),
        (
            "create_named_region",
            {
                "name": "点-自由端",
                "part_id": "P1",
                "logical_ids": ["point:Tip"],
                "mesh_kind": "node",
            },
            "tip-region",
        ),
        (
            "create_named_region",
            {
                "name": "域-梁",
                "part_id": "P1",
                "logical_ids": ["edge:Bar"],
                "mesh_kind": "element",
            },
            "beam-region",
        ),
        (
            "create_material",
            {
                "name": "材料-钢",
                "properties": {"E": 210000.0, "nu": 0.3},
            },
            "material",
        ),
    ):
        apply_definition_action(controller, session, action, parameters, suffix)


def apply_beam_definitions(
    controller: object,
    session: ModelSession,
) -> None:
    apply_line_scopes_and_material(controller, session)
    for action, parameters, suffix in (
        (
            "create_section",
            {
                "name": "截面-矩形梁",
                "material": "材料-钢",
                "section_type": "rectangle",
                "properties": {"height": 20.0, "width": 10.0},
            },
            "section",
        ),
        (
            "assign_section",
            {
                "section_name": "截面-矩形梁",
                "region_name": "域-梁",
                "local_y_reference": [0.0, 1.0, 0.0],
            },
            "assignment",
        ),
        ("create_static_step", {"name": BEAM_STEP_NAME}, "step"),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": BEAM_STEP_NAME,
                "target_scope": "点-固定端",
                "target_kind": "node_set",
                "first_component": 1,
                "last_component": 6,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            "fixed",
        ),
        (
            "create_load",
            {
                "name": "载荷-轴向力",
                "step_name": BEAM_STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 1,
                "vector": None,
                "magnitude": 1000.0,
                "direction": "global_x",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
            "axial-force",
        ),
        (
            "create_load",
            {
                "name": "载荷-横向力",
                "step_name": BEAM_STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 2,
                "vector": None,
                "magnitude": -100.0,
                "direction": "global_y",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
            "bending-force",
        ),
        (
            "create_load",
            {
                "name": "载荷-扭矩",
                "step_name": BEAM_STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 4,
                "vector": None,
                "magnitude": 500.0,
                "direction": "global_rx",
                "unit": "N*mm",
                "distribution": "concentrated",
                "confirmed": True,
            },
            "torque",
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-节点",
                "step_name": BEAM_STEP_NAME,
                "target": "node",
                "variables": ["U", "RF"],
                "units": ["mm", "N"],
                "confirmed": True,
            },
            "node-output",
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-应力",
                "step_name": BEAM_STEP_NAME,
                "target": "element",
                "variables": ["S"],
                "units": ["MPa"],
                "confirmed": True,
            },
            "element-output",
        ),
    ):
        apply_definition_action(controller, session, action, parameters, suffix)
