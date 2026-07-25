from __future__ import annotations

import pytest

from fem.abaqus import read
from fem.application import (
    BeamOrientation,
    ModelDefinitions,
    RegionAssignment,
    SectionDefinition,
    resolve_effective_beam_frames,
)
from fem.solvers.static_linear import solve
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    ElementSet,
    FEMModel,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    SectionAssignment,
)
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def _page(inspection, title):
    return next(page for page in inspection.pages if page.title == title)


def _fields(page):
    return dict(page.fields)


def test_service_builds_node_element_and_assignment_indexes_once(gui_inp_path):
    model = read(gui_inp_path)
    service = InspectionService(model)

    assert not hasattr(service, "_element_records")
    assert service._element_record_cached.cache_info().currsize == 0

    assert service.node_sets_by_node[2] == ["RIGHT"]
    assert service.adjacent_elements[2] == [1]
    node = service.inspect("node", 2)
    basic = _page(node, "基本信息")
    assert _fields(basic)["坐标"] == "1, 0"
    assert _fields(basic)["所属节点集"] == "RIGHT"
    assert basic.tables[0].rows == (("1", "Quad4"),)
    assert _page(node, "分析定义").tables[0].rows[0][:3] == ("Static-1", "节点载荷", "U1")
    assert all(page.title != "结果" for page in node.pages)

    record = service.element_record(1)
    assert service._element_record_cached.cache_info().currsize == 1
    assert record["abaqus_type"] == "CPS4"
    assert record["material"] == "STEEL"
    assert record["section_index"] == 0
    assert record["properties"]["E"] == 210000.0
    assert record["properties"]["nu"] == 0.3
    assert record["properties"]["plane_type"] == "stress"


def test_collection_material_section_and_step_information_is_structured(gui_inp_path):
    service = InspectionService(read(gui_inp_path))

    node_set = service.inspect("node_set", "LEFT")
    assert _fields(node_set.pages[0])["节点数量"] == "2"
    assert len(node_set.pages[0].tables[0].rows) == 2
    assert "边界条件" in _fields(node_set.pages[0])["边界条件引用"]

    element_set = service.inspect("element_set", "SOLID")
    assert _fields(element_set.pages[0])["使用的材料"] == "STEEL"
    assert element_set.pages[0].tables[0].rows[0][:3] == ("1", "Quad4", "STEEL")

    material = _fields(service.inspect("material", "STEEL").pages[0])
    assert material["弹性模量 E"] == "210000"
    assert material["泊松比 ν"] == "0.3"
    assert material["作用单元数量"] == "1"
    section = _fields(service.inspect("section", 0).pages[0])
    assert section["材料"] == "STEEL"
    assert section["平面类型"] == "平面应力"
    assert section["厚度"] == "1"

    step_index = next(index for index, step in enumerate(service.model.steps) if step.name == "Static-1")
    step = service.inspect("step", step_index)
    assert [page.title for page in step.pages] == ["概况", "载荷", "输出请求"]
    assert _page(step, "载荷").tables[0].rows[0][1:] == ("节点载荷", "RIGHT", "U1", "10")
    assert _page(step, "输出请求").tables[0].rows[0][1:] == ("场输出", "节点", "U, RF")


def test_result_pages_use_existing_result_data_and_hide_missing_3d_components(gui_inp_path):
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    result = solve(model)
    data = build_result_data(result, geometry)
    service = InspectionService(model, data)

    node_fields = _fields(_page(service.inspect("node", 2), "结果"))
    assert "U1" in node_fields and "U2" in node_fields and "位移模" in node_fields
    assert "RF1" in node_fields and "RF2" in node_fields and "反力模" in node_fields
    assert "U3" not in node_fields and "RF3" not in node_fields
    assert "Mises" in node_fields

    element_result = _page(service.inspect("element", 1), "结果")
    assert _fields(element_result)["结果位置"] == "单元质心"
    values = dict(element_result.tables[0].rows)
    assert "Mises" in values
    assert "最大主应力" in values
    assert "最小主应力" in values


def test_beam_section_and_line_load_use_the_common_inspection_service():
    mesh = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        [Element3D(10, [1, 2], "Beam2", {})],
        dofs_per_node=6,
    )
    model = FEMModel(
        mesh,
        element_sets={"BEAMS": ElementSet("BEAMS", (10,))},
        materials={"STEEL": MaterialDefinition("STEEL", {"E": 210000.0})},
        sections=[SectionAssignment(
            "BEAMS", "STEEL", "beam",
            {"section_type": "rectangle", "height": 0.1, "width": 0.02},
        )],
        steps=[AnalysisStep(
            "Load",
            line_loads=(LineLoad("BEAMS", (0.0, -5.0, 0.0), "local"),),
        )],
    )
    service = InspectionService(model)

    element_properties = dict(
        service.inspect("element", 10).pages[1].tables[1].rows
    )
    assert element_properties["截面类型"] == "rectangle"
    assert element_properties["矩形高度（局部 y）"] == "0.1"
    assert element_properties["矩形宽度（局部 z）"] == "0.02"
    load_fields = _fields(service.inspect("line_load", (0, 0)).pages[0])
    assert (
        load_fields["坐标系"]
        == "局部（Beam 已解析局部坐标）"
    )
    assert load_fields["载荷向量"] == "0, -5, 0"
    assert service.selection_for("line_load", (0, 0)).element_ids == (10,)


@pytest.mark.parametrize(
    ("orientation", "source", "reference"),
    [
        (BeamOrientation((0.0, 1.0, 0.0)), "explicit", "0, 1, 0"),
        (None, "automatic", "—"),
    ],
)
def test_assignment_and_element_inspection_use_effective_frame_query(
    orientation,
    source,
    reference,
):
    properties = {
        "height": 0.1,
        "width": 0.02,
    }
    if orientation is not None:
        properties["beam_local_y_reference"] = (
            orientation.local_y_reference
        )
    mesh = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        [Element3D(10, [1, 2], "Beam2", {})],
        dofs_per_node=6,
    )
    model = FEMModel(
        mesh,
        element_sets={"BEAMS": ElementSet("BEAMS", (10,))},
        materials={
            "STEEL": MaterialDefinition(
                "STEEL",
                {"E": 210000.0, "nu": 0.3},
            )
        },
        sections=[
            SectionAssignment(
                "BEAMS",
                "STEEL",
                "rectangle",
                properties,
            )
        ],
    )
    definitions = ModelDefinitions(
        materials=tuple(model.materials.values()),
        sections=(
            SectionDefinition(
                "Beam Section",
                "STEEL",
                "rectangle",
                {"height": 0.1, "width": 0.02},
            ),
        ),
        assignments=(
            RegionAssignment(
                "Beam Section",
                "BEAMS",
                orientation,
            ),
        ),
    )
    queried = []

    def query(target):
        queried.append(target)
        return resolve_effective_beam_frames(model, target)

    service = InspectionService(
        model,
        definitions=definitions,
        effective_frame_query=query,
    )

    assignment = _fields(
        service.inspect("assignment", 0).pages[0]
    )
    assert assignment["orientation source"] == source
    assert assignment["authored reference"] == reference
    assert assignment["effective frame source"] == source
    assert assignment["有效元素数量"] == "1"
    assert assignment["无效元素数量"] == "0"
    assert assignment["validity"] == "valid"
    assert assignment["矩形高度（local y）"] == "0.1"
    assert assignment["矩形宽度（local z）"] == "0.02"
    assert service.selection_for("assignment", 0).element_ids == (10,)

    frame = _fields(
        _page(service.inspect("element", 10), "Beam 局部坐标")
    )
    assert frame["frame source"] == source
    assert frame["effective properties 来源"] == "截面分配 1"
    assert frame["local x"] == "1, 0, 0"
    assert frame["local y"] == "0, 1, 0"
    assert frame["local z"] == "0, 0, 1"

    service.inspect("element", 10)
    assert len(queried) == 2


def test_non_beam_assignment_marks_orientation_not_applicable():
    mesh = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 0.0, 0.0)],
        [Element3D(10, [1, 2], "Truss2", {})],
        dofs_per_node=3,
    )
    model = FEMModel(
        mesh,
        element_sets={"TRUSSES": ElementSet("TRUSSES", (10,))},
        materials={
            "STEEL": MaterialDefinition("STEEL", {"E": 210000.0})
        },
        sections=[
            SectionAssignment(
                "TRUSSES",
                "STEEL",
                "truss",
                {"area": 0.1},
            )
        ],
    )
    definitions = ModelDefinitions(
        materials=tuple(model.materials.values()),
        sections=(
            SectionDefinition(
                "Truss Section",
                "STEEL",
                "truss",
                {"area": 0.1},
            ),
        ),
        assignments=(
            RegionAssignment("Truss Section", "TRUSSES"),
        ),
    )

    service = InspectionService(model, definitions=definitions)
    assignment = _fields(service.inspect("assignment", 0).pages[0])

    assert assignment["orientation source"] == "not applicable"
    assert assignment["effective frame source"] == "not applicable"
    assert assignment["有效元素数量"] == "0"
    assert assignment["无效元素数量"] == "0"
    assert assignment["validity"] == "not applicable"
    assert "diagnostics" not in assignment


def test_global_gravity_uses_the_common_inspection_and_selection(gui_inp_path):
    model = read(gui_inp_path)
    step_index = next(
        index
        for index, step in enumerate(model.steps)
        if step.name == "Static-1"
    )
    model.steps[step_index].gravity_loads = (
        GravityLoad((0.0, -9.81)),
    )
    service = InspectionService(model)

    inspection = service.inspect("gravity_load", (step_index, 0))

    assert inspection.title == "重力"
    assert _fields(inspection.pages[0])["目标"] == "整个模型"
    assert service.selection_for(
        "gravity_load",
        (step_index, 0),
    ).element_ids == tuple(sorted(service.elements))
