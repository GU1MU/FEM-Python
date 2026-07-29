from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from fem.abaqus import read
from fem.application import (
    BeamOrientation,
    ModelDefinitions,
    RegionAssignment,
    SectionDefinition,
    resolve_effective_beam_frames,
)
from fem.application.results import (
    ElementResultInspectionRequest,
    FieldPosition,
    FieldRequest,
    FieldState,
    NodeResultInspectionRequest,
    ResultCatalog,
    ResultDiagnostic,
    ResultFieldId,
    ResultProvider,
    ResultSourceKey,
    ResultVariable,
    advance_materialization,
    build_result_provider,
    restore_result_provider,
)
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
from fem.post.averaging import NodalAveragingPolicy
from fem_gui.inspection_service import InspectionService
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _page(inspection, title):
    return next(page for page in inspection.pages if page.title == title)


def _fields(page):
    return dict(page.fields)


def _provider_source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="inspection-result",
        session_id="inspection-session",
        artifact_id="inspection-artifact",
        model_revision=3,
        step_name="Step-1",
        run_id="inspection-run",
    )


def _provider_with_all_continuum_fields():
    result = make_continuum_nodal_semantics_result()
    provider = build_result_provider(_provider_source(), result)
    keys = tuple(
        provider.resolve_request(
            FieldRequest(
                ResultFieldId(ResultVariable.S, position),
                averaging_policy=(
                    NodalAveragingPolicy()
                    if position is FieldPosition.RESOLVED_NODAL
                    else None
                ),
            )
        )
        for position in (
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
        )
    )
    provider = restore_result_provider(
        result,
        advance_materialization(
            provider.snapshot,
            provider.materialize(keys),
        ),
    )
    return result, provider


def _provider_with_unavailable_field():
    result = make_continuum_nodal_semantics_result()
    provider = build_result_provider(_provider_source(), result)
    catalog = provider.catalog()
    target = next(
        availability
        for availability in catalog.fields
        if (
            availability.state is FieldState.LAZY
            and availability.descriptor.field_id.position
            is FieldPosition.CENTROID
        )
    )
    unavailable = replace(
        target,
        state=FieldState.UNAVAILABLE,
        diagnostics=(
            ResultDiagnostic(
                code="result.field.unavailable",
                severity="warning",
                message="Field is unavailable.",
                path=("inspection",),
                remediation="Choose another field.",
                details={"position": "centroid"},
            ),
        ),
    )
    fields = tuple(
        unavailable if availability is target else availability
        for availability in catalog.fields
    )
    provider = replace(
        provider,
        _catalog=ResultCatalog(
            source=catalog.source,
            fields=fields,
            default_selection=catalog.default_selection,
            diagnostics=catalog.diagnostics,
        ),
    )
    return result, provider


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
    assert _page(node, "分析定义").tables[0].rows[0][:3] == ("Static-1", "节点力", "U1")
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
    assert _page(step, "载荷").tables[0].rows[0][1:] == ("节点力", "RIGHT", "U1", "10")
    assert _page(step, "输出请求").tables[0].rows[0][1:] == ("场输出", "节点", "U, RF")


def test_typed_provider_drives_node_and_element_result_pages_in_catalog_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, provider = _provider_with_all_continuum_fields()
    observed = []
    original = ResultProvider.inspect_result

    def inspect_result(self, request):
        observed.append(request)
        return original(self, request)

    monkeypatch.setattr(ResultProvider, "inspect_result", inspect_result)

    service = InspectionService(
        result.model,
        result_provider=provider,
    )
    node_page = _page(service.inspect("node", 1), "结果")
    element_page = _page(service.inspect("element", 2), "结果")

    assert tuple(type(request) for request in observed) == (
        NodeResultInspectionRequest,
        ElementResultInspectionRequest,
    )
    assert tuple(table.title for table in node_page.tables) == (
        "位移 U（就绪）",
        "反力 RF（就绪）",
        "应力 S（节点）（就绪）",
    )
    assert tuple(table.title for table in element_page.tables) == (
        "应力 S（节点）（就绪）",
    )
    assert all(
        table.columns
        == (
            "状态",
            "分量",
            "数值",
            "节点",
            "单元",
            "积分点",
            "局部节点",
            "结果区域",
            "平均",
            "诊断",
        )
        for table in (*node_page.tables, *element_page.tables)
    )


def test_typed_result_rows_preserve_all_location_provenance() -> None:
    result, provider = _provider_with_all_continuum_fields()
    service = InspectionService(result.model, result_provider=provider)
    page = _page(service.inspect("node", 1), "结果")
    by_title = {table.title: table for table in page.tables}

    element_node = tuple(
        row
        for row in by_title["应力 S（节点）（就绪）"].rows
        if row[1] == "S11"
    )

    assert len(element_node) == 3
    assert tuple(row[4] for row in element_node) == ("1", "2", "3")
    assert all(row[6] != "—" for row in element_node)


def test_typed_provider_update_is_exact_and_clearable() -> None:
    result, provider = _provider_with_all_continuum_fields()
    service = InspectionService(result.model)

    with pytest.raises(TypeError, match="positional"):
        InspectionService(result.model, object())
    with pytest.raises(TypeError, match="exactly ResultProvider"):
        InspectionService(result.model, result_provider=object())
    with pytest.raises(TypeError, match="exactly ResultProvider"):
        service.update_result_provider(object())

    service.update_result_provider(provider)
    assert service.result_provider is provider
    assert _page(service.inspect("node", 1), "结果").tables

    service.update_result_provider(None)
    assert service.result_provider is None
    assert all(
        page.title != "结果"
        for page in service.inspect("node", 1).pages
    )


def test_lazy_and_unavailable_provider_fields_do_not_materialize_or_block_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, provider = _provider_with_unavailable_field()
    calls = []

    def materialize(self, keys, *, cancellation=None):
        del self, cancellation
        calls.append(tuple(keys))
        raise AssertionError("inspection must not materialize fields")

    monkeypatch.setattr(ResultProvider, "materialize", materialize)
    service = InspectionService(result.model, result_provider=provider)

    assert service.inspect("model", None).pages[0].title == "概况"
    node_page = _page(service.inspect("node", 1), "结果")
    element_page = _page(service.inspect("element", 1), "结果")
    titles = tuple(
        table.title
        for table in (*node_page.tables, *element_page.tables)
    )

    assert any(title.endswith("（按需加载）") for title in titles)
    assert all("单元质心" not in title for title in titles)
    assert calls == []


def test_typed_inspection_path_has_no_legacy_or_support_order_dependency() -> None:
    module_path = Path(
        InspectionService.__init__.__code__.co_filename
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    typed_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {
            "_provider_result_page",
            "_provider_result_table",
        }
    }
    typed_source = "\n".join(
        ast.unparse(typed_functions[name])
        for name in sorted(typed_functions)
    )

    assert not any(
        module.endswith("visualization.result_adapter")
        or module.endswith("widgets.result_tree")
        for module in imported_modules
    )
    assert set(typed_functions) == {
        "_provider_result_page",
        "_provider_result_table",
    }
    assert ".inspect_result(" in typed_source
    for forbidden in (
        ".materialize(",
        "result_data",
        "nodal_values",
        "nodal_stress",
        "element_stress",
        "field_family",
        "sorted(",
    ):
        assert forbidden not in typed_source

    assert not hasattr(InspectionService, "update_result_data")
    legacy_names = {
        "ResultData",
        "result_data",
        "update_result_data",
        "_node_result_fields",
        "nodal_stress",
        "element_stress",
    }
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert legacy_names.isdisjoint(names | attributes | functions)
    assert all(name not in source for name in legacy_names)


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
