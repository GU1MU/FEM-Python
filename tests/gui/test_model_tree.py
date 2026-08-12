from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

import fem_gui.widgets.model_tree as model_tree_module
from fem.io.inp import read
from fem.application import (
    RegionAssignment,
    SectionDefinition,
    describe_model_capabilities,
)
from fem.application.results import project_output_requests
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    GravityLoad,
    OutputRequest,
)
from fem.elements import BeamOrientation
from fem_gui.widgets.model_tree import (
    ModelTree,
    ROLE_INHERITED,
    ROLE_KEY,
    ROLE_KIND,
)
from tests.helpers.model_builders import make_two_step_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _items(tree: ModelTree):
    root = tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    values = []
    while stack:
        item = stack.pop()
        values.append(item)
        stack.extend(item.child(index) for index in range(item.childCount()))
    return values


def test_model_tree_uses_agent_chat_scrollbar_style():
    _application()
    tree = ModelTree()

    stylesheet = tree.styleSheet()

    assert "QTreeWidget#modelTree QScrollBar:vertical" in stylesheet
    assert "background: transparent" in stylesheet
    assert "border-radius: 4px" in stylesheet
    assert "agent_chat_scroll_up.svg" in stylesheet
    assert "agent_chat_scroll_down.svg" in stylesheet


def test_model_tree_is_compact_and_keeps_real_engineering_objects(gui_inp_path):
    _application()
    model = read(gui_inp_path)
    tree = ModelTree()
    tree.set_model(model)
    items = _items(tree)
    text = [item.text(0) for item in items]
    kinds = [item.data(0, ROLE_KIND) for item in items]

    assert text[0] == model.name
    assert "网格" in text
    assert "节点集 (2)" in text
    assert "单元集 (1)" in text
    assert "材料 (1)" in text
    assert "截面 (1)" in text
    assert "分析 (2)" in text
    assert "node" not in kinds
    assert "element" not in kinds
    assert not any(value.startswith(("节点 1", "单元 1")) for value in text)
    assert not tree.topLevelItem(0).icon(0).isNull()
    mesh = next(item for item in items if item.data(0, ROLE_KIND) == "mesh")
    assert not mesh.icon(0).isNull()
    categories = [item for item in items if item.data(0, ROLE_KIND) == "category"]
    assert categories
    assert all(not item.icon(0).isNull() for item in categories)
    section = next(
        item for item in items
        if item.data(0, ROLE_KIND) == "section"
    )
    assert section.text(0) == "截面 1（平面应力）"


def test_model_tree_rebuild_preserves_navigation_state(gui_inp_path):
    _application()
    model = read(gui_inp_path)
    tree = ModelTree()
    tree.set_model(model)
    items = _items(tree)
    mesh = next(item for item in items if item.data(0, ROLE_KIND) == "mesh")
    analysis = next(
        item
        for item in items
        if item.data(0, ROLE_KIND) == "category"
        and item.text(0).startswith("分析 (")
    )
    step = next(item for item in items if item.data(0, ROLE_KIND) == "step")
    boundaries = next(
        step.child(index)
        for index in range(step.childCount())
        if step.child(index).text(0).startswith("边界条件 (")
    )
    material = next(
        item for item in items if item.data(0, ROLE_KIND) == "material"
    )
    mesh.setExpanded(False)
    analysis.setExpanded(True)
    step.setExpanded(True)
    boundaries.setExpanded(True)
    tree.setCurrentItem(material)
    step_name = step.text(0)
    material_key = material.data(0, ROLE_KEY)
    model.steps.append(AnalysisStep("Additional"))

    tree.set_model(model)

    refreshed = _items(tree)
    assert not next(
        item for item in refreshed if item.data(0, ROLE_KIND) == "mesh"
    ).isExpanded()
    assert next(
        item
        for item in refreshed
        if item.data(0, ROLE_KIND) == "category"
        and item.text(0).startswith("分析 (")
    ).isExpanded()
    refreshed_step = next(
        item
        for item in refreshed
        if item.data(0, ROLE_KIND) == "step" and item.text(0) == step_name
    )
    assert refreshed_step.isExpanded()
    assert next(
        refreshed_step.child(index)
        for index in range(refreshed_step.childCount())
        if refreshed_step.child(index).text(0).startswith("边界条件 (")
    ).isExpanded()
    assert tree.currentItem().data(0, ROLE_KIND) == "material"
    assert tree.currentItem().data(0, ROLE_KEY) == material_key


def test_node_and_element_selection_safely_selects_mesh_summary(gui_inp_path):
    _application()
    tree = ModelTree()
    tree.set_model(read(gui_inp_path))

    tree.select_entity("node", 4)
    assert tree.currentItem().data(0, ROLE_KIND) == "mesh"
    tree.select_entity("element", 1)
    assert tree.currentItem().data(0, ROLE_KIND) == "mesh"


def test_tree_item_count_does_not_scale_with_node_or_element_count():
    _application()
    model = SimpleNamespace(
        name="大型模型",
        mesh=SimpleNamespace(nodes=[None] * 100_000, elements=[None] * 50_000),
        node_sets={}, element_sets={}, surfaces={}, edges={},
        materials={}, sections=[], steps=[],
    )
    tree = ModelTree()
    tree.set_model(model)

    assert len(_items(tree)) == 8
    assert tree.topLevelItem(0).text(0) == "大型模型"


def test_tree_click_and_double_click_keep_object_signals(gui_inp_path):
    _application()
    tree = ModelTree()
    tree.set_model(read(gui_inp_path))
    material = next(item for item in _items(tree) if item.data(0, ROLE_KIND) == "material")
    highlighted = []
    informed = []
    edited = []
    tree.highlightRequested.connect(lambda kind, key: highlighted.append((kind, key)))
    tree.informationRequested.connect(lambda kind, key: informed.append((kind, key)))
    tree.editRequested.connect(lambda kind, key: edited.append((kind, key)))

    tree._on_clicked(material)
    tree._on_double_clicked(material)

    assert highlighted == [("material", "STEEL")]
    assert informed == []
    assert edited == [("material", "STEEL")]
    assert tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_boundary_and_load_context_menus_emit_delete_request(
    gui_inp_path,
    monkeypatch,
):
    _application()
    tree = ModelTree()
    tree.set_model(read(gui_inp_path))
    boundary = next(
        item
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "boundary"
        and not item.data(0, ROLE_INHERITED)
    )
    load = next(
        item
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "cload"
    )
    selected = [boundary]
    action_labels = []

    class Menu:
        def __init__(self, _parent):
            self.actions = {}

        def addAction(self, label):
            action = object()
            action_labels.append(label)
            self.actions[label] = action
            return action

        def exec(self, _position):
            return self.actions["删除"]

    deleted = []
    tree.deleteRequested.connect(
        lambda kind, key: deleted.append((kind, key))
    )
    monkeypatch.setattr(model_tree_module, "QMenu", Menu)
    monkeypatch.setattr(
        ModelTree,
        "itemAt",
        lambda _tree, _position: selected[0],
    )

    tree._show_context_menu(QPoint())

    assert action_labels == ["高亮", "编辑", "删除", "查看信息"]
    assert deleted == [("boundary", boundary.data(0, ROLE_KEY))]

    selected[0] = load
    action_labels.clear()
    tree._show_context_menu(QPoint())

    assert action_labels == ["高亮", "编辑", "删除", "查看信息"]
    assert deleted[-1] == ("cload", load.data(0, ROLE_KEY))


def test_runnable_steps_show_boundaries_inherited_from_every_previous_step():
    _application()
    model = make_two_step_static_pull_truss_model()
    model.steps[1].boundaries = (
        DisplacementConstraint("TIP", 1, 1, 0.125),
    )
    tree = ModelTree()
    tree.set_model(model)
    step = next(
        item
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "step" and item.text(0) == "pull2"
    )
    boundary_root = next(
        step.child(index)
        for index in range(step.childCount())
        if step.child(index).text(0).startswith("边界条件")
    )

    assert boundary_root.text(0) == "边界条件 (3)"
    inherited = tuple(
        boundary_root.child(index)
        for index in range(boundary_root.childCount())
    )
    assert [item.text(0) for item in inherited] == [
        "位移约束 1",
        "位移约束 2",
        "位移约束 1",
    ]
    assert all(item.data(0, ROLE_INHERITED) is True for item in inherited)
    assert all(
        item.data(0, ROLE_KIND) == "inherited_boundary"
        for item in inherited
    )
    assert [item.data(0, ROLE_KEY) for item in inherited] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert all(not item.toolTip(0) for item in inherited)

    edited = []
    informed = []
    tree.editRequested.connect(lambda kind, key: edited.append((kind, key)))
    tree.informationRequested.connect(
        lambda kind, key: informed.append((kind, key))
    )
    tree._on_double_clicked(inherited[0])

    assert edited == []
    assert informed == [("boundary", (0, 0))]


def test_line_load_is_a_regular_load_tree_item():
    _application()
    step = SimpleNamespace(
        name="Load",
        boundaries=(),
        cloads=(),
        surface_loads=(),
        edge_loads=(),
        line_loads=(SimpleNamespace(),),
        outputs=(),
    )
    model = SimpleNamespace(
        name="Beam model",
        mesh=SimpleNamespace(nodes=[None, None], elements=[None]),
        node_sets={}, element_sets={}, surfaces={}, edges={},
        materials={}, sections=[], steps=[step],
    )
    tree = ModelTree()
    tree.set_model(model)

    line_load = next(
        item for item in _items(tree)
        if item.data(0, ROLE_KIND) == "line_load"
    )
    assert line_load.text(0) == "边力 1"
    assert not line_load.icon(0).isNull()


def test_output_request_tree_items_show_only_variables():
    _application()
    step = AnalysisStep(
        "Load",
        outputs=(
            OutputRequest("field", "preselect", ("PRESELECT",)),
            OutputRequest("field", "node", ("RF", "U")),
            OutputRequest("field", "element", ("S",)),
        ),
    )
    model = SimpleNamespace(
        name="Output model",
        mesh=SimpleNamespace(nodes=[], elements=[]),
        node_sets={},
        element_sets={},
        surfaces={},
        edges={},
        materials={},
        sections=[],
        steps=[step],
    )
    tree = ModelTree()

    tree.set_model(model)

    outputs = [
        item.text(0)
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "output"
    ]
    assert set(outputs) == {
        "PRESELECT",
        "RF",
        "U",
        "S",
    }
    output_category = next(
        item
        for item in _items(tree)
        if item.text(0) == "输出请求 (4)"
    )
    assert output_category.childCount() == 4


def test_model_tree_hides_output_requests_without_executable_projection(
    gui_inp_path,
):
    _application()
    model = read(gui_inp_path)
    step = model.steps[0]
    step.outputs = (
        OutputRequest("history", "node", ("U", "RF")),
        OutputRequest("history", "element", ("S", "MISES")),
    )
    catalog = describe_model_capabilities(model).output_request_catalog
    assert catalog is not None
    projections = project_output_requests(tuple(step.outputs), catalog)
    tree = ModelTree()

    tree.set_model(
        model,
        output_request_projections_by_step={step.name: projections},
    )

    output_category = next(
        item
        for item in _items(tree)
        if item.text(0) == "输出请求 (0)"
    )
    assert output_category.childCount() == 0
    assert not any(
        item.data(0, ROLE_KIND) == "output"
        for item in _items(tree)
    )


def test_section_tree_uses_cae_labels_instead_of_backend_identifiers():
    _application()
    model = SimpleNamespace(
        name="Section labels",
        mesh=SimpleNamespace(nodes=[None], elements=[None]),
        node_sets={},
        element_sets={},
        surfaces={},
        edges={},
        materials={},
        sections=(
            SimpleNamespace(section_type="solid", properties={}),
            SimpleNamespace(section_type="beam", properties={}),
        ),
        steps=[],
    )
    tree = ModelTree()

    tree.set_model(model)

    sections = [
        item.text(0)
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "section"
    ]
    assert "截面 1（三维实体）" in sections
    assert "截面 2（梁截面）" in sections


def test_native_geometry_tree_is_shallow_model_part_feature_history():
    _application()
    tree = ModelTree()

    tree.set_geometry_preview(
        "模型-1",
        ("Sketch-1", "Extrude-1", "Cut-1"),
        part_name="部件-1",
    )

    root = tree.topLevelItem(0)
    part = root.child(0)
    assert root.text(0) == "模型-1"
    assert part.text(0) == "部件-1"
    assert [part.child(index).text(0) for index in range(part.childCount())] == [
        "草图-1", "拉伸-1", "切除-1",
    ]
    assert len(_items(tree)) == 5


def test_native_model_part_and_feature_menus_omit_highlight_and_route_actions(
    monkeypatch,
):
    _application()
    tree = ModelTree()
    tree.set_geometry_preview(
        "模型-1",
        ("Sketch-1",),
        part_name="部件-1",
    )
    root = tree.topLevelItem(0)
    part = root.child(0)
    feature = part.child(0)
    selected = [root]
    requested_action = ["重命名"]
    action_labels: list[str] = []

    class Menu:
        def __init__(self, _parent):
            self.actions = {}

        def addAction(self, label):
            action = object()
            action_labels.append(label)
            self.actions[label] = action
            return action

        def exec(self, _position):
            return self.actions[requested_action[0]]

    renamed = []
    informed = []
    highlighted = []
    tree.renameRequested.connect(
        lambda kind, key: renamed.append((kind, key))
    )
    tree.informationRequested.connect(
        lambda kind, key: informed.append((kind, key))
    )
    tree.highlightRequested.connect(
        lambda kind, key: highlighted.append((kind, key))
    )
    monkeypatch.setattr(model_tree_module, "QMenu", Menu)
    monkeypatch.setattr(
        ModelTree,
        "itemAt",
        lambda _tree, _position: selected[0],
    )

    tree._show_context_menu(QPoint())
    assert action_labels == ["重命名", "查看信息"]
    assert renamed == [("model", None)]

    selected[0] = part
    action_labels.clear()
    tree._show_context_menu(QPoint())
    assert action_labels == ["重命名", "查看信息"]
    assert renamed[-1] == ("part", None)

    selected[0] = feature
    requested_action[0] = "查看信息"
    action_labels.clear()
    tree._show_context_menu(QPoint())
    tree._on_clicked(feature)
    assert action_labels == ["查看信息"]
    assert informed == [("feature", "Sketch-1")]
    assert highlighted == []


def test_gravity_is_a_regular_load_tree_item(gui_inp_path):
    _application()
    model = read(gui_inp_path)
    step_index = next(
        index
        for index, step in enumerate(model.steps)
        if step.name == "Static-1"
    )
    model.steps[step_index].gravity_loads = (
        GravityLoad((0.0, -9.81)),
    )
    tree = ModelTree()
    tree.set_model(model)

    gravity = next(
        item
        for item in _items(tree)
        if item.data(0, ROLE_KIND) == "gravity_load"
    )
    assert gravity.text(0) == "重力 1"
    assert not gravity.icon(0).isNull()


def test_native_meshed_tree_keeps_the_part_feature_history(gui_inp_path):
    _application()
    tree = ModelTree()
    tree.set_model(
        read(gui_inp_path),
        feature_rows=("Sketch-1", "Extrude-1"),
        part_name="部件-1",
    )

    root = tree.topLevelItem(0)
    part = root.child(0)
    mesh = root.child(1)
    assert part.text(0) == "部件-1"
    assert [part.child(index).text(0) for index in range(part.childCount())] == [
        "草图-1",
        "拉伸-1",
    ]
    assert mesh.text(0) == "网格"
    assert not mesh.isExpanded()


def test_assignment_nodes_show_orientation_and_route_edit_by_index():
    _application()
    model = SimpleNamespace(
        name="Beam assignments",
        mesh=SimpleNamespace(nodes=[None, None], elements=[None]),
        node_sets={},
        element_sets={},
        surfaces={},
        edges={},
        materials={},
        sections=[],
        steps=[],
    )
    sections = (
        SectionDefinition("Beam-A", "Steel", "rectangle", {}),
        SectionDefinition("Beam-B", "Steel", "solid_circle", {}),
    )
    assignments = (
        RegionAssignment(
            "Beam-A",
            "SET-A",
            BeamOrientation((0.0, 1.0, 0.0)),
        ),
        RegionAssignment("Beam-B", "SET-B"),
    )
    tree = ModelTree()
    tree.set_model(
        model,
        section_definitions=sections,
        region_assignments=assignments,
    )

    items = _items(tree)
    assignment_items = sorted(
        (
            item
            for item in items
            if item.data(0, ROLE_KIND) == "assignment"
        ),
        key=lambda item: item.data(0, ROLE_KEY),
    )
    assignment_category = next(
        item
        for item in items
        if item.data(0, ROLE_KIND) == "category"
        and item.text(0) == "截面分配 (2)"
    )
    section_assign_icon_key = model_tree_module.icon("section_assign").cacheKey()
    assert assignment_category.icon(0).cacheKey() == section_assign_icon_key
    assert all(
        item.icon(0).cacheKey() == section_assign_icon_key
        for item in assignment_items
    )
    assert [item.text(0) for item in assignment_items] == [
        "Beam-A → SET-A",
        "Beam-B → SET-B",
    ]
    assert [
        item.child(0).text(0)
        for item in assignment_items
    ] == [
        "orientation: explicit",
        "orientation: automatic",
    ]

    edited = []
    tree.editRequested.connect(
        lambda kind, key: edited.append((kind, key))
    )
    tree._on_double_clicked(assignment_items[1])

    assert edited == [("assignment", 1)]
