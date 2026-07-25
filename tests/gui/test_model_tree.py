from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.application import RegionAssignment, SectionDefinition
from fem.core.model import GravityLoad
from fem.elements import BeamOrientation
from fem_gui.widgets.model_tree import ModelTree, ROLE_KEY, ROLE_KIND


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


def test_model_tree_is_compact_and_keeps_real_engineering_objects(gui_inp_path):
    _application()
    model = read(gui_inp_path)
    tree = ModelTree()
    tree.set_model(model)
    items = _items(tree)
    text = [item.text(0) for item in items]
    kinds = [item.data(0, ROLE_KIND) for item in items]

    assert text[0] == model.name
    assert "网格（4 节点，1 单元）" in text
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
    assert line_load.text(0) == "梁均布载荷 1"
    assert not line_load.icon(0).isNull()


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
        "Model-1",
        ("Sketch-1", "Extrude-1", "Cut-1"),
        part_name="Part-1",
    )

    root = tree.topLevelItem(0)
    part = root.child(0)
    assert root.text(0) == "Model-1"
    assert part.text(0) == "Part-1"
    assert [part.child(index).text(0) for index in range(part.childCount())] == [
        "Sketch-1", "Extrude-1", "Cut-1",
    ]
    assert len(_items(tree)) == 5


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
        part_name="Part-1",
    )

    root = tree.topLevelItem(0)
    part = root.child(0)
    mesh = root.child(1)
    assert part.text(0) == "Part-1"
    assert [part.child(index).text(0) for index in range(part.childCount())] == [
        "Sketch-1",
        "Extrude-1",
    ]
    assert mesh.text(0).startswith("网格（")
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
