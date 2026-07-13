from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem_gui.widgets.model_tree import ModelTree, ROLE_KIND


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
    tree.highlightRequested.connect(lambda kind, key: highlighted.append((kind, key)))
    tree.informationRequested.connect(lambda kind, key: informed.append((kind, key)))

    tree._on_clicked(material)
    tree._on_double_clicked(material)

    assert highlighted == [("material", "STEEL")]
    assert informed == [("material", "STEEL")]
    assert tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
