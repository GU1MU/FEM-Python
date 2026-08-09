from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import RegionRef
from fem.core.mesh import Mesh2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    FEMModel,
    OutputRequest,
)
from fem_gui.analysis_definition_dialogs import AnalysisDefinitionManagerDialog
from fem_gui.widgets.model_tree import ModelTree, ROLE_KEY, ROLE_KIND


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _step() -> AnalysisStep:
    return AnalysisStep(
        "分析步-静力",
        boundaries=(
            DisplacementConstraint(
                "边-固定端",
                1,
                2,
                target_kind="edge",
                name="位移-固定端",
            ),
        ),
        edge_loads=(
            EdgeLoad(
                "边-加载端",
                (10.0, 0.0),
                name="载荷-拉伸",
            ),
        ),
        outputs=(
            OutputRequest(
                "field",
                "node",
                ("U",),
                name="结果请求-位移",
            ),
        ),
        metadata={"nlgeom": False},
    )


def _items(tree: ModelTree):
    root = tree.invisibleRootItem()
    pending = [root.child(index) for index in range(root.childCount())]
    values = []
    while pending:
        item = pending.pop()
        values.append(item)
        pending.extend(
            item.child(index) for index in range(item.childCount())
        )
    return values


def test_a5_model_tree_uses_named_stable_edit_identity() -> None:
    _application()
    model = FEMModel(Mesh2D([], []), name="模型-板", steps=[_step()])
    tree = ModelTree()
    tree.set_model(model)
    by_kind = {
        item.data(0, ROLE_KIND): item
        for item in _items(tree)
        if item.data(0, ROLE_KIND)
        in {"boundary", "edge_load", "output"}
    }

    assert by_kind["boundary"].text(0) == "位移-固定端"
    assert by_kind["boundary"].data(0, ROLE_KEY) == (
        "分析步-静力",
        "位移-固定端",
    )
    assert by_kind["edge_load"].text(0) == "载荷-拉伸"
    assert by_kind["edge_load"].data(0, ROLE_KEY) == (
        "分析步-静力",
        "载荷-拉伸",
    )
    assert by_kind["output"].text(0) == "U"


def test_a5_manager_displays_name_separate_from_target_identity() -> None:
    _application()
    dialog = AnalysisDefinitionManagerDialog(
        [_step()],
        (),
        (RegionRef("edge", "边-固定端"), RegionRef("edge", "边-加载端")),
        (),
        2,
    )
    rows = [
        tuple(
            dialog.table.item(row, column).text()
            for column in range(dialog.table.columnCount())
        )
        for row in range(dialog.table.rowCount())
    ]

    assert any("位移-固定端 · 边-固定端" in row[2] for row in rows)
    assert any("载荷-拉伸 · 边-加载端" in row[2] for row in rows)
    assert any("结果请求-位移 · 节点" in row[2] for row in rows)
