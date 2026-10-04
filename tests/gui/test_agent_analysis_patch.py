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
        "Step-静力",
        boundaries=(
            DisplacementConstraint(
                "Edge-固定端",
                1,
                2,
                target_kind="edge",
                name="Displacement-固定端",
            ),
        ),
        edge_loads=(
            EdgeLoad(
                "Edge-加载端",
                (10.0, 0.0),
                name="Load-拉伸",
            ),
        ),
        outputs=(
            OutputRequest(
                "field",
                "node",
                ("U",),
                name="Output-位移",
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


def test_model_tree_uses_named_stable_edit_identity(gui_application) -> None:
    _application()
    model = FEMModel(Mesh2D([], []), name="Model-板", steps=[_step()])
    tree = ModelTree()
    tree.set_model(model)
    by_kind = {
        item.data(0, ROLE_KIND): item
        for item in _items(tree)
        if item.data(0, ROLE_KIND)
        in {"boundary", "edge_load", "output"}
    }

    assert by_kind["boundary"].text(0) == "Displacement-固定端"
    assert by_kind["boundary"].data(0, ROLE_KEY) == (
        "Step-静力",
        "Displacement-固定端",
    )
    assert by_kind["edge_load"].text(0) == "Load-拉伸"
    assert by_kind["edge_load"].data(0, ROLE_KEY) == (
        "Step-静力",
        "Load-拉伸",
    )
    assert by_kind["output"].text(0) == "U"


def test_manager_displays_name_separate_from_target_identity(gui_application) -> None:
    _application()
    dialog = AnalysisDefinitionManagerDialog(
        [_step()],
        (),
        (RegionRef("edge", "Edge-固定端"), RegionRef("edge", "Edge-加载端")),
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

    assert any("Displacement-固定端 - Edge-固定端" in row[2] for row in rows)
    assert any("Load-拉伸 - Edge-加载端" in row[2] for row in rows)
    assert any("Output-位移 - Node" in row[2] for row in rows)
