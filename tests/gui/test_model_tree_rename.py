from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from fem.application import (
    NamedRegion,
    RegionAssignment,
    SectionDefinition,
    generate_fem_model,
)
from fem.core.model import (
    DisplacementConstraint,
    MaterialDefinition,
    NodeSet,
)
from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    namespace_part_logical_id,
)
from fem.steps.factory import static
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
import fem_gui.widgets.model_tree as model_tree_module
from fem_gui.widgets.model_tree import ModelTree, ROLE_KEY, ROLE_KIND


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _tree_items(tree: ModelTree):
    root = tree.topLevelItem(0)
    pending = [root]
    while pending:
        item = pending.pop()
        yield item
        pending.extend(
            item.child(index) for index in range(item.childCount())
        )


def _part_reference(logical_id: str) -> LogicalEntityRef:
    return LogicalEntityRef(namespace_part_logical_id("P1", logical_id))


def test_native_engineering_rows_offer_routed_rename_action(monkeypatch):
    _application()
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint(
            "Support",
            1,
            2,
            target_kind="edge",
            name="Clamp",
        ),
    )
    model = SimpleNamespace(
        name="Model-1",
        mesh=SimpleNamespace(nodes=(), elements=()),
        node_sets={"Support": NodeSet("Support", (1,))},
        element_sets={},
        surfaces={},
        edges={},
        materials={"Steel": MaterialDefinition("Steel", {})},
        sections=(),
        steps=(step,),
    )
    part = SimpleNamespace(
        id="P1",
        name="Part-1",
        suppressed=False,
        feature_history=(),
    )
    tree = ModelTree()
    tree.set_model(
        model,
        section_definitions=(SectionDefinition("Solid", "Steel"),),
        scope_names={"Support"},
        native_parts=(part,),
        active_part_id="P1",
        document_id=7,
    )
    selected = [tree.topLevelItem(0)]
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
            return self.actions["重命名"]

    routed = []
    tree.renameRequested[int, str, object].connect(
        lambda document_id, kind, key: routed.append(
            (document_id, kind, key)
        )
    )
    monkeypatch.setattr(model_tree_module, "QMenu", Menu)
    monkeypatch.setattr(
        ModelTree,
        "itemAt",
        lambda _tree, _position: selected[0],
    )
    rows = {
        item.data(0, ROLE_KIND): item
        for item in _tree_items(tree)
        if item.data(0, ROLE_KIND)
        in {"model", "part", "material", "section", "node_set", "boundary"}
    }

    for kind in (
        "model",
        "part",
        "material",
        "section",
        "node_set",
        "boundary",
    ):
        selected[0] = rows[kind]
        action_labels.clear()
        tree._show_context_menu(QPoint())
        assert "重命名" in action_labels

    assert routed == [
        (7, "model", None),
        (7, "part", "P1"),
        (7, "material", "Steel"),
        (7, "section", 0),
        (7, "node_set", "Support"),
        (7, "boundary", ("Load", "Clamp")),
    ]


def test_tree_renames_cascade_engineering_name_references(monkeypatch):
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Support",
                    (_part_reference("edge:bottom"),),
                ),
                NamedRegion(
                    "Volume",
                    (_part_reference("body:domain"),),
                ),
            )
        )
    )
    step = static("Load")
    step.boundaries = (
        DisplacementConstraint(
            "Support",
            1,
            2,
            target_kind="edge",
            name="Clamp",
        ),
    )
    window._apply_session_delta(
        window.session.replace_model_definitions(
            (
                MaterialDefinition(
                    "Steel",
                    {"E": 210000.0, "nu": 0.3},
                ),
            ),
            (SectionDefinition("Solid", "Steel"),),
            (RegionAssignment("Solid", "Volume"),),
            (step,),
        )
    )
    task = window.session.prepare_mesh_generation()
    model = generate_fem_model(task)
    window._generated_model_loaded(
        (model, build_model_geometry(model)),
        token=task.token,
    )
    names = iter((
        ("Aluminium", True),
        ("PlateSection", True),
        ("FixedEdge", True),
        ("FixedSupport", True),
    ))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: next(names),
    )

    window._rename_tree_entry("material", "Steel")
    window._rename_tree_entry("section", 0)
    window._rename_tree_entry("edge", "Support")
    window._rename_tree_entry("boundary", ("Load", "Clamp"))

    assert window.document.materials[0].name == "Aluminium"
    assert window.document.sections[0].name == "PlateSection"
    assert window.document.sections[0].material == "Aluminium"
    assert window.document.assignments[0].section_name == "PlateSection"
    assert window.document.assignments[0].region_name == "Volume"
    assert set(window.document.named_regions) == {"FixedEdge", "Volume"}
    boundary = window.document.steps[0].boundaries[0]
    assert boundary.name == "FixedSupport"
    assert boundary.target == "FixedEdge"

    rows = tuple(_tree_items(window.model_tree))
    assert any(
        item.data(0, ROLE_KIND) == "material"
        and item.text(0) == "Aluminium"
        for item in rows
    )
    assert any(
        item.data(0, ROLE_KIND) == "section"
        and item.text(0).startswith("PlateSection（")
        for item in rows
    )
    assert any(
        item.data(0, ROLE_KIND) == "edge"
        and item.data(0, ROLE_KEY) == "FixedEdge"
        for item in rows
    )
    assert any(
        item.data(0, ROLE_KIND) == "boundary"
        and item.text(0) == "FixedSupport"
        for item in rows
    )
    window.close()
