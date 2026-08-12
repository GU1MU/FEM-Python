from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import MeshEntityRef
from fem.application.definitions import mesh_entity_ref_sort_key
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    NATIVE_SCOPE_CATALOG_KEY,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel
import fem_gui.scope_selection as scope_selection_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.scope_selection import build_mesh_selection_topology
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _two_imported_parts_model(*, reverse: bool = False) -> FEMModel:
    nodes = [
        Node2D(1, 0.0, 0.0),
        Node2D(2, 1.0, 0.0),
        Node2D(3, 2.0, 0.0),
        Node2D(4, 0.0, 1.0),
        Node2D(5, 1.0, 1.0),
        Node2D(6, 2.0, 1.0),
        Node2D(11, 10.0, 0.0),
        Node2D(12, 11.0, 0.0),
        Node2D(13, 12.0, 0.0),
        Node2D(14, 10.0, 1.0),
        Node2D(15, 11.0, 1.0),
        Node2D(16, 12.0, 1.0),
    ]
    elements = [
        Element2D(1, [1, 2, 5, 4], type="Quad4"),
        Element2D(2, [2, 3, 6, 5], type="Quad4"),
        Element2D(10, [11, 12, 15, 14], type="Quad4"),
        Element2D(11, [12, 13, 16, 15], type="Quad4"),
    ]
    return FEMModel(
        Mesh2D(
            list(reversed(nodes)) if reverse else nodes,
            list(reversed(elements)) if reverse else elements,
        )
    )


def test_imported_mesh_topology_has_stable_parts_and_whole_edges() -> None:
    first = build_mesh_selection_topology(_two_imported_parts_model())
    repeated = build_mesh_selection_topology(
        _two_imported_parts_model(reverse=True)
    )

    assert first.part_elements == repeated.part_elements
    assert first.edge_expansions == repeated.edge_expansions
    assert {
        part_id: tuple(reference.element_id for reference in references)
        for part_id, references in first.part_elements.items()
    } == {"P1": (1, 2), "P2": (10, 11)}

    p2_body = first.expand("body", MeshEntityRef.element(10))
    assert tuple(reference.element_id for reference in p2_body) == (10, 11)
    assert {reference.part_id for reference in p2_body} == {"P2"}

    whole_edge = next(
        group
        for group in set(first.edge_expansions.values())
        if len(group) == 2
        and {reference.part_id for reference in group} == {"P1"}
    )
    assert first.expand("edge", whole_edge[-1]) == whole_edge


def test_mesh_modules_defer_and_share_imported_topology_inference(
    monkeypatch,
) -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("lazy-selection-topology.inp"),
        (model, build_model_geometry(model)),
    )
    window._scope_selection_topology_cache = None
    window._mesh_selection_topology_cache = None
    calls = []
    original = scope_selection_module._inferred_scope_selection_topology

    def record_inference(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scope_selection_module,
        "_inferred_scope_selection_topology",
        record_inference,
    )

    for module_name in ("网格", "模型", "分析"):
        window.ribbon.set_current(module_name)

    assert calls == []
    assert window._scope_selection_topology_cache is None
    assert window._mesh_selection_topology_cache is None

    window._set_selection_filter("edge")

    assert calls == [True]
    assert window._scope_selection_topology_cache is not None
    assert window._mesh_selection_topology_cache is not None
    window._scope_selection_topology()
    assert calls == [True]
    window.close()


def test_large_mesh_topology_is_prepared_before_enabling_edge_picks(
    monkeypatch,
) -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("background-selection-topology.inp"),
        (model, build_model_geometry(model)),
    )
    window.scope_background_reference_threshold = 0
    window._scope_selection_topology_cache = None
    window._mesh_selection_topology_cache = None
    task = {}

    def capture_task(
        workload,
        on_success,
        error_title,
        on_failure=None,
        **options,
    ):
        task.update(
            workload=workload,
            on_success=on_success,
            error_title=error_title,
            on_failure=on_failure,
            **options,
        )
        return True

    monkeypatch.setattr(window, "_start_task", capture_task)

    window._set_selection_filter("edge")

    assert task["task_name"] == "准备网格选择拓扑"
    assert window._scope_selection_topology_cache is None
    assert window._mesh_selection_topology_cache is None
    assert window.viewport._selection_mode != "mesh_edge"

    class ImmediateContext:
        def report(self, _message):
            return None

        def checkpoint(self):
            return None

    payload = task["workload"](ImmediateContext())
    outcome = task["apply_result"](payload)
    task["on_success"](outcome.projection_value)

    assert window._scope_selection_topology_cache is not None
    assert window._mesh_selection_topology_cache is not None
    assert window.viewport._selection_mode == "mesh_edge"
    window.close()


def test_analysis_node_selection_does_not_materialize_scope_topology() -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("node-selection-topology.inp"),
        (model, build_model_geometry(model)),
    )
    window._scope_selection_topology_cache = None
    window._mesh_selection_topology_cache = None

    window._request_analysis_geometry_selection("load", "node")

    assert window._scope_selection_topology_cache is None
    assert window._mesh_selection_topology_cache is None
    assert window._temporary_selection_owner == "analysis_scope"
    assert window.viewport._selection_mode == "mesh_node"
    window.close()


def test_native_mesh_topology_uses_exact_catalog_and_numeric_part_order() -> None:
    model = _two_imported_parts_model()
    model.metadata[NATIVE_PART_OWNERSHIP_KEY] = {
        "P2": {
            "node_ids": (1, 2, 3, 4, 5, 6),
            "element_ids": (1, 2),
        },
        "P10": {
            "node_ids": (11, 12, 13, 14, 15, 16),
            "element_ids": (10, 11),
        },
    }
    model.metadata[NATIVE_SCOPE_CATALOG_KEY] = {
        "edge:P2/bottom": {
            "kind": "edge",
            "node_ids": (1, 2, 3),
            "element_ids": (1, 2),
            "edges": ((1, 0, (1, 2)), (2, 0, (2, 3))),
            "faces": (),
        },
        "edge:P10/bottom": {
            "kind": "edge",
            "node_ids": (11, 12, 13),
            "element_ids": (10, 11),
            "edges": ((10, 0, (11, 12)), (11, 0, (12, 13))),
            "faces": (),
        },
        "face:P2/domain": {
            "kind": "face",
            "node_ids": (1, 2, 3, 4, 5, 6),
            "element_ids": (1, 2),
            "edges": (),
            "faces": (),
        },
        "face:P10/domain": {
            "kind": "face",
            "node_ids": (11, 12, 13, 14, 15, 16),
            "element_ids": (10, 11),
            "edges": (),
            "faces": (),
        },
    }

    topology = build_mesh_selection_topology(model)

    assert tuple(topology.part_elements) == ("P2", "P10")
    exact = topology.expand(
        "edge",
        MeshEntityRef.edge(11, 0, (12, 13)),
    )
    assert tuple(reference.element_id for reference in exact) == (10, 11)
    assert {reference.part_id for reference in exact} == {"P10"}
    exact_face = topology.expand("face", MeshEntityRef.element(11))
    assert tuple(reference.element_id for reference in exact_face) == (10, 11)
    assert {reference.part_id for reference in exact_face} == {"P10"}
    ordered = sorted(
        (
            MeshEntityRef.element(10, part_id="P10"),
            MeshEntityRef.element(1, part_id="P2"),
        ),
        key=mesh_entity_ref_sort_key,
    )
    assert [reference.part_id for reference in ordered] == ["P2", "P10"]


def test_mesh_filters_invalidate_previous_selection_and_group_toggle(
    monkeypatch,
) -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("selection-semantics.inp"),
        (model, build_model_geometry(model)),
    )
    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: False,
    )

    window._set_selection_filter("point")
    window._on_mesh_scope_entity_pick(MeshEntityRef.node(1))
    assert window._selected_mesh_scope_refs == {
        MeshEntityRef.node(1, part_id="P1")
    }
    assert window.selection.node_id is None
    window._on_viewport_pick_missed("mesh_node")
    assert not window._selected_mesh_scope_refs
    window._on_mesh_scope_entity_pick(MeshEntityRef.node(1))
    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: True,
    )
    window._on_viewport_pick_missed("mesh_node")
    assert window._selected_mesh_scope_refs == {
        MeshEntityRef.node(1, part_id="P1")
    }
    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: False,
    )

    window._set_selection_filter("element")
    assert not window._selected_mesh_scope_refs
    window._on_mesh_scope_entity_pick(MeshEntityRef.element(1))
    assert window._selected_mesh_scope_refs == {
        MeshEntityRef.element(1, part_id="P1")
    }
    assert window.selection.element_id is None

    window._set_selection_filter("edge")
    topology = window._mesh_selection_topology()
    whole_edge = next(
        group
        for group in set(topology.edge_expansions.values())
        if len(group) == 2
    )
    window._on_mesh_scope_entity_pick(whole_edge[0])
    assert window._selected_mesh_scope_refs == set(whole_edge)
    assert window.status_panel.object_label.text() == "对象：1 个拓扑边"

    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: True,
    )
    window._on_mesh_scope_entity_pick(whole_edge[-1])
    assert not window._selected_mesh_scope_refs

    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: False,
    )
    window._set_selection_filter("body")
    window._on_mesh_scope_entity_pick(MeshEntityRef.element(10))
    assert window._canonical_mesh_scope_selection() == (
        MeshEntityRef.element(10, part_id="P2"),
        MeshEntityRef.element(11, part_id="P2"),
    )
    assert window.status_panel.object_label.text() == "对象：1 个部件"
    window._set_selection_filter("element")
    assert not window._selected_mesh_scope_refs
    window._on_mesh_scope_entity_pick(MeshEntityRef.element(10))
    assert window._selected_mesh_scope_refs == {
        MeshEntityRef.element(10, part_id="P2"),
    }
    window._set_selection_filter("body")
    assert not window._selected_mesh_scope_refs
    window._set_selection_filter("point")
    assert not window._selected_mesh_scope_refs
    window.close()


def test_mesh_box_ctrl_toggles_each_whole_topology_group_once(monkeypatch) -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("selection-box-semantics.inp"),
        (model, build_model_geometry(model)),
    )
    window._set_selection_filter("edge")
    topology = window._mesh_selection_topology()
    whole_edge = next(
        group
        for group in set(topology.edge_expansions.values())
        if len(group) == 2
    )
    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: False,
    )
    window._on_mesh_entities_box_selected(whole_edge)
    assert window._selected_mesh_scope_refs == set(whole_edge)

    monkeypatch.setattr(
        window,
        "_geometry_pick_is_additive",
        lambda: True,
    )
    window._on_mesh_entities_box_selected(whole_edge)
    assert not window._selected_mesh_scope_refs
    window.close()


def test_mesh_body_box_expands_shared_group_once(monkeypatch) -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("selection-body-box-performance.inp"),
        (model, build_model_geometry(model)),
    )
    window._set_selection_filter("body")
    calls = 0
    original = window._expand_mesh_selection_reference

    def expand(selection_filter, reference):
        nonlocal calls
        calls += 1
        return original(selection_filter, reference)

    monkeypatch.setattr(window, "_expand_mesh_selection_reference", expand)
    monkeypatch.setattr(window, "_geometry_pick_is_additive", lambda: False)

    window._on_mesh_entities_box_selected(
        tuple(MeshEntityRef.element(element.id) for element in model.mesh.elements)
    )

    assert calls == len(model.mesh.elements)
    assert len(window._selected_mesh_scope_refs) == len(model.mesh.elements)
    window.close()


def test_guided_scope_cancel_restores_module_selection_context() -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("temporary-scope-context.inp"),
        (model, build_model_geometry(model)),
    )
    window._set_selection_filter("body")

    window._request_analysis_geometry_selection("scope", "edge")

    assert window._temporary_selection_owner == "analysis_scope"
    assert window._selection_context.space == "geometry"
    assert window._selection_context.active_filter == "edge"
    assert window.viewport._selection_mode == "geometry_edge"

    window._cancel_guided_selection()

    assert window._temporary_selection_context is None
    assert window._selection_context.space == "mesh"
    assert window._selection_context.active_filter == "body"
    assert window.viewport._selection_mode == "mesh_body"
    assert not window._selected_geometry_refs
    assert not window._selected_mesh_scope_refs

    window._request_analysis_geometry_selection("scope", "edge")
    window._finish_scope_creation_from_bar("CreatedEdge")

    assert window._temporary_selection_context is None
    assert window._selection_context.space == "mesh"
    assert window._selection_context.active_filter == "body"
    assert window.viewport._selection_mode == "mesh_body"
    window.close()


def test_close_model_clears_contextual_selection_and_viewport_pick_state() -> None:
    _application()
    model = _two_imported_parts_model()
    window = FEMMainWindow()
    window._model_loaded(
        Path("close-selection-state.inp"),
        (model, build_model_geometry(model)),
    )
    window._set_selection_filter("body")
    window._on_mesh_scope_entity_pick(MeshEntityRef.element(10))
    assert window._selected_mesh_scope_refs

    assert window.close_model(confirm=False)

    assert not window._selected_geometry_refs
    assert not window._selected_mesh_scope_refs
    assert window.viewport._hover_hit is None
    assert window.viewport._result_render_payload is None
    assert not window.viewport._mesh_scope_highlight_pipelines
    assert "selection" not in window.viewport._actors
    assert "preselection" not in window.viewport._actors
    window.close()
