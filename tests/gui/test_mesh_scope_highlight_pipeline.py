from __future__ import annotations

import os
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyvista
from PySide6.QtWidgets import QApplication

from fem.application import MeshEntityRef
from fem.core.model import FEMModel
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    _MeshScopeHighlightPipeline,
    _positive_id_indices,
)
from tests.helpers.mesh_builders import make_selection_hex_mesh


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Plotter:
    def __init__(self) -> None:
        self.inner = pyvista.Plotter(off_screen=True)
        self.mesh_calls: list[str] = []
        self.mesh_options: dict[str, dict[str, object]] = {}
        self.render_count = 0

    def add_mesh(self, dataset, **kwargs):
        self.mesh_calls.append(kwargs["name"])
        self.mesh_options[kwargs["name"]] = dict(kwargs)
        return self.inner.add_mesh(dataset, **kwargs)

    def render(self) -> None:
        self.render_count += 1

    def close(self) -> None:
        self.inner.close()


def _viewport_with_scope_pipelines() -> tuple[FEMViewport, _Plotter]:
    _application()
    viewport_module._pyvista = pyvista
    model = FEMModel(make_selection_hex_mesh())
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._model = model
    viewport._geometry = build_model_geometry(model)
    viewport._grid = viewport._make_grid(viewport._geometry.points)
    viewport._pick_grid = viewport._grid
    viewport._install_mesh_scope_pick_bindings()
    viewport._install_mesh_scope_highlight_pipelines()
    return viewport, plotter


def _first_reference(viewport: FEMViewport, kind: str) -> MeshEntityRef:
    if kind == "node":
        return MeshEntityRef.node(1)
    if kind == "element":
        return MeshEntityRef.element(1)
    return next(
        reference
        for reference in viewport._mesh_scope_ref_to_pick_id
        if reference.kind == kind
    )


def test_four_mesh_scope_kinds_reuse_actor_and_threshold_pipeline() -> None:
    viewport, plotter = _viewport_with_scope_pipelines()
    try:
        assert set(viewport._mesh_scope_highlight_pipelines) == {
            "node",
            "element",
            "edge",
            "face",
        }
        assert len(plotter.mesh_calls) == 4
        actor_ids = {
            kind: id(pipeline.actor)
            for kind, pipeline in viewport._mesh_scope_highlight_pipelines.items()
        }

        for kind in ("node", "element", "edge", "face"):
            reference = _first_reference(viewport, kind)
            viewport.highlight_mesh_entities({reference})
            pipeline = viewport._mesh_scope_highlight_pipelines[kind]
            pipeline.algorithm.Update()

            assert pipeline.selected_indices == {
                viewport._mesh_scope_highlight_index(reference)
            }
            assert pipeline.algorithm.GetOutput().GetNumberOfCells() == 1
            assert id(pipeline.actor) == actor_ids[kind]
            assert len(plotter.mesh_calls) == 4

        viewport.clear_selection()
        assert all(
            not pipeline.selected_indices
            for pipeline in viewport._mesh_scope_highlight_pipelines.values()
        )
        assert {
            kind: id(pipeline.actor)
            for kind, pipeline in viewport._mesh_scope_highlight_pipelines.items()
        } == actor_ids
        assert len(plotter.mesh_calls) == 4
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


def test_incremental_updates_do_not_extract_or_create_actors(monkeypatch) -> None:
    viewport, plotter = _viewport_with_scope_pipelines()
    try:
        def forbidden_extract(*_args, **_kwargs):
            raise AssertionError("selection update must not extract cells")

        monkeypatch.setattr(pyvista.DataSet, "extract_cells", forbidden_extract)
        first = MeshEntityRef.node(1)
        second = MeshEntityRef.node(2)
        selected = {first}
        viewport.highlight_mesh_entities(selected)
        selected.add(second)
        viewport.highlight_mesh_entities(
            selected,
            changed_references={second},
        )
        selected.remove(first)
        viewport.highlight_mesh_entities(
            selected,
            changed_references={first},
        )

        pipeline = viewport._mesh_scope_highlight_pipelines["node"]
        assert pipeline.selected_indices == {1}
        assert len(plotter.mesh_calls) == 4
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


def test_body_scope_uses_surface_and_geometry_edges_without_element_wireframe() -> None:
    viewport, plotter = _viewport_with_scope_pipelines()
    try:
        viewport.highlight_mesh_entities(
            {MeshEntityRef.element(1)},
            entity_kind="body",
        )

        assert viewport._mesh_scope_highlight_kind == "body"
        assert not viewport._mesh_scope_highlight_pipelines[
            "element"
        ].selected_indices
        assert {
            "mesh_scope_selection_body",
            "mesh_scope_selection_body_edges",
        }.issubset(viewport._actors)
        surface_options = plotter.mesh_options["mesh_scope_selection_body"]
        assert surface_options["show_edges"] is False
        assert "style" not in surface_options
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


def test_body_surface_is_reused_for_hover_and_selection(monkeypatch) -> None:
    viewport, plotter = _viewport_with_scope_pipelines()
    calls = 0
    original = pyvista.DataSet.extract_cells

    def record_extract(dataset, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(pyvista.DataSet, "extract_cells", record_extract)
    try:
        first = viewport._mesh_body_render_data((1,))
        second = viewport._mesh_body_render_data((1,))

        assert second[0] is first[0]
        assert second[1] is first[1]
        assert calls == 1
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


def test_result_provenance_ids_are_grouped_in_one_pass() -> None:
    identifiers = np.tile(np.arange(1, 50_001, dtype=np.int64), 3)

    grouped = _positive_id_indices(identifiers)

    assert len(grouped) == 50_000
    assert grouped[1] == (0, 50_000, 100_000)
    assert grouped[50_000] == (49_999, 99_999, 149_999)


def test_scope_render_requests_are_coalesced_and_keep_final_selection() -> None:
    application = _application()
    viewport, plotter = _viewport_with_scope_pipelines()
    try:
        selected: set[MeshEntityRef] = set()
        for node_id in (1, 2, 3):
            reference = MeshEntityRef.node(node_id)
            selected.add(reference)
            viewport.highlight_mesh_entities(
                selected,
                changed_references={reference},
            )

        assert plotter.render_count == 0
        application.processEvents()
        assert plotter.render_count == 1
        assert viewport._mesh_scope_highlight_pipelines[
            "node"
        ].selected_indices == {0, 1, 2}

        viewport.set_selection_mode("mesh_element")
        application.processEvents()
        assert not viewport._mesh_scope_highlight_pipelines[
            "node"
        ].selected_indices
        assert viewport._mesh_scope_highlight_kind is None
        assert plotter.render_count == 2
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


def test_definition_rebind_reuses_pipelines_and_remesh_rebuilds_them() -> None:
    viewport, plotter = _viewport_with_scope_pipelines()
    try:
        original = {
            kind: pipeline.actor
            for kind, pipeline in viewport._mesh_scope_highlight_pipelines.items()
        }
        rebound_model = FEMModel(make_selection_hex_mesh(), name="definitions changed")
        rebound_geometry = build_model_geometry(rebound_model)
        viewport.rebind_model_artifact(rebound_model, rebound_geometry)
        assert {
            kind: pipeline.actor
            for kind, pipeline in viewport._mesh_scope_highlight_pipelines.items()
        } == original

        viewport._reset_mesh_scope_highlight_pipelines()
        viewport._clear_mesh_scope_pick_bindings()
        viewport._model = rebound_model
        viewport._geometry = rebound_geometry
        viewport._grid = viewport._make_grid(rebound_geometry.points)
        viewport._pick_grid = viewport._grid
        viewport._install_mesh_scope_pick_bindings()
        viewport._install_mesh_scope_highlight_pipelines()
        assert all(
            viewport._mesh_scope_highlight_pipelines[kind].actor
            is not original[kind]
            for kind in original
        )
        assert all(
            not pipeline.selected_indices
            for pipeline in viewport._mesh_scope_highlight_pipelines.values()
        )
    finally:
        viewport._mesh_scope_render_timer.stop()
        plotter.close()
        viewport._plotter = None
        viewport.close()


class _Modified:
    def Modified(self) -> None:
        pass


class _Actor:
    def SetVisibility(self, _visible: bool) -> None:
        pass


class _NoIterationSet(set):
    def __iter__(self):
        raise AssertionError("incremental update must not scan all selected indices")


def test_main_window_forwards_live_set_and_changed_refs_without_sorting() -> None:
    selected = {MeshEntityRef.node(1), MeshEntityRef.node(2)}
    changed = {MeshEntityRef.node(2)}
    calls = []
    fake = SimpleNamespace(
        _selected_mesh_scope_refs=selected,
        viewport_panel=SimpleNamespace(
            scope_creation_bar=SimpleNamespace(
                set_selection_ready=lambda ready: calls.append(("ready", ready))
            )
        ),
        viewport=SimpleNamespace(
            highlight_mesh_entities=lambda refs, **kwargs: calls.append(
                ("highlight", refs, kwargs)
            )
        ),
        status_panel=SimpleNamespace(
            set_selection_mode=lambda mode: None,
            set_object=lambda label: None,
        ),
        actions={
            "selected_info": SimpleNamespace(setEnabled=lambda enabled: None)
        },
    )

    FEMMainWindow._refresh_mesh_scope_selection(
        fake,
        "node",
        changed_references=changed,
    )

    assert calls == [
        ("ready", True),
        (
            "highlight",
            selected,
            {"changed_references": changed, "entity_kind": "node"},
        ),
    ]


def test_main_window_preserves_body_highlight_semantics() -> None:
    selected = {MeshEntityRef.element(1, part_id="P1")}
    calls = []
    topology = SimpleNamespace(
        expand=lambda _kind, _reference: tuple(selected),
    )
    fake = SimpleNamespace(
        document=SimpleNamespace(model=object()),
        _selected_mesh_scope_refs=selected,
        _mesh_selection_topology=lambda: topology,
        viewport_panel=SimpleNamespace(
            scope_creation_bar=SimpleNamespace(set_selection_ready=lambda _ready: None)
        ),
        viewport=SimpleNamespace(
            highlight_mesh_entities=lambda refs, **kwargs: calls.append(
                (refs, kwargs)
            )
        ),
        status_panel=SimpleNamespace(
            set_selection_mode=lambda _mode: None,
            set_object=lambda _label: None,
        ),
        actions={
            "selected_info": SimpleNamespace(setEnabled=lambda _enabled: None)
        },
    )

    FEMMainWindow._refresh_mesh_scope_selection(fake, "body")

    assert calls == [
        (
            selected,
            {"changed_references": None, "entity_kind": "body"},
        )
    ]


def test_element_pick_uses_loaded_owner_map_without_topology_rebuild() -> None:
    fake = SimpleNamespace(
        document=SimpleNamespace(model=object()),
        _pending_analysis_selection=None,
        viewport=SimpleNamespace(_mesh_body_owner_by_element_id={7: "P2"}),
        _mesh_selection_topology=lambda: (_ for _ in ()).throw(
            AssertionError("element pick must not rebuild topology")
        ),
    )

    expanded = FEMMainWindow._expand_mesh_selection_reference(
        fake,
        "element",
        MeshEntityRef.element(7),
    )

    assert expanded == (MeshEntityRef.element(7, part_id="P2"),)


def test_completed_scope_clears_persistent_viewport_selection() -> None:
    calls = []
    selected = {MeshEntityRef.node(1)}
    fake = SimpleNamespace(
        _pending_analysis_selection="scope",
        _pending_analysis_edit=None,
        _pending_scope_kind="node",
        _scope_selection_overlay_active=False,
        _selected_geometry_refs=set(),
        _selected_mesh_scope_refs=selected,
        viewport_panel=SimpleNamespace(
            scope_creation_bar=SimpleNamespace(
                finish=lambda: calls.append("finish")
            )
        ),
        viewport=SimpleNamespace(
            clear_selection=lambda: calls.append("clear")
        ),
        status_panel=SimpleNamespace(set_object=lambda: None),
        actions={
            "selected_info": SimpleNamespace(setEnabled=lambda enabled: None)
        },
        _update_action_states=lambda: None,
        _restore_temporary_selection_context=lambda _owner: None,
        create_displacement_boundary=lambda: None,
        create_load=lambda: None,
        assign_section_to_region=lambda: None,
    )

    FEMMainWindow._finish_scope_creation_from_bar(fake, "NodeSet-1")

    assert calls == ["finish", "clear"]
    assert not selected
    assert fake._pending_analysis_selection is None
    assert fake._pending_scope_kind is None


def test_single_toggle_in_100k_selection_is_below_old_rebuild_cost() -> None:
    _application()
    count = 100_000
    references = {MeshEntityRef.node(index + 1) for index in range(count)}
    geometry = SimpleNamespace(
        node_id_to_point_index={index + 1: index for index in range(count)},
        element_id_to_cell_index={},
        points=np.zeros((count, 3), dtype=float),
    )
    mask = np.ones(count, dtype=np.uint8)
    pipeline = _MeshScopeHighlightPipeline(
        "node",
        _Modified(),
        _Modified(),
        mask,
        _Modified(),
        _Actor(),
        _NoIterationSet(range(count)),
    )
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(render=lambda: None)
    viewport._geometry = geometry
    viewport._mesh_scope_highlight_pipelines = {"node": pipeline}
    viewport._mesh_scope_highlight_kind = "node"
    removed = MeshEntityRef.node(count)
    references.remove(removed)

    started = perf_counter()
    viewport.highlight_mesh_entities(
        references,
        changed_references={removed},
        entity_kind="node",
    )
    incremental = perf_counter() - started

    started = perf_counter()
    indices = np.fromiter(
        (
            geometry.node_id_to_point_index[int(reference.node_id)]
            for reference in references
        ),
        dtype=np.int64,
        count=len(references),
    )
    geometry.points[indices].copy()
    old_rebuild = perf_counter() - started

    assert incremental <= old_rebuild * 0.30
    assert len(pipeline.selected_indices) == count - 1
    assert count - 1 not in pipeline.selected_indices
    assert mask[-1] == 0
    viewport._mesh_scope_render_timer.stop()
    viewport._plotter = None
    viewport.close()
