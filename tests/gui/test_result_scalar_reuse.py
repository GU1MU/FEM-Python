from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyvista
import pytest
from PySide6.QtWidgets import QApplication

from fem.application.results import (
    FieldAssociation,
    FieldLocation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultCellKind,
    ResultFieldId,
    ResultFieldTopology,
    ResultSourceKey,
    ResultValueLayout,
    ResultVariable,
    ScalarFieldSelection,
)
from fem_gui.visualization.result_renderer import (
    build_result_render_payload,
    validate_result_render_payload,
)
from fem_gui.visualization.scene import DisplayState
import fem_gui.widgets.viewport as viewport_module
from fem_gui.widgets.viewport import FEMViewport


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _payload(component: str, values: tuple[float, ...]):
    source = ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=1,
        step_name="Step-1",
        run_id="run-1",
    )
    key = FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.ELEMENT_NODAL,
            )
        ),
        recovery_contract=1,
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    locations = tuple(
        FieldLocation(
            association=FieldAssociation.ELEMENT_NODE,
            coordinates=tuple(point),
            displacement=(0.0, 0.0, 0.0),
            node_id=node_id,
            element_id=1,
            local_node=local_node,
        )
        for node_id, local_node, point in zip(
            (1, 2, 3),
            (1, 2, 3),
            points,
            strict=True,
        )
    )
    topology = ResultFieldTopology(
        source=source,
        materialization_generation=0,
        selection=ScalarFieldSelection(key, component),
        deformation_scale=1.0,
        points=points,
        cells=((0, 1, 2),),
        cell_kinds=(ResultCellKind.FEM_ELEMENT,),
        canonical_element_types=("Tri3",),
        values=np.asarray(values),
        value_layout=ResultValueLayout.POINT,
        point_locations=locations,
        cell_locations=(None,),
    )
    return build_result_render_payload(topology)


def test_component_switch_reuses_grid_and_actor() -> None:
    _application()
    first = _payload("S11", (1.0, 2.0, 3.0))
    second = _payload("S22", (4.0, 5.0, 6.0))
    viewport = FEMViewport()
    viewport._plotter = pyvista.Plotter(off_screen=True)
    viewport._display = DisplayState("deformed", True)

    viewport.set_result_render_payload(first)
    viewport._update_result_layer()
    rendered_grid = viewport._result_grid
    rendered_actor = viewport._actors["result"]

    viewport.set_result_render_payload(second)
    viewport.set_display("deformed", True)

    installed = viewport._result_render_payload
    assert installed is not None
    assert viewport._result_grid is rendered_grid
    assert viewport._actors["result"] is rendered_actor
    assert installed.dataset is rendered_grid
    assert installed.topology.selection.component == "S22"
    assert rendered_actor.mapper.array_name == installed.scalar_name
    assert tuple(viewport._plotter.scalar_bars.keys()) == ("S, S22",)
    np.testing.assert_array_equal(
        installed.dataset.point_data[installed.scalar_name],
        (4.0, 5.0, 6.0),
    )
    assert validate_result_render_payload(first) is first
    assert validate_result_render_payload(installed) is installed
    viewport.close()


def test_failed_component_switch_can_restore_exact_previous_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    first = _payload("S11", (1.0, 2.0, 3.0))
    second = _payload("S22", (4.0, 5.0, 6.0))
    viewport = FEMViewport()
    viewport._plotter = pyvista.Plotter(off_screen=True)
    viewport._display = DisplayState("deformed", True)
    viewport.set_result_render_payload(first)
    viewport._update_result_layer()
    rendered_actor = viewport._actors["result"]
    original_render = viewport._render

    def fail_render() -> None:
        raise RuntimeError("render failed")

    viewport.set_result_render_payload(second)
    monkeypatch.setattr(viewport, "_render", fail_render)
    with pytest.raises(RuntimeError, match="render failed"):
        viewport.set_display("deformed", True)

    monkeypatch.setattr(viewport, "_render", original_render)
    viewport.set_result_render_payload(first)
    viewport.set_display("deformed", True)

    assert viewport._result_render_payload is first
    assert viewport._actors["result"] is rendered_actor
    assert rendered_actor.mapper.array_name == first.scalar_name
    assert validate_result_render_payload(first) is first
    viewport.close()


def test_cached_layout_reuses_dataset_and_provenance_indexes() -> None:
    _application()
    first = _payload("S11", (1.0, 2.0, 3.0))
    first_topology = first.topology
    second_topology = ResultFieldTopology(
        source=first_topology.source,
        materialization_generation=first_topology.materialization_generation,
        selection=ScalarFieldSelection(
            first_topology.selection.field_key,
            "S22",
        ),
        deformation_scale=first_topology.deformation_scale,
        points=first_topology.points,
        cells=first_topology.cells,
        cell_kinds=first_topology.cell_kinds,
        canonical_element_types=first_topology.canonical_element_types,
        values=np.asarray((4.0, 5.0, 6.0)),
        value_layout=first_topology.value_layout,
        point_locations=first_topology.point_locations,
        cell_locations=first_topology.cell_locations,
    )
    second = build_result_render_payload(
        second_topology,
        reusable=first,
    )
    assert second.dataset is first.dataset

    viewport = FEMViewport()
    viewport._plotter = pyvista.Plotter(off_screen=True)
    viewport._display = DisplayState("deformed", True)
    viewport.set_result_render_payload(first)
    viewport._update_result_layer()
    node_index = viewport._result_point_index_to_node_id
    element_index = viewport._result_point_index_to_element_id
    cell_index = viewport._result_cell_index_to_element_id

    viewport.set_result_render_payload(second)

    assert viewport._result_point_index_to_node_id is node_index
    assert viewport._result_point_index_to_element_id is element_index
    assert viewport._result_cell_index_to_element_id is cell_index
    viewport.close()


def test_install_transaction_reuses_one_payload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    payload = _payload("S11", (1.0, 2.0, 3.0))
    calls: list[object] = []
    original = viewport_module._require_result_render_payload

    def record_validation(candidate: object):
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(
        viewport_module,
        "_require_result_render_payload",
        record_validation,
    )
    viewport = FEMViewport()

    viewport.set_result_render_payload(payload)
    viewport._update_result_layer()

    assert calls == [payload]

    viewport._update_result_layer()

    assert calls == [payload, payload]
    viewport.close()


class _CountingMapper:
    def __init__(self, array_name: str) -> None:
        self.array_name = array_name
        self.scalar_visibility = True
        self.scalar_range = (0.0, 1.0)
        self.update_count = 0

    def Update(self) -> None:
        self.update_count += 1


class _CountingActor:
    def __init__(self, array_name: str) -> None:
        self.mapper = _CountingMapper(array_name)
        self.visible = True

    def SetVisibility(self, visible: bool) -> None:
        self.visible = bool(visible)


class _CountingPlotter:
    def __init__(self) -> None:
        self.mesh_calls: list[tuple[object, dict[str, object]]] = []
        self.remove_actor_calls = 0
        self.render_count = 0
        self.scalar_bars: dict[str, object] = {}

    def add_mesh(self, dataset: object, **options: object) -> _CountingActor:
        self.mesh_calls.append((dataset, options))
        return _CountingActor(str(options.get("scalars", "")))

    def remove_actor(self, _actor: object, **_options: object) -> None:
        self.remove_actor_calls += 1

    def remove_scalar_bar(self, title: str, **_options: object) -> None:
        self.scalar_bars.pop(title, None)

    def add_scalar_bar(self, *, title: str, **_options: object) -> None:
        self.scalar_bars[title] = object()

    def render(self) -> None:
        self.render_count += 1


def test_stress_component_switch_has_constant_geometry_work() -> None:
    _application()
    viewport = FEMViewport()
    plotter = _CountingPlotter()
    viewport._plotter = plotter
    viewport._display = DisplayState("deformed", True)
    first = _payload("S11", (1.0, 2.0, 3.0))
    viewport.set_result_render_payload(first)
    viewport._update_result_layer()
    rendered_grid = viewport._result_grid
    rendered_actor = viewport._actors["result"]
    node_index = viewport._result_point_index_to_node_id
    element_index = viewport._result_point_index_to_element_id
    cell_index = viewport._result_cell_index_to_element_id

    for offset, component in enumerate(
        ("S22", "S33", "S12", "Mises"),
        start=1,
    ):
        current = viewport._result_render_payload
        assert current is not None
        topology = current.topology
        next_topology = ResultFieldTopology(
            source=topology.source,
            materialization_generation=topology.materialization_generation,
            selection=ScalarFieldSelection(
                topology.selection.field_key,
                component,
            ),
            deformation_scale=topology.deformation_scale,
            points=topology.points,
            cells=topology.cells,
            cell_kinds=topology.cell_kinds,
            canonical_element_types=topology.canonical_element_types,
            values=np.asarray(
                (offset + 1.0, offset + 2.0, offset + 3.0)
            ),
            value_layout=topology.value_layout,
            point_locations=topology.point_locations,
            cell_locations=topology.cell_locations,
        )
        next_payload = build_result_render_payload(
            next_topology,
            reusable=current,
        )
        viewport.set_result_render_payload(next_payload)
        viewport.set_display("deformed", True)

    assert viewport._result_grid is rendered_grid
    assert viewport._actors["result"] is rendered_actor
    assert [
        options["name"]
        for _dataset, options in plotter.mesh_calls
    ] == ["result", "result_edges"]
    assert plotter.remove_actor_calls == 0
    assert plotter.render_count == 5
    assert viewport._result_point_index_to_node_id is node_index
    assert viewport._result_point_index_to_element_id is element_index
    assert viewport._result_cell_index_to_element_id is cell_index
    viewport.close()
