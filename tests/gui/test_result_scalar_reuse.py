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
