from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
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
    RESULT_SCALAR_NAME,
    build_result_render_payload,
)
from fem_gui.visualization.scene import DisplayState
from fem_gui.widgets.viewport import FEMViewport


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=2,
        step_name="Step-1",
        run_id="run-1",
    )


def _selection(position: FieldPosition, component: str) -> ScalarFieldSelection:
    return ScalarFieldSelection(
        FieldMaterializationKey(
            FieldRequest(ResultFieldId(ResultVariable.S, position)),
            recovery_contract=3,
        ),
        component,
    )


def _element_node_location(
    node_id: int,
    element_id: int,
    local_node: int,
    coordinates: tuple[float, float, float],
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.ELEMENT_NODE,
        coordinates=coordinates,
        displacement=(0.0, 0.0, 0.0),
        node_id=node_id,
        element_id=element_id,
        local_node=local_node,
    )


def _element_location(
    element_id: int,
    coordinates: tuple[float, float, float],
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.ELEMENT,
        coordinates=coordinates,
        displacement=(0.0, 0.0, 0.0),
        element_id=element_id,
    )


def _point_payload():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    locations = tuple(
        _element_node_location(
            node_id,
            201,
            local_node,
            tuple(point),
        )
        for node_id, local_node, point in zip(
            (10, 20, 30),
            (1, 2, 3),
            points,
            strict=True,
        )
    )
    topology = ResultFieldTopology(
        source=_source(),
        materialization_generation=4,
        selection=_selection(FieldPosition.ELEMENT_NODAL, "Mises"),
        deformation_scale=2.5,
        points=points,
        cells=((0, 1, 2),),
        cell_kinds=(ResultCellKind.FEM_ELEMENT,),
        canonical_element_types=("Tri3",),
        values=np.asarray((4.0, 1.0, 9.0)),
        value_layout=ResultValueLayout.POINT,
        point_locations=locations,
        cell_locations=(None,),
    )
    return build_result_render_payload(topology)


def _cell_payload():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    topology = ResultFieldTopology(
        source=_source(),
        materialization_generation=5,
        selection=_selection(FieldPosition.CENTROID, "S11"),
        deformation_scale=1.25,
        points=points,
        cells=((0, 1), (1, 2)),
        cell_kinds=(
            ResultCellKind.FEM_ELEMENT,
            ResultCellKind.FEM_ELEMENT,
        ),
        canonical_element_types=("Truss2", "Truss2"),
        values=np.asarray((8.0, 3.0)),
        value_layout=ResultValueLayout.CELL,
        point_locations=(None, None, None),
        cell_locations=(
            _element_location(401, (0.5, 0.0, 0.0)),
            _element_location(402, (1.5, 0.0, 0.0)),
        ),
    )
    return build_result_render_payload(topology)


class _Actor:
    def __init__(self) -> None:
        self.visible = True
        self.pickable = None

    def SetVisibility(self, visible: bool) -> None:
        self.visible = bool(visible)

    def SetPickable(self, pickable: bool) -> None:
        self.pickable = bool(pickable)


class _Plotter:
    def __init__(self) -> None:
        self.mesh_calls: list[tuple[object, dict[str, object]]] = []
        self.label_calls: list[tuple[object, list[str]]] = []
        self.scalar_bars: dict[str, object] = {}
        self.render_count = 0

    def add_mesh(self, dataset, **kwargs):
        self.mesh_calls.append((dataset, kwargs))
        return _Actor()

    def add_point_labels(self, points, labels, **_kwargs):
        self.label_calls.append((points, list(labels)))
        return _Actor()

    def remove_actor(self, _actor, **_kwargs) -> None:
        return None

    def render(self) -> None:
        self.render_count += 1


@pytest.mark.parametrize(
    ("factory", "expected_layout"),
    (
        (_point_payload, ResultValueLayout.POINT),
        (_cell_payload, ResultValueLayout.CELL),
    ),
)
def test_typed_payload_renders_owned_dataset_without_reprojection(
    factory,
    expected_layout: ResultValueLayout,
) -> None:
    _application()
    payload = factory()
    original_points = np.asarray(payload.dataset.points).copy()
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._display = DisplayState("deformed", True, "ignored")
    viewport._deformation_scale = 99.0

    viewport.set_result_render_payload(payload)
    viewport._update_result_layer()

    rendered, options = plotter.mesh_calls[0]
    assert rendered is payload.dataset
    assert viewport._result_grid is payload.dataset
    assert viewport._pick_grid is payload.dataset
    assert viewport._result_scalar is None
    assert options["scalars"] == RESULT_SCALAR_NAME
    assert payload.topology.value_layout is expected_layout
    np.testing.assert_array_equal(payload.dataset.points, original_points)
    assert viewport.artifact_id == "artifact-1"
    assert viewport.run_id == "run-1"
    viewport.close()


def test_typed_entry_requires_exact_payload_and_scalar_association() -> None:
    _application()
    viewport = FEMViewport()
    with pytest.raises(TypeError, match="exactly ResultRenderPayload"):
        viewport.set_result_render_payload(object())  # type: ignore[arg-type]

    payload = _point_payload()
    values = np.asarray(payload.dataset.point_data[RESULT_SCALAR_NAME]).copy()
    del payload.dataset.point_data[RESULT_SCALAR_NAME]
    payload.dataset.cell_data[RESULT_SCALAR_NAME] = values[:1]
    payload.dataset.set_active_scalars(
        RESULT_SCALAR_NAME,
        preference="cell",
    )

    with pytest.raises(ValueError, match="association"):
        viewport.set_result_render_payload(payload)
    viewport.close()


def test_typed_entry_rejects_mismatched_artifact_without_mutation() -> None:
    _application()
    viewport = FEMViewport()
    viewport._artifact_id = "artifact-other"

    with pytest.raises(ValueError, match="artifact provenance"):
        viewport.set_result_render_payload(_point_payload())

    assert viewport._result_render_payload is None
    assert viewport.run_id is None
    viewport.close()


def test_typed_extrema_labels_use_point_and_cell_location_provenance() -> None:
    _application()
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._contour.update(
        show_minimum=True,
        show_maximum=True,
        show_ids=True,
    )

    point_payload = _point_payload()
    viewport._add_result_render_payload_extrema_labels(point_payload)
    point_labels = plotter.label_calls[-1][1]
    assert any(
        "节点 20" in label and "单元 201" in label and "局部节点 2" in label
        for label in point_labels
    )
    assert any("节点 30" in label for label in point_labels)

    cell_payload = _cell_payload()
    viewport._add_result_render_payload_extrema_labels(cell_payload)
    cell_labels = plotter.label_calls[-1][1]
    assert any("单元 402" in label for label in cell_labels)
    assert any("单元 401" in label for label in cell_labels)
    viewport.close()


def test_typed_picking_exposes_location_ids_through_existing_pick_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    viewport._pick_grid = payload.dataset
    monkeypatch.setattr(
        viewport,
        "_world_points_to_display",
        lambda _points: np.asarray(
            (
                (1.0, 2.0, 0.5),
                (100.0, 100.0, 0.5),
                (200.0, 200.0, 0.5),
            )
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_display_candidate_is_visible",
        lambda *_args: True,
    )
    viewport._selection_mode = "node"
    node_hit = viewport._resolve_pick(1, 2)
    monkeypatch.setattr(
        viewport,
        "_intersect_dataset",
        lambda *_args: (0, np.asarray((0.25, 0.25, 0.0))),
    )
    viewport._selection_mode = "element"
    element_hit = viewport._resolve_pick(1, 2)

    assert node_hit is not None and node_hit.pick_id == 10
    assert element_hit is not None and element_hit.pick_id == 201
    assert viewport._result_point_index_to_node_id == {
        0: 10,
        1: 20,
        2: 30,
    }
    assert viewport._result_cell_index_to_element_id == {0: 201}
    assert RESULT_SCALAR_NAME in payload.dataset.point_data
    assert "node_id" not in payload.dataset.point_data
    assert "element_id" not in payload.dataset.cell_data
    viewport.close()


def test_background_refresh_reuses_installed_typed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    viewport._display = DisplayState("deformed", True, "ignored")
    viewport._contour["show_maximum"] = True
    calls = []
    monkeypatch.setattr(
        viewport,
        "_add_result_render_payload_extrema_labels",
        calls.append,
    )

    viewport._refresh_extrema_for_background()

    assert calls == [payload]
    viewport.close()


def test_typed_viewport_path_has_no_engineering_or_materialization_calls() -> None:
    path = Path(__file__).parents[2] / "src" / "fem_gui" / "widgets" / "viewport.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    typed_names = {
        "_require_result_render_payload",
        "set_result_render_payload",
        "_index_result_render_provenance",
        "_update_result_render_payload_layer",
        "_add_result_render_payload_extrema_labels",
        "_result_location_identity",
        "_pick_screen_point",
        "_pick_cell",
    }
    segments = {
        node.name: ast.get_source_segment(
            path.read_text(encoding="utf-8"),
            node,
        )
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in typed_names
    }
    source = "\n".join(segments.values())

    assert set(segments) == typed_names
    for forbidden in (
        "ResultData",
        "deformed_points",
        "build_stress_render_geometry",
        "dispatch",
        "natural_shape_values",
        "project_scalar_field_topology",
        "build_result_render_payload",
        "provider",
        "query",
        "materialize",
        "_make_grid",
    ):
        assert forbidden not in source
