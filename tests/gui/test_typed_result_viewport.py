from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

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
    RESULT_SCALAR_NAME,
    build_result_render_payload,
)
from fem_gui.visualization.colormaps import abaqus_rainbow_colors
from fem_gui.visualization.contour_rendering import (
    CONTOUR_EDGE_ALL,
    CONTOUR_RENDER_FILLED,
)
from fem_gui.visualization.scene import DisplayState
from fem_gui.viewport_background import ViewportBackgroundSettings
import fem_gui.widgets.viewport as viewport_module
from fem_gui.widgets.viewport import FEMViewport, PickHit


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


def _node_location(
    node_id: int,
    coordinates: tuple[float, float, float],
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.NODE,
        coordinates=coordinates,
        displacement=(0.0, 0.0, 0.0),
        node_id=node_id,
    )


def _integration_point_location(
    element_id: int,
    integration_point: int,
    coordinates: tuple[float, float, float],
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.INTEGRATION_POINT,
        coordinates=coordinates,
        displacement=(0.0, 0.0, 0.0),
        element_id=element_id,
        integration_point=integration_point,
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


def _node_payload():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    topology = ResultFieldTopology(
        source=_source(),
        materialization_generation=6,
        selection=ScalarFieldSelection(
            FieldMaterializationKey(
                FieldRequest(
                    ResultFieldId(
                        ResultVariable.U,
                        FieldPosition.NODE,
                    )
                ),
                recovery_contract=3,
            ),
            "U1",
        ),
        deformation_scale=0.5,
        points=points,
        cells=((0, 1, 2),),
        cell_kinds=(ResultCellKind.FEM_ELEMENT,),
        canonical_element_types=("Tri3",),
        values=np.asarray((2.0, 3.0, 4.0)),
        value_layout=ResultValueLayout.POINT,
        point_locations=tuple(
            _node_location(node_id, tuple(point))
            for node_id, point in zip(
                (10, 20, 30),
                points,
                strict=True,
            )
        ),
        cell_locations=(None,),
    )
    return build_result_render_payload(topology)


def _duplicate_point_payload(
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
    ) + np.asarray(translation, dtype=float)
    locations = tuple(
        _element_node_location(
            node_id,
            element_id,
            local_node,
            tuple(points[index]),
        )
        for index, (node_id, element_id, local_node) in enumerate(
            (
                (10, 201, 1),
                (20, 201, 2),
                (30, 201, 3),
                (10, 202, 1),
                (30, 202, 2),
                (40, 202, 3),
            )
        )
    )
    topology = ResultFieldTopology(
        source=_source(),
        materialization_generation=7,
        selection=_selection(FieldPosition.ELEMENT_NODAL, "Mises"),
        deformation_scale=2.0,
        points=points,
        cells=((0, 1, 2), (3, 4, 5)),
        cell_kinds=(
            ResultCellKind.FEM_ELEMENT,
            ResultCellKind.FEM_ELEMENT,
        ),
        canonical_element_types=("Tri3", "Tri3"),
        values=np.arange(1.0, 7.0),
        value_layout=ResultValueLayout.POINT,
        point_locations=locations,
        cell_locations=(None, None),
    )
    return build_result_render_payload(topology)


def _integration_point_payload():
    points = np.asarray(
        (
            (0.15, 0.15, 0.0),
            (0.35, 0.15, 0.0),
            (0.35, 0.35, 0.0),
            (0.15, 0.35, 0.0),
            (1.25, 0.25, 0.0),
            (1.75, 0.25, 0.0),
        )
    )
    locations = tuple(
        _integration_point_location(
            element_id,
            integration_point,
            tuple(points[index]),
        )
        for index, (element_id, integration_point) in enumerate(
            (
                (401, 1),
                (401, 2),
                (401, 3),
                (401, 4),
                (402, 1),
                (402, 2),
            )
        )
    )
    topology = ResultFieldTopology(
        source=_source(),
        materialization_generation=8,
        selection=_selection(FieldPosition.INTEGRATION_POINT, "Mises"),
        deformation_scale=1.0,
        points=points,
        cells=tuple((index,) for index in range(len(points))),
        cell_kinds=(ResultCellKind.SAMPLE_VERTEX,) * len(points),
        canonical_element_types=(None,) * len(points),
        values=np.arange(1.0, 7.0),
        value_layout=ResultValueLayout.POINT,
        point_locations=locations,
        cell_locations=locations,
    )
    return build_result_render_payload(topology)


def _model_grid(
    *,
    points: np.ndarray,
    cells: tuple[tuple[int, ...], ...],
    cell_types: tuple[int, ...],
    node_ids: tuple[int, ...],
    element_ids: tuple[int, ...],
):
    cell_array = np.fromiter(
        (
            value
            for cell in cells
            for value in (len(cell), *cell)
        ),
        dtype=np.int64,
        count=sum(len(cell) + 1 for cell in cells),
    )
    grid = pyvista.UnstructuredGrid(
        cell_array,
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )
    grid.point_data["node_id"] = np.asarray(node_ids, dtype=np.int64)
    grid.cell_data["element_id"] = np.asarray(element_ids, dtype=np.int64)
    return grid


def _geometry(
    *,
    points: np.ndarray,
    cells: tuple[tuple[int, ...], ...],
    node_ids: tuple[int, ...],
    element_ids: tuple[int, ...],
):
    return SimpleNamespace(
        points=points,
        cells=cells,
        cell_types=tuple(3 if len(cell) == 2 else 5 for cell in cells),
        node_id_to_point_index={
            node_id: index for index, node_id in enumerate(node_ids)
        },
        point_index_to_node_id={
            index: node_id for index, node_id in enumerate(node_ids)
        },
        element_id_to_cell_index={
            element_id: index
            for index, element_id in enumerate(element_ids)
        },
        cell_index_to_element_id={
            index: element_id
            for index, element_id in enumerate(element_ids)
        },
    )


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
        self.label_option_calls: list[dict[str, object]] = []
        self.scalar_bars: dict[str, object] = {}
        self.render_count = 0
        self.background_calls: list[tuple[str, str | None]] = []
        self.hide_axes_count = 0
        self.axes_calls: list[dict[str, object]] = []

    def add_mesh(self, dataset, **kwargs):
        self.mesh_calls.append((dataset, kwargs))
        return _Actor()

    def add_point_labels(self, points, labels, **kwargs):
        self.label_calls.append((points, list(labels)))
        self.label_option_calls.append(kwargs)
        return _Actor()

    def remove_actor(self, _actor, **_kwargs) -> None:
        return None

    def set_background(self, color: str, *, top: str | None = None) -> None:
        self.background_calls.append((color, top))

    def hide_axes(self) -> None:
        self.hide_axes_count += 1

    def add_axes(self, **kwargs) -> None:
        self.axes_calls.append(kwargs)

    def _getPixelRatio(self) -> float:
        return 1.0

    def render(self) -> None:
        self.render_count += 1


def test_coordinate_system_option_updates_viewport_axes() -> None:
    _application()
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter

    viewport.set_contour_options(
        {"show_coordinate_system": False}
    )

    assert plotter.hide_axes_count == 1
    assert plotter.axes_calls == []

    viewport.set_contour_options(
        {"show_coordinate_system": True}
    )

    assert plotter.hide_axes_count == 2
    assert plotter.axes_calls == [
        {"color": viewport._background_settings.foreground_color}
    ]
    viewport.close()


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
    model_pick_grid = object()
    viewport._plotter = plotter
    viewport._pick_grid = model_pick_grid
    viewport._display = DisplayState("deformed", True)

    viewport.set_result_render_payload(payload)
    viewport._update_result_layer()

    rendered, options = plotter.mesh_calls[0]
    assert rendered is payload.dataset
    assert viewport._result_grid is payload.dataset
    assert viewport._pick_grid is model_pick_grid
    assert not hasattr(viewport, "_result_data")
    assert not hasattr(viewport, "_result_scalar")
    assert not hasattr(viewport, "_deformation_scale")
    assert options["scalars"] == RESULT_SCALAR_NAME
    assert options["cmap"] == abaqus_rainbow_colors(12)
    assert options["n_colors"] == 12
    assert not options["interpolate_before_map"]
    assert options["scalar_bar_args"]["vertical"]
    assert options["scalar_bar_args"]["n_labels"] == 7
    assert options["scalar_bar_args"]["title"] == (
        f"S, {payload.topology.selection.component}"
    )
    assert not options["scalar_bar_args"]["outline"]
    assert options["scalar_bar_args"]["width"] == 0.045
    assert options["scalar_bar_args"]["height"] == 0.62
    assert options["scalar_bar_args"]["position_x"] == 0.78
    assert options["scalar_bar_args"]["position_y"] == 0.19
    assert options["scalar_bar_args"]["title_font_size"] == 14
    assert options["scalar_bar_args"]["label_font_size"] == 14
    assert options["scalar_bar_args"]["font_family"] == "arial"
    assert options["scalar_bar_args"]["unconstrained_font_size"]
    assert options["lighting"]
    assert options["smooth_shading"]
    assert options["ambient"] == 0.35
    assert options["diffuse"] == 0.65
    assert options["specular"] == 0.0
    assert payload.topology.value_layout is expected_layout
    np.testing.assert_array_equal(payload.dataset.points, original_points)
    assert viewport.artifact_id == "artifact-1"
    assert viewport.run_id == "run-1"
    viewport.close()


def test_scientific_scalar_format_uses_compact_uppercase_exponent() -> None:
    _application()
    viewport = FEMViewport()

    assert viewport._contour["number_format"] == "scientific"
    assert viewport._contour["decimals"] == 2
    assert viewport._format_scalar(1.89e-4) == "1.89E-4"
    assert viewport._format_scalar(-2.5e3) == "-2.50E+3"
    assert viewport._format_scalar(0.0) == "0"

    viewport.close()


def test_engineering_scalar_format_and_legend_typography_are_configurable() -> None:
    _application()
    viewport = FEMViewport()
    viewport.set_contour_options(
        {
            "number_format": "engineering",
            "decimals": 2,
            "legend_font": "Times New Roman",
            "legend_font_size": 16,
        }
    )

    assert viewport._format_scalar(12345.0) == "12.35E+3"
    assert viewport._format_scalar(0.00125) == "1.25E-3"
    assert viewport._format_scalar(0.0) == "0"
    scalar_bar_args = viewport._contour_bar_args(_point_payload())
    assert scalar_bar_args["font_family"] == "times"
    assert scalar_bar_args["title_font_size"] == 16
    assert scalar_bar_args["label_font_size"] == 16

    viewport.close()


def test_filled_result_and_all_edge_modes_reach_pyvista() -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._display = DisplayState("deformed", True)
    viewport.set_contour_options(
        {
            "render_mode": CONTOUR_RENDER_FILLED,
            "edge_mode": CONTOUR_EDGE_ALL,
            "edges": True,
        }
    )

    viewport.set_result_render_payload(payload)
    viewport._update_result_layer()

    result_data, result_options = plotter.mesh_calls[0]
    edge_data, edge_options = plotter.mesh_calls[1]
    assert result_data is payload.dataset
    assert not result_options["lighting"]
    assert not result_options["smooth_shading"]
    assert edge_data.n_cells == 3
    assert edge_options["name"] == "result_edges"
    assert not edge_options["lighting"]
    assert not edge_options["show_scalar_bar"]
    assert not edge_options["pickable"]

    viewport.set_contour_options(
        {
            "edge_style": "bold",
            "edge_width": 2.5,
        }
    )
    assert plotter.mesh_calls[-1][1]["line_width"] == 5.0

    viewport.set_edges_visible(False, render=False)
    viewport._update_result_layer()
    assert "result_edges" not in viewport._actors
    assert plotter.mesh_calls[-1][1]["name"] == "result"

    viewport.set_edges_visible(True, render=False)
    assert "result_edges" in viewport._actors
    assert plotter.mesh_calls[-1][1]["name"] == "result_edges"
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


def test_typed_entry_rejects_mutated_dataset_geometry() -> None:
    _application()
    payload = _point_payload()
    payload.dataset.points[0, 0] = 0.25
    viewport = FEMViewport()

    with pytest.raises(ValueError, match="points"):
        viewport.set_result_render_payload(payload)

    assert viewport._result_render_payload is None
    viewport.close()


def test_typed_consumer_revalidates_vtk_reported_modification() -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    validated_mtime = viewport._result_render_validated_mtime
    payload.dataset.points[0, 0] = 0.25
    assert int(payload.dataset.GetMTime()) != validated_mtime

    with pytest.raises(ValueError, match="points"):
        viewport._resolve_pick(1, 2)

    viewport.close()


def test_typed_render_boundary_does_not_rely_on_runtime_mtime_cache() -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    payload.dataset.points[0, 0] = 0.25
    # Simulate an unreported change by making the runtime cache look current.
    viewport._result_render_validated_mtime = int(payload.dataset.GetMTime())

    with pytest.raises(ValueError, match="points"):
        viewport._update_result_layer()

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
    point_label_options = plotter.label_option_calls[-1]
    assert point_label_options["point_color"] == "#d69a3a"
    assert point_label_options["point_size"] == 14
    assert point_label_options["render_points_as_spheres"] is True
    assert point_label_options["font_size"] == 14
    assert point_label_options["shape"] is None
    assert point_label_options["text_color"] == "#000000"
    assert point_label_options["always_visible"] is True

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
    model_grid = _model_grid(
        points=np.asarray(payload.dataset.points),
        cells=((0, 1, 2),),
        cell_types=(5,),
        node_ids=(110, 120, 130),
        element_ids=(1201,),
    )
    payload.dataset.point_data["node_id"] = np.asarray((910, 920, 930))
    payload.dataset.cell_data["element_id"] = np.asarray((9201,))
    payload.dataset.set_active_scalars(
        RESULT_SCALAR_NAME,
        preference="point",
    )
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    viewport._pick_grid = model_grid
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
    assert node_hit.dataset_name == "typed_result_grid"
    assert element_hit.dataset_name == "typed_result_grid"
    assert viewport._result_point_index_to_node_id == {
        0: 10,
        1: 20,
        2: 30,
    }
    assert viewport._result_cell_index_to_element_id == {0: 201}
    assert RESULT_SCALAR_NAME in payload.dataset.point_data
    np.testing.assert_array_equal(
        payload.dataset.point_data["node_id"],
        (910, 920, 930),
    )
    np.testing.assert_array_equal(
        payload.dataset.cell_data["element_id"],
        (9201,),
    )
    viewport.close()


def test_typed_picking_falls_back_to_model_when_field_lacks_entity_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    viewport = FEMViewport()
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

    cell_payload = _cell_payload()
    viewport.set_result_render_payload(cell_payload)
    viewport._result_grid = cell_payload.dataset
    viewport._pick_grid = _model_grid(
        points=np.asarray(cell_payload.dataset.points),
        cells=((0, 1), (1, 2)),
        cell_types=(3, 3),
        node_ids=(10, 20, 30),
        element_ids=(401, 402),
    )
    viewport._selection_mode = "node"
    node_hit = viewport._resolve_pick(1, 2)

    node_payload = _node_payload()
    viewport.set_result_render_payload(node_payload)
    viewport._result_grid = node_payload.dataset
    viewport._pick_grid = _model_grid(
        points=np.asarray(node_payload.dataset.points),
        cells=((0, 1, 2),),
        cell_types=(5,),
        node_ids=(10, 20, 30),
        element_ids=(201,),
    )
    monkeypatch.setattr(
        viewport,
        "_intersect_dataset",
        lambda *_args: (0, np.asarray((0.25, 0.25, 0.0))),
    )
    viewport._selection_mode = "element"
    element_hit = viewport._resolve_pick(1, 2)

    assert node_hit is not None
    assert node_hit.pick_id == 10
    assert node_hit.dataset_name == "model_pick_grid"
    assert element_hit is not None
    assert element_hit.pick_id == 201
    assert element_hit.dataset_name == "model_pick_grid"
    viewport.close()


def test_typed_highlight_uses_all_matches_or_falls_back_for_partial_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    monkeypatch.setattr(viewport_module, "_pyvista", pyvista)
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter

    node_payload = _duplicate_point_payload()
    model_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        )
    )
    model_cells = ((0, 1, 2), (0, 2, 3))
    viewport._geometry = _geometry(
        points=model_points,
        cells=model_cells,
        node_ids=(10, 20, 30, 40, 50),
        element_ids=(201, 202),
    )
    viewport._pick_grid = _model_grid(
        points=model_points,
        cells=model_cells,
        cell_types=(5, 5),
        node_ids=(10, 20, 30, 40, 50),
        element_ids=(201, 202),
    )
    viewport.set_result_render_payload(node_payload)
    viewport._update_result_layer()

    viewport.highlight_node(10)
    assert plotter.mesh_calls[-1][0].n_points == 2
    viewport._show_preselection(
        PickHit(
            "node",
            10,
            "typed_result_grid",
            (1.0, 2.0),
            (0.0, 0.0, 0.0),
        )
    )
    assert plotter.mesh_calls[-1][0].n_points == 2
    viewport.highlight_nodes((10, 30))
    assert plotter.mesh_calls[-1][0].n_points == 4
    viewport.highlight_nodes((10, 50))
    partial_node_highlight = plotter.mesh_calls[-1][0]
    np.testing.assert_array_equal(
        partial_node_highlight.points,
        model_points[[0, 4]],
    )

    element_payload = _integration_point_payload()
    element_model_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        )
    )
    element_model_cells = ((0, 1), (1, 2), (2, 3))
    viewport._geometry = _geometry(
        points=element_model_points,
        cells=element_model_cells,
        node_ids=(10, 20, 30, 40),
        element_ids=(401, 402, 403),
    )
    viewport._pick_grid = _model_grid(
        points=element_model_points,
        cells=element_model_cells,
        cell_types=(3, 3, 3),
        node_ids=(10, 20, 30, 40),
        element_ids=(401, 402, 403),
    )
    viewport.set_result_render_payload(element_payload)
    viewport._update_result_layer()

    monkeypatch.setattr(
        viewport,
        "_world_points_to_display",
        lambda _points: np.asarray(
            (
                (1.0, 2.0, 0.5),
                (100.0, 100.0, 0.5),
                (200.0, 200.0, 0.5),
                (300.0, 300.0, 0.5),
                (400.0, 400.0, 0.5),
                (500.0, 500.0, 0.5),
            )
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_display_candidate_is_visible",
        lambda *_args: True,
    )
    viewport._selection_mode = "element"
    element_hit = viewport._resolve_pick(1, 2)
    assert element_hit is not None
    assert element_hit.pick_id == 401
    assert element_hit.dataset_name == "typed_result_grid"

    viewport.highlight_element(401)
    assert plotter.mesh_calls[-1][0].n_cells == 4
    viewport._show_preselection(element_hit)
    assert plotter.mesh_calls[-1][0].n_cells == 4
    viewport.highlight_elements((401, 402))
    assert plotter.mesh_calls[-1][0].n_cells == 6
    viewport.highlight_elements((401, 403))
    partial_element_highlight = plotter.mesh_calls[-1][0]
    np.testing.assert_array_equal(
        partial_element_highlight.cell_data["element_id"],
        (401, 403),
    )
    viewport.close()


def test_typed_payload_refresh_clears_batch_and_restores_persistent_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    monkeypatch.setattr(viewport_module, "_pyvista", pyvista)
    model_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        )
    )
    model_cells = ((0, 1, 2), (0, 2, 3))
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._geometry = _geometry(
        points=model_points,
        cells=model_cells,
        node_ids=(10, 20, 30, 40),
        element_ids=(201, 202),
    )
    viewport._pick_grid = _model_grid(
        points=model_points,
        cells=model_cells,
        cell_types=(5, 5),
        node_ids=(10, 20, 30, 40),
        element_ids=(201, 202),
    )
    viewport._display = DisplayState("deformed", True)
    viewport.set_result_render_payload(_duplicate_point_payload())
    viewport._update_result_layer()
    viewport.highlight_node(10)
    viewport.highlight_nodes((10, 30))
    assert "set_highlight" in viewport._actors

    translated = _duplicate_point_payload((3.0, 4.0, 0.0))
    viewport.set_result_render_payload(translated)
    viewport._update_result_layer()

    assert "set_highlight" not in viewport._actors
    assert "selection" in viewport._actors
    assert viewport._selected_kind == "node"
    assert viewport._selected_id == 10
    np.testing.assert_array_equal(
        plotter.mesh_calls[-1][0].points,
        np.asarray(translated.dataset.points)[[0, 3]],
    )

    viewport.highlight_nodes((10, 30))
    assert "set_highlight" in viewport._actors
    viewport.set_display("deformed", False)

    assert "set_highlight" not in viewport._actors
    assert "selection" in viewport._actors
    np.testing.assert_array_equal(
        plotter.mesh_calls[-1][0].points,
        np.asarray(translated.dataset.points)[[0, 3]],
    )
    viewport.close()


def test_typed_sample_result_keeps_model_grid_for_labels_and_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    monkeypatch.setattr(viewport_module, "_pyvista", pyvista)
    payload = _integration_point_payload()
    model_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    model_cells = ((0, 1), (1, 2))
    model_grid = _model_grid(
        points=model_points,
        cells=model_cells,
        cell_types=(3, 3),
        node_ids=(10, 20, 30),
        element_ids=(401, 402),
    )
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    viewport._geometry = _geometry(
        points=model_points,
        cells=model_cells,
        node_ids=(10, 20, 30),
        element_ids=(401, 402),
    )
    viewport._grid = model_grid
    viewport._pick_grid = model_grid
    viewport.set_result_render_payload(payload)
    viewport._update_result_layer()
    viewport.set_element_labels_visible(True)
    viewport.set_background_settings(
        ViewportBackgroundSettings(
            style="solid",
            bottom_color="#202020",
            top_color="#202020",
        )
    )

    assert viewport._pick_grid is model_grid
    label_points, labels = plotter.label_calls[-1]
    assert len(label_points) == 2
    assert labels == ["401", "402"]
    assert plotter.background_calls[-1] == ("#202020", None)
    viewport.close()


def test_background_refresh_reuses_installed_typed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _application()
    payload = _point_payload()
    viewport = FEMViewport()
    viewport.set_result_render_payload(payload)
    viewport._result_grid = payload.dataset
    viewport._display = DisplayState("deformed", True)
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
    module_source = path.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    typed_names = {
        "_require_result_render_payload",
        "set_result_render_payload",
        "_index_result_render_provenance",
        "_rendered_result_payload",
        "_provenance_ids",
        "_typed_result_point_ids",
        "_typed_result_point_element_ids",
        "_typed_result_cell_ids",
        "_typed_result_node_points",
        "_typed_result_element_cells",
        "_update_result_render_payload_layer",
        "_add_result_render_payload_extrema_labels",
        "_result_location_identity",
        "_resolve_pick",
        "_pick_screen_point",
        "_pick_cell",
        "_show_preselection",
    }
    segments = {
        node.name: ast.get_source_segment(
            module_source,
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
    assert "_pick_grid = dataset" not in source
    for forbidden in (
        "ResultData",
        "ScalarField",
        "deformed_points",
        "stress_adapter",
        "fem.post.stress",
        "build_stress_render_geometry",
        "averaging_threshold",
        "_result_data",
        "_result_scalar",
        "_deformation_scale",
        "set_result_data",
        "set_deformation_scale",
        "_add_extrema_labels",
    ):
        assert forbidden not in module_source
