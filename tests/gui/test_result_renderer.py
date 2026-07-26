from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

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
    validate_result_render_payload,
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-1",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=2,
        step_name="Step-1",
        run_id="run-1",
    )


def _selection() -> ScalarFieldSelection:
    key = FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(
                ResultVariable.U,
                FieldPosition.NODE,
            )
        ),
        recovery_contract=3,
    )
    return ScalarFieldSelection(key, "U1")


def _location(
    node_id: int,
    point: tuple[float, float, float],
) -> FieldLocation:
    return FieldLocation(
        association=FieldAssociation.NODE,
        coordinates=point,
        displacement=(0.0, 0.0, 0.0),
        node_id=node_id,
    )


def _topology(
    *,
    points: np.ndarray,
    cells: tuple[tuple[int, ...], ...],
    cell_kinds: tuple[ResultCellKind, ...],
    canonical_element_types: tuple[str | None, ...],
    values: np.ndarray,
    value_layout: ResultValueLayout,
    point_locations: tuple[FieldLocation | None, ...],
    cell_locations: tuple[FieldLocation | None, ...],
) -> ResultFieldTopology:
    return ResultFieldTopology(
        source=_source(),
        materialization_generation=4,
        selection=_selection(),
        deformation_scale=1.5,
        points=points,
        cells=cells,
        cell_kinds=cell_kinds,
        canonical_element_types=canonical_element_types,
        values=values,
        value_layout=value_layout,
        point_locations=point_locations,
        cell_locations=cell_locations,
    )


def test_maps_mixed_fem_cells_and_point_scalars_without_reprojection() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        )
    )
    values = np.asarray((0.5, 1.5, 2.5, 3.5))
    locations = tuple(
        _location(10 + index, tuple(point)) for index, point in enumerate(points)
    )
    topology = _topology(
        points=points,
        cells=((0, 1, 2), (1, 3)),
        cell_kinds=(
            ResultCellKind.FEM_ELEMENT,
            ResultCellKind.FEM_ELEMENT,
        ),
        canonical_element_types=("Tri3", "Truss2"),
        values=values,
        value_layout=ResultValueLayout.POINT,
        point_locations=locations,
        cell_locations=(None, None),
    )

    payload = build_result_render_payload(topology)

    assert payload.topology is topology
    assert payload.scalar_name == RESULT_SCALAR_NAME
    assert payload.dataset.n_points == 4
    assert payload.dataset.n_cells == 2
    assert payload.dataset.celltypes.tolist() == [5, 3]
    assert payload.dataset.get_cell(0).point_ids == [0, 1, 2]
    assert payload.dataset.get_cell(1).point_ids == [1, 3]
    np.testing.assert_array_equal(
        payload.dataset.point_data[RESULT_SCALAR_NAME],
        values,
    )
    assert RESULT_SCALAR_NAME not in payload.dataset.cell_data
    assert payload.dataset.active_scalars_info.association.name == "POINT"
    assert payload.topology.point_locations == locations


def test_maps_sample_vertices_and_cell_scalars_exactly() -> None:
    points = np.asarray(((0.25, 0.25, 0.0), (0.75, 0.25, 0.0)))
    values = np.asarray((11.0, 22.0))
    topology = _topology(
        points=points,
        cells=((0,), (1,)),
        cell_kinds=(
            ResultCellKind.SAMPLE_VERTEX,
            ResultCellKind.SAMPLE_VERTEX,
        ),
        canonical_element_types=(None, None),
        values=values,
        value_layout=ResultValueLayout.CELL,
        point_locations=(None, None),
        cell_locations=(None, None),
    )

    payload = build_result_render_payload(topology)

    assert payload.dataset.celltypes.tolist() == [1, 1]
    np.testing.assert_array_equal(
        payload.dataset.cell_data[RESULT_SCALAR_NAME],
        values,
    )
    assert RESULT_SCALAR_NAME not in payload.dataset.point_data
    assert payload.dataset.active_scalars_info.association.name == "CELL"


def test_keeps_zero_cell_node_only_topology_renderable() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    topology = _topology(
        points=points,
        cells=(),
        cell_kinds=(),
        canonical_element_types=(),
        values=np.asarray((3.0, 4.0)),
        value_layout=ResultValueLayout.POINT,
        point_locations=(
            _location(1, (0.0, 0.0, 0.0)),
            _location(2, (1.0, 0.0, 0.0)),
        ),
        cell_locations=(),
    )

    payload = build_result_render_payload(topology)

    assert payload.dataset.n_points == 2
    assert payload.dataset.n_cells == 0
    np.testing.assert_array_equal(
        payload.dataset.point_data[RESULT_SCALAR_NAME],
        (3.0, 4.0),
    )


def test_dataset_owns_points_and_values_independently() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    values = np.asarray((3.0, 4.0))
    topology = _topology(
        points=points,
        cells=((0, 1),),
        cell_kinds=(ResultCellKind.FEM_ELEMENT,),
        canonical_element_types=("Truss2",),
        values=values,
        value_layout=ResultValueLayout.POINT,
        point_locations=(
            _location(1, (0.0, 0.0, 0.0)),
            _location(2, (1.0, 0.0, 0.0)),
        ),
        cell_locations=(None,),
    )

    payload = build_result_render_payload(topology)
    detached_points = topology.points
    detached_values = topology.values
    detached_points.setflags(write=True)
    detached_values.setflags(write=True)
    detached_points[0, 0] = 99.0
    detached_values[0] = 99.0

    np.testing.assert_array_equal(
        payload.dataset.points,
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    np.testing.assert_array_equal(
        payload.dataset.point_data[RESULT_SCALAR_NAME],
        (3.0, 4.0),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("points", "points"),
        ("connectivity", "connectivity"),
        ("cell_types", "cell types"),
        ("scalar", "scalar values"),
    ),
)
def test_validation_rejects_mutated_dataset_representation(
    mutation: str,
    message: str,
) -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    topology = _topology(
        points=points,
        cells=((0, 1, 2),),
        cell_kinds=(ResultCellKind.FEM_ELEMENT,),
        canonical_element_types=("Tri3",),
        values=np.asarray((3.0, 4.0, 5.0)),
        value_layout=ResultValueLayout.POINT,
        point_locations=tuple(
            _location(index + 1, tuple(point))
            for index, point in enumerate(points)
        ),
        cell_locations=(None,),
    )
    payload = build_result_render_payload(topology)

    if mutation == "points":
        payload.dataset.points[0, 0] = 0.25
    elif mutation == "connectivity":
        connectivity = payload.dataset.GetCells().GetConnectivityArray()
        connectivity.SetTuple1(1, 2)
        payload.dataset.Modified()
    elif mutation == "cell_types":
        payload.dataset.celltypes[0] = 3
        payload.dataset.Modified()
    else:
        payload.dataset.point_data[RESULT_SCALAR_NAME][0] = 99.0

    with pytest.raises(ValueError, match=message):
        validate_result_render_payload(payload)


@pytest.mark.parametrize("value", [None, object()])
def test_requires_exact_neutral_topology(value: object) -> None:
    with pytest.raises(TypeError, match="exactly ResultFieldTopology"):
        build_result_render_payload(value)  # type: ignore[arg-type]


def test_renderer_has_no_engineering_or_qt_dependencies() -> None:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "fem_gui"
        / "visualization"
        / "result_renderer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    source = path.read_text(encoding="utf-8")

    assert not any(name.startswith(("PySide6", "PyQt6")) for name in imports)
    assert "result_adapter" not in source
    assert "stress_adapter" not in source
    assert "fem.post.stress" not in source
    assert "fem.elements" not in source
    assert "ResultData" not in source
