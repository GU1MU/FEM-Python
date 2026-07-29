"""Map neutral application result topology onto an owned PyVista dataset."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
import pyvista

from fem.application.results import (
    ResultCellKind,
    ResultFieldTopology,
    ResultValueLayout,
)
from fem.post.vtk.cells import vtk_cell_type


RESULT_SCALAR_NAME = "result_scalar"
_VTK_VERTEX = 1


@dataclass(frozen=True, slots=True, eq=False)
class ResultRenderPayload:
    """One exact topology together with its detached PyVista representation."""

    topology: ResultFieldTopology
    dataset: pyvista.UnstructuredGrid
    scalar_name: str = RESULT_SCALAR_NAME

    def __post_init__(self) -> None:
        if type(self.topology) is not ResultFieldTopology:
            raise TypeError("topology must be exactly ResultFieldTopology")
        if not isinstance(self.dataset, pyvista.UnstructuredGrid):
            raise TypeError("dataset must be a PyVista UnstructuredGrid")
        if type(self.scalar_name) is not str:
            raise TypeError("scalar_name must be a string")
        if not self.scalar_name.strip():
            raise ValueError("scalar_name must not be blank")
        validate_result_render_payload(self)


def build_result_render_payload(
    topology: ResultFieldTopology,
) -> ResultRenderPayload:
    """Build a PyVista grid without reinterpreting result-field semantics."""

    if type(topology) is not ResultFieldTopology:
        raise TypeError("topology must be exactly ResultFieldTopology")

    cells = _flat_cell_array(topology)
    cell_types = _cell_type_array(topology)
    grid = pyvista.UnstructuredGrid(
        cells,
        cell_types,
        topology.points,
        deep=True,
    )
    values = topology.values
    target = (
        grid.point_data
        if topology.value_layout is ResultValueLayout.POINT
        else grid.cell_data
    )
    target.set_array(
        values,
        RESULT_SCALAR_NAME,
        deep_copy=True,
    )
    grid.set_active_scalars(
        RESULT_SCALAR_NAME,
        preference=(
            "point" if topology.value_layout is ResultValueLayout.POINT else "cell"
        ),
    )
    return ResultRenderPayload(
        topology=topology,
        dataset=grid,
    )


def validate_result_render_payload(
    payload: object,
) -> ResultRenderPayload:
    """Require an exact dataset representation at a consumer boundary.

    The PyVista dataset remains mutable, so callers repeat this check whenever
    they install or render a payload rather than treating construction as a
    permanent integrity guarantee.
    """

    if type(payload) is not ResultRenderPayload:
        raise TypeError("payload must be exactly ResultRenderPayload")
    topology = payload.topology
    dataset = payload.dataset
    if type(topology) is not ResultFieldTopology:
        raise TypeError("payload topology must be exactly ResultFieldTopology")
    if not isinstance(dataset, pyvista.UnstructuredGrid):
        raise TypeError("payload dataset must be a PyVista UnstructuredGrid")
    if int(dataset.n_points) != len(topology.point_locations):
        raise ValueError("payload point provenance must match dataset points")
    if int(dataset.n_cells) != len(topology.cell_locations):
        raise ValueError("payload cell provenance must match dataset cells")
    if not np.array_equal(np.asarray(dataset.points), topology.points):
        raise ValueError("payload dataset points must exactly match topology points")
    if not np.array_equal(
        np.asarray(dataset.cells, dtype=np.int64),
        _flat_cell_array(topology),
    ):
        raise ValueError(
            "payload dataset connectivity must exactly match topology cells"
        )
    if not np.array_equal(
        np.asarray(dataset.celltypes, dtype=np.uint8),
        _cell_type_array(topology),
    ):
        raise ValueError(
            "payload dataset cell types must exactly match topology cell types"
        )

    scalar_name = payload.scalar_name
    in_points = scalar_name in dataset.point_data
    in_cells = scalar_name in dataset.cell_data
    if topology.value_layout is ResultValueLayout.POINT:
        values = dataset.point_data[scalar_name] if in_points else None
        expected_values = int(dataset.n_points)
    elif topology.value_layout is ResultValueLayout.CELL:
        values = dataset.cell_data[scalar_name] if in_cells else None
        expected_values = int(dataset.n_cells)
    else:
        raise TypeError("payload value_layout must be exactly ResultValueLayout")
    if values is None or in_points == in_cells:
        raise ValueError(
            "payload scalar association must exactly match topology value_layout"
        )
    scalar_values = np.asarray(values)
    if scalar_values.shape != (expected_values,):
        raise ValueError("payload scalar must be one-dimensional")
    if not np.array_equal(scalar_values, topology.values):
        raise ValueError("payload scalar values must exactly match topology values")
    return payload


def reuse_result_render_dataset(
    current: ResultRenderPayload,
    candidate: ResultRenderPayload,
) -> tuple[ResultRenderPayload, bool]:
    """Rebind a component-only payload to the currently rendered dataset."""

    checked_current = validate_result_render_payload(current)
    checked_candidate = validate_result_render_payload(candidate)
    if not _has_reusable_geometry(checked_current, checked_candidate):
        return checked_candidate, False

    topology = checked_candidate.topology
    dataset = checked_current.dataset
    if checked_candidate.dataset is dataset:
        dataset.set_active_scalars(
            checked_candidate.scalar_name,
            preference=(
                "point"
                if topology.value_layout is ResultValueLayout.POINT
                else "cell"
            ),
        )
        return checked_candidate, True
    source_values = (
        checked_candidate.dataset.point_data[checked_candidate.scalar_name]
        if topology.value_layout is ResultValueLayout.POINT
        else checked_candidate.dataset.cell_data[checked_candidate.scalar_name]
    )
    target = (
        dataset.point_data
        if topology.value_layout is ResultValueLayout.POINT
        else dataset.cell_data
    )
    scalar_name = _component_scalar_name(checked_candidate)
    if scalar_name in target:
        if not np.array_equal(np.asarray(target[scalar_name]), source_values):
            return checked_candidate, False
    else:
        target.set_array(
            source_values,
            scalar_name,
            deep_copy=True,
        )
    dataset.set_active_scalars(
        scalar_name,
        preference=(
            "point"
            if topology.value_layout is ResultValueLayout.POINT
            else "cell"
        ),
    )
    return (
        ResultRenderPayload(
            topology=topology,
            dataset=dataset,
            scalar_name=scalar_name,
        ),
        True,
    )


def _has_reusable_geometry(
    current: ResultRenderPayload,
    candidate: ResultRenderPayload,
) -> bool:
    current_topology = current.topology
    candidate_topology = candidate.topology
    return (
        current_topology.source == candidate_topology.source
        and current_topology.materialization_generation
        == candidate_topology.materialization_generation
        and current_topology.selection.field_key
        == candidate_topology.selection.field_key
        and current_topology.deformation_scale
        == candidate_topology.deformation_scale
        and current_topology.value_layout is candidate_topology.value_layout
        and current_topology.cells == candidate_topology.cells
        and current_topology.cell_kinds == candidate_topology.cell_kinds
        and current_topology.canonical_element_types
        == candidate_topology.canonical_element_types
        and current_topology.point_locations
        == candidate_topology.point_locations
        and current_topology.cell_locations
        == candidate_topology.cell_locations
        and np.array_equal(
            np.asarray(current.dataset.points),
            np.asarray(candidate.dataset.points),
        )
        and np.array_equal(
            np.asarray(current.dataset.cells),
            np.asarray(candidate.dataset.cells),
        )
        and np.array_equal(
            np.asarray(current.dataset.celltypes),
            np.asarray(candidate.dataset.celltypes),
        )
    )


def _component_scalar_name(payload: ResultRenderPayload) -> str:
    component = payload.topology.selection.component
    return f"{RESULT_SCALAR_NAME}:{component}"


def _flat_cell_array(topology: ResultFieldTopology) -> np.ndarray:
    return np.fromiter(
        chain.from_iterable((len(cell), *cell) for cell in topology.cells),
        dtype=np.int64,
        count=sum(len(cell) + 1 for cell in topology.cells),
    )


def _cell_type_array(topology: ResultFieldTopology) -> np.ndarray:
    return np.asarray(
        tuple(
            _cell_type(kind, element_type)
            for kind, element_type in zip(
                topology.cell_kinds,
                topology.canonical_element_types,
                strict=True,
            )
        ),
        dtype=np.uint8,
    )


def _cell_type(
    kind: ResultCellKind,
    canonical_element_type: str | None,
) -> int:
    if kind is ResultCellKind.SAMPLE_VERTEX:
        return _VTK_VERTEX
    if kind is not ResultCellKind.FEM_ELEMENT:
        raise ValueError(f"unsupported result cell kind {kind!r}")
    if type(canonical_element_type) is not str:
        raise TypeError("FEM-element cells require a canonical element type")
    return vtk_cell_type(canonical_element_type)


__all__ = [
    "RESULT_SCALAR_NAME",
    "ResultRenderPayload",
    "build_result_render_payload",
    "reuse_result_render_dataset",
    "validate_result_render_payload",
]
