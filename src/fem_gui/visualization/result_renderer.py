"""Map neutral application result topology onto an owned PyVista dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import pyvista

from fem.application.results import (
    ResultCellKind,
    ResultFieldTopology,
    ResultValueLayout,
    field_materialization_sort_key,
)
from fem.post.vtk.cells import vtk_cell_type


RESULT_SCALAR_NAME = "result_scalar"
_VTK_VERTEX = 1


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResultRenderPayload:
    """One exact topology together with its detached PyVista representation."""

    topology: ResultFieldTopology
    dataset: pyvista.UnstructuredGrid
    scalar_name: str = RESULT_SCALAR_NAME
    _cell_array: np.ndarray = field(repr=False)
    _cell_types: np.ndarray = field(repr=False)

    def __init__(
        self,
        topology: ResultFieldTopology,
        dataset: pyvista.UnstructuredGrid,
        scalar_name: str = RESULT_SCALAR_NAME,
    ) -> None:
        if type(topology) is not ResultFieldTopology:
            raise TypeError("topology must be exactly ResultFieldTopology")
        self._initialize(
            topology,
            dataset,
            scalar_name,
            _flat_cell_array(topology),
            _cell_type_array(topology),
        )
        validate_result_render_payload(self)

    @classmethod
    def _from_renderer_arrays(
        cls,
        topology: ResultFieldTopology,
        dataset: pyvista.UnstructuredGrid,
        scalar_name: str,
        cell_array: np.ndarray,
        cell_types: np.ndarray,
    ) -> "ResultRenderPayload":
        """Transfer renderer-owned immutable topology arrays without rescanning."""

        payload = object.__new__(cls)
        payload._initialize(
            topology,
            dataset,
            scalar_name,
            cell_array,
            cell_types,
        )
        return payload

    def _initialize(
        self,
        topology: ResultFieldTopology,
        dataset: pyvista.UnstructuredGrid,
        scalar_name: str,
        cell_array: np.ndarray,
        cell_types: np.ndarray,
    ) -> None:
        if type(topology) is not ResultFieldTopology:
            raise TypeError("topology must be exactly ResultFieldTopology")
        if not isinstance(dataset, pyvista.UnstructuredGrid):
            raise TypeError("dataset must be a PyVista UnstructuredGrid")
        if type(scalar_name) is not str:
            raise TypeError("scalar_name must be a string")
        if not scalar_name.strip():
            raise ValueError("scalar_name must not be blank")
        if type(cell_array) is not np.ndarray or cell_array.dtype != np.int64:
            raise TypeError("cell_array must be an int64 numpy.ndarray")
        if type(cell_types) is not np.ndarray or cell_types.dtype != np.uint8:
            raise TypeError("cell_types must be a uint8 numpy.ndarray")
        cell_array.setflags(write=False)
        cell_types.setflags(write=False)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "scalar_name", scalar_name)
        object.__setattr__(self, "_cell_array", cell_array)
        object.__setattr__(self, "_cell_types", cell_types)


def build_result_render_payload(
    topology: ResultFieldTopology,
    *,
    reusable: ResultRenderPayload | None = None,
) -> ResultRenderPayload:
    """Build a PyVista grid without reinterpreting result-field semantics."""

    if type(topology) is not ResultFieldTopology:
        raise TypeError("topology must be exactly ResultFieldTopology")
    if reusable is not None:
        checked_reusable = validate_result_render_payload(reusable)
        if _has_reusable_topology(checked_reusable, topology):
            rebound = _payload_on_reused_dataset(
                checked_reusable,
                topology,
                topology.values,
            )
            if rebound is not None:
                return rebound

    cells = _flat_cell_array(topology)
    cell_types = _cell_type_array(topology)
    grid = pyvista.UnstructuredGrid(
        cells,
        cell_types,
        topology._points,
        deep=True,
    )
    values = topology._values
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
    return ResultRenderPayload._from_renderer_arrays(
        topology=topology,
        dataset=grid,
        scalar_name=RESULT_SCALAR_NAME,
        cell_array=cells,
        cell_types=cell_types,
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
    if not np.array_equal(np.asarray(dataset.points), topology._points):
        raise ValueError("payload dataset points must exactly match topology points")
    if not np.array_equal(
        np.asarray(dataset.cells, dtype=np.int64),
        payload._cell_array,
    ):
        raise ValueError(
            "payload dataset connectivity must exactly match topology cells"
        )
    if not np.array_equal(
        np.asarray(dataset.celltypes, dtype=np.uint8),
        payload._cell_types,
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
    if not np.array_equal(scalar_values, topology._values):
        raise ValueError("payload scalar values must exactly match topology values")
    return payload


def reuse_result_render_dataset(
    current: ResultRenderPayload,
    candidate: ResultRenderPayload,
    *,
    current_validated: bool = False,
    candidate_validated: bool = False,
) -> tuple[ResultRenderPayload, bool]:
    """Rebind a layout-compatible payload to the rendered VTK dataset."""

    checked_current = (
        current
        if current_validated
        else validate_result_render_payload(current)
    )
    if type(checked_current) is not ResultRenderPayload:
        raise TypeError("current must be exactly ResultRenderPayload")
    checked_candidate = (
        candidate
        if candidate_validated
        else validate_result_render_payload(candidate)
    )
    if type(checked_candidate) is not ResultRenderPayload:
        raise TypeError("candidate must be exactly ResultRenderPayload")
    geometry_matches = _has_reusable_geometry(
        checked_current,
        checked_candidate,
    )
    if not geometry_matches and not _has_reusable_layout(
        checked_current,
        checked_candidate.topology,
    ):
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
    if not geometry_matches:
        dataset.points = np.asarray(
            checked_candidate.dataset.points,
            dtype=float,
        ).copy()
    source_values = (
        checked_candidate.dataset.point_data[checked_candidate.scalar_name]
        if topology.value_layout is ResultValueLayout.POINT
        else checked_candidate.dataset.cell_data[checked_candidate.scalar_name]
    )
    rebound = _payload_on_reused_dataset(
        checked_current,
        topology,
        source_values,
    )
    return (
        (checked_candidate, False)
        if rebound is None
        else (rebound, True)
    )


def _has_reusable_geometry(
    current: ResultRenderPayload,
    candidate: ResultRenderPayload,
) -> bool:
    return _has_reusable_topology(current, candidate.topology)


def _has_reusable_topology(
    current: ResultRenderPayload,
    candidate_topology: ResultFieldTopology,
) -> bool:
    current_topology = current.topology
    return (
        _has_reusable_layout(current, candidate_topology)
        and current_topology.deformation_scale
        == candidate_topology.deformation_scale
        and (
            current_topology._points is candidate_topology._points
            or np.array_equal(
                current_topology._points,
                candidate_topology._points,
            )
        )
    )


def _has_reusable_layout(
    current: ResultRenderPayload,
    candidate_topology: ResultFieldTopology,
) -> bool:
    current_topology = current.topology
    return (
        current_topology.source == candidate_topology.source
        and current_topology.materialization_generation
        == candidate_topology.materialization_generation
        and current_topology.value_layout is candidate_topology.value_layout
        and current_topology.cells == candidate_topology.cells
        and current_topology.cell_kinds == candidate_topology.cell_kinds
        and current_topology.canonical_element_types
        == candidate_topology.canonical_element_types
        and current_topology.point_locations
        == candidate_topology.point_locations
        and current_topology.cell_locations
        == candidate_topology.cell_locations
    )


def _payload_on_reused_dataset(
    current: ResultRenderPayload,
    topology: ResultFieldTopology,
    values: np.ndarray,
) -> ResultRenderPayload | None:
    dataset = current.dataset
    target = (
        dataset.point_data
        if topology.value_layout is ResultValueLayout.POINT
        else dataset.cell_data
    )
    scalar_name = _component_scalar_name(topology)
    if scalar_name in target:
        if not np.array_equal(np.asarray(target[scalar_name]), values):
            return None
    else:
        target.set_array(
            values,
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
    return ResultRenderPayload._from_renderer_arrays(
        topology=topology,
        dataset=dataset,
        scalar_name=scalar_name,
        cell_array=current._cell_array,
        cell_types=current._cell_types,
    )


def _component_scalar_name(topology: ResultFieldTopology) -> str:
    component = topology.selection.component
    field_token = ":".join(
        str(value)
        for value in field_materialization_sort_key(topology.selection.field_key)
    )
    return f"{RESULT_SCALAR_NAME}:{field_token}:{component}"


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
