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


def build_result_render_payload(
    topology: ResultFieldTopology,
) -> ResultRenderPayload:
    """Build a PyVista grid without reinterpreting result-field semantics."""

    if type(topology) is not ResultFieldTopology:
        raise TypeError("topology must be exactly ResultFieldTopology")

    cells = np.fromiter(
        chain.from_iterable((len(cell), *cell) for cell in topology.cells),
        dtype=np.int64,
        count=sum(len(cell) + 1 for cell in topology.cells),
    )
    cell_types = np.asarray(
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
]
