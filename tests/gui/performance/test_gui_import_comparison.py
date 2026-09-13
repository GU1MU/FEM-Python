"""GUI import results remain equivalent to the reference implementation."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from fem.boundary.step import boundary_for_step
from fem.post.vtk import cells as vtk_cells
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import build_model_geometry, pyvista_cell_array
from tests.gui.performance.test_gui_import_performance import _plate_model


def _timed(function):
    started = perf_counter()
    value = function()
    return value, perf_counter() - started


def _legacy_geometry(model):
    point_rows = [
        (float(node.x), float(node.y), float(getattr(node, "z", 0.0)))
        for node in model.mesh.nodes
    ]
    cells, cell_types, _elements = vtk_cells.build(model.mesh)
    connectivity = tuple(tuple(int(value) for value in cell[1:]) for cell in cells)
    flattened = []
    for cell in connectivity:
        flattened.append(len(cell))
        flattened.extend(cell)
    return np.asarray(point_rows), np.asarray(flattened), np.asarray(cell_types)


def _legacy_boundary_scan(model):
    for edge in model.edges["TOP"].edges:
        node_lookup = {node.id: node for node in model.mesh.nodes}
        element_lookup = {element.id: element for element in model.mesh.elements}
        first = node_lookup[edge.node_ids[0]]
        last = node_lookup[edge.node_ids[-1]]
        tangent = np.array([last.x - first.x, last.y - first.y])
        normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        element = element_lookup[edge.elem_id]
        center = np.mean([[node_lookup[node_id].x, node_lookup[node_id].y]
                          for node_id in element.node_ids], axis=0)
        edge_center = np.array([(first.x + last.x) / 2, (first.y + last.y) / 2])
        if np.dot(normal, center - edge_center) < 0.0:
            normal = -normal


def _legacy_viewport_connectivity(geometry):
    values = []
    for cell in geometry.cells:
        values.append(len(cell))
        values.extend(cell)
    return np.asarray(values, dtype=np.int64)


def test_gui_import_matches_reference(record_property) -> None:
    for element_count, pressure in ((10, False), (1_000, False), (50_000, False), (50_000, True)):
        prefix = f"{element_count}_{'pressure' if pressure else 'unloaded'}"
        model = _plate_model(element_count, pressure)
        legacy, old_geometry = _timed(lambda: _legacy_geometry(model))
        geometry, new_geometry = _timed(lambda: build_model_geometry(model))
        service, new_inspection = _timed(lambda: InspectionService(model))
        assert service._element_record_cached.cache_info().currsize == 0
        details, eager_details = _timed(
            lambda: [service.element_record(element.id) for element in model.mesh.elements]
        )
        old_cells, old_refresh = _timed(lambda: _legacy_viewport_connectivity(geometry))
        new_cells, new_refresh = _timed(lambda: pyvista_cell_array(geometry))
        np.testing.assert_array_equal(old_cells, new_cells)
        np.testing.assert_array_equal(legacy[0], geometry.points)
        np.testing.assert_array_equal(legacy[1], new_cells)
        np.testing.assert_array_equal(legacy[2], geometry.cell_types)
        assert len(details) == element_count
        for stage, before, after in (
            ("geometry", old_geometry, new_geometry),
            ("viewport_connectivity", old_refresh, new_refresh),
            ("inspection", new_inspection + eager_details, new_inspection),
        ):
            record_property(f"{prefix}_{stage}_before_seconds", before)
            record_property(f"{prefix}_{stage}_after_seconds", after)
        if pressure:
            _, old_boundary = _timed(lambda: _legacy_boundary_scan(model))
            boundary, new_boundary = _timed(lambda: boundary_for_step(model, "load"))
            assert len(boundary.edge_tractions) == len(model.edges["TOP"].edges)
            record_property(f"{prefix}_pressure_boundary_before_seconds", old_boundary)
            record_property(f"{prefix}_pressure_boundary_after_seconds", new_boundary)
