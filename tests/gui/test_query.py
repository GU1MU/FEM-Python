from __future__ import annotations

import pytest

from fem.abaqus import read
from fem.solvers.static_linear import solve
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.query import (
    ELEMENT_STRESS,
    NODE_DISPLACEMENT,
    NODAL_STRESS,
    available_components,
    parse_object_ids,
    query_records,
)
from fem_gui.visualization.result_adapter import build_result_data


def _data(path):
    model = read(path)
    geometry = build_model_geometry(model)
    return geometry, build_result_data(solve(model), geometry)


def test_parse_object_ids_supports_lists_ranges_and_real_id_validation():
    assert parse_object_ids("1, 3-4", (1, 2, 3, 4)) == (1, 3, 4)
    assert parse_object_ids("4-3 1", (1, 2, 3, 4)) == (4, 3, 1)
    with pytest.raises(ValueError, match="不存在"):
        parse_object_ids("5", (1, 2, 3, 4))


def test_query_records_use_fem_ids_and_real_components(gui_inp_path):
    geometry, data = _data(gui_inp_path)
    node_id = next(iter(geometry.node_id_to_point_index))
    element_id = next(iter(geometry.element_id_to_cell_index))

    displacement = query_records(data, NODE_DISPLACEMENT, (node_id,))
    assert displacement[0].object_id == node_id
    assert "U" in displacement[0].values
    assert "U3" not in displacement[0].values

    stress = query_records(data, ELEMENT_STRESS, (element_id,))
    assert stress[0].object_id == element_id
    assert "Mises" in stress[0].values
    assert "Mises" in available_components(data, NODAL_STRESS)
