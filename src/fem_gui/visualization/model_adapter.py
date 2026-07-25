"""有限元网格到 VTK 拓扑的无界面适配。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np

from fem.post.vtk import cells as vtk_cells


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """PyVista 可直接消费的拓扑与真实编号映射。"""

    points: np.ndarray
    cells: tuple[tuple[int, ...], ...]
    cell_array: np.ndarray
    cell_types: np.ndarray
    node_id_to_point_index: dict[int, int]
    point_index_to_node_id: dict[int, int]
    element_id_to_cell_index: dict[int, int]
    cell_index_to_element_id: dict[int, int]
    artifact_id: str | None = None


def build_model_geometry(model: Any) -> ModelGeometry:
    """使用正式 VTK 单元映射构造 GUI 几何。"""
    mesh = model.mesh
    point_rows = [
        (float(node.x), float(node.y), float(getattr(node, "z", 0.0)))
        for node in mesh.nodes
    ]
    node_id_to_point_index = {
        int(node.id): index for index, node in enumerate(mesh.nodes)
    }
    if len(node_id_to_point_index) != len(mesh.nodes):
        raise ValueError("节点编号必须唯一")
    cells, cell_types, elements = vtk_cells.build(mesh)
    if len(cells) != len(mesh.elements):
        converted = {int(element.id) for element in elements}
        missing = [int(element.id) for element in mesh.elements if int(element.id) not in converted]
        raise ValueError(f"以下单元无法转换为 VTK：{missing}")
    element_id_to_cell_index = {
        int(element.id): index for index, element in enumerate(elements)
    }
    if len(element_id_to_cell_index) != len(elements):
        raise ValueError("单元编号必须唯一")
    connectivity = tuple(tuple(int(value) for value in cell[1:]) for cell in cells)
    flat_cells = np.fromiter(
        chain.from_iterable(cells),
        dtype=np.int64,
        count=sum(len(cell) for cell in cells),
    )
    return ModelGeometry(
        points=np.asarray(point_rows, dtype=float).reshape((-1, 3)),
        cells=connectivity,
        cell_array=flat_cells,
        cell_types=np.asarray(cell_types, dtype=np.uint8),
        node_id_to_point_index=node_id_to_point_index,
        point_index_to_node_id={index: node_id for node_id, index in node_id_to_point_index.items()},
        element_id_to_cell_index=element_id_to_cell_index,
        cell_index_to_element_id={index: element_id for element_id, index in element_id_to_cell_index.items()},
    )


def pyvista_cell_array(geometry: ModelGeometry) -> np.ndarray:
    """返回 PyVista legacy 单元数组。"""
    return geometry.cell_array
