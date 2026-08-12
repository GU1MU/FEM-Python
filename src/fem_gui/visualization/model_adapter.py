"""有限元网格到 VTK 拓扑的无界面适配。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from itertools import chain
from typing import Any

import numpy as np

from fem.application.results import ElementResultProfile, ResultArchiveModelProjection
from fem.core.model import (
    ElementSet,
    NodeSet,
)
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

    @property
    def is_line_mesh(self) -> bool:
        """Return whether every converted cell is a first-order VTK line."""

        return bool(len(self.cell_types)) and bool(
            np.all(np.asarray(self.cell_types) == 3)
        )


@dataclass(frozen=True, slots=True)
class ArchiveNodeView:
    """Immutable node facade used by result-only GUI consumers."""

    id: int
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class ArchiveElementView:
    """Immutable element facade used by result-only GUI consumers."""

    id: int
    node_ids: tuple[int, ...]
    type: str
    props: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ArchiveMeshView:
    """Read-only structural mesh facade; it is never a solver input."""

    nodes: tuple[ArchiveNodeView, ...]
    elements: tuple[ArchiveElementView, ...]
    dofs_per_node: int
    spatial_dimension: int

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def num_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(node.id for node in self.nodes)


@dataclass(frozen=True, slots=True)
class ArchiveModelView:
    """Read-only topology facade for result inspection and viewport display."""

    mesh: ArchiveMeshView
    name: str
    node_sets: Mapping[str, NodeSet]
    element_sets: Mapping[str, ElementSet]
    surfaces: Mapping[str, object]
    materials: Mapping[str, object]
    sections: tuple[object, ...]
    steps: tuple[object, ...]
    metadata: Mapping[str, object]
    edges: Mapping[str, object]


def _profile_spatial_dimension(profile: ElementResultProfile) -> int:
    return 2 if profile.family.value == "plane_continuum" else 3


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
        missing = [
            int(element.id)
            for element in mesh.elements
            if int(element.id) not in converted
        ]
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
        point_index_to_node_id={
            index: node_id for node_id, index in node_id_to_point_index.items()
        },
        element_id_to_cell_index=element_id_to_cell_index,
        cell_index_to_element_id={
            index: element_id for element_id, index in element_id_to_cell_index.items()
        },
    )


def pyvista_cell_array(geometry: ModelGeometry) -> np.ndarray:
    """返回 PyVista legacy 单元数组。"""
    return geometry.cell_array


def build_result_archive_geometry(
    projection: ResultArchiveModelProjection,
    *,
    artifact_id: str | None = None,
) -> ModelGeometry:
    """Build display topology directly from an immutable archive projection."""

    if type(projection) is not ResultArchiveModelProjection:
        raise TypeError("projection must be ResultArchiveModelProjection")
    topology = projection.topology
    coordinates = topology._node_coordinates
    node_ids = tuple(int(value) for value in topology.node_ids)
    node_id_to_point_index = {node_id: index for index, node_id in enumerate(node_ids)}
    cells: list[tuple[int, ...]] = []
    flat: list[int] = []
    cell_types: list[int] = []
    element_id_to_cell_index: dict[int, int] = {}
    for index, (element_id, element_type, connectivity) in enumerate(
        zip(
            topology.element_ids,
            topology.element_types,
            topology.connectivity,
            strict=True,
        )
    ):
        point_ids = tuple(node_id_to_point_index[int(value)] for value in connectivity)
        cells.append(point_ids)
        flat.extend((len(point_ids), *point_ids))
        cell_types.append(vtk_cells.vtk_cell_type(element_type))
        element_id_to_cell_index[int(element_id)] = index
    return ModelGeometry(
        points=coordinates,
        cells=tuple(cells),
        cell_array=np.asarray(flat, dtype=np.int64),
        cell_types=np.asarray(cell_types, dtype=np.uint8),
        node_id_to_point_index=node_id_to_point_index,
        point_index_to_node_id={
            index: node_id for node_id, index in node_id_to_point_index.items()
        },
        element_id_to_cell_index=element_id_to_cell_index,
        cell_index_to_element_id={
            index: element_id for element_id, index in element_id_to_cell_index.items()
        },
        artifact_id=artifact_id,
    )


def build_result_archive_model_view(
    projection: ResultArchiveModelProjection,
    profile: ElementResultProfile,
    *,
    name: str = "结果",
) -> ArchiveModelView:
    """Create an immutable structural view consumed by GUI widgets."""

    if type(projection) is not ResultArchiveModelProjection:
        raise TypeError("projection must be ResultArchiveModelProjection")
    if type(profile) is not ElementResultProfile:
        raise TypeError("profile must be ElementResultProfile")
    if profile.dofs_per_node is None:
        raise ValueError("result archive profile must define dofs_per_node")
    topology = projection.topology
    coordinates = topology._node_coordinates
    nodes = tuple(
        ArchiveNodeView(
            int(node_id),
            float(row[0]),
            float(row[1]),
            float(row[2]) if coordinates.shape[1] >= 3 else 0.0,
        )
        for node_id, row in zip(topology.node_ids, coordinates, strict=True)
    )
    elements = tuple(
        ArchiveElementView(int(element_id), tuple(connectivity), element_type)
        for element_id, connectivity, element_type in zip(
            topology.element_ids,
            topology.connectivity,
            topology.element_types,
            strict=True,
        )
    )
    mesh = ArchiveMeshView(
        nodes,
        elements,
        dofs_per_node=profile.dofs_per_node,
        spatial_dimension=_profile_spatial_dimension(profile),
    )
    node_sets = {
        region_name: NodeSet(region_name, values)
        for region_name, values in projection.named_region_node_ids.items()
    }
    element_sets = {
        region_name: ElementSet(region_name, values)
        for region_name, values in projection.named_region_element_ids.items()
    }
    return ArchiveModelView(
        mesh=mesh,
        name=str(name or "结果"),
        node_sets=MappingProxyType(node_sets),
        element_sets=MappingProxyType(element_sets),
        surfaces=MappingProxyType({}),
        materials=MappingProxyType({}),
        sections=(),
        steps=(),
        metadata=MappingProxyType({}),
        edges=MappingProxyType({}),
    )
