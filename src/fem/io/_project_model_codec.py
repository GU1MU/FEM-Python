"""Strict JSON codec for a persisted native finite-element model artifact."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from fem.core.mesh import (
    Element2D,
    Element3D,
    Mesh2D,
    Mesh3D,
    Node2D,
    Node3D,
)
from fem.core.model import (
    Edge,
    ElementEdge,
    ElementFace,
    ElementSet,
    FEMModel,
    NodeSet,
    Surface,
    _unstamped_element_properties,
)


_SECTION_METADATA_KEYS = frozenset(
    {
        "_section_property_keys_by_element",
        "_section_original_properties_by_element",
        "_section_property_element_identity_by_element",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "dimension",
        "dofs_per_node",
        "name",
        "nodes",
        "elements",
        "node_sets",
        "element_sets",
        "surfaces",
        "edges",
        "metadata",
    }
)


def encode_project_model(model: Any, *, error_type: type[Exception]) -> dict[str, Any]:
    """Encode one native model as definition-neutral mesh state."""

    if type(model) is not FEMModel:
        raise error_type("snapshot.model 必须是 FEMModel")
    mesh = model.mesh
    if type(mesh) is Mesh2D:
        dimension = 2
        node_type = Node2D
        element_type = Element2D
    elif type(mesh) is Mesh3D:
        dimension = 3
        node_type = Node3D
        element_type = Element3D
    else:
        raise error_type("snapshot.model.mesh 必须是 Mesh2D 或 Mesh3D")

    nodes = tuple(mesh.nodes)
    elements = tuple(mesh.elements)
    if any(type(node) is not node_type for node in nodes):
        raise error_type("snapshot.model.mesh.nodes 包含不匹配的节点类型")
    if any(type(element) is not element_type for element in elements):
        raise error_type("snapshot.model.mesh.elements 包含不匹配的单元类型")

    metadata = {
        str(key): value
        for key, value in dict(model.metadata).items()
        if str(key) not in _SECTION_METADATA_KEYS
    }
    return {
        "dimension": dimension,
        "dofs_per_node": _positive_integer(
            mesh.dofs_per_node,
            "snapshot.model.mesh.dofs_per_node",
            error_type,
        ),
        "name": None if model.name is None else str(model.name),
        "nodes": [
            [
                _positive_integer(
                    node.id, f"snapshot.model.nodes[{index}].id", error_type
                ),
                *_node_coordinates(node, dimension, index, error_type),
            ]
            for index, node in enumerate(nodes)
        ],
        "elements": [
            {
                "id": _positive_integer(
                    element.id,
                    f"snapshot.model.elements[{index}].id",
                    error_type,
                ),
                "node_ids": [
                    _positive_integer(
                        node_id,
                        f"snapshot.model.elements[{index}].node_ids[{node_index}]",
                        error_type,
                    )
                    for node_index, node_id in enumerate(element.node_ids)
                ],
                "type": _nonempty_string(
                    element.type,
                    f"snapshot.model.elements[{index}].type",
                    error_type,
                ),
                "properties": _json_value(
                    _unstamped_element_properties(model, int(element.id), element),
                    f"snapshot.model.elements[{index}].properties",
                    error_type,
                ),
            }
            for index, element in enumerate(elements)
        ],
        "node_sets": [
            {
                "name": str(name),
                "node_ids": [int(value) for value in node_set.node_ids],
            }
            for name, node_set in sorted(model.node_sets.items())
        ],
        "element_sets": [
            {
                "name": str(name),
                "element_ids": [int(value) for value in element_set.element_ids],
            }
            for name, element_set in sorted(model.element_sets.items())
        ],
        "surfaces": [
            {
                "name": str(name),
                "faces": [
                    [
                        int(face.elem_id),
                        int(face.local_index),
                        [int(value) for value in face.node_ids],
                    ]
                    for face in surface.faces
                ],
            }
            for name, surface in sorted(model.surfaces.items())
        ],
        "edges": [
            {
                "name": str(name),
                "edges": [
                    [
                        int(edge.elem_id),
                        int(edge.local_index),
                        [int(value) for value in edge.node_ids],
                    ]
                    for edge in collection.edges
                ],
            }
            for name, collection in sorted(model.edges.items())
        ],
        "metadata": _json_value(
            metadata,
            "snapshot.model.metadata",
            error_type,
        ),
    }


def decode_project_model(value: Any, *, error_type: type[Exception]) -> FEMModel:
    """Decode and validate one detached native finite-element model."""

    data = _mapping(value, "$.project.model_artifact", error_type)
    _exact_keys(data, _MODEL_FIELDS, "$.project.model_artifact", error_type)
    dimension = _integer(
        data["dimension"], "$.project.model_artifact.dimension", error_type
    )
    if dimension not in {2, 3}:
        raise error_type("$.project.model_artifact.dimension 必须是 2 或 3")
    dofs_per_node = _positive_integer(
        data["dofs_per_node"],
        "$.project.model_artifact.dofs_per_node",
        error_type,
    )
    raw_name = data["name"]
    if raw_name is not None and type(raw_name) is not str:
        raise error_type("$.project.model_artifact.name 必须是 string 或 null")

    nodes = []
    node_ids: set[int] = set()
    for index, raw_node in enumerate(
        _array(data["nodes"], "$.project.model_artifact.nodes", error_type)
    ):
        path = f"$.project.model_artifact.nodes[{index}]"
        row = _array(raw_node, path, error_type)
        if len(row) != dimension + 1:
            raise error_type(f"{path} 坐标维数与模型不一致")
        node_id = _positive_integer(row[0], f"{path}[0]", error_type)
        if node_id in node_ids:
            raise error_type(f"{path}[0] 包含重复节点编号")
        node_ids.add(node_id)
        coordinates = tuple(
            _finite_number(item, f"{path}[{coordinate_index}]", error_type)
            for coordinate_index, item in enumerate(row[1:], start=1)
        )
        nodes.append(
            Node2D(node_id, *coordinates)
            if dimension == 2
            else Node3D(node_id, *coordinates)
        )

    elements = []
    element_ids: set[int] = set()
    for index, raw_element in enumerate(
        _array(data["elements"], "$.project.model_artifact.elements", error_type)
    ):
        path = f"$.project.model_artifact.elements[{index}]"
        element = _mapping(raw_element, path, error_type)
        _exact_keys(element, {"id", "node_ids", "type", "properties"}, path, error_type)
        element_id = _positive_integer(element["id"], f"{path}.id", error_type)
        if element_id in element_ids:
            raise error_type(f"{path}.id 包含重复单元编号")
        element_ids.add(element_id)
        connectivity = tuple(
            _positive_integer(item, f"{path}.node_ids[{node_index}]", error_type)
            for node_index, item in enumerate(
                _array(element["node_ids"], f"{path}.node_ids", error_type)
            )
        )
        if not connectivity or any(node_id not in node_ids for node_id in connectivity):
            raise error_type(f"{path}.node_ids 引用了不存在的节点")
        properties = _json_object(
            element["properties"], f"{path}.properties", error_type
        )
        element_class = Element2D if dimension == 2 else Element3D
        elements.append(
            element_class(
                element_id,
                list(connectivity),
                _nonempty_string(element["type"], f"{path}.type", error_type),
                properties,
            )
        )

    mesh = (
        Mesh2D(nodes, elements, dofs_per_node=dofs_per_node)
        if dimension == 2
        else Mesh3D(nodes, elements, dofs_per_node=dofs_per_node)
    )
    node_sets = _decode_sets(
        data["node_sets"],
        path="$.project.model_artifact.node_sets",
        ids_key="node_ids",
        available=node_ids,
        factory=NodeSet,
        error_type=error_type,
    )
    element_sets = _decode_sets(
        data["element_sets"],
        path="$.project.model_artifact.element_sets",
        ids_key="element_ids",
        available=element_ids,
        factory=ElementSet,
        error_type=error_type,
    )
    surfaces = _decode_boundaries(
        data["surfaces"],
        path="$.project.model_artifact.surfaces",
        collection_key="faces",
        available_elements=element_ids,
        available_nodes=node_ids,
        item_factory=ElementFace,
        collection_factory=Surface,
        error_type=error_type,
    )
    edges = _decode_boundaries(
        data["edges"],
        path="$.project.model_artifact.edges",
        collection_key="edges",
        available_elements=element_ids,
        available_nodes=node_ids,
        item_factory=ElementEdge,
        collection_factory=Edge,
        error_type=error_type,
    )
    return FEMModel(
        mesh=mesh,
        name=raw_name,
        node_sets=node_sets,
        element_sets=element_sets,
        surfaces=surfaces,
        materials={},
        sections=[],
        steps=[],
        metadata=_json_object(
            data["metadata"],
            "$.project.model_artifact.metadata",
            error_type,
        ),
        edges=edges,
    )


def _decode_sets(
    value: Any,
    *,
    path: str,
    ids_key: str,
    available: set[int],
    factory: Any,
    error_type: type[Exception],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, raw_set in enumerate(_array(value, path, error_type)):
        item_path = f"{path}[{index}]"
        data = _mapping(raw_set, item_path, error_type)
        _exact_keys(data, {"name", ids_key}, item_path, error_type)
        name = _nonempty_string(data["name"], f"{item_path}.name", error_type)
        if name in result:
            raise error_type(f"{item_path}.name 包含重复名称")
        identities = tuple(
            _positive_integer(item, f"{item_path}.{ids_key}[{item_index}]", error_type)
            for item_index, item in enumerate(
                _array(data[ids_key], f"{item_path}.{ids_key}", error_type)
            )
        )
        if any(identity not in available for identity in identities):
            raise error_type(f"{item_path}.{ids_key} 引用了不存在的实体")
        result[name] = factory(name, identities)
    return result


def _decode_boundaries(
    value: Any,
    *,
    path: str,
    collection_key: str,
    available_elements: set[int],
    available_nodes: set[int],
    item_factory: Any,
    collection_factory: Any,
    error_type: type[Exception],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, raw_collection in enumerate(_array(value, path, error_type)):
        item_path = f"{path}[{index}]"
        data = _mapping(raw_collection, item_path, error_type)
        _exact_keys(data, {"name", collection_key}, item_path, error_type)
        name = _nonempty_string(data["name"], f"{item_path}.name", error_type)
        if name in result:
            raise error_type(f"{item_path}.name 包含重复名称")
        entries = []
        for entry_index, raw_entry in enumerate(
            _array(data[collection_key], f"{item_path}.{collection_key}", error_type)
        ):
            entry_path = f"{item_path}.{collection_key}[{entry_index}]"
            row = _array(raw_entry, entry_path, error_type)
            if len(row) != 3:
                raise error_type(f"{entry_path} 必须包含单元、局部编号和节点")
            element_id = _positive_integer(row[0], f"{entry_path}[0]", error_type)
            local_index = _nonnegative_integer(
                row[1],
                f"{entry_path}[1]",
                error_type,
            )
            boundary_nodes = tuple(
                _positive_integer(item, f"{entry_path}[2][{node_index}]", error_type)
                for node_index, item in enumerate(
                    _array(row[2], f"{entry_path}[2]", error_type)
                )
            )
            if element_id not in available_elements or any(
                node_id not in available_nodes for node_id in boundary_nodes
            ):
                raise error_type(f"{entry_path} 引用了不存在的网格实体")
            entries.append(item_factory(element_id, local_index, boundary_nodes))
        result[name] = collection_factory(name, entries)
    return result


def _node_coordinates(
    node: Any,
    dimension: int,
    index: int,
    error_type: type[Exception],
) -> list[float]:
    names = ("x", "y") if dimension == 2 else ("x", "y", "z")
    return [
        _finite_number(
            getattr(node, name),
            f"snapshot.model.nodes[{index}].{name}",
            error_type,
        )
        for name in names
    ]


def _json_object(value: Any, path: str, error_type: type[Exception]) -> dict[str, Any]:
    if type(value) is not dict:
        raise error_type(f"{path} 必须是 JSON object")
    return _json_value(value, path, error_type)


def _json_value(value: Any, path: str, error_type: type[Exception]) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} 必须是有限数值")
        return float(value)
    if type(value) in {list, tuple}:
        return [
            _json_value(item, f"{path}[{index}]", error_type)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise error_type(f"{path} 的键必须是 string")
            result[key] = _json_value(item, f"{path}.{key}", error_type)
        return result
    raise error_type(f"{path} 的 {type(value).__name__} 无法由 JSON 无损表示")


def _mapping(value: Any, path: str, error_type: type[Exception]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} 必须是 object")
    return dict(value)


def _array(value: Any, path: str, error_type: type[Exception]) -> list[Any]:
    if not isinstance(value, list):
        raise error_type(f"{path} 必须是 array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    path: str,
    error_type: type[Exception],
) -> None:
    if set(value) != set(expected):
        raise error_type(f"{path} 字段不符合 schema")


def _integer(value: Any, path: str, error_type: type[Exception]) -> int:
    if type(value) is not int:
        raise error_type(f"{path} 必须是 integer")
    return int(value)


def _positive_integer(value: Any, path: str, error_type: type[Exception]) -> int:
    result = _integer(value, path, error_type)
    if result <= 0:
        raise error_type(f"{path} 必须大于零")
    return result


def _nonnegative_integer(value: Any, path: str, error_type: type[Exception]) -> int:
    result = _integer(value, path, error_type)
    if result < 0:
        raise error_type(f"{path} 必须大于等于零")
    return result


def _finite_number(value: Any, path: str, error_type: type[Exception]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{path} 必须是 number")
    result = float(value)
    if not math.isfinite(result):
        raise error_type(f"{path} 必须是有限数值")
    return result


def _nonempty_string(value: Any, path: str, error_type: type[Exception]) -> str:
    if type(value) is not str or not value.strip():
        raise error_type(f"{path} 必须是非空 string")
    return value


__all__ = ["decode_project_model", "encode_project_model"]
