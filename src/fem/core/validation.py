from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .model import model_element_info


_MESH_ATTRIBUTES = (
    "nodes",
    "elements",
    "dofs_per_node",
    "dof_map",
    "num_dofs",
    "element_dofs",
)
_MODEL_ATTRIBUTES = (
    "mesh",
    "node_sets",
    "element_sets",
    "materials",
    "sections",
    "surfaces",
    "edges",
    "steps",
    "metadata",
)


def validate_mesh(mesh: Any) -> None:
    """Validate mesh identity, connectivity, coordinates, and DOF state."""
    missing = [name for name in _MESH_ATTRIBUTES if not hasattr(mesh, name)]
    if missing:
        raise TypeError(
            "mesh validation requires attributes "
            + ", ".join(_MESH_ATTRIBUTES)
            + f"; missing {', '.join(missing)}"
        )

    nodes = list(mesh.nodes)
    elements = list(mesh.elements)
    if not nodes:
        raise ValueError("mesh must contain at least one node")
    if not elements:
        raise ValueError("mesh must contain at least one element")
    node_ids = _unique_entity_ids(nodes, "node")
    _unique_entity_ids(elements, "element")
    _validate_coordinates(nodes)
    _validate_connectivity(elements, set(node_ids))
    _validate_dof_map(mesh, node_ids)


def validate_model_structure(model: Any) -> None:
    """Validate model-owned structure without inspecting Step references."""
    _require_model_attributes(model)
    validate_mesh(model.mesh)
    node_ids = {_entity_id(node, "node") for node in model.mesh.nodes}
    element_lookup = {
        _entity_id(element, "element"): element for element in model.mesh.elements
    }
    element_ids = set(element_lookup)

    node_sets = _mapping(model.node_sets, "model.node_sets")
    element_sets = _mapping(model.element_sets, "model.element_sets")
    internal_element_sets = _internal_element_sets(model)
    materials = _mapping(model.materials, "model.materials")

    for name, node_set in node_sets.items():
        _validate_set_members(
            getattr(node_set, "node_ids", None),
            node_ids,
            f"node set {name}",
            "node",
        )
    for name, element_set in element_sets.items():
        _validate_set_members(
            getattr(element_set, "element_ids", None),
            element_ids,
            f"element set {name}",
            "element",
        )
    for name, element_set in internal_element_sets.items():
        if name in element_sets:
            continue
        _validate_set_members(
            getattr(element_set, "element_ids", None),
            element_ids,
            f"internal element set {name}",
            "element",
        )

    all_element_set_names = set(element_sets) | set(internal_element_sets)
    for section in model.sections:
        if section.material not in materials:
            raise KeyError(f"material {section.material} is not defined")
        if section.element_set not in all_element_set_names:
            raise KeyError(f"element set {section.element_set} is not defined")

    _validate_boundaries(model.surfaces, element_lookup, node_ids, "surface", "faces")
    _validate_boundaries(model.edges, element_lookup, node_ids, "edge", "edges")
    _validate_unique_step_names(list(model.steps))


def validate_analysis_step(model: Any, step: Any | None) -> None:
    """Validate references and effective Initial boundaries for one Step."""
    if step is None:
        return
    _require_model_attributes(model)
    node_ids = {_entity_id(node, "node") for node in model.mesh.nodes}
    element_lookup = {
        _entity_id(element, "element"): element for element in model.mesh.elements
    }
    node_sets = _mapping(model.node_sets, "model.node_sets")
    element_sets = _mapping(model.element_sets, "model.element_sets")
    all_element_sets = dict(_internal_element_sets(model))
    all_element_sets.update(element_sets)
    _validate_step_references(
        step,
        node_ids,
        node_sets,
        element_lookup,
        all_element_sets,
        _mapping(model.surfaces, "model.surfaces"),
        _mapping(model.edges, "model.edges"),
        int(model.mesh.dofs_per_node),
        _model_spatial_dimension(element_lookup),
        model,
        effective_boundaries=_effective_step_boundaries(model, step),
    )


def validate_model(model: Any, step: Any | None = None) -> None:
    """Validate model structure and either all Steps or one selected Step."""
    validate_model_structure(model)
    steps = list(model.steps) if step is None else [step]
    for candidate in steps:
        validate_analysis_step(model, candidate)


def _require_model_attributes(model: Any) -> None:
    missing = [name for name in _MODEL_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "model validation requires attributes "
            + ", ".join(_MODEL_ATTRIBUTES)
            + f"; missing {', '.join(missing)}"
        )


def _unique_entity_ids(entities: Sequence[Any], kind: str) -> list[int]:
    """Return entity ids and reject duplicate or non-integral values."""
    ids = [_entity_id(entity, kind) for entity in entities]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{kind} ids must be unique")
    return ids


def _entity_id(entity: Any, kind: str) -> int:
    if not hasattr(entity, "id"):
        raise TypeError(f"{kind} is missing id")
    return _integer_id(entity.id, f"{kind} id")


def _integer_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer, got {value!r}") from exc


def _validate_coordinates(nodes: Sequence[Any]) -> None:
    """Require consistent 2D/3D nodes with finite coordinates."""
    z_flags = [hasattr(node, "z") for node in nodes]
    if z_flags and any(z_flags) and not all(z_flags):
        raise ValueError("mesh nodes must use one consistent coordinate dimension")

    coordinate_names = ("x", "y", "z") if z_flags and all(z_flags) else ("x", "y")
    for node in nodes:
        node_id = _entity_id(node, "node")
        for name in coordinate_names:
            if not hasattr(node, name):
                raise TypeError(f"node {node_id} is missing coordinate {name}")
            raw_value = getattr(node, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(
                    f"node {node_id} coordinate {name} must be a real number, "
                    f"got {raw_value!r}"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"node {node_id} coordinate {name} must be finite, got {value}"
                )


def _validate_connectivity(elements: Sequence[Any], node_ids: set[int]) -> None:
    """Require every element connectivity id to reference a mesh node."""
    for element in elements:
        element_id = _entity_id(element, "element")
        if not hasattr(element, "node_ids"):
            raise TypeError(f"element {element_id} is missing node_ids")
        connectivity = [
            _integer_id(raw_node_id, f"element {element_id} node id")
            for raw_node_id in element.node_ids
        ]
        if len(set(connectivity)) != len(connectivity):
            raise ValueError(f"element {element_id} node_ids must be unique")
        for node_id in connectivity:
            if node_id not in node_ids:
                raise KeyError(
                    f"element {element_id} references missing node {node_id}"
                )


def _validate_dof_map(mesh: Any, node_ids: Sequence[int]) -> None:
    """Require the cached DOF map to match the mesh's current nodes."""
    dofs_per_node = _integer_id(mesh.dofs_per_node, "dofs_per_node")
    if dofs_per_node <= 0:
        raise ValueError("dofs_per_node must be positive")

    dof_map = mesh.dof_map
    expected_ids = sorted(node_ids)
    expected_lookup = {
        node_id: node_index for node_index, node_id in enumerate(expected_ids)
    }
    consistent = (
        int(getattr(dof_map, "dofs_per_node", -1)) == dofs_per_node
        and list(getattr(dof_map, "node_ids", ())) == expected_ids
        and dict(getattr(dof_map, "node_id_to_index", {})) == expected_lookup
        and int(mesh.num_dofs) == len(expected_ids) * dofs_per_node
    )
    if not consistent:
        raise ValueError(
            "mesh DofMap is inconsistent with current nodes; "
            "call mesh.rebuild_dof_map()"
        )


def _mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _internal_element_sets(model: Any) -> Mapping[Any, Any]:
    metadata = _mapping(model.metadata, "model.metadata")
    value = metadata.get("_abaqus_internal_element_sets", {})
    return _mapping(value, "model internal element sets")


def _validate_set_members(
    raw_ids: Any,
    valid_ids: set[int],
    label: str,
    member_kind: str,
) -> None:
    if raw_ids is None:
        raise TypeError(f"{label} is missing {member_kind}_ids")
    ids = [_integer_id(value, f"{label} {member_kind} id") for value in raw_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate {member_kind} ids")
    missing = [value for value in ids if value not in valid_ids]
    if missing:
        raise KeyError(f"{label} references missing {member_kind} {missing[0]}")


def _validate_boundaries(
    collections: Any,
    element_lookup: Mapping[int, Any],
    node_ids: set[int],
    collection_kind: str,
    entries_attribute: str,
) -> None:
    for name, collection in _mapping(
        collections, f"model.{collection_kind}s"
    ).items():
        entries = getattr(collection, entries_attribute, None)
        if entries is None:
            raise TypeError(
                f"{collection_kind} {name} is missing {entries_attribute}"
            )
        for entry in entries:
            element_id = _integer_id(
                getattr(entry, "elem_id", None),
                f"{collection_kind} {name} element id",
            )
            if element_id not in element_lookup:
                raise KeyError(
                    f"{collection_kind} {name} references missing element {element_id}"
                )
            local_index = _integer_id(
                getattr(entry, "local_index", None),
                f"{collection_kind} {name} local_index",
            )
            element = element_lookup[element_id]
            expected_node_ids = _kernel_boundary_node_ids(
                element,
                local_index,
                collection_kind,
            )
            element_node_ids = {
                _integer_id(value, f"element {element_id} node id")
                for value in element.node_ids
            }
            raw_entry_node_ids = getattr(entry, "node_ids", None)
            if raw_entry_node_ids is None:
                raise TypeError(f"{collection_kind} {name} is missing node_ids")
            entry_node_ids = tuple(
                _integer_id(raw_node_id, f"{collection_kind} {name} node id")
                for raw_node_id in raw_entry_node_ids
            )
            for node_id in entry_node_ids:
                if node_id not in node_ids:
                    raise KeyError(
                        f"{collection_kind} {name} references missing node {node_id}"
                    )
                if node_id not in element_node_ids:
                    raise ValueError(
                        f"{collection_kind} {name} node {node_id} is not connected "
                        f"to element {element_id}"
                    )
            if (
                expected_node_ids is not None
                and not _boundary_node_ids_are_equivalent(
                    entry_node_ids,
                    expected_node_ids,
                    collection_kind,
                )
            ):
                raise ValueError(
                    f"{collection_kind} {name} element {element_id} local_index "
                    f"{local_index} expects node_ids {expected_node_ids}, "
                    f"got {entry_node_ids}"
                )


def _kernel_boundary_node_ids(
    element: Any,
    local_index: int,
    collection_kind: str,
) -> tuple[int, ...] | None:
    """Return boundary node ids from the element kernel's local topology."""
    from ..elements import get_element_kernel

    kernel = get_element_kernel(element.type)
    if collection_kind == "surface":
        topology = getattr(kernel, "face_node_indices", None)
        local_kind = "face"
    elif collection_kind == "edge":
        topology = getattr(kernel, "edge_node_indices", None)
        local_kind = "edge"
    else:
        raise ValueError(f"unsupported boundary collection kind: {collection_kind}")

    if topology is None:
        return None
    if local_index < 0 or local_index >= len(topology):
        raise ValueError(
            f"{collection_kind} local_index {local_index} is invalid for element "
            f"{element.id} type {element.type}; expected 0 through {len(topology) - 1}"
        )

    try:
        return tuple(
            _integer_id(element.node_ids[node_index], f"element {element.id} node id")
            for node_index in topology[local_index]
        )
    except IndexError as exc:
        raise ValueError(
            f"element {element.id} type {element.type} connectivity is incompatible "
            f"with local {local_kind} topology"
        ) from exc


def _boundary_node_ids_are_equivalent(
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    collection_kind: str,
) -> bool:
    """Accept valid boundary orientations without changing midside roles."""
    if len(actual) != len(expected):
        return False
    if collection_kind == "edge":
        return actual == expected or actual == tuple(reversed(expected))
    if collection_kind != "surface":
        return False

    if len(expected) in (6, 8):
        corner_count = len(expected) // 2
        corners = expected[:corner_count]
        midsides = expected[corner_count:]
    else:
        corner_count = len(expected)
        corners = expected
        midsides = ()

    for direction in (1, -1):
        for start in range(corner_count):
            corner_order = tuple(
                corners[(start + direction * offset) % corner_count]
                for offset in range(corner_count)
            )
            if not midsides:
                if actual == corner_order:
                    return True
                continue
            midside_order = tuple(
                midsides[
                    (
                        start
                        + direction * offset
                        + (0 if direction == 1 else -1)
                    )
                    % corner_count
                ]
                for offset in range(corner_count)
            )
            if actual == corner_order + midside_order:
                return True
    return False


def _validate_step_references(
    step: Any,
    node_ids: set[int],
    node_sets: Mapping[Any, Any],
    element_lookup: Mapping[int, Any],
    all_element_sets: Mapping[Any, Any],
    surfaces: Mapping[Any, Any],
    edges: Mapping[Any, Any],
    dofs_per_node: int,
    spatial_dimension: int,
    model: Any,
    *,
    effective_boundaries: Sequence[Any] | None = None,
) -> None:
    boundaries = (
        getattr(step, "boundaries", ())
        if effective_boundaries is None
        else effective_boundaries
    )
    for constraint in boundaries:
        _validate_node_target(constraint.target, node_ids, node_sets, step.name)
        first = _integer_id(
            constraint.first_component,
            f"analysis step {step.name} constraint first component",
        )
        last = _integer_id(
            constraint.last_component,
            f"analysis step {step.name} constraint last component",
        )
        if not 1 <= first <= last <= dofs_per_node:
            raise ValueError(
                f"analysis step {step.name} constraint components must satisfy "
                f"1 <= first <= last <= {dofs_per_node}, got {first}..{last}"
            )
        _finite_scalar(
            constraint.value,
            f"analysis step {step.name} constraint value",
        )
    for load in getattr(step, "cloads", ()):
        _validate_node_target(load.target, node_ids, node_sets, step.name)
        component = _integer_id(
            load.component,
            f"analysis step {step.name} load component",
        )
        if component < 1 or component > dofs_per_node:
            raise ValueError(
                f"analysis step {step.name} load component must be from 1 through "
                f"{dofs_per_node}, got {component}"
            )
        _finite_scalar(load.value, f"analysis step {step.name} load value")
    for load in getattr(step, "surface_loads", ()):
        if load.surface not in surfaces:
            raise KeyError(
                f"analysis step {step.name} references missing surface {load.surface}"
            )
    for load in getattr(step, "edge_loads", ()):
        if load.edge not in edges:
            raise KeyError(
                f"analysis step {step.name} references missing edge {load.edge}"
            )
    for load in getattr(step, "line_loads", ()):
        element_ids = _line_load_element_ids(
            load.target,
            element_lookup,
            all_element_sets,
            step.name,
        )
        for element_id in element_ids:
            capabilities = _element_capabilities(
                element_lookup[element_id].type
            )
            if "line" not in capabilities.load_kinds:
                raise ValueError("line loads may target only Beam2 elements")
        vector = getattr(load, "vector", None)
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError("line load vector must contain three finite numbers")
        if len(vector) != 3:
            raise ValueError("line load vector must contain three finite numbers")
        for value in vector:
            try:
                _finite_scalar(value, "line load vector component")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "line load vector must contain three finite numbers"
                ) from exc
        if getattr(load, "coordinate_system", None) not in {"global", "local"}:
            raise ValueError(
                "line load coordinate_system must be 'global' or 'local', "
                f"got {getattr(load, 'coordinate_system', None)!r}"
            )
    for load in getattr(step, "gravity_loads", ()):
        _validate_gravity_load(
            model,
            step,
            load,
            element_lookup,
            all_element_sets,
            spatial_dimension,
        )


def _effective_step_boundaries(model: Any, step: Any) -> tuple[Any, ...]:
    """Return the Initial boundaries inherited by one selected Step."""
    initial = next(
        (
            candidate
            for candidate in model.steps
            if str(candidate.name).casefold() == "initial"
        ),
        None,
    )
    selected = tuple(getattr(step, "boundaries", ()))
    if initial is None or initial is step:
        return selected
    return tuple(getattr(initial, "boundaries", ())) + selected


def _model_spatial_dimension(
    element_lookup: Mapping[int, Any],
) -> int:
    """Return the common catalog spatial dimension for model elements."""
    dimensions = {
        _element_capabilities(element.type).spatial_dimension
        for element in element_lookup.values()
    }
    if not dimensions:
        raise ValueError("model must contain at least one element")
    if len(dimensions) != 1:
        raise ValueError(
            "model elements must share one spatial dimension, got "
            + ", ".join(str(value) for value in sorted(dimensions))
        )
    return next(iter(dimensions))


def _element_capabilities(element_type: Any) -> Any:
    """Resolve capabilities lazily to keep core package initialization acyclic."""
    from ..elements import get_element_capabilities

    return get_element_capabilities(element_type)


def _validate_gravity_load(
    model: Any,
    step: Any,
    load: Any,
    element_lookup: Mapping[int, Any],
    element_sets: Mapping[Any, Any],
    spatial_dimension: int,
) -> None:
    """Validate one global or explicitly targeted gravity acceleration."""
    acceleration = getattr(load, "acceleration", None)
    if not isinstance(acceleration, Sequence) or isinstance(
        acceleration,
        (str, bytes),
    ):
        raise TypeError(
            f"analysis step {step.name} gravity acceleration must be a sequence"
        )
    if len(acceleration) != spatial_dimension:
        raise ValueError(
            f"analysis step {step.name} gravity acceleration must have "
            f"{spatial_dimension} components, got {len(acceleration)}"
        )
    for component in acceleration:
        _finite_scalar(
            component,
            f"analysis step {step.name} gravity acceleration component",
        )

    target = getattr(load, "target", None)
    if target is None:
        return
    element_ids = _gravity_element_ids(
        target,
        element_lookup,
        element_sets,
        step.name,
    )
    for element_id in element_ids:
        _validate_effective_gravity_density(model, step.name, target, element_id)


def _gravity_element_ids(
    target: Any,
    element_lookup: Mapping[int, Any],
    element_sets: Mapping[Any, Any],
    step_name: str,
) -> tuple[int, ...]:
    """Resolve and validate one explicit gravity target."""
    if isinstance(target, str):
        if target not in element_sets:
            raise KeyError(
                f"analysis step {step_name} gravity target references missing "
                f"element set {target}"
            )
        return tuple(
            _integer_id(
                value,
                f"analysis step {step_name} gravity target element id",
            )
            for value in element_sets[target].element_ids
        )
    element_id = _integer_id(
        target,
        f"analysis step {step_name} gravity target",
    )
    if element_id not in element_lookup:
        raise KeyError(
            f"analysis step {step_name} gravity target references missing "
            f"element {element_id}"
        )
    return (element_id,)


def _validate_effective_gravity_density(
    model: Any,
    step_name: str,
    target: Any,
    element_id: int,
) -> None:
    """Require a finite non-negative effective density for targeted gravity."""
    density = model_element_info(model, element_id).properties.get("rho")
    context = (
        f"analysis step {step_name} gravity target {target!r} element {element_id}"
    )
    if density is None:
        raise ValueError(f"{context} requires an effective density rho")
    try:
        value = float(density)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} density rho must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{context} density rho must be finite and >= 0, got {density!r}"
        )


def _line_load_element_ids(
    target: Any,
    element_lookup: Mapping[int, Any],
    element_sets: Mapping[Any, Any],
    step_name: str,
) -> tuple[int, ...]:
    """Resolve and validate one line-load element target."""
    if isinstance(target, str):
        if target not in element_sets:
            raise KeyError(
                f"analysis step {step_name} references missing element set {target}"
            )
        return tuple(
            _integer_id(value, f"analysis step {step_name} line load element id")
            for value in element_sets[target].element_ids
        )
    element_id = _integer_id(target, f"analysis step {step_name} line load target")
    if element_id not in element_lookup:
        raise KeyError(
            f"analysis step {step_name} references missing element {element_id}"
        )
    return (element_id,)


def _validate_unique_step_names(steps: Sequence[Any]) -> None:
    """Require model step names to be unique without case distinctions."""
    seen: dict[str, str] = {}
    for step in steps:
        name = str(step.name)
        key = name.casefold()
        if key in seen:
            raise ValueError(
                "analysis step names must be unique ignoring case; "
                f"got {seen[key]!r} and {name!r}"
            )
        seen[key] = name


def _validate_node_target(
    target: Any,
    node_ids: set[int],
    node_sets: Mapping[Any, Any],
    step_name: str,
) -> None:
    if isinstance(target, str):
        if target not in node_sets:
            raise KeyError(
                f"analysis step {step_name} references missing node set {target}"
            )
        return
    node_id = _integer_id(target, f"analysis step {step_name} node target")
    if node_id not in node_ids:
        raise KeyError(
            f"analysis step {step_name} references missing node {node_id}"
        )


def _finite_scalar(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result
