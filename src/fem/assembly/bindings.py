"""Immutable compiled bindings between fields, cells, and physics operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from fem.model import DofSpace
from fem.physics.contracts import PhysicsOperator
from fem.state import LocalFieldState, SolutionState


@dataclass(frozen=True, slots=True)
class LocalFieldBinding:
    """Map one named global field to an operator's local array shape."""

    name: str
    dofs: tuple[int, ...]
    shape: tuple[int, ...]
    _indices: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        dofs = tuple(_dof(value) for value in self.dofs)
        shape = tuple(int(value) for value in self.shape)
        if not name:
            raise ValueError("local field name must be nonblank")
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("local field shape entries must be positive")
        if int(np.prod(shape)) != len(dofs):
            raise ValueError("local field shape must contain exactly its DOFs")
        if len(dofs) != len(set(dofs)):
            raise ValueError("local field DOFs must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "shape", shape)
        indices = np.asarray(dofs, dtype=int)
        indices.flags.writeable = False
        object.__setattr__(self, "_indices", indices)

    def extract(self, solution: SolutionState) -> LocalFieldState:
        indices = self._indices
        first = (
            None
            if solution.first_derivative is None
            else solution.first_derivative[indices].reshape(self.shape)
        )
        second = (
            None
            if solution.second_derivative is None
            else solution.second_derivative[indices].reshape(self.shape)
        )
        return LocalFieldState(
            solution.values[indices].reshape(self.shape),
            first,
            second,
        )


@dataclass(frozen=True, slots=True)
class CompiledElement:
    """Owned element identity used by the executable assembly graph."""

    id: int
    type: str
    node_ids: tuple[int, ...]
    props: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        element_id = int(self.id)
        element_type = str(self.type).strip()
        node_ids = tuple(int(node_id) for node_id in self.node_ids)
        if element_id < 1:
            raise ValueError("compiled element id must be positive")
        if not element_type:
            raise ValueError("compiled element type must be nonblank")
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("compiled element node_ids must be unique and nonempty")
        if not isinstance(self.props, Mapping):
            raise TypeError("compiled element props must be a mapping")
        object.__setattr__(self, "id", element_id)
        object.__setattr__(self, "type", element_type)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(
            self,
            "props",
            MappingProxyType(dict(self.props)),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "CompiledElement":
        memo[id(self)] = self
        return self


@dataclass(frozen=True, slots=True)
class OperatorBinding:
    """One compiled local equation contribution in a global DOF space."""

    binding_id: str
    entity: CompiledElement
    reference_coordinates: np.ndarray
    fields: tuple[LocalFieldBinding, ...]
    operator: PhysicsOperator
    resources: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)
    state_namespace: str = "operator"
    _dofs: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        binding_id = str(self.binding_id).strip()
        if not binding_id:
            raise ValueError("binding_id must be nonblank")
        reference = np.asarray(self.reference_coordinates, dtype=float)
        if reference.ndim != 2 or not np.all(np.isfinite(reference)):
            raise ValueError("reference_coordinates must be a finite 2D array")
        fields = tuple(self.fields)
        if not fields or any(type(item) is not LocalFieldBinding for item in fields):
            raise TypeError("fields must contain LocalFieldBinding values")
        names = tuple(item.name for item in fields)
        if len(names) != len(set(names)):
            raise ValueError("local field names must be unique")
        dofs = tuple(dof for item in fields for dof in item.dofs)
        if len(dofs) != len(set(dofs)):
            raise ValueError("one operator binding cannot repeat a global DOF")
        if not isinstance(self.operator, PhysicsOperator):
            raise TypeError("operator must implement PhysicsOperator")
        if type(self.entity) is not CompiledElement:
            raise TypeError("entity must be exactly CompiledElement")
        if not isinstance(self.resources, Mapping):
            raise TypeError("resources must be a mapping")
        if not isinstance(self.properties, Mapping):
            raise TypeError("properties must be a mapping")
        namespace = str(self.state_namespace).strip()
        if not namespace:
            raise ValueError("state_namespace must be nonblank")
        owned_reference = np.array(reference, dtype=float, copy=True)
        owned_reference.flags.writeable = False
        object.__setattr__(self, "reference_coordinates", owned_reference)
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))
        object.__setattr__(self, "state_namespace", namespace)
        object.__setattr__(
            self,
            "_dofs",
            tuple(dof for item in fields for dof in item.dofs),
        )

    @property
    def dofs(self) -> tuple[int, ...]:
        return self._dofs

    def validate_dof_space(
        self,
        dof_space: DofSpace,
        *,
        allowed_dofs: Mapping[str, frozenset[int]] | None = None,
    ) -> None:
        for local in self.fields:
            out_of_bounds = sorted(
                dof
                for dof in local.dofs
                if dof < 0 or dof >= dof_space.num_dofs
            )
            if out_of_bounds:
                raise IndexError(
                    f"binding field {local.name!r} uses DOFs out of bounds "
                    f"[0, {dof_space.num_dofs}): {out_of_bounds}"
                )
            allowed = (
                dof_space.field(local.name).dofs
                if allowed_dofs is None
                else allowed_dofs[local.name]
            )
            invalid = sorted(set(local.dofs).difference(allowed))
            if invalid:
                raise ValueError(
                    f"binding field {local.name!r} uses foreign DOFs {invalid}"
                )

    def local_fields(self, solution: SolutionState) -> Mapping[str, LocalFieldState]:
        return MappingProxyType(
            {local.name: local.extract(solution) for local in self.fields}
        )


def displacement_bindings_from_mesh(
    mesh: Any,
    operator_by_type: Mapping[str, PhysicsOperator] | PhysicsOperator,
    *,
    properties_by_element: Mapping[int, Mapping[str, Any]] | None = None,
    material_by_element: Mapping[int, Any] | None = None,
    state_namespace: str = "mechanics.material",
) -> tuple[OperatorBinding, ...]:
    """Compile the current displacement mesh into explicit local bindings."""

    properties = properties_by_element or {}
    materials = material_by_element or {}
    node_lookup = {int(node.id): node for node in mesh.nodes}
    bindings: list[OperatorBinding] = []
    for element in mesh.elements:
        if not tuple(element.node_ids):
            raise ValueError(
                f"element {int(element.id)} must contain at least one node"
            )
        nodes = [node_lookup[int(node_id)] for node_id in element.node_ids]
        dimensions = 3 if all(hasattr(node, "z") for node in nodes) else 2
        names = ("x", "y", "z")[:dimensions]
        reference = np.asarray(
            [[float(getattr(node, name)) for name in names] for node in nodes],
            dtype=float,
        )
        if isinstance(operator_by_type, Mapping):
            try:
                operator = operator_by_type[str(element.type).casefold()]
            except KeyError as exc:
                raise NotImplementedError(
                    f"physics operator is not bound for element type {element.type}"
                ) from exc
        else:
            operator = operator_by_type
        element_id = int(element.id)
        compiled_element = CompiledElement(
            id=element_id,
            type=str(element.type),
            node_ids=tuple(element.node_ids),
            props=getattr(element, "props", {}),
        )
        bindings.append(
            OperatorBinding(
                binding_id=f"{state_namespace}:{element_id}",
                entity=compiled_element,
                reference_coordinates=reference,
                fields=(
                    LocalFieldBinding(
                        "U",
                        tuple(mesh.element_dofs(element)),
                        (len(element.node_ids), int(mesh.dofs_per_node)),
                    ),
                ),
                operator=operator,
                resources={"material": materials.get(element_id)},
                properties=dict(
                    properties.get(
                        element_id,
                        getattr(element, "props", {}),
                    )
                ),
                state_namespace=state_namespace,
            )
        )
    return tuple(bindings)


def _dof(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("local DOF ids must be integers")
    if value < 0:
        raise ValueError("local DOF ids must be non-negative")
    return int(value)


__all__ = [
    "CompiledElement",
    "LocalFieldBinding",
    "OperatorBinding",
    "displacement_bindings_from_mesh",
]
