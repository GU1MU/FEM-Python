from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dynamics import DynamicFrameData


@dataclass
class ModelResult:
    """Result data for one solved model step."""

    model: Any
    step: Any
    U: np.ndarray
    reactions: np.ndarray
    name: str | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    load_factor: float | None = None
    iterations: int | None = None
    residual_norm: float | None = None
    frames: tuple["ResultFrame", ...] = field(default_factory=tuple)
    compiled_model: Any | None = None
    dynamic_data: DynamicFrameData | None = None

    def __post_init__(self) -> None:
        num_dofs = int(self.model.mesh.num_dofs)
        self.U = _result_vector("U", self.U, num_dofs)
        self.reactions = _result_vector("reactions", self.reactions, num_dofs)
        self.outputs = dict(self.outputs)
        if self.load_factor is not None:
            self.load_factor = _finite_scalar("load_factor", self.load_factor)
        if self.iterations is not None:
            if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
                raise ValueError("iterations must be an integer >= 0")
            if self.iterations < 0:
                raise ValueError("iterations must be an integer >= 0")
        if self.residual_norm is not None:
            self.residual_norm = _finite_scalar("residual_norm", self.residual_norm)
        self.frames = tuple(self.frames)
        if any(type(frame) is not ResultFrame for frame in self.frames):
            raise TypeError("frames must contain only ResultFrame values")
        if self.dynamic_data is not None and type(self.dynamic_data) is not DynamicFrameData:
            raise TypeError("dynamic_data must be DynamicFrameData or None")

    def nodal_displacement(self, node_id: int, component: int) -> float:
        """Return one nodal displacement component using 1-based numbering."""

        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.U[dof])

    def nodal_reaction(self, node_id: int, component: int) -> float:
        """Return one nodal reaction component using 1-based numbering."""

        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.reactions[dof])


@dataclass(frozen=True, slots=True)
class ResultFrame:
    """One converged static increment retained for post-processing."""

    model: Any
    step: Any
    U: np.ndarray
    reactions: np.ndarray
    frame_index: int
    load_factor: float
    name: str | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    iterations: int | None = None
    residual_norm: float | None = None
    converged: bool = True
    compiled_model: Any | None = None
    dynamic_data: DynamicFrameData | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(
            self.frame_index,
            int,
        ) or self.frame_index < 1:
            raise ValueError("frame_index must be an integer >= 1")
        num_dofs = int(self.model.mesh.num_dofs)
        displacement = _result_vector("U", self.U, num_dofs)
        reactions = _result_vector("reactions", self.reactions, num_dofs)
        displacement.setflags(write=False)
        reactions.setflags(write=False)
        object.__setattr__(self, "U", displacement)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(
            self,
            "load_factor",
            _finite_scalar("load_factor", self.load_factor),
        )
        if self.iterations is not None:
            if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
                raise ValueError("iterations must be an integer >= 0")
            if self.iterations < 0:
                raise ValueError("iterations must be an integer >= 0")
        if self.residual_norm is not None:
            object.__setattr__(
                self,
                "residual_norm",
                _finite_scalar("residual_norm", self.residual_norm),
            )
        if type(self.converged) is not bool:
            raise TypeError("converged must be a bool")
        if self.dynamic_data is not None and type(self.dynamic_data) is not DynamicFrameData:
            raise TypeError("dynamic_data must be DynamicFrameData or None")
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        object.__setattr__(self, "outputs", dict(self.outputs))

    def nodal_displacement(self, node_id: int, component: int) -> float:
        """Return one nodal displacement component using 1-based numbering."""

        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.U[dof])

    def nodal_reaction(self, node_id: int, component: int) -> float:
        """Return one nodal reaction component using 1-based numbering."""

        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.reactions[dof])


@dataclass
class ModelResults:
    """Collection of solved model step results."""

    model: Any
    results: tuple[ModelResult, ...]

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(
        self,
        index: int | slice,
    ) -> ModelResult | tuple[ModelResult, ...]:
        return self.results[index]


def _nodal_dof(mesh: Any, node_id: int, component: int) -> int:
    if isinstance(component, bool):
        raise TypeError("component must be an integer")
    try:
        component_number = operator.index(component)
    except TypeError as exc:
        raise TypeError("component must be an integer") from exc
    if component_number < 1 or component_number > mesh.dofs_per_node:
        raise IndexError(
            f"component {component_number} out of range for "
            f"{mesh.dofs_per_node} DOFs per node; components are 1-based"
        )
    return mesh.global_dof(node_id, component_number - 1)


def _result_vector(name: str, values: Any, num_dofs: int) -> np.ndarray:
    """Return an owned, finite one-dimensional result vector."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.shape[0] != num_dofs:
        raise ValueError(
            f"{name} must have length {num_dofs}, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_scalar(name: str, value: Any) -> float:
    """Return one finite result scalar."""

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar
