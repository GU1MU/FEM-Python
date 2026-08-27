"""Single sparse scatter implementation for compiled operator bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from fem.model import DofSpace, validate_mesh
from fem.physics.contracts import (
    LocalContribution,
    LocalContributionBatch,
    LocalOutputBatch,
    PhysicsOperator,
)
from fem.state import EvaluationContext, SolutionState, StateManager

from .bindings import OperatorBinding, displacement_bindings_from_mesh
from .contracts import AssemblyResult


@dataclass(slots=True)
class SparseAssembler:
    """Evaluate compiled local bindings and scatter them once to global CSR.

    The assembler knows neither element families nor material algorithms. A
    binding supplies the operator, local field map, reference geometry,
    material, and properties. The same traversal therefore supports linear
    and nonlinear statics, dynamics, and coupled fields.
    """

    dof_space: DofSpace
    bindings: tuple[OperatorBinding, ...]
    state: StateManager | None = None
    output_metadata: Mapping[str, Any] = field(default_factory=dict)
    require_symmetric_tangent: bool = False
    _scatter_plan: "_SparseScatterPlan" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.dof_space) is not DofSpace:
            raise TypeError("dof_space must be exactly DofSpace")
        bindings = tuple(self.bindings)
        if not bindings or any(type(item) is not OperatorBinding for item in bindings):
            raise TypeError("bindings must contain OperatorBinding values")
        if type(self.require_symmetric_tangent) is not bool:
            raise TypeError("require_symmetric_tangent must be bool")
        binding_ids: set[str] = set()
        allowed_dofs = {
            field.name: frozenset(field.dofs)
            for field in self.dof_space.fields
        }
        for binding in bindings:
            binding.validate_dof_space(
                self.dof_space,
                allowed_dofs=allowed_dofs,
            )
            if binding.binding_id in binding_ids:
                raise ValueError(
                    f"duplicate operator binding identity {binding.binding_id!r}"
                )
            binding_ids.add(binding.binding_id)
            binding.operator.initialize(
                binding.entity,
                binding.resources,
                self.state,
                binding.properties,
                state_namespace=binding.state_namespace,
            )
            prepare_geometry = getattr(
                binding.operator,
                "prepare_reference_geometry",
                None,
            )
            if prepare_geometry is not None:
                if not callable(prepare_geometry):
                    raise TypeError(
                        "operator.prepare_reference_geometry must be callable"
                    )
                prepare_geometry(
                    binding.entity,
                    binding.reference_coordinates,
                    binding.properties,
                )
        self.bindings = bindings
        self.output_metadata = MappingProxyType(dict(self.output_metadata))
        self._scatter_plan = _SparseScatterPlan.from_bindings(
            bindings,
            self.dof_space.num_dofs,
        )

    @classmethod
    def from_displacement_mesh(
        cls,
        mesh: Any,
        operator: PhysicsOperator | Mapping[str, PhysicsOperator],
        *,
        element_properties_by_element: Mapping[int, Mapping[str, Any]] | None = None,
        material_by_element: Mapping[int, Any] | None = None,
        state: StateManager | None = None,
        output_metadata: Mapping[str, Any] | None = None,
        require_symmetric_tangent: bool = False,
        state_namespace: str = "mechanics.material",
    ) -> "SparseAssembler":
        """Compile the current displacement mesh into explicit bindings."""

        validate_mesh(mesh)
        normalized = _operator_map(operator)
        dof_space = DofSpace.displacement_for_mesh(mesh)
        bindings = displacement_bindings_from_mesh(
            mesh,
            normalized,
            properties_by_element=element_properties_by_element,
            material_by_element=material_by_element,
            state_namespace=state_namespace,
        )
        return cls(
            dof_space=dof_space,
            bindings=bindings,
            state=state,
            output_metadata=output_metadata or {},
            require_symmetric_tangent=require_symmetric_tangent,
        )

    @property
    def num_dofs(self) -> int:
        return self.dof_space.num_dofs

    def assemble(
        self,
        solution: SolutionState,
        *,
        context: EvaluationContext | None = None,
    ) -> AssemblyResult:
        """Evaluate all local equations and scatter residual/matrices to CSR."""

        if type(solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if solution.dof_space != self.dof_space:
            raise ValueError("solution DOF space must match the assembler")
        evaluation_context = context or EvaluationContext()
        if type(evaluation_context) is not EvaluationContext:
            raise TypeError("context must be exactly EvaluationContext")

        global_residual = np.zeros(self.num_dofs, dtype=float)
        need_tangent = bool(
            evaluation_context.parameters.get("_fem_need_tangent", True)
        )
        tangent_parts = (
            _SparseAccumulator(self._scatter_plan) if need_tangent else None
        )
        # Most static/nonlinear mechanics evaluations do not contribute mass
        # or damping.  Allocate those global sparse value buffers only after
        # the first local contribution actually requests them.
        mass_parts: _SparseAccumulator | None = None
        damping_parts: _SparseAccumulator | None = None
        point_records: list[Mapping[str, Any]] = []
        local_outputs: list[LocalOutputBatch] = []

        batch_evaluations = _batch_evaluations(
            self.bindings,
            solution,
            self.state,
            context=evaluation_context,
        )
        if batch_evaluations is None:
            evaluated = (
                (
                    binding_index,
                    binding,
                    binding.operator.evaluate(
                        binding.entity,
                        binding.reference_coordinates,
                        binding.local_fields(solution),
                        binding.dofs,
                        binding.resources,
                        self.state,
                        binding.properties,
                        context=evaluation_context,
                        state_namespace=binding.state_namespace,
                    ),
                )
                for binding_index, binding in enumerate(self.bindings)
            )
            for binding_index, binding, contribution in evaluated:
                _validate_contribution(
                    contribution,
                    binding,
                    self.num_dofs,
                    require_symmetric=self.require_symmetric_tangent,
                )
                local_dofs = np.asarray(contribution.dofs, dtype=int)
                global_residual[local_dofs] += contribution.residual
                if tangent_parts is not None:
                    tangent_parts.add(binding_index, contribution.tangent)
                if contribution.mass is not None:
                    if mass_parts is None:
                        mass_parts = _SparseAccumulator(self._scatter_plan)
                    mass_parts.add(binding_index, contribution.mass)
                if contribution.damping is not None:
                    if damping_parts is None:
                        damping_parts = _SparseAccumulator(self._scatter_plan)
                    damping_parts.add(binding_index, contribution.damping)

                if contribution.output_batch is not None:
                    local_outputs.append(contribution.output_batch)
                    point_records.extend(contribution.output_batch.as_records())
                else:
                    records = contribution.outputs.get("integration_points", ())
                    point_records.extend(dict(record) for record in records)
        else:
            for contribution_batch in batch_evaluations:
                if type(contribution_batch) is not LocalContributionBatch:
                    raise TypeError(
                        "batched operator evaluation must yield "
                        "LocalContributionBatch values"
                    )
                mass_parts, damping_parts = _scatter_batch(
                    contribution_batch,
                    self.bindings,
                    self._scatter_plan,
                    global_residual,
                    tangent_parts,
                    mass_parts,
                    damping_parts,
                    require_symmetric=self.require_symmetric_tangent,
                )
                for output_batch in contribution_batch.output_batches:
                    local_outputs.append(output_batch)
                    point_records.extend(output_batch.as_records())

        outputs = dict(self.output_metadata)
        outputs.update(
            {
                "load_factor": evaluation_context.load_factor,
                "time": evaluation_context.time,
                "integration_points": _pack_integration_points(point_records),
            }
        )
        return AssemblyResult(
            residual=global_residual,
            tangent=(
                tangent_parts.matrix(self.num_dofs)
                if tangent_parts is not None
                else csr_matrix((self.num_dofs, self.num_dofs), dtype=float)
            ),
            mass=(mass_parts.matrix(self.num_dofs) if mass_parts else None),
            damping=(damping_parts.matrix(self.num_dofs) if damping_parts else None),
            outputs=outputs,
            local_outputs=tuple(local_outputs),
        )

    def begin_increment(self) -> None:
        if self.state is not None:
            self.state.begin_increment()

    def commit(self) -> None:
        if self.state is not None:
            self.state.commit()

    def rollback(self) -> None:
        if self.state is not None:
            self.state.rollback()


@dataclass(frozen=True, slots=True)
class _SparseScatterPlan:
    """Immutable CSR pattern and local-to-global data positions."""

    indices: np.ndarray
    indptr: np.ndarray
    local_positions: tuple[np.ndarray, ...]
    size: int

    @classmethod
    def from_bindings(
        cls,
        bindings: tuple[OperatorBinding, ...],
        size: int,
    ) -> "_SparseScatterPlan":
        rows = np.concatenate(
            tuple(
                np.repeat(np.asarray(binding.dofs, dtype=int), len(binding.dofs))
                for binding in bindings
            )
        )
        columns = np.concatenate(
            tuple(
                np.tile(np.asarray(binding.dofs, dtype=int), len(binding.dofs))
                for binding in bindings
            )
        )
        pattern = coo_matrix(
            (np.ones(rows.size, dtype=float), (rows, columns)),
            shape=(size, size),
        ).tocsr()
        indices = np.asarray(pattern.indices, dtype=int)
        indptr = np.asarray(pattern.indptr, dtype=int)
        local_positions: list[np.ndarray] = []
        for binding in bindings:
            dofs = np.asarray(binding.dofs, dtype=int)
            row_positions = []
            for row in dofs:
                start = int(indptr[row])
                stop = int(indptr[row + 1])
                row_positions.append(
                    start
                    + np.searchsorted(
                        indices[start:stop],
                        dofs,
                    )
                )
            position_array = np.concatenate(row_positions).astype(
                int,
                copy=False,
            )
            position_array.flags.writeable = False
            local_positions.append(position_array)
        indices.flags.writeable = False
        indptr.flags.writeable = False
        return cls(
            indices=indices,
            indptr=indptr,
            local_positions=tuple(local_positions),
            size=int(size),
        )


@dataclass(slots=True)
class _SparseAccumulator:
    plan: _SparseScatterPlan
    values: np.ndarray = field(init=False, repr=False)
    used: bool = False

    def __post_init__(self) -> None:
        self.values = np.zeros(self.plan.indices.size, dtype=float)

    def __bool__(self) -> bool:
        return self.used

    def add(self, binding_index: int, matrix: np.ndarray) -> None:
        values = np.asarray(matrix, dtype=float).reshape(-1)
        positions = self.plan.local_positions[binding_index]
        if values.size != positions.size:
            raise ValueError("local sparse contribution size does not match its binding")
        self.values[positions] += values
        self.used = True

    def matrix(self, size: int) -> csr_matrix:
        if int(size) != self.plan.size:
            raise ValueError("sparse accumulator size does not match its plan")
        return csr_matrix(
            (self.values, self.plan.indices, self.plan.indptr),
            shape=(size, size),
            copy=False,
        )


def _batch_evaluations(
    bindings: tuple[OperatorBinding, ...],
    solution: SolutionState,
    state: StateManager | None,
    *,
    context: EvaluationContext,
):
    if not bindings:
        return None
    operator = bindings[0].operator
    can_batch = getattr(operator, "can_evaluate_batch", None)
    evaluate_batch = getattr(operator, "evaluate_batch", None)
    if not callable(can_batch) or not callable(evaluate_batch):
        return None
    if any(binding.operator is not operator for binding in bindings):
        return None
    if not can_batch(bindings, context=context):
        return None
    return evaluate_batch(
        bindings,
        solution,
        state,
        context=context,
    )


def _scatter_batch(
    contribution: LocalContributionBatch,
    bindings: tuple[OperatorBinding, ...],
    plan: _SparseScatterPlan,
    global_residual: np.ndarray,
    tangent_parts: _SparseAccumulator | None,
    mass_parts: _SparseAccumulator | None,
    damping_parts: _SparseAccumulator | None,
    *,
    require_symmetric: bool,
) -> tuple[_SparseAccumulator | None, _SparseAccumulator | None]:
    """Scatter one validated numerical block without per-element wrappers."""

    binding_indices = contribution.binding_indices
    if binding_indices.size == 0:
        return mass_parts, damping_parts
    if np.any(binding_indices < 0) or np.any(binding_indices >= len(bindings)):
        raise IndexError("batched contribution binding index is out of bounds")
    expected_dofs = np.asarray(
        [bindings[int(index)].dofs for index in binding_indices],
        dtype=int,
    )
    if not np.array_equal(expected_dofs, contribution.dofs):
        raise ValueError("batched contribution DOFs do not match bindings")
    if np.any(contribution.dofs < 0) or np.any(
        contribution.dofs >= global_residual.size
    ):
        raise IndexError("batched contribution DOF is out of bounds")
    if require_symmetric and not np.allclose(
        contribution.tangent,
        np.swapaxes(contribution.tangent, 1, 2),
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError("batched contribution tangent is not symmetric")

    np.add.at(
        global_residual,
        contribution.dofs.reshape(-1),
        contribution.residual.reshape(-1),
    )
    positions = np.asarray(
        [plan.local_positions[int(index)] for index in binding_indices],
        dtype=int,
    )
    if positions.shape != contribution.tangent.reshape(
        contribution.tangent.shape[0], -1
    ).shape:
        raise ValueError("batched local tangent does not match scatter pattern")
    if tangent_parts is not None:
        np.add.at(
            tangent_parts.values,
            positions.reshape(-1),
            contribution.tangent.reshape(-1),
        )
    if contribution.mass is not None:
        if mass_parts is None:
            mass_parts = _SparseAccumulator(plan)
        np.add.at(
            mass_parts.values,
            positions.reshape(-1),
            contribution.mass.reshape(-1),
        )
        mass_parts.used = True
    if contribution.damping is not None:
        if damping_parts is None:
            damping_parts = _SparseAccumulator(plan)
        np.add.at(
            damping_parts.values,
            positions.reshape(-1),
            contribution.damping.reshape(-1),
        )
        damping_parts.used = True
    return mass_parts, damping_parts


def _operator_map(
    value: PhysicsOperator | Mapping[str, PhysicsOperator],
) -> PhysicsOperator | dict[str, PhysicsOperator]:
    if isinstance(value, Mapping):
        normalized: dict[str, PhysicsOperator] = {}
        for element_type, candidate in value.items():
            if not isinstance(candidate, PhysicsOperator):
                raise TypeError("operator map values must implement PhysicsOperator")
            normalized[str(element_type).casefold()] = candidate
        if not normalized:
            raise ValueError("operator map must not be empty")
        return normalized
    if not isinstance(value, PhysicsOperator):
        raise TypeError("operator must implement PhysicsOperator")
    return value


def _validate_contribution(
    contribution: LocalContribution,
    binding: OperatorBinding,
    num_dofs: int,
    *,
    require_symmetric: bool,
) -> None:
    if type(contribution) is not LocalContribution:
        raise TypeError(
            f"binding {binding.binding_id!r} operator must return LocalContribution"
        )
    if contribution.dofs != binding.dofs:
        raise ValueError(
            f"binding {binding.binding_id!r} contribution DOFs must match binding"
        )
    if any(dof < 0 or dof >= num_dofs for dof in contribution.dofs):
        raise IndexError(
            f"binding {binding.binding_id!r} contribution DOF is out of bounds"
        )
    if require_symmetric and not np.allclose(
        contribution.tangent,
        contribution.tangent.T,
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError(f"binding {binding.binding_id!r} tangent is not symmetric")


def _pack_integration_points(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "element_id": np.empty(0, dtype=int),
            "integration_point": np.empty(0, dtype=int),
            "history": (),
        }
    keys = set(records[0])
    for record in records[1:]:
        keys.intersection_update(record)
    packed: dict[str, Any] = {}
    for key in sorted(keys):
        items = [record[key] for record in records]
        if key == "history":
            packed[key] = tuple(dict(item) for item in items)
        elif key in {"element_id", "integration_point"}:
            packed[key] = np.asarray(items, dtype=int)
        elif key == "natural_coordinates":
            packed[key] = np.asarray(items, dtype=float)
        elif isinstance(items[0], (np.ndarray, float, int, np.number)):
            packed[key] = np.asarray(items, dtype=float)
        else:
            packed[key] = tuple(items)
    return packed


__all__ = ["SparseAssembler"]
