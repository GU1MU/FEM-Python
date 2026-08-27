"""Continuum-mechanics local operator shared by plane and solid elements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ....materials.contracts import (
    KinematicMeasure,
    MaterialModel,
    MaterialPointInput,
    MaterialResponse,
    StressMeasure,
)
from ....state import EvaluationContext, LocalFieldState, StateKey, StateManager
from ...contracts import (
    LocalContribution,
    LocalContributionBatch,
    LocalOutputBatch,
    LocalPointOutput,
)
from ....elements.contracts import ElementDefinition
from ..kinematics import KinematicsModel
from .plane import plane_thickness


@dataclass(frozen=True, slots=True)
class _ReferencePointData:
    natural_coordinates: tuple[float, ...]
    weight: float
    reference_gradients: np.ndarray
    reference_det: float


@dataclass(slots=True)
class ContinuumMechanicsOperator:
    """Evaluate a plane or three-dimensional continuum element.

    The weak form is independent of element topology, material model, and
    spatial dimension.  The reference element supplies interpolation and
    quadrature, the kinematics model supplies ``F``/``E``, and the material
    supplies ``P`` and ``dP/dF``.  A two-dimensional element uses its section
    thickness as the out-of-plane measure; a three-dimensional element uses
    its reference volume measure directly.
    """

    kinematics: KinematicsModel
    definition: ElementDefinition
    _reference_geometry: dict[int, tuple["_ReferencePointData", ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _reference_geometry_templates: dict[
        tuple[tuple[int, ...], bytes],
        tuple["_ReferencePointData", ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _reference_measure_scale: dict[int, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _state_key_cache: dict[tuple[str, int], tuple[StateKey, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _reference_gradient_cache: dict[int, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _reference_weight_cache: dict[int, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def prepare_reference_geometry(
        self,
        element: Any,
        reference_coordinates: np.ndarray,
        properties: Mapping[str, Any] | None = None,
    ) -> None:
        """Precompute geometry terms that do not depend on displacement.

        The reference array is owned by its compiled binding and remains alive
        for the lifetime of the operator.  Using its identity as the cache key
        avoids hashing or copying coordinates during every Newton evaluation.
        """

        reference = _coordinates(reference_coordinates, "reference_coordinates")
        cache_key = id(reference_coordinates)
        # Element reference coordinates often differ only by translation on
        # structured meshes.  Their Jacobians, reference gradients, and
        # quadrature measures are then identical.  Use an exact
        # translation-normalized byte signature to share the immutable
        # template while retaining a per-binding identity lookup for the hot
        # Newton path.  A byte-level key avoids geometric tolerances and can
        # therefore never merge two numerically different geometries.
        signature = _reference_geometry_signature(reference)
        reference_geometry = self._reference_geometry_templates.get(signature)
        if reference_geometry is None:
            reference_geometry = _build_reference_geometry(
                self.definition,
                reference,
            )
            self._reference_geometry_templates[signature] = reference_geometry
        geometry_identity = id(reference_geometry)
        if geometry_identity not in self._reference_gradient_cache:
            gradients = np.asarray(
                [point.reference_gradients for point in reference_geometry],
                dtype=float,
            )
            weights = np.asarray(
                [point.reference_det * point.weight for point in reference_geometry],
                dtype=float,
            )
            gradients.flags.writeable = False
            weights.flags.writeable = False
            self._reference_gradient_cache[geometry_identity] = gradients
            self._reference_weight_cache[geometry_identity] = weights
        self._reference_geometry[cache_key] = reference_geometry
        self._reference_measure_scale[cache_key] = _measure_scale(
            properties or {},
            int(reference.shape[1]),
            element_id=int(element.id),
        )

    def initialize(
        self,
        element: Any,
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        state_namespace: str,
    ) -> None:
        del properties
        material = _material(resources)
        if state is None:
            raise TypeError("continuum mechanics requires a StateManager")
        element_id = int(element.id)
        keys = self._state_keys(element_id, state_namespace)
        for key in keys:
            state.register(key, material.initial_state())

    def _state_keys(
        self,
        element_id: int,
        state_namespace: str,
    ) -> tuple[StateKey, ...]:
        """Return stable integration-point keys without rebuilding them per trial."""

        cache_key = (str(state_namespace), int(element_id))
        cached = self._state_key_cache.get(cache_key)
        if cached is not None:
            return cached
        keys = tuple(
            StateKey.material_point(
                int(element_id),
                point_id,
                namespace=str(state_namespace),
            )
            for point_id, _ in enumerate(self.definition.gauss_points(), start=1)
        )
        self._state_key_cache[cache_key] = keys
        return keys

    def evaluate(
        self,
        element: Any,
        reference_coordinates: np.ndarray,
        fields: Mapping[str, LocalFieldState],
        dofs: tuple[int, ...],
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        context: EvaluationContext,
        state_namespace: str,
    ) -> LocalContribution:
        element_id = int(element.id)
        material = _material(resources)
        reference = _coordinates(reference_coordinates, "reference_coordinates")
        if state is None:
            raise TypeError("continuum mechanics requires a StateManager")
        try:
            displacement = fields["U"].values
        except KeyError as exc:
            raise ValueError("continuum mechanics requires local field 'U'") from exc
        trial = _coordinates(displacement, "displacement")
        node_count = int(self.definition.node_count)
        if reference.shape != trial.shape:
            raise ValueError(
                f"{self.definition.canonical_type} reference and displacement "
                "shapes must match"
            )
        if reference.shape[0] != node_count:
            raise ValueError(
                f"{self.definition.canonical_type} requires {node_count} nodes"
            )
        spatial_dimension = int(reference.shape[1])
        if spatial_dimension not in (2, 3):
            raise ValueError(
                f"{self.definition.canonical_type} requires a 2D or 3D geometry"
            )
        if not isinstance(properties, Mapping):
            raise TypeError("compiled continuum properties must be a mapping")

        local_size = spatial_dimension * node_count
        local_residual = np.zeros(local_size, dtype=float)
        local_tangent = np.zeros((local_size, local_size), dtype=float)
        records: list[LocalPointOutput] = []
        material_fields = _material_fields(fields, context)
        need_tangent = bool(
            context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            context.parameters.get("_fem_capture_outputs", True)
        )
        reference_geometry = self._reference_geometry.get(
            id(reference_coordinates)
        )
        if reference_geometry is None:
            self.prepare_reference_geometry(
                element,
                reference_coordinates,
                properties,
            )
            reference_geometry = self._reference_geometry[id(reference_coordinates)]
        measure_scale = self._reference_measure_scale.get(id(reference_coordinates))
        if measure_scale is None:
            measure_scale = _measure_scale(
                properties,
                spatial_dimension,
                element_id=element_id,
            )
            self._reference_measure_scale[id(reference_coordinates)] = measure_scale
        state_keys = self._state_keys(
            element_id,
            state_namespace,
        )

        for point_id, point_data in enumerate(reference_geometry, start=1):
            natural_coordinates = point_data.natural_coordinates
            weight = point_data.weight
            reference_det = point_data.reference_det
            reference_gradients = point_data.reference_gradients
            kinematic = self.kinematics.evaluate(
                reference,
                trial,
                reference_gradients,
                reference_det,
            )
            key = state_keys[point_id - 1]
            response = material.evaluate(
                MaterialPointInput(
                    kinematics=kinematic,
                    committed_state=state.committed(key),
                    context=context,
                    fields=material_fields,
                )
            )
            state.stage(key, response.trial_state)
            scale = measure_scale * reference_det * weight
            if _is_small_strain_kinematics(self.kinematics):
                _integrate_small_strain_point(
                    response,
                    reference_gradients,
                    scale,
                    spatial_dimension,
                    local_residual,
                    local_tangent,
                    include_tangent=need_tangent,
                )
                if capture_outputs:
                    records.append(
                        _small_strain_record(
                            element_id,
                            point_id,
                            natural_coordinates,
                            weight,
                            kinematic,
                            response,
                        )
                    )
            else:
                _integrate_total_lagrangian_point(
                    response,
                    reference_gradients,
                    scale,
                    spatial_dimension,
                    local_residual,
                    local_tangent,
                    include_tangent=need_tangent,
                )
                if capture_outputs:
                    records.append(
                        _total_lagrangian_record(
                            element_id,
                            point_id,
                            natural_coordinates,
                            weight,
                            kinematic,
                            response,
                        )
                    )

        output_batch = (
            LocalOutputBatch.element(element_id, tuple(records))
            if capture_outputs
            else None
        )
        return LocalContribution(
            dofs=dofs,
            residual=local_residual,
            tangent=local_tangent,
            outputs=(
                {"integration_points": output_batch.as_records()}
                if output_batch is not None
                else {}
            ),
            output_batch=output_batch,
        )

    def can_evaluate_batch(
        self,
        bindings: tuple[Any, ...],
        *,
        context: EvaluationContext,
    ) -> bool:
        """Return whether the bounded Quad4 batch path is valid."""

        if self.definition.canonical_type != "Quad4":
            return False
        if int(self.definition.node_count) != 4:
            return False
        if _is_small_strain_kinematics(self.kinematics):
            return False
        if not bindings:
            return False
        first = bindings[0]
        if len(first.fields) != 1 or first.fields[0].name != "U":
            return False
        materials = [binding.resources.get("material") for binding in bindings]
        material = materials[0]
        if not callable(getattr(material, "evaluate_batch", None)):
            return False
        if getattr(material, "algorithm", None) != "hencky":
            return False
        return all(
            len(binding.fields) == 1
            and binding.fields[0].name == "U"
            and type(candidate) is type(material)
            and candidate == material
            for binding, candidate in zip(bindings, materials)
        )

    def evaluate_batch(
        self,
        bindings: tuple[Any, ...],
        solution: Any,
        state: StateManager | None,
        *,
        context: EvaluationContext,
    ):
        """Yield local contributions from bounded blocks of Quad4 points."""

        if state is None:
            raise TypeError("continuum mechanics requires a StateManager")
        material = bindings[0].resources["material"]
        need_tangent = bool(
            context.parameters.get("_fem_need_tangent", True)
        )
        capture_outputs = bool(
            context.parameters.get("_fem_capture_outputs", True)
        )
        point_count = len(self.definition.gauss_points())
        # Keep enough points per BLAS/LAPACK block to amortize Python/state
        # bookkeeping while bounding the temporary tangent workset.  The
        # arrays are local to one yield and are released before the next one.
        chunk_size = 2_048
        for chunk_start in range(0, len(bindings), chunk_size):
            chunk = bindings[chunk_start:chunk_start + chunk_size]
            reference = np.asarray(
                [binding.reference_coordinates for binding in chunk],
                dtype=float,
            )
            displacement = np.asarray(
                [
                    solution.values[binding.fields[0]._indices].reshape(
                        binding.fields[0].shape
                    )
                    for binding in chunk
                ],
                dtype=float,
            )
            current = reference + displacement
            reference_geometries = tuple(
                self._reference_geometry[id(binding.reference_coordinates)]
                for binding in chunk
            )
            first_geometry = reference_geometries[0]
            first_gradients = self._reference_gradient_cache[id(first_geometry)]
            if all(geometry is first_geometry for geometry in reference_geometries):
                gradients = np.broadcast_to(
                    first_gradients,
                    (len(chunk),) + first_gradients.shape,
                )
            else:
                gradients = np.asarray(
                    [
                        self._reference_gradient_cache[id(geometry)]
                        for geometry in reference_geometries
                    ],
                    dtype=float,
                )
            if gradients.shape[1] != point_count or gradients.shape[2:] != (2, 4):
                raise ValueError("batched Quad4 reference gradients have an invalid shape")
            deformation_2d = np.einsum(
                "eni,epnj->epij",
                current,
                gradients.transpose(0, 1, 3, 2),
            )
            deformation = np.zeros(
                (len(chunk), point_count, 3, 3),
                dtype=float,
            )
            deformation[:, :, :2, :2] = deformation_2d
            deformation[:, :, 2, 2] = 1.0

            committed_plastic = np.zeros_like(deformation)
            committed_alpha = np.zeros((len(chunk), point_count), dtype=float)
            for element_index, binding in enumerate(chunk):
                element_id = int(binding.entity.id)
                keys = self._state_keys(
                    element_id,
                    binding.state_namespace,
                )
                for point_index, key in enumerate(keys):
                    committed = state.committed(key)
                    plastic = committed.get("plastic_log_strain")
                    if plastic is not None:
                        committed_plastic[element_index, point_index] = np.asarray(
                            plastic,
                            dtype=float,
                        )
                    committed_alpha[element_index, point_index] = float(
                        committed.get("equivalent_plastic_strain", 0.0)
                    )

            flat = len(chunk) * point_count
            batch = material.evaluate_batch(
                deformation.reshape(flat, 3, 3),
                committed_plastic.reshape(flat, 3, 3),
                committed_alpha.reshape(flat),
                need_tangent=need_tangent,
                tangent_columns=(0, 1, 3, 4),
            )
            stress = batch.first_piola_stress.reshape(
                len(chunk), point_count, 3, 3
            )
            tangent = batch.tangent.reshape(
                len(chunk), point_count, 9, 9
            )
            trial_plastic = batch.plastic_log_strain.reshape(
                len(chunk), point_count, 3, 3
            )
            trial_alpha = batch.equivalent_plastic_strain.reshape(
                len(chunk), point_count
            )
            second_piola = batch.second_piola_stress.reshape(
                len(chunk), point_count, 3, 3
            )
            kirchhoff = batch.kirchhoff_stress.reshape(
                len(chunk), point_count, 3, 3
            )
            trial_plastic.flags.writeable = False
            trial_alpha.flags.writeable = False
            state_changed = np.not_equal(
                trial_alpha,
                committed_alpha,
            )
            state_changed |= np.any(
                np.not_equal(trial_plastic, committed_plastic),
                axis=(2, 3),
            )
            measure_scales = np.asarray(
                [
                    self._reference_measure_scale[
                        id(binding.reference_coordinates)
                    ]
                    for binding in chunk
                ],
                dtype=float,
            )
            point_weights = self._reference_weight_cache[id(first_geometry)]
            if all(geometry is first_geometry for geometry in reference_geometries):
                scales = measure_scales[:, None] * point_weights[None, :]
            else:
                scales = np.asarray(
                    [
                        measure_scale
                        * self._reference_weight_cache[id(geometry)]
                        for measure_scale, geometry in zip(
                            measure_scales,
                            reference_geometries,
                            strict=True,
                        )
                    ],
                    dtype=float,
                )
            staged_entries = []
            staged_states: list[tuple[dict[str, object], ...]] = []
            for element_index, binding in enumerate(chunk):
                element_id = int(binding.entity.id)
                keys = self._state_keys(
                    element_id,
                    binding.state_namespace,
                )
                element_states: list[dict[str, object]] = []
                for point_index, key in enumerate(keys):
                    trial_state: dict[str, object] = {
                        "plastic_log_strain": trial_plastic[
                            element_index,
                            point_index,
                        ],
                        "equivalent_plastic_strain": float(
                            trial_alpha[element_index, point_index]
                        ),
                    }
                    if state_changed[element_index, point_index]:
                        staged_entries.append((key, trial_state))
                    if capture_outputs:
                        element_states.append(trial_state)
                if capture_outputs:
                    staged_states.append(tuple(element_states))
            stage_immutable_batch = getattr(
                state,
                "stage_immutable_batch",
                None,
            )
            if callable(stage_immutable_batch):
                stage_immutable_batch(staged_entries)
            else:
                stage_immutable = getattr(state, "stage_immutable", None)
                for key, trial_state in staged_entries:
                    if callable(stage_immutable):
                        stage_immutable(key, trial_state)
                    else:
                        state.stage(key, trial_state)
            local_residual = np.einsum(
                "epij,epja,ep->eia",
                stress[:, :, :2, :2],
                gradients,
                scales,
            )
            local_size = 8
            local_tangent = np.zeros(
                (len(chunk), local_size, local_size),
                dtype=float,
            )
            if need_tangent:
                delta_f = np.zeros(
                    (len(chunk), point_count, local_size, 3, 3),
                    dtype=float,
                )
                gradients_transposed = gradients.transpose(0, 1, 3, 2)
                for component in range(2):
                    delta_f[:, :, component::2, component, :2] = (
                        gradients_transposed
                    )
                delta_p = np.einsum(
                    "epij,eplj->epli",
                    tangent,
                    delta_f.reshape(len(chunk), point_count, local_size, 9),
                ).reshape(len(chunk), point_count, local_size, 3, 3)
                point_tangent = np.einsum(
                    "eplij,epja->eplia",
                    delta_p[:, :, :, :2, :2],
                    gradients,
                )
                point_tangent = np.transpose(
                    point_tangent,
                    (0, 1, 4, 3, 2),
                ).reshape(
                    len(chunk),
                    point_count,
                    local_size,
                    local_size,
                )
                local_tangent = np.einsum(
                    "ep,epij->eij",
                    scales,
                    point_tangent,
                )

            output_batches: tuple[LocalOutputBatch, ...] = ()
            if capture_outputs:
                green_lagrange = 0.5 * (
                    np.einsum(
                        "epji,epjk->epik",
                        deformation,
                        deformation,
                    )
                    - np.eye(3, dtype=float)
                )
                jacobian = np.linalg.det(deformation)
                cauchy = kirchhoff / jacobian[:, :, None, None]
                radial_return = trial_alpha > (
                    committed_alpha + 1.0e-14
                )
                output_list: list[LocalOutputBatch] = []
                for element_index, binding in enumerate(chunk):
                    element_id = int(binding.entity.id)
                    points: list[LocalPointOutput] = []
                    for point_index, point in enumerate(
                        self._reference_geometry[
                            id(binding.reference_coordinates)
                        ],
                        start=0,
                    ):
                        trial_state = staged_states[element_index][point_index]
                        fields = {
                            "second_piola_stress": second_piola[
                                element_index,
                                point_index,
                            ],
                            "kirchhoff_stress": kirchhoff[
                                element_index,
                                point_index,
                            ],
                            "return_algorithm": (
                                "radial_return"
                                if radial_return[element_index, point_index]
                                else "elastic"
                            ),
                            "tangent_algorithm": "analytic_objective",
                            "deformation_gradient": deformation[
                                element_index,
                                point_index,
                            ],
                            "green_lagrange_strain": green_lagrange[
                                element_index,
                                point_index,
                            ],
                            "first_piola_stress": stress[
                                element_index,
                                point_index,
                            ],
                            "cauchy_stress": cauchy[
                                element_index,
                                point_index,
                            ],
                            "jacobian": float(
                                jacobian[element_index, point_index]
                            ),
                            "equivalent_plastic_strain": float(
                                trial_alpha[element_index, point_index]
                            ),
                        }
                        points.append(
                            LocalPointOutput._from_immutable(
                                entity_kind="element",
                                entity_id=element_id,
                                point_id=point_index + 1,
                                local_coordinates=point.natural_coordinates,
                                weight=point.weight,
                                fields=fields,
                                history=trial_state,
                            )
                        )
                    output_list.append(
                        LocalOutputBatch._from_immutable(
                            element_id,
                            tuple(points),
                        )
                    )
                output_batches = tuple(output_list)
            yield LocalContributionBatch(
                binding_indices=np.arange(
                    chunk_start,
                    chunk_start + len(chunk),
                    dtype=int,
                ),
                dofs=np.asarray(
                    [binding.dofs for binding in chunk],
                    dtype=int,
                ),
                residual=local_residual.transpose(0, 2, 1).reshape(
                    len(chunk),
                    -1,
                ),
                tangent=local_tangent,
                output_batches=output_batches,
            )


def _build_reference_geometry(
    definition: ElementDefinition,
    reference: np.ndarray,
) -> tuple[_ReferencePointData, ...]:
    """Build immutable reference integration data once per compiled binding."""

    if reference.ndim != 2 or reference.shape[1] not in (2, 3):
        raise ValueError("reference coordinates must have 2 or 3 spatial dimensions")
    if reference.shape[0] != int(definition.node_count):
        raise ValueError(
            f"{definition.canonical_type} requires {int(definition.node_count)} nodes"
        )
    spatial_dimension = int(reference.shape[1])
    points: list[_ReferencePointData] = []
    for point in definition.gauss_points():
        natural_coordinates = tuple(float(value) for value in point[:-1])
        weight = float(point[-1])
        natural_gradients = np.asarray(
            definition.shape_gradients(*natural_coordinates),
            dtype=float,
        )
        if natural_gradients.shape != (spatial_dimension, int(definition.node_count)):
            raise ValueError(
                f"{definition.canonical_type} shape gradients must have "
                f"shape ({spatial_dimension}, {int(definition.node_count)})"
            )
        reference_jacobian = natural_gradients @ reference
        reference_det = float(np.linalg.det(reference_jacobian))
        if reference_det <= 0.0:
            raise ValueError(
                f"{definition.canonical_type} reference Jacobian determinant must be > 0"
            )
        reference_gradients = np.asarray(
            np.linalg.solve(reference_jacobian, natural_gradients),
            dtype=float,
        )
        reference_gradients.flags.writeable = False
        points.append(
            _ReferencePointData(
                natural_coordinates=natural_coordinates,
                weight=weight,
                reference_gradients=reference_gradients,
                reference_det=reference_det,
            )
        )
    return tuple(points)


def _reference_geometry_signature(
    reference: np.ndarray,
) -> tuple[tuple[int, ...], bytes]:
    """Return an exact translation-invariant key for reference geometry.

    Reference Jacobians are unchanged by a rigid translation of every node.
    Normalizing by the first node lets structured meshes share their
    immutable quadrature template without accepting any tolerance-based
    geometric equivalence.  The exact byte representation keeps distinct
    floating-point geometries on separate paths.
    """

    normalized = np.ascontiguousarray(reference - reference[0])
    return tuple(int(value) for value in normalized.shape), normalized.tobytes()


def _integrate_total_lagrangian_point(
    response: MaterialResponse,
    reference_gradients: np.ndarray,
    scale: float,
    spatial_dimension: int,
    residual: np.ndarray,
    tangent: np.ndarray,
    *,
    include_tangent: bool = True,
) -> None:
    """Integrate a ``P + dP/dF`` response in active spatial dimensions."""

    response.require(
        stress_measure=StressMeasure.FIRST_PIOLA,
        tangent_input=KinematicMeasure.DEFORMATION_GRADIENT,
        stress_shape=(3, 3),
        tangent_shape=(9, 9),
    )
    first_piola = np.asarray(response.stress, dtype=float)
    node_count = int(reference_gradients.shape[1])
    active_first_piola = first_piola[
        :spatial_dimension,
        :spatial_dimension,
    ]
    residual[:] += (
        active_first_piola @ reference_gradients
    ).T.reshape(-1) * scale
    if not include_tangent:
        return
    constitutive = np.asarray(response.tangent, dtype=float)
    local_size = spatial_dimension * node_count
    component_indices = np.tile(np.arange(spatial_dimension), node_count)
    gradients_by_column = np.repeat(
        reference_gradients.T,
        spatial_dimension,
        axis=0,
    )
    delta_f = np.zeros((local_size, 3, 3), dtype=float)
    delta_f[
        np.arange(local_size),
        component_indices,
        :spatial_dimension,
    ] = gradients_by_column
    delta_p = np.einsum(
        "ij,cj->ci",
        constitutive,
        delta_f.reshape(local_size, 9),
    ).reshape(local_size, 3, 3)
    column_contributions = np.einsum(
        "cij,ja->cia",
        delta_p[:, :spatial_dimension, :spatial_dimension],
        reference_gradients,
    )
    tangent[:, :] += (
        np.transpose(column_contributions, (0, 2, 1))
        .reshape(local_size, local_size)
        .T
        * scale
    )


def _integrate_small_strain_point(
    response: MaterialResponse,
    reference_gradients: np.ndarray,
    scale: float,
    spatial_dimension: int,
    residual: np.ndarray,
    tangent: np.ndarray,
    *,
    include_tangent: bool = True,
) -> None:
    """Integrate a Cauchy stress and ``dσ/dε`` in the reference geometry."""

    response.require(
        stress_measure=StressMeasure.CAUCHY,
        tangent_input=KinematicMeasure.SMALL_STRAIN,
        stress_shape=(3, 3),
        tangent_shape=(6, 6),
    )
    stress = _engineering_stress(np.asarray(response.stress, dtype=float))
    matrix = _small_strain_b_matrix(reference_gradients, spatial_dimension)
    active = (0, 1, 3) if spatial_dimension == 2 else tuple(range(6))
    residual[:] += matrix.T @ stress[np.asarray(active, dtype=int)] * scale
    if not include_tangent:
        return
    constitutive = np.asarray(response.tangent, dtype=float)
    tangent[:, :] += (
        matrix.T
        @ constitutive[np.ix_(active, active)]
        @ matrix
        * scale
    )


def _small_strain_b_matrix(
    reference_gradients: np.ndarray,
    spatial_dimension: int,
) -> np.ndarray:
    node_count = int(reference_gradients.shape[1])
    local_size = spatial_dimension * node_count
    if spatial_dimension == 2:
        matrix = np.zeros((3, local_size), dtype=float)
        columns = 2 * np.arange(node_count)
        matrix[0, columns] = reference_gradients[0]
        matrix[1, columns + 1] = reference_gradients[1]
        matrix[2, columns] = reference_gradients[1]
        matrix[2, columns + 1] = reference_gradients[0]
        return matrix
    if spatial_dimension == 3:
        matrix = np.zeros((6, local_size), dtype=float)
        columns = 3 * np.arange(node_count)
        matrix[0, columns] = reference_gradients[0]
        matrix[1, columns + 1] = reference_gradients[1]
        matrix[2, columns + 2] = reference_gradients[2]
        matrix[3, columns] = reference_gradients[1]
        matrix[3, columns + 1] = reference_gradients[0]
        matrix[4, columns] = reference_gradients[2]
        matrix[4, columns + 2] = reference_gradients[0]
        matrix[5, columns + 1] = reference_gradients[2]
        matrix[5, columns + 2] = reference_gradients[1]
        return matrix
    raise ValueError(f"unsupported small-strain spatial dimension {spatial_dimension}")


def _small_strain_record(
    element_id: int,
    point_id: int,
    natural_coordinates: tuple[float, ...],
    weight: float,
    kinematic: Any,
    response: MaterialResponse,
) -> LocalPointOutput:
    strain = _embed_tensor(np.asarray(kinematic.small_strain, dtype=float))
    stress = np.asarray(response.stress, dtype=float)
    fields = {
        **dict(response.outputs),
        "small_strain": strain,
        "cauchy_stress": stress,
        "deformation_gradient": _embed_deformation_gradient(
            np.asarray(kinematic.deformation_gradient, dtype=float)
        ),
        "green_lagrange_strain": strain,
    }
    if "equivalent_plastic_strain" in response.trial_state:
        fields["equivalent_plastic_strain"] = float(
            response.trial_state["equivalent_plastic_strain"]
        )
    return LocalPointOutput.element_point(
        element_id=element_id,
        integration_point=point_id,
        natural_coordinates=natural_coordinates,
        weight=weight,
        fields=fields,
        history=response.trial_state,
    )


def _total_lagrangian_record(
    element_id: int,
    point_id: int,
    natural_coordinates: tuple[float, ...],
    weight: float,
    kinematic: Any,
    response: MaterialResponse,
) -> LocalPointOutput:
    """Create one common result record for plane and 3D TL materials."""

    deformation_gradient = _embed_deformation_gradient(
        np.asarray(kinematic.deformation_gradient, dtype=float)
    )
    green_lagrange = _embed_tensor(
        np.asarray(kinematic.green_lagrange_strain, dtype=float)
    )
    first_piola = np.asarray(response.stress, dtype=float)
    second_piola_output = response.outputs.get("second_piola_stress")
    if second_piola_output is None:
        second_piola = np.asarray(
            np.linalg.solve(deformation_gradient, first_piola),
            dtype=float,
        )
    else:
        second_piola = np.asarray(second_piola_output, dtype=float)
    jacobian = float(np.linalg.det(deformation_gradient))
    kirchhoff_output = response.outputs.get("kirchhoff_stress")
    if kirchhoff_output is None:
        kirchhoff = np.asarray(
            first_piola @ deformation_gradient.T,
            dtype=float,
        )
    else:
        kirchhoff = np.asarray(kirchhoff_output, dtype=float)
    fields = {
        **dict(response.outputs),
        "deformation_gradient": deformation_gradient,
        "green_lagrange_strain": green_lagrange,
        "first_piola_stress": first_piola,
        "second_piola_stress": second_piola,
        "kirchhoff_stress": kirchhoff,
        "cauchy_stress": kirchhoff / jacobian,
        "jacobian": jacobian,
    }
    if "equivalent_plastic_strain" in response.trial_state:
        fields["equivalent_plastic_strain"] = float(
            response.trial_state["equivalent_plastic_strain"]
        )
    return LocalPointOutput.element_point(
        element_id=element_id,
        integration_point=point_id,
        natural_coordinates=natural_coordinates,
        weight=weight,
        fields=fields,
        history=response.trial_state,
    )


def _embed_deformation_gradient(value: np.ndarray) -> np.ndarray:
    if value.shape == (3, 3):
        return np.array(value, dtype=float, copy=True)
    if value.shape != (2, 2):
        raise ValueError("deformation gradient must be 2x2 or 3x3")
    result = np.eye(3, dtype=float)
    result[:2, :2] = value
    return result


def _embed_tensor(value: np.ndarray) -> np.ndarray:
    if value.shape == (3, 3):
        return np.array(value, dtype=float, copy=True)
    if value.shape != (2, 2):
        raise ValueError("continuum tensor must be 2x2 or 3x3")
    result = np.zeros((3, 3), dtype=float)
    result[:2, :2] = value
    return result


def _measure_scale(
    properties: Mapping[str, Any],
    spatial_dimension: int,
    *,
    element_id: int,
) -> float:
    if spatial_dimension == 2:
        return plane_thickness(properties, element_id=element_id)
    if spatial_dimension == 3:
        return 1.0
    raise ValueError(f"unsupported continuum dimension {spatial_dimension}")


def _coordinates(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D array")
    return array


def _material(resources: Mapping[str, Any]) -> MaterialModel:
    if not isinstance(resources, Mapping):
        raise TypeError("continuum mechanics resources must be a mapping")
    material = resources.get("material")
    if not isinstance(material, MaterialModel):
        raise TypeError("continuum mechanics requires resource 'material'")
    return material


def _material_fields(
    fields: Mapping[str, LocalFieldState],
    context: EvaluationContext,
) -> dict[str, Any]:
    coupled = {
        name: local.values
        for name, local in fields.items()
        if name != "U"
    }
    coupled.update(context.parameters)
    return coupled


def _is_small_strain_kinematics(kinematics: Any) -> bool:
    measure = getattr(kinematics, "kinematic_measure", None)
    return measure in {
        KinematicMeasure.SMALL_STRAIN,
        KinematicMeasure.SMALL_STRAIN.value,
    }


def _engineering_stress(stress: np.ndarray) -> np.ndarray:
    if stress.shape != (3, 3):
        raise ValueError("small-strain stress must have shape (3, 3)")
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[0, 1],
            stress[0, 2],
            stress[1, 2],
        ],
        dtype=float,
    )


__all__ = ["ContinuumMechanicsOperator"]
