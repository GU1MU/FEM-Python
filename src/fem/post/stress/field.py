from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from ...physics.mechanics import get_recovery_service
from ...materials import linear_elastic
from ...physics.mechanics.operators.continuum_quad4 import (
    quad4_gauss_points,
    quad4_shape_grad_xi_eta,
)
from ...physics.mechanics.operators.plane import PlaneProperties
from ..averaging import NodalAveragingPolicy, resolve_nodal_stress
from ..fields import (
    MATERIAL_SIGNATURE_KEY as MATERIAL_SIGNATURE_KEY,
    ResultRegionKey,
    SECTION_SIGNATURE_KEY as SECTION_SIGNATURE_KEY,
    _result_region_key_from_compatible_signatures,
    result_region_key_for_element,
    result_region_sort_key,
)
from . import dispatch
from ._common import element_volume, node_lookup, validated_u
from .invariants import (
    StressInvariants,
    complete_plane_components,
    derive_stress_invariants,
)


PLANE_COMPONENT_NAMES = ("sig_x", "sig_y", "tau_xy")
SOLID_COMPONENT_NAMES = (
    "sig_x",
    "sig_y",
    "sig_z",
    "tau_xy",
    "tau_yz",
    "tau_zx",
)
CANONICAL_PLANE_COMPONENT_NAMES = ("S11", "S22", "S33", "S12")
CANONICAL_SOLID_COMPONENT_NAMES = ("S11", "S22", "S33", "S12", "S23", "S13")


def _validate_checkpoint(
    checkpoint: Callable[[], None] | None,
) -> Callable[[], None] | None:
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    return checkpoint


def _run_checkpoint(checkpoint: Callable[[], None] | None) -> None:
    if checkpoint is not None:
        checkpoint()


def _recovery_model(result: Any) -> Any:
    """Return the compiled effective-property model when a result has one."""

    compiled_model = getattr(result, "compiled_model", None)
    return result.model if compiled_model is None else compiled_model


class StressPosition(str, Enum):
    """Supported locations for continuum stress output."""

    INTEGRATION_POINT = "integration_point"
    CENTROID = "centroid"
    ELEMENT_NODAL = "element_nodal"
    NODAL = "nodal"


def StressRegionKey(
    material_signature: Any,
    section_signature: Any,
) -> ResultRegionKey:
    """Compatibility factory returning the sole result-region identity type."""

    return _result_region_key_from_compatible_signatures(
        material_signature,
        section_signature,
    )


@dataclass(frozen=True)
class StressRecord:
    """One complete stress tensor and its derived values at one result location."""

    position: StressPosition
    coordinates: tuple[float, ...]
    components: tuple[float, ...]
    invariants: StressInvariants
    elem_id: int | None = None
    integration_point: int | None = None
    natural_coordinates: tuple[float, ...] | None = None
    node_id: int | None = None
    local_node: int | None = None
    region_key: ResultRegionKey | None = None
    weight: float = 1.0
    displacement: tuple[float, ...] | None = None
    averaged: bool | None = None

    def values(self, component_names: Sequence[str]) -> dict[str, float]:
        """Return named components and invariants for exporters and GUI consumers."""
        values = {
            str(name): float(value)
            for name, value in zip(component_names, self.components)
        }
        values.update({
            "Mises": self.invariants.mises,
            "MaxPrincipal": self.invariants.max_principal,
            "MidPrincipal": self.invariants.mid_principal,
            "MinPrincipal": self.invariants.min_principal,
        })
        return values


@dataclass(frozen=True)
class StressField:
    """A deterministic collection of stress records at one result position."""

    position: StressPosition
    component_names: tuple[str, ...]
    records: tuple[StressRecord, ...]


@dataclass(frozen=True, slots=True)
class PlaneElementNodalRecord:
    """Lightweight canonical plane row prepared for legacy CSV projections."""

    components: tuple[float, float, float, float]
    elem_id: int
    node_id: int
    local_node: int
    region_key: ResultRegionKey
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PlaneElementNodalField:
    """Plane element-nodal rows without unused invariant/result metadata."""

    records: tuple[PlaneElementNodalRecord, ...]
    position: ClassVar[StressPosition] = StressPosition.ELEMENT_NODAL
    component_names: ClassVar[tuple[str, ...]] = CANONICAL_PLANE_COMPONENT_NAMES


@dataclass(frozen=True)
class _ElementIntegrationPointField:
    elem: Any
    type_key: str
    kernel: Any
    gauss_order: int | None
    natural_coordinates: np.ndarray
    components: np.ndarray
    region_key: ResultRegionKey
    weight: float


class StressRecovery:
    """Cache canonical integration-point components for repeated position recovery."""

    def __init__(
        self,
        mesh: Any,
        U: Sequence[float],
        element_type: str | None = None,
        gauss_order: int | None = None,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        type_keys = dispatch.resolve_type_keys(mesh, element_type)
        group = dispatch.stress_group_for_keys(type_keys)
        if group not in {"plane", "solid"}:
            raise ValueError(
                "StressPosition output currently supports plane and solid elements only"
            )
        self.mesh = mesh
        self.lookup = node_lookup(mesh)
        self.component_names = (
            CANONICAL_PLANE_COMPONENT_NAMES
            if group == "plane"
            else CANONICAL_SOLID_COMPONENT_NAMES
        )
        self._U = validated_u(mesh, U)
        self._ip_fields = _collect_element_integration_points(
            mesh,
            self._U,
            self.lookup,
            set(type_keys),
            group,
            gauss_order,
            checkpoint,
        )
        _run_checkpoint(checkpoint)
        self._cache: dict[StressPosition, StressField] = {}

    def collect(
        self,
        position: StressPosition | str = StressPosition.INTEGRATION_POINT,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> StressField:
        """Recover and cache one requested stress position."""
        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        try:
            resolved_position = StressPosition(position)
        except ValueError as exc:
            choices = ", ".join(item.value for item in StressPosition)
            raise ValueError(
                f"Unsupported stress position {position!r}; expected one of {choices}"
            ) from exc
        cached = self._cache.get(resolved_position)
        if cached is not None:
            _run_checkpoint(checkpoint)
            return cached

        if resolved_position is StressPosition.INTEGRATION_POINT:
            records = _integration_point_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is StressPosition.CENTROID:
            records = _centroid_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is StressPosition.ELEMENT_NODAL:
            records = _element_nodal_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                self._U,
                checkpoint,
            )
        else:
            element_nodal_field = self.collect(
                StressPosition.ELEMENT_NODAL,
                checkpoint=checkpoint,
            )
            records = _average_nodal_records(
                self.mesh,
                self.lookup,
                element_nodal_field.records,
                self.component_names,
                checkpoint,
            )
        stress_field = StressField(
            resolved_position,
            self.component_names,
            tuple(records),
        )
        _run_checkpoint(checkpoint)
        self._cache[resolved_position] = stress_field
        return stress_field


class CapturedStressRecovery:
    """Recover continuum stress from captured integration-point outputs.

    The linear StressRecovery path evaluates a small-strain constitutive
    response from nodal displacement. Nonlinear frames already contain the
    committed integration-point response, so this adapter only performs the
    spatial projection from those captured values. The captured contract is
    shared by small-strain material nonlinearity and finite-strain geometry;
    geometry mode is selected by the producer, not by this recovery service.
    """

    def __init__(
        self,
        result: Any,
        gauss_order: int | None = None,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        self.mesh = _recovery_model(result).mesh
        self._U = validated_u(self.mesh, result.U)
        self.lookup = node_lookup(self.mesh)
        type_keys = dispatch.resolve_type_keys(self.mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
        if group not in {"plane", "solid"}:
            raise ValueError(
                "captured stress recovery requires a continuum mesh"
            )
        self.component_names = (
            CANONICAL_PLANE_COMPONENT_NAMES
            if group == "plane"
            else CANONICAL_SOLID_COMPONENT_NAMES
        )
        self._ip_fields = _captured_finite_strain_integration_points(
            result,
            self.mesh,
            self.lookup,
            self.component_names,
            gauss_order,
            checkpoint,
        )
        self._cache: dict[StressPosition, StressField] = {}

    def collect(
        self,
        position: StressPosition | str = StressPosition.INTEGRATION_POINT,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> StressField:
        """Project captured stress to one supported location."""

        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        try:
            resolved_position = StressPosition(position)
        except ValueError as exc:
            choices = ", ".join(item.value for item in StressPosition)
            raise ValueError(
                f"Unsupported stress position {position!r}; expected one of {choices}"
            ) from exc
        cached = self._cache.get(resolved_position)
        if cached is not None:
            return cached
        if resolved_position is StressPosition.INTEGRATION_POINT:
            records = _integration_point_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is StressPosition.CENTROID:
            records = _centroid_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is StressPosition.ELEMENT_NODAL:
            records = _element_nodal_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                self._U,
                checkpoint,
            )
        else:
            element_nodal_field = self.collect(
                StressPosition.ELEMENT_NODAL,
                checkpoint=checkpoint,
            )
            records = _average_nodal_records(
                self.mesh,
                self.lookup,
                element_nodal_field.records,
                self.component_names,
                checkpoint,
            )
        stress_field = StressField(
            resolved_position,
            self.component_names,
            tuple(records),
        )
        self._cache[resolved_position] = stress_field
        return stress_field


FiniteStrainStressRecovery = CapturedStressRecovery


class FiniteStrainStatePosition(str, Enum):
    """Spatial positions supported by captured finite-strain state fields."""

    INTEGRATION_POINT = "integration_point"
    CENTROID = "centroid"
    ELEMENT_NODAL = "element_nodal"
    NODAL = "nodal"


@dataclass(frozen=True)
class FiniteStrainStateRecord:
    """One projected Green strain or scalar plastic-state value."""

    position: FiniteStrainStatePosition
    coordinates: tuple[float, ...]
    components: tuple[float, ...]
    elem_id: int | None = None
    integration_point: int | None = None
    natural_coordinates: tuple[float, ...] | None = None
    node_id: int | None = None
    local_node: int | None = None
    region_key: ResultRegionKey | None = None
    weight: float = 1.0
    displacement: tuple[float, ...] | None = None
    averaged: bool | None = None


@dataclass(frozen=True)
class FiniteStrainStateField:
    """Projected values of one captured finite-strain state variable."""

    position: FiniteStrainStatePosition
    component_names: tuple[str, ...]
    records: tuple[FiniteStrainStateRecord, ...]


class FiniteStrainStateRecovery:
    """Project captured finite-strain state from points to GUI field positions.

    The solver publishes integration-point values once.  This recovery object
    performs only spatial projection; it never recomputes constitutive state
    from nodal displacement.
    """

    _TENSOR_STATE = "green_lagrange_strain"
    _SCALAR_STATE = "equivalent_plastic_strain"

    def __init__(
        self,
        result: Any,
        state_variable: str,
        gauss_order: int | None = None,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        if state_variable not in {
            self._TENSOR_STATE,
            self._SCALAR_STATE,
        }:
            raise ValueError(
                "state_variable must be green_lagrange_strain or "
                "equivalent_plastic_strain"
            )
        self.mesh = _recovery_model(result).mesh
        self._U = validated_u(self.mesh, result.U)
        self.lookup = node_lookup(self.mesh)
        type_keys = dispatch.resolve_type_keys(self.mesh, None)
        group = dispatch.stress_group_for_keys(type_keys)
        if group not in {"plane", "solid"}:
            raise ValueError(
                "finite-strain state recovery requires a continuum mesh"
            )
        if state_variable == self._TENSOR_STATE:
            self.component_names = (
                ("E11", "E22", "E33", "E12")
                if group == "plane"
                else ("E11", "E22", "E33", "E12", "E23", "E13")
            )
        else:
            self.component_names = ("PEEQ",)
        self.state_variable = state_variable
        integration = _captured_finite_strain_integration_mapping(result)
        element_ids, point_numbers, natural, weights = (
            _captured_finite_strain_metadata(
                integration,
                natural_dimension=2 if group == "plane" else 3,
            )
        )
        if state_variable == self._TENSOR_STATE:
            raw_values = np.asarray(
                integration.get(state_variable),
                dtype=float,
            )
            if raw_values.shape != (len(element_ids), 3, 3):
                raise ValueError(
                    "green_lagrange_strain must have shape (n, 3, 3)"
                )
            components = _finite_strain_component_matrix(
                raw_values,
                self.component_names,
            )
        else:
            raw_values = np.asarray(
                integration.get(state_variable),
                dtype=float,
            )
            if raw_values.shape != (len(element_ids),):
                raise ValueError(
                    "equivalent_plastic_strain must have shape (n,)"
                )
            components = raw_values[:, None]
        if not np.all(np.isfinite(components)):
            raise ValueError("finite-strain state values must be finite")
        self._ip_fields = _build_captured_finite_strain_fields(
            self.mesh,
            element_ids,
            point_numbers,
            natural,
            components,
            weights,
            gauss_order,
            checkpoint,
        )
        self._cache: dict[
            FiniteStrainStatePosition,
            FiniteStrainStateField,
        ] = {}

    def collect(
        self,
        position: FiniteStrainStatePosition | str = (
            FiniteStrainStatePosition.INTEGRATION_POINT
        ),
        *,
        averaging_policy: NodalAveragingPolicy | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> FiniteStrainStateField:
        """Return one captured state field at the requested position."""

        checkpoint = _validate_checkpoint(checkpoint)
        _run_checkpoint(checkpoint)
        if (
            averaging_policy is not None
            and type(averaging_policy) is not NodalAveragingPolicy
        ):
            raise TypeError(
                "averaging_policy must be NodalAveragingPolicy or None"
            )
        try:
            resolved_position = FiniteStrainStatePosition(position)
        except ValueError as exc:
            choices = ", ".join(item.value for item in FiniteStrainStatePosition)
            raise ValueError(
                f"Unsupported finite-strain state position {position!r}; "
                f"expected one of {choices}"
            ) from exc
        cacheable = averaging_policy is None
        cached = self._cache.get(resolved_position) if cacheable else None
        if cached is not None:
            return cached
        if resolved_position is FiniteStrainStatePosition.INTEGRATION_POINT:
            records = _state_integration_point_records(
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is FiniteStrainStatePosition.CENTROID:
            records = _state_centroid_records(
                self.lookup,
                self._ip_fields,
                self.component_names,
                checkpoint,
            )
        elif resolved_position is FiniteStrainStatePosition.ELEMENT_NODAL:
            records = _state_element_nodal_records(
                self.mesh,
                self.lookup,
                self._ip_fields,
                self.component_names,
                self._U,
                checkpoint,
            )
        else:
            element_nodal = self.collect(
                FiniteStrainStatePosition.ELEMENT_NODAL,
                checkpoint=checkpoint,
            )
            records = _state_average_nodal_records(
                self.mesh,
                self.lookup,
                element_nodal.records,
                policy=averaging_policy,
                checkpoint=checkpoint,
            )
        field = FiniteStrainStateField(
            resolved_position,
            self.component_names,
            tuple(records),
        )
        if cacheable:
            self._cache[resolved_position] = field
        return field


@dataclass(frozen=True)
class ElementNodalStressContribution:
    """One element-local stress tensor recovered at a mesh node."""

    node_id: int
    elem_id: int
    local_node: int
    components: tuple[float, ...]
    weight: float
    region_key: ResultRegionKey
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class NodalStressField:
    """Raw element-nodal stress contributions grouped in mesh-node order."""

    component_names: tuple[str, ...]
    contributions_by_node: Mapping[int, tuple[ElementNodalStressContribution, ...]]
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedNodalStressRow:
    """One averaged or element-local nodal stress value for export."""

    node_id: int
    components: tuple[float, ...]
    elem_id: int | None
    local_node: int | None
    averaged: bool
    plane_type: str | None = None
    poisson_ratio: float | None = None


@dataclass(frozen=True)
class ResolvedNodalStressField:
    """Resolved nodal stress rows in deterministic mesh and element order."""

    component_names: tuple[str, ...]
    rows: tuple[ResolvedNodalStressRow, ...]


def collect_stress(
    mesh: Any,
    U: Sequence[float],
    position: StressPosition | str = StressPosition.INTEGRATION_POINT,
    element_type: str | None = None,
    gauss_order: int | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> StressField:
    """Collect one continuum stress field from the canonical integration-point data."""
    recovery = StressRecovery(
        mesh,
        U,
        element_type=element_type,
        gauss_order=gauss_order,
        checkpoint=checkpoint,
    )
    return recovery.collect(
        position,
        checkpoint=checkpoint,
    )


def collect_plane_element_nodal(
    mesh: Any,
    U: Sequence[float],
    element_type: str | None = None,
    gauss_order: int | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> PlaneElementNodalField:
    """Recover only the plane element-nodal values needed by legacy CSVs."""

    checkpoint = _validate_checkpoint(checkpoint)
    _run_checkpoint(checkpoint)
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if dispatch.stress_group_for_keys(type_keys) != "plane":
        raise ValueError("plane CSV recovery requires plane elements")
    selected = set(type_keys)
    displacement = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    records: list[PlaneElementNodalRecord] = []
    region_cache: dict[tuple[tuple[object, object], ...], ResultRegionKey] = {}

    for elem in mesh.elements:
        _run_checkpoint(checkpoint)
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in selected:
            continue
        kernel = get_recovery_service(elem.type)
        order = (
            gauss_order
            if gauss_order is not None
            else dispatch.default_gauss_order(type_key)
        )
        if order is None:
            raw_values, plane_type, poisson_ratio = kernel.nodal_stress(
                mesh,
                elem,
                displacement,
                lookup,
            )
        else:
            raw_values, plane_type, poisson_ratio = kernel.nodal_stress(
                mesh,
                elem,
                displacement,
                lookup,
                order,
            )
        values = np.asarray(raw_values, dtype=float)
        if values.shape != (len(elem.node_ids), 3):
            raise ValueError(
                f"Element {elem.id} nodal stress shape {values.shape} "
                f"does not match ({len(elem.node_ids)}, 3)"
            )
        if not np.isfinite(values).all():
            raise ValueError("plane element-nodal stress components must be finite")
        try:
            props_key = tuple(sorted(elem.props.items()))
            region_key = region_cache.get(props_key)
        except TypeError:
            props_key = None
            region_key = None
        if region_key is None:
            region_key = result_region_key_for_element(elem)
            if props_key is not None:
                region_cache[props_key] = region_key
        for local_node, (node_id, components) in enumerate(
            zip(elem.node_ids, values),
            start=1,
        ):
            _run_checkpoint(checkpoint)
            sig_x = float(components[0])
            sig_y = float(components[1])
            tau_xy = float(components[2])
            records.append(
                PlaneElementNodalRecord(
                    components=(
                        sig_x,
                        sig_y,
                        (
                            float(poisson_ratio) * (sig_x + sig_y)
                            if plane_type == "strain"
                            else 0.0
                        ),
                        tau_xy,
                    ),
                    elem_id=int(elem.id),
                    node_id=int(node_id),
                    local_node=local_node,
                    region_key=region_key,
                )
            )

    _run_checkpoint(checkpoint)
    return PlaneElementNodalField(tuple(records))


def _collect_element_integration_points(
    mesh: Any,
    U: np.ndarray,
    lookup: dict[int, Any],
    selected: set[str],
    group: str,
    gauss_order: int | None,
    checkpoint: Callable[[], None] | None,
) -> list[_ElementIntegrationPointField]:
    selected_elements = [
        elem
        for elem in mesh.elements
        if dispatch.type_key_from_name(elem.type) in selected
    ]
    if (
        group == "plane"
        and selected == {"quad4"}
        and selected_elements
        and gauss_order in (None, 2)
    ):
        return _collect_quad4_integration_points_batch(
            mesh,
            U,
            lookup,
            selected_elements,
            checkpoint,
        )
    fields: list[_ElementIntegrationPointField] = []
    region_cache: dict[tuple[tuple[object, object], ...], ResultRegionKey] = {}
    for elem in mesh.elements:
        _run_checkpoint(checkpoint)
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in selected:
            continue
        kernel = get_recovery_service(elem.type)
        order = (
            gauss_order
            if gauss_order is not None
            else dispatch.default_gauss_order(type_key)
        )
        if order is None:
            natural, raw_components = kernel.integration_point_stress(
                mesh, elem, U, lookup
            )
        else:
            natural, raw_components = kernel.integration_point_stress(
                mesh, elem, U, lookup, order
            )
        _run_checkpoint(checkpoint)
        raw_values = np.asarray(raw_components, dtype=float)
        if group == "plane":
            plane_type, nu = kernel._plane_data(elem)
            complete = np.empty((len(raw_values), 4), dtype=float)
            complete[:, :2] = raw_values[:, :2]
            complete[:, 2] = (
                float(nu) * (raw_values[:, 0] + raw_values[:, 1])
                if plane_type == "strain"
                else 0.0
            )
            complete[:, 3] = raw_values[:, 2]
        else:
            complete = raw_values
        expected = len(
            CANONICAL_PLANE_COMPONENT_NAMES
            if group == "plane"
            else CANONICAL_SOLID_COMPONENT_NAMES
        )
        if complete.ndim != 2 or complete.shape[1] != expected:
            raise ValueError(
                f"Element {elem.id} integration-point stress shape {complete.shape} "
                f"does not have {expected} components"
            )
        weight = (
            element_volume(mesh, elem, lookup)
            if type_key in {"tet4", "tet10"}
            else 1.0
        )
        try:
            props_key = tuple(sorted(elem.props.items()))
            region_key = region_cache.get(props_key)
        except TypeError:
            props_key = None
            region_key = None
        if region_key is None:
            region_key = result_region_key_for_element(elem)
            if props_key is not None:
                region_cache[props_key] = region_key
        fields.append(
            _ElementIntegrationPointField(
                elem=elem,
                type_key=type_key,
                kernel=kernel,
                gauss_order=order,
                natural_coordinates=np.asarray(natural, dtype=float),
                components=complete,
                region_key=region_key,
                weight=float(weight),
            )
        )
    return fields


def _collect_quad4_integration_points_batch(
    mesh: Any,
    U: np.ndarray,
    lookup: dict[int, Any],
    elements: Sequence[Any],
    checkpoint: Callable[[], None] | None,
) -> list[_ElementIntegrationPointField]:
    """Recover homogeneous Quad4 plane stress in one NumPy block.

    This is algebraically the same B-matrix, material matrix and Gauss rule
    used by :class:`Quad4LinearOperator`.  Only the traversal changes: all
    element Jacobians, strains and stresses are evaluated as array batches;
    public ``StressField`` records are still built below with the original
    identities and ordering.  Mixed element families and non-default rules
    intentionally remain on the general recovery path.
    """

    count = len(elements)
    reference = np.asarray(
        [
            [
                (float(lookup[int(node_id)].x), float(lookup[int(node_id)].y))
                for node_id in elem.node_ids
            ]
            for elem in elements
        ],
        dtype=float,
    )
    if reference.shape != (count, 4, 2) or not np.all(np.isfinite(reference)):
        raise ValueError("Quad4 reference coordinates must be finite (n, 4, 2)")
    element_displacements = np.asarray(
        [U[np.asarray(mesh.element_dofs(elem), dtype=int)] for elem in elements],
        dtype=float,
    )
    if element_displacements.shape != (count, 8):
        raise ValueError("Quad4 element displacement blocks must have shape (n, 8)")

    gauss = tuple(quad4_gauss_points(2))
    natural = np.asarray([(xi, eta) for xi, eta, _weight in gauss], dtype=float)
    natural_gradients = np.asarray(
        [quad4_shape_grad_xi_eta(xi, eta) for xi, eta, _weight in gauss],
        dtype=float,
    )
    jacobians = np.einsum(
        "qia,eaj->eqij",
        natural_gradients,
        reference,
    )
    determinants = (
        jacobians[:, :, 0, 0] * jacobians[:, :, 1, 1]
        - jacobians[:, :, 0, 1] * jacobians[:, :, 1, 0]
    )
    if np.any(~np.isfinite(determinants)) or np.any(determinants <= 0.0):
        invalid = int(np.flatnonzero(determinants <= 0.0)[0])
        raise ValueError(
            f"Element {elements[invalid].id} has non-positive Jacobian determinant"
        )
    inverse = np.empty_like(jacobians)
    inverse[:, :, 0, 0] = jacobians[:, :, 1, 1] / determinants
    inverse[:, :, 0, 1] = -jacobians[:, :, 0, 1] / determinants
    inverse[:, :, 1, 0] = -jacobians[:, :, 1, 0] / determinants
    inverse[:, :, 1, 1] = jacobians[:, :, 0, 0] / determinants
    gradients = np.einsum(
        "eqij,qja->eqia",
        inverse,
        natural_gradients,
    )
    B = np.zeros((count, len(gauss), 3, 8), dtype=float)
    columns = 2 * np.arange(4)
    B[:, :, 0, columns] = gradients[:, :, 0, :]
    B[:, :, 1, columns + 1] = gradients[:, :, 1, :]
    B[:, :, 2, columns] = gradients[:, :, 1, :]
    B[:, :, 2, columns + 1] = gradients[:, :, 0, :]
    strains = np.einsum("eqij,ej->eqi", B, element_displacements)

    matrices = np.empty((count, 3, 3), dtype=float)
    plane_strain = np.zeros(count, dtype=bool)
    poisson = np.empty(count, dtype=float)
    for index, elem in enumerate(elements):
        properties = PlaneProperties.from_mapping(
            elem.props,
            element_id=elem.id,
            element_type=getattr(elem, "type", None),
        )
        matrices[index] = linear_elastic.plane_matrix(
            properties.E,
            properties.nu,
            properties.plane_type,
        )
        plane_strain[index] = properties.plane_type == "strain"
        poisson[index] = properties.nu
    raw_components = np.einsum("eij,eqj->eqi", matrices, strains)
    complete = np.empty((count, len(gauss), 4), dtype=float)
    complete[:, :, :2] = raw_components[:, :, :2]
    complete[:, :, 2] = np.where(
        plane_strain[:, None],
        poisson[:, None] * (raw_components[:, :, 0] + raw_components[:, :, 1]),
        0.0,
    )
    complete[:, :, 3] = raw_components[:, :, 2]
    if not np.all(np.isfinite(complete)):
        raise ValueError("Quad4 integration-point stress components must be finite")

    kernel = get_recovery_service("Quad4")
    region_cache: dict[tuple[tuple[object, object], ...], ResultRegionKey] = {}
    fields: list[_ElementIntegrationPointField] = []
    for index, elem in enumerate(elements):
        _run_checkpoint(checkpoint)
        try:
            props_key = tuple(sorted(elem.props.items()))
            region_key = region_cache.get(props_key)
        except TypeError:
            props_key = None
            region_key = None
        if region_key is None:
            region_key = result_region_key_for_element(elem)
            if props_key is not None:
                region_cache[props_key] = region_key
        fields.append(
            _ElementIntegrationPointField(
                elem=elem,
                type_key="quad4",
                kernel=kernel,
                gauss_order=2,
                natural_coordinates=natural,
                components=complete[index],
                region_key=region_key,
                weight=1.0,
            )
        )
    return fields


def _captured_finite_strain_integration_points(
    result: Any,
    mesh: Any,
    lookup: dict[int, Any],
    component_names: tuple[str, ...],
    gauss_order: int | None,
    checkpoint: Callable[[], None] | None,
) -> list[_ElementIntegrationPointField]:
    """Build the spatial-recovery input from captured frame outputs."""

    integration = _captured_finite_strain_integration_mapping(result)
    element_ids, point_numbers, natural, weights = (
        _captured_finite_strain_metadata(
            integration,
            natural_dimension=2
            if len(component_names) == len(CANONICAL_PLANE_COMPONENT_NAMES)
            else 3,
        )
    )
    stresses = _captured_cauchy_stress(integration)
    if stresses.shape != (len(element_ids), 3, 3):
        raise ValueError("finite-strain cauchy_stress must have shape (n, 3, 3)")
    components = _finite_strain_component_matrix(
        stresses,
        component_names,
    )
    return _build_captured_finite_strain_fields(
        mesh,
        element_ids,
        point_numbers,
        natural,
        components,
        weights,
        gauss_order,
        checkpoint,
    )


def _captured_finite_strain_integration_mapping(
    result: Any,
) -> Mapping[str, Any]:
    outputs = getattr(result, "outputs", {})
    integration = outputs.get("integration_points")
    if not isinstance(integration, Mapping):
        raise ValueError(
            "finite-strain result frame is missing integration-point outputs"
        )
    required = {
        "element_id",
        "integration_point",
        "natural_coordinates",
    }
    missing = sorted(required - set(integration))
    if missing:
        raise ValueError(
            "finite-strain integration-point outputs are missing: "
            + ", ".join(missing)
        )
    return integration


def _captured_finite_strain_metadata(
    integration: Mapping[str, Any],
    *,
    natural_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    element_ids = np.asarray(integration["element_id"], dtype=int)
    point_numbers = np.asarray(integration["integration_point"], dtype=int)
    natural = np.asarray(integration["natural_coordinates"], dtype=float)
    weight_values = integration.get("weight")
    if weight_values is None:
        weight_values = np.ones(len(element_ids), dtype=float)
    weights = np.asarray(weight_values, dtype=float)
    if point_numbers.shape != (len(element_ids),):
        raise ValueError(
            "finite-strain integration-point number shape is invalid"
        )
    if natural.shape != (len(element_ids), natural_dimension):
        raise ValueError(
            "finite-strain natural_coordinates must have shape "
            f"(n, {natural_dimension})"
        )
    if weights.shape != (len(element_ids),):
        raise ValueError("finite-strain integration-point weight shape is invalid")
    if not (
        np.all(np.isfinite(natural))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    ):
        raise ValueError(
            "finite-strain integration metadata must be finite with positive weights"
        )
    return element_ids, point_numbers, natural, weights


def _build_captured_finite_strain_fields(
    mesh: Any,
    element_ids: np.ndarray,
    point_numbers: np.ndarray,
    natural: np.ndarray,
    components: np.ndarray,
    weights: np.ndarray,
    gauss_order: int | None,
    checkpoint: Callable[[], None] | None,
) -> list[_ElementIntegrationPointField]:
    if components.ndim != 2 or components.shape[0] != len(element_ids):
        raise ValueError(
            "finite-strain captured values must have shape (n, components)"
        )
    if not np.all(np.isfinite(components)):
        raise ValueError("finite-strain captured values must be finite")
    element_lookup = {int(element.id): element for element in mesh.elements}
    # Captured assembly output is normally already element-contiguous, but
    # archived/imported frames are not required to preserve that ordering.
    # The old implementation called ``element_ids == element_id`` once per
    # element, turning a frame with E elements and q points into O(E * E*q)
    # comparisons.  Build one stable group permutation instead; this keeps
    # authored element order while sorting points inside each group.
    unique_ids, first_positions, inverse = np.unique(
        element_ids,
        return_index=True,
        return_inverse=True,
    )
    group_order = np.argsort(first_positions, kind="stable")
    group_rank = np.empty_like(group_order)
    group_rank[group_order] = np.arange(len(group_order), dtype=int)
    ranks = group_rank[inverse]
    stable_indices = np.arange(len(element_ids), dtype=int)
    permutation = np.lexsort((stable_indices, point_numbers, ranks))
    sorted_ranks = ranks[permutation]
    split_points = np.flatnonzero(sorted_ranks[1:] != sorted_ranks[:-1]) + 1
    group_starts = np.concatenate(([0], split_points))
    group_stops = np.concatenate((split_points, [len(permutation)]))
    fields: list[_ElementIntegrationPointField] = []
    ordered_ids = unique_ids[group_order]
    for element_id, start, stop in zip(
        ordered_ids,
        group_starts,
        group_stops,
    ):
        _run_checkpoint(checkpoint)
        element = element_lookup.get(int(element_id))
        if element is None:
            raise ValueError(
                f"finite-strain integration output references unknown element {element_id}"
            )
        selected = permutation[start:stop]
        type_key = dispatch.type_key_from_name(element.type)
        kernel = get_recovery_service(element.type)
        fields.append(
            _ElementIntegrationPointField(
                elem=element,
                type_key=type_key,
                kernel=kernel,
                gauss_order=gauss_order,
                natural_coordinates=natural[selected],
                components=components[selected],
                region_key=result_region_key_for_element(element),
                weight=float(np.sum(weights[selected])),
            )
        )
    if not fields:
        raise ValueError("finite-strain result frame has no integration points")
    return fields


def _captured_cauchy_stress(integration: Mapping[str, Any]) -> np.ndarray:
    if "cauchy_stress" in integration:
        values = np.asarray(integration["cauchy_stress"], dtype=float)
    else:
        kirchhoff = np.asarray(integration.get("kirchhoff_stress"), dtype=float)
        deformation = np.asarray(integration.get("deformation_gradient"), dtype=float)
        if kirchhoff.shape != deformation.shape or kirchhoff.ndim != 3:
            raise ValueError(
                "finite-strain stress outputs cannot derive Cauchy stress"
            )
        jacobian = np.linalg.det(deformation)
        if np.any(~np.isfinite(jacobian)) or np.any(jacobian <= 0.0):
            raise ValueError("finite-strain stress outputs require positive det(F)")
        values = kirchhoff / jacobian[:, None, None]
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("finite-strain cauchy stress must contain 3x3 tensors")
    if not np.all(np.isfinite(values)):
        raise ValueError("finite-strain cauchy stress must be finite")
    return values


def _finite_strain_component_matrix(
    stresses: np.ndarray,
    component_names: tuple[str, ...],
) -> np.ndarray:
    values = np.empty((len(stresses), len(component_names)), dtype=float)
    values[:, 0] = stresses[:, 0, 0]
    values[:, 1] = stresses[:, 1, 1]
    values[:, 2] = stresses[:, 2, 2]
    values[:, 3] = stresses[:, 0, 1]
    if len(component_names) == len(CANONICAL_SOLID_COMPONENT_NAMES):
        values[:, 4] = stresses[:, 1, 2]
        values[:, 5] = stresses[:, 0, 2]
    return values


def _state_integration_point_records(
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    checkpoint: Callable[[], None] | None,
) -> list[FiniteStrainStateRecord]:
    records: list[FiniteStrainStateRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        for index, (natural, components) in enumerate(
            zip(item.natural_coordinates, item.components),
            start=1,
        ):
            _run_checkpoint(checkpoint)
            records.append(
                _make_state_record(
                    FiniteStrainStatePosition.INTEGRATION_POINT,
                    components,
                    component_names,
                    coordinates=_physical_coordinates(
                        item.type_key,
                        item.elem,
                        natural,
                        lookup,
                    ),
                    elem_id=int(item.elem.id),
                    integration_point=index,
                    natural_coordinates=tuple(float(value) for value in natural),
                    region_key=item.region_key,
                    weight=item.weight,
                )
            )
    return records


def _state_centroid_records(
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    checkpoint: Callable[[], None] | None,
) -> list[FiniteStrainStateRecord]:
    records: list[FiniteStrainStateRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        if item.gauss_order is None:
            components = item.kernel.interpolate_stress_to_centroid(
                item.components
            )
        else:
            components = item.kernel.interpolate_stress_to_centroid(
                item.components,
                item.gauss_order,
            )
        natural = _centroid_natural_coordinates(item.type_key)
        records.append(
            _make_state_record(
                FiniteStrainStatePosition.CENTROID,
                components,
                component_names,
                coordinates=_physical_coordinates(
                    item.type_key,
                    item.elem,
                    natural,
                    lookup,
                ),
                elem_id=int(item.elem.id),
                natural_coordinates=natural,
                region_key=item.region_key,
                weight=item.weight,
            )
        )
    return records


def _state_element_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    U: np.ndarray,
    checkpoint: Callable[[], None] | None,
) -> list[FiniteStrainStateRecord]:
    records: list[FiniteStrainStateRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        if item.gauss_order is None:
            node_values = item.kernel.extrapolate_stress_to_nodes(
                item.components
            )
        else:
            node_values = item.kernel.extrapolate_stress_to_nodes(
                item.components,
                item.gauss_order,
            )
        if node_values.shape != (len(item.elem.node_ids), len(component_names)):
            raise ValueError(
                f"Element {item.elem.id} finite-strain state shape "
                f"{node_values.shape} does not match "
                f"({len(item.elem.node_ids)}, {len(component_names)})"
            )
        for local_node, (node_id, components) in enumerate(
            zip(item.elem.node_ids, node_values),
            start=1,
        ):
            _run_checkpoint(checkpoint)
            node = lookup[int(node_id)]
            records.append(
                _make_state_record(
                    FiniteStrainStatePosition.ELEMENT_NODAL,
                    components,
                    component_names,
                    coordinates=_node_coordinates(node),
                    elem_id=int(item.elem.id),
                    node_id=int(node_id),
                    local_node=local_node,
                    region_key=item.region_key,
                    weight=item.weight,
                    displacement=_node_displacement(
                        mesh,
                        int(node_id),
                        U,
                    ),
                )
            )
    return records


def _state_average_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    element_nodal: Sequence[FiniteStrainStateRecord],
    *,
    checkpoint: Callable[[], None] | None,
    policy: NodalAveragingPolicy | None = None,
) -> list[FiniteStrainStateRecord]:
    if policy is None:
        policy = NodalAveragingPolicy()
    elif type(policy) is not NodalAveragingPolicy:
        raise TypeError("policy must be NodalAveragingPolicy")
    grouped: dict[tuple[int, ResultRegionKey], list[FiniteStrainStateRecord]] = {}
    values_by_region: dict[ResultRegionKey, list[tuple[float, ...]]] = {}
    for record in element_nodal:
        _run_checkpoint(checkpoint)
        if (
            record.node_id is None
            or record.region_key is None
            or record.elem_id is None
            or record.local_node is None
        ):
            raise ValueError(
                "finite-strain element-nodal records require node, element, "
                "local-node, and region identities"
            )
        grouped.setdefault((record.node_id, record.region_key), []).append(record)
        values_by_region.setdefault(record.region_key, []).append(record.components)

    region_ranges = {
        region: np.ptp(np.asarray(values, dtype=float), axis=0)
        for region, values in values_by_region.items()
    }
    region_tolerances = {
        region: np.finfo(float).eps
        * np.maximum(1.0, np.max(np.abs(np.asarray(values, dtype=float)), axis=0))
        * 32.0
        for region, values in values_by_region.items()
    }
    node_order = {int(node_id): index for index, node_id in enumerate(mesh.node_ids)}
    element_order = {
        int(element.id): index for index, element in enumerate(mesh.elements)
    }
    grouped_by_node: dict[
        int,
        list[tuple[ResultRegionKey, list[FiniteStrainStateRecord]]],
    ] = {}
    for key, records in grouped.items():
        grouped_by_node.setdefault(key[0], []).append((key[1], records))

    result: list[FiniteStrainStateRecord] = []
    for node_id in mesh.node_ids:
        _run_checkpoint(checkpoint)
        node_regions = sorted(
            grouped_by_node.get(int(node_id), ()),
            key=lambda item: result_region_sort_key(item[0]),
        )
        for region_key, contributions in node_regions:
            ordered = sorted(
                contributions,
                key=lambda record: (
                    element_order[int(record.elem_id)],
                    int(record.local_node),
                ),
            )
            if len(ordered) == 1 or policy.threshold_percent == 0.0:
                result.extend(
                    _state_with_position(record, averaged=False)
                    for record in ordered
                )
                continue
            node_values = np.asarray(
                [record.components for record in ordered],
                dtype=float,
            )
            node_ranges = np.ptp(node_values, axis=0)
            ranges = region_ranges[region_key]
            tolerances = region_tolerances[region_key]
            relative_variation = np.zeros_like(node_ranges)
            outside_tolerance = ranges > tolerances
            relative_variation[outside_tolerance] = (
                100.0
                * node_ranges[outside_tolerance]
                / ranges[outside_tolerance]
            )
            if not np.all(relative_variation <= policy.threshold_percent):
                result.extend(
                    _state_with_position(record, averaged=False)
                    for record in ordered
                )
                continue
            weights = np.asarray([record.weight for record in ordered], dtype=float)
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError(
                    "finite-strain state element-nodal weights must be positive"
                )
            averaged = np.average(node_values, axis=0, weights=weights)
            first = ordered[0]
            result.append(
                FiniteStrainStateRecord(
                    position=FiniteStrainStatePosition.NODAL,
                    coordinates=first.coordinates,
                    components=tuple(float(value) for value in averaged),
                    node_id=int(node_id),
                    region_key=region_key,
                    weight=float(np.sum(weights)),
                    displacement=first.displacement,
                    averaged=True,
                )
            )
    return result


def _state_with_position(
    record: FiniteStrainStateRecord,
    *,
    averaged: bool,
) -> FiniteStrainStateRecord:
    return FiniteStrainStateRecord(
        position=FiniteStrainStatePosition.NODAL,
        coordinates=record.coordinates,
        components=record.components,
        elem_id=record.elem_id,
        node_id=record.node_id,
        local_node=record.local_node,
        region_key=record.region_key,
        weight=record.weight,
        displacement=record.displacement,
        averaged=averaged,
    )


def _make_state_record(
    position: FiniteStrainStatePosition,
    components: Sequence[float],
    component_names: tuple[str, ...],
    *,
    coordinates: tuple[float, ...],
    elem_id: int | None = None,
    integration_point: int | None = None,
    natural_coordinates: tuple[float, ...] | None = None,
    node_id: int | None = None,
    local_node: int | None = None,
    region_key: ResultRegionKey | None = None,
    weight: float = 1.0,
    displacement: tuple[float, ...] | None = None,
    averaged: bool | None = None,
) -> FiniteStrainStateRecord:
    values = tuple(float(value) for value in components)
    if len(values) != len(component_names) or not np.all(np.isfinite(values)):
        raise ValueError("finite-strain state components are invalid")
    return FiniteStrainStateRecord(
        position=position,
        coordinates=coordinates,
        components=values,
        elem_id=elem_id,
        integration_point=integration_point,
        natural_coordinates=natural_coordinates,
        node_id=node_id,
        local_node=local_node,
        region_key=region_key,
        weight=float(weight),
        displacement=displacement,
        averaged=averaged,
    )


def _integration_point_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    checkpoint: Callable[[], None] | None,
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        for index, (natural, components) in enumerate(
            zip(item.natural_coordinates, item.components),
            start=1,
        ):
            _run_checkpoint(checkpoint)
            records.append(
                _make_record(
                    StressPosition.INTEGRATION_POINT,
                    components,
                    component_names,
                    coordinates=_physical_coordinates(
                        item.type_key, item.elem, natural, lookup
                    ),
                    elem_id=int(item.elem.id),
                    integration_point=index,
                    natural_coordinates=tuple(float(value) for value in natural),
                    region_key=item.region_key,
                    weight=item.weight,
                )
            )
    return records


def _centroid_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    checkpoint: Callable[[], None] | None,
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        if item.gauss_order is None:
            components = item.kernel.interpolate_stress_to_centroid(item.components)
        else:
            components = item.kernel.interpolate_stress_to_centroid(
                item.components, item.gauss_order
            )
        _run_checkpoint(checkpoint)
        natural = _centroid_natural_coordinates(item.type_key)
        records.append(
            _make_record(
                StressPosition.CENTROID,
                components,
                component_names,
                coordinates=_physical_coordinates(
                    item.type_key, item.elem, natural, lookup
                ),
                elem_id=int(item.elem.id),
                natural_coordinates=natural,
                region_key=item.region_key,
                weight=item.weight,
            )
        )
    return records


def _element_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    fields: Sequence[_ElementIntegrationPointField],
    component_names: tuple[str, ...],
    U: np.ndarray,
    checkpoint: Callable[[], None] | None,
) -> list[StressRecord]:
    records: list[StressRecord] = []
    for item in fields:
        _run_checkpoint(checkpoint)
        if item.gauss_order is None:
            node_values = item.kernel.extrapolate_stress_to_nodes(item.components)
        else:
            node_values = item.kernel.extrapolate_stress_to_nodes(
                item.components, item.gauss_order
            )
        _run_checkpoint(checkpoint)
        if node_values.shape != (len(item.elem.node_ids), len(component_names)):
            raise ValueError(
                f"Element {item.elem.id} element-nodal stress shape {node_values.shape} "
                f"does not match ({len(item.elem.node_ids)}, {len(component_names)})"
            )
        for local_node, (node_id, components) in enumerate(
            zip(item.elem.node_ids, node_values),
            start=1,
        ):
            _run_checkpoint(checkpoint)
            node = lookup[int(node_id)]
            records.append(
                _make_record(
                    StressPosition.ELEMENT_NODAL,
                    components,
                    component_names,
                    coordinates=_node_coordinates(node),
                    elem_id=int(item.elem.id),
                    node_id=int(node_id),
                    local_node=local_node,
                    region_key=item.region_key,
                    weight=item.weight,
                    displacement=_node_displacement(mesh, int(node_id), U),
                )
            )
    return records


def _average_nodal_records(
    mesh: Any,
    lookup: dict[int, Any],
    element_nodal: Sequence[StressRecord],
    component_names: tuple[str, ...],
    checkpoint: Callable[[], None] | None,
) -> list[StressRecord]:
    grouped: dict[tuple[int, ResultRegionKey], list[StressRecord]] = {}
    for record in element_nodal:
        _run_checkpoint(checkpoint)
        if record.node_id is None or record.region_key is None:
            continue
        grouped.setdefault((record.node_id, record.region_key), []).append(record)

    node_order = {
        int(node_id): index for index, node_id in enumerate(mesh.node_ids)
    }
    records: list[StressRecord] = []
    for (node_id, region_key), contributions in sorted(
        grouped.items(),
        key=lambda item: (
            node_order.get(item[0][0], len(node_order)),
            result_region_sort_key(item[0][1]),
        ),
    ):
        _run_checkpoint(checkpoint)
        weights = np.asarray(
            [record.weight for record in contributions],
            dtype=float,
        )
        components = np.average(
            np.asarray([record.components for record in contributions], dtype=float),
            axis=0,
            weights=weights,
        )
        _run_checkpoint(checkpoint)
        records.append(
            _make_record(
                StressPosition.NODAL,
                components,
                component_names,
                coordinates=_node_coordinates(lookup[node_id]),
                node_id=node_id,
                region_key=region_key,
                weight=float(np.sum(weights)),
                displacement=contributions[0].displacement,
            )
        )
    return records


def _make_record(
    position: StressPosition,
    components,
    component_names: tuple[str, ...],
    *,
    coordinates: tuple[float, ...],
    elem_id: int | None = None,
    integration_point: int | None = None,
    natural_coordinates: tuple[float, ...] | None = None,
    node_id: int | None = None,
    local_node: int | None = None,
    region_key: ResultRegionKey | None = None,
    weight: float = 1.0,
    displacement: tuple[float, ...] | None = None,
    averaged: bool | None = None,
) -> StressRecord:
    values = tuple(float(value) for value in components)
    return StressRecord(
        position=position,
        coordinates=coordinates,
        components=values,
        invariants=derive_stress_invariants(values, component_names),
        elem_id=elem_id,
        integration_point=integration_point,
        natural_coordinates=natural_coordinates,
        node_id=node_id,
        local_node=local_node,
        region_key=region_key,
        weight=float(weight),
        displacement=displacement,
        averaged=averaged,
    )


def _node_coordinates(node: Any) -> tuple[float, ...]:
    coordinates = [float(node.x), float(node.y)]
    if hasattr(node, "z"):
        coordinates.append(float(node.z))
    return tuple(coordinates)


def _node_displacement(
    mesh: Any,
    node_id: int,
    U: np.ndarray,
) -> tuple[float, ...]:
    translation_count = (
        3
        if mesh.nodes and hasattr(mesh.nodes[0], "z")
        else 2
    )
    return tuple(
        float(U[mesh.global_dof(node_id, component)])
        for component in range(min(mesh.dofs_per_node, translation_count))
    )


def _physical_coordinates(
    type_key: str,
    elem: Any,
    natural_coordinates,
    lookup: dict[int, Any],
) -> tuple[float, ...]:
    natural = tuple(float(value) for value in natural_coordinates)
    shape_values = natural_shape_values(type_key, natural)
    node_coordinates = np.asarray(
        [_node_coordinates(lookup[int(node_id)]) for node_id in elem.node_ids],
        dtype=float,
    )
    coordinates = shape_values @ node_coordinates
    return tuple(float(value) for value in coordinates)


def natural_shape_values(
    type_key: str,
    natural_coordinates: Sequence[float],
) -> np.ndarray:
    """Return continuum shape-function values at one natural coordinate."""
    natural = tuple(float(value) for value in natural_coordinates)
    if type_key == "tri3":
        xi, eta = natural
        return np.asarray([1.0 - xi - eta, xi, eta], dtype=float)
    if type_key == "tri6":
        from ...elements.tri6 import tri6_shape_funcs_grads

        return tri6_shape_funcs_grads(*natural)[0]
    if type_key == "quad4":
        xi, eta = natural
        return 0.25 * np.asarray([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ], dtype=float)
    if type_key == "quad8":
        from ...elements.quad8 import quad8_shape_funcs_grads

        return quad8_shape_funcs_grads(*natural)[0]
    if type_key == "tet4":
        from ...elements.tet4 import tet4_shape_funcs_grads

        return tet4_shape_funcs_grads(*natural)[0]
    if type_key == "tet10":
        from ...elements.tet10 import tet10_shape_funcs_grads

        return tet10_shape_funcs_grads(*natural)[0]
    if type_key == "hex8":
        from ...elements.hex8 import hex8_shape_funcs_grads

        return hex8_shape_funcs_grads(*natural)[0]
    if type_key == "hex20":
        from ...elements.hex20 import hex20_shape_funcs_grads

        return hex20_shape_funcs_grads(*natural)[0]
    raise ValueError(f"Unsupported continuum stress element type key: {type_key!r}")


def _centroid_natural_coordinates(type_key: str) -> tuple[float, ...]:
    if type_key in {"tri3", "tri6"}:
        return (1.0 / 3.0, 1.0 / 3.0)
    if type_key in {"quad4", "quad8"}:
        return (0.0, 0.0)
    if type_key in {"tet4", "tet10"}:
        return (0.25, 0.25, 0.25)
    if type_key in {"hex8", "hex20"}:
        return (0.0, 0.0, 0.0)
    raise ValueError(f"Unsupported continuum stress element type key: {type_key!r}")


def resolve(
    field: NodalStressField,
    threshold: float = 75.0,
) -> ResolvedNodalStressField:
    """Compatibility adapter over the canonical complete-tensor resolver.

    The historical component schema and isolated-node zero rows remain here
    only for legacy CSV/VTK callers.
    """

    try:
        policy = NodalAveragingPolicy(threshold_percent=threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "threshold must be a finite number from 0.0 through 100.0"
        ) from error

    canonical, metadata = _canonical_field_from_legacy(field)
    element_ids = tuple(
        dict.fromkeys(
            contribution.elem_id
            for node_id in field.node_ids
            for contribution in field.contributions_by_node.get(node_id, ())
        )
    )
    resolved = resolve_nodal_stress(
        canonical,
        policy,
        node_ids=field.node_ids,
        element_ids=element_ids,
    )
    canonical_plane = (
        canonical.component_names == CANONICAL_PLANE_COMPONENT_NAMES
    )
    rows_by_node_region: dict[
        tuple[int, ResultRegionKey],
        list[ResolvedNodalStressRow],
    ] = {}
    for record in resolved.records:
        if record.node_id is None or record.region_key is None:
            continue
        plane_type, poisson_ratio = metadata[
            (record.node_id, record.region_key)
        ]
        components = (
            (
                record.components[0],
                record.components[1],
                record.components[3],
            )
            if canonical_plane
            else record.components
        )
        rows_by_node_region.setdefault(
            (record.node_id, record.region_key),
            [],
        ).append(
            ResolvedNodalStressRow(
                node_id=record.node_id,
                components=tuple(float(value) for value in components),
                elem_id=record.elem_id,
                local_node=record.local_node,
                averaged=bool(record.averaged),
                plane_type=plane_type,
                poisson_ratio=poisson_ratio,
            )
        )

    rows: list[ResolvedNodalStressRow] = []
    for node_id in field.node_ids:
        contributions = tuple(field.contributions_by_node.get(node_id, ()))
        region_order = tuple(
            dict.fromkeys(
                contribution.region_key for contribution in contributions
            )
        )
        if region_order:
            # Historical field.resolve treated any multi-region node as fully
            # unaveraged. Keep that projection only in this legacy adapter.
            if len(region_order) > 1:
                rows.extend(
                    _legacy_raw_row(contribution)
                    for contribution in contributions
                )
                continue
            for region_key in region_order:
                rows.extend(
                    rows_by_node_region[(node_id, region_key)]
                )
            continue
        rows.append(
            ResolvedNodalStressRow(
                node_id=node_id,
                components=(0.0,) * len(field.component_names),
                elem_id=None,
                local_node=None,
                averaged=True,
            )
        )
    return ResolvedNodalStressField(field.component_names, tuple(rows))


def _legacy_raw_row(
    contribution: ElementNodalStressContribution,
) -> ResolvedNodalStressRow:
    return ResolvedNodalStressRow(
        node_id=contribution.node_id,
        components=contribution.components,
        elem_id=contribution.elem_id,
        local_node=contribution.local_node,
        averaged=False,
        plane_type=contribution.plane_type,
        poisson_ratio=contribution.poisson_ratio,
    )


def collect(
    mesh: Any,
    U: Sequence[float],
    element_type: str | None = None,
    gauss_order: int | None = None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> NodalStressField:
    """Compatibility view of canonical element-nodal stress records."""
    stress_field = collect_stress(
        mesh,
        U,
        position=StressPosition.ELEMENT_NODAL,
        element_type=element_type,
        gauss_order=gauss_order,
        checkpoint=checkpoint,
    )
    return nodal_from_stress_field(
        mesh,
        stress_field,
        checkpoint=checkpoint,
    )


def nodal_from_stress_field(
    mesh: Any,
    stress_field: StressField,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> NodalStressField:
    """Project one recovered element-nodal field to the legacy nodal view."""

    checkpoint = _validate_checkpoint(checkpoint)
    _run_checkpoint(checkpoint)
    if type(stress_field) is not StressField:
        raise TypeError("stress_field must be a StressField")
    if stress_field.position is not StressPosition.ELEMENT_NODAL:
        raise ValueError(
            "legacy nodal projection requires element-nodal stress"
        )
    if stress_field.component_names not in {
        CANONICAL_PLANE_COMPONENT_NAMES,
        CANONICAL_SOLID_COMPONENT_NAMES,
    }:
        raise ValueError(
            "legacy nodal projection requires canonical continuum stress"
        )
    contributions: dict[int, list[ElementNodalStressContribution]] = {
        int(node_id): [] for node_id in mesh.node_ids
    }
    is_plane = stress_field.component_names == CANONICAL_PLANE_COMPONENT_NAMES
    element_lookup = {int(elem.id): elem for elem in mesh.elements}
    for record in stress_field.records:
        _run_checkpoint(checkpoint)
        if (
            record.node_id is None
            or record.elem_id is None
            or record.local_node is None
            or record.region_key is None
        ):
            continue
        plane_type = None
        poisson_ratio = None
        if is_plane:
            plane_type, poisson_ratio = get_recovery_service(
                element_lookup[record.elem_id].type
            )._plane_data(element_lookup[record.elem_id])
        contributions[record.node_id].append(
            ElementNodalStressContribution(
                node_id=record.node_id,
                elem_id=record.elem_id,
                local_node=record.local_node,
                components=(
                    (
                        record.components[0],
                        record.components[1],
                        record.components[3],
                    )
                    if is_plane
                    else record.components
                ),
                weight=record.weight,
                region_key=record.region_key,
                plane_type=plane_type,
                poisson_ratio=poisson_ratio,
            )
        )

    return NodalStressField(
        component_names=(
            PLANE_COMPONENT_NAMES if is_plane else SOLID_COMPONENT_NAMES
        ),
        contributions_by_node={
            node_id: tuple(values) for node_id, values in contributions.items()
        },
        node_ids=tuple(int(node_id) for node_id in mesh.node_ids),
    )


def _canonical_field_from_legacy(
    legacy: NodalStressField,
) -> tuple[
    StressField,
    dict[tuple[int, ResultRegionKey], tuple[str | None, float | None]],
]:
    is_plane = legacy.component_names == PLANE_COMPONENT_NAMES
    if not is_plane and legacy.component_names != SOLID_COMPONENT_NAMES:
        raise ValueError("legacy nodal stress field has unsupported components")
    component_names = (
        CANONICAL_PLANE_COMPONENT_NAMES
        if is_plane
        else CANONICAL_SOLID_COMPONENT_NAMES
    )
    metadata: dict[
        tuple[int, ResultRegionKey],
        tuple[str | None, float | None],
    ] = {}
    records: list[StressRecord] = []
    for node_id in legacy.node_ids:
        for contribution in legacy.contributions_by_node.get(node_id, ()):
            if contribution.node_id != node_id:
                raise ValueError(
                    "legacy contribution node id does not match its node group"
                )
            if is_plane:
                plane_type = contribution.plane_type or "stress"
                poisson_ratio = (
                    0.0
                    if contribution.poisson_ratio is None
                    else contribution.poisson_ratio
                )
                components = complete_plane_components(
                    contribution.components,
                    plane_type,
                    poisson_ratio,
                )
            else:
                components = contribution.components
            metadata.setdefault(
                (node_id, contribution.region_key),
                (
                    contribution.plane_type,
                    contribution.poisson_ratio,
                ),
            )
            records.append(
                _make_record(
                    StressPosition.ELEMENT_NODAL,
                    components,
                    component_names,
                    coordinates=(),
                    elem_id=contribution.elem_id,
                    node_id=node_id,
                    local_node=contribution.local_node,
                    region_key=contribution.region_key,
                    weight=contribution.weight,
                )
            )
    return (
        StressField(
            StressPosition.ELEMENT_NODAL,
            component_names,
            tuple(records),
        ),
        metadata,
    )
