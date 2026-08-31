from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence
from types import MappingProxyType

from .controls import (
    DynamicStepControls,
    GeometryMode,
    InitialConditionSet,
    StaticAnalysisOptions,
    StaticFormulation,
    StaticStepControls,
)
from .immutable_json import freeze_json_mapping


@dataclass(frozen=True)
class NodeSet:
    """Named node id set."""
    name: str
    node_ids: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))


@dataclass(frozen=True)
class ElementSet:
    """Named element id set."""
    name: str
    element_ids: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_ids",
            tuple(int(element_id) for element_id in self.element_ids),
        )


@dataclass(frozen=True)
class ElementFace:
    """Element face identified by element id and local face index."""
    elem_id: int
    local_index: int
    node_ids: Sequence[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(self, "local_index", int(self.local_index))
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))


@dataclass(frozen=True)
class Surface:
    """Named collection of element faces."""
    name: str
    faces: Sequence[ElementFace]

    def __post_init__(self) -> None:
        object.__setattr__(self, "faces", tuple(self.faces))


@dataclass(frozen=True)
class ElementEdge:
    """Element edge identified by element id and local edge index."""
    elem_id: int
    local_index: int
    node_ids: Sequence[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(self, "local_index", int(self.local_index))
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))


@dataclass(frozen=True)
class Edge:
    """Named collection of element edges."""
    name: str
    edges: Sequence[ElementEdge]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class MaterialBehavior:
    """One explicit Abaqus-style material behavior definition."""

    behavior_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        behavior_id = str(self.behavior_id).strip().casefold()
        if not behavior_id:
            raise ValueError("material behavior_id must not be blank")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("material behavior parameters must be a mapping")
        object.__setattr__(self, "behavior_id", behavior_id)
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "MaterialBehavior":
        memo[id(self)] = self
        return self


_ELASTIC_BEHAVIOR = "mechanical.elasticity.isotropic"
_PLASTIC_BEHAVIOR = "mechanical.plasticity.j2"
_DENSITY_BEHAVIOR = "general.density"
_NEO_HOOKEAN_BEHAVIOR = "mechanical.hyperelasticity.neo_hookean"


def _behaviors_from_material_identity(
    constitutive_model: str,
    properties: Mapping[str, Any],
) -> tuple[MaterialBehavior, ...]:
    """Materialize the typed behavior list at the authoring boundary.

    Existing callers still construct ``MaterialDefinition`` from the compact
    property mapping.  The explicit constitutive identity remains authoritative
    for that boundary; properties alone never activate plasticity.
    """

    behaviors: list[MaterialBehavior] = []
    if constitutive_model in {"linear_elastic", "j2_plasticity"}:
        elastic = {
            key: properties[key]
            for key in ("E", "nu")
            if key in properties
        }
        if constitutive_model == "j2_plasticity" or elastic:
            behaviors.append(MaterialBehavior(_ELASTIC_BEHAVIOR, elastic))
    if constitutive_model == "j2_plasticity":
        plastic = {
            key: properties[key]
            for key in ("yield_stress", "hardening_modulus")
            if key in properties
        }
        behaviors.append(MaterialBehavior(_PLASTIC_BEHAVIOR, plastic))
    if constitutive_model == "neo_hookean":
        hyperelastic = {
            key: properties[key]
            for key in ("C10", "D1")
            if key in properties
        }
        behaviors.append(
            MaterialBehavior(_NEO_HOOKEAN_BEHAVIOR, hyperelastic)
        )
    if "rho" in properties:
        behaviors.append(
            MaterialBehavior(_DENSITY_BEHAVIOR, {"rho": properties["rho"]})
        )
    return tuple(behaviors)


def _identity_from_behaviors(
    behaviors: Sequence[MaterialBehavior],
    fallback: str,
) -> str:
    ids = {behavior.behavior_id for behavior in behaviors}
    if _NEO_HOOKEAN_BEHAVIOR in ids:
        return "neo_hookean"
    if _PLASTIC_BEHAVIOR in ids:
        return "j2_plasticity"
    if _ELASTIC_BEHAVIOR in ids:
        return "linear_elastic"
    return fallback


@dataclass(frozen=True)
class MaterialDefinition:
    """Named material composed from explicit Abaqus-style behaviors."""

    name: str
    properties: Mapping[str, Any] = field(default_factory=dict)
    constitutive_model: str = "linear_elastic"
    algorithm: str | None = None
    behaviors: Sequence[MaterialBehavior] = ()
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        model = str(self.constitutive_model).strip().casefold()
        if not name:
            raise ValueError("material name must not be blank")
        if not model:
            raise ValueError("material constitutive_model must not be blank")
        if not isinstance(self.properties, Mapping):
            raise TypeError("material properties must be a mapping")
        behaviors = tuple(self.behaviors)
        if any(type(item) is not MaterialBehavior for item in behaviors):
            raise TypeError(
                "material behaviors must contain only MaterialBehavior values"
            )
        if not behaviors:
            behaviors = _behaviors_from_material_identity(model, self.properties)
        else:
            model = _identity_from_behaviors(behaviors, model)
        description = str(self.description)
        algorithm = self.algorithm
        if algorithm is not None:
            algorithm = str(algorithm).strip().casefold() or None
        properties = dict(self.properties)
        if behaviors:
            behavior_properties: dict[str, Any] = {}
            for behavior in behaviors:
                behavior_properties.update(dict(behavior.parameters))
            # A caller using replace(..., properties=...) is performing an
            # explicit partial update.  Its new values must win over the
            # behavior snapshot carried by the replaced dataclass.
            behavior_properties.update(properties)
            properties = behavior_properties
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "properties",
            MappingProxyType(deepcopy(properties)),
        )
        object.__setattr__(self, "constitutive_model", model)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "behaviors", behaviors)
        object.__setattr__(self, "description", description)

    def __deepcopy__(self, memo: dict[int, Any]) -> "MaterialDefinition":
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class ElementInfo:
    """Read-only effective model information for one element."""
    elem_id: int
    type: str
    node_ids: Sequence[int]
    material: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    section_type: str | None = None
    element_sets: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(self, "type", str(self.type))
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))
        if self.material is not None:
            object.__setattr__(self, "material", str(self.material))
        object.__setattr__(self, "properties", dict(self.properties))
        if self.section_type is not None:
            object.__setattr__(self, "section_type", str(self.section_type))
        object.__setattr__(self, "element_sets", tuple(str(name) for name in self.element_sets))

    @property
    def element_type(self) -> str:
        """Element formulation name, such as Hex8, Tet4, or Quad4."""
        return self.type


@dataclass(frozen=True)
class SectionAssignment:
    """Assign a material and section definition to an element set."""
    element_set: str
    material: str
    section_type: str = "solid"
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "properties", dict(self.properties))


@dataclass(frozen=True)
class DisplacementConstraint:
    """Abaqus-style displacement constraint using 1-based components."""
    target: str | int
    first_component: int
    last_component: int
    value: float = 0.0
    target_kind: str = "node_set"
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_component", int(self.first_component))
        object.__setattr__(self, "last_component", int(self.last_component))
        object.__setattr__(self, "value", float(self.value))
        target_kind = str(self.target_kind).strip().casefold()
        if target_kind not in {"node_set", "edge", "surface"}:
            raise ValueError(
                "displacement constraint target_kind must be "
                "'node_set', 'edge', or 'surface'"
            )
        if isinstance(self.target, int) and target_kind != "node_set":
            raise ValueError(
                "integer displacement constraint targets require "
                "target_kind='node_set'"
            )
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class NodalLoad:
    """Abaqus-style nodal load using a 1-based component."""
    target: str | int
    component: int
    value: float
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", int(self.component))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class SurfaceLoad:
    """Surface load attached to a named surface."""
    surface: str
    vector: Sequence[float] = ()
    magnitude: float | None = None
    load_type: str = "traction"
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        if self.magnitude is not None:
            object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "load_type", str(self.load_type).lower())
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class EdgeLoad:
    """Line load attached to a named edge collection."""
    edge: str
    vector: Sequence[float] = ()
    magnitude: float | None = None
    load_type: str = "traction"
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", str(self.edge))
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        if self.magnitude is not None:
            object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "load_type", str(self.load_type).lower())
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class LineLoad:
    """Constant Beam2 line load per undeformed length."""
    target: str | int
    vector: Sequence[float]
    coordinate_system: str = "global"
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        object.__setattr__(self, "coordinate_system", str(self.coordinate_system))
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class BodyForce:
    """Constant force per unit volume applied to elements or an element set."""
    target: str | int
    vector: Sequence[float]
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector",
            tuple(float(value) for value in self.vector),
        )
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True)
class GravityLoad:
    """Gravity acceleration applied globally or to selected elements."""
    acceleration: Sequence[float]
    target: str | int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceleration", tuple(self.acceleration))
        object.__setattr__(self, "name", _optional_identity_name(self.name))


@dataclass(frozen=True, slots=True)
class OutputSourceEvidence:
    """Immutable structured source context for an imported output request."""

    source_kind: str
    parent_parameters: tuple[tuple[str, str], ...] = ()
    parent_flags: tuple[str, ...] = ()
    child_parameters: tuple[tuple[str, str], ...] = ()
    child_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source_kind = _exact_nonblank_string(
            self.source_kind,
            name="source_kind",
        )
        object.__setattr__(self, "source_kind", source_kind.lower())
        object.__setattr__(
            self,
            "parent_parameters",
            _freeze_output_evidence_parameters(
                self.parent_parameters,
                name="parent_parameters",
            ),
        )
        object.__setattr__(
            self,
            "parent_flags",
            _freeze_output_evidence_flags(
                self.parent_flags,
                name="parent_flags",
            ),
        )
        object.__setattr__(
            self,
            "child_parameters",
            _freeze_output_evidence_parameters(
                self.child_parameters,
                name="child_parameters",
            ),
        )
        object.__setattr__(
            self,
            "child_flags",
            _freeze_output_evidence_flags(
                self.child_flags,
                name="child_flags",
            ),
        )


@dataclass(frozen=True, slots=True)
class OutputRequest:
    """Output request attached to an analysis step."""

    kind: str
    target: str
    variables: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_evidence: OutputSourceEvidence | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        kind = _exact_nonblank_string(self.kind, name="kind")
        target = _exact_nonblank_string(self.target, name="target")
        variables = _freeze_output_variables(self.variables)
        metadata = freeze_json_mapping(self.metadata, name="metadata")
        evidence = self.source_evidence
        if evidence is not None and type(evidence) is not OutputSourceEvidence:
            raise TypeError(
                "source_evidence must be exactly OutputSourceEvidence or None"
            )

        object.__setattr__(self, "kind", kind.lower())
        object.__setattr__(self, "target", target.lower())
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "name", _optional_identity_name(self.name))


def _exact_nonblank_string(value: Any, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _optional_identity_name(value: Any) -> str | None:
    if value is None:
        return None
    return _exact_nonblank_string(value, name="name")


def _freeze_output_variables(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("variables must be an iterable of exact strings")
    try:
        variables = tuple(value)
    except TypeError as error:
        raise TypeError(
            "variables must be an iterable of exact strings"
        ) from error
    for index, item in enumerate(variables):
        if type(item) is not str:
            raise TypeError(f"variables[{index}] must be an exact string")
    return variables


def _freeze_output_evidence_parameters(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an iterable of string pairs")
    try:
        parameters = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of string pairs") from error

    result: list[tuple[str, str]] = []
    for index, item in enumerate(parameters):
        if isinstance(item, (str, bytes, bytearray, Mapping)):
            raise TypeError(f"{name}[{index}] must be a pair of exact strings")
        try:
            pair = tuple(item)
        except TypeError as error:
            raise TypeError(
                f"{name}[{index}] must be a pair of exact strings"
            ) from error
        if len(pair) != 2 or any(type(part) is not str for part in pair):
            raise TypeError(f"{name}[{index}] must be a pair of exact strings")
        result.append((pair[0], pair[1]))
    return tuple(result)


def _freeze_output_evidence_flags(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an iterable of exact strings")
    try:
        flags = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of exact strings") from error
    for index, flag in enumerate(flags):
        if type(flag) is not str:
            raise TypeError(f"{name}[{index}] must be an exact string")
    return flags


@dataclass
class AnalysisStep:
    """Analysis step with loads, typed controls, and output metadata.

    ``controls`` is the canonical in-memory value object for analysis
    settings.  ``metadata`` remains a persistence mirror; numerical execution
    resolves typed controls once into ``AnalysisRequest``.
    """
    name: str
    procedure: str = "static"
    boundaries: Sequence[DisplacementConstraint] = ()
    cloads: Sequence[NodalLoad] = ()
    surface_loads: Sequence[SurfaceLoad] = ()
    outputs: Sequence[OutputRequest] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    edge_loads: Sequence[EdgeLoad] = ()
    line_loads: Sequence[LineLoad] = ()
    body_loads: Sequence[BodyForce] = ()
    gravity_loads: Sequence[GravityLoad] = ()
    controls: Any | None = None
    formulation: StaticFormulation | None = None
    geometry_mode: GeometryMode | None = None
    options: StaticAnalysisOptions | None = None
    initial_conditions: InitialConditionSet = field(
        default_factory=InitialConditionSet,
    )

    def __post_init__(self) -> None:
        procedure = str(self.procedure).strip().casefold()
        if not procedure:
            raise ValueError("analysis step procedure must not be blank")
        object.__setattr__(self, "procedure", procedure)
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "cloads", tuple(self.cloads))
        object.__setattr__(self, "surface_loads", tuple(self.surface_loads))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "edge_loads", tuple(self.edge_loads))
        object.__setattr__(self, "line_loads", tuple(self.line_loads))
        object.__setattr__(self, "body_loads", tuple(self.body_loads))
        object.__setattr__(self, "gravity_loads", tuple(self.gravity_loads))
        if self.formulation is not None and not isinstance(
            self.formulation,
            StaticFormulation,
        ):
            raise TypeError("formulation must be StaticFormulation or None")
        if self.geometry_mode is not None and not isinstance(
            self.geometry_mode,
            GeometryMode,
        ):
            raise TypeError("geometry_mode must be GeometryMode or None")
        if self.formulation is not None and self.geometry_mode is not None:
            expected = GeometryMode.from_formulation(self.formulation)
            if self.geometry_mode is not expected:
                raise ValueError(
                    "analysis step formulation and geometry_mode must agree"
                )
        if self.geometry_mode is None and self.formulation is not None:
            object.__setattr__(
                self,
                "geometry_mode",
                GeometryMode.from_formulation(self.formulation),
            )
        elif self.formulation is None and self.geometry_mode is not None:
            object.__setattr__(self, "formulation", self.geometry_mode.formulation)
        if procedure == "dynamic":
            if self.formulation is None and self.geometry_mode is None:
                object.__setattr__(self, "formulation", StaticFormulation.LINEAR)
                object.__setattr__(self, "geometry_mode", GeometryMode.SMALL_STRAIN)
        if self.formulation is not None:
            metadata = dict(self.metadata)
            if self.formulation is StaticFormulation.NONLINEAR:
                metadata["NLGEOM"] = True
            else:
                # Linear is the default formulation.  Keep the persistence
                # mirror sparse so changing a nonlinear authored step back to
                # the default does not leave a stale legacy switch behind.
                metadata.pop("NLGEOM", None)
            object.__setattr__(self, "metadata", metadata)
        if self.options is not None and not isinstance(
            self.options,
            StaticAnalysisOptions,
        ):
            raise TypeError("options must be StaticAnalysisOptions or None")
        controls = self.controls
        if controls is not None:
            if procedure == "static" and not isinstance(controls, StaticStepControls):
                raise TypeError("controls must be StaticStepControls")
            if procedure == "dynamic" and not isinstance(controls, DynamicStepControls):
                raise TypeError(
                    "dynamic analysis step controls must be DynamicStepControls"
                )
            metadata = dict(self.metadata)
            metadata.update(controls.to_metadata())
            object.__setattr__(self, "metadata", metadata)
        if self.options is not None:
            metadata = dict(self.metadata)
            metadata["material_algorithm"] = self.options.material_algorithm
            object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.initial_conditions, InitialConditionSet):
            raise TypeError(
                "analysis step initial_conditions must be InitialConditionSet"
            )
        if not self.initial_conditions.is_empty:
            metadata = dict(self.metadata)
            metadata["initial_conditions"] = self.initial_conditions.to_metadata()
            object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class AnalysisStepSnapshot:
    """Detached executable view of one authored analysis step.

    Persistence metadata and mutable authoring identity deliberately do not
    cross this boundary.  The snapshot contains only the typed records that
    boundary compilation and result publication need.
    """

    name: str
    procedure: str
    boundaries: tuple[DisplacementConstraint, ...] = ()
    cloads: tuple[NodalLoad, ...] = ()
    surface_loads: tuple[SurfaceLoad, ...] = ()
    outputs: tuple[OutputRequest, ...] = ()
    edge_loads: tuple[EdgeLoad, ...] = ()
    line_loads: tuple[LineLoad, ...] = ()
    body_loads: tuple[BodyForce, ...] = ()
    gravity_loads: tuple[GravityLoad, ...] = ()
    controls: Any = field(default_factory=StaticStepControls)
    formulation: StaticFormulation = StaticFormulation.LINEAR
    geometry_mode: GeometryMode | None = None
    initial_conditions: InitialConditionSet = field(
        default_factory=InitialConditionSet,
    )

    @classmethod
    def from_step(
        cls,
        step: AnalysisStep,
        *,
        controls: Any,
        formulation: StaticFormulation | None = None,
        geometry_mode: GeometryMode | None = None,
        initial_conditions: InitialConditionSet | None = None,
    ) -> "AnalysisStepSnapshot":
        if not isinstance(step, AnalysisStep):
            raise TypeError("step must be an AnalysisStep")
        procedure = str(step.procedure).strip().casefold()
        if procedure == "static" and not isinstance(controls, StaticStepControls):
            raise TypeError("controls must be StaticStepControls")
        if procedure == "dynamic" and not isinstance(controls, DynamicStepControls):
            raise TypeError("dynamic controls must be DynamicStepControls")
        selected_formulation = (
            step.formulation
            if formulation is None and step.formulation is not None
            else (
                StaticFormulation.LINEAR
                if formulation is None
                else formulation
            )
        )
        if not isinstance(selected_formulation, StaticFormulation):
            raise TypeError("formulation must be StaticFormulation")
        selected_geometry = geometry_mode
        if selected_geometry is None:
            selected_geometry = getattr(step, "geometry_mode", None)
        if selected_geometry is None:
            selected_geometry = GeometryMode.from_formulation(selected_formulation)
        if not isinstance(selected_geometry, GeometryMode):
            raise TypeError("geometry_mode must be GeometryMode")
        if selected_geometry.formulation is not selected_formulation:
            raise ValueError(
                "analysis step formulation and geometry_mode must agree"
            )
        return cls(
            name=str(step.name),
            procedure=str(step.procedure),
            boundaries=deepcopy(tuple(step.boundaries)),
            cloads=deepcopy(tuple(step.cloads)),
            surface_loads=deepcopy(tuple(step.surface_loads)),
            outputs=deepcopy(tuple(step.outputs)),
            edge_loads=deepcopy(tuple(step.edge_loads)),
            line_loads=deepcopy(tuple(step.line_loads)),
            body_loads=deepcopy(tuple(step.body_loads)),
            gravity_loads=deepcopy(tuple(step.gravity_loads)),
            controls=controls,
            formulation=selected_formulation,
            geometry_mode=selected_geometry,
            initial_conditions=(
                deepcopy(step.initial_conditions)
                if initial_conditions is None
                else deepcopy(initial_conditions)
            ),
        )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        procedure = str(self.procedure).strip().casefold()
        if not name:
            raise ValueError("analysis step snapshot name must not be blank")
        if not procedure:
            raise ValueError("analysis step snapshot procedure must not be blank")
        if self.procedure == "static" and not isinstance(self.controls, StaticStepControls):
            raise TypeError("analysis step snapshot controls must be StaticStepControls")
        if self.procedure == "dynamic" and not isinstance(self.controls, DynamicStepControls):
            raise TypeError(
                "dynamic analysis step snapshot controls must be DynamicStepControls"
            )
        if not isinstance(self.formulation, StaticFormulation):
            raise TypeError(
                "analysis step snapshot formulation must be StaticFormulation"
            )
        geometry_mode = self.geometry_mode
        if geometry_mode is None:
            geometry_mode = GeometryMode.from_formulation(self.formulation)
        if not isinstance(geometry_mode, GeometryMode):
            raise TypeError(
                "analysis step snapshot geometry_mode must be GeometryMode"
            )
        if geometry_mode.formulation is not self.formulation:
            raise ValueError(
                "analysis step snapshot formulation and geometry_mode must agree"
            )
        object.__setattr__(self, "geometry_mode", geometry_mode)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "procedure", procedure)
        for field_name in (
            "boundaries",
            "cloads",
            "surface_loads",
            "outputs",
            "edge_loads",
            "line_loads",
            "body_loads",
            "gravity_loads",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(deepcopy(getattr(self, field_name))),
            )
        if not isinstance(self.initial_conditions, InitialConditionSet):
            raise TypeError(
                "analysis step snapshot initial_conditions must be InitialConditionSet"
            )


@dataclass
class FEMModel:
    """Finite element model data independent of input format."""
    mesh: Any
    name: str | None = None
    node_sets: dict[str, NodeSet] = field(default_factory=dict)
    element_sets: dict[str, ElementSet] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    materials: dict[str, MaterialDefinition] = field(default_factory=dict)
    sections: list[SectionAssignment] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)


def model_element_info(model: FEMModel, elem_id: int) -> ElementInfo:
    """Return effective type, set, section, and material data for one element id."""
    elem_id = int(elem_id)
    elem = _model_element(model, elem_id)
    properties = _unstamped_element_properties(model, elem_id, elem)
    section = _matching_section(model, elem_id)

    material = properties.get("material")
    section_type = None
    if section is not None:
        if section.material not in model.materials:
            raise KeyError(f"material {section.material} is not defined")
        effective = dict(model.materials[section.material].properties)
        effective.update(section.properties)
        effective["material"] = section.material
        properties.update(effective)
        material = section.material
        section_type = section.section_type

    return ElementInfo(
        elem_id=elem.id,
        type=elem.type,
        node_ids=elem.node_ids,
        material=material,
        properties=properties,
        section_type=section_type,
        element_sets=_element_set_names(model, elem_id),
    )


def _unstamped_element_properties(
    model: FEMModel,
    elem_id: int,
    elem: Any,
) -> dict[str, Any]:
    """Return base properties with prior section-derived values restored."""
    properties = dict(getattr(elem, "props", {}))
    tracked = model.metadata.get("_section_property_keys_by_element", {})
    originals = model.metadata.get("_section_original_properties_by_element", {})
    identities = model.metadata.get("_section_property_element_identity_by_element", {})
    expected_identity = identities.get(elem_id)
    if expected_identity is not None:
        if isinstance(expected_identity, int):
            identity_matches = expected_identity == id(elem)
        else:
            identity_matches = (
                getattr(
                    elem,
                    "_fem_section_property_tracking_identity",
                    None,
                )
                is expected_identity
            )
        if not identity_matches:
            return properties

    baseline = originals.get(elem_id, {})
    for key in tracked.get(elem_id, ()):
        existed, value = baseline.get(key, (False, None))
        if existed:
            properties[key] = value
        else:
            properties.pop(key, None)
    return properties


def _model_element(model: FEMModel, elem_id: int) -> Any:
    """Return a mesh element by id."""
    for elem in model.mesh.elements:
        if int(elem.id) == elem_id:
            return elem
    raise KeyError(f"element {elem_id} is not defined")


def _matching_section(model: FEMModel, elem_id: int) -> SectionAssignment | None:
    """Return the last section assignment covering an element."""
    match = None
    for section in model.sections:
        element_set = _model_element_set(model, section.element_set)
        if elem_id in element_set.element_ids:
            match = section
    return match


def _element_set_names(model: FEMModel, elem_id: int) -> tuple[str, ...]:
    """Return public and importer-internal element set names containing an element."""
    names: list[str] = []
    for name, element_set in _all_model_element_sets(model).items():
        if elem_id in element_set.element_ids:
            names.append(str(name))
    return tuple(names)


def _model_element_set(model: FEMModel, name: str) -> ElementSet:
    """Return a public or importer-internal element set."""
    element_sets = _all_model_element_sets(model)
    if name in element_sets:
        return element_sets[name]
    raise KeyError(f"element set {name} is not defined")


def _all_model_element_sets(model: FEMModel) -> dict[str, ElementSet]:
    """Return public and importer-internal element sets."""
    result = dict(
        model.metadata.get("_abaqus_internal_element_sets", {})
    )
    result.update(model.element_sets)
    return result


__all__ = [
    "AnalysisStep",
    "AnalysisStepSnapshot",
    "BodyForce",
    "DisplacementConstraint",
    "Edge",
    "EdgeLoad",
    "ElementEdge",
    "ElementFace",
    "ElementInfo",
    "ElementSet",
    "FEMModel",
    "GravityLoad",
    "LineLoad",
    "MaterialBehavior",
    "MaterialDefinition",
    "NodalLoad",
    "NodeSet",
    "OutputRequest",
    "OutputSourceEvidence",
    "SectionAssignment",
    "Surface",
    "SurfaceLoad",
    "model_element_info",
]
