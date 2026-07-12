from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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
class MaterialDefinition:
    """Named material properties."""
    name: str
    properties: Mapping[str, Any] = field(default_factory=dict)


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
        """Element formulation name, such as Hex8, Tet4, or Quad4Plane."""
        return self.type


@dataclass(frozen=True)
class SectionAssignment:
    """Assign a material to an element set."""
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_component", int(self.first_component))
        object.__setattr__(self, "last_component", int(self.last_component))
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class NodalLoad:
    """Abaqus-style nodal load using a 1-based component."""
    target: str | int
    component: int
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", int(self.component))
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class SurfaceLoad:
    """Surface load attached to a named surface."""
    surface: str
    vector: Sequence[float] = ()
    magnitude: float | None = None
    load_type: str = "traction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        if self.magnitude is not None:
            object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "load_type", str(self.load_type).lower())


@dataclass(frozen=True)
class EdgeLoad:
    """Line load attached to a named edge collection."""
    edge: str
    vector: Sequence[float] = ()
    magnitude: float | None = None
    load_type: str = "traction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", str(self.edge))
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        if self.magnitude is not None:
            object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "load_type", str(self.load_type).lower())


@dataclass(frozen=True)
class LineLoad:
    """Constant Beam2 line load per undeformed length."""
    target: str | int
    vector: Sequence[float]
    coordinate_system: str = "global"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        object.__setattr__(self, "coordinate_system", str(self.coordinate_system))


@dataclass(frozen=True)
class OutputRequest:
    """Output request attached to an analysis step."""
    kind: str
    target: str
    variables: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind).lower())
        object.__setattr__(self, "target", str(self.target).lower())
        object.__setattr__(self, "variables", tuple(str(value) for value in self.variables))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class AnalysisStep:
    """Analysis step with loads and output metadata."""
    name: str
    procedure: str = "static"
    boundaries: Sequence[DisplacementConstraint] = ()
    cloads: Sequence[NodalLoad] = ()
    surface_loads: Sequence[SurfaceLoad] = ()
    outputs: Sequence[OutputRequest] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    edge_loads: Sequence[EdgeLoad] = ()
    line_loads: Sequence[LineLoad] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "cloads", tuple(self.cloads))
        object.__setattr__(self, "surface_loads", tuple(self.surface_loads))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "edge_loads", tuple(self.edge_loads))
        object.__setattr__(self, "line_loads", tuple(self.line_loads))


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
    properties = dict(getattr(elem, "props", {}))
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
    result = dict(model.element_sets)
    result.update(model.metadata.get("_abaqus_internal_element_sets", {}))
    return result
