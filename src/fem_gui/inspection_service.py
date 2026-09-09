"""Read-only indexing and structured information for finite element model entities."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from typing import Any

from fem.application import RegionRef, resolve_effective_beam_frames
from fem.core._constraint_targets import (
    displacement_target_kind,
    resolve_displacement_node_ids,
)
from fem.application.results import (
    ElementResultInspectionRequest,
    FieldPosition,
    FieldState,
    NodeResultInspectionRequest,
    ResultInspectionField,
    ResultInspectionRequest,
    ResultInspectionResult,
    ResultProvider,
    ResultQueryValidationError,
)
from fem.post.fields import encode_result_region_key
from .result_presentation import (
    result_field_has_section_points,
    result_field_position_label,
    result_field_is_visible,
    result_provider_section_point_labels,
    section_point_relative_position_label,
)


_RESULT_VARIABLE_LABELS = {
    "U": "Displacement U",
    "UR": "Rotation UR",
    "RF": "Reaction Force RF",
    "RM": "Reaction Moment RM",
    "LE": "Logarithmic Strain LE",
    "S": "Stress S",
}
_RESULT_COMPONENT_LABELS = {
    "Magnitude": "Magnitude",
    "Mises": "Mises equivalent stress",
    "MaxPrincipal": "Maximum principal stress",
    "MidPrincipal": "Middle principal stress",
    "MinPrincipal": "Minimum principal stress",
    "S11Max": "Maximum axial stress",
    "S11Min": "Minimum axial stress",
    "S11AbsMax": "Maximum absolute axial stress",
}
_RESULT_STATE_LABELS = {
    FieldState.READY: "Ready",
    FieldState.LAZY: "Load on demand",
    FieldState.UNAVAILABLE: "Unavailable",
}
_RESULT_COLUMNS = (
    "Status",
    "Component",
    "Value",
    "Node",
    "Element",
    "Integration point",
    "Local node",
    "Result region",
    "Averaging",
    "Diagnostics",
)
_SECTION_POINT_RESULT_COLUMNS = (
    *_RESULT_COLUMNS[:7],
    "Section position",
    "Section local Y",
    "Section local Z",
    *_RESULT_COLUMNS[7:],
)


@dataclass(frozen=True, slots=True)
class EntityReference:
    kind: str
    key: object


@dataclass(frozen=True, slots=True)
class InspectionTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    references: tuple[EntityReference | None, ...] = ()


@dataclass(frozen=True, slots=True)
class InspectionPage:
    title: str
    fields: tuple[tuple[str, str], ...] = ()
    tables: tuple[InspectionTable, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityInspection:
    title: str
    kind: str
    key: object
    pages: tuple[InspectionPage, ...]


@dataclass(frozen=True, slots=True)
class EntitySelection:
    node_ids: tuple[int, ...] = ()
    element_ids: tuple[int, ...] = ()


class InspectionService:
    """Build a read-only reverse index once for an FEMModel."""

    def __init__(
        self,
        model: Any,
        *,
        result_provider: ResultProvider | None = None,
        definitions: Any | None = None,
        effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None,
    ) -> None:
        self.model = model
        self.result_provider = _require_result_provider(result_provider)
        self.definitions = definitions
        self.section_definitions = tuple(
            getattr(definitions, "sections", ())
        )
        self.region_assignments = tuple(
            getattr(definitions, "assignments", ())
        )
        self._effective_frame_query = (
            effective_frame_query
            if effective_frame_query is not None
            else lambda target: resolve_effective_beam_frames(
                self.model,
                target,
            )
        )
        self.nodes = {int(node.id): node for node in model.mesh.nodes}
        self.elements = {int(element.id): element for element in model.mesh.elements}
        self.node_sets_by_node: dict[int, list[str]] = defaultdict(list)
        self.element_sets_by_element: dict[int, list[str]] = defaultdict(list)
        self.adjacent_elements: dict[int, list[int]] = defaultdict(list)
        self.element_material: dict[int, str] = {}
        self.element_section: dict[int, int] = {}
        self.section_elements: dict[int, tuple[int, ...]] = {}
        self.material_elements: dict[str, set[int]] = defaultdict(set)
        self.node_analysis: dict[int, list[tuple[str, str, str, float, EntityReference]]] = defaultdict(list)
        self.node_set_boundaries: dict[str, list[EntityReference]] = defaultdict(list)
        self.node_set_loads: dict[str, list[EntityReference]] = defaultdict(list)
        self.region_boundaries: dict[
            tuple[str, str],
            list[EntityReference],
        ] = defaultdict(list)
        self.region_loads: dict[tuple[str, str], list[EntityReference]] = defaultdict(list)
        self._all_element_sets = dict(model.element_sets)
        self._all_element_sets.update(model.metadata.get("_abaqus_internal_element_sets", {}))
        self._build_indexes()
        self._element_record_cached = lru_cache(maxsize=4096)(self._make_element_record)
        self._beam_frame_report_cached = lru_cache(maxsize=4096)(
            self._query_effective_beam_frames
        )

    def update_result_provider(
        self,
        result_provider: ResultProvider | None,
    ) -> None:
        """Install one exact immutable provider without materializing fields."""

        self.result_provider = _require_result_provider(result_provider)

    def _query_effective_beam_frames(
        self,
        target: RegionRef | int,
    ) -> Any:
        return self._effective_frame_query(target)

    def _build_indexes(self) -> None:
        for name, node_set in self.model.node_sets.items():
            for node_id in node_set.node_ids:
                self.node_sets_by_node[int(node_id)].append(str(name))
        for name, element_set in self.model.element_sets.items():
            for element_id in element_set.element_ids:
                self.element_sets_by_element[int(element_id)].append(str(name))
        for element_id, element in self.elements.items():
            for node_id in element.node_ids:
                self.adjacent_elements[int(node_id)].append(element_id)
        for section_index, section in enumerate(self.model.sections):
            element_set = self._all_element_sets.get(section.element_set)
            ids = () if element_set is None else tuple(int(value) for value in element_set.element_ids)
            self.section_elements[section_index] = ids
            for element_id in ids:
                self.element_section[element_id] = section_index
                self.element_material[element_id] = section.material
                self.material_elements[section.material].add(element_id)
        for element_id, element in self.elements.items():
            material = getattr(element, "props", {}).get("material")
            if material and element_id not in self.element_material:
                self.element_material[element_id] = str(material)
                self.material_elements[str(material)].add(element_id)
        self._index_analysis_definitions()

    def _index_analysis_definitions(self) -> None:
        for step_index, step in enumerate(self.model.steps):
            for index, boundary in enumerate(step.boundaries):
                reference = EntityReference("boundary", (step_index, index))
                target_kind = displacement_target_kind(boundary)
                if isinstance(boundary.target, str):
                    if target_kind == "node_set":
                        self.node_set_boundaries[str(boundary.target)].append(
                            reference
                        )
                    else:
                        self.region_boundaries[
                            (target_kind, str(boundary.target))
                        ].append(reference)
                component = _component_range(boundary.first_component, boundary.last_component)
                for node_id in resolve_displacement_node_ids(
                    self.model,
                    boundary,
                ):
                    self.node_analysis[node_id].append(
                        (step.name, "Displacement BC", component, float(boundary.value), reference)
                    )
            for index, load in enumerate(step.cloads):
                reference = EntityReference("cload", (step_index, index))
                if isinstance(load.target, str):
                    self.node_set_loads[str(load.target)].append(reference)
                for node_id in self.target_node_ids(load.target):
                    self.node_analysis[node_id].append(
                        (step.name, "Nodal force", f"U{load.component}", float(load.value), reference)
                    )
            for index, load in enumerate(step.surface_loads):
                self.region_loads[("surface", load.surface)].append(
                    EntityReference("surface_load", (step_index, index))
                )
            for index, load in enumerate(step.edge_loads):
                self.region_loads[("edge", load.edge)].append(
                    EntityReference("edge_load", (step_index, index))
                )

    def target_node_ids(self, target: str | int) -> tuple[int, ...]:
        if isinstance(target, int):
            return (int(target),) if int(target) in self.nodes else ()
        node_set = self.model.node_sets.get(str(target))
        return () if node_set is None else tuple(int(value) for value in node_set.node_ids)

    def element_record(self, element_id: int) -> dict[str, Any]:
        return self._element_record_cached(int(element_id))

    def _make_element_record(self, element_id: int) -> dict[str, Any]:
        element_id = int(element_id)
        element = self.elements[element_id]
        props = dict(getattr(element, "props", {}))
        section_index = self.element_section.get(element_id)
        section = self.model.sections[section_index] if section_index is not None else None
        material = self.element_material.get(element_id)
        effective: dict[str, Any] = {}
        if material in self.model.materials:
            effective.update(self.model.materials[material].properties)
        if section is not None:
            effective.update(section.properties)
        effective.update({key: value for key, value in props.items() if _is_physical_property(key, value)})
        return {
            "id": element_id,
            "type": str(element.type),
            "abaqus_type": props.get("abaqus_type"),
            "node_ids": tuple(int(value) for value in element.node_ids),
            "sets": tuple(self.element_sets_by_element.get(element_id, ())),
            "material": material,
            "section_index": section_index,
            "section_type": None if section is None else section.section_type,
            "properties": effective,
        }

    def node_row(self, node_id: int) -> tuple[object, ...]:
        node = self.nodes[int(node_id)]
        return (
            int(node.id), float(node.x), float(node.y),
            float(getattr(node, "z", 0.0)),
            ", ".join(self.node_sets_by_node.get(int(node.id), ())),
        )

    def element_row(self, element_id: int) -> tuple[object, ...]:
        record = self.element_record(element_id)
        nodes = record["node_ids"]
        preview = ", ".join(str(value) for value in nodes[:4]) + (", …" if len(nodes) > 4 else "")
        section = "—" if record["section_index"] is None else f"Section {record['section_index'] + 1}"
        return (
            record["id"], record["type"], preview,
            ", ".join(record["sets"]), record["material"] or "—", section,
        )

    def inspect(self, kind: str, key: object) -> EntityInspection:
        handlers = {
            "model": self._inspect_model, "node": self._inspect_node,
            "element": self._inspect_element, "node_set": self._inspect_node_set,
            "element_set": self._inspect_element_set, "surface": self._inspect_surface,
            "edge": self._inspect_edge, "material": self._inspect_material,
            "section": self._inspect_section, "step": self._inspect_step,
            "boundary": self._inspect_boundary, "cload": self._inspect_cload,
            "surface_load": self._inspect_surface_load, "edge_load": self._inspect_edge_load,
            "line_load": self._inspect_line_load,
            "body_load": self._inspect_body_load,
            "gravity_load": self._inspect_gravity_load,
            "assignment": self._inspect_assignment,
            "output": self._inspect_output,
        }
        if kind not in handlers:
            raise KeyError(f"Unsupported inspection entity: {kind}")
        return handlers[kind](key)

    def selection_for(self, kind: str, key: object) -> EntitySelection:
        if kind == "node":
            return EntitySelection(node_ids=(int(key),))
        if kind == "element":
            return EntitySelection(element_ids=(int(key),))
        if kind == "node_set":
            return EntitySelection(node_ids=tuple(self.model.node_sets[str(key)].node_ids))
        if kind == "element_set":
            return EntitySelection(element_ids=tuple(self.model.element_sets[str(key)].element_ids))
        if kind == "surface":
            return EntitySelection(element_ids=tuple(sorted({face.elem_id for face in self.model.surfaces[str(key)].faces})))
        if kind == "edge":
            return EntitySelection(element_ids=tuple(sorted({edge.elem_id for edge in self.model.edges[str(key)].edges})))
        if kind == "material":
            return EntitySelection(element_ids=tuple(sorted(self.material_elements.get(str(key), ()))))
        if kind == "section":
            return EntitySelection(element_ids=self.section_elements.get(int(key), ()))
        if kind == "assignment":
            assignment = self.region_assignments[int(key)]
            return EntitySelection(
                element_ids=self.target_element_ids(
                    str(assignment.region_name)
                )
            )
        if kind == "boundary":
            step_index, index = key
            return EntitySelection(
                node_ids=resolve_displacement_node_ids(
                    self.model,
                    self.model.steps[step_index].boundaries[index],
                )
            )
        if kind == "cload":
            step_index, index = key
            return EntitySelection(node_ids=self.target_node_ids(self.model.steps[step_index].cloads[index].target))
        if kind in {"surface_load", "edge_load"}:
            step_index, index = key
            step = self.model.steps[step_index]
            region_kind = "surface" if kind == "surface_load" else "edge"
            region = step.surface_loads[index].surface if region_kind == "surface" else step.edge_loads[index].edge
            return self.selection_for(region_kind, region)
        if kind == "line_load":
            step_index, index = key
            return EntitySelection(element_ids=self.target_element_ids(
                self.model.steps[step_index].line_loads[index].target
            ))
        if kind == "body_load":
            step_index, index = key
            return EntitySelection(element_ids=self.target_element_ids(
                self.model.steps[step_index].body_loads[index].target
            ))
        if kind == "gravity_load":
            step_index, index = key
            target = self.model.steps[step_index].gravity_loads[index].target
            if target is None:
                return EntitySelection(
                    element_ids=tuple(sorted(self.elements))
                )
            return EntitySelection(
                element_ids=self.target_element_ids(target)
            )
        return EntitySelection()

    def target_element_ids(self, target: str | int) -> tuple[int, ...]:
        if isinstance(target, int):
            return (int(target),) if int(target) in self.elements else ()
        element_set = self._all_element_sets.get(str(target))
        return () if element_set is None else tuple(
            int(value) for value in element_set.element_ids
        )

    def _inspect_model(self, _key: object) -> EntityInspection:
        mesh = self.model.mesh
        spatial_dimension = getattr(mesh, "spatial_dimension", None)
        dimension = (
            f"{spatial_dimension}D"
            if spatial_dimension in {1, 2, 3}
            else "3D"
            if mesh.nodes and hasattr(mesh.nodes[0], "z")
            else "2D"
        )
        fields = (
            ("Model name", str(self.model.name or "Model")), ("Spatial dimension", dimension),
            ("Node count", str(len(mesh.nodes))), ("Element count", str(len(mesh.elements))),
            ("Total DOF count", str(mesh.num_dofs)),
            ("Element type counts", _counter_text(element.type for element in mesh.elements)),
            ("Node set count", str(len(self.model.node_sets))),
            ("Element set count", str(len(self.model.element_sets))),
            ("Surface and edge count", str(len(self.model.surfaces) + len(self.model.edges))),
            ("Material count", str(len(self.model.materials))), ("Section count", str(len(self.model.sections))),
            ("Step count", str(len(self.model.steps))),
        )
        return EntityInspection("Model Overview", "model", None, (InspectionPage("Overview", fields),))

    def _inspect_node(self, key: object) -> EntityInspection:
        node_id = int(key)
        node = self.nodes[node_id]
        coords = [float(node.x), float(node.y)]
        if getattr(self.model.mesh, "spatial_dimension", None) != 2 and hasattr(node, "z"):
            coords.append(float(node.z))
        adjacent = tuple(
            (str(element_id), str(self.elements[element_id].type))
            for element_id in self.adjacent_elements.get(node_id, ())
        )
        references = tuple(EntityReference("element", int(row[0])) for row in adjacent)
        pages = [InspectionPage(
            "Basic information",
            (("Node ID", str(node_id)), ("Coordinates", ", ".join(format_number(value) for value in coords)),
             ("Node sets", ", ".join(self.node_sets_by_node.get(node_id, ())) or "—")),
            (InspectionTable("Adjacent elements", ("Element ID", "Element type"), adjacent, references),),
        )]
        analysis_rows = tuple(
            (step, kind, component, format_number(value))
            for step, kind, component, value, _reference in self.node_analysis.get(node_id, ())
        )
        if analysis_rows:
            pages.append(InspectionPage(
                "Analysis definitions", tables=(InspectionTable(
                    "Analysis definitions", ("Step", "Type", "Component", "Value"), analysis_rows,
                    tuple(row[4] for row in self.node_analysis[node_id]),
                ),),
            ))
        result_page = self._provider_result_page(
            NodeResultInspectionRequest(node_id)
        )
        if result_page is not None:
            pages.append(result_page)
        return EntityInspection(
            f"Node {node_id}",
            "node",
            node_id,
            tuple(pages),
        )

    def _inspect_element(self, key: object) -> EntityInspection:
        element_id = int(key)
        record = self.element_record(element_id)
        props = record["properties"]
        fields = [
            ("Element ID", str(element_id)), ("Native element type", record["type"]),
            ("Element sets", ", ".join(record["sets"]) or "—"),
            ("Material", record["material"] or "—"),
            ("Section", "—" if record["section_index"] is None else f"Section {record['section_index'] + 1}"),
        ]
        if record["abaqus_type"]:
            fields.insert(2, ("Abaqus element type", str(record["abaqus_type"])))
        if "plane_type" in props:
            fields.append(("Plane type", _plane_label(props["plane_type"])))
        if "thickness" in props:
            fields.append(("Thickness", format_number(props["thickness"])))
        connection_rows = []
        references = []
        for local_index, node_id in enumerate(record["node_ids"], 1):
            node = self.nodes[node_id]
            connection_rows.append((
                str(local_index), str(node_id), format_number(node.x), format_number(node.y),
                format_number(getattr(node, "z", 0.0)),
            ))
            references.append(EntityReference("node", node_id))
        property_rows = tuple(
            (_property_label(name), format_number(value))
            for name, value in props.items()
            if name not in {"plane_type", "thickness"}
        )
        tables = [InspectionTable(
            "Connectivity", ("Local node", "Global node", "X", "Y", "Z"),
            tuple(connection_rows), tuple(references),
        )]
        if property_rows:
            tables.append(InspectionTable("Effective properties", ("Property", "Value"), property_rows))
        pages = [InspectionPage("Basic information", tuple(fields)), InspectionPage("Connectivity and properties", tables=tuple(tables))]
        frame_report = self._beam_frame_report_cached(element_id)
        frame_entry = (
            frame_report.for_element(element_id)
            if hasattr(frame_report, "for_element")
            else next(
                (
                    entry
                    for entry in tuple(
                        getattr(frame_report, "entries", ())
                    )
                    if int(entry.element_id) == element_id
                ),
                None,
            )
        )
        if frame_entry is not None:
            frame = frame_entry.frame
            provenance = (
                "direct element"
                if frame_entry.assignment_index is None
                else f"Section assignment {int(frame_entry.assignment_index) + 1}"
            )
            frame_fields = [
                ("Element ID", str(element_id)),
                ("frame source", str(frame.source)),
                ("Effective properties source", provenance),
                (
                    "assignment element set",
                    str(frame_entry.element_set or "—"),
                ),
                (
                    "effective section type",
                    str(frame_entry.section_type or "—"),
                ),
                ("local x", _vector_text(frame.local_x)),
                ("local y", _vector_text(frame.local_y)),
                ("local z", _vector_text(frame.local_z)),
            ]
            frame_diagnostics = tuple(
                getattr(frame_report, "diagnostics", ())
            )
            if frame_diagnostics:
                frame_fields.append(
                    (
                        "diagnostics",
                        _diagnostic_summary(frame_diagnostics),
                    )
                )
            pages.append(
                InspectionPage("Beam local coordinates", tuple(frame_fields))
            )
        result_page = self._provider_result_page(
            ElementResultInspectionRequest(element_id)
        )
        if result_page is not None:
            pages.append(result_page)
        return EntityInspection(f"Element {element_id}", "element", element_id, tuple(pages))

    def _provider_result_page(
        self,
        request: ResultInspectionRequest,
    ) -> InspectionPage | None:
        """Render only the provider's catalog-ordered typed inspection."""

        provider = self.result_provider
        if provider is None:
            return None
        try:
            result = provider.inspect_result(request)
        except ResultQueryValidationError:
            return None
        if type(result) is not ResultInspectionResult:
            raise TypeError(
                "ResultProvider.inspect_result() must return "
                "ResultInspectionResult"
            )
        fields = tuple(
            field_entry
            for field_entry in result.fields
            if result_field_is_visible(field_entry.availability)
        )
        if not fields:
            return None
        section_point_labels = result_provider_section_point_labels(provider)
        return InspectionPage(
            "Results",
            tables=tuple(
                _provider_result_table(
                    field_entry,
                    section_point_labels=section_point_labels,
                )
                for field_entry in fields
            ),
        )

    def _inspect_node_set(self, key: object) -> EntityInspection:
        name = str(key)
        item = self.model.node_sets[name]
        rows = tuple(
            (str(node_id), format_number(self.nodes[node_id].x), format_number(self.nodes[node_id].y),
             format_number(getattr(self.nodes[node_id], "z", 0.0)))
            for node_id in item.node_ids
        )
        fields = (
            ("Name", name), ("Node count", str(len(item.node_ids))),
            ("Boundary condition references", self._reference_names(self.node_set_boundaries.get(name, ()))),
            ("Nodal force references", self._reference_names(self.node_set_loads.get(name, ()))),
        )
        table = InspectionTable("Member nodes", ("Node ID", "X", "Y", "Z"), rows,
                                tuple(EntityReference("node", int(row[0])) for row in rows))
        return EntityInspection(f"Node set {name}", "node_set", name, (InspectionPage("Node set", fields, (table,)),))

    def _inspect_element_set(self, key: object) -> EntityInspection:
        name = str(key)
        item = self.model.element_sets[name]
        records = [self.element_record(element_id) for element_id in item.element_ids]
        rows = tuple((
            str(record["id"]), record["type"], record["material"] or "—",
            "—" if record["section_index"] is None else f"Section {record['section_index'] + 1}",
        ) for record in records)
        fields = (
            ("Name", name), ("Element count", str(len(records))),
            ("Element type counts", _counter_text(record["type"] for record in records)),
            ("Materials used", ", ".join(sorted({record["material"] for record in records if record["material"]})) or "—"),
            ("Sections used", ", ".join(sorted({f"Section {record['section_index'] + 1}" for record in records if record["section_index"] is not None})) or "—"),
        )
        table = InspectionTable("Member elements", ("Element ID", "Type", "Material", "Section"), rows,
                                tuple(EntityReference("element", int(row[0])) for row in rows))
        return EntityInspection(f"Element set {name}", "element_set", name, (InspectionPage("Element set", fields, (table,)),))

    def _inspect_surface(self, key: object) -> EntityInspection:
        name = str(key)
        return self._inspect_region("surface", name, self.model.surfaces[name].faces)

    def _inspect_edge(self, key: object) -> EntityInspection:
        name = str(key)
        return self._inspect_region("edge", name, self.model.edges[name].edges)

    def _inspect_region(self, kind: str, name: str, members: tuple[Any, ...]) -> EntityInspection:
        label = "Surface" if kind == "surface" else "Edge"
        rows = tuple((
            str(member.elem_id), str(member.local_index),
            ", ".join(str(value) for value in member.node_ids),
        ) for member in members)
        fields = (
            ("Name", name), ("Type", label), (f"{label} count", str(len(members))),
            ("Involved element count", str(len({member.elem_id for member in members}))),
            ("Boundary condition references", self._reference_names(self.region_boundaries.get((kind, name), ()))),
            ("Load references", self._reference_names(self.region_loads.get((kind, name), ()))),
        )
        table = InspectionTable(f"Member {label.lower()}s", ("Element ID", f"Local {label.lower()} ID", "Node ID"), rows,
                                tuple(EntityReference("element", int(row[0])) for row in rows))
        return EntityInspection(f"{label} {name}", kind, name, (InspectionPage(label, fields, (table,)),))

    def _inspect_material(self, key: object) -> EntityInspection:
        name = str(key)
        material = self.model.materials[name]
        fields = [("Material name", name)]
        for property_name in ("E", "nu", "rho", "density"):
            if property_name in material.properties:
                fields.append((_property_label(property_name), format_number(material.properties[property_name])))
        sections = [f"Section {index + 1}" for index, section in enumerate(self.model.sections) if section.material == name]
        fields.extend((("Referenced sections", ", ".join(sections) or "—"),
                       ("Affected element count", str(len(self.material_elements.get(name, ()))))))
        return EntityInspection(f"Material {name}", "material", name, (InspectionPage("Material", tuple(fields)),))

    def _inspect_section(self, key: object) -> EntityInspection:
        index = int(key)
        section = self.model.sections[index]
        ids = self.section_elements.get(index, ())
        fields = [
            ("Section ID", str(index + 1)), ("Section type", section.section_type),
            ("Material", section.material), ("Affected element sets", section.element_set),
            ("Affected element count", str(len(ids))),
        ]
        plane_types = {self.element_record(value)["properties"].get("plane_type") for value in ids}
        thicknesses = {self.element_record(value)["properties"].get("thickness") for value in ids}
        plane_types.discard(None)
        thicknesses.discard(None)
        if plane_types:
            fields.append(("Plane type", ", ".join(sorted(_plane_label(value) for value in plane_types))))
        if thicknesses:
            fields.append(("Thickness", ", ".join(sorted(format_number(value) for value in thicknesses))))
        for name, value in section.properties.items():
            if _is_physical_property(name, value) and name not in {"plane_type", "thickness"}:
                fields.append((_property_label(name), format_number(value)))
        return EntityInspection(f"Section {index + 1}", "section", index, (InspectionPage("Section", tuple(fields)),))

    def _inspect_assignment(self, key: object) -> EntityInspection:
        index = int(key)
        assignment = self.region_assignments[index]
        section = next(
            (
                item
                for item in self.section_definitions
                if str(item.name) == str(assignment.section_name)
            ),
            None,
        )
        target = RegionRef(
            "element_set",
            str(assignment.region_name),
        )
        report = self._beam_frame_report_cached(target)
        entries = tuple(getattr(report, "entries", ()))
        element_ids = tuple(getattr(report, "element_ids", ()))
        diagnostics = tuple(getattr(report, "diagnostics", ()))
        orientation = getattr(assignment, "beam_orientation", None)
        not_applicable = (
            orientation is None
            and not entries
            and bool(diagnostics)
            and all(
                str(getattr(item, "code", ""))
                == "beam.orientation.unsupported_target"
                for item in diagnostics
            )
        )
        if not_applicable:
            diagnostics = ()
        reference = (
            None
            if orientation is None
            else getattr(orientation, "local_y_reference", None)
        )
        sources = sorted(
            {
                str(entry.frame.source)
                for entry in entries
            }
        )
        fields = [
            ("Assignment ID", str(index + 1)),
            ("Section", str(assignment.section_name)),
            ("Target element set", str(assignment.region_name)),
            (
                "section type",
                str(getattr(section, "section_type", "—")),
            ),
            (
                "orientation source",
                (
                    "not applicable"
                    if not_applicable
                    else (
                        "explicit"
                        if orientation is not None
                        else "automatic"
                    )
                ),
            ),
            (
                "authored reference",
                "—" if reference is None else _vector_text(reference),
            ),
            (
                "effective frame source",
                (
                    "not applicable"
                    if not_applicable
                    else (", ".join(sources) or "—")
                ),
            ),
            (
                "Valid element count",
                "0" if not_applicable else str(len(entries)),
            ),
            (
                "Invalid element count",
                (
                    "0"
                    if not_applicable
                    else str(max(0, len(element_ids) - len(entries)))
                ),
            ),
            (
                "validity",
                (
                    "not applicable"
                    if not_applicable
                    else (
                        "valid"
                        if bool(getattr(report, "passed", False))
                        else "invalid"
                    )
                ),
            ),
        ]
        properties = dict(getattr(section, "properties", {}))
        if (
            str(getattr(section, "section_type", "")).casefold()
            == "rectangle"
        ):
            fields.extend(
                (
                    ("Rectangle height (local z)", format_number(properties.get("height"))),
                    ("Rectangle width (local y)", format_number(properties.get("width"))),
                    (
                        "Section axis mapping",
                        "width → local y; height → local z",
                    ),
                )
            )
        if diagnostics:
            fields.append(("diagnostics", _diagnostic_summary(diagnostics)))
        tables = ()
        if diagnostics:
            tables = (
                InspectionTable(
                    "Orientation diagnostics",
                    ("Code", "Severity", "Description"),
                    tuple(
                        (
                            str(getattr(item, "code", "")),
                            str(
                                getattr(
                                    getattr(item, "severity", ""),
                                    "value",
                                    getattr(item, "severity", ""),
                                )
                            ),
                            str(getattr(item, "message", item)),
                        )
                        for item in diagnostics
                    ),
                ),
            )
        return EntityInspection(
            f"Section assignment {index + 1}",
            "assignment",
            index,
            (InspectionPage("Section assignment", tuple(fields), tables),),
        )

    def _inspect_step(self, key: object) -> EntityInspection:
        step_index = int(key)
        step = self.model.steps[step_index]
        load_count = getattr(step, "summary_load_count", None)
        if load_count is None:
            load_count = (
                len(step.cloads)
                + len(step.surface_loads)
                + len(step.edge_loads)
                + len(step.line_loads)
                + len(step.body_loads)
                + len(step.gravity_loads)
            )
        boundary_count = getattr(
            step,
            "summary_boundary_count",
            len(step.boundaries),
        )
        output_count = getattr(
            step,
            "summary_output_count",
            len(step.outputs),
        )
        pages = [InspectionPage("Overview", (
            ("Step name", step.name), ("Analysis type", _procedure_label(step.procedure)),
            ("Boundary condition count", str(boundary_count)), ("Load count", str(load_count)),
            ("Output request count", str(output_count)),
        ))]
        boundary_rows = tuple((str(index + 1), "Displacement BC", str(item.target),
                               _component_range(item.first_component, item.last_component), format_number(item.value))
                              for index, item in enumerate(step.boundaries))
        if boundary_rows:
            pages.append(InspectionPage("Boundary conditions", tables=(InspectionTable(
                "Boundary conditions", ("Index", "Type", "Target", "Component", "Value"), boundary_rows,
                tuple(EntityReference("boundary", (step_index, index)) for index in range(len(boundary_rows))),
            ),)))
        load_rows: list[tuple[str, ...]] = []
        load_refs: list[EntityReference] = []
        for index, item in enumerate(step.cloads):
            load_rows.append((str(len(load_rows) + 1), "Nodal force", str(item.target), f"U{item.component}", format_number(item.value)))
            load_refs.append(EntityReference("cload", (step_index, index)))
        for kind, items, target_name in (("surface_load", step.surface_loads, "surface"), ("edge_load", step.edge_loads, "edge")):
            for index, item in enumerate(items):
                target = getattr(item, target_name)
                direction = _load_direction(item)
                load_rows.append((str(len(load_rows) + 1), _load_type_label(item.load_type), target, direction, format_number(item.magnitude)))
                load_refs.append(EntityReference(kind, (step_index, index)))
        for index, item in enumerate(step.line_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "Edge force",
                str(item.target),
                (
                    "Local (resolved beam coordinates)"
                    if item.coordinate_system == "local"
                    else "Global coordinates"
                ),
                ", ".join(format_number(value) for value in item.vector),
            ))
            load_refs.append(EntityReference("line_load", (step_index, index)))
        for index, item in enumerate(step.body_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "Body force",
                str(item.target),
                "Global coordinates",
                ", ".join(format_number(value) for value in item.vector),
            ))
            load_refs.append(EntityReference("body_load", (step_index, index)))
        for index, item in enumerate(step.gravity_loads):
            load_rows.append((
                str(len(load_rows) + 1),
                "Gravity",
                "Entire model" if item.target is None else str(item.target),
                "Global coordinates",
                ", ".join(
                    format_number(value)
                    for value in item.acceleration
                ),
            ))
            load_refs.append(
                EntityReference("gravity_load", (step_index, index))
            )
        if load_rows:
            pages.append(InspectionPage("Loads", tables=(InspectionTable(
                "Loads", ("Index", "Type", "Target", "Component or direction", "Value"), tuple(load_rows), tuple(load_refs),
            ),)))
        output_rows = tuple((str(index + 1), _output_kind(item.kind), _output_target(item.target), ", ".join(item.variables))
                            for index, item in enumerate(step.outputs))
        if output_rows:
            pages.append(InspectionPage("Output requests", tables=(InspectionTable(
                "Output requests", ("Index", "Type", "Position", "Variables"), output_rows,
                tuple(EntityReference("output", (step_index, index)) for index in range(len(output_rows))),
            ),)))
        return EntityInspection(f"Step {step.name}", "step", step_index, tuple(pages))

    def _inspect_boundary(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.boundaries[index]
        node_ids = resolve_displacement_node_ids(self.model, item)
        fields = (
            ("Step", step.name), ("Type", "Displacement BC"), ("Target region", str(item.target)),
            ("Scope type", displacement_target_kind(item)),
            ("Target node count", str(len(node_ids))),
            ("Constrained components", _component_range(item.first_component, item.last_component)),
            ("Value", format_number(item.value)),
        )
        rows = tuple((f"U{component}", format_number(item.value))
                     for component in range(item.first_component, item.last_component + 1))
        tables = (InspectionTable("Components and values", ("Component", "Value"), rows),) if len(rows) > 1 else ()
        return EntityInspection("Boundary conditions", "boundary", key, (InspectionPage("Boundary conditions", fields, tables),))

    def _inspect_cload(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.cloads[index]
        fields = (("Step", step.name), ("Type", "Nodal force"), ("Target", str(item.target)),
                  ("Target node count", str(len(self.target_node_ids(item.target)))),
                  ("Component", f"U{item.component}"), ("Value", format_number(item.value)))
        return EntityInspection("Nodal force", "cload", key, (InspectionPage("Loads", fields),))

    def _inspect_surface_load(self, key: object) -> EntityInspection:
        return self._inspect_distributed_load("surface_load", key)

    def _inspect_edge_load(self, key: object) -> EntityInspection:
        return self._inspect_distributed_load("edge_load", key)

    def _inspect_line_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.line_loads[index]
        fields = (
            ("Step", step.name),
            ("Type", "Edge force (distributed beam load)"),
            ("Target", str(item.target)),
            ("Target element count", str(len(self.target_element_ids(item.target)))),
            (
                "Coordinate system",
                (
                    "Local (resolved beam coordinates)"
                    if item.coordinate_system == "local"
                    else "Global"
                ),
            ),
            ("Load vector", ", ".join(format_number(value) for value in item.vector)),
        )
        return EntityInspection(
            "Edge force", "line_load", key,
            (InspectionPage("Loads", fields),),
        )

    def _inspect_gravity_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.gravity_loads[index]
        target = (
            "Entire model"
            if item.target is None
            else str(item.target)
        )
        fields = (
            ("Step", step.name),
            ("Type", "Gravity"),
            ("Target", target),
            (
                "Acceleration vector",
                ", ".join(
                    format_number(value)
                    for value in item.acceleration
                ),
            ),
        )
        return EntityInspection(
            "Gravity", "gravity_load", key,
            (InspectionPage("Loads", fields),),
        )

    def _inspect_body_load(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.body_loads[index]
        fields = (
            ("Step", step.name),
            ("Type", "Body force"),
            ("Target", str(item.target)),
            (
                "Force density vector",
                ", ".join(format_number(value) for value in item.vector),
            ),
        )
        return EntityInspection(
            "Body force",
            "body_load",
            key,
            (InspectionPage("Loads", fields),),
        )

    def _inspect_distributed_load(self, kind: str, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        is_surface = kind == "surface_load"
        item = step.surface_loads[index] if is_surface else step.edge_loads[index]
        region = item.surface if is_surface else item.edge
        members = self.model.surfaces[region].faces if is_surface else self.model.edges[region].edges
        fields = [("Step", step.name), ("Type", _load_type_label(item.load_type)),
                  ("Target", region), ("Face or edge count", str(len(members)))]
        if item.load_type != "pressure" and item.vector:
            fields.append(("Direction or vector", ", ".join(format_number(value) for value in item.vector)))
        if item.magnitude is not None:
            fields.append((("Pressure magnitude" if item.load_type == "pressure" else "Value"), format_number(item.magnitude)))
        return EntityInspection("Surface force" if is_surface else "Edge force", kind, key, (InspectionPage("Loads", tuple(fields)),))

    def _inspect_output(self, key: object) -> EntityInspection:
        step_index, index = key
        step = self.model.steps[step_index]
        item = step.outputs[index]
        fields = (("Step", step.name), ("Type", _output_kind(item.kind)),
                  ("Output position", _output_target(item.target)), ("Variables", ", ".join(item.variables)))
        return EntityInspection("Output requests", "output", key, (InspectionPage("Output requests", fields),))

    def _reference_names(self, references: tuple[EntityReference, ...] | list[EntityReference]) -> str:
        names = []
        for reference in references:
            if reference.kind in {
                "boundary",
                "cload",
                "surface_load",
                "edge_load",
                "line_load",
                "body_load",
                "gravity_load",
            }:
                step_index, index = reference.key
                step = self.model.steps[step_index]
                labels = {
                    "boundary": "Boundary conditions",
                    "cload": "Nodal force",
                    "surface_load": "Surface force",
                    "edge_load": "Edge force",
                    "line_load": "Edge force",
                    "body_load": "Body force",
                    "gravity_load": "Gravity",
                }
                names.append(f"{step.name} / {labels[reference.kind]} {index + 1}")
        return ", ".join(names) or "—"


def _require_result_provider(
    value: ResultProvider | None,
) -> ResultProvider | None:
    if value is not None and type(value) is not ResultProvider:
        raise TypeError("result_provider must be exactly ResultProvider or None")
    return value


def _provider_result_table(
    field_entry: ResultInspectionField,
    *,
    section_point_labels: Mapping[int, str] | None = None,
) -> InspectionTable:
    if type(field_entry) is not ResultInspectionField:
        raise TypeError(
            "provider result fields must be ResultInspectionField values"
        )
    availability = field_entry.availability
    descriptor = availability.descriptor
    include_section_points = result_field_has_section_points(
        descriptor.field_id
    )
    columns = (
        _SECTION_POINT_RESULT_COLUMNS
        if include_section_points
        else _RESULT_COLUMNS
    )
    state_label = _RESULT_STATE_LABELS[availability.state]
    field_label = _localized_result_field(
        descriptor,
        section_point_labels=section_point_labels,
    )
    if descriptor.unit_label is not None:
        field_label = f"{field_label} [{descriptor.unit_label}]"
    title = f"{field_label} ({state_label})"
    diagnostic = _diagnostic_summary(availability.diagnostics)
    if availability.state is not FieldState.READY:
        variable = descriptor.field_id.variable.value
        rows = tuple(
            (
                state_label,
                _localized_result_component(component, variable),
                *("—" for _column in columns[2:-1]),
                diagnostic,
            )
            for component in descriptor.columns
        )
        return InspectionTable(
            title,
            columns,
            rows,
            tuple(None for _row in rows),
        )

    rows: list[tuple[str, ...]] = []
    references: list[EntityReference | None] = []
    variable = descriptor.field_id.variable.value
    for component_result in field_entry.component_results:
        component = _localized_result_component(
            component_result.query.component,
            variable,
        )
        if not component_result.records:
            rows.append(
                (
                    state_label,
                    component,
                    *("—" for _column in columns[2:-1]),
                    diagnostic,
                )
            )
            references.append(None)
            continue
        for record in component_result.records:
            location = record.location
            section_values = (
                (
                    (
                        "—"
                        if location.section_point is None
                        else section_point_relative_position_label(
                            location.section_point
                        )
                    ),
                    format_number(
                        None
                        if location.section_point is None
                        else location.section_point.local_y
                    ),
                    format_number(
                        None
                        if location.section_point is None
                        else location.section_point.local_z
                    ),
                )
                if include_section_points
                else ()
            )
            rows.append(
                (
                    state_label,
                    component,
                    format_number(record.value),
                    format_number(location.node_id),
                    format_number(location.element_id),
                    format_number(location.integration_point),
                    format_number(location.local_node),
                    *section_values,
                    (
                        "—"
                        if location.region_key is None
                        else encode_result_region_key(location.region_key)
                    ),
                    _averaged_label(location.averaged),
                    diagnostic,
                )
            )
            references.append(_result_location_reference(location))
    return InspectionTable(
        title,
        columns,
        tuple(rows),
        tuple(references),
    )


def _localized_result_component(component: str, variable: str) -> str:
    if component == "Magnitude":
        return {
            "U": "Displacement magnitude",
            "RF": "Reaction Force magnitude",
        }.get(variable, "Magnitude")
    return _RESULT_COMPONENT_LABELS.get(component, component)


def _localized_result_field(
    descriptor: Any,
    *,
    section_point_labels: Mapping[int, str] | None = None,
) -> str:
    field_id = descriptor.field_id
    base = _RESULT_VARIABLE_LABELS.get(
        field_id.variable.value,
        descriptor.label_key,
    )
    if field_id.position is FieldPosition.NODE:
        return base
    position = result_field_position_label(
        field_id,
        section_point_labels=section_point_labels,
    )
    return f"{base} ({position})"


def _averaged_label(value: bool | None) -> str:
    if value is None:
        return "—"
    return "Yes" if value else "No"


def _result_location_reference(location: Any) -> EntityReference | None:
    if location.element_id is not None:
        return EntityReference("element", int(location.element_id))
    if location.node_id is not None:
        return EntityReference("node", int(location.node_id))
    return None


def format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, Real):
        number = float(value)
        if abs(number) < 5.0e-13:
            return "0"
        return f"{number:.6g}"
    return str(value)


def _vector_text(values: object) -> str:
    return ", ".join(
        format_number(value)
        for value in tuple(values)
    )


def _diagnostic_summary(diagnostics: object) -> str:
    values = tuple(diagnostics)
    return "; ".join(
        (
            f"[{getattr(item, 'code', 'diagnostic')}] "
            f"{getattr(item, 'message', str(item))}"
        )
        for item in values
    ) or "—"


def _counter_text(values: Any) -> str:
    return ", ".join(f"{name}: {count}" for name, count in Counter(values).items()) or "—"


def _component_range(first: int, last: int) -> str:
    return f"U{first}" if first == last else ", ".join(f"U{value}" for value in range(first, last + 1))


def _is_physical_property(name: str, value: object) -> bool:
    if name.startswith("_") or name in {"material", "element_set", "abaqus_type"}:
        return False
    return isinstance(value, (Real, str)) and name in {
        "E", "nu", "rho", "density", "thickness", "plane_type",
        "area", "A", "I", "Iyy", "Izz", "J", "section_type",
        "height", "width", "radius", "inner_radius", "outer_radius",
    }


def _property_label(name: str) -> str:
    return {
        "E": "Elastic modulus E", "nu": "Poisson's ratio ν", "rho": "Density ρ",
        "density": "Density ρ", "area": "Section area", "A": "Section area",
        "I": "Moment of inertia", "Iyy": "Moment of inertia Iyy", "Izz": "Moment of inertia Izz",
        "J": "Torsion constant J", "section_type": "Section type",
        "height": "Rectangle height (local z)", "width": "Rectangle width (local y)",
        "radius": "Radius", "inner_radius": "Inner radius", "outer_radius": "Outer radius",
    }.get(name, name)


def _plane_label(value: object) -> str:
    return "Plane strain" if str(value).lower().startswith("strain") else "Plane stress"


def _procedure_label(value: str) -> str:
    return "Linear static" if str(value).lower() == "static" else str(value)


def _output_kind(value: str) -> str:
    return {"field": "Field output", "history": "History output"}.get(value, value)


def _output_target(value: str) -> str:
    return {"node": "Node", "element": "Element"}.get(value, value)


def _load_type_label(value: str) -> str:
    return "Pressure" if str(value).lower() == "pressure" else "Surface or edge force"


def _load_direction(item: Any) -> str:
    if str(item.load_type).lower() == "pressure":
        return "Normal"
    return ", ".join(format_number(value) for value in item.vector) or "—"
