"""Conservative, provider-safe inspection of Abaqus input capabilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..capabilities import (
    CapabilityDisposition,
    SUPPORTED_BOUNDARY_LABELS,
    SUPPORTED_DLOAD_LABELS,
    SUPPORTED_DSLOAD_LABELS,
    SUPPORTED_ELEMENT_TYPES,
    keyword_capability,
)
from ..diagnostics import DiagnosticCode, has_errors, make_diagnostic
from ..schemas import Diagnostic, DiagnosticSeverity, ResourceLimits


_COMMON_UNSUPPORTED_PROCEDURES = frozenset(
    {
        "buckle",
        "coupled temperature-displacement",
        "direct cyclic",
        "dynamic",
        "frequency",
        "geostatic",
        "heat transfer",
        "mass diffusion",
        "modal dynamic",
        "random response",
        "response spectrum",
        "soils",
        "steady state dynamics",
        "visco",
    }
)
_LOAD_KEYWORDS = frozenset({"cload", "dload", "dsload"})
_MAX_DIAGNOSTICS = 96
_MAX_SET_MEMBERSHIP_IDS = 2_000_000
_SAFE_TOKEN_PATTERN = re.compile(r"[^a-z0-9 _-]+")
_FACE_LABEL_PATTERN = re.compile(r"S[1-6]\Z")
_ELEMENT_NODE_COUNTS = {
    "CPS3": 3,
    "CPE3": 3,
    "CPS4": 4,
    "CPE4": 4,
    "CPS6": 6,
    "CPE6": 6,
    "CPS8": 8,
    "CPE8": 8,
    "C3D4": 4,
    "C3D8": 8,
    "C3D10": 10,
    "C3D20": 20,
}


@dataclass(frozen=True)
class AbaqusKeywordInspection:
    """Safe aggregate of a local keyword scan.

    This object intentionally contains neither source lines nor a local path.
    ``keyword_inventory`` reports names and option names only; parameter values
    and data records remain local.
    """

    keyword_inventory: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Diagnostic, ...]
    keyword_count: int
    part_count: int
    instance_count: int
    explicit_step_count: int
    node_record_count: int
    element_record_count: int
    estimated_dofs: int
    collections_truncated: bool = False

    @property
    def has_blocking_diagnostics(self) -> bool:
        return has_errors(self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return only provider-safe, JSON-compatible scan data."""

        return {
            "keyword_inventory": [dict(item) for item in self.keyword_inventory],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "keyword_count": self.keyword_count,
            "part_count": self.part_count,
            "instance_count": self.instance_count,
            "explicit_step_count": self.explicit_step_count,
            "node_record_count": self.node_record_count,
            "element_record_count": self.element_record_count,
            "estimated_dofs": self.estimated_dofs,
            "collections_truncated": self.collections_truncated,
        }


@dataclass(frozen=True)
class _Keyword:
    name: str
    parameters: Mapping[str, str]
    flags: frozenset[str]


@dataclass
class _StepScan:
    name: str
    static_count: int = 0
    unsupported_procedure: bool = False


@dataclass
class _InventoryItem:
    name: str
    disposition: CapabilityDisposition
    count: int = 0
    parameters: set[str] = field(default_factory=set)
    flags: set[str] = field(default_factory=set)


class _DiagnosticCollector:
    def __init__(self) -> None:
        self._items: list[Diagnostic] = []
        self._seen: set[tuple[str, str | None, str]] = set()
        self._truncated = False

    def add(
        self,
        code: DiagnosticCode,
        message: str,
        *,
        severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
        entity: str | None = None,
        step: str | None = None,
        remediation: str | None = None,
    ) -> None:
        key = (code.value, entity, message)
        if key in self._seen:
            return
        self._seen.add(key)
        if len(self._items) >= _MAX_DIAGNOSTICS - 1:
            self._truncated = True
            return
        self._items.append(
            make_diagnostic(
                code,
                message,
                source="abaqus_inspection",
                severity=severity,
                entity=entity,
                step=step,
                remediation=remediation,
            )
        )

    def finish(self) -> tuple[Diagnostic, ...]:
        if self._truncated:
            self._items.append(
                make_diagnostic(
                    DiagnosticCode.INVALID_INPUT,
                    "Additional input diagnostics were omitted to keep the report bounded.",
                    source="abaqus_inspection",
                    remediation="Reduce unsupported content and inspect the input again.",
                )
            )
        ignored = sorted(
            {
                item.entity.upper()
                for item in self._items
                if item.code == DiagnosticCode.IGNORED_METADATA.value
                and item.entity is not None
            }
        )
        if not ignored:
            return tuple(self._items)
        summary = make_diagnostic(
            DiagnosticCode.IGNORED_METADATA,
            "Recognized metadata with no effect on this analysis: "
            + ", ".join(f"*{name}" for name in ignored)
            + ".",
            source="abaqus_inspection",
            severity=DiagnosticSeverity.INFO,
        )
        result: list[Diagnostic] = []
        inserted = False
        for item in self._items:
            if item.code == DiagnosticCode.IGNORED_METADATA.value:
                if not inserted:
                    result.append(summary)
                    inserted = True
                continue
            result.append(item)
        return tuple(result)


class _ScanState:
    def __init__(
        self,
        max_inventory_entries: int,
        resource_limits: ResourceLimits,
    ) -> None:
        if (
            isinstance(max_inventory_entries, bool)
            or not isinstance(max_inventory_entries, int)
            or max_inventory_entries <= 0
        ):
            raise ValueError("max_inventory_entries must be a positive integer")
        self.max_inventory_entries = int(max_inventory_entries)
        self.resource_limits = resource_limits
        self.current_keyword: _Keyword | None = None
        self.current_keyword_data_record_count = 0
        self.current_step_index: int | None = None
        self.steps: list[_StepScan] = []
        self.scope = "model"
        self.part_count = 0
        self.instance_count = 0
        self.keyword_count = 0
        self.node_record_count = 0
        self.element_record_count = 0
        self.max_dofs_per_node = 0
        self.pending_element_value_count = 0
        self.element_record_size: int | None = None
        self.set_membership_id_count = 0
        self._part_names: list[str] = []
        self._instance_parts: list[str] = []
        self._part_open = False
        self._assembly_open = False
        self._instance_open = False
        self._node_ids: set[int] = set()
        self._element_ids: set[int] = set()
        self._material_names: set[str] = set()
        self._material_properties: set[tuple[str, str]] = set()
        self._current_material_name: str | None = None
        self._inventory: dict[str, _InventoryItem] = {}
        self._inventory_truncated = False
        self._blocking_keywords: set[str] = set()
        self.diagnostics = _DiagnosticCollector()

    def handle_keyword(self, keyword: _Keyword) -> None:
        self._finish_element_block()
        self.current_keyword = keyword
        self.current_keyword_data_record_count = 0
        self.keyword_count += 1
        safe_name = _safe_token(keyword.name, fallback="invalid-keyword")
        capability = keyword_capability(keyword.name)
        self._record_inventory(safe_name, keyword, capability.disposition)

        if capability.disposition == CapabilityDisposition.BLOCKING:
            if keyword.name in _COMMON_UNSUPPORTED_PROCEDURES:
                self._block(
                    DiagnosticCode.UNSUPPORTED_PROCEDURE,
                    "The input requests an analysis procedure that V0 cannot solve.",
                    safe_name,
                )
                if self.current_step_index is not None:
                    self.steps[self.current_step_index].unsupported_procedure = True
            elif keyword.name == "include":
                self._block(
                    DiagnosticCode.UNSUPPORTED_KEYWORD,
                    "Abaqus INCLUDE files are not expanded by the local V0 importer.",
                    safe_name,
                    remediation="Create one self-contained input deck before attaching it.",
                )
            else:
                self._block(
                    DiagnosticCode.UNSUPPORTED_KEYWORD,
                    (
                        f"*{safe_name.upper()} is not implemented by the "
                        "V0 importer."
                    ),
                    safe_name,
                )
            return

        missing = capability.required_parameters.difference(keyword.parameters)
        for parameter in sorted(missing):
            display_parameter = _safe_token(
                parameter,
                fallback="parameter",
            ).upper()
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                (
                    f"*{safe_name.upper()} requires the "
                    f"{display_parameter} parameter."
                ),
                f"{safe_name}:{_safe_token(parameter, fallback='parameter')}",
            )

        unknown_parameters = set(keyword.parameters).difference(
            capability.allowed_parameters
        )
        for parameter in sorted(unknown_parameters):
            display_parameter = _safe_token(
                parameter,
                fallback="parameter",
            ).upper()
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                (
                    f"*{safe_name.upper()} uses the unsupported "
                    f"{display_parameter} parameter."
                ),
                f"{safe_name}:{_safe_token(parameter, fallback='parameter')}",
            )

        unknown_flags = set(keyword.flags).difference(capability.allowed_flags)
        for flag in sorted(unknown_flags):
            display_flag = _safe_token(flag, fallback="flag").upper()
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                (
                    f"*{safe_name.upper()} uses the unsupported "
                    f"{display_flag} flag."
                ),
                f"{safe_name}:{_safe_token(flag, fallback='flag')}",
            )

        if capability.disposition == CapabilityDisposition.IGNORED:
            self.diagnostics.add(
                DiagnosticCode.IGNORED_METADATA,
                (
                    f"*{safe_name.upper()} is recognized metadata and does "
                    "not affect this analysis."
                ),
                severity=DiagnosticSeverity.INFO,
                entity=safe_name,
            )

        self._inspect_keyword_semantics(keyword, safe_name)

    def handle_data(self, line: str) -> None:
        keyword = self.current_keyword
        if keyword is None:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "The input contains a data record without a preceding keyword.",
                entity="orphan-data",
            )
            return

        name = keyword.name
        self.current_keyword_data_record_count += 1
        if name == "node":
            self._inspect_node_data(line)
        elif name == "element":
            self._inspect_element_data(line)
        elif name == "instance":
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "Instance translation or rotation data is not supported by V0.",
                "instance:transform",
            )
            return
        if (
            name in {"elastic", "density"}
            and self.current_keyword_data_record_count > 1
        ):
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                (
                    "V0 supports exactly one data record in each "
                    f"{name.upper()} keyword block."
                ),
                f"{name}:data",
            )
            return
        if name == "elastic":
            self._inspect_elastic_data(line)
        elif name == "density":
            self._inspect_density_data(line)
        elif name == "solid section":
            self._inspect_solid_section_data(line)
        elif name == "surface":
            self._inspect_surface_data(line)
        elif name == "boundary":
            self._inspect_boundary_data(line)
        elif name == "cload":
            self._inspect_cload_data(line)
        elif name in {"dload", "dsload"}:
            self._inspect_distributed_load_data(line, name)
        elif name in {"nset", "elset"}:
            self._inspect_set_data(line, keyword)
        elif name == "static":
            self._inspect_static_data(line)

    def finish(self) -> AbaqusKeywordInspection:
        self._finish_element_block()
        if self.part_count > 1:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 supports at most one Abaqus part.",
                "part:count",
                inventory_name="part",
            )
        if self.instance_count > 1:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 supports at most one Abaqus instance.",
                "instance:count",
                inventory_name="instance",
            )
        if self.instance_count and self.part_count != 1:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "An identity instance requires exactly one local part in V0.",
                "instance:part",
                inventory_name="instance",
            )
        if (
            self.part_count == 1
            and self.instance_count == 1
            and self._part_names
            and self._instance_parts
            and self._part_names[0].casefold() != self._instance_parts[0].casefold()
        ):
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "The single instance must reference the single local part.",
                "instance:part",
                inventory_name="instance",
            )
        if self._part_open or self._assembly_open or self._instance_open:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "An Abaqus part, assembly, instance, or step block is not closed.",
                entity="input-structure",
            )
        if self.current_step_index is not None:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "An Abaqus STEP block is not closed.",
                entity="step",
            )

        if len(self.steps) == 0:
            self.diagnostics.add(
                DiagnosticCode.UNSUPPORTED_PROCEDURE,
                "V0 requires exactly one explicit runnable linear static step.",
                entity="step",
            )
        elif len(self.steps) > 1:
            self.diagnostics.add(
                DiagnosticCode.MULTI_STEP_HISTORY_UNSUPPORTED,
                "V0 does not reproduce Abaqus multi-step history.",
                entity="step",
                remediation="Provide an input deck with one runnable static step.",
            )

        for step in self.steps:
            if step.static_count != 1 and not step.unsupported_procedure:
                self.diagnostics.add(
                    DiagnosticCode.UNSUPPORTED_PROCEDURE,
                    "Each runnable V0 step must contain exactly one STATIC procedure.",
                    entity="static",
                    step=_safe_token(step.name, fallback="step"),
                )

        inventory = []
        for name in sorted(self._inventory):
            item = self._inventory[name]
            disposition = (
                CapabilityDisposition.BLOCKING
                if name in self._blocking_keywords
                else item.disposition
            )
            inventory.append(
                {
                    "name": item.name,
                    "count": item.count,
                    "parameters": tuple(sorted(item.parameters)),
                    "flags": tuple(sorted(item.flags)),
                    "disposition": disposition.value,
                }
            )

        return AbaqusKeywordInspection(
            keyword_inventory=tuple(inventory),
            diagnostics=self.diagnostics.finish(),
            keyword_count=self.keyword_count,
            part_count=self.part_count,
            instance_count=self.instance_count,
            explicit_step_count=len(self.steps),
            node_record_count=self.node_record_count,
            element_record_count=self.element_record_count,
            estimated_dofs=(
                self.node_record_count * self.max_dofs_per_node
            ),
            collections_truncated=self._inventory_truncated,
        )

    def _record_inventory(
        self,
        safe_name: str,
        keyword: _Keyword,
        disposition: CapabilityDisposition,
    ) -> None:
        item = self._inventory.get(safe_name)
        if item is None:
            if len(self._inventory) >= self.max_inventory_entries:
                self._inventory_truncated = True
                return
            item = _InventoryItem(safe_name, disposition)
            self._inventory[safe_name] = item
        item.count += 1
        item.parameters.update(
            _safe_token(name, fallback="parameter")
            for name in keyword.parameters
        )
        item.flags.update(
            _safe_token(name, fallback="flag")
            for name in keyword.flags
        )

    def _inspect_keyword_semantics(self, keyword: _Keyword, safe_name: str) -> None:
        if keyword.name == "part":
            self.part_count += 1
            self.scope = "part"
            self._part_open = True
            self._part_names.append(keyword.parameters.get("name", ""))
        elif keyword.name == "end part":
            self.scope = "model"
            self._part_open = False
        elif keyword.name == "assembly":
            self.scope = "assembly"
            self._assembly_open = True
        elif keyword.name == "end assembly":
            self.scope = "model"
            self._assembly_open = False
        elif keyword.name == "instance":
            self.instance_count += 1
            self._instance_open = True
            self._instance_parts.append(keyword.parameters.get("part", ""))
        elif keyword.name == "end instance":
            self._instance_open = False
        elif keyword.name == "material":
            material_name = keyword.parameters.get("name", "")
            self._current_material_name = material_name or None
            if material_name:
                if material_name in self._material_names:
                    self._block(
                        DiagnosticCode.INVALID_INPUT,
                        (
                            "MATERIAL names must be unique within the "
                            "attached input."
                        ),
                        "material:name",
                    )
                else:
                    self._material_names.add(material_name)
        elif keyword.name in {"elastic", "density"}:
            self._inspect_material_property_keyword(keyword.name)
        elif keyword.name == "step":
            self.steps.append(_StepScan(keyword.parameters.get("name", "step")))
            self.current_step_index = len(self.steps) - 1
            nlgeom = keyword.parameters.get("nlgeom")
            if nlgeom is not None and _truthy_option(nlgeom):
                self._block(
                    DiagnosticCode.UNSUPPORTED_PROCEDURE,
                    "The linear static V0 solver does not support NLGEOM.",
                    "step:nlgeom",
                )
        elif keyword.name == "static":
            if self.current_step_index is None:
                self._block(
                    DiagnosticCode.UNSUPPORTED_PROCEDURE,
                    "STATIC must appear inside the single explicit STEP block.",
                    safe_name,
                )
            else:
                self.steps[self.current_step_index].static_count += 1
        elif keyword.name == "end step":
            self.current_step_index = None
        elif keyword.name == "element":
            element_type = keyword.parameters.get("type", "").upper()
            if element_type and element_type not in SUPPORTED_ELEMENT_TYPES:
                self._block(
                    DiagnosticCode.UNSUPPORTED_ELEMENT,
                    "The input contains an element formulation unsupported by V0.",
                    "element:type",
                )
            node_count = _ELEMENT_NODE_COUNTS.get(element_type)
            self.element_record_size = (
                None if node_count is None else node_count + 1
            )
            if element_type.startswith("C3D"):
                self.max_dofs_per_node = max(self.max_dofs_per_node, 3)
            elif element_type:
                self.max_dofs_per_node = max(self.max_dofs_per_node, 2)
            self._check_dof_limit()
        elif keyword.name == "surface":
            surface_type = keyword.parameters.get("type", "").upper()
            if surface_type and surface_type != "ELEMENT":
                self._block(
                    DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                    (
                        "*SURFACE TYPE="
                        f"{_safe_token(surface_type, fallback='value')} "
                        "is unsupported; V0 accepts only TYPE=ELEMENT."
                    ),
                    "surface:type",
                )
            if self.scope == "part":
                self._block(
                    DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                    "Part-level surfaces are not instantiated by the V0 importer.",
                    "surface:scope",
                    remediation="Define the element surface at model or assembly scope.",
                )
        elif keyword.name in _LOAD_KEYWORDS and self.current_step_index is None:
            self._block(
                DiagnosticCode.UNSUPPORTED_PROCEDURE,
                "Loads must be defined inside the single runnable STEP block.",
                safe_name,
            )

    def _inspect_node_data(self, line: str) -> None:
        values = _split_values(line)
        valid = (
            len(values) in {3, 4}
            and _is_integer(values[0])
            and _finite_numbers(values[1:]) is not None
        )
        if not valid:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "A NODE record must contain an integer ID and finite coordinates.",
                entity="node:data",
            )
            return
        if self.node_record_count < self.resource_limits.max_nodes:
            node_id = int(values[0])
            if node_id in self._node_ids:
                self._block(
                    DiagnosticCode.INVALID_INPUT,
                    "NODE identifiers must be unique within the attached input.",
                    "node:id",
                )
            else:
                self._node_ids.add(node_id)
        self.node_record_count += 1
        if self.node_record_count > self.resource_limits.max_nodes:
            self.diagnostics.add(
                DiagnosticCode.RESOURCE_LIMIT,
                "NODE records exceed the configured preflight limit.",
                entity="node:count",
            )
        self._check_dof_limit()

    def _inspect_element_data(self, line: str) -> None:
        values = _split_values(line)
        if not values or not all(_is_integer(value) for value in values):
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "An ELEMENT record must contain integer identifiers.",
                entity="element:data",
            )
            return
        if self.element_record_size is None:
            return
        completed = 0
        for value in values:
            if self.pending_element_value_count == 0:
                if (
                    self.element_record_count + completed
                    < self.resource_limits.max_elements
                ):
                    element_id = int(value)
                    if element_id in self._element_ids:
                        self._block(
                            DiagnosticCode.INVALID_INPUT,
                            (
                                "ELEMENT identifiers must be unique within "
                                "the attached input."
                            ),
                            "element:id",
                        )
                    else:
                        self._element_ids.add(element_id)
            self.pending_element_value_count += 1
            if self.pending_element_value_count == self.element_record_size:
                completed += 1
                self.pending_element_value_count = 0
        self.element_record_count += completed
        if self.element_record_count > self.resource_limits.max_elements:
            self.diagnostics.add(
                DiagnosticCode.RESOURCE_LIMIT,
                "ELEMENT records exceed the configured preflight limit.",
                entity="element:count",
            )

    def _finish_element_block(self) -> None:
        if self.current_keyword is not None and self.current_keyword.name == "element":
            if self.pending_element_value_count:
                self.diagnostics.add(
                    DiagnosticCode.INVALID_INPUT,
                    "An ELEMENT connectivity record is incomplete.",
                    entity="element:data",
                )
        self.pending_element_value_count = 0
        self.element_record_size = None

    def _check_dof_limit(self) -> None:
        estimated = self.node_record_count * self.max_dofs_per_node
        if estimated > self.resource_limits.max_dofs:
            self.diagnostics.add(
                DiagnosticCode.RESOURCE_LIMIT,
                "Estimated DOFs exceed the configured preflight limit.",
                entity="dof:count",
            )

    def _inspect_elastic_data(self, line: str) -> None:
        values = _split_values(line)
        if len(values) != 2:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 supports one isotropic ELASTIC record containing E and nu.",
                "elastic:data",
            )
            return
        numbers = _finite_numbers(values)
        if numbers is None or numbers[0] <= 0.0 or not -1.0 < numbers[1] < 0.5:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "The isotropic elastic constants are invalid or non-finite.",
                entity="elastic:data",
            )

    def _inspect_material_property_keyword(self, property_name: str) -> None:
        material_name = self._current_material_name
        if material_name is None:
            return
        identity = (material_name, property_name)
        if identity in self._material_properties:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                (
                    f"Each material may contain at most one {property_name.upper()} "
                    "keyword block."
                ),
                f"{property_name}:block",
            )
            return
        self._material_properties.add(identity)

    def _inspect_density_data(self, line: str) -> None:
        values = _split_values(line)
        numbers = _finite_numbers(values)
        if len(values) != 1:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 supports one scalar DENSITY record.",
                "density:data",
            )
        elif numbers is None or numbers[0] < 0.0:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "Density must be a finite non-negative number.",
                entity="density:data",
            )

    def _inspect_solid_section_data(self, line: str) -> None:
        values = _split_values(line, preserve_leading_empty=True)
        numbers = _finite_numbers(values)
        if self.current_keyword_data_record_count > 1 or len(values) != 1:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 supports at most one scalar SOLID SECTION thickness value.",
                "solid section:data",
            )
        elif numbers is None or numbers[0] <= 0.0:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "SOLID SECTION thickness must be finite and greater than zero.",
                entity="solid section:data",
            )

    def _inspect_surface_data(self, line: str) -> None:
        values = _split_values(line)
        if (
            len(values) != 2
            or not values[0]
            or _FACE_LABEL_PATTERN.fullmatch(values[1].upper()) is None
        ):
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "V0 surfaces require an element target and an S1 through S6 face label.",
                "surface:data",
            )

    def _inspect_boundary_data(self, line: str) -> None:
        values = _split_values(line)
        valid = 2 <= len(values) <= 4 and bool(values[0])
        if valid and not _is_integer(values[1]):
            valid = len(values) == 2 and values[1].upper() in SUPPORTED_BOUNDARY_LABELS
        elif valid:
            valid = (
                (len(values) < 3 or _is_integer(values[2]))
                and _finite_numbers(values[1:]) is not None
            )
        if not valid:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "The BOUNDARY record uses an unsupported label or data form.",
                "boundary:data",
            )

    def _inspect_cload_data(self, line: str) -> None:
        values = _split_values(line)
        valid = (
            len(values) == 3
            and bool(values[0])
            and _is_integer(values[1])
            and _finite_numbers(values[2:]) is not None
        )
        if not valid:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "CLOAD requires target, integer component, and finite value.",
                "cload:data",
            )

    def _inspect_distributed_load_data(self, line: str, source: str) -> None:
        values = _split_values(line, preserve_leading_empty=True)
        if len(values) < 2:
            self._unsupported_load_form(source)
            return
        label = values[1].upper()
        supported = (
            label in SUPPORTED_DLOAD_LABELS
            if source == "dload"
            else label in SUPPORTED_DSLOAD_LABELS
        )
        if not supported:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "The distributed-load label is not supported by V0.",
                f"{source}:label",
            )
            return

        if source == "dload" and label == "GRAV":
            valid = (
                len(values) == 6
                and _finite_numbers(values[2:]) is not None
                and _nonzero_direction(values[3:])
            )
        elif label in {"TRVEC", "TRSHR"}:
            expected_lengths = {5, 6} if label == "TRVEC" else {6}
            valid = (
                source == "dsload"
                and len(values) in expected_lengths
                and bool(values[0])
                and _finite_numbers(values[2:]) is not None
                and _nonzero_direction(values[3:])
            )
        else:
            valid = (
                len(values) == 3
                and bool(values[0])
                and _finite_numbers(values[2:]) is not None
            )
        if not valid:
            self._unsupported_load_form(source)

    def _unsupported_load_form(self, source: str) -> None:
        self._block(
            DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
            "The distributed-load record has an unsupported target or data shape.",
            f"{source}:data",
        )

    def _inspect_set_data(self, line: str, keyword: _Keyword) -> None:
        values = _split_values(line)
        valid = bool(values) and all(_is_integer(value) for value in values)
        if valid and "generate" in keyword.flags:
            valid = len(values) % 3 == 0 and all(
                int(values[index]) > 0 for index in range(2, len(values), 3)
            )
            if valid:
                expanded = sum(
                    max(
                        0,
                        (
                            int(values[index + 1])
                            - int(values[index])
                        )
                        // int(values[index + 2])
                        + 1,
                    )
                    for index in range(0, len(values), 3)
                )
                self.set_membership_id_count += expanded
                if self.set_membership_id_count > _MAX_SET_MEMBERSHIP_IDS:
                    self.diagnostics.add(
                        DiagnosticCode.RESOURCE_LIMIT,
                        "The GENERATE record exceeds the safe local expansion limit.",
                        entity=f"{keyword.name}:generate",
                    )
                    return
        elif valid:
            self.set_membership_id_count += len(values)
            if self.set_membership_id_count > _MAX_SET_MEMBERSHIP_IDS:
                self.diagnostics.add(
                    DiagnosticCode.RESOURCE_LIMIT,
                    "Set membership data exceeds the safe local expansion limit.",
                    entity=f"{keyword.name}:data",
                )
                return
        if not valid:
            self.diagnostics.add(
                DiagnosticCode.INVALID_INPUT,
                "The set data record is not a valid integer list or GENERATE triplet.",
                entity=f"{keyword.name}:data",
            )

    def _inspect_static_data(self, line: str) -> None:
        values = _split_values(line)
        if len(values) > 4 or _finite_numbers(values) is None:
            self._block(
                DiagnosticCode.UNSUPPORTED_KEYWORD_OPTION,
                "The STATIC time-control record has an unsupported data form.",
                "static:data",
            )

    def _block(
        self,
        code: DiagnosticCode,
        message: str,
        entity: str,
        *,
        remediation: str | None = None,
        inventory_name: str | None = None,
    ) -> None:
        keyword_name = (
            _safe_token(inventory_name, fallback="invalid-keyword")
            if inventory_name is not None
            else (
                _safe_token(self.current_keyword.name, fallback="invalid-keyword")
                if self.current_keyword is not None
                else entity.split(":", 1)[0]
            )
        )
        self._blocking_keywords.add(keyword_name)
        self.diagnostics.add(
            code,
            message,
            entity=entity,
            remediation=remediation,
        )


def inspect_abaqus_keywords(
    path: str | Path,
    *,
    max_inventory_entries: int = 64,
    resource_limits: ResourceLimits | None = None,
) -> AbaqusKeywordInspection:
    """Inspect one local ``.inp`` without retaining comments or data records."""

    limits = resource_limits or ResourceLimits()
    if not isinstance(limits, ResourceLimits):
        raise TypeError("resource_limits must be a ResourceLimits instance")
    state = _ScanState(max_inventory_entries, limits)
    input_path = Path(path)
    try:
        if input_path.stat().st_size > limits.max_input_bytes:
            state.diagnostics.add(
                DiagnosticCode.RESOURCE_LIMIT,
                "The attached input exceeds the configured preflight byte limit.",
                entity="input:bytes",
            )
            return state.finish()
        with input_path.open("r", encoding="utf-8", errors="strict") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("**"):
                    continue
                if line.startswith("*"):
                    state.handle_keyword(_parse_keyword(line))
                else:
                    state.handle_data(line)
    except UnicodeError:
        state.diagnostics.add(
            DiagnosticCode.INVALID_INPUT,
            "The attached Abaqus input must be valid UTF-8 text.",
            entity="input",
        )
    except OSError:
        state.diagnostics.add(
            DiagnosticCode.INVALID_INPUT,
            "The attached Abaqus input could not be opened locally.",
            entity="input",
        )
    return state.finish()


def _parse_keyword(line: str) -> _Keyword:
    parts = [part.strip() for part in line[1:].split(",") if part.strip()]
    if not parts:
        return _Keyword("", {}, frozenset())
    name = parts[0].lower()
    parameters: dict[str, str] = {}
    flags: set[str] = set()
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            parameters[key.strip().lower()] = value.strip()
        else:
            flags.add(part.lower())
    return _Keyword(name, parameters, frozenset(flags))


def _split_values(
    line: str,
    *,
    preserve_leading_empty: bool = False,
) -> list[str]:
    parts = [part.strip() for part in line.split(",")]
    while parts and not parts[-1]:
        parts.pop()
    if preserve_leading_empty:
        return parts
    return [part for part in parts if part]


def _finite_numbers(values: list[str]) -> tuple[float, ...] | None:
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return numbers if all(math.isfinite(value) for value in numbers) else None


def _nonzero_direction(values: list[str]) -> bool:
    numbers = _finite_numbers(values)
    return numbers is not None and any(value != 0.0 for value in numbers)


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _truthy_option(value: str) -> bool:
    return value.strip().casefold() not in {"", "0", "false", "no", "off"}


def _safe_token(value: str, *, fallback: str) -> str:
    normalized = " ".join(str(value).strip().casefold().split())
    normalized = _SAFE_TOKEN_PATTERN.sub("_", normalized).strip(" _-")
    return normalized[:64] or fallback


__all__ = ["AbaqusKeywordInspection", "inspect_abaqus_keywords"]
