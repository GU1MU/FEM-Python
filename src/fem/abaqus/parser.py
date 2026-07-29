from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .contracts import RETIRED_ELEMENT_TYPES
from .deck import (
    AbaqusBeamSectionData,
    AbaqusBoundary,
    AbaqusCload,
    AbaqusDataRecordEvidence,
    AbaqusDeck,
    AbaqusDistributedLoad,
    AbaqusElement,
    AbaqusKeywordOccurrence,
    AbaqusMaterial,
    AbaqusNodeRecord,
    AbaqusOutputRequest,
    AbaqusSection,
    AbaqusSolidSectionData,
    AbaqusSourceSpan,
    AbaqusStep,
    AbaqusSurfaceFace,
)
from .errors import (
    AbaqusParseError,
    AbaqusSourceLocation,
    UnsupportedAbaqusFeatureError,
)


_SUPPORTED_ELEMENT_NODE_COUNTS = {
    "T3D2": 2,
    "B31": 2,
    "CPS3": 3,
    "CPE3": 3,
    "CPS6": 6,
    "CPE6": 6,
    "CPS4": 4,
    "CPE4": 4,
    "CPS8": 8,
    "CPE8": 8,
    "C3D4": 4,
    "C3D10": 10,
    "C3D8": 8,
    "C3D20": 20,
}

_ABAQUS_REAL_PATTERN = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?\Z"
)
_OUTPUT_CONTEXT_KEYWORDS = frozenset(
    {
        "output",
        "field output",
        "history output",
        "node output",
        "element output",
    }
)


@dataclass(frozen=True)
class Keyword:
    """Parsed Abaqus keyword line."""
    name: str
    params: dict[str, str]
    flags: set[str]
    occurrence: AbaqusKeywordOccurrence

    @property
    def location(self) -> AbaqusSourceLocation:
        return self.occurrence.location


def parse_abaqus_real(
    value: str,
    *,
    location: AbaqusSourceLocation | None = None,
    field: str = "Abaqus numeric field",
    allow_nonfinite: bool = False,
) -> float:
    """Parse one finite Abaqus real using E/e/D/d exponent notation."""

    if not isinstance(value, str):
        raise AbaqusParseError(
            f"{field} must be text",
            code="abaqus.real.type",
            location=location,
            record=value,
        )
    token = value.strip()
    if allow_nonfinite:
        try:
            result = float(
                token.replace("D", "E").replace("d", "e")
            )
        except (TypeError, ValueError) as exc:
            raise AbaqusParseError(
                f"{field} must be an Abaqus real, got {value!r}",
                code="abaqus.real.invalid",
                location=location,
                record=value,
            ) from exc
        return result
    if not token or _ABAQUS_REAL_PATTERN.fullmatch(token) is None:
        raise AbaqusParseError(
            f"{field} must be a finite Abaqus real, got {value!r}",
            code="abaqus.real.invalid",
            location=location,
            record=value,
        )
    result = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise AbaqusParseError(
            f"{field} must be finite, got {value!r}",
            code="abaqus.real.nonfinite",
            location=location,
            record=value,
        )
    return result


def parse_file(path: str | Path) -> AbaqusDeck:
    """Parse a supported subset of an Abaqus input file."""
    inp_path = Path(path)
    deck = AbaqusDeck(name=inp_path.stem)
    state = _ParserState(deck, inp_path)

    raw_bytes = inp_path.read_bytes()
    text = _decode_abaqus_text(raw_bytes, inp_path)

    physical_lines = text.splitlines()
    index = 0
    while index < len(physical_lines):
        raw_line = physical_lines[index]
        stripped = raw_line.strip()
        line_number = index + 1
        if stripped.startswith("**"):
            index += 1
            continue
        if stripped.startswith("*"):
            keyword, index = _assemble_keyword(
                physical_lines,
                index,
                inp_path,
            )
            deck.keyword_occurrences.append(keyword.occurrence)
            state.handle_keyword(keyword)
            continue

        location = AbaqusSourceLocation(
            inp_path,
            line_number,
            state.keyword.name if state.keyword is not None else None,
        )
        if not stripped:
            state.handle_blank(location, raw_line)
        else:
            state.handle_data(
                _split_values(
                    raw_line,
                    preserve_leading_empty=(
                        state.mode
                        in {
                            "dload",
                            "dsload",
                            "solid_section",
                            "beam_section",
                        }
                    ),
                    preserve_all_empty=(state.mode == "node"),
                ),
                location,
                raw_line,
            )
        index += 1

    state.finish()
    return deck


def _decode_abaqus_text(raw_bytes: bytes, inp_path: Path) -> str:
    """Decode an Abaqus deck without dropping or replacing source bytes."""

    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as utf8_error:
        if not raw_bytes.startswith(b"\xef\xbb\xbf"):
            try:
                return raw_bytes.decode("gb18030")
            except UnicodeDecodeError:
                pass
        line = raw_bytes[: utf8_error.start].count(b"\n") + 1
        raise AbaqusParseError(
            "Abaqus input is not valid UTF-8 or GB18030 text",
            code="abaqus.text.invalid_utf8",
            location=AbaqusSourceLocation(inp_path, line),
            record=raw_bytes[utf8_error.start : utf8_error.end],
            remediation=(
                "Save the input as valid UTF-8 or GB18030/GBK text."
            ),
        ) from utf8_error


class _ParserState:
    """State machine for the supported Abaqus keyword subset."""

    def __init__(self, deck: AbaqusDeck, path: Path):
        self.deck = deck
        self.path = path
        self.mode: str | None = None
        self.keyword: Keyword | None = None
        self.current_material: AbaqusMaterial | None = None
        self.current_step: AbaqusStep | None = None
        self.current_output_kind: str | None = None
        self.current_output_target: str | None = None
        self.current_output_parent: Keyword | None = None
        self.scope = "model"
        self.current_set_accepts_data = True
        self.current_surface_accepts_data = True
        self.pending_element_values: list[str] = []
        self.pending_element_locations: list[AbaqusSourceLocation] = []

    def handle_keyword(self, keyword: Keyword) -> None:
        """Dispatch a keyword line and update data mode."""
        self._finish_active_mode()
        self.keyword = keyword
        self.mode = None
        self.current_set_accepts_data = True
        self.current_surface_accepts_data = True
        if keyword.name not in _OUTPUT_CONTEXT_KEYWORDS:
            self._clear_output_context()

        if keyword.name == "part":
            self.scope = "part"
            return

        if keyword.name == "end part":
            self.scope = "model"
            return

        if keyword.name == "assembly":
            self.scope = "assembly"
            return

        if keyword.name == "end assembly":
            self.scope = "model"
            return

        if keyword.name == "node":
            self.mode = "node"
            return

        if keyword.name == "element":
            element_type = _required_param(keyword, "type").upper()
            if element_type in RETIRED_ELEMENT_TYPES:
                replacement = "B31" if element_type == "BEAM2" else "T3D2"
                raise UnsupportedAbaqusFeatureError(
                    (
                        f"Abaqus wire element type {element_type} is a retired "
                        "custom alias"
                    ),
                    code="abaqus.element_type.retired",
                    location=keyword.location,
                    record=element_type,
                    remediation=f"Use the standard Abaqus type {replacement}.",
                )
            if element_type in {"B31", "T3D2"}:
                _validate_keyword_parameters(
                    keyword,
                    allowed=frozenset({"type", "elset"}),
                    required=frozenset({"type"}),
                )
            self.mode = "element"
            return

        if keyword.name == "nset":
            self._start_set("nset")
            return

        if keyword.name == "elset":
            self._start_set("elset")
            return

        if keyword.name == "surface":
            name = _required_param(keyword, "name")
            self.current_surface_accepts_data = self._prepare_scoped_collection(
                self.deck.surfaces,
                self.deck.surface_scopes,
                name,
                self._keyword_scope(keyword),
            )
            self.mode = "surface"
            return

        if keyword.name == "material":
            name = _required_param(keyword, "name")
            self.current_material = AbaqusMaterial(name)
            self.deck.materials[name] = self.current_material
            return

        if keyword.name == "density":
            if self.current_material is None:
                raise AbaqusParseError(
                    "*Density must follow *Material",
                    location=keyword.location,
                )
            self.current_material.density_keyword_count += 1
            self.mode = "density"
            return

        if keyword.name == "elastic":
            if self.current_material is None:
                raise AbaqusParseError(
                    "*Elastic must follow *Material",
                    location=keyword.location,
                )
            self.current_material.elastic_keyword_count += 1
            self.mode = "elastic"
            return

        if keyword.name == "solid section":
            self._start_solid_section(keyword)
            return

        if keyword.name == "beam section":
            self._start_beam_section(keyword)
            return

        if keyword.name == "truss section":
            raise UnsupportedAbaqusFeatureError(
                "*TRUSS SECTION is a retired custom input form",
                code="abaqus.truss_section.retired",
                location=keyword.location,
                remediation=(
                    "Use *SOLID SECTION, ELSET=..., MATERIAL=... followed by "
                    "one area data record."
                ),
            )

        if keyword.name.endswith("section"):
            raise UnsupportedAbaqusFeatureError(
                f"*{keyword.name.upper()} is outside the supported subset",
                code="abaqus.section.keyword_unsupported",
                location=keyword.location,
                remediation="Use exact *SOLID SECTION or *BEAM SECTION syntax.",
            )
            return

        if keyword.name == "step":
            self.current_step = AbaqusStep(
                _required_param(keyword, "name"),
                metadata=dict(keyword.params),
                keyword_location=keyword.location,
            )
            self._clear_output_context()
            self.deck.steps.append(self.current_step)
            return

        if keyword.name == "static":
            step = self._ensure_step()
            step.procedure = "static"
            step.procedure_present = True
            step.procedure_location = keyword.location
            step.procedure_count += 1
            self.mode = "static"
            return

        if keyword.name == "boundary":
            self._ensure_step()
            self.mode = "boundary"
            return

        if keyword.name == "cload":
            self._ensure_step()
            self.mode = "cload"
            return

        if keyword.name == "dload":
            self._ensure_step()
            self.mode = "dload"
            return

        if keyword.name == "dsload":
            self._ensure_step()
            self.mode = "dsload"
            return

        if keyword.name == "output":
            self._start_output_block(keyword)
            return

        if keyword.name in ("field output", "history output"):
            self._start_named_output(keyword)
            return

        if keyword.name == "node output":
            self._start_output_data(keyword, "node")
            return

        if keyword.name == "element output":
            self._start_output_data(keyword, "element")
            return

        if keyword.name == "end step":
            self.current_step = None
            self._clear_output_context()

    def handle_blank(
        self,
        location: AbaqusSourceLocation,
        raw: str,
    ) -> None:
        """Preserve a positional blank only inside a section block."""

        if self.mode == "solid_section":
            section = self.deck.sections[-1]
            data = section.data
            if (
                isinstance(data, AbaqusSolidSectionData)
                and not data.record_present
            ):
                self.handle_data([""], location, raw)
            return
        if self.mode == "beam_section":
            section = self.deck.sections[-1]
            data = section.data
            if (
                isinstance(data, AbaqusBeamSectionData)
                and data.geometry.present
                and not data.orientation.present
            ):
                self.handle_data([""], location, raw)
            return

    def handle_data(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
        raw: str,
    ) -> None:
        """Handle one data line in the current mode."""
        if not values and self.mode not in {"solid_section", "beam_section"}:
            return

        if self.mode == "node":
            self._add_node(values, location)
        elif self.mode == "element":
            self._consume_element_values(values, location)
        elif self.mode == "nset":
            self._extend_set(
                self.deck.node_sets,
                self.deck.node_set_scopes,
                values,
            )
        elif self.mode == "elset":
            self._extend_set(
                self.deck.element_sets,
                self.deck.element_set_scopes,
                values,
            )
        elif self.mode == "surface":
            self._add_surface(values)
        elif self.mode == "density":
            self._add_density(values, location, raw)
        elif self.mode == "elastic":
            self._add_elastic(values, location, raw)
        elif self.mode == "solid_section":
            self._add_solid_section_data(values, location)
        elif self.mode == "beam_section":
            self._add_beam_section_data(values, location, raw)
        elif self.mode == "static":
            self._ensure_step().metadata["time"] = tuple(
                parse_abaqus_real(
                    value,
                    location=location,
                    field="*STATIC value",
                )
                for value in values
            )
        elif self.mode == "boundary":
            self._add_boundary(values, location)
        elif self.mode == "cload":
            self._add_cload(values, location)
        elif self.mode == "dload":
            self._add_distributed_load(values, "dload", location)
        elif self.mode == "dsload":
            self._add_distributed_load(values, "dsload", location)
        elif self.mode == "output":
            self._add_output_request(values)

    def _start_set(self, mode: str) -> None:
        name_key = "nset" if mode == "nset" else "elset"
        name = _required_param(self.keyword, name_key)
        target = self.deck.node_sets if mode == "nset" else self.deck.element_sets
        scopes = self.deck.node_set_scopes if mode == "nset" else self.deck.element_set_scopes
        self.current_set_accepts_data = self._prepare_scoped_collection(
            target,
            scopes,
            name,
            self._keyword_scope(self.keyword),
        )
        self.mode = mode

    def _extend_set(
        self,
        target: dict[str, list[int]],
        scopes: dict[str, str],
        values: list[str],
    ) -> None:
        if not self.current_set_accepts_data:
            return
        name = _required_param(self.keyword, self.mode)
        scope = self._keyword_scope(self.keyword)
        self._prepare_scoped_collection(target, scopes, name, scope)
        if self.keyword and "generate" in self.keyword.flags:
            ids = _generate_ids(values)
        else:
            ids = [int(value) for value in values]
        target[name].extend(ids)

    def _add_node(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        if len(values) < 3:
            raise AbaqusParseError(
                "*NODE requires an ID and at least two coordinates",
                code="abaqus.node.record_shape",
                location=location,
                record=tuple(values),
            )
        node_id = _parse_int(values[0], location, "node ID")
        x = parse_abaqus_real(values[1], location=location, field="node x")
        y = parse_abaqus_real(values[2], location=location, field="node y")
        z = (
            parse_abaqus_real(values[3], location=location, field="node z")
            if len(values) > 3 and values[3]
            else 0.0
        )
        coordinates = (x, y, z)
        self.deck.nodes[node_id] = coordinates
        self.deck.node_records[node_id] = AbaqusNodeRecord(
            node_id,
            coordinates,
            tuple(values[4:]),
            self.keyword.location if self.keyword is not None else None,
            location,
            tuple(values),
        )

    def _add_element(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        element_type = _required_param(self.keyword, "type")
        element_set = self.keyword.params.get("elset") if self.keyword else None
        element = AbaqusElement(
            _parse_int(values[0], location, "element ID"),
            tuple(
                _parse_int(value, location, "element connectivity")
                for value in values[1:]
            ),
            element_type,
            element_set,
            self.keyword.location if self.keyword is not None else None,
            location,
            tuple(values),
        )
        self.deck.elements.append(element)
        if element_set is not None:
            scope = self._keyword_scope(self.keyword)
            if self._prepare_scoped_collection(
                self.deck.element_sets,
                self.deck.element_set_scopes,
                element_set,
                scope,
            ):
                self.deck.element_sets[element_set].append(element.id)

    def _consume_element_values(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        element_type = _required_param(self.keyword, "type").upper()
        if element_type in {"B31", "T3D2"}:
            if element_type == "B31" and len(values) == 4:
                raise UnsupportedAbaqusFeatureError(
                    (
                        "B31 additional orientation nodes are outside the "
                        "assignment-scoped orientation subset"
                    ),
                    code="abaqus.b31.orientation_node_unsupported",
                    location=location,
                    record=tuple(values),
                    remediation=(
                        "Remove the third connectivity node and provide "
                        "approximate n1 on the *BEAM SECTION data record."
                    ),
                )
            if len(values) != 3:
                raise AbaqusParseError(
                    (
                        f"{element_type} connectivity must be one physical "
                        "record containing element ID and two node IDs"
                    ),
                    code="abaqus.line.connectivity_shape",
                    location=location,
                    record=tuple(values),
                )
            self._add_element(values, location)
            return

        node_count = _SUPPORTED_ELEMENT_NODE_COUNTS.get(element_type)
        if node_count is None:
            self._add_element(values, location)
            return
        self.pending_element_values.extend(values)
        self.pending_element_locations.extend(
            location for _ in values
        )
        record_size = node_count + 1
        while len(self.pending_element_values) >= record_size:
            record = self.pending_element_values[:record_size]
            self.pending_element_values = self.pending_element_values[record_size:]
            record_location = self.pending_element_locations[0]
            self.pending_element_locations = (
                self.pending_element_locations[record_size:]
            )
            self._add_element(record, record_location)

    def finish(self) -> None:
        """Finalize the active block and reject partial records."""

        self._finish_active_mode()

    def _finish_active_mode(self) -> None:
        if self.mode == "beam_section":
            self._finalize_beam_section()
        if self.pending_element_values:
            element_type = _required_param(self.keyword, "type")
            location = (
                self.pending_element_locations[0]
                if self.pending_element_locations
                else self.keyword.location
            )
            raise AbaqusParseError(
                (
                    f"Incomplete {element_type} connectivity record: "
                    f"{self.pending_element_values}"
                ),
                code="abaqus.element.connectivity_incomplete",
                location=location,
                record=tuple(self.pending_element_values),
            )
        self.pending_element_values = []
        self.pending_element_locations = []

    def _add_surface(self, values: list[str]) -> None:
        if len(values) < 2 or not self.current_surface_accepts_data:
            return
        name = _required_param(self.keyword, "name")
        self.deck.surfaces[name].append(
            AbaqusSurfaceFace(_parse_target(values[0]), values[1].upper())
        )

    def _keyword_scope(self, keyword: Keyword | None) -> str:
        """Return the scope for a keyword definition."""
        if keyword is not None and "instance" in keyword.params:
            return "assembly"
        return self.scope

    def _prepare_scoped_collection(
        self,
        target: dict[str, list],
        scopes: dict[str, str],
        name: str,
        scope: str,
    ) -> bool:
        """Prepare a named collection and handle cross-scope redefinitions."""
        existing_scope = scopes.get(name)
        if existing_scope is None:
            target[name] = []
            scopes[name] = scope
            return True
        if existing_scope == scope:
            return True
        if scope == "assembly":
            target[name] = []
            scopes[name] = scope
            return True
        if existing_scope == "assembly":
            return False
        target[name] = []
        scopes[name] = scope
        return True

    def _add_density(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
        raw: str,
    ) -> None:
        if self.current_material is None:
            raise AbaqusParseError(
                "*Density must follow *Material",
                location=location,
            )
        fields = tuple(values)
        parsed = tuple(
            parse_abaqus_real(
                value,
                location=location,
                field="density",
            )
            for value in fields
        )
        if not parsed:
            raise AbaqusParseError(
                "*Density requires at least one value",
                location=location,
                record=fields,
            )
        self.current_material.density_records.append(
            AbaqusDataRecordEvidence(
                present=True,
                blank=False,
                field_count=len(fields),
                location=location,
                fields=fields,
                raw=raw,
                values=parsed,
            )
        )
        self.current_material.properties["rho"] = parsed[0]

    def _add_elastic(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
        raw: str,
    ) -> None:
        if self.current_material is None:
            raise AbaqusParseError(
                "*Elastic must follow *Material",
                location=location,
            )
        if len(values) < 2:
            raise AbaqusParseError(
                "*Elastic requires E and nu",
                location=location,
                record=tuple(values),
            )
        fields = tuple(values)
        parsed = tuple(
            parse_abaqus_real(
                value,
                location=location,
                field=(
                    "Young's modulus"
                    if index == 0
                    else "Poisson ratio"
                    if index == 1
                    else "*ELASTIC value"
                ),
            )
            for index, value in enumerate(fields)
        )
        self.current_material.elastic_records.append(
            AbaqusDataRecordEvidence(
                present=True,
                blank=False,
                field_count=len(fields),
                location=location,
                fields=fields,
                raw=raw,
                values=parsed,
            )
        )
        self.current_material.properties["E"] = parsed[0]
        self.current_material.properties["nu"] = parsed[1]

    def _start_solid_section(self, keyword: Keyword) -> None:
        element_set = _required_param(keyword, "elset")
        material = _required_param(keyword, "material")
        target_was_defined = element_set in self.deck.element_sets
        element_ids = tuple(self.deck.element_sets.get(element_set, ()))
        self.deck.sections.append(
            AbaqusSection(
                element_set=element_set,
                material=material,
                section_type="solid",
                element_ids=element_ids,
                data=AbaqusSolidSectionData(
                    attribute=None,
                    record_present=False,
                    blank=False,
                    field_count=0,
                    location=None,
                    fields=(),
                ),
                keyword_location=keyword.location,
                target_was_defined=target_was_defined,
            )
        )
        self.mode = "solid_section"

    def _start_beam_section(self, keyword: Keyword) -> None:
        _validate_keyword_parameters(
            keyword,
            allowed=frozenset({"elset", "material", "section"}),
            required=frozenset({"elset", "material", "section"}),
        )
        element_set = _required_param(keyword, "elset")
        material = _required_param(keyword, "material")
        profile = " ".join(_required_param(keyword, "section").upper().split())
        target_was_defined = element_set in self.deck.element_sets
        element_ids = tuple(self.deck.element_sets.get(element_set, ()))
        missing = AbaqusDataRecordEvidence.missing()
        self.deck.sections.append(
            AbaqusSection(
                element_set=element_set,
                material=material,
                section_type="beam",
                element_ids=element_ids,
                data=AbaqusBeamSectionData(
                    profile=profile,
                    dimensions=(),
                    approximate_n1=None,
                    geometry=missing,
                    orientation=missing,
                ),
                keyword_location=keyword.location,
                target_was_defined=target_was_defined,
            )
        )
        self.mode = "beam_section"

    def _add_solid_section_data(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        section = self.deck.sections[-1]
        data = section.data
        if not isinstance(data, AbaqusSolidSectionData):
            raise RuntimeError("solid-section parser state lost its typed data")
        if data.record_present:
            raise AbaqusParseError(
                "*SOLID SECTION supports at most one data record",
                code="abaqus.solid_section.extra_record",
                location=location,
                record=tuple(values),
            )
        fields = _normalized_positional_fields(values)
        blank = len(fields) == 1 and fields[0] == ""
        parsed = tuple(
            parse_abaqus_real(
                value,
                location=location,
                field="*SOLID SECTION value",
            )
            for value in fields
            if value
        )
        attribute = None if not fields or not fields[0] else parsed[0]
        self.deck.sections[-1] = replace(
            section,
            data=AbaqusSolidSectionData(
                attribute=attribute,
                record_present=True,
                blank=blank,
                field_count=len(fields),
                location=location,
                fields=fields,
            ),
        )

    def _add_beam_section_data(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
        raw: str,
    ) -> None:
        section = self.deck.sections[-1]
        data = section.data
        if not isinstance(data, AbaqusBeamSectionData):
            raise RuntimeError("beam-section parser state lost its typed data")
        fields = _normalized_positional_fields(values)
        blank = len(fields) == 1 and fields[0] == ""
        evidence = AbaqusDataRecordEvidence(
            present=True,
            blank=blank,
            field_count=len(fields),
            location=location,
            fields=fields,
            raw=raw,
        )
        if not data.geometry.present:
            if blank:
                raise AbaqusParseError(
                    "*BEAM SECTION geometry record cannot be blank",
                    code="abaqus.b31.section.geometry_missing",
                    location=location,
                    record=fields,
                )
            dimensions = tuple(
                parse_abaqus_real(
                    value,
                    location=location,
                    field="*BEAM SECTION geometry value",
                )
                for value in fields
            )
            self.deck.sections[-1] = replace(
                section,
                data=replace(
                    data,
                    dimensions=dimensions,
                    geometry=evidence,
                ),
            )
            return

        if not data.orientation.present:
            if blank:
                orientation = None
            else:
                if len(fields) != 3:
                    raise AbaqusParseError(
                        (
                            "*BEAM SECTION approximate n1 record must contain "
                            "exactly three direction cosines"
                        ),
                        code="abaqus.b31.orientation_shape",
                        location=location,
                        record=fields,
                    )
                orientation = tuple(
                    parse_abaqus_real(
                        value,
                        location=location,
                        field="*BEAM SECTION n1 component",
                    )
                    for value in fields
                )
            self.deck.sections[-1] = replace(
                section,
                data=replace(
                    data,
                    approximate_n1=orientation,
                    orientation=evidence,
                ),
            )
            return

        raise UnsupportedAbaqusFeatureError(
            "a third *BEAM SECTION integration-point record is unsupported",
            code="abaqus.b31.section.integration_record_unsupported",
            location=location,
            record=fields,
            remediation=(
                "Remove the third data record and use default section "
                "integration."
            ),
        )

    def _finalize_beam_section(self) -> None:
        section = self.deck.sections[-1]
        data = section.data
        if (
            not isinstance(data, AbaqusBeamSectionData)
            or not data.geometry.present
        ):
            raise AbaqusParseError(
                "*BEAM SECTION requires one geometry data record",
                code="abaqus.b31.section.geometry_missing",
                location=section.keyword_location,
                remediation="Add the required profile dimensions data record.",
            )

    def _add_boundary(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        target = _parse_target(values[0])
        if len(values) >= 2 and not _is_int(values[1]):
            first_component: int | str = values[1].upper()
            last_component = None
            value = 0.0
        else:
            first_component = _parse_int(
                values[1],
                location,
                "boundary first component",
            )
            last_component = (
                _parse_int(
                    values[2],
                    location,
                    "boundary last component",
                )
                if len(values) > 2
                else first_component
            )
            value = (
                parse_abaqus_real(
                    values[3],
                    location=location,
                    field="boundary value",
                )
                if len(values) > 3
                else 0.0
            )
        self._ensure_step().boundaries.append(
            AbaqusBoundary(target, first_component, last_component, value)
        )

    def _add_cload(
        self,
        values: list[str],
        location: AbaqusSourceLocation,
    ) -> None:
        if len(values) < 3:
            raise AbaqusParseError(
                "*Cload requires target, component, and value",
                location=location,
                record=tuple(values),
            )
        self._ensure_step().cloads.append(
            AbaqusCload(
                _parse_target(values[0]),
                _parse_int(values[1], location, "CLOAD component"),
                parse_abaqus_real(
                    values[2],
                    location=location,
                    field="CLOAD value",
                ),
            )
        )

    def _add_distributed_load(
        self,
        values: list[str],
        source: str,
        location: AbaqusSourceLocation,
    ) -> None:
        label = values[1].upper() if len(values) > 1 else ""
        if label == "GRAV" and len(values) != 6:
            component_count = max(len(values) - 3, 0)
            raise AbaqusParseError(
                "GRAV requires a magnitude and 3 direction components, "
                f"got {component_count} direction components",
                location=location,
                record=tuple(values),
            )
        if len(values) < 3:
            raise AbaqusParseError(
                f"*{source} requires target, label, and magnitude",
                location=location,
                record=tuple(values),
            )
        try:
            magnitude = parse_abaqus_real(
                values[2],
                location=location,
                field=f"{label or source} magnitude",
                allow_nonfinite=(label == "GRAV"),
            )
            extra = tuple(
                parse_abaqus_real(
                    value,
                    location=location,
                    field=f"{label or source} component",
                    allow_nonfinite=(label == "GRAV"),
                )
                for value in values[3:]
            )
        except AbaqusParseError as exc:
            if label == "GRAV":
                raise AbaqusParseError(
                    (
                        "GRAV magnitude and direction components must be "
                        "numeric"
                    ),
                    code="abaqus.grav.numeric_invalid",
                    location=location,
                    record=tuple(values),
                ) from exc
            raise
        keyword = self.keyword
        self._ensure_step().distributed_loads.append(
            AbaqusDistributedLoad(
                _parse_optional_target(values[0]),
                label,
                magnitude,
                source,
                extra,
                keyword.location if keyword is not None else None,
                location,
                tuple(values),
                (
                    tuple(keyword.params.items())
                    if keyword is not None
                    else ()
                ),
                (
                    tuple(sorted(keyword.flags))
                    if keyword is not None
                    else ()
                ),
            )
        )

    def _start_output_block(self, keyword: Keyword) -> None:
        kind = "field" if "field" in keyword.flags else "history"
        if "history" in keyword.flags:
            kind = "history"
        self.current_output_kind = kind
        self.current_output_target = None
        self.current_output_parent = keyword
        variable = keyword.params.get("variable")
        if variable is not None:
            self._ensure_step().output_requests.append(
                AbaqusOutputRequest(
                    kind,
                    (
                        "preselect"
                        if variable.casefold() == "preselect"
                        else "output"
                    ),
                    (variable,),
                    _effective_output_metadata(keyword, None),
                    _keyword_parameters(keyword),
                    _keyword_flags(keyword),
                )
            )

    def _start_named_output(self, keyword: Keyword) -> None:
        kind = keyword.name.split(" ", 1)[0]
        self.current_output_kind = kind
        self.current_output_target = kind
        self.current_output_parent = None
        variable = keyword.params.get("variable")
        if variable is not None:
            self._ensure_step().output_requests.append(
                AbaqusOutputRequest(
                    kind,
                    kind,
                    (variable,),
                    _effective_output_metadata(None, keyword),
                    (),
                    (),
                    _keyword_parameters(keyword),
                    _keyword_flags(keyword),
                )
            )
            self.mode = None
        else:
            self.keyword = keyword
            self.mode = "output"

    def _start_output_data(self, keyword: Keyword, target: str) -> None:
        self.current_output_kind = self.current_output_kind or "field"
        self.current_output_target = target
        self.keyword = keyword
        self.mode = "output"

    def _add_output_request(self, values: list[str]) -> None:
        kind = self.current_output_kind or "field"
        target = self.current_output_target or kind
        child = self.keyword
        self._ensure_step().output_requests.append(
            AbaqusOutputRequest(
                kind,
                target,
                tuple(values),
                _effective_output_metadata(
                    self.current_output_parent,
                    child,
                ),
                _keyword_parameters(self.current_output_parent),
                _keyword_flags(self.current_output_parent),
                _keyword_parameters(child),
                _keyword_flags(child),
            )
        )

    def _clear_output_context(self) -> None:
        self.current_output_kind = None
        self.current_output_target = None
        self.current_output_parent = None

    def _ensure_step(self) -> AbaqusStep:
        if self.current_step is None:
            self.current_step = AbaqusStep("Initial")
            self.deck.steps.append(self.current_step)
        return self.current_step


def _keyword_parameters(
    keyword: Keyword | None,
) -> tuple[tuple[str, str], ...]:
    if keyword is None:
        return ()
    return keyword.occurrence.params


def _keyword_flags(keyword: Keyword | None) -> tuple[str, ...]:
    if keyword is None:
        return ()
    return keyword.occurrence.flags


def _effective_output_metadata(
    parent: Keyword | None,
    child: Keyword | None,
) -> dict[str, str]:
    """Merge output options while letting child keys override by casefold."""

    metadata: dict[str, str] = {}
    for keyword in (parent, child):
        if keyword is None:
            continue
        for key, value in keyword.occurrence.params:
            collision = next(
                (
                    existing
                    for existing in metadata
                    if existing.casefold() == key.casefold()
                ),
                None,
            )
            if collision is not None and collision != key:
                del metadata[collision]
            metadata[key] = value
    return metadata


def _assemble_keyword(
    physical_lines: list[str],
    start_index: int,
    path: Path,
) -> tuple[Keyword, int]:
    """Assemble one logical keyword and retain every physical source line."""

    raw_lines = [physical_lines[start_index]]
    index = start_index
    while raw_lines[-1].rstrip().endswith(","):
        next_index = index + 1
        first_location = AbaqusSourceLocation(path, start_index + 1)
        if next_index >= len(physical_lines):
            raise AbaqusParseError(
                "keyword continuation reaches end of file",
                code="abaqus.keyword.continuation_eof",
                location=first_location,
                record=tuple(raw_lines),
            )
        candidate = physical_lines[next_index]
        stripped = candidate.strip()
        if not stripped or stripped.startswith("**"):
            raise AbaqusParseError(
                "keyword continuation cannot be empty or a comment",
                code="abaqus.keyword.continuation_empty",
                location=AbaqusSourceLocation(path, next_index + 1),
                record=candidate,
            )
        if stripped.startswith("*"):
            raise AbaqusParseError(
                "keyword continuation was interrupted by another keyword",
                code="abaqus.keyword.continuation_interrupted",
                location=AbaqusSourceLocation(path, next_index + 1),
                record=candidate,
            )
        raw_lines.append(candidate)
        index = next_index

    logical = "".join(line.strip() for line in raw_lines)
    keyword = _parse_keyword(
        logical,
        path=path,
        line_number=start_index + 1,
        end_line=index + 1,
        raw_lines=tuple(raw_lines),
    )
    return keyword, index + 1


def _parse_keyword(
    line: str,
    *,
    path: Path | None = None,
    line_number: int = 1,
    end_line: int | None = None,
    raw_lines: tuple[str, ...] | None = None,
) -> Keyword:
    """Parse one logical Abaqus keyword line with duplicate detection."""

    parts = [part.strip() for part in line[1:].split(",")]
    if not parts or not parts[0]:
        location = AbaqusSourceLocation(path, line_number)
        raise AbaqusParseError(
            "Abaqus keyword name cannot be empty",
            code="abaqus.keyword.name_missing",
            location=location,
            record=line,
        )
    name = parts[0].lower()
    location = AbaqusSourceLocation(path, line_number, name)
    params: dict[str, str] = {}
    flags: set[str] = set()
    ordered_flags: list[str] = []
    for part in parts[1:]:
        if not part:
            raise AbaqusParseError(
                f"*{name.upper()} contains an empty keyword option",
                code="abaqus.keyword.option_empty",
                location=location,
                record=line,
            )
        if "=" in part:
            key, value = part.split("=", 1)
            normalized_key = key.strip().lower()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                raise AbaqusParseError(
                    f"*{name.upper()} has an empty parameter name or value",
                    code="abaqus.keyword.parameter_empty",
                    location=location,
                    record=part,
                )
            if normalized_key in params:
                duplicate_line = line_number
                if raw_lines:
                    matches = [
                        line_number + offset
                        for offset, raw in enumerate(raw_lines)
                        if re.search(
                            (
                                rf"(?:^|,)\s*{re.escape(normalized_key)}"
                                r"\s*="
                            ),
                            raw,
                            flags=re.IGNORECASE,
                        )
                    ]
                    if len(matches) > 1:
                        duplicate_line = matches[1]
                raise AbaqusParseError(
                    (
                        f"*{name.upper()} repeats parameter "
                        f"{normalized_key!r}"
                    ),
                    code="abaqus.keyword.parameter_duplicate",
                    location=AbaqusSourceLocation(
                        path,
                        duplicate_line,
                        name,
                    ),
                    record=normalized_key,
                    remediation=(
                        f"Keep exactly one {normalized_key.upper()} parameter "
                        f"on *{name.upper()}."
                    ),
                )
            params[normalized_key] = normalized_value
        else:
            flag = part.lower()
            flags.add(flag)
            if flag not in ordered_flags:
                ordered_flags.append(flag)

    physical_count = len(raw_lines or (line,))
    physical_locations = tuple(
        AbaqusSourceLocation(path, line_number + offset, name)
        for offset in range(physical_count)
    )
    span = AbaqusSourceSpan(
        physical_locations[0],
        AbaqusSourceLocation(
            path,
            line_number if end_line is None else end_line,
            name,
        ),
        physical_locations,
    )
    occurrence = AbaqusKeywordOccurrence(
        name=name,
        params=tuple(params.items()),
        flags=tuple(ordered_flags),
        span=span,
        raw_lines=raw_lines or (line,),
    )
    return Keyword(name, params, flags, occurrence)


def _split_values(
    line: str,
    *,
    preserve_leading_empty: bool = False,
    preserve_all_empty: bool = False,
) -> list[str]:
    """Split an Abaqus comma-separated data line."""
    parts = [part.strip() for part in line.split(",")]
    if preserve_all_empty:
        return parts
    while parts and not parts[-1]:
        parts.pop()
    if preserve_leading_empty:
        return parts
    return [part for part in parts if part]


def _required_param(keyword: Keyword | None, name: str) -> str:
    """Return a required keyword parameter."""
    if keyword is None or name not in keyword.params:
        raise AbaqusParseError(
            f"missing Abaqus keyword parameter: {name}",
            code="abaqus.keyword.parameter_missing",
            location=keyword.location if keyword is not None else None,
            record=name,
        )
    return keyword.params[name]


def _validate_keyword_parameters(
    keyword: Keyword,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    unknown = tuple(sorted(set(keyword.params) - set(allowed)))
    missing = tuple(sorted(set(required) - set(keyword.params)))
    if missing:
        raise AbaqusParseError(
            f"*{keyword.name.upper()} is missing parameters {missing}",
            code="abaqus.keyword.parameter_missing",
            location=keyword.location,
            record=missing,
        )
    if unknown or keyword.flags:
        options = (*unknown, *tuple(sorted(keyword.flags)))
        if keyword.name == "beam section" and unknown:
            remediation = (
                "Keep only ELSET, MATERIAL, and SECTION on *BEAM SECTION; "
                "move profile geometry to the following data record."
            )
        else:
            remediation = (
                "Remove unsupported options and use the exact supported "
                "keyword form."
            )
        raise AbaqusParseError(
            (
                f"*{keyword.name.upper()} uses unsupported parameters or "
                f"flags {options}"
            ),
            code="abaqus.keyword.option_unsupported",
            location=keyword.location,
            record=options,
            remediation=remediation,
        )


def _normalized_positional_fields(values: list[str]) -> tuple[str, ...]:
    """Represent an explicit blank record as one positional blank field."""

    fields = tuple(values)
    return fields if fields else ("",)


def _parse_int(
    value: str,
    location: AbaqusSourceLocation | None,
    field: str,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AbaqusParseError(
            f"{field} must be an integer, got {value!r}",
            code="abaqus.integer.invalid",
            location=location,
            record=value,
        ) from exc


def _generate_ids(values: Iterable[str]) -> list[int]:
    """Expand Abaqus generate triplets."""
    numbers = [int(value) for value in values]
    if len(numbers) % 3 != 0:
        raise ValueError("Abaqus generate set data must use start,end,step triplets")
    ids: list[int] = []
    for start, end, step in zip(numbers[0::3], numbers[1::3], numbers[2::3]):
        ids.extend(range(start, end + 1, step))
    return ids


def _parse_target(value: str) -> str | int:
    """Parse a node/element id target or keep a set name."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_optional_target(value: str) -> str | int | None:
    """Parse an optional distributed-load target."""
    return None if not value else _parse_target(value)


def _is_int(value: str) -> bool:
    """Return whether a string can be parsed as int."""
    try:
        int(value)
    except ValueError:
        return False
    return True
