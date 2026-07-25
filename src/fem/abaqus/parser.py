from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .deck import (
    AbaqusBoundary,
    AbaqusCload,
    AbaqusDeck,
    AbaqusDistributedLoad,
    AbaqusElement,
    AbaqusMaterial,
    AbaqusOutputRequest,
    AbaqusSection,
    AbaqusStep,
    AbaqusSurfaceFace,
)


_SUPPORTED_ELEMENT_NODE_COUNTS = {
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


@dataclass(frozen=True)
class Keyword:
    """Parsed Abaqus keyword line."""
    name: str
    params: dict[str, str]
    flags: set[str]


def parse_file(path: str | Path) -> AbaqusDeck:
    """Parse a supported subset of an Abaqus input file."""
    inp_path = Path(path)
    deck = AbaqusDeck(name=inp_path.stem)
    state = _ParserState(deck)

    with inp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                state.handle_keyword(_parse_keyword(line))
            else:
                state.handle_data(
                    _split_values(
                        line,
                        preserve_leading_empty=(
                            state.mode
                            in {"dload", "dsload", "solid_section"}
                        ),
                    )
                )

    state.finish()
    return deck


class _ParserState:
    """State machine for the supported Abaqus keyword subset."""

    def __init__(self, deck: AbaqusDeck):
        self.deck = deck
        self.mode: str | None = None
        self.keyword: Keyword | None = None
        self.current_material: AbaqusMaterial | None = None
        self.current_step: AbaqusStep | None = None
        self.current_output_kind: str | None = None
        self.current_output_target: str | None = None
        self.scope = "model"
        self.current_set_accepts_data = True
        self.current_surface_accepts_data = True
        self.pending_element_values: list[str] = []

    def handle_keyword(self, keyword: Keyword) -> None:
        """Dispatch a keyword line and update data mode."""
        if self.mode == "element":
            self.finish()
            self.pending_element_values = []
        self.keyword = keyword
        self.mode = None
        self.current_set_accepts_data = True
        self.current_surface_accepts_data = True

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
            self.mode = "density"
            return

        if keyword.name == "elastic":
            self.mode = "elastic"
            return

        if keyword.name.endswith("section"):
            self._add_section(keyword)
            return

        if keyword.name == "step":
            self.current_step = AbaqusStep(
                _required_param(keyword, "name"),
                metadata=dict(keyword.params),
            )
            self.deck.steps.append(self.current_step)
            return

        if keyword.name == "static":
            self._ensure_step().procedure = "static"
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

    def handle_data(self, values: list[str]) -> None:
        """Handle one data line in the current mode."""
        if not values:
            return

        if self.mode == "node":
            self._add_node(values)
        elif self.mode == "element":
            self._consume_element_values(values)
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
            self._add_density(values)
        elif self.mode == "elastic":
            self._add_elastic(values)
        elif self.mode == "solid_section":
            self._add_solid_section_data(values)
        elif self.mode == "static":
            self._ensure_step().metadata["time"] = tuple(float(value) for value in values)
        elif self.mode == "boundary":
            self._add_boundary(values)
        elif self.mode == "cload":
            self._add_cload(values)
        elif self.mode == "dload":
            self._add_distributed_load(values, "dload")
        elif self.mode == "dsload":
            self._add_distributed_load(values, "dsload")
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

    def _add_node(self, values: list[str]) -> None:
        node_id = int(values[0])
        x = float(values[1])
        y = float(values[2])
        z = float(values[3]) if len(values) > 3 else 0.0
        self.deck.nodes[node_id] = (x, y, z)

    def _add_element(self, values: list[str]) -> None:
        element_type = _required_param(self.keyword, "type")
        element_set = self.keyword.params.get("elset") if self.keyword else None
        element = AbaqusElement(
            int(values[0]),
            tuple(int(value) for value in values[1:]),
            element_type,
            element_set,
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

    def _consume_element_values(self, values: list[str]) -> None:
        element_type = _required_param(self.keyword, "type").upper()
        node_count = _SUPPORTED_ELEMENT_NODE_COUNTS.get(element_type)
        if node_count is None:
            self._add_element(values)
            return
        self.pending_element_values.extend(values)
        record_size = node_count + 1
        while len(self.pending_element_values) >= record_size:
            record = self.pending_element_values[:record_size]
            self.pending_element_values = self.pending_element_values[record_size:]
            self._add_element(record)

    def finish(self) -> None:
        """Reject an incomplete wrapped element record."""
        if self.pending_element_values:
            element_type = _required_param(self.keyword, "type")
            raise ValueError(
                f"Incomplete {element_type} connectivity record: "
                f"{self.pending_element_values}"
            )

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

    def _add_density(self, values: list[str]) -> None:
        if self.current_material is None:
            raise ValueError("*Density must follow *Material")
        self.current_material.properties["rho"] = float(values[0])

    def _add_elastic(self, values: list[str]) -> None:
        if self.current_material is None:
            raise ValueError("*Elastic must follow *Material")
        if len(values) < 2:
            raise ValueError("*Elastic requires E and nu")
        self.current_material.properties["E"] = float(values[0])
        self.current_material.properties["nu"] = float(values[1])

    def _add_section(self, keyword: Keyword) -> None:
        element_set = _required_param(keyword, "elset")
        material = _required_param(keyword, "material")
        section_type = keyword.name.replace(" section", "")
        element_ids = tuple(self.deck.element_sets.get(element_set, ()))
        self.deck.sections.append(
            AbaqusSection(element_set, material, section_type, element_ids)
        )
        if section_type == "solid":
            self.mode = "solid_section"

    def _add_solid_section_data(self, values: list[str]) -> None:
        if len(values) != 1:
            raise ValueError(
                "*Solid Section data must contain exactly one scalar "
                "thickness value"
            )
        thickness = float(values[0])
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "*Solid Section thickness must be finite and greater than zero"
            )
        section = self.deck.sections[-1]
        if section.thickness is not None:
            raise ValueError(
                "*Solid Section supports at most one thickness data record"
            )
        self.deck.sections[-1] = replace(section, thickness=thickness)

    def _add_boundary(self, values: list[str]) -> None:
        target = _parse_target(values[0])
        if len(values) >= 2 and not _is_int(values[1]):
            first_component: int | str = values[1].upper()
            last_component = None
            value = 0.0
        else:
            first_component = int(values[1])
            last_component = int(values[2]) if len(values) > 2 else first_component
            value = float(values[3]) if len(values) > 3 else 0.0
        self._ensure_step().boundaries.append(
            AbaqusBoundary(target, first_component, last_component, value)
        )

    def _add_cload(self, values: list[str]) -> None:
        if len(values) < 3:
            raise ValueError("*Cload requires target, component, and value")
        self._ensure_step().cloads.append(
            AbaqusCload(_parse_target(values[0]), int(values[1]), float(values[2]))
        )

    def _add_distributed_load(self, values: list[str], source: str) -> None:
        label = values[1].upper() if len(values) > 1 else ""
        if label == "GRAV" and len(values) != 6:
            component_count = max(len(values) - 3, 0)
            raise ValueError(
                "GRAV requires a magnitude and 3 direction components, "
                f"got {component_count} direction components"
            )
        if len(values) < 3:
            raise ValueError(f"*{source} requires target, label, and magnitude")
        try:
            magnitude = float(values[2])
            extra = tuple(float(value) for value in values[3:])
        except (TypeError, ValueError) as exc:
            if label == "GRAV":
                raise ValueError(
                    "GRAV magnitude and direction components must be numeric"
                ) from exc
            raise
        self._ensure_step().distributed_loads.append(
            AbaqusDistributedLoad(
                _parse_optional_target(values[0]),
                label,
                magnitude,
                source,
                extra,
            )
        )

    def _start_output_block(self, keyword: Keyword) -> None:
        kind = "field" if "field" in keyword.flags else "history"
        if "history" in keyword.flags:
            kind = "history"
        self.current_output_kind = kind
        self.current_output_target = None
        variable = keyword.params.get("variable")
        if variable is not None:
            self._ensure_step().output_requests.append(
                AbaqusOutputRequest(
                    kind,
                    "preselect" if variable.upper() == "PRESELECT" else "output",
                    (variable.upper(),),
                    dict(keyword.params),
                )
            )

    def _start_named_output(self, keyword: Keyword) -> None:
        kind = keyword.name.split(" ", 1)[0]
        self.current_output_kind = kind
        self.current_output_target = kind
        variable = keyword.params.get("variable")
        if variable is not None:
            self._ensure_step().output_requests.append(
                AbaqusOutputRequest(kind, kind, (variable.upper(),), dict(keyword.params))
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
        metadata = dict(self.keyword.params) if self.keyword is not None else {}
        self._ensure_step().output_requests.append(
            AbaqusOutputRequest(kind, target, tuple(value.upper() for value in values), metadata)
        )

    def _ensure_step(self) -> AbaqusStep:
        if self.current_step is None:
            self.current_step = AbaqusStep("Initial")
            self.deck.steps.append(self.current_step)
        return self.current_step


def _parse_keyword(line: str) -> Keyword:
    """Parse one Abaqus keyword line."""
    parts = [part.strip() for part in line[1:].split(",") if part.strip()]
    name = parts[0].lower()
    params: dict[str, str] = {}
    flags: set[str] = set()
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().lower()] = value.strip()
        else:
            flags.add(part.lower())
    return Keyword(name, params, flags)


def _split_values(
    line: str,
    *,
    preserve_leading_empty: bool = False,
) -> list[str]:
    """Split an Abaqus comma-separated data line."""
    parts = [part.strip() for part in line.split(",")]
    while parts and not parts[-1]:
        parts.pop()
    if preserve_leading_empty:
        return parts
    return [part for part in parts if part]


def _required_param(keyword: Keyword | None, name: str) -> str:
    """Return a required keyword parameter."""
    if keyword is None or name not in keyword.params:
        raise ValueError(f"missing Abaqus keyword parameter: {name}")
    return keyword.params[name]


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
