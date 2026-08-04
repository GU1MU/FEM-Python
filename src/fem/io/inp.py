from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from ..core.model import FEMModel


@dataclass(frozen=True, slots=True)
class InpSourceLocation:
    """One physical source location in an INP input artifact."""

    path: Path | None
    line: int
    keyword: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.line, bool) or int(self.line) < 1:
            raise ValueError("INP source line must be a positive integer")
        object.__setattr__(self, "line", int(self.line))
        if self.path is not None and not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    def format(self) -> str:
        prefix = str(self.path) if self.path is not None else "<input>"
        context = f"{prefix}:{self.line}"
        if self.keyword:
            context += f" [*{self.keyword.upper()}]"
        return context


@dataclass(frozen=True, slots=True)
class InpSourceSpan:
    """A logical keyword span and all of its physical source locations."""

    start: InpSourceLocation
    end: InpSourceLocation
    physical_locations: tuple[InpSourceLocation, ...]

    @property
    def location(self) -> InpSourceLocation:
        return self.start


class InpKeywordCategory(str, Enum):
    """The single import classification used for source keyword evidence."""

    EXECUTED = "executed"
    POSTPROCESS_CANDIDATE = "postprocess candidate"
    PRESERVED = "preserved"
    HARMLESS_IGNORED = "harmless ignored"
    UNSUPPORTED_ENGINEERING_SEMANTICS = "unsupported engineering semantics"


# A shorter discoverable name for callers that describe the field as a
# disposition rather than a category.
InpKeywordDisposition = InpKeywordCategory
KeywordCategory = InpKeywordCategory


def classify_keyword(name: str) -> InpKeywordCategory:
    """Classify one normalized or raw INP keyword using the adapter contract."""

    from ..abaqus.contracts import STANDARD_LINE_SUBSET

    normalized = str(name).strip().casefold().lstrip("*")
    if normalized in STANDARD_LINE_SUBSET.executed_keywords:
        return InpKeywordCategory.EXECUTED
    if normalized in STANDARD_LINE_SUBSET.postprocess_candidate_keywords:
        return InpKeywordCategory.POSTPROCESS_CANDIDATE
    if normalized in STANDARD_LINE_SUBSET.preserved_output_keywords:
        return InpKeywordCategory.PRESERVED
    if normalized in STANDARD_LINE_SUBSET.harmless_ignored_keywords:
        return InpKeywordCategory.HARMLESS_IGNORED
    return InpKeywordCategory.UNSUPPORTED_ENGINEERING_SEMANTICS


@dataclass(frozen=True, slots=True)
class InpSourceOccurrence:
    """Detached source evidence for one keyword occurrence."""

    name: str
    params: tuple[tuple[str, str], ...]
    flags: tuple[str, ...]
    span: InpSourceSpan
    raw_lines: tuple[str, ...] = ()
    category: InpKeywordCategory = (
        InpKeywordCategory.UNSUPPORTED_ENGINEERING_SEMANTICS
    )

    @property
    def location(self) -> InpSourceLocation:
        return self.span.start

    @property
    def keyword(self) -> str:
        """Return the normalized keyword name for inspection clients."""

        return self.name

    @property
    def classification(self) -> InpKeywordCategory:
        """Return the unified category under an inspection-friendly name."""

        return self.category

    @property
    def disposition(self) -> InpKeywordCategory:
        """Return the category under the report/disposition terminology."""

        return self.category


@dataclass(frozen=True, slots=True)
class InpSourceSummary:
    """Read-only source evidence retained by the complete-model facade."""

    occurrences: tuple[InpSourceOccurrence, ...] = ()

    @property
    def keyword_occurrences(self) -> tuple[InpSourceOccurrence, ...]:
        """Compatibility spelling for clients that use parser terminology."""

        return self.occurrences

    @property
    def source_occurrences(self) -> tuple[InpSourceOccurrence, ...]:
        """Return all preserved source keyword occurrences."""

        return self.occurrences


@dataclass(frozen=True, slots=True)
class InpImportNotice:
    """One non-authoritative limitation reported by an INP import."""

    code: str
    message: str
    locations: tuple[InpSourceLocation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "locations", tuple(self.locations))


@dataclass(frozen=True, slots=True)
class InpImportResult:
    """A detached model, notices, and optional read-only source evidence."""

    model: FEMModel
    notices: tuple[InpImportNotice, ...] = ()
    source_summary: InpSourceSummary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "notices", tuple(self.notices))


class InpInputError(ValueError):
    """Base input error retaining source and remediation evidence."""

    default_code = "abaqus.input.invalid"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        location: InpSourceLocation | None = None,
        record: Any = None,
        remediation: str | None = None,
        locations: Iterable[InpSourceLocation] = (),
        path: str | Path | None = None,
        line: int | None = None,
        keyword: str | None = None,
    ) -> None:
        if location is None and (path is not None or line is not None):
            location = InpSourceLocation(
                None if path is None else Path(path),
                1 if line is None else line,
                keyword,
            )

        ordered_locations: list[InpSourceLocation] = []
        if location is not None:
            ordered_locations.append(location)
        for item in locations:
            if item not in ordered_locations:
                ordered_locations.append(item)

        self.code = str(code or self.default_code)
        self.location = ordered_locations[0] if ordered_locations else None
        self.locations = tuple(ordered_locations)
        self.path = self.location.path if self.location is not None else None
        self.line = self.location.line if self.location is not None else None
        self.keyword = (
            self.location.keyword if self.location is not None else keyword
        )
        self.record = record
        self.remediation = remediation
        self.message = str(message)

        rendered = self.message
        if self.location is not None:
            rendered = f"{self.location.format()}: {rendered}"
        if remediation:
            rendered += f" Remediation: {remediation}"
        super().__init__(rendered)


class InpParseError(InpInputError):
    """A lexical or structural INP input error."""

    default_code = "abaqus.parse.invalid"


class InpBuildError(InpInputError):
    """A semantic construction error for a parsed INP source."""

    default_code = "abaqus.build.invalid"


class UnsupportedInpFeatureError(InpInputError):
    """A valid INP feature outside the currently implemented capability."""

    default_code = "abaqus.feature.unsupported"


def read(path: str | Path) -> FEMModel:
    """Read a complete INP model while discarding non-authoritative notices."""

    return read_with_report(path).model


def read_with_report(path: str | Path) -> InpImportResult:
    """Read a complete INP model through the public facade."""

    from ..abaqus.builder import build_model_with_report
    from ..abaqus.parser import parse_file

    deck = parse_file(path)
    built = build_model_with_report(deck)
    return InpImportResult(
        model=built.model,
        notices=built.notices,
        source_summary=_source_summary(deck),
    )


def _source_summary(deck: Any) -> InpSourceSummary:
    """Copy parser keyword evidence into an owned, immutable public value."""

    occurrences = tuple(
        _copy_source_occurrence(occurrence)
        for occurrence in tuple(getattr(deck, "keyword_occurrences", ()))
    )
    return InpSourceSummary(occurrences=occurrences)


def _copy_source_occurrence(occurrence: Any) -> InpSourceOccurrence:
    source_span = occurrence.span
    physical_locations = tuple(
        _copy_source_location(location)
        for location in source_span.physical_locations
    )
    span = InpSourceSpan(
        start=_copy_source_location(source_span.start),
        end=_copy_source_location(source_span.end),
        physical_locations=physical_locations,
    )
    return InpSourceOccurrence(
        name=str(occurrence.name),
        params=tuple(
            (str(key), str(value))
            for key, value in tuple(occurrence.params)
        ),
        flags=tuple(str(flag) for flag in tuple(occurrence.flags)),
        span=span,
        raw_lines=tuple(str(line) for line in tuple(occurrence.raw_lines)),
        category=classify_keyword(str(occurrence.name)),
    )


def _copy_source_location(location: Any) -> InpSourceLocation:
    return InpSourceLocation(
        getattr(location, "path", None),
        int(location.line),
        getattr(location, "keyword", None),
    )


_MIXED2D_TYPES = {
    "CPS3": ("Tri3", 3, "linear"),
    "CPE3": ("Tri3", 3, "linear"),
    "CPS4": ("Quad4", 4, "linear"),
    "CPE4": ("Quad4", 4, "linear"),
    "CPS6": ("Tri6", 6, "quadratic"),
    "CPE6": ("Tri6", 6, "quadratic"),
    "CPS8": ("Quad8", 8, "quadratic"),
    "CPE8": ("Quad8", 8, "quadratic"),
}

_UNSUPPORTED_REDUCED_INTEGRATION_TYPES = frozenset(
    {"C3D8R", "CPS4R", "CPE4R", "CPS8R", "CPE8R", "C3D20R"}
)
_UNSUPPORTED_COUPLED_ELEMENT_TYPES = frozenset({"C3D4T", "C3D10T"})


def _split_nums(line: str) -> List[str]:
    parts = [p.strip() for p in line.strip().split(",")]
    return [p for p in parts if p]


def _keyword_type(kw: str) -> str | None:
    for part in [p.strip() for p in kw.split(",")]:
        if part.startswith("TYPE="):
            return part.split("=", 1)[1].strip().upper()
    return None


def _reject_unsupported_reduced_integration(element_type: str | None) -> None:
    """Reject reduced-integration aliases that have no local formulation."""
    if element_type in _UNSUPPORTED_REDUCED_INTEGRATION_TYPES:
        raise NotImplementedError(
            f"Unsupported element type: {element_type}; "
            "reduced integration is not implemented"
        )


def _reject_unsupported_coupled_element(element_type: str | None) -> None:
    """Reject coupled formulations that require unsupported extra DOFs."""
    if element_type in _UNSUPPORTED_COUPLED_ELEMENT_TYPES:
        raise NotImplementedError(
            f"Unsupported element type: {element_type}; "
            "coupled temperature-displacement elements are not implemented"
        )


def _integer_token(
    raw_value: object,
    inp_path: str,
    line_no: int,
    field: str,
    record_position: int,
) -> int:
    """Parse one INP integer token with source context."""
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Abaqus INP {inp_path} line {line_no} field {field} "
            f"at record position {record_position} has raw value {raw_value!r}; "
            "expected an integer"
        ) from exc


def _numeric_token(
    raw_value: object,
    inp_path: str,
    line_no: int,
    field: str,
    record_position: int,
) -> float:
    """Parse one INP numeric token with source context."""
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Abaqus INP {inp_path} line {line_no} field {field} "
            f"at record position {record_position} has raw value {raw_value!r}; "
            "expected a numeric value"
        ) from exc


def _node_2d(parts: List[str], inp_path: str, line_no: int) -> Node2D:
    """Parse one already-split 2D node record."""
    return Node2D(
        id=_integer_token(parts[0], inp_path, line_no, "node_id", 1),
        x=_numeric_token(parts[1], inp_path, line_no, "x", 2),
        y=_numeric_token(parts[2], inp_path, line_no, "y", 3),
    )


def _node_3d(parts: List[str], inp_path: str, line_no: int) -> Node3D:
    """Parse one already-split 3D node record."""
    return Node3D(
        id=_integer_token(parts[0], inp_path, line_no, "node_id", 1),
        x=_numeric_token(parts[1], inp_path, line_no, "x", 2),
        y=_numeric_token(parts[2], inp_path, line_no, "y", 3),
        z=_numeric_token(parts[3], inp_path, line_no, "z", 4),
    )


def _fixed_connectivity(
    parts: List[str],
    node_count: int,
    inp_path: str,
    line_no: int,
    token_line_numbers: Optional[List[int]] = None,
) -> tuple[int, List[int]] | None:
    """Parse a fixed-width element record, or return ``None`` when incomplete."""
    if len(parts) < node_count + 1:
        return None
    source_lines = token_line_numbers or [line_no] * (node_count + 1)
    elem_id = _integer_token(parts[0], inp_path, source_lines[0], "elem_id", 1)
    node_ids = [
        _integer_token(
            value,
            inp_path,
            source_lines[local_node],
            f"node{local_node}",
            local_node + 1,
        )
        for local_node, value in enumerate(parts[1:node_count + 1], start=1)
    ]
    return elem_id, node_ids


def _normalize_plane_type(plane_type: Optional[str], elem_type: str) -> str:
    if plane_type is None:
        return "strain" if elem_type.upper().startswith("CPE") else "stress"
    pt = str(plane_type).lower()
    if pt.startswith("stress"):
        return "stress"
    if pt.startswith("strain"):
        return "strain"
    raise ValueError(
        f"plane_type {plane_type!r} is invalid; expected 'stress' or 'strain'"
    )


def _signed_area_2d(node_lookup: Dict[int, Node2D], node_ids: List[int]) -> float:
    corners = node_ids[:3] if len(node_ids) in (3, 6) else node_ids[:4]
    area = 0.0
    for i, node_id in enumerate(corners):
        current = node_lookup[node_id]
        nxt = node_lookup[corners[(i + 1) % len(corners)]]
        area += current.x * nxt.y - nxt.x * current.y
    return 0.5 * area


def _fix_plane_orientation(elem: Element2D, node_lookup: Dict[int, Node2D]) -> None:
    area = _signed_area_2d(node_lookup, elem.node_ids)
    if area >= 0.0:
        return
    if elem.type == "Tri3":
        n1, n2, n3 = elem.node_ids
        elem.node_ids = [n1, n3, n2]
    elif elem.type == "Tri6":
        n1, n2, n3, n4, n5, n6 = elem.node_ids
        elem.node_ids = [n1, n3, n2, n6, n5, n4]
    elif elem.type == "Quad4":
        n1, n2, n3, n4 = elem.node_ids
        elem.node_ids = [n1, n4, n3, n2]
    elif elem.type == "Quad8":
        n1, n2, n3, n4, n5, n6, n7, n8 = elem.node_ids
        elem.node_ids = [n1, n4, n3, n2, n8, n7, n6, n5]


def read_tri3(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
) -> Mesh2D:
    """Read a Tri3 plane mesh from Abaqus .inp files."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    node_lookup: Dict[int, Node2D] = {}

    in_node_block = False
    in_elem_block = False
    elem_abaqus_type: Optional[str] = None  # "CPS3" or "CPE3"

    def _infer_plane_type_from_elem_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS3"):
            return "stress"
        if etu.startswith("CPE3"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.upper()

                in_node_block = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                    etype = _keyword_type(kw)
                    if etype in ("CPS3", "CPE3"):
                        in_elem_block = True
                        elem_abaqus_type = etype
                    continue

                if not in_node_block:
                    pass
                if not kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                continue

            # 数据行
            if in_node_block:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = _node_2d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node

            elif in_elem_block and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 3, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record

                if plane_type is None:
                    pt = _infer_plane_type_from_elem_type(elem_abaqus_type)
                else:
                    pt = str(plane_type).lower()
                    if pt.startswith("stress"):
                        pt = "stress"
                    elif pt.startswith("strain"):
                        pt = "strain"
                    else:
                        raise ValueError(
                            f"plane_type {plane_type!r} is invalid for {elem_abaqus_type}; "
                            "expected 'stress' or 'strain'"
                        )

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elem = Element2D(
                    id=elem_id,
                    node_ids=node_ids,
                    type="Tri3",
                    props=props,
                )
                elements.append(elem)

    if not nodes:
        raise ValueError(f"No *Node data found in Abaqus input file {inp_path!r}")
    if not elements:
        raise ValueError(
            f"No CPS3/CPE3 *Element data found in Abaqus input file {inp_path!r}"
        )

    return Mesh2D(nodes=nodes, elements=elements)


def read_tri6(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> Mesh2D:
    """Read Tri6 plane mesh (CPS6/CPE6) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}
    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    elem_abaqus_type = _keyword_type(kw)
                    in_elem = elem_abaqus_type in ("CPS6", "CPE6")
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = _node_2d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 6, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record
                props: Dict[str, Any] = {
                    "thickness": float(default_thickness),
                    "plane_type": _normalize_plane_type(plane_type, elem_abaqus_type),
                }
                elements.append(
                    Element2D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Tri6",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS6/CPE6 *Element data found in {inp_path}")
    if fix_orientation:
        for elem in elements:
            _fix_plane_orientation(elem, node_lookup)
    return Mesh2D(nodes=nodes, elements=elements)


def read_quad4(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
    enforce_parallelogram: bool = False,
    tol: float = 1e-10,
) -> Mesh2D:
    """Read Quad4 plane mesh (CPS4/CPE4) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS4"):
            return "stress"
        if etu.startswith("CPE4"):
            return "strain"
        return "stress"

    def signed_area_quad(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        return 0.5 * (
            x1 * y2 - x2 * y1 +
            x2 * y3 - x3 * y2 +
            x3 * y4 - x4 * y3 +
            x4 * y1 - x1 * y4
        )

    def is_parallelogram(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        d1 = (x1 + x3 - x2 - x4, y1 + y3 - y2 - y4)  # diag midpoints: p1+p3 == p2+p4
        return (d1[0] * d1[0] + d1[1] * d1[1]) <= tol

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    et = _keyword_type(kw)
                    _reject_unsupported_reduced_integration(et)
                    if et in ("CPS4", "CPE4"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = _node_2d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 4, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError(
                        f"plane_type {plane_type!r} is invalid for {elem_abaqus_type}; "
                        "expected 'stress' or 'strain'"
                    )

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elements.append(
                    Element2D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Quad4",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS4/CPE4 *Element data found in {inp_path}")

    if enforce_parallelogram or fix_orientation:
        for e in elements:
            n1, n2, n3, n4 = e.node_ids
            try:
                p1 = (node_lookup[n1].x, node_lookup[n1].y)
                p2 = (node_lookup[n2].x, node_lookup[n2].y)
                p3 = (node_lookup[n3].x, node_lookup[n3].y)
                p4 = (node_lookup[n4].x, node_lookup[n4].y)
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")

            if enforce_parallelogram and not is_parallelogram(p1, p2, p3, p4):
                raise ValueError(f"Element {e.id} is not a parallelogram by tolerance {tol}")

            if fix_orientation:
                A = signed_area_quad(p1, p2, p3, p4)
                if A < 0.0:
                    e.node_ids = [n1, n4, n3, n2]

    return Mesh2D(nodes=nodes, elements=elements)


def read_quad8(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> Mesh2D:
    """Read Quad8 plane mesh (CPS8/CPE8) from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS8"):
            return "stress"
        if etu.startswith("CPE8"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    et = _keyword_type(kw)
                    _reject_unsupported_reduced_integration(et)
                    if et in ("CPS8", "CPE8"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = _node_2d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 8, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError(
                        f"plane_type {plane_type!r} is invalid for {elem_abaqus_type}; "
                        "expected 'stress' or 'strain'"
                    )

                props: Dict[str, any] = {
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                elements.append(
                    Element2D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Quad8",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS8/CPE8 *Element data found in {inp_path}")

    if fix_orientation:
        for e in elements:
            try:
                n1, n2, n3, n4 = (node_lookup[e.node_ids[i]] for i in range(4))
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")
            area = 0.5 * (
                n1.x * n2.y - n2.x * n1.y
                + n2.x * n3.y - n3.x * n2.y
                + n3.x * n4.y - n4.x * n3.y
                + n4.x * n1.y - n1.x * n4.y
            )
            if area < 0.0:
                if len(e.node_ids) != 8:
                    raise ValueError(f"Element {e.id} expected 8 nodes for orientation fix, got {len(e.node_ids)}")
                n1_id, n2_id, n3_id, n4_id, n5_id, n6_id, n7_id, n8_id = e.node_ids
                e.node_ids = [n1_id, n4_id, n3_id, n2_id, n8_id, n7_id, n6_id, n5_id]

    return Mesh2D(nodes=nodes, elements=elements)


def read_mixed2d(
    inp_path: str,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> Mesh2D:
    """Read a same-order mixed 2D plane mesh from Abaqus INP file."""
    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}
    orders: set[str] = set()
    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    elem_abaqus_type = _keyword_type(kw)
                    _reject_unsupported_reduced_integration(elem_abaqus_type)
                    in_elem = elem_abaqus_type in _MIXED2D_TYPES
                    if (
                        elem_abaqus_type is not None
                        and not in_elem
                        and elem_abaqus_type.startswith(("CPS", "CPE"))
                    ):
                        raise ValueError(
                            f"unsupported mixed 2D element type: {elem_abaqus_type}"
                        )
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 3:
                    continue
                node = _node_2d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                local_type, node_count, order = _MIXED2D_TYPES[elem_abaqus_type]
                record = _fixed_connectivity(
                    _split_nums(line), node_count, inp_path, line_no
                )
                if record is None:
                    continue
                elem_id, node_ids = record
                orders.add(order)
                props: Dict[str, Any] = {
                    "thickness": float(default_thickness),
                    "plane_type": _normalize_plane_type(plane_type, elem_abaqus_type),
                }
                elements.append(
                    Element2D(
                        id=elem_id,
                        node_ids=node_ids,
                        type=local_type,
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No supported 2D *Element data found in {inp_path}")
    if len(orders) > 1:
        raise ValueError("read_mixed2d requires elements with the same polynomial order")
    if fix_orientation:
        for elem in elements:
            _fix_plane_orientation(elem, node_lookup)
    return Mesh2D(nodes=nodes, elements=elements)


def read_tet10(inp_path: str) -> Mesh3D:
    """Read a Tet10 3D mesh from Abaqus .inp file (C3D10 elements).

    Node ordering (Abaqus convention):
        Corner nodes:  1-4
        Edge midnodes: 5=edge(1,2), 6=edge(2,3), 7=edge(3,1),
                       8=edge(1,4), 9=edge(2,4), 10=edge(3,4)
    """
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None
    pending_elem_parts: List[str] = []
    pending_elem_line_numbers: List[int] = []

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                if in_elem and pending_elem_parts:
                    raise ValueError(
                        f"Incomplete C3D10 connectivity record in {inp_path}: {pending_elem_parts}"
                    )
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet10 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    pending_elem_parts = []
                    pending_elem_line_numbers = []
                    et = _keyword_type(kw)
                    _reject_unsupported_coupled_element(et)
                    if et == "C3D10":
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 4:
                    continue
                node = _node_3d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                values = _split_nums(line)
                pending_elem_parts.extend(values)
                pending_elem_line_numbers.extend([line_no] * len(values))
                while len(pending_elem_parts) >= 11:
                    record = _fixed_connectivity(
                        pending_elem_parts[:11],
                        10,
                        inp_path,
                        line_no,
                        pending_elem_line_numbers[:11],
                    )
                    assert record is not None
                    elem_id, node_ids = record
                    pending_elem_parts = pending_elem_parts[11:]
                    pending_elem_line_numbers = pending_elem_line_numbers[11:]

                    elements.append(
                        Element3D(
                            id=elem_id,
                            node_ids=node_ids,
                            type="Tet10",
                        )
                    )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if pending_elem_parts:
        raise ValueError(
            f"Incomplete C3D10 connectivity record at end of file {inp_path}: {pending_elem_parts}"
        )
    if not elements:
        raise ValueError(f"No C3D10 *Element data found in {inp_path}")

    from ..elements.tetrahedron import tet10_gauss_points, tet10_shape_funcs_grads

    for e in elements:
        coords = [node_lookup[node_id] for node_id in e.node_ids]
        x = np.array([n.x for n in coords], dtype=float)
        y = np.array([n.y for n in coords], dtype=float)
        z = np.array([n.z for n in coords], dtype=float)

        for xi, eta, zeta, _ in tet10_gauss_points():
            _, dN_dxi, dN_deta, dN_dzeta = tet10_shape_funcs_grads(xi, eta, zeta)
            J = np.array([
                [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
                [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
                [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
            ], dtype=float)
            if np.linalg.det(J) <= 0.0:
                raise ValueError(
                    f"Element {e.id} has zero or negative Jacobian determinant. Check node ordering."
                )

    return Mesh3D(nodes=nodes, elements=elements)


def read_tet4(inp_path: str) -> Mesh3D:
    """Read a Tet4 3D mesh from Abaqus C3D4 input data."""
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet4 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    et = _keyword_type(kw)
                    _reject_unsupported_coupled_element(et)
                    if et == "C3D4":
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 4:
                    continue
                node = _node_3d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 4, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Tet4",
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D4 *Element data found in {inp_path}")

    # Check volume (Jacobian determinant) for each element
    for e in elements:
        n1, n2, n3, n4 = (node_lookup[node_id] for node_id in e.node_ids)
        # Volume = det(J)/6 where J columns are (x2-x1, x3-x1, x4-x1)
        v1 = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z])
        v2 = np.array([n3.x - n1.x, n3.y - n1.y, n3.z - n1.z])
        v3 = np.array([n4.x - n1.x, n4.y - n1.y, n4.z - n1.z])
        vol = np.dot(v1, np.cross(v2, v3)) / 6.0
        if vol <= 0.0:
            raise ValueError(
                f"Element {e.id} has zero or negative volume "
                f"(nodes: {e.node_ids}). Check node ordering."
            )

    return Mesh3D(nodes=nodes, elements=elements)


def read_hex8(inp_path: str) -> Mesh3D:
    """Read a Hex8 3D mesh from Abaqus .inp file (C3D8 elements)."""
    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    et = _keyword_type(kw)
                    _reject_unsupported_reduced_integration(et)
                    if et == "C3D8":
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = _split_nums(line)
                if len(parts) < 4:
                    continue
                node = _node_3d(parts, inp_path, line_no)
                nodes.append(node)
                node_lookup[node.id] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                record = _fixed_connectivity(_split_nums(line), 8, inp_path, line_no)
                if record is None:
                    continue
                elem_id, node_ids = record

                elements.append(
                    Element3D(
                        id=elem_id,
                        node_ids=node_ids,
                        type="Hex8",
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D8 *Element data found in {inp_path}")

    return Mesh3D(nodes=nodes, elements=elements)


def read_hex20(inp_path: str) -> Mesh3D:
    """Read a Hex20 mesh from Abaqus C3D20 input data."""
    from ..elements.hexahedron import hex20_gauss_points, hex20_shape_funcs_grads

    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}
    in_node = False
    in_elem = False
    pending_elem_parts: List[str] = []
    pending_elem_line_numbers: List[int] = []

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                if in_elem and pending_elem_parts:
                    raise ValueError(
                        f"Incomplete C3D20 connectivity record in {inp_path}: "
                        f"{pending_elem_parts}"
                    )
                keyword = line.upper()
                if keyword.startswith("*ASSEMBLY"):
                    break
                in_node = keyword.startswith("*NODE")
                elem_abaqus_type = (
                    _keyword_type(keyword)
                    if keyword.startswith("*ELEMENT")
                    else None
                )
                _reject_unsupported_reduced_integration(elem_abaqus_type)
                in_elem = (
                    keyword.startswith("*ELEMENT")
                    and elem_abaqus_type == "C3D20"
                )
                if keyword.startswith("*ELEMENT"):
                    pending_elem_parts = []
                    pending_elem_line_numbers = []
                continue

            values = _split_nums(line)
            if in_node:
                if len(values) >= 4:
                    node = _node_3d(values, inp_path, line_no)
                    nodes.append(node)
                    node_lookup[node.id] = node
                continue

            if in_elem:
                pending_elem_parts.extend(values)
                pending_elem_line_numbers.extend([line_no] * len(values))
                while len(pending_elem_parts) >= 21:
                    record = pending_elem_parts[:21]
                    record_line_numbers = pending_elem_line_numbers[:21]
                    pending_elem_parts = pending_elem_parts[21:]
                    pending_elem_line_numbers = pending_elem_line_numbers[21:]
                    connectivity = _fixed_connectivity(
                        record, 20, inp_path, line_no, record_line_numbers
                    )
                    assert connectivity is not None
                    elem_id, node_ids = connectivity
                    elements.append(
                        Element3D(
                            id=elem_id,
                            node_ids=node_ids,
                            type="Hex20",
                        )
                    )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if pending_elem_parts:
        raise ValueError(
            f"Incomplete C3D20 connectivity record at end of file {inp_path}: "
            f"{pending_elem_parts}"
        )
    if not elements:
        raise ValueError(f"No C3D20 *Element data found in {inp_path}")

    for elem in elements:
        coords = [node_lookup[node_id] for node_id in elem.node_ids]
        xyz = np.array([[node.x, node.y, node.z] for node in coords])
        for xi, eta, zeta, _ in hex20_gauss_points():
            _, dN_dxi, dN_deta, dN_dzeta = hex20_shape_funcs_grads(
                xi, eta, zeta
            )
            J = np.vstack([dN_dxi, dN_deta, dN_dzeta]) @ xyz
            if np.linalg.det(J) <= 0.0:
                raise ValueError(
                    f"Element {elem.id} has zero or negative Jacobian determinant. "
                    "Check node ordering."
                )

    return Mesh3D(nodes=nodes, elements=elements)
