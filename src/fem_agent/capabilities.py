"""Versioned V0 capabilities for local Abaqus import and result queries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .schemas import ResultQueryKind, SCHEMA_VERSION


class CapabilityDisposition(str, Enum):
    SUPPORTED = "supported"
    IGNORED = "inspected_but_ignored"
    WARNING = "unsupported_warning"
    BLOCKING = "unsupported_blocking"


@dataclass(frozen=True)
class KeywordCapability:
    disposition: CapabilityDisposition
    allowed_parameters: frozenset[str] = frozenset()
    required_parameters: frozenset[str] = frozenset()
    allowed_flags: frozenset[str] = frozenset()


SUPPORTED_ELEMENT_TYPES = frozenset(
    {
        "CPS3",
        "CPE3",
        "CPS6",
        "CPE6",
        "CPS4",
        "CPE4",
        "CPS8",
        "CPE8",
        "C3D4",
        "C3D10",
        "C3D8",
        "C3D20",
        "B31",
        "BEAM2",
        "T3D2",
        "TRUSS2",
    }
)
SUPPORTED_SECTION_TYPES = frozenset({"beam", "solid", "truss"})
SUPPORTED_PROCEDURES = frozenset({"static"})
SUPPORTED_BOUNDARY_LABELS = frozenset({"ENCASTRE", "XSYMM", "YSYMM", "ZSYMM"})
SUPPORTED_DLOAD_LABELS = frozenset(
    {"GRAV", "P1", "P2", "P3", "P4", "P5", "P6", "QGLOBAL", "QLOCAL"}
)
SUPPORTED_DSLOAD_LABELS = frozenset(
    {"P", "P1", "P2", "P3", "P4", "P5", "P6", "TRVEC", "TRSHR"}
)


KEYWORD_CAPABILITIES: Mapping[str, KeywordCapability] = {
    "heading": KeywordCapability(CapabilityDisposition.IGNORED),
    "preprint": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"echo", "model", "history", "contact"}),
    ),
    "part": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name"}),
        required_parameters=frozenset({"name"}),
    ),
    "end part": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "assembly": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name"}),
    ),
    "end assembly": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "instance": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name", "part"}),
        required_parameters=frozenset({"name", "part"}),
    ),
    "end instance": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "node": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "element": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"type", "elset"}),
        required_parameters=frozenset({"type"}),
    ),
    "nset": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"nset", "instance"}),
        required_parameters=frozenset({"nset"}),
        allowed_flags=frozenset({"generate", "unsorted", "internal"}),
    ),
    "elset": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"elset", "instance"}),
        required_parameters=frozenset({"elset"}),
        allowed_flags=frozenset({"generate", "unsorted", "internal"}),
    ),
    "surface": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name", "type", "instance"}),
        required_parameters=frozenset({"name", "type"}),
    ),
    "material": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name"}),
        required_parameters=frozenset({"name"}),
    ),
    "elastic": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "density": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "solid section": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"elset", "material"}),
        required_parameters=frozenset({"elset", "material"}),
    ),
    "beam section": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset(
            {
                "elset",
                "material",
                "section",
                "height",
                "width",
                "radius",
                "outer_radius",
                "inner_radius",
            }
        ),
        required_parameters=frozenset({"elset", "material", "section"}),
    ),
    "truss section": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"elset", "material", "area"}),
        required_parameters=frozenset({"elset", "material", "area"}),
    ),
    "step": KeywordCapability(
        CapabilityDisposition.SUPPORTED,
        allowed_parameters=frozenset({"name", "nlgeom"}),
        required_parameters=frozenset({"name"}),
    ),
    "static": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "boundary": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "cload": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "dload": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "dsload": KeywordCapability(CapabilityDisposition.SUPPORTED),
    "output": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"variable", "frequency", "number interval"}),
        allowed_flags=frozenset({"field", "history"}),
    ),
    "field output": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"name", "variable", "frequency"}),
    ),
    "history output": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"name", "variable", "frequency"}),
    ),
    "node output": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"nset", "frequency"}),
    ),
    "element output": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset(
            {"elset", "position", "directions", "frequency"}
        ),
    ),
    "restart": KeywordCapability(
        CapabilityDisposition.IGNORED,
        allowed_parameters=frozenset({"frequency"}),
        allowed_flags=frozenset({"write"}),
    ),
    "end step": KeywordCapability(CapabilityDisposition.SUPPORTED),
}


V0_CAPABILITIES = {
    "schema_version": SCHEMA_VERSION,
    "input_formats": ["abaqus_inp"],
    "procedures": sorted(SUPPORTED_PROCEDURES),
    "element_types": sorted(SUPPORTED_ELEMENT_TYPES),
    "section_types": sorted(SUPPORTED_SECTION_TYPES),
    "boundary_labels": sorted(SUPPORTED_BOUNDARY_LABELS),
    "dload_labels": sorted(SUPPORTED_DLOAD_LABELS),
    "dsload_labels": sorted(SUPPORTED_DSLOAD_LABELS),
    "result_queries": [item.value for item in ResultQueryKind],
    "post_solve_result_queries": True,
    "precomputed_result_queries_required": False,
    "reusable_solution_state": True,
    "export_formats": ["csv", "vtk"],
    "multiple_runnable_steps": False,
    "geometric_nonlinearity": False,
    "raw_input_sent_to_provider": False,
    "arbitrary_code_execution": False,
}


def keyword_capability(name: str) -> KeywordCapability:
    """Return a conservative capability classification for one keyword."""

    return KEYWORD_CAPABILITIES.get(
        str(name).strip().casefold(),
        KeywordCapability(CapabilityDisposition.BLOCKING),
    )


def show_capabilities() -> dict[str, object]:
    """Return a fresh JSON-compatible capability declaration."""

    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in V0_CAPABILITIES.items()
    }


__all__ = [
    "CapabilityDisposition",
    "KEYWORD_CAPABILITIES",
    "KeywordCapability",
    "SUPPORTED_BOUNDARY_LABELS",
    "SUPPORTED_DLOAD_LABELS",
    "SUPPORTED_DSLOAD_LABELS",
    "SUPPORTED_ELEMENT_TYPES",
    "SUPPORTED_PROCEDURES",
    "SUPPORTED_SECTION_TYPES",
    "V0_CAPABILITIES",
    "keyword_capability",
    "show_capabilities",
]
