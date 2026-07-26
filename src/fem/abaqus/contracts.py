from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AbaqusLineSubset:
    """Exact Abaqus vocabulary executed for the Phase 6 line subset."""

    element_types: frozenset[str]
    section_profiles: frozenset[str]
    distributed_load_labels: frozenset[str]
    executed_keywords: frozenset[str]
    harmless_ignored_keywords: frozenset[str]


STANDARD_LINE_SUBSET = AbaqusLineSubset(
    element_types=frozenset({"B31", "T3D2"}),
    section_profiles=frozenset({"RECT", "CIRC", "THICK PIPE"}),
    distributed_load_labels=frozenset(
        {"PX", "PY", "PZ", "P1", "P2", "GRAV"}
    ),
    executed_keywords=frozenset(
        {
            "node",
            "element",
            "nset",
            "elset",
            "part",
            "end part",
            "assembly",
            "end assembly",
            "instance",
            "end instance",
            "material",
            "elastic",
            "density",
            "solid section",
            "beam section",
            "step",
            "static",
            "boundary",
            "cload",
            "dload",
            "output",
            "field output",
            "history output",
            "node output",
            "element output",
            "end step",
        }
    ),
    harmless_ignored_keywords=frozenset({"heading"}),
)

# Keep one descriptor object while offering a package-specific discoverable name.
ABAQUS_LINE_SUBSET = STANDARD_LINE_SUBSET

RETIRED_ELEMENT_TYPES = frozenset({"BEAM2", "TRUSS2"})
RETIRED_DLOAD_LABELS = frozenset({"QGLOBAL", "QLOCAL"})
UNSUPPORTED_BEAM_SECTION_PROFILES = frozenset(
    {
        "PIPE",
        "BOX",
        "I",
        "L",
        "HEX",
        "ARBITRARY",
        "TRAPEZOID",
        "GENERAL",
        "NONLINEAR GENERAL",
        "MESHED",
    }
)
UNSUPPORTED_LINE_DLOAD_LABELS = frozenset(
    {
        "QGLOBAL",
        "QLOCAL",
        "PXNU",
        "PYNU",
        "PZNU",
        "P1NU",
        "P2NU",
    }
)


__all__ = [
    "ABAQUS_LINE_SUBSET",
    "AbaqusLineSubset",
    "RETIRED_DLOAD_LABELS",
    "RETIRED_ELEMENT_TYPES",
    "STANDARD_LINE_SUBSET",
    "UNSUPPORTED_BEAM_SECTION_PROFILES",
    "UNSUPPORTED_LINE_DLOAD_LABELS",
]
