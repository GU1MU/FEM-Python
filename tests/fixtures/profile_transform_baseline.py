"""Small, deterministic fixtures used by the Profile transform baseline.

The fixture deliberately stays at the native recipe/topology boundary.  It
does not construct a GUI object, open a project file, or retain any local
session/path information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fem.geometry import (
    ExtrudedGeometry,
    SketchCircle,
    SketchGeometry,
    analyze_sketch_profiles,
    describe_recipe_topology,
)
from fem_agent.geometry_authoring import (
    feature_topology_catalog,
    planar_sketch_geometry,
)


RING_OUTER_RADIUS = 50.0
RING_INNER_RADIUS = 25.0
RING_EXTRUSION_HEIGHT = 10.0


@dataclass(frozen=True, slots=True)
class ConcentricRingFixture:
    """Canonical two-circle material Profile with one nested hole."""

    sketch: SketchGeometry
    source_face_id: str
    extrusion: ExtrudedGeometry
    feature_catalog: Mapping[str, object]


def concentric_ring_sketch(
    *,
    outer_radius: float = RING_OUTER_RADIUS,
    inner_radius: float = RING_INNER_RADIUS,
) -> SketchGeometry:
    """Return a strict XY sketch containing one material ring Profile."""

    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("ring radii must satisfy 0 < inner < outer")
    return planar_sketch_geometry(
        "ring-profile",
        contours=(
            SketchCircle("material", 0.0, 0.0, outer_radius),
            SketchCircle("cut", 0.0, 0.0, inner_radius),
        ),
    ).recipe


def concentric_ring_source_face_id(sketch: SketchGeometry) -> str:
    """Return the canonical face ID for the unique material Profile."""

    profiles = analyze_sketch_profiles(sketch).profiles
    material = tuple(profile for profile in profiles if profile.role == "outer")
    if len(material) != 1:
        raise AssertionError("the ring fixture must have one material Profile")
    return f"face:{material[0].id}"


def concentric_ring_fixture() -> ConcentricRingFixture:
    """Build the reusable Phase 0 sketch, catalog, and extrusion recipe."""

    sketch = concentric_ring_sketch()
    source_face_id = concentric_ring_source_face_id(sketch)
    extrusion = ExtrudedGeometry(
        sketch,
        RING_EXTRUSION_HEIGHT,
        (source_face_id,),
    )
    return ConcentricRingFixture(
        sketch=sketch,
        source_face_id=source_face_id,
        extrusion=extrusion,
        feature_catalog=feature_topology_catalog(sketch, part_id="P1"),
    )


def ring_extrusion_logical_ids(fixture: ConcentricRingFixture) -> tuple[str, ...]:
    """Return the detached extrusion topology IDs used by baseline checks."""

    return describe_recipe_topology(fixture.extrusion).signature.logical_ids


__all__ = [
    "ConcentricRingFixture",
    "RING_EXTRUSION_HEIGHT",
    "RING_INNER_RADIUS",
    "RING_OUTER_RADIUS",
    "concentric_ring_fixture",
    "concentric_ring_sketch",
    "concentric_ring_source_face_id",
    "ring_extrusion_logical_ids",
]
