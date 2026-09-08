from __future__ import annotations

from fem.geometry import (
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    analyze_sketch_profiles,
)


def two_profile_sketch() -> SketchGeometry:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 2.0, 1.0),
        SketchPoint("P4", 0.0, 1.0),
        SketchPoint("P5", 5.0, 0.0),
        SketchPoint("P6", 7.0, 0.0),
        SketchPoint("P7", 7.0, 1.0),
        SketchPoint("P8", 5.0, 1.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
        SketchLine("L5", "P5", "P6"),
        SketchLine("L6", "P6", "P7"),
        SketchLine("L7", "P7", "P8"),
        SketchLine("L8", "P8", "P5"),
    )
    return SketchGeometry("Profiles", SketchPlane.xy(), points, curves)


def profile_face_id(sketch: SketchGeometry, edge_id: str) -> str:
    profile = next(
        profile
        for profile in analyze_sketch_profiles(sketch).profiles
        if edge_id in {item.lstrip("-") for item in profile.curve_ids}
    )
    return f"face:{profile.id}"


def hole_profile_sketch() -> SketchGeometry:
    points = (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 4.0, 0.0),
        SketchPoint("P3", 4.0, 3.0),
        SketchPoint("P4", 0.0, 3.0),
        SketchPoint("P5", 1.0, 1.0),
        SketchPoint("P6", 2.0, 1.0),
        SketchPoint("P7", 2.0, 2.0),
        SketchPoint("P8", 1.0, 2.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
        SketchLine("L5", "P5", "P6"),
        SketchLine("L6", "P6", "P7"),
        SketchLine("L7", "P7", "P8"),
        SketchLine("L8", "P8", "P5"),
    )
    return SketchGeometry("Perforated", SketchPlane.xy(), points, curves)
