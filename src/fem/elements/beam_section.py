from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, pi, tanh
from typing import Any, Mapping


_DIMENSION_FIELDS = (
    "radius",
    "outer_radius",
    "inner_radius",
    "height",
    "width",
)
_SECTION_FIELDS = {
    "solid_circle": ("radius",),
    "hollow_circle": ("outer_radius", "inner_radius"),
    "rectangle": ("height", "width"),
}


@dataclass(frozen=True)
class Beam2Section:
    """Validated Beam2 section properties and source dimensions."""

    section_type: str
    area: float
    Iyy: float
    Izz: float
    J: float
    radius: float | None = None
    outer_radius: float | None = None
    inner_radius: float | None = None
    height: float | None = None
    width: float | None = None


def parse_beam2_section(props: Mapping[str, Any]) -> Beam2Section:
    """Validate a standard Beam2 section and derive its stiffness properties."""
    if "section_type" not in props:
        raise KeyError("Beam2 section missing property section_type")
    section_type = props["section_type"]
    if not isinstance(section_type, str) or section_type not in _SECTION_FIELDS:
        raise ValueError(
            "Beam2 section_type must be one of solid_circle, hollow_circle, "
            f"rectangle; got {section_type!r}"
        )

    required = _SECTION_FIELDS[section_type]
    irrelevant = set(_DIMENSION_FIELDS).difference(required)
    for name in irrelevant:
        if name in props and props[name] not in (None, ""):
            raise ValueError(
                f"Beam2 {section_type} section does not use property {name}"
            )

    dimensions = {name: _positive_dimension(props, name) for name in required}
    if section_type == "solid_circle":
        return _circle_section(section_type, dimensions["radius"], 0.0)

    if section_type == "hollow_circle":
        outer_radius = dimensions["outer_radius"]
        inner_radius = dimensions["inner_radius"]
        if outer_radius <= inner_radius:
            raise ValueError(
                "Beam2 hollow_circle outer_radius must be greater than inner_radius"
            )
        return _circle_section(section_type, outer_radius, inner_radius)

    height = dimensions["height"]
    width = dimensions["width"]
    return Beam2Section(
        section_type=section_type,
        area=height * width,
        Iyy=height * width**3 / 12.0,
        Izz=width * height**3 / 12.0,
        J=_rectangle_torsion_constant(height, width),
        height=height,
        width=width,
    )


def axial_stress_extrema(
    section: Beam2Section,
    axial_force: float,
    moment_y: float,
    moment_z: float,
) -> tuple[float, float, float]:
    """Return maximum, minimum, and maximum-absolute longitudinal stress."""
    forces = (float(axial_force), float(moment_y), float(moment_z))
    if not all(isfinite(value) for value in forces):
        raise ValueError("Beam2 axial force and bending moments must be finite")
    axial = forces[0] / section.area
    if section.section_type in {"solid_circle", "hollow_circle"}:
        outer_radius = (
            section.radius
            if section.section_type == "solid_circle"
            else section.outer_radius
        )
        assert outer_radius is not None
        increment = outer_radius * hypot(
            forces[1] / section.Iyy,
            forces[2] / section.Izz,
        )
    else:
        assert section.height is not None and section.width is not None
        increment = (
            abs(forces[1] / section.Iyy) * section.width / 2.0
            + abs(forces[2] / section.Izz) * section.height / 2.0
        )
    maximum = axial + increment
    minimum = axial - increment
    return maximum, minimum, max(abs(maximum), abs(minimum))


def _positive_dimension(props: Mapping[str, Any], name: str) -> float:
    """Return one required finite positive section dimension."""
    if name not in props or props[name] in (None, ""):
        raise KeyError(f"Beam2 section missing property {name}")
    try:
        value = float(props[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Beam2 section property {name} must be finite and > 0, got {props[name]!r}"
        ) from exc
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Beam2 section property {name} must be finite and > 0, got {props[name]!r}"
        )
    return value


def _circle_section(
    section_type: str,
    outer_radius: float,
    inner_radius: float,
) -> Beam2Section:
    """Return circular properties using the common annulus formulas."""
    radius_delta_2 = outer_radius**2 - inner_radius**2
    radius_delta_4 = outer_radius**4 - inner_radius**4
    kwargs: dict[str, float | str | None] = {
        "section_type": section_type,
        "area": pi * radius_delta_2,
        "Iyy": pi * radius_delta_4 / 4.0,
        "Izz": pi * radius_delta_4 / 4.0,
        "J": pi * radius_delta_4 / 2.0,
    }
    if section_type == "solid_circle":
        kwargs["radius"] = outer_radius
    else:
        kwargs["outer_radius"] = outer_radius
        kwargs["inner_radius"] = inner_radius
    return Beam2Section(**kwargs)


def _rectangle_torsion_constant(height: float, width: float) -> float:
    """Return the Saint-Venant torsion constant from the convergent odd series."""
    long_side = max(height, width)
    short_side = min(height, width)
    series = 0.0
    for odd in range(1, 10000, 2):
        term = tanh(odd * pi * long_side / (2.0 * short_side)) / odd**5
        series += term
        if term < 1e-15:
            break
    correction = 192.0 * short_side * series / (pi**5 * long_side)
    return long_side * short_side**3 * (1.0 - correction) / 3.0
