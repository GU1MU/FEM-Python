from __future__ import annotations

from dataclasses import dataclass
from math import cosh, hypot, isfinite, pi, sqrt
import operator
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
ABAQUS_B31_SLENDERNESS_COMPENSATION = 0.25


@dataclass(frozen=True)
class Beam2Section:
    """Validated Beam2 section properties and source dimensions.

    For rectangular sections, ``width`` spans local ``y`` and ``height``
    spans local ``z``.
    """

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

    def shear_correction_factors(
        self,
        poisson_ratio: float,
    ) -> tuple[float, float]:
        """Return Abaqus ``(kappa_y, kappa_z)`` for the local shear axes.

        Abaqus uses 0.85 for RECT and 0.89 for CIRC.  Its THICK PIPE factor
        follows the Cowper annulus expression and therefore depends on the
        isotropic material Poisson ratio.  All supported sections are doubly
        symmetric, so both directions have the same coefficient.
        """

        nu = _validated_poisson_ratio(poisson_ratio)
        if self.section_type == "rectangle":
            kappa = 0.85
        elif self.section_type == "solid_circle":
            kappa = 0.89
        else:
            assert self.outer_radius is not None and self.inner_radius is not None
            radius_ratio_squared = (self.inner_radius / self.outer_radius) ** 2
            radius_term = (1.0 + radius_ratio_squared) ** 2
            kappa = 6.0 * (1.0 + nu) * radius_term / (
                (7.0 + 6.0 * nu) * radius_term
                + (20.0 + 12.0 * nu) * radius_ratio_squared
            )
        return kappa, kappa

    def effective_shear_rigidities(
        self,
        shear_modulus: float,
        poisson_ratio: float,
    ) -> tuple[float, float]:
        """Return effective local ``(kappa_y G A, kappa_z G A)``."""

        try:
            shear = float(shear_modulus)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Beam2 shear modulus must be finite and > 0, "
                f"got {shear_modulus!r}"
            ) from error
        if not isfinite(shear) or shear <= 0.0:
            raise ValueError(
                "Beam2 shear modulus must be finite and > 0, "
                f"got {shear_modulus!r}"
            )
        kappa_y, kappa_z = self.shear_correction_factors(poisson_ratio)
        return kappa_y * shear * self.area, kappa_z * shear * self.area

    def abaqus_b31_shear_rigidities(
        self,
        shear_modulus: float,
        poisson_ratio: float,
        length: float,
        slenderness_compensation: float = ABAQUS_B31_SLENDERNESS_COMPENSATION,
    ) -> tuple[float, float]:
        """Return directionally compensated B31 shear rigidities.

        The local-y shear mode bends about local z and therefore consumes
        ``Izz``; the local-z shear mode consumes ``Iyy``.  Abaqus applies the
        first-order factor independently to those two modes.
        """

        actual_y, actual_z = self.effective_shear_rigidities(
            shear_modulus,
            poisson_ratio,
        )
        try:
            element_length = float(length)
            compensation = float(slenderness_compensation)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Beam2 B31 length must be finite and > 0 and slenderness "
                "compensation must be finite and >= 0"
            ) from error
        if not isfinite(element_length) or element_length <= 0.0:
            raise ValueError(
                "Beam2 B31 length must be finite and > 0, "
                f"got {length!r}"
            )
        if not isfinite(compensation) or compensation < 0.0:
            raise ValueError(
                "Beam2 B31 slenderness compensation must be finite and >= 0, "
                f"got {slenderness_compensation!r}"
            )
        factor_y = 1.0 / (
            1.0
            + compensation
            * element_length**2
            * self.area
            / (12.0 * self.Izz)
        )
        factor_z = 1.0 / (
            1.0
            + compensation
            * element_length**2
            * self.area
            / (12.0 * self.Iyy)
        )
        return factor_y * actual_y, factor_z * actual_z


@dataclass(frozen=True, slots=True)
class BeamSectionPoint:
    """One immutable point in the Beam2 local ``(y, z)`` section plane."""

    number: int
    local_y: float
    local_z: float

    def __post_init__(self) -> None:
        if isinstance(self.number, bool):
            raise TypeError("Beam2 section point number must be an integer")
        try:
            number = operator.index(self.number)
        except TypeError as error:
            raise TypeError("Beam2 section point number must be an integer") from error
        if number <= 0:
            raise ValueError("Beam2 section point number must be positive")
        local_y = float(self.local_y)
        local_z = float(self.local_z)
        if not isfinite(local_y) or not isfinite(local_z):
            raise ValueError("Beam2 section point coordinates must be finite")
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "local_y", local_y)
        object.__setattr__(self, "local_z", local_z)

    @property
    def local_coordinates(self) -> tuple[float, float]:
        """Return the point coordinates in local ``(y, z)`` order."""

        return self.local_y, self.local_z


@dataclass(frozen=True, slots=True)
class BeamSectionEndForces:
    """Tension-positive Beam2 resultants at one element-end section.

    The first four fields retain the historical positional constructor order.
    Transverse shear resultants are appended so existing
    ``BeamSectionEndForces(N, My, Mz, T)`` callers remain valid.
    """

    axial_force: float
    moment_y: float
    moment_z: float
    torque: float
    shear_y: float = 0.0
    shear_z: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "axial_force",
            "moment_y",
            "moment_z",
            "torque",
            "shear_y",
            "shear_z",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"Beam2 section force {name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def N(self) -> float:
        return self.axial_force

    @property
    def Vy(self) -> float:
        return self.shear_y

    @property
    def Vz(self) -> float:
        return self.shear_z

    @property
    def My(self) -> float:
        return self.moment_y

    @property
    def Mz(self) -> float:
        return self.moment_z

    @property
    def T(self) -> float:
        return self.torque


@dataclass(frozen=True, slots=True)
class BeamSectionPointStress:
    """Stress components and invariants evaluated at one section point."""

    point: BeamSectionPoint
    s11: float
    s12: float
    mises: float
    max_principal: float
    mid_principal: float
    min_principal: float

    def values(self) -> dict[str, float]:
        """Return the canonical point-stress component mapping."""

        return {
            "S11": self.s11,
            "S12": self.s12,
            "Mises": self.mises,
            "MaxPrincipal": self.max_principal,
            "MidPrincipal": self.mid_principal,
            "MinPrincipal": self.min_principal,
        }


@dataclass(frozen=True, slots=True)
class BeamSectionStress:
    """Point stresses and true-section extrema for one Beam2 section end."""

    forces: BeamSectionEndForces
    point_stresses: tuple[BeamSectionPointStress, ...]
    s11_max: float
    s11_min: float
    s11_abs_max: float
    s12_abs_max: float

    def section_values(self) -> dict[str, float]:
        """Return the canonical section-level stress component mapping."""

        return {
            "S11Max": self.s11_max,
            "S11Min": self.s11_min,
            "S11AbsMax": self.s11_abs_max,
            "S12AbsMax": self.s12_abs_max,
        }


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
        Iyy=width * height**3 / 12.0,
        Izz=height * width**3 / 12.0,
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
            abs(forces[1] / section.Iyy) * section.height / 2.0
            + abs(forces[2] / section.Izz) * section.width / 2.0
        )
    maximum = axial + increment
    minimum = axial - increment
    return maximum, minimum, max(abs(maximum), abs(minimum))


def default_section_points(
    section: Beam2Section,
) -> tuple[BeamSectionPoint, ...]:
    """Return the four canonical Beam2 section points in ring order.

    Rectangles start at ``(+y, +z)`` and proceed from local ``+y`` toward
    local ``+z``, with ``y = +/-width/2`` and ``z = +/-height/2``.  Circular
    sections start at ``+y`` and follow the same sense through ``+z``.
    Circular points always lie on the outer radius.
    """

    if not isinstance(section, Beam2Section):
        raise TypeError("section must be Beam2Section")
    if section.section_type == "rectangle":
        assert section.height is not None and section.width is not None
        half_y = section.width / 2.0
        half_z = section.height / 2.0
        coordinates = (
            (half_y, half_z),
            (-half_y, half_z),
            (-half_y, -half_z),
            (half_y, -half_z),
        )
    else:
        radius = (
            section.radius
            if section.section_type == "solid_circle"
            else section.outer_radius
        )
        assert radius is not None
        coordinates = (
            (radius, 0.0),
            (0.0, radius),
            (-radius, 0.0),
            (0.0, -radius),
        )
    return tuple(
        BeamSectionPoint(number, local_y, local_z)
        for number, (local_y, local_z) in enumerate(coordinates, start=1)
    )


def recover_section_point_stress(
    section: Beam2Section,
    forces: BeamSectionEndForces,
) -> BeamSectionStress:
    """Recover point stresses and true-section extrema from end resultants."""

    if not isinstance(section, Beam2Section):
        raise TypeError("section must be Beam2Section")
    if not isinstance(forces, BeamSectionEndForces):
        raise TypeError("forces must be BeamSectionEndForces")

    axial = forces.axial_force / section.area
    circle = section.section_type in {"solid_circle", "hollow_circle"}
    circle_s12 = 0.0
    if circle:
        radius = (
            section.radius
            if section.section_type == "solid_circle"
            else section.outer_radius
        )
        assert radius is not None
        circle_s12 = forces.torque * radius / section.J

    point_stresses = []
    for point in default_section_points(section):
        s11 = (
            axial
            + forces.moment_y * point.local_z / section.Iyy
            - forces.moment_z * point.local_y / section.Izz
        )
        # S12 is signed in the point's tangential transverse basis.  Circular
        # values therefore share the torque sign.  Rectangle corners lie on
        # two traction-free edges and have zero Saint-Venant shear.
        s12 = circle_s12 if circle else 0.0
        principal_span = sqrt(s11**2 + 4.0 * s12**2)
        point_stresses.append(
            BeamSectionPointStress(
                point=point,
                s11=s11,
                s12=s12,
                mises=sqrt(s11**2 + 3.0 * s12**2),
                max_principal=(s11 + principal_span) / 2.0,
                mid_principal=0.0,
                min_principal=(s11 - principal_span) / 2.0,
            )
        )

    s11_max, s11_min, s11_abs_max = axial_stress_extrema(
        section,
        forces.axial_force,
        forces.moment_y,
        forces.moment_z,
    )
    s12_abs_max = (
        abs(circle_s12)
        if circle
        else _rectangle_torsion_shear_abs_max(section, forces.torque)
    )
    return BeamSectionStress(
        forces=forces,
        point_stresses=tuple(point_stresses),
        s11_max=s11_max,
        s11_min=s11_min,
        s11_abs_max=s11_abs_max,
        s12_abs_max=s12_abs_max,
    )


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


def _validated_poisson_ratio(value: float) -> float:
    """Return one finite isotropic Poisson ratio in its admissible range."""

    try:
        poisson_ratio = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Beam2 Poisson ratio must be finite and satisfy -1 < nu < 0.5, "
            f"got {value!r}"
        ) from error
    if not isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError(
            "Beam2 Poisson ratio must be finite and satisfy -1 < nu < 0.5, "
            f"got {value!r}"
        )
    return poisson_ratio


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
    """Return Abaqus' effective default 5x5 integrated-RECT torsion constant.

    Abaqus first obtains a normalized Prandtl stress function on its local
    second-order cross-section model, then integrates the resulting torsional
    shear-strain energy at the default 5x5 Simpson section points.  Eliminating
    that fixed finite-dimensional solve gives the symmetric rational form
    below.  It intentionally represents the default integrated-section
    discretization, not the continuous rectangular Saint-Venant series.
    """

    long_side = max(height, width)
    short_side = min(height, width)
    aspect_ratio = long_side / short_side
    transformed = ((aspect_ratio - 1.0) / (aspect_ratio + 1.0)) ** 2
    reduced_numerator = (
        (
            (
                (169.0 / 2400.0 * transformed - 0.18806666666666667)
                * transformed
                + 0.26943333333333336
            )
            * transformed
            - 0.18806666666666667
        )
        * transformed
        + 169.0 / 2400.0
    )
    denominator = (
        (
            (
                (
                    (
                        (transformed - 0.72) * transformed - 0.5904
                    )
                    * transformed
                    + 2.2592
                )
                * transformed
                - 0.5904
            )
            * transformed
            - 0.72
        )
        * transformed
        + 1.0
    )
    dimensionless_constant = (
        (1.0 + aspect_ratio**2)
        * (1.0 - transformed) ** 2
        * reduced_numerator
        / denominator
    )
    return long_side * short_side**3 * dimensionless_constant


def _rectangle_torsion_shear_abs_max(
    section: Beam2Section,
    torque: float,
) -> float:
    """Return the Saint-Venant boundary maximum for a rectangle."""

    assert section.height is not None and section.width is not None
    long_side = max(section.height, section.width)
    short_side = min(section.height, section.width)
    inverse_cosh_series = 0.0
    for odd in range(1, 10000, 2):
        argument = odd * pi * long_side / (2.0 * short_side)
        term = (0.0 if argument > 40.0 else 1.0 / cosh(argument)) / odd**2
        inverse_cosh_series += term
        if term < 1e-15:
            break
    series = pi**2 / 8.0 - inverse_cosh_series
    shear_per_twist = 8.0 * short_side * series / pi**2
    return abs(float(torque)) * shear_per_twist / section.J
