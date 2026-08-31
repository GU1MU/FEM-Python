"""Public controls for the currently supported static-step path.

The controls belong to an analysis step rather than to a particular solver.
This keeps the GUI, application layer, and future Agent on the same vocabulary
while the numerical implementation remains free to change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


_DEFAULT_INITIAL_INCREMENT = 0.1
_DEFAULT_MAXIMUM_INCREMENTS = 100
_DEFAULT_NEWTON_MAX_ITERATIONS = 25
_DEFAULT_RESIDUAL_TOLERANCE = 1.0e-6
_DEFAULT_MINIMUM_INCREMENT = 1.0e-6
_DEFAULT_MAXIMUM_INCREMENT = 1.0
_DEFAULT_GROWTH_FACTOR = 1.5
_DEFAULT_GROWTH_ITERATION_THRESHOLD = 4


class StaticFormulation(StrEnum):
    """Persistence-facing spelling of the static geometry mode.

    New execution code uses :class:`GeometryMode` directly.  The formulation
    values remain as a deliberately narrow authoring/persistence projection so
    existing project documents can still describe the same geometry choice.
    """

    LINEAR = "linear_static"
    NONLINEAR = "nonlinear_static"


class GeometryMode(StrEnum):
    """Kinematic model selected independently from constitutive behavior."""

    SMALL_STRAIN = "small_strain"
    FINITE_STRAIN = "finite_strain"

    @classmethod
    def from_formulation(cls, formulation: StaticFormulation) -> "GeometryMode":
        if not isinstance(formulation, StaticFormulation):
            raise TypeError("formulation must be StaticFormulation")
        return (
            cls.FINITE_STRAIN
            if formulation is StaticFormulation.NONLINEAR
            else cls.SMALL_STRAIN
        )

    @property
    def formulation(self) -> StaticFormulation:
        """Return the compatibility projection used at the file boundary."""

        return (
            StaticFormulation.NONLINEAR
            if self is GeometryMode.FINITE_STRAIN
            else StaticFormulation.LINEAR
        )


class StaticControlMode(StrEnum):
    """Proportional control channels used by a static analysis step."""

    LOAD = "load"
    DISPLACEMENT = "displacement"
    MIXED = "mixed"


class NewtonStrategy(StrEnum):
    """Tangent update policy used inside one Newton increment."""

    FULL = "full"
    MODIFIED = "modified"


class DynamicProcedureKind(StrEnum):
    """Time-integration procedure selected by a dynamic analysis step."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class DynamicIntegrationMethod(StrEnum):
    """Time-integration methods owned by the selected dynamic procedure."""

    NEWMARK = "newmark"
    CENTRAL_DIFFERENCE = "central_difference"


class MassMatrixPolicy(StrEnum):
    """Mass representation used by one compiled dynamic system."""

    CONSISTENT = "consistent"
    LUMPED = "lumped"


class DampingModel(StrEnum):
    """Procedure-level damping models."""

    NONE = "none"
    RAYLEIGH = "rayleigh"


@dataclass(frozen=True, slots=True)
class TimeAmplitude:
    """Piecewise-linear scalar history evaluated in step time."""

    points: tuple[tuple[float, float], ...] = ((0.0, 1.0),)

    def __post_init__(self) -> None:
        points = tuple(
            (float(time), float(value))
            for time, value in self.points
        )
        if not points:
            raise ValueError("time amplitude requires at least one point")
        previous_time: float | None = None
        for time, value in points:
            if not math.isfinite(time) or time < 0.0:
                raise ValueError("time amplitude times must be finite and >= 0")
            if not math.isfinite(value):
                raise ValueError("time amplitude values must be finite")
            if previous_time is not None and time <= previous_time:
                raise ValueError("time amplitude times must be strictly increasing")
            previous_time = time
        object.__setattr__(self, "points", points)

    @classmethod
    def constant(cls, value: float = 1.0) -> "TimeAmplitude":
        return cls(((0.0, float(value)),))

    def value_at(self, time: float) -> float:
        """Return the linearly interpolated value, clamped at both ends."""

        target = float(time)
        if not math.isfinite(target):
            raise ValueError("time must be finite")
        if target <= self.points[0][0]:
            return self.points[0][1]
        for left, right in zip(self.points, self.points[1:]):
            if target <= right[0]:
                span = right[0] - left[0]
                fraction = (target - left[0]) / span
                return left[1] + fraction * (right[1] - left[1])
        return self.points[-1][1]


@dataclass(frozen=True, slots=True)
class InitialConditionSet:
    """Sparse initial displacement, velocity, and acceleration values."""

    displacement: Mapping[int, float] = ()
    velocity: Mapping[int, float] = ()
    acceleration: Mapping[int, float] = ()

    def __deepcopy__(self, memo: dict[int, Any]) -> "InitialConditionSet":
        memo[id(self)] = self
        return self

    @property
    def is_empty(self) -> bool:
        return not (
            self.displacement
            or self.velocity
            or self.acceleration
        )

    def to_metadata(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                str(dof): float(value)
                for dof, value in getattr(self, name).items()
            }
            for name in ("displacement", "velocity", "acceleration")
            if getattr(self, name)
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> "InitialConditionSet":
        source = {} if metadata is None else dict(metadata)
        values: dict[str, dict[int, float]] = {}
        for name in ("displacement", "velocity", "acceleration"):
            raw = source.get(name, {})
            if not isinstance(raw, Mapping):
                raise TypeError(f"initial condition {name} must be a mapping")
            values[name] = {int(dof): float(value) for dof, value in raw.items()}
        return cls(**values)

    def __post_init__(self) -> None:
        for name in ("displacement", "velocity", "acceleration"):
            value = getattr(self, name)
            if value == ():
                value = {}
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping of DOF values")
            normalized: dict[int, float] = {}
            for raw_dof, raw_value in value.items():
                if isinstance(raw_dof, bool):
                    raise TypeError(f"{name} DOF ids must be integers")
                dof = int(raw_dof)
                numeric = float(raw_value)
                if dof < 0 or not math.isfinite(numeric):
                    raise ValueError(f"{name} contains an invalid DOF value")
                normalized[dof] = numeric
            object.__setattr__(self, name, MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class DynamicStepControls:
    """Controls shared by Abaqus-style implicit and explicit dynamics.

    The procedure is selected explicitly.  The linear/nonlinear response
    regime is deliberately not a user setting; it is derived from the
    compiled geometry and material behavior.
    """

    procedure = "dynamic"

    time_period: float = 1.0
    initial_time_increment: float = 0.1
    minimum_time_increment: float = 1.0e-8
    maximum_time_increment: float = 1.0
    maximum_increments: int = 1000
    integration_method: DynamicIntegrationMethod = DynamicIntegrationMethod.NEWMARK
    beta: float = 0.25
    gamma: float = 0.5
    mass_matrix: MassMatrixPolicy = MassMatrixPolicy.CONSISTENT
    damping_model: DampingModel = DampingModel.NONE
    rayleigh_mass: float = 0.0
    rayleigh_stiffness: float = 0.0
    amplitude: TimeAmplitude = TimeAmplitude()
    procedure_kind: DynamicProcedureKind = DynamicProcedureKind.IMPLICIT

    def __post_init__(self) -> None:
        period = _positive_finite("time_period", self.time_period)
        initial = _positive_finite(
            "initial_time_increment",
            self.initial_time_increment,
        )
        minimum = _positive_finite(
            "minimum_time_increment",
            self.minimum_time_increment,
        )
        maximum = _positive_finite(
            "maximum_time_increment",
            self.maximum_time_increment,
        )
        if not minimum <= initial <= maximum:
            raise ValueError(
                "dynamic time increments must satisfy minimum <= initial <= maximum"
            )
        maximum_increments = _positive_integer(
            "maximum_increments",
            self.maximum_increments,
        )
        try:
            procedure_kind = DynamicProcedureKind(
                str(self.procedure_kind).strip().casefold()
            )
            integration_method = DynamicIntegrationMethod(
                str(self.integration_method).strip().casefold()
            )
            mass_matrix = MassMatrixPolicy(
                str(self.mass_matrix).strip().casefold()
            )
            damping_model = DampingModel(
                str(self.damping_model).strip().casefold()
            )
        except ValueError as error:
            raise ValueError(
                "unsupported dynamic integration, mass, or damping model"
            ) from error
        if procedure_kind is DynamicProcedureKind.IMPLICIT:
            if integration_method is not DynamicIntegrationMethod.NEWMARK:
                raise ValueError(
                    "implicit dynamics requires the Newmark integration method"
                )
        else:
            if integration_method is DynamicIntegrationMethod.NEWMARK:
                integration_method = DynamicIntegrationMethod.CENTRAL_DIFFERENCE
            if integration_method is not DynamicIntegrationMethod.CENTRAL_DIFFERENCE:
                raise ValueError(
                    "explicit dynamics requires the central-difference integration method"
                )
            # Explicit integration uses a diagonal mass inverse.  Treat the
            # default consistent choice as an authoring shorthand and publish
            # the actual compiled policy as lumped.
            mass_matrix = MassMatrixPolicy.LUMPED
        beta = _positive_finite("beta", self.beta)
        gamma = _positive_finite("gamma", self.gamma)
        if beta <= 0.0 or gamma <= 0.0:
            raise ValueError("Newmark beta and gamma must be > 0")
        rayleigh_mass = _nonnegative_finite(
            "rayleigh_mass",
            self.rayleigh_mass,
        )
        rayleigh_stiffness = _nonnegative_finite(
            "rayleigh_stiffness",
            self.rayleigh_stiffness,
        )
        if not isinstance(self.amplitude, TimeAmplitude):
            raise TypeError("amplitude must be a TimeAmplitude")
        required = max(1, math.ceil(period / initial - 1.0e-12))
        if maximum_increments < required:
            raise ValueError(
                "maximum_increments is too small for the initial time increment"
            )
        object.__setattr__(self, "time_period", period)
        object.__setattr__(self, "initial_time_increment", initial)
        object.__setattr__(self, "minimum_time_increment", minimum)
        object.__setattr__(self, "maximum_time_increment", maximum)
        object.__setattr__(self, "maximum_increments", maximum_increments)
        object.__setattr__(self, "integration_method", integration_method)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "mass_matrix", mass_matrix)
        object.__setattr__(self, "damping_model", damping_model)
        object.__setattr__(self, "rayleigh_mass", rayleigh_mass)
        object.__setattr__(self, "rayleigh_stiffness", rayleigh_stiffness)
        object.__setattr__(self, "procedure_kind", procedure_kind)

    @property
    def time_grid(self) -> tuple[float, ...]:
        """Return accepted target times for the fixed-increment MVP."""

        times: list[float] = []
        current = 0.0
        while current < self.time_period - 1.0e-12:
            increment = min(
                self.initial_time_increment,
                self.maximum_time_increment,
                self.time_period - current,
            )
            if increment < self.minimum_time_increment and times:
                raise ValueError(
                    "final dynamic time increment is below minimum_time_increment"
                )
            current += increment
            if self.time_period - current <= 1.0e-12:
                current = self.time_period
            times.append(current)
            if len(times) > self.maximum_increments:
                raise ValueError("dynamic time integration exceeded maximum_increments")
        return tuple(times)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "time_period": self.time_period,
            "initial_time_increment": self.initial_time_increment,
            "minimum_time_increment": self.minimum_time_increment,
            "maximum_time_increment": self.maximum_time_increment,
            "maximum_increments": self.maximum_increments,
            "integration_method": self.integration_method.value,
            "dynamic_procedure": self.procedure_kind.value,
            "beta": self.beta,
            "gamma": self.gamma,
            "mass_matrix": self.mass_matrix.value,
            "damping_model": self.damping_model.value,
            "rayleigh_mass": self.rayleigh_mass,
            "rayleigh_stiffness": self.rayleigh_stiffness,
            # Project JSON arrays are mutable lists at the wire boundary;
            # keep the in-memory amplitude immutable and convert only here.
            "amplitude_points": [list(point) for point in self.amplitude.points],
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "DynamicStepControls":
        source = {} if metadata is None else dict(metadata)
        points = source.get("amplitude_points", ((0.0, 1.0),))
        return cls(
            time_period=source.get("time_period", 1.0),
            initial_time_increment=source.get("initial_time_increment", 0.1),
            minimum_time_increment=source.get("minimum_time_increment", 1.0e-8),
            maximum_time_increment=source.get("maximum_time_increment", 1.0),
            maximum_increments=source.get("maximum_increments", 1000),
            integration_method=source.get("integration_method", DynamicIntegrationMethod.NEWMARK),
            beta=source.get("beta", 0.25),
            gamma=source.get("gamma", 0.5),
            mass_matrix=source.get("mass_matrix", MassMatrixPolicy.CONSISTENT),
            damping_model=source.get("damping_model", DampingModel.NONE),
            rayleigh_mass=source.get("rayleigh_mass", 0.0),
            rayleigh_stiffness=source.get("rayleigh_stiffness", 0.0),
            amplitude=TimeAmplitude(tuple(tuple(item) for item in points)),
            procedure_kind=source.get(
                "dynamic_procedure",
                source.get("procedure_kind", DynamicProcedureKind.IMPLICIT),
            ),
        )


@runtime_checkable
class ProcedureControls(Protocol):
    """Marker contract implemented by one procedure's typed controls."""

    procedure: str


@dataclass(frozen=True, slots=True)
class StaticAnalysisOptions:
    """Typed procedure options consumed by the static compiler."""

    material_algorithm: str = "hencky"

    def __post_init__(self) -> None:
        algorithm = str(self.material_algorithm).strip().casefold()
        if not algorithm:
            raise ValueError("material_algorithm must be nonblank")
        object.__setattr__(self, "material_algorithm", algorithm)

    def as_mapping(self) -> dict[str, str]:
        """Return the narrow compiler view without exposing a request mapping."""

        return {"material_algorithm": self.material_algorithm}


@dataclass(frozen=True, slots=True)
class StaticStepControls:
    """Supported static-step increment and Newton controls.

    ``initial_increment`` is the starting load-factor increment from zero to
    one.  The numerical path may cut back a failed increment and, when opted
    in, grow an easy path.  ``maximum_increment`` caps the size of an
    automatically grown increment; ``maximum_increments`` caps the number of
    accepted and retried increments.  Stabilization, arc length, and dynamic
    controls are deliberately outside this contract.
    """

    procedure = "static"

    initial_increment: float = _DEFAULT_INITIAL_INCREMENT
    maximum_increments: int = _DEFAULT_MAXIMUM_INCREMENTS
    newton_max_iterations: int = _DEFAULT_NEWTON_MAX_ITERATIONS
    residual_tolerance: float = _DEFAULT_RESIDUAL_TOLERANCE
    control_mode: StaticControlMode = StaticControlMode.MIXED
    relative_residual_tolerance: float = 1.0e-8
    displacement_tolerance: float = 1.0e-8
    energy_tolerance: float = 1.0e-8
    constraint_tolerance: float = 1.0e-10
    newton_strategy: NewtonStrategy = NewtonStrategy.FULL
    line_search: bool = True
    predictor: bool = True
    automatic_cutback: bool = True
    minimum_increment: float = _DEFAULT_MINIMUM_INCREMENT
    maximum_increment: float = _DEFAULT_MAXIMUM_INCREMENT
    adaptive_growth: bool = False
    growth_factor: float = _DEFAULT_GROWTH_FACTOR
    growth_iteration_threshold: int = _DEFAULT_GROWTH_ITERATION_THRESHOLD

    def __post_init__(self) -> None:
        increment = _positive_finite(
            "initial_increment",
            self.initial_increment,
            upper=1.0,
        )
        maximum_increments = _positive_integer(
            "maximum_increments",
            self.maximum_increments,
        )
        max_iterations = _positive_integer(
            "newton_max_iterations",
            self.newton_max_iterations,
        )
        tolerance = _positive_finite(
            "residual_tolerance",
            self.residual_tolerance,
        )
        try:
            control_mode = StaticControlMode(
                str(self.control_mode).strip().casefold()
            )
        except ValueError as error:
            raise ValueError(
                "control_mode must be 'load', 'displacement', or 'mixed'"
            ) from error
        relative_tolerance = _positive_finite(
            "relative_residual_tolerance",
            self.relative_residual_tolerance,
        )
        displacement_tolerance = _positive_finite(
            "displacement_tolerance",
            self.displacement_tolerance,
        )
        energy_tolerance = _positive_finite(
            "energy_tolerance",
            self.energy_tolerance,
        )
        constraint_tolerance = _positive_finite(
            "constraint_tolerance",
            self.constraint_tolerance,
        )
        try:
            newton_strategy = NewtonStrategy(
                str(self.newton_strategy).strip().casefold()
            )
        except ValueError as error:
            raise ValueError(
                "newton_strategy must be 'full' or 'modified'"
            ) from error
        for name in (
            "line_search",
            "predictor",
            "automatic_cutback",
            "adaptive_growth",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        minimum_increment = _positive_finite(
            "minimum_increment",
            self.minimum_increment,
        )
        maximum_increment = _positive_finite(
            "maximum_increment",
            self.maximum_increment,
            upper=1.0,
        )
        if maximum_increment < increment:
            raise ValueError(
                "maximum_increment must be at least initial_increment"
            )
        growth_factor = _positive_finite(
            "growth_factor",
            self.growth_factor,
        )
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1")
        growth_iteration_threshold = _positive_integer(
            "growth_iteration_threshold",
            self.growth_iteration_threshold,
        )
        required = _required_increments(increment)
        if maximum_increments < required and not adaptive_growth:
            raise ValueError(
                "maximum_increments must be at least "
                f"{required} for initial_increment={increment:g}"
            )
        object.__setattr__(self, "initial_increment", increment)
        object.__setattr__(self, "maximum_increments", maximum_increments)
        object.__setattr__(self, "newton_max_iterations", max_iterations)
        object.__setattr__(self, "residual_tolerance", tolerance)
        object.__setattr__(self, "control_mode", control_mode)
        object.__setattr__(
            self,
            "relative_residual_tolerance",
            relative_tolerance,
        )
        object.__setattr__(
            self,
            "displacement_tolerance",
            displacement_tolerance,
        )
        object.__setattr__(self, "energy_tolerance", energy_tolerance)
        object.__setattr__(self, "constraint_tolerance", constraint_tolerance)
        object.__setattr__(self, "newton_strategy", newton_strategy)
        object.__setattr__(self, "minimum_increment", minimum_increment)
        object.__setattr__(self, "maximum_increment", maximum_increment)
        object.__setattr__(self, "growth_factor", growth_factor)
        object.__setattr__(
            self,
            "growth_iteration_threshold",
            growth_iteration_threshold,
        )

    @property
    def required_increments(self) -> int:
        """Return the number of fixed increments needed to reach factor one."""

        return _required_increments(self.initial_increment)

    @property
    def load_factors(self) -> tuple[float, ...]:
        """Return converged target factors for the current fixed-step path."""

        factors = tuple(
            min(1.0, index * self.initial_increment)
            for index in range(1, self.required_increments + 1)
        )
        # Avoid a representation such as 0.9999999999999999 at the final
        # target; the analysis step always ends at the normalized factor one.
        return factors[:-1] + (1.0,)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the public controls into step metadata."""

        return {
            "initial_increment": self.initial_increment,
            "maximum_increments": self.maximum_increments,
            "newton_max_iterations": self.newton_max_iterations,
            "residual_tolerance": self.residual_tolerance,
            "control_mode": self.control_mode.value,
            "relative_residual_tolerance": self.relative_residual_tolerance,
            "displacement_tolerance": self.displacement_tolerance,
            "energy_tolerance": self.energy_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
            "newton_strategy": self.newton_strategy.value,
            "line_search": self.line_search,
            "predictor": self.predictor,
            "automatic_cutback": self.automatic_cutback,
            "minimum_increment": self.minimum_increment,
            "maximum_increment": self.maximum_increment,
            "adaptive_growth": self.adaptive_growth,
            "growth_factor": self.growth_factor,
            "growth_iteration_threshold": self.growth_iteration_threshold,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "StaticStepControls":
        """Build controls from public metadata and known legacy aliases."""

        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("analysis step metadata must be a mapping")
        return cls(
            initial_increment=_metadata_value(
                metadata,
                ("initial_increment", "initial_load_increment"),
                _DEFAULT_INITIAL_INCREMENT,
            ),
            maximum_increments=_metadata_value(
                metadata,
                ("maximum_increments", "max_increments"),
                _DEFAULT_MAXIMUM_INCREMENTS,
            ),
            newton_max_iterations=_metadata_value(
                metadata,
                (
                    "newton_max_iterations",
                    "finite_strain_max_iterations",
                    "max_iterations",
                ),
                _DEFAULT_NEWTON_MAX_ITERATIONS,
            ),
            residual_tolerance=_metadata_value(
                metadata,
                (
                    "residual_tolerance",
                    "newton_residual_tolerance",
                    "finite_strain_residual_tolerance",
                ),
                _DEFAULT_RESIDUAL_TOLERANCE,
            ),
            control_mode=_metadata_value(
                metadata,
                ("control_mode",),
                StaticControlMode.MIXED,
            ),
            relative_residual_tolerance=_metadata_value(
                metadata,
                ("relative_residual_tolerance",),
                1.0e-8,
            ),
            displacement_tolerance=_metadata_value(
                metadata,
                ("displacement_tolerance",),
                1.0e-8,
            ),
            energy_tolerance=_metadata_value(
                metadata,
                ("energy_tolerance",),
                1.0e-8,
            ),
            constraint_tolerance=_metadata_value(
                metadata,
                ("constraint_tolerance",),
                1.0e-10,
            ),
            newton_strategy=_metadata_value(
                metadata,
                ("newton_strategy", "tangent_strategy"),
                NewtonStrategy.FULL,
            ),
            line_search=_metadata_value(metadata, ("line_search",), True),
            predictor=_metadata_value(metadata, ("predictor",), True),
            automatic_cutback=_metadata_value(
                metadata,
                ("automatic_cutback", "adaptive_cutback"),
                True,
            ),
            minimum_increment=_metadata_value(
                metadata,
                ("minimum_increment", "minimum_load_increment"),
                _DEFAULT_MINIMUM_INCREMENT,
            ),
            maximum_increment=_metadata_value(
                metadata,
                ("maximum_increment", "maximum_load_increment"),
                _DEFAULT_MAXIMUM_INCREMENT,
            ),
            adaptive_growth=_metadata_value(
                metadata,
                ("adaptive_growth", "adaptive_increment"),
                False,
            ),
            growth_factor=_metadata_value(
                metadata,
                ("growth_factor", "increment_growth_factor"),
                _DEFAULT_GROWTH_FACTOR,
            ),
            growth_iteration_threshold=_metadata_value(
                metadata,
                ("growth_iteration_threshold", "easy_iteration_threshold"),
                _DEFAULT_GROWTH_ITERATION_THRESHOLD,
            ),
        )


def _metadata_value(
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    normalized = {
        str(key).strip().casefold(): value
        for key, value in metadata.items()
    }
    for key in keys:
        if key in normalized:
            return normalized[key]
    return default


def _positive_finite(
    name: str,
    value: Any,
    *,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and > 0")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and > 0") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be <= {upper:g}")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= 1") from error
    if result < 1 or result != value:
        raise ValueError(f"{name} must be an integer >= 1")
    return result


def _nonnegative_finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and >= 0")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and >= 0") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _required_increments(increment: float) -> int:
    return max(1, math.ceil(1.0 / increment - 1.0e-12))


__all__ = [
    "DampingModel",
    "DynamicProcedureKind",
    "DynamicIntegrationMethod",
    "DynamicStepControls",
    "GeometryMode",
    "InitialConditionSet",
    "MassMatrixPolicy",
    "NewtonStrategy",
    "ProcedureControls",
    "StaticControlMode",
    "StaticFormulation",
    "StaticAnalysisOptions",
    "StaticStepControls",
    "TimeAmplitude",
]
