"""Explicit result-display intent kept separate from field recovery data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .fields import FieldPosition, ScalarFieldSelection
from .frames import ResultFrameKey


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


class ResultDisplayComputation(str, Enum):
    """How a selected field is represented in the viewport."""

    NATIVE = "native"
    INTEGRATION_POINT_MARKERS = "integration_point_markers"
    CENTROID_CELLS = "centroid_cells"
    ELEMENT_NODAL_QUILT = "element_nodal_quilt"
    NODAL_AVERAGED = "nodal_averaged"


def display_computation_for_position(
    position: FieldPosition,
) -> ResultDisplayComputation:
    """Resolve the default render meaning for one stored field position.

    ``NATIVE`` remains useful for callers that do not choose a display mode,
    but it must resolve to one explicit continuum representation before rendering.
    Legacy node-region and resolved-nodal positions are accepted here only as
    compatibility inputs; new callers should use ``NODAL_AVERAGED`` with an
    averaging policy.
    """

    if type(position) is not FieldPosition:
        raise TypeError("position must be FieldPosition")
    return {
        FieldPosition.NODE: ResultDisplayComputation.NATIVE,
        FieldPosition.INTEGRATION_POINT: (
            ResultDisplayComputation.INTEGRATION_POINT_MARKERS
        ),
        FieldPosition.CENTROID: ResultDisplayComputation.CENTROID_CELLS,
        FieldPosition.ELEMENT_NODAL: (
            ResultDisplayComputation.ELEMENT_NODAL_QUILT
        ),
        FieldPosition.NODE_REGION: ResultDisplayComputation.NODAL_AVERAGED,
        FieldPosition.RESOLVED_NODAL: ResultDisplayComputation.NODAL_AVERAGED,
        FieldPosition.SECTION_END: ResultDisplayComputation.NATIVE,
        FieldPosition.SECTION_POINT: ResultDisplayComputation.NATIVE,
        FieldPosition.SECTION_NODE_ENVELOPE: ResultDisplayComputation.NATIVE,
    }[position]


class ResultDisplayScopeKind(str, Enum):
    """Geometry scope, independent of result sampling position."""

    WHOLE_MODEL = "whole_model"
    NODE_SET = "node_set"
    ELEMENT_SET = "element_set"
    EXTERIOR = "exterior"


class ResultDeformationMode(str, Enum):
    """Whether the viewport uses the undeformed or deformed geometry."""

    UNDEFORMED = "undeformed"
    DEFORMED = "deformed"


class ResultLegendMode(str, Enum):
    """Range policy for a scalar legend during frame navigation."""

    PER_FRAME = "per_frame"
    GLOBAL_STEP = "global_step"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ResultDisplayScope:
    """Named geometric display scope, not a result-field domain."""

    kind: ResultDisplayScopeKind = ResultDisplayScopeKind.WHOLE_MODEL
    name: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ResultDisplayScopeKind:
            raise TypeError("kind must be ResultDisplayScopeKind")
        if self.name is not None:
            if type(self.name) is not str or not self.name.strip():
                raise ValueError("name must be a nonblank string or None")
        if self.kind in {
            ResultDisplayScopeKind.NODE_SET,
            ResultDisplayScopeKind.ELEMENT_SET,
        } and self.name is None:
            raise ValueError("named display scopes require a name")
        if self.kind in {
            ResultDisplayScopeKind.WHOLE_MODEL,
            ResultDisplayScopeKind.EXTERIOR,
        } and self.name is not None:
            raise ValueError("this display scope kind cannot have a name")


@dataclass(frozen=True, slots=True)
class ResultAveragingOptions:
    """Display-time averaging options, separate from field position."""

    threshold_percent: float = 75.0
    preserve_region_boundaries: bool = True
    displayed_elements_only: bool = False
    invariant_after_averaging: bool = False

    def __post_init__(self) -> None:
        threshold = _finite_real(
            self.threshold_percent,
            name="threshold_percent",
        )
        if not 0.0 <= threshold <= 100.0:
            raise ValueError("threshold_percent must be from 0.0 through 100.0")
        object.__setattr__(self, "threshold_percent", threshold)
        for name in (
            "preserve_region_boundaries",
            "displayed_elements_only",
            "invariant_after_averaging",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class ResultLegendPolicy:
    """Scalar legend range policy, including explicit manual bounds."""

    mode: ResultLegendMode = ResultLegendMode.PER_FRAME
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not ResultLegendMode:
            raise TypeError("mode must be ResultLegendMode")
        if self.minimum is not None:
            _finite_real(self.minimum, name="minimum")
        if self.maximum is not None:
            _finite_real(self.maximum, name="maximum")
        if self.mode is ResultLegendMode.MANUAL:
            if self.minimum is None or self.maximum is None:
                raise ValueError("manual legend mode requires both bounds")
            if float(self.minimum) >= float(self.maximum):
                raise ValueError("manual legend minimum must be below maximum")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError(
                "non-manual legend modes cannot carry explicit bounds"
            )


@dataclass(frozen=True, slots=True)
class ResultDisplayQuery:
    """Complete, serializable intent for one field visualization."""

    frame: ResultFrameKey
    selection: ScalarFieldSelection
    computation: ResultDisplayComputation = ResultDisplayComputation.NATIVE
    scope: ResultDisplayScope = ResultDisplayScope()
    deformation: ResultDeformationMode = ResultDeformationMode.DEFORMED
    deformation_scale: float | None = None
    averaging: ResultAveragingOptions = ResultAveragingOptions()
    legend: ResultLegendPolicy = ResultLegendPolicy()

    def __post_init__(self) -> None:
        if type(self.frame) is not ResultFrameKey:
            raise TypeError("frame must be ResultFrameKey")
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        if type(self.computation) is not ResultDisplayComputation:
            raise TypeError(
                "computation must be ResultDisplayComputation"
            )
        if type(self.scope) is not ResultDisplayScope:
            raise TypeError("scope must be ResultDisplayScope")
        if type(self.deformation) is not ResultDeformationMode:
            raise TypeError(
                "deformation must be ResultDeformationMode"
            )
        if self.deformation_scale is not None:
            scale = _finite_real(
                self.deformation_scale,
                name="deformation_scale",
            )
            if scale < 0.0:
                raise ValueError("deformation_scale must be non-negative")
            object.__setattr__(self, "deformation_scale", scale)
        if type(self.averaging) is not ResultAveragingOptions:
            raise TypeError("averaging must be ResultAveragingOptions")
        if type(self.legend) is not ResultLegendPolicy:
            raise TypeError("legend must be ResultLegendPolicy")
        _validate_computation_position(
            self.selection.field_key.request.field_id.position,
            self.computation,
        )

    @property
    def field_position(self) -> FieldPosition:
        """Return the selected field's source position."""

        return self.selection.field_key.request.field_id.position

    @property
    def effective_computation(self) -> ResultDisplayComputation:
        """Return the explicit render mode after resolving ``NATIVE``."""

        if self.computation is not ResultDisplayComputation.NATIVE:
            return self.computation
        return display_computation_for_position(self.field_position)


def _validate_computation_position(
    position: FieldPosition,
    computation: ResultDisplayComputation,
) -> None:
    if type(position) is not FieldPosition:
        raise TypeError("position must be FieldPosition")
    if computation is ResultDisplayComputation.INTEGRATION_POINT_MARKERS:
        if position is not FieldPosition.INTEGRATION_POINT:
            raise ValueError(
                "integration-point markers require an integration-point field"
            )
    elif computation is ResultDisplayComputation.CENTROID_CELLS:
        if position is not FieldPosition.CENTROID:
            raise ValueError("centroid cells require a centroid field")
    elif computation is ResultDisplayComputation.ELEMENT_NODAL_QUILT:
        if position is not FieldPosition.ELEMENT_NODAL:
            raise ValueError(
                "element-nodal quilt requires an element-nodal field"
            )
    elif computation is ResultDisplayComputation.NODAL_AVERAGED:
        if position not in {
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
        }:
            raise ValueError(
                "nodal averaging requires an element-based continuum field"
            )


__all__ = [
    "ResultAveragingOptions",
    "ResultDeformationMode",
    "ResultDisplayComputation",
    "ResultDisplayQuery",
    "ResultDisplayScope",
    "ResultDisplayScopeKind",
    "ResultLegendMode",
    "ResultLegendPolicy",
    "display_computation_for_position",
]
