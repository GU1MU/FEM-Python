"""Canonical result-field identities without solver or GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fem.post.averaging import NodalAveragingPolicy


class FieldAssociation(str, Enum):
    """Identity association for one result-field record."""

    NODE = "node"
    ELEMENT = "element"
    INTEGRATION_POINT = "integration_point"
    ELEMENT_NODE = "element_node"
    NODE_REGION = "node_region"
    RESOLVED_NODAL = "resolved_nodal"


class PhysicalQuantity(str, Enum):
    """Physical dimension represented by a result field."""

    DISPLACEMENT = "displacement"
    ROTATION = "rotation"
    FORCE = "force"
    MOMENT = "moment"
    STRESS = "stress"
    STRAIN = "strain"


class ResultVariable(str, Enum):
    """Canonical engineering variables published by a result provider."""

    U = "U"
    UR = "UR"
    RF = "RF"
    RM = "RM"
    SF = "SF"
    SM = "SM"
    S = "S"
    LE = "LE"


class FieldPosition(str, Enum):
    """Canonical recovery or sampling position for one result variable."""

    NODE = "node"
    INTEGRATION_POINT = "integration_point"
    CENTROID = "centroid"
    ELEMENT_NODAL = "element_nodal"
    NODE_REGION = "node_region"
    RESOLVED_NODAL = "resolved_nodal"
    SECTION_POINT = "section_point"
    SECTION_END = "section_end"
    SECTION_NODE_ENVELOPE = "section_node_envelope"


_VARIABLE_POSITIONS = {
    ResultVariable.U: frozenset({FieldPosition.NODE}),
    ResultVariable.UR: frozenset({FieldPosition.NODE}),
    ResultVariable.RF: frozenset({FieldPosition.NODE}),
    ResultVariable.RM: frozenset({FieldPosition.NODE}),
    ResultVariable.SF: frozenset({FieldPosition.INTEGRATION_POINT}),
    ResultVariable.SM: frozenset({FieldPosition.INTEGRATION_POINT}),
    ResultVariable.S: frozenset(
        {
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
            FieldPosition.SECTION_END,
            FieldPosition.SECTION_POINT,
            FieldPosition.SECTION_NODE_ENVELOPE,
        }
    ),
    ResultVariable.LE: frozenset({FieldPosition.CENTROID}),
}

_CONTINUUM_STRESS_RECOVERY_POSITIONS = frozenset(
    {
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
        FieldPosition.NODE_REGION,
        FieldPosition.RESOLVED_NODAL,
    }
)

_FIELD_VARIABLE_SORT_ORDER = {
    ResultVariable.U: 0,
    ResultVariable.UR: 1,
    ResultVariable.RF: 2,
    ResultVariable.RM: 3,
    ResultVariable.SF: 4,
    ResultVariable.SM: 5,
    ResultVariable.LE: 6,
    ResultVariable.S: 7,
}
_POSITION_ORDER = {
    value: index for index, value in enumerate(FieldPosition)
}


@dataclass(frozen=True, slots=True)
class ResultFieldId:
    """Unique typed identity for one variable at one recovery position."""

    variable: ResultVariable
    position: FieldPosition
    section_point_number: int | None = None

    def __post_init__(self) -> None:
        if type(self.variable) is not ResultVariable:
            raise TypeError("variable must be ResultVariable")
        if type(self.position) is not FieldPosition:
            raise TypeError("position must be FieldPosition")
        if self.position not in _VARIABLE_POSITIONS[self.variable]:
            raise ValueError(
                f"{self.variable.value} is not available at "
                f"{self.position.value}"
            )
        if self.position is FieldPosition.SECTION_POINT:
            if type(self.section_point_number) is not int:
                raise TypeError(
                    "section_point fields require an integer point number"
                )
            if self.section_point_number <= 0:
                raise ValueError("section point number must be positive")
        elif self.section_point_number is not None:
            if (
                self.variable is ResultVariable.S
                and self.position is FieldPosition.INTEGRATION_POINT
                and type(self.section_point_number) is int
                and self.section_point_number > 0
            ):
                return
            raise ValueError(
                "section_point_number is only valid for S at section_point or "
                "integration_point position"
            )


@dataclass(frozen=True, slots=True)
class FieldRequest:
    """Intrinsic numerical intent before a provider resolves its contract."""

    field_id: ResultFieldId
    averaging_policy: NodalAveragingPolicy | None = None
    gauss_order: int | None = None

    def __post_init__(self) -> None:
        if type(self.field_id) is not ResultFieldId:
            raise TypeError("field_id must be ResultFieldId")
        if (
            self.averaging_policy is not None
            and type(self.averaging_policy) is not NodalAveragingPolicy
        ):
            raise TypeError(
                "averaging_policy must be NodalAveragingPolicy or None"
            )

        position = self.field_id.position
        if position is FieldPosition.RESOLVED_NODAL:
            if self.averaging_policy is None:
                raise ValueError(
                    "resolved_nodal requests require an averaging policy"
                )
        elif self.averaging_policy is not None:
            raise ValueError(
                "averaging policy is only valid for resolved_nodal requests"
            )

        if self.gauss_order is None:
            return
        if type(self.gauss_order) is not int:
            raise TypeError("gauss_order must be an integer or None")
        if self.gauss_order <= 0:
            raise ValueError("gauss_order must be positive")
        if (
            self.field_id.variable is not ResultVariable.S
            or position not in _CONTINUUM_STRESS_RECOVERY_POSITIONS
        ):
            raise ValueError(
                "gauss_order is only valid for continuum stress recovery "
                "positions"
            )


@dataclass(frozen=True, slots=True)
class FieldMaterializationKey:
    """Fully resolved numerical identity including its recovery contract."""

    request: FieldRequest
    recovery_contract: int

    def __post_init__(self) -> None:
        if type(self.request) is not FieldRequest:
            raise TypeError("request must be FieldRequest")
        if type(self.recovery_contract) is not int:
            raise TypeError("recovery_contract must be an integer")
        if self.recovery_contract <= 0:
            raise ValueError("recovery_contract must be positive")


@dataclass(frozen=True, slots=True)
class ScalarFieldSelection:
    """One scalar component selected from a complete materialized field."""

    field_key: FieldMaterializationKey
    component: str

    def __post_init__(self) -> None:
        if type(self.field_key) is not FieldMaterializationKey:
            raise TypeError("field_key must be FieldMaterializationKey")
        _require_nonblank_string(self.component, label="component")


@dataclass(frozen=True, slots=True)
class ResultSourceKey:
    """Immutable identity binding a provider to one accepted solve result."""

    result_id: str
    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str
    run_id: str

    def __post_init__(self) -> None:
        _require_nonblank_string(self.result_id, label="result_id")
        _require_nonblank_string(self.session_id, label="session_id")
        _require_nonblank_string(self.artifact_id, label="artifact_id")
        if type(self.model_revision) is not int:
            raise TypeError("model_revision must be an integer")
        if self.model_revision < 0:
            raise ValueError("model_revision must be non-negative")
        _require_nonblank_string(self.step_name, label="step_name")
        _require_nonblank_string(self.run_id, label="run_id")


def field_materialization_sort_key(
    key: FieldMaterializationKey,
) -> tuple[int, int, int, int, float, int, int, int, int]:
    """Return the sole deterministic ordering key for materialized fields."""

    if type(key) is not FieldMaterializationKey:
        raise TypeError("key must be FieldMaterializationKey")
    request = key.request
    policy = request.averaging_policy
    if policy is None:
        policy_key = (0, 0.0, 0)
    else:
        policy_key = (
            1,
            policy.threshold_percent,
            int(policy.preserve_region_boundaries),
        )
    if request.gauss_order is None:
        gauss_key = (0, 0)
    else:
        gauss_key = (1, request.gauss_order)
    return (
        _FIELD_VARIABLE_SORT_ORDER[request.field_id.variable],
        _POSITION_ORDER[request.field_id.position],
        request.field_id.section_point_number or 0,
        *policy_key,
        *gauss_key,
        key.recovery_contract,
    )


def _require_nonblank_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


__all__ = [
    "FieldAssociation",
    "FieldMaterializationKey",
    "FieldPosition",
    "FieldRequest",
    "PhysicalQuantity",
    "ResultFieldId",
    "ResultSourceKey",
    "ResultVariable",
    "ScalarFieldSelection",
    "field_materialization_sort_key",
]
