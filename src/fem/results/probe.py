"""Exact result-location probes built on the typed result-query contract.

The probe layer deliberately does not know anything about a viewport.  It
turns one human-facing location (node, element, integration point, or local
element node) into the smallest existing :class:`ResultQuery`, then filters
the returned records without averaging or reordering them.  This keeps a
probe useful for diagnosing the actual stored result locations rather than
inventing a second post-processing data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .data import FieldLocation, ResultMaterializationSnapshot
from .fields import ResultSourceKey, ScalarFieldSelection
from .frames import ResultFrameKey
from .query import (
    ResultQuery,
    ResultQueryRecord,
    ResultQueryResult,
    evaluate_result_query,
)


class ResultProbeValidationError(ValueError):
    """Typed validation failure for one exact result probe."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code.strip():
            raise ValueError("code must be a non-blank string")
        self.code = code
        super().__init__(message)


class ResultProbeKind(str, Enum):
    """FEM identity selected by a graphical result probe."""

    NODE = "node"
    ELEMENT = "element"
    INTEGRATION_POINT = "integration_point"
    ELEMENT_NODE = "element_node"


@dataclass(frozen=True, slots=True)
class ResultProbeTarget:
    """One exact FEM location, with no implicit averaging semantics."""

    kind: ResultProbeKind
    node_id: int | None = None
    element_id: int | None = None
    integration_point: int | None = None
    local_node: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ResultProbeKind:
            raise TypeError("kind must be ResultProbeKind")

        required: tuple[str, ...]
        if self.kind is ResultProbeKind.NODE:
            required = ("node_id",)
        elif self.kind is ResultProbeKind.ELEMENT:
            required = ("element_id",)
        elif self.kind is ResultProbeKind.INTEGRATION_POINT:
            required = ("element_id", "integration_point")
        else:
            required = ("element_id", "local_node")

        values = {
            "node_id": self.node_id,
            "element_id": self.element_id,
            "integration_point": self.integration_point,
            "local_node": self.local_node,
        }
        for name, value in values.items():
            if name in required:
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        f"{name} must be a positive integer for "
                        f"{self.kind.value} probes"
                    )
            elif value is not None:
                raise ValueError(
                    f"{name} is not valid for {self.kind.value} probes"
                )


@dataclass(frozen=True, slots=True)
class ResultProbeRequest:
    """One scalar field selection and one exact probe target."""

    selection: ScalarFieldSelection
    target: ResultProbeTarget

    def __post_init__(self) -> None:
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        if type(self.target) is not ResultProbeTarget:
            raise TypeError("target must be ResultProbeTarget")


@dataclass(frozen=True, slots=True)
class ResultProbeResult:
    """Probe records bound to one materialization generation and frame."""

    source: ResultSourceKey
    materialization_generation: int
    frame_key: ResultFrameKey | None
    request: ResultProbeRequest
    records: tuple[ResultQueryRecord, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError(
                "materialization_generation must be non-negative"
            )
        if self.frame_key is not None:
            if type(self.frame_key) is not ResultFrameKey:
                raise TypeError("frame_key must be ResultFrameKey or None")
            if self.frame_key.source != self.source:
                raise ValueError("frame key source must match probe source")
        if type(self.request) is not ResultProbeRequest:
            raise TypeError("request must be ResultProbeRequest")
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        for record in self.records:
            if type(record) is not ResultQueryRecord:
                raise TypeError(
                    "records must contain only ResultQueryRecord values"
                )
            if record.source != self.source:
                raise ValueError(
                    "every probe record source must match the probe source"
                )


def result_query_for_probe(request: ResultProbeRequest) -> ResultQuery:
    """Translate a probe request to the existing exact result-query API."""

    if type(request) is not ResultProbeRequest:
        raise TypeError("request must be ResultProbeRequest")
    target = request.target
    if target.kind is ResultProbeKind.NODE:
        return ResultQuery(
            field_key=request.selection.field_key,
            component=request.selection.component,
            node_ids=(target.node_id,),
        )
    return ResultQuery(
        field_key=request.selection.field_key,
        component=request.selection.component,
        element_ids=(target.element_id,),
    )


def probe_result(
    materialization: ResultMaterializationSnapshot,
    request: ResultProbeRequest,
) -> ResultProbeResult:
    """Evaluate one probe directly against an accepted materialization."""

    if type(materialization) is not ResultMaterializationSnapshot:
        raise TypeError(
            "materialization must be ResultMaterializationSnapshot"
        )
    query_result = evaluate_result_query(
        materialization,
        result_query_for_probe(request),
    )
    return probe_result_from_query_result(
        query_result,
        request,
        frame_key=materialization.frame_key,
    )


def probe_result_from_query_result(
    query_result: ResultQueryResult,
    request: ResultProbeRequest,
    *,
    frame_key: ResultFrameKey | None = None,
) -> ResultProbeResult:
    """Filter a completed query result into the requested probe location."""

    if type(query_result) is not ResultQueryResult:
        raise TypeError("query_result must be ResultQueryResult")
    if type(request) is not ResultProbeRequest:
        raise TypeError("request must be ResultProbeRequest")
    expected_query = result_query_for_probe(request)
    if query_result.query != expected_query:
        raise ResultProbeValidationError(
            "result.probe.query_mismatch",
            "query result does not match the probe request",
        )
    if frame_key is not None and type(frame_key) is not ResultFrameKey:
        raise TypeError("frame_key must be ResultFrameKey or None")
    if frame_key is not None and frame_key.source != query_result.source:
        raise ValueError("frame key source must match query result source")

    records = tuple(
        record
        for record in query_result.records
        if _matches_probe_target(record.location, request.target)
    )
    return ResultProbeResult(
        source=query_result.source,
        materialization_generation=query_result.materialization_generation,
        frame_key=frame_key,
        request=request,
        records=records,
    )


def _matches_probe_target(
    location: FieldLocation,
    target: ResultProbeTarget,
) -> bool:
    if type(location) is not FieldLocation:
        raise TypeError("location must be FieldLocation")
    if type(target) is not ResultProbeTarget:
        raise TypeError("target must be ResultProbeTarget")
    if target.kind is ResultProbeKind.NODE:
        return location.node_id == target.node_id
    if target.kind is ResultProbeKind.ELEMENT:
        return location.element_id == target.element_id
    if target.kind is ResultProbeKind.INTEGRATION_POINT:
        return (
            location.element_id == target.element_id
            and location.integration_point == target.integration_point
        )
    return (
        location.element_id == target.element_id
        and location.local_node == target.local_node
    )


__all__ = [
    "ResultProbeKind",
    "ResultProbeRequest",
    "ResultProbeResult",
    "ResultProbeTarget",
    "ResultProbeValidationError",
    "probe_result",
    "probe_result_from_query_result",
    "result_query_for_probe",
]
