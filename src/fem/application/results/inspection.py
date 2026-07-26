"""Typed result inspection built only from catalog and query contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .data import (
    FieldAvailability,
    FieldState,
    ResultCatalog,
    ResultMaterializationSnapshot,
)
from .fields import FieldAssociation, ResultSourceKey
from .query import (
    ResultQuery,
    ResultQueryResult,
    ResultQueryValidationError,
    evaluate_result_query,
)


@dataclass(frozen=True, slots=True)
class NodeResultInspectionRequest:
    """Inspect every result field relevant to one FEM node."""

    node_id: int

    def __post_init__(self) -> None:
        _require_positive_id(self.node_id, label="node_id")


@dataclass(frozen=True, slots=True)
class ElementResultInspectionRequest:
    """Inspect every result field relevant to one FEM element."""

    element_id: int

    def __post_init__(self) -> None:
        _require_positive_id(self.element_id, label="element_id")


ResultInspectionRequest: TypeAlias = (
    NodeResultInspectionRequest | ElementResultInspectionRequest
)


@dataclass(frozen=True, slots=True)
class ResultInspectionField:
    """One catalog entry and its ordered component queries when ready."""

    availability: FieldAvailability
    component_results: tuple[ResultQueryResult, ...]

    def __post_init__(self) -> None:
        if type(self.availability) is not FieldAvailability:
            raise TypeError("availability must be FieldAvailability")
        if type(self.component_results) is not tuple:
            raise TypeError("component_results must be a tuple")
        if any(
            type(result) is not ResultQueryResult
            for result in self.component_results
        ):
            raise TypeError(
                "component_results must contain ResultQueryResult values"
            )

        availability = self.availability
        if availability.state is not FieldState.READY:
            if self.component_results:
                raise ValueError(
                    "non-ready inspection fields cannot contain query results"
                )
            return
        components = tuple(
            result.query.component for result in self.component_results
        )
        if components != availability.descriptor.columns:
            raise ValueError(
                "ready inspection component results must follow descriptor "
                "column order"
            )
        if any(
            result.query.field_key != availability.key
            for result in self.component_results
        ):
            raise ValueError(
                "inspection query field keys must match availability"
            )


@dataclass(frozen=True, slots=True)
class ResultInspectionResult:
    """Catalog-ordered inspection bound to one materialization generation."""

    source: ResultSourceKey
    materialization_generation: int
    request: ResultInspectionRequest
    fields: tuple[ResultInspectionField, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError(
                "materialization_generation must be non-negative"
            )
        if type(self.request) not in {
            NodeResultInspectionRequest,
            ElementResultInspectionRequest,
        }:
            raise TypeError(
                "request must be a typed result inspection request"
            )
        if type(self.fields) is not tuple:
            raise TypeError("fields must be a tuple")
        associations = (
            _NODE_ASSOCIATIONS
            if type(self.request) is NodeResultInspectionRequest
            else _ELEMENT_ASSOCIATIONS
        )
        for field_entry in self.fields:
            if type(field_entry) is not ResultInspectionField:
                raise TypeError(
                    "fields must contain ResultInspectionField values"
                )
            if (
                field_entry.availability.descriptor.association
                not in associations
            ):
                raise ValueError(
                    "inspection field association is not relevant to request"
                )
            for query_result in field_entry.component_results:
                if query_result.source != self.source:
                    raise ValueError(
                        "inspection query source must match result source"
                    )
                if (
                    query_result.materialization_generation
                    != self.materialization_generation
                ):
                    raise ValueError(
                        "inspection query generation must match result "
                        "generation"
                    )
                _validate_query_filters(query_result, self.request)


def inspect_result_snapshot(
    materialization: ResultMaterializationSnapshot,
    catalog: ResultCatalog,
    request: ResultInspectionRequest,
) -> ResultInspectionResult:
    """Inspect relevant fields without materializing or applying GUI policy."""

    if type(materialization) is not ResultMaterializationSnapshot:
        raise TypeError(
            "materialization must be ResultMaterializationSnapshot"
        )
    if type(catalog) is not ResultCatalog:
        raise TypeError("catalog must be ResultCatalog")
    if type(request) not in {
        NodeResultInspectionRequest,
        ElementResultInspectionRequest,
    }:
        raise TypeError("request must be a typed result inspection request")
    if catalog.source != materialization.source:
        raise ValueError(
            "inspection catalog source must match materialization source"
        )

    _validate_entity_exists(materialization, request)
    associations = (
        _NODE_ASSOCIATIONS
        if type(request) is NodeResultInspectionRequest
        else _ELEMENT_ASSOCIATIONS
    )
    fields = tuple(
        _inspect_availability(materialization, availability, request)
        for availability in catalog.fields
        if availability.descriptor.association in associations
    )
    return ResultInspectionResult(
        source=materialization.source,
        materialization_generation=materialization.generation,
        request=request,
        fields=fields,
    )


_NODE_ASSOCIATIONS = frozenset(
    {
        FieldAssociation.NODE,
        FieldAssociation.NODE_REGION,
        FieldAssociation.RESOLVED_NODAL,
        FieldAssociation.ELEMENT_NODE,
    }
)
_ELEMENT_ASSOCIATIONS = frozenset(
    {
        FieldAssociation.ELEMENT,
        FieldAssociation.INTEGRATION_POINT,
        FieldAssociation.ELEMENT_NODE,
    }
)


def _inspect_availability(
    materialization: ResultMaterializationSnapshot,
    availability: FieldAvailability,
    request: ResultInspectionRequest,
) -> ResultInspectionField:
    if availability.state is not FieldState.READY:
        return ResultInspectionField(
            availability=availability,
            component_results=(),
        )
    filters = (
        {"node_ids": (request.node_id,)}
        if type(request) is NodeResultInspectionRequest
        else {"element_ids": (request.element_id,)}
    )
    results = tuple(
        evaluate_result_query(
            materialization,
            ResultQuery(
                field_key=availability.key,
                component=component,
                **filters,
            ),
        )
        for component in availability.descriptor.columns
    )
    return ResultInspectionField(
        availability=availability,
        component_results=results,
    )


def _validate_entity_exists(
    materialization: ResultMaterializationSnapshot,
    request: ResultInspectionRequest,
) -> None:
    topology = materialization.topology
    if type(request) is NodeResultInspectionRequest:
        if request.node_id not in frozenset(topology.node_ids):
            raise ResultQueryValidationError(
                "result.query.unknown_node_ids",
                f"inspection contains unknown node ID: {request.node_id}",
            )
        return
    if request.element_id not in frozenset(topology.element_ids):
        raise ResultQueryValidationError(
            "result.query.unknown_element_ids",
            f"inspection contains unknown element ID: {request.element_id}",
        )


def _validate_query_filters(
    result: ResultQueryResult,
    request: ResultInspectionRequest,
) -> None:
    query = result.query
    if type(request) is NodeResultInspectionRequest:
        valid = (
            query.node_ids == (request.node_id,)
            and not query.element_ids
            and not query.region_keys
        )
    else:
        valid = (
            query.element_ids == (request.element_id,)
            and not query.node_ids
            and not query.region_keys
        )
    if not valid:
        raise ValueError(
            "inspection query filters must exactly match the request"
        )


def _require_positive_id(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


__all__ = [
    "ElementResultInspectionRequest",
    "NodeResultInspectionRequest",
    "ResultInspectionField",
    "ResultInspectionRequest",
    "ResultInspectionResult",
    "inspect_result_snapshot",
]
