"""Bounded, read-only queries over completed FEM model results."""

from __future__ import annotations

from collections.abc import Sequence
import math
from operator import attrgetter
from typing import Any

import numpy as np

from fem.post.stress import beam, dispatch, field, invariants

from ..diagnostics import (
    DiagnosticCode,
    exception_diagnostic,
    make_diagnostic,
)
from ..schemas import (
    Diagnostic,
    ResultQuery,
    ResultQueryKind,
    ResultSummary,
    ScalarResult,
    UnitContext,
)


MAX_PROVIDER_SCALARS = 64
_SOURCE = "fem.results"


def query_results(
    result: Any,
    queries: Sequence[ResultQuery],
    *,
    run_id: str,
    unit_context: UnitContext,
    max_scalars: int = MAX_PROVIDER_SCALARS,
) -> ResultSummary:
    """Evaluate bounded result requests and normalize failures per query."""

    if isinstance(max_scalars, bool) or not isinstance(max_scalars, int):
        raise TypeError("max_scalars must be an integer")
    if max_scalars <= 0 or max_scalars > MAX_PROVIDER_SCALARS:
        raise ValueError(
            f"max_scalars must be from 1 through {MAX_PROVIDER_SCALARS}"
        )

    requested = tuple(queries)
    step_name = _step_name(result)
    finite_vectors = _has_finite_vectors(result)
    if not finite_vectors:
        return ResultSummary(
            run_id=run_id,
            step=step_name,
            finite_vectors=False,
            scalars=(),
            diagnostics=(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    "The solved displacement or reaction vector is not finite.",
                    source=_SOURCE,
                    step=step_name,
                    remediation="Inspect the solver diagnostics before querying results.",
                ),
            ),
        )

    estimated_count = sum(
        2
        if isinstance(query, ResultQuery)
        and query.kind == ResultQueryKind.STRESS_EXTREMA
        else 1
        for query in requested
    )
    if estimated_count > max_scalars:
        return ResultSummary(
            run_id=run_id,
            step=step_name,
            finite_vectors=True,
            scalars=(),
            diagnostics=(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    f"Result requests would return {estimated_count} scalars; "
                    f"the configured limit is {max_scalars}.",
                    source=_SOURCE,
                    step=step_name,
                    remediation="Request fewer result quantities.",
                ),
            ),
        )

    scalars: list[ScalarResult] = []
    diagnostics: list[Diagnostic] = []
    for query in requested:
        if not isinstance(query, ResultQuery):
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    "Every result request must be a ResultQuery.",
                    source=_SOURCE,
                    step=step_name,
                )
            )
            continue
        try:
            scalars.extend(
                _evaluate_query(
                    result,
                    query,
                    run_id=run_id,
                    step_name=step_name,
                    units=unit_context,
                )
            )
        except Exception as error:
            diagnostics.append(_query_failure(query, error, step_name))

    return ResultSummary(
        run_id=run_id,
        step=step_name,
        finite_vectors=True,
        scalars=tuple(scalars),
        diagnostics=tuple(diagnostics),
    )


def _evaluate_query(
    result: Any,
    query: ResultQuery,
    *,
    run_id: str,
    step_name: str,
    units: UnitContext,
) -> tuple[ScalarResult, ...]:
    kind = query.kind
    if kind == ResultQueryKind.DISPLACEMENT_COMPONENT:
        value = result.nodal_displacement(query.node_id, query.component)
        return (
            _scalar(
                query,
                value,
                units.length,
                f"component_{query.component}",
                run_id,
                step_name,
                node_id=query.node_id,
            ),
        )
    if kind == ResultQueryKind.DISPLACEMENT_MAGNITUDE:
        value = _nodal_displacement_magnitude(result, query.node_id)
        return (
            _scalar(
                query,
                value,
                units.length,
                "magnitude",
                run_id,
                step_name,
                node_id=query.node_id,
            ),
        )
    if kind == ResultQueryKind.MAX_DISPLACEMENT_COMPONENT:
        node_ids, region = _query_node_region(result.model, query)
        node_id, value = _absolute_component_extreme(
            result,
            node_ids,
            query.component,
        )
        return (
            _scalar(
                query,
                value,
                units.length,
                f"absolute_max_component_{query.component}",
                run_id,
                step_name,
                node_id=node_id,
                region=region,
            ),
        )
    if kind == ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE:
        node_ids, region = _query_node_region(result.model, query)
        node_id, value = max(
            (
                (node_id, _nodal_displacement_magnitude(result, node_id))
                for node_id in node_ids
            ),
            key=lambda item: item[1],
        )
        return (
            _scalar(
                query,
                value,
                units.length,
                "max_magnitude",
                run_id,
                step_name,
                node_id=node_id,
                region=region,
            ),
        )
    if kind == ResultQueryKind.REACTION_COMPONENT:
        value = result.nodal_reaction(query.node_id, query.component)
        return (
            _scalar(
                query,
                value,
                units.force,
                f"component_{query.component}",
                run_id,
                step_name,
                node_id=query.node_id,
            ),
        )
    if kind == ResultQueryKind.REACTION_SUM:
        node_ids, region = _query_node_region(result.model, query)
        value = math.fsum(
            result.nodal_reaction(node_id, query.component)
            for node_id in node_ids
        )
        return (
            _scalar(
                query,
                value,
                units.force,
                f"sum_component_{query.component}",
                run_id,
                step_name,
                region=region,
            ),
        )
    if kind == ResultQueryKind.STRESS_EXTREMA:
        return _stress_extrema(
            result,
            query,
            run_id=run_id,
            step_name=step_name,
            stress_unit=units.stress,
        )
    raise ValueError(f"Unsupported result query kind {kind.value!r}")


def _scalar(
    query: ResultQuery,
    value: float,
    unit: str,
    measure: str,
    run_id: str,
    step_name: str,
    *,
    node_id: int | None = None,
    element_id: int | None = None,
    region: str | None = None,
) -> ScalarResult:
    return ScalarResult(
        query_kind=query.kind,
        value=float(value),
        unit=unit,
        measure=measure,
        run_id=run_id,
        step=step_name,
        node_id=node_id,
        element_id=element_id,
        region=region,
    )


def _query_node_region(
    model: Any,
    query: ResultQuery,
) -> tuple[tuple[int, ...], str]:
    if query.node_set is not None:
        node_set = getattr(model, "node_sets", {}).get(query.node_set)
        if node_set is None:
            if query.node_set in getattr(model, "edges", {}):
                raise ValueError(
                    f"{query.node_set!r} is an edge; use "
                    f"edge={query.node_set!r}"
                )
            if query.node_set in getattr(model, "surfaces", {}):
                raise ValueError(
                    f"{query.node_set!r} is a surface; use "
                    f"surface={query.node_set!r}"
                )
            raise KeyError(f"node set {query.node_set!r} is not defined")
        node_ids = tuple(int(node_id) for node_id in node_set.node_ids)
        region = query.node_set
    elif query.edge is not None:
        edge = getattr(model, "edges", {}).get(query.edge)
        if edge is None:
            if query.edge in getattr(model, "surfaces", {}):
                raise ValueError(
                    f"{query.edge!r} is a surface; use "
                    f"surface={query.edge!r}"
                )
            if query.edge in getattr(model, "node_sets", {}):
                raise ValueError(
                    f"{query.edge!r} is a node set; use "
                    f"node_set={query.edge!r}"
                )
            raise KeyError(f"edge {query.edge!r} is not defined")
        node_ids = _topology_node_ids(edge.edges)
        region = query.edge
    elif query.surface is not None:
        surface = getattr(model, "surfaces", {}).get(query.surface)
        if surface is None:
            if query.surface in getattr(model, "edges", {}):
                raise ValueError(
                    f"{query.surface!r} is an edge; use "
                    f"edge={query.surface!r}"
                )
            if query.surface in getattr(model, "node_sets", {}):
                raise ValueError(
                    f"{query.surface!r} is a node set; use "
                    f"node_set={query.surface!r}"
                )
            raise KeyError(f"surface {query.surface!r} is not defined")
        node_ids = _topology_node_ids(surface.faces)
        region = query.surface
    else:
        node_ids = tuple(int(node_id) for node_id in model.mesh.node_ids)
        region = "all_nodes"
    if not node_ids:
        raise ValueError("the selected node region is empty")
    return node_ids, region


def _topology_node_ids(entries: Sequence[Any]) -> tuple[int, ...]:
    node_ids: list[int] = []
    seen: set[int] = set()
    for entry in entries:
        for node_id in getattr(entry, "node_ids", ()):
            normalized = int(node_id)
            if normalized not in seen:
                seen.add(normalized)
                node_ids.append(normalized)
    return tuple(node_ids)


def _query_element_ids(model: Any, element_set_name: str | None) -> set[int] | None:
    if element_set_name is None:
        return None
    element_set = getattr(model, "element_sets", {}).get(element_set_name)
    if element_set is None:
        raise KeyError(f"element set {element_set_name!r} is not defined")
    element_ids = {int(element_id) for element_id in element_set.element_ids}
    if not element_ids:
        raise ValueError("the selected element region is empty")
    return element_ids


def _nodal_displacement_magnitude(result: Any, node_id: int) -> float:
    mesh = result.model.mesh
    spatial_components = 3 if hasattr(mesh.nodes[0], "z") else 2
    component_count = min(int(mesh.dofs_per_node), spatial_components)
    values = (
        result.nodal_displacement(node_id, component)
        for component in range(1, component_count + 1)
    )
    return math.sqrt(math.fsum(value * value for value in values))


def _absolute_component_extreme(
    result: Any,
    node_ids: tuple[int, ...],
    component: int,
) -> tuple[int, float]:
    return max(
        (
            (node_id, result.nodal_displacement(node_id, component))
            for node_id in node_ids
        ),
        key=lambda item: abs(item[1]),
    )


def _stress_extrema(
    result: Any,
    query: ResultQuery,
    *,
    run_id: str,
    step_name: str,
    stress_unit: str,
) -> tuple[ScalarResult, ScalarResult]:
    element_ids = _query_element_ids(result.model, query.element_set)
    type_keys = dispatch.resolve_type_keys(result.model.mesh, None)
    if type_keys == ("beam2",):
        if element_ids is not None:
            raise ValueError("Beam2 stress queries do not support element_set filtering")
        values = _beam_stress_values(result, query.measure)
    else:
        values = _continuum_stress_values(result, query.measure, element_ids)
    if not values:
        raise ValueError("no stress values are available for the selected region")

    minimum = min(values, key=lambda item: item[0])
    maximum = max(values, key=lambda item: item[0])
    measure = _canonical_stress_measure(query.measure)
    region = query.element_set or "all_elements"
    return (
        _scalar(
            query,
            minimum[0],
            stress_unit,
            f"{measure}_minimum",
            run_id,
            step_name,
            node_id=minimum[1],
            element_id=minimum[2],
            region=region,
        ),
        _scalar(
            query,
            maximum[0],
            stress_unit,
            f"{measure}_maximum",
            run_id,
            step_name,
            node_id=maximum[1],
            element_id=maximum[2],
            region=region,
        ),
    )


def _continuum_stress_values(
    result: Any,
    measure: str | None,
    element_ids: set[int] | None,
) -> list[tuple[float, int, int]]:
    stress_field = field.collect(result.model.mesh, result.U)
    canonical_measure = _canonical_stress_measure(measure)
    component_index = (
        stress_field.component_names.index(canonical_measure)
        if canonical_measure in stress_field.component_names
        else None
    )
    if canonical_measure != "von_mises" and component_index is None:
        available = ", ".join(("von_mises", *stress_field.component_names))
        raise ValueError(
            f"stress measure {measure!r} is unsupported; available measures: {available}"
        )

    values: list[tuple[float, int, int]] = []
    for node_id in stress_field.node_ids:
        for contribution in stress_field.contributions_by_node.get(node_id, ()):
            if element_ids is not None and contribution.elem_id not in element_ids:
                continue
            if canonical_measure == "von_mises":
                if len(contribution.components) == 3:
                    value = invariants.von_mises_plane(
                        *contribution.components,
                        plane_type=contribution.plane_type or "stress",
                        nu=contribution.poisson_ratio or 0.0,
                    )
                elif len(contribution.components) == 6:
                    value = invariants.von_mises_3d(*contribution.components)
                else:
                    raise ValueError(
                        "von Mises stress is unavailable for this element family"
                    )
            else:
                value = contribution.components[component_index]
            if not math.isfinite(float(value)):
                raise ValueError("stress recovery produced a non-finite value")
            values.append(
                (float(value), int(contribution.node_id), int(contribution.elem_id))
            )
    return values


def _beam_stress_values(
    result: Any,
    measure: str | None,
) -> list[tuple[float, int, int | None]]:
    canonical = _canonical_stress_measure(measure)
    rows = beam.nodal_envelope(result)
    if canonical in {"von_mises", "axial_stress_abs_max"}:
        selector = attrgetter("absolute_maximum")
    elif canonical == "axial_stress_max":
        selector = attrgetter("maximum")
    elif canonical == "axial_stress_min":
        selector = attrgetter("minimum")
    else:
        raise ValueError(
            f"stress measure {measure!r} is unsupported for Beam2; available "
            "measures: von_mises, axial_stress_max, axial_stress_min, "
            "axial_stress_abs_max"
        )
    return [
        (float(selector(row)), int(row.node_id), None)
        for row in rows
    ]


def _canonical_stress_measure(measure: str | None) -> str:
    normalized = str(measure or "von_mises").strip().casefold().replace(" ", "_")
    aliases = {
        "mises": "von_mises",
        "vonmises": "von_mises",
        "von_mises_stress": "von_mises",
    }
    return aliases.get(normalized, normalized)


def _has_finite_vectors(result: Any) -> bool:
    try:
        displacement = np.asarray(result.U, dtype=float)
        reactions = np.asarray(result.reactions, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        displacement.ndim == 1
        and reactions.ndim == 1
        and np.all(np.isfinite(displacement))
        and np.all(np.isfinite(reactions))
    )


def _step_name(result: Any) -> str:
    name = str(getattr(getattr(result, "step", None), "name", "")).strip()
    return name or "step"


def _query_failure(
    query: ResultQuery,
    error: Exception,
    step_name: str,
) -> Diagnostic:
    message = exception_diagnostic(
        DiagnosticCode.RESULT_QUERY_FAILED,
        error,
        source=_SOURCE,
    ).message
    return make_diagnostic(
        DiagnosticCode.RESULT_QUERY_FAILED,
        f"{query.kind.value} failed: {message}",
        source=_SOURCE,
        step=step_name,
        remediation=_query_failure_remediation(query),
    )


def _query_failure_remediation(query: ResultQuery) -> str:
    if query.kind in {
        ResultQueryKind.DISPLACEMENT_COMPONENT,
        ResultQueryKind.DISPLACEMENT_MAGNITUDE,
        ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
    }:
        return (
            "Check the requested node or nodal region "
            "(node_set, edge, or surface) and displacement component."
        )
    if query.kind in {
        ResultQueryKind.REACTION_COMPONENT,
        ResultQueryKind.REACTION_SUM,
    }:
        return (
            "Check the requested node or nodal region "
            "(node_set, edge, or surface) and reaction component."
        )
    return (
        "Check the requested element_set, stress measure, and available "
        "stress output."
    )


summarize_results = query_results


__all__ = [
    "MAX_PROVIDER_SCALARS",
    "query_results",
    "summarize_results",
]
