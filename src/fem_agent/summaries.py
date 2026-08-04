"""Deterministic, bounded summaries of locally imported FEM models."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

from fem.core import model_element_info

from .diagnostics import DiagnosticCode, make_diagnostic
from .schemas import (
    AnalysisSummary,
    Diagnostic,
    ImportAnalysisSpec,
    ResultQueryKind,
)
from .tools.importing import AbaqusImportResult


DEFAULT_MAX_COLLECTION_ITEMS = 32


def build_analysis_summary(
    import_result: AbaqusImportResult,
    spec: ImportAnalysisSpec,
    revision_hash: str,
    *,
    max_collection_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
) -> AnalysisSummary:
    """Build the provider-safe confirmation summary for one import revision."""

    if (
        isinstance(max_collection_items, bool)
        or not isinstance(max_collection_items, int)
        or max_collection_items <= 0
    ):
        raise ValueError("max_collection_items must be a positive integer")
    max_items = int(max_collection_items)
    diagnostics = list(import_result.diagnostics)
    model = import_result.model
    truncated = import_result.collections_truncated

    if model is None:
        model_name = "unavailable_model"
        node_count = int(import_result.keyword_inspection.node_record_count)
        element_count = int(import_result.keyword_inspection.element_record_count)
        total_dofs = int(import_result.keyword_inspection.estimated_dofs)
        dofs_per_node = (
            total_dofs // node_count if node_count else 0
        )
        element_types = {}
        node_sets: tuple[dict[str, Any], ...] = ()
        element_sets: tuple[dict[str, Any], ...] = ()
        edges: tuple[dict[str, Any], ...] = ()
        surfaces: tuple[dict[str, Any], ...] = ()
        materials: tuple[dict[str, Any], ...] = ()
        sections: tuple[dict[str, Any], ...] = ()
        analysis_step = None
        constraints: tuple[dict[str, Any], ...] = ()
        loads: tuple[dict[str, Any], ...] = ()
    else:
        model_name = _bounded_text(model.name or "abaqus_model")
        node_count = int(model.mesh.num_nodes)
        element_count = int(model.mesh.num_elements)
        dofs_per_node = int(model.mesh.dofs_per_node)
        total_dofs = int(model.mesh.num_dofs)
        element_types = dict(
            sorted(Counter(str(elem.type) for elem in model.mesh.elements).items())
        )

        node_sets, was_truncated = _bounded_items(
            (
                {"name": _bounded_text(name), "size": len(node_set.node_ids)}
                for name, node_set in _sorted_mapping_items(model.node_sets)
            ),
            max_items,
        )
        truncated |= was_truncated
        element_sets, was_truncated = _bounded_items(
            (
                {
                    "name": _bounded_text(name),
                    "size": len(element_set.element_ids),
                }
                for name, element_set in _sorted_mapping_items(model.element_sets)
            ),
            max_items,
        )
        truncated |= was_truncated
        edges, was_truncated = _bounded_items(
            (
                {"name": _bounded_text(name), "size": len(edge.edges)}
                for name, edge in _sorted_mapping_items(model.edges)
            ),
            max_items,
        )
        truncated |= was_truncated
        surfaces, was_truncated = _bounded_items(
            (
                {"name": _bounded_text(name), "size": len(surface.faces)}
                for name, surface in _sorted_mapping_items(model.surfaces)
            ),
            max_items,
        )
        truncated |= was_truncated
        materials, was_truncated = _bounded_items(
            (
                _material_summary(name, material, diagnostics)
                for name, material in _sorted_mapping_items(model.materials)
            ),
            max_items,
        )
        truncated |= was_truncated
        sections, was_truncated = _bounded_items(
            (_section_summary(section) for section in model.sections),
            max_items,
        )
        truncated |= was_truncated

        selected_step = import_result.runnable_step
        analysis_step, step_truncated = _step_summary(selected_step, max_items)
        truncated |= step_truncated
        constraints, was_truncated = _constraint_summaries(
            model,
            selected_step,
            max_items,
            diagnostics,
        )
        truncated |= was_truncated
        loads, was_truncated = _load_summaries(
            model,
            selected_step,
            max_items,
            diagnostics,
        )
        truncated |= was_truncated

    _append_spec_diagnostics(
        diagnostics,
        import_result,
        spec,
    )
    resource_class = _resource_class(
        node_count,
        element_count,
        total_dofs,
        import_result.input_size_bytes,
        spec,
        diagnostics,
    )

    return AnalysisSummary(
        revision=spec.revision,
        revision_hash=revision_hash,
        source_artifact_id=spec.source_artifact_id,
        source_sha256=spec.source_sha256,
        model_name=model_name,
        node_count=node_count,
        element_count=element_count,
        dofs_per_node=dofs_per_node,
        total_dofs=total_dofs,
        element_types=element_types,
        node_sets=node_sets,
        element_sets=element_sets,
        edges=edges,
        surfaces=surfaces,
        materials=materials,
        sections=sections,
        analysis_step=analysis_step,
        constraints=constraints,
        loads=loads,
        unit_context=spec.unit_context,
        requested_queries=spec.requested_queries,
        export_formats=spec.export_formats,
        keyword_inventory=import_result.keyword_inventory,
        diagnostics=_deduplicate_diagnostics(diagnostics),
        resource_class=resource_class,
        collections_truncated=bool(truncated),
    )


def _material_summary(
    name: object,
    material: Any,
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    properties: dict[str, float | None] = {}
    raw_properties = dict(getattr(material, "properties", {}))
    for property_name in ("E", "nu", "rho"):
        if property_name not in raw_properties:
            continue
        properties[property_name] = _safe_number(
            raw_properties[property_name],
            f"material:{property_name}",
            diagnostics,
        )
    return {
        "name": _bounded_text(name),
        "properties": properties,
    }


def _section_summary(section: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "element_set": _bounded_text(section.element_set),
        "material": _bounded_text(section.material),
        "section_type": _bounded_text(section.section_type),
    }
    thickness = getattr(section, "properties", {}).get("thickness")
    if thickness is not None:
        summary["thickness"] = float(thickness)
    return summary


def _step_summary(
    step: Any | None,
    max_items: int,
) -> tuple[dict[str, Any] | None, bool]:
    if step is None:
        return None, False
    outputs, truncated = _bounded_items(
        (
            {
                "kind": _bounded_text(output.kind),
                "target": _bounded_text(output.target),
                "variable_count": len(output.variables),
            }
            for output in step.outputs
        ),
        max_items,
    )
    return (
        {
            "name": _bounded_text(step.name),
            "procedure": _bounded_text(step.procedure),
            "nlgeom": _truthy_option(step.metadata.get("nlgeom")),
            "constraint_count": len(step.boundaries),
            "nodal_load_count": len(step.cloads),
            "surface_load_count": len(step.surface_loads),
            "edge_load_count": len(step.edge_loads),
            "line_load_count": len(step.line_loads),
            "gravity_load_count": len(step.gravity_loads),
            "output_requests": outputs,
        },
        truncated,
    )


def _constraint_summaries(
    model: Any,
    step: Any | None,
    max_items: int,
    diagnostics: list[Diagnostic],
) -> tuple[tuple[dict[str, Any], ...], bool]:
    if step is None:
        return (), False
    initial = next(
        (
            candidate
            for candidate in model.steps
            if str(candidate.name).strip().casefold() == "initial"
        ),
        None,
    )

    def rows() -> Iterable[dict[str, Any]]:
        if initial is not None and initial is not step:
            for constraint in initial.boundaries:
                yield _constraint_summary(constraint, "initial", diagnostics)
        for constraint in step.boundaries:
            yield _constraint_summary(constraint, "step", diagnostics)

    return _bounded_items(rows(), max_items)


def _constraint_summary(
    constraint: Any,
    scope: str,
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    return {
        "target": _safe_target(constraint.target),
        "first_component": int(constraint.first_component),
        "last_component": int(constraint.last_component),
        "value": _safe_number(
            constraint.value,
            "constraint:value",
            diagnostics,
        ),
        "scope": scope,
    }


def _load_summaries(
    model: Any,
    step: Any | None,
    max_items: int,
    diagnostics: list[Diagnostic],
) -> tuple[tuple[dict[str, Any], ...], bool]:
    if step is None:
        return (), False

    def rows() -> Iterable[dict[str, Any]]:
        for load in step.cloads:
            yield {
                "type": "nodal",
                "target": _safe_target(load.target),
                "component": int(load.component),
                "value": _safe_number(load.value, "cload:value", diagnostics),
            }
        for load in step.surface_loads:
            yield _boundary_load_summary(
                "surface",
                load.surface,
                load,
                diagnostics,
            )
        for load in step.edge_loads:
            yield _boundary_load_summary(
                "edge",
                load.edge,
                load,
                diagnostics,
            )
        for load in step.line_loads:
            yield {
                "type": "line",
                "target": _safe_target(load.target),
                "vector": _safe_vector(
                    load.vector,
                    "line-load:vector",
                    diagnostics,
                ),
                "coordinate_system": _bounded_text(load.coordinate_system),
            }
        for load in step.gravity_loads:
            yield {
                "type": "gravity",
                "target": (
                    "all-density-bearing-elements"
                    if load.target is None
                    else _safe_target(load.target)
                ),
                "acceleration": _safe_vector(
                    load.acceleration,
                    "gravity:acceleration",
                    diagnostics,
                ),
                "density": _gravity_density_summary(model, load),
            }

    return _bounded_items(rows(), max_items)


def _gravity_density_summary(model: Any, load: Any) -> dict[str, Any]:
    element_ids = _gravity_target_element_ids(model, load.target)
    valid_count = 0
    invalid_count = 0
    for element_id in element_ids:
        try:
            density = model_element_info(model, element_id).properties.get("rho")
            density_value = float(density)
        except (KeyError, TypeError, ValueError):
            density = None
            density_value = math.nan
        if density is None:
            continue
        if math.isfinite(density_value) and density_value >= 0.0:
            valid_count += 1
        else:
            invalid_count += 1

    total = len(element_ids)
    if invalid_count:
        status = "invalid"
    elif valid_count == total and total:
        status = "complete"
    elif valid_count:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "target_element_count": total,
        "elements_with_valid_density": valid_count,
    }


def _gravity_target_element_ids(model: Any, target: Any) -> tuple[int, ...]:
    if target is None:
        return tuple(int(element.id) for element in model.mesh.elements)
    if isinstance(target, str):
        element_sets = dict(model.element_sets)
        element_sets.update(model.metadata.get("_abaqus_internal_element_sets", {}))
        element_set = element_sets.get(target)
        if element_set is None:
            return ()
        return tuple(int(element_id) for element_id in element_set.element_ids)
    try:
        return (int(target),)
    except (TypeError, ValueError):
        return ()


def _boundary_load_summary(
    location: str,
    target: object,
    load: Any,
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    return {
        "type": _bounded_text(load.load_type),
        "location": location,
        "target": _bounded_text(target),
        "vector": _safe_vector(
            load.vector,
            f"{location}-load:vector",
            diagnostics,
        ),
        "magnitude": (
            None
            if load.magnitude is None
            else _safe_number(
                load.magnitude,
                f"{location}-load:magnitude",
                diagnostics,
            )
        ),
        "coordinate_system": "global",
    }


def _append_spec_diagnostics(
    diagnostics: list[Diagnostic],
    import_result: AbaqusImportResult,
    spec: ImportAnalysisSpec,
) -> None:
    if spec.unit_context is None:
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.UNIT_CONTEXT_REQUIRED,
                "A declared unit context is required before confirmation.",
                source="analysis_summary",
            )
        )
    if spec.requested_queries and import_result.model is not None:
        _append_result_query_diagnostics(
            diagnostics,
            import_result.model,
            spec,
        )

    runnable = import_result.runnable_step
    if runnable is not None:
        if spec.analysis_step is None:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.INVALID_INPUT,
                    "The analysis specification does not select the imported runnable step.",
                    source="analysis_summary",
                    entity="analysis_step",
                )
            )
        elif spec.analysis_step != runnable.name:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.INVALID_INPUT,
                    "The selected analysis step does not match the imported runnable step.",
                    source="analysis_summary",
                    entity="analysis_step",
                )
            )


def _append_result_query_diagnostics(
    diagnostics: list[Diagnostic],
    model: Any,
    spec: ImportAnalysisSpec,
) -> None:
    node_query_kinds = {
        ResultQueryKind.DISPLACEMENT_COMPONENT,
        ResultQueryKind.DISPLACEMENT_MAGNITUDE,
        ResultQueryKind.REACTION_COMPONENT,
    }
    component_query_kinds = {
        ResultQueryKind.DISPLACEMENT_COMPONENT,
        ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
        ResultQueryKind.REACTION_COMPONENT,
        ResultQueryKind.REACTION_SUM,
    }
    aggregate_node_kinds = {
        ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
        ResultQueryKind.REACTION_SUM,
    }
    mesh_node_ids = {
        int(node_id) for node_id in getattr(model.mesh, "node_ids", ())
    }
    mesh_element_ids = {
        int(element.id) for element in getattr(model.mesh, "elements", ())
    }
    dofs_per_node = int(getattr(model.mesh, "dofs_per_node", 0))

    for index, query in enumerate(spec.requested_queries):
        prefix = f"requested_queries[{index}]"
        if (
            query.kind in node_query_kinds
            and query.node_id not in mesh_node_ids
        ):
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    (
                        f"The {query.kind.value} query references node_id "
                        f"{query.node_id}, which is not defined in the model."
                    ),
                    source="analysis_summary",
                    entity=f"{prefix}.node_id",
                    remediation=(
                        "Choose a node_id that is present in the imported model."
                    ),
                )
            )
        if (
            query.kind in component_query_kinds
            and (
                query.component is None
                or query.component > dofs_per_node
            )
        ):
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    (
                        f"The {query.kind.value} query requests component "
                        f"{query.component}; this model has {dofs_per_node} "
                        "degrees of freedom per node."
                    ),
                    source="analysis_summary",
                    entity=f"{prefix}.component",
                    remediation=(
                        "Choose a component from 1 through "
                        f"{dofs_per_node}."
                    ),
                )
            )
        if query.kind in aggregate_node_kinds:
            _append_nodal_region_diagnostics(
                diagnostics,
                model,
                query,
                prefix,
                mesh_node_ids,
            )
        if (
            query.kind == ResultQueryKind.STRESS_EXTREMA
            and query.element_set is not None
        ):
            _append_element_region_diagnostics(
                diagnostics,
                model,
                query.kind,
                query.element_set,
                prefix,
                mesh_element_ids,
            )


def _append_nodal_region_diagnostics(
    diagnostics: list[Diagnostic],
    model: Any,
    query: Any,
    prefix: str,
    mesh_node_ids: set[int],
) -> None:
    selectors = (
        (
            "node_set",
            query.node_set,
            getattr(model, "node_sets", {}),
            lambda region: tuple(int(value) for value in region.node_ids),
        ),
        (
            "edge",
            query.edge,
            getattr(model, "edges", {}),
            lambda region: _topology_region_node_ids(region.edges),
        ),
        (
            "surface",
            query.surface,
            getattr(model, "surfaces", {}),
            lambda region: _topology_region_node_ids(region.faces),
        ),
    )
    for field_name, target, collection, node_ids_for_region in selectors:
        if target is None:
            continue
        region = collection.get(target)
        if region is None:
            actual_field = _target_namespace(model, target, field_name)
            if actual_field is not None:
                message = (
                    f"The {query.kind.value} query references {field_name} "
                    f"{target!r}, but that name is defined as "
                    f"{_selector_with_article(actual_field)}."
                )
                remediation = (
                    f"Use {actual_field}={target!r} and remove "
                    f"{field_name} from this query."
                )
            else:
                message = (
                    f"The {query.kind.value} query references {field_name} "
                    f"{target!r}, which is not defined in the model."
                )
                remediation = _available_selector_remediation(
                    field_name,
                    collection,
                )
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    message,
                    source="analysis_summary",
                    entity=f"{prefix}.{field_name}",
                    remediation=remediation,
                )
            )
            return

        region_node_ids = node_ids_for_region(region)
        if not region_node_ids:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    (
                        f"The selected {field_name} {target!r} contains no "
                        "nodes."
                    ),
                    source="analysis_summary",
                    entity=f"{prefix}.{field_name}",
                    remediation=(
                        f"Choose a non-empty {_selector_label(field_name)}."
                    ),
                )
            )
        elif any(node_id not in mesh_node_ids for node_id in region_node_ids):
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.RESULT_QUERY_FAILED,
                    (
                        f"The selected {field_name} {target!r} references "
                        "nodes that are not defined in the model."
                    ),
                    source="analysis_summary",
                    entity=f"{prefix}.{field_name}",
                    remediation=(
                        f"Choose a valid {_selector_label(field_name)}."
                    ),
                )
            )
        return


def _append_element_region_diagnostics(
    diagnostics: list[Diagnostic],
    model: Any,
    query_kind: ResultQueryKind,
    target: str,
    prefix: str,
    mesh_element_ids: set[int],
) -> None:
    element_sets = getattr(model, "element_sets", {})
    element_set = element_sets.get(target)
    if element_set is None:
        actual_field = _target_namespace(model, target, "element_set")
        if actual_field is not None:
            message = (
                f"The {query_kind.value} query references element_set "
                f"{target!r}, but that name is defined as "
                f"{_selector_with_article(actual_field)}."
            )
        else:
            message = (
                f"The {query_kind.value} query references element_set "
                f"{target!r}, which is not defined in the model."
            )
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.RESULT_QUERY_FAILED,
                message,
                source="analysis_summary",
                entity=f"{prefix}.element_set",
                remediation=_available_selector_remediation(
                    "element_set",
                    element_sets,
                ),
            )
        )
        return

    element_ids = tuple(int(value) for value in element_set.element_ids)
    if not element_ids:
        message = f"The selected element_set {target!r} contains no elements."
    elif any(element_id not in mesh_element_ids for element_id in element_ids):
        message = (
            f"The selected element_set {target!r} references elements that "
            "are not defined in the model."
        )
    else:
        return
    diagnostics.append(
        make_diagnostic(
            DiagnosticCode.RESULT_QUERY_FAILED,
            message,
            source="analysis_summary",
            entity=f"{prefix}.element_set",
            remediation="Choose a valid, non-empty element set.",
        )
    )


def _topology_region_node_ids(entries: Iterable[Any]) -> tuple[int, ...]:
    node_ids: list[int] = []
    seen: set[int] = set()
    for entry in entries:
        for node_id in getattr(entry, "node_ids", ()):
            normalized = int(node_id)
            if normalized not in seen:
                seen.add(normalized)
                node_ids.append(normalized)
    return tuple(node_ids)


def _target_namespace(
    model: Any,
    target: str,
    requested_field: str,
) -> str | None:
    namespaces = (
        ("node_set", getattr(model, "node_sets", {})),
        ("edge", getattr(model, "edges", {})),
        ("surface", getattr(model, "surfaces", {})),
        ("element_set", getattr(model, "element_sets", {})),
    )
    for field_name, collection in namespaces:
        if field_name != requested_field and target in collection:
            return field_name
    return None


def _selector_label(field_name: str) -> str:
    return {
        "node_set": "node set",
        "edge": "edge",
        "surface": "surface",
        "element_set": "element set",
    }[field_name]


def _selector_with_article(field_name: str) -> str:
    article = "an" if field_name in {"edge", "element_set"} else "a"
    return f"{article} {_selector_label(field_name)}"


def _available_selector_remediation(
    field_name: str,
    collection: Any,
) -> str:
    names = [
        _bounded_text(name)
        for name, _ in _sorted_mapping_items(collection)[:8]
    ]
    if names:
        return (
            f"Choose an available {field_name}: "
            + ", ".join(repr(name) for name in names)
            + "."
        )
    return (
        f"No {_selector_label(field_name)} is available in this model; "
        "choose another supported query target."
    )


def _resource_class(
    node_count: int,
    element_count: int,
    total_dofs: int,
    input_size_bytes: int | None,
    spec: ImportAnalysisSpec,
    diagnostics: list[Diagnostic],
) -> str:
    limits = spec.resource_limits
    exceeded = []
    if input_size_bytes is not None and input_size_bytes > limits.max_input_bytes:
        exceeded.append("input_bytes")
    if node_count > limits.max_nodes:
        exceeded.append("nodes")
    if element_count > limits.max_elements:
        exceeded.append("elements")
    if total_dofs > limits.max_dofs:
        exceeded.append("dofs")
    if exceeded:
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.RESOURCE_LIMIT,
                "The imported model exceeds the configured local execution limits.",
                source="analysis_summary",
                entity=",".join(exceeded),
            )
        )
        return "exceeds_limits"
    if total_dofs <= 50_000:
        return "small"
    if total_dofs <= 500_000:
        return "medium"
    return "large"


def _safe_number(
    value: object,
    entity: str,
    diagnostics: list[Diagnostic],
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = math.nan
    if not math.isfinite(result):
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.INVALID_MODEL,
                "A summarized model value is non-numeric or non-finite.",
                source="analysis_summary",
                entity=entity,
            )
        )
        return None
    return result


def _safe_vector(
    values: Iterable[object],
    entity: str,
    diagnostics: list[Diagnostic],
) -> tuple[float | None, ...]:
    return tuple(_safe_number(value, entity, diagnostics) for value in values)


def _safe_target(value: object) -> str | int:
    if isinstance(value, bool):
        return _bounded_text(value)
    if isinstance(value, int):
        return value
    return _bounded_text(value)


def _bounded_items(
    values: Iterable[dict[str, Any]],
    limit: int,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    result: list[dict[str, Any]] = []
    for value in values:
        if len(result) >= limit:
            return tuple(result), True
        result.append(value)
    return tuple(result), False


def _sorted_mapping_items(mapping: Any) -> list[tuple[Any, Any]]:
    return sorted(
        mapping.items(),
        key=lambda item: (str(item[0]).casefold(), str(item[0])),
    )


def _deduplicate_diagnostics(
    diagnostics: Iterable[Diagnostic],
) -> tuple[Diagnostic, ...]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, str | None, str]] = set()
    for diagnostic in diagnostics:
        key = (diagnostic.code, diagnostic.entity, diagnostic.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return tuple(result)


def _truthy_option(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _bounded_text(value: object, *, max_length: int = 128) -> str:
    text = "".join(
        " "
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in str(value)
    ).strip()
    return text[:max_length] or "unnamed"


__all__ = [
    "DEFAULT_MAX_COLLECTION_ITEMS",
    "build_analysis_summary",
]
