from __future__ import annotations

from fem_agent.schemas import (
    ExportFormat,
    ImportAnalysisSpec,
    ResourceLimits,
    ResultQuery,
    ResultQueryKind,
    UnitContext,
)
from fem_agent.summaries import build_analysis_summary
from fem_agent.tools import inspect_abaqus
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.file_builders import write_inp


def _summary_input(tmp_path):
    return write_perforated_plate_style_inp(
        tmp_path,
        "summary_model.inp",
        ("*Dsload", "Surf-right, P, 2."),
        section_data=("2.5,",),
    )


def _spec(
    *,
    step: str | None = "Step-1",
    units: UnitContext | None = None,
    queries: tuple[ResultQuery, ...] | None = None,
    limits: ResourceLimits | None = None,
) -> ImportAnalysisSpec:
    return ImportAnalysisSpec(
        session_id="session-1",
        revision=1,
        source_artifact_id="artifact-1",
        source_sha256="a" * 64,
        unit_context=units
        if units is not None
        else UnitContext(
            length="mm",
            force="N",
            stress="MPa",
            density="tonne/mm3",
            acceleration="mm/s2",
        ),
        analysis_step=step,
        requested_queries=queries
        if queries is not None
        else (
            ResultQuery(ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE),
            ResultQuery(
                ResultQueryKind.REACTION_SUM,
                component=2,
                node_set="Set-left",
            ),
        ),
        export_formats=(ExportFormat.CSV, ExportFormat.VTK),
        resource_limits=limits or ResourceLimits(),
    )


def test_small_model_summary_is_complete_and_deterministic(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))

    first = build_analysis_summary(imported, _spec(), "b" * 64)
    second = build_analysis_summary(imported, _spec(), "b" * 64)

    assert first.to_json() == second.to_json()
    assert first.node_count == 6
    assert first.element_count == 2
    assert first.dofs_per_node == 2
    assert first.total_dofs == 12
    assert first.element_types == {"Quad4": 2}
    assert first.resource_class == "small"
    assert not first.has_blocking_diagnostics

    assert {"name": "Set-left", "size": 2} in first.node_sets
    assert {"name": "SOLID", "size": 2} in first.element_sets
    assert {"name": "Surf-right", "size": 1} in first.edges
    assert first.materials == (
        {
            "name": "STEEL",
            "properties": {
                "E": 210000.0,
                "nu": 0.3,
            },
        },
    )
    assert first.sections == (
        {
            "element_set": "SOLID",
            "material": "STEEL",
            "section_type": "solid",
            "thickness": 2.5,
        },
    )
    assert first.analysis_step is not None
    assert first.analysis_step["name"] == "Step-1"
    assert first.analysis_step["procedure"] == "static"
    assert first.analysis_step["nlgeom"] is False
    assert first.analysis_step["output_requests"] == []
    assert first.constraints == (
        {
            "target": "Set-left",
            "first_component": 1,
            "last_component": 2,
            "value": 0.0,
            "scope": "initial",
        },
    )
    assert first.loads == (
        {
            "type": "pressure",
            "location": "edge",
            "target": "Surf-right",
            "vector": [],
            "magnitude": 2.0,
            "coordinate_system": "global",
        },
    )


def test_summary_contains_no_node_coordinates_connectivity_or_comments(tmp_path):
    path = write_inp(
        tmp_path,
        "private_model.inp",
        [
            "*Heading",
            "PRIVATE_HEADING_SENTINEL",
            "** PRIVATE_COMMENT_SENTINEL",
            "*Node",
            "1, 912345.6789, 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4",
            "7654321, 1,2,3,4",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )
    imported = inspect_abaqus(path)

    payload = build_analysis_summary(
        imported,
        _spec(step="LOAD"),
        "b" * 64,
    ).to_json()

    assert "PRIVATE_HEADING_SENTINEL" not in payload
    assert "PRIVATE_COMMENT_SENTINEL" not in payload
    assert "912345.6789" not in payload
    assert "7654321" not in payload
    assert str(path) not in payload


def test_summary_includes_inherited_initial_constraints(tmp_path):
    path = write_inp(
        tmp_path,
        "initial_constraint.inp",
        [
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4",
            "1, 1,2,3,4",
            "*Nset, nset=FIXED",
            "1",
            "*Boundary",
            "FIXED, 1, 3, 0.",
            "*Step, name=LOAD",
            "*Static",
            "*Cload",
            "2, 1, 5.",
            "*End Step",
        ],
    )
    imported = inspect_abaqus(path)

    summary = build_analysis_summary(
        imported,
        _spec(step="LOAD"),
        "b" * 64,
    )

    assert summary.constraints == (
        {
            "target": "FIXED",
            "first_component": 1,
            "last_component": 3,
            "value": 0.0,
            "scope": "initial",
        },
    )
    assert summary.loads == (
        {
            "type": "nodal",
            "target": 2,
            "component": 1,
            "value": 5.0,
        },
    )


def test_summary_marks_missing_confirmation_context_as_blocking(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))
    spec = ImportAnalysisSpec(
        session_id="session-1",
        revision=1,
        source_artifact_id="artifact-1",
        source_sha256="a" * 64,
        unit_context=None,
        analysis_step=None,
        requested_queries=(),
        export_formats=(),
    )

    summary = build_analysis_summary(imported, spec, "b" * 64)
    codes = {diagnostic.code for diagnostic in summary.diagnostics}

    assert summary.has_blocking_diagnostics
    assert "UNIT_CONTEXT_REQUIRED" in codes
    assert "INVALID_INPUT" in codes


def test_summary_allows_confirmation_without_precomputed_queries(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))

    summary = build_analysis_summary(
        imported,
        _spec(queries=()),
        "b" * 64,
    )

    assert not summary.has_blocking_diagnostics
    assert not any(
        item.code == "RESULT_REQUEST_REQUIRED"
        for item in summary.diagnostics
    )


def test_summary_blocks_node_set_name_that_is_an_edge(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))

    summary = build_analysis_summary(
        imported,
        _spec(
            queries=(
                ResultQuery(
                    ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                    node_set="Surf-right",
                ),
            )
        ),
        "b" * 64,
    )

    diagnostic = next(
        item
        for item in summary.diagnostics
        if item.code == "RESULT_QUERY_FAILED"
    )
    assert summary.has_blocking_diagnostics
    assert diagnostic.entity == "requested_queries[0].node_set"
    assert "defined as an edge" in diagnostic.message
    assert "Use edge='Surf-right'" in diagnostic.remediation


def test_summary_accepts_edge_as_a_nodal_result_region(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))

    summary = build_analysis_summary(
        imported,
        _spec(
            queries=(
                ResultQuery(
                    ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                    edge="Surf-right",
                ),
            )
        ),
        "b" * 64,
    )

    assert not summary.has_blocking_diagnostics
    assert not any(
        item.code == "RESULT_QUERY_FAILED"
        for item in summary.diagnostics
    )


def test_summary_blocks_invalid_result_node_and_component(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))

    summary = build_analysis_summary(
        imported,
        _spec(
            queries=(
                ResultQuery(
                    ResultQueryKind.DISPLACEMENT_COMPONENT,
                    node_id=999,
                    component=3,
                ),
            )
        ),
        "b" * 64,
    )

    entities = {
        item.entity
        for item in summary.diagnostics
        if item.code == "RESULT_QUERY_FAILED"
    }
    assert entities == {
        "requested_queries[0].node_id",
        "requested_queries[0].component",
    }


def test_summary_does_not_add_step_selection_error_after_import_failure(
    tmp_path,
):
    path = write_inp(
        tmp_path,
        "blocked_import.inp",
        [
            "*Unsupported Physical Keyword",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )
    imported = inspect_abaqus(path)

    summary = build_analysis_summary(
        imported,
        _spec(step=None),
        "b" * 64,
    )

    assert imported.runnable_step is None
    assert not any(
        diagnostic.code == "INVALID_INPUT"
        and diagnostic.entity == "analysis_step"
        for diagnostic in summary.diagnostics
    )


def test_summary_enforces_collection_and_resource_bounds(tmp_path):
    imported = inspect_abaqus(_summary_input(tmp_path))
    limits = ResourceLimits(
        max_nodes=5,
        max_elements=100,
        max_dofs=10_000,
    )

    summary = build_analysis_summary(
        imported,
        _spec(limits=limits),
        "b" * 64,
        max_collection_items=1,
    )

    assert summary.collections_truncated
    assert len(summary.node_sets) == 1
    assert len(summary.element_sets) == 1
    assert len(summary.materials) == 1
    assert len(summary.loads) == 1
    assert summary.resource_class == "exceeds_limits"
    assert "RESOURCE_LIMIT" in {
        diagnostic.code for diagnostic in summary.diagnostics
    }
