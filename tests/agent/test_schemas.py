import json
from dataclasses import replace
from pathlib import Path

import pytest

from fem_agent.diagnostics import DiagnosticCode, exception_diagnostic
from fem_agent.schemas import (
    Diagnostic,
    DiagnosticSeverity,
    ExportFormat,
    ImportAnalysisSpec,
    ResourceLimits,
    ResultQuery,
    ResultQueryKind,
    SchemaValidationError,
    UnitContext,
)


def _units():
    return UnitContext(
        length="mm",
        force="N",
        stress="MPa",
        density="tonne/mm^3",
        acceleration="mm/s^2",
    )


def _spec():
    return ImportAnalysisSpec(
        session_id="ses_test",
        revision=1,
        source_artifact_id="art_input",
        source_sha256="a" * 64,
        unit_context=_units(),
        analysis_step="Step-1",
        requested_queries=(
            ResultQuery(ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE),
            ResultQuery(
                ResultQueryKind.REACTION_SUM,
                component=2,
                node_set="Set-Fixed",
            ),
        ),
        export_formats=(ExportFormat.CSV, ExportFormat.VTK),
        resource_limits=ResourceLimits(worker_timeout_seconds=30),
    )


def test_import_analysis_spec_round_trips_through_json():
    original = _spec()

    restored = ImportAnalysisSpec.from_dict(json.loads(original.to_json()))

    assert restored == original
    assert restored.to_json() == original.to_json()


def test_analysis_spec_is_ready_without_precomputed_result_queries():
    spec = replace(_spec(), requested_queries=())

    assert spec.ready_for_confirmation


def test_unknown_schema_version_is_rejected_at_every_contract_boundary():
    payload = json.loads(_spec().to_json())
    payload["schema_version"] = 99

    with pytest.raises(SchemaValidationError, match="unsupported schema_version"):
        ImportAnalysisSpec.from_dict(payload)

    nested = json.loads(_units().to_json())
    nested["schema_version"] = 99
    with pytest.raises(SchemaValidationError, match="unsupported schema_version"):
        UnitContext.from_dict(nested)


def test_unknown_fields_are_rejected():
    payload = json.loads(_spec().to_json())
    payload["source_path"] = "secret.inp"

    with pytest.raises(SchemaValidationError, match="unknown fields"):
        ImportAnalysisSpec.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["length", "force", "stress", "density", "acceleration"],
)
def test_unit_context_requires_each_nonblank_unit(field):
    values = _units().to_dict()
    values[field] = " "

    with pytest.raises(SchemaValidationError, match=field):
        UnitContext.from_dict(values)


@pytest.mark.parametrize("control", ["\u2028", "\u2029", "\u061c", "\u200e", "\u200f"])
def test_contract_labels_reject_invisible_and_line_separator_controls(control):
    values = _units().to_dict()
    values["length"] = f"m{control}m"

    with pytest.raises(SchemaValidationError, match="control characters"):
        UnitContext.from_dict(values)


def test_result_query_shape_is_validated_before_fem_code_runs():
    with pytest.raises(
        SchemaValidationError,
        match="requires node_set, edge, or surface",
    ):
        ResultQuery(ResultQueryKind.REACTION_SUM, component=2)

    with pytest.raises(SchemaValidationError, match="requires component"):
        ResultQuery(ResultQueryKind.MAX_DISPLACEMENT_COMPONENT)


@pytest.mark.parametrize("field_name", ["edge", "surface"])
def test_result_query_round_trips_topology_region_selectors(field_name):
    query = ResultQuery(
        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
        **{field_name: "Surf-right"},
    )

    restored = ResultQuery.from_dict(query.to_dict())

    assert restored == query
    assert query.to_dict()[field_name] == "Surf-right"


def test_result_query_new_null_fields_preserve_legacy_json_shape():
    payload = ResultQuery(
        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE
    ).to_dict()

    assert "edge" not in payload
    assert "surface" not in payload


def test_result_query_preserves_legacy_positional_field_order():
    query = ResultQuery(
        ResultQueryKind.STRESS_EXTREMA,
        None,
        None,
        None,
        "plate",
        "von_mises",
    )

    assert query.element_set == "plate"
    assert query.measure == "von_mises"
    assert query.edge is None
    assert query.surface is None


@pytest.mark.parametrize("field_name", ["edge", "surface"])
def test_reaction_sum_accepts_each_topology_region(field_name):
    query = ResultQuery(
        ResultQueryKind.REACTION_SUM,
        component=1,
        **{field_name: "loaded"},
    )

    assert getattr(query, field_name) == "loaded"


def test_result_query_rejects_multiple_nodal_region_selectors():
    with pytest.raises(SchemaValidationError, match="mutually exclusive"):
        ResultQuery(
            ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
            node_set="tip",
            edge="loaded",
        )


@pytest.mark.parametrize(
    "kind, extra",
    [
        (
            ResultQueryKind.DISPLACEMENT_MAGNITUDE,
            {"node_id": 1, "edge": "loaded"},
        ),
        (
            ResultQueryKind.STRESS_EXTREMA,
            {"surface": "loaded"},
        ),
    ],
)
def test_result_query_rejects_topology_region_for_nonaggregate_query(
    kind,
    extra,
):
    with pytest.raises(
        SchemaValidationError,
        match="valid only for aggregate nodal queries",
    ):
        ResultQuery(kind, **extra)


def test_diagnostic_rejects_invalid_severity():
    with pytest.raises(SchemaValidationError, match="severity"):
        Diagnostic.from_dict(
            {
                "schema_version": 1,
                "code": "INVALID_INPUT",
                "severity": "fatal",
                "message": "bad input",
                "source": "test",
            }
        )


def test_contract_json_preserves_non_ascii_text():
    diagnostic = Diagnostic(
        code="INVALID_INPUT",
        severity=DiagnosticSeverity.ERROR,
        message="单位信息缺失",
        source="test",
    )

    assert "单位信息缺失" in diagnostic.to_json()


def test_exception_diagnostic_strips_terminal_controls_and_local_paths():
    diagnostic = exception_diagnostic(
        DiagnosticCode.INVALID_INPUT,
        ValueError(
            "bad\nvalue \u202e at C:\\private\\model.inp "
            "api_key=do-not-display"
        ),
        source="test",
    )

    assert "\n" not in diagnostic.message
    assert "\u202e" not in diagnostic.message
    assert "C:\\private" not in diagnostic.message
    assert "do-not-display" not in diagnostic.message


def test_fem_package_does_not_import_agent_package():
    fem_root = Path(__file__).resolve().parents[2] / "src" / "fem"

    offenders = [
        str(path.relative_to(fem_root))
        for path in fem_root.rglob("*.py")
        if "fem_agent" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
