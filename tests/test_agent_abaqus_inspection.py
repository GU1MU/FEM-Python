from __future__ import annotations

import json
from pathlib import Path

import pytest

from fem.io import inp as abaqus
from fem_agent.schemas import DiagnosticSeverity, ResourceLimits
from fem_agent.tools import inspect_abaqus, inspect_abaqus_keywords
from fem_agent.tools import inspection as inspection_module
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.file_builders import write_inp


LINE_FIXTURES = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _codes(result) -> set[str]:
    return {diagnostic.code for diagnostic in result.diagnostics}


@pytest.mark.parametrize(
    "name, element_type, dofs_per_node",
    (
        ("truss2_tension.inp", "Truss2", 3),
        ("beam2_rectangle_uniform_load.inp", "Beam2", 6),
        ("beam2_solid_circle_inclined.inp", "Beam2", 6),
    ),
)
def test_inspection_accepts_supported_line_element_inputs(
    name,
    element_type,
    dofs_per_node,
):
    inspected = inspect_abaqus(LINE_FIXTURES / name)

    assert inspected.ok
    assert inspected.model is not None
    assert {element.type for element in inspected.model.mesh.elements} == {
        element_type
    }
    assert inspected.model.mesh.dofs_per_node == dofs_per_node
    assert not (
        _codes(inspected)
        & {
            "UNSUPPORTED_ELEMENT",
            "UNSUPPORTED_KEYWORD",
            "UNSUPPORTED_KEYWORD_OPTION",
        }
    )


def _minimal_tet_lines(*extra_lines: str) -> list[str]:
    return [
        "*Heading",
        "local heading text",
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "3, 0., 1., 0.",
        "4, 0., 0., 1.",
        "*Element, type=C3D4, elset=SOLID",
        "1, 1,2,3,4",
        *extra_lines,
        "*Step, name=LOAD, nlgeom=NO",
        "*Static",
        "1., 1.",
        "*End Step",
    ]


def _hex20_lines() -> list[str]:
    coordinates = (
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
        (0, -1, -1),
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, 1),
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (-1, -1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
    )
    return [
        "*Node",
        *[
            f"{node_id}, {x}, {y}, {z}"
            for node_id, (x, y, z) in enumerate(coordinates, start=1)
        ],
        "*Element, type=C3D20, elset=SOLID",
        "1, " + ",".join(str(node_id) for node_id in range(1, 21)),
        "*Material, name=STEEL",
        "*Elastic",
        "206000., 0.3",
        "*Solid Section, elset=SOLID, material=STEEL",
        "*Step, name=Step-1, nlgeom=NO",
        "*Static",
        "*Output, field, variable=PRESELECT",
        "*Restart, write, frequency=0",
        "*End Step",
    ]


def test_inspect_hex20_input_matches_direct_import(tmp_path):
    path = write_inp(tmp_path, "hex20.inp", _hex20_lines())
    inspected = inspect_abaqus(path)
    direct = abaqus.read(path)

    assert inspected.ok
    assert inspected.model is not None
    assert inspected.runnable_step is not None
    assert inspected.runnable_step.name == "Step-1"
    assert inspected.model.mesh.num_nodes == direct.mesh.num_nodes == 20
    assert inspected.model.mesh.num_elements == direct.mesh.num_elements == 1
    assert {element.type for element in inspected.model.mesh.elements} == {"Hex20"}
    assert inspected.keyword_inspection.node_record_count == 20
    assert inspected.keyword_inspection.element_record_count == 1
    assert inspected.keyword_inspection.estimated_dofs == 60
    assert not any(
        diagnostic.severity == DiagnosticSeverity.ERROR
        for diagnostic in inspected.diagnostics
    )

    inventory = {
        item["name"]: item for item in inspected.keyword_inspection.keyword_inventory
    }
    assert inventory["element"]["parameters"] == ("elset", "type")
    assert inventory["output"]["count"] == 1
    assert inventory["output"]["disposition"] == "preserved_output"
    assert inventory["restart"]["disposition"] == "inspected_but_ignored"
    assert "IGNORED_METADATA" in _codes(inspected)
    ignored = [
        diagnostic
        for diagnostic in inspected.diagnostics
        if diagnostic.code == "IGNORED_METADATA"
    ]
    assert len(ignored) == 1
    assert ignored[0].severity == DiagnosticSeverity.INFO
    assert "*RESTART" in ignored[0].message
    assert "*OUTPUT" not in ignored[0].message


def test_output_inventory_preserves_unknown_options_without_solver_claims(
    tmp_path,
):
    lines = (
        LINE_FIXTURES / "truss2_tension.inp"
    ).read_text(encoding="utf-8").splitlines()
    output_index = lines.index("*Output, field")
    lines[output_index] = (
        "*Output, field, frequency=1, FutureParentFlag"
    )
    node_output_index = lines.index("*Node Output")
    lines[node_output_index] = (
        "*Node Output, FutureOption=kept, FutureChildFlag"
    )
    path = write_inp(tmp_path, "classified_output.inp", lines)

    inspected = inspect_abaqus(path)
    inventory = {
        item["name"]: item
        for item in inspected.keyword_inspection.keyword_inventory
    }

    assert inspected.ok
    assert inventory["output"]["disposition"] == "preserved_output"
    assert inventory["node output"]["disposition"] == "preserved_output"
    assert (
        inventory["element output"]["disposition"]
        == "postprocess_candidate"
    )
    assert set(inventory["output"]) == {
        "name",
        "count",
        "parameters",
        "flags",
        "disposition",
    }
    assert "UNSUPPORTED_KEYWORD_OPTION" not in _codes(inspected)


@pytest.mark.parametrize(
    ("output_keyword", "expected_disposition"),
    (
        ("*Output, field", "postprocess_candidate"),
        ("*Output, history", "preserved_output"),
        ("*Output, field, variable=PRESELECT", "preserved_output"),
    ),
)
def test_output_parent_structure_has_static_non_solver_disposition(
    tmp_path,
    output_keyword,
    expected_disposition,
):
    path = write_inp(
        tmp_path,
        "output_parent_classification.inp",
        [
            "*Step, name=OUTPUT",
            "*Static",
            output_keyword,
            "*End Step",
        ],
    )

    report = inspect_abaqus_keywords(path)
    inventory = {item["name"]: item for item in report.keyword_inventory}

    assert inventory["output"]["disposition"] == expected_disposition


def test_inspection_accepts_positive_2d_solid_section_thickness(tmp_path):
    path = write_perforated_plate_style_inp(
        tmp_path,
        "plate_thickness.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=("2.5,",),
    )

    inspected = inspect_abaqus(path)

    assert inspected.ok
    assert inspected.model is not None
    assert inspected.model.sections[0].properties == {"thickness": 2.5}
    assert "UNSUPPORTED_KEYWORD_OPTION" not in _codes(inspected)


@pytest.mark.parametrize(
    "section_data",
    (
        ("0.",),
        ("-1.",),
        ("NaN",),
        ("1., 2.",),
        (", 2.",),
        ("1.", "2."),
    ),
)
def test_inspection_rejects_invalid_solid_section_thickness(
    tmp_path,
    section_data,
):
    path = write_perforated_plate_style_inp(
        tmp_path,
        "invalid_plate_thickness.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=section_data,
    )

    inspected = inspect_abaqus(path)

    assert not inspected.ok
    assert _codes(inspected) & {"INVALID_INPUT", "UNSUPPORTED_KEYWORD_OPTION"}


def test_inspection_projection_never_contains_comments_or_data_records(tmp_path):
    comment = "SYSTEM_OVERRIDE_SENTINEL"
    coordinate = "987654.321"
    path = write_inp(
        tmp_path,
        "privacy.inp",
        [
            "*Heading",
            "RAW_HEADING_SENTINEL",
            f"** {comment}",
            "*Node",
            f"1, {coordinate}, 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4",
            "777, 1,2,3,4",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )

    result = inspect_abaqus(path)
    projection = json.dumps(result.provider_data(), ensure_ascii=False)

    assert comment not in projection
    assert "RAW_HEADING_SENTINEL" not in projection
    assert coordinate not in projection
    assert "777" not in projection
    assert str(path) not in projection
    assert result.model is not None


def test_include_is_blocked_without_exposing_include_target(tmp_path):
    path = write_inp(
        tmp_path,
        "include.inp",
        [
            "*Include, input=SECRET_MODEL_PATH.inp",
            *_minimal_tet_lines(),
        ],
    )

    result = inspect_abaqus(path)
    projection = json.dumps(result.provider_data())

    assert "UNSUPPORTED_KEYWORD" in _codes(result)
    assert "SECRET_MODEL_PATH" not in projection
    assert not result.ok


@pytest.mark.parametrize(
    ("extra_lines", "expected_code"),
    [
        (("*Plastic", "100., 0.1"), "UNSUPPORTED_KEYWORD"),
        (("*Element, type=C3D8R", "2, 1,2,3,4,1,2,3,4"), "UNSUPPORTED_ELEMENT"),
        (("*Surface, type=NODE, name=BAD", "1, S1"), "UNSUPPORTED_KEYWORD_OPTION"),
        (("*Boundary, amplitude=A", "1, 1, 1, 0."), "UNSUPPORTED_KEYWORD_OPTION"),
    ],
)
def test_unknown_physical_content_and_options_are_blocking(
    tmp_path,
    extra_lines,
    expected_code,
):
    path = write_inp(
        tmp_path,
        "unsupported.inp",
        _minimal_tet_lines(*extra_lines),
    )

    result = inspect_abaqus(path)

    assert expected_code in _codes(result)
    assert any(
        diagnostic.severity == DiagnosticSeverity.ERROR
        for diagnostic in result.diagnostics
    )
    assert not result.ok


def test_unsupported_option_diagnostic_names_keyword_and_option(tmp_path):
    path = write_inp(
        tmp_path,
        "unsupported_surface_option.inp",
        _minimal_tet_lines(
            "*Surface, type=NODE, name=BAD",
            "1, S1",
        ),
    )

    result = inspect_abaqus(path)
    diagnostic = next(
        item
        for item in result.diagnostics
        if item.code == "UNSUPPORTED_KEYWORD_OPTION"
        and item.entity == "surface:type"
    )

    assert "*SURFACE" in diagnostic.message
    assert "TYPE" in diagnostic.message
    assert "The keyword" not in diagnostic.message
    assert not result.ok


@pytest.mark.parametrize(
    ("record_kind", "expected_entity"),
    [
        ("node", "node:id"),
        ("element", "element:id"),
    ],
)
def test_duplicate_mesh_identifiers_are_blocked_before_parser_overwrite(
    monkeypatch,
    tmp_path,
    record_kind,
    expected_entity,
):
    lines = _minimal_tet_lines()
    if record_kind == "node":
        element_keyword_index = lines.index("*Element, type=C3D4, elset=SOLID")
        lines.insert(element_keyword_index, "1, 9., 9., 9.")
    else:
        element_record_index = lines.index("1, 1,2,3,4")
        lines.insert(element_record_index + 1, "1, 4,3,2,1")
    path = write_inp(tmp_path, f"duplicate_{record_kind}.inp", lines)
    parser_calls = []

    def record_unexpected_parse(*args, **kwargs):
        parser_calls.append((args, kwargs))
        raise AssertionError("duplicate IDs must be blocked during preflight")

    monkeypatch.setattr(abaqus, "read_with_report", record_unexpected_parse)

    result = inspect_abaqus(path)

    assert not parser_calls
    assert result.model is None
    assert any(
        diagnostic.code == "INVALID_INPUT"
        and diagnostic.entity == expected_entity
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize(
    ("keyword", "records"),
    [
        ("Elastic", ("210000., 0.3", "190000., 0.29")),
        ("Density", ("7800.", "7850.")),
    ],
)
def test_material_property_tables_are_blocked_after_first_record(
    tmp_path,
    keyword,
    records,
):
    path = write_inp(
        tmp_path,
        f"multi_record_{keyword.casefold()}.inp",
        _minimal_tet_lines(
            "*Material, name=STEEL",
            f"*{keyword}",
            *records,
        ),
    )

    report = inspect_abaqus_keywords(path)
    inventory = {item["name"]: item for item in report.keyword_inventory}

    assert any(
        diagnostic.code == "UNSUPPORTED_KEYWORD_OPTION"
        and diagnostic.entity == f"{keyword.casefold()}:data"
        for diagnostic in report.diagnostics
    )
    assert (
        inventory[keyword.casefold()]["disposition"]
        == "unsupported_blocking"
    )
    assert report.has_blocking_diagnostics


def test_single_material_records_are_counted_per_keyword_block(tmp_path):
    path = write_inp(
        tmp_path,
        "independent_material_blocks.inp",
        _minimal_tet_lines(
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Density",
            "7800.",
            "*Material, name=steel",
            "*Elastic",
            "70000., 0.33",
            "*Density",
            "2700.",
        ),
    )

    report = inspect_abaqus_keywords(path)

    assert not any(
        diagnostic.entity
        in {
            "elastic:data",
            "density:data",
            "elastic:block",
            "density:block",
            "material:name",
        }
        for diagnostic in report.diagnostics
    )


@pytest.mark.parametrize(
    ("keyword", "first_record", "second_record"),
    [
        ("Elastic", "210000., 0.3", "190000., 0.29"),
        ("Density", "7800.", "7850."),
    ],
)
def test_repeated_material_property_keyword_blocks_are_blocked(
    tmp_path,
    keyword,
    first_record,
    second_record,
):
    path = write_inp(
        tmp_path,
        f"repeated_{keyword.casefold()}_block.inp",
        _minimal_tet_lines(
            "*Material, name=STEEL",
            f"*{keyword}",
            first_record,
            f"*{keyword}",
            second_record,
        ),
    )

    report = inspect_abaqus_keywords(path)

    assert any(
        diagnostic.code == "UNSUPPORTED_KEYWORD_OPTION"
        and diagnostic.entity == f"{keyword.casefold()}:block"
        for diagnostic in report.diagnostics
    )
    assert report.has_blocking_diagnostics


def test_duplicate_material_name_is_blocked_before_parser_overwrite(tmp_path):
    path = write_inp(
        tmp_path,
        "duplicate_material.inp",
        _minimal_tet_lines(
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Material, name=STEEL",
        ),
    )

    report = inspect_abaqus_keywords(path)

    assert any(
        diagnostic.code == "INVALID_INPUT"
        and diagnostic.entity == "material:name"
        for diagnostic in report.diagnostics
    )
    assert report.has_blocking_diagnostics


def test_identifier_tracking_sets_stop_at_resource_record_caps():
    limits = ResourceLimits(
        max_nodes=2,
        max_elements=1,
        max_dofs=100,
    )
    state = inspection_module._ScanState(64, limits)
    state.handle_keyword(inspection_module._parse_keyword("*Node"))
    for node_id in range(1, 11):
        state.handle_data(f"{node_id}, 0., 0., 0.")

    state.handle_keyword(
        inspection_module._parse_keyword("*Element, type=C3D4")
    )
    for element_id in range(1, 11):
        state.handle_data(f"{element_id}, 1, 2, 1, 2")

    assert state.node_record_count == 10
    assert state.element_record_count == 10
    assert len(state._node_ids) == limits.max_nodes
    assert len(state._element_ids) == limits.max_elements
    assert "RESOURCE_LIMIT" in _codes(state.finish())


@pytest.mark.parametrize(
    "procedure_lines",
    [
        ("*Dynamic", "1., 1."),
        ("*Step, name=LOAD, nlgeom=YES", "*Static", "*End Step"),
    ],
)
def test_nonstatic_or_geometrically_nonlinear_procedure_is_blocked(
    tmp_path,
    procedure_lines,
):
    base = _minimal_tet_lines()
    if procedure_lines[0].startswith("*Step"):
        base = base[:-4] + list(procedure_lines)
    else:
        base = base[:-4] + ["*Step, name=LOAD", *procedure_lines, "*End Step"]
    path = write_inp(tmp_path, "procedure.inp", base)

    result = inspect_abaqus(path)

    assert "UNSUPPORTED_PROCEDURE" in _codes(result)
    assert not result.ok


def test_dload_trvec_is_blocked_even_though_dsload_trvec_is_supported(tmp_path):
    path = write_inp(
        tmp_path,
        "dload_trvec.inp",
        _minimal_tet_lines(
            "*Dload",
            "SOLID, TRVEC, 1., 1., 0., 0.",
        ),
    )

    result = inspect_abaqus(path)

    assert "UNSUPPORTED_KEYWORD_OPTION" in _codes(result)
    assert not result.ok


def test_multiple_steps_are_blocked_as_history(tmp_path):
    lines = _minimal_tet_lines()
    lines.extend(
        [
            "*Step, name=SECOND",
            "*Static",
            "*End Step",
        ]
    )
    path = write_inp(tmp_path, "multi_step.inp", lines)

    result = inspect_abaqus(path)

    assert "MULTI_STEP_HISTORY_UNSUPPORTED" in _codes(result)
    assert result.runnable_step is None
    assert not result.ok


@pytest.mark.parametrize(
    "assembly_lines",
    [
        (
            "*Part, name=P1",
            "*End Part",
            "*Part, name=P2",
            "*End Part",
        ),
        (
            "*Part, name=P1",
            "*End Part",
            "*Assembly, name=A",
            "*Instance, name=I1, part=P1",
            "*End Instance",
            "*Instance, name=I2, part=P1",
            "*End Instance",
            "*End Assembly",
        ),
        (
            "*Part, name=P1",
            "*End Part",
            "*Assembly, name=A",
            "*Instance, name=I1, part=P1",
            "1., 0., 0.",
            "*End Instance",
            "*End Assembly",
        ),
    ],
)
def test_multiple_parts_instances_and_instance_transforms_are_blocked(
    tmp_path,
    assembly_lines,
):
    path = write_inp(
        tmp_path,
        "assembly.inp",
        [*assembly_lines, *_minimal_tet_lines()],
    )

    report = inspect_abaqus_keywords(path)

    assert "UNSUPPORTED_KEYWORD_OPTION" in _codes(report)
    assert report.has_blocking_diagnostics


def test_keyword_inventory_and_diagnostics_are_bounded(tmp_path):
    unknown = [f"*Unknown Physical Keyword {index}" for index in range(100)]
    path = write_inp(
        tmp_path,
        "bounded.inp",
        [*unknown, *_minimal_tet_lines()],
    )

    report = inspect_abaqus_keywords(path, max_inventory_entries=8)

    assert len(report.keyword_inventory) == 8
    assert len(report.diagnostics) <= 96
    assert report.collections_truncated
    assert report.has_blocking_diagnostics


def test_generate_set_expansion_is_blocked_before_model_construction(tmp_path):
    path = write_inp(
        tmp_path,
        "generate_bomb.inp",
        [
            "*Nset, nset=TOO_LARGE, generate",
            "1, 2147483647, 1",
            "*Step, name=Step-1",
            "*Static",
            "*End Step",
        ],
    )

    result = inspect_abaqus(path)

    assert result.model is None
    assert "RESOURCE_LIMIT" in _codes(result)


def test_invalid_utf8_is_rejected_without_lossy_keyword_scanning(tmp_path):
    path = tmp_path / "invalid_utf8.inp"
    path.write_bytes(b"*Heading\n\xff\n")

    result = inspect_abaqus(path)

    assert not result.ok
    assert any(
        diagnostic.code == "INVALID_INPUT"
        and "UTF-8" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_node_and_dof_preflight_blocks_model_construction(
    monkeypatch,
    tmp_path,
):
    path = write_inp(
        tmp_path,
        "preflight_limit.inp",
        _minimal_tet_lines(),
    )

    def fail_if_parsed(*args, **kwargs):
        raise AssertionError("resource-limited input must not be parsed")

    monkeypatch.setattr(abaqus, "read_with_report", fail_if_parsed)
    result = inspect_abaqus(
        path,
        resource_limits=ResourceLimits(
            max_nodes=3,
            max_elements=2,
            max_dofs=8,
        ),
    )

    assert result.model is None
    assert result.keyword_inspection.node_record_count == 4
    assert result.keyword_inspection.element_record_count == 1
    assert result.keyword_inspection.estimated_dofs == 12
    assert "RESOURCE_LIMIT" in _codes(result)


def test_malformed_public_import_returns_diagnostic_without_model_or_summary(
    tmp_path,
):
    path = write_inp(
        tmp_path,
        "malformed_b31.inp",
        [
            "*Heading",
            "Agent malformed import",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.2, 0.1",
            "1., 0., 0.",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )

    result = inspect_abaqus(path)

    assert not result.ok
    assert result.model is None
    assert result.source_summary is None
    assert "IMPORT_FAILED" in _codes(result)
