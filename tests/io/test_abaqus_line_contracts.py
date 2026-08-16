from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fem.io import inp as abaqus
from fem.application import (
    RegionRef,
    compile_model_definitions,
    definitions_from_model,
    resolve_effective_beam_frames,
)
from fem.elements import (
    BEAM_DEFAULT_LOCAL_Y_REFERENCE,
    BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY,
    BEAM_LOCAL_Y_REFERENCE_KEY,
    get_element_kernel,
)
from fem.solvers.static_linear import solve
from tests.helpers.file_builders import write_inp


FIXTURES = Path(__file__).resolve().parents[1] / "helpers" / "fixtures" / "inp"
STANDARD = FIXTURES / "abaqus_standard"
RETIRED = FIXTURES / "abaqus_retired"


def _step(model, name: str):
    return next(step for step in model.steps if step.name == name)


def _minimal_t3d2(
    *,
    element_keyword: str = "*Element, type=T3D2, elset=TRUSS",
    section_keyword: str = (
        "*Solid Section, elset=TRUSS, material=STEEL"
    ),
    section_data: tuple[str, ...] = ("0.01",),
    extra_before_section: tuple[str, ...] = (),
    extra_after_section: tuple[str, ...] = (),
) -> list[str]:
    return [
        "*Heading",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        element_keyword,
        "1, 1, 2",
        "*Material, name=STEEL",
        "*Elastic",
        "2.10E11, 0.30",
        *extra_before_section,
        section_keyword,
        *section_data,
        *extra_after_section,
    ]


def _minimal_b31(
    *,
    element_keyword: str = "*Element, type=B31, elset=BEAM",
    connectivity: str = "1, 1, 2",
    section_keyword: str = (
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT"
    ),
    geometry: tuple[str, ...] = ("0.20, 0.10",),
    orientation: tuple[str, ...] = ("0.0, 1.0, 0.0",),
    load_keyword: str = "*Dload",
    load_records: tuple[str, ...] = (),
) -> list[str]:
    lines = [
        "*Heading",
        "*Node",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        element_keyword,
        connectivity,
        "*Material, name=STEEL",
        "*Elastic",
        "2.10E11, 0.30",
        section_keyword,
        *geometry,
        *orientation,
    ]
    if load_records:
        lines.extend(
            (
                "*Step, name=LOAD",
                "*Static",
                load_keyword,
                *load_records,
                "*End Step",
            )
        )
    return lines


@pytest.mark.parametrize(
    ("filename", "required_lines"),
    (
        (
            "t3d2_tension.inp",
            (
                "*Element, type=T3D2, elset=TRUSS",
                "*Solid Section, elset=TRUSS, material=STEEL",
                "1.0E-4",
            ),
        ),
        (
            "t3d2_default_area.inp",
            (
                "*Element, type=T3D2, elset=TRUSS",
                "*Solid Section, elset=TRUSS, material=STEEL",
                ",",
            ),
        ),
        (
            "b31_rect_explicit_n1_loads.inp",
            (
                "*Element, type=B31, elset=BEAM",
                (
                    "*Beam Section, elset=BEAM, material=STEEL, "
                    "section=RECT"
                ),
                "0.20, 0.10",
                "0.0, 1.0, 0.0",
                "BEAM, PX, 11.0",
                "BEAM, P1, -14.0",
            ),
        ),
        (
            "b31_rect_default_n1.inp",
            (
                "*Element, type=B31, elset=BEAM",
                (
                    "*Beam Section, elset=BEAM, material=STEEL, "
                    "section=RECT"
                ),
                "0.30, 0.10",
            ),
        ),
        (
            "b31_circ.inp",
            (
                "*Element, type=B31, elset=BEAM",
                (
                    "*Beam Section, elset=BEAM, material=STEEL, "
                    "section=CIRC"
                ),
                "0.075",
            ),
        ),
        (
            "b31_thick_pipe.inp",
            (
                "*Element, type=B31, elset=BEAM",
                (
                    "*Beam Section, elset=BEAM, material=STEEL, "
                    "section=THICK PIPE"
                ),
                "0.10, 0.025",
            ),
        ),
    ),
)
def test_standard_line_fixtures_are_literal_utf8_official_shapes(
    filename: str,
    required_lines: tuple[str, ...],
) -> None:
    raw = (STANDARD / filename).read_bytes()
    text = raw.decode("utf-8")

    assert text.encode("utf-8") == raw
    for line in required_lines:
        assert line in text.splitlines()
    assert "*Truss Section" not in text
    assert "QGLOBAL" not in text
    assert "QLOCAL" not in text
    assert "type=BEAM2" not in text
    assert "type=TRUSS2" not in text


def test_public_line_import_error_hierarchy_remains_value_error_compatible():
    for name in (
        "InpInputError",
        "InpParseError",
        "InpBuildError",
        "UnsupportedInpFeatureError",
    ):
        error_type = getattr(abaqus, name)
        assert issubclass(error_type, abaqus.InpInputError)
        assert issubclass(error_type, ValueError)


def test_standard_t3d2_maps_solid_section_scalar_to_canonical_area():
    model = abaqus.read(STANDARD / "t3d2_tension.inp")

    assert model.mesh.elements[0].type == "Truss2"
    assert model.mesh.dofs_per_node == 3
    assert model.sections[0].section_type == "truss"
    assert model.sections[0].properties == {"area": pytest.approx(1.0e-4)}


def test_standard_t3d2_explicit_blank_first_field_defaults_area_to_one():
    model = abaqus.read(STANDARD / "t3d2_default_area.inp")

    assert model.sections[0].section_type == "truss"
    assert model.sections[0].properties == {"area": pytest.approx(1.0)}


def test_standard_t3d2_tension_has_analytical_displacement_strain_and_stress():
    model = abaqus.read(STANDARD / "t3d2_tension.inp")
    result = solve(model, "TENSION")
    element = model.mesh.elements[0]
    strain, stress, mises = get_element_kernel(
        element.type
    ).element_stress(model.mesh, element, result.U)

    assert result.nodal_displacement(2, 1) == pytest.approx(2.0e-3)
    assert strain == pytest.approx(1.0e-3)
    assert stress == pytest.approx(2.10e8)
    assert mises == pytest.approx(2.10e8)


@pytest.mark.parametrize(
    ("filename", "section_type", "expected_properties"),
    (
        (
            "b31_rect_explicit_n1_loads.inp",
            "rectangle",
            {"height": 0.10, "width": 0.20},
        ),
        (
            "b31_rect_default_n1.inp",
            "rectangle",
            {"height": 0.10, "width": 0.30},
        ),
        (
            "b31_circ.inp",
            "solid_circle",
            {"radius": 0.075},
        ),
        (
            "b31_thick_pipe.inp",
            "hollow_circle",
            {"outer_radius": 0.10, "inner_radius": 0.075},
        ),
    ),
)
def test_standard_b31_profiles_map_to_canonical_section_schema(
    filename: str,
    section_type: str,
    expected_properties: dict[str, float],
) -> None:
    model = abaqus.read(STANDARD / filename)
    section = model.sections[0]

    assert {element.type for element in model.mesh.elements} == {"Beam2"}
    assert model.mesh.dofs_per_node == 6
    assert section.section_type == section_type
    for key, value in expected_properties.items():
        assert section.properties[key] == pytest.approx(value)


@pytest.mark.parametrize(
    ("filename", "expected_reference"),
    (
        ("b31_rect_explicit_n1_loads.inp", (0.0, 1.0, 0.0)),
        ("b31_rect_default_n1.inp", None),
    ),
)
def test_b31_n1_is_assignment_owned_and_not_a_section_definition_property(
    filename: str,
    expected_reference: tuple[float, float, float] | None,
) -> None:
    model = abaqus.read(STANDARD / filename)
    element = model.mesh.elements[0]
    definitions = definitions_from_model(model)

    assert BEAM_LOCAL_Y_REFERENCE_KEY not in element.props
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in definitions.sections[0].properties
    if expected_reference is None:
        assert element.props[BEAM_DEFAULT_LOCAL_Y_REFERENCE_KEY] == pytest.approx(
            BEAM_DEFAULT_LOCAL_Y_REFERENCE
        )
        assert definitions.assignments[0].beam_orientation is None
    else:
        assert definitions.assignments[0].beam_orientation is not None
        assert (
            definitions.assignments[0]
            .beam_orientation.local_y_reference
            == pytest.approx(expected_reference)
        )

    compiled = compile_model_definitions(model, definitions).require_model()
    if expected_reference is None:
        assert BEAM_LOCAL_Y_REFERENCE_KEY not in compiled.sections[0].properties
    else:
        assert compiled.sections[0].properties[
            BEAM_LOCAL_Y_REFERENCE_KEY
        ] == pytest.approx(expected_reference)


def test_explicit_b31_n1_resolves_t_n1_n2_as_right_handed_xyz_frame():
    model = abaqus.read(
        STANDARD / "b31_rect_explicit_n1_loads.inp"
    )
    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAM"),
    )

    assert report.passed
    assert len(report.frames) == 2
    for frame in report.frames:
        assert frame.source == "explicit"
        assert frame.local_x == pytest.approx((1.0, 0.0, 0.0))
        assert frame.local_y == pytest.approx((0.0, 1.0, 0.0))
        assert frame.local_z == pytest.approx((0.0, 0.0, 1.0))
        assert np.linalg.det(frame.rotation) == pytest.approx(1.0)


def test_missing_b31_n1_keeps_default_frame_provenance():
    model = abaqus.read(STANDARD / "b31_rect_default_n1.inp")
    report = resolve_effective_beam_frames(
        model,
        RegionRef("element_set", "BEAM"),
    )
    tangent = np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0)

    assert report.passed
    frame = report.frames[0]
    assert frame.source == "default"
    assert frame.orientation is None
    assert frame.local_x == pytest.approx(tangent)
    assert frame.local_y == pytest.approx((0.0, 0.0, -1.0))
    assert frame.local_z == pytest.approx(
        (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0)
    )


def test_standard_b31_dload_records_preserve_order_sign_and_coordinate_system():
    model = abaqus.read(
        STANDARD / "b31_rect_explicit_n1_loads.inp"
    )
    loads = _step(model, "STANDARD_LINE_LOADS").line_loads

    assert [
        (load.target, load.vector, load.coordinate_system)
        for load in loads
    ] == [
        ("BEAM", (11.0, 0.0, 0.0), "global"),
        ("BEAM", (0.0, -12.0, 0.0), "global"),
        ("BEAM", (0.0, 0.0, 13.0), "global"),
        ("BEAM", (0.0, -14.0, 0.0), "local"),
        ("BEAM", (0.0, 0.0, 15.0), "local"),
        ("BEAM", (0.0, 1.5, 0.0), "local"),
    ]


@pytest.mark.parametrize(
    ("filename", "diagnostic_tokens"),
    (
        ("truss_section.inp", ("Truss Section", "Solid Section")),
        ("beam_keyword_geometry.inp", ("Beam Section", "data")),
        ("qglobal.inp", ("QGLOBAL", "PX")),
        ("qlocal.inp", ("QLOCAL", "P1")),
        ("beam2_alias.inp", ("BEAM2", "B31")),
        ("truss2_alias.inp", ("TRUSS2", "T3D2")),
    ),
)
def test_retired_line_dialects_fail_closed_with_migration_guidance(
    filename: str,
    diagnostic_tokens: tuple[str, str],
) -> None:
    with pytest.raises(abaqus.InpInputError) as caught:
        abaqus.read(RETIRED / filename)

    diagnostic = f"{caught.value} {caught.value.remediation}"
    for token in diagnostic_tokens:
        assert token.casefold() in diagnostic.casefold()
    assert caught.value.remediation


@pytest.mark.parametrize(
    "area_record",
    ("0.0", "-0.01", "NaN", "Infinity", "True", "0.01, 2.0"),
)
def test_t3d2_invalid_area_records_fail_before_model_construction(
    tmp_path,
    area_record: str,
) -> None:
    path = write_inp(
        tmp_path,
        "invalid_t3d2_area.inp",
        _minimal_t3d2(section_data=(area_record,)),
    )

    with pytest.raises(abaqus.InpInputError):
        abaqus.read(path)


def test_t3d2_missing_entire_area_record_is_not_treated_as_default(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "missing_t3d2_area.inp",
        _minimal_t3d2(section_data=()),
    )

    with pytest.raises(abaqus.InpBuildError, match="area"):
        abaqus.read(path)


@pytest.mark.parametrize(
    ("profile", "geometry"),
    (
        ("RECT", ("0.2",)),
        ("RECT", ("0.2, 0.1, 0.05",)),
        ("CIRC", ("0.1, 0.2",)),
        ("THICK PIPE", ("0.1",)),
        ("THICK PIPE", ("0.1, 0.1",)),
        ("THICK PIPE", ("0.1, 0.2",)),
        ("THICK PIPE", ("0.1, -0.01",)),
    ),
)
def test_b31_invalid_profile_geometry_fails_closed(
    tmp_path,
    profile: str,
    geometry: tuple[str, ...],
) -> None:
    path = write_inp(
        tmp_path,
        "invalid_b31_geometry.inp",
        _minimal_b31(
            section_keyword=(
                "*Beam Section, elset=BEAM, material=STEEL, "
                f"section={profile}"
            ),
            geometry=geometry,
        ),
    )

    with pytest.raises(abaqus.InpInputError):
        abaqus.read(path)


def test_b31_pipe_is_not_silently_mapped_to_hollow_circle(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "unsupported_pipe.inp",
        _minimal_b31(
            section_keyword=(
                "*Beam Section, elset=BEAM, material=STEEL, "
                "section=PIPE"
            ),
            geometry=("0.1, 0.01",),
        ),
    )

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as caught:
        abaqus.read(path)

    assert "THICK PIPE" in caught.value.remediation.upper()


@pytest.mark.parametrize(
    "load_record",
    (
        "BEAM, P3, 1.0",
        "BEAM, PXNU, 1.0",
        "BEAM, QGLOBAL, 0.0, 1.0, 0.0",
        "BEAM, QLOCAL, 0.0, 1.0, 0.0",
        "BEAM, PX, 1.0, 2.0",
        "BEAM, P1, NaN",
    ),
)
def test_unsupported_or_malformed_b31_dloads_fail_closed(
    tmp_path,
    load_record: str,
) -> None:
    path = write_inp(
        tmp_path,
        "invalid_b31_dload.inp",
        _minimal_b31(load_records=(load_record,)),
    )

    with pytest.raises(abaqus.InpInputError):
        abaqus.read(path)


def test_t3d2_target_rejects_line_dload_labels(tmp_path) -> None:
    lines = _minimal_t3d2(
        extra_after_section=(
            "*Step, name=LOAD",
            "*Static",
            "*Dload",
            "TRUSS, P1, 1.0",
            "*End Step",
        )
    )
    path = write_inp(tmp_path, "truss_line_load.inp", lines)

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as caught:
        abaqus.read(path)

    assert "T3D2" in (
        f"{caught.value} {caught.value.remediation}"
    ).upper()


def test_b31_rejects_dsload_even_when_target_name_is_valid(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "beam_dsload.inp",
        _minimal_b31(
            load_keyword="*Dsload",
            load_records=("BEAM, P1, 1.0",),
        ),
    )

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as caught:
        abaqus.read(path)

    assert "DSLOAD" in (
        f"{caught.value} {caught.value.remediation}"
    ).upper()


def test_utf8_bom_is_accepted_at_lexer_boundary(tmp_path) -> None:
    source = STANDARD / "t3d2_default_area.inp"
    path = tmp_path / "bom_t3d2.inp"
    path.write_bytes(b"\xef\xbb\xbf" + source.read_bytes())

    model = abaqus.read(path)

    assert model.mesh.elements[0].type == "Truss2"


def test_gb18030_abaqus_comments_are_accepted_without_lossy_decode(
    tmp_path,
) -> None:
    source = STANDARD / "t3d2_default_area.inp"
    path = tmp_path / "gb18030_t3d2.inp"
    chinese_comment = "** 名称：完全固定；类型：位移/转角\n"
    path.write_bytes(
        chinese_comment.encode("gb18030") + source.read_bytes()
    )

    model = abaqus.read(path)

    assert model.mesh.elements[0].type == "Truss2"


def test_invalid_utf8_raises_typed_parse_error_without_dropping_cause(
    tmp_path,
) -> None:
    path = tmp_path / "invalid_utf8.inp"
    path.write_bytes(b"*Heading\n** invalid byte: \xff\n")

    with pytest.raises(abaqus.InpParseError) as caught:
        abaqus.read_with_report(path)

    assert Path(caught.value.path) == path
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)
    assert caught.value.remediation


def test_keyword_parameter_continuation_accepts_d_exponent_data(
    tmp_path,
) -> None:
    lines = _minimal_t3d2(
        element_keyword="*Element, type=T3D2,\n elset=TRUSS",
        section_data=("1.0d-2",),
    )
    path = write_inp(tmp_path, "continued_keyword.inp", lines)

    model = abaqus.read(path)

    assert model.sections[0].properties["area"] == pytest.approx(1.0e-2)


def test_duplicate_parameter_across_keyword_continuation_is_rejected(
    tmp_path,
) -> None:
    lines = _minimal_t3d2(
        element_keyword=(
            "*Element, type=T3D2,\n type=T3D2, elset=TRUSS"
        ),
    )
    path = write_inp(tmp_path, "duplicate_continued_parameter.inp", lines)

    with pytest.raises(abaqus.InpParseError, match="type") as caught:
        abaqus.read_with_report(path)

    assert caught.value.line == 6
    assert "element" in caught.value.keyword.casefold()
    assert caught.value.remediation


def test_unknown_line_keyword_parameter_is_rejected_with_source_context(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "unknown_element_parameter.inp",
        _minimal_t3d2(
            element_keyword=(
                "*Element, type=T3D2, elset=TRUSS, mystery=YES"
            )
        ),
    )

    with pytest.raises(abaqus.InpParseError) as caught:
        abaqus.read_with_report(path)

    assert Path(caught.value.path) == path
    assert caught.value.line == 5
    assert "element" in caught.value.keyword.casefold()
    assert caught.value.remediation


def test_exact_section_dispatch_rejects_beam_general_section(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "beam_general_section.inp",
        _minimal_b31(
            section_keyword=(
                "*Beam General Section, elset=BEAM, material=STEEL"
            ),
            geometry=("0.01, 1.0E-6, 1.0E-6",),
        ),
    )

    with pytest.raises(
        abaqus.UnsupportedInpFeatureError,
    ) as caught:
        abaqus.read(path)

    assert "BEAM GENERAL SECTION" in (
        f"{caught.value} {caught.value.remediation}"
    ).upper()


def test_line_deck_unknown_engineering_keyword_is_default_denied(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "line_orientation_keyword.inp",
        _minimal_t3d2(
            extra_before_section=(
                "*Orientation, name=LOCAL",
                "1.0, 0.0, 0.0, 0.0, 1.0, 0.0",
            )
        ),
    )

    with pytest.raises(abaqus.UnsupportedInpFeatureError) as caught:
        abaqus.read(path)

    assert "ORIENTATION" in str(caught.value).upper()
    assert caught.value.remediation
