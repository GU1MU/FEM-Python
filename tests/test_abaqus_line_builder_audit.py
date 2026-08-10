from __future__ import annotations

import pytest

from fem.io import inp as abaqus
from tests.helpers.file_builders import write_inp


def _material() -> list[str]:
    return [
        "*Material, name=STEEL",
        "*Elastic",
        "2.10E11, 0.30",
    ]


def _mixed_line_deck(*, shared_set: bool) -> list[str]:
    first_set = "MIX" if shared_set else "BEAM"
    second_set = "MIX" if shared_set else "TRUSS"
    return [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "3, 2., 0., 0.",
        f"*Element, type=B31, elset={first_set}",
        "1, 1, 2",
        f"*Element, type=T3D2, elset={second_set}",
        "2, 2, 3",
        *_material(),
    ]


def test_mixed_line_section_target_fails_before_mesh_build(tmp_path) -> None:
    lines = [
        *_mixed_line_deck(shared_set=True),
        "*Beam Section, elset=MIX, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 1., 0.",
    ]
    path = write_inp(tmp_path, "mixed_section.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == "abaqus.target.family_mixed"
    assert error.line == lines.index(
        "*Beam Section, elset=MIX, material=STEEL, section=RECT"
    ) + 1
    assert error.keyword == "beam section"


def test_mixed_line_dload_target_fails_before_mesh_build(tmp_path) -> None:
    lines = [
        *_mixed_line_deck(shared_set=True),
        "*Step, name=LOAD",
        "*Static",
        "*Dload",
        "MIX, P1, -4.",
        "*End Step",
    ]
    path = write_inp(tmp_path, "mixed_dload.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == "abaqus.target.family_mixed"
    assert error.line == lines.index("MIX, P1, -4.") + 1
    assert error.keyword == "dload"


def test_whole_mixed_model_fails_with_typed_mesh_capability(tmp_path) -> None:
    lines = [
        *_mixed_line_deck(shared_set=False),
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 1., 0.",
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
    ]
    path = write_inp(tmp_path, "mixed_model.inp", lines)

    with pytest.raises(abaqus.UnsupportedInpFeatureError) as caught:
        abaqus.read(path)

    assert caught.value.code == "abaqus.mesh.element_family_mixed"


@pytest.mark.parametrize(
    ("procedure_lines", "code", "offending"),
    (
        ((), "abaqus.line.step.static_missing", "*Step, name=LOAD"),
        (
            ("*Static", "*Static"),
            "abaqus.line.step.static_count",
            "*Static",
        ),
    ),
)
def test_explicit_line_step_requires_exactly_one_static(
    tmp_path,
    procedure_lines: tuple[str, ...],
    code: str,
    offending: str,
) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=T3D2, elset=TRUSS",
        "1, 1, 2",
        *_material(),
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
        "*Step, name=LOAD",
        *procedure_lines,
        "*End Step",
    ]
    path = write_inp(tmp_path, "step_procedure.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == code
    if code.endswith("static_count"):
        assert error.line == max(
            index + 1
            for index, line in enumerate(lines)
            if line == offending
        )
    else:
        assert error.line == lines.index(offending) + 1


def test_static_outside_explicit_step_is_rejected(tmp_path) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=T3D2, elset=TRUSS",
        "1, 1, 2",
        *_material(),
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
        "*Static",
    ]
    path = write_inp(tmp_path, "static_outside_step.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == "abaqus.line.step.static_outside_step"
    assert error.line == lines.index("*Static") + 1
    assert error.keyword == "static"


@pytest.mark.parametrize(
    ("material_lines", "code", "offending"),
    (
        (
            ("*Material, name=STEEL",),
            "abaqus.line.material.elastic_missing",
            "*Material, name=STEEL",
        ),
        (
            (
                "*Material, name=STEEL",
                "*Elastic",
                "2.10E11, 0.30, 20.",
            ),
            "abaqus.line.material.elastic_shape",
            "2.10E11, 0.30, 20.",
        ),
        (
            (
                "*Material, name=STEEL",
                "*Elastic",
                "2.10E11, 0.30",
                "2.00E11, 0.29",
            ),
            "abaqus.line.material.elastic_record_count",
            "2.00E11, 0.29",
        ),
        (
            (
                "*Material, name=STEEL",
                "*Elastic",
                "2.10E11, 0.30",
                "*Density",
                "7850., 20.",
            ),
            "abaqus.line.material.density_shape",
            "7850., 20.",
        ),
        (
            (
                "*Material, name=STEEL",
                "*Elastic",
                "2.10E11, 0.30",
                "*Density",
                "7850.",
                "7800.",
            ),
            "abaqus.line.material.density_record_count",
            "7800.",
        ),
    ),
)
def test_line_material_must_be_one_constant_record(
    tmp_path,
    material_lines: tuple[str, ...],
    code: str,
    offending: str,
) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=T3D2, elset=TRUSS",
        "1, 1, 2",
        *material_lines,
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
    ]
    path = write_inp(tmp_path, "material_table.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == code
    assert error.line == lines.index(offending) + 1


@pytest.mark.parametrize(
    ("material_lines", "code"),
    (
        ((), "definition.material.missing"),
        (
            (
                "*Material, name=STEEL",
                "*Elastic",
                "-1., 0.30",
            ),
            "definition.material.invalid",
        ),
    ),
)
def test_continuum_declared_sections_use_canonical_material_validation(
    tmp_path,
    material_lines: tuple[str, ...],
    code: str,
) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "3, 0., 1., 0.",
        "4, 0., 0., 1.",
        "*Element, type=C3D4, elset=SOLID",
        "1, 1, 2, 3, 4",
        *material_lines,
        "*Solid Section, elset=SOLID, material=STEEL",
    ]
    path = write_inp(tmp_path, "continuum_material.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == code
    assert error.keyword == "solid section"


def test_definition_time_empty_set_does_not_fall_through(tmp_path) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=T3D2",
        "1, 1, 2",
        "*Elset, elset=TRUSS",
        *_material(),
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
        "*Elset, elset=TRUSS",
        "1",
    ]
    path = write_inp(tmp_path, "empty_then_redefined.inp", lines)

    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == "abaqus.section.target_empty"
    assert error.line == lines.index(
        "*Solid Section, elset=TRUSS, material=STEEL"
    ) + 1


def test_forward_defined_section_set_is_resolved(tmp_path) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=T3D2",
        "1, 1, 2",
        *_material(),
        "*Solid Section, elset=TRUSS, material=STEEL",
        "0.01",
        "*Elset, elset=TRUSS",
        "1",
    ]
    path = write_inp(tmp_path, "forward_set.inp", lines)

    model = abaqus.read(path)

    assert model.sections[0].element_set == "TRUSS"
    assert model.element_sets["TRUSS"].element_ids == (1,)


def test_parallel_orientation_locations_cover_all_source_evidence(
    tmp_path,
) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        *_material(),
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "1., 0., 0.",
    ]
    path = write_inp(tmp_path, "parallel_orientation.inp", lines)
    with pytest.raises(abaqus.InpBuildError) as caught:
        abaqus.read(path)

    error = caught.value
    assert error.code == "beam.orientation.parallel"
    assert [location.line for location in error.locations] == [
        lines.index("*Element, type=B31, elset=BEAM") + 1,
        lines.index("1, 1, 2") + 1,
        lines.index("1., 0., 0.") + 1,
    ]


def test_kinked_frame_notice_is_structured_and_does_not_recommend_disconnect(
    tmp_path,
) -> None:
    lines = [
        "*Node",
        "1, 0., 0., 0.",
        "2, 1., 0., 0.",
        "3, 1., 1., 0.",
        "*Element, type=B31, elset=BEAM",
        "1, 1, 2",
        "2, 2, 3",
        *_material(),
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.2, 0.1",
        "0., 0., 1.",
    ]
    path = write_inp(tmp_path, "kink_locations.inp", lines)

    result = abaqus.read_with_report(path)

    assert tuple(
        notice.code for notice in result.notices
    ) == (
        "abaqus.b31.linear_timoshenko_support_boundary",
        "abaqus.b31.nodal_normal_generation_approximation",
    )
    notice = result.notices[1]
    assert notice.locations
    assert "element-end normals" in notice.message.casefold()
    assert "connectivity" in notice.message.casefold()
    assert "disconnect" not in notice.message.casefold()
