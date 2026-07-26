from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fem import abaqus
from fem.core.model import FEMModel
from tests.helpers.file_builders import write_inp


STANDARD = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)
B31_NOTICE = "abaqus.b31.euler_bernoulli_approximation"


@pytest.mark.parametrize(
    "fixture_name",
    (
        "b31_rect_explicit_n1_loads.inp",
        "b31_rect_default_n1.inp",
        "b31_circ.inp",
        "b31_thick_pipe.inp",
    ),
)
def test_each_successful_b31_read_reports_exactly_one_formulation_notice(
    fixture_name: str,
) -> None:
    source = STANDARD / fixture_name
    result = abaqus.read_with_report(source)

    assert isinstance(result, abaqus.AbaqusBuildResult)
    assert isinstance(result.model, FEMModel)
    assert isinstance(result.notices, tuple)
    assert len(result.notices) == 1

    notice = result.notices[0]
    assert isinstance(notice, abaqus.AbaqusImportNotice)
    assert notice.code == B31_NOTICE
    assert notice.locations
    assert all(Path(location.path) == source for location in notice.locations)
    assert all(location.line > 0 for location in notice.locations)
    assert any(
        "element" in (location.keyword or "").casefold()
        for location in notice.locations
    )

    message = notice.message.casefold()
    assert "b31" in message
    assert "timoshenko" in message
    assert "euler" in message
    assert "shear" in message
    assert any(
        token in message
        for token in ("short", "thick", "sensitive", "reproduce")
    )


def test_multiple_b31_elements_do_not_duplicate_formulation_notice():
    result = abaqus.read_with_report(
        STANDARD / "b31_rect_explicit_n1_loads.inp"
    )

    assert len(result.model.mesh.elements) == 2
    assert tuple(notice.code for notice in result.notices) == (
        B31_NOTICE,
    )


@pytest.mark.parametrize(
    "fixture_name",
    ("t3d2_tension.inp", "t3d2_default_area.inp"),
)
def test_t3d2_only_deck_has_no_b31_formulation_notice(
    fixture_name: str,
) -> None:
    result = abaqus.read_with_report(STANDARD / fixture_name)

    assert result.notices == ()


def test_build_model_with_report_matches_read_with_report_contract():
    source = STANDARD / "b31_circ.inp"
    deck = abaqus.parse_file(source)

    built = abaqus.build_model_with_report(deck)
    read = abaqus.read_with_report(source)

    assert isinstance(built, abaqus.AbaqusBuildResult)
    assert built.model.name == read.model.name
    assert built.model.mesh.num_nodes == read.model.mesh.num_nodes
    assert [
        (section.section_type, dict(section.properties))
        for section in built.model.sections
    ] == [
        (section.section_type, dict(section.properties))
        for section in read.model.sections
    ]
    assert tuple(notice.code for notice in built.notices) == (
        B31_NOTICE,
    )


@pytest.mark.parametrize(
    "function",
    (abaqus.build_model, abaqus.read),
)
def test_model_only_compatibility_wrappers_document_notice_discard(
    function,
) -> None:
    documentation = (inspect.getdoc(function) or "").casefold()

    assert "notice" in documentation
    assert any(
        token in documentation
        for token in ("discard", "drop", "丢弃")
    )


def test_model_only_compatibility_wrapper_still_returns_plain_model():
    model = abaqus.read(STANDARD / "b31_circ.inp")

    assert isinstance(model, FEMModel)
    assert not isinstance(model, abaqus.AbaqusBuildResult)


def test_b31_notice_is_immutable_and_owns_immutable_locations():
    result = abaqus.read_with_report(STANDARD / "b31_circ.inp")
    notice = result.notices[0]

    assert isinstance(notice.locations, tuple)
    with pytest.raises(FrozenInstanceError):
        notice.code = "changed"


def test_formulation_notice_does_not_leak_into_authoritative_model_data():
    model = abaqus.read_with_report(
        STANDARD / "b31_rect_explicit_n1_loads.inp"
    ).model
    authoritative_data = (
        model.metadata,
        tuple(
            (material.name, dict(material.properties))
            for material in model.materials.values()
        ),
        tuple(
            (
                section.section_type,
                dict(section.properties),
            )
            for section in model.sections
        ),
        tuple(dict(element.props) for element in model.mesh.elements),
    )

    serialized = repr(authoritative_data).casefold()
    assert B31_NOTICE not in serialized
    assert "euler_bernoulli_approximation" not in serialized


def test_short_thick_b31_is_reported_without_mesh_slenderness_block(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "short_thick_b31.inp",
        [
            "*Heading",
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 0.01, 0.0, 0.0",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2",
            "*Material, name=STEEL",
            "*Elastic",
            "2.10E11, 0.30",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.20, 0.10",
            "0.0, 1.0, 0.0",
        ],
    )

    result = abaqus.read_with_report(path)

    assert result.model.mesh.elements[0].type == "Beam2"
    assert tuple(notice.code for notice in result.notices) == (
        B31_NOTICE,
    )


def test_report_and_model_only_build_paths_share_the_same_model_contract():
    deck = abaqus.parse_file(STANDARD / "t3d2_tension.inp")

    reported = abaqus.build_model_with_report(deck)
    compatible = abaqus.build_model(deck)

    assert reported.notices == ()
    assert reported.model.mesh.num_dofs == compatible.mesh.num_dofs
    assert reported.model.sections == compatible.sections
    assert reported.model.steps == compatible.steps
