from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fem import abaqus
from fem.application import RegionRef, resolve_effective_beam_frames
from tests.helpers.file_builders import write_inp


STANDARD = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)
TOPOLOGY_CODE = "abaqus.b31.nodal_normal_averaging_unsupported"


def _single_section_deck(
    nodes: tuple[tuple[int, float, float, float], ...],
    elements: tuple[tuple[int, int, int], ...],
    *,
    orientation: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> list[str]:
    return [
        "*Heading",
        "*Node",
        *(
            f"{node_id}, {x}, {y}, {z}"
            for node_id, x, y, z in nodes
        ),
        "*Element, type=B31, elset=BEAM",
        *(
            f"{element_id}, {node_i}, {node_j}"
            for element_id, node_i, node_j in elements
        ),
        "*Material, name=STEEL",
        "*Elastic",
        "2.10E11, 0.30",
        "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
        "0.20, 0.10",
        ", ".join(str(value) for value in orientation),
    ]


def _assert_topology_rejection(
    path: Path,
    *,
    element_ids: tuple[int, ...],
    node_ids: tuple[int, ...],
) -> None:
    with pytest.raises(abaqus.UnsupportedAbaqusFeatureError) as caught:
        abaqus.read_with_report(path)

    error = caught.value
    assert error.code == TOPOLOGY_CODE
    assert Path(error.path) == path
    assert error.line > 0
    assert error.keyword
    assert error.remediation
    assert (
        "straight" in error.remediation.casefold()
        or "直" in error.remediation
    )
    assert error.locations
    assert all(Path(location.path) == path for location in error.locations)

    diagnostic = repr(
        (
            str(error),
            error.record,
            getattr(error, "details", None),
        )
    )
    for element_id in element_ids:
        assert str(element_id) in diagnostic
    for node_id in node_ids:
        assert str(node_id) in diagnostic


@pytest.mark.parametrize(
    "fixture_name",
    (
        "b31_rect_explicit_n1_loads.inp",
        "b31_rect_default_n1.inp",
    ),
)
def test_literal_isolated_or_directed_straight_b31_fixtures_are_accepted(
    fixture_name: str,
) -> None:
    result = abaqus.read_with_report(STANDARD / fixture_name)

    assert {element.type for element in result.model.mesh.elements} == {
        "Beam2"
    }


def test_disconnected_straight_b31_members_are_accepted(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "disconnected_straight_members.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 0.0, 2.0, 0.0),
                (4, 1.0, 2.0, 0.0),
            ),
            ((1, 1, 2), (2, 3, 4)),
        ),
    )

    result = abaqus.read_with_report(path)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )

    assert frames.passed
    assert frames.element_ids == (1, 2)
    assert all(
        frame.local_x == pytest.approx((1.0, 0.0, 0.0))
        for frame in frames.frames
    )


def test_collinear_directed_open_chain_is_accepted(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "straight_chain.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 2.0, 0.0, 0.0),
                (4, 3.0, 0.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3), (3, 3, 4)),
        ),
    )

    result = abaqus.read_with_report(path)

    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (2, 3), (3, 4))


def test_kinked_shared_node_requires_unsupported_normal_averaging(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "kinked_chain.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3)),
            orientation=(0.0, 0.0, 1.0),
        ),
    )

    _assert_topology_rejection(
        path,
        element_ids=(1, 2),
        node_ids=(2,),
    )


def test_branching_shared_node_requires_unsupported_normal_averaging(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "branching_beam.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 2.0, 0.0, 0.0),
                (4, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3), (3, 2, 4)),
            orientation=(0.0, 0.0, 1.0),
        ),
    )

    _assert_topology_rejection(
        path,
        element_ids=(1, 2, 3),
        node_ids=(2,),
    )


def test_closed_b31_loop_requires_unsupported_normal_averaging(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "closed_beam_loop.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 0.5, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3), (3, 3, 1)),
            orientation=(0.0, 0.0, 1.0),
        ),
    )

    _assert_topology_rejection(
        path,
        element_ids=(1, 2, 3),
        node_ids=(1, 2, 3),
    )


def test_reversed_element_connectivity_rejects_discontinuous_frames(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "reversed_connectivity.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 2.0, 0.0, 0.0),
            ),
            ((1, 1, 2), (2, 3, 2)),
        ),
    )

    _assert_topology_rejection(
        path,
        element_ids=(1, 2),
        node_ids=(2,),
    )


def test_shared_node_rejects_incompatible_section_orientations(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "incompatible_shared_frames.inp",
        [
            "*Heading",
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "3, 2.0, 0.0, 0.0",
            "*Element, type=B31, elset=LEFT",
            "1, 1, 2",
            "*Element, type=B31, elset=RIGHT",
            "2, 2, 3",
            "*Material, name=STEEL",
            "*Elastic",
            "2.10E11, 0.30",
            "*Beam Section, elset=LEFT, material=STEEL, section=RECT",
            "0.20, 0.10",
            "0.0, 1.0, 0.0",
            "*Beam Section, elset=RIGHT, material=STEEL, section=RECT",
            "0.20, 0.10",
            "0.0, 0.0, 1.0",
        ],
    )

    _assert_topology_rejection(
        path,
        element_ids=(1, 2),
        node_ids=(2,),
    )


def test_orientation_parallel_to_later_target_element_fails_transactionally(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "parallel_later_element.inp",
        _single_section_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3)),
            orientation=(0.0, 1.0, 0.0),
        ),
    )
    deck = abaqus.parse_file(path)
    snapshot = deepcopy(deck)

    with pytest.raises(abaqus.AbaqusInputError) as caught:
        abaqus.build_model_with_report(deck)

    assert caught.value.code == "beam.orientation.parallel"
    assert "2" in str(caught.value)
    assert caught.value.remediation
    assert deck == snapshot


def test_additional_b31_orientation_node_is_not_truncated(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "additional_orientation_node.inp",
        [
            "*Heading",
            "*Node",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "3, 0.0, 1.0, 0.0",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2, 3",
            "*Material, name=STEEL",
            "*Elastic",
            "2.10E11, 0.30",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.20, 0.10",
            "0.0, 1.0, 0.0",
        ],
    )

    with pytest.raises(
        abaqus.UnsupportedAbaqusFeatureError
    ) as caught:
        abaqus.read_with_report(path)

    assert "additional orientation node" in str(caught.value).casefold()
    assert "n1" in caught.value.remediation.casefold()
    assert caught.value.record


def test_node_normal_components_are_not_silently_discarded(
    tmp_path,
) -> None:
    path = write_inp(
        tmp_path,
        "node_normal_components.inp",
        [
            "*Heading",
            "*Node",
            "1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0",
            "2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0",
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

    with pytest.raises(
        abaqus.UnsupportedAbaqusFeatureError
    ) as caught:
        abaqus.read_with_report(path)

    assert "normal" in str(caught.value).casefold()
    assert caught.value.line == 3
    assert caught.value.remediation


def test_normal_keyword_is_explicitly_rejected_for_b31_source(
    tmp_path,
) -> None:
    lines = _single_section_deck(
        (
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
        ),
        ((1, 1, 2),),
    )
    lines.extend(("*Normal", "1, 1, 0.0, 1.0, 0.0"))
    path = write_inp(tmp_path, "normal_keyword.inp", lines)

    with pytest.raises(
        abaqus.UnsupportedAbaqusFeatureError
    ) as caught:
        abaqus.read_with_report(path)

    assert "NORMAL" in str(caught.value).upper()
    assert caught.value.remediation
