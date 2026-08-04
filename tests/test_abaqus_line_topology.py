from __future__ import annotations

from pathlib import Path

import pytest

from fem.io import inp as abaqus
from fem.application import RegionRef, resolve_effective_beam_frames
from tests.helpers.file_builders import write_inp


STANDARD = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)
FRAME_NOTICE_CODE = "abaqus.b31.nodal_normal_generation_approximation"


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


def _assert_frame_approximation_notice(result: abaqus.InpImportResult) -> None:
    notices = [
        notice for notice in result.notices if notice.code == FRAME_NOTICE_CODE
    ]
    assert len(notices) == 1
    notice = notices[0]
    assert notice.locations
    assert "element-end normals" in notice.message.casefold()
    assert "connectivity" in notice.message.casefold()
    assert "disconnect" not in notice.message.casefold()


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


def test_kinked_shared_node_uses_independent_element_frames(
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

    result = abaqus.read_with_report(path)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )

    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (2, 3))
    assert frames.passed
    assert frames.frames[0].local_x == pytest.approx((1.0, 0.0, 0.0))
    assert frames.frames[1].local_x == pytest.approx((0.0, 1.0, 0.0))
    _assert_frame_approximation_notice(result)


def test_branching_shared_node_preserves_one_shared_global_dof_identity(
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

    result = abaqus.read_with_report(path)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )

    assert result.model.mesh.num_nodes == 4
    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (2, 3), (2, 4))
    assert frames.passed
    assert frames.element_ids == (1, 2, 3)
    assert frames.frames[0].local_x == pytest.approx((1.0, 0.0, 0.0))
    assert frames.frames[2].local_x == pytest.approx((0.0, 1.0, 0.0))
    _assert_frame_approximation_notice(result)


def test_closed_b31_loop_preserves_connectivity_and_builds(
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

    result = abaqus.read_with_report(path)

    assert result.model.mesh.num_nodes == 3
    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (2, 3), (3, 1))
    _assert_frame_approximation_notice(result)


def test_reversed_element_connectivity_preserves_source_order_and_frame_direction(
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

    result = abaqus.read_with_report(path)
    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )

    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (3, 2))
    assert frames.passed
    assert frames.frames[0].local_x == pytest.approx((1.0, 0.0, 0.0))
    assert frames.frames[1].local_x == pytest.approx((-1.0, 0.0, 0.0))
    _assert_frame_approximation_notice(result)


def test_shared_node_allows_assignment_scoped_section_orientations(
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

    result = abaqus.read_with_report(path)
    left_frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "LEFT"),
    )
    right_frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "RIGHT"),
    )

    assert left_frames.passed
    assert right_frames.passed
    assert left_frames.frames[0].source == "explicit"
    assert right_frames.frames[0].source == "explicit"
    assert left_frames.frames[0].local_y == pytest.approx((0.0, 1.0, 0.0))
    assert right_frames.frames[0].local_y == pytest.approx((0.0, 0.0, 1.0))
    _assert_frame_approximation_notice(result)


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
    with pytest.raises(abaqus.InpInputError) as caught:
        abaqus.read_with_report(path)

    assert caught.value.code == "beam.orientation.parallel"
    assert "2" in str(caught.value)
    assert caught.value.path == path
    assert caught.value.keyword in {"element", "beam section"}
    assert caught.value.locations


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

    result = abaqus.read_with_report(path)

    assert tuple(result.model.mesh.elements[0].node_ids) == (1, 2)
    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_dofs == 12


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

    result = abaqus.read_with_report(path)

    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert result.source_summary is not None
    assert any(
        occurrence.name == "node"
        for occurrence in result.source_summary.occurrences
    )


def test_normal_keyword_is_supported_for_b31_source(
    tmp_path,
) -> None:
    lines = _single_section_deck(
        (
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
        ),
        ((1, 1, 2),),
    )
    lines.extend(
        (
            "*Normal",
            "1, 1, 0.0, 1.0, 0.0",
            "1, 2, 0.0, 1.0, 0.0",
        )
    )
    path = write_inp(tmp_path, "normal_keyword.inp", lines)

    result = abaqus.read_with_report(path)

    assert result.model.mesh.num_nodes == 2
    assert result.model.mesh.num_elements == 1
    assert result.source_summary is not None
    assert sum(
        occurrence.name == "normal"
        for occurrence in result.source_summary.occurrences
    ) == 1
