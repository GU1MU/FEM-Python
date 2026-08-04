from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import numpy as np
import pytest

from fem import abaqus
from fem.assemble import assemble_global_stiffness
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.abaqus.deck import (
    AbaqusBeamSectionData,
    AbaqusDataRecordEvidence,
    AbaqusDeck,
    AbaqusElement,
    AbaqusNodeNormalRecord,
    AbaqusNodeRecord,
    AbaqusNormalRecord,
    AbaqusSection,
    AbaqusSourceSpan,
)
from fem.abaqus.errors import AbaqusSourceLocation
from fem.abaqus.orientation import (
    AbaqusOrientationPolicy,
    resolve_b31_orientations,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import AnalysisStep, FEMModel, LineLoad
from fem.elements import (
    BEAM_LOCAL_Y_REFERENCE_KEY,
    get_element_kernel,
    resolve_beam_frame,
)
from tests.helpers.file_builders import write_inp


def _span(line: int, keyword: str) -> AbaqusSourceSpan:
    location = AbaqusSourceLocation(None, line, keyword)
    return AbaqusSourceSpan(location, location, (location,))


def _deck(
    nodes: dict[int, tuple[float, float, float]],
    elements: list[AbaqusElement],
    *,
    node_records: dict[int, AbaqusNodeRecord] | None = None,
    normal_records: list[AbaqusNormalRecord] | None = None,
    element_sets: dict[str, list[int]] | None = None,
    node_sets: dict[str, list[int]] | None = None,
    sections: list[AbaqusSection] | None = None,
) -> AbaqusDeck:
    return AbaqusDeck(
        name="phase4",
        nodes=nodes,
        node_records={} if node_records is None else node_records,
        elements=elements,
        element_sets={} if element_sets is None else element_sets,
        node_sets={} if node_sets is None else node_sets,
        sections=[] if sections is None else sections,
        normal_records=[] if normal_records is None else normal_records,
    )


def _star(angles: tuple[float, ...]) -> AbaqusDeck:
    nodes = {0: (0.0, 0.0, 0.0)}
    elements = []
    for index, angle in enumerate(angles, start=1):
        node_id = index
        nodes[node_id] = (cos(radians(angle)), sin(radians(angle)), 0.0)
        elements.append(AbaqusElement(index, (0, node_id), "B31"))
    return _deck(
        nodes,
        elements,
        element_sets={"BEAMS": [element.id for element in elements]},
        node_sets={"CENTER": [0]},
    )


def _field_signature(result) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            entry.identity,
            entry.tangent,
            entry.n1,
            entry.normal,
            entry.resolution_kind,
            entry.normal_group,
        )
        for entry in result.field.entries
    )


def _beam2_mesh(*, reversed_connectivity: bool = False) -> Mesh3D:
    return Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 2.0, 3.0, 6.0)],
        elements=[
            Element3D(
                10,
                [2, 1] if reversed_connectivity else [1, 2],
                "Beam2",
                {
                    "E": 210.0,
                    "nu": 0.25,
                    "section_type": "rectangle",
                    "height": 3.0,
                    "width": 2.0,
                    BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
                },
            )
        ],
        dofs_per_node=6,
    )


def _line_load_vector(
    mesh: Mesh3D,
    vector: tuple[float, float, float],
    coordinate_system: str,
) -> np.ndarray:
    model = FEMModel(
        mesh=mesh,
        steps=[
            AnalysisStep(
                "LOAD",
                line_loads=(LineLoad(10, vector, coordinate_system),),
            )
        ],
    )
    return build_load_vector(mesh, boundary_for_step(model, "LOAD"))


def test_isolated_default_field_is_unit_right_handed_and_stably_sorted() -> None:
    result = resolve_b31_orientations(
        _deck(
            {2: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)},
            [AbaqusElement(20, (1, 2), "B31")],
        )
    )

    assert result.report.ok
    assert tuple(entry.identity.local_end for entry in result.field.entries) == (1, 2)
    assert {entry.resolution_kind for entry in result.field.entries} == {
        "default-generated"
    }
    for entry in result.field.entries:
        np.testing.assert_allclose(np.linalg.norm(entry.tangent), 1.0)
        np.testing.assert_allclose(np.linalg.norm(entry.n1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(entry.normal), 1.0)
        np.testing.assert_allclose(np.cross(entry.tangent, entry.n1), entry.normal)


def test_explicit_normal_precedes_node_normal_and_orientation_node_precedes_section_n1() -> None:
    section = AbaqusSection(
        "BEAMS",
        "STEEL",
        "beam",
        element_ids=(1,),
        data=AbaqusBeamSectionData(
            "RECT",
            (0.2, 0.1),
            (0.0, 1.0, 0.0),
            AbaqusDataRecordEvidence.missing(),
            AbaqusDataRecordEvidence.missing(),
        ),
    )
    node_records = {
        node_id: AbaqusNodeRecord(
            node_id,
            coordinates,
            normal=AbaqusNodeNormalRecord((0.0, 0.0, 1.0), node_id),
        )
        for node_id, coordinates in {
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
        }.items()
    }
    result = resolve_b31_orientations(
        _deck(
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (0.0, 0.0, 1.0),
            },
            [AbaqusElement(1, (1, 2), "B31", orientation_node_id=3)],
            node_records=node_records,
            normal_records=[
                AbaqusNormalRecord(
                    1,
                    1,
                    (0.0, 0.0, 1.0),
                    _span(40, "normal"),
                )
            ],
            element_sets={"BEAMS": [1]},
            sections=[section],
        )
    )

    assert result.report.ok
    first, second = result.field.entries
    assert first.normal_source == "element-normal"
    assert second.normal_source == "node-normal"
    assert first.reference_source == second.reference_source == "orientation-node"
    assert first.n1 == pytest.approx((0.0, 1.0, 0.0))
    assert second.n1 == pytest.approx((0.0, 1.0, 0.0))
    assert len(result.report.explicit) == 2


def test_continuous_group_uses_official_pairwise_20_degree_average() -> None:
    result = resolve_b31_orientations(_star((0.0, 10.0, 15.0)))

    center_entries = [entry for entry in result.field.entries if entry.node_id == 0]
    assert len(center_entries) == 3
    assert {entry.resolution_kind for entry in center_entries} == {"averaged"}
    assert len({entry.normal_group for entry in center_entries}) == 1
    assert len(result.report.averaged) == 1
    assert result.report.averaged[0].identities == tuple(
        entry.identity for entry in center_entries
    )
    assert result.report.groups[0].averaged


def test_official_disjoint_groups_average_two_and_keep_one_independent() -> None:
    result = resolve_b31_orientations(_star((0.0, 10.0, 40.0)))

    center = [entry for entry in result.field.entries if entry.node_id == 0]
    assert [entry.resolution_kind for entry in center] == [
        "averaged",
        "averaged",
        "split-group",
    ]
    assert len(result.report.averaged) == 1
    assert result.report.averaged[0].identities == tuple(
        entry.identity for entry in center[:2]
    )
    groups = [group for group in result.report.groups if group.node_id == 0]
    assert [len(group.identities) for group in groups] == [2, 1]
    assert groups[0].averaged
    assert not groups[1].averaged
    assert groups[1].split_reason == "disjoint-continuity-group"
    assert result.report.split_groups


def test_non_clique_continuity_group_is_split_without_element_order_dependence() -> None:
    forward = resolve_b31_orientations(_star((0.0, 10.0, 30.0)))
    reversed_input = _star((0.0, 10.0, 30.0))
    reversed_input.elements.reverse()
    permuted = resolve_b31_orientations(reversed_input)

    center = [entry for entry in forward.field.entries if entry.node_id == 0]
    assert {entry.resolution_kind for entry in center} == {"split-group"}
    assert len({entry.normal_group for entry in center}) == 3
    assert forward.report.split_groups
    assert _field_signature(forward) == _field_signature(permuted)


def test_kink_keeps_independent_element_end_normal_groups() -> None:
    result = resolve_b31_orientations(
        _deck(
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (1.0, 1.0, 0.0),
            },
            [AbaqusElement(1, (1, 2), "B31"), AbaqusElement(2, (2, 3), "B31")],
        )
    )

    center = [entry for entry in result.field.entries if entry.node_id == 2]
    assert len(center) == 2
    assert {entry.resolution_kind for entry in center} == {"split-group"}
    assert len({entry.normal_group for entry in center}) == 2
    assert result.report.split_groups


def test_t_junction_keeps_shared_node_element_end_identity() -> None:
    result = resolve_b31_orientations(
        _deck(
            {
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
                3: (2.0, 0.0, 0.0),
                4: (1.0, 1.0, 0.0),
            },
            [
                AbaqusElement(1, (1, 2), "B31"),
                AbaqusElement(2, (2, 3), "B31"),
                AbaqusElement(3, (2, 4), "B31"),
            ],
        )
    )

    center = [entry for entry in result.field.entries if entry.node_id == 2]
    assert len(center) == 3
    assert len({entry.identity for entry in center}) == 3
    assert len({entry.normal_group for entry in center}) == 2
    assert {entry.resolution_kind for entry in center} == {
        "averaged",
        "split-group",
    }


def test_more_than_thirty_remaining_elements_are_kept_as_separate_groups() -> None:
    result = resolve_b31_orientations(
        _star(tuple(0.0 for _ in range(31)))
    )

    center = [entry for entry in result.field.entries if entry.node_id == 0]
    assert len(center) == 31
    assert {entry.resolution_kind for entry in center} == {"split-group"}
    assert result.report.split_groups[0].record["reason"] == (
        "more-than-30-remaining-elements"
    )


def test_loop_and_multibranch_topology_are_element_end_owned() -> None:
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (2.0, 0.0, 0.0),
    }
    elements = [
        AbaqusElement(10, (1, 2), "B31"),
        AbaqusElement(20, (2, 3), "B31"),
        AbaqusElement(30, (3, 4), "B31"),
        AbaqusElement(40, (4, 1), "B31"),
        AbaqusElement(50, (2, 5), "B31"),
    ]
    result = resolve_b31_orientations(_deck(nodes, elements))

    assert result.report.ok
    identities = tuple(entry.identity for entry in result.field.entries)
    assert len(identities) == 10
    assert len(set(identity.node_id for identity in identities)) == 5
    assert any(
        len({entry.normal_group for entry in result.field.for_element(element.id)})
        >= 1
        for element in elements
    )


def test_input_permutation_and_connectivity_reversal_are_covariant() -> None:
    original = resolve_b31_orientations(
        _deck(
            {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
            [AbaqusElement(7, (1, 2), "B31")],
        )
    )
    reversed_result = resolve_b31_orientations(
        _deck(
            {2: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)},
            [AbaqusElement(7, (2, 1), "B31")],
        )
    )

    first = original.field.entries[0]
    second = reversed_result.field.entries[0]
    assert second.identity.node_id == 2
    assert second.n1 == pytest.approx(first.n1)
    assert second.tangent == pytest.approx(tuple(-value for value in first.tangent))
    assert second.normal == pytest.approx(tuple(-value for value in first.normal))


def test_connectivity_reversal_covaries_global_stiffness_and_local_global_loads() -> None:
    forward = _beam2_mesh()
    reversed_mesh = _beam2_mesh(reversed_connectivity=True)
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]

    forward_stiffness = assemble_global_stiffness(forward)
    reversed_stiffness = assemble_global_stiffness(reversed_mesh)
    kernel = get_element_kernel("Beam2")
    forward_element_stiffness = kernel.stiffness(forward, forward.elements[0])
    reversed_element_stiffness = kernel.stiffness(
        reversed_mesh,
        reversed_mesh.elements[0],
    )
    np.testing.assert_allclose(
        reversed_element_stiffness,
        permutation @ forward_element_stiffness @ permutation.T,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        reversed_stiffness,
        forward_stiffness,
        rtol=1e-10,
        atol=1e-10,
    )

    global_vector = (1.5, -2.0, 0.25)
    frame = resolve_beam_frame(forward, forward.elements[0])
    local_vector = tuple(frame.rotation @ np.asarray(global_vector, dtype=float))
    local_axis_reversal = np.diag((-1.0, 1.0, -1.0))

    forward_global = _line_load_vector(forward, global_vector, "global")
    forward_local = _line_load_vector(forward, local_vector, "local")
    reversed_global = _line_load_vector(reversed_mesh, global_vector, "global")
    reversed_local = _line_load_vector(
        reversed_mesh,
        tuple(local_axis_reversal @ np.asarray(local_vector)),
        "local",
    )
    np.testing.assert_allclose(forward_global, forward_local, atol=1e-12)
    np.testing.assert_allclose(reversed_global, forward_global, atol=1e-12)
    np.testing.assert_allclose(reversed_local, forward_local, atol=1e-12)


def test_custom_frame_tolerance_is_carried_into_resolved_entries() -> None:
    policy = AbaqusOrientationPolicy(frame_tolerance=1.0e-5)
    result = resolve_b31_orientations(
        _deck(
            {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
            [AbaqusElement(1, (1, 2), "B31")],
        ),
        policy=policy,
    )

    assert {entry.frame_tolerance for entry in result.field.entries} == {1.0e-5}


def test_set_and_mapping_order_permutation_is_stable() -> None:
    normal = AbaqusNormalRecord(
        "BEAMS",
        "CENTER",
        (0.0, 0.0, 1.0),
        _span(22, "normal"),
    )
    forward = resolve_b31_orientations(
        _deck(
            {
                0: (0.0, 0.0, 0.0),
                1: (1.0, 0.0, 0.0),
                2: (0.0, 1.0, 0.0),
            },
            [AbaqusElement(2, (0, 2), "B31"), AbaqusElement(1, (0, 1), "B31")],
            normal_records=[normal],
            element_sets={"BEAMS": [2, 1], "UNUSED": [99]},
            node_sets={"CENTER": [0], "UNUSED_NODES": [99]},
        )
    )
    permuted = resolve_b31_orientations(
        _deck(
            {
                2: (0.0, 1.0, 0.0),
                1: (1.0, 0.0, 0.0),
                0: (0.0, 0.0, 0.0),
            },
            [AbaqusElement(1, (0, 1), "B31"), AbaqusElement(2, (0, 2), "B31")],
            normal_records=[normal],
            element_sets={"UNUSED": [99], "BEAMS": [1, 2]},
            node_sets={"UNUSED_NODES": [99], "CENTER": [0]},
        )
    )

    assert _field_signature(forward) == _field_signature(permuted)
    assert tuple(
        (item.kind, item.code, item.identities, item.normal_group)
        for item in forward.report.events
    ) == tuple(
        (item.kind, item.code, item.identities, item.normal_group)
        for item in permuted.report.events
    )


def test_conflict_is_reported_and_phase5_variation_is_legal() -> None:
    conflict = resolve_b31_orientations(
        _deck(
            {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
            [AbaqusElement(1, (1, 2), "B31")],
            normal_records=[
                AbaqusNormalRecord(1, 1, (0.0, 0.0, 1.0), _span(8, "normal")),
                AbaqusNormalRecord(1, 1, (0.0, 1.0, 0.0), _span(9, "normal")),
            ],
        )
    )
    assert conflict.report.conflicts
    assert {location.line for location in conflict.report.conflicts[0].locations} == {
        8,
        9,
    }
    with pytest.raises(abaqus.AbaqusBuildError) as caught:
        conflict.raise_if_invalid()
    assert caught.value.code == "abaqus.b31.normal.conflict"

    varying = resolve_b31_orientations(
        _deck(
            {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)},
            [AbaqusElement(1, (1, 2), "B31")],
            normal_records=[
                AbaqusNormalRecord(1, 1, (0.0, 0.0, 1.0), _span(12, "normal")),
                AbaqusNormalRecord(1, 2, (0.0, 1.0, 0.0), _span(13, "normal")),
            ],
        )
    )
    assert varying.report.ok
    assert not varying.report.unsupported_variations
    assert varying.field.varies_by_element()


def test_builder_consumes_orientation_node_field(tmp_path: Path) -> None:
    path = write_inp(
        tmp_path,
        "phase4_orientation_node.inp",
        [
            "*Heading",
            "*Node",
            "1, 0., 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 0., 1.",
            "*Element, type=B31, elset=BEAM",
            "1, 1, 2, 3",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.2, 0.1",
            "0., 1., 0.",
        ],
    )

    result = abaqus.read_with_report(path)
    element = result.model.mesh.elements[0]
    assert element.props["beam_element_local_y_reference"] == pytest.approx(
        (0.0, 0.0, 1.0)
    )
