from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fem import abaqus
from fem.application import RegionRef, resolve_effective_beam_frames
from fem.assemble import assemble_global_stiffness
from fem.boundary.loads import build_load_vector
from fem.boundary.step import boundary_for_step
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    SectionAssignment,
)
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY, get_element_kernel
from fem.materials import apply_sections
from fem.solvers.static_linear import solve
from tests.helpers.file_builders import write_inp


FRAME_NOTICE = "abaqus.b31.nodal_normal_generation_approximation"


def _b31_deck(
    nodes: tuple[tuple[int, float, float, float], ...],
    elements: tuple[tuple[int, int, int], ...],
    *,
    orientation: tuple[float, float, float] | None = (0.0, 0.0, 1.0),
    orientation_record: str | None = None,
    fixed_nodes: tuple[int, ...] = (),
    tip_nodes: tuple[int, ...] = (),
    cloads: tuple[tuple[int, int, float], ...] = (),
    dloads: tuple[str, ...] = (),
) -> list[str]:
    lines = [
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
    ]
    if fixed_nodes:
        lines.extend(("*Nset, nset=FIXED", ", ".join(map(str, fixed_nodes))))
    if tip_nodes:
        lines.extend(("*Nset, nset=TIP", ", ".join(map(str, tip_nodes))))
    lines.extend(
        (
            "*Material, name=STEEL",
            "*Elastic",
            "210000000000.0, 0.30",
            "*Beam Section, elset=BEAM, material=STEEL, section=RECT",
            "0.20, 0.10",
        )
    )
    if orientation_record is not None:
        lines.append(orientation_record)
    elif orientation is not None:
        lines.append(", ".join(str(value) for value in orientation))
    if fixed_nodes or cloads or dloads:
        lines.extend(("*Step, name=LOAD", "*Static"))
        if fixed_nodes:
            lines.extend(("*Boundary", "FIXED, 1, 6, 0.0"))
        if cloads:
            lines.extend(
                (
                    "*Cload",
                    *(
                        f"TIP, {component}, {value}"
                        for _node_id, component, value in cloads
                    ),
                )
            )
        if dloads:
            lines.extend(("*Dload", *dloads))
        lines.append("*End Step")
    return lines


def _read_deck(tmp_path, name: str, lines: list[str]):
    return abaqus.read_with_report(write_inp(tmp_path, name, lines))


def test_noncollinear_two_span_b31_imports_and_solves(tmp_path) -> None:
    result = _read_deck(
        tmp_path,
        "two_span_kink.inp",
        _b31_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3)),
            fixed_nodes=(1,),
            tip_nodes=(3,),
            cloads=((3, 3, 10.0),),
        ),
    )

    frames = resolve_effective_beam_frames(
        result.model,
        RegionRef("element_set", "BEAM"),
    )
    displacement = solve(result.model, "LOAD").U

    assert frames.passed
    assert frames.frames[0].local_x == pytest.approx((1.0, 0.0, 0.0))
    assert frames.frames[1].local_x == pytest.approx((0.0, 1.0, 0.0))
    assert np.all(np.isfinite(displacement))
    assert np.linalg.norm(displacement) > 0.0
    assert tuple(notice.code for notice in result.notices) == (
        "abaqus.b31.euler_bernoulli_approximation",
        FRAME_NOTICE,
    )


def test_t_junction_preserves_shared_node_and_solves(tmp_path) -> None:
    result = _read_deck(
        tmp_path,
        "t_junction.inp",
        _b31_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 2.0, 0.0, 0.0),
                (4, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3), (3, 2, 4)),
            fixed_nodes=(1,),
            tip_nodes=(4,),
            cloads=((4, 3, 10.0),),
        ),
    )

    result_value = solve(result.model, "LOAD")
    shared = [
        element
        for element in result.model.mesh.elements
        if 2 in element.node_ids
    ]

    assert result.model.mesh.num_nodes == 4
    assert len(shared) == 3
    assert tuple(
        tuple(element.node_ids) for element in result.model.mesh.elements
    ) == ((1, 2), (2, 3), (2, 4))
    assert np.all(np.isfinite(result_value.U))
    assert any(notice.code == FRAME_NOTICE for notice in result.notices)


def _mixed_connectivity_load_model(
    tmp_path,
    name: str,
    elements: tuple[tuple[int, int, int], ...],
    record: str,
    *,
    fixed_nodes: tuple[int, ...] = (),
):
    return _read_deck(
        tmp_path,
        name,
        _b31_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 2.0, 0.0, 0.0),
            ),
            elements,
            orientation=(0.0, 1.0, 0.0),
            fixed_nodes=fixed_nodes,
            dloads=(record,),
        ),
    ).model


def _materialized_load_vector(model: FEMModel) -> np.ndarray:
    materialized = deepcopy(model)
    apply_sections(materialized)
    return build_load_vector(
        materialized.mesh,
        boundary_for_step(materialized, "LOAD"),
    )


def test_global_and_local_dloads_preserve_mixed_connectivity_and_signs(
    tmp_path,
) -> None:
    forward = ((1, 1, 2), (2, 2, 3))
    mixed = ((1, 1, 2), (2, 3, 2))
    global_forward = _mixed_connectivity_load_model(
        tmp_path,
        "global_forward.inp",
        forward,
        "BEAM, PY, 2.5",
    )
    global_mixed = _mixed_connectivity_load_model(
        tmp_path,
        "global_mixed.inp",
        mixed,
        "BEAM, PY, 2.5",
    )
    global_z_forward = _mixed_connectivity_load_model(
        tmp_path,
        "global_z_forward.inp",
        forward,
        "BEAM, PZ, 2.5",
    )
    global_z_mixed = _mixed_connectivity_load_model(
        tmp_path,
        "global_z_mixed.inp",
        mixed,
        "BEAM, PZ, 2.5",
    )
    global_forward_solvable = _mixed_connectivity_load_model(
        tmp_path,
        "global_forward_solvable.inp",
        forward,
        "BEAM, PY, 2.5",
        fixed_nodes=(1,),
    )
    global_mixed_solvable = _mixed_connectivity_load_model(
        tmp_path,
        "global_mixed_solvable.inp",
        mixed,
        "BEAM, PY, 2.5",
        fixed_nodes=(1,),
    )
    local_p1 = _mixed_connectivity_load_model(
        tmp_path,
        "local_p1.inp",
        mixed,
        "BEAM, P1, 2.5",
    )
    local_p2 = _mixed_connectivity_load_model(
        tmp_path,
        "local_p2.inp",
        mixed,
        "BEAM, P2, -1.5",
    )

    global_forward_load = _materialized_load_vector(global_forward)
    global_mixed_load = _materialized_load_vector(global_mixed)
    global_z_forward_load = _materialized_load_vector(global_z_forward)
    global_z_mixed_load = _materialized_load_vector(global_z_mixed)
    local_p1_load = _materialized_load_vector(local_p1)

    np.testing.assert_allclose(global_forward_load, global_mixed_load)
    np.testing.assert_allclose(global_z_forward_load, global_z_mixed_load)
    np.testing.assert_allclose(global_mixed_load, local_p1_load)

    forward_displacement = solve(global_forward_solvable, "LOAD").U
    mixed_displacement = solve(global_mixed_solvable, "LOAD").U
    np.testing.assert_allclose(
        forward_displacement,
        mixed_displacement,
        rtol=1e-9,
        atol=1e-12,
    )

    local_p2_owned = deepcopy(local_p2)
    apply_sections(local_p2_owned)
    boundary = boundary_for_step(local_p2_owned, "LOAD")
    actual = build_load_vector(local_p2_owned.mesh, boundary)
    expected = np.zeros(local_p2_owned.mesh.num_dofs)
    for element in local_p2_owned.mesh.elements:
        force = get_element_kernel(element.type).line_load(
            local_p2_owned.mesh,
            element,
            (0.0, 0.0, -1.5),
            "local",
        )
        expected[local_p2_owned.mesh.element_dofs(element)] += force
    np.testing.assert_allclose(actual, expected)

    frames = resolve_effective_beam_frames(
        local_p2,
        RegionRef("element_set", "BEAM"),
    )
    assert frames.passed
    assert frames.frames[0].local_z == pytest.approx((0.0, 0.0, 1.0))
    assert frames.frames[1].local_z == pytest.approx((0.0, 0.0, -1.0))


def _direct_two_span_model() -> FEMModel:
    return FEMModel(
        name="direct_beam2",
        mesh=Mesh3D(
            nodes=[
                Node3D(1, 0.0, 0.0, 0.0),
                Node3D(2, 1.0, 0.0, 0.0),
                Node3D(3, 1.0, 1.0, 0.0),
            ],
            elements=[
                Element3D(1, [1, 2], "Beam2"),
                Element3D(2, [2, 3], "Beam2"),
            ],
            dofs_per_node=6,
        ),
        node_sets={
            "FIXED": NodeSet("FIXED", (1,)),
            "TIP": NodeSet("TIP", (3,)),
        },
        element_sets={"BEAM": ElementSet("BEAM", (1, 2))},
        materials={
            "STEEL": MaterialDefinition(
                "STEEL",
                {"E": 210000000000.0, "nu": 0.30},
            )
        },
        sections=[
            SectionAssignment(
                "BEAM",
                "STEEL",
                "rectangle",
                {
                    "height": 0.20,
                    "width": 0.10,
                    BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 1.0),
                },
            )
        ],
        steps=[
            AnalysisStep(
                "LOAD",
                boundaries=(DisplacementConstraint("FIXED", 1, 6),),
                cloads=(NodalLoad("TIP", 3, 10.0),),
            )
        ],
    )


def _materialized_stiffness_and_load(
    model: FEMModel,
) -> tuple[np.ndarray, np.ndarray]:
    materialized = deepcopy(model)
    apply_sections(materialized)
    stiffness = assemble_global_stiffness(materialized.mesh)
    load = build_load_vector(
        materialized.mesh,
        boundary_for_step(materialized, "LOAD"),
    )
    return stiffness, load


def test_imported_b31_matches_direct_beam2_frame_stiffness_load_and_solution(
    tmp_path,
) -> None:
    imported = _read_deck(
        tmp_path,
        "imported_two_span.inp",
        _b31_deck(
            (
                (1, 0.0, 0.0, 0.0),
                (2, 1.0, 0.0, 0.0),
                (3, 1.0, 1.0, 0.0),
            ),
            ((1, 1, 2), (2, 2, 3)),
            orientation=(0.0, 0.0, 1.0),
            fixed_nodes=(1,),
            tip_nodes=(3,),
            cloads=((3, 3, 10.0),),
        ),
    ).model
    direct = _direct_two_span_model()

    imported_frames = resolve_effective_beam_frames(
        imported,
        RegionRef("element_set", "BEAM"),
    )
    direct_frames = resolve_effective_beam_frames(
        direct,
        RegionRef("element_set", "BEAM"),
    )

    assert [
        (node.id, node.x, node.y, node.z) for node in imported.mesh.nodes
    ] == [
        (node.id, node.x, node.y, node.z) for node in direct.mesh.nodes
    ]
    assert [
        (element.id, tuple(element.node_ids))
        for element in imported.mesh.elements
    ] == [
        (element.id, tuple(element.node_ids))
        for element in direct.mesh.elements
    ]
    assert imported_frames.passed and direct_frames.passed
    for imported_entry, direct_entry in zip(
        imported_frames.entries,
        direct_frames.entries,
    ):
        assert imported_entry.frame.source == direct_entry.frame.source == "explicit"
        np.testing.assert_allclose(
            imported_entry.frame.rotation,
            direct_entry.frame.rotation,
        )
        assert imported_entry.frame.length == pytest.approx(
            direct_entry.frame.length
        )

    imported_stiffness, imported_load = _materialized_stiffness_and_load(imported)
    direct_stiffness, direct_load = _materialized_stiffness_and_load(direct)
    np.testing.assert_allclose(imported_stiffness, direct_stiffness)
    np.testing.assert_allclose(imported_load, direct_load)
    np.testing.assert_allclose(
        solve(imported, "LOAD").U,
        solve(direct, "LOAD").U,
    )


@pytest.mark.parametrize(
    ("orientation", "expected_code"),
    (
        ((0.0, 0.0, 0.0), "beam.orientation.invalid"),
        ((1.0, 0.0, 0.0), "beam.orientation.parallel"),
    ),
)
def test_invalid_b31_n1_remains_a_typed_build_error(
    tmp_path,
    orientation: tuple[float, float, float],
    expected_code: str,
) -> None:
    path = write_inp(
        tmp_path,
        f"invalid_{expected_code.rsplit('.', 1)[-1]}.inp",
        _b31_deck(
            ((1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0)),
            ((1, 1, 2),),
            orientation=orientation,
        ),
    )

    with pytest.raises(abaqus.AbaqusInputError) as caught:
        abaqus.read(path)

    assert caught.value.code == expected_code
    assert len(caught.value.locations) <= 4
    assert caught.value.record["element"] == 1
    assert caught.value.record["nodes"] == (1, 2)


def test_nonfinite_b31_n1_remains_a_typed_parse_error(tmp_path) -> None:
    path = write_inp(
        tmp_path,
        "nonfinite_n1.inp",
        _b31_deck(
            ((1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0)),
            ((1, 1, 2),),
            orientation_record="1e400, 0.0, 1.0",
        ),
    )

    with pytest.raises(abaqus.AbaqusInputError) as caught:
        abaqus.read(path)

    assert caught.value.code == "abaqus.real.nonfinite"
