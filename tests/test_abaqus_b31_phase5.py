from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fem.assemble import assemble_global_stiffness
from fem.application import ModelSession, resolve_effective_beam_frames
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    LineLoad,
    NodalLoad,
)
from fem.core.result import ModelResult
from fem.elements import (
    BEAM_FRAME_FIELD_KEY,
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamFrameField,
    BeamFrameFieldInvalidError,
    get_element_kernel,
    resolve_beam_frame,
)
from fem.post.stress.beam import recover_section_end_stress
from fem.solvers import static_linear


def _beam_properties(field: BeamFrameField | None = None) -> dict[str, object]:
    properties: dict[str, object] = {
        "E": 210.0,
        "nu": 0.25,
        "section_type": "rectangle",
        "height": 0.3,
        "width": 0.2,
    }
    if field is not None:
        properties[BEAM_FRAME_FIELD_KEY] = field
    return properties


def _beam_mesh(
    *,
    field: BeamFrameField | None = None,
    reversed_connectivity: bool = False,
) -> Mesh3D:
    node_ids = [1, 2]
    if reversed_connectivity:
        node_ids.reverse()
    return Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
        ),
        elements=(
            Element3D(10, node_ids, "Beam2", _beam_properties(field)),
        ),
        dofs_per_node=6,
    )


def _twisted_field(length: float = 2.0) -> BeamFrameField:
    start = np.eye(3)
    end = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    return BeamFrameField.from_rotations(length, start, end)


def _fixed_tip_model(mesh: Mesh3D) -> FEMModel:
    return FEMModel(
        mesh=mesh,
        steps=(
            AnalysisStep(
                "solve",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=(NodalLoad(2, 2, 4.0),),
            ),
        ),
    )


def _rigid_modes(mesh: Mesh3D) -> tuple[np.ndarray, ...]:
    start = np.array((0.0, 0.0, 0.0))
    end = np.array((2.0, 0.0, 0.0))
    modes: list[np.ndarray] = []
    for translation in np.eye(3):
        modes.append(
            np.concatenate((translation, np.zeros(3), translation, np.zeros(3)))
        )
    for rotation in np.eye(3):
        modes.append(
            np.concatenate(
                (
                    np.cross(rotation, start),
                    rotation,
                    np.cross(rotation, end),
                    rotation,
                )
            )
        )
    return tuple(modes)


def _rotation_dof_map(rotation: np.ndarray) -> np.ndarray:
    transform = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transform[start : start + 3, start : start + 3] = rotation
    return transform


def test_constant_field_preserves_stiffness_load_solution_and_stress() -> None:
    legacy_mesh = _beam_mesh()
    legacy_frame = resolve_beam_frame(legacy_mesh, legacy_mesh.elements[0])
    contract_mesh = _beam_mesh(field=BeamFrameField.constant(legacy_frame))
    legacy_kernel = get_element_kernel("Beam2")
    contract_kernel = get_element_kernel("Beam2")

    np.testing.assert_array_equal(
        legacy_kernel.stiffness(legacy_mesh, legacy_mesh.elements[0]),
        contract_kernel.stiffness(contract_mesh, contract_mesh.elements[0]),
    )
    np.testing.assert_array_equal(
        legacy_kernel.line_load(
            legacy_mesh,
            legacy_mesh.elements[0],
            (0.5, -1.0, 2.0),
            "global",
        ),
        contract_kernel.line_load(
            contract_mesh,
            contract_mesh.elements[0],
            (0.5, -1.0, 2.0),
            "global",
        ),
    )

    legacy_result = static_linear.solve(_fixed_tip_model(legacy_mesh), "solve")
    contract_result = static_linear.solve(
        _fixed_tip_model(contract_mesh),
        "solve",
    )
    np.testing.assert_array_equal(legacy_result.U, contract_result.U)
    np.testing.assert_array_equal(legacy_result.reactions, contract_result.reactions)

    legacy_stress = recover_section_end_stress(legacy_result)
    contract_stress = recover_section_end_stress(contract_result)
    np.testing.assert_array_equal(
        [row.values() for row in legacy_stress.rows],
        [row.values() for row in contract_stress.rows],
    )


def test_public_report_keeps_element_field_and_uses_its_endpoint_frames() -> None:
    field = _twisted_field()
    model = FEMModel(mesh=_beam_mesh(field=field))

    report = resolve_effective_beam_frames(model, 10)

    assert report.passed
    assert report.entries[0].field == field
    np.testing.assert_array_equal(
        report.entries[0].field.start.rotation,
        field.start.rotation,
    )
    np.testing.assert_array_equal(
        report.entries[0].field.end.rotation,
        field.end.rotation,
    )


def test_field_is_canonical_when_a_legacy_reference_conflicts() -> None:
    field = BeamFrameField.from_rotations(2.0, np.eye(3), np.eye(3))
    mesh = _beam_mesh(field=field)
    mesh.elements[0].props[BEAM_LOCAL_Y_REFERENCE_KEY] = (0.0, 0.0, 1.0)

    frame = resolve_beam_frame(
        mesh,
        mesh.elements[0],
        properties={BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 1.0)},
    )
    report = resolve_effective_beam_frames(
        FEMModel(mesh=mesh),
        10,
    )

    np.testing.assert_array_equal(frame.rotation, field.start.rotation)
    assert frame.source == "explicit"
    np.testing.assert_array_equal(
        report.entries[0].frame.rotation,
        field.start.rotation,
    )


def test_twisted_field_is_symmetric_has_rigid_modes_and_solves() -> None:
    mesh = _beam_mesh(field=_twisted_field())
    stiffness = assemble_global_stiffness(mesh)

    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0.0, atol=1.0e-12)
    for mode in _rigid_modes(mesh):
        np.testing.assert_allclose(stiffness @ mode, 0.0, rtol=0.0, atol=1.0e-10)

    result = static_linear.solve(_fixed_tip_model(mesh), "solve")
    assert np.all(np.isfinite(result.U))
    assert abs(result.U[mesh.global_dof(2, 1)]) > 0.0


def test_twisted_field_is_covariant_under_global_rotation() -> None:
    field = _twisted_field()
    original = _beam_mesh(field=field)
    quarter_turn = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotated_field = BeamFrameField.from_rotations(
        field.length,
        field.start.rotation @ quarter_turn.T,
        field.end.rotation @ quarter_turn.T,
    )
    rotated = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 0.0, 2.0, 0.0),
        ),
        elements=(
            Element3D(10, (1, 2), "Beam2", _beam_properties(rotated_field)),
        ),
        dofs_per_node=6,
    )
    dof_rotation = _rotation_dof_map(quarter_turn)
    original_stiffness = assemble_global_stiffness(original)
    rotated_stiffness = assemble_global_stiffness(rotated)

    np.testing.assert_allclose(
        rotated_stiffness,
        dof_rotation @ original_stiffness @ dof_rotation.T,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_twisted_field_covaries_with_connectivity_reversal() -> None:
    field = _twisted_field()
    forward = _beam_mesh(field=field)
    axis_reversal = np.diag((-1.0, 1.0, -1.0))
    reversed_field = BeamFrameField.from_rotations(
        field.length,
        axis_reversal @ field.end.rotation,
        axis_reversal @ field.start.rotation,
    )
    reversed_mesh = _beam_mesh(
        field=reversed_field,
        reversed_connectivity=True,
    )
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    kernel = get_element_kernel("Beam2")

    forward_element = kernel.stiffness(forward, forward.elements[0])
    reversed_element = kernel.stiffness(
        reversed_mesh,
        reversed_mesh.elements[0],
    )
    np.testing.assert_allclose(
        reversed_element,
        permutation @ forward_element @ permutation.T,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        assemble_global_stiffness(reversed_mesh),
        assemble_global_stiffness(forward),
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    forward_load = kernel.line_load(forward, forward.elements[0], (0.0, 2.0, 1.0), "local")
    reversed_load = kernel.line_load(
        reversed_mesh,
        reversed_mesh.elements[0],
        tuple(axis_reversal @ np.array((0.0, 2.0, 1.0))),
        "local",
    )
    np.testing.assert_allclose(
        reversed_load,
        permutation @ forward_load,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def _constant_field_for_tangent(
    tangent: tuple[float, float, float],
    local_y: tuple[float, float, float],
    length: float,
) -> BeamFrameField:
    local_x = np.asarray(tangent, dtype=float)
    local_y_array = np.asarray(local_y, dtype=float)
    local_y_array /= np.linalg.norm(local_y_array)
    local_z = np.cross(local_x, local_y_array)
    rotation = np.vstack((local_x, local_y_array, local_z))
    return BeamFrameField.from_rotations(length, rotation, rotation)


def test_kink_and_branch_assemblies_keep_independent_end_frames() -> None:
    kink_fields = (
        _constant_field_for_tangent((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0),
        _constant_field_for_tangent((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.0),
    )
    kink = Mesh3D(
        nodes=(
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 1.0, 0.0, 0.0),
            Node3D(3, 1.0, 1.0, 0.0),
        ),
        elements=(
            Element3D(10, (1, 2), "Beam2", _beam_properties(kink_fields[0])),
            Element3D(20, (2, 3), "Beam2", _beam_properties(kink_fields[1])),
        ),
        dofs_per_node=6,
    )
    kink_stiffness = assemble_global_stiffness(kink)
    np.testing.assert_allclose(kink_stiffness, kink_stiffness.T, atol=1.0e-12)

    branch_nodes = (
        Node3D(1, 0.0, 0.0, 0.0),
        Node3D(2, 1.0, 0.0, 0.0),
        Node3D(3, 0.0, 1.0, 0.0),
        Node3D(4, 0.0, 0.0, 1.0),
    )
    branch_fields = (
        kink_fields[0],
        kink_fields[1],
        _constant_field_for_tangent((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0),
    )
    branch = Mesh3D(
        nodes=branch_nodes,
        elements=tuple(
            Element3D(
                element_id,
                (1, node_id),
                "Beam2",
                _beam_properties(field),
            )
            for element_id, node_id, field in zip(
                (10, 20, 30),
                (2, 3, 4),
                branch_fields,
                strict=True,
            )
        ),
        dofs_per_node=6,
    )
    branch_model = FEMModel(
        mesh=branch,
        steps=(
            AnalysisStep(
                "branch",
                boundaries=(DisplacementConstraint(1, 1, 6),),
                cloads=(
                    NodalLoad(2, 1, 1.0),
                    NodalLoad(3, 2, 1.0),
                    NodalLoad(4, 3, 1.0),
                ),
            ),
        ),
    )
    result = static_linear.solve(branch_model, "branch")
    assert np.all(np.isfinite(result.U))


def test_local_p1_p2_loads_and_section_recovery_share_field_effective_load() -> None:
    field = _twisted_field()
    mesh = _beam_mesh(field=field)
    elem = mesh.elements[0]
    kernel = get_element_kernel("Beam2")
    p1 = kernel.line_load(mesh, elem, (0.0, 2.0, 0.0), "local")
    p2 = kernel.line_load(mesh, elem, (0.0, 0.0, 3.0), "local")
    resultant_scale = 2.0 * field.length / np.pi
    np.testing.assert_allclose(
        p1[:3] + p1[6:9],
        (0.0, 2.0 * resultant_scale, 2.0 * resultant_scale),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        p2[:3] + p2[6:9],
        (0.0, -3.0 * resultant_scale, 3.0 * resultant_scale),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(
        p1,
        kernel.local_line_load(mesh, elem, (0.0, 2.0, 0.0), "local"),
    )
    np.testing.assert_array_equal(
        p2,
        kernel.local_line_load(mesh, elem, (0.0, 0.0, 3.0), "local"),
    )

    step = AnalysisStep(
        "loaded",
        line_loads=(
            LineLoad(10, (0.0, 2.0, 0.0), "local"),
            LineLoad(10, (0.0, 0.0, 3.0), "local"),
        ),
    )
    result = ModelResult(
        FEMModel(mesh=mesh),
        step,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )
    recovered = recover_section_end_stress(result)
    expected_actions = kernel.local_end_actions(
        mesh,
        elem,
        result.U,
        p1 + p2,
    )
    np.testing.assert_allclose(
        [
            (row.axial_force, row.moment_y, row.moment_z)
            for row in recovered.rows
        ],
        expected_actions,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert all(np.isfinite(row.s11_abs_max) for row in recovered.rows)


def test_invalid_field_is_rejected_before_kernel_and_session_install() -> None:
    invalid_field = BeamFrameField.from_rotations(
        3.0,
        np.eye(3),
        np.eye(3),
    )
    mesh = _beam_mesh(field=invalid_field)
    kernel = get_element_kernel("Beam2")
    with pytest.raises(BeamFrameFieldInvalidError):
        kernel.stiffness(mesh, mesh.elements[0])

    session = ModelSession()
    token = session.prepare_import("inline-invalid.inp")
    before = session.snapshot()
    invalid_model = FEMModel(mesh=mesh)
    with pytest.raises(BeamFrameFieldInvalidError):
        session.accept_imported_model(token.token, invalid_model)
    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.model_revision == before.model_revision
    assert after.model is None

    valid_field = _twisted_field()
    valid_model = FEMModel(mesh=_beam_mesh(field=valid_field))
    delta = session.accept_imported_model(
        token.token,
        valid_model,
    )
    assert delta.accepted
    projection = session.projection_snapshot()
    assert projection.source_kind == "imported"
    installed_field = projection.model.mesh.elements[0].props[BEAM_FRAME_FIELD_KEY]
    assert installed_field == valid_field
    assert projection.model is not valid_model

    public = session.snapshot()
    assert public.model is not projection.model
    assert public.model.mesh.elements[0].props[BEAM_FRAME_FIELD_KEY] == valid_field
    copied = deepcopy(public.model)
    assert copied.mesh.elements[0].props[BEAM_FRAME_FIELD_KEY] == valid_field
