from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fem.abaqus import read
from fem.application.results import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultProvider,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    build_result_provider,
)
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import FEMModel
from fem.core.result import ModelResult
from fem.solvers.static_linear import solve
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import (
    ResultData, ScalarField, StressSample, automatic_deformation_scale,
    build_result_data, build_result_data_from_provider, deformed_points,
    ensure_stress_data, field_family,
)
from fem_gui.visualization import result_adapter as result_adapter_module
from fem_gui.visualization.model_adapter import ModelGeometry
from fem_gui.visualization.stress_adapter import build_stress_render_geometry


def _accepted_provider(result: ModelResult) -> ResultProvider:
    return build_result_provider(
        ResultSourceKey(
            result_id="result-accepted",
            session_id="session-provider-adapter",
            artifact_id="artifact-provider-adapter",
            model_revision=7,
            step_name=str(getattr(result.step, "name", "Static-1")),
            run_id="run-provider-adapter",
        ),
        result,
    )


def test_provider_adapter_reads_primary_fields_without_recovery(
    monkeypatch,
    gui_inp_path,
):
    model = read(gui_inp_path)
    result = solve(model)
    provider = _accepted_provider(result)
    geometry = build_model_geometry(model)

    def reject_recovery(*_args, **_kwargs):
        raise AssertionError("provider adaptation must not recover fields")

    monkeypatch.setattr(
        result_adapter_module.dispatch,
        "resolve_type_keys",
        reject_recovery,
    )
    monkeypatch.setattr(
        result_adapter_module.field,
        "StressRecovery",
        reject_recovery,
    )
    monkeypatch.setattr(
        result_adapter_module.truss,
        "recover",
        reject_recovery,
    )
    monkeypatch.setattr(
        result_adapter_module.beam,
        "recover_section_end_stress",
        reject_recovery,
    )

    data = build_result_data_from_provider(
        provider,
        geometry,
        legacy_result=result,
    )

    u_availability = next(
        item
        for item in provider.catalog().fields
        if item.descriptor.field_id.variable is ResultVariable.U
    )
    u_field = provider.field(u_availability.key)
    u_values = u_field.values
    u1_column = u_field.descriptor.columns.index("U1")
    magnitude_column = u_field.descriptor.columns.index("Magnitude")
    assert data.fields["U1"].values == pytest.approx(
        u_values[:, u1_column]
    )
    assert data.fields["U"].values == pytest.approx(
        u_values[:, magnitude_column]
    )
    assert data.displacement_vectors == pytest.approx(
        provider.snapshot.topology.nodal_displacements
    )
    assert data.field_selections["U"] == ScalarFieldSelection(
        u_availability.key,
        "Magnitude",
    )
    assert not data.field_ready("IP:Mises")
    assert data._source_result is result
    assert data.artifact_id == provider.source.artifact_id
    assert data.run_id == provider.source.run_id
    assert data.result_id == provider.source.result_id
    assert data.materialization_generation == 0


def test_provider_adapter_keeps_the_legacy_lazy_recovery_fallback(
    gui_inp_path,
):
    model = read(gui_inp_path)
    result = solve(model)
    provider = _accepted_provider(result)
    data = build_result_data_from_provider(
        provider,
        build_model_geometry(model),
        legacy_result=result,
    )

    assert not data.field_ready("CENTROID:Mises")
    assert ensure_stress_data(data, "CENTROID")
    assert data.field_ready("CENTROID:Mises")


def test_provider_adapter_fails_closed_when_complete_keys_share_a_shortcut(
    gui_inp_path,
):
    model = read(gui_inp_path)
    result = solve(model)
    provider = _accepted_provider(result)
    explicit_key = provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
            ),
            gauss_order=1,
        )
    )
    draft = provider.apply(provider.materialize((explicit_key,)))

    with pytest.raises(
        ValueError,
        match="cannot represent multiple complete field keys",
    ):
        build_result_data_from_provider(
            draft,
            build_model_geometry(model),
            legacy_result=result,
        )


def test_displacement_reaction_stress_and_deformation_coordinates(gui_inp_path):
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    data = build_result_data(result, geometry)

    for node_id, point_index in geometry.node_id_to_point_index.items():
        expected = [result.U[model.mesh.global_dof(node_id, component)] for component in range(2)]
        expected.append(0.0)
        assert data.displacement_vectors[point_index] == pytest.approx(expected)
    assert data.fields["U"].values == pytest.approx(np.linalg.norm(data.displacement_vectors, axis=1))
    assert "RF" in data.fields
    assert set(data.nodal_values) == set(geometry.node_id_to_point_index)
    assert data.element_stress
    assert all("Mises" in values for values in data.element_stress.values())
    assert data.nodal_stress_samples

    scale = automatic_deformation_scale(geometry, data)
    assert deformed_points(geometry, data, scale) == pytest.approx(
        geometry.points + scale * data.displacement_vectors
    )


def test_stress_recovery_is_lazy_and_cached_by_position(monkeypatch, gui_inp_path):
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    calls = {"recovery": 0}
    original_recovery = result_adapter_module.field.StressRecovery

    class CountedRecovery(original_recovery):
        def __init__(self, *args, **kwargs):
            calls["recovery"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        result_adapter_module.field,
        "StressRecovery",
        CountedRecovery,
    )

    data = build_result_data(result, geometry, include_stress=False)

    assert data.field_ready("U")
    assert not data.field_ready("NODAL:Mises")
    assert not data.field_ready("CENTROID:Mises")
    assert not data.field_ready("IP:Mises")
    assert data.element_stress == {}
    assert data.nodal_stress_samples == {}
    assert calls == {"recovery": 0}

    assert ensure_stress_data(data, "NODAL")
    assert data.field_ready("NODAL:Mises")
    assert data.field_ready("EN:Mises")
    assert data.field_ready("CENTROID:Mises")
    assert data.field_ready("IP:Mises")
    assert calls == {"recovery": 1}

    assert not ensure_stress_data(data, "EN")
    assert not ensure_stress_data(data, "CENTROID")
    assert not ensure_stress_data(data, ("IP", "CENTROID", "EN", "NODAL"))
    assert calls == {"recovery": 1}


def test_beam_rotations_are_separate_from_translations():
    mesh = Mesh3D(
        [Node3D(10, 0.0, 0.0, 0.0), Node3D(20, 1.0, 0.0, 0.0)],
        [Element3D(
            30,
            [10, 20],
            "Beam2",
            {
                "E": 200.0,
                "nu": 0.3,
                "section_type": "rectangle",
                "height": 0.1,
                "width": 0.2,
            },
        )],
        dofs_per_node=6,
    )
    model = FEMModel(mesh)
    displacement = np.array([
        0.0, 0.0, 0.0, 0.1, 0.2, 0.3,
        1.0, 2.0, 3.0, 0.4, 0.5, 0.6,
    ])
    result = ModelResult(model, None, displacement, np.zeros(12))
    geometry = build_model_geometry(model)

    data = build_result_data(result, geometry)

    assert data.fields["U3"].values.tolist() == pytest.approx([0.0, 3.0])
    assert data.fields["R1"].values.tolist() == pytest.approx([0.1, 0.4])
    assert data.fields["R2"].values.tolist() == pytest.approx([0.2, 0.5])
    assert data.fields["R3"].values.tolist() == pytest.approx([0.3, 0.6])
    assert data.displacement_vectors[:, 2].tolist() == [0.0, 3.0]
    assert data.element_stress == {}
    assert set(data.nodal_stress) == {10, 20}
    assert data.field_ready("NODAL:S11Max")
    assert data.field_ready("NODAL:S11Min")
    assert data.field_ready("NODAL:S11AbsMax")
    assert data.stress_position_label("NODAL") == "节点包络"
    assert field_family("R2") == "R"
    assert field_family("RM2") == "RM"
    assert field_family("NODAL:S11AbsMax") == "S"


def test_beam_stress_consumes_typed_envelope_without_isolated_zero(
    monkeypatch,
):
    mesh = Mesh3D(
        [
            Node3D(10, 0.0, 0.0, 0.0),
            Node3D(20, 1.0, 0.0, 0.0),
            Node3D(30, 2.0, 0.0, 0.0),
        ],
        [Element3D(
            40,
            [10, 20],
            "Beam2",
            {
                "E": 200.0,
                "nu": 0.3,
                "section_type": "rectangle",
                "height": 0.1,
                "width": 0.2,
            },
        )],
        dofs_per_node=6,
    )
    model = FEMModel(mesh)
    result = ModelResult(
        model,
        None,
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )
    original_recover = (
        result_adapter_module.beam.recover_section_end_stress
    )
    calls = 0

    def counted_recover(result_):
        nonlocal calls
        calls += 1
        return original_recover(result_)

    def reject_legacy(_result):
        raise AssertionError("GUI Beam path must not call the legacy envelope")

    monkeypatch.setattr(
        result_adapter_module.beam,
        "recover_section_end_stress",
        counted_recover,
    )
    monkeypatch.setattr(
        result_adapter_module.beam,
        "nodal_envelope",
        reject_legacy,
    )

    data = build_result_data(result, build_model_geometry(model))

    assert calls == 1
    assert set(data.nodal_stress) == {10, 20}
    assert 30 not in data.nodal_stress
    assert data.fields["NODAL:S11AbsMax"].values[:2] == pytest.approx(
        [0.0, 0.0]
    )
    assert np.isnan(data.fields["NODAL:S11AbsMax"].values[2])


def test_truss_stress_is_adapted_from_one_canonical_recovery(monkeypatch):
    mesh = Mesh3D(
        [Node3D(10, 0.0, 0.0, 0.0), Node3D(20, 2.0, 0.0, 0.0)],
        [
            Element3D(
                30,
                [10, 20],
                "Truss2",
                {"E": 100.0, "area": 2.0},
            )
        ],
    )
    model = FEMModel(mesh)
    result = ModelResult(
        model,
        None,
        np.asarray([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
        np.zeros(mesh.num_dofs),
    )
    original_recover = result_adapter_module.truss.recover
    calls = 0

    def counted_recover(mesh_, displacement):
        nonlocal calls
        calls += 1
        return original_recover(mesh_, displacement)

    monkeypatch.setattr(
        result_adapter_module.truss,
        "recover",
        counted_recover,
    )

    data = build_result_data(result, build_model_geometry(model))

    assert calls == 1
    assert data.element_stress == {
        30: {"LE11": 0.1, "S11": 10.0, "Mises": 10.0}
    }
    assert list(data.fields)[-3:] == [
        "CENTROID:LE11",
        "CENTROID:S11",
        "CENTROID:Mises",
    ]


def test_automatic_deformation_scale_uses_actual_small_model_span(gui_inp_path):
    model = read(gui_inp_path)
    result = solve(model)
    geometry = build_model_geometry(model)
    data = build_result_data(result, geometry)
    small_geometry = replace(geometry, points=geometry.points * 1.0e-3)
    small_data = replace(data, displacement_vectors=data.displacement_vectors * 1.0e-3)
    span = float(np.linalg.norm(np.ptp(small_geometry.points, axis=0)))
    maximum = float(np.max(np.linalg.norm(small_data.displacement_vectors, axis=1)))

    assert automatic_deformation_scale(small_geometry, small_data) == pytest.approx(
        0.1 * span / maximum
    )


def _two_cell_stress_data() -> tuple[ModelGeometry, ResultData]:
    points = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0],
    ])
    cells = ((0, 1, 3, 2), (1, 4, 5, 3))
    geometry = ModelGeometry(
        points, cells, np.asarray([4, *cells[0], 4, *cells[1]], dtype=np.int64),
        np.asarray([9, 9], dtype=np.uint8),
        {node_id: index for index, node_id in enumerate((10, 20, 30, 40, 50, 60))},
        {index: node_id for index, node_id in enumerate((10, 20, 30, 40, 50, 60))},
        {101: 0, 102: 1}, {0: 101, 1: 102},
    )
    samples: dict[int, tuple[StressSample, ...]] = {}
    for cell_index, cell in enumerate(cells):
        element_id = 101 + cell_index
        for local_node, point_index in enumerate(cell, 1):
            node_id = geometry.point_index_to_node_id[point_index]
            # S33 varies mildly and should average; S23 jumps at the shared edge.
            values = {
                "S33": float(node_id) + cell_index,
                "S23": (-100.0 if cell_index == 0 else 100.0),
            }
            sample = StressSample(element_id, local_node, "region-a", 1.0, values)
            samples[node_id] = (*samples.get(node_id, ()), sample)
    fields = {
        key: ScalarField(key, key, "point" if key.startswith("N:") else "element_node", np.empty(0))
        for key in ("N:S33", "N:S23", "EN:S33")
    }
    data = ResultData(
        np.zeros((6, 3)), np.zeros((6, 3)), fields, {}, {}, {}, samples
    )
    return geometry, data


def test_stress_averaging_threshold_is_component_specific():
    geometry, data = _two_cell_stress_data()

    smooth = build_stress_render_geometry(geometry, data, "N:S33", 75.0)
    discontinuous = build_stress_render_geometry(geometry, data, "N:S23", 75.0)

    assert len(smooth.points) == 6
    assert len(discontinuous.points) == 8
    assert np.isfinite(smooth.values).all()
    assert np.isfinite(discontinuous.values).all()


def test_element_nodal_mode_duplicates_shared_coordinates_and_keeps_fem_maps():
    geometry, data = _two_cell_stress_data()

    rendered = build_stress_render_geometry(geometry, data, "EN:S33")

    assert len(rendered.points) == 8
    assert rendered.points[1] == pytest.approx(rendered.points[4])
    assert rendered.point_index_to_node_id[1] == rendered.point_index_to_node_id[4] == 20
    assert rendered.point_index_to_element_id[1] == 101
    assert rendered.point_index_to_element_id[4] == 102
    assert rendered.cell_index_to_element_id == {0: 101, 1: 102}


def test_nodal_averaging_never_crosses_material_or_section_region():
    geometry, data = _two_cell_stress_data()
    changed = dict(data.nodal_stress_samples)
    for node_id in (20, 40):
        first, second = changed[node_id]
        changed[node_id] = (first, replace(second, region_key="region-b"))
    separated = build_stress_render_geometry(
        geometry, replace(data, nodal_stress_samples=changed), "N:S33", 100.0
    )

    assert len(separated.points) == 8
