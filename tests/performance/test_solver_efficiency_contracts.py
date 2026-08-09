from __future__ import annotations

import gc

import numpy as np
import pytest

from fem.assemble import stiffness as stiffness_module
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import AnalysisStep, FEMModel
from fem.solvers import static_linear
from fem_gui.inspection_service import InspectionService
from fem_gui.visualization.model_adapter import (
    build_model_geometry,
    pyvista_cell_array,
)
from tests.helpers.model_builders import (
    make_two_step_static_pull_truss_model,
)


def test_prepared_system_reuses_work_without_sharing_public_models(
    monkeypatch,
):
    calls = {"sections": 0, "assembly": 0, "factor": 0}
    original_sections = static_linear.materials.apply_sections
    original_assembly = static_linear.assemble_global_stiffness_sparse
    original_factor = static_linear.factorize_spd

    def apply_sections(model):
        calls["sections"] += 1
        return original_sections(model)

    def assemble(mesh):
        calls["assembly"] += 1
        return original_assembly(mesh)

    def factor(stiffness):
        calls["factor"] += 1
        return original_factor(stiffness)

    monkeypatch.setattr(
        static_linear.materials,
        "apply_sections",
        apply_sections,
    )
    monkeypatch.setattr(
        static_linear,
        "assemble_global_stiffness_sparse",
        assemble,
    )
    monkeypatch.setattr(static_linear, "factorize_spd", factor)

    prepared = static_linear.prepare(
        make_two_step_static_pull_truss_model()
    )
    prepared.validate_stiffness("pull1")
    first = prepared.solve("pull1")
    second = prepared.solve("pull2")
    cloned = prepared.clone()
    third = cloned.solve("pull1")

    assert calls == {"sections": 1, "assembly": 1, "factor": 1}
    assert cloned._base_stiffness is prepared._base_stiffness
    assert cloned._factor_cache is prepared._factor_cache
    assert len(prepared._factor_cache._entries) == 1
    constrained_pattern = next(iter(prepared._factor_cache._entries))
    assert constrained_pattern == tuple(sorted(constrained_pattern))
    assert (
        len(prepared._factor_cache._entries)
        <= static_linear._FACTOR_CACHE_MAX_ENTRIES
    )
    assert first.model is not second.model
    assert first.model is not third.model
    assert second.model is not third.model
    assert first.model is not prepared._trusted_model_for_task()

    original_x = second.model.mesh.nodes[1].x
    first.model.mesh.nodes[1].x = original_x + 100.0
    repeated = prepared.solve("pull1")
    assert repeated.model.mesh.nodes[1].x == original_x
    np.testing.assert_allclose(repeated.U, first.U)
    with pytest.raises(ValueError):
        prepared._base_stiffness.data[0] = 0.0


def test_prepared_system_clone_keeps_shared_factor_alive_until_last_owner(
    monkeypatch,
):
    factors = []
    original_factor = static_linear.factorize_spd

    class TrackedFactor:
        def __init__(self, factor):
            self.factor = factor
            self.close_calls = 0

        def solve(self, rhs):
            return self.factor.solve(rhs)

        def close(self):
            self.close_calls += 1
            self.factor.close()

    def factor(stiffness):
        tracked = TrackedFactor(original_factor(stiffness))
        factors.append(tracked)
        return tracked

    monkeypatch.setattr(static_linear, "factorize_spd", factor)
    prepared = static_linear.prepare(
        make_two_step_static_pull_truss_model()
    )
    prepared.validate_stiffness("pull1")
    cloned = prepared.clone()

    del prepared
    gc.collect()
    assert len(factors) == 1
    assert factors[0].close_calls == 0

    result = cloned.solve("pull2")
    assert result.U[
        result.model.mesh.global_dof(2, 0)
    ] == pytest.approx(1.0)
    assert factors[0].close_calls == 0

    del cloned
    gc.collect()
    assert factors[0].close_calls == 1


def test_sparse_assembly_plan_has_exact_flat_storage():
    model = _plate_model(400)
    plan = stiffness_module._build_assembly_plan(model.mesh)
    element_count = len(model.mesh.elements)
    entry_count = sum(
        len(model.mesh.element_dofs(element)) ** 2
        for element in model.mesh.elements
    )
    expected_bytes = (
        2 * (element_count + 1) * np.dtype(np.int64).itemsize
        + 2 * entry_count * np.dtype(np.int64).itemsize
    )

    assert not hasattr(plan, "__dict__")
    assert plan.dof_offsets.shape == (element_count + 1,)
    assert plan.entry_offsets.shape == (element_count + 1,)
    assert plan.rows.shape == (entry_count,)
    assert plan.cols.shape == (entry_count,)
    assert all(
        values.dtype == np.int64
        and values.ndim == 1
        and values.flags.c_contiguous
        for values in (
            plan.dof_offsets,
            plan.entry_offsets,
            plan.rows,
            plan.cols,
        )
    )
    assert sum(
        values.nbytes
        for values in (
            plan.dof_offsets,
            plan.entry_offsets,
            plan.rows,
            plan.cols,
        )
    ) == expected_bytes


def test_large_import_projection_stays_flat_and_inspection_stays_lazy():
    model = _plate_model(5_000)

    geometry = build_model_geometry(model)
    service = InspectionService(model)

    assert geometry.points.shape == (len(model.mesh.nodes), 3)
    assert geometry.cell_array.shape == (5 * len(model.mesh.elements),)
    assert geometry.cell_array.dtype == np.int64
    assert geometry.cell_array.flags.c_contiguous
    assert pyvista_cell_array(geometry) is geometry.cell_array
    assert len(geometry.cells) == len(model.mesh.elements)
    assert service._element_record_cached.cache_info().currsize == 0

    service.element_record(model.mesh.elements[-1].id)
    assert service._element_record_cached.cache_info().currsize == 1


def _plate_model(element_count: int) -> FEMModel:
    columns = 100
    rows = (element_count + columns - 1) // columns
    nodes = [
        Node2D(
            row * (columns + 1) + column + 1,
            float(column),
            float(row),
        )
        for row in range(rows + 1)
        for column in range(columns + 1)
    ]
    elements = []
    for index in range(element_count):
        row, column = divmod(index, columns)
        lower_left = row * (columns + 1) + column + 1
        elements.append(
            Element2D(
                index + 1,
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + columns + 2,
                    lower_left + columns + 1,
                ],
                "Quad4",
            )
        )
    return FEMModel(
        Mesh2D(nodes, elements),
        steps=[AnalysisStep("load")],
    )
