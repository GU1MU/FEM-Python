from __future__ import annotations

import numpy as np

from fem.application.results import (
    SolveResultBundle,
    build_solve_result_bundle,
)
from fem.application.revisions import SolveTaskSnapshot
from fem.application.runs import ResultRecord
from fem.core.model import AnalysisStep, FEMModel
from fem.core.result import ModelResult


def make_zero_result(mesh, model_name):
    model = FEMModel(mesh=mesh, name=model_name)
    return ModelResult(
        model,
        AnalysisStep("load"),
        np.zeros(mesh.num_dofs),
        np.zeros(mesh.num_dofs),
    )


def make_solve_result_bundle(
    task: SolveTaskSnapshot,
    *,
    marker: float = 0.0,
) -> SolveResultBundle:
    """Build a coherent typed bundle for Session lifecycle tests."""

    if type(task) is not SolveTaskSnapshot:
        raise TypeError("task must be exactly SolveTaskSnapshot")
    matching_steps = tuple(
        step
        for step in task.model.steps
        if step.name == task.step_name
    )
    if len(matching_steps) != 1:
        raise ValueError("task model must contain exactly one matching step")
    displacement = np.zeros(task.model.mesh.num_dofs, dtype=float)
    displacement[0] = float(marker)
    result = ModelResult(
        model=task.model,
        step=matching_steps[0],
        U=displacement,
        reactions=np.zeros(task.model.mesh.num_dofs, dtype=float),
        name=task.run_name,
    )
    return build_solve_result_bundle(task, result)


def assert_result_records_equivalent(
    actual: ResultRecord,
    expected: ResultRecord,
) -> None:
    """Compare the complete accepted result payload without NumPy equality."""

    assert actual.result_id == expected.result_id
    assert actual.provenance == expected.provenance
    assert actual.output_report == expected.output_report
    assert actual.created_at == expected.created_at
    assert actual.materialization.source == expected.materialization.source
    assert (
        actual.materialization.generation
        == expected.materialization.generation
    )
    assert (
        actual.materialization.topology.node_ids
        == expected.materialization.topology.node_ids
    )
    assert (
        actual.materialization.topology.element_ids
        == expected.materialization.topology.element_ids
    )
    assert (
        actual.materialization.topology.element_types
        == expected.materialization.topology.element_types
    )
    assert (
        actual.materialization.topology.connectivity
        == expected.materialization.topology.connectivity
    )
    assert (
        actual.materialization.topology.element_region_keys
        == expected.materialization.topology.element_region_keys
    )
    np.testing.assert_array_equal(actual.result.U, expected.result.U)
    np.testing.assert_array_equal(
        actual.result.reactions,
        expected.result.reactions,
    )
    np.testing.assert_array_equal(
        actual.materialization.topology.node_coordinates,
        expected.materialization.topology.node_coordinates,
    )
    np.testing.assert_array_equal(
        actual.materialization.topology.nodal_displacements,
        expected.materialization.topology.nodal_displacements,
    )
    assert len(actual.materialization.fields) == len(
        expected.materialization.fields
    )
    for actual_field, expected_field in zip(
        actual.materialization.fields,
        expected.materialization.fields,
        strict=True,
    ):
        assert actual_field.descriptor == expected_field.descriptor
        assert actual_field.source == expected_field.source
        assert actual_field.key == expected_field.key
        assert actual_field.locations == expected_field.locations
        np.testing.assert_array_equal(
            actual_field.values,
            expected_field.values,
        )
