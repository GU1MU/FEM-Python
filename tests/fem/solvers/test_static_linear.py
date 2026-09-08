from copy import deepcopy

import numpy as np
import pytest

from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    NodeSet,
    MaterialDefinition,
    NodalLoad,
    SectionAssignment,
)
from fem.solvers import static_linear
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh, make_mixed_tri3_quad4_mesh
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)


def _assert_bar_result(result, tip_displacement, fixed_reaction, tip_reaction=0.0):
    # Two-node axial bar: EA/L = 100 * 2 / 1 = 200.
    np.testing.assert_allclose(
        result.U, (0.0, 0.0, 0.0, tip_displacement, 0.0, 0.0), atol=1e-12,
    )
    np.testing.assert_allclose(
        result.reactions,
        (fixed_reaction, 0.0, 0.0, tip_reaction, 0.0, 0.0),
        atol=1e-12,
    )


def test_prepared_system_reuses_assembly_and_isolates_model_snapshots(monkeypatch):
    model = make_two_step_static_pull_truss_model()
    model.mesh.elements[0].props["E"] = 7.0
    model.materials["steel"] = MaterialDefinition("steel", {"E": 100.0})
    model.element_sets["bar"] = ElementSet("bar", (1,))
    model.sections = [SectionAssignment("bar", "steel", "truss", {"area": 2.0})]
    assemble_calls = 0
    original_assemble = static_linear.assemble_global_stiffness_sparse

    def assemble(mesh):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(mesh)

    monkeypatch.setattr(static_linear, "assemble_global_stiffness_sparse", assemble)
    prepared = static_linear.prepare(model)
    assert model.mesh.elements[0].props["E"] == 7.0

    # Subsequent edits by the caller must not alter the prepared bar or its loads.
    model.mesh.nodes[1].x = 10.0
    model.materials["steel"].properties["E"] = 9.0
    model.steps[1].cloads = [NodalLoad("TIP", 1, 999.0)]
    pull1, pull2 = prepared.solve(steps="all")
    assert pull1.model.mesh.nodes[1].x == 1.0
    assert pull1.model.materials["steel"].properties["E"] == 100.0
    _assert_bar_result(pull1, 0.5, -100.0)
    _assert_bar_result(pull2, 1.0, -200.0)

    pull1.model.mesh.nodes[1].x = 20.0
    pull1.model.steps[1].cloads = [NodalLoad("TIP", 1, 888.0)]
    prepared.validate_step("pull1").name = "edited-validation-result"
    prepared.validate_stiffness("pull1").name = "edited-stiffness-result"
    clone = prepared.clone()
    cloned_result = clone.solve("pull1")
    assert cloned_result.model.mesh.nodes[1].x == 1.0
    _assert_bar_result(cloned_result, 0.5, -100.0)
    cloned_result.model.mesh.elements[0].props["area"] = 99.0
    _assert_bar_result(clone.solve("pull2"), 1.0, -200.0)
    _assert_bar_result(prepared.solve("pull1"), 0.5, -100.0)
    assert assemble_calls == 1


def test_prepared_load_cases_have_independent_absolute_displacements_and_reactions():
    model = make_two_step_static_pull_truss_model()
    settlement = AnalysisStep(
        "settlement",
        boundaries=[DisplacementConstraint("TIP", 1, 1, 0.25)],
        cloads=[NodalLoad("TIP", 1, 30.0)],
    )
    prepared = static_linear.prepare(model)

    pull1, pull2, settled = prepared.solve(steps=("pull1", "pull2", settlement))

    _assert_bar_result(pull1, 0.5, -100.0)
    _assert_bar_result(pull2, 1.0, -200.0)
    # Settlement imposes u=0.25 absolutely; tip reaction = 200*0.25 - 30.
    _assert_bar_result(settled, 0.25, -50.0, 20.0)
    pull1.U[:] = 999.0
    pull1.reactions[:] = 999.0
    _assert_bar_result(pull2, 1.0, -200.0)
    _assert_bar_result(settled, 0.25, -50.0, 20.0)


@pytest.mark.parametrize(
    ("invalid_step", "error", "message"),
    [
        (AnalysisStep("late_dynamic", procedure="dynamic"), ValueError, "procedure"),
        (AnalysisStep("late_invalid", cloads=[NodalLoad("MISSING", 1, 1.0)]),
         KeyError, "MISSING"),
    ],
    ids=["procedure", "reference"],
)
def test_invalid_later_case_does_not_leave_a_partly_prepared_model(
    invalid_step, error, message,
):
    model = make_two_step_static_pull_truss_model()
    model.materials["steel"] = MaterialDefinition("steel", {"E": 300.0})
    model.element_sets["bar"] = ElementSet("bar", (1,))
    model.sections = [SectionAssignment("bar", "steel", "truss", {"area": 4.0})]
    before = deepcopy((model.mesh.elements, model.metadata, model.sections, model.steps))

    with pytest.raises(error, match=message):
        static_linear.solve(model, steps=("pull1", invalid_step))

    assert (model.mesh.elements, model.metadata, model.sections, model.steps) == before


@pytest.mark.parametrize("operation", [static_linear.validate_problem, static_linear.solve],
                         ids=["validate", "solve"])
@pytest.mark.parametrize(
    ("invalid", "error", "message"),
    [
        ("procedure", ValueError, "requires procedure 'static'"),
        ("nlgeom-bool", ValueError, "does not support nlgeom"),
        ("nlgeom-text", ValueError, "does not support nlgeom"),
        ("reference", KeyError, "MISSING"),
    ],
)
def test_static_operations_reject_unsupported_or_invalid_steps(
    operation, invalid, error, message,
):
    model = make_static_pull_truss_model()
    if invalid == "procedure":
        model.steps[0].procedure = "dynamic"
    elif invalid == "nlgeom-bool":
        model.steps[0].metadata["nlgeom"] = True
    elif invalid == "nlgeom-text":
        model.steps[0].metadata["nlgeom"] = "YES"
    else:
        model.steps[0].cloads = [NodalLoad("MISSING", 1, 1.0)]

    with pytest.raises(error, match=message):
        operation(model, "pull")


def test_static_solve_accepts_false_nlgeom_and_records_usable_timings():
    model = make_static_pull_truss_model()
    model.steps[0].metadata["nlgeom"] = "NO"
    timings = {}

    assert static_linear.validate_problem(model, "pull").name == "pull"
    result = static_linear.solve(model, "pull", timings=timings)

    _assert_bar_result(result, 0.5, -100.0)
    assert timings
    assert all(np.isfinite(seconds) and seconds >= 0.0 for seconds in timings.values())


def test_static_validation_and_solve_ignore_unselected_step_references():
    model = make_two_step_static_pull_truss_model()
    model.steps[2].cloads = [NodalLoad("MISSING_IN_PULL2", 1, 1.0)]

    assert static_linear.validate_problem(model, "pull1").name == "pull1"
    result = static_linear.solve(model, steps=("pull1",))[0]
    _assert_bar_result(result, 0.5, -100.0)
    with pytest.raises(KeyError, match="MISSING_IN_PULL2"):
        static_linear.validate_problem(model, "pull2")


@pytest.mark.parametrize(
    ("mesh_builder", "fixed_nodes", "loaded_node", "components"),
    [
        (make_mixed_tri3_quad4_mesh, (1, 4), 5, (1, 2)),
        (make_mixed_hex8_tet4_mesh, (1, 4, 5, 8), 9, (1, 2, 3)),
    ],
    ids=["connected_plane", "connected_solid"],
)
def test_connected_mixed_models_preserve_global_force_balance(
    mesh_builder, fixed_nodes, loaded_node, components
):
    mesh = mesh_builder()
    model = FEMModel(
        mesh=mesh,
        node_sets={
            "fixed": NodeSet("fixed", fixed_nodes),
            "loaded": NodeSet("loaded", (loaded_node,)),
        },
        steps=[
            AnalysisStep(
                "pull",
                boundaries=(
                    DisplacementConstraint(
                        "fixed", min(components), max(components), 0.0
                    ),
                ),
                cloads=(NodalLoad("loaded", 1, 1.0),),
            )
        ],
    )

    result = static_linear.solve(model, "pull")

    assert np.all(np.isfinite(result.U))
    assert float(result.reactions[0::mesh.dofs_per_node].sum()) == pytest.approx(-1.0)
    for component in range(1, mesh.dofs_per_node):
        assert float(result.reactions[component::mesh.dofs_per_node].sum()) == pytest.approx(
            0.0, abs=1e-10
        )
