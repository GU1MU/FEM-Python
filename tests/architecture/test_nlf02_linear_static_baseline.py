from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fem import materials
from fem.model import authoring as steps
from fem.model import add_material, assign_section
from fem.model import ElementSet, FEMModel, NodeSet
from fem.io import inp
from fem.selection import nodes
from fem.analysis import linear_static as static_linear
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INP_FIXTURE = PROJECT_ROOT / "examples" / "examples_data" / "cantilever_beam_3d.inp"


def _make_quad4_baseline_model() -> FEMModel:
    """Build the deterministic 2D model used by the NLF-02 baseline."""
    model = FEMModel(
        mesh=make_quad4_stiffness_mesh(),
        name="nlf02_quad4_baseline",
        node_sets={
            "fixed": NodeSet("fixed", (1, 4)),
            "loaded": NodeSet("loaded", (2, 3)),
        },
        element_sets={"plate": ElementSet("plate", (1,))},
    )
    add_material(
        model,
        materials.linear_elastic.material("steel", E=210.0, nu=0.3),
    )
    assign_section(model, "steel", "plate")

    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=1.0)
    steps.add(model, step)
    return model


def test_nlf02_2d_quad4_linear_static_result_baseline() -> None:
    model = _make_quad4_baseline_model()

    result = static_linear.solve(model, "pull")

    assert type(result).__name__ == "ModelResult"
    assert result.step.name == "pull"
    np.testing.assert_allclose(
        result.U,
        np.array(
            [
                0.0,
                0.0,
                0.0185592665356909,
                0.00204322200392928,
                0.0185592665356909,
                -0.00204322200392926,
                0.0,
                0.0,
            ]
        ),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.reactions,
        np.array(
            [
                -1.0,
                -0.3418467583497053,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.34184675834970535,
            ]
        ),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert result.nodal_displacement(2, 1) == pytest.approx(
        result.nodal_displacement(3, 1)
    )


def test_nlf02_3d_inp_linear_static_result_baseline() -> None:
    model = inp.read(INP_FIXTURE)

    result = static_linear.solve(model)

    assert model.mesh.num_nodes == 621
    assert model.mesh.num_elements == 80
    assert model.mesh.num_dofs == 1863
    assert {str(element.type) for element in model.mesh.elements} == {"Hex20"}
    assert [step.name for step in model.steps] == ["Step-1"]

    tip_node = nodes.nearest(model.mesh, 5.0, 5.0, 0.0)
    fixed_node_ids = model.node_sets["Set-Fixed"].node_ids
    tip_uy = result.nodal_displacement(tip_node, component=2)
    reaction_y = sum(
        result.nodal_reaction(node_id, component=2)
        for node_id in fixed_node_ids
    )

    assert tip_node == 125
    assert tip_uy == pytest.approx(-1.936825523289897, rel=1.0e-8, abs=1.0e-8)
    assert reaction_y == pytest.approx(
        1000.7700849774269,
        rel=1.0e-8,
        abs=1.0e-8,
    )


def test_nlf02_linear_solver_rejects_geometric_nonlinearity() -> None:
    model = _make_quad4_baseline_model()
    model.steps.clear()
    steps.add(model, steps.static("nlgeom", NLGEOM=True))

    with pytest.raises(ValueError, match="does not support nlgeom"):
        static_linear.solve(model, "nlgeom")
