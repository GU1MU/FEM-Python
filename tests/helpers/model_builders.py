from fem.core.mesh import Element2D, Node2D, TrussMesh2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementSet,
    FEMModel,
    NodalLoad,
    NodeSet,
)


def make_simple_truss_mesh(E=100.0, area=2.0, length=1.0):
    return TrussMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, length, 0.0)],
        elements=[Element2D(1, [1, 2], type="Truss2D", props={"E": E, "area": area})],
    )


def make_bare_truss_mesh(length=1.0, props=None):
    return TrussMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, length, 0.0)],
        elements=[Element2D(1, [1, 2], type="Truss2D", props=props or {})],
    )


def make_truss_workflow_model(name="manual_bar", loaded_set_name="tip", element_props=None):
    mesh = make_bare_truss_mesh(props=element_props)
    return FEMModel(
        mesh=mesh,
        name=name,
        node_sets={
            "fixed": NodeSet("fixed", [1]),
            loaded_set_name: NodeSet(loaded_set_name, [2]),
        },
        element_sets={"bar": ElementSet("bar", [1])},
    )


def make_static_pull_truss_model(load=100.0):
    mesh = make_simple_truss_mesh()
    step = AnalysisStep(
        "pull",
        boundaries=[
            DisplacementConstraint("FIXED", 1, 2, 0.0),
            DisplacementConstraint(2, 2, 2, 0.0),
        ],
        cloads=[NodalLoad("TIP", 1, load)],
    )
    return FEMModel(
        mesh=mesh,
        node_sets={
            "FIXED": NodeSet("FIXED", [1]),
            "TIP": NodeSet("TIP", [2]),
        },
        steps=[step],
    )


def make_two_step_static_pull_truss_model():
    mesh = make_simple_truss_mesh()
    return FEMModel(
        mesh=mesh,
        node_sets={
            "FIXED": NodeSet("FIXED", [1]),
            "TIP": NodeSet("TIP", [2]),
        },
        steps=[
            AnalysisStep(
                "Initial",
                boundaries=[
                    DisplacementConstraint("FIXED", 1, 2, 0.0),
                    DisplacementConstraint(2, 2, 2, 0.0),
                ],
            ),
            AnalysisStep("pull1", cloads=[NodalLoad("TIP", 1, 100.0)]),
            AnalysisStep("pull2", cloads=[NodalLoad("TIP", 1, 200.0)]),
        ],
        name="bar",
    )
