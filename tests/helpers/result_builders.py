import numpy as np

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
