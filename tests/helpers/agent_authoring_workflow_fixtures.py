from __future__ import annotations

from fem_agent.authoring import AuthoringContext, LocalModelBinding
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.tools.registry import ToolExecutionContext


_REQUIREMENT_GROUP_KEYS = {
    "geometry": (
        "length_unit",
        "force_unit",
        "stress_unit",
    ),
    "mesh": (
        "mesh_cell_shape",
        "mesh_order",
        "mesh_global_size",
    ),
    "definitions": (
        "modeling_assumption",
        "plate_thickness",
        "young_modulus",
        "poisson_ratio",
    ),
    "analysis": (
        "fixed_dofs",
        "load_type",
        "load_direction",
        "load_magnitude",
        "load_unit",
        "load_distribution",
        "analysis_procedure",
        "nlgeom",
        "result_requests",
    ),
}


def _context() -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding(
            "document:a8",
            "native-a8",
            0,
            "native",
            True,
        ),
        model_name="模型-偏心孔板",
        active_part_id=None,
    )


def _requirements() -> dict[str, object]:
    return {
        "modeling_assumption": "plane_stress",
        "length_unit": "mm",
        "force_unit": "N",
        "stress_unit": "MPa",
        "plate_thickness": 2.0,
        "young_modulus": 210000.0,
        "poisson_ratio": 0.3,
        "mesh_cell_shape": "quadrilateral",
        "mesh_order": 1,
        "mesh_global_size": 5.0,
        "fixed_dofs": [1, 2],
        "load_type": "edge_traction",
        "load_direction": "x",
        "load_magnitude": 100.0,
        "load_unit": "N/mm",
        "load_distribution": "uniform",
        "analysis_procedure": "static",
        "nlgeom": False,
        "result_requests": ["U", "S", "RF"],
    }


def _requirements_for(group: str) -> dict[str, object]:
    values = _requirements()
    return {key: values[key] for key in _REQUIREMENT_GROUP_KEYS[group]}


def _dispatch(
    controller: AuthoringWorkflowController,
    name: str,
    arguments: dict[str, object],
    index: int,
):
    return controller.dispatch(
        name,
        arguments,
        ToolExecutionContext("session-a8", 0, f"key-{index}"),
    )
