"""Model capability and engineering rules shared by definition creation and edits."""

from __future__ import annotations

from fem.application import UnitContext, describe_model_capabilities

from .analysis_authoring import (
    AnalysisAuthoringError,
    ConfirmedDisplacement,
    ConfirmedLoad,
)


NODAL_LOAD_DIRECTIONS = {
    1: "global_x",
    2: "global_y",
    3: "global_z",
    4: "global_rx",
    5: "global_ry",
    6: "global_rz",
}


def model_dofs_per_node(model: object | None) -> int:
    if model is None:
        raise AnalysisAuthoringError(
            "analysis requires a current realized model capability"
        )
    report = describe_model_capabilities(model)
    if not report.compatible or report.dofs_per_node is None:
        raise AnalysisAuthoringError(
            "analysis model has no compatible nodal DOF capability"
        )
    return int(report.dofs_per_node)


def boundary_unit(first_component: int, units: UnitContext) -> str:
    return units.length if first_component <= 3 else "rad"


def nodal_load_unit(component: int, units: UnitContext) -> str:
    return units.force if component <= 3 else f"{units.force}*{units.length}"


def validate_boundary_components_and_unit(
    boundary: ConfirmedDisplacement,
    dofs_per_node: int,
    units: UnitContext,
) -> None:
    if boundary.last_component > dofs_per_node:
        raise AnalysisAuthoringError(
            "boundary component exceeds the current model capability"
        )
    if boundary.first_component <= 3 < boundary.last_component:
        if boundary.value != 0.0 or boundary.unit != boundary_unit(
            boundary.first_component, units
        ):
            raise AnalysisAuthoringError(
                "a mixed translation/rotation range must be a zero fixed "
                "constraint using the project length unit"
            )
    elif boundary.unit != boundary_unit(boundary.first_component, units):
        raise AnalysisAuthoringError(
            "displacement unit must exactly match the project length unit"
            if boundary.first_component <= 3
            else "rotation unit must be rad"
        )


def validate_load_components_and_unit(
    load: ConfirmedLoad,
    dimension: int,
    dofs_per_node: int,
    units: UnitContext,
) -> None:
    if load.load_type == "nodal":
        if int(load.component or 0) > dofs_per_node:
            raise AnalysisAuthoringError(
                "load component must be an integer from 1 to "
                f"{dofs_per_node} for the current model capability"
            )
        if load.direction != NODAL_LOAD_DIRECTIONS[int(load.component)]:
            raise AnalysisAuthoringError(
                "nodal load direction does not match its component"
            )
        expected_unit = nodal_load_unit(int(load.component), units)
    elif load.load_type.startswith("edge_"):
        if dimension != 2:
            raise AnalysisAuthoringError(
                "edge load requires a two-dimensional Part"
            )
        if load.load_type == "edge_traction" and load.direction != "global_xy":
            raise AnalysisAuthoringError(
                "two-dimensional traction direction must be global_xy"
            )
        expected_unit = f"{units.force}/{units.length}"
    else:
        if dimension != 3:
            raise AnalysisAuthoringError(
                "surface load requires a three-dimensional Part"
            )
        if load.load_type == "surface_traction" and load.direction != "global_xyz":
            raise AnalysisAuthoringError(
                "three-dimensional traction direction must be global_xyz"
            )
        expected_unit = units.stress
    if load.vector and len(load.vector) != dimension:
        raise AnalysisAuthoringError(
            "load vector dimension does not match the current Part"
        )
    if load.unit != expected_unit:
        raise AnalysisAuthoringError(
            "load unit does not match the project unit context"
        )
