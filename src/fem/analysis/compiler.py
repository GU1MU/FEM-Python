"""Sole compiler from authored analysis data to executable equations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from scipy.sparse import csr_matrix

from fem.assembly import SparseAssembler, assemble_mass_matrix
from fem.model import (
    AnalysisStepSnapshot,
    DampingModel,
    DofSpace,
    DynamicStepControls,
    StaticControlMode,
)
from fem.problem import (
    ExplicitDynamicProblem,
    LinearDynamicProblem,
    NonlinearDynamicProblem,
    StaticEquilibriumProblem,
)
from fem.state import EvaluationContext, SolutionState, TransactionalStateManager

from .compilation.boundary.compiled import compile_boundary
from .compilation.boundary.loads import build_load_vector
from .compilation.capabilities import (
    DEFAULT_NONLINEAR_STATIC_CAPABILITIES,
    NonlinearStaticCapabilityRegistry,
)
from .compilation.materials import compile_material_assignments
from .compiled import CompiledAnalysis, CompiledSystem
from .contracts import AnalysisRequest
from .execution_plan import ExecutionStrategy, resolve_execution_plan


def compile_analysis(
    model: Any,
    request: AnalysisRequest,
    *,
    capability_registry: NonlinearStaticCapabilityRegistry | None = None,
) -> CompiledAnalysis:
    """Compile one resolved request into the canonical numerical graph."""

    if type(request) is not AnalysisRequest:
        raise TypeError("request must be exactly AnalysisRequest")
    if request.step.procedure == "dynamic":
        plan = resolve_execution_plan(model, request)
        if plan.strategy is ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR:
            return _compile_linear_dynamic(model, request)
        if plan.strategy is ExecutionStrategy.EXPLICIT_DYNAMIC_LINEAR:
            return _compile_linear_explicit_dynamic(model, request)
        return _compile_assembled_dynamic(model, request, plan)
    if request.step.procedure != "static":
        raise NotImplementedError(
            f"analysis procedure {request.step.procedure!r} is not compiled yet"
        )
    if capability_registry is None:
        capability_registry = DEFAULT_NONLINEAR_STATIC_CAPABILITIES
    if type(capability_registry) is not NonlinearStaticCapabilityRegistry:
        raise TypeError(
            "capability_registry must be exactly "
            "NonlinearStaticCapabilityRegistry"
        )
    step = request.step
    plan = resolve_execution_plan(model, request)
    options = request.options.as_mapping()
    options["geometry_mode"] = request.geometry_mode
    problem = _compile_nonlinear_static(
        model,
        step,
        options,
        capability_registry,
        control_mode=request.controls.control_mode,
        material_assignments=plan.material_assignments,
        execution_plan=plan,
    )
    return CompiledAnalysis(
        request=request,
        system=CompiledSystem(problem),
        result_model=_compiled_result_model(
            model,
            plan.material_assignments,
        ),
    )


def _compile_linear_dynamic(
    model: Any,
    request: AnalysisRequest,
) -> CompiledAnalysis:
    """Compile the linear implicit transient equation system."""

    if not isinstance(request.controls, DynamicStepControls):
        raise TypeError("dynamic request must carry DynamicStepControls")
    plan = resolve_execution_plan(model, request)
    if plan.strategy is not ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR:
        raise ValueError("dynamic request did not resolve to linear implicit dynamics")

    # Reuse the canonical linear section application and stiffness assembly.
    # The returned stiffness is detached from the authored model; no dynamic
    # solver reads GUI metadata or mutates the session model.
    from . import linear_static

    prepared = linear_static.prepare(model, copy_model=True)
    owned_model = prepared.model_for_task()
    boundary = compile_boundary(owned_model, request.step)
    reference_load = build_load_vector(owned_model.mesh, boundary.loads)
    mass = assemble_mass_matrix(
        owned_model.mesh,
        policy=request.controls.mass_matrix,
    )
    if request.controls.damping_model is DampingModel.NONE:
        damping = csr_matrix(mass.shape, dtype=float)
    else:
        damping = (
            request.controls.rayleigh_mass * mass
            + request.controls.rayleigh_stiffness * prepared.base_stiffness
        ).tocsr()
    problem = LinearDynamicProblem(
        dof_space=DofSpace.displacement_for_mesh(owned_model.mesh),
        stiffness=prepared.base_stiffness,
        mass=mass,
        damping=damping,
        constraints=boundary.constraints,
        reference_load=reference_load,
        amplitude=request.controls.amplitude,
        initial_conditions=request.step.initial_conditions,
        output_metadata={
            "procedure": "dynamic_implicit",
            "integration_method": request.controls.integration_method.value,
            "mass_matrix": request.controls.mass_matrix.value,
            "damping_model": request.controls.damping_model.value,
            "material_models": plan.material_models,
        },
    )
    return CompiledAnalysis(
        request=request,
        system=CompiledSystem(problem),
        result_model=_compiled_result_model(
            model,
            plan.material_assignments,
            base_model=owned_model,
        ),
    )


def _compile_linear_explicit_dynamic(
    model: Any,
    request: AnalysisRequest,
) -> CompiledAnalysis:
    """Compile the linear central-difference equation system.

    Linear explicit dynamics still uses the linear-static section and element
    kernel path for stiffness assembly.  This is important for truss and beam
    elements that are intentionally outside the nonlinear continuum registry.
    The explicit procedure has already normalized the mass policy to lumped
    mass at the typed-controls boundary.
    """

    if not isinstance(request.controls, DynamicStepControls):
        raise TypeError("dynamic request must carry DynamicStepControls")
    plan = resolve_execution_plan(model, request)
    if plan.strategy is not ExecutionStrategy.EXPLICIT_DYNAMIC_LINEAR:
        raise ValueError(
            "dynamic request did not resolve to linear explicit dynamics"
        )

    from . import linear_static

    prepared = linear_static.prepare(model, copy_model=True)
    owned_model = prepared.model_for_task()
    boundary = compile_boundary(owned_model, request.step)
    reference_load = build_load_vector(owned_model.mesh, boundary.loads)
    mass = assemble_mass_matrix(
        owned_model.mesh,
        policy=request.controls.mass_matrix,
    )
    if request.controls.damping_model is DampingModel.NONE:
        damping = csr_matrix(mass.shape, dtype=float)
    else:
        damping = (
            request.controls.rayleigh_mass * mass
            + request.controls.rayleigh_stiffness * prepared.base_stiffness
        ).tocsr()
    problem = LinearDynamicProblem(
        dof_space=DofSpace.displacement_for_mesh(owned_model.mesh),
        stiffness=prepared.base_stiffness,
        mass=mass,
        damping=damping,
        constraints=boundary.constraints,
        reference_load=reference_load,
        amplitude=request.controls.amplitude,
        initial_conditions=request.step.initial_conditions,
        output_metadata={
            "procedure": "dynamic_explicit",
            "integration_method": request.controls.integration_method.value,
            "mass_matrix": request.controls.mass_matrix.value,
            "damping_model": request.controls.damping_model.value,
            "material_models": plan.material_models,
        },
    )
    return CompiledAnalysis(
        request=request,
        system=CompiledSystem(problem),
        result_model=_compiled_result_model(
            model,
            plan.material_assignments,
            base_model=owned_model,
        ),
    )


def _compile_assembled_dynamic(
    model: Any,
    request: AnalysisRequest,
    plan: Any,
) -> CompiledAnalysis:
    """Compile nonlinear implicit and both explicit dynamic variants."""

    if not isinstance(request.controls, DynamicStepControls):
        raise TypeError("dynamic request must carry DynamicStepControls")
    options = request.options.as_mapping()
    options["geometry_mode"] = request.geometry_mode
    try:
        static_problem = _compile_nonlinear_static(
            model,
            request.step,
            options,
            DEFAULT_NONLINEAR_STATIC_CAPABILITIES,
            control_mode=StaticControlMode.LOAD,
            material_assignments=plan.material_assignments,
            execution_plan=plan,
        )
    except NotImplementedError as error:
        # A dynamic procedure shares the canonical nonlinear assembly path,
        # but the public failure must describe the dynamic capability
        # boundary rather than leaking the static compiler's internal error.
        raise ValueError(
            "dynamic analysis does not support this element/material "
            f"combination: {error}"
        ) from error
    assembly = static_problem.assembly
    mass = assemble_mass_matrix(
        model.mesh,
        policy=request.controls.mass_matrix,
    )
    zero = SolutionState.zeros(assembly.dof_space)
    if request.controls.damping_model is DampingModel.NONE:
        damping = csr_matrix(mass.shape, dtype=float)
    else:
        try:
            assembly.begin_increment()
            initial = assembly.assemble(
                zero,
                context=EvaluationContext(
                    time=0.0,
                    load_factor=request.controls.amplitude.value_at(0.0),
                ),
            )
            assembly.commit()
        except BaseException:
            assembly.rollback()
            raise
        damping = (
            request.controls.rayleigh_mass * mass
            + request.controls.rayleigh_stiffness * initial.tangent
        ).tocsr()
    metadata = {
        "procedure": (
            "dynamic_implicit"
            if plan.strategy is ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR
            else "dynamic_explicit"
        ),
        "response_regime": (
            "nonlinear"
            if plan.strategy
            in {
                ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
                ExecutionStrategy.EXPLICIT_DYNAMIC_NONLINEAR,
            }
            else "linear"
        ),
        "mass_matrix": request.controls.mass_matrix.value,
        "damping_model": request.controls.damping_model.value,
        "material_models": plan.material_models,
        "geometry_mode": request.geometry_mode.value,
    }
    problem_type = (
        NonlinearDynamicProblem
        if plan.strategy is ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR
        else ExplicitDynamicProblem
    )
    problem = problem_type(
        assembly=assembly,
        mass=mass,
        damping=damping,
        constraints=static_problem.constraints,
        reference_load=static_problem.reference_load,
        amplitude=request.controls.amplitude,
        initial_conditions=request.step.initial_conditions,
        output_metadata=metadata,
    )
    return CompiledAnalysis(
        request=request,
        system=CompiledSystem(problem),
        result_model=_compiled_result_model(
            model,
            plan.material_assignments,
        ),
    )


def _compile_nonlinear_static(
    model: Any,
    step: AnalysisStepSnapshot,
    options: Mapping[str, Any],
    capability_registry: NonlinearStaticCapabilityRegistry,
    *,
    control_mode: Any,
    material_assignments: Any | None = None,
    execution_plan: Any | None = None,
) -> StaticEquilibriumProblem:
    """Compose current nonlinear statics without creating combination types."""

    elements = tuple(getattr(model.mesh, "elements", ()))
    if not elements:
        raise ValueError("nonlinear static analysis requires at least one element")
    compiled_assignments = (
        compile_material_assignments(model)
        if material_assignments is None
        else material_assignments
    )
    assignment_by_element = compiled_assignments.by_element
    operator_by_type = {}
    material_by_element = {}
    material_labels: set[str] = set()
    material_algorithms: set[str] = set()
    for element in elements:
        element_type = str(getattr(element, "type", "")).strip()
        capability = capability_registry.resolve(element_type)
        operator_by_type.setdefault(
            element_type.casefold(),
            capability.build_operator(options),
        )
        assignment = assignment_by_element[int(element.id)]
        material_properties = dict(assignment.properties)
        material_properties["constitutive_model"] = (
            assignment.constitutive_model
        )
        if assignment.algorithm is not None:
            material_properties["algorithm"] = assignment.algorithm
        material = capability.build_material(
            material_properties,
            options,
            element_id=int(element.id),
            element_type=element_type,
        )
        material_by_element[int(element.id)] = material
        material_labels.add(assignment.constitutive_model)
        algorithm = getattr(material, "algorithm", None)
        if algorithm is not None:
            material_algorithms.add(str(algorithm))
    element_properties = {
        element_id: dict(assignment.properties)
        for element_id, assignment in assignment_by_element.items()
    }
    assembly = SparseAssembler.from_displacement_mesh(
        model.mesh,
        operator_by_type,
        state=TransactionalStateManager(),
        material_by_element=material_by_element,
        element_properties_by_element=element_properties,
        output_metadata={
            "material": (
                next(iter(material_labels))
                if len(material_labels) == 1
                else "mixed"
            ),
            "material_algorithm": (
                next(iter(material_algorithms))
                if len(material_algorithms) == 1
                else None
            ),
            "geometry_mode": str(options.get("geometry_mode", "finite_strain")),
            "execution_strategy": (
                None
                if execution_plan is None
                else str(execution_plan.strategy)
            ),
            "material_models": tuple(sorted(material_labels)),
        },
    )
    boundary = compile_boundary(model, step)
    return StaticEquilibriumProblem(
        assembly=assembly,
        constraints=boundary.constraints,
        reference_load=build_load_vector(model.mesh, boundary.loads),
        control_mode=control_mode,
    )


def _compiled_result_model(
    model: Any,
    material_assignments: Any,
    *,
    base_model: Any | None = None,
) -> Any:
    """Build the detached model used by result recovery.

    The public result must remain attached to the accepted authoring model,
    but post-processing needs the exact effective properties that were used by
    the compiler.  Keeping that projection here makes the compiler the only
    source of truth for section/material resolution and avoids mutating the
    authoring session just to display stress.
    """

    if not hasattr(material_assignments, "by_element"):
        raise TypeError("material assignments must expose by_element")
    owned_model = deepcopy(model if base_model is None else base_model)
    elements = tuple(getattr(getattr(owned_model, "mesh", None), "elements", ()))
    assignments = material_assignments.by_element
    for element in elements:
        try:
            assignment = assignments[int(element.id)]
        except KeyError as error:
            raise ValueError(
                f"compiled material assignment is missing element {element.id}"
            ) from error
        properties = dict(getattr(element, "props", {}) or {})
        properties.update(dict(assignment.properties))
        if assignment.material_name is not None:
            properties["material"] = assignment.material_name
        if assignment.section_type is not None:
            properties["section_type"] = assignment.section_type
        properties["constitutive_model"] = assignment.constitutive_model
        if assignment.algorithm is not None:
            properties["algorithm"] = assignment.algorithm
        element.props = properties
    return owned_model


__all__ = ["compile_analysis"]
