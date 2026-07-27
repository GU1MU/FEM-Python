"""Application-level model and authoring capability reports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable

from fem.elements import (
    ElementCapabilityDescriptor,
    ElementCapabilityRequirement,
    get_element_capabilities,
)

from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)
from .results.output_requests import (
    OutputRequestProjection,
    ResultCapabilityCatalog,
    project_output_request,
)
from .results.registry import (
    classify_result_element_types,
    classify_result_model,
)
from .native_mesh_contract import (
    NativeMeshContract,
    describe_native_mesh_contract,
    describe_native_mesh_settings_contract,
)

if TYPE_CHECKING:
    from fem.core.model import LineLoad

    from .definitions import ModelDefinitions, RegionAssignment


_REGION_KINDS = frozenset(
    {"node_set", "element_set", "edge", "surface"}
)
_DISTRIBUTED_LOAD_KINDS = frozenset({"edge", "surface", "line"})
_EXPLICIT_BEAM_ORIENTATION_REQUIREMENT = "beam.orientation.explicit"

class AuthoringStatus(str, Enum):
    """Product-level availability derived from domain capabilities."""

    ENABLED = "enabled"
    LIMITED = "limited"
    READ_ONLY = "read_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RegionRef:
    """Typed reference that preserves region namespace identity."""

    kind: str
    name: str

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold()
        name = str(self.name).strip()
        if kind not in _REGION_KINDS:
            raise ValueError(f"unsupported region kind: {self.kind!r}")
        if not name:
            raise ValueError("region name must not be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class AuthoringCapability:
    """Availability and reason for one application authoring operation."""

    operation: str
    status: AuthoringStatus
    diagnostics: tuple[PreflightDiagnostic, ...] = ()

    @property
    def can_submit(self) -> bool:
        """Return the strict Phase 4 candidate-submission decision."""

        return (
            self.status is AuthoringStatus.ENABLED
            and not any(item.blocking for item in self.diagnostics)
        )

    @property
    def can_enter(self) -> bool:
        """Return whether a manager may expose this operation for inspection."""

        return self.status is not AuthoringStatus.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class AuthoringTarget:
    """One namespace-preserving target and its contextual operations."""

    region: RegionRef
    operations: tuple[AuthoringCapability, ...]

    def __post_init__(self) -> None:
        if type(self.region) is not RegionRef:
            raise TypeError("authoring target region must be RegionRef")
        operations = tuple(self.operations)
        if any(type(item) is not AuthoringCapability for item in operations):
            raise TypeError(
                "authoring target operations must be AuthoringCapability values"
            )
        names = tuple(item.operation for item in operations)
        if len(names) != len(set(names)):
            raise ValueError("authoring target operation names must be unique")
        object.__setattr__(self, "operations", operations)

    def operation(self, name: str) -> AuthoringCapability:
        normalized = str(name).strip().casefold()
        for capability in self.operations:
            if capability.operation == normalized:
                return capability
        return AuthoringCapability(normalized, AuthoringStatus.UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class StepLifecycleProjection:
    """Detached check/submit state for one effective analysis Step."""

    step_name: str
    can_check: bool
    can_submit: bool
    check_reason: str
    submit_reason: str


@dataclass(frozen=True, slots=True)
class SessionAuthoringProjection:
    """Complete headless authoring projection for one Session snapshot."""

    report: ModelCapabilityReport
    targets: tuple[AuthoringTarget, ...]
    step_lifecycle: tuple[StepLifecycleProjection, ...]
    operations: tuple[AuthoringCapability, ...]

    @property
    def output_request_catalog(self) -> ResultCapabilityCatalog | None:
        """Return the source-independent candidate catalog."""

        return self.report.output_request_catalog

    def operation(self, name: str) -> AuthoringCapability:
        """Return a Session-contextual operation before model-level fallback."""

        normalized = _normalize_operation(name)
        for capability in self.operations:
            if capability.operation == normalized:
                return capability
        return self.report.operation(normalized)

    def target(self, region: RegionRef) -> AuthoringTarget:
        if type(region) is not RegionRef:
            raise TypeError("authoring projection target lookup requires RegionRef")
        for target in self.targets:
            if target.region == region:
                return target
        return AuthoringTarget(region, ())

    def step(self, step_name: str | None) -> StepLifecycleProjection | None:
        normalized = str(step_name or "").strip()
        for item in self.step_lifecycle:
            if item.step_name == normalized:
                return item
        return None


@dataclass(frozen=True, slots=True)
class RegionCapability:
    """Safe capability intersection for one typed model region."""

    region: RegionRef
    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    homogeneous: bool
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    distributed_load_kinds: tuple[str, ...]
    diagnostics: tuple[PreflightDiagnostic, ...] = ()
    operations: tuple[AuthoringCapability, ...] = ()

    @property
    def status(self) -> AuthoringStatus:
        if not self.compatible or any(item.blocking for item in self.diagnostics):
            return AuthoringStatus.UNAVAILABLE
        if self.diagnostics:
            return AuthoringStatus.LIMITED
        return AuthoringStatus.ENABLED

    def supports_section(self, section_type: str) -> bool:
        return (
            self.compatible
            and _supports_section_preset(
                self.section_presets,
                section_type,
            )
        )

    def supports_distributed_load(self, load_kind: str) -> bool:
        return (
            self.compatible
            and str(load_kind).strip().casefold()
            in self.distributed_load_kinds
        )

    def operation(self, name: str) -> AuthoringCapability:
        """Return the contextual decision for one catalog operation."""

        normalized = _normalize_operation(name)
        for capability in self.operations:
            if capability.operation == normalized:
                return capability
        return AuthoringCapability(
            normalized,
            AuthoringStatus.UNAVAILABLE,
            (),
        )

    def status_for(self, operation: str) -> AuthoringStatus:
        """Return one operation status without changing the base status."""

        return self.operation(operation).status

    def diagnostics_for(
        self,
        operation: str,
    ) -> tuple[PreflightDiagnostic, ...]:
        """Return only diagnostics relevant to one operation."""

        return self.operation(operation).diagnostics


@dataclass(frozen=True, slots=True)
class ModelCapabilityReport:
    """Aggregated model facts and product authoring policy."""

    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    diagnostics: tuple[PreflightDiagnostic, ...]
    regions: tuple[RegionCapability, ...]
    authoring: tuple[AuthoringCapability, ...]
    output_request_catalog: ResultCapabilityCatalog | None

    def __post_init__(self) -> None:
        if (
            self.output_request_catalog is not None
            and type(self.output_request_catalog) is not ResultCapabilityCatalog
        ):
            raise TypeError(
                "output_request_catalog must be ResultCapabilityCatalog or None"
            )

    @property
    def output_request_candidates(
        self,
    ) -> tuple[OutputRequestProjection, ...]:
        """Return source-independent, canonically projected create choices."""

        if self.output_request_catalog is None:
            return ()
        return self.output_request_catalog.candidates

    @property
    def status(self) -> AuthoringStatus:
        if not self.compatible or any(item.blocking for item in self.diagnostics):
            return AuthoringStatus.UNAVAILABLE
        if self.diagnostics:
            return AuthoringStatus.LIMITED
        return AuthoringStatus.ENABLED

    def region(self, region: RegionRef) -> RegionCapability:
        """Return a region report without collapsing its namespace."""

        for capability in self.regions:
            if capability.region == region:
                return capability
        return _missing_region_capability(region)

    def operation(self, name: str) -> AuthoringCapability:
        for capability in self.authoring:
            if capability.operation == str(name):
                return capability
        return AuthoringCapability(
            str(name),
            AuthoringStatus.UNAVAILABLE,
            (),
        )

    def supports_section(self, section_type: str) -> bool:
        """Return whether the model contract can author this section type."""

        return self.compatible and _supports_section_preset(
            self.section_presets,
            section_type,
        )


@dataclass(frozen=True, slots=True)
class _OutputSupportEvaluation:
    """Private compatibility projection through the canonical result catalog."""

    catalog: ResultCapabilityCatalog | None
    projections: tuple[OutputRequestProjection, ...]
    request_count: int
    complete: bool

    @property
    def supports_output_authoring(self) -> bool:
        """Whether the catalog proves at least one output field combination."""

        return (
            self.catalog is not None
            and bool(self.catalog.candidates)
        )


def _evaluate_output_requests(
    model: Any,
    outputs: Iterable[Any],
) -> _OutputSupportEvaluation:
    """Evaluate authoring values through the sole result support projection."""

    try:
        profile = classify_result_model(model)
        catalog = ResultCapabilityCatalog.from_profile(profile)
    except Exception:
        return _OutputSupportEvaluation(None, (), 0, False)

    try:
        requests = tuple(outputs)
    except Exception:
        return _OutputSupportEvaluation(catalog, (), 0, False)

    projections: list[OutputRequestProjection] = []
    for request_index, request in enumerate(requests):
        try:
            projection = project_output_request(
                request,
                catalog,
                request_index=request_index,
            )
        except Exception:
            return _OutputSupportEvaluation(
                catalog,
                tuple(projections),
                len(requests),
                False,
            )
        projections.append(projection)
    return _OutputSupportEvaluation(
        catalog,
        tuple(projections),
        len(requests),
        True,
    )


def _model_output_requests(model: Any) -> Iterable[Any]:
    """Yield preserved requests without assigning support semantics here."""

    for step in getattr(model, "steps", ()):
        yield from getattr(step, "outputs", ())


def _output_authoring_capabilities(
    *,
    supports_candidate: bool,
) -> tuple[AuthoringCapability]:
    """Expose intrinsic create support from the canonical candidate catalog."""

    return (
        AuthoringCapability(
            "output_request.create",
            (
                AuthoringStatus.ENABLED
                if supports_candidate
                else AuthoringStatus.UNAVAILABLE
            ),
        ),
    )


def require_region_kind(region: RegionRef, expected_kind: str) -> str:
    """Validate an application command target before writing a string DTO."""

    if not isinstance(region, RegionRef):
        raise TypeError("authoring target must be RegionRef")
    normalized = str(expected_kind).strip().casefold()
    if region.kind != normalized:
        raise ValueError(
            f"operation requires region kind {normalized!r}, "
            f"got {region.kind!r}"
        )
    return region.name


def evaluate_authoring_candidate(
    model: Any,
    definitions: ModelDefinitions,
    *,
    operation: str,
    candidate: RegionAssignment | LineLoad,
    step_name: str | None = None,
    candidate_index: int | None = None,
) -> AuthoringCapability:
    """Evaluate an uninstalled assignment or line load on detached state."""

    normalized_operation = _normalize_operation(operation)
    from fem.core.model import LineLoad

    from .definitions import RegionAssignment

    if isinstance(candidate, RegionAssignment):
        return _evaluate_assignment_candidate(
            model,
            definitions,
            normalized_operation,
            candidate,
            candidate_index,
        )
    if isinstance(candidate, LineLoad):
        return _evaluate_line_load_candidate(
            model,
            definitions,
            normalized_operation,
            candidate,
            step_name,
            candidate_index,
        )
    return _unsupported_candidate(
        normalized_operation,
        subject=type(candidate).__name__,
        message=(
            "authoring candidate must be RegionAssignment or LineLoad"
        ),
    )


def _evaluate_assignment_candidate(
    model: Any,
    definitions: Any,
    operation: str,
    candidate: Any,
    candidate_index: int | None,
) -> AuthoringCapability:
    from .definitions import (
        DefinitionRejected,
        ModelDefinitions,
        compile_model_definitions,
        normalize_model_definitions,
    )

    try:
        normalized = normalize_model_definitions(definitions)
        candidate_assignments = list(normalized.assignments)
        if candidate_index is None:
            candidate_assignments = [
                assignment
                for assignment in candidate_assignments
                if assignment.region_name != candidate.region_name
            ]
            candidate_assignments.append(deepcopy(candidate))
            owned_candidate_index = len(candidate_assignments) - 1
        else:
            owned_candidate_index = _validated_candidate_index(
                candidate_index,
                len(candidate_assignments),
                "assignment",
            )
            candidate_assignments[owned_candidate_index] = deepcopy(
                candidate
            )
        candidate_definitions = ModelDefinitions(
            materials=normalized.materials,
            sections=normalized.sections,
            assignments=tuple(candidate_assignments),
            steps=normalized.steps,
        )
    except (DefinitionRejected, IndexError, TypeError, ValueError) as error:
        if isinstance(error, DefinitionRejected):
            return _candidate_decision(operation, error.diagnostics)
        return _unsupported_candidate(
            operation,
            subject=getattr(candidate, "region_name", None),
            message=str(error),
        )

    compile_result = compile_model_definitions(
        model,
        candidate_definitions,
    )
    if not compile_result.passed:
        return _candidate_decision(
            operation,
            compile_result.diagnostics,
        )
    compiled = compile_result.require_model()
    compiled_definitions = compile_result.definitions
    assert compiled_definitions is not None
    owned_candidate = compiled_definitions.assignments[
        owned_candidate_index
    ]
    section = next(
        (
            item
            for item in compiled_definitions.sections
            if item.name == owned_candidate.section_name
        ),
        None,
    )
    target = RegionRef("element_set", owned_candidate.region_name)
    if section is None:
        return _unsupported_candidate(
            operation,
            subject=target,
            message=(
                f"section {owned_candidate.section_name!r} is not defined"
            ),
        )
    expected_operation = _section_operation(section)
    if operation != expected_operation:
        return _unsupported_candidate(
            operation,
            subject=target,
            message=(
                f"operation {operation!r} does not match "
                f"section type {section.section_type!r}"
            ),
        )
    return _evaluate_compiled_orientation_operation(
        compiled,
        target,
        operation,
    )


def _evaluate_line_load_candidate(
    model: Any,
    definitions: Any,
    operation: str,
    candidate: Any,
    step_name: str | None,
    candidate_index: int | None,
) -> AuthoringCapability:
    from .definitions import (
        DefinitionRejected,
        compile_model_definitions,
        normalize_model_definitions,
    )

    try:
        normalized = normalize_model_definitions(definitions)
    except DefinitionRejected as error:
        return _candidate_decision(operation, error.diagnostics)
    compile_result = compile_model_definitions(model, normalized)
    if not compile_result.passed:
        return _candidate_decision(
            operation,
            compile_result.diagnostics,
        )
    compiled = compile_result.require_model()

    if step_name is not None:
        from fem.boundary.step import get_step

        try:
            selected_step = get_step(compiled, step_name)
        except Exception as error:
            return AuthoringCapability(
                operation,
                AuthoringStatus.UNAVAILABLE,
                (
                    PreflightDiagnostic(
                        code="step.reference.invalid",
                        severity=PreflightSeverity.ERROR,
                        stage=PreflightStage.STEP,
                        message=str(error),
                        subject=str(step_name),
                        path=("steps", str(step_name)),
                        remediation="请选择当前 definitions 中存在的分析步。",
                    ),
                ),
            )
        line_loads = list(getattr(selected_step, "line_loads", ()))
        try:
            if candidate_index is None:
                line_loads.append(deepcopy(candidate))
            else:
                owned_index = _validated_candidate_index(
                    candidate_index,
                    len(line_loads),
                    "line load",
                )
                line_loads[owned_index] = deepcopy(candidate)
        except (IndexError, TypeError, ValueError) as error:
            return _unsupported_candidate(
                operation,
                subject=str(step_name),
                message=str(error),
            )
        selected_step.line_loads = tuple(line_loads)

    coordinate_system = str(
        getattr(candidate, "coordinate_system", "")
    ).strip().casefold()
    expected_operation = f"load.line.{coordinate_system}"
    if (
        operation != expected_operation
        or coordinate_system not in {"global", "local"}
    ):
        return _unsupported_candidate(
            operation,
            subject=getattr(candidate, "target", None),
            message=(
                f"operation {operation!r} does not match line load "
                f"coordinate system {coordinate_system!r}"
            ),
        )
    raw_target = getattr(candidate, "target", None)
    try:
        target: RegionRef | int = (
            RegionRef("element_set", raw_target)
            if isinstance(raw_target, str)
            else raw_target
        )
    except (TypeError, ValueError) as error:
        return _unsupported_candidate(
            operation,
            subject=raw_target,
            message=str(error),
        )
    return _evaluate_compiled_orientation_operation(
        compiled,
        target,
        operation,
    )


def _evaluate_compiled_orientation_operation(
    model: Any,
    target: RegionRef | int,
    operation: str,
) -> AuthoringCapability:
    if not _supports_target_operation(model, target, operation):
        return _unsupported_candidate(
            operation,
            subject=target,
            message=f"target does not support {operation!r}",
        )
    if not _requires_explicit_beam_orientation(
        model,
        target,
        operation,
    ):
        return AuthoringCapability(
            operation,
            AuthoringStatus.ENABLED,
            (),
        )

    # Lazy import avoids capabilities <-> beam_frames module initialization.
    from .beam_frames import resolve_effective_beam_frames

    report = resolve_effective_beam_frames(model, target)
    diagnostics = tuple(
        _diagnostic_for_operation(item, operation)
        for item in report.diagnostics
    )
    if any(item.blocking for item in diagnostics):
        return AuthoringCapability(
            operation,
            AuthoringStatus.UNAVAILABLE,
            diagnostics,
        )
    automatic = tuple(
        entry
        for entry in report.entries
        if entry.frame.source != "explicit"
    )
    if automatic:
        warning = _assumed_orientation_diagnostic(
            target,
            operation,
            automatic,
        )
        return AuthoringCapability(
            operation,
            AuthoringStatus.LIMITED,
            (*diagnostics, warning),
        )
    return AuthoringCapability(
        operation,
        AuthoringStatus.ENABLED,
        diagnostics,
    )


def _candidate_decision(
    operation: str,
    diagnostics: Iterable[PreflightDiagnostic],
) -> AuthoringCapability:
    owned = tuple(
        _diagnostic_for_operation(item, operation)
        for item in deepcopy(tuple(diagnostics))
    )
    status = (
        AuthoringStatus.UNAVAILABLE
        if any(item.blocking for item in owned)
        else (
            AuthoringStatus.LIMITED
            if owned
            else AuthoringStatus.ENABLED
        )
    )
    return AuthoringCapability(operation, status, owned)


def _unsupported_candidate(
    operation: str,
    *,
    subject: Any,
    message: str,
) -> AuthoringCapability:
    diagnostic = PreflightDiagnostic(
        code="beam.orientation.unsupported_target",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=subject,
        path=("capabilities", "operations", operation),
        remediation="请选择支持该操作且只包含 Beam2 单元的目标区域。",
        details={"operation": operation},
    )
    return AuthoringCapability(
        operation,
        AuthoringStatus.UNAVAILABLE,
        (diagnostic,),
    )


def describe_model_capabilities(model: Any) -> ModelCapabilityReport:
    """Describe intrinsic facts and installed Phase 4 authoring policy."""

    output_support = _evaluate_output_requests(
        model,
        _model_output_requests(model),
    )
    elements = tuple(getattr(getattr(model, "mesh", None), "elements", ()))
    aggregate = _aggregate_capabilities(
        (getattr(element, "type", "") for element in elements),
        subject="model",
    )
    regions = tuple(
        describe_region_capabilities(model, reference)
        for reference in _model_region_refs(model)
    )
    installed_region_names = {
        str(getattr(section, "element_set", ""))
        for section in getattr(model, "sections", ())
    }
    installed_diagnostics = tuple(
        diagnostic
        for region in regions
        if (
            region.region.kind == "element_set"
            and region.region.name in installed_region_names
        )
        for diagnostic in region.diagnostics
    )
    model_diagnostics = _deduplicate_diagnostics(
        (*aggregate.diagnostics, *installed_diagnostics)
    )
    output_capabilities = _output_authoring_capabilities(
        supports_candidate=output_support.supports_output_authoring,
    )
    section_status = (
        AuthoringStatus.UNAVAILABLE
        if not aggregate.compatible or not aggregate.section_families
        else (
            AuthoringStatus.LIMITED
            if aggregate.diagnostics
            else AuthoringStatus.ENABLED
        )
    )
    line_regions = tuple(
        item
        for item in regions
        if item.region.kind == "element_set"
        and item.compatible
        and item.distributed_load_kinds == ("line",)
        and item.families == ("beam",)
    )
    line_status = (
        AuthoringStatus.ENABLED
        if line_regions
        else AuthoringStatus.UNAVAILABLE
    )
    authoring = (
        AuthoringCapability("section.create", section_status),
        AuthoringCapability("line_load.create", line_status),
        *output_capabilities,
    )
    return ModelCapabilityReport(
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        diagnostics=model_diagnostics,
        regions=regions,
        authoring=authoring,
        output_request_catalog=output_support.catalog,
    )


def describe_region_capabilities(
    model: Any,
    region: RegionRef,
) -> RegionCapability:
    """Describe a typed region using a safe element capability intersection."""

    if not isinstance(region, RegionRef):
        raise TypeError("region must be RegionRef")
    try:
        element_lookup = {
            int(element.id): element
            for element in getattr(
                getattr(model, "mesh", None),
                "elements",
                (),
            )
        }
        element_ids = _region_element_ids(model, region, element_lookup)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _missing_region_capability(region, str(error))
    element_types = tuple(
        element_lookup[element_id].type for element_id in element_ids
    )
    aggregate = _aggregate_capabilities(element_types, subject=region)
    operations, installed_diagnostics = _region_operation_capabilities(
        model,
        region,
        element_ids,
        aggregate,
    )
    return RegionCapability(
        region=region,
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        homogeneous=len(aggregate.canonical_element_types) <= 1,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        distributed_load_kinds=aggregate.distributed_load_kinds,
        diagnostics=_deduplicate_diagnostics(
            (*aggregate.diagnostics, *installed_diagnostics)
        ),
        operations=operations,
    )


def describe_native_authoring_capabilities(
    recipe: Any,
    mesh_settings: Any,
    *,
    named_regions: Iterable[Any] = (),
) -> ModelCapabilityReport:
    """Describe a native recipe before a mesh artifact exists."""

    from fem.geometry.recipes import NATIVE_GEOMETRY_TYPES

    if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        contract = describe_native_mesh_contract(recipe, mesh_settings)
        region_recipe = recipe
    else:
        # Keep the catalog-only compatibility seam used by existing callers
        # that ask about a prospective continuum shape before a recipe has
        # been constructed.  Resolve it through the same shared contract
        # helper as real authoring paths.
        from fem.mesh.settings import MeshSettings

        prospective_settings = (
            MeshSettings(1.0) if mesh_settings is None else mesh_settings
        )
        contract = describe_native_mesh_settings_contract(prospective_settings)
        region_recipe = None
    line_diagnostic = None
    if not contract.complete:
        line_diagnostic = PreflightDiagnostic(
            code="native.line.formulation_required",
            severity=PreflightSeverity.ERROR,
            stage=PreflightStage.CAPABILITY,
            message=(
                "native 1D geometry requires an explicit Truss2 or Beam2 "
                "line formulation before mesh generation"
            ),
            subject=getattr(recipe, "name", "native_mesh"),
            path=("mesh_settings", "line_element_type"),
            remediation="设置 MeshSettings.line_element_type 为 Truss2 或 Beam2。",
        )
        aggregate = _empty_capability_aggregate((line_diagnostic,))
        result_catalog = None
    else:
        aggregate = _aggregate_capabilities(
            (contract.canonical_element_type,),
            subject="native_mesh",
        )
        profile = classify_result_element_types(
            (contract.canonical_element_type,),
            dofs_per_node=aggregate.dofs_per_node,
        )
        result_catalog = ResultCapabilityCatalog.from_profile(profile)
    native_regions = (
        _native_region_capabilities(
            region_recipe,
            named_regions,
            contract,
            aggregate,
            line_diagnostic,
        )
        if region_recipe is not None
        else ()
    )
    output_capabilities = _output_authoring_capabilities(
        supports_candidate=(
            result_catalog is not None and bool(result_catalog.candidates)
        ),
    )
    unavailable_operations = (
        "mesh.generate",
        "model.validate",
        "analysis.run",
        "line_load.create",
    )
    if contract.complete:
        authoring = (
            AuthoringCapability(
                "section.create",
                (
                    AuthoringStatus.ENABLED
                    if aggregate.compatible
                    else AuthoringStatus.UNAVAILABLE
                ),
            ),
            AuthoringCapability(
                "line_load.create",
                (
                    AuthoringStatus.ENABLED
                    if aggregate.families == ("beam",)
                    else AuthoringStatus.UNAVAILABLE
                ),
            ),
            *output_capabilities,
        )
    else:
        authoring = tuple(
            AuthoringCapability(
                operation,
                AuthoringStatus.UNAVAILABLE,
                (line_diagnostic,),
            )
            for operation in (
                "section.create",
                *unavailable_operations,
                "output_request.create",
            )
        )
    return ModelCapabilityReport(
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        compatible=aggregate.compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        diagnostics=aggregate.diagnostics,
        regions=native_regions,
        authoring=authoring,
        output_request_catalog=result_catalog,
    )


def describe_session_authoring(snapshot: Any) -> SessionAuthoringProjection:
    """Project one detached Session snapshot into typed authoring facts.

    This function owns target namespace construction and Step lifecycle
    decisions for front ends.  It is intentionally Qt-free and never mutates
    the supplied snapshot or any model it references.
    """

    if snapshot is None or not hasattr(snapshot, "source_kind"):
        raise TypeError("snapshot must be a SessionSnapshot-compatible value")

    model = getattr(snapshot, "model", None)
    recipe = getattr(snapshot, "geometry_recipe", None)
    mesh_settings = getattr(snapshot, "mesh_settings", None)
    if model is not None:
        report = describe_model_capabilities(model)
    elif recipe is not None and getattr(snapshot, "source_kind", None) == "native":
        report = describe_native_authoring_capabilities(
            recipe,
            mesh_settings,
            named_regions=getattr(snapshot, "named_regions", ()),
        )
    else:
        report = _empty_model_capability_report()

    target_regions: list[tuple[RegionRef, frozenset[str]]] = []
    if recipe is not None and getattr(snapshot, "source_kind", None) == "native":
        from .native_regions import describe_native_regions

        descriptors = describe_native_regions(
            recipe,
            getattr(snapshot, "named_regions", ()),
            mesh_settings=mesh_settings,
        )
        for descriptor in descriptors:
            for product, kind in (
                ("node_set", "node_set"),
                ("element_set", "element_set"),
                ("beam_element_set", "element_set"),
                ("edge", "edge"),
                ("surface", "surface"),
            ):
                if product in descriptor.products:
                    _append_target_region(
                        target_regions,
                        RegionRef(kind, descriptor.name),
                        descriptor.products,
                    )

    for item in report.regions:
        _append_target_region(
            target_regions,
            item.region,
            frozenset({item.region.kind}),
        )

    targets = tuple(
        AuthoringTarget(
            region,
            _operations_for_target(region, products, report),
        )
        for region, products in sorted(
            target_regions,
            key=lambda item: (
                {"node_set": 0, "element_set": 1, "edge": 2, "surface": 3}[
                    item[0].kind
                ],
                item[0].name.casefold(),
                item[0].name,
            ),
        )
    )

    model_current = bool(getattr(snapshot, "model_current", False))
    lifecycles: list[StepLifecycleProjection] = []
    for name in tuple(getattr(snapshot, "runnable_step_names")()):
        can_check = model_current
        validation_current = bool(
            getattr(snapshot, "validation_current")(name)
        ) if can_check else False
        check_reason = (
            "当前模型制品可检查"
            if can_check
            else "请先生成网格或打开包含分析步的 INP 模型"
        )
        submit_reason = (
            "当前分析步已通过模型检查"
            if validation_current
            else (
                "请先通过当前分析步的模型检查"
                if can_check
                else check_reason
            )
        )
        lifecycles.append(
            StepLifecycleProjection(
                step_name=name,
                can_check=can_check,
                can_submit=validation_current,
                check_reason=check_reason,
                submit_reason=submit_reason,
            )
        )
    return SessionAuthoringProjection(
        report,
        targets,
        tuple(lifecycles),
        _session_output_authoring_capabilities(snapshot, report),
    )


def _session_output_authoring_capabilities(
    snapshot: Any,
    report: ModelCapabilityReport,
) -> tuple[AuthoringCapability, ...]:
    """Merge intrinsic output support with Session-owned lifecycle facts."""

    steps = tuple(getattr(snapshot, "steps", ()))
    editable_steps = tuple(
        step
        for step in steps
        if (
            (name := str(getattr(step, "name", "")).strip())
            and name.casefold() != "initial"
        )
    )
    existing = tuple(
        request
        for step in steps
        for request in tuple(getattr(step, "outputs", ()))
    )
    deletable = tuple(
        request
        for step in editable_steps
        for request in tuple(getattr(step, "outputs", ()))
    )
    idle = (
        getattr(snapshot, "running_run_id", None) is None
        and not bool(getattr(snapshot, "busy", False))
    )
    intrinsic_create = report.operation("output_request.create")
    create_enabled = (
        intrinsic_create.status is AuthoringStatus.ENABLED
        and bool(editable_steps)
        and idle
    )
    return (
        AuthoringCapability(
            "output_request.create",
            (
                AuthoringStatus.ENABLED
                if create_enabled
                else AuthoringStatus.UNAVAILABLE
            ),
            intrinsic_create.diagnostics,
        ),
        AuthoringCapability(
            "output_request.view",
            (
                AuthoringStatus.READ_ONLY
                if existing
                else AuthoringStatus.UNAVAILABLE
            ),
        ),
        AuthoringCapability(
            "output_request.delete",
            (
                AuthoringStatus.ENABLED
                if deletable and idle
                else AuthoringStatus.UNAVAILABLE
            ),
        ),
    )


def _append_target_region(
    target_regions: list[tuple[RegionRef, frozenset[str]]],
    region: RegionRef,
    products: frozenset[str],
) -> None:
    for index, (current, current_products) in enumerate(target_regions):
        if current == region:
            target_regions[index] = (current, current_products | products)
            return
    target_regions.append((region, products))


def _operations_for_target(
    region: RegionRef,
    products: frozenset[str],
    report: ModelCapabilityReport,
) -> tuple[AuthoringCapability, ...]:
    region_report = next(
        (item for item in report.regions if item.region == region),
        None,
    )
    diagnostics = () if region_report is None else region_report.diagnostics
    compatible = report.compatible and (
        region_report is None or region_report.compatible
    )
    load_kinds = set(report.load_kinds)
    operations: list[AuthoringCapability] = []

    def add(name: str, enabled: bool) -> None:
        operations.append(
            AuthoringCapability(
                name,
                AuthoringStatus.ENABLED if enabled else AuthoringStatus.UNAVAILABLE,
                diagnostics if not enabled else (),
            )
        )

    if region.kind == "node_set" or "node_set" in products:
        add("boundary.displacement", True)
        add("load.node", compatible and "node" in load_kinds)
    if region.kind == "edge" or "edge" in products:
        add("load.edge", compatible and "edge" in load_kinds)
    if region.kind == "surface" or "surface" in products:
        add("load.surface", compatible and "surface" in load_kinds)
    if region.kind == "element_set" or "element_set" in products:
        add("section.assignment", compatible and bool(report.section_families))
        line_enabled = bool(
            region_report is not None
            and region_report.supports_distributed_load("line")
        )
        add("load.line.global", line_enabled)
        local = (
            None
            if region_report is None
            else region_report.operation("load.line.local")
        )
        if local is not None and local.status is not AuthoringStatus.UNAVAILABLE:
            operations[-1] = region_report.operation("load.line.global")
            operations.append(local)

    if region_report is not None:
        by_name = {item.operation: item for item in operations}
        by_name.update({item.operation: item for item in region_report.operations})
        operations = list(by_name.values())
    return tuple(operations)


def _empty_model_capability_report() -> ModelCapabilityReport:
    return ModelCapabilityReport(
        canonical_element_types=(),
        families=(),
        compatible=False,
        topological_dimension=None,
        spatial_dimension=None,
        dofs_per_node=None,
        dof_labels=(),
        force_labels=(),
        section_families=(),
        section_presets=(),
        load_kinds=(),
        diagnostics=(),
        regions=(),
        authoring=(
            AuthoringCapability("section.create", AuthoringStatus.UNAVAILABLE),
            AuthoringCapability(
                "output_request.create",
                AuthoringStatus.UNAVAILABLE,
            ),
        ),
        output_request_catalog=None,
    )


def _empty_capability_aggregate(
    diagnostics: tuple[PreflightDiagnostic, ...],
) -> _CapabilityAggregate:
    return _CapabilityAggregate(
        canonical_element_types=(),
        families=(),
        compatible=False,
        topological_dimension=None,
        spatial_dimension=None,
        dofs_per_node=None,
        dof_labels=(),
        force_labels=(),
        section_families=(),
        section_presets=(),
        load_kinds=(),
        distributed_load_kinds=(),
        requirements=(),
        diagnostics=diagnostics,
    )


def _native_region_capabilities(
    recipe: Any,
    named_regions: Iterable[Any],
    contract: NativeMeshContract,
    aggregate: _CapabilityAggregate,
    line_diagnostic: PreflightDiagnostic | None,
) -> tuple[RegionCapability, ...]:
    """Project stable pre-mesh region products into capability reports."""

    from .native_regions import describe_native_regions

    descriptors = describe_native_regions(
        recipe,
        named_regions,
        mesh_contract=contract,
    )
    result: list[RegionCapability] = []
    seen: set[RegionRef] = set()
    diagnostics = (
        (line_diagnostic,)
        if line_diagnostic is not None
        else aggregate.diagnostics
    )
    for descriptor in descriptors:
        products = set(descriptor.products)
        if "node_set" in products:
            region = RegionRef("node_set", descriptor.name)
            if region not in seen:
                result.append(
                    _native_region_capability(
                        region,
                        aggregate,
                        diagnostics,
                        contract,
                    )
                )
                seen.add(region)
        if "element_set" in products or "beam_element_set" in products:
            region = RegionRef("element_set", descriptor.name)
            if region not in seen:
                result.append(
                    _native_region_capability(
                        region,
                        aggregate,
                        diagnostics,
                        contract,
                    )
                )
                seen.add(region)
        for product in ("edge", "surface"):
            if product not in products:
                continue
            region = RegionRef(product, descriptor.name)
            if region in seen:
                continue
            result.append(
                _native_region_capability(
                    region,
                    aggregate,
                    diagnostics,
                    contract,
                )
            )
            seen.add(region)
    return tuple(result)


def _native_region_capability(
    region: RegionRef,
    aggregate: _CapabilityAggregate,
    diagnostics: tuple[PreflightDiagnostic, ...],
    contract: NativeMeshContract,
) -> RegionCapability:
    compatible = aggregate.compatible and not any(
        item.blocking for item in diagnostics
    )
    operations: list[AuthoringCapability] = []

    def add(name: str, enabled: bool, extra: tuple[PreflightDiagnostic, ...] = ()) -> None:
        status = (
            AuthoringStatus.LIMITED
            if enabled and extra
            else AuthoringStatus.ENABLED
            if enabled
            else AuthoringStatus.UNAVAILABLE
        )
        operations.append(
            AuthoringCapability(
                name,
                status,
                extra if enabled else diagnostics,
            )
        )

    if region.kind == "node_set":
        add("boundary.displacement", compatible)
        add("load.node", compatible and "node" in aggregate.load_kinds)
    elif region.kind == "element_set":
        add(
            "section.assignment",
            compatible and bool(aggregate.section_families),
        )
        is_beam = compatible and aggregate.families == ("beam",)
        add("load.line.global", is_beam and "line" in aggregate.load_kinds)
        if is_beam:
            automatic = PreflightDiagnostic(
                code="beam.orientation.assumed",
                severity=PreflightSeverity.WARNING,
                stage=PreflightStage.CAPABILITY,
                message=(
                    "Beam2 local line-load authoring uses the automatic local "
                    "frame until an explicit orientation is installed"
                ),
                subject=region,
                path=("capabilities", "operations", "load.line.local"),
                remediation="如需方向敏感的局部线载荷，请提供 Beam2 显式方向。",
            )
            add("load.line.local", True, (automatic,))
    elif region.kind == "edge":
        add("load.edge", compatible and "edge" in aggregate.load_kinds)
    elif region.kind == "surface":
        add("load.surface", compatible and "surface" in aggregate.load_kinds)

    return RegionCapability(
        region=region,
        canonical_element_types=aggregate.canonical_element_types,
        families=aggregate.families,
        homogeneous=len(aggregate.families) == 1,
        compatible=compatible,
        topological_dimension=aggregate.topological_dimension,
        spatial_dimension=aggregate.spatial_dimension,
        dofs_per_node=aggregate.dofs_per_node,
        dof_labels=aggregate.dof_labels,
        force_labels=aggregate.force_labels,
        section_families=aggregate.section_families,
        section_presets=aggregate.section_presets,
        load_kinds=aggregate.load_kinds,
        distributed_load_kinds=aggregate.distributed_load_kinds,
        diagnostics=diagnostics,
        operations=tuple(operations),
    )


@dataclass(frozen=True, slots=True)
class _CapabilityAggregate:
    canonical_element_types: tuple[str, ...]
    families: tuple[str, ...]
    compatible: bool
    topological_dimension: int | None
    spatial_dimension: int | None
    dofs_per_node: int | None
    dof_labels: tuple[str, ...]
    force_labels: tuple[str, ...]
    section_families: tuple[str, ...]
    section_presets: tuple[str, ...]
    load_kinds: tuple[str, ...]
    distributed_load_kinds: tuple[str, ...]
    requirements: tuple[ElementCapabilityRequirement, ...]
    diagnostics: tuple[PreflightDiagnostic, ...]


def _aggregate_capabilities(
    element_types: Iterable[Any],
    *,
    subject: Any,
) -> _CapabilityAggregate:
    descriptors: list[ElementCapabilityDescriptor] = []
    diagnostics: list[PreflightDiagnostic] = []
    seen: set[str] = set()
    for element_type in element_types:
        try:
            descriptor = get_element_capabilities(str(element_type))
        except (NotImplementedError, TypeError, ValueError) as error:
            diagnostics.append(
                _unsupported_mix_diagnostic(subject, str(error))
            )
            continue
        if descriptor.canonical_type.casefold() not in seen:
            descriptors.append(descriptor)
            seen.add(descriptor.canonical_type.casefold())
    if not descriptors:
        if not diagnostics:
            diagnostics.append(
                _unsupported_mix_diagnostic(
                    subject,
                    "region contains no elements",
                )
            )
        return _CapabilityAggregate(
            canonical_element_types=(),
            families=(),
            compatible=False,
            topological_dimension=None,
            spatial_dimension=None,
            dofs_per_node=None,
            dof_labels=(),
            force_labels=(),
            section_families=(),
            section_presets=(),
            load_kinds=(),
            distributed_load_kinds=(),
            requirements=(),
            diagnostics=tuple(diagnostics),
        )

    families = _ordered_unique(item.family for item in descriptors)
    topologies = _ordered_unique(
        item.topological_dimension for item in descriptors
    )
    spatial_dimensions = _ordered_unique(
        item.spatial_dimension for item in descriptors
    )
    dof_profiles = _ordered_unique(
        (item.dofs_per_node, item.dof_labels, item.force_labels)
        for item in descriptors
    )
    section_families = _intersection(
        item.section_families for item in descriptors
    )
    load_kinds = _intersection(item.load_kinds for item in descriptors)
    requirements = _aggregate_requirements(descriptors)
    compatible = (
        len(families) == 1
        and len(topologies) == 1
        and len(spatial_dimensions) == 1
        and len(dof_profiles) == 1
        and bool(section_families)
        and bool(load_kinds)
        and not diagnostics
    )
    if not compatible:
        diagnostics.append(
            _unsupported_mix_diagnostic(
                subject,
                "element families or DOF profiles have no safe common contract",
            )
        )
    for descriptor in descriptors:
        for limitation in descriptor.limitations:
            if any(
                existing.code == limitation.code
                for existing in diagnostics
            ):
                continue
            diagnostics.append(
                PreflightDiagnostic(
                    code=limitation.code,
                    severity=PreflightSeverity.WARNING,
                    stage=PreflightStage.CAPABILITY,
                    message=limitation.message,
                    subject=subject,
                    path=("capabilities", descriptor.canonical_type),
                    remediation=(
                        "当前局部轴由单元几何自动确定；请核对方向假设。"
                    ),
                    details={"operations": limitation.operations},
                )
            )
    family = families[0] if len(families) == 1 else None
    profile = dof_profiles[0] if len(dof_profiles) == 1 else None
    return _CapabilityAggregate(
        canonical_element_types=tuple(
            item.canonical_type for item in descriptors
        ),
        families=families,
        compatible=compatible,
        topological_dimension=(
            topologies[0] if len(topologies) == 1 else None
        ),
        spatial_dimension=(
            spatial_dimensions[0]
            if len(spatial_dimensions) == 1
            else None
        ),
        dofs_per_node=None if profile is None else profile[0],
        dof_labels=() if profile is None else profile[1],
        force_labels=() if profile is None else profile[2],
        section_families=section_families,
        section_presets=(
            () if family is None else _section_presets(family)
        ),
        load_kinds=load_kinds,
        distributed_load_kinds=tuple(
            kind for kind in load_kinds
            if kind in _DISTRIBUTED_LOAD_KINDS
        ),
        requirements=requirements,
        diagnostics=tuple(diagnostics),
    )


def _region_operation_capabilities(
    model: Any,
    region: RegionRef,
    element_ids: tuple[int, ...],
    aggregate: _CapabilityAggregate,
) -> tuple[
    tuple[AuthoringCapability, ...],
    tuple[PreflightDiagnostic, ...],
]:
    requirement_operations = _ordered_unique(
        operation
        for requirement in aggregate.requirements
        for operation in requirement.operations
    )
    operation_names = requirement_operations
    if (
        aggregate.compatible
        and aggregate.families == ("beam",)
        and region.kind == "element_set"
    ):
        operation_names = _ordered_unique(
            (
                *requirement_operations,
                *(
                    f"section.{preset}"
                    for preset in aggregate.section_presets
                ),
                *(
                    ("load.line.global",)
                    if "line" in aggregate.distributed_load_kinds
                    else ()
                ),
            )
        )
    if not operation_names:
        return (), ()

    if (
        not aggregate.compatible
        or aggregate.families != ("beam",)
        or region.kind != "element_set"
    ):
        diagnostics = aggregate.diagnostics or (
            _unsupported_mix_diagnostic(
                region,
                "operation requires a compatible Beam2 element set",
            ),
        )
        return (
            tuple(
                AuthoringCapability(
                    operation,
                    AuthoringStatus.UNAVAILABLE,
                    diagnostics,
                )
                for operation in operation_names
            ),
            (),
        )

    entries: tuple[Any, ...] = ()
    frame_diagnostics: tuple[PreflightDiagnostic, ...] = ()
    mesh_nodes = tuple(
        getattr(getattr(model, "mesh", None), "nodes", ())
    )
    if mesh_nodes:
        # Lazy import avoids capabilities <-> beam_frames initialization.
        from .beam_frames import resolve_effective_beam_frames

        report = resolve_effective_beam_frames(model, region)
        entries = report.entries
        frame_diagnostics = report.diagnostics

    automatic_entries = tuple(
        entry
        for entry in entries
        if entry.frame.source != "explicit"
    )
    fallback_automatic = not entries and not frame_diagnostics
    decisions: list[AuthoringCapability] = []
    for operation in operation_names:
        operation_diagnostics = tuple(
            _diagnostic_for_operation(item, operation)
            for item in frame_diagnostics
        )
        if _aggregate_requires(
            aggregate,
            _EXPLICIT_BEAM_ORIENTATION_REQUIREMENT,
            operation,
        ) and (automatic_entries or fallback_automatic):
            operation_diagnostics = (
                *operation_diagnostics,
                _assumed_orientation_diagnostic(
                    region,
                    operation,
                    automatic_entries,
                    fallback_element_ids=(
                        element_ids if fallback_automatic else ()
                    ),
                ),
            )
        decisions.append(
            AuthoringCapability(
                operation,
                _status_from_diagnostics(operation_diagnostics),
                operation_diagnostics,
            )
        )

    installed_diagnostics = tuple(
        _diagnostic_for_operation(item, "section.assignment")
        for item in frame_diagnostics
    )
    rectangle_automatic = tuple(
        entry
        for entry in automatic_entries
        if _is_rectangle_section_type(entry.section_type)
    )
    if (
        rectangle_automatic
        and _aggregate_requires(
            aggregate,
            _EXPLICIT_BEAM_ORIENTATION_REQUIREMENT,
            "section.rectangle",
        )
    ):
        installed_diagnostics = (
            *installed_diagnostics,
            _assumed_orientation_diagnostic(
                region,
                "section.rectangle",
                rectangle_automatic,
            ),
        )
    return tuple(decisions), installed_diagnostics


def _aggregate_requirements(
    descriptors: Iterable[ElementCapabilityDescriptor],
) -> tuple[ElementCapabilityRequirement, ...]:
    operations_by_code: dict[str, list[str]] = {}
    order: list[str] = []
    for descriptor in descriptors:
        for requirement in descriptor.requirements:
            if requirement.code not in operations_by_code:
                operations_by_code[requirement.code] = []
                order.append(requirement.code)
            operations = operations_by_code[requirement.code]
            for operation in requirement.operations:
                if operation not in operations:
                    operations.append(operation)
    return tuple(
        ElementCapabilityRequirement(
            code=code,
            operations=tuple(operations_by_code[code]),
        )
        for code in order
    )


def _aggregate_requires(
    aggregate: _CapabilityAggregate,
    requirement_code: str,
    operation: str,
) -> bool:
    normalized = _normalize_operation(operation)
    return any(
        requirement.code == requirement_code
        and normalized in requirement.operations
        for requirement in aggregate.requirements
    )


def _requires_explicit_beam_orientation(
    model: Any,
    target: RegionRef | int,
    operation: str,
) -> bool:
    elements = tuple(
        getattr(getattr(model, "mesh", None), "elements", ())
    )
    lookup = {int(element.id): element for element in elements}
    try:
        if isinstance(target, bool):
            return False
        if isinstance(target, int):
            element_types = (lookup[target].type,)
        elif isinstance(target, RegionRef):
            element_types = tuple(
                lookup[element_id].type
                for element_id in _region_element_ids(
                    model,
                    target,
                    lookup,
                )
            )
        else:
            return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    aggregate = _aggregate_capabilities(
        element_types,
        subject=target,
    )
    return (
        aggregate.compatible
        and aggregate.families == ("beam",)
        and _aggregate_requires(
            aggregate,
            _EXPLICIT_BEAM_ORIENTATION_REQUIREMENT,
            operation,
        )
    )


def _supports_target_operation(
    model: Any,
    target: RegionRef | int,
    operation: str,
) -> bool:
    elements = tuple(
        getattr(getattr(model, "mesh", None), "elements", ())
    )
    lookup = {int(element.id): element for element in elements}
    try:
        if isinstance(target, bool):
            return False
        if isinstance(target, int):
            element_types = (lookup[target].type,)
        elif isinstance(target, RegionRef):
            element_types = tuple(
                lookup[element_id].type
                for element_id in _region_element_ids(
                    model,
                    target,
                    lookup,
                )
            )
        else:
            return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False

    aggregate = _aggregate_capabilities(
        element_types,
        subject=target,
    )
    if not aggregate.compatible:
        return False
    normalized = _normalize_operation(operation)
    if normalized.startswith("section."):
        return _supports_section_preset(
            aggregate.section_presets,
            normalized.removeprefix("section."),
        )
    if normalized in {"load.line.global", "load.line.local"}:
        return (
            aggregate.families == ("beam",)
            and "line" in aggregate.distributed_load_kinds
        )
    return False


def _validated_candidate_index(
    value: Any,
    size: int,
    label: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} candidate_index must be an integer")
    index = int(value)
    if index < 0 or index >= size:
        raise IndexError(
            f"{label} candidate_index {index} is outside 0..{size - 1}"
        )
    return index


def _assumed_orientation_diagnostic(
    subject: RegionRef | int,
    operation: str,
    entries: Iterable[Any],
    *,
    fallback_element_ids: tuple[int, ...] = (),
) -> PreflightDiagnostic:
    owned_entries = tuple(entries)
    element_ids = tuple(
        int(entry.element_id) for entry in owned_entries
    ) or tuple(int(value) for value in fallback_element_ids)
    assignment_indexes = _ordered_unique(
        int(entry.assignment_index)
        for entry in owned_entries
        if entry.assignment_index is not None
    )
    tangents = tuple(
        (
            int(entry.element_id),
            tuple(float(value) for value in entry.frame.local_x),
        )
        for entry in owned_entries
    )
    details: dict[str, Any] = {
        "operation": _normalize_operation(operation),
        "element_ids": element_ids,
        "assignment_indexes": assignment_indexes,
        "reference": None,
        "element_tangents": tangents,
    }
    if len(element_ids) == 1:
        details["element_id"] = element_ids[0]
    if len(assignment_indexes) == 1:
        details["assignment_index"] = assignment_indexes[0]
    return PreflightDiagnostic(
        code="beam.orientation.assumed",
        severity=PreflightSeverity.WARNING,
        stage=PreflightStage.CAPABILITY,
        message=(
            f"{operation} uses an automatic Beam2 local frame for "
            f"{len(element_ids)} target element(s)."
        ),
        subject=subject,
        path=(
            "capabilities",
            "operations",
            _normalize_operation(operation),
        ),
        remediation=(
            "旧模型可继续执行；新建或保存该方向敏感定义时请提供显式"
            "局部 y 参考方向。"
        ),
        details=details,
    )


def _diagnostic_for_operation(
    diagnostic: PreflightDiagnostic,
    operation: str,
) -> PreflightDiagnostic:
    details = diagnostic.details_dict()
    details["operation"] = _normalize_operation(operation)
    return PreflightDiagnostic(
        code=diagnostic.code,
        severity=diagnostic.severity,
        stage=diagnostic.stage,
        message=diagnostic.message,
        subject=diagnostic.subject,
        path=diagnostic.path,
        remediation=diagnostic.remediation,
        details=details,
    )


def _status_from_diagnostics(
    diagnostics: Iterable[PreflightDiagnostic],
) -> AuthoringStatus:
    owned = tuple(diagnostics)
    if any(item.blocking for item in owned):
        return AuthoringStatus.UNAVAILABLE
    if owned:
        return AuthoringStatus.LIMITED
    return AuthoringStatus.ENABLED


def _section_operation(section: Any) -> str:
    section_type = str(
        getattr(section, "section_type", "")
    ).strip().casefold()
    if section_type == "beam":
        section_type = str(
            getattr(section, "properties", {}).get(
                "section_type",
                section_type,
            )
        ).strip().casefold()
    return f"section.{section_type}"


def _is_rectangle_section_type(section_type: Any) -> bool:
    return str(section_type or "").strip().casefold() == "rectangle"


def _normalize_operation(operation: Any) -> str:
    return str(operation).strip().casefold()


def _deduplicate_diagnostics(
    diagnostics: Iterable[PreflightDiagnostic],
) -> tuple[PreflightDiagnostic, ...]:
    result: list[PreflightDiagnostic] = []
    seen: set[str] = set()
    for diagnostic in diagnostics:
        identity = repr(
            (
                diagnostic.code,
                diagnostic.severity.value,
                diagnostic.stage.value,
                diagnostic.subject,
                diagnostic.path,
                diagnostic.details,
            )
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(diagnostic)
    return tuple(result)


def _model_region_refs(model: Any) -> tuple[RegionRef, ...]:
    refs: list[RegionRef] = []
    for kind, attribute in (
        ("node_set", "node_sets"),
        ("element_set", "element_sets"),
        ("edge", "edges"),
        ("surface", "surfaces"),
    ):
        refs.extend(
            RegionRef(kind, str(name))
            for name in getattr(model, attribute, {})
        )
    internal = getattr(model, "metadata", {}).get(
        "_abaqus_internal_element_sets",
        {},
    )
    public_names = {
        item.name
        for item in refs
        if item.kind == "element_set"
    }
    refs.extend(
        RegionRef("element_set", str(name))
        for name in internal
        if str(name) not in public_names
    )
    return tuple(refs)


def _region_element_ids(
    model: Any,
    region: RegionRef,
    element_lookup: dict[int, Any],
) -> tuple[int, ...]:
    if region.kind == "element_set":
        public = getattr(model, "element_sets", {})
        internal = getattr(model, "metadata", {}).get(
            "_abaqus_internal_element_sets",
            {},
        )
        collection = (
            public[region.name]
            if region.name in public
            else internal[region.name]
        )
        ids = tuple(int(value) for value in collection.element_ids)
    elif region.kind == "edge":
        collection = getattr(model, "edges", {})[region.name]
        ids = tuple(int(entry.elem_id) for entry in collection.edges)
    elif region.kind == "surface":
        collection = getattr(model, "surfaces", {})[region.name]
        ids = tuple(int(entry.elem_id) for entry in collection.faces)
    else:
        node_set = getattr(model, "node_sets", {})[region.name]
        node_ids = {int(value) for value in node_set.node_ids}
        ids = tuple(
            element_id
            for element_id, element in element_lookup.items()
            if any(int(node_id) in node_ids for node_id in element.node_ids)
        )
    unique = _ordered_unique(ids)
    missing = tuple(
        element_id
        for element_id in unique
        if element_id not in element_lookup
    )
    if missing:
        raise KeyError(f"region references missing element {missing[0]}")
    return unique


def _intersection(
    values: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    collections = tuple(values)
    if not collections:
        return ()
    common = set(collections[0])
    for collection in collections[1:]:
        common.intersection_update(collection)
    return tuple(value for value in collections[0] if value in common)


def _supports_section_preset(
    presets: tuple[str, ...],
    section_type: str,
) -> bool:
    normalized = str(section_type).strip().casefold()
    if normalized == "solid":
        return any(
            preset == "solid" or preset.startswith("solid_plane_")
            for preset in presets
        )
    return normalized in presets


def _section_presets(element_family: str) -> tuple[str, ...]:
    from fem.materials import section_presets_for_element_family

    return section_presets_for_element_family(element_family)


def _ordered_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _unsupported_mix_diagnostic(
    subject: Any,
    message: str,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        code="model.capability.unsupported_mix",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message,
        subject=subject,
        path=("capabilities",),
        remediation="请选择具有共同单元族与自由度契约的区域。",
    )


def _missing_region_capability(
    region: RegionRef,
    message: str | None = None,
) -> RegionCapability:
    diagnostic = PreflightDiagnostic(
        code="step.reference.invalid",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.CAPABILITY,
        message=message or f"region {region.name!r} is not defined",
        subject=region,
        path=("regions", region.kind, region.name),
        remediation="请选择当前模型中存在的同类命名区域。",
    )
    return RegionCapability(
        region=region,
        canonical_element_types=(),
        families=(),
        homogeneous=False,
        compatible=False,
        topological_dimension=None,
        spatial_dimension=None,
        dofs_per_node=None,
        dof_labels=(),
        force_labels=(),
        section_families=(),
        section_presets=(),
        load_kinds=(),
        distributed_load_kinds=(),
        diagnostics=(diagnostic,),
    )


__all__ = [
    "AuthoringCapability",
    "AuthoringStatus",
    "AuthoringTarget",
    "ModelCapabilityReport",
    "RegionCapability",
    "RegionRef",
    "SessionAuthoringProjection",
    "StepLifecycleProjection",
    "describe_model_capabilities",
    "describe_native_authoring_capabilities",
    "describe_region_capabilities",
    "describe_session_authoring",
    "evaluate_authoring_candidate",
    "require_region_kind",
]
