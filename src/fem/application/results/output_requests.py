"""Pure projection from preserved output authoring to executable field requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fem.core.model import OutputRequest, OutputSourceEvidence

from .data import ResultDiagnostic
from .fields import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultVariable,
)
from .registry import (
    ElementResultProfile,
    FieldRegistryEntry,
    ResultModelFamily,
    catalog_diagnostics,
    catalog_entries,
)


_VARIABLE_ORDER = (
    ResultVariable.U,
    ResultVariable.UR,
    ResultVariable.RF,
    ResultVariable.RM,
    ResultVariable.S,
)
_VARIABLE_LOOKUP = {
    variable.value.casefold(): variable for variable in _VARIABLE_ORDER
}
_PRIMARY_TARGETS = {
    ResultVariable.U: "node",
    ResultVariable.UR: "node",
    ResultVariable.RF: "node",
    ResultVariable.RM: "node",
    ResultVariable.S: "element",
}
_EXECUTABLE_STRESS_POSITIONS = {
    ResultModelFamily.PLANE_CONTINUUM: frozenset(
        {
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
        }
    ),
    ResultModelFamily.SOLID_CONTINUUM: frozenset(
        {
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
        }
    ),
    ResultModelFamily.TRUSS: frozenset({FieldPosition.CENTROID}),
    ResultModelFamily.BEAM: frozenset({FieldPosition.SECTION_END}),
}
_DEFAULT_STRESS_POSITION = {
    ResultModelFamily.PLANE_CONTINUUM: FieldPosition.INTEGRATION_POINT,
    ResultModelFamily.SOLID_CONTINUUM: FieldPosition.INTEGRATION_POINT,
    ResultModelFamily.TRUSS: FieldPosition.CENTROID,
    ResultModelFamily.BEAM: FieldPosition.SECTION_END,
}
_POSITION_LOOKUP = {
    position.value.casefold(): position for position in FieldPosition
}
_STRUCTURAL_ABAQUS_FLAGS = frozenset({"field", "history"})
_ABSENT = object()


@dataclass(frozen=True, slots=True)
class ResultCapabilityCatalog:
    """Exact registry projection used by every OutputRequest consumer."""

    profile: ElementResultProfile
    entries: tuple[FieldRegistryEntry, ...]

    def __post_init__(self) -> None:
        if type(self.profile) is not ElementResultProfile:
            raise TypeError("profile must be ElementResultProfile")
        if type(self.entries) is not tuple:
            raise TypeError("entries must be a tuple")
        if any(type(entry) is not FieldRegistryEntry for entry in self.entries):
            raise TypeError("entries must contain only FieldRegistryEntry values")
        expected = catalog_entries(self.profile)
        if self.entries != expected:
            raise ValueError(
                "entries must exactly match the contextual registry catalog"
            )

    @classmethod
    def from_profile(
        cls,
        profile: ElementResultProfile,
    ) -> ResultCapabilityCatalog:
        """Build capabilities from the sole contextual registry factory."""

        if type(profile) is not ElementResultProfile:
            raise TypeError("profile must be ElementResultProfile")
        return cls(profile, catalog_entries(profile))

    @classmethod
    def from_entries(
        cls,
        profile: ElementResultProfile,
        entries: tuple[FieldRegistryEntry, ...],
    ) -> ResultCapabilityCatalog:
        """Validate already-computed contextual registry entries."""

        return cls(profile, entries)

    @property
    def diagnostics(self) -> tuple[ResultDiagnostic, ...]:
        """Return the registry-owned profile diagnostics."""

        return catalog_diagnostics(self.profile)

    def entry_for(
        self,
        field_id: ResultFieldId,
    ) -> FieldRegistryEntry | None:
        """Return one exact registry entry without inventing fallback fields."""

        if type(field_id) is not ResultFieldId:
            raise TypeError("field_id must be ResultFieldId")
        for entry in self.entries:
            if entry.descriptor.field_id == field_id:
                return entry
        return None


@dataclass(frozen=True, slots=True)
class OutputVariableProjection:
    """One canonical variable and all authoring occurrences that map to it."""

    source_variable_indices: tuple[int, ...]
    source_variables: tuple[str, ...]
    canonical_variable: ResultVariable | None
    field_requests: tuple[FieldRequest, ...]
    diagnostics: tuple[ResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        indices = _strict_source_indices(self.source_variable_indices)
        variables = _strict_string_tuple(
            self.source_variables,
            label="source_variables",
        )
        if len(indices) != len(variables):
            raise ValueError(
                "source_variable_indices and source_variables must align"
            )
        if (
            self.canonical_variable is not None
            and type(self.canonical_variable) is not ResultVariable
        ):
            raise TypeError(
                "canonical_variable must be ResultVariable or None"
            )
        _validate_field_requests(self.field_requests)
        _validate_diagnostics(self.diagnostics)
        if self.canonical_variable is None and self.field_requests:
            raise ValueError(
                "unknown variable projections cannot contain field requests"
            )
        if any(
            request.field_id.variable is not self.canonical_variable
            for request in self.field_requests
        ):
            raise ValueError(
                "field requests must match the canonical variable"
            )


@dataclass(frozen=True, slots=True)
class ExecutableOutputRequest:
    """Canonical request that may be eagerly materialized atomically."""

    request_index: int
    kind: str
    target: str
    frequency: int
    variables: tuple[OutputVariableProjection, ...]
    field_requests: tuple[FieldRequest, ...]

    def __post_init__(self) -> None:
        _strict_request_index(self.request_index)
        if type(self.kind) is not str:
            raise TypeError("executable output kind must be a string")
        if self.kind != "field":
            raise ValueError("executable output kind must be 'field'")
        if type(self.target) is not str:
            raise TypeError("executable output target must be a string")
        if self.target not in {"node", "element"}:
            raise ValueError(
                "executable output target must be 'node' or 'element'"
            )
        if type(self.frequency) is not int:
            raise TypeError("executable output frequency must be an integer")
        if self.frequency != 1:
            raise ValueError("executable output frequency must be integer 1")
        if type(self.variables) is not tuple or not self.variables:
            raise ValueError(
                "executable variables must be a nonempty tuple"
            )
        for variable in self.variables:
            if type(variable) is not OutputVariableProjection:
                raise TypeError(
                    "variables must contain OutputVariableProjection values"
                )
            if variable.canonical_variable is None:
                raise ValueError(
                    "executable variables require a canonical variable"
                )
            if variable.diagnostics:
                raise ValueError(
                    "executable variables cannot contain diagnostics"
                )
            if not variable.field_requests:
                raise ValueError(
                    "executable variables require field requests"
                )
        canonical_variables = tuple(
            variable.canonical_variable for variable in self.variables
        )
        expected_variables = tuple(
            variable
            for variable in _VARIABLE_ORDER
            if variable in canonical_variables
        )
        if canonical_variables != expected_variables:
            raise ValueError(
                "executable variables must be unique and canonically ordered"
            )
        _validate_field_requests(self.field_requests)
        expected_requests = tuple(
            field_request
            for variable in self.variables
            for field_request in variable.field_requests
        )
        if self.field_requests != expected_requests:
            raise ValueError(
                "field_requests must flatten canonical variable requests"
            )


@dataclass(frozen=True, slots=True)
class OutputRequestProjection:
    """Atomic executable-or-unsupported projection of one authoring request."""

    request_index: int
    authoring_request: OutputRequest
    variables: tuple[OutputVariableProjection, ...]
    executable_request: ExecutableOutputRequest | None
    diagnostics: tuple[ResultDiagnostic, ...]

    def __post_init__(self) -> None:
        _strict_request_index(self.request_index)
        if type(self.authoring_request) is not OutputRequest:
            raise TypeError("authoring_request must be exactly OutputRequest")
        if type(self.variables) is not tuple:
            raise TypeError("variables must be a tuple")
        for variable in self.variables:
            if type(variable) is not OutputVariableProjection:
                raise TypeError(
                    "variables must contain OutputVariableProjection values"
                )
        if (
            self.executable_request is not None
            and type(self.executable_request) is not ExecutableOutputRequest
        ):
            raise TypeError(
                "executable_request must be ExecutableOutputRequest or None"
            )
        _validate_diagnostics(self.diagnostics)
        variable_diagnostics = tuple(
            diagnostic
            for variable in self.variables
            for diagnostic in variable.diagnostics
        )
        for diagnostic in variable_diagnostics:
            if diagnostic not in self.diagnostics:
                raise ValueError(
                    "variable diagnostics must appear in projection diagnostics"
                )
        source_occurrences = tuple(
            sorted(
                (
                    source_index,
                    source_variable,
                )
                for variable in self.variables
                for source_index, source_variable in zip(
                    variable.source_variable_indices,
                    variable.source_variables,
                    strict=True,
                )
            )
        )
        expected_occurrences = tuple(
            enumerate(self.authoring_request.variables)
        )
        if source_occurrences != expected_occurrences:
            raise ValueError(
                "variable projections must exactly cover authoring occurrences"
            )
        if self.diagnostics and self.executable_request is not None:
            raise ValueError(
                "diagnostic projections cannot contain an executable request"
            )
        if not self.diagnostics and self.executable_request is None:
            raise ValueError(
                "supported projections require an executable request"
            )
        if (
            self.executable_request is not None
            and self.executable_request.request_index != self.request_index
        ):
            raise ValueError(
                "executable request index must match the projection"
            )
        if (
            self.executable_request is not None
            and self.executable_request.variables != self.variables
        ):
            raise ValueError(
                "executable variables must match the supported projection"
            )

    @property
    def executable(self) -> bool:
        """Whether this authoring request passed atomic projection."""

        return self.executable_request is not None


def project_output_request(
    request: OutputRequest,
    capabilities: ResultCapabilityCatalog,
    *,
    request_index: int,
) -> OutputRequestProjection:
    """Project one preserved authoring request without mutating it."""

    if type(request) is not OutputRequest:
        raise TypeError("request must be exactly OutputRequest")
    if type(capabilities) is not ResultCapabilityCatalog:
        raise TypeError(
            "capabilities must be exactly ResultCapabilityCatalog"
        )
    _strict_request_index(request_index)

    grouped = _group_variables(request.variables)
    request_diagnostics: list[ResultDiagnostic] = []
    kind_supported = request.kind == "field"
    target_supported = request.target in {"node", "element"}
    if not kind_supported:
        request_diagnostics.append(
            _diagnostic(
                "output.request.kind_unsupported",
                request_index,
                suffix=("kind",),
                message=f"Output kind {request.kind!r} is not executable.",
                remediation="Use a field output request.",
                details={"kind": request.kind},
            )
        )
    if not target_supported:
        request_diagnostics.append(
            _diagnostic(
                "output.request.target_unsupported",
                request_index,
                suffix=("target",),
                message=f"Output target {request.target!r} is not executable.",
                remediation="Use the node or element output target.",
                details={"target": request.target},
            )
        )
    if not request.variables:
        request_diagnostics.append(
            _diagnostic(
                "output.request.variables_empty",
                request_index,
                suffix=("variables",),
                message="The output request has no variables.",
                remediation="Select at least one supported output variable.",
                details={},
            )
        )

    effective_metadata, metadata_diagnostics = _effective_metadata(
        request,
        request_index=request_index,
    )
    request_diagnostics.extend(metadata_diagnostics)

    semantic_supported = (
        kind_supported
        and target_supported
        and bool(request.variables)
    )
    frequency = 1
    position_value: object = _ABSENT
    if kind_supported and bool(request.variables):
        allow_position = _position_option_is_applicable(
            grouped,
            target=request.target,
        )
        metadata_result = _validate_metadata(
            effective_metadata,
            allow_position=allow_position,
            request_index=request_index,
        )
        frequency = metadata_result.frequency
        position_value = metadata_result.position
        request_diagnostics.extend(metadata_result.diagnostics)

    variables: list[OutputVariableProjection] = []
    for group in grouped:
        variable_diagnostics: tuple[ResultDiagnostic, ...] = ()
        field_requests: tuple[FieldRequest, ...] = ()
        if semantic_supported:
            field_requests, variable_diagnostics = _project_variable(
                group,
                target=request.target,
                position_value=position_value,
                capabilities=capabilities,
                request_index=request_index,
            )
        projection = OutputVariableProjection(
            source_variable_indices=group.source_indices,
            source_variables=group.source_variables,
            canonical_variable=group.canonical_variable,
            field_requests=field_requests,
            diagnostics=variable_diagnostics,
        )
        variables.append(projection)
        request_diagnostics.extend(variable_diagnostics)

    variable_tuple = tuple(variables)
    diagnostics = tuple(request_diagnostics)
    executable_request = None
    if not diagnostics:
        executable_variables = tuple(
            variable
            for variable in variable_tuple
            if variable.canonical_variable is not None
        )
        field_requests = tuple(
            field_request
            for variable in executable_variables
            for field_request in variable.field_requests
        )
        executable_request = ExecutableOutputRequest(
            request_index=request_index,
            kind="field",
            target=request.target,
            frequency=frequency,
            variables=executable_variables,
            field_requests=field_requests,
        )
    return OutputRequestProjection(
        request_index=request_index,
        authoring_request=request,
        variables=variable_tuple,
        executable_request=executable_request,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class _VariableGroup:
    source_indices: tuple[int, ...]
    source_variables: tuple[str, ...]
    canonical_variable: ResultVariable | None


@dataclass(frozen=True, slots=True)
class _MetadataProjection:
    frequency: int
    position: object
    diagnostics: tuple[ResultDiagnostic, ...]


def _group_variables(
    variables: tuple[str, ...],
) -> tuple[_VariableGroup, ...]:
    known: dict[ResultVariable, list[tuple[int, str]]] = {
        variable: [] for variable in _VARIABLE_ORDER
    }
    unknown: dict[str, list[tuple[int, str]]] = {}
    for index, source_variable in enumerate(variables):
        canonical = _VARIABLE_LOOKUP.get(source_variable.casefold())
        if canonical is not None:
            known[canonical].append((index, source_variable))
            continue
        unknown.setdefault(source_variable.casefold(), []).append(
            (index, source_variable)
        )

    result: list[_VariableGroup] = []
    for canonical in _VARIABLE_ORDER:
        occurrences = known[canonical]
        if occurrences:
            result.append(
                _VariableGroup(
                    tuple(index for index, _value in occurrences),
                    tuple(value for _index, value in occurrences),
                    canonical,
                )
            )
    for occurrences in unknown.values():
        result.append(
            _VariableGroup(
                tuple(index for index, _value in occurrences),
                tuple(value for _index, value in occurrences),
                None,
            )
        )
    return tuple(result)


def _position_option_is_applicable(
    groups: tuple[_VariableGroup, ...],
    *,
    target: str,
) -> bool:
    return (
        target == "element"
        and bool(groups)
        and all(
            group.canonical_variable is ResultVariable.S
            for group in groups
        )
    )


def _project_variable(
    group: _VariableGroup,
    *,
    target: str,
    position_value: object,
    capabilities: ResultCapabilityCatalog,
    request_index: int,
) -> tuple[tuple[FieldRequest, ...], tuple[ResultDiagnostic, ...]]:
    canonical = group.canonical_variable
    if canonical is None:
        diagnostic = _diagnostic(
            "output.request.variable_unsupported",
            request_index,
            suffix=("variables", group.source_indices[0]),
            message=(
                "Output variable "
                f"{group.source_variables[0]!r} is not executable."
            ),
            remediation="Choose U, UR, RF, RM, or S for a supported target.",
            details={
                "source_indices": list(group.source_indices),
                "source_variables": list(group.source_variables),
            },
        )
        return (), (diagnostic,)

    expected_target = _PRIMARY_TARGETS[canonical]
    if target != expected_target:
        diagnostic = _diagnostic(
            "output.request.variable_unsupported",
            request_index,
            suffix=("variables", group.source_indices[0]),
            message=(
                f"{canonical.value} is not executable for target {target!r}."
            ),
            remediation=(
                f"Use target {expected_target!r} for {canonical.value}."
            ),
            details={
                "canonical_variable": canonical.value,
                "source_indices": list(group.source_indices),
                "target": target,
            },
        )
        return (), (diagnostic,)

    if canonical is ResultVariable.S:
        return _project_stress_variable(
            group,
            position_value=position_value,
            capabilities=capabilities,
            request_index=request_index,
        )

    field_id = ResultFieldId(canonical, FieldPosition.NODE)
    entry = capabilities.entry_for(field_id)
    if entry is None:
        diagnostic = _model_family_diagnostic(
            group,
            capabilities=capabilities,
            request_index=request_index,
        )
        return (), (diagnostic,)
    return (entry.default_request(),), ()


def _project_stress_variable(
    group: _VariableGroup,
    *,
    position_value: object,
    capabilities: ResultCapabilityCatalog,
    request_index: int,
) -> tuple[tuple[FieldRequest, ...], tuple[ResultDiagnostic, ...]]:
    family = capabilities.profile.family
    if (
        not capabilities.profile.stress_compatible
        or family not in _EXECUTABLE_STRESS_POSITIONS
    ):
        return (), (
            _model_family_diagnostic(
                group,
                capabilities=capabilities,
                request_index=request_index,
            ),
        )

    if position_value is _ABSENT:
        position = _DEFAULT_STRESS_POSITION[family]
    elif type(position_value) is not str:
        return (), (
            _position_diagnostic(
                group,
                position_value=position_value,
                family=family,
                request_index=request_index,
            ),
        )
    else:
        position = _POSITION_LOOKUP.get(position_value.casefold())
        if (
            position is None
            or position not in _EXECUTABLE_STRESS_POSITIONS[family]
        ):
            return (), (
                _position_diagnostic(
                    group,
                    position_value=position_value,
                    family=family,
                    request_index=request_index,
                ),
            )

    field_id = ResultFieldId(ResultVariable.S, position)
    entry = capabilities.entry_for(field_id)
    if entry is None:
        return (), (
            _model_family_diagnostic(
                group,
                capabilities=capabilities,
                request_index=request_index,
            ),
        )
    return (entry.default_request(),), ()


def _model_family_diagnostic(
    group: _VariableGroup,
    *,
    capabilities: ResultCapabilityCatalog,
    request_index: int,
) -> ResultDiagnostic:
    canonical = group.canonical_variable
    return _diagnostic(
        "output.request.model_family_unsupported",
        request_index,
        suffix=("variables", group.source_indices[0]),
        message=(
            f"{canonical.value if canonical is not None else 'variable'} "
            f"is unavailable for model family "
            f"{capabilities.profile.family.value!r}."
        ),
        remediation=(
            "Choose a field published by this model's result capability catalog."
        ),
        details={
            "canonical_variable": (
                None if canonical is None else canonical.value
            ),
            "model_family": capabilities.profile.family.value,
            "source_indices": list(group.source_indices),
        },
    )


def _position_diagnostic(
    group: _VariableGroup,
    *,
    position_value: object,
    family: ResultModelFamily,
    request_index: int,
) -> ResultDiagnostic:
    supported = tuple(
        position.value
        for position in FieldPosition
        if position in _EXECUTABLE_STRESS_POSITIONS[family]
    )
    return _diagnostic(
        "output.request.position_unsupported",
        request_index,
        suffix=("metadata", "position"),
        message=(
            f"Stress position {position_value!r} is not executable for "
            f"model family {family.value!r}."
        ),
        remediation=(
            "Use one of the family-specific canonical stress positions."
        ),
        details={
            "model_family": family.value,
            "position": position_value,
            "source_indices": list(group.source_indices),
            "supported_positions": list(supported),
        },
    )


def _effective_metadata(
    request: OutputRequest,
    *,
    request_index: int,
) -> tuple[dict[str, tuple[str, Any]], tuple[ResultDiagnostic, ...]]:
    request_layer, request_collisions = _metadata_layer(
        tuple(request.metadata.items()),
        layer="metadata",
        request_index=request_index,
        path_suffix=("metadata",),
    )
    evidence = request.source_evidence
    if evidence is None or evidence.source_kind != "abaqus":
        return request_layer, request_collisions

    parent, parent_collisions = _metadata_layer(
        evidence.parent_parameters,
        layer="parent_parameters",
        request_index=request_index,
        path_suffix=("source_evidence", "parent_parameters"),
    )
    child, child_collisions = _metadata_layer(
        evidence.child_parameters,
        layer="child_parameters",
        request_index=request_index,
        path_suffix=("source_evidence", "child_parameters"),
    )
    diagnostics = list(parent_collisions)
    diagnostics.extend(child_collisions)
    diagnostics.extend(request_collisions)
    diagnostics.extend(
        _unknown_flag_diagnostics(
            evidence,
            request_index=request_index,
        )
    )

    effective = dict(parent)
    effective.update(child)
    effective.update(request_layer)
    collision_keys = {
        str(diagnostic.details["canonical_key"])
        for diagnostic in diagnostics
        if diagnostic.details.get("reason") == "casefold_collision"
    }
    for key in collision_keys:
        effective.pop(key, None)
    return effective, tuple(diagnostics)


def _metadata_layer(
    items: tuple[tuple[str, Any], ...],
    *,
    layer: str,
    request_index: int,
    path_suffix: tuple[object, ...],
) -> tuple[
    dict[str, tuple[str, Any]],
    tuple[ResultDiagnostic, ...],
]:
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for key, value in items:
        grouped.setdefault(key.casefold(), []).append((key, value))

    result: dict[str, tuple[str, Any]] = {}
    diagnostics: list[ResultDiagnostic] = []
    for canonical_key, occurrences in grouped.items():
        if len(occurrences) == 1:
            result[canonical_key] = occurrences[0]
            continue
        source_keys = [key for key, _value in occurrences]
        diagnostics.append(
            _diagnostic(
                "output.request.metadata_unsupported",
                request_index,
                suffix=path_suffix,
                message=(
                    "Output metadata contains a case-insensitive key collision "
                    f"for {canonical_key!r}."
                ),
                remediation=(
                    "Keep exactly one spelling of each metadata option per layer."
                ),
                details={
                    "canonical_key": canonical_key,
                    "layer": layer,
                    "reason": "casefold_collision",
                    "source_keys": source_keys,
                },
            )
        )
    return result, tuple(diagnostics)


def _unknown_flag_diagnostics(
    evidence: OutputSourceEvidence,
    *,
    request_index: int,
) -> tuple[ResultDiagnostic, ...]:
    diagnostics: list[ResultDiagnostic] = []
    layers = (
        ("parent_flags", evidence.parent_flags),
        ("child_flags", evidence.child_flags),
    )
    for layer, flags in layers:
        unknown = tuple(
            flag
            for flag in flags
            if flag.casefold() not in _STRUCTURAL_ABAQUS_FLAGS
        )
        if not unknown:
            continue
        diagnostics.append(
            _diagnostic(
                "output.request.metadata_unsupported",
                request_index,
                suffix=("source_evidence", layer),
                message=(
                    f"Abaqus output {layer} contains unsupported bare flags."
                ),
                remediation=(
                    "Remove unsupported bare flags or keep the request "
                    "as preserved-only authoring."
                ),
                details={
                    "flags": list(unknown),
                    "layer": layer,
                    "reason": "unknown_bare_flag",
                },
            )
        )
    return tuple(diagnostics)


def _validate_metadata(
    effective: dict[str, tuple[str, Any]],
    *,
    allow_position: bool,
    request_index: int,
) -> _MetadataProjection:
    allowed = {"frequency"}
    if allow_position:
        allowed.add("position")
    unknown = tuple(
        source_key
        for canonical_key, (source_key, _value) in effective.items()
        if canonical_key not in allowed
    )
    diagnostics: list[ResultDiagnostic] = []
    if unknown:
        diagnostics.append(
            _diagnostic(
                "output.request.metadata_unsupported",
                request_index,
                suffix=("metadata",),
                message="The output request contains unsupported metadata.",
                remediation=(
                    "Remove unsupported options or retain the request "
                    "as preserved-only authoring."
                ),
                details={
                    "options": list(unknown),
                    "reason": "option_not_allowed",
                },
            )
        )

    frequency = 1
    frequency_entry = effective.get("frequency")
    if frequency_entry is not None:
        source_key, value = frequency_entry
        if not (
            (type(value) is int and value == 1)
            or (type(value) is str and value == "1")
        ):
            diagnostics.append(
                _diagnostic(
                    "output.request.frequency_unsupported",
                    request_index,
                    suffix=("metadata", source_key),
                    message=(
                        f"Output frequency {value!r} is not executable."
                    ),
                    remediation="Use strict integer 1 or exact string '1'.",
                    details={
                        "frequency": value,
                        "source_key": source_key,
                    },
                )
            )

    position = _ABSENT
    if allow_position and "position" in effective:
        position = effective["position"][1]
    return _MetadataProjection(
        frequency=frequency,
        position=position,
        diagnostics=tuple(diagnostics),
    )


def _diagnostic(
    code: str,
    request_index: int,
    *,
    suffix: tuple[object, ...],
    message: str,
    remediation: str,
    details: Mapping[str, Any],
) -> ResultDiagnostic:
    return ResultDiagnostic(
        code=code,
        severity="warning",
        message=message,
        path=("outputs", request_index, *suffix),
        remediation=remediation,
        details={"request_index": request_index, **details},
    )


def _strict_request_index(value: object) -> int:
    if type(value) is not int:
        raise TypeError("request_index must be an integer")
    if value < 0:
        raise ValueError("request_index must be non-negative")
    return value


def _strict_source_indices(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("source_variable_indices must be a tuple")
    if not value:
        raise ValueError("source_variable_indices must be a nonempty tuple")
    for index in value:
        if type(index) is not int:
            raise TypeError("source variable indices must be integers")
        if index < 0:
            raise ValueError("source variable indices must be non-negative")
    if value != tuple(sorted(set(value))):
        raise ValueError(
            "source variable indices must be unique and increasing"
        )
    return value


def _strict_string_tuple(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    if not value:
        raise ValueError(f"{label} must be a nonempty tuple")
    if any(type(item) is not str for item in value):
        raise TypeError(f"{label} must contain exact strings")
    return value


def _validate_field_requests(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("field_requests must be a tuple")
    if any(type(item) is not FieldRequest for item in value):
        raise TypeError("field_requests must contain FieldRequest values")
    field_ids = tuple(item.field_id for item in value)
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("field_requests must use unique field IDs")


def _validate_diagnostics(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("diagnostics must be a tuple")
    if any(type(item) is not ResultDiagnostic for item in value):
        raise TypeError("diagnostics must contain ResultDiagnostic values")


__all__ = [
    "ExecutableOutputRequest",
    "OutputRequestProjection",
    "OutputVariableProjection",
    "ResultCapabilityCatalog",
    "project_output_request",
]
