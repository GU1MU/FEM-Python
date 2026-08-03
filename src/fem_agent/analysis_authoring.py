"""A5 strict, named linear-static analysis authoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from fem.application import (
    ModelDefinitions,
    NamedRegion,
    ScopedDefinitionBatch,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)

from .authoring import AgentProposal, AuthoringContext, ModelPatch, ProposalKind
from .definition_authoring import definition_state_operations
from .naming import NameAllocator, NamePolicy


class AnalysisAuthoringError(ValueError):
    """An A5 analysis definition is incomplete, ambiguous, or unsupported."""


class _Snapshot(Protocol):
    session_id: str
    session_revision: int
    source_kind: str | None
    named_regions: Mapping[str, NamedRegion]
    materials: Sequence[object]
    sections: Sequence[object]
    assignments: Sequence[object]
    steps: Sequence[object]
    artifact: object | None
    runs: Sequence[object]
    model_current: bool


@dataclass(frozen=True, slots=True)
class ConfirmedDisplacement:
    name: str
    step_name: str
    target_scope: str
    entity_type: str
    first_component: int
    last_component: int
    value: float
    unit: str
    distribution: str
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not self.name.startswith("位移-"):
            raise AnalysisAuthoringError("displacement name must use type 位移")
        _nonblank(self.step_name, "displacement step")
        _nonblank(self.target_scope, "displacement target scope")
        if self.entity_type not in {"node_set", "edge", "surface"}:
            raise AnalysisAuthoringError(
                "displacement entity_type must be node_set, edge, or surface"
            )
        if (
            type(self.first_component) is not int
            or type(self.last_component) is not int
            or self.first_component < 1
            or self.last_component < self.first_component
        ):
            raise AnalysisAuthoringError(
                "displacement components must be one explicit contiguous range"
            )
        _finite(self.value, "displacement value")
        _nonblank(self.unit, "displacement unit")
        if self.distribution != "uniform":
            raise AnalysisAuthoringError(
                "A5 displacement distribution must be explicitly uniform"
            )
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "displacement engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        translations = tuple(
            component
            for component in range(self.first_component, self.last_component + 1)
            if component <= 3
        )
        rotations = tuple(
            component
            for component in range(self.first_component, self.last_component + 1)
            if component >= 4
        )
        return {
            "name": self.name,
            "step": self.step_name,
            "target_scope": self.target_scope,
            "entity_type": self.entity_type,
            "degrees_of_freedom": [
                self.first_component,
                self.last_component,
            ],
            "component_direction": (
                f"U{self.first_component}"
                if self.first_component == self.last_component
                else f"U{self.first_component}..U{self.last_component}"
            ),
            "translation_components": list(translations),
            "rotation_components": list(rotations),
            "translation_unit": self.unit if translations else None,
            "rotation_unit": "rad" if rotations else None,
            "signed_value": self.value,
            "unit": self.unit,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class ConfirmedLoad:
    name: str
    step_name: str
    target_scope: str
    entity_type: str
    load_type: str
    component: int | None
    vector: tuple[float, ...]
    magnitude: float | None
    direction: str
    unit: str
    distribution: str
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not self.name.startswith("载荷-"):
            raise AnalysisAuthoringError("load name must use type 载荷")
        _nonblank(self.step_name, "load step")
        _nonblank(self.target_scope, "load target scope")
        if self.load_type not in {
            "nodal",
            "edge_traction",
            "edge_pressure",
            "surface_traction",
            "surface_pressure",
        }:
            raise AnalysisAuthoringError("load_type is unsupported by A5")
        expected_entity = {
            "nodal": "node",
            "edge_traction": "edge",
            "edge_pressure": "edge",
            "surface_traction": "surface",
            "surface_pressure": "surface",
        }[self.load_type]
        if self.entity_type != expected_entity:
            raise AnalysisAuthoringError(
                "load entity_type does not match load_type"
            )
        values = tuple(self.vector)
        if any(not _is_finite(value) for value in values):
            raise AnalysisAuthoringError("load vector must contain finite numbers")
        object.__setattr__(self, "vector", tuple(float(value) for value in values))
        if self.magnitude is not None:
            _finite(self.magnitude, "load magnitude")
        if self.load_type == "nodal":
            if type(self.component) is not int or self.component < 1:
                raise AnalysisAuthoringError(
                    "nodal load requires one explicit component"
                )
            if values or self.magnitude is None:
                raise AnalysisAuthoringError(
                    "nodal load requires only one signed magnitude"
                )
            if self.distribution != "concentrated":
                raise AnalysisAuthoringError(
                    "nodal load distribution must be concentrated"
                )
        elif self.load_type.endswith("traction"):
            if self.component is not None or not values or self.magnitude is not None:
                raise AnalysisAuthoringError(
                    "traction requires an explicit signed vector only"
                )
            if self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    "traction distribution must be explicitly uniform"
                )
        else:
            if self.component is not None or values or self.magnitude is None:
                raise AnalysisAuthoringError(
                    "pressure requires one signed magnitude only"
                )
            if self.direction not in {"outward_normal", "inward_normal"}:
                raise AnalysisAuthoringError(
                    "pressure requires an explicit normal direction"
                )
            if (
                self.direction == "inward_normal"
                and float(self.magnitude) <= 0.0
            ) or (
                self.direction == "outward_normal"
                and float(self.magnitude) >= 0.0
            ):
                raise AnalysisAuthoringError(
                    "pressure sign must be positive inward and negative outward"
                )
            if self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    "pressure distribution must be explicitly uniform"
                )
        _nonblank(self.direction, "load direction")
        _nonblank(self.unit, "load unit")
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "load engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        nodal_family = (
            None
            if self.load_type != "nodal"
            else "force" if int(self.component or 0) <= 3 else "moment"
        )
        return {
            "name": self.name,
            "step": self.step_name,
            "target_scope": self.target_scope,
            "entity_type": self.entity_type,
            "load_type": self.load_type,
            "component": self.component,
            "nodal_family": nodal_family,
            "direction": self.direction,
            "signed_components": list(self.vector),
            "signed_magnitude": self.magnitude,
            "unit": self.unit,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class ConfirmedResultRequest:
    name: str
    step_name: str
    kind: str
    target: str
    variables: tuple[str, ...]
    units: tuple[str, ...]
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not self.name.startswith("结果请求-"):
            raise AnalysisAuthoringError(
                "result request name must use type 结果请求"
            )
        _nonblank(self.step_name, "result request step")
        if self.kind != "field":
            raise AnalysisAuthoringError("A5 supports only field output requests")
        supported = {
            "node": frozenset({"U", "RF"}),
            "element": frozenset({"S"}),
        }
        if self.target not in supported:
            raise AnalysisAuthoringError("result request target is unsupported")
        variables = tuple(self.variables)
        if (
            not variables
            or any(type(item) is not str for item in variables)
            or not set(variables).issubset(supported[self.target])
        ):
            raise AnalysisAuthoringError(
                "result variables are missing or unsupported for their target"
            )
        if len(self.units) != len(variables) or any(
            type(item) is not str or not item.strip() for item in self.units
        ):
            raise AnalysisAuthoringError(
                "result variables require explicit aligned units"
            )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "units", tuple(self.units))
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "result request engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "step": self.step_name,
            "kind": self.kind,
            "target": self.target,
            "variables": list(self.variables),
            "units": list(self.units),
        }


@dataclass(frozen=True, slots=True)
class LinearStaticAnalysis:
    step_name: str
    dimension: int
    procedure: str
    nlgeom: bool
    displacements: tuple[ConfirmedDisplacement, ...]
    loads: tuple[ConfirmedLoad, ...]
    results: tuple[ConfirmedResultRequest, ...]
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.step_name)
        if not self.step_name.startswith("分析步-"):
            raise AnalysisAuthoringError("step name must use type 分析步")
        if type(self.dimension) is not int or self.dimension not in {2, 3}:
            raise AnalysisAuthoringError(
                "analysis dimension must be explicitly 2 or 3"
            )
        if self.procedure != "static" or self.nlgeom is not False:
            raise AnalysisAuthoringError(
                "A5 requires one linear static step with NLGEOM disabled"
            )
        displacements = tuple(self.displacements)
        loads = tuple(self.loads)
        results = tuple(self.results)
        if not displacements or not loads or not results:
            raise AnalysisAuthoringError(
                "analysis requires confirmed constraints, loads, and results"
            )
        for item in (*displacements, *loads, *results):
            if item.step_name != self.step_name:
                raise AnalysisAuthoringError(
                    "every analysis object must belong to the single static step"
                )
        if any(
            item.last_component > self.dimension for item in displacements
        ):
            raise AnalysisAuthoringError(
                "displacement DOF exceeds the confirmed model dimension"
            )
        for item in loads:
            if item.load_type == "nodal" and int(item.component or 0) > self.dimension:
                raise AnalysisAuthoringError(
                    "nodal load component exceeds model dimension"
                )
            if item.load_type.startswith("edge_") and self.dimension != 2:
                raise AnalysisAuthoringError(
                    "edge loads require a confirmed two-dimensional model"
                )
            if item.load_type.startswith("surface_") and self.dimension != 3:
                raise AnalysisAuthoringError(
                    "surface loads require a confirmed three-dimensional model"
                )
            if item.vector and len(item.vector) != self.dimension:
                raise AnalysisAuthoringError(
                    "load vector dimension does not match the model"
                )
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "analysis engineering fields are not confirmed"
            )
        object.__setattr__(self, "displacements", displacements)
        object.__setattr__(self, "loads", loads)
        object.__setattr__(self, "results", results)

    def to_step(self) -> AnalysisStep:
        boundaries = tuple(
            DisplacementConstraint(
                item.target_scope,
                item.first_component,
                item.last_component,
                item.value,
                item.entity_type,
                item.name,
            )
            for item in self.displacements
        )
        cloads: list[NodalLoad] = []
        edge_loads: list[EdgeLoad] = []
        surface_loads: list[SurfaceLoad] = []
        for item in self.loads:
            if item.load_type == "nodal":
                cloads.append(
                    NodalLoad(
                        item.target_scope,
                        int(item.component),
                        float(item.magnitude),
                        item.name,
                    )
                )
            elif item.load_type.startswith("edge_"):
                edge_loads.append(
                    EdgeLoad(
                        item.target_scope,
                        item.vector,
                        item.magnitude,
                        item.load_type.removeprefix("edge_"),
                        item.name,
                    )
                )
            else:
                surface_loads.append(
                    SurfaceLoad(
                        item.target_scope,
                        item.vector,
                        item.magnitude,
                        item.load_type.removeprefix("surface_"),
                        item.name,
                    )
                )
        outputs = tuple(
            OutputRequest(
                item.kind,
                item.target,
                item.variables,
                name=item.name,
            )
            for item in self.results
        )
        return AnalysisStep(
            self.step_name,
            procedure="static",
            boundaries=boundaries,
            cloads=tuple(cloads),
            edge_loads=tuple(edge_loads),
            surface_loads=tuple(surface_loads),
            outputs=outputs,
            metadata={"nlgeom": False},
        )


def create_analysis_definition_change(
    *,
    patch_id: str,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    snapshot: _Snapshot,
    draft_revision: int,
    analysis: LinearStaticAnalysis,
) -> ModelPatch | AgentProposal:
    """Create one complete A5 definition patch or destructive proposal."""

    if type(analysis) is not LinearStaticAnalysis:
        raise TypeError("analysis must be exactly LinearStaticAnalysis")
    _require_live_context(context, snapshot)
    if context.unit_context is None:
        raise AnalysisAuthoringError("analysis requires a confirmed unit context")
    _require_unit_semantics(context, analysis)
    if snapshot.artifact is None or not snapshot.model_current:
        raise AnalysisAuthoringError(
            "analysis requires a current accepted generated model"
        )
    _require_names_are_next(snapshot, analysis)
    _require_scope_kinds(snapshot, analysis)

    step = analysis.to_step()
    definitions = ModelDefinitions(
        tuple(snapshot.materials),
        tuple(snapshot.sections),
        tuple(snapshot.assignments),
        (step,),
    )
    operations = definition_state_operations(
        tuple(snapshot.named_regions.values()),
        definitions,
    )
    details = {
        "step": {
            "name": step.name,
            "procedure": "linear_static",
            "nlgeom": False,
        },
        "constraints": [item.summary() for item in analysis.displacements],
        "loads": [item.summary() for item in analysis.loads],
        "result_requests": [item.summary() for item in analysis.results],
        "unit_context": context.unit_context.to_dict(),
    }
    existing_steps = tuple(snapshot.steps)
    result_invalidating = _has_accepted_result(snapshot)
    destructive = bool(existing_steps) or result_invalidating
    summary = {
        "title": "Agent 已创建完整线性静力分析定义",
        "summary": (
            f"{step.name}：{len(step.boundaries)} 个位移、"
            f"{len(step.cloads) + len(step.edge_loads) + len(step.surface_loads)} "
            f"个载荷、{len(step.outputs)} 个结果请求"
        ),
        "analysis": details,
        "objects": [
            step.name,
            *(item.name for item in analysis.displacements),
            *(item.name for item in analysis.loads),
            *(item.name for item in analysis.results),
        ],
        "undo_label": "撤销本次 Agent 修改",
    }
    common = {
        "agent_session_id": agent_session_id,
        "turn_id": turn_id,
        "source_tool_call_ids": tuple(source_tool_call_ids),
        "target_document_id": context.binding.document_id,
        "target_session_id": context.binding.session_id,
        "base_session_revision": context.binding.session_revision,
        "draft_revision": draft_revision,
        "operations": operations,
        "preconditions": {
            "authoring_phase": "A5",
            "source_kind": "native",
            "model_current": True,
            "requirements_confirmed": True,
            "single_linear_static_step": True,
            "nlgeom": False,
            "no_destructive_name_overwrite": not destructive,
        },
        "expected_changes": {
            "analysis_steps": 1,
            "boundaries": len(step.boundaries),
            "loads": (
                len(step.cloads)
                + len(step.edge_loads)
                + len(step.surface_loads)
            ),
            "result_requests": len(step.outputs),
        },
        "invalidation_impact": {
            "model": True,
            "validation": True,
            "results": result_invalidating,
        },
        "display_summary": summary,
    }
    if destructive:
        return AgentProposal.create(
            proposal_id=proposal_id,
            proposal_kind=ProposalKind.DESTRUCTIVE_EDIT,
            **{
                **common,
                "display_summary": {
                    **summary,
                    "title": "分析定义修改需要确认",
                    "impact": (
                        "将替换已有分析定义并使已有结果失效"
                        if result_invalidating
                        else "将替换已有分析定义"
                    ),
                    "confirm_label": "确认修改",
                },
            },
        )
    return ModelPatch.create(patch_id=patch_id, **common)


def require_non_destructive_a5_batch(
    snapshot: _Snapshot,
    batch: ScopedDefinitionBatch,
) -> None:
    """Reject any automatic A5 change beyond one fresh static step."""

    if tuple(snapshot.steps):
        raise AnalysisAuthoringError(
            "automatic A5 patch cannot overwrite existing analysis steps"
        )
    if tuple(batch.regions) != tuple(snapshot.named_regions.values()):
        raise AnalysisAuthoringError(
            "automatic A5 patch cannot edit named regions"
        )
    for label, before, after in (
        ("materials", snapshot.materials, batch.materials),
        ("sections", snapshot.sections, batch.sections),
        ("assignments", snapshot.assignments, batch.assignments),
    ):
        if tuple(after) != tuple(before):
            raise AnalysisAuthoringError(
                f"automatic A5 patch cannot edit {label}"
            )
    if len(batch.steps) != 1:
        raise AnalysisAuthoringError(
            "A5 requires exactly one analysis step"
        )
    step = batch.steps[0]
    if (
        type(step) is not AnalysisStep
        or type(step.procedure) is not str
        or step.procedure != "static"
        or "nlgeom" not in step.metadata
        or step.metadata["nlgeom"] is not False
    ):
        raise AnalysisAuthoringError(
            "A5 requires one linear static step with NLGEOM disabled"
        )


def _require_live_context(
    context: AuthoringContext,
    snapshot: _Snapshot,
) -> None:
    if (
        snapshot.source_kind != "native"
        or context.binding.source_kind != "native"
        or context.binding.session_id != snapshot.session_id
        or context.binding.document_id != f"document:{snapshot.session_id}"
        or context.binding.session_revision != snapshot.session_revision
    ):
        raise AnalysisAuthoringError("authoring context is stale")


def _require_names_are_next(
    snapshot: _Snapshot,
    analysis: LinearStaticAnalysis,
) -> None:
    existing_steps = tuple(snapshot.steps)
    allocator = NameAllocator(
        {
            "steps": (str(item.name) for item in existing_steps),
            "boundaries": (
                str(item.name)
                for step in existing_steps
                for item in tuple(step.boundaries)
                if item.name is not None
            ),
            "loads": (
                str(item.name)
                for step in existing_steps
                for collection in (
                    step.cloads,
                    step.edge_loads,
                    step.surface_loads,
                    step.line_loads,
                    step.body_loads,
                    step.gravity_loads,
                )
                for item in tuple(collection)
                if item.name is not None
            ),
            "outputs": (
                str(item.name)
                for step in existing_steps
                for item in tuple(step.outputs)
                if item.name is not None
            ),
        }
    )
    _require_allocated(allocator, "steps", "分析步", analysis.step_name)
    for item in analysis.displacements:
        _require_allocated(allocator, "boundaries", "位移", item.name)
    for item in analysis.loads:
        _require_allocated(allocator, "loads", "载荷", item.name)
    for item in analysis.results:
        _require_allocated(allocator, "outputs", "结果请求", item.name)


def _require_allocated(
    allocator: NameAllocator,
    namespace: str,
    object_type: str,
    name: str,
) -> None:
    try:
        allocator.require_next(namespace, object_type, name)
    except ValueError as error:
        raise AnalysisAuthoringError(str(error)) from error


def _require_scope_kinds(
    snapshot: _Snapshot,
    analysis: LinearStaticAnalysis,
) -> None:
    for item in (*analysis.displacements, *analysis.loads):
        try:
            region = snapshot.named_regions[item.target_scope]
        except KeyError as error:
            raise AnalysisAuthoringError(
                f"target scope {item.target_scope!r} does not exist"
            ) from error
        actual = {getattr(reference, "kind", None) for reference in region.references}
        expected = {
            "node_set": "node",
            "node": "node",
            "edge": "edge",
            "surface": "face",
        }[item.entity_type]
        if actual != {expected}:
            raise AnalysisAuthoringError(
                f"target scope {item.target_scope!r} is not a proven "
                f"{item.entity_type} scope"
            )


def _require_unit_semantics(
    context: AuthoringContext,
    analysis: LinearStaticAnalysis,
) -> None:
    units = context.unit_context
    if units is None:
        raise AnalysisAuthoringError("analysis requires a confirmed unit context")
    for item in analysis.displacements:
        if item.unit != units.length:
            raise AnalysisAuthoringError(
                f"{item.name} unit must exactly match project length "
                f"unit {units.length!r}; conversion is unavailable"
            )
    edge_unit = f"{units.force}/{units.length}"
    for item in analysis.loads:
        expected = {
            "nodal": units.force,
            "edge_traction": edge_unit,
            "edge_pressure": edge_unit,
            "surface_traction": units.stress,
            "surface_pressure": units.stress,
        }[item.load_type]
        if item.unit != expected:
            raise AnalysisAuthoringError(
                f"{item.name} unit must exactly match project analysis "
                f"unit {expected!r}; conversion is unavailable"
            )
    expected_result_units = {
        "U": units.length,
        "RF": units.force,
        "S": units.stress,
    }
    for item in analysis.results:
        expected = tuple(
            expected_result_units[variable]
            for variable in item.variables
        )
        if item.units != expected:
            raise AnalysisAuthoringError(
                f"{item.name} units must be {expected!r} from the "
                "confirmed project unit context"
            )


def _has_accepted_result(snapshot: _Snapshot) -> bool:
    return any(bool(getattr(run, "has_result", False)) for run in snapshot.runs)


def _nonblank(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise AnalysisAuthoringError(
            f"{label} must be a nonblank string without outer whitespace"
        )
    return value


def _is_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and type(value) in {int, float}
        and math.isfinite(float(value))
    )


def _finite(value: object, label: str) -> float:
    if not _is_finite(value):
        raise AnalysisAuthoringError(f"{label} must be a finite number")
    return float(value)


__all__ = [
    "AnalysisAuthoringError",
    "ConfirmedDisplacement",
    "ConfirmedLoad",
    "ConfirmedResultRequest",
    "LinearStaticAnalysis",
    "create_analysis_definition_change",
    "require_non_destructive_a5_batch",
]
