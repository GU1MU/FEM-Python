"""Headless, revisioned ownership of one FEM project lifecycle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fem.geometry import (
    BooleanBodyContext,
    BooleanGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    logical_ref_sort_key,
    MovedGeometry,
    MultiBodyGeometry,
    PlanarBooleanContext,
    PartBooleanContext,
    geometry_dimension,
    historical_recipe_ids,
    recipe_characteristic_size,
    retired_recipe_ids,
    RevolvedGeometry,
    RotatedGeometry,
    supports_structured_hexahedron,
    namespace_part_logical_id,
    part_id_from_logical_id,
    strip_part_logical_id,
)
from fem.geometry.recipe_topology import (
    can_preserve_logical_references,
    logical_reference_transition_map,
    surviving_logical_reference_ids,
)
from fem.mesh.settings import MeshSettings

from .results import (
    FieldMaterializationKey,
    ResultMaterializationPatch,
    ResultProvider,
    ResultSourceKey,
    SolveResultBundle,
    field_materialization_sort_key,
    validate_solve_result_model_identity,
)
from .changes import (
    ArtifactKind,
    ChangeKind,
    SessionDelta,
    TransitionEffect,
)
from .commands import (
    UNSET,
    DefinitionEditBatch,
    DeleteIntent,
    NamedRegionEditBatch,
    RenameIntent,
    Unset,
)
from .definitions import (
    FeatureRecord,
    MeshEntityRef,
    ModelDefinitions,
    NamedRegion,
    RegionAssignment,
    SectionDefinition,
    compile_model_definitions,
    definitions_from_model,
    normalize_model_definitions,
)
from .native_part import (
    NativePart,
    PartBooleanProvenance,
    next_part_boolean_feature_id,
    next_part_id,
    normalize_part_boolean_feature_id,
    normalize_part_id,
    part_boolean_feature_id_sort_key,
    part_id_sort_key,
    validate_native_parts,
)
from .diagnostics import PreflightReport, internal_error_report
from .feature_history import derive_feature_history
from .project_validation import (
    analysis_step_has_native_region_target,
    analysis_steps_have_native_region_targets,
    validate_native_project_inputs,
)
from .native_mesh_contract import require_complete_native_mesh_contract
from .native_regions import describe_native_regions
from .native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    can_materialize_native_scopes,
    materialize_native_scopes,
)
from .revisions import (
    ImportTaskSnapshot,
    MeshTaskSnapshot,
    ModelArtifact,
    ResultMaterializationTaskSnapshot,
    ResultTaskSnapshot,
    SolveTaskSnapshot,
    TaskToken,
    TokenStatus,
    ValidationTaskSnapshot,
    new_identity,
)
from .runs import (
    AnalysisRun,
    ResultProvenance,
    ResultRecord,
    RunStatus,
    advance_result_record,
    detached_result_record,
    result_record_provider,
    utc_now,
)
from .validation import ValidationRecord, ValidationStamp


_ALL_INVALIDATIONS = frozenset(
    {
        ArtifactKind.MODEL,
        ArtifactKind.VALIDATIONS,
        ArtifactKind.RUNS,
        ArtifactKind.RESULTS,
        ArtifactKind.DISPLAYED_RESULT,
        ArtifactKind.TASKS,
    }
)
_MODEL_INVALIDATIONS = frozenset(
    {
        ArtifactKind.MODEL,
        ArtifactKind.VALIDATIONS,
        ArtifactKind.RUNS,
        ArtifactKind.RESULTS,
        ArtifactKind.DISPLAYED_RESULT,
    }
)
_COMPUTATION_INVALIDATIONS = frozenset(
    {
        ArtifactKind.VALIDATIONS,
        ArtifactKind.RUNS,
        ArtifactKind.RESULTS,
        ArtifactKind.DISPLAYED_RESULT,
    }
)


class RevisionConflictError(RuntimeError):
    """A compare-and-swap command was based on an older session revision."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"expected session revision {expected}, current revision is {actual}"
        )
        self.expected = int(expected)
        self.actual = int(actual)


class PartRevisionConflictError(RuntimeError):
    """A compare-and-swap Part edit was based on an older Part revision."""

    def __init__(
        self,
        part_id: str,
        expected: int,
        actual: int,
    ) -> None:
        normalized = normalize_part_id(part_id)
        super().__init__(
            f"expected Part {normalized} revision {expected}, "
            f"current revision is {actual}"
        )
        self.part_id = normalized
        self.expected = int(expected)
        self.actual = int(actual)


class SessionStateError(RuntimeError):
    """The requested operation is not valid for the current session state."""


def _mapping_values(value: Mapping[Any, Any] | Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    return tuple(value)


@dataclass(frozen=True, slots=True)
class _IssuedSolvePayload:
    """Exact detached model objects issued for one solve task."""

    model: Any
    step: Any


@dataclass(frozen=True, slots=True)
class _IssuedMaterializationPayload:
    """Expected patch identity for one generation-bound recovery task."""

    source: ResultSourceKey
    generation: int
    expected_patch_keys: tuple[FieldMaterializationKey, ...]


@dataclass(frozen=True, slots=True)
class BooleanReferenceUndoRecord:
    """Exact pre/post reference state for one reversible strict Boolean."""

    feature_id: str
    target_body_id: str  # Frozen schema field; stores a target Face for PB*.
    before_geometry: Any
    after_geometry: Any
    before_named_regions: tuple[NamedRegion, ...]
    after_named_regions: tuple[NamedRegion, ...]
    before_mesh_settings: Any | None
    after_mesh_settings: Any | None
    before_materials: tuple[Any, ...]
    after_materials: tuple[Any, ...]
    before_sections: tuple[SectionDefinition, ...]
    after_sections: tuple[SectionDefinition, ...]
    before_assignments: tuple[RegionAssignment, ...]
    after_assignments: tuple[RegionAssignment, ...]
    before_steps: tuple[Any, ...]
    after_steps: tuple[Any, ...]

    def __post_init__(self) -> None:
        transition = _strict_boolean_transition(
            self.before_geometry,
            self.after_geometry,
        )
        if (
            transition is None
            or transition[0] != "forward"
            or transition[1].feature_id != self.feature_id
            or _boolean_context_target(transition[1]) != self.target_body_id
        ):
            raise ValueError(
                "Boolean undo record geometries must describe its exact "
                "forward feature transition"
            )


@dataclass(frozen=True, slots=True)
class PartBooleanUndoRecord:
    """Exact source and definition state for one reversible Part Boolean."""

    feature_id: str
    result_part_id: str
    source_parts: tuple[NativePart, NativePart]
    result_part: NativePart
    before_named_regions: tuple[NamedRegion, ...]
    after_named_regions: tuple[NamedRegion, ...]
    before_assignments: tuple[RegionAssignment, ...]
    after_assignments: tuple[RegionAssignment, ...]
    before_steps: tuple[Any, ...]
    after_steps: tuple[Any, ...]

    def __post_init__(self) -> None:
        feature_id = normalize_part_boolean_feature_id(self.feature_id)
        result_id = normalize_part_id(self.result_part_id)
        sources = deepcopy(tuple(self.source_parts))
        if len(sources) != 2 or any(
            type(part) is not NativePart for part in sources
        ):
            raise TypeError(
                "Part Boolean undo source_parts must contain two NativeParts"
            )
        if sources[0].id == sources[1].id:
            raise ValueError("Part Boolean undo sources must differ")
        result = deepcopy(self.result_part)
        if type(result) is not NativePart or result.id != result_id:
            raise ValueError(
                "Part Boolean undo result_part must match result_part_id"
            )
        provenance = result.provenance
        if (
            provenance is None
            or provenance.feature_id != feature_id
            or provenance.source_part_ids
            != (sources[0].id, sources[1].id)
        ):
            raise ValueError(
                "Part Boolean undo result provenance does not match sources"
            )
        recipe = result.geometry_recipe
        context = (
            recipe.part_context
            if isinstance(recipe, BooleanGeometry)
            else None
        )
        if (
            context is None
            or context.feature_id != feature_id
            or context.result_part_id != result_id
            or context.target_part_id != sources[0].id
            or context.tool_part_id != sources[1].id
            or recipe.object_geometry != sources[0].geometry_recipe
            or recipe.tool_geometry != sources[1].geometry_recipe
            or recipe.operation != provenance.operation
        ):
            raise ValueError(
                "Part Boolean undo result recipe is not bound to its exact "
                "source Part states"
            )
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "result_part_id", result_id)
        object.__setattr__(self, "source_parts", sources)
        object.__setattr__(self, "result_part", result)
        for field_name in (
            "before_named_regions",
            "after_named_regions",
            "before_assignments",
            "after_assignments",
            "before_steps",
            "after_steps",
        ):
            object.__setattr__(
                self,
                field_name,
                deepcopy(tuple(getattr(self, field_name))),
            )


@dataclass(frozen=True, slots=True)
class _PartExtrusionUndoRecord:
    primary_part_id: str
    before_part: NativePart
    after_parts: tuple[NativePart, ...]
    before_named_regions: tuple[NamedRegion, ...]
    after_named_regions: tuple[NamedRegion, ...]
    before_assignments: tuple[RegionAssignment, ...]
    after_assignments: tuple[RegionAssignment, ...]
    before_steps: tuple[Any, ...]
    after_steps: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ProjectSnapshot:
    """Detached, fully decoded project inputs suitable for atomic installation."""

    source_kind: str = "native"
    source_path: Path | None = None
    parts: tuple[NativePart, ...] = ()
    geometry_recipe: Any | None = None
    mesh_settings: Any | None = None
    feature_history: tuple[FeatureRecord, ...] = ()
    named_regions: tuple[NamedRegion, ...] = ()
    material_definitions: tuple[Any, ...] = ()
    section_definitions: tuple[SectionDefinition, ...] = ()
    region_assignments: tuple[RegionAssignment, ...] = ()
    analysis_definitions: tuple[Any, ...] = ()
    model: Any | None = None
    boolean_reference_undo_records: tuple[
        BooleanReferenceUndoRecord,
        ...,
    ] = ()
    model_name: str = "Model-1"
    part_boolean_undo_records: tuple[PartBooleanUndoRecord, ...] = ()
    retired_part_ids: tuple[str, ...] = ()
    retired_part_boolean_feature_ids: tuple[str, ...] = ()
    active_part_id: str | None = None

    def __post_init__(self) -> None:
        source_kind = _canonical_source_kind(self.source_kind)
        source_path = (
            None if self.source_path is None else Path(self.source_path)
        )
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "parts", deepcopy(tuple(self.parts)))
        object.__setattr__(self, "geometry_recipe", deepcopy(self.geometry_recipe))
        object.__setattr__(self, "mesh_settings", deepcopy(self.mesh_settings))
        object.__setattr__(
            self, "feature_history", deepcopy(tuple(self.feature_history))
        )
        object.__setattr__(
            self,
            "named_regions",
            deepcopy(_mapping_values(self.named_regions)),
        )
        object.__setattr__(
            self,
            "material_definitions",
            deepcopy(_mapping_values(self.material_definitions)),
        )
        object.__setattr__(
            self,
            "section_definitions",
            deepcopy(tuple(self.section_definitions)),
        )
        object.__setattr__(
            self,
            "region_assignments",
            deepcopy(tuple(self.region_assignments)),
        )
        object.__setattr__(
            self,
            "analysis_definitions",
            deepcopy(tuple(self.analysis_definitions)),
        )
        object.__setattr__(self, "model", deepcopy(self.model))
        records = tuple(self.boolean_reference_undo_records)
        if any(type(record) is not BooleanReferenceUndoRecord for record in records):
            raise TypeError(
                "boolean_reference_undo_records must contain only "
                "BooleanReferenceUndoRecord values"
            )
        feature_ids = tuple(record.feature_id for record in records)
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("Boolean undo record feature IDs must be unique")
        canonical = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.feature_id[:2],
                    int(record.feature_id[2:]),
                ),
            )
        )
        if records != canonical:
            raise ValueError("Boolean undo records must use canonical feature order")
        object.__setattr__(
            self,
            "boolean_reference_undo_records",
            deepcopy(records),
        )
        part_records = tuple(self.part_boolean_undo_records)
        if any(type(record) is not PartBooleanUndoRecord for record in part_records):
            raise TypeError(
                "part_boolean_undo_records must contain PartBooleanUndoRecord"
            )
        part_record_ids = tuple(record.feature_id for record in part_records)
        if len(part_record_ids) != len(set(part_record_ids)):
            raise ValueError("Part Boolean undo feature IDs must be unique")
        if part_records != tuple(
            sorted(
                part_records,
                key=lambda record: part_boolean_feature_id_sort_key(
                    record.feature_id
                ),
            )
        ):
            raise ValueError(
                "Part Boolean undo records must use canonical feature order"
            )
        object.__setattr__(
            self,
            "part_boolean_undo_records",
            deepcopy(part_records),
        )
        retired_parts = tuple(
            sorted(
                {
                    normalize_part_id(value, "retired Part ID")
                    for value in self.retired_part_ids
                },
                key=part_id_sort_key,
            )
        )
        retired_features = tuple(
            sorted(
                {
                    normalize_part_boolean_feature_id(
                        value,
                        "retired Part Boolean feature ID",
                    )
                    for value in self.retired_part_boolean_feature_ids
                },
                key=part_boolean_feature_id_sort_key,
            )
        )
        active_part_ids = {part.id for part in self.parts}
        if active_part_ids & set(retired_parts):
            raise ValueError("active and retired Part IDs must be disjoint")
        active_feature_ids = {
            part.provenance.feature_id
            for part in self.parts
            if part.provenance is not None
        }
        if active_feature_ids & set(retired_features):
            raise ValueError(
                "active and retired Part Boolean feature IDs must be disjoint"
            )
        active_part_id = (
            None
            if self.active_part_id is None
            else normalize_part_id(
                self.active_part_id,
                "ProjectSnapshot.active_part_id",
            )
        )
        if active_part_id is not None:
            active_part = next(
                (part for part in self.parts if part.id == active_part_id),
                None,
            )
            if active_part is None:
                raise ValueError(
                    "ProjectSnapshot.active_part_id must identify an active Part"
                )
            if active_part.suppressed:
                raise ValueError(
                    "ProjectSnapshot.active_part_id cannot identify a suppressed Part"
                )
        object.__setattr__(self, "active_part_id", active_part_id)
        object.__setattr__(self, "retired_part_ids", retired_parts)
        object.__setattr__(
            self,
            "retired_part_boolean_feature_ids",
            retired_features,
        )
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("ProjectSnapshot.model_name must not be empty")
        object.__setattr__(self, "model_name", model_name)

    @property
    def project_path(self) -> Path | None:
        return self.source_path if self.source_kind == "native" else None

    @property
    def path(self) -> Path | None:
        return self.source_path

    @property
    def materials(self) -> tuple[Any, ...]:
        return self.material_definitions

    @property
    def sections(self) -> tuple[SectionDefinition, ...]:
        return self.section_definitions

    @property
    def assignments(self) -> tuple[RegionAssignment, ...]:
        return self.region_assignments

    @property
    def steps(self) -> tuple[Any, ...]:
        return self.analysis_definitions


@dataclass(frozen=True, slots=True, init=False)
class ProjectSaveSnapshot:
    """Immutable inputs and token for a transactional project save."""

    token: TaskToken
    project_revision: int
    _snapshot: ProjectSnapshot

    def __init__(
        self,
        token: TaskToken,
        project_revision: int,
        snapshot: ProjectSnapshot,
    ) -> None:
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "project_revision", int(project_revision))
        object.__setattr__(self, "_snapshot", deepcopy(snapshot))

    @property
    def snapshot(self) -> ProjectSnapshot:
        """Return a detached copy so callers cannot mutate save-at-revision data."""

        return deepcopy(self._snapshot)


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Read-only, detached projection of all authoritative session state."""

    is_open: bool
    session_id: str
    session_revision: int
    project_revision: int
    mesh_input_revision: int
    model_revision: int
    saved_project_revision: int
    source_kind: str | None
    source_path: Path | None
    project_path: Path | None
    model_name: str | None
    geometry_recipe: Any | None
    mesh_settings: Any | None
    parts: tuple[NativePart, ...]
    active_part_id: str | None
    part_revisions: Mapping[str, int]
    feature_history: tuple[FeatureRecord, ...]
    named_regions: Mapping[str, NamedRegion]
    materials: tuple[Any, ...]
    sections: tuple[SectionDefinition, ...]
    assignments: tuple[RegionAssignment, ...]
    steps: tuple[Any, ...]
    artifact: ModelArtifact | None
    validations: Mapping[str, ValidationRecord]
    runs: tuple[AnalysisRun, ...]
    selected_run_id: str | None
    displayed_result_run_id: str | None
    displayed_result: ResultRecord | None
    dirty: bool
    can_save: bool

    @property
    def path(self) -> Path | None:
        if self.source_kind == "imported":
            return self.source_path
        return self.project_path

    @property
    def model(self) -> Any | None:
        return None if self.artifact is None else self.artifact.model

    @property
    def has_model(self) -> bool:
        return self.artifact is not None

    @property
    def has_result(self) -> bool:
        return self.displayed_result is not None

    @property
    def has_native_geometry(self) -> bool:
        return self.source_kind == "native" and any(
            part.geometry_recipe is not None for part in self.parts
        )

    @property
    def active_part(self) -> NativePart | None:
        if self.active_part_id is None:
            return None
        return next(
            (
                part
                for part in self.parts
                if part.id == self.active_part_id
            ),
            None,
        )

    def part(self, part_id: str) -> NativePart:
        normalized = normalize_part_id(part_id)
        for part in self.parts:
            if part.id == normalized:
                return deepcopy(part)
        raise KeyError(normalized)

    def part_revision(self, part_id: str) -> int:
        normalized = normalize_part_id(part_id)
        try:
            return int(self.part_revisions[normalized])
        except KeyError:
            raise KeyError(normalized) from None

    @property
    def model_current(self) -> bool:
        return (
            self.artifact is not None
            and self.artifact.session_id == self.session_id
            and self.artifact.model_revision == self.model_revision
        )

    @property
    def mesh_current(self) -> bool:
        if not self.model_current:
            return False
        if self.source_kind == "imported":
            return self.artifact.source_kind == "imported"
        return (
            self.artifact.source_kind == "native"
            and self.artifact.mesh_input_revision == self.mesh_input_revision
        )

    @property
    def mesh_is_current(self) -> bool:
        return self.mesh_current

    @property
    def native_mesh_current(self) -> bool:
        return self.source_kind == "native" and self.mesh_current

    @property
    def can_reload(self) -> bool:
        return self.source_kind == "imported" and self.source_path is not None

    @property
    def running_run_id(self) -> str | None:
        for run in reversed(self.runs):
            if run.status is RunStatus.RUNNING:
                return run.run_id
        return None

    @property
    def active_job_name(self) -> str | None:
        running_id = self.running_run_id
        if running_id is None:
            return None
        for run in self.runs:
            if run.run_id == running_id:
                return run.name
        return None

    @property
    def step_name(self) -> str | None:
        return self.default_step_name()

    @property
    def needs_model_check(self) -> bool:
        step_name = self.default_step_name()
        return self.model_current and (
            step_name is None or not self.validation_current(step_name)
        )

    def runnable_step_names(self) -> tuple[str, ...]:
        names = tuple(
            str(step.name)
            for step in self.steps
            if str(step.name).casefold() != "initial"
        )
        return names or tuple(str(step.name) for step in self.steps)

    def default_step_name(self) -> str | None:
        names = self.runnable_step_names()
        return names[0] if names else None

    def validation_for(self, step_name: str) -> ValidationRecord | None:
        record = self.validations.get(str(step_name))
        if record is None or self.artifact is None:
            return None
        stamp = record.stamp
        if (
            stamp.session_id != self.session_id
            or stamp.artifact_id != self.artifact.artifact_id
            or stamp.model_revision != self.model_revision
            or stamp.step_name != str(step_name)
        ):
            return None
        return record

    def validation_current(self, step_name: str) -> bool:
        record = self.validation_for(step_name)
        return record is not None and record.passed


class ModelSession:
    """Single writer for project inputs, derived artifacts, runs, and results."""

    def __init__(self) -> None:
        self._session_id = new_identity("session")
        self._session_revision = 0
        self._project_revision = 0
        self._mesh_input_revision = 0
        self._model_revision = 0
        self._saved_project_revision = 0
        self._clear_content()

    # ------------------------------------------------------------------
    # Stable scalar queries
    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_revision(self) -> int:
        return self._session_revision

    @property
    def project_revision(self) -> int:
        return self._project_revision

    @property
    def mesh_input_revision(self) -> int:
        return self._mesh_input_revision

    @property
    def model_revision(self) -> int:
        return self._model_revision

    @property
    def saved_project_revision(self) -> int:
        return self._saved_project_revision

    @property
    def dirty(self) -> bool:
        return (
            self._is_open
            and self._project_revision != self._saved_project_revision
        )

    @property
    def can_save(self) -> bool:
        return (
            self._is_open
            and self._source_kind == "native"
        )

    @property
    def active_part_id(self) -> str | None:
        return self._active_part_id

    @property
    def retired_part_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._retired_part_ids, key=part_id_sort_key))

    @property
    def retired_part_boolean_feature_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._retired_part_boolean_feature_ids,
                key=part_boolean_feature_id_sort_key,
            )
        )

    @property
    def next_native_part_id(self) -> str:
        return next_part_id(
            (part.id for part in self._parts),
            self._retired_part_ids,
        )

    @property
    def next_part_boolean_feature_id(self) -> str:
        active = (
            part.provenance.feature_id
            for part in self._parts
            if part.provenance is not None
        )
        return next_part_boolean_feature_id(
            active,
            self._retired_part_boolean_feature_ids,
        )

    def part_revision(self, part_id: str) -> int:
        normalized = normalize_part_id(part_id)
        if normalized not in self._part_revisions:
            raise KeyError(normalized)
        return int(self._part_revisions[normalized])

    def can_undo_part_extrusion(self, part_id: str) -> bool:
        return (
            normalize_part_id(part_id)
            in self._part_extrusion_undo_records
        )

    @property
    def retired_body_ids(self) -> tuple[str, ...]:
        """Body IDs observed earlier in this open authoring session."""

        return tuple(
            sorted(self._retired_body_ids, key=lambda value: int(value[1:]))
        )

    @property
    def retired_boolean_feature_ids(self) -> tuple[str, ...]:
        """Strict-Boolean IDs observed earlier in this open authoring session."""

        return tuple(
            sorted(
                self._retired_boolean_feature_ids,
                key=lambda value: int(value[2:]),
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    def new_native_project(
        self,
        name: str = "Model-1",
        *,
        part_name: str = "Part-1",
        body_name: str = "Body-1",
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        project_name = str(name).strip()
        if not project_name:
            raise ValueError("project name must not be empty")
        del part_name, body_name

        self._session_id = new_identity("session")
        self._clear_content()
        self._is_open = True
        self._source_kind = "native"
        self._model_name = project_name
        # A native Model starts empty.  Detached editors allocate the first
        # Part only when Finish commits valid geometry.
        self._parts = ()
        self._active_part_id = None
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        self._saved_project_revision = self._project_revision
        return self._emit(
            {
                ChangeKind.SESSION,
                ChangeKind.SOURCE,
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
                ChangeKind.SAVED_STATE,
            },
            _ALL_INVALIDATIONS,
            "new native project",
        )

    def close(
        self, *, expected_session_revision: int | None = None
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._session_id = new_identity("session")
        self._clear_content()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        self._saved_project_revision = self._project_revision
        return self._emit(
            {
                ChangeKind.SESSION,
                ChangeKind.SOURCE,
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
                ChangeKind.SAVED_STATE,
            },
            _ALL_INVALIDATIONS,
            "session closed",
        )

    def replace_from_snapshot(
        self,
        snapshot: ProjectSnapshot,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        detached = (
            snapshot
            if isinstance(snapshot, ProjectSnapshot)
            else ProjectSnapshot(
                snapshot.source_kind,
                snapshot.source_path,
                snapshot.parts,
                snapshot.geometry_recipe,
                snapshot.mesh_settings,
                snapshot.feature_history,
                snapshot.named_regions,
                snapshot.material_definitions,
                snapshot.section_definitions,
                snapshot.region_assignments,
                snapshot.analysis_definitions,
                getattr(snapshot, "model", None),
                getattr(snapshot, "boolean_reference_undo_records", ()),
                model_name=getattr(snapshot, "model_name", "Model-1"),
                part_boolean_undo_records=getattr(
                    snapshot,
                    "part_boolean_undo_records",
                    (),
                ),
                retired_part_ids=getattr(snapshot, "retired_part_ids", ()),
                retired_part_boolean_feature_ids=getattr(
                    snapshot,
                    "retired_part_boolean_feature_ids",
                    (),
                ),
                active_part_id=getattr(snapshot, "active_part_id", None),
            )
        )
        # ProjectSnapshot already owns a detached copy; copy once more so the
        # caller may retain and mutate its own nested values after installation.
        detached = deepcopy(detached)
        source_kind = _canonical_source_kind(detached.source_kind)
        raw_parts = deepcopy(detached.parts)
        canonical_part_mode = (
            source_kind == "native"
            and bool(raw_parts)
            and all(part.geometry_recipe is not None for part in raw_parts)
        )
        if canonical_part_mode:
            parts = validate_native_parts(raw_parts)
            active_part = (
                next(
                    (
                        part
                        for part in parts
                        if part.id == detached.active_part_id
                    ),
                    None,
                )
                if detached.active_part_id is not None
                else None
            )
            if active_part is None:
                active_part = next(
                    (part for part in parts if not part.suppressed),
                    parts[0],
                )
            geometry_recipe = deepcopy(active_part.geometry_recipe)
            mesh_settings = deepcopy(active_part.mesh_settings)
        else:
            parts = raw_parts or (
                (NativePart(),)
                if source_kind == "native"
                and detached.geometry_recipe is not None
                else ()
            )
            geometry_recipe = deepcopy(detached.geometry_recipe)
            mesh_settings = deepcopy(detached.mesh_settings)
        named_regions = _regions_by_name(detached.named_regions)
        definitions = normalize_model_definitions(
            detached.material_definitions,
            detached.section_definitions,
            detached.region_assignments,
            detached.analysis_definitions,
        )
        materials = definitions.materials
        sections = definitions.sections
        assignments = definitions.assignments
        steps = definitions.steps
        if source_kind == "native" and not canonical_part_mode:
            feature_history = (
                ()
                if geometry_recipe is None
                else derive_feature_history(geometry_recipe)
            )
            if tuple(detached.feature_history) != tuple(feature_history):
                raise ValueError(
                    "project feature_history is not the canonical projection "
                    "derived from geometry_recipe"
                )
            if geometry_recipe is not None:
                if (
                    mesh_settings is not None
                    and type(mesh_settings) is MeshSettings
                    and (
                        geometry_dimension(geometry_recipe) == 1
                        or mesh_settings.cell_shape == "line"
                    )
                ):
                    _validate_explicit_mesh_settings(mesh_settings, geometry_recipe)
                validate_native_project_inputs(
                    geometry_recipe,
                    mesh_settings,
                    tuple(named_regions.values()),
                    materials,
                    sections,
                    assignments,
                    steps,
                )
            elif (
                named_regions
                or (
                    mesh_settings is not None
                    and bool(getattr(mesh_settings, "local_controls", ()))
                )
                or assignments
                or analysis_steps_have_native_region_targets(steps)
            ):
                raise ValueError(
                    "native project inputs cannot reference geometry before "
                    "a geometry recipe exists"
                )
        elif source_kind != "native":
            feature_history = deepcopy(detached.feature_history)
        else:
            feature_history = deepcopy(active_part.feature_history)
            _validate_native_parts_project_inputs(
                parts,
                tuple(named_regions.values()),
                materials,
                sections,
                assignments,
                steps,
                authenticate_geometry=True,
            )
        model = deepcopy(detached.model)
        if model is not None:
            model = compile_model_definitions(
                model,
                definitions,
            ).require_model()

        self._session_id = new_identity("session")
        self._clear_content()
        self._is_open = True
        self._source_kind = source_kind
        self._model_name = (
            detached.model_name
            if source_kind == "native"
            else str(getattr(model, "name", "") or "").strip() or None
        )
        if source_kind == "native":
            self._project_path = detached.source_path
        else:
            self._source_path = detached.source_path
        self._geometry_recipe = geometry_recipe
        (
            self._retired_body_ids,
            self._retired_boolean_feature_ids,
        ) = (
            set(values)
            for values in retired_recipe_ids(geometry_recipe)
        )
        self._mesh_settings = mesh_settings
        self._parts = parts
        if canonical_part_mode:
            self._active_part_id = detached.active_part_id or next(
                (
                    part.id
                    for part in parts
                    if not part.suppressed
                ),
                None,
            )
        else:
            self._active_part_id = (
                parts[0].id
                if source_kind == "native" and parts
                else None
            )
        self._part_revisions = {
            part.id: 0 for part in parts
        }
        self._retired_part_ids = set(detached.retired_part_ids)
        self._retired_part_boolean_feature_ids = set(
            detached.retired_part_boolean_feature_ids
        )
        self._feature_history = feature_history
        if canonical_part_mode:
            self._sync_active_part_projection()
        self._named_regions = named_regions
        self._materials = materials
        self._sections = sections
        self._assignments = assignments
        self._steps = steps
        self._boolean_reference_undo_records = {
            record.feature_id: deepcopy(record)
            for record in detached.boolean_reference_undo_records
        }
        self._part_boolean_undo_records = {
            record.feature_id: deepcopy(record)
            for record in detached.part_boolean_undo_records
        }
        self._definitions_explicit = True
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        if model is not None:
            self._artifact = self._new_artifact(model, source_kind)
        self._saved_project_revision = self._project_revision
        return self._emit(
            {
                ChangeKind.SESSION,
                ChangeKind.SOURCE,
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
                ChangeKind.SAVED_STATE,
            },
            _ALL_INVALIDATIONS,
            "project snapshot installed",
        )

    # ------------------------------------------------------------------
    # Editable project inputs
    def replace_geometry(
        self,
        parts: Iterable[NativePart],
        recipe: Any,
        *,
        mesh_settings: MeshSettings | None | Unset = UNSET,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Compatibility spelling for the atomic native-input command."""

        return self.replace_native_geometry_inputs(
            parts,
            recipe,
            mesh_settings=mesh_settings,
            expected_session_revision=expected_session_revision,
        )

    def rename_native_model(
        self,
        name: str,
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Rename native-project model metadata without invalidating its mesh."""

        self._check_expected(expected_session_revision)
        self._require_native()
        model_name = str(name).strip()
        if not model_name:
            raise ValueError("model name must not be empty")
        if model_name == self._model_name:
            return self._emit(frozenset(), frozenset(), "model name unchanged")
        self._model_name = model_name
        if self._artifact is not None:
            renamed_model = deepcopy(self._artifact.model)
            renamed_model.name = model_name
            self._artifact = replace(
                self._artifact,
                model=renamed_model,
            )
        self._increment_domain_revisions(project=True)
        return self._emit(
            {ChangeKind.PROJECT_INPUTS},
            frozenset(),
            "native model renamed",
        )

    def set_active_native_part(
        self,
        part_id: str | None,
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Project one stable Part selection without changing project inputs."""

        self._check_expected(expected_session_revision)
        self._require_native()
        normalized = None if part_id is None else normalize_part_id(part_id)
        if normalized is not None:
            part = self._require_part(normalized)
            if part.suppressed:
                raise SessionStateError(
                    f"Part {normalized} is suppressed and cannot become active"
                )
        if normalized == self._active_part_id:
            return self._emit(
                frozenset(),
                frozenset(),
                "active Part unchanged",
            )
        self._active_part_id = normalized
        self._sync_active_part_projection()
        return self._emit(
            {ChangeKind.SESSION},
            frozenset(),
            "active Part changed",
        )

    def add_native_part(
        self,
        draft: NativePart | Any,
        *,
        name: str | None = None,
        mesh_settings: MeshSettings | None | Unset = UNSET,
        expected_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Atomically allocate and append one detached native Part draft."""

        expected = (
            expected_session_revision
            if expected_session_revision is not None
            else expected_revision
        )
        self._check_expected(expected)
        self._require_native()
        allocated_id = self.next_native_part_id
        if type(draft) is NativePart:
            recipe = deepcopy(draft.geometry_recipe)
            part_name = draft.name if name is None else str(name)
            requested_settings: MeshSettings | None | Unset = (
                draft.mesh_settings
                if isinstance(mesh_settings, Unset)
                else mesh_settings
            )
        else:
            recipe = deepcopy(draft)
            part_name = (
                f"部件-{part_id_sort_key(allocated_id)}"
                if name is None
                else str(name)
            )
            requested_settings = mesh_settings
        if recipe is None:
            raise ValueError("new Part draft must contain geometry")
        if any(part.name == part_name.strip() for part in self._parts):
            raise ValueError(f"Part name already exists: {part_name.strip()!r}")
        local_settings = (
            _default_mesh_settings(recipe)
            if isinstance(requested_settings, Unset)
            else deepcopy(requested_settings)
        )
        owned_settings = _namespace_part_mesh_settings(
            allocated_id,
            local_settings,
        )
        part = NativePart(
            id=allocated_id,
            name=part_name,
            geometry_recipe=recipe,
            mesh_settings=owned_settings,
        )
        _validate_native_part_inputs(part, authenticate_geometry=True)
        self._parts = validate_native_parts((*self._parts, part))
        self._part_revisions[part.id] = 0
        self._active_part_id = part.id
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "native Part added",
        )

    def rename_native_part(
        self,
        part_id_or_name: str,
        name: str | None = None,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Rename one Part while preserving its stable identity and refs."""

        self._check_expected(expected_session_revision)
        self._require_native()
        if name is None:
            if len(self._parts) != 1:
                raise SessionStateError(
                    "native part rename requires a Part ID"
                )
            part_id = self._parts[0].id
            part_name = str(part_id_or_name).strip()
        else:
            part_id = normalize_part_id(part_id_or_name)
            part_name = str(name).strip()
        legacy_projection = (
            part_id not in self._part_revisions
            and len(self._parts) == 1
            and self._parts[0].geometry_recipe is None
            and self._geometry_recipe is not None
        )
        if legacy_projection:
            if expected_part_revision is not None:
                raise SessionStateError(
                    "legacy authoring state has no Part revision"
                )
        else:
            self._check_part_revision(part_id, expected_part_revision)
        if not part_name:
            raise ValueError("part name must not be empty")
        current = (
            self._require_part(part_id)
            if legacy_projection
            else self._require_editable_part(part_id)
        )
        if part_name == current.name:
            return self._emit(frozenset(), frozenset(), "part name unchanged")
        if any(
            part.id != part_id and part.name == part_name
            for part in self._parts
        ):
            raise ValueError(f"Part name already exists: {part_name!r}")
        self._replace_part(replace(current, name=part_name))
        if not legacy_projection:
            self._part_revisions[part_id] += 1
            self._sync_active_part_projection()
        self._increment_domain_revisions(project=True)
        return self._emit(
            {ChangeKind.PROJECT_INPUTS},
            frozenset(),
            "native part renamed",
        )

    def replace_part_geometry(
        self,
        part_id: str,
        recipe: Any,
        *,
        mesh_settings: MeshSettings | None | Unset = UNSET,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Replace only one Part recipe and transition only its namespace."""

        self._check_expected(expected_session_revision)
        self._require_native()
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        current = self._require_editable_part(normalized)
        before_regions = tuple(self._named_regions.values())
        before_assignments = self._assignments
        before_steps = self._steps
        candidate_recipe = deepcopy(recipe)
        candidate_regions = _transition_part_named_regions(
            before_regions,
            normalized,
            current.geometry_recipe,
            candidate_recipe,
        )
        removed_region_names = set(self._named_regions).difference(
            region.name for region in candidate_regions
        )
        candidate_assignments = tuple(
            assignment
            for assignment in before_assignments
            if assignment.region_name not in removed_region_names
        )
        candidate_steps = (
            _without_geometry_dependent_steps(before_steps)
            if _region_references((), before_steps) & removed_region_names
            else before_steps
        )
        current_local_settings = _localize_part_mesh_settings(
            normalized,
            current.mesh_settings,
        )
        preserve = can_preserve_logical_references(
            current.geometry_recipe,
            candidate_recipe,
        )
        surviving = (
            frozenset()
            if preserve
            else surviving_logical_reference_ids(
                current.geometry_recipe,
                candidate_recipe,
            )
        )
        rewrites = (
            {}
            if preserve
            else logical_reference_transition_map(
                current.geometry_recipe,
                candidate_recipe,
            )
        )
        local_settings, mesh_effects = _transition_mesh_settings(
            current_local_settings,
            candidate_recipe,
            preserve_references=preserve,
            surviving_logical_ids=surviving,
            reference_rewrites=rewrites,
            requested=(
                mesh_settings
                if not isinstance(mesh_settings, MeshSettings)
                else _requested_local_part_mesh_settings(
                    normalized,
                    mesh_settings,
                )
            ),
        )
        updated = replace(
            current,
            geometry_recipe=candidate_recipe,
            mesh_settings=_namespace_part_mesh_settings(
                normalized,
                local_settings,
            ),
        )
        _validate_native_part_inputs(updated, authenticate_geometry=True)
        self._replace_part(updated)
        self._named_regions = {
            region.name: region for region in candidate_regions
        }
        self._assignments = candidate_assignments
        self._steps = candidate_steps
        self._part_revisions[normalized] += 1
        self._active_part_id = normalized
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.GEOMETRY,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        if updated.mesh_settings != current.mesh_settings:
            changed.add(ChangeKind.MESH_SETTINGS)
        if candidate_regions != before_regions:
            changed.add(ChangeKind.NAMED_REGIONS)
        if (
            candidate_assignments != before_assignments
            or candidate_steps != before_steps
        ):
            changed.add(ChangeKind.DEFINITIONS)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "Part geometry replaced",
            effects=mesh_effects,
        )

    def replace_part_with_extruded_siblings(
        self,
        part_id: str,
        recipes: Iterable[ExtrudedGeometry | RevolvedGeometry],
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Atomically turn canonical Profiles into independent solid Parts."""

        self._check_expected(expected_session_revision)
        self._require_native()
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        current = self._require_editable_part(normalized)
        requested_candidates = deepcopy(tuple(recipes))
        recipe_type = (
            None
            if not requested_candidates
            else type(requested_candidates[0])
        )
        if (
            recipe_type not in {ExtrudedGeometry, RevolvedGeometry}
            or any(type(recipe) is not recipe_type for recipe in requested_candidates)
        ):
            raise ValueError(
                "profile solid creation requires one or more recipes "
                "of the same supported feature type"
            )
        candidates = tuple(
            sorted(
                requested_candidates,
                key=lambda recipe: tuple(recipe.source_face_ids),
            )
        )
        if len(
            {tuple(recipe.source_face_ids) for recipe in candidates}
        ) != len(candidates):
            raise ValueError("Profile sources must be unique")

        before_regions = tuple(self._named_regions.values())
        before_assignments = self._assignments
        before_steps = self._steps
        primary_recipe = candidates[0]
        primary_regions = _transition_part_named_regions(
            before_regions,
            normalized,
            current.geometry_recipe,
            primary_recipe,
        )
        current_local_settings = _localize_part_mesh_settings(
            normalized,
            current.mesh_settings,
        )
        preserve = can_preserve_logical_references(
            current.geometry_recipe,
            primary_recipe,
        )
        rewrites = (
            {}
            if preserve
            else logical_reference_transition_map(
                current.geometry_recipe,
                primary_recipe,
            )
        )
        primary_settings, _effects = _transition_mesh_settings(
            current_local_settings,
            primary_recipe,
            preserve_references=preserve,
            surviving_logical_ids=(
                frozenset()
                if preserve
                else surviving_logical_reference_ids(
                    current.geometry_recipe,
                    primary_recipe,
                )
            ),
            reference_rewrites=rewrites,
            requested=UNSET,
        )
        primary = replace(
            current,
            geometry_recipe=primary_recipe,
            mesh_settings=_namespace_part_mesh_settings(
                normalized,
                primary_settings,
            ),
        )
        _validate_native_part_inputs(primary, authenticate_geometry=True)

        allocated_ids: list[str] = []
        active_ids = [part.id for part in self._parts]
        siblings: list[NativePart] = []
        for recipe in candidates[1:]:
            sibling_id = next_part_id(
                (*active_ids, *allocated_ids),
                self._retired_part_ids,
            )
            allocated_ids.append(sibling_id)
            sibling_preserve = can_preserve_logical_references(
                current.geometry_recipe,
                recipe,
            )
            sibling_rewrites = (
                {}
                if sibling_preserve
                else logical_reference_transition_map(
                    current.geometry_recipe,
                    recipe,
                )
            )
            sibling_settings, _sibling_effects = _transition_mesh_settings(
                current_local_settings,
                recipe,
                preserve_references=sibling_preserve,
                surviving_logical_ids=(
                    frozenset()
                    if sibling_preserve
                    else surviving_logical_reference_ids(
                        current.geometry_recipe,
                        recipe,
                    )
                ),
                reference_rewrites=sibling_rewrites,
                requested=UNSET,
            )
            sibling = NativePart(
                id=sibling_id,
                name=f"部件-{part_id_sort_key(sibling_id)}",
                geometry_recipe=recipe,
                mesh_settings=_namespace_part_mesh_settings(
                    sibling_id,
                    (
                        sibling_settings
                        if sibling_settings is not None
                        else _default_mesh_settings(recipe)
                    ),
                ),
            )
            _validate_native_part_inputs(sibling, authenticate_geometry=True)
            siblings.append(sibling)

        after_regions = _merge_extrusion_region_transitions(
            before_regions,
            normalized,
            current.geometry_recipe,
            primary_regions,
            tuple(
                (sibling.id, sibling.geometry_recipe)
                for sibling in siblings
            ),
        )
        removed_names = set(self._named_regions).difference(
            region.name for region in after_regions
        )
        after_assignments = tuple(
            assignment
            for assignment in before_assignments
            if assignment.region_name not in removed_names
        )
        after_steps = (
            _without_geometry_dependent_steps(before_steps)
            if _region_references((), before_steps) & removed_names
            else before_steps
        )
        after_parts = (primary, *siblings)
        prospective_parts = validate_native_parts(
            (
                *(
                    primary if part.id == normalized else part
                    for part in self._parts
                ),
                *siblings,
            )
        )
        self._parts = prospective_parts
        self._named_regions = {
            region.name: region for region in after_regions
        }
        self._assignments = after_assignments
        self._steps = after_steps
        self._part_revisions[normalized] += 1
        for sibling in siblings:
            self._part_revisions[sibling.id] = 0
        self._part_extrusion_undo_records[normalized] = (
            _PartExtrusionUndoRecord(
                normalized,
                current,
                after_parts,
                before_regions,
                after_regions,
                before_assignments,
                after_assignments,
                before_steps,
                after_steps,
            )
        )
        self._active_part_id = normalized
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(
            project=True,
            mesh=True,
            model=True,
        )
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            (
                "Profiles extruded into Parts"
                if recipe_type is ExtrudedGeometry
                else "Profiles swept into Parts"
            ),
        )

    def replace_part_with_revolved_siblings(
        self,
        part_id: str,
        recipes: Iterable[RevolvedGeometry],
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Atomically revolve canonical Profiles into independent Parts."""

        return self.replace_part_with_extruded_siblings(
            part_id,
            recipes,
            expected_part_revision=expected_part_revision,
            expected_session_revision=expected_session_revision,
        )

    def undo_part_extrusion(
        self,
        part_id: str,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Undo the latest shared multi-Profile extrusion group."""

        self._check_expected(expected_session_revision)
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        record = self._part_extrusion_undo_records.get(normalized)
        if record is None:
            raise SessionStateError(
                f"Part {normalized} has no reversible extrusion group"
            )
        by_id = {part.id: part for part in self._parts}
        if any(
            by_id.get(part.id) != part for part in record.after_parts
        ):
            raise SessionStateError(
                "extrusion.undo-conflict: one result Part changed"
            )
        if (
            tuple(self._named_regions.values())
            != record.after_named_regions
            or self._assignments != record.after_assignments
            or self._steps != record.after_steps
        ):
            raise SessionStateError(
                "extrusion.undo-reference-conflict: definitions changed"
            )
        sibling_ids = {
            part.id for part in record.after_parts[1:]
        }
        self._parts = validate_native_parts(
            tuple(
                record.before_part
                if part.id == normalized
                else part
                for part in self._parts
                if part.id not in sibling_ids
            )
        )
        self._retired_part_ids.update(sibling_ids)
        for sibling_id in sibling_ids:
            self._part_revisions.pop(sibling_id, None)
        self._part_revisions[normalized] += 1
        self._named_regions = {
            region.name: region
            for region in record.before_named_regions
        }
        self._assignments = record.before_assignments
        self._steps = record.before_steps
        self._part_extrusion_undo_records.pop(normalized, None)
        self._active_part_id = normalized
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(
            project=True,
            mesh=True,
            model=True,
        )
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "Part extrusion group undone",
        )

    def replace_part_mesh_settings(
        self,
        part_id: str,
        settings: MeshSettings | None,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Replace mesh policy on exactly one active Part."""

        self._check_expected(expected_session_revision)
        self._require_native()
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        current = self._require_editable_part(normalized)
        owned = _namespace_part_mesh_settings(normalized, deepcopy(settings))
        updated = replace(current, mesh_settings=owned)
        _validate_native_part_inputs(updated)
        if updated == current:
            return self._emit(
                frozenset(),
                frozenset(),
                "Part mesh settings unchanged",
            )
        self._replace_part(updated)
        self._part_revisions[normalized] += 1
        self._active_part_id = normalized
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "Part mesh settings replaced",
        )

    def delete_native_part(
        self,
        part_id: str,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Delete one ordinary Part or undo one Boolean result Part."""

        self._check_expected(expected_session_revision)
        self._require_native()
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        part = self._require_part(normalized)
        if part.provenance is not None:
            return self.undo_part_boolean(
                normalized,
                expected_part_revision=expected_part_revision,
            )
        if self._part_has_active_descendant(normalized):
            raise SessionStateError(
                f"Part {normalized} is locked by an active Boolean result"
            )
        before_regions = tuple(self._named_regions.values())
        before_assignments = self._assignments
        before_steps = self._steps
        candidate_regions = _remove_part_from_named_regions(
            before_regions,
            normalized,
        )
        removed_names = set(self._named_regions).difference(
            region.name for region in candidate_regions
        )
        candidate_assignments = tuple(
            assignment
            for assignment in before_assignments
            if assignment.region_name not in removed_names
        )
        candidate_steps = (
            _without_geometry_dependent_steps(before_steps)
            if _region_references((), before_steps) & removed_names
            else before_steps
        )
        self._parts = tuple(
            value for value in self._parts if value.id != normalized
        )
        self._part_revisions.pop(normalized, None)
        self._retired_part_ids.add(normalized)
        self._named_regions = {
            region.name: region for region in candidate_regions
        }
        self._assignments = candidate_assignments
        self._steps = candidate_steps
        if self._active_part_id == normalized:
            self._active_part_id = next(
                (
                    value.id
                    for value in self._parts
                    if not value.suppressed
                ),
                None,
            )
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.GEOMETRY,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        effects: set[TransitionEffect] = set()
        if candidate_regions != before_regions:
            changed.add(ChangeKind.NAMED_REGIONS)
            effects.add(TransitionEffect.NAMED_REGIONS_CLEARED)
        if candidate_assignments != before_assignments:
            changed.add(ChangeKind.DEFINITIONS)
            effects.add(TransitionEffect.ASSIGNMENTS_CLEARED)
        if candidate_steps != before_steps:
            changed.add(ChangeKind.DEFINITIONS)
            effects.add(TransitionEffect.STEPS_CLEARED)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "native Part deleted",
            effects=effects,
        )

    def suppress_native_part(
        self,
        part_id: str,
        suppressed: bool = True,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Set explicit Part suppression while respecting the provenance DAG."""

        self._check_expected(expected_session_revision)
        self._require_native()
        if type(suppressed) is not bool:
            raise TypeError("suppressed must be a bool")
        normalized = normalize_part_id(part_id)
        self._check_part_revision(normalized, expected_part_revision)
        current = self._require_part(normalized)
        if not suppressed and self._part_has_active_descendant(normalized):
            raise SessionStateError(
                f"Part {normalized} cannot be unsuppressed while an active "
                "Boolean descendant exists"
            )
        if current.suppressed == suppressed:
            return self._emit(
                frozenset(),
                frozenset(),
                "Part suppression unchanged",
            )
        self._replace_part(replace(current, suppressed=suppressed))
        self._part_revisions[normalized] += 1
        if suppressed and self._active_part_id == normalized:
            self._active_part_id = next(
                (
                    part.id
                    for part in self._parts
                    if not part.suppressed
                ),
                None,
            )
        elif not suppressed:
            self._active_part_id = normalized
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "Part suppression changed",
        )

    def apply_part_boolean(
        self,
        target_id: str,
        tool_id: str,
        operation: str,
        result_name: str,
        *,
        result: Any | None = None,
        result_recipe: Any | None = None,
        expected_target_revision: int | None = None,
        expected_tool_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Commit one already-proven detached Part Boolean atomically."""

        self._check_expected(expected_session_revision)
        self._require_native()
        target_key = normalize_part_id(target_id, "target_id")
        tool_key = normalize_part_id(tool_id, "tool_id")
        if target_key == tool_key:
            raise ValueError("target and tool Parts must differ")
        self._check_part_revision(target_key, expected_target_revision)
        self._check_part_revision(tool_key, expected_tool_revision)
        target = self._require_editable_part(target_key)
        tool = self._require_editable_part(tool_key)
        if target.dimension != 3 or tool.dimension != 3:
            raise ValueError("Part Boolean requires two 3D Parts")
        if operation not in {"fuse", "cut"}:
            raise ValueError("Part Boolean operation must be fuse or cut")
        normalized_name = str(result_name).strip()
        if not normalized_name:
            raise ValueError("result Part name must not be empty")
        if any(part.name == normalized_name for part in self._parts):
            raise ValueError(f"Part name already exists: {normalized_name!r}")

        from .part_boolean import StrictPartBooleanResult

        recipe = (
            deepcopy(result.recipe)
            if type(result) is StrictPartBooleanResult
            else deepcopy(result_recipe)
        )
        if not isinstance(recipe, BooleanGeometry):
            raise TypeError(
                "Part Boolean commit requires a proven BooleanGeometry result"
            )
        context = recipe.part_context
        if context is None or not context.proven:
            raise ValueError("Part Boolean result requires complete lineage proof")
        allocated_part_id = self.next_native_part_id
        allocated_feature_id = self.next_part_boolean_feature_id
        if (
            context.target_part_id != target_key
            or context.tool_part_id != tool_key
            or context.result_part_id != allocated_part_id
            or context.feature_id != allocated_feature_id
            or recipe.operation != operation
        ):
            raise SessionStateError(
                "detached Part Boolean identity no longer matches Session"
            )
        if (
            recipe.object_geometry != target.geometry_recipe
            or recipe.tool_geometry != tool.geometry_recipe
        ):
            raise SessionStateError(
                "detached Part Boolean proof no longer matches source geometry"
            )

        before_regions = tuple(self._named_regions.values())
        before_assignments = self._assignments
        before_steps = self._steps
        forward_map = _part_boolean_forward_map(context)
        after_regions = _rewrite_part_boolean_regions(
            before_regions,
            forward_map,
            {target_key, tool_key},
        )
        removed_names = set(self._named_regions).difference(
            region.name for region in after_regions
        )
        after_assignments = tuple(
            assignment
            for assignment in before_assignments
            if assignment.region_name not in removed_names
        )
        after_steps = (
            _without_geometry_dependent_steps(before_steps)
            if _region_references((), before_steps) & removed_names
            else before_steps
        )
        result_settings = _transition_part_boolean_mesh_settings(
            target.mesh_settings,
            forward_map,
            target_key,
            allocated_part_id,
        )
        result_part = NativePart(
            id=allocated_part_id,
            name=normalized_name,
            geometry_recipe=recipe,
            mesh_settings=result_settings,
            provenance=PartBooleanProvenance(
                allocated_feature_id,
                target_key,
                tool_key,
                operation,
            ),
        )
        _validate_native_part_inputs(result_part, authenticate_geometry=True)
        record = PartBooleanUndoRecord(
            allocated_feature_id,
            allocated_part_id,
            (target, tool),
            result_part,
            before_regions,
            after_regions,
            before_assignments,
            after_assignments,
            before_steps,
            after_steps,
        )
        self._replace_part(replace(target, suppressed=True))
        self._replace_part(replace(tool, suppressed=True))
        self._parts = validate_native_parts((*self._parts, result_part))
        self._part_revisions[target_key] += 1
        self._part_revisions[tool_key] += 1
        self._part_revisions[allocated_part_id] = 0
        self._part_boolean_undo_records[allocated_feature_id] = record
        self._named_regions = {
            region.name: region for region in after_regions
        }
        self._assignments = after_assignments
        self._steps = after_steps
        self._active_part_id = allocated_part_id
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.GEOMETRY,
            ChangeKind.MESH_SETTINGS,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        if after_regions != before_regions:
            changed.add(ChangeKind.NAMED_REGIONS)
        if (
            after_assignments != before_assignments
            or after_steps != before_steps
        ):
            changed.add(ChangeKind.DEFINITIONS)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "Part Boolean applied",
            effects={TransitionEffect.REFERENCES_PRESERVED},
        )

    def undo_part_boolean(
        self,
        result_part_id: str,
        *,
        expected_part_revision: int | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Delete one result Part and exactly restore its direct sources."""

        self._check_expected(expected_session_revision)
        self._require_native()
        result_id = normalize_part_id(result_part_id)
        self._check_part_revision(result_id, expected_part_revision)
        result = self._require_part(result_id)
        provenance = result.provenance
        if provenance is None:
            raise SessionStateError("Part is not a Boolean result")
        if self._part_has_active_descendant(result_id):
            raise SessionStateError(
                "undo later Part Boolean descendants before this result"
            )
        record = self._part_boolean_undo_records.get(provenance.feature_id)
        if record is None or record.result_part_id != result_id:
            raise SessionStateError(
                "Part Boolean undo record is missing or stale"
            )
        current_sources = tuple(
            self._require_part(part.id) for part in record.source_parts
        )
        if (
            result.provenance != record.result_part.provenance
            or not _recipe_contains_geometry_state(
                result.geometry_recipe,
                record.result_part.geometry_recipe,
            )
            or tuple(self._named_regions.values())
            != record.after_named_regions
            or self._assignments != record.after_assignments
            or self._steps != record.after_steps
            or any(not part.suppressed for part in current_sources)
        ):
            raise SessionStateError(
                "Part Boolean state changed after commit; undo later edits first"
            )
        self._parts = tuple(
            part
            for part in self._parts
            if part.id not in {
                result_id,
                *(source.id for source in record.source_parts),
            }
        )
        self._parts = validate_native_parts(
            tuple(
                sorted(
                    (*self._parts, *record.source_parts),
                    key=lambda part: part_id_sort_key(part.id),
                )
            )
        )
        self._part_revisions.pop(result_id, None)
        for source in record.source_parts:
            self._part_revisions[source.id] = (
                self._part_revisions.get(source.id, 0) + 1
            )
        self._retired_part_ids.add(result_id)
        self._retired_part_boolean_feature_ids.add(provenance.feature_id)
        self._part_boolean_undo_records.pop(provenance.feature_id, None)
        self._named_regions = {
            region.name: region for region in record.before_named_regions
        }
        self._assignments = record.before_assignments
        self._steps = record.before_steps
        self._active_part_id = record.source_parts[0].id
        self._sync_active_part_projection()
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "Part Boolean undone",
            effects={TransitionEffect.REFERENCES_PRESERVED},
        )

    def replace_native_geometry_inputs(
        self,
        parts: Iterable[NativePart],
        recipe: Any,
        *,
        mesh_settings: MeshSettings | None | Unset = UNSET,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        """Replace geometry and mesh inputs in one validated Session commit."""

        self._check_expected(expected_session_revision)
        self._require_native()
        prior_parts = {part.id: part for part in self._parts}
        prior_part_revisions = dict(self._part_revisions)
        owned_parts = deepcopy(tuple(parts))
        owned_recipe = deepcopy(recipe)
        if owned_recipe is not None and not owned_parts:
            owned_parts = (NativePart(),)
        before_body_ids, before_feature_ids = historical_recipe_ids(
            self._geometry_recipe
        )
        after_body_ids, after_feature_ids = historical_recipe_ids(
            owned_recipe
        )
        explicit_body_ids, explicit_feature_ids = retired_recipe_ids(
            owned_recipe
        )
        next_retired_body_ids = {
            *self._retired_body_ids,
            *explicit_body_ids,
            *(before_body_ids - after_body_ids),
        }
        next_retired_feature_ids = {
            *self._retired_boolean_feature_ids,
            *explicit_feature_ids,
            *(before_feature_ids - after_feature_ids),
        }
        if isinstance(owned_recipe, MultiBodyGeometry):
            active_body_ids = {body.id for body in owned_recipe.bodies}
            active_feature_ids = set(after_feature_ids) - set(
                explicit_feature_ids
            )
            owned_recipe = replace(
                owned_recipe,
                retired_body_ids=tuple(
                    next_retired_body_ids - active_body_ids
                ),
                retired_boolean_feature_ids=tuple(
                    next_retired_feature_ids - active_feature_ids
                ),
            )
        owned_history = (
            () if owned_recipe is None else derive_feature_history(owned_recipe)
        )
        preserve_references = can_preserve_logical_references(
            self._geometry_recipe,
            owned_recipe,
        ) and not _regions_use_mesh_entities(self._named_regions.values())
        surviving_logical_ids = (
            frozenset()
            if preserve_references
            or _regions_use_mesh_entities(self._named_regions.values())
            else surviving_logical_reference_ids(
                self._geometry_recipe,
                owned_recipe,
            )
        )
        reference_rewrites = (
            {}
            if preserve_references
            or _regions_use_mesh_entities(self._named_regions.values())
            else logical_reference_transition_map(
                self._geometry_recipe,
                owned_recipe,
            )
        )
        boolean_transition = _strict_boolean_transition(
            self._geometry_recipe,
            owned_recipe,
        )
        reverse_undo_record: BooleanReferenceUndoRecord | None = None
        if (
            boolean_transition is not None
            and boolean_transition[0] == "reverse"
        ):
            context = boolean_transition[1]
            reverse_undo_record = self._boolean_reference_undo_records.get(
                context.feature_id
            )
            if reverse_undo_record is not None and (
                self._geometry_recipe != reverse_undo_record.after_geometry
                or not _same_active_multi_body_geometry(
                    owned_recipe,
                    reverse_undo_record.before_geometry,
                )
                or tuple(self._named_regions.values())
                != reverse_undo_record.after_named_regions
                or self._mesh_settings
                != reverse_undo_record.after_mesh_settings
                or self._materials != reverse_undo_record.after_materials
                or self._sections != reverse_undo_record.after_sections
                or self._assignments != reverse_undo_record.after_assignments
                or self._steps != reverse_undo_record.after_steps
            ):
                diagnostic = (
                    "boolean.planar.undo-reference-conflict"
                    if type(context) is PlanarBooleanContext
                    else "boolean.body.undo-reference-conflict"
                )
                raise SessionStateError(
                    f"{diagnostic}: reference state changed after the Boolean"
                )
        candidate_regions = (
            deepcopy(reverse_undo_record.before_named_regions)
            if reverse_undo_record is not None
            else (
                tuple(self._named_regions.values())
                if preserve_references
                else tuple(
                    rewritten
                    for region in self._named_regions.values()
                    if (
                        rewritten := _rewrite_named_region(
                            region,
                            reference_rewrites,
                        )
                    )
                    is not None
                )
            )
        )
        prior_region_count = len(self._named_regions)
        valid_region_names = {
            "DOMAIN",
            *(region.name for region in candidate_regions),
        }
        candidate_assignments = (
            deepcopy(reverse_undo_record.before_assignments)
            if reverse_undo_record is not None
            else (
                self._assignments
                if preserve_references
                else tuple(
                    assignment
                    for assignment in self._assignments
                    if assignment.region_name in valid_region_names
                )
            )
        )
        referenced_step_regions = _region_references((), self._steps)
        candidate_steps = (
            deepcopy(reverse_undo_record.before_steps)
            if reverse_undo_record is not None
            else (
                self._steps
                if preserve_references
                or referenced_step_regions.issubset(valid_region_names)
                else _without_geometry_dependent_steps(self._steps)
            )
        )
        if reverse_undo_record is not None and isinstance(
            mesh_settings,
            Unset,
        ):
            candidate_mesh_settings = deepcopy(
                reverse_undo_record.before_mesh_settings
            )
            mesh_effects = frozenset()
        else:
            candidate_mesh_settings, mesh_effects = _transition_mesh_settings(
                self._mesh_settings,
                owned_recipe,
                preserve_references=preserve_references,
                surviving_logical_ids=surviving_logical_ids,
                reference_rewrites=reference_rewrites,
                requested=mesh_settings,
            )
        mesh_settings_changed = candidate_mesh_settings != self._mesh_settings
        named_regions_changed = candidate_regions != tuple(
            self._named_regions.values()
        )
        assignments_changed = candidate_assignments != self._assignments
        steps_changed = candidate_steps != self._steps
        assignments_cleared = any(
            assignment not in candidate_assignments
            for assignment in self._assignments
        )
        steps_cleared = any(
            step not in candidate_steps for step in self._steps
        )
        definitions_changed = (
            not self._definitions_explicit
            or assignments_changed
            or steps_changed
        )
        if owned_recipe is not None:
            validate_native_project_inputs(
                owned_recipe,
                candidate_mesh_settings,
                candidate_regions,
                self._materials,
                self._sections,
                candidate_assignments,
                candidate_steps,
            )

        forward_undo_record: BooleanReferenceUndoRecord | None = None
        if (
            boolean_transition is not None
            and boolean_transition[0] == "forward"
        ):
            context = boolean_transition[1]
            if context.feature_id in self._boolean_reference_undo_records:
                raise SessionStateError(
                    "boolean.body.undo-reference-conflict: duplicate "
                    f"transition record {context.feature_id!r}"
                )
            forward_undo_record = BooleanReferenceUndoRecord(
                context.feature_id,
                _boolean_context_target(context),
                deepcopy(self._geometry_recipe),
                deepcopy(owned_recipe),
                deepcopy(tuple(self._named_regions.values())),
                deepcopy(candidate_regions),
                deepcopy(self._mesh_settings),
                deepcopy(candidate_mesh_settings),
                deepcopy(self._materials),
                deepcopy(self._materials),
                deepcopy(self._sections),
                deepcopy(self._sections),
                deepcopy(self._assignments),
                deepcopy(candidate_assignments),
                deepcopy(self._steps),
                deepcopy(candidate_steps),
            )

        self._parts = owned_parts
        self._part_revisions = {
            part.id: (
                prior_part_revisions.get(part.id, 0)
                + int(
                    part.id in prior_parts
                    and prior_parts[part.id] != part
                )
            )
            for part in owned_parts
        }
        if (
            self._active_part_id not in self._part_revisions
            or self._require_part(self._active_part_id).suppressed
        ):
            self._active_part_id = next(
                (
                    part.id
                    for part in owned_parts
                    if not part.suppressed
                ),
                None,
            )
        self._retired_body_ids = set(next_retired_body_ids)
        self._retired_boolean_feature_ids = set(
            next_retired_feature_ids
        )
        self._geometry_recipe = owned_recipe
        self._feature_history = owned_history
        self._mesh_settings = candidate_mesh_settings
        if not preserve_references:
            self._named_regions = {
                region.name: region for region in candidate_regions
            }
            self._assignments = candidate_assignments
        self._steps = candidate_steps
        if forward_undo_record is not None:
            self._boolean_reference_undo_records[
                forward_undo_record.feature_id
            ] = forward_undo_record
        if reverse_undo_record is not None:
            self._boolean_reference_undo_records.pop(
                reverse_undo_record.feature_id,
                None,
            )
        active_boolean_feature_ids = {
            context.feature_id
            for context in _active_boolean_contexts(owned_recipe)
        }
        self._boolean_reference_undo_records = {
            feature_id: record
            for feature_id, record in self._boolean_reference_undo_records.items()
            if feature_id in active_boolean_feature_ids
        }
        self._definitions_explicit = True
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.GEOMETRY,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        if mesh_settings_changed:
            changed.add(ChangeKind.MESH_SETTINGS)
        if named_regions_changed:
            changed.add(ChangeKind.NAMED_REGIONS)
        if definitions_changed:
            changed.add(ChangeKind.DEFINITIONS)
        effects = set(mesh_effects)
        if preserve_references or (
            reference_rewrites and candidate_regions
        ):
            effects.add(TransitionEffect.REFERENCES_PRESERVED)
        if len(candidate_regions) < prior_region_count:
            effects.add(TransitionEffect.NAMED_REGIONS_CLEARED)
        if assignments_cleared:
            effects.add(TransitionEffect.ASSIGNMENTS_CLEARED)
        if steps_cleared:
            effects.add(TransitionEffect.STEPS_CLEARED)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "geometry replaced",
            effects=effects,
        )

    def clear_geometry(
        self, *, expected_session_revision: int | None = None
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._require_open()
        self._retired_part_ids.update(part.id for part in self._parts)
        self._retired_part_boolean_feature_ids.update(
            part.provenance.feature_id
            for part in self._parts
            if part.provenance is not None
        )
        self._parts = ()
        self._active_part_id = None
        self._part_revisions.clear()
        self._part_boolean_undo_records.clear()
        self._part_extrusion_undo_records.clear()
        body_ids, feature_ids = historical_recipe_ids(self._geometry_recipe)
        self._retired_body_ids.update(body_ids)
        self._retired_boolean_feature_ids.update(feature_ids)
        self._geometry_recipe = None
        self._feature_history = ()
        self._boolean_reference_undo_records.clear()
        self._mesh_settings = _without_mesh_topology_references(
            self._mesh_settings
        )
        self._named_regions = {}
        self._assignments = ()
        self._steps = ()
        self._definitions_explicit = True
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.GEOMETRY,
                ChangeKind.MESH_SETTINGS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _MODEL_INVALIDATIONS,
            "geometry cleared",
        )

    def replace_mesh_settings(
        self,
        settings: Any,
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._require_native()
        if self._canonical_part_state():
            if self._active_part_id is None:
                raise SessionStateError(
                    "Part mesh settings require an active Part"
                )
            return self.replace_part_mesh_settings(
                self._active_part_id,
                settings,
                expected_session_revision=expected_session_revision,
            )
        owned = deepcopy(settings)
        clears_mesh_scopes = _regions_use_mesh_entities(
            self._named_regions.values()
        )
        candidate_regions = () if clears_mesh_scopes else tuple(
            self._named_regions.values()
        )
        candidate_assignments = () if clears_mesh_scopes else self._assignments
        candidate_steps = (
            _without_geometry_dependent_steps(self._steps)
            if clears_mesh_scopes
            else self._steps
        )
        had_regions = clears_mesh_scopes and bool(self._named_regions)
        had_assignments = clears_mesh_scopes and bool(self._assignments)
        steps_cleared = candidate_steps != self._steps
        if self._geometry_recipe is not None:
            if (
                owned is not None
                and type(owned) is MeshSettings
                and (
                    geometry_dimension(self._geometry_recipe) == 1
                    or owned.cell_shape == "line"
                )
            ):
                _validate_explicit_mesh_settings(owned, self._geometry_recipe)
            validate_native_project_inputs(
                self._geometry_recipe,
                owned,
                candidate_regions,
                self._materials,
                self._sections,
                candidate_assignments,
                candidate_steps,
            )
        elif owned is not None and bool(
            getattr(owned, "local_controls", ())
        ):
            raise ValueError(
                "local mesh controls require a geometry recipe"
            )
        self._mesh_settings = owned
        if clears_mesh_scopes:
            self._named_regions = {}
            self._assignments = ()
            self._steps = candidate_steps
            self._definitions_explicit = True
        self._drop_model_state()
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.MESH_SETTINGS,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        effects: set[TransitionEffect] = set()
        if had_regions:
            changed.add(ChangeKind.NAMED_REGIONS)
            effects.add(TransitionEffect.NAMED_REGIONS_CLEARED)
        if had_assignments or steps_cleared:
            changed.add(ChangeKind.DEFINITIONS)
        if had_assignments:
            effects.add(TransitionEffect.ASSIGNMENTS_CLEARED)
        if steps_cleared:
            effects.add(TransitionEffect.STEPS_CLEARED)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "mesh settings replaced",
            effects=effects,
        )

    def replace_named_regions(
        self,
        regions: Mapping[str, NamedRegion] | Iterable[NamedRegion],
        *,
        renames: Mapping[str, str] | None = None,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._require_native()
        owned = _regions_by_name(deepcopy(_mapping_values(regions)))
        if self._canonical_part_state() and self._artifact is not None:
            owned = _regions_by_name(
                _attach_mesh_reference_owners(
                    tuple(owned.values()),
                    self._parts,
                    self._artifact.model,
                )
            )
        rename_map = {
            str(old_name): str(new_name)
            for old_name, new_name in (renames or {}).items()
        }
        old_names = set(self._named_regions)
        for old_name, new_name in rename_map.items():
            if old_name not in old_names:
                raise KeyError(f"unknown named region to rename: {old_name}")
            if new_name not in owned:
                raise ValueError(
                    f"renamed region {new_name!r} is absent from replacement"
                )
        if len(set(rename_map.values())) != len(rename_map):
            raise ValueError("multiple named regions cannot be renamed to one name")

        assignments = tuple(
            replace(
                assignment,
                region_name=rename_map.get(
                    str(assignment.region_name),
                    str(assignment.region_name),
                ),
            )
            for assignment in self._assignments
        )
        steps = _rename_step_region_references(self._steps, rename_map)
        referenced_old_names = _region_references(
            self._assignments,
            self._steps,
        ) & old_names
        removed_references = sorted(
            name
            for name in referenced_old_names
            if name not in rename_map and name not in owned
        )
        if removed_references:
            raise ValueError(
                "named region replacement would remove referenced regions: "
                + ", ".join(removed_references)
            )
        if self._geometry_recipe is None:
            if owned and (
                self._source_kind != "imported"
                or self._artifact is None
                or not _regions_use_mesh_entities(owned.values())
            ):
                raise ValueError(
                    "named regions require a geometry recipe or imported mesh"
                )
        elif (
            _regions_use_mesh_entities(owned.values())
            and self._artifact is None
        ):
            raise ValueError("mesh scopes require a generated mesh")
        elif not _regions_use_mesh_entities(owned.values()):
            if self._canonical_part_state():
                _validate_part_namespaced_references(
                    self._parts,
                    tuple(owned.values()),
                )
            else:
                validate_native_project_inputs(
                    self._geometry_recipe,
                    self._mesh_settings,
                    tuple(owned.values()),
                    self._materials,
                    self._sections,
                    assignments,
                    steps,
                )

        return self._commit_named_region_state(
            owned,
            assignments,
            steps,
            "named regions replaced",
        )

    def apply_named_region_edit(
        self,
        batch: NamedRegionEditBatch,
    ) -> SessionDelta:
        """Atomically apply one explicit named-region post-state and ledger."""

        if type(batch) is not NamedRegionEditBatch:
            raise TypeError("batch must be a NamedRegionEditBatch")
        self._check_expected(batch.base_session_revision)
        self._require_open()

        owned = _regions_by_name(batch.regions)
        if self._canonical_part_state() and self._artifact is not None:
            owned = _regions_by_name(
                _attach_mesh_reference_owners(
                    tuple(owned.values()),
                    self._parts,
                    self._artifact.model,
                )
            )
        rename_map = _validate_edit_ledger(
            tuple(self._named_regions.values()),
            tuple(owned.values()),
            batch.renames,
            batch.deletes,
            label="named region",
        )
        delete_names = {intent.name for intent in batch.deletes}
        referenced_deletes = sorted(
            delete_names
            & _region_references(self._assignments, self._steps)
        )
        if referenced_deletes:
            raise ValueError(
                "cannot delete referenced named regions: "
                + ", ".join(referenced_deletes)
            )

        assignments = tuple(
            replace(
                assignment,
                region_name=rename_map.get(
                    str(assignment.region_name),
                    str(assignment.region_name),
                ),
            )
            for assignment in self._assignments
        )
        steps = _rename_step_region_references(self._steps, rename_map)
        if self._geometry_recipe is None:
            if owned and (
                self._source_kind != "imported"
                or self._artifact is None
                or not _regions_use_mesh_entities(owned.values())
            ):
                raise ValueError(
                    "named regions require a geometry recipe or imported mesh"
                )
        elif (
            _regions_use_mesh_entities(owned.values())
            and self._artifact is None
        ):
            raise ValueError("mesh scopes require a generated mesh")
        elif not _regions_use_mesh_entities(owned.values()):
            if self._canonical_part_state():
                _validate_part_namespaced_references(
                    self._parts,
                    tuple(owned.values()),
                )
            else:
                validate_native_project_inputs(
                    self._geometry_recipe,
                    self._mesh_settings,
                    tuple(owned.values()),
                    self._materials,
                    self._sections,
                    assignments,
                    steps,
                )

        return self._commit_named_region_state(
            owned,
            assignments,
            steps,
            "named region edit applied",
        )

    def _commit_named_region_state(
        self,
        owned: dict[str, NamedRegion],
        assignments: tuple[RegionAssignment, ...],
        steps: tuple[Any, ...],
        reason: str,
    ) -> SessionDelta:
        """Commit scope definitions, preserving an already-generated mesh."""

        previous_artifact = self._artifact
        compiled_model = None
        if (
            previous_artifact is not None
            and can_materialize_native_scopes(
                previous_artifact.model,
                owned.values(),
            )
        ):
            scoped_model = materialize_native_scopes(
                previous_artifact.model,
                previous_names=tuple(self._named_regions),
                regions=tuple(owned.values()),
            )
            compiled_model = compile_model_definitions(
                scoped_model,
                ModelDefinitions(
                    self._materials,
                    self._sections,
                    assignments,
                    steps,
                ),
            ).require_model()

        self._named_regions = owned
        self._assignments = assignments
        self._steps = steps
        if compiled_model is None:
            self._drop_model_state()
        else:
            self._drop_computations()
        edits_mesh_scopes = (
            _regions_use_mesh_entities(self._named_regions.values())
            or _regions_use_mesh_entities(owned.values())
        )
        self._increment_domain_revisions(
            project=True,
            mesh=not edits_mesh_scopes,
            model=True,
        )
        if compiled_model is not None:
            self._artifact = self._new_artifact(
                compiled_model,
                (
                    previous_artifact.source_kind
                    if previous_artifact is not None
                    else "native"
                ),
            )
        invalidated = (
            _MODEL_INVALIDATIONS
            if compiled_model is None
            else _COMPUTATION_INVALIDATIONS
            | frozenset({ArtifactKind.MODEL})
        )
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.NAMED_REGIONS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            invalidated,
            reason,
        )

    def replace_model_definitions(
        self,
        materials: Mapping[str, Any] | Iterable[Any],
        sections: Iterable[SectionDefinition],
        assignments: Iterable[RegionAssignment],
        steps: Iterable[Any],
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._require_open()
        owned = normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
        return self._commit_model_definitions(
            owned,
            reason="model definitions replaced",
        )

    def apply_definition_edit(
        self,
        batch: DefinitionEditBatch,
    ) -> SessionDelta:
        """Atomically apply an explicit material/section definition batch."""

        if type(batch) is not DefinitionEditBatch:
            raise TypeError("batch must be a DefinitionEditBatch")
        self._check_expected(batch.base_session_revision)
        self._require_open()

        material_renames = _validate_edit_ledger(
            self._materials,
            batch.materials,
            batch.material_renames,
            batch.material_deletes,
            label="material",
        )
        section_renames = _validate_edit_ledger(
            self._sections,
            batch.sections,
            batch.section_renames,
            batch.section_deletes,
            label="section",
        )
        deleted_materials = {
            intent.name for intent in batch.material_deletes
        }
        referenced_materials = {
            str(section.material) for section in self._sections
        }
        invalid_material_deletes = sorted(
            deleted_materials & referenced_materials
        )
        if invalid_material_deletes:
            raise ValueError(
                "cannot delete referenced materials: "
                + ", ".join(invalid_material_deletes)
            )
        deleted_sections = {
            intent.name for intent in batch.section_deletes
        }
        assigned_sections = {
            str(assignment.section_name)
            for assignment in self._assignments
        }
        invalid_section_deletes = sorted(
            deleted_sections & assigned_sections
        )
        if invalid_section_deletes:
            raise ValueError(
                "cannot delete assigned sections: "
                + ", ".join(invalid_section_deletes)
            )

        sections = tuple(
            replace(
                section,
                material=material_renames.get(
                    str(section.material),
                    str(section.material),
                ),
            )
            for section in batch.sections
        )
        assignments = tuple(
            replace(
                assignment,
                section_name=section_renames.get(
                    str(assignment.section_name),
                    str(assignment.section_name),
                ),
            )
            for assignment in batch.assignments
        )
        owned = normalize_model_definitions(
            batch.materials,
            sections,
            assignments,
            batch.steps,
        )
        return self._commit_model_definitions(
            owned,
            reason="definition edit applied",
        )

    def _commit_model_definitions(
        self,
        owned: ModelDefinitions,
        *,
        reason: str,
    ) -> SessionDelta:
        """Validate and commit one already-owned definitions post-state."""

        if self._source_kind == "native":
            if self._canonical_part_state():
                _validate_native_parts_project_inputs(
                    self._parts,
                    tuple(self._named_regions.values()),
                    owned.materials,
                    owned.sections,
                    owned.assignments,
                    owned.steps,
                )
            elif self._geometry_recipe is None:
                if (
                    owned.assignments
                    or analysis_steps_have_native_region_targets(owned.steps)
                ):
                    raise ValueError(
                        "native region assignments and named Step targets "
                        "require a geometry recipe"
                    )
            else:
                validate_native_project_inputs(
                    self._geometry_recipe,
                    self._mesh_settings,
                    tuple(self._named_regions.values()),
                    owned.materials,
                    owned.sections,
                    owned.assignments,
                    owned.steps,
                )

        compiled_model = None
        previous_artifact = self._artifact
        if previous_artifact is not None:
            compiled_model = compile_model_definitions(
                previous_artifact.model,
                owned,
            ).require_model()

        self._materials = owned.materials
        self._sections = owned.sections
        self._assignments = owned.assignments
        self._steps = owned.steps
        self._definitions_explicit = True
        self._drop_computations()
        self._increment_domain_revisions(project=True, model=True)
        if compiled_model is not None and previous_artifact is not None:
            self._artifact = ModelArtifact(
                session_id=self._session_id,
                artifact_id=new_identity("artifact"),
                model_revision=self._model_revision,
                mesh_input_revision=previous_artifact.mesh_input_revision,
                source_kind=previous_artifact.source_kind,
                model=compiled_model,
            )
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _COMPUTATION_INVALIDATIONS
            | (
                frozenset({ArtifactKind.MODEL})
                if previous_artifact is not None
                else frozenset()
            ),
            reason,
        )

    def clear_generated_model(
        self, *, expected_session_revision: int | None = None
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        self._require_native()
        clears_mesh_scopes = _regions_use_mesh_entities(
            self._named_regions.values()
        )
        candidate_steps = (
            _without_geometry_dependent_steps(self._steps)
            if clears_mesh_scopes
            else self._steps
        )
        had_regions = clears_mesh_scopes and bool(self._named_regions)
        had_assignments = clears_mesh_scopes and bool(self._assignments)
        steps_cleared = candidate_steps != self._steps
        if clears_mesh_scopes:
            self._named_regions = {}
            self._assignments = ()
            self._steps = candidate_steps
            self._definitions_explicit = True
        self._drop_model_state()
        self._increment_domain_revisions(
            project=True,
            mesh=clears_mesh_scopes,
            model=True,
        )
        changed = {
            ChangeKind.PROJECT_INPUTS,
            ChangeKind.MODEL,
            ChangeKind.VALIDATIONS,
            ChangeKind.RUNS,
            ChangeKind.DISPLAYED_RESULT,
        }
        effects: set[TransitionEffect] = set()
        if had_regions:
            changed.add(ChangeKind.NAMED_REGIONS)
            effects.add(TransitionEffect.NAMED_REGIONS_CLEARED)
        if had_assignments or steps_cleared:
            changed.add(ChangeKind.DEFINITIONS)
        if had_assignments:
            effects.add(TransitionEffect.ASSIGNMENTS_CLEARED)
        if steps_cleared:
            effects.add(TransitionEffect.STEPS_CLEARED)
        return self._emit(
            changed,
            _MODEL_INVALIDATIONS,
            "generated model cleared",
            effects=effects,
        )

    # ------------------------------------------------------------------
    # Asynchronous model artifacts
    def prepare_import(self, source_path: str | Path) -> ImportTaskSnapshot:
        path = Path(source_path)
        token = self._issue_token(
            "import",
            (("session_revision", self._session_revision),),
        )
        self._task_data[token.task_id] = path
        return ImportTaskSnapshot(token, path)

    def accept_imported_model(
        self, token: TaskToken, model: Any
    ) -> SessionDelta:
        status = self._token_status_for(token, "import")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale imported model")
        owned_model = deepcopy(model)
        definitions = definitions_from_model(owned_model)
        source_path = Path(self._task_data[token.task_id])

        self._session_id = new_identity("session")
        self._clear_content()
        self._is_open = True
        self._source_kind = "imported"
        self._source_path = source_path
        self._materials = definitions.materials
        self._sections = definitions.sections
        self._assignments = definitions.assignments
        self._steps = definitions.steps
        self._definitions_explicit = True
        self._increment_domain_revisions(project=True, mesh=True, model=True)
        self._artifact = self._new_artifact(owned_model, "imported")
        self._saved_project_revision = self._project_revision
        return self._emit(
            {
                ChangeKind.SESSION,
                ChangeKind.SOURCE,
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.DEFINITIONS,
                ChangeKind.MODEL,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
                ChangeKind.SAVED_STATE,
            },
            _ALL_INVALIDATIONS,
            "imported model installed",
        )

    def prepare_mesh_generation(self) -> MeshTaskSnapshot:
        self._require_native()
        if self._canonical_part_state():
            active_parts = tuple(
                part for part in self._parts if not part.suppressed
            )
            if not active_parts:
                raise SessionStateError(
                    "mesh generation requires an unsuppressed Part"
                )
            if (
                _regions_use_mesh_entities(self._named_regions.values())
                and self._artifact is not None
            ):
                raise SessionStateError(
                    "mesh scopes must be cleared before generating a new mesh"
                )
            for part in active_parts:
                if part.mesh_settings is None:
                    raise SessionStateError(
                        f"Part {part.id} requires mesh settings"
                    )
                require_complete_native_mesh_contract(
                    part.geometry_recipe,
                    _localize_part_mesh_settings(
                        part.id,
                        part.mesh_settings,
                    ),
                )
            _validate_native_parts_project_inputs(
                self._parts,
                tuple(self._named_regions.values()),
                self._materials,
                self._sections,
                self._assignments,
                self._steps,
            )
            _validate_suppressed_part_analysis_targets(
                self._parts,
                tuple(self._named_regions.values()),
                self._assignments,
                self._steps,
            )
            token = self._issue_token(
                "mesh",
                (
                    ("mesh_input_revision", self._mesh_input_revision),
                    ("model_revision", self._model_revision),
                ),
            )
            active = active_parts[0]
            return MeshTaskSnapshot(
                token=token,
                model_name=str(self._model_name or "模型-1"),
                # Compatibility projections are intentionally present on the
                # detached task, while generate_fem_model consumes Parts.
                geometry_recipe=deepcopy(active.geometry_recipe),
                mesh_settings=deepcopy(active.mesh_settings),
                parts=deepcopy(self._parts),
                feature_history=tuple(
                    deepcopy(feature)
                    for part in self._parts
                    for feature in part.feature_history
                ),
                named_regions=deepcopy(
                    tuple(self._named_regions.values())
                ),
                material_definitions=deepcopy(self._materials),
                section_definitions=deepcopy(self._sections),
                region_assignments=deepcopy(self._assignments),
                analysis_definitions=deepcopy(self._steps),
            )
        if self._geometry_recipe is None:
            raise SessionStateError("mesh generation requires geometry")
        if (
            _regions_use_mesh_entities(self._named_regions.values())
            and self._artifact is not None
        ):
            raise SessionStateError(
                "mesh scopes must be cleared before generating a new mesh"
            )
        require_complete_native_mesh_contract(
            self._geometry_recipe,
            self._mesh_settings,
        )
        validate_native_project_inputs(
            self._geometry_recipe,
            self._mesh_settings,
            tuple(self._named_regions.values()),
            self._materials,
            self._sections,
            self._assignments,
            self._steps,
        )
        token = self._issue_token(
            "mesh",
            (
                ("mesh_input_revision", self._mesh_input_revision),
                ("model_revision", self._model_revision),
            ),
        )
        return MeshTaskSnapshot(
            token=token,
            model_name=str(self._model_name or "Model-1"),
            geometry_recipe=deepcopy(self._geometry_recipe),
            mesh_settings=deepcopy(self._mesh_settings),
            parts=deepcopy(self._parts),
            feature_history=deepcopy(self._feature_history),
            named_regions=deepcopy(tuple(self._named_regions.values())),
            material_definitions=deepcopy(self._materials),
            section_definitions=deepcopy(self._sections),
            region_assignments=deepcopy(self._assignments),
            analysis_definitions=deepcopy(self._steps),
        )

    def accept_generated_model(
        self, token: TaskToken, model: Any
    ) -> SessionDelta:
        status = self._token_status_for(token, "mesh")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale generated model")
        self._require_native()
        owned_model = deepcopy(model)
        definitions = None
        if self._definitions_explicit:
            current_definitions = ModelDefinitions(
                self._materials,
                self._sections,
                self._assignments,
                self._steps,
            )
            owned_model = compile_model_definitions(
                owned_model,
                current_definitions,
            ).require_model()
        else:
            definitions = definitions_from_model(owned_model)

        self._drop_computations()
        if definitions is not None:
            self._materials = definitions.materials
            self._sections = definitions.sections
            self._assignments = definitions.assignments
            self._steps = definitions.steps
        self._increment_domain_revisions(project=True, model=True)
        self._artifact = self._new_artifact(owned_model, "native")
        self._complete_token(token)
        return self._emit(
            {
                ChangeKind.PROJECT_INPUTS,
                ChangeKind.MODEL,
                ChangeKind.DEFINITIONS,
                ChangeKind.VALIDATIONS,
                ChangeKind.RUNS,
                ChangeKind.DISPLAYED_RESULT,
            },
            _COMPUTATION_INVALIDATIONS,
            "generated model installed",
        )

    # ------------------------------------------------------------------
    # Per-step validation
    def prepare_validation(
        self,
        step_name: str | None = None,
        *,
        detach_model: bool = True,
    ) -> ValidationTaskSnapshot:
        """Prepare validation, optionally deferring the model copy to a worker."""

        if type(detach_model) is not bool:
            raise TypeError("detach_model must be bool")
        artifact = self._require_current_artifact()
        resolved = self._resolve_step_name(step_name)
        token = self._issue_token(
            "validation",
            (("model_revision", self._model_revision),),
            artifact_id=artifact.artifact_id,
            step_name=resolved,
        )
        return ValidationTaskSnapshot(
            token=token,
            model=(
                deepcopy(artifact.model)
                if detach_model
                else artifact.model
            ),
            step_name=resolved,
        )

    def accept_validation(
        self,
        token: TaskToken,
        report: PreflightReport,
    ) -> SessionDelta:
        status = self._token_status_for(token, "validation")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale validation report")
        if not isinstance(report, PreflightReport):
            raise TypeError("validation report must be PreflightReport")
        self._validate_report_provenance(token, report)
        owned_report = deepcopy(report)
        artifact = self._require_current_artifact()
        stamp = ValidationStamp(
            session_id=self._session_id,
            artifact_id=artifact.artifact_id,
            model_revision=self._model_revision,
            step_name=str(token.step_name),
        )
        self._validations[str(token.step_name)] = ValidationRecord(
            stamp=stamp,
            report=owned_report,
        )
        self._complete_token(token)
        return self._emit(
            {ChangeKind.VALIDATIONS},
            frozenset(),
            (
                "validation passed"
                if owned_report.passed
                else "validation failed"
            ),
        )

    def accept_validation_failed(
        self,
        token: TaskToken,
        report: PreflightReport,
    ) -> SessionDelta:
        if not isinstance(report, PreflightReport):
            raise TypeError("validation report must be PreflightReport")
        if report.passed:
            raise ValueError(
                "failed validation callback requires an error diagnostic"
            )
        return self.accept_validation(token, report)

    # ------------------------------------------------------------------
    # Runs and solver results
    def prepare_solve(
        self,
        step_name: str | None = None,
        run_name: str | None = None,
        *,
        expected_session_revision: int | None = None,
    ) -> SolveTaskSnapshot:
        self._check_expected(expected_session_revision)
        artifact = self._require_current_artifact()
        resolved_step = self._resolve_step_name(step_name)
        if not self.can_submit(resolved_step):
            raise SessionStateError(
                f"step {resolved_step!r} does not have a current passing validation"
            )
        resolved_name = (
            self.next_run_name() if run_name is None else str(run_name).strip()
        )
        if not resolved_name:
            raise ValueError("run name must not be empty")
        if self._find_run_by_name(resolved_name) is not None:
            raise ValueError(f"run name already exists: {resolved_name}")
        solve_model = deepcopy(artifact.model)
        solve_steps = tuple(
            step
            for step in getattr(solve_model, "steps", ())
            if str(getattr(step, "name", "")) == resolved_step
        )
        if len(solve_steps) != 1:
            raise SessionStateError(
                "current model artifact must contain exactly one matching "
                f"analysis step: {resolved_step}"
            )
        run_id = new_identity("run")
        result_id = new_identity("result")
        run = AnalysisRun(
            run_id=run_id,
            name=resolved_name,
            step_name=resolved_step,
            artifact_id=artifact.artifact_id,
            model_revision=self._model_revision,
        )
        self._runs[run_id] = run
        delta = self._emit(
            {ChangeKind.RUNS},
            frozenset(),
            "analysis run prepared",
        )
        token = self._issue_token(
            "solve",
            (("model_revision", self._model_revision),),
            artifact_id=artifact.artifact_id,
            step_name=resolved_step,
            run_id=run_id,
            result_id=result_id,
        )
        self._task_data[token.task_id] = _IssuedSolvePayload(
            model=solve_model,
            step=solve_steps[0],
        )
        return SolveTaskSnapshot(
            token=token,
            model=solve_model,
            step_name=resolved_step,
            run_name=resolved_name,
            run_id=run_id,
            result_id=result_id,
            delta=delta,
        )

    def begin_run(self, token: TaskToken) -> SessionDelta:
        status = self._token_status_for(token, "solve")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale run start")
        run = self._runs[str(token.run_id)]
        if run.status is not RunStatus.PENDING:
            return self._rejected(
                TokenStatus.INVALID_STATE,
                "run is not pending",
            )
        self._runs[run.run_id] = replace(
            run,
            status=RunStatus.RUNNING,
            started_at=utc_now(),
        )
        return self._emit(
            {ChangeKind.RUNS}, frozenset(), "analysis run started"
        )

    def accept_run_succeeded(
        self,
        token: TaskToken,
        bundle: SolveResultBundle,
        *,
        timings: Mapping[str, float] | None = None,
    ) -> SessionDelta:
        status = self._token_status_for(token, "solve")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale run result")
        run = self._runs[str(token.run_id)]
        if run.status is not RunStatus.RUNNING:
            return self._rejected(
                TokenStatus.INVALID_STATE,
                "run is not running",
            )
        if type(bundle) is not SolveResultBundle:
            raise TypeError("bundle must be exactly SolveResultBundle")
        issued_payload = self._task_data.get(token.task_id)
        if type(issued_payload) is not _IssuedSolvePayload:
            raise SessionStateError(
                "solve task has no issued model payload"
            )
        if (
            bundle.result.model is not issued_payload.model
            or bundle.result.step is not issued_payload.step
        ):
            raise ValueError(
                "solve result must use the exact model and step issued "
                "for the solve task"
            )
        owned_timings = {
            str(name): float(seconds)
            for name, seconds in (timings or {}).items()
        }
        artifact = self._require_current_artifact()
        result_id = str(token.result_id)
        expected_source = ResultSourceKey(
            result_id=result_id,
            session_id=self._session_id,
            artifact_id=artifact.artifact_id,
            model_revision=self._model_revision,
            step_name=str(token.step_name),
            run_id=run.run_id,
        )
        if bundle.source != expected_source:
            raise ValueError(
                "solve result bundle source must match the reserved result"
            )
        if getattr(bundle.result.step, "name", None) != expected_source.step_name:
            raise ValueError(
                "solve result step must match the reserved result step"
            )
        validate_solve_result_model_identity(
            bundle.result,
            artifact.model,
            expected_source.step_name,
        )
        provenance = ResultProvenance(
            session_id=self._session_id,
            artifact_id=artifact.artifact_id,
            model_revision=self._model_revision,
            step_name=str(token.step_name),
            run_id=run.run_id,
        )
        provider = bundle._provider
        record = ResultRecord(
            result_id=result_id,
            provenance=provenance,
            result=(
                bundle.result
                if provider is None
                else provider._owned_result
            ),
            output_report=bundle.execution_report,
            materialization=bundle.initial_materialization,
            _provider=provider,
        )
        self._results[run.run_id] = record
        if provider is not None:
            object.__setattr__(bundle, "_provider", None)
        self._runs[run.run_id] = replace(
            run,
            status=RunStatus.SUCCEEDED,
            started_at=run.started_at or utc_now(),
            finished_at=utc_now(),
            result_id=result_id,
            error=None,
            timings=owned_timings,
        )
        self._selected_run_id = run.run_id
        self._displayed_result_run_id = run.run_id
        self._complete_token(token)
        return self._emit(
            {
                ChangeKind.RUNS,
                ChangeKind.RESULTS,
                ChangeKind.DISPLAYED_RESULT,
            },
            frozenset(),
            "analysis run succeeded",
        )

    def accept_run_failed(
        self, token: TaskToken, error: Any
    ) -> SessionDelta:
        status = self._token_status_for(token, "solve")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale run failure")
        run = self._runs[str(token.run_id)]
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return self._rejected(
                TokenStatus.INVALID_STATE,
                "run is already terminal",
            )
        self._runs[run.run_id] = replace(
            run,
            status=RunStatus.FAILED,
            started_at=run.started_at or utc_now(),
            finished_at=utc_now(),
            error=str(error),
        )
        self._complete_token(token)
        return self._emit(
            {ChangeKind.RUNS}, frozenset(), "analysis run failed"
        )

    def request_cancel(
        self,
        run_id: str,
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        run = self._runs.get(str(run_id))
        if run is None:
            raise KeyError(f"unknown run id: {run_id}")
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise SessionStateError("only pending or running runs can be cancelled")
        self._runs[run.run_id] = replace(run, cancellation_requested=True)
        return self._emit(
            {ChangeKind.RUNS}, frozenset(), "run cancellation requested"
        )

    def accept_run_cancelled(self, token: TaskToken) -> SessionDelta:
        status = self._token_status_for(token, "solve")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale run cancellation")
        run = self._runs[str(token.run_id)]
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return self._rejected(
                TokenStatus.INVALID_STATE,
                "run is already terminal",
            )
        self._runs[run.run_id] = replace(
            run,
            status=RunStatus.CANCELLED,
            started_at=run.started_at or utc_now(),
            finished_at=utc_now(),
            cancellation_requested=True,
        )
        self._complete_token(token)
        return self._emit(
            {ChangeKind.RUNS}, frozenset(), "analysis run cancelled"
        )

    def select_result(
        self,
        run_id: str,
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        record = self._current_result_record(str(run_id))
        if record is None:
            raise SessionStateError(
                f"run {run_id!r} has no current successful result"
            )
        self._selected_run_id = str(run_id)
        self._displayed_result_run_id = str(run_id)
        return self._emit(
            {ChangeKind.DISPLAYED_RESULT},
            frozenset(),
            "displayed result selected",
        )

    def prepare_result_projection(self, run_id: str) -> ResultTaskSnapshot:
        record = self._current_result_record(str(run_id))
        if record is None:
            raise SessionStateError(
                f"run {run_id!r} has no current successful result"
            )
        token = self._issue_token(
            "result_projection",
            (
                (
                    "materialization_generation",
                    record.materialization.generation,
                ),
                ("model_revision", self._model_revision),
            ),
            artifact_id=record.provenance.artifact_id,
            step_name=record.provenance.step_name,
            run_id=record.provenance.run_id,
            result_id=record.result_id,
        )
        return ResultTaskSnapshot(
            token=token,
            run_id=str(run_id),
            record=detached_result_record(record),
        )

    def accept_result_projection(self, token: TaskToken) -> SessionDelta:
        """Atomically validate and consume a derived-result task token."""

        status = self._token_status_for(token, "result_projection")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale result projection")
        self._complete_token(token)
        return SessionDelta(
            session_revision=self._session_revision,
            reason="result projection accepted",
        )

    def prepare_result_materialization(
        self,
        run_id: str,
        field_keys: Iterable[FieldMaterializationKey],
    ) -> ResultMaterializationTaskSnapshot:
        """Bind lazy field recovery to one exact accepted generation."""

        record = self._current_result_record(str(run_id))
        if record is None:
            raise SessionStateError(
                f"run {run_id!r} has no current successful result"
            )
        try:
            requested = tuple(field_keys)
        except TypeError as error:
            raise TypeError(
                "field_keys must be an iterable of "
                "FieldMaterializationKey values"
            ) from error
        if not requested:
            raise ValueError("field_keys must not be empty")
        if any(type(key) is not FieldMaterializationKey for key in requested):
            raise TypeError(
                "field_keys must contain only FieldMaterializationKey values"
            )
        ordered = tuple(
            sorted(
                set(requested),
                key=field_materialization_sort_key,
            )
        )
        ready_keys = {
            field_data.key for field_data in record.materialization.fields
        }
        expected_patch_keys = tuple(
            key for key in ordered if key not in ready_keys
        )
        generation = record.materialization.generation
        token = self._issue_token(
            "result_materialization",
            (
                ("materialization_generation", generation),
                ("model_revision", self._model_revision),
            ),
            artifact_id=record.provenance.artifact_id,
            step_name=record.provenance.step_name,
            run_id=record.provenance.run_id,
            result_id=record.result_id,
        )
        self._task_data[token.task_id] = _IssuedMaterializationPayload(
            source=record.materialization.source,
            generation=generation,
            expected_patch_keys=expected_patch_keys,
        )
        return ResultMaterializationTaskSnapshot(
            token=token,
            run_id=str(run_id),
            record=detached_result_record(record),
            field_keys=deepcopy(ordered),
        )

    def accept_result_materialization(
        self,
        token: TaskToken,
        patch: ResultMaterializationPatch,
    ) -> SessionDelta:
        """Install one complete patch through a generation compare-and-swap."""

        status = self._token_status_for(token, "result_materialization")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale result materialization")
        if type(patch) is not ResultMaterializationPatch:
            raise TypeError(
                "patch must be exactly ResultMaterializationPatch"
            )
        record = self._current_result_record(token.run_id)
        payload = self._task_data.get(token.task_id)
        if record is None or type(payload) is not _IssuedMaterializationPayload:
            raise SessionStateError(
                "result materialization task has no current issued payload"
            )
        if (
            patch.source != payload.source
            or patch.source != record.materialization.source
        ):
            raise ValueError(
                "materialization patch source must match the target result"
            )
        if record.materialization.generation != payload.generation:
            return self._rejected(
                TokenStatus.STALE_REVISION,
                "stale result materialization generation",
            )
        patch_keys = tuple(field_data.key for field_data in patch.fields)
        if patch_keys != payload.expected_patch_keys:
            raise ValueError(
                "materialization patch fields must exactly match the "
                "requested lazy keys"
            )
        if not patch.fields:
            self._complete_token(token)
            return SessionDelta(
                session_revision=self._session_revision,
                reason="result materialization cache hit",
            )

        updated_record = advance_result_record(record, patch)
        self._results[str(token.run_id)] = updated_record
        for task_id, issued in self._issued_tokens.items():
            if (
                issued.task_kind == "result_materialization"
                and issued.run_id == token.run_id
            ):
                self._task_data.pop(task_id, None)
        self._complete_token(token)
        return self._emit(
            {ChangeKind.RESULTS},
            frozenset(),
            "result materialization advanced",
        )

    def accept_task_failed(
        self, token: TaskToken, error: Any
    ) -> SessionDelta:
        """Consume a current task failure through the shared stale-token gate."""

        status = self.validate_task_token(token)
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale task failure")
        if token.task_kind == "solve":
            return self.accept_run_failed(token, error)
        if token.task_kind == "validation":
            report = internal_error_report(
                str(token.step_name),
                error,
                session_id=token.session_id,
                artifact_id=token.artifact_id,
                model_revision=_token_dependency(
                    token,
                    "model_revision",
                ),
            )
            return self.accept_validation_failed(token, report)
        self._complete_token(token)
        if token.task_kind in {
            "result_materialization",
            "result_projection",
        }:
            return SessionDelta(
                session_revision=self._session_revision,
                reason=f"{token.task_kind} task failed",
            )
        return self._emit(
            {ChangeKind.SESSION},
            frozenset(),
            f"{token.task_kind} task failed",
        )

    def accept_task_cancelled(self, token: TaskToken) -> SessionDelta:
        """Consume a current task cancellation through the shared token gate."""

        status = self.validate_task_token(token)
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale task cancellation")
        if token.task_kind == "solve":
            return self.accept_run_cancelled(token)
        self._complete_token(token)
        if token.task_kind in {
            "result_materialization",
            "result_projection",
        }:
            return SessionDelta(
                session_revision=self._session_revision,
                reason=f"{token.task_kind} task cancelled",
            )
        return self._emit(
            {ChangeKind.SESSION},
            frozenset(),
            f"{token.task_kind} task cancelled",
        )

    # ------------------------------------------------------------------
    # Project snapshot save protocol
    def prepare_project_save(self) -> ProjectSaveSnapshot:
        self._require_native()
        canonical_part_mode = self._canonical_part_state()
        save_parts = (
            _canonical_parts_for_save(
                self._parts,
                self._geometry_recipe,
                self._mesh_settings,
            )
            if canonical_part_mode
            else deepcopy(self._parts)
        )
        project = ProjectSnapshot(
            source_kind="native",
            source_path=self._project_path,
            model_name=str(self._model_name or "Model-1"),
            parts=save_parts,
            geometry_recipe=self._geometry_recipe,
            mesh_settings=self._mesh_settings,
            feature_history=self._feature_history,
            named_regions=tuple(self._named_regions.values()),
            material_definitions=self._materials,
            section_definitions=self._sections,
            region_assignments=self._assignments,
            analysis_definitions=self._steps,
            boolean_reference_undo_records=tuple(
                self._boolean_reference_undo_records[feature_id]
                for feature_id in sorted(
                    self._boolean_reference_undo_records,
                    key=lambda value: (value[:2], int(value[2:])),
                )
            ),
            part_boolean_undo_records=tuple(
                self._part_boolean_undo_records[feature_id]
                for feature_id in sorted(
                    self._part_boolean_undo_records,
                    key=part_boolean_feature_id_sort_key,
                )
            ),
            retired_part_ids=self.retired_part_ids,
            retired_part_boolean_feature_ids=(
                self.retired_part_boolean_feature_ids
            ),
            active_part_id=(
                self._active_part_id if canonical_part_mode else None
            ),
        )
        token = self._issue_token(
            "project_save",
            (("project_revision", self._project_revision),),
        )
        return ProjectSaveSnapshot(token, self._project_revision, project)

    def accept_project_saved(
        self, token: TaskToken, path: str | Path
    ) -> SessionDelta:
        status = self._token_status_for(token, "project_save")
        if status is not TokenStatus.CURRENT:
            return self._rejected(status, "stale project save")
        self._project_path = Path(path)
        revision = dict(token.dependency_revisions)["project_revision"]
        self._saved_project_revision = revision
        self._complete_token(token)
        return self._emit(
            {ChangeKind.SAVED_STATE},
            frozenset(),
            "project saved",
        )

    # ------------------------------------------------------------------
    # Queries
    def snapshot(self) -> SessionSnapshot:
        artifact = None
        if self._artifact is not None:
            artifact = ModelArtifact(
                session_id=self._artifact.session_id,
                artifact_id=self._artifact.artifact_id,
                model_revision=self._artifact.model_revision,
                mesh_input_revision=self._artifact.mesh_input_revision,
                source_kind=self._artifact.source_kind,
                model=deepcopy(self._artifact.model),
            )
        displayed = self._current_result_record(
            self._displayed_result_run_id
        )
        return SessionSnapshot(
            is_open=self._is_open,
            session_id=self._session_id,
            session_revision=self._session_revision,
            project_revision=self._project_revision,
            mesh_input_revision=self._mesh_input_revision,
            model_revision=self._model_revision,
            saved_project_revision=self._saved_project_revision,
            source_kind=self._source_kind,
            source_path=self._source_path,
            project_path=self._project_path,
            model_name=self._model_name,
            geometry_recipe=deepcopy(self._geometry_recipe),
            mesh_settings=deepcopy(self._mesh_settings),
            parts=deepcopy(self._parts),
            active_part_id=self._active_part_id,
            part_revisions=MappingProxyType(
                deepcopy(self._part_revisions)
            ),
            feature_history=deepcopy(self._feature_history),
            named_regions=MappingProxyType(deepcopy(self._named_regions)),
            materials=deepcopy(self._materials),
            sections=deepcopy(self._sections),
            assignments=deepcopy(self._assignments),
            steps=deepcopy(self._effective_steps()),
            artifact=artifact,
            validations=MappingProxyType(deepcopy(self._validations)),
            runs=deepcopy(tuple(self._runs.values())),
            selected_run_id=self._selected_run_id,
            displayed_result_run_id=self._displayed_result_run_id,
            displayed_result=(
                None
                if displayed is None
                else detached_result_record(displayed)
            ),
            dirty=self.dirty,
            can_save=self.can_save,
        )

    def can_check(self, step_name: str | None = None) -> bool:
        try:
            self._require_current_artifact()
            self._resolve_step_name(step_name)
        except (SessionStateError, KeyError, ValueError):
            return False
        return True

    def can_submit(self, step_name: str | None = None) -> bool:
        if not self.can_check(step_name):
            return False
        try:
            resolved = self._resolve_step_name(step_name)
        except (SessionStateError, KeyError, ValueError):
            return False
        record = self._validation_record(resolved)
        return record is not None and record.passed

    def validation_for(
        self, step_name: str
    ) -> ValidationRecord | None:
        record = self._validation_record(str(step_name))
        return None if record is None else deepcopy(record)

    def current_result(self) -> ResultRecord | None:
        record = self._current_result_record(self._displayed_result_run_id)
        return (
            None
            if record is None
            else detached_result_record(record)
        )

    def current_result_identity(
        self,
    ) -> tuple[ResultSourceKey, int] | None:
        """Return immutable displayed-result identity without detaching data."""

        record = self._current_result_record(
            self._displayed_result_run_id
        )
        if record is None:
            return None
        return (
            record.materialization.source,
            record.materialization.generation,
        )

    def current_result_provider(self) -> ResultProvider | None:
        """Return the immutable provider projection for the displayed result."""

        record = self._current_result_record(
            self._displayed_result_run_id
        )
        return (
            None
            if record is None
            else result_record_provider(record)
        )

    def find_run(self, run_id_or_name: str | None) -> AnalysisRun | None:
        normalized = str(run_id_or_name or "").strip()
        run = self._runs.get(normalized)
        if run is None:
            run = self._find_run_by_name(normalized)
        return None if run is None else deepcopy(run)

    def next_run_name(self) -> str:
        number = 1
        while self._find_run_by_name(f"Job-{number}") is not None:
            number += 1
        return f"Job-{number}"

    def latest_resubmittable_run(self) -> AnalysisRun | None:
        for run in reversed(tuple(self._runs.values())):
            if run.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return deepcopy(run)
        return None

    def runnable_step_names(self) -> tuple[str, ...]:
        names = tuple(
            str(step.name)
            for step in self._effective_steps()
            if str(step.name).casefold() != "initial"
        )
        return names or tuple(
            str(step.name) for step in self._effective_steps()
        )

    def default_step_name(self) -> str | None:
        names = self.runnable_step_names()
        return names[0] if names else None

    def validate_task_token(self, token: TaskToken) -> TokenStatus:
        if not isinstance(token, TaskToken):
            return TokenStatus.UNKNOWN_TASK
        if token.session_id != self._session_id:
            return TokenStatus.STALE_SESSION
        issued = self._issued_tokens.get(token.task_id)
        if issued is None:
            return TokenStatus.UNKNOWN_TASK
        if token.task_kind != issued.task_kind:
            return TokenStatus.WRONG_KIND
        if token.artifact_id != issued.artifact_id:
            return TokenStatus.STALE_ARTIFACT
        if token.step_name != issued.step_name:
            return TokenStatus.STALE_STEP
        if token.run_id != issued.run_id:
            return TokenStatus.STALE_RUN
        if token.result_id != issued.result_id:
            return TokenStatus.STALE_RESULT
        if token.dependency_revisions != issued.dependency_revisions:
            return TokenStatus.STALE_REVISION
        if token != issued:
            return TokenStatus.UNKNOWN_TASK
        if token.task_id in self._completed_task_ids:
            return TokenStatus.ALREADY_COMPLETED
        for name, revision in token.dependency_revisions:
            if name == "materialization_generation":
                record = self._current_result_record(token.run_id)
                if (
                    record is None
                    or record.result_id != token.result_id
                    or record.materialization.generation != revision
                ):
                    return TokenStatus.STALE_REVISION
                continue
            if self._revision_value(name) != revision:
                return TokenStatus.STALE_REVISION
        if token.artifact_id is not None:
            if (
                self._artifact is None
                or self._artifact.artifact_id != token.artifact_id
                or self._artifact.session_id != token.session_id
                or self._artifact.model_revision != self._model_revision
            ):
                return TokenStatus.STALE_ARTIFACT
        if token.step_name is not None and not self._step_exists(token.step_name):
            return TokenStatus.STALE_STEP
        if token.run_id is not None:
            run = self._runs.get(token.run_id)
            if (
                run is None
                or run.artifact_id != token.artifact_id
                or run.model_revision != self._model_revision
                or run.step_name != token.step_name
            ):
                return TokenStatus.STALE_RUN
        if token.result_id is not None and token.task_kind != "solve":
            record = self._current_result_record(token.run_id)
            if record is None or record.result_id != token.result_id:
                return TokenStatus.STALE_RESULT
        return TokenStatus.CURRENT

    # ------------------------------------------------------------------
    # Internals
    def _canonical_part_state(self) -> bool:
        return bool(self._parts) and all(
            part.geometry_recipe is not None for part in self._parts
        )

    def _require_part(self, part_id: str) -> NativePart:
        normalized = normalize_part_id(part_id)
        for part in self._parts:
            if part.id == normalized:
                return part
        raise KeyError(normalized)

    def _check_part_revision(
        self,
        part_id: str,
        expected: int | None,
    ) -> None:
        normalized = normalize_part_id(part_id)
        if normalized not in self._part_revisions:
            raise KeyError(normalized)
        if expected is None:
            return
        actual = self._part_revisions[normalized]
        if int(expected) != actual:
            raise PartRevisionConflictError(
                normalized,
                int(expected),
                actual,
            )

    def _part_has_active_descendant(self, part_id: str) -> bool:
        normalized = normalize_part_id(part_id)
        return any(
            part.provenance is not None
            and normalized in part.provenance.source_part_ids
            for part in self._parts
        )

    def _require_editable_part(self, part_id: str) -> NativePart:
        part = self._require_part(part_id)
        if part.suppressed:
            raise SessionStateError(f"Part {part.id} is suppressed")
        if self._part_has_active_descendant(part.id):
            raise SessionStateError(
                f"Part {part.id} is locked by an active Boolean result"
            )
        return part

    def _replace_part(self, replacement: NativePart) -> None:
        if type(replacement) is not NativePart:
            raise TypeError("replacement must be a NativePart")
        if replacement.id not in {part.id for part in self._parts}:
            raise KeyError(replacement.id)
        self._parts = tuple(
            replacement if part.id == replacement.id else part
            for part in self._parts
        )

    def _sync_active_part_projection(self) -> None:
        """Keep read-only v1-v6 accessors projected from the active Part."""

        active = None
        if self._active_part_id is not None:
            active = next(
                (
                    part
                    for part in self._parts
                    if part.id == self._active_part_id
                ),
                None,
            )
        if active is None:
            self._geometry_recipe = None
            self._mesh_settings = None
            self._feature_history = ()
            return
        self._geometry_recipe = deepcopy(active.geometry_recipe)
        self._mesh_settings = deepcopy(active.mesh_settings)
        self._feature_history = deepcopy(active.feature_history)

    def _clear_content(self) -> None:
        self._is_open = False
        self._source_kind: str | None = None
        self._source_path: Path | None = None
        self._project_path: Path | None = None
        self._model_name: str | None = None
        self._geometry_recipe: Any | None = None
        self._retired_body_ids: set[str] = set()
        self._retired_boolean_feature_ids: set[str] = set()
        self._boolean_reference_undo_records: dict[
            str,
            BooleanReferenceUndoRecord,
        ] = {}
        self._mesh_settings: Any | None = None
        self._parts: tuple[NativePart, ...] = ()
        self._active_part_id: str | None = None
        self._part_revisions: dict[str, int] = {}
        self._retired_part_ids: set[str] = set()
        self._retired_part_boolean_feature_ids: set[str] = set()
        self._part_boolean_undo_records: dict[
            str,
            PartBooleanUndoRecord,
        ] = {}
        self._part_extrusion_undo_records: dict[
            str,
            _PartExtrusionUndoRecord,
        ] = {}
        self._feature_history: tuple[FeatureRecord, ...] = ()
        self._named_regions: dict[str, NamedRegion] = {}
        self._materials: tuple[Any, ...] = ()
        self._sections: tuple[SectionDefinition, ...] = ()
        self._assignments: tuple[RegionAssignment, ...] = ()
        self._steps: tuple[Any, ...] = ()
        self._definitions_explicit = False
        self._artifact: ModelArtifact | None = None
        self._validations: dict[str, ValidationRecord] = {}
        self._runs: dict[str, AnalysisRun] = {}
        self._results: dict[str, ResultRecord] = {}
        self._selected_run_id: str | None = None
        self._displayed_result_run_id: str | None = None
        self._issued_tokens: dict[str, TaskToken] = {}
        self._completed_task_ids: set[str] = set()
        self._task_data: dict[str, Any] = {}

    def _check_expected(self, expected: int | None) -> None:
        if expected is None:
            return
        if int(expected) != self._session_revision:
            raise RevisionConflictError(int(expected), self._session_revision)

    def _require_open(self) -> None:
        if not self._is_open:
            raise SessionStateError("no project is open")

    def _require_native(self) -> None:
        self._require_open()
        if self._source_kind != "native":
            raise SessionStateError("operation requires a native project")

    def _increment_domain_revisions(
        self,
        *,
        project: bool = False,
        mesh: bool = False,
        model: bool = False,
    ) -> None:
        if project:
            self._project_revision += 1
        if mesh:
            self._mesh_input_revision += 1
        if model:
            self._model_revision += 1

    def _emit(
        self,
        changed: Iterable[ChangeKind],
        invalidated: Iterable[ArtifactKind],
        reason: str,
        *,
        effects: Iterable[TransitionEffect] = (),
    ) -> SessionDelta:
        self._session_revision += 1
        return SessionDelta(
            session_revision=self._session_revision,
            changed=frozenset(changed),
            invalidated=frozenset(invalidated),
            reason=reason,
            effects=frozenset(effects),
        )

    def _rejected(
        self, status: TokenStatus, reason: str
    ) -> SessionDelta:
        return SessionDelta(
            session_revision=self._session_revision,
            changed=frozenset(),
            invalidated=frozenset(),
            reason=reason,
            accepted=False,
            token_status=status,
        )

    def _drop_computations(self) -> None:
        for task_id, token in self._issued_tokens.items():
            if token.task_kind in {
                "result_materialization",
                "solve",
            }:
                self._task_data.pop(task_id, None)
        self._validations.clear()
        self._runs.clear()
        self._results.clear()
        self._selected_run_id = None
        self._displayed_result_run_id = None

    def _drop_model_state(self) -> None:
        self._artifact = None
        self._drop_computations()

    def _new_artifact(self, model: Any, source_kind: str) -> ModelArtifact:
        return ModelArtifact(
            session_id=self._session_id,
            artifact_id=new_identity("artifact"),
            model_revision=self._model_revision,
            mesh_input_revision=(
                self._mesh_input_revision if source_kind == "native" else None
            ),
            source_kind=source_kind,
            model=model,
        )

    def _require_current_artifact(self) -> ModelArtifact:
        artifact = self._artifact
        if (
            artifact is None
            or artifact.session_id != self._session_id
            or artifact.model_revision != self._model_revision
        ):
            raise SessionStateError("no current model artifact")
        return artifact

    def _effective_steps(self) -> tuple[Any, ...]:
        if self._steps:
            return self._steps
        if self._artifact is not None:
            return tuple(getattr(self._artifact.model, "steps", ()))
        return ()

    def _step_exists(self, step_name: str) -> bool:
        return any(
            str(step.name) == str(step_name) for step in self._effective_steps()
        )

    def _resolve_step_name(self, step_name: str | None) -> str:
        resolved = (
            self.default_step_name()
            if step_name is None
            else str(step_name).strip()
        )
        if not resolved:
            raise SessionStateError("no analysis step is available")
        if not self._step_exists(resolved):
            raise KeyError(f"unknown analysis step: {resolved}")
        return resolved

    def _validation_record(
        self, step_name: str
    ) -> ValidationRecord | None:
        record = self._validations.get(str(step_name))
        if record is None or self._artifact is None:
            return None
        stamp = record.stamp
        if (
            stamp.session_id != self._session_id
            or stamp.artifact_id != self._artifact.artifact_id
            or stamp.model_revision != self._model_revision
            or stamp.step_name != str(step_name)
        ):
            return None
        return record

    def _current_result_record(
        self, run_id: str | None
    ) -> ResultRecord | None:
        if run_id is None or self._artifact is None:
            return None
        run = self._runs.get(run_id)
        record = self._results.get(run_id)
        if (
            run is None
            or record is None
            or run.status is not RunStatus.SUCCEEDED
            or run.result_id != record.result_id
        ):
            return None
        provenance = record.provenance
        if (
            provenance.session_id != self._session_id
            or provenance.artifact_id != self._artifact.artifact_id
            or provenance.model_revision != self._model_revision
            or provenance.step_name != run.step_name
            or provenance.run_id != run.run_id
        ):
            return None
        return record

    def _find_run_by_name(self, name: str) -> AnalysisRun | None:
        normalized = str(name).strip().casefold()
        return next(
            (
                run
                for run in self._runs.values()
                if run.name.casefold() == normalized
            ),
            None,
        )

    def _issue_token(
        self,
        task_kind: str,
        dependencies: Iterable[tuple[str, int]],
        *,
        artifact_id: str | None = None,
        step_name: str | None = None,
        run_id: str | None = None,
        result_id: str | None = None,
    ) -> TaskToken:
        token = TaskToken(
            session_id=self._session_id,
            task_id=new_identity("task"),
            task_kind=task_kind,
            dependency_revisions=tuple(dependencies),
            artifact_id=artifact_id,
            step_name=step_name,
            run_id=run_id,
            result_id=result_id,
        )
        self._issued_tokens[token.task_id] = token
        return token

    def _complete_token(self, token: TaskToken) -> None:
        self._completed_task_ids.add(token.task_id)
        self._task_data.pop(token.task_id, None)

    def _token_status_for(
        self, token: TaskToken, expected_kind: str
    ) -> TokenStatus:
        status = self.validate_task_token(token)
        if (
            status is TokenStatus.CURRENT
            and token.task_kind != expected_kind
        ):
            return TokenStatus.WRONG_KIND
        return status

    def _validate_report_provenance(
        self,
        token: TaskToken,
        report: PreflightReport,
    ) -> None:
        """Require a report to identify the exact validation snapshot."""

        expected_step = str(token.step_name)
        expected_revision = _token_dependency(
            token,
            "model_revision",
        )
        mismatches: list[str] = []
        if report.step_name != expected_step:
            mismatches.append(
                f"step {report.step_name!r} != {expected_step!r}"
            )
        if report.session_id != token.session_id:
            mismatches.append("session_id")
        if report.artifact_id != token.artifact_id:
            mismatches.append("artifact_id")
        if report.model_revision != expected_revision:
            mismatches.append("model_revision")
        if mismatches:
            raise ValueError(
                "preflight report provenance does not match validation token: "
                + ", ".join(mismatches)
            )

    def _revision_value(self, name: str) -> int | None:
        return {
            "session_revision": self._session_revision,
            "project_revision": self._project_revision,
            "mesh_input_revision": self._mesh_input_revision,
            "model_revision": self._model_revision,
            "saved_project_revision": self._saved_project_revision,
        }.get(str(name))

def _canonical_source_kind(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if normalized == "inp":
        normalized = "imported"
    if normalized not in {"native", "imported"}:
        raise ValueError(f"unsupported project source kind: {value!r}")
    return normalized


def _regions_by_name(
    regions: Mapping[str, NamedRegion] | Iterable[NamedRegion],
) -> dict[str, NamedRegion]:
    owned = deepcopy(_mapping_values(regions))
    result: dict[str, NamedRegion] = {}
    folded: dict[str, str] = {}
    for region in owned:
        name = str(region.name).strip()
        if not name:
            raise ValueError("named region name must not be empty")
        key = name.casefold()
        if key in folded:
            raise ValueError(
                "named region names must be unique ignoring case: "
                f"{folded[key]!r} and {name!r}"
            )
        folded[key] = name
        result[name] = region
    return result


def _validate_edit_ledger(
    before: Iterable[Any],
    after: Iterable[Any],
    renames: Iterable[RenameIntent],
    deletes: Iterable[DeleteIntent],
    *,
    label: str,
) -> dict[str, str]:
    """Validate explicit identity intents against a detached full post-state."""

    before_names = tuple(_edit_value_name(value, label) for value in before)
    after_names = tuple(_edit_value_name(value, label) for value in after)
    _require_unique_edit_names(before_names, f"existing {label}")
    _require_unique_edit_names(after_names, f"replacement {label}")
    before_set = set(before_names)
    after_set = set(after_names)
    before_by_fold = {name.casefold(): name for name in before_names}

    rename_values = tuple(renames)
    delete_values = tuple(deletes)
    if any(type(intent) is not RenameIntent for intent in rename_values):
        raise TypeError("renames must contain only RenameIntent values")
    if any(type(intent) is not DeleteIntent for intent in delete_values):
        raise TypeError("deletes must contain only DeleteIntent values")

    rename_old_names = tuple(intent.old_name for intent in rename_values)
    rename_new_names = tuple(intent.new_name for intent in rename_values)
    delete_names = tuple(intent.name for intent in delete_values)
    _require_unique_edit_names(rename_old_names, f"{label} rename source")
    _require_unique_edit_names(rename_new_names, f"{label} rename target")
    _require_unique_edit_names(delete_names, f"{label} delete target")

    overlap = set(rename_old_names) & set(delete_names)
    if overlap:
        raise ValueError(
            f"{label} rename and delete intents overlap: "
            + ", ".join(sorted(overlap))
        )

    rename_map: dict[str, str] = {}
    for intent in rename_values:
        old_name = intent.old_name
        new_name = intent.new_name
        if old_name not in before_set:
            raise KeyError(f"unknown {label} to rename: {old_name}")
        if old_name in after_set:
            raise ValueError(
                f"renamed {label} {old_name!r} must be absent from replacement"
            )
        if new_name not in after_set:
            raise ValueError(
                f"renamed {label} {new_name!r} is absent from replacement"
            )
        occupied = before_by_fold.get(new_name.casefold())
        if occupied is not None and occupied != old_name:
            raise ValueError(
                f"{label} rename target {new_name!r} collides with "
                f"existing {occupied!r}"
            )
        rename_map[old_name] = new_name

    for name in delete_names:
        if name not in before_set:
            raise KeyError(f"unknown {label} to delete: {name}")
        if name in after_set:
            raise ValueError(
                f"deleted {label} {name!r} must be absent from replacement"
            )

    removed = before_set - after_set
    explained = set(rename_old_names) | set(delete_names)
    missing = sorted(removed - explained)
    if missing:
        raise ValueError(
            f"removed {label} names require an explicit rename or delete intent: "
            + ", ".join(missing)
        )
    unexpected = sorted(explained - removed)
    if unexpected:
        raise ValueError(
            f"{label} intents do not describe removed names: "
            + ", ".join(unexpected)
        )
    return rename_map


def _edit_value_name(value: Any, label: str) -> str:
    name = getattr(value, "name", None)
    if type(name) is not str or not name.strip():
        raise ValueError(f"{label} name must be a non-empty string")
    return name.strip()


def _require_unique_edit_names(
    names: Iterable[str],
    label: str,
) -> None:
    seen: dict[str, str] = {}
    for name in names:
        folded = name.casefold()
        if folded in seen:
            raise ValueError(
                f"{label} names must be unique ignoring case: "
                f"{seen[folded]!r} and {name!r}"
            )
        seen[folded] = name


def _token_dependency(token: TaskToken, name: str) -> int | None:
    for dependency_name, value in token.dependency_revisions:
        if dependency_name == name:
            return int(value)
    return None


def _rename_step_region_references(
    steps: tuple[Any, ...],
    renames: Mapping[str, str],
) -> tuple[Any, ...]:
    if not renames:
        return deepcopy(steps)
    renamed_steps: list[Any] = []
    for step in deepcopy(steps):
        renamed_steps.append(
            replace(
                step,
                boundaries=tuple(
                    _rename_dataclass_field(item, "target", renames)
                    for item in getattr(step, "boundaries", ())
                ),
                cloads=tuple(
                    _rename_dataclass_field(item, "target", renames)
                    for item in getattr(step, "cloads", ())
                ),
                edge_loads=tuple(
                    _rename_dataclass_field(item, "edge", renames)
                    for item in getattr(step, "edge_loads", ())
                ),
                surface_loads=tuple(
                    _rename_dataclass_field(item, "surface", renames)
                    for item in getattr(step, "surface_loads", ())
                ),
                line_loads=tuple(
                    _rename_dataclass_field(item, "target", renames)
                    for item in getattr(step, "line_loads", ())
                ),
                body_loads=tuple(
                    _rename_dataclass_field(item, "target", renames)
                    for item in getattr(step, "body_loads", ())
                ),
                gravity_loads=tuple(
                    _rename_dataclass_field(item, "target", renames)
                    for item in getattr(step, "gravity_loads", ())
                ),
            )
        )
    return tuple(renamed_steps)


def _rename_dataclass_field(
    value: Any,
    field_name: str,
    renames: Mapping[str, str],
) -> Any:
    current = getattr(value, field_name, None)
    if not isinstance(current, str) or current not in renames:
        return value
    return replace(value, **{field_name: renames[current]})


def _region_references(
    assignments: tuple[RegionAssignment, ...],
    steps: tuple[Any, ...],
) -> set[str]:
    references = {
        str(assignment.region_name) for assignment in assignments
    }
    for step in steps:
        for collection_name, field_name in (
            ("boundaries", "target"),
            ("cloads", "target"),
            ("edge_loads", "edge"),
            ("surface_loads", "surface"),
            ("line_loads", "target"),
            ("body_loads", "target"),
            ("gravity_loads", "target"),
        ):
            for value in getattr(step, collection_name, ()):
                target = getattr(value, field_name, None)
                if isinstance(target, str):
                    references.add(target)
    return references


def _rewrite_named_region(
    region: NamedRegion,
    rewrites: Mapping[str, tuple[str, ...]],
) -> NamedRegion | None:
    rewritten: list[LogicalEntityRef] = []
    for reference in region.references:
        if type(reference) is not LogicalEntityRef:
            return None
        targets = rewrites.get(reference.logical_id, ())
        if not targets:
            return None
        rewritten.extend(LogicalEntityRef(target) for target in targets)
    unique = tuple(
        dict.fromkeys(
            sorted(
                rewritten,
                key=lambda item: item.logical_id,
            )
        )
    )
    return replace(region, references=unique) if unique else None


def _canonical_parts_for_save(
    parts: tuple[NativePart, ...],
    legacy_recipe: object | None,
    legacy_mesh_settings: MeshSettings | None,
) -> tuple[NativePart, ...]:
    """Materialize compatibility state into canonical Part ownership."""

    if parts and all(part.geometry_recipe is not None for part in parts):
        return validate_native_parts(parts)
    if legacy_recipe is None:
        return ()
    if isinstance(legacy_recipe, MultiBodyGeometry):
        owned = tuple(
            NativePart(
                id=f"P{int(body.id[1:])}",
                name=body.name,
                geometry_recipe=body.recipe,
                mesh_settings=_namespace_part_mesh_settings(
                    f"P{int(body.id[1:])}",
                    legacy_mesh_settings,
                ),
            )
            for body in legacy_recipe.bodies
        )
        return validate_native_parts(owned)
    metadata = parts[0] if parts else NativePart()
    return (
        NativePart(
            id=metadata.id,
            name=metadata.name,
            geometry_recipe=legacy_recipe,
            mesh_settings=_namespace_part_mesh_settings(
                metadata.id,
                legacy_mesh_settings,
            ),
            body_name=metadata.body_name,
        ),
    )


def _namespace_part_mesh_settings(
    part_id: str,
    settings: MeshSettings | None,
) -> MeshSettings | None:
    if settings is None:
        return None
    if type(settings) is not MeshSettings:
        raise TypeError("Part mesh settings must be MeshSettings or None")
    normalized = normalize_part_id(part_id)
    controls = []
    for control in settings.local_controls:
        owner = part_id_from_logical_id(control.target.logical_id)
        if owner is None:
            target = LogicalEntityRef(
                namespace_part_logical_id(
                    normalized,
                    control.target.logical_id,
                )
            )
        elif owner == normalized:
            target = control.target
        else:
            raise ValueError(
                f"Part {normalized} mesh settings contain target owned by "
                f"{owner}"
            )
        controls.append(replace(control, target=target))
    return replace(deepcopy(settings), local_controls=tuple(controls))


def _localize_part_mesh_settings(
    part_id: str,
    settings: MeshSettings | None,
) -> MeshSettings | None:
    if settings is None:
        return None
    if type(settings) is not MeshSettings:
        raise TypeError("Part mesh settings must be MeshSettings or None")
    normalized = normalize_part_id(part_id)
    return replace(
        deepcopy(settings),
        local_controls=tuple(
            replace(
                control,
                target=LogicalEntityRef(
                    strip_part_logical_id(
                        normalized,
                        control.target.logical_id,
                    )
                ),
            )
            for control in settings.local_controls
        ),
    )


def _requested_local_part_mesh_settings(
    part_id: str,
    settings: MeshSettings,
) -> MeshSettings:
    owners = {
        part_id_from_logical_id(control.target.logical_id)
        for control in settings.local_controls
    }
    if owners == {None} or not owners:
        return deepcopy(settings)
    if owners == {normalize_part_id(part_id)}:
        localized = _localize_part_mesh_settings(part_id, settings)
        if localized is None:  # pragma: no cover - exact input type above
            raise TypeError("settings unexpectedly localized to None")
        return localized
    raise ValueError("Part mesh settings mix local or foreign namespaces")


def _validate_native_part_inputs(
    part: NativePart,
    *,
    authenticate_geometry: bool = False,
) -> None:
    if type(part) is not NativePart or part.geometry_recipe is None:
        raise ValueError("canonical Part must own geometry")
    settings = _localize_part_mesh_settings(part.id, part.mesh_settings)
    if (
        settings is not None
        and (
            geometry_dimension(part.geometry_recipe) == 1
            or settings.cell_shape == "line"
        )
    ):
        _validate_explicit_mesh_settings(settings, part.geometry_recipe)
    validate_native_project_inputs(
        part.geometry_recipe,
        settings,
        (),
        (),
        (),
        (),
        (),
    )
    if authenticate_geometry and part.dimension == 3:
        _authenticate_native_part_single_solid(part)


def _authenticate_native_part_single_solid(part: NativePart) -> None:
    """Require OCC replay to produce exactly one three-dimensional domain."""

    from fem.geometry import model as geometry_model

    from .recipe_compiler import compile_recipe

    if _contains_unproven_boolean(part.geometry_recipe):
        raise ValueError(
            f"Part {part.id} 的三维布尔几何缺少完整拓扑证明"
        )
    try:
        with geometry_model(
            f"native-part-{part.id}-single-solid-authentication",
            dimension=3,
        ) as cad:
            compiled = compile_recipe(cad, part.geometry_recipe)
            domains = tuple(compiled.domain)
    except Exception as error:
        raise ValueError(
            f"Part {part.id} 的三维几何无法通过 OCC 单实体认证：{error}"
        ) from error
    if len(domains) != 1 or domains[0].dimension != 3:
        raise ValueError(
            f"Part {part.id} 的三维几何必须精确生成一个实体；"
            f"当前生成 {len(domains)} 个"
        )


def _recipe_contains_geometry_state(current: Any, expected: Any) -> bool:
    """Return whether *current* is the expected recipe plus later local features."""

    if current == expected:
        return True
    if isinstance(current, (MovedGeometry, RotatedGeometry)):
        return _recipe_contains_geometry_state(current.base, expected)
    return False


def _contains_unproven_boolean(recipe: Any) -> bool:
    if isinstance(recipe, BooleanGeometry):
        context = (
            recipe.body_context
            or recipe.planar_context
            or recipe.part_context
        )
        return (
            context is None
            or not context.proven
            or _contains_unproven_boolean(recipe.object_geometry)
            or _contains_unproven_boolean(recipe.tool_geometry)
        )
    if isinstance(
        recipe,
        (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry),
    ):
        return _contains_unproven_boolean(recipe.base)
    return False


def _validate_native_parts_project_inputs(
    parts: tuple[NativePart, ...],
    regions: tuple[NamedRegion, ...],
    materials: tuple[Any, ...],
    sections: tuple[SectionDefinition, ...],
    assignments: tuple[RegionAssignment, ...],
    steps: tuple[Any, ...],
    *,
    authenticate_geometry: bool = False,
) -> None:
    """Validate aggregate authoring without projecting it onto one Part."""

    for part in parts:
        _validate_native_part_inputs(
            part,
            authenticate_geometry=authenticate_geometry,
        )
    _validate_part_namespaced_references(parts, regions)
    localized_by_part = {
        part.id: _localized_regions_for_part(part.id, parts, regions)
        for part in parts
    }
    builtins_by_part = {
        part.id: {
            descriptor.name
            for descriptor in describe_native_regions(
                part.geometry_recipe,
                (),
                mesh_settings=_localize_part_mesh_settings(
                    part.id,
                    part.mesh_settings,
                ),
            )
        }
        for part in parts
    }
    known_region_names = {
        region.name for region in regions
    } | {
        name
        for names in builtins_by_part.values()
        for name in names
    }
    unknown_targets = sorted(
        _region_references(assignments, steps) - known_region_names
    )
    if unknown_targets:
        raise ValueError(
            "multi-Part definitions reference unknown named regions: "
            + ", ".join(unknown_targets)
        )
    for part in parts:
        part_names = builtins_by_part[part.id] | {
            region.name for region in localized_by_part[part.id]
        }
        validate_native_project_inputs(
            part.geometry_recipe,
            _localize_part_mesh_settings(part.id, part.mesh_settings),
            localized_by_part[part.id],
            materials,
            sections,
            tuple(
                assignment
                for assignment in assignments
                if assignment.region_name in part_names
            ),
            tuple(
                _step_projected_to_regions(step, part_names)
                for step in steps
            ),
        )


def _localized_regions_for_part(
    part_id: str,
    parts: tuple[NativePart, ...],
    regions: tuple[NamedRegion, ...],
) -> tuple[NamedRegion, ...]:
    localized: list[NamedRegion] = []
    for region in regions:
        references: list[LogicalEntityRef | MeshEntityRef] = []
        for reference in region.references:
            if type(reference) is LogicalEntityRef:
                if part_id_from_logical_id(reference.logical_id) == part_id:
                    references.append(
                        LogicalEntityRef(
                            strip_part_logical_id(
                                part_id,
                                reference.logical_id,
                            )
                        )
                    )
            elif (
                reference.part_id == part_id
                or (reference.part_id is None and len(parts) == 1)
            ):
                references.append(replace(reference, part_id=None))
        if references:
            localized.append(NamedRegion(region.name, tuple(references)))
    return tuple(localized)


def _attach_mesh_reference_owners(
    regions: tuple[NamedRegion, ...],
    parts: tuple[NativePart, ...],
    model: Any,
) -> tuple[NamedRegion, ...]:
    metadata = getattr(model, "metadata", None)
    ownership = (
        metadata.get(NATIVE_PART_OWNERSHIP_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(ownership, Mapping):
        if any(
            type(reference) is MeshEntityRef
            for region in regions
            for reference in region.references
        ):
            raise ValueError("generated multi-Part mesh lacks ownership metadata")
        return regions
    known_ids = {part.id for part in parts}
    result: list[NamedRegion] = []
    for region in regions:
        references: list[LogicalEntityRef | MeshEntityRef] = []
        for reference in region.references:
            if type(reference) is not MeshEntityRef:
                references.append(reference)
                continue
            if reference.part_id is not None:
                if reference.part_id not in known_ids:
                    raise ValueError(
                        f"mesh reference owner {reference.part_id!r} is unknown"
                    )
                references.append(reference)
                continue
            identity = (
                int(reference.node_id)
                if reference.kind == "node"
                else int(reference.element_id)
            )
            field = "node_ids" if reference.kind == "node" else "element_ids"
            candidates = tuple(
                part_id
                for part_id, row in ownership.items()
                if part_id in known_ids
                and isinstance(row, Mapping)
                and identity
                in {int(value) for value in row.get(field, ())}
            )
            if len(candidates) != 1:
                raise ValueError(
                    "mesh reference cannot be assigned to exactly one Part"
                )
            references.append(replace(reference, part_id=candidates[0]))
        result.append(NamedRegion(region.name, tuple(references)))
    return tuple(result)


def _step_projected_to_regions(step: Any, names: set[str]) -> Any:
    projected = deepcopy(step)
    projected.boundaries = tuple(
        value for value in step.boundaries if value.target in names
    )
    projected.cloads = tuple(
        value for value in step.cloads if value.target in names
    )
    projected.edge_loads = tuple(
        value for value in step.edge_loads if value.edge in names
    )
    projected.surface_loads = tuple(
        value for value in step.surface_loads if value.surface in names
    )
    projected.line_loads = tuple(
        value for value in step.line_loads if value.target in names
    )
    projected.body_loads = tuple(
        value for value in step.body_loads if value.target in names
    )
    projected.gravity_loads = tuple(
        value
        for value in step.gravity_loads
        if value.target is None or value.target in names
    )
    return projected


def _validate_part_namespaced_references(
    parts: tuple[NativePart, ...],
    regions: tuple[NamedRegion, ...],
) -> None:
    by_id = {part.id: part for part in parts}
    for region in regions:
        for reference in region.references:
            if type(reference) is MeshEntityRef:
                if reference.part_id is None:
                    if len(parts) != 1:
                        raise ValueError(
                            f"named region {region.name!r} contains an "
                            "ownerless mesh reference in a multi-Part project"
                        )
                elif reference.part_id not in by_id:
                    raise ValueError(
                        f"named region {region.name!r} contains a mesh "
                        "reference owned by an unknown Part"
                    )
                continue
            if type(reference) is not LogicalEntityRef:
                continue
            owner = part_id_from_logical_id(reference.logical_id)
            if owner is None or owner not in by_id:
                raise ValueError(
                    f"named region {region.name!r} contains a reference "
                    "without an active Part owner"
                )
            local = strip_part_logical_id(owner, reference.logical_id)
            from fem.geometry.recipe_topology import describe_recipe_topology

            topology = describe_recipe_topology(
                by_id[owner].geometry_recipe
            )
            try:
                topology.entity(local)
            except KeyError as error:
                raise ValueError(
                    f"named region {region.name!r} contains unknown Part "
                    f"reference {reference.logical_id!r}"
                ) from error


def _validate_suppressed_part_analysis_targets(
    parts: tuple[NativePart, ...],
    regions: tuple[NamedRegion, ...],
    assignments: tuple[RegionAssignment, ...],
    steps: tuple[Any, ...],
) -> None:
    suppressed = {part.id for part in parts if part.suppressed}
    if not suppressed:
        return
    used_region_names = _region_references(assignments, steps)
    for region in regions:
        if region.name not in used_region_names:
            continue
        blocked = sorted(
            {
                owner
                for reference in region.references
                if type(reference) is LogicalEntityRef
                and (
                    owner := part_id_from_logical_id(
                        reference.logical_id
                    )
                )
                in suppressed
            },
            key=part_id_sort_key,
        )
        if blocked:
            raise SessionStateError(
                "suppressed Part references cannot be active analysis "
                f"targets: {region.name!r} -> {', '.join(blocked)}"
            )


def _transition_part_named_regions(
    regions: tuple[NamedRegion, ...],
    part_id: str,
    before_recipe: object,
    after_recipe: object,
) -> tuple[NamedRegion, ...]:
    normalized = normalize_part_id(part_id)
    preserve = can_preserve_logical_references(before_recipe, after_recipe)
    local_rewrites = (
        {}
        if preserve
        else logical_reference_transition_map(before_recipe, after_recipe)
    )
    transitioned: list[NamedRegion] = []
    for region in regions:
        if any(type(item) is not LogicalEntityRef for item in region.references):
            transitioned.append(region)
            continue
        references: list[LogicalEntityRef] = []
        for reference in region.references:
            owner = part_id_from_logical_id(reference.logical_id)
            if owner != normalized:
                references.append(reference)
                continue
            local_id = strip_part_logical_id(
                normalized,
                reference.logical_id,
            )
            targets = (
                (local_id,)
                if preserve
                else local_rewrites.get(local_id, ())
            )
            references.extend(
                LogicalEntityRef(
                    namespace_part_logical_id(normalized, target)
                )
                for target in targets
            )
        unique = tuple(
            dict.fromkeys(
                sorted(references, key=logical_ref_sort_key)
            )
        )
        if unique:
            transitioned.append(replace(region, references=unique))
    return tuple(transitioned)


def _merge_extrusion_region_transitions(
    regions: tuple[NamedRegion, ...],
    primary_part_id: str,
    before_recipe: object,
    primary_regions: tuple[NamedRegion, ...],
    siblings: tuple[tuple[str, object], ...],
) -> tuple[NamedRegion, ...]:
    by_name: dict[str, list[LogicalEntityRef | MeshEntityRef]] = {
        region.name: list(region.references)
        for region in primary_regions
    }
    for sibling_id, recipe in siblings:
        transitioned = _transition_part_named_regions(
            regions,
            primary_part_id,
            before_recipe,
            recipe,
        )
        for region in transitioned:
            for reference in region.references:
                if (
                    type(reference) is LogicalEntityRef
                    and part_id_from_logical_id(reference.logical_id)
                    == primary_part_id
                ):
                    local_id = strip_part_logical_id(
                        primary_part_id,
                        reference.logical_id,
                    )
                    by_name.setdefault(region.name, []).append(
                        LogicalEntityRef(
                            namespace_part_logical_id(
                                sibling_id,
                                local_id,
                            )
                        )
                    )
    merged: list[NamedRegion] = []
    for region in regions:
        references = tuple(
            dict.fromkeys(
                sorted(
                    by_name.get(region.name, ()),
                    key=lambda reference: (
                        logical_ref_sort_key(reference)
                        if type(reference) is LogicalEntityRef
                        else (
                            reference.kind,
                            reference.identity,
                            reference.node_ids,
                        )
                    ),
                )
            )
        )
        if references:
            merged.append(replace(region, references=references))
    return tuple(merged)


def _remove_part_from_named_regions(
    regions: tuple[NamedRegion, ...],
    part_id: str,
) -> tuple[NamedRegion, ...]:
    normalized = normalize_part_id(part_id)
    result: list[NamedRegion] = []
    for region in regions:
        if any(type(item) is not LogicalEntityRef for item in region.references):
            result.append(region)
            continue
        references = tuple(
            reference
            for reference in region.references
            if part_id_from_logical_id(reference.logical_id) != normalized
        )
        if references:
            result.append(replace(region, references=references))
    return tuple(result)


def _part_boolean_forward_map(
    context: PartBooleanContext,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for mapping in context.topology_mappings:
        grouped.setdefault(mapping.source_logical_id, []).append(
            mapping.target_logical_id
        )
    return {
        source: tuple(
            sorted(set(targets), key=lambda value: logical_ref_sort_key(
                LogicalEntityRef(value)
            ))
        )
        for source, targets in grouped.items()
    }


def _rewrite_part_boolean_regions(
    regions: tuple[NamedRegion, ...],
    rewrites: Mapping[str, tuple[str, ...]],
    source_part_ids: set[str],
) -> tuple[NamedRegion, ...]:
    result: list[NamedRegion] = []
    for region in regions:
        if any(type(item) is not LogicalEntityRef for item in region.references):
            result.append(region)
            continue
        references: list[LogicalEntityRef] = []
        for reference in region.references:
            owner = part_id_from_logical_id(reference.logical_id)
            if owner not in source_part_ids:
                references.append(reference)
                continue
            references.extend(
                LogicalEntityRef(target)
                for target in rewrites.get(reference.logical_id, ())
            )
        unique = tuple(
            dict.fromkeys(
                sorted(references, key=logical_ref_sort_key)
            )
        )
        if unique:
            result.append(replace(region, references=unique))
    return tuple(result)


def _transition_part_boolean_mesh_settings(
    settings: MeshSettings | None,
    rewrites: Mapping[str, tuple[str, ...]],
    source_part_id: str,
    result_part_id: str,
) -> MeshSettings | None:
    del result_part_id
    if settings is None:
        return None
    controls = tuple(
        replace(control, target=LogicalEntityRef(target))
        for control in settings.local_controls
        if part_id_from_logical_id(control.target.logical_id)
        == source_part_id
        for target in rewrites.get(control.target.logical_id, ())
    )
    return replace(deepcopy(settings), local_controls=controls)


def _strict_boolean_transition(
    before: object,
    after: object,
) -> tuple[str, BooleanBodyContext | PlanarBooleanContext] | None:
    """Identify one exact top-level strict Boolean commit or undo."""

    if (
        isinstance(after, BooleanGeometry)
        and after.planar_context is not None
        and after.object_geometry == before
    ):
        return "forward", after.planar_context
    if (
        isinstance(before, BooleanGeometry)
        and before.planar_context is not None
        and before.object_geometry == after
    ):
        return "reverse", before.planar_context
    if not isinstance(before, MultiBodyGeometry) or not isinstance(
        after,
        MultiBodyGeometry,
    ):
        return None
    before_by_id = {body.id: body for body in before.bodies}
    after_by_id = {body.id: body for body in after.bodies}
    matches: list[tuple[str, BooleanBodyContext]] = []
    for target_body_id in before_by_id.keys() & after_by_id.keys():
        before_body = before_by_id[target_body_id]
        after_body = after_by_id[target_body_id]
        forward = after_body.recipe
        if isinstance(forward, BooleanGeometry):
            context = forward.body_context
            tool = (
                None
                if context is None
                else before_by_id.get(context.tool_body_id)
            )
            if (
                context is not None
                and context.target_body_id == target_body_id
                and context.tool_body_id not in after_by_id
                and forward.object_geometry == before_body.recipe
                and tool is not None
                and forward.tool_geometry == tool.recipe
            ):
                matches.append(("forward", context))
        reverse = before_body.recipe
        if isinstance(reverse, BooleanGeometry):
            context = reverse.body_context
            tool = (
                None
                if context is None
                else after_by_id.get(context.tool_body_id)
            )
            if (
                context is not None
                and context.target_body_id == target_body_id
                and context.tool_body_id not in before_by_id
                and reverse.object_geometry == after_body.recipe
                and tool is not None
                and reverse.tool_geometry == tool.recipe
            ):
                matches.append(("reverse", context))
    return matches[0] if len(matches) == 1 else None


def _boolean_context_target(
    context: BooleanBodyContext | PlanarBooleanContext,
) -> str:
    if type(context) is BooleanBodyContext:
        return context.target_body_id
    if type(context) is PlanarBooleanContext:
        return context.target_face_id
    raise TypeError("unsupported strict Boolean context")


def _active_boolean_contexts(
    recipe: object | None,
) -> tuple[BooleanBodyContext | PlanarBooleanContext, ...]:
    contexts: list[BooleanBodyContext | PlanarBooleanContext] = []

    def visit(item: object | None) -> None:
        if isinstance(item, MultiBodyGeometry):
            for body in item.bodies:
                visit(body.recipe)
            return
        if isinstance(item, BooleanGeometry):
            if item.body_context is not None:
                contexts.append(item.body_context)
            if item.planar_context is not None:
                contexts.append(item.planar_context)
            visit(item.object_geometry)
            visit(item.tool_geometry)
            return
        base = getattr(item, "base", None)
        if base is not None:
            visit(base)

    visit(recipe)
    return tuple(contexts)


def _same_active_multi_body_geometry(left: object, right: object) -> bool:
    """Compare active Body state while allowing monotonic retirement ledgers."""

    if not isinstance(left, MultiBodyGeometry) or not isinstance(
        right,
        MultiBodyGeometry,
    ):
        return left == right
    return replace(
        left,
        retired_body_ids=(),
        retired_boolean_feature_ids=(),
    ) == replace(
        right,
        retired_body_ids=(),
        retired_boolean_feature_ids=(),
    )


def _transition_mesh_settings(
    current: Any,
    recipe: Any,
    *,
    preserve_references: bool,
    surviving_logical_ids: frozenset[str] = frozenset(),
    reference_rewrites: Mapping[str, tuple[str, ...]] | None = None,
    requested: MeshSettings | None | Unset,
) -> tuple[MeshSettings | None, frozenset[TransitionEffect]]:
    """Apply the three-state mesh-input policy without mutating Session state."""

    if not isinstance(requested, Unset):
        if requested is None:
            effects = (
                frozenset({TransitionEffect.LOCAL_CONTROLS_CLEARED})
                if _has_local_mesh_controls(current)
                else frozenset()
            )
            return None, effects
        if type(requested) is not MeshSettings:
            raise TypeError(
                "mesh_settings must be MeshSettings, None, or UNSET"
            )
        owned = deepcopy(requested)
        _validate_explicit_mesh_settings(owned, recipe)
        effects = (
            frozenset({TransitionEffect.LOCAL_CONTROLS_CLEARED})
            if _has_local_mesh_controls(current)
            and not owned.local_controls
            else frozenset()
        )
        return owned, effects

    if recipe is None:
        transitioned = _without_mesh_topology_references(current)
        effects: set[TransitionEffect] = set()
        if _has_local_mesh_controls(current) and not _has_local_mesh_controls(
            transitioned
        ):
            effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
        if transitioned is not None and type(transitioned) is not MeshSettings:
            raise TypeError("existing mesh_settings must be MeshSettings or None")
        return transitioned, frozenset(effects)

    if current is None:
        return _default_mesh_settings(recipe), frozenset()
    if type(current) is not MeshSettings:
        raise TypeError("existing mesh_settings must be MeshSettings or None")

    rewrites = {
        logical_id: (logical_id,)
        for logical_id in surviving_logical_ids
    }
    rewrites.update(reference_rewrites or {})
    controls = current.local_controls if preserve_references else tuple(
        replace(
            control,
            target=LogicalEntityRef(rewrites[control.target.logical_id][0]),
        )
        for control in current.local_controls
        if len(rewrites.get(control.target.logical_id, ())) == 1
    )
    if geometry_dimension(recipe) == 1:
        if current.cell_shape == "line":
            transitioned = replace(deepcopy(current), local_controls=controls)
            effects = set()
            if len(controls) < len(current.local_controls):
                effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
            return transitioned, frozenset(effects)
        effects = {TransitionEffect.MESH_SHAPE_NORMALIZED}
        if current.local_controls:
            effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
        return None, frozenset(effects)

    transitioned = replace(deepcopy(current), local_controls=controls)
    effects = set()
    if len(controls) < len(current.local_controls):
        effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
    if not _mesh_shape_supported(transitioned.cell_shape, recipe):
        updates = {"cell_shape": _default_cell_shape(recipe)}
        if geometry_dimension(recipe) != 1:
            updates["line_element_type"] = None
        transitioned = replace(
            transitioned,
            **updates,
        )
        effects.add(TransitionEffect.MESH_SHAPE_NORMALIZED)
    return transitioned, frozenset(effects)


def _validate_explicit_mesh_settings(
    settings: MeshSettings,
    recipe: Any,
) -> None:
    if recipe is None:
        if settings.local_controls:
            raise ValueError(
                "local mesh controls require a geometry recipe"
            )
        return
    from .native_mesh_contract import describe_native_mesh_contract

    describe_native_mesh_contract(recipe, settings)
    if not _mesh_shape_supported(settings.cell_shape, recipe):
        raise ValueError(
            f"mesh cell shape {settings.cell_shape!r} is not supported "
            "by the submitted geometry recipe"
        )


def _mesh_shape_supported(cell_shape: str, recipe: Any) -> bool:
    dimension = geometry_dimension(recipe)
    if dimension == 1:
        return cell_shape == "line"
    if dimension == 2:
        return cell_shape in {"triangle", "quadrilateral"}
    if cell_shape == "tetrahedron":
        return True
    return (
        cell_shape == "hexahedron"
        and supports_structured_hexahedron(recipe)
    )


def _default_mesh_settings(recipe: Any) -> MeshSettings | None:
    if geometry_dimension(recipe) == 1:
        return None
    return MeshSettings(
        recipe_characteristic_size(recipe) / 10.0,
        cell_shape=_default_cell_shape(recipe),
    )


def _default_cell_shape(recipe: Any) -> str:
    dimension = geometry_dimension(recipe)
    if dimension == 1:
        return "line"
    return "tetrahedron" if dimension == 3 else "triangle"


def _has_local_mesh_controls(settings: Any) -> bool:
    return bool(
        settings is not None
        and getattr(settings, "local_controls", ())
    )


def _without_mesh_topology_references(settings: Any) -> Any:
    """Preserve global settings while dropping geometry-entity references."""

    owned = deepcopy(settings)
    if owned is None:
        return None
    updates: dict[str, Any] = {}
    if hasattr(owned, "local_controls"):
        updates["local_controls"] = ()
    if updates:
        try:
            return replace(owned, **updates)
        except TypeError:
            pass
    if isinstance(owned, dict):
        if "local_controls" in owned:
            owned["local_controls"] = ()
    return owned


def _without_geometry_dependent_steps(
    steps: Iterable[Any],
) -> tuple[Any, ...]:
    """Keep analysis steps whose inputs do not depend on geometry regions."""

    return tuple(
        step
        for step in steps
        if not analysis_step_has_native_region_target(step)
    )


def _regions_use_mesh_entities(regions: Iterable[Any]) -> bool:
    """Return whether any supplied scope stores generated-mesh entities."""

    return any(
        type(reference) is MeshEntityRef
        for region in regions
        for reference in tuple(getattr(region, "references", ()))
    )
