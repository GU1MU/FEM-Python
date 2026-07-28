"""Headless, revisioned ownership of one FEM project lifecycle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fem.geometry import (
    geometry_dimension,
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.geometry.recipe_topology import can_preserve_logical_references
from fem.mesh.settings import MeshSettings

from .results import (
    FieldMaterializationKey,
    ResultMaterializationPatch,
    ResultSourceKey,
    SolveResultBundle,
    advance_materialization,
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
    NativePart,
    RegionAssignment,
    SectionDefinition,
    compile_model_definitions,
    definitions_from_model,
    normalize_model_definitions,
)
from .diagnostics import PreflightReport, internal_error_report
from .feature_history import derive_feature_history
from .project_validation import (
    analysis_step_has_native_region_target,
    analysis_steps_have_native_region_targets,
    validate_native_project_inputs,
)
from .native_mesh_contract import require_complete_native_mesh_contract
from .native_scope_materialization import (
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
    detached_result_record,
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
    geometry_recipe: Any | None
    mesh_settings: Any | None
    parts: tuple[NativePart, ...]
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
        return (
            self.source_kind == "native" and self.geometry_recipe is not None
        )

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
            and self._geometry_recipe is not None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    def new_native_project(
        self,
        name: str = "Model-1",
        *,
        expected_session_revision: int | None = None,
    ) -> SessionDelta:
        self._check_expected(expected_session_revision)
        project_name = str(name).strip()
        if not project_name:
            raise ValueError("project name must not be empty")

        self._session_id = new_identity("session")
        self._clear_content()
        self._is_open = True
        self._source_kind = "native"
        self._parts = (NativePart(),)
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
            )
        )
        # ProjectSnapshot already owns a detached copy; copy once more so the
        # caller may retain and mutate its own nested values after installation.
        detached = deepcopy(detached)
        source_kind = _canonical_source_kind(detached.source_kind)
        parts = deepcopy(detached.parts) or (
            (NativePart(),) if source_kind == "native" else ()
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
        if source_kind == "native":
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
        else:
            feature_history = deepcopy(detached.feature_history)
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
        if source_kind == "native":
            self._project_path = detached.source_path
        else:
            self._source_path = detached.source_path
        self._geometry_recipe = geometry_recipe
        self._mesh_settings = mesh_settings
        self._parts = parts
        self._feature_history = feature_history
        self._named_regions = named_regions
        self._materials = materials
        self._sections = sections
        self._assignments = assignments
        self._steps = steps
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
        owned_parts = deepcopy(tuple(parts))
        owned_recipe = deepcopy(recipe)
        if owned_recipe is not None and not owned_parts:
            owned_parts = (NativePart(),)
        owned_history = (
            () if owned_recipe is None else derive_feature_history(owned_recipe)
        )
        preserve_references = can_preserve_logical_references(
            self._geometry_recipe,
            owned_recipe,
        ) and not _regions_use_mesh_entities(self._named_regions.values())
        candidate_mesh_settings, mesh_effects = _transition_mesh_settings(
            self._mesh_settings,
            owned_recipe,
            preserve_references=preserve_references,
            requested=mesh_settings,
        )
        candidate_steps = (
            self._steps
            if preserve_references
            else _without_geometry_dependent_steps(self._steps)
        )
        mesh_settings_changed = candidate_mesh_settings != self._mesh_settings
        named_regions_changed = (
            not preserve_references and bool(self._named_regions)
        )
        assignments_cleared = (
            not preserve_references and bool(self._assignments)
        )
        steps_cleared = candidate_steps != self._steps
        definitions_changed = (
            not self._definitions_explicit
            or assignments_cleared
            or steps_cleared
        )
        candidate_regions = (
            tuple(self._named_regions.values())
            if preserve_references
            else ()
        )
        candidate_assignments = self._assignments if preserve_references else ()
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

        self._parts = owned_parts
        self._geometry_recipe = owned_recipe
        self._feature_history = owned_history
        self._mesh_settings = candidate_mesh_settings
        if not preserve_references:
            self._named_regions = {}
            self._assignments = ()
            self._steps = candidate_steps
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
        if preserve_references:
            effects.add(TransitionEffect.REFERENCES_PRESERVED)
        if named_regions_changed:
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
        self._parts = ()
        self._geometry_recipe = None
        self._feature_history = ()
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
            if self._geometry_recipe is None:
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
        self, step_name: str | None = None
    ) -> ValidationTaskSnapshot:
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
            model=deepcopy(artifact.model),
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
        record = ResultRecord(
            result_id=result_id,
            provenance=provenance,
            result=bundle.result,
            output_report=bundle.execution_report,
            materialization=bundle.initial_materialization,
        )
        self._results[run.run_id] = record
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

        advanced = advance_materialization(
            record.materialization,
            patch,
        )
        updated_record = replace(
            record,
            materialization=advanced,
        )
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
        if self._geometry_recipe is None:
            raise SessionStateError("project save requires geometry")
        project = ProjectSnapshot(
            source_kind="native",
            source_path=self._project_path,
            parts=self._parts,
            geometry_recipe=self._geometry_recipe,
            mesh_settings=self._mesh_settings,
            feature_history=self._feature_history,
            named_regions=tuple(self._named_regions.values()),
            material_definitions=self._materials,
            section_definitions=self._sections,
            region_assignments=self._assignments,
            analysis_definitions=self._steps,
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
            geometry_recipe=deepcopy(self._geometry_recipe),
            mesh_settings=deepcopy(self._mesh_settings),
            parts=deepcopy(self._parts),
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
    def _clear_content(self) -> None:
        self._is_open = False
        self._source_kind: str | None = None
        self._source_path: Path | None = None
        self._project_path: Path | None = None
        self._geometry_recipe: Any | None = None
        self._mesh_settings: Any | None = None
        self._parts: tuple[NativePart, ...] = ()
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
            ("gravity_loads", "target"),
        ):
            for value in getattr(step, collection_name, ()):
                target = getattr(value, field_name, None)
                if isinstance(target, str):
                    references.add(target)
    return references


def _transition_mesh_settings(
    current: Any,
    recipe: Any,
    *,
    preserve_references: bool,
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

    controls = current.local_controls if preserve_references else ()
    if geometry_dimension(recipe) == 1:
        if current.cell_shape == "line":
            transitioned = replace(deepcopy(current), local_controls=controls)
            effects = set()
            if current.local_controls and not controls:
                effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
            return transitioned, frozenset(effects)
        effects = {TransitionEffect.MESH_SHAPE_NORMALIZED}
        if current.local_controls:
            effects.add(TransitionEffect.LOCAL_CONTROLS_CLEARED)
        return None, frozenset(effects)

    transitioned = replace(deepcopy(current), local_controls=controls)
    effects = set()
    if current.local_controls and not controls:
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
