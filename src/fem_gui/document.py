"""GUI 文档状态，避免在窗口中复制模型状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .analysis_jobs import AnalysisJob, JobStatus


SourceKind = Literal["inp", "native"]


@dataclass(slots=True)
class NativePart:
    """Small, serialisable GUI-side representation of one editable part.

    The existing preprocessing kernel remains the source of CAD/mesh truth.  This
    object deliberately only stores workflow metadata so the GUI does not invent
    a second geometry kernel.
    """

    name: str = "Part-1"
    body_name: str = "Body-1"


@dataclass(slots=True)
class FeatureRecord:
    """One item in the shallow native feature history."""

    name: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NamedRegion:
    """A logical native region mapped to mesh sets after regeneration."""

    name: str
    entity_kind: Literal["point", "edge", "face", "body"]
    entity_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class SectionDefinition:
    """GUI-side section definition; material is linked by name."""

    name: str
    material: str
    section_type: str = "solid"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegionAssignment:
    """Assign one named section to an existing element region."""

    section_name: str
    region_name: str


@dataclass(slots=True)
class WorkflowState:
    """Explicit dependency state shared by native and INP workflows."""

    mesh_current: bool = False
    model_checked: bool = False
    results_current: bool = False
    reason: str | None = None

    def invalidate_mesh(self, reason: str) -> None:
        self.mesh_current = False
        self.invalidate_model(reason)

    def invalidate_model(self, reason: str) -> None:
        self.model_checked = False
        self.invalidate_results(reason)

    def invalidate_results(self, reason: str) -> None:
        self.results_current = False
        self.reason = str(reason)

    def mark_mesh_current(self) -> None:
        self.mesh_current = True
        self.invalidate_model("网格已更新，模型需要检查")

    def mark_checked(self) -> None:
        self.model_checked = True
        self.reason = None

    def mark_results_current(self) -> None:
        self.results_current = True
        self.reason = None


@dataclass(slots=True)
class FEMDocument:
    """当前打开的只读模型及其结果。"""

    path: Path | None = None
    source_kind: SourceKind | None = None
    geometry_recipe: Any | None = None
    mesh_settings: Any | None = None
    native_mesh_current: bool = False
    native_project_path: Path | None = None
    parts: list[NativePart] = field(default_factory=list)
    feature_history: list[FeatureRecord] = field(default_factory=list)
    named_regions: dict[str, NamedRegion] = field(default_factory=dict)
    material_definitions: list[Any] = field(default_factory=list)
    section_definitions: list[SectionDefinition] = field(default_factory=list)
    region_assignments: list[RegionAssignment] = field(default_factory=list)
    analysis_definitions: list[Any] = field(default_factory=list)
    workflow: WorkflowState = field(default_factory=WorkflowState)
    model: Any | None = None
    result: Any | None = None
    step_name: str | None = None
    jobs: list[AnalysisJob] = field(default_factory=list)
    active_job_name: str | None = None
    dirty: bool = False
    revision: int = 0

    @property
    def has_model(self) -> bool:
        return self.model is not None

    @property
    def has_result(self) -> bool:
        return self.result is not None

    @property
    def has_native_geometry(self) -> bool:
        return self.source_kind == "native" and self.geometry_recipe is not None

    @property
    def mesh_is_current(self) -> bool:
        return self.has_native_geometry and self.native_mesh_current and self.workflow.mesh_current

    @property
    def needs_model_check(self) -> bool:
        return self.model is not None and not self.workflow.model_checked

    @property
    def can_reload(self) -> bool:
        return self.source_kind == "inp" and self.path is not None

    def set_model(self, path: str | Path, model: Any) -> None:
        self.clear_jobs()
        self.path = Path(path)
        self.source_kind = "inp"
        self.geometry_recipe = None
        self.mesh_settings = None
        self.native_mesh_current = False
        self.native_project_path = None
        self.parts.clear()
        self.feature_history.clear()
        self.named_regions.clear()
        self.material_definitions.clear()
        self.section_definitions.clear()
        self.region_assignments.clear()
        self.analysis_definitions.clear()
        self.model = model
        self.result = None
        self.step_name = self.default_step_name()
        self.workflow = WorkflowState(
            mesh_current=True,
            model_checked=False,
            results_current=False,
            reason="已导入 INP，提交前请检查模型",
        )
        self.dirty = False
        self._bump_revision()

    def set_generated_model(
        self,
        model: Any,
        *,
        geometry_recipe: Any | None = None,
        mesh_settings: Any | None = None,
    ) -> None:
        """Install one generated native model without pretending it came from a file."""
        self.clear_jobs()
        self.path = None
        self.source_kind = "native"
        self.geometry_recipe = geometry_recipe
        self.mesh_settings = mesh_settings
        self.native_mesh_current = True
        if not self.parts:
            self.parts.append(NativePart())
        self.model = model
        self.result = None
        self.step_name = self.default_step_name()
        self.workflow.mark_mesh_current()
        self._bump_revision()

    def begin_native_model(self, recipe: Any, *, feature: FeatureRecord | None = None) -> None:
        """Record an editable geometry change and invalidate dependent FE data."""
        self.clear_jobs()
        self.path = None
        self.source_kind = "native"
        if not self.parts:
            self.parts.append(NativePart())
        self.geometry_recipe = recipe
        self.native_mesh_current = False
        self.model = None
        self.result = None
        self.step_name = self.default_step_name()
        if feature is not None:
            self.feature_history.append(feature)
        self.dirty = True
        self.workflow.invalidate_mesh("几何已修改，网格需要重新生成")
        self._bump_revision()

    def new_native_model(self, name: str = "Model-1") -> None:
        """Start an empty editable project without manufacturing a geometry."""
        self.close()
        self.source_kind = "native"
        self.parts.append(NativePart())
        self.dirty = True
        self.workflow.reason = f"{name} 已创建，请先创建草图"
        self._bump_revision()

    def replace_native_geometry(
        self,
        recipe: Any,
        *,
        feature_history: list[FeatureRecord],
    ) -> None:
        """Replace the regenerated history after edit, undo, or delete."""
        self.begin_native_model(recipe)
        self.feature_history = list(feature_history)

    def set_mesh_settings(self, settings: Any) -> None:
        self.mesh_settings = settings
        self.native_mesh_current = False
        self.clear_jobs()
        self.result = None
        self.dirty = True
        self.workflow.invalidate_mesh("网格设置已修改，网格需要重新生成")
        self._bump_revision()

    def mark_model_definition_changed(self, reason: str) -> None:
        """Invalidate validation/results after material or analysis input edits."""
        self.clear_jobs()
        self.result = None
        self.dirty = True
        self.workflow.invalidate_model(reason)
        self._bump_revision()

    def mark_dirty(self) -> None:
        self.dirty = True
        self._bump_revision()

    def mark_model_checked(self) -> None:
        self.workflow.mark_checked()

    def mark_result_current(self) -> None:
        self.workflow.mark_results_current()

    def close(self) -> None:
        self.clear_jobs()
        self.path = None
        self.source_kind = None
        self.geometry_recipe = None
        self.mesh_settings = None
        self.native_mesh_current = False
        self.native_project_path = None
        self.parts.clear()
        self.feature_history.clear()
        self.named_regions.clear()
        self.material_definitions.clear()
        self.section_definitions.clear()
        self.region_assignments.clear()
        self.analysis_definitions.clear()
        self.workflow = WorkflowState()
        self.model = None
        self.result = None
        self.step_name = None
        self.dirty = False
        self._bump_revision()

    def clear_generated_mesh(self) -> None:
        """Drop a generated FE mesh while keeping its native geometry inputs."""
        if self.source_kind != "native":
            return
        self.clear_jobs()
        self.path = None
        self.native_mesh_current = False
        self.model = None
        self.result = None
        self.step_name = self.default_step_name()
        self.dirty = True
        self.workflow.invalidate_mesh("网格已清除，请重新生成")
        self._bump_revision()

    def _bump_revision(self) -> None:
        """Advance the identity of mutable model inputs for stale-task guards."""
        self.revision += 1

    def add_job(self, job: AnalysisJob) -> None:
        """添加名称唯一的会话作业。"""
        job.name = job.name.strip()
        if not job.name:
            raise ValueError("作业名称不能为空")
        if self.find_job(job.name) is not None:
            raise ValueError(f"作业名称已存在：{job.name}")
        self.jobs.append(job)

    def find_job(self, name: str | None) -> AnalysisJob | None:
        """按忽略大小写的名称查找作业。"""
        normalized = str(name or "").strip().casefold()
        return next((job for job in self.jobs if job.name.casefold() == normalized), None)

    def next_job_name(self) -> str:
        """返回下一个不与现有名称冲突的 Job-N。"""
        number = 1
        while self.find_job(f"Job-{number}") is not None:
            number += 1
        return f"Job-{number}"

    def latest_resubmittable_job(self) -> AnalysisJob | None:
        """返回最近完成、失败或取消的作业。"""
        for job in reversed(self.jobs):
            if job.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return job
        return None

    def clear_jobs(self) -> None:
        """清除当前模型全部会话作业和活动作业指针。"""
        self.jobs.clear()
        self.active_job_name = None

    def default_step_name(self) -> str | None:
        steps = self.analysis_definitions or (
            list(self.model.steps) if self.model is not None else []
        )
        runnable = [step.name for step in steps if step.name.lower() != "initial"]
        if runnable:
            return runnable[0]
        return steps[0].name if steps else None

    def runnable_step_names(self) -> tuple[str, ...]:
        steps = self.analysis_definitions or (
            list(self.model.steps) if self.model is not None else []
        )
        names = tuple(step.name for step in steps if step.name.lower() != "initial")
        return names or tuple(step.name for step in steps)
