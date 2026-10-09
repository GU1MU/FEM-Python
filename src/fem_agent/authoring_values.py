"""Immutable authoring records and bounded provider-turn snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import re

from .authoring import AuthoringContext, ProposalState
from .authoring_payload import _provider_safe_summary


class AuthoringWorkflowStage(str, Enum):
    REQUIREMENTS = "requirements"
    GEOMETRY_READY = "geometry_ready"
    GEOMETRY_PENDING = "geometry_pending"
    MESH_READY = "mesh_ready"
    MESH_PENDING = "mesh_pending"
    DEFINITIONS_READY = "definitions_ready"
    ANALYSIS_DEFINITIONS_READY = "analysis_definitions_ready"
    PREFLIGHT_READY = "preflight_ready"
    PREFLIGHT_PENDING = "preflight_pending"
    SOLVE_READY = "solve_ready"
    SOLVE_PENDING = "solve_pending"
    RESULTS_READY = "results_ready"
    PROJECT_SAVE_PENDING = "project_save_pending"
    DESTRUCTIVE_EDIT_PENDING = "destructive_edit_pending"
    STALE = "stale"
    CANCELLED = "cancelled"


# A turn snapshot is deliberately smaller than a tool result.  It is sent on
# every provider round, so keeping this budget independent from the general
# authoring payload prevents a large feature catalog from leaking into the
# conversation context.
AUTHORING_TURN_SNAPSHOT_MAX_BYTES = 8_192
AUTHORING_TURN_SNAPSHOT_MAX_ITEMS = 96
AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION = "1"
_SNAPSHOT_BUDGET_MARGIN_BYTES = 128
_SNAPSHOT_PREFERRED_NAMES = (
    "read_authoring_context",
    "read_geometry_feature_catalog",
    "read_geometry_edit_context",
    "prepare_geometry_edit",
    "edit_native_geometry",
)


@dataclass(frozen=True, slots=True)
class AuthoringToolOutcome:
    summary: str
    data: Mapping[str, object]
    ok: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("authoring tool summary must be non-blank")
        if not isinstance(self.data, Mapping):
            raise TypeError("authoring tool data must be an object")
        if not isinstance(self.ok, bool):
            raise TypeError("authoring tool ok must be boolean")
        object.__setattr__(
            self,
            "summary",
            _provider_safe_summary(self.summary),
        )
        object.__setattr__(self, "data", dict(self.data))


@dataclass(frozen=True, slots=True)
class AuthoringTurnSnapshot:
    """Minimal immutable authoring state projected at a provider turn.

    The snapshot is intentionally not an ``AuthoringContext`` replacement:
    canonical profile IDs and complete feature catalogs remain local tools.
    ``available`` is false until a typed context has been observed on the GUI
    owner thread; callers must not fill an unavailable snapshot from history.
    """

    available: bool = False
    source_kind: str | None = None
    workflow_stage: str | None = None
    document_id: str | None = None
    session_id: str | None = None
    session_revision: int | None = None
    active_part_id: str | None = None
    active_part_dimension: int | None = None
    active_part_recipe_kind: str | None = None
    active_part_suppressed: bool | None = None
    mesh_present: bool = False
    mesh_current: bool = False
    enabled_capabilities: tuple[str, ...] = ()
    published_tool_names: tuple[str, ...] = ()
    snapshot_generation: int = 0
    truncated: bool = False
    schema_version: str = AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("snapshot available must be boolean")
        if self.schema_version != AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported authoring turn snapshot schema")
        if (
            isinstance(self.snapshot_generation, bool)
            or not isinstance(self.snapshot_generation, int)
            or self.snapshot_generation < 0
        ):
            raise ValueError("snapshot_generation must be a non-negative integer")
        if self.session_revision is not None and (
            isinstance(self.session_revision, bool)
            or not isinstance(self.session_revision, int)
            or self.session_revision < 0
        ):
            raise ValueError("session_revision must be a non-negative integer")
        if (
            self.active_part_dimension is not None
            and self.active_part_dimension
            not in {
                1,
                2,
                3,
            }
        ):
            raise ValueError("active_part_dimension must be 1, 2, or 3")
        if type(self.mesh_present) is not bool or type(self.mesh_current) is not bool:
            raise TypeError("snapshot mesh flags must be boolean")
        if type(self.truncated) is not bool:
            raise TypeError("snapshot truncated must be boolean")

        for name in (
            "source_kind",
            "workflow_stage",
            "document_id",
            "session_id",
            "active_part_id",
            "active_part_recipe_kind",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _snapshot_text(value, name))
        if (
            self.active_part_suppressed is not None
            and type(self.active_part_suppressed) is not bool
        ):
            raise TypeError("active_part_suppressed must be boolean or null")
        object.__setattr__(
            self,
            "enabled_capabilities",
            _snapshot_names(self.enabled_capabilities),
        )
        object.__setattr__(
            self,
            "published_tool_names",
            _snapshot_names(self.published_tool_names),
        )

        # Enforce the byte budget after normalizing names.  A deterministic
        # tail trim is used only for non-preferred names; core read/prepare
        # capabilities are retained whenever they are present.
        payload = self._to_dict_unbounded()
        budget = AUTHORING_TURN_SNAPSHOT_MAX_BYTES - _SNAPSHOT_BUDGET_MARGIN_BYTES
        if _json_size_bytes(payload) > budget:
            payload["truncated"] = True
            capabilities = list(self.enabled_capabilities)
            tools = list(self.published_tool_names)
            while _json_size_bytes(payload) > budget:
                removed = False
                for values, field_name in (
                    (capabilities, "enabled_capabilities"),
                    (tools, "published_tool_names"),
                ):
                    preferred = {
                        item for item in _SNAPSHOT_PREFERRED_NAMES if item in values
                    }
                    candidate_index = next(
                        (
                            index
                            for index in range(len(values) - 1, -1, -1)
                            if values[index] not in preferred
                        ),
                        None,
                    )
                    if candidate_index is None:
                        continue
                    values.pop(candidate_index)
                    payload[field_name] = list(values)
                    removed = True
                    break
                if not removed:
                    break
            object.__setattr__(self, "enabled_capabilities", tuple(capabilities))
            object.__setattr__(self, "published_tool_names", tuple(tools))
            object.__setattr__(self, "truncated", True)

    @classmethod
    def unavailable(cls, *, generation: int = 0) -> "AuthoringTurnSnapshot":
        return cls(snapshot_generation=generation)

    @classmethod
    def from_context(
        cls,
        context: AuthoringContext | None,
        *,
        workflow_stage: AuthoringWorkflowStage | str | None,
        published_tool_names: Sequence[str] = (),
        generation: int = 0,
    ) -> "AuthoringTurnSnapshot":
        if context is None:
            return cls.unavailable(generation=generation)
        active = next(
            (item for item in context.parts if item.part_id == context.active_part_id),
            None,
        )
        callable_names = {
            str(name).strip() for name in published_tool_names if str(name).strip()
        }
        return cls(
            available=True,
            source_kind=context.binding.source_kind,
            workflow_stage=(
                None
                if workflow_stage is None
                else str(getattr(workflow_stage, "value", workflow_stage))
            ),
            document_id=context.binding.document_id,
            session_id=context.binding.session_id,
            session_revision=context.binding.session_revision,
            active_part_id=context.active_part_id,
            active_part_dimension=(None if active is None else active.dimension),
            active_part_recipe_kind=(None if active is None else active.recipe_kind),
            active_part_suppressed=(None if active is None else active.suppressed),
            mesh_present=context.mesh.present,
            mesh_current=context.mesh.current,
            enabled_capabilities=tuple(
                item.operation
                for item in context.capabilities
                if item.enabled and item.operation in callable_names
            ),
            published_tool_names=published_tool_names,
            snapshot_generation=generation,
        )

    def _to_dict_unbounded(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "source_kind": self.source_kind,
            "workflow_stage": self.workflow_stage,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "session_revision": self.session_revision,
            "active_part_id": self.active_part_id,
            "active_part_dimension": self.active_part_dimension,
            "active_part_recipe_kind": self.active_part_recipe_kind,
            "active_part_suppressed": self.active_part_suppressed,
            "mesh_present": self.mesh_present,
            "mesh_current": self.mesh_current,
            "enabled_capabilities": list(self.enabled_capabilities),
            "published_tool_names": list(self.published_tool_names),
            "snapshot_generation": self.snapshot_generation,
            "truncated": self.truncated,
        }

    def to_provider_dict(self) -> dict[str, object]:
        """Return a detached bounded JSON projection for the provider."""

        return dict(self._to_dict_unbounded())

    def to_dict(self) -> dict[str, object]:
        return self.to_provider_dict()


@dataclass(frozen=True, slots=True)
class AuthoringTerminalRecord:
    operation: str
    state: str
    message: str


@dataclass(frozen=True, slots=True)
class PlanarConstructionAuditRecord:
    construction_digest: str
    stage: str
    diagnostic_code: str | None
    proposal_id: str | None
    terminal_state: str | None


@dataclass(frozen=True, slots=True)
class ProjectSaveProposalRecord:
    proposal_id: str
    proposal_hash: str
    target_document_id: str
    target_session_id: str
    base_session_revision: int
    resume_stage: AuthoringWorkflowStage
    state: ProposalState = ProposalState.PENDING_CONFIRMATION
    message: str = ""

    def __post_init__(self) -> None:
        identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
        for field_name in (
            "proposal_id",
            "target_document_id",
            "target_session_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or identifier.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a bounded identifier")
        if (
            not isinstance(self.proposal_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.proposal_hash) is None
        ):
            raise ValueError("proposal_hash must be a SHA-256 hexadecimal value")
        if (
            type(self.base_session_revision) is not int
            or self.base_session_revision < 0
        ):
            raise ValueError("base_session_revision must be non-negative")
        object.__setattr__(
            self,
            "resume_stage",
            AuthoringWorkflowStage(self.resume_stage),
        )
        object.__setattr__(self, "state", ProposalState(self.state))
        object.__setattr__(self, "message", str(self.message).strip()[:512])


def _snapshot_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    normalized = value.strip()
    if len(normalized.encode("utf-8")) > 128:
        raise ValueError(f"{field_name} exceeds the snapshot identity budget")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} contains control characters")
    return normalized


def _snapshot_names(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        values = tuple(values)
    normalized = {
        _snapshot_text(value, "snapshot name")
        for value in values
        if isinstance(value, str) and value.strip()
    }
    ordered = sorted(normalized)
    preferred = [item for item in _SNAPSHOT_PREFERRED_NAMES if item in normalized]
    remainder = [item for item in ordered if item not in preferred]
    return tuple((preferred + remainder)[:AUTHORING_TURN_SNAPSHOT_MAX_ITEMS])


def _json_size_bytes(payload: Mapping[str, object]) -> int:
    return (
        len(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        + 1
    )
