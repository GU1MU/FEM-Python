"""Pure, local contracts for FEM Agent V1 authoring.

This module deliberately has no GUI, Qt, VTK, Gmsh, filesystem, or
``ModelSession`` dependency. A1 uses it for detached requirements, drafts,
patch/proposal envelopes, and a no-side-effect fake port.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence


AUTHORING_SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_POSIX_PATH = re.compile(r"^/")
_EXECUTABLE_KEYS = frozenset(
    {
        "callback",
        "code",
        "command",
        "executable",
        "python",
        "script",
        "shell",
    }
)


class AuthoringContractError(ValueError):
    """A detached authoring value violates the locked A1 contract."""


class AuthoringAuthorizationError(PermissionError):
    """An operation was not authorized by a local GUI control."""


class ClarificationRequiredError(RuntimeError):
    """A deterministic authoring stage gate is not satisfied."""

    code = "clarification_required"

    def __init__(self, stage: str, missing_fields: Sequence[str]) -> None:
        self.stage = _require_text(stage, "stage")
        self.missing_fields = tuple(
            sorted({_require_key(item) for item in missing_fields})
        )
        super().__init__(
            f"{self.stage} requires confirmed fields: "
            + ", ".join(self.missing_fields)
        )


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthoringContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_identifier(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if not _IDENTIFIER.fullmatch(text):
        raise AuthoringContractError(f"{field_name} is not a valid identifier")
    return text


def _require_key(value: object) -> str:
    return _require_identifier(value, "requirement key")


def _reject_local_or_executable_data(value: object, *, key: str = "") -> None:
    normalized_key = key.casefold()
    if normalized_key in _EXECUTABLE_KEYS:
        raise AuthoringContractError(f"{key} is not allowed in authoring DTOs")
    if normalized_key.endswith("_path") or normalized_key == "path":
        raise AuthoringContractError(
            "filesystem paths are not allowed in authoring DTOs"
        )
    if isinstance(value, str):
        if _WINDOWS_PATH.match(value) or _POSIX_PATH.match(value):
            raise AuthoringContractError(
                "absolute paths are not allowed in authoring DTOs"
            )
        return
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise AuthoringContractError("JSON object keys must be strings")
            _reject_local_or_executable_data(child, key=child_key)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _reject_local_or_executable_data(child)


def _json_copy(value: object, field_name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AuthoringContractError(
            f"{field_name} must contain only finite JSON values"
        ) from exc
    decoded = json.loads(encoded)
    _reject_local_or_executable_data(decoded)
    return decoded


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_fields(
    record: Mapping[str, object],
    expected: Iterable[str],
    field_name: str,
) -> None:
    if not isinstance(record, Mapping):
        raise AuthoringContractError(f"{field_name} must be an object")
    expected_set = frozenset(expected)
    actual = frozenset(record)
    missing = expected_set - actual
    unknown = actual - expected_set
    if missing:
        raise AuthoringContractError(
            f"{field_name} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise AuthoringContractError(
            f"{field_name} has unknown fields: {', '.join(sorted(unknown))}"
        )


@dataclass(frozen=True, slots=True)
class LocalModelBinding:
    """Local identity for one GUI document/session/revision."""

    document_id: str
    session_id: str
    session_revision: int
    source_kind: str
    supported: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _require_identifier(self.document_id, "document_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_identifier(self.session_id, "session_id"),
        )
        if (
            not isinstance(self.session_revision, int)
            or isinstance(self.session_revision, bool)
            or self.session_revision < 0
        ):
            raise AuthoringContractError(
                "session_revision must be a non-negative integer"
            )
        source_kind = _require_text(self.source_kind, "source_kind")
        if source_kind not in {"blank", "native", "imported"}:
            raise AuthoringContractError("source_kind is not supported")
        object.__setattr__(self, "source_kind", source_kind)
        if not isinstance(self.supported, bool):
            raise AuthoringContractError("supported must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "session_id": self.session_id,
            "session_revision": self.session_revision,
            "source_kind": self.source_kind,
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class PartSummary:
    part_id: str
    name: str
    recipe_kind: str | None
    dimension: int | None
    suppressed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "part_id",
            _require_identifier(self.part_id, "part_id"),
        )
        object.__setattr__(self, "name", _require_text(self.name, "part name"))
        if self.recipe_kind is not None:
            object.__setattr__(
                self,
                "recipe_kind",
                _require_identifier(self.recipe_kind, "recipe_kind"),
            )
        if self.dimension not in {None, 1, 2, 3}:
            raise AuthoringContractError("dimension must be 1, 2, 3, or null")
        if not isinstance(self.suppressed, bool):
            raise AuthoringContractError("suppressed must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "recipe_kind": self.recipe_kind,
            "dimension": self.dimension,
            "suppressed": self.suppressed,
        }


@dataclass(frozen=True, slots=True)
class MeshSummary:
    present: bool
    current: bool
    node_count: int | None = None
    element_count: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("present", "current"):
            if not isinstance(getattr(self, field_name), bool):
                raise AuthoringContractError(f"{field_name} must be boolean")
        for field_name in ("node_count", "element_count"):
            value = getattr(self, field_name)
            if (
                value is not None
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                )
            ):
                raise AuthoringContractError(
                    f"{field_name} must be a non-negative integer or null"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "current": self.current,
            "node_count": self.node_count,
            "element_count": self.element_count,
        }


@dataclass(frozen=True, slots=True)
class DefinitionSummary:
    named_region_count: int = 0
    material_count: int = 0
    section_count: int = 0
    assignment_count: int = 0
    analysis_step_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "named_region_count",
            "material_count",
            "section_count",
            "assignment_count",
            "analysis_step_count",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise AuthoringContractError(
                    f"{field_name} must be a non-negative integer"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "named_region_count": self.named_region_count,
            "material_count": self.material_count,
            "section_count": self.section_count,
            "assignment_count": self.assignment_count,
            "analysis_step_count": self.analysis_step_count,
        }


@dataclass(frozen=True, slots=True)
class UnitContextSummary:
    length: str
    force: str
    stress: str
    density: str | None = None
    acceleration: str | None = None
    convention: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("length", "force", "stress"):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        for field_name in ("density", "acceleration", "convention"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _require_text(value, field_name),
                )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "length": self.length,
            "force": self.force,
            "stress": self.stress,
            "density": self.density,
            "acceleration": self.acceleration,
            "convention": self.convention,
        }


@dataclass(frozen=True, slots=True)
class CapabilitySummary:
    operation: str
    enabled: bool
    blocking_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _require_identifier(self.operation, "capability operation"),
        )
        if not isinstance(self.enabled, bool):
            raise AuthoringContractError("capability enabled must be boolean")
        if self.blocking_reason is not None:
            object.__setattr__(
                self,
                "blocking_reason",
                _require_text(self.blocking_reason, "blocking_reason"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "enabled": self.enabled,
            "blocking_reason": self.blocking_reason,
        }


@dataclass(frozen=True, slots=True)
class AuthoringContext:
    """Bounded provider-safe snapshot of accepted GUI state."""

    binding: LocalModelBinding
    model_name: str | None
    active_part_id: str | None
    parts: tuple[PartSummary, ...] = ()
    mesh: MeshSummary = field(default_factory=lambda: MeshSummary(False, False))
    definitions: DefinitionSummary = field(default_factory=DefinitionSummary)
    validation_status: str = "not_run"
    job_status: str = "idle"
    result_available: bool = False
    capabilities: tuple[CapabilitySummary, ...] = ()
    unit_context: UnitContextSummary | None = None
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.binding) is not LocalModelBinding:
            raise AuthoringContractError("binding must be LocalModelBinding")
        if self.model_name is not None:
            object.__setattr__(
                self,
                "model_name",
                _require_text(self.model_name, "model_name"),
            )
        if self.active_part_id is not None:
            object.__setattr__(
                self,
                "active_part_id",
                _require_identifier(self.active_part_id, "active_part_id"),
            )
        parts = tuple(self.parts)
        if any(type(item) is not PartSummary for item in parts):
            raise AuthoringContractError("parts must contain PartSummary values")
        if len(parts) > 128:
            raise AuthoringContractError("AuthoringContext part summary is unbounded")
        object.__setattr__(self, "parts", parts)
        if type(self.mesh) is not MeshSummary:
            raise AuthoringContractError("mesh must be MeshSummary")
        if type(self.definitions) is not DefinitionSummary:
            raise AuthoringContractError("definitions must be DefinitionSummary")
        object.__setattr__(
            self,
            "validation_status",
            _require_identifier(self.validation_status, "validation_status"),
        )
        object.__setattr__(
            self,
            "job_status",
            _require_identifier(self.job_status, "job_status"),
        )
        if not isinstance(self.result_available, bool):
            raise AuthoringContractError("result_available must be boolean")
        capabilities = tuple(self.capabilities)
        if any(type(item) is not CapabilitySummary for item in capabilities):
            raise AuthoringContractError(
                "capabilities must contain CapabilitySummary values"
            )
        object.__setattr__(self, "capabilities", capabilities)
        if (
            self.unit_context is not None
            and type(self.unit_context) is not UnitContextSummary
        ):
            raise AuthoringContractError(
                "unit_context must be UnitContextSummary or null"
            )
        if self.schema_version != AUTHORING_SCHEMA_VERSION:
            raise AuthoringContractError("unknown AuthoringContext schema_version")

    def to_provider_dict(self) -> dict[str, object]:
        """Return the only A1 representation intended for Provider context."""

        return {
            "schema_version": self.schema_version,
            "binding": self.binding.to_dict(),
            "model_name": self.model_name,
            "active_part_id": self.active_part_id,
            "parts": [item.to_dict() for item in self.parts],
            "mesh": self.mesh.to_dict(),
            "definitions": self.definitions.to_dict(),
            "validation_status": self.validation_status,
            "job_status": self.job_status,
            "result_available": self.result_available,
            "capabilities": [
                item.to_dict() for item in self.capabilities
            ],
            "unit_context": (
                None
                if self.unit_context is None
                else self.unit_context.to_dict()
            ),
        }


class RequirementStatus(str, Enum):
    MISSING = "missing"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class RequirementEntry:
    key: str
    field_type: str
    stage: str
    value: object
    source_turn_id: str
    status: RequirementStatus
    dependencies: tuple[str, ...] = ()
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _require_key(self.key))
        object.__setattr__(
            self,
            "field_type",
            _require_identifier(self.field_type, "field_type"),
        )
        object.__setattr__(
            self,
            "stage",
            _require_identifier(self.stage, "stage"),
        )
        object.__setattr__(
            self,
            "source_turn_id",
            _require_identifier(self.source_turn_id, "source_turn_id"),
        )
        try:
            status = RequirementStatus(self.status)
        except ValueError as exc:
            raise AuthoringContractError("unknown requirement status") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "value",
            _json_copy(self.value, "requirement value"),
        )
        dependencies = tuple(
            sorted({_require_key(item) for item in self.dependencies})
        )
        if self.key in dependencies:
            raise AuthoringContractError("a requirement cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        if self.invalidation_reason is not None:
            object.__setattr__(
                self,
                "invalidation_reason",
                _require_text(
                    self.invalidation_reason,
                    "invalidation_reason",
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "field_type": self.field_type,
            "stage": self.stage,
            "value": deepcopy(self.value),
            "source_turn_id": self.source_turn_id,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "invalidation_reason": self.invalidation_reason,
        }


class RequirementReviewStatus(str, Enum):
    PENDING = "pending_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class RequirementReview:
    review_id: str
    ledger_revision: int
    fields: tuple[RequirementEntry, ...]
    status: RequirementReviewStatus
    review_hash: str
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "review_id",
            _require_identifier(self.review_id, "review_id"),
        )
        if (
            not isinstance(self.ledger_revision, int)
            or isinstance(self.ledger_revision, bool)
            or self.ledger_revision < 0
        ):
            raise AuthoringContractError(
                "ledger_revision must be a non-negative integer"
            )
        fields = tuple(self.fields)
        if not fields or any(
            type(item) is not RequirementEntry for item in fields
        ):
            raise AuthoringContractError(
                "RequirementReview requires RequirementEntry fields"
            )
        object.__setattr__(self, "fields", fields)
        try:
            status = RequirementReviewStatus(self.status)
        except ValueError as exc:
            raise AuthoringContractError("unknown review status") from exc
        object.__setattr__(self, "status", status)
        if self.schema_version != AUTHORING_SCHEMA_VERSION:
            raise AuthoringContractError("unknown RequirementReview schema_version")
        if self.review_hash != _hash(self._hash_payload()):
            raise AuthoringContractError("RequirementReview hash does not match")

    @classmethod
    def create(
        cls,
        review_id: str,
        ledger_revision: int,
        fields: Sequence[RequirementEntry],
    ) -> RequirementReview:
        payload = {
            "schema_version": AUTHORING_SCHEMA_VERSION,
            "review_id": review_id,
            "ledger_revision": ledger_revision,
            "fields": [item.to_dict() for item in fields],
        }
        return cls(
            review_id=review_id,
            ledger_revision=ledger_revision,
            fields=tuple(fields),
            status=RequirementReviewStatus.PENDING,
            review_hash=_hash(payload),
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "ledger_revision": self.ledger_revision,
            "fields": [item.to_dict() for item in self.fields],
        }


class RequirementLedger:
    """Agent-private requirements with dependency invalidation."""

    def __init__(self) -> None:
        self._entries: dict[str, RequirementEntry] = {}
        self._revision = 0
        self._reviews: dict[str, RequirementReview] = {}

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def entries(self) -> tuple[RequirementEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def record(
        self,
        key: str,
        *,
        field_type: str,
        stage: str,
        value: object = None,
        source_turn_id: str,
        status: RequirementStatus = RequirementStatus.PROPOSED,
        dependencies: Sequence[str] = (),
    ) -> RequirementEntry:
        if RequirementStatus(status) is RequirementStatus.CONFIRMED:
            raise AuthoringAuthorizationError(
                "confirmed requirements require a GUI RequirementReview"
            )
        normalized = _require_key(key)
        prior = self._entries.get(normalized)
        entry = RequirementEntry(
            key=normalized,
            field_type=field_type,
            stage=stage,
            value=value,
            source_turn_id=source_turn_id,
            status=status,
            dependencies=tuple(dependencies),
        )
        if prior == entry:
            return entry
        self._entries[normalized] = entry
        self._revision += 1
        if prior is not None and prior.value != entry.value:
            self._invalidate_dependents(
                normalized,
                f"upstream requirement {normalized} changed",
            )
        self._stale_reviews()
        return entry

    def invalidate(self, key: str, reason: str) -> RequirementEntry:
        normalized = _require_key(key)
        try:
            prior = self._entries[normalized]
        except KeyError as exc:
            raise AuthoringContractError(
                f"unknown requirement: {normalized}"
            ) from exc
        invalidated = replace(
            prior,
            status=RequirementStatus.INVALIDATED,
            invalidation_reason=_require_text(reason, "reason"),
        )
        if invalidated != prior:
            self._entries[normalized] = invalidated
            self._revision += 1
            self._invalidate_dependents(normalized, reason)
            self._stale_reviews()
        return invalidated

    def create_review(
        self,
        review_id: str,
        keys: Sequence[str],
    ) -> RequirementReview:
        normalized = tuple(
            dict.fromkeys(_require_key(item) for item in keys)
        )
        if not normalized:
            raise AuthoringContractError("RequirementReview cannot be empty")
        fields: list[RequirementEntry] = []
        for key in normalized:
            try:
                entry = self._entries[key]
            except KeyError as exc:
                raise AuthoringContractError(
                    f"unknown requirement: {key}"
                ) from exc
            if entry.status not in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }:
                raise ClarificationRequiredError(entry.stage, (key,))
            fields.append(entry)
        review = RequirementReview.create(
            review_id,
            self._revision,
            fields,
        )
        self._reviews[review.review_id] = review
        return review

    def _confirm_review_from_gui(
        self,
        review: RequirementReview,
    ) -> RequirementReview:
        """Apply a review only through the GUI bridge's capability path."""

        current = self._require_current_review(review)
        if current.status is not RequirementReviewStatus.PENDING:
            raise AuthoringAuthorizationError(
                "RequirementReview is no longer pending"
            )
        for reviewed in current.fields:
            live = self._entries.get(reviewed.key)
            if live != reviewed:
                self._stale_reviews()
                raise AuthoringContractError("RequirementReview is stale")
        for reviewed in current.fields:
            self._entries[reviewed.key] = replace(
                reviewed,
                status=RequirementStatus.CONFIRMED,
                invalidation_reason=None,
            )
        self._revision += 1
        confirmed = replace(
            current,
            status=RequirementReviewStatus.CONFIRMED,
        )
        self._reviews[current.review_id] = confirmed
        return confirmed

    def _reject_review_from_gui(
        self,
        review: RequirementReview,
    ) -> RequirementReview:
        current = self._require_current_review(review)
        if current.status is not RequirementReviewStatus.PENDING:
            raise AuthoringAuthorizationError(
                "RequirementReview is no longer pending"
            )
        rejected = replace(current, status=RequirementReviewStatus.REJECTED)
        self._reviews[current.review_id] = rejected
        return rejected

    def require_confirmed(
        self,
        stage: str,
        required_keys: Sequence[str],
    ) -> tuple[RequirementEntry, ...]:
        normalized = tuple(
            dict.fromkeys(_require_key(item) for item in required_keys)
        )
        missing = [
            key
            for key in normalized
            if (
                key not in self._entries
                or self._entries[key].status is not RequirementStatus.CONFIRMED
            )
        ]
        if missing:
            raise ClarificationRequiredError(stage, missing)
        return tuple(self._entries[key] for key in normalized)

    def confirmed_hash(self) -> str:
        confirmed = [
            item.to_dict()
            for item in self.entries
            if item.status is RequirementStatus.CONFIRMED
        ]
        return _hash(confirmed)

    def _require_current_review(
        self,
        review: RequirementReview,
    ) -> RequirementReview:
        if type(review) is not RequirementReview:
            raise AuthoringContractError("review must be RequirementReview")
        current = self._reviews.get(review.review_id)
        if current is None or current.review_hash != review.review_hash:
            raise AuthoringContractError("RequirementReview is not registered")
        return current

    def _invalidate_dependents(self, key: str, reason: str) -> None:
        pending = [key]
        changed = False
        seen: set[str] = set()
        while pending:
            upstream = pending.pop()
            if upstream in seen:
                continue
            seen.add(upstream)
            for dependent_key, entry in tuple(self._entries.items()):
                if (
                    upstream in entry.dependencies
                    and entry.status is not RequirementStatus.INVALIDATED
                ):
                    self._entries[dependent_key] = replace(
                        entry,
                        status=RequirementStatus.INVALIDATED,
                        invalidation_reason=_require_text(reason, "reason"),
                    )
                    pending.append(dependent_key)
                    changed = True
        if changed:
            self._revision += 1

    def _stale_reviews(self) -> None:
        for review_id, review in tuple(self._reviews.items()):
            if (
                review.status is RequirementReviewStatus.PENDING
                and review.ledger_revision != self._revision
            ):
                self._reviews[review_id] = replace(
                    review,
                    status=RequirementReviewStatus.STALE,
                )


@dataclass(frozen=True, slots=True)
class AgentDraft:
    draft_id: str
    agent_session_id: str
    base_document_id: str
    base_session_id: str
    base_session_revision: int
    draft_revision: int
    confirmed_requirements_hash: str
    candidate_model_hash: str
    candidate_summary: object
    pending_proposal_ids: tuple[str, ...] = ()
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "draft_id",
            "agent_session_id",
            "base_document_id",
            "base_session_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identifier(getattr(self, field_name), field_name),
            )
        for field_name in ("base_session_revision", "draft_revision"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise AuthoringContractError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in (
            "confirmed_requirements_hash",
            "candidate_model_hash",
        ):
            if not _SHA256.fullmatch(str(getattr(self, field_name))):
                raise AuthoringContractError(
                    f"{field_name} must be a SHA-256 hash"
                )
        object.__setattr__(
            self,
            "candidate_summary",
            _json_copy(self.candidate_summary, "candidate_summary"),
        )
        proposal_ids = tuple(
            _require_identifier(item, "proposal_id")
            for item in self.pending_proposal_ids
        )
        if len(proposal_ids) != len(set(proposal_ids)):
            raise AuthoringContractError("pending proposal IDs must be unique")
        object.__setattr__(self, "pending_proposal_ids", proposal_ids)
        if self.schema_version != AUTHORING_SCHEMA_VERSION:
            raise AuthoringContractError("unknown AgentDraft schema_version")

    @classmethod
    def create(
        cls,
        *,
        draft_id: str,
        agent_session_id: str,
        binding: LocalModelBinding,
        confirmed_requirements: object,
        candidate_summary: object,
    ) -> AgentDraft:
        safe_requirements = _json_copy(
            confirmed_requirements,
            "confirmed_requirements",
        )
        safe_candidate = _json_copy(candidate_summary, "candidate_summary")
        return cls(
            draft_id=draft_id,
            agent_session_id=agent_session_id,
            base_document_id=binding.document_id,
            base_session_id=binding.session_id,
            base_session_revision=binding.session_revision,
            draft_revision=0,
            confirmed_requirements_hash=_hash(safe_requirements),
            candidate_model_hash=_hash(safe_candidate),
            candidate_summary=safe_candidate,
        )

    def revise(
        self,
        *,
        confirmed_requirements: object,
        candidate_summary: object,
        pending_proposal_ids: Sequence[str] = (),
    ) -> AgentDraft:
        safe_requirements = _json_copy(
            confirmed_requirements,
            "confirmed_requirements",
        )
        safe_candidate = _json_copy(candidate_summary, "candidate_summary")
        return replace(
            self,
            draft_revision=self.draft_revision + 1,
            confirmed_requirements_hash=_hash(safe_requirements),
            candidate_model_hash=_hash(safe_candidate),
            candidate_summary=safe_candidate,
            pending_proposal_ids=tuple(pending_proposal_ids),
        )


class OperationKind(str, Enum):
    CREATE_NATIVE_PROJECT = "create_native_project"
    ADD_NATIVE_PART = "add_native_part"
    REPLACE_PART_GEOMETRY = "replace_part_geometry"
    EXTRUDE_PART_PROFILES = "extrude_part_profiles"
    SET_PART_MESH_INTENT = "set_part_mesh_intent"
    UPSERT_NAMED_REGIONS = "upsert_named_regions"
    UPSERT_MODEL_DEFINITIONS = "upsert_model_definitions"
    REQUEST_MESH = "request_mesh"
    REQUEST_PREFLIGHT = "request_preflight"
    REQUEST_SOLVE = "request_solve"
    REQUEST_RESULT_QUERY = "request_result_query"
    DELETE_MODEL_OBJECT = "delete_model_object"
    EDIT_MODEL_OBJECT = "edit_model_object"


_OPERATION_PARAMETER_FIELDS: dict[
    OperationKind,
    tuple[frozenset[str], frozenset[str]],
] = {
    OperationKind.CREATE_NATIVE_PROJECT: (
        frozenset({"project_name", "part_name", "recipe", "unit_context"}),
        frozenset({"project_name", "part_name", "recipe", "unit_context"}),
    ),
    OperationKind.ADD_NATIVE_PART: (
        frozenset({"part_name", "recipe", "unit_context"}),
        frozenset({"part_name", "recipe"}),
    ),
    OperationKind.REPLACE_PART_GEOMETRY: (
        frozenset({"part_id", "recipe"}),
        frozenset({"part_id", "recipe"}),
    ),
    OperationKind.EXTRUDE_PART_PROFILES: (
        frozenset({"part_id", "base_recipe", "source_face_ids", "height"}),
        frozenset({"part_id", "base_recipe", "source_face_ids", "height"}),
    ),
    OperationKind.SET_PART_MESH_INTENT: (
        frozenset({"part_id", "mesh_intent"}),
        frozenset({"part_id", "mesh_intent"}),
    ),
    OperationKind.UPSERT_NAMED_REGIONS: (
        frozenset({"regions"}),
        frozenset({"regions"}),
    ),
    OperationKind.UPSERT_MODEL_DEFINITIONS: (
        frozenset({"definitions"}),
        frozenset({"definitions"}),
    ),
    OperationKind.REQUEST_MESH: (
        frozenset({"part_id", "mesh_intent_hash"}),
        frozenset({"part_id", "mesh_intent_hash"}),
    ),
    OperationKind.REQUEST_PREFLIGHT: (
        frozenset({"step_name"}),
        frozenset({"step_name"}),
    ),
    OperationKind.REQUEST_SOLVE: (
        frozenset(
            {
                "step_name",
                "job_name",
                "artifact_id",
                "model_revision",
                "validation_stamp",
            }
        ),
        frozenset({"step_name", "validation_stamp"}),
    ),
    OperationKind.REQUEST_RESULT_QUERY: (
        frozenset({"query"}),
        frozenset({"query"}),
    ),
    OperationKind.DELETE_MODEL_OBJECT: (
        frozenset({"object_type", "target_id", "step_name"}),
        frozenset({"object_type", "target_id"}),
    ),
    OperationKind.EDIT_MODEL_OBJECT: (
        frozenset({"object_type", "target_id", "step_name", "changes"}),
        frozenset({"object_type", "target_id", "changes"}),
    ),
}


_DELETE_MODEL_OBJECT_TYPES = frozenset(
    {
        "part",
        "generated_mesh",
        "named_region",
        "analysis_step",
        "boundary_condition",
        "load",
        "result_request",
    }
)
_STEP_CHILD_DELETE_TYPES = frozenset(
    {"boundary_condition", "load", "result_request"}
)
_EDIT_MODEL_OBJECT_TYPES = frozenset(
    {"named_region", "boundary_condition", "load"}
)
_STEP_CHILD_EDIT_TYPES = frozenset({"boundary_condition", "load"})


@dataclass(frozen=True, slots=True)
class ModelOperation:
    kind: OperationKind
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            kind = OperationKind(self.kind)
        except ValueError as exc:
            raise AuthoringContractError("unknown model operation") from exc
        object.__setattr__(self, "kind", kind)
        parameters = _json_copy(self.parameters, "operation parameters")
        if not isinstance(parameters, dict):
            raise AuthoringContractError("operation parameters must be an object")
        allowed, required = _OPERATION_PARAMETER_FIELDS[kind]
        keys = frozenset(parameters)
        unknown = keys - allowed
        missing = required - keys
        if unknown:
            raise AuthoringContractError(
                f"{kind.value} has unknown parameter fields: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise AuthoringContractError(
                f"{kind.value} is missing parameter fields: "
                + ", ".join(sorted(missing))
            )
        if kind is OperationKind.REQUEST_SOLVE and keys not in {
            frozenset({"step_name", "validation_stamp"}),
            frozenset(
                {
                    "step_name",
                    "job_name",
                    "artifact_id",
                    "model_revision",
                    "validation_stamp",
                }
            ),
        }:
            raise AuthoringContractError(
                "request_solve must use either the legacy or exact A6 field set"
            )
        if kind is OperationKind.DELETE_MODEL_OBJECT:
            object_type = parameters.get("object_type")
            target_id = parameters.get("target_id")
            step_name = parameters.get("step_name")
            if object_type not in _DELETE_MODEL_OBJECT_TYPES:
                raise AuthoringContractError(
                    "delete_model_object has an unsupported object_type"
                )
            if type(target_id) is not str or not target_id.strip():
                raise AuthoringContractError(
                    "delete_model_object target_id must be non-blank"
                )
            if object_type in _STEP_CHILD_DELETE_TYPES:
                if type(step_name) is not str or not step_name.strip():
                    raise AuthoringContractError(
                        "step_name is required for this delete target"
                    )
            elif "step_name" in parameters:
                raise AuthoringContractError(
                    "step_name is only valid for step child delete targets"
                )
        if kind is OperationKind.EDIT_MODEL_OBJECT:
            object_type = parameters.get("object_type")
            target_id = parameters.get("target_id")
            step_name = parameters.get("step_name")
            changes = parameters.get("changes")
            if object_type not in _EDIT_MODEL_OBJECT_TYPES:
                raise AuthoringContractError(
                    "edit_model_object has an unsupported object_type"
                )
            if type(target_id) is not str or not target_id.strip():
                raise AuthoringContractError(
                    "edit_model_object target_id must be non-blank"
                )
            if not isinstance(changes, Mapping) or not changes:
                raise AuthoringContractError(
                    "edit_model_object changes must be a non-empty object"
                )
            if object_type in _STEP_CHILD_EDIT_TYPES:
                if type(step_name) is not str or not step_name.strip():
                    raise AuthoringContractError(
                        "step_name is required for this edit target"
                    )
            elif "step_name" in parameters:
                raise AuthoringContractError(
                    "step_name is only valid for step child edit targets"
                )
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "parameters": deepcopy(dict(self.parameters)),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ModelOperation:
        _strict_fields(record, ("kind", "parameters"), "operation")
        return cls(
            kind=OperationKind(str(record["kind"])),
            parameters=record["parameters"],  # type: ignore[arg-type]
        )


def _validate_common_envelope(
    *,
    schema_version: str,
    envelope_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    target_document_id: str,
    target_session_id: str,
    base_session_revision: int,
    draft_revision: int,
    operations: Sequence[ModelOperation],
    preconditions: object,
    expected_changes: object,
    invalidation_impact: object,
    display_summary: object,
    idempotency_key: str,
    content_hash: str,
) -> tuple[
    tuple[str, ...],
    tuple[ModelOperation, ...],
    object,
    object,
    object,
    object,
]:
    if schema_version != AUTHORING_SCHEMA_VERSION:
        raise AuthoringContractError("unknown authoring envelope schema_version")
    for field_name, value in (
        ("envelope_id", envelope_id),
        ("agent_session_id", agent_session_id),
        ("turn_id", turn_id),
        ("target_document_id", target_document_id),
        ("target_session_id", target_session_id),
    ):
        _require_identifier(value, field_name)
    calls = tuple(
        _require_identifier(item, "source_tool_call_id")
        for item in source_tool_call_ids
    )
    if not calls or len(calls) != len(set(calls)):
        raise AuthoringContractError(
            "source_tool_call_ids must be non-empty and unique"
        )
    for field_name, value in (
        ("base_session_revision", base_session_revision),
        ("draft_revision", draft_revision),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise AuthoringContractError(
                f"{field_name} must be a non-negative integer"
            )
    typed_operations = tuple(operations)
    if not typed_operations or any(
        type(item) is not ModelOperation for item in typed_operations
    ):
        raise AuthoringContractError(
            "operations must contain at least one ModelOperation"
        )
    safe_preconditions = _json_copy(preconditions, "preconditions")
    safe_expected = _json_copy(expected_changes, "expected_changes")
    safe_invalidation = _json_copy(
        invalidation_impact,
        "invalidation_impact",
    )
    safe_summary = _json_copy(display_summary, "display_summary")
    if not _SHA256.fullmatch(idempotency_key):
        raise AuthoringContractError("idempotency_key must be SHA-256")
    if not _SHA256.fullmatch(content_hash):
        raise AuthoringContractError("hash must be SHA-256")
    return (
        calls,
        typed_operations,
        safe_preconditions,
        safe_expected,
        safe_invalidation,
        safe_summary,
    )


@dataclass(frozen=True, slots=True)
class ModelPatch:
    patch_id: str
    agent_session_id: str
    turn_id: str
    source_tool_call_ids: tuple[str, ...]
    target_document_id: str
    target_session_id: str
    base_session_revision: int
    draft_revision: int
    operations: tuple[ModelOperation, ...]
    preconditions: object
    expected_changes: object
    invalidation_impact: object
    display_summary: object
    idempotency_key: str
    patch_hash: str
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validated = _validate_common_envelope(
            schema_version=self.schema_version,
            envelope_id=self.patch_id,
            agent_session_id=self.agent_session_id,
            turn_id=self.turn_id,
            source_tool_call_ids=self.source_tool_call_ids,
            target_document_id=self.target_document_id,
            target_session_id=self.target_session_id,
            base_session_revision=self.base_session_revision,
            draft_revision=self.draft_revision,
            operations=self.operations,
            preconditions=self.preconditions,
            expected_changes=self.expected_changes,
            invalidation_impact=self.invalidation_impact,
            display_summary=self.display_summary,
            idempotency_key=self.idempotency_key,
            content_hash=self.patch_hash,
        )
        (
            calls,
            operations,
            preconditions,
            expected,
            invalidation,
            summary,
        ) = validated
        object.__setattr__(self, "source_tool_call_ids", calls)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "expected_changes", expected)
        object.__setattr__(self, "invalidation_impact", invalidation)
        object.__setattr__(self, "display_summary", summary)
        _require_identifier(self.patch_id, "patch_id")
        if self.idempotency_key != _hash(self._idempotency_payload()):
            raise AuthoringContractError("ModelPatch idempotency key does not match")
        if self.patch_hash != _hash(self._hash_payload()):
            raise AuthoringContractError("ModelPatch hash does not match")

    @classmethod
    def create(cls, **values: object) -> ModelPatch:
        core = _envelope_core("patch_id", values)
        serialized = _envelope_hash_payload(core)
        idempotency_key = _hash(_idempotency_payload(serialized))
        payload = {
            **serialized,
            "idempotency_key": idempotency_key,
        }
        patch_hash = _hash(payload)
        return cls(
            **core,
            idempotency_key=idempotency_key,
            patch_hash=patch_hash,
        )

    def _idempotency_payload(self) -> dict[str, object]:
        return _idempotency_payload(self._core_dict())

    def _hash_payload(self) -> dict[str, object]:
        return {
            **self._core_dict(),
            "idempotency_key": self.idempotency_key,
        }

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "patch_id": self.patch_id,
            "agent_session_id": self.agent_session_id,
            "turn_id": self.turn_id,
            "source_tool_call_ids": list(self.source_tool_call_ids),
            "target_document_id": self.target_document_id,
            "target_session_id": self.target_session_id,
            "base_session_revision": self.base_session_revision,
            "draft_revision": self.draft_revision,
            "operations": [item.to_dict() for item in self.operations],
            "preconditions": deepcopy(self.preconditions),
            "expected_changes": deepcopy(self.expected_changes),
            "invalidation_impact": deepcopy(self.invalidation_impact),
            "display_summary": deepcopy(self.display_summary),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "patch_hash": self.patch_hash,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ModelPatch:
        fields = (
            "schema_version",
            "patch_id",
            "agent_session_id",
            "turn_id",
            "source_tool_call_ids",
            "target_document_id",
            "target_session_id",
            "base_session_revision",
            "draft_revision",
            "operations",
            "preconditions",
            "expected_changes",
            "invalidation_impact",
            "display_summary",
            "idempotency_key",
            "patch_hash",
        )
        _strict_fields(record, fields, "ModelPatch")
        return cls(
            schema_version=str(record["schema_version"]),
            patch_id=str(record["patch_id"]),
            agent_session_id=str(record["agent_session_id"]),
            turn_id=str(record["turn_id"]),
            source_tool_call_ids=tuple(record["source_tool_call_ids"]),  # type: ignore[arg-type]
            target_document_id=str(record["target_document_id"]),
            target_session_id=str(record["target_session_id"]),
            base_session_revision=record["base_session_revision"],  # type: ignore[arg-type]
            draft_revision=record["draft_revision"],  # type: ignore[arg-type]
            operations=tuple(
                ModelOperation.from_dict(item)
                for item in record["operations"]  # type: ignore[union-attr]
            ),
            preconditions=record["preconditions"],
            expected_changes=record["expected_changes"],
            invalidation_impact=record["invalidation_impact"],
            display_summary=record["display_summary"],
            idempotency_key=str(record["idempotency_key"]),
            patch_hash=str(record["patch_hash"]),
        )


class ProposalKind(str, Enum):
    GEOMETRY = "geometry"
    MESH = "mesh"
    SOLVE = "solve"
    DESTRUCTIVE_EDIT = "destructive_edit"
    REQUIREMENT_REVIEW = "requirement_review"


@dataclass(frozen=True, slots=True)
class AgentProposal:
    proposal_id: str
    proposal_kind: ProposalKind
    agent_session_id: str
    turn_id: str
    source_tool_call_ids: tuple[str, ...]
    target_document_id: str
    target_session_id: str
    base_session_revision: int
    draft_revision: int
    operations: tuple[ModelOperation, ...]
    preconditions: object
    expected_changes: object
    invalidation_impact: object
    display_summary: object
    idempotency_key: str
    proposal_hash: str
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validated = _validate_common_envelope(
            schema_version=self.schema_version,
            envelope_id=self.proposal_id,
            agent_session_id=self.agent_session_id,
            turn_id=self.turn_id,
            source_tool_call_ids=self.source_tool_call_ids,
            target_document_id=self.target_document_id,
            target_session_id=self.target_session_id,
            base_session_revision=self.base_session_revision,
            draft_revision=self.draft_revision,
            operations=self.operations,
            preconditions=self.preconditions,
            expected_changes=self.expected_changes,
            invalidation_impact=self.invalidation_impact,
            display_summary=self.display_summary,
            idempotency_key=self.idempotency_key,
            content_hash=self.proposal_hash,
        )
        (
            calls,
            operations,
            preconditions,
            expected,
            invalidation,
            summary,
        ) = validated
        object.__setattr__(self, "source_tool_call_ids", calls)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "expected_changes", expected)
        object.__setattr__(self, "invalidation_impact", invalidation)
        object.__setattr__(self, "display_summary", summary)
        _require_identifier(self.proposal_id, "proposal_id")
        try:
            kind = ProposalKind(self.proposal_kind)
        except ValueError as exc:
            raise AuthoringContractError("unknown proposal kind") from exc
        object.__setattr__(self, "proposal_kind", kind)
        if self.idempotency_key != _hash(self._idempotency_payload()):
            raise AuthoringContractError(
                "AgentProposal idempotency key does not match"
            )
        if self.proposal_hash != _hash(self._hash_payload()):
            raise AuthoringContractError("AgentProposal hash does not match")

    @classmethod
    def create(
        cls,
        *,
        proposal_kind: ProposalKind | str,
        **values: object,
    ) -> AgentProposal:
        core = _envelope_core("proposal_id", values)
        core["proposal_kind"] = ProposalKind(proposal_kind).value
        serialized = _envelope_hash_payload(core)
        idempotency_key = _hash(_idempotency_payload(serialized))
        payload = {
            **serialized,
            "idempotency_key": idempotency_key,
        }
        proposal_hash = _hash(payload)
        constructor = dict(core)
        constructor.pop("proposal_kind")
        return cls(
            **constructor,
            proposal_kind=ProposalKind(proposal_kind),
            idempotency_key=idempotency_key,
            proposal_hash=proposal_hash,
        )

    def _idempotency_payload(self) -> dict[str, object]:
        return _idempotency_payload(self._core_dict())

    def _hash_payload(self) -> dict[str, object]:
        return {
            **self._core_dict(),
            "idempotency_key": self.idempotency_key,
        }

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "proposal_kind": self.proposal_kind.value,
            "agent_session_id": self.agent_session_id,
            "turn_id": self.turn_id,
            "source_tool_call_ids": list(self.source_tool_call_ids),
            "target_document_id": self.target_document_id,
            "target_session_id": self.target_session_id,
            "base_session_revision": self.base_session_revision,
            "draft_revision": self.draft_revision,
            "operations": [item.to_dict() for item in self.operations],
            "preconditions": deepcopy(self.preconditions),
            "expected_changes": deepcopy(self.expected_changes),
            "invalidation_impact": deepcopy(self.invalidation_impact),
            "display_summary": deepcopy(self.display_summary),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "proposal_hash": self.proposal_hash,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> AgentProposal:
        fields = (
            "schema_version",
            "proposal_id",
            "proposal_kind",
            "agent_session_id",
            "turn_id",
            "source_tool_call_ids",
            "target_document_id",
            "target_session_id",
            "base_session_revision",
            "draft_revision",
            "operations",
            "preconditions",
            "expected_changes",
            "invalidation_impact",
            "display_summary",
            "idempotency_key",
            "proposal_hash",
        )
        _strict_fields(record, fields, "AgentProposal")
        return cls(
            schema_version=str(record["schema_version"]),
            proposal_id=str(record["proposal_id"]),
            proposal_kind=ProposalKind(str(record["proposal_kind"])),
            agent_session_id=str(record["agent_session_id"]),
            turn_id=str(record["turn_id"]),
            source_tool_call_ids=tuple(record["source_tool_call_ids"]),  # type: ignore[arg-type]
            target_document_id=str(record["target_document_id"]),
            target_session_id=str(record["target_session_id"]),
            base_session_revision=record["base_session_revision"],  # type: ignore[arg-type]
            draft_revision=record["draft_revision"],  # type: ignore[arg-type]
            operations=tuple(
                ModelOperation.from_dict(item)
                for item in record["operations"]  # type: ignore[union-attr]
            ),
            preconditions=record["preconditions"],
            expected_changes=record["expected_changes"],
            invalidation_impact=record["invalidation_impact"],
            display_summary=record["display_summary"],
            idempotency_key=str(record["idempotency_key"]),
            proposal_hash=str(record["proposal_hash"]),
        )


def _envelope_core(id_field: str, values: Mapping[str, object]) -> dict[str, object]:
    expected = {
        id_field,
        "agent_session_id",
        "turn_id",
        "source_tool_call_ids",
        "target_document_id",
        "target_session_id",
        "base_session_revision",
        "draft_revision",
        "operations",
        "preconditions",
        "expected_changes",
        "invalidation_impact",
        "display_summary",
    }
    _strict_fields(values, expected, id_field.removesuffix("_id"))
    operations = tuple(values["operations"])  # type: ignore[arg-type]
    if any(type(item) is not ModelOperation for item in operations):
        raise AuthoringContractError("operations must contain ModelOperation")
    return {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        id_field: values[id_field],
        "agent_session_id": values["agent_session_id"],
        "turn_id": values["turn_id"],
        "source_tool_call_ids": tuple(values["source_tool_call_ids"]),  # type: ignore[arg-type]
        "target_document_id": values["target_document_id"],
        "target_session_id": values["target_session_id"],
        "base_session_revision": values["base_session_revision"],
        "draft_revision": values["draft_revision"],
        "operations": operations,
        "preconditions": _json_copy(values["preconditions"], "preconditions"),
        "expected_changes": _json_copy(
            values["expected_changes"],
            "expected_changes",
        ),
        "invalidation_impact": _json_copy(
            values["invalidation_impact"],
            "invalidation_impact",
        ),
        "display_summary": _json_copy(
            values["display_summary"],
            "display_summary",
        ),
    }


def _idempotency_payload(core: Mapping[str, object]) -> dict[str, object]:
    payload = dict(core)
    payload.pop("patch_id", None)
    payload.pop("proposal_id", None)
    return payload


def _envelope_hash_payload(core: Mapping[str, object]) -> dict[str, object]:
    payload = dict(core)
    payload["operations"] = [
        item.to_dict()
        for item in payload["operations"]  # type: ignore[union-attr]
    ]
    proposal_kind = payload.get("proposal_kind")
    if isinstance(proposal_kind, ProposalKind):
        payload["proposal_kind"] = proposal_kind.value
    return payload


class ProposalState(str, Enum):
    DRAFT = "draft"
    PENDING_CONFIRMATION = "pending_confirmation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProposalPortRecord:
    proposal: AgentProposal
    state: ProposalState
    message: str = ""


class AuthoringPort(Protocol):
    """Port used by the GUI bridge; implementations must not expose Qt."""

    def set_context(self, context: AuthoringContext) -> None: ...

    def present(self, proposal: AgentProposal) -> ProposalPortRecord: ...

    def accept(self, proposal_id: str) -> ProposalPortRecord: ...

    def reject(self, proposal_id: str) -> ProposalPortRecord: ...

    def stale(self, proposal_id: str, reason: str) -> ProposalPortRecord: ...


class FakeAuthoringPort:
    """Deterministic A1 port that never owns or mutates a ModelSession."""

    def __init__(self) -> None:
        self._context: AuthoringContext | None = None
        self._records: dict[str, ProposalPortRecord] = {}
        self.calls: list[tuple[str, str]] = []
        self.accept_error: Exception | None = None

    @property
    def context(self) -> AuthoringContext | None:
        return self._context

    @property
    def records(self) -> tuple[ProposalPortRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def set_context(self, context: AuthoringContext) -> None:
        if type(context) is not AuthoringContext:
            raise AuthoringContractError("context must be AuthoringContext")
        self._context = context
        self.calls.append(("set_context", context.binding.document_id))

    def present(self, proposal: AgentProposal) -> ProposalPortRecord:
        if type(proposal) is not AgentProposal:
            raise AuthoringContractError("proposal must be AgentProposal")
        if proposal.proposal_id in self._records:
            raise AuthoringContractError("proposal_id is already registered")
        record = ProposalPortRecord(
            proposal,
            ProposalState.PENDING_CONFIRMATION,
        )
        self._records[proposal.proposal_id] = record
        self.calls.append(("present", proposal.proposal_id))
        return record

    def accept(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        if self.accept_error is not None:
            raise self.accept_error
        accepted = replace(record, state=ProposalState.ACCEPTED)
        self._records[proposal_id] = accepted
        self.calls.append(("accept", proposal_id))
        return accepted

    def reject(self, proposal_id: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        rejected = replace(record, state=ProposalState.REJECTED)
        self._records[proposal_id] = rejected
        self.calls.append(("reject", proposal_id))
        return rejected

    def stale(self, proposal_id: str, reason: str) -> ProposalPortRecord:
        record = self._pending(proposal_id)
        stale = replace(
            record,
            state=ProposalState.STALE,
            message=_require_text(reason, "reason"),
        )
        self._records[proposal_id] = stale
        self.calls.append(("stale", proposal_id))
        return stale

    def mark_failed(self, proposal_id: str, message: str) -> ProposalPortRecord:
        record = self._records[proposal_id]
        failed = replace(
            record,
            state=ProposalState.FAILED,
            message=_require_text(message, "message"),
        )
        self._records[proposal_id] = failed
        self.calls.append(("failed", proposal_id))
        return failed

    def _pending(self, proposal_id: str) -> ProposalPortRecord:
        normalized = _require_identifier(proposal_id, "proposal_id")
        try:
            record = self._records[normalized]
        except KeyError as exc:
            raise AuthoringContractError("proposal is not registered") from exc
        if record.state is not ProposalState.PENDING_CONFIRMATION:
            raise AuthoringAuthorizationError(
                f"proposal is already {record.state.value}"
            )
        return record


__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "AgentDraft",
    "AgentProposal",
    "AuthoringAuthorizationError",
    "AuthoringContext",
    "AuthoringContractError",
    "AuthoringPort",
    "CapabilitySummary",
    "ClarificationRequiredError",
    "DefinitionSummary",
    "FakeAuthoringPort",
    "LocalModelBinding",
    "MeshSummary",
    "ModelOperation",
    "ModelPatch",
    "OperationKind",
    "PartSummary",
    "ProposalKind",
    "ProposalPortRecord",
    "ProposalState",
    "RequirementEntry",
    "RequirementLedger",
    "RequirementReview",
    "RequirementReviewStatus",
    "RequirementStatus",
    "UnitContextSummary",
]
