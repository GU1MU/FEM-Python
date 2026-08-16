"""Strict contracts for Agent-authored result exports and display context.

Phase 1 exposes exactly two tools: ``export_accepted_result_csv`` (write the
currently accepted READY result table into the user workspace) and the
read-only ``read_result_display_context`` (enumerate the READY field catalog
and current display settings).  Both contracts stay fail-closed: every DTO is
bounded, every schema is closed, and a missing user workspace returns one
short diagnostic that the engine relays verbatim without retrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import json
import math
from typing import Mapping, Protocol

from .result_authoring import AcceptedResultSource


EXPORT_AUTHORING_SCHEMA_VERSION = "1.0"
EXPORT_CSV_TOOL_NAME = "export_accepted_result_csv"
RESULT_DISPLAY_CONTEXT_TOOL_NAME = "read_result_display_context"
NO_WORKSPACE_DIAGNOSTIC_CODE = "export.no_workspace"
NO_WORKSPACE_DIAGNOSTIC_MESSAGE = (
    "尚未选择工作区，请先执行 /workspace 选择目录，"
    "导出文件将保存到该目录下的 agent_exports 中"
)
EXPORT_KINDS = {"csv", "png"}
EXPORT_NAME_MAX_LENGTH = 180
FIELD_REF_MAX_LENGTH = 256
COMPONENT_MAX_LENGTH = 128
DISPLAY_SETTING_KEYS = {
    "edge_mode",
    "edge_style",
    "edge_width",
    "number_format",
    "decimals",
    "orientation",
    "legend_font",
    "legend_font_size",
    "legend",
    "show_ids",
    "show_coordinate_system",
    "edges",
}
CONTOUR_SETTING_KEYS = {
    "manual",
    "minimum",
    "maximum",
    "colormap",
    "style",
    "render_mode",
    "levels",
    "show_minimum",
    "show_maximum",
    "averaging_threshold",
}


class ExportAuthoringError(ValueError):
    """Fail-closed export contract error."""


def _strict_mapping(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be strings")
    if set(value) != expected:
        raise ExportAuthoringError(f"{label} fields do not match the schema")
    return value


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _exact_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if value != value.strip() or not value:
        raise ExportAuthoringError(
            f"{label} must be nonblank without surrounding whitespace"
        )
    if len(value) > maximum:
        raise ExportAuthoringError(f"{label} exceeds its bound")
    if "\x00" in value:
        raise ExportAuthoringError(f"{label} contains a null character")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ExportAuthoringError(f"{label} must be non-negative")
    return value


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ExportAuthoringError(f"{label} must be finite")
    return numeric


def _sha256_digest(value: object, label: str) -> str:
    digest = _exact_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ExportAuthoringError(f"{label} must be lowercase hex sha256")
    return digest


@dataclass(frozen=True, slots=True)
class ExportFileReceipt:
    """Proof of one landed export file; only workspace-relative paths leave."""

    workspace_relative_path: str
    filename: str
    sha256: str
    size_bytes: int
    kind: str

    def __post_init__(self) -> None:
        _bounded_text(
            self.workspace_relative_path,
            "workspace_relative_path",
            maximum=512,
        )
        if "\\" in self.workspace_relative_path:
            raise ExportAuthoringError(
                "workspace_relative_path must use posix separators"
            )
        _bounded_text(self.filename, "filename", maximum=255)
        _sha256_digest(self.sha256, "sha256")
        _nonnegative_integer(self.size_bytes, "size_bytes")
        if self.kind not in EXPORT_KINDS:
            raise ExportAuthoringError("kind must be one of csv or png")

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace_relative_path": self.workspace_relative_path,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExportFileReceipt:
        data = _strict_mapping(
            value,
            {
                "workspace_relative_path",
                "filename",
                "sha256",
                "size_bytes",
                "kind",
            },
            "export receipt",
        )
        return cls(
            workspace_relative_path=_exact_string(
                data["workspace_relative_path"],
                "workspace_relative_path",
            ),
            filename=_exact_string(data["filename"], "filename"),
            sha256=_sha256_digest(data["sha256"], "sha256"),
            size_bytes=_nonnegative_integer(data["size_bytes"], "size_bytes"),
            kind=_exact_string(data["kind"], "kind"),
        )


@dataclass(frozen=True, slots=True)
class ExportDiagnostic:
    """Bounded, stable export failure without any file identity."""

    code: str
    message: str
    retryable: bool
    clarification_required: bool
    phase: str = "export"

    def __post_init__(self) -> None:
        _bounded_text(self.code, "code", maximum=128)
        _bounded_text(self.message, "message", maximum=1024)
        _bounded_text(self.phase, "phase", maximum=16)
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be boolean")
        if type(self.clarification_required) is not bool:
            raise TypeError("clarification_required must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "clarification_required": self.clarification_required,
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class ExportCsvRequest:
    """One complete CSV export request bound to an exact accepted result."""

    expected_source: AcceptedResultSource
    expected_materialization_generation: int
    field_ref: str
    component: str
    name: str | None = None

    def __post_init__(self) -> None:
        if type(self.expected_source) is not AcceptedResultSource:
            raise TypeError("expected_source must be AcceptedResultSource")
        _nonnegative_integer(
            self.expected_materialization_generation,
            "expected_materialization_generation",
        )
        _bounded_text(self.field_ref, "field_ref", maximum=FIELD_REF_MAX_LENGTH)
        _bounded_text(
            self.component,
            "component",
            maximum=COMPONENT_MAX_LENGTH,
        )
        if self.name is not None:
            _bounded_text(self.name, "name", maximum=EXPORT_NAME_MAX_LENGTH)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXPORT_AUTHORING_SCHEMA_VERSION,
            "expected_source": self.expected_source.to_dict(),
            "expected_materialization_generation": (
                self.expected_materialization_generation
            ),
            "field_ref": self.field_ref,
            "component": self.component,
            "name": self.name,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, value: object) -> ExportCsvRequest:
        if not isinstance(value, Mapping):
            raise TypeError("export csv request must be an object")
        if any(type(key) is not str for key in value):
            raise TypeError("export csv request keys must be strings")
        allowed = {
            "schema_version",
            "expected_source",
            "expected_materialization_generation",
            "field_ref",
            "component",
            "name",
        }
        required = allowed - {"name"}
        if not required <= set(value) <= allowed:
            raise ExportAuthoringError(
                "export csv request fields do not match the schema"
            )
        data = value
        schema_version = _exact_string(data["schema_version"], "schema_version")
        if schema_version != EXPORT_AUTHORING_SCHEMA_VERSION:
            raise ExportAuthoringError(
                "export csv request has an unsupported schema version"
            )
        name = data["name"]
        return cls(
            expected_source=AcceptedResultSource.from_dict(
                data["expected_source"]
            ),
            expected_materialization_generation=_nonnegative_integer(
                data["expected_materialization_generation"],
                "expected_materialization_generation",
            ),
            field_ref=_bounded_text(
                data["field_ref"],
                "field_ref",
                maximum=FIELD_REF_MAX_LENGTH,
            ),
            component=_bounded_text(
                data["component"],
                "component",
                maximum=COMPONENT_MAX_LENGTH,
            ),
            name=None if name is None else _exact_string(name, "name"),
        )


@dataclass(frozen=True, slots=True)
class ExportCsvResponse:
    """Exactly one landed receipt or one-or-more bounded diagnostics."""

    receipt: ExportFileReceipt | None = None
    diagnostics: tuple[ExportDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.receipt is not None and type(self.receipt) is not ExportFileReceipt:
            raise TypeError("receipt must be ExportFileReceipt or None")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not ExportDiagnostic for item in self.diagnostics
        ):
            raise TypeError("diagnostics must be a tuple of ExportDiagnostic")
        if (self.receipt is None) == (not self.diagnostics):
            raise ExportAuthoringError(
                "response requires exactly one receipt or diagnostics"
            )
        if len(self.diagnostics) > 8:
            raise ExportAuthoringError("response diagnostics exceed the bound")

    @property
    def ok(self) -> bool:
        return self.receipt is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXPORT_AUTHORING_SCHEMA_VERSION,
            "ok": self.ok,
            "export_receipt": (
                None if self.receipt is None else self.receipt.to_dict()
            ),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def success(cls, receipt: ExportFileReceipt) -> ExportCsvResponse:
        return cls(receipt=receipt)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool,
        clarification_required: bool,
    ) -> ExportCsvResponse:
        return cls(
            diagnostics=(
                ExportDiagnostic(
                    code=code,
                    message=message,
                    retryable=retryable,
                    clarification_required=clarification_required,
                ),
            )
        )

    @classmethod
    def no_workspace(cls) -> ExportCsvResponse:
        return cls.failure(
            NO_WORKSPACE_DIAGNOSTIC_CODE,
            NO_WORKSPACE_DIAGNOSTIC_MESSAGE,
            retryable=False,
            clarification_required=True,
        )


@dataclass(frozen=True, slots=True)
class ResultDisplayField:
    """One READY field identity exposed to the Agent with a stable ref."""

    field_ref: str
    display_name: str
    components: tuple[str, ...]
    unit: str

    def __post_init__(self) -> None:
        _bounded_text(self.field_ref, "field_ref", maximum=FIELD_REF_MAX_LENGTH)
        _bounded_text(self.display_name, "display_name", maximum=256)
        if type(self.unit) is not str or len(self.unit) > 128 or "\x00" in self.unit:
            raise ExportAuthoringError("unit must be a string within its bound")
        if type(self.components) is not tuple:
            raise TypeError("components must be a tuple")
        if not self.components or len(self.components) > 32:
            raise ExportAuthoringError(
                "components must contain from 1 through 32 values"
            )
        for component in self.components:
            _bounded_text(
                component,
                "component",
                maximum=COMPONENT_MAX_LENGTH,
            )
        if len(set(self.components)) != len(self.components):
            raise ExportAuthoringError("components must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "field_ref": self.field_ref,
            "display_name": self.display_name,
            "components": list(self.components),
            "unit": self.unit,
        }


def _bounded_setting_value(value: object, label: str) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return _finite_real(value, label)
    if type(value) is str:
        return _bounded_text(value, label, maximum=128)
    raise TypeError(f"{label} must be a JSON scalar")


def _bounded_settings(
    value: object,
    allowed_keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be strings")
    if not set(value) <= allowed_keys:
        raise ExportAuthoringError(f"{label} contains unsupported keys")
    if len(value) > 32:
        raise ExportAuthoringError(f"{label} exceeds the key bound")
    return {
        key: _bounded_setting_value(item, f"{label}.{key}")
        for key, item in value.items()
    }


@dataclass(frozen=True, slots=True)
class ResultDisplayContext:
    """Read-only snapshot of the READY field catalog and display state."""

    fields: tuple[ResultDisplayField, ...]
    display_settings: Mapping[str, object]
    contour_settings: Mapping[str, object]
    selected_field_ref: str | None
    selected_component: str | None
    deformation_scale: float

    def __post_init__(self) -> None:
        if type(self.fields) is not tuple or any(
            type(item) is not ResultDisplayField for item in self.fields
        ):
            raise TypeError("fields must be a tuple of ResultDisplayField")
        if len(self.fields) > 64:
            raise ExportAuthoringError("field catalog exceeds the bound")
        refs = [item.field_ref for item in self.fields]
        if len(set(refs)) != len(refs):
            raise ExportAuthoringError("field_ref values must be unique")
        object.__setattr__(
            self,
            "display_settings",
            _bounded_settings(
                self.display_settings,
                DISPLAY_SETTING_KEYS,
                "display_settings",
            ),
        )
        object.__setattr__(
            self,
            "contour_settings",
            _bounded_settings(
                self.contour_settings,
                CONTOUR_SETTING_KEYS,
                "contour_settings",
            ),
        )
        if self.selected_field_ref is not None:
            _bounded_text(
                self.selected_field_ref,
                "selected_field_ref",
                maximum=FIELD_REF_MAX_LENGTH,
            )
            if self.selected_field_ref not in {item.field_ref for item in self.fields}:
                raise ExportAuthoringError(
                    "selected_field_ref must come from the READY catalog"
                )
        if self.selected_component is not None:
            _bounded_text(
                self.selected_component,
                "selected_component",
                maximum=COMPONENT_MAX_LENGTH,
            )
            if self.selected_field_ref is None:
                raise ExportAuthoringError(
                    "selected_component requires selected_field_ref"
                )
            selected = next(
                item
                for item in self.fields
                if item.field_ref == self.selected_field_ref
            )
            if self.selected_component not in selected.components:
                raise ExportAuthoringError(
                    "selected_component must come from the selected field"
                )
        _finite_real(self.deformation_scale, "deformation_scale")
        if self.deformation_scale < 0.0:
            raise ExportAuthoringError(
                "deformation_scale must be non-negative"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "fields": [item.to_dict() for item in self.fields],
            "display_settings": dict(self.display_settings),
            "contour_settings": dict(self.contour_settings),
            "selected_field_ref": self.selected_field_ref,
            "selected_component": self.selected_component,
            "deformation_scale": self.deformation_scale,
        }


@dataclass(frozen=True, slots=True)
class ResultDisplayContextResponse:
    """Exactly one display-context snapshot or bounded diagnostics."""

    context: ResultDisplayContext | None = None
    diagnostics: tuple[ExportDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.context is not None
            and type(self.context) is not ResultDisplayContext
        ):
            raise TypeError("context must be ResultDisplayContext or None")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not ExportDiagnostic for item in self.diagnostics
        ):
            raise TypeError("diagnostics must be a tuple of ExportDiagnostic")
        if (self.context is None) == (not self.diagnostics):
            raise ExportAuthoringError(
                "response requires exactly one context or diagnostics"
            )
        if len(self.diagnostics) > 8:
            raise ExportAuthoringError("response diagnostics exceed the bound")

    @property
    def ok(self) -> bool:
        return self.context is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXPORT_AUTHORING_SCHEMA_VERSION,
            "ok": self.ok,
            "display_context": (
                None if self.context is None else self.context.to_dict()
            ),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def success(
        cls,
        context: ResultDisplayContext,
    ) -> ResultDisplayContextResponse:
        return cls(context=context)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool,
        clarification_required: bool,
    ) -> ResultDisplayContextResponse:
        return cls(
            diagnostics=(
                ExportDiagnostic(
                    code=code,
                    message=message,
                    retryable=retryable,
                    clarification_required=clarification_required,
                ),
            )
        )


class AgentExportPort(Protocol):
    """Cross-layer export protocol; implementations return bounded DTOs."""

    def export_accepted_result_csv(
        self,
        request: ExportCsvRequest,
    ) -> ExportCsvResponse: ...

    def read_result_display_context(self) -> ResultDisplayContextResponse: ...


class FakeAgentExportPort:
    """Deterministic Fake Provider boundary for export contract tests."""

    def __init__(
        self,
        *,
        export_response: ExportCsvResponse | None = None,
        display_context_response: ResultDisplayContextResponse | None = None,
    ) -> None:
        self._export_response = export_response
        self._display_context_response = display_context_response
        self.export_calls: list[ExportCsvRequest] = []
        self.display_context_calls = 0

    def register_export(self, response: ExportCsvResponse) -> None:
        if type(response) is not ExportCsvResponse:
            raise TypeError("response must be ExportCsvResponse")
        self._export_response = response

    def register_display_context(
        self,
        response: ResultDisplayContextResponse,
    ) -> None:
        if type(response) is not ResultDisplayContextResponse:
            raise TypeError("response must be ResultDisplayContextResponse")
        self._display_context_response = response

    def export_accepted_result_csv(
        self,
        request: ExportCsvRequest,
    ) -> ExportCsvResponse:
        if type(request) is not ExportCsvRequest:
            raise TypeError("request must be ExportCsvRequest")
        self.export_calls.append(request)
        if self._export_response is not None:
            return self._export_response
        return ExportCsvResponse.failure(
            "export.not_configured",
            "The export port is not configured.",
            retryable=False,
            clarification_required=True,
        )

    def read_result_display_context(self) -> ResultDisplayContextResponse:
        self.display_context_calls += 1
        if self._display_context_response is not None:
            return self._display_context_response
        return ResultDisplayContextResponse.failure(
            "export.context.not_configured",
            "The display context port is not configured.",
            retryable=False,
            clarification_required=True,
        )


class AgentExportBridge:
    """Strict model-callable export boundary over an injected local port."""

    def __init__(self, port: AgentExportPort) -> None:
        if not all(
            callable(getattr(port, name, None))
            for name in (
                "export_accepted_result_csv",
                "read_result_display_context",
            )
        ):
            raise TypeError("port must implement the Agent export protocol")
        self._port = port

    @property
    def port(self) -> AgentExportPort:
        return self._port

    def export_csv(
        self,
        request: ExportCsvRequest | Mapping[str, object],
    ) -> ExportCsvResponse:
        normalized = (
            request
            if type(request) is ExportCsvRequest
            else ExportCsvRequest.from_dict(request)
        )
        response = self._port.export_accepted_result_csv(normalized)
        if type(response) is not ExportCsvResponse:
            raise TypeError("export port must return ExportCsvResponse")
        return response

    def display_context(self) -> ResultDisplayContextResponse:
        response = self._port.read_result_display_context()
        if type(response) is not ResultDisplayContextResponse:
            raise TypeError(
                "export port must return ResultDisplayContextResponse"
            )
        return response


_SOURCE_PROPERTIES: dict[str, object] = {
    "result_id": {"type": "string", "minLength": 1, "maxLength": 256},
    "session_id": {"type": "string", "minLength": 1, "maxLength": 256},
    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 256},
    "model_revision": {"type": "integer", "minimum": 0},
    "step_name": {"type": "string", "minLength": 1, "maxLength": 256},
    "run_id": {"type": "string", "minLength": 1, "maxLength": 256},
}


def export_result_csv_tool_schema() -> dict[str, object]:
    """Return the closed schema for export_accepted_result_csv."""

    properties: dict[str, object] = {
        "schema_version": {"type": "string", "const": EXPORT_AUTHORING_SCHEMA_VERSION},
        "expected_source": {
            "type": "object",
            "additionalProperties": False,
            "properties": _SOURCE_PROPERTIES,
            "required": sorted(_SOURCE_PROPERTIES),
        },
        "expected_materialization_generation": {
            "type": "integer",
            "minimum": 0,
        },
        "field_ref": {
            "type": "string",
            "minLength": 1,
            "maxLength": FIELD_REF_MAX_LENGTH,
        },
        "component": {
            "type": "string",
            "minLength": 1,
            "maxLength": COMPONENT_MAX_LENGTH,
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": EXPORT_NAME_MAX_LENGTH,
        },
    }
    return {
        "name": EXPORT_CSV_TOOL_NAME,
        "description": (
            "Export the currently accepted READY result table for one exact "
            "field_ref and component as a CSV file into agent_exports under "
            "the selected user workspace. The field_ref and component must "
            "come from read_result_display_context, and the source identity "
            "block plus expected_materialization_generation must match the "
            "values returned by the result catalogs. The receipt returns only "
            "the workspace-relative path. If the response carries the "
            "export.no_workspace diagnostic, relay that exact short message "
            "to the user in one sentence and do not retry the export."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": [
                "schema_version",
                "expected_source",
                "expected_materialization_generation",
                "field_ref",
                "component",
            ],
        },
    }


def result_display_context_tool_schema() -> dict[str, object]:
    """Return the closed schema for read_result_display_context."""

    return {
        "name": RESULT_DISPLAY_CONTEXT_TOOL_NAME,
        "description": (
            "Read the current READY result display context: the field "
            "catalog with stable field_ref values, display names, and "
            "available components, plus the current display and contour "
            "settings, the selected field and component, and the current "
            "deformation scale. Fields that are not READY are never listed."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
    }


__all__ = [
    "COMPONENT_MAX_LENGTH",
    "CONTOUR_SETTING_KEYS",
    "DISPLAY_SETTING_KEYS",
    "EXPORT_AUTHORING_SCHEMA_VERSION",
    "EXPORT_CSV_TOOL_NAME",
    "EXPORT_KINDS",
    "EXPORT_NAME_MAX_LENGTH",
    "FIELD_REF_MAX_LENGTH",
    "NO_WORKSPACE_DIAGNOSTIC_CODE",
    "NO_WORKSPACE_DIAGNOSTIC_MESSAGE",
    "RESULT_DISPLAY_CONTEXT_TOOL_NAME",
    "AgentExportBridge",
    "AgentExportPort",
    "ExportAuthoringError",
    "ExportCsvRequest",
    "ExportCsvResponse",
    "ExportDiagnostic",
    "ExportFileReceipt",
    "FakeAgentExportPort",
    "ResultDisplayContext",
    "ResultDisplayContextResponse",
    "ResultDisplayField",
    "export_result_csv_tool_schema",
    "result_display_context_tool_schema",
]
