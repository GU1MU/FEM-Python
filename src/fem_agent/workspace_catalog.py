"""Strict provider-safe contracts for the local FEM document workspace."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol


WORKSPACE_DOCUMENTS_TOOL_NAME = "read_workspace_documents"
WORKSPACE_DOCUMENT_CATALOG_SCHEMA_VERSION = "1.0"
WORKSPACE_DOCUMENT_CATALOG_LIMIT = 128
WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES = 24_000


def _text(value: object, label: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    normalized = value.strip()
    if len(normalized.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds its bound")
    return normalized


@dataclass(frozen=True, slots=True)
class WorkspaceDocumentIdentity:
    document_id: str
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, "document_id"))
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))

    def to_dict(self) -> dict[str, object]:
        return {"document_id": self.document_id, "session_id": self.session_id}


@dataclass(frozen=True, slots=True)
class WorkspaceDocumentSummary:
    target: WorkspaceDocumentIdentity
    session_revision: int
    document_kind: str
    source_kind: str
    display_name: str
    model_name: str | None

    def __post_init__(self) -> None:
        if type(self.target) is not WorkspaceDocumentIdentity:
            raise TypeError("target must be WorkspaceDocumentIdentity")
        if type(self.session_revision) is not int or self.session_revision < 0:
            raise ValueError("session_revision must be non-negative")
        for field_name in ("document_kind", "source_kind", "display_name"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if self.document_kind not in {"model", "result"}:
            raise ValueError("document_kind must be model or result")
        if self.source_kind not in {"blank", "native", "imported", "result"}:
            raise ValueError("source_kind is unsupported")
        if self.model_name is not None:
            object.__setattr__(self, "model_name", _text(self.model_name, "model_name"))

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "session_revision": self.session_revision,
            "document_kind": self.document_kind,
            "source_kind": self.source_kind,
            "display_name": self.display_name,
            "model_name": self.model_name,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceDocumentCatalog:
    active_target: WorkspaceDocumentIdentity | None
    documents: tuple[WorkspaceDocumentSummary, ...]
    truncated: bool = False
    omitted_document_count: int = 0

    def __post_init__(self) -> None:
        if self.active_target is not None and type(self.active_target) is not WorkspaceDocumentIdentity:
            raise TypeError("active_target must be WorkspaceDocumentIdentity or None")
        if type(self.documents) is not tuple or any(
            type(item) is not WorkspaceDocumentSummary for item in self.documents
        ):
            raise TypeError("documents must be a tuple of WorkspaceDocumentSummary")
        if len(self.documents) > WORKSPACE_DOCUMENT_CATALOG_LIMIT:
            raise ValueError("workspace document catalog exceeds its bound")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be boolean")
        if type(self.omitted_document_count) is not int or self.omitted_document_count < 0:
            raise ValueError("omitted_document_count must be non-negative")
        if self.truncated != (self.omitted_document_count > 0):
            raise ValueError("truncation metadata is inconsistent")
        identities = tuple(item.target for item in self.documents)
        if len(set(identities)) != len(identities):
            raise ValueError("workspace document identities must be unique")
        if self.active_target is not None and self.active_target not in identities:
            raise ValueError("active_target must be retained in documents")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKSPACE_DOCUMENT_CATALOG_SCHEMA_VERSION,
            "active_target": None if self.active_target is None else self.active_target.to_dict(),
            "documents": [item.to_dict() for item in self.documents],
            "truncated": self.truncated,
            "omitted_document_count": self.omitted_document_count,
        }


class WorkspaceCatalogPort(Protocol):
    def catalog(self) -> WorkspaceDocumentCatalog: ...


class WorkspaceCatalogBridge:
    def __init__(self, port: WorkspaceCatalogPort) -> None:
        if not callable(getattr(port, "catalog", None)):
            raise TypeError("port must implement WorkspaceCatalogPort")
        self._port = port

    @property
    def port(self) -> WorkspaceCatalogPort:
        return self._port

    def catalog(self) -> WorkspaceDocumentCatalog:
        value = self._port.catalog()
        if type(value) is not WorkspaceDocumentCatalog:
            raise TypeError("workspace catalog port returned an invalid DTO")
        encoded = json.dumps(
            value.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES:
            raise ValueError("workspace catalog exceeds its provider-safe budget")
        return value


class FakeWorkspaceCatalogPort:
    def __init__(self, catalog: WorkspaceDocumentCatalog) -> None:
        if type(catalog) is not WorkspaceDocumentCatalog:
            raise TypeError("catalog must be WorkspaceDocumentCatalog")
        self.value = catalog
        self.calls = 0

    def catalog(self) -> WorkspaceDocumentCatalog:
        self.calls += 1
        return self.value


def workspace_documents_tool_schema() -> dict[str, object]:
    return {
        "name": WORKSPACE_DOCUMENTS_TOOL_NAME,
        "description": "Read the bounded local model/result document directory.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


read_workspace_documents_tool_schema = workspace_documents_tool_schema


__all__ = [
    "FakeWorkspaceCatalogPort",
    "WORKSPACE_DOCUMENTS_TOOL_NAME",
    "WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES",
    "WORKSPACE_DOCUMENT_CATALOG_LIMIT",
    "WORKSPACE_DOCUMENT_CATALOG_SCHEMA_VERSION",
    "WorkspaceCatalogBridge",
    "WorkspaceCatalogPort",
    "WorkspaceDocumentCatalog",
    "WorkspaceDocumentIdentity",
    "WorkspaceDocumentSummary",
    "read_workspace_documents_tool_schema",
    "workspace_documents_tool_schema",
]
