"""Privacy-safe provider-round auditing with deferred atomic storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStore, atomic_write_json, read_json_file, safe_child
from .conversation_store import json_storage_size_bytes
from .providers.base import ToolCall
from .tools.registry import tool_schema_hash


_TOOL_AUDIT_SCHEMA_VERSION = 2


def build_round_audit_entry(
    session_id: str,
    snapshot: Mapping[str, Any],
    revision: int,
    available_tools: Sequence[object],
    tool_calls: Sequence[ToolCall],
) -> dict[str, Any]:
    """Project identities and call flags without arguments or tool results."""

    tool_names = tuple(
        sorted(
            {
                str(getattr(item, "name", ""))
                for item in available_tools
                if isinstance(getattr(item, "name", None), str)
            }
        )
    )
    schema_hashes = {
        name: tool_schema_hash(item)
        for item in available_tools
        if isinstance(getattr(item, "name", None), str)
        for name in (str(item.name),)
    }
    called_names = tuple(
        sorted(
            {str(item.name) for item in tool_calls if isinstance(item, ToolCall)}
        )
    )
    entry = {
        "session_id": session_id,
        "workflow_stage": snapshot.get("workflow_stage"),
        "revision": revision,
        "published_tool_names": list(tool_names),
        "schema_hashes": schema_hashes,
        "tool_call_flags": {
            "provider_called": bool(tool_calls),
            "called_tool_names": list(called_names),
            "accepted_tool_names": [],
            "read_tool_called": any(
                name.startswith("read_") for name in called_names
            ),
            "prepare_tool_called": any(
                name.startswith("prepare_") for name in called_names
            ),
        },
    }
    return entry


class ToolAuditStore:
    """Write bounded audit records under the engine-owned write lock."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        max_storage_bytes: int,
    ) -> None:
        self.artifacts = artifacts
        self.max_storage_bytes = max_storage_bytes

    def path(self, session_id: str) -> Path:
        session = self.artifacts.session_path(session_id)
        return safe_child(session, "tool-audit.json")

    def persist(
        self,
        session_id: str,
        new_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically persist a turn batch while the caller holds its write lock."""

        if not new_entries:
            return
        path = self.path(session_id)
        entries: list[dict[str, Any]] = []
        if path.exists():
            payload = read_json_file(
                path,
                max_bytes=self.max_storage_bytes,
            )
            if (
                payload.get("schema_version") != _TOOL_AUDIT_SCHEMA_VERSION
                or not isinstance(payload.get("entries"), list)
            ):
                raise ValueError("tool audit storage is invalid")
            entries = [
                dict(item)
                for item in payload["entries"]
                if isinstance(item, Mapping)
            ]
        entries.extend(dict(item) for item in new_entries)
        entries = entries[-512:]
        payload = {
            "schema_version": _TOOL_AUDIT_SCHEMA_VERSION,
            "entries": entries,
        }
        while (
            len(entries) > 1
            and json_storage_size_bytes(payload) > self.max_storage_bytes
        ):
            entries.pop(0)
            payload = {
                "schema_version": _TOOL_AUDIT_SCHEMA_VERSION,
                "entries": entries,
            }
        if json_storage_size_bytes(payload) > self.max_storage_bytes:
            # Keep the round identity and call flags while deterministically
            # clipping the published schema catalog.  No arguments/results are
            # ever retained in this file.
            entry = entries[-1]
            names = list(entry.get("published_tool_names", ()))
            hashes = dict(entry.get("schema_hashes", {}))
            while (
                names
                and json_storage_size_bytes(payload) > self.max_storage_bytes
            ):
                removed = names.pop()
                hashes.pop(removed, None)
                entry["published_tool_names"] = names
                entry["schema_hashes"] = hashes
            payload = {
                "schema_version": _TOOL_AUDIT_SCHEMA_VERSION,
                "entries": entries,
            }
        if json_storage_size_bytes(payload) > self.max_storage_bytes:
            raise ValueError("one tool audit entry exceeds its storage limit")
        atomic_write_json(
            path,
            payload,
            overwrite=path.exists(),
        )
