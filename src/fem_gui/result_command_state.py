"""Small pending result requests owned by one workspace document."""

from __future__ import annotations

from dataclasses import dataclass

from fem.application.results import ResultQuery, ResultSourceKey, ScalarFieldSelection


@dataclass(frozen=True, slots=True)
class PendingResultMaterialization:
    """Identify a request without retaining providers or worker payloads."""

    source: ResultSourceKey
    generation: int
    request: ScalarFieldSelection | ResultQuery


@dataclass(slots=True)
class DocumentResultCommandState:
    """Track the one materialization permitted by a document controller."""

    materialization: PendingResultMaterialization | None = None
