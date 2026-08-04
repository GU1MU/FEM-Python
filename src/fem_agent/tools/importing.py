"""Local Abaqus inspection and model construction for Agent V0."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fem import abaqus
from fem.abaqus.deck import AbaqusDeck
from fem.core.model import AnalysisStep, FEMModel

from ..diagnostics import DiagnosticCode, has_errors, make_diagnostic
from ..schemas import Diagnostic, DiagnosticSeverity, ResourceLimits
from .inspection import AbaqusKeywordInspection, inspect_abaqus_keywords


_NOTICE_DIAGNOSTIC_CODES = {
    abaqus.B31_EULER_BERNOULLI_NOTICE_CODE: (
        DiagnosticCode.ABAQUS_B31_EULER_BERNOULLI_APPROXIMATION
    ),
    abaqus.B31_SHARED_NODE_FRAME_NOTICE_CODE: (
        DiagnosticCode.ABAQUS_B31_SHARED_NODE_FRAME_APPROXIMATION
    ),
}


@dataclass(frozen=True)
class AbaqusImportResult:
    """Local import result with an explicitly bounded provider projection.

    ``deck`` and ``model`` are worker-local objects and must never be serialized
    into provider messages or persisted as revision truth.
    """

    keyword_inspection: AbaqusKeywordInspection
    diagnostics: tuple[Diagnostic, ...]
    input_size_bytes: int | None = None
    deck: AbaqusDeck | None = None
    model: FEMModel | None = None
    runnable_step: AnalysisStep | None = None

    @property
    def ok(self) -> bool:
        return self.model is not None and not has_errors(self.diagnostics)

    @property
    def keyword_inventory(self) -> tuple[Mapping[str, Any], ...]:
        return self.keyword_inspection.keyword_inventory

    @property
    def collections_truncated(self) -> bool:
        return self.keyword_inspection.collections_truncated

    def provider_data(self) -> dict[str, Any]:
        """Return counts and safe names without raw deck or model arrays."""

        if self.model is None:
            model_data = {
                "model_name": _model_name(self.deck, None),
                "node_count": len(self.deck.nodes) if self.deck is not None else 0,
                "element_count": (
                    len(self.deck.elements) if self.deck is not None else 0
                ),
                "runnable_step": None,
            }
        else:
            model_data = {
                "model_name": _model_name(self.deck, self.model),
                "node_count": int(self.model.mesh.num_nodes),
                "element_count": int(self.model.mesh.num_elements),
                "runnable_step": (
                    _bounded_text(self.runnable_step.name)
                    if self.runnable_step is not None
                    else None
                ),
            }
        return {
            **model_data,
            "keyword_inventory": [
                dict(item) for item in self.keyword_inspection.keyword_inventory
            ],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "input_size_bytes": self.input_size_bytes,
            "collections_truncated": self.collections_truncated,
        }


def inspect_abaqus(
    path: str | Path,
    *,
    max_inventory_entries: int = 64,
    resource_limits: ResourceLimits | None = None,
) -> AbaqusImportResult:
    """Inspect, parse, and build one local Abaqus input artifact."""

    keyword_inspection = inspect_abaqus_keywords(
        path,
        max_inventory_entries=max_inventory_entries,
        resource_limits=resource_limits,
    )
    diagnostics = list(keyword_inspection.diagnostics)
    deck: AbaqusDeck | None = None
    model: FEMModel | None = None
    try:
        input_size_bytes = Path(path).stat().st_size
    except OSError:
        input_size_bytes = None

    if has_errors(diagnostics):
        return AbaqusImportResult(
            keyword_inspection=keyword_inspection,
            diagnostics=_deduplicate_diagnostics(diagnostics),
            input_size_bytes=input_size_bytes,
        )

    try:
        deck = abaqus.parse_file(path)
        built = abaqus.build_model_with_report(deck)
        model = built.model
        diagnostics.extend(_notice_diagnostics(built.notices))
    except Exception:
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.IMPORT_FAILED,
                "The attached Abaqus input could not be parsed and built locally.",
                source="abaqus_import",
                remediation="Review the capability report and input structure.",
            )
        )

    runnable: tuple[AnalysisStep, ...] = ()
    if model is not None:
        runnable = runnable_steps(model)
        if not runnable:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.UNSUPPORTED_PROCEDURE,
                    "The imported model does not contain one runnable non-Initial step.",
                    source="abaqus_import",
                    entity="step",
                )
            )
        elif len(runnable) > 1:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.MULTI_STEP_HISTORY_UNSUPPORTED,
                    "The imported model contains multiple runnable steps.",
                    source="abaqus_import",
                    entity="step",
                    remediation="Provide an input deck with one runnable static step.",
                )
            )

    normalized = _deduplicate_diagnostics(diagnostics)
    return AbaqusImportResult(
        keyword_inspection=keyword_inspection,
        diagnostics=normalized,
        input_size_bytes=input_size_bytes,
        deck=deck,
        model=model,
        runnable_step=runnable[0] if len(runnable) == 1 else None,
    )


def runnable_steps(model: FEMModel) -> tuple[AnalysisStep, ...]:
    """Return explicit runnable steps using the solver's Initial convention."""

    return tuple(
        step
        for step in model.steps
        if str(step.name).strip().casefold() != "initial"
    )


def _notice_diagnostics(notices: tuple[Any, ...]) -> tuple[Diagnostic, ...]:
    return tuple(
        make_diagnostic(
            _NOTICE_DIAGNOSTIC_CODES.get(
                notice.code,
                DiagnosticCode.IMPORT_APPROXIMATION,
            ),
            notice.message,
            source="abaqus_import",
            severity=DiagnosticSeverity.WARNING,
            entity="beam",
        )
        for notice in notices
    )


def _deduplicate_diagnostics(
    diagnostics: list[Diagnostic],
) -> tuple[Diagnostic, ...]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, str | None, str]] = set()
    for diagnostic in diagnostics:
        key = (diagnostic.code, diagnostic.entity, diagnostic.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return tuple(result)


def _model_name(deck: AbaqusDeck | None, model: FEMModel | None) -> str:
    value = (
        getattr(model, "name", None)
        if model is not None
        else getattr(deck, "name", None)
    )
    return _bounded_text(value or "abaqus_model")


def _bounded_text(value: object, *, max_length: int = 128) -> str:
    text = "".join(
        " "
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in str(value)
    ).strip()
    return text[:max_length] or "unnamed"


__all__ = [
    "AbaqusImportResult",
    "inspect_abaqus",
    "runnable_steps",
]
