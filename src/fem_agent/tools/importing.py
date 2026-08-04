"""Local Abaqus inspection and model construction for Agent V0."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fem.core.model import AnalysisStep, FEMModel
from fem.io import inp

from ..diagnostics import DiagnosticCode, has_errors, make_diagnostic
from ..schemas import Diagnostic, ResourceLimits
from .inspection import AbaqusKeywordInspection, inspect_abaqus_keywords


@dataclass(frozen=True)
class AbaqusImportResult:
    """Local import result with an explicitly bounded provider projection.

    The model and source summary are worker-local objects and must never be
    serialized into provider messages or persisted as revision truth.
    """

    keyword_inspection: AbaqusKeywordInspection
    diagnostics: tuple[Diagnostic, ...]
    input_size_bytes: int | None = None
    source_summary: inp.InpSourceSummary | None = None
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
                "model_name": "abaqus_model",
                "node_count": self.keyword_inspection.node_record_count,
                "element_count": self.keyword_inspection.element_record_count,
                "runnable_step": None,
            }
        else:
            model_data = {
                "model_name": _model_name(self.model),
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
    model: FEMModel | None = None
    source_summary: inp.InpSourceSummary | None = None
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
        imported = inp.read_with_report(path)
        source_summary = imported.source_summary
        model = imported.model
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
        source_summary=source_summary,
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


def _model_name(model: FEMModel | None) -> str:
    value = getattr(model, "name", None)
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
