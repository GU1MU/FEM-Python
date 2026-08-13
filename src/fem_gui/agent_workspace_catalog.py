"""Provider-safe workspace directory adapter for :class:`FEMWorkspace`."""

from __future__ import annotations

import json

from fem_agent.workspace_catalog import (
    WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES,
    WORKSPACE_DOCUMENT_CATALOG_LIMIT,
    WorkspaceCatalogBridge,
    WorkspaceDocumentCatalog,
    WorkspaceDocumentIdentity,
    WorkspaceDocumentLineage,
    WorkspaceDocumentSummary,
)

from .workspace import FEMWorkspace, WorkspaceDocument


class FEMWorkspaceCatalogPort:
    def __init__(self, workspace: FEMWorkspace) -> None:
        if type(workspace) is not FEMWorkspace:
            raise TypeError("workspace must be FEMWorkspace")
        self._workspace = workspace

    def catalog(self) -> WorkspaceDocumentCatalog:
        documents = sorted(
            self._workspace.documents(),
            key=lambda item: item.document_id,
        )
        active_id = self._workspace.active_document_id
        visible = documents[:WORKSPACE_DOCUMENT_CATALOG_LIMIT]
        if active_id is not None and all(
            item.document_id != active_id for item in visible
        ):
            active = next(
                (item for item in documents if item.document_id == active_id),
                None,
            )
            if active is not None:
                visible = [*visible[:-1], active]
                visible.sort(key=lambda item: item.document_id)
        summaries = [self._summary(item) for item in visible]
        while True:
            catalog = self._catalog(documents, summaries, active_id)
            encoded = json.dumps(
                catalog.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) <= WORKSPACE_DOCUMENT_CATALOG_MAX_BYTES:
                return catalog
            removable = next(
                (
                    index
                    for index in range(len(summaries) - 1, -1, -1)
                    if summaries[index].target.document_id != str(active_id)
                ),
                None,
            )
            if removable is None:
                raise ValueError("active workspace document exceeds catalog budget")
            summaries.pop(removable)

    @staticmethod
    def _catalog(
        documents: list[WorkspaceDocument],
        summaries: list[WorkspaceDocumentSummary],
        active_id: int | None,
    ) -> WorkspaceDocumentCatalog:
        active_target = next(
            (
                item.target
                for item in summaries
                if item.target.document_id == str(active_id)
            ),
            None,
        )
        omitted = max(0, len(documents) - len(summaries))
        return WorkspaceDocumentCatalog(
            active_target=active_target,
            documents=tuple(summaries),
            truncated=omitted > 0,
            omitted_document_count=omitted,
        )

    @staticmethod
    def _summary(document: WorkspaceDocument) -> WorkspaceDocumentSummary:
        projection = document.projection
        lineage = document.lineage
        return WorkspaceDocumentSummary(
            target=WorkspaceDocumentIdentity(
                document_id=str(document.document_id),
                session_id=str(document.session.session_id),
            ),
            session_revision=int(projection.session_revision),
            document_kind=str(document.kind),
            source_kind=str(projection.source_kind or "blank"),
            display_name=str(document.display_name),
            model_name=(
                None if projection.model_name is None else str(projection.model_name)
            ),
            lineage=(
                None
                if lineage is None
                else WorkspaceDocumentLineage(
                    source_document_id=str(lineage.source_document_id),
                    source_session_id=lineage.source_session_id,
                    source_session_revision=lineage.source_session_revision,
                    source_project_revision=lineage.source_project_revision,
                    source_run_id=lineage.source_run_id,
                    reason=lineage.reason,
                )
            ),
        )


AgentWorkspaceCatalogPort = FEMWorkspaceCatalogPort


def create_workspace_catalog_bridge(
    workspace: FEMWorkspace,
) -> WorkspaceCatalogBridge:
    """Create the provider boundary without leaking fem_agent into the window."""

    return WorkspaceCatalogBridge(FEMWorkspaceCatalogPort(workspace))


__all__ = [
    "AgentWorkspaceCatalogPort",
    "FEMWorkspaceCatalogPort",
    "create_workspace_catalog_bridge",
]
