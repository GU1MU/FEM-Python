"""GUI authoring handlers for workspace results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable
from fem_agent.authoring import AuthoringContractError
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome
from fem_agent.result_authoring import ANALYSIS_RUN_CATALOG_MAX_LIMIT, AgentResultQueryBridge
from fem_agent.workspace_catalog import WorkspaceCatalogBridge
from fem_agent.export_authoring import AgentExportBridge, ExportAuthoringError
from .binding import AuthoringHandlerBinding, envelope


class WorkspaceResultHandlers:
    def __init__(
        self,
        binding: AuthoringHandlerBinding,
        result_bridge: AgentResultQueryBridge,
        *,
        workspace_catalog_bridge: WorkspaceCatalogBridge | None = None,
        create_model_document: Callable[[str | None], str] | None = None,
        export_bridge: AgentExportBridge | None = None,
        export_facade: object | None = None,
    ) -> None:
        self.binding = binding
        self.result_bridge = result_bridge
        self.workspace_catalog_bridge = workspace_catalog_bridge
        self.create_model_document = create_model_document
        self.export_bridge = export_bridge
        self.export_facade = export_facade

    def request_project_save(
        self,
        _arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        metadata = envelope(controller, "project-save")
        suffix = str(metadata["identity_suffix"])
        context = self.binding.current_context()
        preview = controller.preview_project_save_proposal(
            f"proposal-{suffix}",
            context,
        )
        provisional = AuthoringToolOutcome(
            "Project save is waiting for the local GUI control.",
            {
                "proposal_id": preview.proposal_id,
                "proposal_hash": preview.proposal_hash,
                "state": preview.state.value,
                "proposal_view": {
                    "proposal_id": preview.proposal_id,
                    "proposal_hash": preview.proposal_hash,
                    "proposal_kind": "project_save",
                    "title": "Save current native project",
                    "summary": "Save current accepted model state",
                    "impact": "Confirm to save the project locally; unconfirmed drafts are not written",
                    "confirm_label": "Save model",
                    "target_document_id": preview.target_document_id,
                    "target_session_id": preview.target_session_id,
                    "base_session_revision": preview.base_session_revision,
                },
                "continuation_checkpoint": {
                    "session_id": str(metadata["agent_session_id"]),
                    "source_turn_id": str(metadata["turn_id"]),
                    "proposal_id": preview.proposal_id,
                    "proposal_hash": preview.proposal_hash,
                    "model_revision": preview.base_session_revision,
                    "proposal_kind": "project_save",
                },
            },
        )
        provider_safe_authoring_payload(provisional.data)
        controller.register_project_save_proposal(
            preview.proposal_id,
            context,
        )
        return provisional

    def create_native_model_document(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"model_name"}:
            raise AuthoringContractError(
                "create_native_model_document has unknown fields"
            )
        if self.create_model_document is None:
            raise AuthoringContractError(
                "native model document creation is unavailable"
            )
        requested_name = arguments.get("model_name")
        if requested_name is not None and (
            type(requested_name) is not str
            or not requested_name.strip()
            or requested_name != requested_name.strip()
            or len(requested_name) > 160
        ):
            raise AuthoringContractError("model_name must be a nonblank trimmed string")
        previous = self.binding.current_context().binding
        model_name = self.create_model_document(requested_name)
        if type(model_name) is not str or not model_name.strip():
            raise AuthoringContractError(
                "model document creation returned no model name"
            )
        context = self.binding.current_context()
        if (
            context.binding.document_id == previous.document_id
            or context.binding.session_id == previous.session_id
        ):
            raise AuthoringContractError(
                "model document creation did not activate a new binding"
            )
        return AuthoringToolOutcome(
            f"Created and activated additional native model {model_name.strip()}.",
            {
                "state": "succeeded",
                "model_name": model_name.strip(),
                "target_document_id": context.binding.document_id,
                "target_session_id": context.binding.session_id,
                "model_revision": context.binding.session_revision,
                "preserved_existing_documents": True,
            },
        )

    def read_catalog(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"run_id", "target"}:
            raise AuthoringContractError("result catalog has unknown fields")
        response = self.result_bridge.catalog(
            arguments.get("run_id"),
            target=arguments.get("target"),
        )
        return AuthoringToolOutcome(
            "Accepted result catalog read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def read_analysis_run_catalog(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) - {"cursor", "limit", "target"}:
            raise AuthoringContractError("run catalog has unknown fields")
        cursor = arguments.get("cursor", 0)
        limit = arguments.get("limit", 20)
        if type(cursor) is not int or cursor < 0:
            raise AuthoringContractError("cursor must be a non-negative integer")
        if (
            type(limit) is not int
            or limit < 1
            or limit > ANALYSIS_RUN_CATALOG_MAX_LIMIT
        ):
            raise AuthoringContractError("limit is outside the run catalog bound")
        target = arguments.get("target")
        if target is None and self.workspace_catalog_bridge is None:
            binding = self.binding.current_context().binding
            target = {
                "document_id": binding.document_id,
                "session_id": binding.session_id,
            }
        catalog = self.result_bridge.analysis_runs(
            cursor=cursor,
            limit=limit,
            target=target,
        )
        return AuthoringToolOutcome(
            "Analysis run catalog read locally.",
            catalog.to_dict(),
        )

    def read_workspace_documents(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if arguments:
            raise AuthoringContractError(
                "read_workspace_documents accepts no arguments"
            )
        if self.workspace_catalog_bridge is None:
            raise AuthoringContractError("workspace catalog is unavailable")
        return AuthoringToolOutcome(
            "Workspace document catalog read locally.",
            self.workspace_catalog_bridge.catalog().to_dict(),
        )

    def query_result(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        response = self.result_bridge.query(arguments)
        return AuthoringToolOutcome(
            "One accepted result scalar read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def compare_results(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        response = self.result_bridge.compare(arguments)
        return AuthoringToolOutcome(
            "Two accepted result scalars compared locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def workspace_result_inventory(self) -> tuple[int, int]:
        if self.workspace_catalog_bridge is None:
            return 0, 0
        catalog = self.workspace_catalog_bridge.catalog()
        return (
            sum(item.run_count for item in catalog.documents),
            sum(item.result_count for item in catalog.documents),
        )

    def export_result_csv(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if self.export_bridge is None:
            raise AuthoringContractError("export support is unavailable")
        try:
            response = self.export_bridge.export_csv(arguments)
        except (TypeError, ExportAuthoringError) as error:
            raise AuthoringContractError(
                f"export csv request is invalid: {error}"
            ) from error
        return AuthoringToolOutcome(
            "One accepted result table exported to the workspace.",
            response.to_dict(),
            ok=response.ok,
        )

    def read_result_display_context(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if self.export_bridge is None:
            raise AuthoringContractError("export support is unavailable")
        if arguments:
            raise AuthoringContractError(
                "read_result_display_context accepts no arguments"
            )
        response = self.export_bridge.display_context()
        return AuthoringToolOutcome(
            "Current READY result display context read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def workspace_export_available(self) -> bool:
        if self.export_facade is None:
            return False
        try:
            return self.export_facade.current_workspace() is not None
        except Exception:
            return False
