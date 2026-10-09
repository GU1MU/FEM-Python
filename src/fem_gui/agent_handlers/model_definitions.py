"""GUI authoring handlers for model definitions."""

from __future__ import annotations

from collections.abc import Mapping
from fem_agent.authoring import AuthoringContractError
from fem_agent.authoring_payload import provider_safe_authoring_payload
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome
from fem_agent.definition_action_authoring import create_definition_change
from fem_agent.deletion_authoring import create_delete_proposal, deletable_object_catalog
from fem_agent.editing_authoring import create_edit_patch, editable_object_catalog
from .binding import AuthoringHandlerBinding, envelope


class ModelDefinitionHandlers:
    def __init__(self, binding: AuthoringHandlerBinding) -> None:
        self.binding = binding

    def apply_model_definition(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if set(arguments) != {"action", "parameters"}:
            raise AuthoringContractError(
                "apply_model_definition requires action and parameters"
            )
        metadata = envelope(controller, "definition")
        suffix = str(metadata.pop("identity_suffix"))
        change = create_definition_change(
            patch_id=f"patch-{suffix}",
            context=self.binding.current_context(),
            snapshot=self.binding.current_session().snapshot(),
            action=arguments["action"],
            parameters=arguments["parameters"],
            **metadata,
        )
        patch = change.value
        provisional = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "action": arguments["action"],
                "patch_id": patch.patch_id,
                "undo_available": True,
                "objects": list(patch.expected_changes.get("created_names", ())),
                "gui_synchronized": True,
                "definition_object_type": change.resume_object_type,
            },
        )
        provider_safe_authoring_payload(provisional.data)
        applied = self.binding.bridge.apply_automatic_patch(patch)
        final = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "action": arguments["action"],
                "patch_id": applied.patch.patch_id,
                "undo_available": applied.undo_available,
                "objects": list(patch.expected_changes.get("created_names", ())),
                "gui_synchronized": True,
                "definition_object_type": change.resume_object_type,
            },
        )
        provider_safe_authoring_payload(final.data)
        return final

    def read_deletable_objects(
        self,
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        catalog = deletable_object_catalog(self.binding.current_session().snapshot(), limit=128)
        visible = catalog[:100]
        return AuthoringToolOutcome(
            "Current deletable native model objects read locally.",
            {
                "objects": [item.to_provider_dict() for item in visible],
                "count": len(visible),
                "truncated": len(catalog) > len(visible),
            },
        )

    def prepare_delete(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        allowed = {"object_type", "target_id", "step_name"}
        if set(arguments) - allowed:
            raise AuthoringContractError("prepare_delete_proposal has unknown fields")
        if not {"object_type", "target_id"}.issubset(arguments):
            raise AuthoringContractError(
                "prepare_delete_proposal requires object_type and target_id"
            )
        metadata = envelope(controller, "delete")
        suffix = str(metadata.pop("identity_suffix"))
        proposal, target = create_delete_proposal(
            proposal_id=f"proposal-{suffix}",
            context=self.binding.current_context(),
            snapshot=self.binding.current_session().snapshot(),
            object_type=arguments["object_type"],
            target_id=arguments["target_id"],
            step_name=arguments.get("step_name"),
            **metadata,
        )
        return self.binding.proposal_outcome(
            proposal,
            summary=str(proposal.display_summary["summary"]),
            impact=str(proposal.display_summary["impact"]),
            confirm_label=str(proposal.display_summary["confirm_label"]),
            extra_data={"delete_object_type": target.object_type},
        )

    def read_editable_objects(
        self,
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        catalog = editable_object_catalog(self.binding.current_session().snapshot(), limit=128)
        visible: list[object] = []
        for item in catalog[:100]:
            candidate = [
                *(entry.to_provider_dict() for entry in visible),
                item.to_provider_dict(),
            ]
            try:
                provider_safe_authoring_payload(
                    {
                        "objects": candidate,
                        "count": len(candidate),
                        "truncated": len(candidate) < len(catalog),
                    }
                )
            except ValueError:
                break
            visible.append(item)
        return AuthoringToolOutcome(
            "Current editable model objects read locally.",
            {
                "objects": [item.to_provider_dict() for item in visible],
                "count": len(visible),
                "truncated": len(catalog) > len(visible),
            },
        )

    def edit_model_object(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        allowed = {"object_type", "target_id", "step_name", "changes"}
        if set(arguments) - allowed:
            raise AuthoringContractError("edit_model_object has unknown fields")
        if not {"object_type", "target_id", "changes"}.issubset(arguments):
            raise AuthoringContractError(
                "edit_model_object requires object_type, target_id, and changes"
            )
        metadata = envelope(controller, "edit")
        suffix = str(metadata.pop("identity_suffix"))
        patch, target = create_edit_patch(
            patch_id=f"patch-{suffix}",
            context=self.binding.current_context(),
            snapshot=self.binding.current_session().snapshot(),
            object_type=arguments["object_type"],
            target_id=arguments["target_id"],
            step_name=arguments.get("step_name"),
            changes=arguments["changes"],
            **metadata,
        )
        provisional = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "edit_object_type": target.object_type,
                "target_id": target.target_id,
                "patch_id": patch.patch_id,
                "undo_available": True,
                "gui_synchronized": True,
            },
        )
        provider_safe_authoring_payload(provisional.data)
        applied = self.binding.bridge.apply_automatic_patch(patch)
        final = AuthoringToolOutcome(
            str(patch.display_summary["summary"]),
            {
                "state": "succeeded",
                "edit_object_type": target.object_type,
                "target_id": target.target_id,
                "patch_id": applied.patch.patch_id,
                "undo_available": applied.undo_available,
                "gui_synchronized": True,
            },
        )
        provider_safe_authoring_payload(final.data)
        return final
