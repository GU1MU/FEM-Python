"""GUI authoring handlers for mesh solve."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable
from fem.application.native_scope_materialization import mesh_references_for_logical_entities
from fem_agent.authoring import AuthoringContractError
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.authoring_values import AuthoringToolOutcome
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.solve_authoring import create_solve_proposal
from fem.geometry import (
    LogicalEntityRef,
    describe_recipe_topology,
    namespace_part_logical_id,
    resolve_target_radius,
)
from fem.mesh.settings import LocalMeshControl, MeshSizeFalloff
from .binding import AuthoringHandlerBinding, envelope
from ..agent_authoring_values import AgentPreflightState


class MeshSolveHandlers:
    def __init__(
        self,
        binding: AuthoringHandlerBinding,
        next_job_name: Callable[[], str] | None = None,
    ) -> None:
        self.binding = binding
        self.next_job_name = next_job_name

    def resolved_step_name(
        self,
        arguments: Mapping[str, object],
        *,
        operation: str,
    ) -> str:
        if set(arguments) - {"step_name"}:
            raise AuthoringContractError(f"{operation} has unknown fields")
        runnable = self.binding.current_session().runnable_step_names()
        if "step_name" not in arguments:
            if len(runnable) != 1:
                raise AuthoringContractError(
                    f"{operation} requires step_name when multiple steps are runnable"
                )
            return runnable[0]
        value = arguments["step_name"]
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or len(value) > 160
        ):
            raise AuthoringContractError("step_name must be a nonblank string")
        step_name = value
        if step_name not in runnable:
            raise AuthoringContractError(
                f"{operation} requires a current runnable analysis step"
            )
        return step_name

    def mesh_refinement_entities(self):
        snapshot = self.binding.current_session().snapshot()
        part = next(
            (
                candidate
                for candidate in snapshot.parts
                if candidate.id == snapshot.active_part_id and not candidate.suppressed
            ),
            None,
        )
        if part is None or part.geometry_recipe is None:
            raise AuthoringContractError(
                "there is no active Part with editable mesh topology"
            )
        topology = describe_recipe_topology(part.geometry_recipe)
        if not topology.exact:
            raise AuthoringContractError(
                "active Part topology is not exact enough for local refinement"
            )
        entities = tuple(
            entity
            for entity in topology.selectable_entities()
            if entity.kind in {"point", "edge", "face"}
        )
        return part, entities

    def read_mesh_refinement_context(
        self,
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        part, entities = self.mesh_refinement_entities()
        visible = entities[:256]
        rows = []
        for entity in visible:
            falloff_references = ["global_size"]
            try:
                resolve_target_radius(
                    part.geometry_recipe,
                    LogicalEntityRef(entity.logical_id),
                )
            except ValueError:
                pass
            else:
                falloff_references.append("target_radius")
            rows.append(
                {
                    "logical_id": entity.logical_id,
                    "kind": entity.kind,
                    "semantic_role": entity.semantic_role,
                    "allowed_falloff_references": falloff_references,
                }
            )
        return AuthoringToolOutcome(
            f"Read {len(visible)} stable logical entities available for local refinement.",
            {
                "part_id": str(part.id),
                "entities": rows,
                "entity_count": len(entities),
                "truncated": len(entities) > len(visible),
            },
        )

    def read_model_topology_context(
        self,
        _arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        snapshot = self.binding.current_session().snapshot()
        model = getattr(snapshot.artifact, "model", None)
        if (
            snapshot.source_kind != "native"
            or not snapshot.model_current
            or not snapshot.mesh_current
            or model is None
        ):
            raise AuthoringContractError(
                "model topology context requires one current native mesh"
            )
        entries: list[dict[str, object]] = []
        truncated = False
        for part in sorted(snapshot.parts, key=lambda item: str(item.id)):
            if part.suppressed or part.geometry_recipe is None:
                continue
            topology = describe_recipe_topology(part.geometry_recipe)
            if not topology.exact:
                raise AuthoringContractError(
                    f"Part {part.id} has no exact logical topology"
                )
            for entity in topology.selectable_entities():
                reference = LogicalEntityRef(
                    namespace_part_logical_id(
                        str(part.id),
                        entity.logical_id,
                    )
                )
                for mesh_kind in ("node", "edge", "face", "element"):
                    try:
                        materialized = mesh_references_for_logical_entities(
                            model,
                            (reference,),
                            mesh_kind=mesh_kind,
                        )
                    except ValueError:
                        continue
                    # The provider payload budget (64 KiB) stays the hard
                    # backstop; 256 entries keeps rich planar-boolean
                    # topologies fully readable instead of dropping the
                    # tail, which previously hid outer-edge matched counts.
                    if len(entries) == 256:
                        truncated = True
                        break
                    entries.append(
                        {
                            "part_id": str(part.id),
                            "part_name": str(part.name),
                            "logical_id": entity.logical_id,
                            "kind": entity.kind,
                            "semantic_role": entity.semantic_role,
                            "mesh_kind": mesh_kind,
                            "matched_count": len(materialized),
                        }
                    )
                if truncated:
                    break
            if truncated:
                break
        if not entries:
            raise AuthoringContractError(
                "current native mesh exposes no materializable logical entities"
            )
        return AuthoringToolOutcome(
            f"Read {len(entries)} materializable model topology entries.",
            {
                "entries": entries,
                "entry_count": len(entries),
                "truncated": truncated,
            },
        )

    def prepare_mesh(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        if not set(arguments) <= {"local_refinements"}:
            raise AuthoringContractError(
                "prepare_mesh_proposal accepts only local_refinements"
            )
        requirements = controller.collected_requirements("mesh")
        metadata = envelope(controller, "mesh")
        context = self.binding.current_context()
        part_id = context.active_part_id
        if part_id is None:
            raise AuthoringContractError("there is no active Part to mesh")
        part, entities = self.mesh_refinement_entities()
        available_targets = {entity.logical_id for entity in entities}
        raw_refinements = arguments.get("local_refinements", [])
        if not isinstance(raw_refinements, list):
            raise TypeError("local_refinements must be an array")
        if len(raw_refinements) > 32:
            raise ValueError("local_refinements exceeds the 32-item bound")
        local_controls = []
        local_summary = []
        for raw_refinement in raw_refinements:
            if not isinstance(raw_refinement, Mapping):
                raise TypeError("each local refinement must be an object")
            refinement = dict(raw_refinement)
            if set(refinement) != {"target", "size", "falloff"}:
                raise ValueError("local refinement fields do not match")
            target = str(refinement["target"])
            if target not in available_targets:
                raise ValueError(
                    "local refinement target is not one current selectable "
                    "logical entity"
                )
            raw_falloff = refinement["falloff"]
            if not isinstance(raw_falloff, Mapping):
                raise TypeError("local refinement falloff must be an object")
            falloff = dict(raw_falloff)
            if set(falloff) != {
                "reference",
                "start_factor",
                "end_factor",
            }:
                raise ValueError("local refinement falloff fields do not match")
            if falloff["reference"] == "target_radius":
                try:
                    resolve_target_radius(
                        part.geometry_recipe,
                        LogicalEntityRef(target),
                    )
                except ValueError as error:
                    raise ValueError(
                        "target_radius falloff is unavailable for the selected "
                        "logical entity"
                    ) from error
            control = LocalMeshControl(
                LogicalEntityRef(target),
                refinement["size"],
                MeshSizeFalloff(
                    falloff["reference"],
                    falloff["start_factor"],
                    falloff["end_factor"],
                ),
            )
            local_controls.append(control)
            local_summary.append(
                {
                    "target": control.target.logical_id,
                    "size": control.size,
                    "falloff": {
                        "reference": control.falloff.reference,
                        "start_factor": control.falloff.start_factor,
                        "end_factor": control.falloff.end_factor,
                    },
                }
            )
        intent = MeshIntent(
            str(requirements["mesh_cell_shape"]),
            int(requirements["mesh_order"]),
            global_size=float(requirements["mesh_global_size"]),
            local_controls=tuple(local_controls),
            line_element_type=(
                str(requirements["line_element_type"]) if part.dimension == 1 else None
            ),
        )
        try:
            intent.validate_recipe_capability(part.geometry_recipe)
        except ValueError as error:
            message = str(error)
            if not message.startswith("mesh.hex.unsupported-shape:"):
                raise
            return AuthoringToolOutcome(
                message,
                {
                    "state": "failed",
                    "diagnostic_code": "mesh.hex.unsupported-shape",
                },
                ok=False,
            )
        suffix = str(metadata.pop("identity_suffix"))
        proposal = create_mesh_proposal(
            proposal_id=f"proposal-{suffix}",
            context=context,
            part_id=part_id,
            mesh_intent=intent,
            **metadata,
        )
        return self.binding.proposal_outcome(
            proposal,
            summary=(
                f"Generate order-{requirements['mesh_order']} "
                f"{requirements['mesh_cell_shape']} mesh; global size "
                f"{requirements['mesh_global_size']}; local refinements "
                f"{len(local_controls)}"
                + (
                    f"; line elements {requirements['line_element_type']}"
                    if part.dimension == 1
                    else ""
                )
            ),
            impact="Confirm to generate the mesh; install it and refresh the GUI on success",
            confirm_label="Generate mesh",
            extra_data={
                "local_refinements": local_summary,
            },
        )

    def run_preflight(
        self,
        arguments: Mapping[str, object],
        _controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        step_name = self.resolved_step_name(arguments, operation="preflight")
        if not self.binding.current_session().can_check(step_name):
            raise AuthoringContractError(
                "preflight requires a current runnable analysis step"
            )
        record = self.binding.bridge.request_preflight(step_name)
        if record.state is AgentPreflightState.RUNNING:
            # Only reachable when the GUI wait budget elapsed before the
            # background check finished; the provider is told to stop
            # polling and report the still-running state instead.
            summary = (
                "Native preflight is still running in the background GUI "
                "task after its local wait budget."
            )
        else:
            summary = (
                "Deterministic native preflight finished with state "
                f"{record.state.value}."
            )
        return AuthoringToolOutcome(
            summary,
            {
                "request_id": record.request_id,
                "state": record.state.value,
                "passed": record.state is AgentPreflightState.PASSED,
                "message": record.message,
            },
        )

    def prepare_solve(
        self,
        arguments: Mapping[str, object],
        controller: AuthoringWorkflowController,
    ) -> AuthoringToolOutcome:
        metadata = envelope(controller, "solve")
        suffix = str(metadata.pop("identity_suffix"))
        step_name = self.resolved_step_name(arguments, operation="solve")
        if not self.binding.current_session().can_submit(step_name):
            raise AuthoringContractError(
                "solve requires a current validated runnable analysis step"
            )
        supplied_job_name = (
            self.binding.current_session().next_run_name()
            if self.next_job_name is None
            else self.next_job_name()
        )
        if type(supplied_job_name) is not str or not supplied_job_name.strip():
            raise AuthoringContractError("next_job_name must return a nonblank string")
        proposal = create_solve_proposal(
            proposal_id=f"proposal-{suffix}",
            snapshot=self.binding.current_session().snapshot(),
            step_name=step_name,
            job_name=supplied_job_name.strip(),
            target_document_id=self.binding.current_context().binding.document_id,
            **metadata,
        )
        return self.binding.proposal_outcome(
            proposal,
            summary=f"Submit {step_name} with the current validation stamp",
            impact="Accept to run the current preflighted linear static model in the background",
            confirm_label="Start solve",
        )
