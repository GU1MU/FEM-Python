"""Typed A3 mesh intent and revision-bound mesh proposals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Sequence

from fem.geometry import NATIVE_GEOMETRY_TYPES, geometry_dimension
from fem.geometry.recipe_analysis import recipe_characteristic_size
from fem.mesh.gmsh import AutoMeshSpec
from fem.mesh.settings import LocalMeshControl, MeshSettings

from .authoring import (
    AgentProposal,
    AuthoringContext,
    AuthoringContractError,
    ModelOperation,
    OperationKind,
    ProposalKind,
)


MESH_INTENT_SCHEMA_VERSION = "1.0"
_A3_CELL_SHAPES = frozenset({"triangle", "quadrilateral"})


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a positive finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return normalized


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MeshIntent:
    """Provider-safe, persistable A3 mesh policy for one two-dimensional Part.

    ``effective MeshSettings.size`` is derived only when an automatic intent
    needs an absolute far-field size for an existing typed local refinement.
    The authoritative mode remains ``auto_level`` and native generation uses
    :class:`AutoMeshSpec`.
    """

    cell_shape: str
    order: int
    global_size: float | None = None
    auto_level: int | None = None
    local_controls: tuple[LocalMeshControl, ...] = ()
    schema_version: str = MESH_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MESH_INTENT_SCHEMA_VERSION:
            raise ValueError("unknown MeshIntent schema_version")
        if type(self.cell_shape) is not str or self.cell_shape not in _A3_CELL_SHAPES:
            raise ValueError(
                "A3 cell_shape must be 'triangle' or 'quadrilateral'"
            )
        if isinstance(self.order, bool) or type(self.order) is not int:
            raise TypeError("MeshIntent order must be an integer")
        if self.order not in {1, 2}:
            raise ValueError("MeshIntent order must be 1 or 2")
        explicit = self.global_size is not None
        automatic = self.auto_level is not None
        if explicit == automatic:
            raise ValueError(
                "MeshIntent requires exactly one of global_size or auto_level"
            )
        if explicit:
            object.__setattr__(
                self,
                "global_size",
                _positive_float(self.global_size, "global_size"),
            )
        if automatic and (
            isinstance(self.auto_level, bool)
            or type(self.auto_level) is not int
            or self.auto_level not in {1, 2, 3, 4, 5}
        ):
            raise ValueError("MeshIntent auto_level must be an integer from 1 to 5")
        controls = tuple(self.local_controls)
        if any(type(item) is not LocalMeshControl for item in controls):
            raise TypeError(
                "MeshIntent local_controls must contain LocalMeshControl values"
            )
        if explicit and any(
            control.size >= float(self.global_size) for control in controls
        ):
            raise ValueError("local mesh sizes must be smaller than global_size")
        keys = {(item.target, item.falloff) for item in controls}
        if len(keys) != len(controls):
            raise ValueError(
                "one logical target and falloff may have only one local size"
            )
        object.__setattr__(
            self,
            "local_controls",
            MeshSettings(
                float(self.global_size or max(item.size for item in controls) * 2.0)
                if controls
                else float(self.global_size or 1.0),
                order=self.order,
                cell_shape=self.cell_shape,
                local_controls=controls,
            ).local_controls,
        )

    @property
    def mode(self) -> str:
        return "explicit" if self.global_size is not None else "automatic"

    @property
    def intent_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "global_size": self.global_size,
            "auto_level": self.auto_level,
            "cell_shape": self.cell_shape,
            "order": self.order,
            "local_controls": [
                {
                    "target": item.target.logical_id,
                    "size": item.size,
                    "falloff": {
                        "reference": item.falloff.reference,
                        "start_factor": item.falloff.start_factor,
                        "end_factor": item.falloff.end_factor,
                    },
                }
                for item in self.local_controls
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MeshIntent:
        if not isinstance(value, Mapping):
            raise TypeError("MeshIntent payload must be an object")
        expected = {
            "schema_version",
            "mode",
            "global_size",
            "auto_level",
            "cell_shape",
            "order",
            "local_controls",
        }
        if set(value) != expected:
            raise ValueError("MeshIntent payload fields do not match schema 1.0")
        mode = value["mode"]
        if mode not in {"explicit", "automatic"}:
            raise ValueError("MeshIntent mode must be explicit or automatic")
        raw_controls = value["local_controls"]
        if isinstance(raw_controls, (str, bytes, bytearray)) or not isinstance(
            raw_controls,
            Sequence,
        ):
            raise TypeError("MeshIntent local_controls must be an array")
        from fem.geometry import LogicalEntityRef
        from fem.mesh.settings import MeshSizeFalloff

        controls: list[LocalMeshControl] = []
        for raw in raw_controls:
            if not isinstance(raw, Mapping) or set(raw) != {
                "target",
                "size",
                "falloff",
            }:
                raise ValueError("local mesh control fields do not match schema")
            falloff = raw["falloff"]
            if not isinstance(falloff, Mapping) or set(falloff) != {
                "reference",
                "start_factor",
                "end_factor",
            }:
                raise ValueError("local mesh falloff fields do not match schema")
            controls.append(
                LocalMeshControl(
                    LogicalEntityRef(str(raw["target"])),
                    _positive_float(raw["size"], "local control size"),
                    MeshSizeFalloff(
                        str(falloff["reference"]),  # type: ignore[arg-type]
                        falloff["start_factor"],  # type: ignore[arg-type]
                        falloff["end_factor"],  # type: ignore[arg-type]
                    ),
                )
            )
        intent = cls(
            schema_version=str(value["schema_version"]),
            cell_shape=str(value["cell_shape"]),
            order=value["order"],  # type: ignore[arg-type]
            global_size=value["global_size"],  # type: ignore[arg-type]
            auto_level=value["auto_level"],  # type: ignore[arg-type]
            local_controls=tuple(controls),
        )
        if intent.mode != mode:
            raise ValueError("MeshIntent mode conflicts with its size selector")
        return intent

    def to_mesh_settings(self, recipe: object) -> MeshSettings:
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError("recipe must be native geometry")
        dimension = geometry_dimension(recipe)
        if dimension != 2:
            raise ValueError("A3 MeshIntent supports two-dimensional Parts")
        if self.global_size is not None:
            effective_size = self.global_size
        else:
            effective_size = (
                recipe_characteristic_size(recipe)
                / 10.0
                * 2.0 ** ((3 - int(self.auto_level)) / dimension)
            )
        if any(item.size >= effective_size for item in self.local_controls):
            raise ValueError(
                "local mesh sizes must be smaller than the effective far-field size"
            )
        return MeshSettings(
            effective_size,
            order=self.order,  # type: ignore[arg-type]
            cell_shape=self.cell_shape,  # type: ignore[arg-type]
            local_controls=self.local_controls,
            auto_level=self.auto_level,  # type: ignore[arg-type]
            strict_cell_shape=True,
        )

    def to_auto_mesh_spec(self) -> AutoMeshSpec | None:
        if self.auto_level is None:
            return None
        return AutoMeshSpec(
            level=self.auto_level,  # type: ignore[arg-type]
            cell_shape=(
                "tri" if self.cell_shape == "triangle" else "quad"
            ),
            order=self.order,  # type: ignore[arg-type]
        )


def create_mesh_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    context: AuthoringContext,
    draft_revision: int,
    part_id: str,
    mesh_intent: MeshIntent,
) -> AgentProposal:
    """Create one local-summary mesh proposal without touching Gmsh."""

    if type(context) is not AuthoringContext:
        raise TypeError("context must be AuthoringContext")
    if type(mesh_intent) is not MeshIntent:
        raise TypeError("mesh_intent must be MeshIntent")
    binding = context.binding
    if not binding.supported or binding.source_kind != "native":
        raise AuthoringContractError("mesh proposal requires a native context")
    part = next((item for item in context.parts if item.part_id == part_id), None)
    if part is None or part.suppressed:
        raise AuthoringContractError("mesh proposal target Part is unavailable")
    if part.dimension != 2:
        raise AuthoringContractError("A3 mesh proposal requires a 2D Part")
    mode_value: object = (
        mesh_intent.global_size
        if mesh_intent.global_size is not None
        else mesh_intent.auto_level
    )
    mode_label = (
        "global_size" if mesh_intent.global_size is not None else "auto_level"
    )
    local_summary = [
        {
            "target": control.target.logical_id,
            "size": control.size,
            "falloff": {
                "reference": control.falloff.reference,
                "start_factor": control.falloff.start_factor,
                "end_factor": control.falloff.end_factor,
            },
        }
        for control in mesh_intent.local_controls
    ]
    resource_level = (
        "high"
        if mesh_intent.auto_level in {4, 5}
        else "medium"
        if mesh_intent.auto_level in {2, 3}
        else "bounded_unknown"
    )
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.MESH,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=tuple(source_tool_call_ids),
        target_document_id=binding.document_id,
        target_session_id=binding.session_id,
        base_session_revision=binding.session_revision,
        draft_revision=draft_revision,
        operations=(
            ModelOperation(
                OperationKind.SET_PART_MESH_INTENT,
                {
                    "part_id": part_id,
                    "mesh_intent": mesh_intent.to_dict(),
                },
            ),
            ModelOperation(
                OperationKind.REQUEST_MESH,
                {
                    "part_id": part_id,
                    "mesh_intent_hash": mesh_intent.intent_hash,
                },
            ),
        ),
        preconditions={
            "source_kind": "native",
            "part_id": part_id,
            "part_suppressed": False,
        },
        expected_changes={
            "mesh_intent_committed": True,
            "generated_model_replaced": True,
            "projection_refresh_count": 1,
        },
        invalidation_impact={
            "mesh": context.mesh.present,
            "named_regions": context.definitions.named_region_count > 0,
            "definitions": (
                context.definitions.material_count
                + context.definitions.section_count
                + context.definitions.assignment_count
                + context.definitions.analysis_step_count
            )
            > 0,
            "validation": context.validation_status != "not_run",
            "results": context.result_available,
        },
        display_summary={
            "title": "生成网格",
            "target_model": context.model_name,
            "target_part": {"part_id": part.part_id, "name": part.name},
            mode_label: mode_value,
            "cell_shape": mesh_intent.cell_shape,
            "order": mesh_intent.order,
            "local_refinements": local_summary,
            "resource_level": resource_level,
            "estimate_only": True,
            "confirm_label": "开始划分",
            "mesh_intent_hash": mesh_intent.intent_hash,
            "base_session_revision": binding.session_revision,
        },
    )


__all__ = [
    "MESH_INTENT_SCHEMA_VERSION",
    "MeshIntent",
    "create_mesh_proposal",
]
