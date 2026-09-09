from fem_agent.providers.base import ToolDefinition
from fem_agent.schemas import ToolResult


class _AdditionalModelToolRegistry:
    def __init__(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = tuple(
            ToolDefinition(name, name, no_arguments)
            for name in (
                "create_native_model_document",
                "read_authoring_context",
                "prepare_planar_construction_proposal",
            )
        )
        self.calls = []

    provider_snapshot = None

    def refresh_turn_snapshot(self, published_tool_names=()):
        del published_tool_names
        return None

    def dispatch(self, name, arguments, context):
        self.calls.append((name, dict(arguments)))
        data = {}
        if name == "create_native_model_document":
            data = {
                "state": "succeeded",
                "next_action": (
                    "read_authoring_context_then_prepare_requested_geometry"
                ),
            }
        elif name == "prepare_planar_construction_proposal":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "proposal_id": "proposal-new-model-geometry",
                    "proposal_hash": "b" * 64,
                    "proposal_kind": "geometry",
                    "title": "Add part",
                    "summary": (
                        "设计提案：2D 平面构造（节点=1，材料区=1，孔洞=0）；"
                        "单位制 mm-N-MPa（默认）"
                    ),
                    "impact": "确认后创建该二维几何并刷新 GUI",
                    "confirm_label": "Add part",
                    "target_document_id": "1",
                    "target_session_id": "native-session",
                    "base_session_revision": 0,
                },
                "proof_summary": {
                    "material_profile_count": 1,
                    "hole_count": 0,
                    "component_count": 1,
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-model",
                    "proposal_id": "proposal-new-model-geometry",
                    "proposal_hash": "b" * 64,
                    "model_revision": 0,
                    "proposal_kind": "geometry",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
        )


class _GeometryEditToolRegistry:
    def __init__(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = tuple(
            ToolDefinition(name, name, no_arguments)
            for name in (
                "read_geometry_edit_context",
                "prepare_geometry_edit",
            )
        )
        self.calls = []

    provider_snapshot = None

    def refresh_turn_snapshot(self, published_tool_names=()):
        del published_tool_names
        return None

    def dispatch(self, name, arguments, context):
        self.calls.append((name, dict(arguments)))
        data = {}
        if name == "prepare_geometry_edit":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "summary": "修改当前二维草图",
                    "impact": "确认后更新该部件并刷新 GUI",
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-edit",
                    "proposal_id": "proposal-planar-edit",
                    "proposal_hash": "c" * 64,
                    "model_revision": context.expected_revision,
                    "proposal_kind": "geometry",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
        )


class _GeometryEditWithCatalogToolRegistry(_GeometryEditToolRegistry):
    def __init__(self):
        super().__init__()
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = (
            *self.definitions,
            ToolDefinition(
                "read_geometry_feature_catalog",
                "read_geometry_feature_catalog",
                no_arguments,
            ),
        )


class _RetryingGeometryEditWithCatalogToolRegistry(
    _GeometryEditWithCatalogToolRegistry
):
    def __init__(self):
        super().__init__()
        self.prepare_attempts = 0

    def dispatch(self, name, arguments, context):
        if name == "prepare_geometry_edit":
            self.prepare_attempts += 1
            if self.prepare_attempts == 1:
                self.calls.append((name, dict(arguments)))
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=context.expected_revision,
                    idempotency_key=context.idempotency_key,
                    summary="geometry validation needs refreshed feature context",
                    data={"retryable": True},
                )
        return super().dispatch(name, arguments, context)
