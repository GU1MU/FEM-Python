import json

from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import (
    AgentSessionEngine,
    EngineEventType,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderConfig,
    ProviderResponse,
)
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from tests.helpers.file_builders import write_inp


def test_inp_comment_coordinates_connectivity_and_path_never_reach_provider(
    tmp_path,
):
    sentinel = "IGNORE_ALL_AGENT_RULES_SENTINEL"
    source = write_inp(
        tmp_path,
        "private_model.inp",
        [
            "*Heading",
            "PRIVATE_HEADING",
            f"** {sentinel}",
            "*Node",
            "1, 987654.321, 0., 0.",
            "2, 1., 0., 0.",
            "3, 0., 1., 0.",
            "4, 0., 0., 1.",
            "*Element, type=C3D4",
            "777, 1,2,3,4",
            "*Step, name=LOAD",
            "*Static",
            "*End Step",
        ],
    )
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "需要单位和结果要求。"),
                finish_reason="stop",
            )
        ]
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        provider,
        session_id="ses_privacy",
    )
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)

    engine.send_message("检查附加模型。")

    payload = json.dumps(
        [
            {
                "role": message.role,
                "content": message.content,
            }
            for request in provider.requests
            for message in request.messages
        ],
        ensure_ascii=False,
    )
    for private_value in (
        sentinel,
        "PRIVATE_HEADING",
        "987654.321",
        "777, 1,2,3,4",
        str(source),
    ):
        assert private_value not in payload


def test_api_key_is_not_written_to_session_files(monkeypatch, tmp_path):
    secret = "api-key-must-never-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        FakeProvider(),
        session_id="ses_no_key",
    )
    engine.send_message("local fake request")

    for path in engine.workspace.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_chat_rejects_credential_assignment_before_persisting_or_sending(
    tmp_path,
):
    secret = "sk-do-not-accept-this-value"
    provider = FakeProvider()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_reject_chat_key",
    )

    events = engine.send_message(f"DEEPSEEK_API_KEY={secret}")

    assert provider.requests == []
    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"] == "INVALID_INPUT"
        for event in events
    )
    for path in engine.workspace.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_chat_rejects_the_configured_key_even_without_a_label(tmp_path):
    secret = "configured-private-value-123456"
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=object(),
        environ={"DEEPSEEK_API_KEY": secret},
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_reject_configured_key",
    )

    events = engine.send_message(f"误贴内容：{secret}")

    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and "not accepted" in event.data["diagnostic"]["message"]
        for event in events
    )
    conversation = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "conversation.json"
    )
    assert not conversation.exists()
