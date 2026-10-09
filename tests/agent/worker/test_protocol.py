import io
import json

import pytest

import fem_agent.worker_protocol as protocol_module
from fem_agent.worker_protocol import WorkerRequest


def test_control_json_uses_utf8_bytes_and_enforces_the_byte_limit():
    value = {
        "acceleration": "mm/s²",
        "density": "tonne/mm³",
        "region": "自由端",
    }

    payload = protocol_module._encode_control_json(value)

    assert json.loads(payload.decode("utf-8")) == value
    assert protocol_module._read_control_json(
        io.BytesIO(payload),
        description="inspection request",
    ) == value
    with pytest.raises(ValueError, match="control payload limit"):
        protocol_module._read_control_json(
            io.BytesIO(
                b"x"
                * (protocol_module._FIXED_CONTROL_PAYLOAD_LIMIT_BYTES + 1)
            ),
            description="inspection request",
        )


def test_worker_contract_round_trips_through_json():
    request = WorkerRequest(
        "ses_worker",
        1,
        "a" * 64,
        "run_worker",
        "solve_worker",
        "2026-07-24T12:00:00Z",
    )

    assert WorkerRequest.from_dict(json.loads(json.dumps(request.to_dict()))) == request
