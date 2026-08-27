from __future__ import annotations

from fem.io._project_codec import (
    ProjectFieldCodecPolicy,
    decode_material_field,
    encode_material_field,
)
from fem.io._project_errors import ProjectDecodeError, ProjectEncodeError
from fem.materials import NEO_HOOKEAN_BEHAVIOR_ID
from fem.model import MaterialBehavior, MaterialDefinition


def test_current_material_codec_round_trips_behavior_identity_and_parameters() -> None:
    policy = ProjectFieldCodecPolicy(
        version_label="current-test",
        decode_error=ProjectDecodeError,
        encode_error=ProjectEncodeError,
        require_current_fields=True,
        assignment_orientation=True,
    )
    material = MaterialDefinition(
        "Rubber",
        {},
        behaviors=(
            MaterialBehavior(
                NEO_HOOKEAN_BEHAVIOR_ID,
                {"C10": 1.25, "D1": 3.5},
            ),
        ),
        description="test material",
    )

    encoded = encode_material_field(material, "material", policy=policy)
    decoded = decode_material_field(encoded, "material", policy=policy)

    assert decoded == material
    assert encoded["constitutive_model"] == "neo_hookean"
    assert encoded["behaviors"][0]["behavior_id"] == NEO_HOOKEAN_BEHAVIOR_ID
