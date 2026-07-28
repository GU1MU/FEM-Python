from __future__ import annotations

from fem.application import MeshEntityRef, NamedRegion, NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry import BoxGeometry
from fem.io.project_v3 import decode_project_v3, encode_project_v3
from fem.mesh.settings import MeshSettings


def test_v3_roundtrip_preserves_all_mesh_scope_reference_kinds() -> None:
    recipe = BoxGeometry("ScopedBox", 2.0, 1.0, 0.5)
    regions = (
        NamedRegion("Nodes", (MeshEntityRef.node(11),)),
        NamedRegion(
            "Edges",
            (MeshEntityRef.edge(21, 2, (11, 12)),),
        ),
        NamedRegion(
            "Faces",
            (MeshEntityRef.face(21, 3, (11, 12, 13)),),
        ),
        NamedRegion("Elements", (MeshEntityRef.element(21),)),
    )
    original = ProjectSnapshot(
        source_kind="native",
        parts=(NativePart(),),
        geometry_recipe=recipe,
        mesh_settings=MeshSettings(0.25, cell_shape="tetrahedron"),
        feature_history=derive_feature_history(recipe),
        named_regions=regions,
    )

    payload = encode_project_v3(original)
    reopened = decode_project_v3(payload)

    assert reopened == original
    encoded_regions = payload["project"]["authoring"]["named_regions"]
    assert {
        item["references"][0]["kind"]
        for item in encoded_regions
    } == {"node", "edge", "face", "element"}
