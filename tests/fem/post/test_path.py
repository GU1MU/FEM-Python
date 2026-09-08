import csv

import pytest

from fem.core.mesh import (
    Mesh2D,
    Node2D,
)
from fem.post import (
    path,
)


def test_path_stress_entrypoint_accepts_current_metadata(tmp_path):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "current_nodal_stress.csv"
    out_path = tmp_path / "nodes.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )

    path.extract_nodes_data(
        mesh,
        [1, 2],
        ["sig_x"],
        path=out_path,
        stress_csv_path=stress_path,
    )

    with out_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["sig_x"]) for row in rows] == [10.0, 20.0]


@pytest.mark.parametrize("target", ("elem_id", "local_node", "averaged"))
def test_path_stress_entrypoints_reject_metadata_targets(tmp_path, target):
    mesh = Mesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 1.0, 0.0)],
        elements=[],
    )
    stress_path = tmp_path / "current_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,,,true,10\n"
        "2,1,0,,,true,20\n",
        encoding="utf-8",
    )
    calls = (
        lambda: path.extract_path_data(
            mesh,
            1,
            2,
            2,
            target,
            path=tmp_path / "path.csv",
            stress_csv_path=stress_path,
        ),
        lambda: path.extract_nodes_data(
            mesh,
            [1],
            [target],
            path=tmp_path / "nodes.csv",
            stress_csv_path=stress_path,
        ),
    )

    for call in calls:
        with pytest.raises(ValueError, match=rf"target {target} not found"):
            call()


def test_path_stress_reader_rejects_duplicate_node_rows(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    stress_path = tmp_path / "duplicate_nodal_stress.csv"
    stress_path.write_text(
        "node_id,x,y,elem_id,local_node,averaged,sig_x\n"
        "1,0,0,1,1,false,10\n"
        "1,0,0,2,1,false,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        path.extract_nodes_data(
            mesh,
            [1],
            ["sig_x"],
            path=tmp_path / "nodes.csv",
            stress_csv_path=stress_path,
        )

    message = str(exc_info.value)
    assert str(stress_path) in message
    assert "node_id 1" in message
    assert "multiple element-nodal contributions" in message
    assert "explicit selection or averaging" in message


def test_path_nodal_reader_reports_invalid_node_id_with_context(tmp_path):
    mesh = Mesh2D(nodes=[Node2D(1, 0.0, 0.0)], elements=[])
    csv_path = tmp_path / "invalid_node_id.csv"
    csv_path.write_text(
        "node_id,x,y,ux,uy\nbad,0,0,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        path.extract_nodes_data(
            mesh,
            [1],
            ["ux"],
            path=tmp_path / "nodes.csv",
            disp_csv_path=csv_path,
        )

    message = str(exc_info.value)
    assert str(csv_path) in message
    assert "line 2" in message
    assert "field node_id" in message
    assert "raw value 'bad'" in message
    assert "expected an integer" in message
