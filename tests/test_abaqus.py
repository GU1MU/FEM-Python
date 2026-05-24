import tempfile
import unittest

import numpy as np

from fem import abaqus
from fem.core.model import (
    DisplacementConstraint,
    ElementFace,
    NodalLoad,
    OutputRequest,
    SectionAssignment,
)
from fem.solvers import static_linear
from tests.helpers.file_builders import write_inp


def _assert_pressure_points_inward(test_case, model, bc):
    node_lookup = {node.id: node for node in model.mesh.nodes}
    elem = model.mesh.elements[0]
    elem_xyz = np.array(
        [
            [node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
            for node_id in elem.node_ids
        ],
        dtype=float,
    )
    elem_center = elem_xyz.mean(axis=0)

    for face, traction in zip(model.surfaces["LOADED"].faces, bc.surface_tractions):
        face_xyz = np.array(
            [
                [node_lookup[node_id].x, node_lookup[node_id].y, node_lookup[node_id].z]
                for node_id in face.node_ids
            ],
            dtype=float,
        )
        inward = elem_center - face_xyz.mean(axis=0)
        test_case.assertGreater(np.dot(np.array(traction.vector), inward), 0.0)


class AbaqusModelReaderTests(unittest.TestCase):
    def test_abaqus_read_builds_model_with_sets_surfaces_materials_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_model.inp",
                [
                    "*Heading",
                    "** minimal model reader fixture",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Nset, nset=FIXED",
                    "1,4,5,8",
                    "*Nset, nset=TIP",
                    "2,3,6,7",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Density",
                    "7.85",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD, nlgeom=NO",
                    "*Static",
                    "1., 1., 1e-05, 1.",
                    "*Boundary",
                    "FIXED, 1, 3, 0.",
                    "*Cload",
                    "TIP, 3, -50.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        self.assertEqual(model.name, "test_abaqus_model")
        self.assertEqual(model.mesh.num_nodes, 8)
        self.assertEqual(model.mesh.elements[0].type, "Hex8")
        self.assertNotIn("E", model.mesh.elements[0].props)
        self.assertEqual(model.materials["STEEL"].properties["E"], 210.0)
        self.assertEqual(model.materials["STEEL"].properties["nu"], 0.3)
        self.assertEqual(model.materials["STEEL"].properties["rho"], 7.85)
        self.assertEqual(model.node_sets["FIXED"].node_ids, (1, 4, 5, 8))
        self.assertEqual(model.element_sets["SOLID"].element_ids, (1,))
        self.assertEqual(model.surfaces["TIP_FACE"].faces[0], ElementFace(1, 5, (2, 3, 7, 6)))
        self.assertEqual(model.sections[0], SectionAssignment("SOLID", "STEEL"))
        self.assertEqual(model.steps[0].name, "LOAD")
        self.assertEqual(model.steps[0].boundaries[0], DisplacementConstraint("FIXED", 1, 3, 0.0))
        self.assertEqual(model.steps[0].cloads[0], NodalLoad("TIP", 3, -50.0))
        self.assertFalse(hasattr(model, "boundary"))

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(len(bc.prescribed_displacements), 12)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), -200.0)

    def test_abaqus_read_inherits_initial_boundaries_across_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_initial_steps.inp",
                [
                    "*Node",
                    "1, 0., 0.",
                    "2, 1., 0.",
                    "*Element, type=CPS3, elset=SOLID",
                    "1, 1,2,1",
                    "*Nset, nset=FIXED",
                    "1",
                    "*Nset, nset=TIP",
                    "2",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Boundary",
                    "FIXED, 1, 2, 0.",
                    "*Step, name=STEP-1",
                    "*Static",
                    "*Cload",
                    "TIP, 1, 10.",
                    "*End Step",
                    "*Step, name=STEP-2",
                    "*Static",
                    "*Cload",
                    "TIP, 2, 20.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        self.assertEqual([step.name for step in model.steps], ["Initial", "STEP-1", "STEP-2"])
        self.assertEqual(static_linear.get_step(model).name, "STEP-1")
        step2_bc = static_linear.boundary_for_step(model, "STEP-2")
        self.assertEqual(len(step2_bc.prescribed_displacements), 2)
        self.assertAlmostEqual(sum(step2_bc.nodal_forces.values()), 20.0)

    def test_abaqus_read_prefers_assembly_node_set_over_part_set_for_load_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_scoped_node_sets.inp",
                [
                    "*Part, name=BLOCK",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Nset, nset=LOADSET, generate",
                    "1, 4, 1",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*End Part",
                    "*Assembly, name=Assembly",
                    "*Instance, name=BLOCK-1, part=BLOCK",
                    "*End Instance",
                    "*Nset, nset=LOADSET, instance=BLOCK-1",
                    "2",
                    "*Nset, nset=FIXED, instance=BLOCK-1",
                    "1",
                    "*End Assembly",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Boundary",
                    "FIXED, 1, 3, 0.",
                    "*Cload",
                    "LOADSET, 2, -1000.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(model.node_sets["LOADSET"].node_ids, (2,))
        self.assertEqual(len(bc.nodal_forces), 1)
        self.assertAlmostEqual(sum(bc.nodal_forces.values()), -1000.0)

    def test_abaqus_read_converts_dsload_and_dload_pressure_to_surface_tractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_surface_loads.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TIP_FACE, P, 2.",
                    "*Dload",
                    "SOLID, P4, 3.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(len(model.steps[0].surface_loads), 2)
        self.assertIn("TIP_FACE", model.surfaces)
        self.assertEqual(len(bc.surface_tractions), 2)
        self.assertTrue(np.allclose(bc.surface_tractions[0].vector, (-2.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(bc.surface_tractions[1].vector, (-3.0, 0.0, 0.0)))

    def test_abaqus_read_projects_trshr_direction_to_surface_tangent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_trshr_surface_load.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TOP",
                    "SOLID, S2",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TOP, TRSHR, 10., 2., 0., 2.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(len(model.steps[0].surface_loads), 1)
        self.assertEqual(model.steps[0].surface_loads[0].load_type, "shear_traction")
        self.assertEqual(len(bc.surface_tractions), 1)
        self.assertTrue(np.allclose(bc.surface_tractions[0].vector, (10.0, 0.0, 0.0)))

    def test_abaqus_trshr_rejects_nonplanar_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_trshr_nonplanar_face.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.2",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=WARPED_TOP",
                    "SOLID, S2",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "WARPED_TOP, TRSHR, 10., 1., 0., 0.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        with self.assertRaisesRegex(ValueError, "non-planar"):
            static_linear.boundary_for_step(model, "LOAD")

    def test_abaqus_read_maps_tetra_face_labels_to_local_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_tet_face_labels.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                ],
            )

            model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 3, (1, 2, 3)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 2, (1, 2, 4)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 0, (2, 3, 4)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 1, (1, 3, 4)))

    def test_abaqus_read_maps_hex_face_labels_to_local_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_hex_face_labels.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                    "*Surface, type=ELEMENT, name=FACE_5",
                    "SOLID, S5",
                    "*Surface, type=ELEMENT, name=FACE_6",
                    "SOLID, S6",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "SOLID, S5",
                    "SOLID, S6",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 0, (1, 4, 3, 2)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 1, (5, 6, 7, 8)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 2, (1, 2, 6, 5)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 5, (2, 3, 7, 6)))
        self.assertEqual(model.surfaces["FACE_5"].faces[0], ElementFace(1, 3, (3, 4, 8, 7)))
        self.assertEqual(model.surfaces["FACE_6"].faces[0], ElementFace(1, 4, (1, 5, 8, 4)))

        bc = static_linear.boundary_for_step(model, "LOAD")
        _assert_pressure_points_inward(self, model, bc)

    def test_abaqus_read_maps_tet10_face_labels_and_pressure_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_tet10_face_labels.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "5, 0.5, 0., 0.",
                    "6, 0.5, 0.5, 0.",
                    "7, 0., 0.5, 0.",
                    "8, 0., 0., 0.5",
                    "9, 0.5, 0., 0.5",
                    "10, 0., 0.5, 0.5",
                    "*Element, type=C3D10, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8,9,10",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=FACE_1",
                    "SOLID, S1",
                    "*Surface, type=ELEMENT, name=FACE_2",
                    "SOLID, S2",
                    "*Surface, type=ELEMENT, name=FACE_3",
                    "SOLID, S3",
                    "*Surface, type=ELEMENT, name=FACE_4",
                    "SOLID, S4",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        self.assertEqual(model.surfaces["FACE_1"].faces[0], ElementFace(1, 3, (1, 2, 3, 5, 6, 7)))
        self.assertEqual(model.surfaces["FACE_2"].faces[0], ElementFace(1, 2, (1, 2, 4, 5, 9, 8)))
        self.assertEqual(model.surfaces["FACE_3"].faces[0], ElementFace(1, 0, (2, 3, 4, 6, 10, 9)))
        self.assertEqual(model.surfaces["FACE_4"].faces[0], ElementFace(1, 1, (1, 3, 4, 7, 10, 8)))

        bc = static_linear.boundary_for_step(model, "LOAD")
        _assert_pressure_points_inward(self, model, bc)

    def test_abaqus_pressure_points_into_tetra_element_for_all_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_tet_pressure_direction.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1,2,3,4",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=LOADED",
                    "SOLID, S1",
                    "SOLID, S2",
                    "SOLID, S3",
                    "SOLID, S4",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "LOADED, P, 2.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        bc = static_linear.boundary_for_step(model, "LOAD")
        _assert_pressure_points_inward(self, model, bc)

    def test_abaqus_read_accumulates_repeated_sets_and_scales_trvec_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_sets_trvec.inp",
                [
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 1., 1., 0.",
                    "4, 0., 1., 0.",
                    "5, 0., 0., 1.",
                    "6, 1., 0., 1.",
                    "7, 1., 1., 1.",
                    "8, 0., 1., 1.",
                    "*Element, type=C3D8, elset=SOLID",
                    "1, 1,2,3,4,5,6,7,8",
                    "*Nset, nset=FIXED",
                    "1,4",
                    "*Nset, nset=FIXED",
                    "5,8",
                    "*Elset, elset=SOLID",
                    "1",
                    "*Surface, type=ELEMENT, name=TIP_FACE",
                    "SOLID, S6",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=LOAD",
                    "*Static",
                    "*Dsload",
                    "TIP_FACE, TRVEC, 10., 0., 0., -2.",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        bc = static_linear.boundary_for_step(model, "LOAD")
        self.assertEqual(model.node_sets["FIXED"].node_ids, (1, 4, 5, 8))
        self.assertEqual(bc.surface_tractions[0].vector, (0.0, 0.0, -10.0))

    def test_abaqus_read_stores_output_requests_on_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inp(
                tmp,
                "test_abaqus_output_requests.inp",
                [
                    "*Node",
                    "1, 0., 0.",
                    "2, 1., 0.",
                    "*Element, type=CPS3, elset=SOLID",
                    "1, 1,2,1",
                    "*Material, name=STEEL",
                    "*Elastic",
                    "210., 0.3",
                    "*Solid Section, elset=SOLID, material=STEEL",
                    "*Step, name=OUTPUT",
                    "*Static",
                    "*Output, field, variable=PRESELECT",
                    "*Node Output",
                    "U, RF",
                    "*Element Output, directions=YES",
                    "S, E",
                    "*Output, history, variable=PRESELECT",
                    "*End Step",
                ],
            )

            model = abaqus.read(path)

        outputs = model.steps[0].outputs
        self.assertEqual(outputs[0], OutputRequest("field", "preselect", ("PRESELECT",), {"variable": "PRESELECT"}))
        self.assertEqual(outputs[1], OutputRequest("field", "node", ("U", "RF"), {}))
        self.assertEqual(outputs[2], OutputRequest("field", "element", ("S", "E"), {"directions": "YES"}))
        self.assertEqual(outputs[3], OutputRequest("history", "preselect", ("PRESELECT",), {"variable": "PRESELECT"}))


if __name__ == "__main__":
    unittest.main()
