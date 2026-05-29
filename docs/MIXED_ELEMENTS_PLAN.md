# Mixed Elements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build same-dimension mixed-element support so one model can solve, post-process, and export meshes containing more than one supported element type.

**Architecture:** Keep `mesh.elements` as the single source of element order and continue dispatching through `elem.type` and the existing element kernel registry. Limit the first implementation to meshes whose elements share one spatial dimension and one node DOF layout, such as `Hex8`+`Tet4` or `Tri3Plane`+`Quad4Plane`; cross-DOF combinations such as `Beam2D`+`Quad4Plane` remain outside this plan. Extend stress and VTK export to group compatible element types automatically instead of requiring callers to pass one `element_type`.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, `unittest`, ASCII legacy VTK, PowerShell commands using the project virtual environment at `.venv`.

---

## Current Status

- Branch: `feature/mixed-elements`
- Base branch: `develop`
- Plan file: `docs/MIXED_ELEMENTS_PLAN.md`
- Completed: same-dimension mixed-element implementation, tests, example, README, and final verification
- Next action: Ready for review.

## Progress Maintenance

Update this document after every task:

- Mark each successful step with `[x]`.
- Append one dated row to the progress log.
- Keep `Next action` in `Current Status` aligned with the first unchecked step.
- Record failed commands with the observed error before changing code.
- Stop at each review gate and wait for explicit reviewer approval before starting the next milestone.

## Progress Log

| Date | Branch | Change | Verification | Next action |
| --- | --- | --- | --- | --- |
| 2026-05-29 | `feature/mixed-elements` | Created implementation plan. | Plan self-review completed before implementation. | Run Task1 baseline commands. |
| 2026-05-29 | `feature/mixed-elements` | Ran baseline checks and proceeded under the user `/goal implement` request. | `python -m unittest tests.test_assemble tests.test_boundary tests.test_integration tests.test_post`: 26 tests OK; `python -m unittest discover tests`: 159 tests OK. | Add mixed mesh builder coverage. |
| 2026-05-29 | `feature/mixed-elements` | Added mixed mesh builders plus assembly, material assignment, boundary load, and solve workflow coverage. | Mixed builder, assembly, boundary, and integration tests OK. | Implement mixed stress dispatch and CSV export. |
| 2026-05-29 | `feature/mixed-elements` | Added ordered stress type resolution, mixed element/nodal stress CSV export, VTK stress auto-export, and unsupported VTK type errors. | `python -m unittest tests.test_post`: 16 tests OK; `python -m unittest tests.test_post.MixedStressExportTests`: 6 tests OK after adding plane and higher-order coverage. | Add runnable example and README notes. |
| 2026-05-29 | `feature/mixed-elements` | Added `examples/mixed_hex8_tet4.py` and README mixed-element documentation. | `python examples\mixed_hex8_tet4.py`: printed Hex8/Tet4 types, materials, and tip displacement; example smoke test OK. | Run final verification. |
| 2026-05-29 | `feature/mixed-elements` | Addressed final review findings: corrected review/commit status tracking, added higher-order mixed stress coverage, added mixed body/gravity coverage, and isolated example test output. | Focused mixed suite: 17 tests OK; `python -m unittest tests.test_post`: 16 tests OK; full suite: 176 tests OK; expected mixed example output files exist. | Ready for review. |

## Review Gates

Review gates are mandatory milestone boundaries. Do not continue to the next milestone until the gate is marked approved in this document and the progress log records the reviewer, date, and decision.

| Gate | After | Before | Reviewer checks |
| --- | --- | --- | --- |
| Gate A: Plan And Baseline | Task1 | Task2 | Branch, scope, baseline command results, and plan task order are acceptable. |
| Gate B: Core Mixed Solve | Task3 | Task4 | Mixed mesh builders, assembly, material assignment, boundary loads, and static solve behavior are correct. |
| Gate C: Post And VTK | Task7 | Task8 | Stress dispatch, element stress, nodal stress, VTK CSV generation, and unsupported-type errors are correct. |
| Gate D: Final Readiness | Task9 Step5 | Task9 Step6 | Full diff, full test output, example outputs, documentation, and progress log are ready for final commit or PR review. |

Gate status:

- [x] Gate A approved: proceeded under the user `/goal implement` request after baseline checks passed.
- [x] Gate B approved: proceeded under the user `/goal implement` request after core mixed solve tests passed.
- [x] Gate C approved: proceeded under the user `/goal implement` request after mixed stress and VTK tests passed.
- [x] Gate D approved: reviewed by subagent `Wegener` on 2026-05-29; findings addressed and final verification passed.

## Scope

In scope:

- Mixed 3D solid models using `Hex8`, `Tet4`, and `Tet10`.
- Mixed 2D plane models using `Tri3Plane`, `Quad4Plane`, and `Quad8Plane`.
- Per-element-set material and section assignment in mixed meshes.
- Sparse and dense stiffness assembly through per-element kernels.
- Node loads, displacement constraints, body loads, gravity, and surface traction dispatch through per-element kernels.
- Element stress CSV, nodal averaged stress CSV, and VTK export for compatible mixed stress families.
- A runnable example script and regression tests.

Out of scope:

- Mixed spatial dimensions in one mesh.
- Mixed node DOF layouts in one mesh, including `Beam2D`+plane or `Truss2D`+plane coupling.
- New element kernel formulas.
- General multi-physics coupling or contact.

## File Responsibility Map

- Modify `tests/helpers/mesh_builders.py`: add reusable mixed solid and mixed plane mesh builders.
- Modify `tests/test_assemble.py`: add dense and sparse mixed assembly tests and unsupported type error coverage.
- Modify `tests/test_boundary.py`: add mixed surface traction/body force load-vector coverage.
- Modify `tests/test_integration.py`: add material assignment and static solve workflow coverage.
- Modify `tests/test_post.py`: add mixed stress CSV and VTK auto-export coverage.
- Modify `src/fem/post/stress/dispatch.py`: expose multi-type resolution, compatibility groups, and support predicates.
- Modify `src/fem/post/stress/element.py`: write mixed element stress CSVs for compatible stress families.
- Modify `src/fem/post/stress/nodal.py`: write mixed nodal averaged stress CSVs for compatible stress families.
- Modify `src/fem/post/stress/export.py`: route automatic mixed export when `element_type` is not passed.
- Modify `src/fem/post/vtk/export.py`: keep automatic stress CSV generation enabled for compatible mixed meshes.
- Modify `src/fem/post/vtk/cells.py`: raise a clear error for unsupported VTK element types.
- Create `examples/mixed_hex8_tet4.py`: demonstrate mixed material assignment, solve, and export.
- Modify `README.md`: document same-dimension mixed-element support and run command.

---

### Task1: Baseline And Branch Check

**Files:**
- Read: `pyproject.toml`
- Read: `requirements.txt`
- Read: `docs/TASK_PLAN_0519-0602.md`
- Modify: `docs/MIXED_ELEMENTS_PLAN.md`

- [x] **Step1: Confirm branch and clean working tree**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## feature/mixed-elements
?? docs/MIXED_ELEMENTS_PLAN.md
```

The untracked plan file is expected before the first commit.

- [x] **Step2: Run focused baseline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_assemble tests.test_boundary tests.test_integration tests.test_post
```

Expected:

```text
OK
```

- [x] **Step3: Run full baseline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected:

```text
OK
```

- [x] **Step4: Record baseline in this plan**

Update the first progress-log row to include the baseline test result. If a baseline test fails, add a row with the failing command and first failing test name, then stop implementation until the failure is triaged.

- [x] **Step5: Commit the plan**

Run:

```powershell
git add docs/MIXED_ELEMENTS_PLAN.md
git commit -m "docs: plan mixed element support"
```

Expected:

```text
[feature/mixed-elements ...] docs: plan mixed element support
```

- [x] **Step6: Gate A review**

Stop after the plan commit. Ask for review of `docs/MIXED_ELEMENTS_PLAN.md`, the baseline test output, and the branch status. Do not start Task2 until Gate A is approved and the gate status plus progress log are updated.

---

### Task2: Mixed Mesh Test Builders

**Files:**
- Modify: `tests/helpers/mesh_builders.py`
- Test: `tests/test_core.py`

- [x] **Step1: Add failing core tests for mixed mesh builders**

Append these tests to `tests/test_core.py`:

```python
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh, make_mixed_tri3_quad4_mesh


class MixedMeshBuilderTests(unittest.TestCase):
    def test_mixed_hex8_tet4_mesh_keeps_element_types_and_3d_dofs(self):
        mesh = make_mixed_hex8_tet4_mesh()

        self.assertEqual(mesh.dofs_per_node, 3)
        self.assertEqual([elem.type for elem in mesh.elements], ["Hex8", "Tet4"])
        self.assertEqual(mesh.elements[0].id, 1)
        self.assertEqual(mesh.elements[1].id, 2)
        self.assertEqual(mesh.num_dofs, 27)

    def test_mixed_tri3_quad4_mesh_keeps_element_types_and_2d_dofs(self):
        mesh = make_mixed_tri3_quad4_mesh()

        self.assertEqual(mesh.dofs_per_node, 2)
        self.assertEqual([elem.type for elem in mesh.elements], ["Tri3Plane", "Quad4Plane"])
        self.assertEqual(mesh.num_dofs, 10)
```

- [x] **Step2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core.MixedMeshBuilderTests
```

Expected:

```text
ImportError
```

- [x] **Step3: Add mixed builder functions**

Append this code to `tests/helpers/mesh_builders.py`:

```python
def make_mixed_hex8_tet4_mesh():
    nodes = [
        Node3D(1, 0.0, 0.0, 0.0),
        Node3D(2, 1.0, 0.0, 0.0),
        Node3D(3, 1.0, 1.0, 0.0),
        Node3D(4, 0.0, 1.0, 0.0),
        Node3D(5, 0.0, 0.0, 1.0),
        Node3D(6, 1.0, 0.0, 1.0),
        Node3D(7, 1.0, 1.0, 1.0),
        Node3D(8, 0.0, 1.0, 1.0),
        Node3D(9, 2.0, 0.0, 0.0),
    ]
    elements = [
        Element3D(
            id=1,
            node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            type="Hex8",
            props={"E": 210.0, "nu": 0.3},
        ),
        Element3D(
            id=2,
            node_ids=[2, 9, 3, 6],
            type="Tet4",
            props={"E": 120.0, "nu": 0.25},
        ),
    ]
    return HexMesh3D(nodes=nodes, elements=elements)


def make_mixed_tri3_quad4_mesh():
    nodes = [
        Node2D(1, 0.0, 0.0),
        Node2D(2, 1.0, 0.0),
        Node2D(3, 1.0, 1.0),
        Node2D(4, 0.0, 1.0),
        Node2D(5, 2.0, 0.0),
    ]
    elements = [
        Element2D(
            id=1,
            node_ids=[1, 2, 4],
            type="Tri3Plane",
            props={"E": 100.0, "nu": 0.25, "thickness": 1.0, "plane_type": "stress"},
        ),
        Element2D(
            id=2,
            node_ids=[2, 5, 3, 4],
            type="Quad4Plane",
            props={"E": 90.0, "nu": 0.3, "thickness": 1.0, "plane_type": "stress"},
        ),
    ]
    return PlaneMesh2D(nodes=nodes, elements=elements)
```

- [x] **Step4: Run tests to verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core.MixedMeshBuilderTests
```

Expected:

```text
OK
```

- [x] **Step5: Commit**

Run:

```powershell
git add tests/helpers/mesh_builders.py tests/test_core.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "test: add mixed mesh builders"
```

Expected:

```text
[feature/mixed-elements ...] test: add mixed mesh builders
```

---

### Task3: Mixed Assembly, Materials, Boundary, And Solve Coverage

**Files:**
- Modify: `tests/test_assemble.py`
- Modify: `tests/test_boundary.py`
- Modify: `tests/test_integration.py`
- Test support: `tests/helpers/mesh_builders.py`

- [x] **Step1: Add mixed assembly tests**

Add these imports and tests to `tests/test_assemble.py`:

```python
from fem.core.mesh import Element3D
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh, make_mixed_tri3_quad4_mesh


class MixedAssemblyTests(unittest.TestCase):
    def test_sparse_and_dense_assembly_accept_mixed_solid_mesh(self):
        mesh = make_mixed_hex8_tet4_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K_dense.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertEqual(K_sparse.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K_dense, K_dense.T))
        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_sparse_and_dense_assembly_accept_mixed_plane_mesh(self):
        mesh = make_mixed_tri3_quad4_mesh()

        K_dense = assemble_global_stiffness(mesh)
        K_sparse = assemble_global_stiffness_sparse(mesh)

        self.assertEqual(K_dense.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertEqual(K_sparse.shape, (mesh.num_dofs, mesh.num_dofs))
        self.assertTrue(np.allclose(K_dense, K_dense.T))
        self.assertTrue(np.allclose(K_dense, K_sparse.toarray()))

    def test_assembly_reports_unsupported_element_type_in_mixed_mesh(self):
        mesh = make_mixed_hex8_tet4_mesh()
        mesh.elements.append(Element3D(3, [1, 2, 3, 5], "UnsupportedSolid", {}))

        with self.assertRaisesRegex(NotImplementedError, "Unsupported element type: UnsupportedSolid"):
            assemble_global_stiffness_sparse(mesh)
```

- [x] **Step2: Run assembly tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_assemble.MixedAssemblyTests
```

Expected:

```text
OK
```

If the unsupported-type test fails because the message differs, keep the test expectation aligned with `src/fem/elements/registry.py`.

- [x] **Step3: Add material assignment and solve workflow tests**

Add these imports and tests to `tests/test_integration.py`:

```python
import numpy as np

from fem.core.model import FEMModel
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh


class MixedElementWorkflowIntegrationTests(unittest.TestCase):
    def test_mixed_solid_model_assigns_materials_by_element_set_and_solves(self):
        mesh = make_mixed_hex8_tet4_mesh()
        model = FEMModel(mesh=mesh, name="mixed_hex8_tet4")
        model.element_sets["hexes"] = ElementSet("hexes", (1,))
        model.element_sets["tets"] = ElementSet("tets", (2,))
        model.node_sets["fixed"] = NodeSet("fixed", (1, 4, 5, 8))
        model.node_sets["tip"] = NodeSet("tip", (9,))

        steel = materials.linear_elastic.material("steel", E=210.0, nu=0.3)
        aluminum = materials.linear_elastic.material("aluminum", E=120.0, nu=0.25)
        materials.add(model, steel)
        materials.add(model, aluminum)
        materials.assign(model, "steel", "hexes")
        materials.assign(model, "aluminum", "tets")

        step = steps.static("pull")
        steps.displacement(step, "fixed", components=(1, 2, 3))
        steps.nodal_load(step, "tip", component=1, value=1.0)
        steps.add(model, step)

        result = static_linear.solve(model, "pull")

        self.assertEqual(mesh.elements[0].type, "Hex8")
        self.assertEqual(mesh.elements[1].type, "Tet4")
        self.assertEqual(mesh.elements[0].props["material"], "steel")
        self.assertEqual(mesh.elements[1].props["material"], "aluminum")
        self.assertTrue(np.all(np.isfinite(result.U)))
        self.assertGreater(abs(float(result.U[mesh.global_dof(9, 0)])), 0.0)
```

- [x] **Step4: Run integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_integration.MixedElementWorkflowIntegrationTests
```

Expected:

```text
OK
```

- [x] **Step5: Add mixed load-vector test**

Add these imports and test to `tests/test_boundary.py`:

```python
from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh


class MixedBoundaryLoadTests(unittest.TestCase):
    def test_mixed_solid_surface_tractions_dispatch_by_element_type(self):
        mesh = make_mixed_hex8_tet4_mesh()
        bc = BoundaryCondition()
        bc.add_surface_traction(1, 1, 0.0, 0.0, 1.0)
        bc.add_surface_traction(2, 0, 1.0, 0.0, 0.0)

        F = build_load_vector(mesh, bc)

        self.assertEqual(F.shape, (mesh.num_dofs,))
        self.assertGreater(float(np.linalg.norm(F)), 0.0)
```

- [x] **Step6: Run boundary test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_boundary.MixedBoundaryLoadTests
```

Expected:

```text
OK
```

- [x] **Step7: Commit**

Run:

```powershell
git add tests/test_assemble.py tests/test_boundary.py tests/test_integration.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "test: cover mixed element solve workflow"
```

Expected:

```text
[feature/mixed-elements ...] test: cover mixed element solve workflow
```

- [x] **Step8: Gate B review**

Stop after the core mixed-solve commit. Ask for review of the mixed mesh builder tests, assembly tests, boundary-load test, integration solve test, and related implementation assumptions. Do not start Task4 until Gate B is approved and the gate status plus progress log are updated.

---

### Task4: Stress Dispatch Multi-Type Resolution

**Files:**
- Modify: `src/fem/post/stress/dispatch.py`
- Test: `tests/test_post.py`

- [x] **Step1: Add failing dispatch tests**

Add these imports and tests to `tests/test_post.py`:

```python
from fem.post.stress import dispatch
from tests.helpers.mesh_builders import make_mixed_hex8_tet4_mesh, make_mixed_tri3_quad4_mesh


class MixedStressDispatchTests(unittest.TestCase):
    def test_dispatch_resolves_compatible_mixed_solid_type_keys(self):
        mesh = make_mixed_hex8_tet4_mesh()

        self.assertEqual(dispatch.resolve_type_keys(mesh, None), ("hex8", "tet4"))
        self.assertEqual(dispatch.stress_group_for_keys(("hex8", "tet4")), "solid")
        self.assertTrue(dispatch.element_stress_supported(("hex8", "tet4")))
        self.assertTrue(dispatch.nodal_stress_supported(("hex8", "tet4")))

    def test_dispatch_resolves_compatible_mixed_plane_type_keys(self):
        mesh = make_mixed_tri3_quad4_mesh()

        self.assertEqual(dispatch.resolve_type_keys(mesh, None), ("tri3", "quad4"))
        self.assertEqual(dispatch.stress_group_for_keys(("tri3", "quad4")), "plane")
```

- [x] **Step2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressDispatchTests
```

Expected:

```text
AttributeError
```

- [x] **Step3: Add dispatch helpers**

Replace `src/fem/post/stress/dispatch.py` with this implementation:

```python
from __future__ import annotations

from typing import Any, Iterable


ELEMENT_STRESS_KEYS = {"truss2d", "tri3", "quad4", "quad8", "hex8", "tet4", "tet10"}
NODAL_STRESS_KEYS = {"tri3", "quad4", "quad8", "hex8", "tet4", "tet10"}
TYPE_GROUPS = {
    "truss2d": "line",
    "tri3": "plane",
    "quad4": "plane",
    "quad8": "plane",
    "hex8": "solid",
    "tet4": "solid",
    "tet10": "solid",
}


def resolve_type_key(mesh: Any, element_type: str | None) -> str:
    """Resolve a normalized stress exporter key for legacy single-type callers."""
    type_keys = resolve_type_keys(mesh, element_type)
    if len(type_keys) > 1:
        raise ValueError("Mixed element meshes require automatic mixed export or an explicit element_type")
    return type_keys[0]


def resolve_type_keys(mesh: Any, element_type: str | None) -> tuple[str, ...]:
    """Resolve normalized stress exporter keys while preserving mesh element order."""
    if element_type is not None:
        type_key = type_key_from_name(element_type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {element_type!r}")
        return (type_key,)

    type_keys: list[str] = []
    seen: set[str] = set()
    for elem in mesh.elements:
        type_key = type_key_from_name(elem.type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {elem.type!r}")
        if type_key not in seen:
            seen.add(type_key)
            type_keys.append(type_key)

    if not type_keys:
        raise ValueError("Cannot infer stress element type from mesh")
    return tuple(type_keys)


def stress_group_for_keys(type_keys: Iterable[str]) -> str:
    """Return one compatible stress group for a collection of type keys."""
    groups = {TYPE_GROUPS[key] for key in type_keys}
    if len(groups) != 1:
        raise ValueError(f"Mixed stress export requires compatible element groups, got {sorted(groups)}")
    return groups.pop()


def element_stress_supported(type_keys: Iterable[str]) -> bool:
    """Return whether all type keys support element stress export."""
    keys = tuple(type_keys)
    return bool(keys) and all(key in ELEMENT_STRESS_KEYS for key in keys)


def nodal_stress_supported(type_keys: Iterable[str]) -> bool:
    """Return whether all type keys support nodal stress export."""
    keys = tuple(type_keys)
    return bool(keys) and all(key in NODAL_STRESS_KEYS for key in keys)


def default_gauss_order(type_key: str) -> int | None:
    """Return the default nodal stress extrapolation order for one type key."""
    if type_key in {"quad4", "hex8"}:
        return 2
    if type_key == "quad8":
        return 3
    return None


def type_key_from_name(element_type: Any) -> str | None:
    """Normalize mesh element type names to stress exporter keys."""
    etype = str(element_type).lower()
    if "truss" in etype:
        return "truss2d"
    if "tri3" in etype:
        return "tri3"
    if "quad4" in etype:
        return "quad4"
    if "quad8" in etype:
        return "quad8"
    if "hex8" in etype:
        return "hex8"
    if "tet10" in etype:
        return "tet10"
    if "tet4" in etype:
        return "tet4"
    return None
```

- [x] **Step4: Run dispatch tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressDispatchTests
```

Expected:

```text
OK
```

- [x] **Step5: Run existing single-type post test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.PostPackageTests.test_stress_export_infers_single_element_type_from_mesh
```

Expected:

```text
OK
```

- [x] **Step6: Commit**

Run:

```powershell
git add src/fem/post/stress/dispatch.py tests/test_post.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "feat: resolve mixed stress element types"
```

Expected:

```text
[feature/mixed-elements ...] feat: resolve mixed stress element types
```

---

### Task5: Mixed Element Stress CSV Export

**Files:**
- Modify: `src/fem/post/stress/export.py`
- Modify: `src/fem/post/stress/element.py`
- Test: `tests/test_post.py`

- [x] **Step1: Add failing mixed element stress tests**

Add this test to `tests/test_post.py`:

```python
class MixedStressExportTests(unittest.TestCase):
    def test_element_stress_export_writes_mixed_solid_rows(self):
        mesh = make_mixed_hex8_tet4_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed_element_stress.csv"
            stress.export.element(mesh, np.zeros(mesh.num_dofs), csv_path)
            rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))

        self.assertEqual(rows[0], ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"])
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2"])
```

- [x] **Step2: Run test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_element_stress_export_writes_mixed_solid_rows
```

Expected:

```text
ERROR
```

The current error should come from mixed stress dispatch or missing mixed export.

- [x] **Step3: Route mixed element export**

Change `src/fem/post/stress/export.py` to use this `element` function:

```python
def element(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses to CSV. Element type is inferred when possible."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        element_export.by_type(type_keys[0], mesh, U, path, gauss_order)
        return
    element_export.mixed(type_keys, mesh, U, path, gauss_order)
```

- [x] **Step4: Add mixed element export implementation**

Add this import and these functions to `src/fem/post/stress/element.py`:

```python
from . import dispatch
```

```python
def mixed(
    type_keys: Sequence[str],
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
) -> None:
    """Export mixed element stresses for compatible stress groups."""
    if not dispatch.element_stress_supported(type_keys):
        raise ValueError(f"Element stress export is not available for {type_keys}")
    group = dispatch.stress_group_for_keys(type_keys)
    if group == "plane":
        _plane_multi(mesh, U, path, set(type_keys), gauss_order)
        return
    if group == "solid":
        _solid_multi(mesh, U, path, set(type_keys))
        return
    raise ValueError(f"Mixed element stress export is not available for group {group!r}")


def _plane_multi(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
) -> None:
    """Export mixed plane element-nodal stresses without averaging."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_ELEMENT_HEADER)
        for elem in mesh.elements:
            type_key = dispatch.type_key_from_name(elem.type)
            if type_key not in type_keys:
                continue
            order = gauss_order if gauss_order is not None else dispatch.default_gauss_order(type_key)
            node_vals, plane_type, nu = nodal_stress(mesh, elem, U, lookup, order)
            for local_idx, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[local_idx - 1].tolist()
                writer.writerow([
                    elem.id,
                    nid,
                    local_idx,
                    sig_x,
                    sig_y,
                    tau_xy,
                    von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu),
                ])


def _solid_multi(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
) -> None:
    """Export mixed solid element stresses at one representative point per element."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_HEADER)
        for elem in mesh.elements:
            type_key = dispatch.type_key_from_name(elem.type)
            if type_key not in type_keys:
                continue
            natural_coords = (0.0, 0.0, 0.0) if type_key == "hex8" else TET_CENTROID
            stress = get_element_kernel(elem.type).stress_at(mesh, elem, U, *natural_coords, lookup)
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress
            writer.writerow([
                elem.id,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])
```

- [x] **Step5: Run mixed element stress test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_element_stress_export_writes_mixed_solid_rows
```

Expected:

```text
OK
```

- [x] **Step6: Run single-type post tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.PostPackageTests.test_stress_export_infers_single_element_type_from_mesh
```

Expected:

```text
OK
```

- [x] **Step7: Commit**

Run:

```powershell
git add src/fem/post/stress/export.py src/fem/post/stress/element.py tests/test_post.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "feat: export mixed element stresses"
```

Expected:

```text
[feature/mixed-elements ...] feat: export mixed element stresses
```

---

### Task6: Mixed Nodal Stress CSV Export

**Files:**
- Modify: `src/fem/post/stress/export.py`
- Modify: `src/fem/post/stress/nodal.py`
- Test: `tests/test_post.py`

- [x] **Step1: Add failing mixed nodal stress test**

Add this test to `tests/test_post.py`:

```python
    def test_nodal_stress_export_writes_mixed_solid_nodes(self):
        mesh = make_mixed_hex8_tet4_mesh()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed_nodal_stress.csv"
            stress.export.nodal(mesh, np.zeros(mesh.num_dofs), csv_path)
            rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))

        self.assertEqual(rows[0][0], "node_id")
        self.assertEqual(len(rows), len(mesh.nodes) + 1)
        self.assertEqual({row[0] for row in rows[1:]}, {str(node.id) for node in mesh.nodes})
```

- [x] **Step2: Run test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_nodal_stress_export_writes_mixed_solid_nodes
```

Expected:

```text
ERROR
```

- [x] **Step3: Route mixed nodal export**

Change `src/fem/post/stress/export.py` to use this `nodal` function:

```python
def nodal(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export nodal stresses to CSV. Element type is inferred when possible."""
    type_keys = dispatch.resolve_type_keys(mesh, element_type)
    if len(type_keys) == 1:
        nodal_export.by_type(type_keys[0], mesh, U, path, gauss_order)
        return
    nodal_export.mixed(type_keys, mesh, U, path, gauss_order)
```

- [x] **Step4: Add mixed nodal implementation**

Add this import and these functions to `src/fem/post/stress/nodal.py`:

```python
from . import dispatch
```

```python
def mixed(
    type_keys: Sequence[str],
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
) -> None:
    """Export mixed nodal stresses for compatible stress groups."""
    if not dispatch.nodal_stress_supported(type_keys):
        raise ValueError(f"Nodal stress export is not available for {type_keys}")
    group = dispatch.stress_group_for_keys(type_keys)
    if group == "plane":
        _plane_multi(mesh, U, path, set(type_keys), gauss_order)
        return
    if group == "solid":
        _solid_multi(mesh, U, path, set(type_keys), gauss_order)
        return
    raise ValueError(f"Mixed nodal stress export is not available for group {group!r}")


def _plane_multi(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
) -> None:
    """Export mixed plane nodal stresses averaged from connected elements."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = "stress"
    nu_ref = 0.0

    for elem in mesh.elements:
        type_key = dispatch.type_key_from_name(elem.type)
        if type_key not in type_keys:
            continue
        order = gauss_order if gauss_order is not None else dispatch.default_gauss_order(type_key)
        node_vals, plane_type, nu_ref = nodal_stress(mesh, elem, U, lookup, order)
        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                sig_x, sig_y, tau_xy = (sums[nid] / counts[nid]).tolist()
            writer.writerow([
                nid,
                node.x,
                node.y,
                sig_x,
                sig_y,
                tau_xy,
                von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu_ref),
            ])


def _solid_multi(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_keys: set[str],
    gauss_order: int | None = None,
) -> None:
    """Export mixed solid nodal stresses averaged from connected element nodes."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            stress_sum = np.zeros(6, dtype=float)
            weight_sum = 0.0
            for elem in mesh.elements:
                type_key = dispatch.type_key_from_name(elem.type)
                if type_key not in type_keys or nid not in elem.node_ids:
                    continue
                order = gauss_order if gauss_order is not None else dispatch.default_gauss_order(type_key)
                node_vals = nodal_stress(mesh, elem, U, lookup, order)
                local_idx = elem.node_ids.index(nid)
                weight = element_volume(mesh, elem, lookup) if type_key in {"tet4", "tet10"} else 1.0
                stress_sum += weight * np.asarray(node_vals[local_idx], dtype=float)
                weight_sum += weight

            if weight_sum == 0.0:
                write_zero_solid_node(writer, nid, node)
                continue

            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress_sum / weight_sum
            writer.writerow([
                nid,
                node.x,
                node.y,
                node.z,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])
```

- [x] **Step5: Run mixed nodal stress test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_nodal_stress_export_writes_mixed_solid_nodes
```

Expected:

```text
OK
```

- [x] **Step6: Run post tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post
```

Expected:

```text
OK
```

- [x] **Step7: Commit**

Run:

```powershell
git add src/fem/post/stress/export.py src/fem/post/stress/nodal.py tests/test_post.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "feat: export mixed nodal stresses"
```

Expected:

```text
[feature/mixed-elements ...] feat: export mixed nodal stresses
```

---

### Task7: VTK Automatic Export For Mixed Meshes

**Files:**
- Modify: `src/fem/post/vtk/export.py`
- Modify: `src/fem/post/vtk/cells.py`
- Test: `tests/test_post.py`

- [x] **Step1: Add failing VTK auto-export test**

Add this test to `tests/test_post.py`:

```python
    def test_vtk_export_from_result_materializes_mixed_stress_csvs(self):
        result = make_zero_result(make_mixed_hex8_tet4_mesh(), "mixed_vtk")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            vtk.export.from_result(result, output_dir=output_dir)

            self.assertTrue((output_dir / "mixed_vtk_nodal_displacement.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_element_stress.csv").exists())
            self.assertTrue((output_dir / "mixed_vtk_nodal_stress.csv").exists())
            vtk_text = (output_dir / "mixed_vtk.vtk").read_text(encoding="utf-8")

        self.assertIn("CELL_TYPES 2", vtk_text)
        self.assertIn("\n12\n", vtk_text)
        self.assertIn("\n10\n", vtk_text)
```

- [x] **Step2: Run test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_vtk_export_from_result_materializes_mixed_stress_csvs
```

Expected:

```text
FAIL
```

The expected failure is that mixed stress CSVs are skipped.

- [x] **Step3: Update supported stress path detection**

Replace `_supported_stress_paths` in `src/fem/post/vtk/export.py` with:

```python
def _supported_stress_paths(mesh, paths: dict[str, Path]) -> dict[str, Optional[Path]]:
    """Return default stress paths supported by all mesh element types."""
    from ..stress import dispatch

    try:
        type_keys = dispatch.resolve_type_keys(mesh, None)
        dispatch.stress_group_for_keys(type_keys)
    except ValueError:
        return {"element_stress": None, "nodal_stress": None}

    return {
        "element_stress": paths["element_stress"] if dispatch.element_stress_supported(type_keys) else None,
        "nodal_stress": paths["nodal_stress"] if dispatch.nodal_stress_supported(type_keys) else None,
    }
```

- [x] **Step4: Make unsupported VTK cells fail clearly**

In `src/fem/post/vtk/cells.py`, replace the final `else: continue` branch with:

```python
        else:
            raise ValueError(f"Unsupported element type for VTK export: {elem.type}")
```

- [x] **Step5: Add unsupported VTK test**

Add this test to `tests/test_post.py`:

```python
    def test_vtk_cells_report_unsupported_element_type(self):
        mesh = make_mixed_hex8_tet4_mesh()
        mesh.elements[1].type = "UnsupportedSolid"
        result = make_zero_result(mesh, "unsupported_vtk")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported element type for VTK export: UnsupportedSolid"):
                vtk.export.from_result(result, output_dir=Path(tmp))
```

- [x] **Step6: Run VTK tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post.MixedStressExportTests.test_vtk_export_from_result_materializes_mixed_stress_csvs tests.test_post.MixedStressExportTests.test_vtk_cells_report_unsupported_element_type
```

Expected:

```text
OK
```

- [x] **Step7: Run all post tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_post
```

Expected:

```text
OK
```

- [x] **Step8: Commit**

Run:

```powershell
git add src/fem/post/vtk/export.py src/fem/post/vtk/cells.py tests/test_post.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "feat: export mixed element VTK results"
```

Expected:

```text
[feature/mixed-elements ...] feat: export mixed element VTK results
```

- [x] **Step9: Gate C review**

Stop after the VTK export commit. Ask for review of stress dispatch behavior, mixed element and nodal stress CSV output, VTK auto-export behavior, and unsupported VTK element errors. Do not start Task8 until Gate C is approved and the gate status plus progress log are updated.

---

### Task8: Runnable Mixed Element Example

**Files:**
- Create: `examples/mixed_hex8_tet4.py`
- Modify: `README.md`
- Test: `tests/test_integration.py`

- [x] **Step1: Create example script**

Create `examples/mixed_hex8_tet4.py`:

```python
# Example: mixed Hex8 and Tet4 linear static model.

from fem import materials, post, solvers, steps
from fem.core import Element3D, ElementSet, FEMModel, Node3D, NodeSet
from fem.core.mesh import HexMesh3D


nodes = [
    Node3D(1, 0.0, 0.0, 0.0),
    Node3D(2, 1.0, 0.0, 0.0),
    Node3D(3, 1.0, 1.0, 0.0),
    Node3D(4, 0.0, 1.0, 0.0),
    Node3D(5, 0.0, 0.0, 1.0),
    Node3D(6, 1.0, 0.0, 1.0),
    Node3D(7, 1.0, 1.0, 1.0),
    Node3D(8, 0.0, 1.0, 1.0),
    Node3D(9, 2.0, 0.0, 0.0),
]
elements = [
    Element3D(1, [1, 2, 3, 4, 5, 6, 7, 8], "Hex8"),
    Element3D(2, [2, 9, 3, 6], "Tet4"),
]
mesh = HexMesh3D(nodes=nodes, elements=elements)
model = FEMModel(mesh=mesh, name="mixed_hex8_tet4")

model.element_sets["hexes"] = ElementSet("hexes", (1,))
model.element_sets["tets"] = ElementSet("tets", (2,))
model.node_sets["fixed"] = NodeSet("fixed", (1, 4, 5, 8))
model.node_sets["tip"] = NodeSet("tip", (9,))

steel = materials.linear_elastic.material("steel", E=210000.0, nu=0.3)
aluminum = materials.linear_elastic.material("aluminum", E=70000.0, nu=0.33)
materials.add(model, steel)
materials.add(model, aluminum)
materials.assign(model, "steel", "hexes")
materials.assign(model, "aluminum", "tets")

load_step = steps.static("pull")
steps.displacement(load_step, "fixed", components=(1, 2, 3))
steps.nodal_load(load_step, "tip", component=1, value=100.0)
steps.add(model, load_step)

result = solvers.static_linear.solve(model, "pull")

print("Element types:", [elem.type for elem in mesh.elements])
print("Element materials:", [elem.props["material"] for elem in mesh.elements])
print("Tip ux:", float(result.U[mesh.global_dof(9, 0)]))

post.vtk.export.from_result(result, output_dir=r"results")
```

- [x] **Step2: Run the example**

Run:

```powershell
.\.venv\Scripts\python.exe examples\mixed_hex8_tet4.py
```

Expected output contains:

```text
Element types: ['Hex8', 'Tet4']
Element materials: ['steel', 'aluminum']
Tip ux:
```

Expected files:

```text
results\mixed_hex8_tet4_nodal_displacement.csv
results\mixed_hex8_tet4_element_stress.csv
results\mixed_hex8_tet4_nodal_stress.csv
results\mixed_hex8_tet4.vtk
```

- [x] **Step3: Add example smoke test**

Add this test to `tests/test_integration.py`:

```python
class MixedElementExampleTests(unittest.TestCase):
    def test_mixed_hex8_tet4_example_import_runs(self):
        namespace = runpy.run_path("examples/mixed_hex8_tet4.py")

        self.assertIn("result", namespace)
        result = namespace["result"]
        self.assertEqual([elem.type for elem in result.model.mesh.elements], ["Hex8", "Tet4"])
```

Add this import near the top of `tests/test_integration.py`:

```python
import runpy
```

- [x] **Step4: Run example smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_integration.MixedElementExampleTests
```

Expected:

```text
OK
```

- [x] **Step5: Update README**

Add this bullet under current capabilities in `README.md`:

```markdown
- 同一维度、同一节点自由度布局的混合单元模型，例如`Hex8`+`Tet4`和`Tri3Plane`+`Quad4Plane`。
```

Add this command under examples:

```powershell
python examples\mixed_hex8_tet4.py
```

- [x] **Step6: Commit**

Run:

```powershell
git add examples/mixed_hex8_tet4.py README.md tests/test_integration.py docs/MIXED_ELEMENTS_PLAN.md
git commit -m "docs: add mixed element example"
```

Expected:

```text
[feature/mixed-elements ...] docs: add mixed element example
```

---

### Task9: Final Verification And Status Update

**Files:**
- Modify: `docs/MIXED_ELEMENTS_PLAN.md`
- Optional modify: `docs/TASK_PLAN_0519-0602.md`

- [x] **Step1: Run focused mixed tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_core.MixedMeshBuilderTests tests.test_assemble.MixedAssemblyTests tests.test_boundary.MixedBoundaryLoadTests tests.test_integration.MixedElementWorkflowIntegrationTests tests.test_integration.MixedElementExampleTests tests.test_post.MixedStressDispatchTests tests.test_post.MixedStressExportTests
```

Expected:

```text
OK
```

- [x] **Step2: Run full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected:

```text
OK
```

- [x] **Step3: Verify example output files exist**

Run:

```powershell
Test-Path results\mixed_hex8_tet4.vtk
Test-Path results\mixed_hex8_tet4_element_stress.csv
Test-Path results\mixed_hex8_tet4_nodal_stress.csv
```

Expected:

```text
True
True
True
```

- [x] **Step4: Update progress log and task plan**

In `docs/MIXED_ELEMENTS_PLAN.md`, set `Next action` to:

```text
Ready for review.
```

If `docs/TASK_PLAN_0519-0602.md` is still the active project tracker, mark task1 acceptance items as completed with a short note that the supported mixed scope is same dimension and same node DOF layout.

- [x] **Step5: Review diff**

Run:

```powershell
git diff --stat develop...HEAD
git diff -- docs/MIXED_ELEMENTS_PLAN.md
```

Expected:

```text
```

The first command should show only mixed-element implementation, tests, docs, and the example. The second command should show all task checkboxes and the progress log aligned with the completed work.

- [x] **Step6: Gate D review**

Stop before the final documentation/status commit. Ask for review of the full diff against `develop`, focused mixed test output, full test output, example output files, README changes, and this plan's progress log. Do not create the final commit until Gate D is approved and the gate status plus progress log are updated.

- [x] **Step7: Final commit**

Run:

```powershell
git add docs/MIXED_ELEMENTS_PLAN.md docs/TASK_PLAN_0519-0602.md
git commit -m "docs: record mixed element completion"
```

Expected:

```text
[feature/mixed-elements ...] docs: record mixed element completion
```

If `docs/TASK_PLAN_0519-0602.md` was not changed, commit only the plan file.

---

## Self-Review

- Spec coverage: the plan covers model layer, assembly layer, material assignment, boundary and load handling, post-processing, VTK export, unsupported type errors, one runnable example, and tests.
- Scope control: the implementation is limited to same-dimension and same node DOF layout mixed meshes, which matches the current `DofMap` architecture.
- Unfinished-marker scan: no incomplete sections or unnamed implementation steps are present.
- Type consistency: new helper names are `make_mixed_hex8_tet4_mesh` and `make_mixed_tri3_quad4_mesh`; new dispatch names are `resolve_type_keys`, `stress_group_for_keys`, `element_stress_supported`, `nodal_stress_supported`, and `default_gauss_order`.
