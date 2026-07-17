# Gmsh Truss2/Beam2 Line-Element Integration Plan

## Goal

Extend the existing scripted Gmsh geometry and mesh-import workflow to spatial
two-node line elements:

- model one-dimensional topology with typed OCC point and straight-line
  entities;
- generate a Gmsh `Line 2` mesh through the existing facade lifecycle;
- require the caller to select either `Truss2` or `Beam2` explicitly when a
  one-dimensional mesh is imported;
- construct a spatial `Mesh3D` with three degrees of freedom per node for
  `Truss2` or six degrees of freedom per node for `Beam2`;
- convert physical curve groups into `ElementSet` objects and physical point
  groups into `NodeSet` objects;
- reuse the existing material, section, load, solver, stress-recovery, and VTK
  paths without inventing analysis semantics in the geometry layer.

The target pipeline is:

```text
typed OCC points/lines -> gmsh.model.mesh.generate(1)
    -> fem.io.gmsh.from_model(dimension=1, line_element_type=...)
    -> Mesh3D + physical sets -> FEMModel
    -> materials/sections + steps -> solve -> CSV/VTK
```

This plan supports one line formulation per imported mesh. A caller can build
either a complete `Truss2` model or a complete `Beam2` model from the same
geometry workflow. Mixing both formulations in one mesh remains a separate
future task because the current `DofMap` assigns one uniform number of degrees
of freedom to every node.

## Verified Starting State

The current checkout already provides the analysis kernels and downstream
workflow:

- `src/fem/elements/line.py` contains `Truss2Kernel` and `Beam2Kernel`;
- the spatial geometry helpers are named `line3d_geometry()` and
  `beam3d_geometry()`;
- `Truss2` uses three translational degrees of freedom per node and requires
  positive `E` and `area`, with optional nonnegative `rho`;
- `Beam2` uses three translations plus three rotations per node and requires
  `E`, `nu`, and one of the existing standard sections;
- `src/fem/elements/beam_section.py` remains the single authority for Beam2
  section validation and derived properties;
- `materials.assign()` and `materials.apply_sections()` already populate
  effective element properties before stiffness assembly;
- nodal loads, body forces, Beam2 line loads, linear-static solution,
  Truss2 element stress, Beam2 nodal axial-stress envelopes, and VTK line
  cells are already supported.

The missing bridge is concentrated in the Gmsh geometry and adapter layers:

- `GeometryModel` accepts only topological dimensions two and three;
- the typed facade does not expose point or straight-line primitives;
- `fem.io.gmsh.from_model()` accepts only dimensions two and three;
- Gmsh type 1, `Line 2`, is registered only as a boundary element and not as
  a top-dimensional FEM element;
- three-dimensional imports currently create `Mesh3D` with three degrees of
  freedom per node unconditionally;
- the codimension-one physical-group branch assumes a two-dimensional edge or
  a three-dimensional surface and would misclassify a point group in a
  one-dimensional model.

The verified pre-plan regression baseline after the spatial-helper rename is:

- affected line-element tests: 140 passed;
- complete test suite: 853 passed;
- Ruff checks for the changed line-element source and tests: passed.

## Global Constraints

- Keep `fem.io.gmsh` as the single Gmsh-to-FEM conversion bridge. The geometry
  facade must call it rather than duplicate mesh conversion.
- Keep the geometry layer responsible only for geometry, topology, physical
  labels, meshing, and explicit formulation selection. It must not create
  materials, sections, loads, constraints, or analysis steps.
- Define `dimension=1` as one-dimensional topology embedded in three-dimensional
  space. Preserve all node coordinates as `Node3D`; do not project line models
  into an XY plane.
- Require one canonical `line_element_type`, exactly `"Truss2"` or `"Beam2"`,
  for every one-dimensional import or facade mesh generation.
- Reject `line_element_type` for dimensions two and three so that an irrelevant
  argument is never silently ignored.
- Keep the first implementation homogeneous: every top-dimensional element in
  one imported mesh uses the selected line formulation.
- Support only first-order Gmsh `Line 2` elements. Do not reinterpret a
  second-order `Line 3` element as a two-node FEM element.
- Reject `order=2` and `recombine=True` for a one-dimensional facade before
  changing Gmsh options or attempting mesh generation.
- Preserve the existing one-mesh-attempt lifecycle, Gmsh option restoration,
  typed-reference ownership, stale-reference checks, and contextual errors.
- Require explicit shared point entities for connected members. Do not merge
  coincident points or infer connectivity from equal coordinates.
- Keep the existing Beam2 automatic local frame unchanged.
- For rectangular Beam2 sections, keep `height` along local y and `width` along
  local z. Consequently, (I_{yy}=height \times width^3/12) and
  (I_{zz}=width \times height^3/12).
- Do not add `local_y`, `local_z`, a section roll angle, an orientation vector,
  or any independent cross-section rotation input.
- Do not add compatibility aliases for unsupported line formulations or higher
  order line elements.
- Do not modify README files or CI configuration.

## Final Public API Contract

### One-Dimensional Geometry Model

Extend the constructor contract:

```python
def model(
    name: str,
    *,
    dimension: Literal[1, 2, 3],
) -> GeometryModel:
    ...
```

`GeometryModel.dimension` remains immutable. For `dimension=1`, the facade
accepts arbitrary finite three-dimensional coordinates and treats curves as
top-dimensional entities.

The existing two- and three-dimensional contracts remain unchanged.

### Point and Straight-Line Primitives

Add the typed one-dimensional primitives:

```python
def point(
    self,
    x: float,
    y: float,
    z: float = 0.0,
) -> EntityRef:
    ...

def line(
    self,
    start: EntityRef,
    end: EntityRef,
) -> EntityRef:
    ...
```

For this plan, these primitives are public on a `dimension=1` facade.

`point()` must:

- require the `BUILDING` state;
- validate finite x, y, and z coordinates before calling Gmsh;
- call OCC `addPoint()` and let Gmsh allocate the tag;
- return a live facade-owned dimension-zero `EntityRef`.

`line()` must:

- require the `BUILDING` state;
- require two live dimension-zero references owned by the same facade;
- reject the same endpoint reference in both positions;
- call OCC `addLine(start.tag, end.tag)` and let Gmsh allocate the tag;
- return a live facade-owned dimension-one `EntityRef`.

Different point entities with equal coordinates are not automatically merged.
Callers create connectivity by reusing the same point reference. Intersecting
members that do not share an endpoint must be split deliberately through the
existing typed `fragment()` workflow before physical groups are frozen.

Rectangle, disk, box, cylinder, and extrusion operations that would create an
entity above dimension one must fail before a backend mutation in a
one-dimensional facade.

### Explicit Line Formulation Selection

Extend both low-level and facade generation entry points with:

```python
line_element_type: Literal["Truss2", "Beam2"] | None = None
```

The affected methods are:

- `fem.io.gmsh.from_model()`;
- `GeometryModel.generate_mesh()`;
- `GeometryModel.generate_fem_model()`;
- the facade's private mesh-generation helper.

Validation rules:

- `dimension=1` requires exactly `"Truss2"` or `"Beam2"`;
- dimensions two and three require `None`;
- invalid values fail before reading mesh blocks or changing Gmsh state;
- the facade passes the validated canonical value explicitly to
  `fem.io.gmsh.from_model()`;
- the selected value applies to all top-dimensional Gmsh elements in that
  import and is stored in adapter metadata.

Gmsh `Line 2` describes interpolation topology only. The adapter must never
guess the FEM formulation from physical names, entity tags, material names, or
model names.

### One-Dimensional Import Result

For `line_element_type="Truss2"`, construct:

```python
Mesh3D(
    nodes=[Node3D(...)],
    elements=[Element3D(..., type="Truss2", props={})],
    dofs_per_node=3,
)
```

For `line_element_type="Beam2"`, construct:

```python
Mesh3D(
    nodes=[Node3D(...)],
    elements=[Element3D(..., type="Beam2", props={})],
    dofs_per_node=6,
)
```

Both paths must:

- preserve positive Gmsh node and element tags;
- preserve full x, y, and z coordinates;
- preserve Gmsh `Line 2` endpoint ordering;
- reject missing, duplicate, malformed, or zero/negative tags through the
  existing adapter validation style;
- reject an empty top-dimensional mesh;
- leave element properties empty so that the existing model-assignment layer
  remains authoritative.

Register Gmsh type 1 as a top-dimensional topology specification without
hard-coding either FEM formulation into the topology record. Element building
uses the separately validated `line_element_type`.

A low-level import that encounters Gmsh type 8, `Line 3`, must raise a clear
unsupported-element error stating that the current FEM line formulations are
two-node, first-order elements.

### Physical Group Mapping

For a one-dimensional import:

- a dimension-one physical curve group becomes an `ElementSet`;
- a dimension-zero physical point group becomes a `NodeSet`;
- point groups do not create `Edge` or `Surface` objects;
- `GmshImportResult.edges` and `.surfaces` remain empty;
- physical names retain the existing deterministic namespace and duplicate
  checks;
- empty physical groups follow the existing skipped-group metadata contract;
- metadata records the physical group dimension, tag, and resulting kind.

The physical-group implementation must distinguish these cases explicitly:

```text
model dimension 1, group dimension 0 -> NodeSet only
model dimension 2, group dimension 1 -> NodeSet + Edge
model dimension 3, group dimension 2 -> NodeSet + Surface
```

Do not route a one-dimensional endpoint through boundary-owner matching.

### Truss2 Analysis Contract

The imported mesh supplies only topology. A normal Truss2 model uses the
existing assignment workflow:

```python
steel = materials.linear_elastic.material("steel", E=210.0e9, nu=0.3)
materials.add(model, steel)
materials.assign(model, steel, "MEMBERS", area=1.0e-4)
```

The Truss2 kernel continues to consume `E`, `area`, and optional `rho`.
No new truss section object or section-type registry is introduced.

The end-to-end acceptance model is a straight spatial bar with physical groups
for all members, the fixed endpoint, and the loaded endpoint. Constrain the
transverse degrees of freedom so that the linear system contains only the
intended axial deformation. Verify the analytical displacement

\[
u=\frac{FL}{EA}
\]

and axial stress

\[
\sigma=\frac{F}{A}.
\]

### Beam2 Analysis and Section Contract

The imported Beam2 elements use the existing six-degree-of-freedom mesh,
material assignment, section parsing, loads, recovery, and output paths.

The automatic frame remains:

1. local x is the normalized direction from node i to node j;
2. global Z is projected onto the plane normal to local x to obtain local z;
3. a beam parallel to global Z uses the existing global-Y fallback;
4. local y is `cross(local_z, local_x)`;
5. the global-to-local rotation rows are local x, local y, and local z.

For a rectangular section:

- `height` is the dimension along the automatically generated local y axis;
- `width` is the dimension along the automatically generated local z axis;
- bending displacement in local y uses `Izz`;
- bending displacement in local z uses `Iyy`;
- bending stress about local y uses `width / 2`;
- bending stress about local z uses `height / 2`.

Gmsh supplies the centerline and node order only. It does not add or override a
Beam2 cross-section direction. This plan contains no section-rotation feature.

The end-to-end acceptance model is a straight cantilever aligned with global X
and assigned a nonsquare rectangle. Verify separately that:

- a global-Y tip load follows the local-y bending response governed by `Izz`;
- a global-Z tip load follows the local-z bending response governed by `Iyy`;
- the fixed endpoint uses all six displacement/rotation constraints;
- the model can also receive the existing Beam2 distributed line load through
  the imported member `ElementSet`;
- VTK output contains displacement, rotation, and the existing Beam2 axial
  stress-envelope scalars.

### Lifecycle and Failure Contract

One-dimensional generation preserves the current facade state machine:

- geometry construction is available only in `BUILDING`;
- a successful physical group moves the facade to `LABELED` and freezes
  topology;
- only one mesh attempt is permitted;
- successful import moves to `MESHED`;
- mesh, adapter, or `to_fem_model()` failure moves to `MESH_FAILED`;
- option restoration runs after both successful and failed mesh attempts;
- an imported `FEMModel` remains usable after the Gmsh context exits.

Validate the line formulation, order, recombination flag, and presence of at
least one dimension-one OCC entity before committing the mesh attempt whenever
the existing lifecycle permits that validation order.

## Task 1: Add Top-Dimensional Gmsh Line 2 Import

### Target Files

- `src/fem/io/gmsh.py`
- `tests/test_gmsh_io.py`

### Steps

1. Expand adapter dimension types and validation to accept one, two, or three.
2. Add the optional `line_element_type` argument and validate its
   dimension-dependent contract before backend access.
3. Register Gmsh type 1 as a valid top-dimensional first-order topology.
4. Thread the selected formulation through record building and element
   construction without embedding it in Gmsh topology metadata.
5. Build one-dimensional nodes as `Node3D` and elements as `Element3D`.
6. Select three or six degrees of freedom per node from the formulation.
7. Preserve the current two-dimensional orientation normalization and
   three-dimensional element mapping unchanged.
8. Adjust physical-group classification so endpoint groups create only
   `NodeSet` objects.
9. Add formulation metadata to one-dimensional import results.
10. Add explicit rejection coverage for Gmsh `Line 3` and unsupported types.

### Acceptance Criteria

- The same valid Gmsh type-1 block imports as Truss2 or Beam2 according to the
  explicit argument.
- Truss2 imports use `Mesh3D(..., dofs_per_node=3)`.
- Beam2 imports use `Mesh3D(..., dofs_per_node=6)`.
- Arbitrary nonzero z coordinates are retained exactly.
- Missing or invalid `line_element_type` fails before any mesh-block read.
- A line formulation supplied for dimension two or three is rejected.
- Physical curves become element sets and physical points become node sets.
- One-dimensional imports produce no edges or surfaces.
- Gmsh `Line 3` is rejected with a first-order/two-node explanation.
- Existing two- and three-dimensional adapter tests remain unchanged and pass.

## Task 2: Add Typed One-Dimensional OCC Geometry and Generation

### Target Files

- `src/fem/geometry/gmsh.py`
- `src/fem/geometry/__init__.py` only if export metadata needs adjustment
- `tests/test_gmsh_geometry.py`

### Steps

1. Expand `GeometryModel`, `model()`, and the mesh-dimension validator to
   accept `dimension=1`.
2. Add typed `point()` and `line()` primitives with pre-backend validation,
   ownership checks, liveness checks, and Gmsh-allocated tags.
3. Reject higher-dimensional primitives and extrusion in a one-dimensional
   facade before backend mutation.
4. Preserve spatial translation and rotation for one-dimensional entities;
   retain the existing XY-plane restrictions only for dimension two.
5. Thread `line_element_type` through `generate_mesh()`,
   `generate_fem_model()`, and the private generation helper.
6. Reject one-dimensional `order=2` and `recombine=True` before option changes.
7. Require at least one live dimension-one OCC entity before generation.
8. Call `gmsh.model.mesh.generate(1)` and pass the active model plus canonical
   formulation to `fem.io.gmsh.from_model()`.
9. Preserve option snapshot/restoration and one-attempt state transitions.
10. Add a real-Gmsh test in which two lines reuse one point and import as a
    connected spatial mesh.
11. Add a real-Gmsh fragment test for intersecting curves if Gmsh OCC returns
    stable point/curve ownership under the existing typed boolean contract.

### Acceptance Criteria

- A one-dimensional context can create, query, transform, label, and mesh
  typed points and lines.
- Reusing a point reference produces one shared mesh node.
- Separate coincident points are not silently merged by the facade.
- Invalid endpoints, cross-model references, stale references, and identical
  endpoint references fail before `addLine()`.
- Topology cannot be changed after the first successful physical group.
- The selected line formulation is forwarded exactly once to the adapter.
- Mesh options are restored after success, Gmsh failure, adapter failure, and
  FEM-model conversion failure.
- Existing dimension-two and dimension-three lifecycle tests remain green.

## Task 3: Complete the Truss2 Vertical Slice

### Target Files

- `examples/gmsh_geometry_truss2.py`
- `tests/test_gmsh_geometry.py` or a focused integration-test module
- existing Truss2 solver/post-processing tests only where shared fixtures are
  required

### Steps

1. Build a spatial straight bar entirely through the typed geometry facade.
2. Create `MEMBERS`, `FIXED`, and `TIP` physical groups from the curve and
   endpoint references.
3. Generate a `Truss2` FEM model with an explicit mesh size.
4. Register a linear-elastic material and assign cross-sectional area to the
   imported member set.
5. Apply a fully fixed first endpoint, constrain transverse displacement at
   the loaded endpoint, and apply an axial nodal load.
6. Solve after leaving the Gmsh context.
7. Verify analytical tip displacement and axial stress.
8. Export VTK and verify line-cell type plus displacement and Truss2 stress
   fields.
9. Keep the example free of direct Gmsh initialization, synchronization,
   finalization, and raw dimension-tag tuples.

### Acceptance Criteria

- Imported member elements are all `Truss2` with two-node connectivity.
- The mesh has exactly three degrees of freedom per node.
- Physical names are directly usable as model sets.
- Material and area assignment requires no direct element-property mutation.
- Numerical displacement and stress match the analytical bar solution within
  an explicit tolerance.
- The model solves and exports after the geometry context has closed.
- The example runs headlessly from a clean process.

## Task 4: Complete the Beam2 Vertical Slice

### Target Files

- `examples/gmsh_geometry_beam2.py`
- `tests/test_gmsh_geometry.py` or a focused integration-test module
- existing Beam2 load and post-processing tests only where shared fixtures are
  required

### Steps

1. Build a straight cantilever through typed points and one or more connected
   lines.
2. Create `MEMBERS`, `FIXED`, and `TIP` physical groups.
3. Generate a `Beam2` FEM model explicitly.
4. Register a linear-elastic material and assign a nonsquare rectangular
   section using `height,width`.
5. Fix all six degrees of freedom at the root.
6. Verify local-y bending with a global-Y tip load and local-z bending with a
   global-Z tip load.
7. Verify the existing distributed Beam2 line-load path against the imported
   member set.
8. Recover the existing Beam2 nodal axial-stress envelope.
9. Export VTK and inspect displacement, rotation, and stress scalars.
10. Keep the automatic frame and fixed section-dimension convention unchanged.

### Acceptance Criteria

- Imported member elements are all `Beam2` with two-node connectivity.
- The mesh has exactly six degrees of freedom per node.
- Global-Y and global-Z cantilever responses use the expected `Izz` and `Iyy`
  values derived from `height,width`.
- `height` remains local-y size and `width` remains local-z size in stiffness
  and stress recovery.
- No orientation or section-rotation property appears in the example, adapter,
  element properties, or public API.
- Existing nodal, body, and distributed line loads remain available.
- The model solves and exports after the geometry context has closed.
- The example runs headlessly from a clean process.

## Task 5: Regression, Error Quality, and Contract Audit

### Target Files

- `tests/test_gmsh_io.py`
- `tests/test_gmsh_geometry.py`
- `tests/test_elements.py`
- `tests/test_line3d.py`
- `tests/test_line_load.py`
- `tests/test_material_validation.py`
- `tests/test_model_solver_validation.py`
- source and example files touched by Tasks 1 through 4

### Steps

1. Audit every new validation path for pre-backend failure where promised.
2. Verify top-dimensional Line 2 and boundary Line 2 specifications coexist
   without changing two-dimensional boundary matching.
3. Verify malformed block counts, connectivity, tags, coordinates, physical
   groups, and backend return values retain contextual adapter errors.
4. Verify order and recombination rejection does not leak changed Gmsh global
   options.
5. Search active source, tests, and examples for accidental `Line3`, `Beam3`,
   mixed line-formulation inference, or section-orientation APIs.
6. Confirm the spatial helper names remain `line3d_geometry()` and
   `beam3d_geometry()`.
7. Run both new examples in fresh processes.
8. Run focused adapter/facade, line-element, post-processing, and full-suite
   tests.
9. Run Ruff and whitespace checks.
10. Confirm README and CI files remain unchanged.

### Acceptance Criteria

- Error messages identify the requested dimension, Gmsh element family, and
  invalid formulation or option where relevant.
- Adapter failures do not return partially built FEM data.
- Facade failures preserve the documented terminal state and restore Gmsh
  options.
- Existing 2D/3D physical boundary topology remains unchanged.
- Existing Truss2 and Beam2 CSV workflows remain unchanged.
- Both new Gmsh line examples run successfully.
- Ruff, focused tests, and the complete suite pass.
- No README or CI file is modified.

## Final Verification Gate

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gmsh_io.py tests\test_gmsh_geometry.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_elements.py tests\test_line3d.py tests\test_line_load.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_material_validation.py tests\test_model_solver_validation.py tests\test_post.py -q
.\.venv\Scripts\python.exe examples\gmsh_geometry_truss2.py
.\.venv\Scripts\python.exe examples\gmsh_geometry_beam2.py
.\.venv\Scripts\python.exe -m pytest -q
ruff check src tests examples
git diff --check
git status --short
```

If pytest temporary-directory creation is blocked by the Codex filesystem
sandbox, rerun the same project-virtual-environment commands outside the
sandbox. Do not change repository code to work around that environment issue.

The examples must run in separate fresh processes so one successful run cannot
hide leaked Gmsh model, option, or initialization state from the other.

## Overall Acceptance Criteria

- `GeometryModel` and `fem.io.gmsh` accept topological dimension one without
  changing existing dimensions two and three.
- Typed point and straight-line geometry can produce a connected spatial mesh.
- The caller explicitly selects `Truss2` or `Beam2`; no name-based formulation
  inference exists.
- Truss2 imports use three degrees of freedom per node.
- Beam2 imports use six degrees of freedom per node.
- Gmsh `Line 3` and one-dimensional second-order generation are rejected.
- Physical curves become element sets and physical points become node sets.
- One-dimensional imports create no edge or surface collections.
- Imported elements receive material and section data only through existing
  model APIs.
- Truss2 displacement and stress match the analytical axial-bar solution.
- Beam2 weak- and strong-axis responses match the fixed
  `height -> local y`, `width -> local z` contract.
- Beam2 keeps the current automatic local frame and exposes no section-rotation
  input.
- Both models solve and export valid line-cell VTK output after the Gmsh
  context exits.
- Existing 2D/3D Gmsh, CSV, solver, load, stress, and VTK tests remain green.
- The complete test suite and Ruff checks pass.
- README and CI files remain unchanged.

## Out of Scope

- mixing `Truss2` and `Beam2` in one mesh or selecting formulation by physical
  group;
- variable degrees of freedom per node or element-specific global DOF maps;
- Gmsh `Line 3`, `Truss3`, `Beam3`, or polynomial order above one;
- curved typed primitives such as arcs, circles, splines, or B-splines;
- automatic coincident-point merging or automatic intersection splitting;
- user-defined Beam2 local axes, orientation vectors, section roll angles, or
  arbitrary cross-section rotation;
- changes to the fixed rectangle convention where `height` follows local y and
  `width` follows local z;
- automatic material, section, load, support, or analysis-step creation;
- new line-element kernels, Timoshenko theory, geometric nonlinearity,
  tension-only trusses, cable behavior, releases, hinges, or offsets;
- `.msh` file reading, STEP/IGES/BREP import, Gmsh GUI integration, or a generic
  CAD backend;
- new Truss2/Beam2 result formats unrelated to proving the Gmsh bridge;
- README or CI changes.

## Definition of Done

The plan is complete when a caller can use only public `fem.geometry.gmsh`
operations to build a connected spatial line model, label its members and
endpoints, select either Truss2 or Beam2 explicitly, leave the Gmsh context,
assign existing materials and sections, solve, and export the existing result
formats. The implementation must preserve the fixed Beam2 section convention,
must not introduce section rotation, and must retain all existing two- and
three-dimensional behavior.
