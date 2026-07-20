# `fem.geometry` Stateless Core Refactor Phase 2 Implementation Handoff

## Outcome

Phase 2 is implemented as a behavior-preserving extraction from the temporary
Gmsh model monolith. Public geometry errors and value objects now have stable
backend-independent modules, shared scalar validation is backend-free, and pure
topology mathematics is separated from stateful OCC orchestration.

The root `fem.geometry` facade still exports exactly the same 16 names.
`GeometryModel` and `model()` remain Gmsh-backed, the external `gmsh` package is
still loaded lazily, and mesh-only references, errors, policies, and runtime
operations remain in `_gmsh.model`.

No files were staged or committed, following the implementation instruction.

## Phase 1 Review Boundary

Phase 1 was still an uncommitted working-tree patch on top of:

~~~text
cc95aa6a897b756ebd88f665ac3445187c65e993
fix(gmsh): harden advanced feature topology
~~~

Before Phase 2 edits, the current Phase 1 tree was recorded as a canonical
content manifest. Each existing file uses its SHA-256 content hash; the deleted
tracked module uses its Git blob identity. Entries were sorted, joined with LF,
encoded as UTF-8, and hashed with SHA-256.

~~~text
manifest-sha256=01447e99ff264510a9a9b32b115393377025a2fb0ebf56aa328d385d7e6d3067

 D|src/fem/geometry/gmsh.py|git-blob:d7252c0db99940eaaf2b40be018f376c53c27e8d
 M|examples/gmsh_geometry_auto_mesh.py|a3866804ef6a7b8f8b0570026ecf9c413d6ead4a1bb1f0a66b8897d6e7aee8ff
 M|examples/gmsh_geometry_beam2.py|a39731043324f0beeaf01e8dedc99ceb1ba09689d6873d3175275e30c7e26456
 M|examples/gmsh_geometry_filleted_box.py|8f21dfbd2acf1f3edbfb25210beb28846d84094907ba8531f71874f72459e2b0
 M|examples/gmsh_geometry_irregular_plate.py|a9f10ad58516e00065a16d27c447dad8b2e8e5d1dfc28ba73987ddb5f2d2e918
 M|examples/gmsh_geometry_local_refinement.py|7e75b8ca4718d6a1e4e7ca04c593569315a48d508e846fb270e810c4adf1d391
 M|examples/gmsh_geometry_partitioned_plate.py|bb1de92fe538bfd40f90edcdc775bec341cb0001885a714d3dd94dff0c9eb1fe
 M|examples/gmsh_geometry_plate.py|258626cf5e02f409b5137370c5a0a63d09f597ee394fb355bc9a9e0ca3fa04c8
 M|examples/gmsh_geometry_revolved_solid.py|353a529d1ff723b458f99d16c5deb22a42c31ee6f76287df88f0e9a60549f4d1
 M|examples/gmsh_geometry_structured_quad8.py|3f1af02279d10f65a035c015eb668b983db47ac2a2161e9d2c566832bb6a375c
 M|examples/gmsh_geometry_swept_solid.py|6f7c8566c97b46b9a082775733043fbe24aad22074d11496c9be0828cc5006c2
 M|examples/gmsh_geometry_truss2.py|f4513fdd2368757dd407ebedfee14da54be4fa298557b456341d1f707dfdbb1d
 M|src/fem/geometry/__init__.py|e6fdeee4bfd3035e705d521edcfb7edd5aa062dda396d06cb2ebfd2f77d1480e
 M|src/fem/mesh/gmsh.py|675620681c2ed98f0848ee880c012bc0312f3fb504d17585fd927f01d753ceeb
 M|tests/test_gmsh_advanced_operations.py|3c187f427025f5108a9a4c2196827d308e32d218def7073db254f8c2c404f5bf
 M|tests/test_gmsh_foundational_operations.py|7f23c3d481646ed32a34d8e7a01bae543899b01e726efb9ba16a16eb7ef22810
 M|tests/test_gmsh_geometry.py|d5d9c06249125fba3e2716a29d21419dae1367207feed1e8d6825ca527c8ecdb
 M|tests/test_gmsh_local_refinement.py|d8b54bd8190bf2b8e6847eab4c39fc9d33ffc0c4c3c55011b0fbc7f7807d32fb
 M|tests/test_gmsh_meshing.py|5fc5d1929bc02133a4e3b6cb4e207a32c6b93d764295eea593d6fb717a32f856
 M|tests/test_gmsh_profiles.py|62ee8b514b104bfb134fea3aec73993f7ce9a66b657b10812fd26b0c0209de68
 M|tests/test_project_layout.py|595ec358f9ff09b389b18233ea1607bafa157960ccee2beed0aea3c3caa1f35f
??|src/fem/geometry/_gmsh/__init__.py|98be95eb951b16057e75bb5bb0a56bccb5320c0de8c140cc32c43adb96f34d9b
??|src/fem/geometry/_gmsh/backend.py|67389d25d96a5d4977003bff1a7abed91cfe3b1914e0ec626429001ab8934b51
??|src/fem/geometry/_gmsh/model.py|04f23e435a1a57d46b1997e62a86fba1bc101a931324e00f4a936153523d896d
~~~

The user-owned deletion of
`docs/2026-07-20-gmsh-structured-feature-results-and-foundational-operations-implementation-handoff.md`
was deliberately excluded from that phase manifest and remains untouched.

The plan-recorded Phase 1 baseline was Python 3.13.11, Gmsh 4.15.2, 12
architecture tests, 892 focused Gmsh tests, and 1,499 complete-suite tests. A
pre-edit local recheck confirmed the 12 architecture tests passed in 1.56 s.

## Implementation

### Public stateless modules

- `fem.geometry.errors` defines the four public geometry errors once and has an
  exact four-name `__all__`.
- `fem.geometry.types` defines the three public literal aliases and seven
  frozen, slotted value objects once. It also owns the private
  `_unique_first_seen()` invariant helper.
- `fem.geometry._validation` contains the eight selected scalar validators and
  imports only the standard library.
- `fem.geometry.__init__` imports public errors and types from their canonical
  modules and imports only `GeometryModel` and `model` from `_gmsh.model`.

### Private Gmsh stateless modules

- `_gmsh.constants` owns the three unchanged OCC/topology constants.
- `_gmsh.predicates` owns the five private aliases and 21 mechanically moved
  pure geometric functions.
- `_gmsh.model` imports all canonical definitions, retains stateful OCC and
  mesh-runtime behavior, and now contains 5,373 physical lines instead of
  6,276. Its `__all__` is narrowed to `GeometryModel` and `model`.

`src/fem/mesh/gmsh.py` required no Phase 2 change. Its SHA-256 hash remains the
Phase 1 boundary value
`675620681c2ed98f0848ee880c012bc0312f3fb504d17585fd927f01d753ceeb`.

## Test Additions and Count Reconciliation

The new characterization coverage is:

- `tests/test_geometry_types.py`: 43 tests;
- `tests/test_geometry_validation.py`: 69 tests;
- `tests/test_geometry_predicates.py`: 23 tests;
- `tests/test_project_layout.py`: 3 additional architecture tests, increasing
  that module from 12 to 15 tests;
- `tests/test_gmsh_profiles.py`: 3 additional public elliptical-arc caller
  validation-order tests.

That is 141 added tests. The complete count therefore reconciles from 1,499 to
1,640 with no removed behavior tests.

Coverage includes canonical identity and `__module__`, exact exports and
inheritance, frozen/slotted value-object behavior, result invariants, fresh
process import permutations, lazy external-Gmsh loading, scalar conversion and
validation order, translated/rotated/rigid signature matching, elliptical-arc
degeneracies, plane projection, winding ambiguity, and 2D segment contact.

Architecture tests enforce each stateless module's import allowlist and verify
that `_gmsh.model` no longer defines any moved class, alias, constant,
validator, helper, or predicate.

## Verification Results

Environment:

~~~text
Python 3.13.11
Gmsh 4.15.2
pytest 9.1.1
Ruff 0.14.10
~~~

The project virtual environment does not contain Ruff, so the globally
available Ruff executable was used as required by the repository instructions.

Static and structural verification:

- `ruff check src tests examples`: passed;
- `python -m compileall -q src tests examples`: passed with an isolated
  `PYTHONPYCACHEPREFIX` because old sandbox-created `__pycache__` directories
  have unusable Windows ACLs;
- legacy `fem.geometry.gmsh` search: no matches;
- legacy `fem.meshing` search: no matches;
- duplicate public-class search in `_gmsh.model`: no matches;
- `git diff --check`: passed, with only existing LF-to-CRLF checkout warnings.

Test verification:

~~~text
Stateless contract and architecture: 150 passed in 5.41s
Profiles/foundational/advanced:     142 passed in 1.87s
Gmsh geometry:                     604 passed in 2.60s
Local refinement/meshing/IO:       149 passed in 1.16s
Focused Gmsh total:                895 passed
Complete repository suite:       1640 passed in 13.57s
~~~

The first sandboxed complete-suite run reproduced the known `tmp_path` ACL
failure. The authoritative complete run used a fresh absolute basetemp outside
the sandbox and had no failures or errors.

Required real-Gmsh examples all exited with status 0:

~~~text
gmsh_geometry_irregular_plate.py
Solved 607 irregular-profile Tri3 elements and wrote results\gmsh_geometry_irregular_plate\gmsh_geometry_irregular_plate.vtk

gmsh_geometry_revolved_solid.py
Solved 145 nodes and 379 Tet4 elements; wrote D:\gu1mu\Code\projects\FEM-Python\results\gmsh_geometry_revolved_solid\gmsh_geometry_revolved_solid.vtk

gmsh_geometry_swept_solid.py
Imported swept solid: 60 nodes, 127 Tet4 elements

gmsh_geometry_filleted_box.py
Imported filleted box: 139 nodes, 358 Tet4 elements
~~~

The irregular-plate and revolved-solid examples updated their ignored result
artifacts. The swept-solid and filleted-box examples created no persistent
result files.

## Working-Tree State and Next Phase

Phase 2 remains uncommitted and unstaged. The pre-existing user-owned document
deletion remains present and untouched. Phase 1 and Phase 2 can be reviewed
against the recorded starting manifest even though the explicit instruction for
this run prohibited committing the completed implementation.

The next phase may plan the stateful runtime boundary: session ownership,
model activation, entity/topology registries, and a narrow geometry-to-mesh
port. No Phase 3 work is included here.
