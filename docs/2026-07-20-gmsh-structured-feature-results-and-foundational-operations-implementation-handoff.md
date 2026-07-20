# Gmsh Structured Feature Results and Foundational Operations Handoff

## Outcome

Plan 1 is implemented against Gmsh 4.15.2. The geometry layer now provides a
validated `FeatureResult`, structured extrusion topology, typed copy and
in-place transforms, cross-dimensional intersection, and mixed-dimensional
fragment/imprint behavior. The dependency direction remains:

~~~text
fem.geometry.gmsh
    -> fem.mesh.gmsh
    -> fem.io.gmsh
    -> explicit FEMModel definition
~~~

`FeatureResult` is exported only by `fem.geometry.gmsh`. The mesh layer refers
to it in the `Mesher.structured_extrude()` return annotation without
re-exporting it. The historical `fem.meshing` package remains absent.

## Intentional Breaking Change

`GeometryModel.extrude()` and `Mesher.structured_extrude()` now return
`FeatureResult` instead of a flat tuple. Callers must select explicit fields:

- `primary` for generated entities one dimension above the inputs;
- `ends` for terminal translated copies;
- `sides` for generated lateral topology;
- `outputs` when normalized native order and multiplicity matter;
- `of_dimension()` for an order-preserving dimensional view of `outputs`.

No iterable or tuple-compatibility shim was added. Stored references describe
the current model state and do not provide persistent topological naming after
later destructive operations.

## Gmsh 4.15.2 Characterization

Extrusion classification is based on input dimension, generated boundary
topology, translated bounding boxes, centers of mass, and measures. Numeric
tags and native position alone are not semantic inputs. Real-backend coverage
confirmed:

- point, curve, rectangular surface, and holed-surface extrusion;
- pure and structured extrusion with positive and negative vectors;
- one-call extrusion of multiple disjoint inputs;
- adjacent inputs whose native output contains a repeated shared side;
- short local features positioned at world coordinates of magnitude `1e9`;
- deterministic primary, end, and side classification for all cases.

Coordinate comparisons use translation-normalized deltas and scale modeling
tolerance with local entity extent. Absolute world coordinates contribute only
their floating-point ULP resolution. The classifier also requires a complete
generated primary boundary, topology-prefilters terminal candidates, and
enforces a one-to-one assignment between unique sides and source-boundary
entities. Incomplete or ambiguous topology raises `GeometryError`; a
post-mutation failure clears typed identity assumptions, and structured
extrusion enters its terminal mesh-failed state.

One mixed-dimensional native `occ.copy()` call returns only the highest
dimension observed in Gmsh 4.15.2. The facade therefore copies homogeneous
dimension batches and restores caller order. It verifies output count,
dimension, freshness, existence, and source liveness before returning fresh
references.

Gmsh 4.15.2 preserves representable geometry and mesh orientation under
negative dilation factors in the characterized 1D, 2D, and 3D cases. Negative
factors are supported; zero and non-finite factors are rejected before native
mutation. Two-dimensional mirror and scale operations must preserve the global
XY plane. Explicit transformed references remain live, affected boundary and
loop identities are invalidated, and unrelated identities remain live.

Intersection was verified for every cross-dimensional pair among dimensions
0 through 3 in both object/tool directions. Each side must be internally
homogeneous; object and tool dimensions may differ. Empty intersections are
valid and return empty `outputs` with a valid input map.

Native fragment outputs and input maps are not subset-related in either
direction. The facade retains native output order and multiplicity, then
appends unique map-only lower-dimensional entities in first-seen order. Real
meshes prove shared-node conformance for point into curve, crossing curves,
curve into surface, surface into volume, and multiple lower-dimensional tools.

The inherited boolean exception contract is unchanged: a native OCC call that
raises before returning is treated as atomic and leaves its input references
live, while malformed returned output or mapping data clears typed identities.
Exception atomicity should be recharacterized when the supported Gmsh version
changes.

## Vertical Slice

`examples/gmsh_geometry_partitioned_plate.py` copies and translates a typed
partition curve, fragments a plate with two lower-dimensional tools, generates
the native mesh through `Mesher`, imports it through `gmsh_io.read()`, builds
explicit FEM selections/material/step data, solves, and exports VTK. It does
not use raw OCC access, physical groups, or automatic FEM model construction.

The verified run solved 416 conformal partitioned Tri3 elements. The existing
irregular-plate example also remained valid and solved 607 Tri3 elements.

## Final Verification

The final working tree passed:

- Python compilation of `src`, `tests`, and `examples`;
- focused Plan 1 suite: 771 passed in 9.07 seconds;
- complete repository suite: 1374 passed in 11.06 seconds;
- dedicated real-Gmsh foundational suite: 34 passed in 0.77 seconds;
- complete geometry module: 546 passed in 4.21 seconds;
- both end-to-end examples described above;
- Ruff checks for `src`, `tests`, and `examples`;
- architecture audits for layer direction, exports, tuple-style extrusion
  consumers, and the absent `fem.meshing` name;
- `git diff --check`.

The test runs used explicitly writable absolute `--basetemp` directories to
isolate the host temporary-directory ACL issue. Compilation used a writable
`PYTHONPYCACHEPREFIX` for the same reason.

This implementation is the stable Plan 1 baseline required before work begins
on the advanced 3D features and edge treatments in Plan 2.
