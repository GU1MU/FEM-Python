# FEM Python

FEM Python is a Python project for script-based finite element modeling and linear static analysis
It covers geometry creation, mesh generation or import, model definition, solving, result queries, and CSV/VTK export

Two primary workflows are supported:

- Create geometry and meshes in Python with OCC/Gmsh
- Build an `FEMModel` from supported content in Abaqus `.inp` files

## Installation

Python 3.13 or later is required, and the following commands should be run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[cad]"
```

The optional `cad` dependencies provide OCC/Gmsh modeling and meshing support

## Examples

The repository contains three representative examples:

| Example | Description |
| --- | --- |
| [`frame.py`](examples/frame.py) | Beam2 frame, OCC geometry, automatic Gmsh mesh, fixed constraints, and multiple static load cases |
| [`perforated_plate.py`](examples/perforated_plate.py) | First-order Tri3 plane-stress plate with a circular hole, curve selection, local mesh refinement, and edge traction |
| [`cantilever_beam.py`](examples/cantilever_beam.py) | Abaqus `.inp` import of a quadratic Hex20 cantilever with surface traction and gravity |

Run the examples:

```powershell
.\.venv\Scripts\python.exe examples\frame.py
.\.venv\Scripts\python.exe examples\perforated_plate.py
.\.venv\Scripts\python.exe examples\cantilever_beam.py
```

Results are written to `results/` by default

## Desktop GUI

Install the GUI dependencies and start the Chinese desktop application:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[cad,gui]"
.\.venv\Scripts\python.exe -m fem_gui.app
```

The GUI supports native sketch/feature modeling, Abaqus `.inp` import,
materials and sections, analysis definitions, mesh generation, background
linear-static jobs, result queries, and CSV/VTK export.

## Workflow

```text
OCC/Gmsh modeling or Abaqus .inp import
    → FEMModel
    → Sets, materials, and sections
    → AnalysisStep
    → static_linear.solve()
    → ModelResult / ModelResults
    → CSV / VTK
```

Solving one load case returns a `ModelResult`; solving multiple load cases returns an iterable `ModelResults`

## Capabilities

| Category | Supported features |
| --- | --- |
| Geometry and meshing | OCC geometry and Boolean operations, CAD entity selection, automatic Gmsh meshing and local refinement, Gmsh/CSV mesh readers |
| Model import | Build an `FEMModel` from supported Abaqus `.inp` keywords and analysis data |
| 1D elements | `Truss2`, `Beam2` |
| 2D elements | `Tri3`, `Tri6`, `Quad4`, `Quad8` with plane-stress and plane-strain formulations |
| 3D elements | `Tet4`, `Tet10`, `Hex8`, `Hex20` |
| Materials | Isotropic linear elasticity |
| Constraints and loads | Zero or nonzero prescribed displacements, nodal forces, Beam2 line loads, 2D edge traction/pressure, 3D surface traction/pressure, gravity |
| Solver | Sparse assembly, single or multiple independent linear static load cases |
| Post-processing | Nodal displacements, reactions, element or nodal stresses, CSV and VTK export |

Beam2 supports solid circular, hollow circular, and rectangular sections

## Scope

- Isotropic linear elasticity, small deformation, and linear static analysis
- The Abaqus adapter parses only the keywords and analysis data explicitly supported by the project
- The project does not enforce a unit system or perform unit conversion, so all input data must use consistent units

## Testing

Install the test dependencies and run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[cad,test]"
.\.venv\Scripts\python.exe -m pytest -q
```
