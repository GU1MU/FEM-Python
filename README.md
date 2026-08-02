# FEM Python

[Chinese](README_Zh.md)

FEM Python is a Python project for script-based finite element modeling and linear static analysis

It covers geometry creation, mesh generation or import, model definition, solving, result queries, and CSV/VTK export

The project provides both a Python API and a desktop GUI with an integrated FEM Agent, and models can be created natively or imported from supported Abaqus `.inp` files

## Installation

Python 3.13 or later and `uv` are required, and the following commands run from the repository root in PowerShell

Create the virtual environment:

```powershell
uv venv --python 3.13
```

Install the desktop GUI, FEM Agent, and meshing support:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[cad,gui,agent]"
```

If you only use the Python API and examples, install only the meshing
dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[cad]"
```

## Desktop GUI

### Launch

Activate the virtual environment and start the GUI:

```powershell
.\.venv\Scripts\Activate.ps1
fem-gui
```

### Features

The desktop GUI brings project management, preprocessing, analysis, and postprocessing into one session

- Create, open, and save native projects, or import Abaqus `.inp` models
- Create and edit geometry, scopes, materials, sections, and analysis definitions
- Configure and generate meshes, assess quality, validate models, and run background analyses
- Display, query, and export results

## FEM Agent

### Configuration

Create `fem-agent.config.json` in the repository root:

```json
{
  "enabled": true,
  "api_key": "your-api-key"
}
```

After starting the GUI, click the `FA` button in the upper-right corner of the
model viewport to open FEM Agent

### Features

FEM Agent uses the current GUI session and user-provided engineering parameters to assist supported native modeling and analysis workflows

- Read the current model state and reference workspace files with `@`
- Create or modify geometry and meshes
- Define and assign materials, analysis steps, and boundary conditions
- Submit solves and query analysis results

## Examples for Python API

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
