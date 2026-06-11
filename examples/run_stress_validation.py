from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from fem import abaqus, post, solvers


DATA_DIR = Path(__file__).resolve().parent / "stress_validation_inputs"
OUT_DIR = Path("results") / "stress_validation"

DEFAULT_CASES = [
    "01_hex8_single_element_C3D8",
    "02_tet4_single_element_C3D4",
    "03_tet10_single_element_C3D10",
    "04_hex8_two_element_average_C3D8",
    "05_material_boundary_two_hex8_C3D8",
    "06_section_boundary_two_hex8_C3D8",
    "07_threshold_split_two_hex8_C3D8",
    "08_tet4_regular_cube_C3D4",
    "09_tet10_regular_cube_C3D10",
    "10_mixed_element_type_hex8_tet4",
    "11_mixed_element_type_hex8_tet10",
]


def run_case(inp_path: Path, output_root: Path) -> None:
    case_name = inp_path.stem

    print(f"\n=== Running {case_name} ===")
    print(f"Input: {inp_path}")

    model = abaqus.read(inp_path)
    model.name = case_name

    step = solvers.static_linear.get_step(model)
    print("Step:", step.name if step is not None else None)
    print("Nodes:", len(model.mesh.nodes))
    print("Elements:", len(model.mesh.elements))
    print("Node sets:", sorted(model.node_sets))
    print("Element sets:", sorted(model.element_sets))
    print("Materials:", sorted(model.materials))

    result = solvers.static_linear.solve(model, step, name=case_name)

    case_out = output_root / case_name
    post.vtk.export.from_result(
        result,
        output_dir=case_out,
        name=case_name,
        overwrite=True,
    )

    print("Output:", case_out)
    print("  ", case_out / f"{case_name}_nodal_displacement.csv")
    print("  ", case_out / f"{case_name}_element_stress.csv")
    print("  ", case_out / f"{case_name}_nodal_stress.csv")
    print("  ", case_out / f"{case_name}.vtk")


def resolve_cases(case_args: list[str], data_dir: Path) -> list[Path]:
    selected = case_args or DEFAULT_CASES
    return [_resolve_case(case, data_dir) for case in selected]


def _resolve_case(case: str, data_dir: Path) -> Path:
    raw_path = Path(case)
    if raw_path.suffix.lower() == ".inp":
        inp_path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
        if inp_path.exists():
            return inp_path

    case_name = raw_path.stem if raw_path.suffix else case
    if case_name.isdigit():
        matches = sorted(data_dir.glob(f"{int(case_name):02d}_*.inp"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"no case starts with {int(case_name):02d}_ in {data_dir}")
        raise ValueError(f"case prefix {case_name!r} is ambiguous: {[path.name for path in matches]}")

    inp_path = data_dir / f"{case_name}.inp"
    if inp_path.exists():
        return inp_path

    raise FileNotFoundError(f"case input not found: {case}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stress-validation Abaqus inp examples and export CSV/VTK results.",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="Case names, numeric prefixes such as 02, or .inp paths. Defaults to cases 01-11.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing inp files. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Root output directory. Default: {OUT_DIR}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List default cases and exit.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed case instead of continuing.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full traceback for failed cases.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir

    if args.list:
        for case_name in DEFAULT_CASES:
            print(case_name)
        return 0

    failures: list[tuple[Path, Exception]] = []
    inp_paths = resolve_cases(args.cases, data_dir)
    for inp_path in inp_paths:
        try:
            run_case(inp_path, output_dir)
        except Exception as exc:
            failures.append((inp_path, exc))
            print(f"\nFAILED: {inp_path.stem}")
            print(f"  {type(exc).__name__}: {exc}")
            if args.traceback:
                traceback.print_exc()
            if args.stop_on_error:
                break

    print("\n=== Summary ===")
    print(f"Passed: {len(inp_paths) - len(failures)}")
    print(f"Failed: {len(failures)}")
    for inp_path, exc in failures:
        print(f"  {inp_path.stem}: {type(exc).__name__}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
