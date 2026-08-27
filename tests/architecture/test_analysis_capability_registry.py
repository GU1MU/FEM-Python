from __future__ import annotations

from pathlib import Path

from fem.analysis import compile_analysis, resolve_analysis_request
from fem.analysis.compilation.capabilities import (
    NonlinearStaticCapability,
    NonlinearStaticCapabilityRegistry,
    nonlinear_static_capability_for,
)
from tests.architecture.test_arch_v2_core import _model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_nonlinear_capability_registry_resolves_aliases_without_compiler_branches():
    capability = nonlinear_static_capability_for("CPS4")

    assert capability.canonical_type == "Quad4"
    assert capability is nonlinear_static_capability_for("Quad4")


def test_compiler_consumes_an_injected_capability_registry():
    base = nonlinear_static_capability_for("Quad4")
    operator_calls: list[str] = []

    def operator_factory(definition, kinematics, options):
        operator_calls.append(definition.canonical_type)
        return base.operator_factory(definition, kinematics, options)

    capability = NonlinearStaticCapability(
        canonical_type=base.canonical_type,
        definition_factory=base.definition_factory,
        kinematics_factory=base.kinematics_factory,
        operator_factory=operator_factory,
        material_factory=base.material_factory,
    )
    registry = NonlinearStaticCapabilityRegistry((capability,))
    model = _model()

    compiled = compile_analysis(
        model,
        resolve_analysis_request(model.steps[0]),
        capability_registry=registry,
    )

    assert compiled.problem.assembly.bindings
    assert operator_calls == ["Quad4"]


def test_analysis_compiler_does_not_own_concrete_quad4_material_or_kinematics_names():
    source = (
        PROJECT_ROOT / "src" / "fem" / "analysis" / "compiler.py"
    ).read_text(encoding="utf-8")

    assert "Quad4Definition" not in source
    assert "TotalLagrangianKinematics" not in source
    assert "J2PlasticityMaterial" not in source
    assert "nonlinear_static_capability_for" not in source
