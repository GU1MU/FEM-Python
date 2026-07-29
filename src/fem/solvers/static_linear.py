from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, overload

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary import step as _boundary_step
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from ..core.validation import validate_analysis_step, validate_model_structure
from . import linear


__all__ = ["PreparedSystem", "prepare", "solve"]


StepSelector = str | int | AnalysisStep


@dataclass(frozen=True)
class _ResolvedSelection:
    """Resolved static steps and the result mode chosen by the call shape."""

    steps: tuple[AnalysisStep | None, ...]
    plural: bool


@dataclass(frozen=True, slots=True, init=False)
class PreparedSystem:
    """Owned linear-static model and immutable assembled base stiffness."""

    _model: Any
    _base_stiffness: Any

    @classmethod
    def _from_owned(
        cls,
        model: Any,
        base_stiffness: Any,
    ) -> PreparedSystem:
        prepared = object.__new__(cls)
        object.__setattr__(prepared, "_model", model)
        object.__setattr__(
            prepared,
            "_base_stiffness",
            _freeze_base_stiffness(base_stiffness),
        )
        return prepared

    def clone(self) -> PreparedSystem:
        """Clone the prepared model while sharing immutable base stiffness."""

        return PreparedSystem._from_owned(
            deepcopy(self._model),
            self._base_stiffness,
        )

    def _trusted_model_for_task(self) -> Any:
        """Return this instance's model only to an owning task boundary."""

        return self._model

    def _shares_base_stiffness_with(
        self,
        other: PreparedSystem,
    ) -> bool:
        """Return whether two trusted instances share the exact frozen K."""

        return (
            type(other) is PreparedSystem
            and self._base_stiffness is other._base_stiffness
        )

    def validate_step(
        self,
        step: StepSelector | None = None,
    ) -> AnalysisStep | None:
        """Validate and return a detached selected Step."""

        selected_step = validate_problem(self._model, step)
        return (
            None
            if selected_step is None
            else deepcopy(selected_step)
        )

    def validate_stiffness(
        self,
        step: StepSelector | None = None,
    ) -> AnalysisStep | None:
        """Factor one constrained Step without rebuilding base stiffness."""

        selected_step = self.validate_step(step)
        self._validate_selected_stiffness(selected_step)
        return selected_step

    def _validate_selected_stiffness(
        self,
        selected_step: AnalysisStep | None,
    ) -> None:
        """Factor one already-validated Step against immutable base K."""

        boundary = _boundary_step.boundary_for_step(
            self._model,
            selected_step,
        )
        zero_load = np.zeros(self._model.mesh.num_dofs, dtype=float)
        constrained_stiffness, _ = apply_dirichlet(
            self._base_stiffness,
            zero_load,
            boundary,
        )
        try:
            _validate_nonsingular_stiffness(constrained_stiffness)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "模型约束不足或刚度矩阵奇异；"
                "请检查刚体位移、材料、截面和单元连接"
            ) from error

    def solve(
        self,
        step: StepSelector | None = None,
        name: str | None = None,
        *,
        steps: Literal["all"] | Iterable[StepSelector] | None = None,
        _validated_step: AnalysisStep | None | object = ...,
        timings: dict[str, float] | None = None,
    ) -> ModelResult | ModelResults:
        """Solve on a disposable model clone without rebuilding base K."""

        return self.clone()._solve_owned(
            step,
            name,
            steps=steps,
            _validated_step=_validated_step,
            timings=timings,
        )

    def _solve_owned(
        self,
        step: StepSelector | None = None,
        name: str | None = None,
        *,
        steps: Literal["all"] | Iterable[StepSelector] | None = None,
        _validated_step: AnalysisStep | None | object = ...,
        timings: dict[str, float] | None = None,
    ) -> ModelResult | ModelResults:
        """Solve against this trusted instance's exact owned model."""

        if _validated_step is not ...:
            if steps is not None:
                raise ValueError(
                    "_validated_step is only valid for a scalar solve"
                )
            selection = _ResolvedSelection((_validated_step,), False)
        else:
            started = perf_counter()
            if steps is None:
                selection = _ResolvedSelection(
                    (validate_problem(self._model, step),),
                    False,
                )
            else:
                selection = _resolve_selection(self._model, step, steps)
                _validate_selection(self._model, selection.steps)
            _record_timing(timings, "模型验证", started)
        return self._solve_selection(selection, name, timings)

    def _solve_selection(
        self,
        selection: _ResolvedSelection,
        name: str | None,
        timings: dict[str, float] | None,
    ) -> ModelResult | ModelResults:
        results = tuple(
            _solve_prepared_step(
                self._model,
                selected_step,
                _boundary_step.boundary_for_step(
                    self._model,
                    selected_step,
                ),
                self._base_stiffness,
                _result_name(
                    self._model,
                    selected_step,
                    name,
                    selection.plural,
                ),
                timings,
            )
            for selected_step in selection.steps
        )
        if selection.plural:
            return ModelResults(self._model, results)
        return results[0]


def prepare(
    model: Any,
    *,
    copy_model: bool = True,
    timings: dict[str, float] | None = None,
) -> PreparedSystem:
    """Apply sections and assemble one reusable linear-static base system."""

    if type(copy_model) is not bool:
        raise TypeError("copy_model must be bool")
    owned_model = deepcopy(model) if copy_model else model

    started = perf_counter()
    materials.apply_sections(owned_model)
    _record_timing(timings, "分析准备", started)

    started = perf_counter()
    base_stiffness = assemble_global_stiffness_sparse(owned_model.mesh)
    _record_timing(timings, "刚度矩阵装配", started)
    return PreparedSystem._from_owned(owned_model, base_stiffness)


@overload
def solve(
    model: Any,
    step: StepSelector | None = None,
    name: str | None = None,
    *,
    steps: None = None,
    _validated_step: AnalysisStep | None | object = ...,
    _prepared_system: PreparedSystem | None = None,
    timings: dict[str, float] | None = None,
) -> ModelResult: ...


@overload
def solve(
    model: Any,
    step: None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector],
    _validated_step: object = ...,
    _prepared_system: PreparedSystem | None = None,
    timings: dict[str, float] | None = None,
) -> ModelResults: ...


def solve(
    model: Any,
    step: StepSelector | None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector] | None = None,
    _validated_step: AnalysisStep | None | object = ...,
    _prepared_system: PreparedSystem | None = None,
    timings: dict[str, float] | None = None,
) -> ModelResult | ModelResults:
    """Solve one or more independent linear static model steps."""
    if _prepared_system is not None:
        if type(_prepared_system) is not PreparedSystem:
            raise TypeError(
                "_prepared_system must be exactly PreparedSystem or None"
            )
        if _prepared_system._trusted_model_for_task() is not model:
            raise ValueError(
                "_prepared_system must own the exact solve model"
            )
    if _validated_step is not ...:
        if steps is not None:
            raise ValueError("_validated_step is only valid for a scalar solve")
        selection = _ResolvedSelection((_validated_step,), False)
    else:
        started = perf_counter()
        if steps is None:
            selection = _ResolvedSelection((validate_problem(model, step),), False)
        else:
            selection = _resolve_selection(model, step, steps)
            _validate_selection(model, selection.steps)
        _record_timing(timings, "模型验证", started)

    prepared = (
        prepare(
            model,
            copy_model=False,
            timings=timings,
        )
        if _prepared_system is None
        else _prepared_system
    )
    return prepared._solve_selection(selection, name, timings)


def _resolve_selection(
    model: Any,
    step: StepSelector | None,
    steps: Literal["all"] | Iterable[StepSelector] | None,
) -> _ResolvedSelection:
    """Resolve scalar or plural selectors before model preparation begins."""
    if steps is None:
        return _ResolvedSelection((_resolve_step(model, step),), False)
    if step is not None:
        raise ValueError("step and steps are mutually exclusive")

    selected_steps = _resolve_plural_steps(model, steps)
    _reject_duplicate_steps(selected_steps)
    return _ResolvedSelection(selected_steps, True)


def _resolve_plural_steps(
    model: Any,
    steps: Literal["all"] | Iterable[StepSelector],
) -> tuple[AnalysisStep | None, ...]:
    """Resolve a plural step selection while preserving its order."""
    if isinstance(steps, str):
        if steps != "all":
            raise TypeError(
                "steps must be the exact string 'all' or an iterable of selectors"
            )
        runnable = tuple(
            candidate
            for candidate in model.steps
            if candidate.name.lower() != "initial"
        )
        if runnable:
            return runnable
        if model.steps:
            return (model.steps[0],)
        return (None,)
    if isinstance(steps, (bytes, bytearray)):
        raise TypeError(
            "steps must be the exact string 'all' or an iterable of selectors"
        )

    try:
        selectors = tuple(steps)
    except TypeError as exc:
        raise TypeError(
            "steps must be the exact string 'all' or an iterable of selectors"
        ) from exc
    if not selectors:
        raise ValueError("steps must contain at least one selector")

    resolved: list[AnalysisStep | None] = []
    for selector in selectors:
        if selector is None:
            raise TypeError("plural step selectors cannot be None")
        resolved.append(_resolve_step(model, selector))
    return tuple(resolved)


def _resolve_step(
    model: Any,
    selector: StepSelector | None,
) -> AnalysisStep | None:
    """Resolve one valid scalar selector through the canonical step resolver."""
    if selector is not None and (
        isinstance(selector, bool)
        or not isinstance(selector, (str, int, AnalysisStep))
    ):
        raise TypeError("step selector must be a step name, index, or AnalysisStep")
    return _boundary_step.get_step(model, selector)


def _reject_duplicate_steps(steps: tuple[AnalysisStep | None, ...]) -> None:
    """Reject repeated steps and result-name collisions in one selection."""
    seen: set[int] = set()
    seen_names: dict[str, str] = {}
    for selected_step in steps:
        identity = id(selected_step)
        if identity in seen:
            step_name = selected_step.name if selected_step is not None else "step"
            raise ValueError(f"analysis step {step_name} was selected more than once")
        seen.add(identity)
        step_name = selected_step.name if selected_step is not None else "step"
        name_key = str(step_name).casefold()
        if name_key in seen_names:
            raise ValueError(
                "selected analysis step names must be unique ignoring case; "
                f"got {seen_names[name_key]!r} and {step_name!r}"
            )
        seen_names[name_key] = str(step_name)


def _validate_selection(
    model: Any,
    steps: tuple[AnalysisStep | None, ...],
) -> None:
    """Validate every selected load case before preparing the shared system."""
    validate_model_structure(model)
    for selected_step in steps:
        validate_analysis_step(model, selected_step)
        _validate_static_step(selected_step)


def _solve_prepared_step(
    model: Any,
    selected_step: AnalysisStep | None,
    boundary: Any,
    base_stiffness: Any,
    name: str | None,
    timings: dict[str, float] | None,
) -> ModelResult:
    """Solve one load case against an already prepared base stiffness."""
    started = perf_counter()
    load = build_load_vector(model.mesh, boundary)
    constrained_stiffness, constrained_load = apply_dirichlet(
        base_stiffness,
        load,
        boundary,
    )
    _record_timing(timings, "载荷与边界条件", started)

    started = perf_counter()
    displacement = linear.solve(constrained_stiffness, constrained_load)
    _record_timing(timings, "线性方程求解", started)

    started = perf_counter()
    reactions = base_stiffness @ displacement - load
    result = ModelResult(
        model,
        selected_step,
        displacement,
        reactions,
        name=name,
    )
    _record_timing(timings, "反力与结果封装", started)
    return result


def _record_timing(
    timings: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    """Record one optional solver stage without coupling the solver to the GUI."""
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + perf_counter() - started


def validate_problem(
    model: Any,
    step: StepSelector | None = None,
) -> AnalysisStep | None:
    """使用与线性静力求解完全一致的规则验证模型和分析步。"""
    selected_step = _resolve_step(model, step)
    validate_model_structure(model)
    validate_analysis_step(model, selected_step)
    _validate_static_step(selected_step)
    return selected_step


def validate_stiffness(
    model: Any,
    step: StepSelector | None = None,
) -> AnalysisStep | None:
    """Verify that assigned sections and constraints produce a solvable matrix."""
    if type(model) is PreparedSystem:
        return model.validate_stiffness(step)
    selected_step = validate_problem(model, step)
    prepared = prepare(
        model,
        copy_model=False,
    )
    prepared._validate_selected_stiffness(selected_step)
    return selected_step


def _freeze_base_stiffness(stiffness: Any) -> Any:
    """Return CSR storage whose arrays cannot be mutated across workers."""

    frozen = stiffness.tocsr(copy=False)
    for values in (frozen.data, frozen.indices, frozen.indptr):
        values.flags.writeable = False
    return frozen


def _validate_nonsingular_stiffness(stiffness: Any) -> None:
    """Check scaled sparse-LU pivots without depending on the load vector."""
    diagonal = np.abs(np.asarray(stiffness.diagonal(), dtype=float))
    if (
        diagonal.size != stiffness.shape[0]
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal <= 0.0)
    ):
        raise ValueError("stiffness matrix has a zero or invalid diagonal")
    inverse_scale = 1.0 / np.sqrt(diagonal)
    scaling = diags(inverse_scale)
    scaled = (scaling @ stiffness @ scaling).tocsc()
    factor = splu(scaled)
    pivots = np.abs(np.asarray(factor.U.diagonal(), dtype=float))
    tolerance = (
        np.finfo(float).eps
        * max(stiffness.shape[0], 1)
        * max(float(np.max(pivots)), 1.0)
    )
    if (
        not np.all(np.isfinite(pivots))
        or float(np.min(pivots)) <= tolerance
    ):
        raise ValueError("stiffness matrix is numerically rank deficient")


def _result_name(
    model: Any,
    step: AnalysisStep | None,
    name: str | None,
    plural: bool,
) -> str | None:
    """Return a scalar name unchanged or suffix a plural result name."""
    if not plural:
        return name
    base = name if name is not None else (model.name or "result")
    step_name = step.name if step is not None else "step"
    return f"{base}_{step_name}"


def _validate_static_step(step: AnalysisStep | None) -> None:
    """Reject procedures and geometric nonlinearity unsupported by this solver."""
    if step is None:
        return
    if str(step.procedure).strip().lower() != "static":
        raise ValueError(
            f"static_linear solver requires procedure 'static', got {step.procedure!r}"
        )
    nlgeom = next(
        (
            value
            for key, value in step.metadata.items()
            if str(key).strip().lower() == "nlgeom"
        ),
        None,
    )
    if _truthy_option(nlgeom):
        raise ValueError("static_linear solver does not support nlgeom")


def _truthy_option(value: Any) -> bool:
    """Return semantic truth for common bool-like analysis options."""
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)
