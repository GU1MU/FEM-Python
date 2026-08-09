from __future__ import annotations

import operator
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Any, Literal, overload

import numpy as np
from scipy.sparse import diags, triu

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary import step as _boundary_step
from ..boundary.loads import build_load_vector
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from ..core.validation import validate_analysis_step, validate_model_structure
from ._pardiso_spd import (
    _PardisoSPDError,
    _PardisoSPDMemoryError,
    factorize_spd,
)


__all__ = ["PreparedSystem", "prepare", "solve"]


StepSelector = str | int | AnalysisStep
_FACTOR_CACHE_MAX_ENTRIES = 1
_PARDISO_MEMORY_FAILURE = "PARDISO SPD solver failed: insufficient memory"


@dataclass(frozen=True)
class _ResolvedSelection:
    """Resolved static steps and the result mode chosen by the call shape."""

    steps: tuple[AnalysisStep | None, ...]
    plural: bool


@dataclass(frozen=True, slots=True)
class _ReducedFactorization:
    """One scaled free-DOF factorization for a constraint pattern."""

    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    inverse_scale: np.ndarray
    factor: Any | None
    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        """Release the owned native factor at most once."""

        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        if self.factor is not None:
            self.factor.close()


class _FactorizationCache:
    """Small thread-safe LRU of factors tied to one exact frozen base K."""

    __slots__ = (
        "_base_stiffness",
        "_entries",
        "_closed",
        "_lock",
        "_max_entries",
    )

    def __init__(
        self,
        base_stiffness: Any,
        *,
        max_entries: int = _FACTOR_CACHE_MAX_ENTRIES,
    ) -> None:
        self._base_stiffness = base_stiffness
        self._entries: OrderedDict[
            tuple[int, ...],
            _ReducedFactorization,
        ] = OrderedDict()
        self._closed = False
        self._lock = RLock()
        self._max_entries = max_entries

    def close(self) -> None:
        """Idempotently release every factor still owned by this cache."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._entries.values())
            self._entries.clear()
            for factorization in entries:
                factorization.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def shares_base_stiffness(self, base_stiffness: Any) -> bool:
        """Return whether the cache is tied to the exact immutable K."""

        return self._base_stiffness is base_stiffness

    def factor_for(
        self,
        constrained_pattern: tuple[int, ...],
    ) -> _ReducedFactorization:
        """Return or build one factor while maintaining LRU order."""

        with self._lock:
            return self._factor_for_locked(constrained_pattern)

    def solve(
        self,
        load: np.ndarray,
        constrained_pattern: tuple[int, ...],
        constrained_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one partitioned system under the cache's serial lock."""

        with self._lock:
            try:
                entry = self._factor_for_locked(constrained_pattern)
            except _PardisoSPDMemoryError as exc:
                raise RuntimeError(_PARDISO_MEMORY_FAILURE) from exc
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "sparse linear solve failed: stiffness matrix "
                    "is singular or under-constrained."
                ) from exc
            return _solve_reduced_system(
                self._base_stiffness,
                load,
                entry,
                constrained_values,
            )

    def _factor_for_locked(
        self,
        constrained_pattern: tuple[int, ...],
    ) -> _ReducedFactorization:
        if self._closed:
            raise RuntimeError("factorization cache is closed")
        cached = self._entries.get(constrained_pattern)
        if cached is not None:
            self._entries.move_to_end(constrained_pattern)
            return cached

        while len(self._entries) >= self._max_entries:
            _, evicted = self._entries.popitem(last=False)
            evicted.close()
        factorization = _build_reduced_factorization(
            self._base_stiffness,
            constrained_pattern,
        )
        self._entries[constrained_pattern] = factorization
        self._entries.move_to_end(constrained_pattern)
        return factorization


@dataclass(frozen=True, slots=True, init=False)
class PreparedSystem:
    """Owned linear-static model and immutable assembled base stiffness."""

    _model: Any
    _base_stiffness: Any
    _factor_cache: _FactorizationCache

    @classmethod
    def _from_owned(
        cls,
        model: Any,
        base_stiffness: Any,
        factor_cache: _FactorizationCache | None = None,
    ) -> PreparedSystem:
        frozen_stiffness = _freeze_base_stiffness(base_stiffness)
        if factor_cache is None:
            factor_cache = _FactorizationCache(frozen_stiffness)
        elif not factor_cache.shares_base_stiffness(frozen_stiffness):
            raise ValueError(
                "factor cache must share the exact base stiffness"
            )

        prepared = object.__new__(cls)
        object.__setattr__(prepared, "_model", model)
        object.__setattr__(prepared, "_base_stiffness", frozen_stiffness)
        object.__setattr__(prepared, "_factor_cache", factor_cache)
        return prepared

    def clone(self) -> PreparedSystem:
        """Clone the prepared model while sharing immutable base stiffness."""

        return PreparedSystem._from_owned(
            deepcopy(self._model),
            self._base_stiffness,
            self._factor_cache,
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
            and self._factor_cache is other._factor_cache
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
        constrained_pattern, _ = _validated_prescribed_displacements(
            boundary,
            self._base_stiffness.shape[0],
        )
        try:
            self._factor_cache.factor_for(constrained_pattern)
        except _PardisoSPDMemoryError as error:
            raise RuntimeError(_PARDISO_MEMORY_FAILURE) from error
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
                self._factor_cache,
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
    factor_cache: _FactorizationCache,
    name: str | None,
    timings: dict[str, float] | None,
) -> ModelResult:
    """Solve one load case against an already prepared base stiffness."""
    started = perf_counter()
    load = build_load_vector(model.mesh, boundary)
    constrained_pattern, constrained_values = (
        _validated_prescribed_displacements(
            boundary,
            base_stiffness.shape[0],
        )
    )
    load = _validated_load_vector(
        load,
        base_stiffness.shape[0],
    )
    _record_timing(timings, "载荷与边界条件", started)

    started = perf_counter()
    displacement, free_dofs = factor_cache.solve(
        load,
        constrained_pattern,
        constrained_values,
    )
    _record_timing(timings, "线性方程求解", started)

    started = perf_counter()
    reactions = base_stiffness @ displacement - load
    _validate_free_dof_equilibrium(reactions, load, free_dofs)
    result = ModelResult(
        model,
        selected_step,
        displacement,
        reactions,
        name=name,
    )
    _record_timing(timings, "反力与结果封装", started)
    return result


def _validated_prescribed_displacements(
    boundary: Any,
    num_dofs: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return a sorted factor key and aligned finite prescribed values."""

    normalized: dict[int, float] = {}
    for raw_dof, raw_value in boundary.prescribed_displacements.items():
        if isinstance(raw_dof, bool):
            raise TypeError(
                "prescribed displacement DOF index must be an integer, "
                f"got {raw_dof!r}"
            )
        try:
            dof = int(operator.index(raw_dof))
        except TypeError as exc:
            raise TypeError(
                "prescribed displacement DOF index must be an integer, "
                f"got {raw_dof!r}"
            ) from exc
        if dof < 0 or dof >= num_dofs:
            raise IndexError(
                f"DOF index {dof} out of bounds [0, {num_dofs})"
            )
        if dof in normalized:
            raise ValueError(
                f"prescribed displacement repeats DOF index {dof}"
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"prescribed displacement at DOF {dof} must be numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"prescribed displacement at DOF {dof} "
                f"must be finite, got {value!r}"
            )
        normalized[dof] = value

    constrained_pattern = tuple(sorted(normalized))
    constrained_values = np.fromiter(
        (normalized[dof] for dof in constrained_pattern),
        dtype=float,
        count=len(constrained_pattern),
    )
    return constrained_pattern, constrained_values


def _validated_load_vector(
    load: Any,
    num_dofs: int,
) -> np.ndarray:
    """Normalize the static load without changing public linear.solve."""

    values = np.asarray(load, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError(
            "load must be one-dimensional or a column vector, "
            f"got shape {values.shape}"
        )
    if values.shape[0] != num_dofs:
        raise ValueError(
            f"load must have length {num_dofs}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("load must contain only finite values")
    return values


def _build_reduced_factorization(
    base_stiffness: Any,
    constrained_pattern: tuple[int, ...],
) -> _ReducedFactorization:
    """Build one diagonally scaled SPD factor for the free-DOF submatrix."""

    num_dofs = base_stiffness.shape[0]
    constrained_dofs = np.fromiter(
        constrained_pattern,
        dtype=np.int64,
        count=len(constrained_pattern),
    )
    free_mask = np.ones(num_dofs, dtype=bool)
    free_mask[constrained_dofs] = False
    free_dofs = np.flatnonzero(free_mask).astype(np.int64, copy=False)
    if free_dofs.size == 0:
        inverse_scale = np.empty(0, dtype=float)
        factor = None
    else:
        free_stiffness = base_stiffness[free_dofs][:, free_dofs]
        diagonal = np.abs(
            np.asarray(free_stiffness.diagonal(), dtype=float)
        )
        if (
            diagonal.size != free_stiffness.shape[0]
            or not np.all(np.isfinite(diagonal))
            or np.any(diagonal <= 0.0)
        ):
            raise ValueError(
                "stiffness matrix has a zero or invalid diagonal"
            )
        inverse_scale = 1.0 / np.sqrt(diagonal)
        if not np.all(np.isfinite(inverse_scale)):
            raise ValueError(
                "stiffness matrix has an invalid diagonal scaling"
            )
        scaling = diags(inverse_scale)
        scaled = scaling @ free_stiffness @ scaling
        scaled_upper = triu(scaled, format="csr")
        scaled_upper.sum_duplicates()
        scaled_upper.sort_indices()
        try:
            factor = factorize_spd(scaled_upper)
        except _PardisoSPDMemoryError:
            raise
        except _PardisoSPDError as exc:
            raise ValueError(
                "stiffness matrix is singular or under-constrained"
            ) from exc

    for values in (constrained_dofs, free_dofs, inverse_scale):
        values.flags.writeable = False
    return _ReducedFactorization(
        constrained_dofs=constrained_dofs,
        free_dofs=free_dofs,
        inverse_scale=inverse_scale,
        factor=factor,
    )
def _solve_reduced_system(
    base_stiffness: Any,
    load: np.ndarray,
    factorization: _ReducedFactorization,
    constrained_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the free system and reconstruct the prescribed global U."""

    if constrained_values.shape != factorization.constrained_dofs.shape:
        raise ValueError(
            "prescribed displacement values do not match constraint pattern"
        )
    displacement = np.zeros(base_stiffness.shape[0], dtype=float)
    displacement[factorization.constrained_dofs] = constrained_values
    if factorization.free_dofs.size == 0:
        return displacement, factorization.free_dofs

    free_load = load[factorization.free_dofs].copy()
    if factorization.constrained_dofs.size:
        coupling = base_stiffness[factorization.free_dofs][
            :,
            factorization.constrained_dofs,
        ]
        free_load -= coupling @ constrained_values

    scaled_load = factorization.inverse_scale * free_load
    try:
        reduced_scaled = np.asarray(
            factorization.factor.solve(scaled_load),
            dtype=float,
        )
    except _PardisoSPDMemoryError as exc:
        raise RuntimeError(_PARDISO_MEMORY_FAILURE) from exc
    except _PardisoSPDError as exc:
        raise RuntimeError(
            "sparse linear solve failed: "
            "stiffness matrix is singular or under-constrained."
        ) from exc
    free_displacement = (
        factorization.inverse_scale * reduced_scaled
    )
    displacement[factorization.free_dofs] = free_displacement
    if (
        displacement.ndim != 1
        or displacement.shape[0] != load.shape[0]
        or not np.all(np.isfinite(displacement))
    ):
        raise RuntimeError(
            "sparse linear solve returned invalid or non-finite values"
        )
    return displacement, factorization.free_dofs


def _validate_free_dof_equilibrium(
    reactions: np.ndarray,
    load: np.ndarray,
    free_dofs: np.ndarray,
) -> None:
    """Require reconstructed free DOFs to have negligible reactions."""

    if free_dofs.size == 0:
        return
    free_reactions = reactions[free_dofs]
    residual_norm = float(
        np.linalg.norm(free_reactions, ord=np.inf)
    )
    free_load = load[free_dofs]
    internal = free_reactions + free_load
    scale = max(
        float(np.linalg.norm(internal, ord=np.inf)),
        float(np.linalg.norm(free_load, ord=np.inf)),
        1.0,
    )
    if residual_norm > 1e-8 * scale:
        raise RuntimeError(
            f"sparse linear solve residual {residual_norm:g} "
            "exceeds tolerance"
        )


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
