"""Private native Gmsh session and model lease management."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from ..errors import GeometryStateError
from . import backend


_MODEL_INCARNATION_ATTRIBUTE = "fem-python.geometry-model-incarnation"
_SESSION_BASELINE_INCARNATION_ATTRIBUTE = (
    "fem-python.session-baseline-model-incarnation"
)


class _NativeModelBorrow:
    """Dormant, revocable authority for one session-owned native model."""

    __slots__ = (
        "_active",
        "_epoch",
        "_incarnation",
        "_model_name",
        "_session",
    )

    def __init__(
        self,
        session: _GmshModelSession,
        model_name: str,
        incarnation: str,
        epoch: object,
    ) -> None:
        self._session = session
        self._model_name = model_name
        self._incarnation = incarnation
        self._epoch = epoch
        self._active = False

    def borrow(self) -> Any:
        """Reactivate and return the exact live facade-owned native model."""
        session = self._session
        if (
            not self._active
            or session._borrow_epoch is not self._epoch
            or session._model_name != self._model_name
            or session._model_incarnation != self._incarnation
        ):
            raise session._state_error(
                "borrow generated mesh",
                "native model borrow authority is inactive or revoked",
            )
        gmsh = session.activate("borrow generated mesh")
        return gmsh.model


class _GmshModelSession:
    """Own one facade model and any process-global session it initializes.

    The native Gmsh API is process-global.  This object deliberately keeps
    resource ownership separate from the public model state so failed cleanup
    can be retried after the public facade has entered ``CLOSED``.
    """

    __slots__ = (
        "_borrow_epoch",
        "_created_model",
        "_facade",
        "_incarnation_verified",
        "_model_incarnation",
        "_model_identity_inspection_failed",
        "_model_name",
        "_owned_session_baseline_incarnations",
        "_owns_session",
        "_pending_options",
        "_prior_current",
        "_prior_current_captured",
    )

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model_incarnation = uuid4().hex
        self._borrow_epoch: object | None = object()
        self._facade: Any | None = None
        self._owns_session = False
        self._created_model = False
        self._incarnation_verified = False
        self._model_identity_inspection_failed = False
        self._owned_session_baseline_incarnations: (
            tuple[tuple[str, str], ...] | None
        ) = None
        self._prior_current: str | None = None
        self._prior_current_captured = False
        self._pending_options: dict[str, float] = {}

    @property
    def created_model(self) -> bool:
        """Return whether native facade-model cleanup is still owned."""
        return self._created_model

    @property
    def has_pending_options(self) -> bool:
        """Return whether numeric options still require restoration."""
        return bool(self._pending_options)

    def enter(self) -> None:
        """Load Gmsh lazily, acquire the session, and create the facade model."""
        gmsh = backend.load_gmsh()
        self._facade = gmsh

        if not bool(gmsh.isInitialized()):
            # Record ownership before initialize: Gmsh can raise after changing
            # its process-global initialized state.
            self._owns_session = True
            # Geometry models are also created by GUI background workers.
            # Let the application retain signal ownership because Python only
            # permits signal handler registration on the main thread.
            gmsh.initialize(interruptible=False)

        self._prior_current = str(gmsh.model.getCurrent())
        self._prior_current_captured = True
        existing_models = tuple(str(item) for item in gmsh.model.list())
        if self._owns_session:
            if len(set(existing_models)) != len(existing_models):
                raise self._state_error(
                    "context entry",
                    "owned Gmsh session baseline has ambiguous duplicate "
                    "model names",
                )
            baseline_incarnations: list[tuple[str, str]] = []
            for baseline_name in sorted(existing_models):
                if str(gmsh.model.getCurrent()) != baseline_name:
                    gmsh.model.setCurrent(baseline_name)
                baseline_incarnation = uuid4().hex
                gmsh.model.setAttribute(
                    _SESSION_BASELINE_INCARNATION_ATTRIBUTE,
                    [baseline_incarnation],
                )
                values = tuple(
                    str(item)
                    for item in gmsh.model.getAttribute(
                        _SESSION_BASELINE_INCARNATION_ATTRIBUTE
                    )
                )
                if values != (baseline_incarnation,):
                    raise self._state_error(
                        "context entry",
                        "owned Gmsh session baseline incarnation could not "
                        "be verified",
                    )
                baseline_incarnations.append(
                    (baseline_name, baseline_incarnation)
                )
            if self._prior_current in existing_models and (
                str(gmsh.model.getCurrent()) != self._prior_current
            ):
                gmsh.model.setCurrent(self._prior_current)
            self._owned_session_baseline_incarnations = tuple(
                baseline_incarnations
            )
        if not isinstance(self._model_name, str) or not self._model_name.strip():
            raise GeometryStateError(
                f"geometry model {self._model_name!r}: model name must be a "
                "nonempty string"
            )
        if self._model_name in existing_models:
            raise GeometryStateError(
                f"geometry model {self._model_name!r}: a Gmsh model with this "
                "name already exists"
            )

        gmsh.model.add(self._model_name)
        self._created_model = True
        try:
            gmsh.model.setAttribute(
                _MODEL_INCARNATION_ATTRIBUTE,
                [self._model_incarnation],
            )
            self._assert_current_model_incarnation(gmsh, "context entry")
            self._incarnation_verified = True
        except BaseException as error:
            try:
                if self._model_name in tuple(
                    str(item) for item in gmsh.model.list()
                ):
                    if str(gmsh.model.getCurrent()) != self._model_name:
                        gmsh.model.setCurrent(self._model_name)
                    if self._current_model_has_expected_incarnation(gmsh):
                        gmsh.model.remove()
                        self._created_model = False
                    else:
                        error.add_note(
                            f"geometry model {self._model_name!r}: entry "
                            "cleanup retained the just-created native model "
                            "because its incarnation could not be verified"
                        )
                else:
                    self._created_model = False
            except BaseException as cleanup_error:
                error.add_note(
                    f"geometry model {self._model_name!r}: entry cleanup also "
                    "failed while trying to remove the just-created native "
                    f"model: {cleanup_error}"
                )
            raise
        gmsh.model.setCurrent(self._model_name)

    def activate(self, operation: str) -> Any:
        """Reactivate the owned native model for one typed operation."""
        gmsh = self._facade
        if gmsh is None or not bool(gmsh.isInitialized()):
            raise self._state_error(operation, "Gmsh session is not active")
        model_names = tuple(str(item) for item in gmsh.model.list())
        if self._model_name not in model_names:
            raise self._state_error(
                operation,
                "facade-owned Gmsh model is missing",
            )
        if str(gmsh.model.getCurrent()) != self._model_name:
            gmsh.model.setCurrent(self._model_name)
        self._assert_current_model_incarnation(gmsh, operation)
        return gmsh

    def prepare_native_borrow(self) -> _NativeModelBorrow:
        """Allocate one dormant capability for this facade-owned model."""
        epoch = self._borrow_epoch
        if epoch is None:
            raise self._state_error(
                "prepare generated mesh borrow",
                "native model borrows have been revoked",
            )
        return _NativeModelBorrow(
            self,
            self._model_name,
            self._model_incarnation,
            epoch,
        )

    def validate_native_borrow(
        self,
        capability: _NativeModelBorrow,
        operation: str,
    ) -> None:
        """Validate one dormant exact-model capability before no-fail commit."""
        if (
            type(capability) is not _NativeModelBorrow
            or capability._session is not self
            or capability._active
            or capability._epoch is not self._borrow_epoch
            or capability._model_name != self._model_name
            or capability._incarnation != self._model_incarnation
        ):
            raise self._state_error(
                operation,
                "native model borrow authority is invalid or revoked",
            )
        self.activate(operation)

    def activate_native_borrow(self, capability: _NativeModelBorrow) -> None:
        """Activate a prepared capability through a no-fail assignment."""
        capability._active = True

    def revoke_borrows(self) -> None:
        """Revoke every issued capability without allocation or native work."""
        self._borrow_epoch = None

    def inspect_initialized(self) -> bool:
        """Inspect the loaded process-global session without guessing state."""
        if self._facade is None:
            return False
        return bool(self._facade.isInitialized())

    def remove_created_model(self) -> None:
        """Remove only the facade-created native model, retaining failures."""
        gmsh = self._facade
        if not self._created_model or gmsh is None:
            return
        model_names = tuple(str(item) for item in gmsh.model.list())
        if self._model_name in model_names:
            if str(gmsh.model.getCurrent()) != self._model_name:
                gmsh.model.setCurrent(self._model_name)
            try:
                incarnation_matches = (
                    self._current_model_has_expected_incarnation(gmsh)
                )
            except BaseException:
                self._model_identity_inspection_failed = True
                raise
            self._model_identity_inspection_failed = False
            if not incarnation_matches:
                if not self._incarnation_verified:
                    raise self._state_error(
                        "remove facade model",
                        "the native model incarnation was never verified",
                    )
                # The facade-created incarnation is gone.  A same-name model
                # belongs to its replacer and must never be removed here.
                self._created_model = False
                return
            gmsh.model.remove()
        self._model_identity_inspection_failed = False
        self._created_model = False

    def restore_prior_model(self) -> None:
        """Restore the captured prior current model when it still exists."""
        gmsh = self._facade
        if gmsh is None or not self._prior_current_captured:
            return
        model_names = tuple(str(item) for item in gmsh.model.list())
        if self._prior_current in model_names:
            gmsh.model.setCurrent(self._prior_current)

    def finalize_owned_session(self, *, initialized: bool) -> None:
        """Finalize only a session initialized by this facade lifecycle."""
        if not initialized:
            self._owns_session = False
            self._created_model = False
            self._incarnation_verified = False
            self._model_identity_inspection_failed = False
            self._owned_session_baseline_incarnations = None
            return
        if not self._owns_session or self._facade is None:
            return
        if self._model_identity_inspection_failed:
            return
        baseline_incarnations = self._owned_session_baseline_incarnations
        if baseline_incarnations is not None:
            baseline_models = tuple(
                sorted(name for name, _incarnation in baseline_incarnations)
            )
            model_names = tuple(
                sorted(str(item) for item in self._facade.model.list())
            )
            expected_with_owned_model = tuple(
                sorted((*baseline_models, self._model_name))
            )
            exact_owned_model_remains = (
                self._created_model
                and model_names == expected_with_owned_model
            )
            if model_names != baseline_models:
                if not exact_owned_model_remains:
                    # A model outside the captured initialization baseline now
                    # relies on this process-global session.
                    self._owns_session = False
                    self._owned_session_baseline_incarnations = None
                    return
            for baseline_name, baseline_incarnation in baseline_incarnations:
                if str(self._facade.model.getCurrent()) != baseline_name:
                    self._facade.model.setCurrent(baseline_name)
                values = tuple(
                    str(item)
                    for item in self._facade.model.getAttribute(
                        _SESSION_BASELINE_INCARNATION_ATTRIBUTE
                    )
                )
                if values != (baseline_incarnation,):
                    self._owns_session = False
                    self._owned_session_baseline_incarnations = None
                    return
            if exact_owned_model_remains:
                if str(self._facade.model.getCurrent()) != self._model_name:
                    self._facade.model.setCurrent(self._model_name)
                if not self._current_model_has_expected_incarnation(
                    self._facade
                ):
                    if self._incarnation_verified:
                        self._created_model = False
                    self._owns_session = False
                    self._owned_session_baseline_incarnations = None
                    return
        self._facade.finalize()
        self._owns_session = False
        self._created_model = False
        self._incarnation_verified = False
        self._model_identity_inspection_failed = False
        self._owned_session_baseline_incarnations = None

    def set_numeric_options(
        self,
        replacements: Iterable[tuple[str, float]],
    ) -> None:
        """Snapshot all originals, then apply ordered numeric replacements."""
        if self._pending_options:
            raise GeometryStateError(
                f"geometry model {self._model_name!r}: mesh options already "
                "have a pending restoration"
            )
        gmsh = self._facade
        if gmsh is None:
            raise self._state_error(
                "mesh option transaction",
                "Gmsh session is not active",
            )
        materialized = tuple(replacements)
        for option_name, _ in materialized:
            self._pending_options[option_name] = float(
                gmsh.option.getNumber(option_name)
            )
        for option_name, value in materialized:
            gmsh.option.setNumber(option_name, float(value))

    def restore_pending_options(self) -> None:
        """Restore every pending option, retaining entries that still fail."""
        gmsh = self._facade
        if gmsh is None:
            return
        first_error: BaseException | None = None
        for option_name, value in tuple(self._pending_options.items()):
            try:
                gmsh.option.setNumber(option_name, value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                del self._pending_options[option_name]
        if first_error is not None:
            raise first_error

    def cleanup_after_failed_entry(
        self,
    ) -> tuple[tuple[str, BaseException], ...]:
        """Best-effort entry rollback with retryable ownership on failure."""
        gmsh = self._facade
        if gmsh is None:
            return ()
        try:
            initialized = self.inspect_initialized()
        except BaseException as error:
            # State is unknown: retain ownership and avoid speculative cleanup.
            return (("inspect Gmsh session state", error),)
        if not initialized:
            self.finalize_owned_session(initialized=False)
            return ()

        errors: list[tuple[str, BaseException]] = []
        for operation, callback in (
            ("restore mesh options", self.restore_pending_options),
            ("remove facade model", self.remove_created_model),
            ("restore prior model", self.restore_prior_model),
        ):
            try:
                callback()
            except BaseException as error:
                errors.append((operation, error))
        if self._owns_session:
            try:
                self.finalize_owned_session(initialized=True)
            except BaseException as error:
                errors.append(("finalize owned session", error))
        return tuple(errors)

    def _state_error(self, operation: str, detail: str) -> GeometryStateError:
        return GeometryStateError(
            f"geometry model {self._model_name!r}: {operation} failed because "
            f"{detail}"
        )

    def _assert_current_model_incarnation(
        self,
        gmsh: Any,
        operation: str,
    ) -> None:
        if not self._current_model_has_expected_incarnation(gmsh):
            raise self._state_error(
                operation,
                "facade-owned Gmsh model incarnation is missing or replaced",
            )

    def _current_model_has_expected_incarnation(self, gmsh: Any) -> bool:
        values = tuple(
            str(item)
            for item in gmsh.model.getAttribute(_MODEL_INCARNATION_ATTRIBUTE)
        )
        return values == (self._model_incarnation,)


__all__: list[str] = []
