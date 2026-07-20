"""Private native Gmsh session and model lease management."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..errors import GeometryStateError
from . import backend


class _GmshModelSession:
    """Own one facade model and any process-global session it initializes.

    The native Gmsh API is process-global.  This object deliberately keeps
    resource ownership separate from the public model state so failed cleanup
    can be retried after the public facade has entered ``CLOSED``.
    """

    __slots__ = (
        "_created_model",
        "_facade",
        "_model_name",
        "_owns_session",
        "_pending_options",
        "_prior_current",
        "_prior_current_captured",
    )

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._facade: Any | None = None
        self._owns_session = False
        self._created_model = False
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
            gmsh.initialize()

        self._prior_current = str(gmsh.model.getCurrent())
        self._prior_current_captured = True
        existing_models = tuple(str(item) for item in gmsh.model.list())
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
        return gmsh

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
            gmsh.model.remove()
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
            return
        if not self._owns_session or self._facade is None:
            return
        self._facade.finalize()
        self._owns_session = False
        self._created_model = False

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


__all__: list[str] = []
