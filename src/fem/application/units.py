"""Typed unit convention stored with native projects."""

from __future__ import annotations

from dataclasses import dataclass


def _unit_label(
    value: object,
    field_name: str,
    *,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > 64:
        raise ValueError(f"{field_name} is too long")
    return value


@dataclass(frozen=True, slots=True)
class UnitContext:
    """Self-consistent labels; numeric project values are never converted."""

    length: str
    force: str
    stress: str
    density: str | None = None
    acceleration: str | None = None
    convention: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("length", "force", "stress"):
            object.__setattr__(
                self,
                field_name,
                _unit_label(getattr(self, field_name), field_name),
            )
        for field_name in ("density", "acceleration", "convention"):
            object.__setattr__(
                self,
                field_name,
                _unit_label(
                    getattr(self, field_name),
                    field_name,
                    optional=True,
                ),
            )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "length": self.length,
            "force": self.force,
            "stress": self.stress,
            "density": self.density,
            "acceleration": self.acceleration,
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, value: object) -> UnitContext:
        if not isinstance(value, dict):
            raise TypeError("unit context must be an object")
        expected = {
            "length",
            "force",
            "stress",
            "density",
            "acceleration",
            "convention",
        }
        if set(value) != expected:
            raise ValueError("unit context fields do not match the schema")
        return cls(
            length=value["length"],  # type: ignore[arg-type]
            force=value["force"],  # type: ignore[arg-type]
            stress=value["stress"],  # type: ignore[arg-type]
            density=value["density"],  # type: ignore[arg-type]
            acceleration=value["acceleration"],  # type: ignore[arg-type]
            convention=value["convention"],  # type: ignore[arg-type]
        )


__all__ = ["UnitContext"]
