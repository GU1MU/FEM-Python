"""Immutable global degree-of-freedom layout for one compiled analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldDofs:
    """Global DOFs owned by one named solution field."""

    name: str
    dofs: tuple[int, ...]
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("field name must be nonblank")
        dofs = tuple(_dof(value) for value in self.dofs)
        if not dofs:
            raise ValueError(f"field {name!r} must own at least one DOF")
        if len(dofs) != len(set(dofs)):
            raise ValueError(f"field {name!r} DOFs must be unique")
        components = tuple(str(value).strip() for value in self.components)
        if any(not value for value in components):
            raise ValueError("field component names must be nonblank")
        if len(components) != len(set(components)):
            raise ValueError("field component names must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, slots=True)
class DofSpace:
    """Complete named-field partition of a global solution vector.

    Field DOFs may be contiguous or interleaved.  Requiring an exact,
    non-overlapping partition keeps solvers vector-based while allowing
    physics operators and result publishers to address named fields.
    """

    num_dofs: int
    fields: tuple[FieldDofs, ...]

    def __post_init__(self) -> None:
        if isinstance(self.num_dofs, bool) or not isinstance(self.num_dofs, int):
            raise TypeError("num_dofs must be an integer")
        if self.num_dofs <= 0:
            raise ValueError("num_dofs must be positive")
        fields = tuple(self.fields)
        if not fields or any(type(field) is not FieldDofs for field in fields):
            raise TypeError("fields must contain FieldDofs values")
        names = tuple(field.name for field in fields)
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        all_dofs = tuple(dof for field in fields for dof in field.dofs)
        if len(all_dofs) != len(set(all_dofs)):
            raise ValueError("solution fields must not share global DOFs")
        expected = set(range(self.num_dofs))
        actual = set(all_dofs)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise ValueError(
                "solution fields must partition [0, num_dofs); "
                f"missing={missing}, extra={extra}"
            )
        object.__setattr__(self, "fields", fields)

    @classmethod
    def single_field(
        cls,
        name: str,
        num_dofs: int,
        *,
        components: tuple[str, ...] = (),
    ) -> "DofSpace":
        return cls(
            num_dofs=int(num_dofs),
            fields=(
                FieldDofs(
                    name=name,
                    dofs=tuple(range(int(num_dofs))),
                    components=components,
                ),
            ),
        )

    @classmethod
    def displacement_for_mesh(cls, mesh: object) -> "DofSpace":
        num_dofs = int(getattr(mesh, "num_dofs"))
        width = int(getattr(mesh, "dofs_per_node"))
        components = tuple(f"U{index}" for index in range(1, width + 1))
        return cls.single_field("U", num_dofs, components=components)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> FieldDofs:
        normalized = str(name).strip()
        for field in self.fields:
            if field.name == normalized:
                return field
        raise KeyError(f"solution field {normalized!r} is not defined")


def _dof(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("DOF ids must be integers")
    if value < 0:
        raise ValueError("DOF ids must be non-negative")
    return int(value)


__all__ = ["DofSpace", "FieldDofs"]
