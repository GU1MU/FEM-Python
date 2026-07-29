"""Deterministic user-visible names for V1 authoring."""

from __future__ import annotations

import unicodedata
import re
from collections.abc import Iterable, Mapping


CONTROLLED_OBJECT_TYPES = frozenset(
    {
        "模型",
        "部件",
        "实体",
        "点",
        "边",
        "面",
        "域",
        "材料",
        "截面",
        "分析步",
        "位移",
        "载荷",
        "结果请求",
        "作业",
    }
)
_SYSTEM_NAMES = frozenset(
    {
        "model-1",
        "part-1",
        "body-1",
        "left",
        "right",
        "hole",
        "domain",
    }
)


class NamePolicyError(ValueError):
    """A requested visible name violates the locked naming boundary."""


class NamePolicy:
    """Validate and normalize ``{type}-{function}`` names."""

    def __init__(self, *, max_name_length: int = 96) -> None:
        if type(max_name_length) is not int or max_name_length < 8:
            raise ValueError("max_name_length must be an integer of at least 8")
        self.max_name_length = max_name_length

    @staticmethod
    def canonical_key(value: str) -> str:
        if type(value) is not str:
            raise TypeError("name must be a string")
        return unicodedata.normalize("NFKC", value).casefold()

    def compose(self, object_type: str, function: str) -> str:
        if type(object_type) is not str or object_type not in CONTROLLED_OBJECT_TYPES:
            raise NamePolicyError("object type is not controlled")
        if type(function) is not str:
            raise TypeError("function must be a string")
        if function != function.strip():
            raise NamePolicyError("function cannot contain surrounding whitespace")
        normalized = unicodedata.normalize("NFKC", function)
        if not normalized:
            raise NamePolicyError("function must not be empty")
        if any(
            unicodedata.category(character).startswith("C") for character in normalized
        ):
            raise NamePolicyError("function cannot contain control characters")
        if self.canonical_key(normalized) in _SYSTEM_NAMES:
            raise NamePolicyError("function cannot impersonate a system name")
        result = f"{object_type}-{normalized}"
        if len(result) > self.max_name_length:
            raise NamePolicyError("name is too long")
        return result

    def validate(self, name: str) -> str:
        if type(name) is not str:
            raise TypeError("name must be a string")
        if name != name.strip():
            raise NamePolicyError("name cannot contain surrounding whitespace")
        object_type, separator, function = name.partition("-")
        if not separator:
            raise NamePolicyError("name must use {type}-{function}")
        normalized = self.compose(object_type, function)
        if normalized != name:
            raise NamePolicyError("name is not Unicode-normalized")
        return normalized


class NameAllocator:
    """Allocate stable short suffixes inside explicit object namespaces."""

    def __init__(
        self,
        existing: Mapping[str, Iterable[str]] | None = None,
        *,
        policy: NamePolicy | None = None,
    ) -> None:
        self.policy = NamePolicy() if policy is None else policy
        if type(self.policy) is not NamePolicy:
            raise TypeError("policy must be NamePolicy")
        self._keys: dict[str, set[str]] = {}
        for namespace, names in ({} if existing is None else existing).items():
            for name in names:
                self.reserve(namespace, name)

    def reserve(self, namespace: str, name: str) -> str:
        normalized_namespace = self._namespace(namespace)
        if type(name) is not str or not name or name != name.strip():
            raise NamePolicyError(
                "existing name must be non-empty without outer whitespace"
            )
        normalized_name = unicodedata.normalize("NFKC", name)
        key = self.policy.canonical_key(normalized_name)
        occupied = self._keys.setdefault(normalized_namespace, set())
        if key in occupied:
            raise NamePolicyError(
                f"name already exists in namespace {normalized_namespace!r}"
            )
        occupied.add(key)
        return normalized_name

    def allocate(
        self,
        namespace: str,
        object_type: str,
        function: str,
    ) -> str:
        normalized_namespace = self._namespace(namespace)
        base = self.policy.compose(object_type, function)
        occupied = self._keys.setdefault(normalized_namespace, set())
        candidate = base
        suffix = 2
        while self.policy.canonical_key(candidate) in occupied:
            candidate = f"{base}-{suffix}"
            if len(candidate) > self.policy.max_name_length:
                raise NamePolicyError("stable suffix would make the name too long")
            suffix += 1
        occupied.add(self.policy.canonical_key(candidate))
        return candidate

    def require_next(
        self,
        namespace: str,
        object_type: str,
        name: str,
    ) -> str:
        """Require *name* to be exactly the next stable allocation."""

        normalized = self.policy.validate(name)
        prefix = f"{object_type}-"
        if not normalized.startswith(prefix):
            raise NamePolicyError("name uses the wrong controlled object type")
        function = normalized[len(prefix) :]
        suffix = re.fullmatch(r"(.+)-([2-9][0-9]*)", function)
        base_function = suffix.group(1) if suffix is not None else function
        expected = self.allocate(namespace, object_type, base_function)
        if normalized != expected:
            raise NamePolicyError(
                f"name is not the next stable allocation: expected {expected!r}"
            )
        return normalized

    @staticmethod
    def _namespace(value: str) -> str:
        if type(value) is not str or not value or value != value.strip():
            raise NamePolicyError(
                "namespace must be non-empty without outer whitespace"
            )
        return unicodedata.normalize("NFKC", value).casefold()


__all__ = [
    "CONTROLLED_OBJECT_TYPES",
    "NameAllocator",
    "NamePolicy",
    "NamePolicyError",
]
