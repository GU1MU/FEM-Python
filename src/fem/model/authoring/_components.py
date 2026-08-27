from __future__ import annotations

from typing import Sequence


def component_ranges(components: int | Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return compact contiguous 1-based component ranges."""
    if isinstance(components, int):
        values = (int(components),)
    else:
        values = tuple(int(component) for component in components)

    if not values:
        raise ValueError("components must not be empty")
    for component in values:
        if component < 1:
            raise ValueError("components are 1-based and must be positive")

    sorted_values = tuple(sorted(set(values)))
    ranges: list[tuple[int, int]] = []
    start = sorted_values[0]
    previous = start
    for component in sorted_values[1:]:
        if component == previous + 1:
            previous = component
            continue
        ranges.append((start, previous))
        start = previous = component
    ranges.append((start, previous))
    return tuple(ranges)
