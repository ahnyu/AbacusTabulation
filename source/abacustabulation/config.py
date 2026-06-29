"""YAML config loading with small inheritance support."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_INHERIT_KEYS = ("inherits", "inherit", "base")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving optional base-config inheritance.

    A config may contain one of ``inherits``, ``inherit``, or ``base`` with a
    path or list of paths. Base paths are resolved relative to the child config
    file. Merge rules are intentionally simple: dictionaries merge recursively,
    while lists, scalars, and explicit ``null`` values replace the base value.
    """

    return _load_config(Path(path), stack=())


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``override``."""

    merged = dict(base)
    for key, value in override.items():
        if key in _INHERIT_KEYS:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    import yaml

    resolved = path.expanduser().resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config {resolved} must contain a YAML mapping at top level.")

    merged: dict[str, Any] = {}
    for parent in _inherit_paths(data, resolved.parent):
        merged = merge_configs(merged, _load_config(parent, stack=(*stack, resolved)))
    return merge_configs(merged, data)


def _inherit_paths(data: Mapping[str, Any], base_dir: Path) -> tuple[Path, ...]:
    value = None
    found = [key for key in _INHERIT_KEYS if key in data]
    if not found:
        return ()
    if len(found) > 1:
        raise ValueError(f"Use only one config inheritance key, not {found}.")
    value = data[found[0]]
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        parent = Path(str(item)).expanduser()
        if not parent.is_absolute():
            parent = base_dir / parent
        paths.append(parent)
    return tuple(paths)
