"""Configuration-driven object construction."""

from __future__ import annotations

import importlib
from typing import Any


def instantiate(config: dict[str, Any]) -> Any:
    values = dict(config)
    target = values.pop("_target_", None)
    if not isinstance(target, str) or "." not in target:
        raise ValueError("component config requires a dotted _target_")
    module_name, attribute = target.rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(**values)
