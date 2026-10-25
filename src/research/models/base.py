"""Framework-light model contracts."""

from __future__ import annotations

from typing import Any, Protocol


class ModelFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...
