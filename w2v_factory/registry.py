from __future__ import annotations
from typing import Any, Callable, Dict

class Registry:
    """Simple registry enabling pluggable factories or plain callables."""
    def __init__(self) -> None:
        self._objs: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._objs:
                raise KeyError(f"Registry duplicate: {name}")
            self._objs[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._objs:
            raise KeyError(f"{name} not found in registry. Available: {list(self._objs)}")
        return self._objs[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Factory: if kwargs are provided, call the registered target; otherwise return it."""
        obj = self.get(name)
        if kwargs:
            return obj(**kwargs)
        return obj

# Global registries
MODEL_REG = Registry()
LOSS_REG = Registry()
TOKENIZER_REG = Registry()
SCHED_REG = Registry()
