"""contextos.eviction package — eviction policies and engine."""
from contextos.eviction.base import EvictionPolicy
from contextos.eviction.engine import EvictionEngine

__all__ = ["EvictionPolicy", "EvictionEngine"]
