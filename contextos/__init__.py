"""
ContextOS — public package.

Usage:
    from contextos import ContextOS, ContextConfig
    from contextos.models import MemoryBlock, MemoryTier, ContextWindow
"""
from contextos.config import ContextConfig
from contextos.contextos import ContextOS
from contextos.models import ContextWindow, MemoryBlock, MemoryTier

__all__ = [
    "ContextOS",
    "ContextConfig",
    "MemoryBlock",
    "MemoryTier",
    "ContextWindow",
]

__version__ = "0.1.0"
