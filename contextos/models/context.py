"""
ContextOS — ContextWindow: the output of the scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from contextos.models.block import MemoryBlock
from contextos.models.tier import MemoryTier


@dataclass
class ContextWindow:
    """
    The packed, budget-respecting set of memory blocks ready to send to an LLM.

    Produced by ContextScheduler.schedule(). Contains ordered MemoryBlocks
    and helpers to format them as a prompt string.
    """

    blocks: list[MemoryBlock] = field(default_factory=list)
    token_budget: int = 0
    tokens_used: int = 0
    query: str = ""

    # ── Accessors ─────────────────────────────────────────────────────────────

    def blocks_by_tier(self, tier: MemoryTier) -> list[MemoryBlock]:
        return [b for b in self.blocks if b.tier == tier]

    def token_usage_by_tier(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for tier in MemoryTier:
            blocks = self.blocks_by_tier(tier)
            usage[tier.value] = sum(b.effective_token_count() for b in blocks)
        return usage

    def utilization(self) -> float:
        """0.0–1.0 fraction of token budget used."""
        if self.token_budget == 0:
            return 0.0
        return self.tokens_used / self.token_budget

    # ── Prompt formatting ──────────────────────────────────────────────────────

    def as_prompt(self) -> str:
        """
        Format the context window as a structured string ready to prepend
        to an LLM prompt.

        Layout:
            [SYSTEM]           ← WORKING blocks tagged 'system'
            [WORKING MEMORY]   ← other WORKING blocks (current turn)
            [RECENT CONTEXT]   ← EPISODIC blocks
            [RETRIEVED]        ← SEMANTIC / ARCHIVED blocks
        """
        sections: list[str] = []

        header = (
            f"[CONTEXT WINDOW — {self.tokens_used:,} / {self.token_budget:,} tokens used]"
        )
        sections.append(header)

        # System / instructions
        system_blocks = [b for b in self.blocks_by_tier(MemoryTier.WORKING) if "system" in b.tags]
        if system_blocks:
            sections.append("\n[SYSTEM INSTRUCTIONS]")
            for b in system_blocks:
                sections.append(b.effective_content())

        # Current turn working memory
        turn_blocks = [
            b for b in self.blocks_by_tier(MemoryTier.WORKING) if "system" not in b.tags
        ]
        if turn_blocks:
            sections.append("\n[WORKING MEMORY — current turn]")
            for b in turn_blocks:
                sections.append(b.effective_content())

        # Episodic — recent conversation
        episodic = self.blocks_by_tier(MemoryTier.EPISODIC)
        if episodic:
            sections.append("\n[RECENT CONTEXT]")
            for b in episodic:
                sections.append(b.effective_content())

        # Semantic / Archived — retrieved
        retrieved = self.blocks_by_tier(MemoryTier.SEMANTIC) + self.blocks_by_tier(
            MemoryTier.ARCHIVED
        )
        if retrieved:
            sections.append("\n[RETRIEVED MEMORIES]")
            for b in retrieved:
                sections.append(b.effective_content())

        return "\n".join(sections)

    def summary(self) -> str:
        """One-line summary for logging / profiler display."""
        usage = self.token_usage_by_tier()
        parts = [f"{k}={v}" for k, v in usage.items() if v > 0]
        return (
            f"ContextWindow({self.tokens_used}/{self.token_budget} tokens, "
            f"blocks={len(self.blocks)}, [{', '.join(parts)}])"
        )

    def __repr__(self) -> str:
        return self.summary()
