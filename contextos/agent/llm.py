"""
ContextOS — LLM Interfaces (Phase 7).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    thought: str
    action_name: str | None = None
    action_input: str | None = None
    final_answer: str | None = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, tools: list[Any]) -> LLMResponse:
        pass


class StubLLM(BaseLLM):
    """
    A fake LLM that parses the prompt to decide what to do.
    Used for testing the agent loop without making real API calls.
    """
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, tools: list[Any]) -> LLMResponse:
        self.call_count += 1
        
        # Simple heuristic to fake an agent reasoning process
        if self.call_count == 1:
            return LLMResponse(
                thought="I need to use the weather tool.",
                action_name="weather",
                action_input="London",
            )
        elif self.call_count == 2:
            return LLMResponse(
                thought="I need to calculate something.",
                action_name="calculator",
                action_input="25 * 4",
            )
        elif self.call_count == 3:
            return LLMResponse(
                thought="I have both results, I can answer the user now.",
                final_answer="The weather is 15°C and Rainy, and 25 * 4 is 100."
            )
        else:
            return LLMResponse(
                thought="I'll just answer directly.",
                final_answer="I am a stub LLM. I don't know the answer."
            )
