"""
ContextOS — Agent Runtime (Phase 7).
"""
import logging
from typing import Any

from contextos import ContextOS
from contextos.agent.llm import BaseLLM
from contextos.agent.tools import Tool
from contextos.models.tier import MemoryTier

logger = logging.getLogger(__name__)


class AgentRuntime:
    """
    An autonomous agent execution loop that uses ContextOS as its memory substrate.
    """

    def __init__(self, cos: ContextOS, llm: BaseLLM, tools: list[Tool], max_steps: int = 10):
        self.cos = cos
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _format_tools_prompt(self) -> str:
        if not self.tools:
            return "No tools available."
        prompt = "Available Tools:\n"
        for name, tool in self.tools.items():
            prompt += f"- {name}: {tool.description}\n"
        return prompt

    def run(self, task: str) -> str:
        """
        Execute the agent loop for a given task.
        """
        logger.info(f"Starting agent task: {task}")
        
        # 1. Store the initial task in Working memory (pinned)
        self.cos.store(f"Current Task: {task}", tier=MemoryTier.WORKING, user_weight=1.0)
        self.cos.store(self._format_tools_prompt(), tier=MemoryTier.WORKING, user_weight=0.8)

        step = 0
        while step < self.max_steps:
            step += 1
            logger.info(f"--- Step {step} ---")

            # 2. Allocate context window through ContextOS Scheduler
            # We use the current task as the retrieval query to pull relevant Semantic/Archived memories
            # while the scheduler perfectly balances Working and Episodic memory.
            context_window = self.cos.retrieve(query=task, budget=self.cos.config.token_budget)
            prompt = context_window.as_prompt()

            # 3. Call the LLM
            response = self.llm.generate(prompt, tools=list(self.tools.values()))

            # 4. Store the agent's thought process
            self.cos.store(f"Thought: {response.thought}", tier=MemoryTier.EPISODIC)

            # 5. Check if finished
            if response.final_answer:
                logger.info(f"Task Complete: {response.final_answer}")
                self.cos.store(f"Final Answer: {response.final_answer}", tier=MemoryTier.EPISODIC)
                return response.final_answer

            # 6. Execute tool action
            if response.action_name:
                action = response.action_name
                input_arg = response.action_input or ""
                
                self.cos.store(f"Action: {action}({input_arg})", tier=MemoryTier.EPISODIC)
                
                if action in self.tools:
                    tool_result = self.tools[action].execute(input_arg)
                    logger.info(f"Tool {action} returned: {tool_result}")
                    self.cos.store(f"Observation: {tool_result}", tier=MemoryTier.EPISODIC)
                else:
                    error_msg = f"Error: Tool '{action}' not found."
                    logger.error(error_msg)
                    self.cos.store(f"Observation: {error_msg}", tier=MemoryTier.EPISODIC)

            # 7. ContextOS GC/Eviction naturally happens inside cos.retrieve/store 
            # if the configured limits are hit.
            
        logger.warning("Agent reached max steps without completing the task.")
        return "Error: Max steps reached."
