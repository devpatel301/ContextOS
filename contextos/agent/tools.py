"""
ContextOS — Agent Tools (Phase 7).
"""
from __future__ import annotations

from typing import Callable, Any
import json
import logging

logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func
        
    def execute(self, arg: str) -> str:
        try:
            logger.info("Executing tool '%s' with arg: %s", self.name, arg)
            return str(self.func(arg))
        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            return f"Error: {e}"

def calc_func(expression: str) -> str:
    """Simple calculator. VERY unsafe for production, but fine for demo."""
    # Strip any potential wrapping quotes
    expression = expression.strip().strip("'").strip('"')
    try:
        # evaluate math string safely-ish
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Invalid expression: {e}"

def weather_func(location: str) -> str:
    """Fake weather API."""
    location = location.strip().lower()
    if "london" in location:
        return "15°C and Rainy"
    elif "tokyo" in location:
        return "22°C and Sunny"
    else:
        return "20°C and Cloudy"

calculator_tool = Tool(
    name="calculator",
    description="Evaluate a math expression. Input should be a mathematical string like '2 + 2'.",
    func=calc_func
)

weather_tool = Tool(
    name="weather",
    description="Get the current weather for a city. Input should be the city name.",
    func=weather_func
)

def get_default_tools() -> list[Tool]:
    return [calculator_tool, weather_tool]
