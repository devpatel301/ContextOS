"""
Tests for Phase 7: Agent Runtime
"""
import pytest
from contextos import ContextOS, ContextConfig
from contextos.agent import AgentRuntime, StubLLM, get_default_tools

@pytest.fixture
def agent_cos(tmp_path):
    config = ContextConfig(
        db_path=str(tmp_path / "test_agent.db"),
        chroma_persist_dir=str(tmp_path / ".chroma_test"),
        embedding_stub=True,
        token_budget=1000,
        working_memory_reserve=200,
    )
    cos = ContextOS(config)
    yield cos
    cos.close()

def test_agent_runtime_weather_task(agent_cos):
    llm = StubLLM()
    tools = get_default_tools()
    
    agent = AgentRuntime(cos=agent_cos, llm=llm, tools=tools, max_steps=5)
    
    result = agent.run("Find out the weather in London and then calculate 25 * 4.")
    
    assert "15°C and Rainy" in result
    assert "100" in result
    assert llm.call_count == 3
    
    # Check that ContextOS recorded the episodic events
    status = agent_cos.status()
    assert status["tiers"]["working"]["blocks"] == 2 # Task + Tools
    assert status["tiers"]["episodic"]["blocks"] == 8 # Thought/Action/Obs(x2) + Thought/Answer
