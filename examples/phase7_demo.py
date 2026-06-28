"""
examples/phase7_demo.py

Demonstrates Phase 7: Agent Runtime using ContextOS Memory
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig
from contextos.agent import AgentRuntime, StubLLM, get_default_tools

# Configure minimal logging so we can see the agent's thought process
logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    print("=" * 60)
    print("ContextOS Phase 7 Demo: Agent Runtime Loop")
    print("=" * 60)
    
    config = ContextConfig(
        db_path="./demo_p7.db",
        chroma_persist_dir="./.chroma_demo_p7",
        embedding_stub=True,
        token_budget=2000,
        working_memory_reserve=500
    )
    cos = ContextOS(config)
    
    llm = StubLLM()
    tools = get_default_tools()
    
    agent = AgentRuntime(cos=cos, llm=llm, tools=tools, max_steps=5)
    
    task = "Find out the weather in London and then calculate 25 * 4."
    print(f"\n[Task] {task}\n")
    
    result = agent.run(task)
    
    print("\n" + "=" * 60)
    print("Final Result:", result)
    print("=" * 60)
    
    print("\n[ContextOS Status after Agent Run]")
    status = cos.status()
    for tier, data in status['tiers'].items():
        print(f"  {tier.upper()}: {data['blocks']} blocks, {data['tokens']} tokens")
        
    cos.close()
    
    # Cleanup
    import os, shutil
    if os.path.exists("./demo_p7.db"):
        os.remove("./demo_p7.db")
    if os.path.exists("./.chroma_demo_p7"):
        shutil.rmtree("./.chroma_demo_p7")

if __name__ == "__main__":
    main()
