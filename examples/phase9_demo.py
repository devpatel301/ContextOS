"""
examples/phase9_demo.py

Demonstrates Multi-Agent Isolation and Context Version Control.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextos import ContextOS, ContextConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    print("=" * 60)
    print("ContextOS Phase 9 Demo: Multi-Agent & Context VCS")
    print("=" * 60)
    
    # --- Multi-Agent Support ---
    print("\n1. Booting Agent 01")
    config1 = ContextConfig(agent_id="agent_01", db_path="./demo_p9.db", chroma_persist_dir="./.chroma_p9", embedding_stub=True)
    agent1_os = ContextOS(config1)
    
    agent1_os.store("Agent 1 discovered the cure for everything is apples.")
    
    print("\n2. Booting Agent 02")
    config2 = ContextConfig(agent_id="agent_02", db_path="./demo_p9.db", chroma_persist_dir="./.chroma_p9", embedding_stub=True)
    agent2_os = ContextOS(config2)
    
    agent2_os.store("Agent 2 thinks the earth is flat.")
    
    # Verify isolation
    print("\n[Isolation Check]")
    print("Agent 1 Memory Blocks:", agent1_os._store.count())
    print("Agent 2 Memory Blocks:", agent2_os._store.count())
    
    # --- Context Version Control ---
    print("\n3. Testing Context VCS on Agent 1")
    print("Snapshotting Agent 1 state to 'pre_experiment'...")
    agent1_os.commit(tag="pre_experiment")
    
    print("Agent 1 makes a mistake and fills memory with junk...")
    for i in range(5):
        agent1_os.store(f"Bad reasoning step {i}")
        
    print(f"Agent 1 Memory Blocks after mistake: {agent1_os._store.count()}")
    
    print("Checking out 'pre_experiment' snapshot...")
    agent1_os.checkout(tag="pre_experiment")
    print(f"Agent 1 Memory Blocks after restore: {agent1_os._store.count()}")
    
    print("\nDemo successful!")
    
    agent1_os.close()
    agent2_os.close()

    # Cleanup
    import os, shutil
    import glob
    for f in glob.glob("./demo_p9*.db"):
        os.remove(f)
    for d in glob.glob("./.chroma_p9*"):
        shutil.rmtree(d)

if __name__ == "__main__":
    main()
