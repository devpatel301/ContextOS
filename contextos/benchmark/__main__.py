"""
ContextOS — Benchmark Harness (Phase 8).
"""
import argparse
from contextos.benchmark.harness import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="ContextOS Benchmark Harness")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--workload", type=str, default="long_conversation", choices=["long_conversation", "qa", "repeated_queries"])
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_benchmark(args.workload)

if __name__ == "__main__":
    main()
