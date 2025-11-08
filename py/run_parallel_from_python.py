#!/usr/bin/env python3
"""
run_parallel_from_python.py
---------------------------
Bridge between Python and the C++ OpenMP binary.

Usage example:
    python py/run_parallel_from_python.py \
        --graph data/graph.txt \
        --states data/states.txt \
        --steps 100 \
        --alpha 0.25 \
        --threads 4 \
        --out_dir results/

This script:
  - Detects and runs cpp/parallel_update (or .exe on Windows)
  - Measures runtime
  - Loads the resulting history and prints summary
"""

import argparse
import os
import sys
import subprocess
import time
import numpy as np

def find_cpp_binary():
    """Find compiled C++ binary (supports Windows & Linux)"""
    bin_unix = os.path.join("cpp", "parallel_update")
    bin_win = os.path.join("cpp", "parallel_update.exe")
    if os.path.exists(bin_win):
        return bin_win
    elif os.path.exists(bin_unix):
        return bin_unix
    else:
        raise FileNotFoundError(
            "parallel_update binary not found. Please build it using:\n"
            "bash cpp/build.sh"
        )

def main():
    parser = argparse.ArgumentParser(description="Run C++ parallel simulation from Python")
    parser.add_argument("--graph", required=True, help="Path to graph.txt")
    parser.add_argument("--states", required=True, help="Path to states.txt")
    parser.add_argument("--steps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha parameter")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--out_dir", default="results/", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cpp_bin = find_cpp_binary()
    history_path = os.path.join(args.out_dir, f"history_{args.threads}.txt")
    out_states = os.path.join(args.out_dir, f"out_states_{args.threads}.txt")

    print(f"Running parallel simulation using {args.threads} threads...")
    start = time.time()

    cmd = [
        cpp_bin,
        args.graph,
        args.states,
        out_states,
        history_path,
        str(args.steps),
        str(args.alpha),
        str(args.threads)
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Error executing C++ binary:", e)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n✅ Completed parallel run in {elapsed:.3f} seconds")
    print(f"History: {history_path}")
    print(f"Final states: {out_states}")

    # Optional: load history and show quick summary
    try:
        hist = np.loadtxt(history_path)
        print(f"Loaded history of {len(hist)} timesteps.")
        print(f"First few values: {hist[:5]}")
    except Exception:
        print("⚠️ Could not read history file (may not be numeric).")

if __name__ == "__main__":
    main()
