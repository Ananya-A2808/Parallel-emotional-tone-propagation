# tests/test_small_graph.py
import subprocess, os, sys
import numpy as np

def write_toy():
    os.makedirs("data/toy", exist_ok=True)
    with open("data/toy/graph.txt","w") as f:
        f.write("3 4\n0 1\n1 2\n2 0\n0 2\n")
    with open("data/toy/states.txt","w") as f:
        f.write("0.5\n-0.6\n0.0\n")
    return "data/toy/graph.txt","data/toy/states.txt"

def find_cpp_binary():
    bin_unix = os.path.join("cpp","parallel_update")
    bin_win = os.path.join("cpp","parallel_update.exe")
    if os.path.exists(bin_win):
        return bin_win
    if os.path.exists(bin_unix):
        return bin_unix
    from shutil import which
    alt = which("parallel_update") or which("parallel_update.exe")
    if alt:
        return alt
    raise FileNotFoundError("C++ binary parallel_update not found. Build it with cpp/build.sh")

def test_serial_vs_cpp():
    g, s = write_toy()
    os.makedirs("results", exist_ok=True)

    # Use the current Python executable to avoid 'python3' vs 'python' issues on Windows
    py_exec = sys.executable

    # run python serial
    cmd_py = [py_exec, "py/serial_sim.py", "--graph", g, "--states", s,
              "--steps", "20", "--alpha", "0.25", "--out", "results/serial_history_toy.txt"]
    subprocess.check_call(cmd_py)

    # run C++ binary (single-thread) - find binary robustly
    cpp_bin = find_cpp_binary()
    cmd_cpp = [cpp_bin, g, s, "data/toy/out_states.txt", "data/toy/history.txt", "20", "0.25", "1"]
    subprocess.check_call(cmd_cpp)

    # compare histories using a tolerant check
    h_py = np.loadtxt("results/serial_history_toy.txt")
    h_cpp = np.loadtxt("data/toy/history.txt")

    # compute max absolute difference (for debug / reporting)
    maxdiff = float(np.max(np.abs(h_py - h_cpp)))
    print("maxdiff", maxdiff)

    # Use numpy.allclose with sensible tolerances:
    # - rtol handles relative differences (small for values ~1)
    # - atol handles small absolute differences from rounding
    # choose atol=1e-7, rtol=1e-6 (conservative & strict enough)
    assert np.allclose(h_py, h_cpp, rtol=1e-6, atol=1e-7), f"Results differ (maxdiff={maxdiff})"
