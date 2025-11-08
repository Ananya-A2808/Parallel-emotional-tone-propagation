import subprocess, numpy as np, os, sys, tempfile

def write_toy():
    os.makedirs("data/toy", exist_ok=True)
    # simple 3-node directed graph
    with open("data/toy/graph.txt","w") as f:
        f.write("3 4\n0 1\n1 2\n2 0\n0 2\n")
    with open("data/toy/states.txt","w") as f:
        f.write("0.5\n-0.6\n0.0\n")
    return "data/toy/graph.txt","data/toy/states.txt"

def test_serial_vs_cpp():
    g,s = write_toy()
    # run python serial
    os.makedirs("results", exist_ok=True)
    cmd_py = ["python3","py/serial_sim.py","--graph",g,"--states",s,"--steps","20","--alpha","0.25","--out","results/serial_history_toy.txt"]
    subprocess.check_call(cmd_py)
    # run C++ (assumes binary at cpp/parallel_update)
    cpp_bin = "cpp/parallel_update"
    assert os.path.exists(cpp_bin), f"C++ binary not found at {cpp_bin}"
    cmd_cpp = [cpp_bin, g, s, "data/toy/out_states.txt", "data/toy/history.txt", "20", "0.25", "1"]
    subprocess.check_call(cmd_cpp)
    # compare histories
    h_py = np.loadtxt("results/serial_history_toy.txt")
    h_cpp = np.loadtxt("data/toy/history.txt")
    maxdiff = float(np.max(np.abs(h_py - h_cpp)))
    print("maxdiff", maxdiff)
    assert maxdiff < 1e-9, f"Results differ (maxdiff={maxdiff})"
