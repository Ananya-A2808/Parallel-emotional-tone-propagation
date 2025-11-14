#!/usr/bin/env python3
"""
plot.py

Supports:
  --history <history.txt>        (one numeric value per line)
  --execution-time <csv>         (threads,time) - plots execution time vs threads
  --speedup <csv>                (threads,time) - plots speedup vs threads
Writes plot to --out (PNG).

Usage:
  python py/plot.py --history results/serial_history.txt --out results/plots/serial_plot.png
  python py/plot.py --execution-time results/execution_time.csv --out results/plots/execution_time.png
  python py/plot.py --speedup results/execution_time.csv --out results/plots/speedup.png
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(path, out):
    # history: one number per line
    try:
        hist = np.loadtxt(path)
    except Exception as e:
        print("Error reading history file:", e, file=sys.stderr)
        raise
    if hist.ndim == 0:
        hist = np.array([float(hist)])
    plt.figure(figsize=(7,3.5))
    plt.plot(hist, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Average sentiment")
    plt.title("Sentiment diffusion over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print("Wrote", out)

def read_execution_data(path):
    """Read execution time CSV and return threads and times arrays"""
    try:
        for sep in [',',';',None]:
            try:
                df = pd.read_csv(path, sep=sep, engine='python', comment='#', skip_blank_lines=True)
                if df.shape[1] >= 2:
                    break
            except Exception:
                df = None
        if df is None or df.shape[1] < 2:
            df = pd.read_csv(path, delim_whitespace=True, header=None, engine='python')
    except Exception as e:
        print("Failed to read CSV with pandas:", e, file=sys.stderr)
        raise

    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    if df.shape[1] < 2:
        raise SystemExit("execution-time file must contain at least two columns (threads,time). Found:\n" + str(df.head(10)))

    # find first two numeric columns
    df_cols = df.columns.tolist()
    numeric_cols = []
    for col in df_cols:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notna().sum() > 0:
            numeric_cols.append(col)
        if len(numeric_cols) >= 2:
            break
    if len(numeric_cols) < 2:
        raise SystemExit("Couldn't find two numeric columns in execution-time CSV. Inspect file.")

    threads_col, time_col = numeric_cols[0], numeric_cols[1]
    df2 = df[[threads_col, time_col]].copy()
    df2[threads_col] = pd.to_numeric(df2[threads_col], errors='coerce')
    df2[time_col] = pd.to_numeric(df2[time_col], errors='coerce')
    df2 = df2.dropna(subset=[threads_col, time_col])

    df2 = df2.sort_values(by=threads_col)
    threads = df2[threads_col].to_numpy()
    times = df2[time_col].to_numpy()
    
    return threads, times, df2

def plot_execution_time(path, out):
    """Plot execution time vs threads (separate PNG)"""
    threads, times, df2 = read_execution_data(path)
    
    plt.figure(figsize=(6, 4))
    plt.plot(threads, times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel("Threads", fontsize=11)
    plt.ylabel("Execution time (s)", fontsize=11)
    plt.title("Execution Time vs Threads", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("Wrote", out)

def plot_speedup(path, out):
    """Plot speedup vs threads (separate PNG)"""
    threads, times, df2 = read_execution_data(path)
    
    # Calculate speedup (relative to 1 thread)
    baseline_time = times[0] if len(times) > 0 else 1.0
    speedup = baseline_time / times

    plt.figure(figsize=(6, 4))
    plt.plot(threads, speedup, marker='s', linewidth=2, markersize=8, color='#A23B72')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1 thread)')
    plt.xlabel("Threads", fontsize=11)
    plt.ylabel("Speedup (relative to 1 thread)", fontsize=11)
    plt.title("Speedup vs Threads", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)
    plt.legend(loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("Wrote", out)
    
    # Also save a speedup CSV
    speedup_df = df2.copy()
    speedup_df['speedup'] = speedup
    speedup_file = os.path.splitext(out)[0] + "_speedup.csv"
    speedup_df.to_csv(speedup_file, index=False)
    print("Wrote speedup CSV to", speedup_file)
    
    cleaned = os.path.splitext(out)[0] + "_cleaned.csv"
    df2.to_csv(cleaned, index=False)
    print("Wrote cleaned CSV to", cleaned)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", help="history file (one numeric per line)")
    parser.add_argument("--speedup", help="speedup CSV file (threads,time). Header allowed.")
    parser.add_argument("--execution-time", help="execution time CSV file (threads,time). Header allowed.")
    parser.add_argument("--out", default="results/plot.png", help="output image path (PNG)")
    opt = parser.parse_args()

    count = sum([bool(opt.history), bool(opt.speedup), bool(opt.execution_time)])
    if count == 0:
        parser.error("Specify one of --history, --speedup, or --execution-time")
    if count > 1:
        parser.error("Specify only one of --history, --speedup, or --execution-time")

    if opt.history:
        plot_history(opt.history, opt.out)
    elif opt.speedup:
        plot_speedup(opt.speedup, opt.out)
    elif opt.execution_time:
        plot_execution_time(opt.execution_time, opt.out)

if __name__ == "__main__":
    main()
