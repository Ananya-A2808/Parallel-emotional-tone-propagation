#!/usr/bin/env python3
"""
plot.py

Supports:
  --history <history.txt>   (one numeric value per line)
  --speedup <speedup.csv>   (threads,time,anything) ; header allowed
Writes plot to --out (PNG).

Usage:
  python py/plot.py --history results/serial_history.txt --out results/plots/serial_plot.png
  python py/plot.py --speedup results/speedup.csv --out results/plots/speedup.png
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

def plot_speedup(path, out):
    # Read with pandas to tolerate header and stray commas (input contains execution times)
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
    
    # Calculate speedup (relative to 1 thread)
    baseline_time = times[0] if len(times) > 0 else 1.0
    speedup = baseline_time / times

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Execution time
    ax1.plot(threads, times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel("Threads", fontsize=11)
    ax1.set_ylabel("Execution time (s)", fontsize=11)
    ax1.set_title("Execution Time vs Threads", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)
    
    # Highlight minimum point
    min_idx = np.argmin(times)
    ax1.plot(threads[min_idx], times[min_idx], marker='o', markersize=12, 
             color='red', markeredgecolor='darkred', markeredgewidth=2, 
             label=f'Optimal: {threads[min_idx]} threads ({times[min_idx]:.2f}s)')
    ax1.legend(loc='best')
    
    # Plot 2: Speedup
    ax2.plot(threads, speedup, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1 thread)')
    ax2.set_xlabel("Threads", fontsize=11)
    ax2.set_ylabel("Speedup (relative to 1 thread)", fontsize=11)
    ax2.set_title("Speedup vs Threads", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads)
    
    # Highlight maximum speedup
    max_speedup_idx = np.argmax(speedup)
    ax2.plot(threads[max_speedup_idx], speedup[max_speedup_idx], marker='s', markersize=12,
             color='green', markeredgecolor='darkgreen', markeredgewidth=2,
             label=f'Max speedup: {speedup[max_speedup_idx]:.2f}x at {threads[max_speedup_idx]} threads')
    ax2.legend(loc='best')
    
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
    parser.add_argument("--out", default="results/plot.png", help="output image path (PNG)")
    opt = parser.parse_args()

    if not opt.history and not opt.speedup:
        parser.error("Specify either --history or --speedup")

    if opt.history and opt.speedup:
        parser.error("Specify only one of --history or --speedup")

    if opt.history:
        plot_history(opt.history, opt.out)
    else:
        plot_speedup(opt.speedup, opt.out)

if __name__ == "__main__":
    main()
