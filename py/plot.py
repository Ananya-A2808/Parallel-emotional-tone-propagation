#!/usr/bin/env python3
"""
plot.py

Supports:
  --history <history.txt>   (one numeric value per line)
  --speedup <speedup.csv>   (threads,time) or (threads,time,anything) ; header allowed
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
    # Read with pandas to tolerate header and stray commas
    try:
        # try common delimiters
        for sep in [',',';',None]:
            try:
                df = pd.read_csv(path, sep=sep, engine='python', comment='#', skip_blank_lines=True)
                # if result has 0 or 1 cols, keep trying
                if df.shape[1] >= 2:
                    break
            except Exception:
                df = None
        if df is None or df.shape[1] < 2:
            # fallback: try whitespace split
            df = pd.read_csv(path, delim_whitespace=True, header=None, engine='python')
    except Exception as e:
        print("Failed to read CSV with pandas:", e, file=sys.stderr)
        raise

    # drop fully-empty columns
    df = df.dropna(axis=1, how='all')
    # drop rows with all NaN
    df = df.dropna(axis=0, how='all')
    if df.shape[1] < 2:
        raise SystemExit("speedup file must contain at least two columns (threads,time). Found:\n" + str(df.head(10)))

    # assume first two numeric columns are threads and time
    # convert columns to numeric where possible
    df_cols = df.columns.tolist()
    # try to coerce first numeric-like pair
    numeric_cols = []
    for col in df_cols:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notna().sum() > 0:
            numeric_cols.append(col)
        if len(numeric_cols) >= 2:
            break
    if len(numeric_cols) < 2:
        raise SystemExit("Couldn't find two numeric columns in speedup CSV. Inspect file.")

    threads_col, time_col = numeric_cols[0], numeric_cols[1]
    df2 = df[[threads_col, time_col]].copy()
    df2[threads_col] = pd.to_numeric(df2[threads_col], errors='coerce')
    df2[time_col] = pd.to_numeric(df2[time_col], errors='coerce')
    df2 = df2.dropna(subset=[threads_col, time_col])

    # ensure sort by threads
    df2 = df2.sort_values(by=threads_col)
    threads = df2[threads_col].to_numpy()
    times = df2[time_col].to_numpy()

    # base time: time where threads is 1 if exists, else first entry
    base_idx = np.where(threads == 1)[0]
    if base_idx.size > 0:
        base_time = float(times[base_idx[0]])
    else:
        base_time = float(times[0])
        print("Warning: no threads==1 found; using first row as base for speedup calculation", file=sys.stderr)

    speedup = base_time / times
    plt.figure(figsize=(6,3.5))
    plt.plot(threads, speedup, marker='o')
    plt.xlabel("Threads")
    plt.ylabel("Speedup (T1 / Tn)")
    plt.title("Parallel speedup")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print("Wrote", out)
    # also write cleaned CSV used
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
