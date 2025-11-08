#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt, os
p=argparse.ArgumentParser()
p.add_argument("--history", required=True)
p.add_argument("--out", default="results/serial_plot.png")
args=p.parse_args()
hist = np.loadtxt(args.history)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.figure(figsize=(6,3))
plt.plot(hist, marker='o', markersize=3)
plt.xlabel("Timestep"); plt.ylabel("Average sentiment")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("Wrote", args.out)
