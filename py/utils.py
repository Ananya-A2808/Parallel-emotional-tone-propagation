#!/usr/bin/env python3
"""
utils.py
--------
Utility functions for the Parallel Emotional Tone Propagation project.

Provides helpers for:
  - Reading and writing graph and state files
  - Initializing random emotional states
  - Computing average sentiment or convergence
  - Simple timer and progress output
"""

import os
import time
import numpy as np


# ---------------------------
# Basic File I/O utilities
# ---------------------------

def read_graph(path):
    """
    Reads a graph from a text file.
    Format:
        N M
        u v
        u v
        ...
    Returns adjacency list (list of lists of ints).
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split()
    n, m = int(header[0]), int(header[1])
    adj = [[] for _ in range(n)]
    for line in lines[1:]:
        u, v = map(int, line.split())
        adj[u].append(v)
    return adj


def write_graph(adj, path):
    """Writes a graph adjacency list to a text file."""
    edges = [(u, v) for u, nbrs in enumerate(adj) for v in nbrs]
    with open(path, "w") as f:
        f.write(f"{len(adj)} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


def read_states(path):
    """Reads one float per line (initial emotional states)."""
    with open(path, "r") as f:
        return np.array([float(x.strip()) for x in f if x.strip()])


def write_states(states, path):
    """Writes emotional state values to a text file."""
    with open(path, "w") as f:
        for val in states:
            f.write(f"{val}\n")


# ---------------------------
# Math helpers
# ---------------------------

def average_sentiment(states):
    """Returns the mean emotional value."""
    return float(np.mean(states))


def update_state_serial(adj, states, alpha=0.25):
    """
    Performs one synchronous update step:
    new_state[i] = (1 - alpha) * old[i] + alpha * avg(neighbor values)
    """
    n = len(states)
    new_states = np.zeros_like(states)
    for i in range(n):
        if not adj[i]:
            new_states[i] = states[i]
        else:
            neighbor_avg = np.mean([states[j] for j in adj[i]])
            new_states[i] = (1 - alpha) * states[i] + alpha * neighbor_avg
    return new_states


def has_converged(old, new, tol=1e-6):
    """Returns True if all states changed less than tolerance."""
    return np.allclose(old, new, atol=tol)


# ---------------------------
# Timer utility
# ---------------------------

class Timer:
    """Simple context manager for timing blocks of code."""
    def __init__(self, label=""):
        self.label = label
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.label:
            print(f"[Timer Start] {self.label} ...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if self.label:
            print(f"[Timer End] {self.label}: {elapsed:.3f} seconds")
        else:
            print(f"Elapsed time: {elapsed:.3f} seconds")
        return False


# ---------------------------
# Random state generation
# ---------------------------

def random_states(n, seed=42):
    """Generates n random emotional state values in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


# ---------------------------
# Diagnostics and progress
# ---------------------------

def print_progress(step, total, msg=""):
    """Simple console progress indicator."""
    pct = 100 * (step + 1) / total
    print(f"\rProgress: {pct:6.2f}% ({step+1}/{total}) {msg}", end="")
    if step + 1 == total:
        print()  # new line at end
