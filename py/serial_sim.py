#!/usr/bin/env python3
import argparse, numpy as np, os, sys, time

def read_graph(graph_path):
    with open(graph_path,'r') as f:
        N,M = map(int, f.readline().split())
        preds=[[] for _ in range(N)]
        for line in f:
            u,v = line.strip().split()
            u,v=int(u),int(v)
            preds[v].append(u)
    return N, preds

def read_states(states_path):
    s=[]
    with open(states_path,'r') as f:
        for line in f: s.append(float(line.strip()))
    return np.array(s, dtype=float)

def simulate(N,preds,states,steps,alpha,verbose=True):
    history=[]
    s=states.copy()
    start_time = time.time()
    print_interval = max(1, steps // 20)  # Print ~20 progress updates
    
    # Print initial status immediately
    if verbose:
        sys.stderr.write(f"[Serial] Starting {steps} steps on {N} nodes...\n")
        sys.stderr.flush()
    
    for t in range(steps):
        new=s.copy()
        for v in range(N):
            neigh = preds[v]
            if len(neigh)==0:
                new[v]=s[v]
            else:
                avg = sum(s[u] for u in neigh)/len(neigh)
                new[v] = (1-alpha)*s[v] + alpha*avg
        s=new
        history.append(float(s.mean()))
        
        # Progress logging - print first step immediately, then at intervals
        if verbose and (t == 0 or t % print_interval == 0 or t == steps - 1):
            elapsed = time.time() - start_time
            progress = 100.0 * (t + 1) / steps
            rate = (t + 1) / elapsed if elapsed > 0 else 0
            eta = (steps - t - 1) / rate if rate > 0 else 0
            avg_sentiment = history[-1]
            sys.stderr.write(f"\r[Serial] Step {t+1}/{steps} ({progress:.1f}%) | "
                           f"Avg sentiment: {avg_sentiment:.6f} | "
                           f"Elapsed: {elapsed:.1f}s | "
                           f"ETA: {eta:.1f}s | "
                           f"Rate: {rate:.0f} steps/s")
            sys.stderr.flush()
    
    if verbose:
        sys.stderr.write("\n")  # New line after progress
        total_time = time.time() - start_time
        print(f"[Serial] Completed {steps} steps in {total_time:.2f}s ({steps/total_time:.0f} steps/s)", file=sys.stderr)
    
    return s, history

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph", default="data/graph.txt")
    p.add_argument("--states", default="data/states.txt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--out", default="results/serial_history.txt")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args=p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    print(f"[Serial] Loading graph from {args.graph}...", file=sys.stderr)
    N,preds = read_graph(args.graph)
    print(f"[Serial] Graph loaded: {N} nodes, {sum(len(p) for p in preds)} edges", file=sys.stderr)
    
    print(f"[Serial] Loading states from {args.states}...", file=sys.stderr)
    states = read_states(args.states)
    if len(states)!=N:
        raise SystemExit("states length does not match N")
    
    print(f"[Serial] Starting simulation: {args.steps} steps, alpha={args.alpha}...", file=sys.stderr)
    final, history = simulate(N,preds,states,args.steps,args.alpha,verbose=not args.quiet)
    
    print(f"[Serial] Writing results...", file=sys.stderr)
    with open(args.out,'w') as f:
        for v in history: f.write(f"{v}\n")
    with open("results/serial_final_states.txt",'w') as f:
        for v in final: f.write(f"{v}\n")
    print(f"[Serial] Wrote history to {args.out} and results/serial_final_states.txt", file=sys.stderr)

if __name__=="__main__":
    main()
