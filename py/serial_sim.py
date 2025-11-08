#!/usr/bin/env python3
import argparse, numpy as np, os

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

def simulate(N,preds,states,steps,alpha):
    history=[]
    s=states.copy()
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
    return s, history

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph", default="data/graph.txt")
    p.add_argument("--states", default="data/states.txt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--out", default="results/serial_history.txt")
    args=p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    N,preds = read_graph(args.graph)
    states = read_states(args.states)
    if len(states)!=N:
        raise SystemExit("states length does not match N")
    final, history = simulate(N,preds,states,args.steps,args.alpha)
    with open(args.out,'w') as f:
        for v in history: f.write(f"{v}\n")
    with open("results/serial_final_states.txt",'w') as f:
        for v in final: f.write(f"{v}\n")
    print("Wrote history to", args.out, "and results/serial_final_states.txt")

if __name__=="__main__":
    main()
