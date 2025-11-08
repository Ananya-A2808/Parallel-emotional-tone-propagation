#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd

def read_edges(path):
    edges=[]; nodes=set()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts)>=2:
                u,v=parts[0],parts[1]
                edges.append((u,v)); nodes.add(u); nodes.add(v)
    return edges, nodes

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--ego-id", default="EGO")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--per-user", default="data/per_user_sentiment.csv")
    p.add_argument("--seed", type=int, default=123)
    args=p.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    edges_raw, nodes_seen = read_edges(args.edges)
    alters = sorted(nodes_seen, key=lambda x: int(x) if x.isdigit() else x)
    ego = str(args.ego_id)
    if ego in alters: alters=[a for a in alters if a!=ego]
    nodes = [ego] + alters
    idx = {nid:i for i,nid in enumerate(nodes)}

    edges=[]
    for u,v in edges_raw:
        if u in idx and v in idx:
            edges.append((idx[u], idx[v]))
    for a in alters:
        edges.append((idx[ego], idx[a]))

    N=len(nodes)
    states = np.zeros(N, dtype=float)

    if os.path.exists(args.per_user):
        df = pd.read_csv(args.per_user, dtype={'user_id':str})
        sentiments = df['sentiment'].astype(float).values
        if len(sentiments)>0:
            states = np.random.choice(sentiments, size=N, replace=True)
            print("Assigned sentiments randomly from Sentiment140 distribution.")
    else:
        print("No per_user_sentiment.csv found; using neutral states (0.0).")

    gpath=os.path.join(args.out_dir,"graph.txt")
    with open(gpath,'w',encoding='utf-8') as gf:
        gf.write(f"{N} {len(edges)}\n")
        for u,v in edges: gf.write(f"{u} {v}\n")
    spath=os.path.join(args.out_dir,"states.txt")
    with open(spath,'w',encoding='utf-8') as sf:
        for v in states: sf.write(f"{float(v)}\n")
    with open("py/node_index.json","w",encoding='utf-8') as jf:
        json.dump(idx,jf)
    print(f"Wrote {gpath} ({N} nodes, {len(edges)} edges) and {spath} and py/node_index.json")

if __name__=="__main__":
    main()
