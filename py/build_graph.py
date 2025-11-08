#!/usr/bin/env python3
"""
build_graph.py
Inputs:
  - raw edgelist file (u v per line) using original user ids/handles
  - per_user_sentiment.csv with columns: user_id,sentiment
Outputs:
  - data/graph.txt (first line: N M, then M lines "u v" where u/v are 0-indexed integers)
  - data/states.txt (N lines, float per node in same node ordering)
  - py/node_index.json (mapping original_id -> index)
"""
import argparse, json
import networkx as nx
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edgelist", required=True, help="Raw edgelist (u v per line) using original user ids")
    p.add_argument("--per-user", required=True, help="CSV user_id,sentiment")
    p.add_argument("--out-dir", default="data")
    opt = p.parse_args()

    # load edgelist as strings (some ids may be numeric or text handles)
    G = nx.DiGraph()
    print("Reading edgelist:", opt.edgelist)
    with open(opt.edgelist, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            u, v = parts[0], parts[1]
            G.add_edge(u, v)
    nodes = list(G.nodes())
    N = len(nodes)
    print("Nodes:", N, "Edges:", G.number_of_edges())

    # build index map
    idx = {node: i for i,node in enumerate(nodes)}
    # write graph.txt
    edges = [(idx[u], idx[v]) for u,v in G.edges()]
    with open(f"{opt.out_dir}/graph.txt", "w", encoding='utf-8') as gf:
        gf.write(f"{N} {len(edges)}\n")
        for u,v in edges:
            gf.write(f"{u} {v}\n")
    print("Wrote", f"{opt.out_dir}/graph.txt")

    # load per-user sentiment
    df = pd.read_csv(opt.per_user, dtype={'user_id': str})
    sentiment_map = dict(zip(df['user_id'].astype(str), df['sentiment'].astype(float)))
    # write states in node order, default 0.0
    with open(f"{opt.out_dir}/states.txt", "w", encoding='utf-8') as sf:
        for node in nodes:
            s = sentiment_map.get(str(node), 0.0)
            sf.write(f"{float(s)}\n")
    print("Wrote", f"{opt.out_dir}/states.txt")

    # save mapping
    with open("data/node_index.json", "w", encoding='utf-8') as jf:
        json.dump(idx, jf)
    print("Wrote data/node_index.json (original_id -> index).")

if __name__ == "__main__":
    main()
