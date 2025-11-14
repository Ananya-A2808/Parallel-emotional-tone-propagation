#!/usr/bin/env python3
"""
generate_large_graph.py

Generate larger synthetic social network graphs for testing parallel performance.
Creates graphs with configurable number of nodes and edges using various models:
- Barabasi-Albert (scale-free, like real social networks)
- Erdos-Renyi (random)
- Watts-Strogatz (small-world)

Usage:
    python py/generate_large_graph.py --nodes 10000 --model barabasi --out data/raw/large_graph.edges
"""
import argparse
import numpy as np
import networkx as nx
import os

def generate_barabasi_albert(nodes, edges_per_node=5, seed=42):
    """Generate Barabasi-Albert scale-free graph (like real social networks)"""
    np.random.seed(seed)
    # BA model: start with m0 nodes, add nodes with m edges each
    m = max(1, edges_per_node // 2)
    G = nx.barabasi_albert_graph(nodes, m, seed=seed)
    return G

def generate_erdos_renyi(nodes, p=None, seed=42):
    """Generate Erdos-Renyi random graph"""
    np.random.seed(seed)
    if p is None:
        # Default: average degree ~10
        p = 10.0 / nodes
    G = nx.erdos_renyi_graph(nodes, p, seed=seed, directed=False)
    return G

def generate_watts_strogatz(nodes, k=10, p=0.1, seed=42):
    """Generate Watts-Strogatz small-world graph"""
    np.random.seed(seed)
    G = nx.watts_strogatz_graph(nodes, k, p, seed=seed)
    return G

def generate_social_like(nodes, avg_degree=10, seed=42):
    """Generate a more realistic social network: combination of models"""
    np.random.seed(seed)
    # Start with BA for hub structure
    G1 = generate_barabasi_albert(nodes, avg_degree // 2, seed=seed)
    # Add some random connections
    G2 = generate_erdos_renyi(nodes, p=avg_degree/(2*nodes), seed=seed+1)
    # Combine
    G = nx.compose(G1, G2)
    return G

def write_edgelist(G, output_path, directed=False):
    """Write graph as edgelist file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for u, v in G.edges():
            if directed:
                f.write(f"{u} {v}\n")
            else:
                # For undirected, write both directions to simulate bidirectional influence
                f.write(f"{u} {v}\n")
                f.write(f"{v} {u}\n")
    return G.number_of_edges() * (2 if not directed else 1)

def generate_states(nodes, output_path, seed=42):
    """Generate random initial emotional states"""
    np.random.seed(seed)
    # Generate states in range [-1, 1] with some bias toward neutral
    states = np.random.normal(0.0, 0.5, nodes)
    states = np.clip(states, -1.0, 1.0)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in states:
            f.write(f"{s:.6f}\n")
    return states

def main():
    parser = argparse.ArgumentParser(description="Generate large synthetic social network graphs")
    parser.add_argument("--nodes", type=int, default=10000, help="Number of nodes")
    parser.add_argument("--model", choices=['barabasi', 'erdos', 'watts', 'social'], 
                       default='social', help="Graph model to use")
    parser.add_argument("--out", default="data/raw/large_graph.edges", help="Output edgelist path")
    parser.add_argument("--states-out", default="data/large_states.txt", help="Output states path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--avg-degree", type=int, default=10, help="Average node degree (for some models)")
    args = parser.parse_args()
    
    print(f"Generating {args.nodes}-node graph using {args.model} model...")
    
    if args.model == 'barabasi':
        G = generate_barabasi_albert(args.nodes, args.avg_degree, args.seed)
    elif args.model == 'erdos':
        G = generate_erdos_renyi(args.nodes, seed=args.seed)
    elif args.model == 'watts':
        G = generate_watts_strogatz(args.nodes, k=args.avg_degree, seed=args.seed)
    else:  # social
        G = generate_social_like(args.nodes, args.avg_degree, args.seed)
    
    num_edges = write_edgelist(G, args.out, directed=False)
    print(f"Generated graph: {args.nodes} nodes, {num_edges} edges")
    print(f"Average degree: {2*num_edges/args.nodes:.2f}")
    print(f"Wrote edgelist to: {args.out}")
    
    # Generate states
    generate_states(args.nodes, args.states_out, args.seed)
    print(f"Generated initial states: {args.states_out}")
    
    print("\nTo use this graph, run:")
    print(f"  python py/build_graph.py --edgelist {args.out} --per-user data/per_user_sentiment.csv --out-dir data")

if __name__ == "__main__":
    main()

