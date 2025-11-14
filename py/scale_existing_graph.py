#!/usr/bin/env python3
"""
scale_existing_graph.py

Scale up an existing graph by creating multiple copies and connecting them.
This preserves the structure while creating a larger graph for parallel testing.

Usage:
    python py/scale_existing_graph.py --input data/raw/12831.edges --copies 10 --out data/raw/scaled_graph.edges
"""
import argparse
import networkx as nx
import numpy as np
import os

def read_edgelist(path):
    """Read edgelist file"""
    G = nx.DiGraph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                try:
                    G.add_edge(int(u), int(v))
                except ValueError:
                    G.add_edge(u, v)
    return G

def scale_graph(G, copies, inter_connect_prob=0.01, seed=42):
    """
    Create multiple copies of the graph and connect them
    
    Args:
        G: Input graph
        copies: Number of copies to make
        inter_connect_prob: Probability of connecting nodes across copies
        seed: Random seed
    """
    np.random.seed(seed)
    original_nodes = list(G.nodes())
    original_edges = list(G.edges())
    max_orig_id = max(int(n) for n in original_nodes if str(n).isdigit()) if original_nodes else 0
    
    scaled_G = nx.DiGraph()
    node_mapping = {}  # (copy_id, orig_node) -> new_node_id
    
    # Create copies
    for copy_id in range(copies):
        offset = copy_id * (max_orig_id + 1)
        for orig_node in original_nodes:
            if isinstance(orig_node, (int, np.integer)):
                new_node = orig_node + offset
            else:
                new_node = f"{orig_node}_copy{copy_id}"
            node_mapping[(copy_id, orig_node)] = new_node
            scaled_G.add_node(new_node)
        
        # Add edges within this copy
        for u, v in original_edges:
            new_u = node_mapping[(copy_id, u)]
            new_v = node_mapping[(copy_id, v)]
            scaled_G.add_edge(new_u, new_v)
    
    # Add inter-copy connections (simulate cross-community influence)
    if copies > 1 and inter_connect_prob > 0:
        all_nodes = list(scaled_G.nodes())
        num_inter_edges = int(len(all_nodes) * inter_connect_prob * copies)
        
        for _ in range(num_inter_edges):
            u = np.random.choice(all_nodes)
            v = np.random.choice(all_nodes)
            if u != v and not scaled_G.has_edge(u, v):
                scaled_G.add_edge(u, v)
    
    return scaled_G

def write_edgelist(G, output_path):
    """Write graph as edgelist"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    return G.number_of_edges()

def main():
    parser = argparse.ArgumentParser(description="Scale up an existing graph")
    parser.add_argument("--input", required=True, help="Input edgelist file")
    parser.add_argument("--copies", type=int, default=10, help="Number of copies to make")
    parser.add_argument("--inter-connect", type=float, default=0.01, 
                       help="Probability of inter-copy connections")
    parser.add_argument("--out", default="data/raw/scaled_graph.edges", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print(f"Reading graph from: {args.input}")
    G = read_edgelist(args.input)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"Scaling by {args.copies}x...")
    scaled_G = scale_graph(G, args.copies, args.inter_connect, args.seed)
    
    num_edges = write_edgelist(scaled_G, args.out)
    print(f"Scaled graph: {scaled_G.number_of_nodes()} nodes, {num_edges} edges")
    print(f"Wrote to: {args.out}")
    
    print("\nTo use this graph:")
    print(f"  python py/build_graph.py --edgelist {args.out} --per-user data/per_user_sentiment.csv --out-dir data")

if __name__ == "__main__":
    main()

