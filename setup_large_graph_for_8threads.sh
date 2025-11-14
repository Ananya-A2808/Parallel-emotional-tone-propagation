#!/usr/bin/env bash
# setup_large_graph_for_8threads.sh
# Generate a very large graph optimized for 8-thread parallel performance
# This script creates graphs with 20K-50K nodes to ensure good scalability

set -euo pipefail

echo "=========================================="
echo "Generating Large Graph for 8-Thread Testing"
echo "=========================================="
echo ""

# Default: 30K nodes (good balance between size and generation time)
NODES="${1:-30000}"
MODEL="${2:-social}"

echo "Configuration:"
echo "  Nodes: ${NODES}"
echo "  Model: ${MODEL}"
echo ""

# Generate the graph
echo "Step 1: Generating ${NODES}-node graph..."
python py/generate_large_graph.py \
    --nodes "${NODES}" \
    --model "${MODEL}" \
    --out data/raw/large_graph_${NODES}.edges \
    --states-out data/large_states_${NODES}.txt

echo ""
echo "Step 2: Building graph format..."
python py/build_graph.py \
    --edgelist data/raw/large_graph_${NODES}.edges \
    --per-user data/per_user_sentiment.csv \
    --out-dir data

echo ""
echo "=========================================="
echo "Graph Generation Complete!"
echo "=========================================="
echo ""
echo "Graph file: data/raw/large_graph_${NODES}.edges"
echo "Graph format: data/graph.txt"
echo "States: data/states.txt"
echo ""
echo "To use this graph, update run_all.sh:"
echo "  GRAPH_EDGELIST=\"data/raw/large_graph_${NODES}.edges\""
echo ""
echo "Recommended settings for 8-thread testing:"
echo "  STEPS=50000  (or higher for very large graphs)"
echo "  THREADS_LIST=(1 2 3 4 5 6 7 8)"
echo ""

