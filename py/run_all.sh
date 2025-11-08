#!/usr/bin/env bash
set -e
# assumes working dir = project root
python py/preprocess_sentiment.py --input data/raw/sentiment140.csv --out data/per_user_sentiment.csv
python py/convert_ego_to_graph.py --edges data/raw/12831.edges --ego-id 12831 --out-dir data --per-user data/per_user_sentiment.csv
mkdir -p results
python py/serial_sim.py --graph data/graph.txt --states data/states.txt --steps 50 --alpha 0.3 --out results/serial_history.txt
python py/plot.py --history results/serial_history.txt --out results/serial_plot.png
echo "Done. Outputs: data/graph.txt, data/states.txt, results/serial_history.txt, results/serial_plot.png"

