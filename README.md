# 🧠 Parallel Emotional Tone Propagation in Social Networks

A simulation of emotional tone diffusion across social networks, implemented in **Python** (serial version) and **C++ with OpenMP** (parallel version). This project demonstrates how emotional states (positive, negative, neutral) spread across a network using message diffusion models, comparing sequential and parallel performance.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Understanding Results](#-understanding-results)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)

---

## 🔧 Prerequisites

Before running this project, ensure you have:

1. **Python 3.8+** installed
   ```bash
   python --version
   ```

2. **g++ compiler with OpenMP support**
   ```bash
   g++ --version
   # On Windows, install MinGW or use WSL
   ```

3. **Bash shell** (Git Bash or WSL on Windows)

4. **Required Python packages** (will be installed in next step)

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Parallel-emotional-tone-propagation
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `networkx` - Graph operations

### Step 3: Verify Installation

```bash
python -c "import networkx, pandas, numpy, matplotlib; print('All packages installed!')"
```

---

## 🚀 Quick Start

### Option 1: Use Existing Graph (Fastest)

If you already have a graph file:

```bash
# 1. Set your graph path in run_all.sh (line 10)
# Edit: GRAPH_EDGELIST="data/raw/your_graph.edges"

# 2. Run the complete pipeline
bash run_all.sh
```

### Option 2: Generate a Large Graph (Recommended for Performance Testing)

```bash
# 1. Generate a 10,000 node graph
python py/generate_large_graph.py --nodes 10000 --model social --out data/raw/large_graph.edges

# 2. Update run_all.sh to use the new graph
# Edit line 10: GRAPH_EDGELIST="data/raw/large_graph.edges"

# 3. Run the pipeline
bash run_all.sh
```

### Option 3: Scale Existing Graph

```bash
# 1. Scale your existing graph by 20x
python py/scale_existing_graph.py --input data/raw/12831.edges --copies 20 --out data/raw/scaled_graph.edges

# 2. Update run_all.sh
# Edit line 10: GRAPH_EDGELIST="data/raw/scaled_graph.edges"

# 3. Run the pipeline
bash run_all.sh
```

---

## 📖 Step-by-Step Guide

### Step 1: Prepare Your Graph Data

#### If you have a raw edgelist file:

```bash
# Build graph from edgelist
python py/build_graph.py --edgelist data/raw/your_graph.edges --per-user data/per_user_sentiment.csv --out-dir data
```

This creates:
- `data/graph.txt` - Graph in required format
- `data/states.txt` - Initial emotional states
- `data/node_index.json` - Node mapping

#### If you need to generate a graph:

```bash
# Generate synthetic graph (10K nodes recommended)
python py/generate_large_graph.py --nodes 10000 --model social --out data/raw/large_graph.edges

# Then build it
python py/build_graph.py --edgelist data/raw/large_graph.edges --per-user data/per_user_sentiment.csv --out-dir data
```

### Step 2: Configure Parameters (Optional)

Edit `run_all.sh` to customize:

```bash
# Line 8: Number of simulation steps
STEPS=50000

# Line 9: Propagation coefficient (0.0 to 1.0)
ALPHA=0.25

# Line 10: Graph file path
GRAPH_EDGELIST="data/raw/large_graph.edges"

# Line 47: Thread counts to test (default: 1-8)
THREADS_LIST=(1 2 3 4 5 6 7 8)
```

### Step 3: Build C++ Binary

```bash
bash cpp/build.sh
```

This compiles the parallel OpenMP implementation.

### Step 4: Run the Complete Pipeline

```bash
bash run_all.sh
```

This automatically:
1. ✅ Builds/loads the graph
2. ✅ Runs serial simulation (optional, can be skipped)
3. ✅ Builds C++ binary
4. ✅ Runs parallel simulations with different thread counts
5. ✅ Generates plots and analysis

### Step 5: View Results

Results are saved in `results/run_<timestamp>/`:

```bash
# View execution times
cat results/run_*/execution_time.csv

# View plots
# Open: results/run_*/plots/execution_time.png
```

---

## 📊 Understanding Results

### Output Structure

```
results/run_<timestamp>/
├── inputs/                    # Input files used
│   ├── graph.txt
│   ├── states.txt
│   └── per_user_sentiment.csv
├── serial/                    # Serial simulation results (if run)
│   ├── serial_history.txt
│   └── serial_final_states.txt
├── parallel/                  # Parallel simulation results
│   ├── history_1.txt          # History for 1 thread
│   ├── history_2.txt          # History for 2 threads
│   ├── ...
│   ├── out_states_<N>.txt    # Final states for N threads
│   └── execution_time.csv    # Performance data
├── plots/                     # Generated visualizations
│   ├── execution_time.png    # Main performance plot
│   ├── execution_time_speedup.csv
│   └── execution_time_cleaned.csv
├── logs/                      # Execution logs
│   └── parallel_threads_<N>_run<M>.log
└── experiment_metadata.json  # Run metadata
```

### Key Files

1. **`execution_time.csv`** - Performance data:
   ```csv
   threads,time
   1,5.45
   2,4.24
   3,3.95
   4,3.80
   ...
   ```

2. **`execution_time.png`** - Dual plot showing:
   - **Left**: Execution time vs threads (lower is better)
   - **Right**: Speedup vs threads (higher is better)
   - **Red marker**: Optimal thread count
   - **Green marker**: Maximum speedup

3. **`experiment_metadata.json`** - Complete run information

### Interpreting Results

- **Optimal thread count**: The point with minimum execution time (usually 3-4 threads)
- **Speedup**: How much faster than single-threaded (e.g., 1.43x = 43% faster)
- **Performance degradation**: When more threads = slower (due to memory bandwidth saturation)

---

## ⚙️ Configuration

### Main Configuration (`run_all.sh`)

```bash
# Simulation parameters
STEPS=50000              # Number of diffusion steps
ALPHA=0.25              # Propagation coefficient (0.0-1.0)
GRAPH_EDGELIST="data/raw/large_graph.edges"  # Graph file path

# Thread testing
THREADS_LIST=(1 2 3 4 5 6 7 8)  # Thread counts to test
```

### Graph Generation Options

```bash
# Generate different graph sizes
python py/generate_large_graph.py --nodes 5000 --model social --out data/raw/graph_5k.edges
python py/generate_large_graph.py --nodes 10000 --model social --out data/raw/graph_10k.edges
python py/generate_large_graph.py --nodes 50000 --model social --out data/raw/graph_50k.edges

# Different graph models
python py/generate_large_graph.py --nodes 10000 --model barabasi --out data/raw/ba_graph.edges
python py/generate_large_graph.py --nodes 10000 --model erdos --out data/raw/er_graph.edges
python py/generate_large_graph.py --nodes 10000 --model watts --out data/raw/ws_graph.edges
```

---

## 🔍 Troubleshooting

### Problem: "Execution time not decreasing with more threads"

**Solution**: Your graph is too small. Generate a larger graph:

```bash
python py/generate_large_graph.py --nodes 10000 --model social --out data/raw/large_graph.edges
python py/build_graph.py --edgelist data/raw/large_graph.edges --per-user data/per_user_sentiment.csv --out-dir data
```

**Why?** Small graphs (< 1000 nodes) have parallelization overhead that exceeds computation time.

### Problem: "C++ binary not found"

**Solution**: Rebuild the binary:

```bash
bash cpp/build.sh
```

On Windows, the binary is `cpp/bin/parallel_update.exe`.

### Problem: "Python module not found"

**Solution**: Install dependencies:

```bash
pip install -r requirements.txt
```

### Problem: "Graph file not found"

**Solution**: Check the path in `run_all.sh` line 10, or generate a graph:

```bash
python py/generate_large_graph.py --nodes 10000 --out data/raw/large_graph.edges
```

### Problem: "Serial simulation taking too long"

**Solution**: Skip serial simulation by commenting it out in `run_all.sh` (already done by default), or reduce STEPS:

```bash
STEPS=1000  # Instead of 50000
```

---

## 📁 Project Structure

```
Parallel-emotional-tone-propagation/
├── cpp/                          # C++ parallel implementation
│   ├── parallel_update.cpp      # Main parallel code
│   ├── build.sh                 # Build script
│   └── bin/                     # Compiled binaries (gitignored)
├── py/                          # Python scripts
│   ├── serial_sim.py           # Serial simulation
│   ├── build_graph.py          # Graph construction
│   ├── generate_large_graph.py # Graph generation
│   ├── scale_existing_graph.py # Graph scaling
│   └── plot.py                 # Visualization
├── data/                        # Data files
│   ├── raw/                     # Raw graph files
│   └── *.txt, *.csv            # Processed data (gitignored)
├── results/                     # Output results (gitignored)
├── run_all.sh                   # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🧩 Key Components

| Component | Description |
|-----------|-------------|
| `py/serial_sim.py` | Serial emotional tone propagation (baseline) |
| `cpp/parallel_update.cpp` | Parallelized diffusion using OpenMP |
| `py/generate_large_graph.py` | Generate synthetic social network graphs |
| `py/scale_existing_graph.py` | Scale existing graphs to larger sizes |
| `py/plot.py` | Generate execution time and speedup plots |
| `py/build_graph.py` | Convert raw edgelist to required format |
| `run_all.sh` | Complete automated pipeline |

---

## 📈 Performance Tips

1. **Use large graphs** (10,000+ nodes) for meaningful speedup
2. **Optimal thread count** is usually 3-4 threads (test to find yours)
3. **More threads ≠ faster** - memory bandwidth limits performance
4. **Increase STEPS** (50,000+) to make parallel work dominate overhead

Performance analysis: Optimal thread count is usually 3-4 threads. Beyond that, memory bandwidth saturation causes performance degradation.

---

## 👩‍💻 Authors

* **Ananya A.**
* **Shree Lakshmi**

---

## 🙏 Acknowledgments

This project demonstrates parallel graph algorithms and emotional diffusion models in social networks.
