# 🧠 Parallel Emotional Tone Propagation

A simulation of emotional tone diffusion across social networks, implemented in **Python** (serial version) and **C++ with OpenMP** (parallel version).
This project demonstrates how emotional states (positive, negative, neutral) spread across a network using message diffusion models, comparing sequential and parallel performance.

---

## ⚙️ Setup

Clone the repository and install dependencies:

```bash
git clone <your_repo_url>
cd Parallel-emotional-tone-propagation
pip install -r requirements.txt
```

Make sure your system has:

* `g++` (with OpenMP support)
* `bash` (Git Bash or WSL on Windows)
* Python ≥ 3.8

---

## 🚀 Run the Project

To execute the full simulation pipeline:

```bash
bash run_all.sh
```

This script automatically performs:

1. (Optional) Sentiment preprocessing if raw data exists
2. Graph construction (`graph.txt`, `states.txt`)
3. Python serial simulation
4. C++ parallel simulation using OpenMP
5. Plot generation and experiment metadata creation

---

## 📊 Outputs

After running, results are saved in a timestamped folder:

```
results/run_<timestamp>/
├── inputs/
│   ├── graph.txt
│   ├── states.txt
│   ├── per_user_sentiment.csv
│   └── node_index.json
├── serial/
│   ├── serial_history.txt
│   └── serial_final_states.txt
├── parallel/
│   ├── history_<threads>.txt
│   ├── out_states_<threads>.txt
│   └── speedup.csv
├── plots/
│   ├── serial_history.png
│   ├── speedup.png
│   └── speedup_cleaned.csv
├── logs/
│   └── parallel_threads_<threads>.log
└── experiment_metadata.json
```

**Key visualizations:**

* `serial_history.png` → Emotional tone diffusion over time
* `speedup.png` → Parallel performance (speedup vs threads)

---

## 🧪 Run Tests

To verify correctness between the serial and parallel implementations:

```bash
pytest -q
```

Expected output:

```
1 passed in 1.2s
```

---

## ⚙️ Configuration

Edit the top section of `run_all.sh` to change parameters:

```bash
THREADS_LIST=(1 2 4 8 16)
STEPS=100
ALPHA=0.25
```

These control thread counts, diffusion steps, and the tone propagation coefficient respectively.

---

## 🧩 Project Components

* **`py/serial_sim.py`** — Serial emotional tone propagation
* **`cpp/parallel_update.cpp`** — Parallelized diffusion using OpenMP
* **`py/run_parallel_from_python.py`** — Python bridge to run the C++ binary
* **`py/plot.py`** — Generates diffusion and speedup plots
* **`py/build_graph.py` / `py/convert_ego_to_graph.py`** — Graph and state preparation
* **`run_all.sh`** — Automates the complete pipeline end-to-end

---

## 👩‍💻 Authors

* **Ananya A.**
* **Shree Lakshmi**
---

