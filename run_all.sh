#!/usr/bin/env bash
# run_all.sh  — full reproducible pipeline (safe when no Sentiment140 present)
set -euo pipefail
IFS=$'\n\t'

# ---------- CONFIG ----------
THREADS_LIST=(1 2 4 8 16)
STEPS=100
ALPHA=0.25
GRAPH_EDGELIST="data/raw/12831.edges"   # set your ego edgelist path
RAW_TWEETS=""                           # leave empty if you don't have sentiment140
PER_USER="data/per_user_sentiment.csv"  # optional aggregated user sentiment
OUT_ROOT="results"
PYTHON="${PYTHON:-python}"
# ------------------------------

timestamp() { date +"%Y%m%d-%H%M%S"; }
RUN_TS=$(timestamp)
OUTDIR="${OUT_ROOT}/run_${RUN_TS}"
mkdir -p "${OUTDIR}"
echo "Running pipeline -> ${OUTDIR}"
echo "$0 $*" > "${OUTDIR}/run_command.txt"

# Check python
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python not found as '${PYTHON}'. Set PYTHON env or install." >&2
  exit 1
fi

# Step 0: Optional preprocessing (only if RAW_TWEETS path provided and file exists)
if [ -n "${RAW_TWEETS}" ] && [ -f "${RAW_TWEETS}" ] && [ ! -f "${PER_USER}" ]; then
  echo "Preprocessing tweets -> ${PER_USER}"
  mkdir -p data
  "${PYTHON}" py/preprocess_sentiment.py --input "${RAW_TWEETS}" --out "${PER_USER}"
else
  if [ -n "${RAW_TWEETS}" ]; then
    if [ ! -f "${RAW_TWEETS}" ]; then
      echo "RAW_TWEETS set but file not found at ${RAW_TWEETS}; skipping preprocess."
    else
      echo "PER_USER already exists at ${PER_USER}; skipping preprocess."
    fi
  else
    echo "No RAW_TWEETS specified — skipping sentiment preprocessing."
  fi
fi

# Step 1: Build graph & states. Use convert_ego_to_graph.py for ego networks.
if [ -f data/graph.txt ] && [ -f data/states.txt ]; then
  echo "Using existing data/graph.txt and data/states.txt"
else
  if [ -f "${GRAPH_EDGELIST}" ]; then
    echo "Building graph & states from ego edges: ${GRAPH_EDGELIST}"
    mkdir -p data
    if [ -f "py/convert_ego_to_graph.py" ]; then
      # convert_ego_to_graph.py will attempt to use per_user_sentiment.csv if present;
      # otherwise it fills states with neutral or sampled values (see below).
      "${PYTHON}" py/convert_ego_to_graph.py --edges "${GRAPH_EDGELIST}" --ego-id 12831 --out-dir data --per-user "${PER_USER}"
    else
      echo "convert_ego_to_graph.py missing; trying generic build_graph.py"
      "${PYTHON}" py/build_graph.py --edgelist "${GRAPH_EDGELIST}" --per-user "${PER_USER}" --out-dir data
    fi
  else
    echo "ERROR: edgelist missing at ${GRAPH_EDGELIST} and data/graph.txt not present. Aborting." >&2
    exit 1
  fi
fi

# If per-user sentiment is missing and states.txt still not found, create neutral states aligned with graph
if [ ! -f data/states.txt ]; then
  if [ -f data/graph.txt ]; then
    echo "data/states.txt missing; creating neutral states (0.0) aligned with data/graph.txt"
    N=$(awk 'NR==1{print $1}' data/graph.txt)
    mkdir -p data
    python - <<PY
N = int(${N})
with open("data/states.txt","w") as f:
    for _ in range(N):
        f.write("0.0\n")
print("Wrote data/states.txt with", N, "neutral values.")
PY
  else
    echo "ERROR: cannot create states.txt because data/graph.txt is missing." >&2
    exit 1
  fi
fi

# copy inputs for provenance
mkdir -p "${OUTDIR}/inputs"
cp -v data/graph.txt data/states.txt "${OUTDIR}/inputs/" || true
[ -f "${PER_USER}" ] && cp -v "${PER_USER}" "${OUTDIR}/inputs/" || true
[ -f data/node_index.json ] && cp -v data/node_index.json "${OUTDIR}/inputs/" || true

# Step 2: Serial baseline
mkdir -p "${OUTDIR}/serial"
echo "Running serial baseline..."
"${PYTHON}" py/serial_sim.py --graph data/graph.txt --states data/states.txt --steps "${STEPS}" --alpha "${ALPHA}" --out "${OUTDIR}/serial/serial_history.txt"

# capture serial final states if available
if [ -f "results/serial_final_states.txt" ]; then
  cp -v results/serial_final_states.txt "${OUTDIR}/serial/serial_final_states.txt" || true
fi

# Step 3: build C++ binary
echo "Building C++ binary..."
bash cpp/build.sh
if [ -x "./cpp/bin/parallel_update" ]; then
  CPP_BIN="./cpp/bin/parallel_update"
elif [ -x "./cpp/parallel_update" ]; then
  CPP_BIN="./cpp/parallel_update"
else
  echo "C++ binary not found after build. Aborting." >&2
  exit 1
fi
echo "Using C++ binary: ${CPP_BIN}"

# Step 4: parallel runs
mkdir -p "${OUTDIR}/parallel" "${OUTDIR}/logs"
CSV="${OUTDIR}/parallel/speedup.csv"
echo "threads,time" > "${CSV}"
echo "Running parallel experiments: steps=${STEPS}, alpha=${ALPHA}"
for T in "${THREADS_LIST[@]}"; do
  echo " -> threads=${T}"
  LOG="${OUTDIR}/logs/parallel_threads_${T}.log"
  OUT_HISTORY="${OUTDIR}/parallel/history_${T}.txt"
  OUT_STATES="${OUTDIR}/parallel/out_states_${T}.txt"
  START=$("${PYTHON}" - <<PY
import time; print(time.time())
PY
)
  "${CPP_BIN}" data/graph.txt data/states.txt "${OUT_STATES}" "${OUT_HISTORY}" "${STEPS}" "${ALPHA}" "${T}" &> "${LOG}" || {
    echo "Parallel run failed for threads=${T}. See ${LOG}" >&2
    exit 1
  }
  END=$("${PYTHON}" - <<PY
import time; print(time.time())
PY
)
  ELAPSED=$("${PYTHON}" - <<PY
print(float(${END}) - float(${START}))
PY
)
  echo "${T},${ELAPSED}" >> "${CSV}"
  echo "Recorded: ${T},${ELAPSED}"
done

cp -v "${CSV}" "${OUTDIR}/speedup.csv"

# Step 5: copy last out_states (last thread)
LAST_INDEX=$(( ${#THREADS_LIST[@]} - 1 ))
LAST_T="${THREADS_LIST[$LAST_INDEX]}"
if [ -f "${OUTDIR}/parallel/out_states_${LAST_T}.txt" ]; then
  cp -v "${OUTDIR}/parallel/out_states_${LAST_T}.txt" "${OUTDIR}/parallel/out_states_last.txt"
fi

# Step 6: plots
mkdir -p "${OUTDIR}/plots"
echo "Generating plots..."
"${PYTHON}" py/plot.py --history "${OUTDIR}/serial/serial_history.txt" --out "${OUTDIR}/plots/serial_history.png"
"${PYTHON}" py/plot.py --speedup "${OUTDIR}/speedup.csv" --out "${OUTDIR}/plots/speedup.png"
cp -v "${OUTDIR}/parallel/history_"*.txt "${OUTDIR}/plots/" 2>/dev/null || true

# Step 7: metadata — create JSON safely (uses $PYTHON and serializes THREADS_LIST)
METAFILE="${OUTDIR}/experiment_metadata.json"
THREADS_JSON=$("${PYTHON}" - <<PY
import json,sys
arr = [${THREADS_LIST[@]}]
print(json.dumps(arr))
PY
)

"${PYTHON}" - <<PY
import json, platform, subprocess, os, multiprocessing
outdir = "${OUTDIR}"
meta = {}
meta['timestamp'] = "${RUN_TS}"
meta['steps'] = ${STEPS}
meta['alpha'] = ${ALPHA}
meta['threads_list'] = ${THREADS_JSON}
try:
    meta['git_commit'] = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception:
    meta['git_commit'] = None
meta['python_version'] = platform.python_version()
try:
    gpp = subprocess.check_output(['g++','--version']).decode().splitlines()[0]
except Exception:
    gpp = None
meta['gpp'] = gpp
try:
    meta['cpu_count_logical'] = multiprocessing.cpu_count()
except Exception:
    meta['cpu_count_logical'] = None
meta['speedup_csv'] = os.path.abspath(os.path.join(outdir, "speedup.csv"))
meta['plots'] = {
    'serial': os.path.abspath(os.path.join(outdir, "plots", "serial_history.png")),
    'speedup': os.path.abspath(os.path.join(outdir, "plots", "speedup.png"))
}
meta['notes'] = "Run produced by run_all.sh"
with open(os.path.join(outdir, "experiment_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote metadata to", os.path.join(outdir, "experiment_metadata.json"))
PY


# Summary
echo "-----------------------------"
echo "Run complete: ${OUTDIR}"
echo "Key outputs:"
echo " - Serial history: ${OUTDIR}/serial/serial_history.txt"
echo " - Parallel histories: ${OUTDIR}/parallel/history_*.txt"
echo " - Speedup CSV: ${OUTDIR}/speedup.csv"
echo " - Plots: ${OUTDIR}/plots/*.png"
echo " - Logs: ${OUTDIR}/logs/*.log"
echo " - Metadata: ${OUTDIR}/experiment_metadata.json"
echo "-----------------------------"

