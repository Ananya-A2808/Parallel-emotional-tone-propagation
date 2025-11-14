#!/usr/bin/env bash
# run_all.sh  — full reproducible pipeline (safe when no Sentiment140 present)
set -euo pipefail
IFS=$'\n\t'

# ---------- CONFIG ----------
# THREADS_LIST=(1 2 4 8 16)   # replaced below by auto-generated list
STEPS=50000            # increased significantly so parallel work dominates overhead
ALPHA=0.25
GRAPH_EDGELIST="data/raw/large_graph.edges"  # set your ego edgelist path
# For better parallel performance, use larger graphs:
# - Generate: python py/generate_large_graph.py --nodes 10000 --out data/raw/large_graph.edges
# - Scale existing: python py/scale_existing_graph.py --input data/raw/12831.edges --copies 20 --out data/raw/scaled_graph.edges
# Then set: GRAPH_EDGELIST="data/raw/large_graph.edges" or "data/raw/scaled_graph.edges"
PER_USER="data/per_user_sentiment.csv"  # aggregated user sentiment (should already exist)
OUT_ROOT="results"
PYTHON="${PYTHON:-python}"
# Default chunk size exposed to C++ via OMP_CHUNK_SIZE (can be overridden in env)
# Auto-tuned in C++ code based on graph size, but can override here
export OMP_CHUNK_SIZE="${OMP_CHUNK_SIZE:-}"
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

# --- THREADS: generate powers-of-two sequence up to available logical CPUs (override by setting THREADS_LIST before calling) ---
if ! declare -p THREADS_LIST >/dev/null 2>&1; then
  CPU_COUNT=1
  if command -v nproc >/dev/null 2>&1; then
    CPU_COUNT=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    CPU_COUNT=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
  fi

  # Test all thread counts from 1 to 8 to find optimal point
  THREADS_LIST=(1 2 3 4 5 6 7 8)
fi

# Step 0: Skip preprocessing (preprocess_sentiment.py removed - not needed without raw tweets)

# Step 1: Build graph & states
if [ -f data/graph.txt ] && [ -f data/states.txt ]; then
  echo "Using existing data/graph.txt and data/states.txt"
  echo "  Graph info: $(head -1 data/graph.txt | awk '{print $1 " nodes, " $2 " edges"}')"
else
  if [ -f "${GRAPH_EDGELIST}" ]; then
    echo "Building graph & states from edgelist: ${GRAPH_EDGELIST}"
    mkdir -p data
    "${PYTHON}" py/build_graph.py --edgelist "${GRAPH_EDGELIST}" --per-user "${PER_USER}" --out-dir data
    if [ -f data/graph.txt ]; then
      echo "  Graph built: $(head -1 data/graph.txt | awk '{print $1 " nodes, " $2 " edges"}')"
    fi
  else
    echo "ERROR: edgelist missing at ${GRAPH_EDGELIST} and data/graph.txt not present." >&2
    echo "" >&2
    echo "To generate a large graph, run:" >&2
    echo "  python py/generate_large_graph.py --nodes 10000 --model social --out data/raw/large_graph.edges" >&2
    echo "" >&2
    echo "Or use the existing graph:" >&2
    echo "  GRAPH_EDGELIST=\"data/raw/12831.edges\"" >&2
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

# Step 2: Serial baseline (COMMENTED OUT - skipping serial to save time)
# mkdir -p "${OUTDIR}/serial"
# echo "Running serial baseline..."
# echo "  Graph: $(head -1 data/graph.txt | awk '{print $1 " nodes, " $2 " edges"}')"
# echo "  Steps: ${STEPS}, Alpha: ${ALPHA}"
# echo ""
# "${PYTHON}" py/serial_sim.py --graph data/graph.txt --states data/states.txt --steps "${STEPS}" --alpha "${ALPHA}" --out "${OUTDIR}/serial/serial_history.txt" 2>&1
# echo ""
# 
# # capture serial final states if available
# if [ -f "results/serial_final_states.txt" ]; then
#   cp -v results/serial_final_states.txt "${OUTDIR}/serial/serial_final_states.txt" || true
# fi
echo "Skipping serial baseline (commented out). Running parallel only."

# Step 3: build C++ binary
echo "Building C++ binary..."
bash cpp/build.sh
# Check for binary (Windows uses .exe, Unix doesn't)
if [ -x "./cpp/bin/parallel_update.exe" ]; then
  CPP_BIN="./cpp/bin/parallel_update.exe"
elif [ -x "./cpp/bin/parallel_update" ]; then
  CPP_BIN="./cpp/bin/parallel_update"
elif [ -x "./cpp/parallel_update.exe" ]; then
  CPP_BIN="./cpp/parallel_update.exe"
elif [ -x "./cpp/parallel_update" ]; then
  CPP_BIN="./cpp/parallel_update"
else
  echo "C++ binary not found after build. Aborting." >&2
  echo "Checked: ./cpp/bin/parallel_update.exe, ./cpp/bin/parallel_update, ./cpp/parallel_update.exe, ./cpp/parallel_update" >&2
  exit 1
fi
echo "Using C++ binary: ${CPP_BIN}"

# Step 4: parallel runs
mkdir -p "${OUTDIR}/parallel" "${OUTDIR}/logs"
CSV="${OUTDIR}/parallel/execution_time.csv"
echo "threads,time" > "${CSV}"
echo "Running parallel experiments: steps=${STEPS}, alpha=${ALPHA}"

# Tunable env defaults to help scaling; overridden per-run by OMP_NUM_THREADS below
export OMP_DYNAMIC=FALSE            # prevent OpenMP from reducing thread count
export OMP_PROC_BIND=close          # prefer close binding of threads to CPUs
export OMP_PLACES=cores             # request placement on physical cores (where supported)
export KMP_AFFINITY="granularity=fine,compact,1"  # Intel OpenMP affinity hint
# Provide a conservative GOMP_CPU_AFFINITY covering all logical CPUs (adjust if needed)
if command -v nproc >/dev/null 2>&1; then
  export GOMP_CPU_AFFINITY="0-$(($(nproc)-1))"
fi

REPEATS=7   # run each thread-count multiple times and take best (min) time to reduce noise

for T in "${THREADS_LIST[@]}"; do
  echo " -> Testing threads=${T}"
  BEST_TIME=""
  BEST_RUN_IDX=0
  # run multiple repeats to get stable timing (take min)
  for runidx in $(seq 1 "${REPEATS}"); do
    LOG="${OUTDIR}/logs/parallel_threads_${T}_run${runidx}.log"
    OUT_HISTORY="${OUTDIR}/parallel/history_${T}_run${runidx}.txt"
    OUT_STATES="${OUTDIR}/parallel/out_states_${T}_run${runidx}.txt"

    START=$(date +%s.%N)

    # Ensure OpenMP uses exactly T threads and provide affinity hints
    # Show progress on first run, suppress on repeats
    if [ "${runidx}" -eq 1 ]; then
      echo "    Run ${runidx}/${REPEATS} (showing progress)..."
      OMP_NUM_THREADS="${T}" \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        KMP_AFFINITY="${KMP_AFFINITY}" \
        GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-}" \
        "${CPP_BIN}" data/graph.txt data/states.txt "${OUT_STATES}" "${OUT_HISTORY}" "${STEPS}" "${ALPHA}" "${T}" 2>&1 | tee "${LOG}" || {
          echo "Parallel run failed for threads=${T} run=${runidx}. See ${LOG}" >&2
          exit 1
        }
    else
      OMP_NUM_THREADS="${T}" \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        KMP_AFFINITY="${KMP_AFFINITY}" \
        GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-}" \
        "${CPP_BIN}" data/graph.txt data/states.txt "${OUT_STATES}" "${OUT_HISTORY}" "${STEPS}" "${ALPHA}" "${T}" &> "${LOG}" || {
          echo "Parallel run failed for threads=${T} run=${runidx}. See ${LOG}" >&2
          exit 1
        }
    fi

    END=$(date +%s.%N)
    ELAPSED=$(awk "BEGIN {print ${END} - ${START}}")

    echo "    Run ${runidx}: ${ELAPSED} s"
    # update best (robust numeric compare)
    if [ -z "${BEST_TIME}" ]; then
      BEST_TIME="${ELAPSED}"
      BEST_RUN_IDX="${runidx}"
    else
      smaller=$(awk "BEGIN{print (${ELAPSED} < ${BEST_TIME}) ? 1 : 0}")
      if [ "${smaller}" -eq 1 ]; then
        BEST_TIME="${ELAPSED}"
        BEST_RUN_IDX="${runidx}"
      fi
    fi
  done

  # copy best run outputs to canonical names
  if [ "${BEST_RUN_IDX}" -ne 0 ]; then
    cp -v "${OUTDIR}/parallel/history_${T}_run${BEST_RUN_IDX}.txt" "${OUTDIR}/parallel/history_${T}.txt" 2>/dev/null || true
    cp -v "${OUTDIR}/parallel/out_states_${T}_run${BEST_RUN_IDX}.txt" "${OUTDIR}/parallel/out_states_${T}.txt" 2>/dev/null || true
  fi

  echo "${T},${BEST_TIME}" >> "${CSV}"
  echo "Recorded best: ${T},${BEST_TIME} (run ${BEST_RUN_IDX})"
done

cp -v "${CSV}" "${OUTDIR}/execution_time.csv"

# Step 5: copy last out_states (last thread)
LAST_INDEX=$(( ${#THREADS_LIST[@]} - 1 ))
LAST_T="${THREADS_LIST[$LAST_INDEX]}"
if [ -f "${OUTDIR}/parallel/out_states_${LAST_T}.txt" ]; then
  cp -v "${OUTDIR}/parallel/out_states_${LAST_T}.txt" "${OUTDIR}/parallel/out_states_last.txt"
fi

# Step 6: plots
mkdir -p "${OUTDIR}/plots"
echo "Generating plots..."
# Skip serial history plot if serial was skipped
if [ -f "${OUTDIR}/serial/serial_history.txt" ]; then
  "${PYTHON}" py/plot.py --history "${OUTDIR}/serial/serial_history.txt" --out "${OUTDIR}/plots/serial_history.png"
else
  echo "Skipping serial history plot (serial simulation was skipped)"
fi
"${PYTHON}" py/plot.py --speedup "${OUTDIR}/execution_time.csv" --out "${OUTDIR}/plots/execution_time.png"
cp -v "${OUTDIR}/parallel/history_"*.txt "${OUTDIR}/plots/" 2>/dev/null || true

# Step 7: metadata — create JSON safely (uses $PYTHON and serializes THREADS_LIST)
METAFILE="${OUTDIR}/experiment_metadata.json"
# THREADS_JSON generation: pipe the numbers to Python so we avoid embedding multiline strings in a heredoc
THREADS_JSON=$(printf "%s\n" "${THREADS_LIST[@]}" | "${PYTHON}" -c 'import sys,json; arr=[int(x) for x in sys.stdin.read().split() if x.strip()]; print(json.dumps(arr))')

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
meta['execution_time_csv'] = os.path.abspath(os.path.join(outdir, "execution_time.csv"))
meta['plots'] = {
    'serial': os.path.abspath(os.path.join(outdir, "plots", "serial_history.png")),
    'execution_time': os.path.abspath(os.path.join(outdir, "plots", "execution_time.png"))
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
if [ -f "${OUTDIR}/serial/serial_history.txt" ]; then
  echo " - Serial history: ${OUTDIR}/serial/serial_history.txt"
else
  echo " - Serial history: SKIPPED"
fi
echo " - Parallel histories: ${OUTDIR}/parallel/history_*.txt"
echo " - Execution time CSV: ${OUTDIR}/execution_time.csv"
echo " - Plots: ${OUTDIR}/plots/*.png"
echo " - Logs: ${OUTDIR}/logs/*.log"
echo " - Metadata: ${OUTDIR}/experiment_metadata.json"
echo "-----------------------------"

