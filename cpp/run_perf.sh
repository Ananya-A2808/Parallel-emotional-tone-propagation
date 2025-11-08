#!/usr/bin/env bash
set -e
BINARY="./cpp/bin/parallel_update"
GRAPH="data/graph.txt"
STATES="data/states.txt"
OUTDIR="results"
mkdir -p "$OUTDIR"
T=100
ALPHA=0.25
THREADS_LIST=(1 2 4 8 16)

CSV="$OUTDIR/speedup.csv"
echo "threads,time" > "$CSV"

if [ ! -x "$BINARY" ]; then
  echo "Binary not found at $BINARY. Try: bash cpp/build.sh" >&2
  exit 1
fi

for THREADS in "${THREADS_LIST[@]}"; do
  echo "Running threads=$THREADS ..."
  LOG="$OUTDIR/log_threads_${THREADS}.txt"
  START_TIME=$(python - <<PY
import time
print(time.time())
PY
)
  # run and capture stdout/stderr to log
  "$BINARY" "$GRAPH" "$STATES" "$OUTDIR/out_states_${THREADS}.txt" "$OUTDIR/history_${THREADS}.txt" "$T" "$ALPHA" "$THREADS" &> "$LOG"
  END_TIME=$(python - <<PY
import time
print(time.time())
PY
)
  ELAPSED=$(python - <<PY
print(float($END_TIME) - float($START_TIME))
PY
)
  echo "${THREADS},${ELAPSED}" >> "$CSV"
  echo "Recorded ${THREADS},${ELAPSED}"
done

echo "Wrote $CSV"
