#!/usr/bin/env bash
set -euo pipefail

# Runs 39–42 qubit EdgeQuantum suite inside a tmux session.
# Logs:
# - Per-run logs + JSON updates handled by run_high_qubit_suite.py
# - This wrapper tees a top-level session log as well.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION="eq3942"
ATTACH=0

if [[ "${1:-}" == "--attach" ]]; then
  ATTACH=1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
TOP_LOG="run_logs/${SESSION}_${TS}.log"
mkdir -p run_logs

CMD=(
  python3 run_high_qubit_suite.py
  --qubits 39 40 41 42
  --results-file comprehensive_results.json
  --log-dir "run_logs/high_qubit/${SESSION}_${TS}"
  --storage-dir state_tmp
  --rerun
)

# Clean leftover storage artifacts for a clean restart.
rm -f \
  state_tmp/state_39q.bin* \
  state_tmp/state_40q.bin* \
  state_tmp/state_41q.bin* \
  state_tmp/state_42q.bin* \
  2>/dev/null || true

# Create session detached, run command, keep shell open on exit
RUNLINE="${CMD[*]} 2>&1 | tee -a ${TOP_LOG}; echo; echo '[done]'; exec bash"

tmux new-session -d -s "$SESSION" "$RUNLINE"

echo "Started tmux session: $SESSION"
echo "Top-level log: $TOP_LOG"
echo "Attach with: tmux attach -t $SESSION"

if [[ "$ATTACH" == "1" ]]; then
  tmux attach -t "$SESSION"
fi
