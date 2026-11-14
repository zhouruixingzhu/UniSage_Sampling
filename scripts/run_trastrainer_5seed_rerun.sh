#!/usr/bin/env bash
# Sequentially run TraStrainer v0.2 seeds 2-5 for hip+TT primary only.
# aiops21 5-seed is handled in a separate session because it is ~50x slower.
# Stamps run-id to match the seed=1 naming convention
# (trastrainer_v02_hipster_train_ticket_primary_seed-N).
set -euo pipefail

# Activate the desired environment before running this script. PYTHON_BIN may
# name a command on PATH or an executable path (relative to the caller's cwd).
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! PYTHON_BIN=$(command -v "$PYTHON_BIN"); then
  echo "Error: Python interpreter not found. Activate your environment or set PYTHON_BIN." >&2
  exit 1
fi
case "$PYTHON_BIN" in
  /*) ;;
  *) PYTHON_BIN="$PWD/$PYTHON_BIN" ;;
esac
if [[ ! -f "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Error: PYTHON_BIN must point to an executable file: $PYTHON_BIN" >&2
  exit 1
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$REPO_ROOT"

LOG_DIR=logs/trastrainer_5seed_rerun
mkdir -p "$LOG_DIR"

start_all=$(date +%s)

for N in 2 3 4 5; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] >>> hip+TT primary seed=${N}"
  /usr/bin/time -p "$PYTHON_BIN" scripts/run_trastrainer_baseline.py \
    --seed ${N} \
    --datasets hipster train_ticket \
    --scope primary \
    --run-id "trastrainer_v02_hipster_train_ticket_primary_seed-${N}" \
    > "${LOG_DIR}/hip_tt_seed${N}.log" 2>&1
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] <<< hip+TT primary seed=${N} done"
done

end_all=$(date +%s)
echo "[done] total elapsed seconds: $((end_all - start_all))"
